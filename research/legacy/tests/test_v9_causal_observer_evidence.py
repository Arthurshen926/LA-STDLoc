import json

import torch

from common.hashing import sha256_file
from scripts.run_v9_causal_observer import (
    _clean_protection_evidence,
    _pose_entered_training_evidence,
)
from scripts.split_v14_feedback_families import split_batches


def test_training_evidence_keeps_only_pose_entered_changed_rows_in_order():
    evidence = {
        "changed_query_rows": torch.tensor([3, 1, 4]),
        "positive_anchor_rows": torch.tensor([13, 11, 14]),
        "negative_anchor_rows": torch.tensor([23, 21, 24]),
        "positive_rank": torch.tensor([2, 3, 1]),
        "positive_scores": torch.tensor([0.3, 0.1, 0.4]),
        "negative_scores": torch.tensor([0.5, 0.5, 0.5]),
    }
    descriptors = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    result = _pose_entered_training_evidence(
        evidence=evidence,
        source_descriptors=descriptors,
        selected_query_rows=torch.tensor([4, 0, 3]),
        authorized=True,
        task_gain=0.25,
    )

    assert result["query_rows"].tolist() == [3, 4]
    assert torch.equal(result["query_descriptors"], descriptors[[3, 4]])
    assert result["positive_anchor_rows"].tolist() == [13, 14]
    assert result["negative_anchor_rows"].tolist() == [23, 24]
    assert result["positive_rank"].tolist() == [2, 1]
    assert result["alternative_pose_entered_mask"].tolist() == [True, True]
    assert result["candidate_changed_alternative_pose_entered_mask"].tolist() == [
        True,
        False,
        True,
    ]


def test_unauthorized_training_evidence_exposes_no_training_row():
    evidence = {
        "changed_query_rows": torch.tensor([1]),
        "positive_anchor_rows": torch.tensor([3]),
        "negative_anchor_rows": torch.tensor([4]),
        "positive_rank": torch.tensor([1]),
        "positive_scores": torch.tensor([0.2]),
        "negative_scores": torch.tensor([0.3]),
    }
    result = _pose_entered_training_evidence(
        evidence=evidence,
        source_descriptors=torch.randn(2, 4),
        selected_query_rows=torch.tensor([1]),
        authorized=False,
        task_gain=float("nan"),
    )

    assert result["query_rows"].numel() == 0
    assert result["query_descriptors"].shape == (0, 4)
    assert result["positive_anchor_rows"].numel() == 0


def test_clean_protection_serializes_row_descriptor_top1_top2_same_order():
    descriptors = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    candidates = torch.tensor(
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], dtype=torch.long
    )
    scores = torch.tensor(
        [[0.9, 0.7], [0.8, 0.5], [0.7, 0.6], [0.9, 0.2], [0.6, 0.1]]
    )
    result = _clean_protection_evidence(
        clean_rows=torch.tensor([3, 1]),
        source_descriptors=descriptors,
        candidate_rows=candidates,
        candidate_scores=scores,
    )

    assert result["query_rows"].tolist() == [3, 1]
    assert torch.equal(result["query_descriptors"], descriptors[[3, 1]])
    assert result["positive_anchor_rows"].tolist() == [6, 2]
    assert result["negative_anchor_rows"].tolist() == [7, 3]
    assert torch.allclose(result["initial_margin"], torch.tensor([0.7, 0.3]))


def test_v14_family_split_preserves_v2_row_binding_contract(tmp_path):
    records = []
    for query, family in enumerate((10, 11)):
        path = tmp_path / f"observer_{query}.pt"
        torch.save(
            {
                "pose_family_id": family,
                "training_evidence": {"query_rows": torch.tensor([0])},
            },
            path,
        )
        records.append(
            {
                "query_index": query,
                "path": str(path),
                "sha256": sha256_file(path),
                "category": "causal_precision_deficit",
                "can_train_metric": True,
            }
        )
    certified_path = tmp_path / "certified.json"
    certified_path.write_text(
        json.dumps({"view_role": "feedback_query", "records": []})
    )
    identity = {
        "map_sha256": "map",
        "certified_batch": str(certified_path),
        "certified_batch_sha256": sha256_file(certified_path),
    }
    payloads = []
    for index, record in enumerate(records):
        payloads.append(
            {
                "schema": "lafgs_v9_no_loo_causal_feedback_batch",
                "version": 2,
                "uses_test_queries": False,
                "loo_used": False,
                "accepted_query_row_policy": "v2_row_valid_only",
                "training_rows_are_alternative_pose_entered_only": True,
                "clean_protection_has_explicit_query_rows": True,
                "input": identity,
                "records": [record],
                "shard_index": index,
                "shard_count": 2,
            }
        )

    design, control = split_batches(payloads, design_fraction=0.5, seed=3)

    for output in (design, control):
        assert output["version"] == 2
        assert output["training_rows_are_alternative_pose_entered_only"] is True
        assert output["clean_protection_has_explicit_query_rows"] is True
        assert output["source_view_role"] == "feedback_query"
        assert len(output["pose_family_ids"]) == output["pose_family_count"]
