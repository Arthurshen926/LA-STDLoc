import json
from pathlib import Path
import sys

import pytest
import torch

from common.hashing import sha256_file
from map_learning.v18_provenance_truth import (
    TRUTH_AMBIGUOUS,
    TRUTH_EQUIVALENT,
    TRUTH_UNIQUE,
)
from scripts.materialize_v20_topk_feedback import (
    _bind_v9_training_rows,
    _truth_bound_clean_protection,
    main,
)


def _truth(status, rows):
    values = [torch.tensor(item, dtype=torch.long) for item in rows]
    counts = torch.tensor([item.numel() for item in values], dtype=torch.long)
    return {
        "row_count": len(values),
        "truth_status": torch.tensor(status, dtype=torch.long),
        "truth_offsets": torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0))),
        "truth_anchor_rows": torch.cat(values),
    }


def test_training_binding_requires_v9_rank_winner_and_v19_truth_agreement():
    candidates = torch.tensor([[0, 1, 2], [0, 3, 2], [4, 5, 1]])
    scores = torch.tensor([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]])
    truth = _truth(
        [TRUTH_UNIQUE, TRUTH_UNIQUE, TRUTH_EQUIVALENT],
        [[1], [2], [1, 5]],
    )
    training = {
        "query_rows": torch.tensor([0, 1, 2]),
        "query_descriptors": torch.randn(3, 4),
        "positive_anchor_rows": torch.tensor([1, 3, 5]),
        "negative_anchor_rows": torch.tensor([0, 0, 4]),
        "positive_rank": torch.tensor([1, 1, 1]),
        "positive_scores": torch.tensor([0.8, 0.8, 0.8]),
        "negative_scores": torch.tensor([0.9, 0.9, 0.9]),
        "alternative_pose_entered_mask": torch.ones(3, dtype=torch.bool),
    }

    keep, counts = _bind_v9_training_rows(
        training=training,
        candidate_anchor_rows=candidates,
        candidate_scores=scores,
        truth=truth,
        equivalence_class_ids=torch.arange(6),
    )

    assert keep.tolist() == [True, False, True]
    assert counts["authorized"] == 2
    assert counts["v9_positive_not_v19_positive"] == 1


def test_training_binding_rejects_stale_v9_current_winner():
    training = {
        "query_rows": torch.tensor([0]),
        "query_descriptors": torch.randn(1, 4),
        "positive_anchor_rows": torch.tensor([1]),
        "negative_anchor_rows": torch.tensor([2]),
        "positive_rank": torch.tensor([1]),
        "positive_scores": torch.tensor([0.8]),
        "negative_scores": torch.tensor([0.9]),
        "alternative_pose_entered_mask": torch.ones(1, dtype=torch.bool),
    }
    keep, counts = _bind_v9_training_rows(
        training=training,
        candidate_anchor_rows=torch.tensor([[0, 1, 2]]),
        candidate_scores=torch.tensor([[0.9, 0.8, 0.7]]),
        truth=_truth([TRUTH_UNIQUE], [[1]]),
        equivalence_class_ids=torch.arange(3),
    )

    assert keep.tolist() == [False]
    assert counts["negative_not_current_winner"] == 1


def test_clean_protection_fails_closed_without_explicit_query_rows():
    clean = {
        "query_descriptors": torch.randn(1, 4),
        "positive_anchor_rows": torch.tensor([1]),
        "negative_anchor_rows": torch.tensor([0]),
        "initial_margin": torch.tensor([0.1]),
    }
    result, counts = _truth_bound_clean_protection(
        clean=clean,
        candidate_anchor_rows=torch.tensor([[1, 0, 2]]),
        candidate_scores=torch.tensor([[0.9, 0.8, 0.7]]),
        truth=_truth([TRUTH_UNIQUE], [[1]]),
        equivalence_class_ids=torch.arange(3),
    )

    assert result["query_descriptors"].shape[0] == 0
    assert counts["missing_explicit_query_rows"] == 1


def test_clean_protection_uses_truth_proven_topk_nonpositives():
    clean = {
        "query_rows": torch.tensor([0, 1]),
        "query_descriptors": torch.randn(2, 4),
        "positive_anchor_rows": torch.tensor([1, 4]),
        "negative_anchor_rows": torch.tensor([2, 5]),
        "initial_margin": torch.tensor([999.0, 999.0]),
    }
    candidates = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 2]])
    scores = torch.tensor([[0.9, 0.8, 0.75, 0.7], [0.9, 0.85, 0.8, 0.7]])
    equivalence = torch.tensor([0, 1, 2, 1, 4, 4])
    result, counts = _truth_bound_clean_protection(
        clean=clean,
        candidate_anchor_rows=candidates,
        candidate_scores=scores,
        truth=_truth([TRUTH_EQUIVALENT, TRUTH_AMBIGUOUS], [[1, 3], [4, 5]]),
        equivalence_class_ids=equivalence,
    )

    assert result["query_rows"].tolist() == [0]
    assert result["positive_anchor_rows"][0].tolist() == [1]
    assert result["negative_anchor_rows"][0].tolist() == [2, 0]
    assert torch.allclose(result["initial_margin"], torch.tensor([0.1]))
    assert counts["authorized"] == 1
    assert counts["v19_truth_not_decisive"] == 1
    assert counts["legacy_top2_is_v19_positive"] == 1


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _full_materializer_fixture(tmp_path, *, duplicate_teacher=False, pose_mismatch=False):
    map_path = tmp_path / "map.pt"
    torch.save(
        {
            "anchor_ids": torch.arange(4),
            "fine_identity_ids": torch.arange(4),
        },
        map_path,
    )
    map_sha = sha256_file(map_path)
    certified_records = []
    observer_records = []
    observer_paths = []
    for query, family in enumerate((10, 11)):
        source_path = tmp_path / f"source_{query}.pt"
        torch.save({"query_index": query}, source_path)
        source_sha = sha256_file(source_path)
        certified_records.append(
            {
                "query_index": query,
                "decision": "ACCEPT",
                "path": str(source_path),
                "sha256": source_sha,
            }
        )
        observer_path = tmp_path / f"observer_{query}.pt"
        torch.save(
            {
                "schema": "lafgs_v9_no_loo_causal_feedback_record",
                "version": 2,
                "loo_used": False,
                "query_index": query,
                "pose_family_id": family,
                "source_record": str(source_path),
                "source_record_sha256": source_sha,
                "source_query_rows": torch.tensor([0, 1]),
                "invalid_source_row_count": 0,
                "can_train_metric": True,
                "actual_task_gain": 0.5,
                "topk_anchor_rows": torch.tensor([[0, 1, 2], [0, 2, 1]]),
                "topk_scores": torch.tensor([[0.9, 0.8, 0.7], [0.9, 0.8, 0.7]]),
                "training_evidence": {
                    "query_rows": torch.tensor([0]),
                    "query_descriptors": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    "positive_anchor_rows": torch.tensor([1]),
                    "negative_anchor_rows": torch.tensor([0]),
                    "positive_rank": torch.tensor([1]),
                    "positive_scores": torch.tensor([0.8]),
                    "negative_scores": torch.tensor([0.9]),
                    "alternative_pose_entered_mask": torch.tensor([True]),
                    "actual_query_task_gain": 0.5,
                },
                "clean_protection_evidence": {
                    "query_rows": torch.tensor([1]),
                    "query_descriptors": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                    "positive_anchor_rows": torch.tensor([0]),
                    "negative_anchor_rows": torch.tensor([2]),
                    "initial_margin": torch.tensor([0.1]),
                },
            },
            observer_path,
        )
        observer_paths.append(observer_path)
        observer_records.append(
            {
                "query_index": query,
                "path": str(observer_path),
                "sha256": sha256_file(observer_path),
                "category": "causal_precision_deficit",
                "can_train_metric": True,
            }
        )

    certified_path = tmp_path / "certified.json"
    _write_json(
        certified_path,
        {
            "schema": "lafgs_v7_certified_clean_render_batch",
            "version": 1,
            "view_role": "feedback_query",
            "uses_test_queries": False,
            "map_mutation_count": 0,
            "records": certified_records,
        },
    )
    certified_sha = sha256_file(certified_path)
    common_input = {
        "map": str(map_path),
        "map_sha256": map_sha,
        "certified_batch": str(certified_path),
        "certified_batch_sha256": certified_sha,
    }
    manifest_paths = []
    for index, record in enumerate(observer_records):
        manifest_path = tmp_path / f"observer_manifest_{index}.json"
        _write_json(
            manifest_path,
            {
                "schema": "lafgs_v9_no_loo_causal_feedback_batch",
                "version": 2,
                "status": "PASS",
                "uses_test_queries": False,
                "loo_used": False,
                "accepted_query_row_policy": "v2_row_valid_only",
                "training_rows_are_alternative_pose_entered_only": True,
                "clean_protection_has_explicit_query_rows": True,
                "shard_count": 2,
                "shard_index": index,
                "input": common_input,
                "records": [record],
            },
        )
        manifest_paths.append(manifest_path)
    design_path = tmp_path / "design.json"
    _write_json(
        design_path,
        {
            "schema": "lafgs_v9_no_loo_causal_feedback_batch",
            "version": 2,
            "status": "PASS",
            "role": "controller_design",
            "uses_test_queries": False,
            "loo_used": False,
            "accepted_query_row_policy": "v2_row_valid_only",
            "training_rows_are_alternative_pose_entered_only": True,
            "clean_protection_has_explicit_query_rows": True,
            "pose_family_count": 2,
            "pose_family_ids": [10, 11],
            "input": common_input,
            "source_observer_batches": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in manifest_paths
            ],
            "records": observer_records,
        },
    )
    validation_path = tmp_path / "validation.pt"
    torch.save(
        {
            "schema": "lafgs_v19_track_extension_teacher_validation",
            "uses_test_queries": False,
            "loo_used": False,
            "feedback_enters_track_registry": False,
            "reference_available_for_novel_query": False,
            "selection_uses_validation": False,
            "authorization_uses_wilson_lower_bound": True,
            "authorization_requires_independent_mapping_families": True,
            "selected_tiers": {"tier_b": {"authorized_actions": True}},
            "inputs": {"anchor_map_sha256": map_sha},
        },
        validation_path,
    )
    teacher_paths = []
    for index in range(2):
        query = 0 if duplicate_teacher and index == 1 else index
        family = (10, 11)[index]
        if pose_mismatch and index == 1:
            family += 100
        source = certified_records[query]
        truth = _truth([TRUTH_UNIQUE, TRUTH_UNIQUE], [[1], [0]])
        teacher_path = tmp_path / f"teacher_{index}.pt"
        torch.save(
            {
                "schema": "lafgs_v19_novel_track_extension_shard",
                "version": 1,
                "uses_test_queries": False,
                "loo_used": False,
                "view_role": "feedback_query",
                "feedback_enters_track_registry": False,
                "reference_available_for_novel_query": False,
                "shard_index": index,
                "shard_count": 2,
                "tier_action_authorization": {"tier_b": True},
                "records": [
                    {
                        "query_index": query,
                        "pose_family_id": family,
                        "source_record": source["path"],
                        "source_record_sha256": source["sha256"],
                        "source_query_rows": torch.tensor([0, 1]),
                        "truth_tiers": {"tier_b": truth},
                    }
                ],
                "inputs": {
                    "anchor_map": str(map_path),
                    "anchor_map_sha256": map_sha,
                    "certified_batch": str(certified_path),
                    "certified_batch_sha256": certified_sha,
                    "teacher_validation": str(validation_path),
                    "teacher_validation_sha256": sha256_file(validation_path),
                },
            },
            teacher_path,
        )
        teacher_paths.append(teacher_path)
    return map_path, manifest_paths, design_path, teacher_paths


def _run_materializer(
    monkeypatch, tmp_path, fixture, teacher_count=2, extra_args=()
):
    map_path, manifests, design, teachers = fixture
    output = tmp_path / "evidence.pt"
    argv = [
        "materialize_v20_topk_feedback.py",
        "--teacher-shards",
        *map(str, teachers[:teacher_count]),
        "--observer-manifests",
        *map(str, manifests),
        "--design-batch",
        str(design),
        "--anchor-map",
        str(map_path),
        "--teacher-tier",
        "tier_b",
        *map(str, extra_args),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return torch.load(output, map_location="cpu", weights_only=False)


def test_materializer_streams_complete_lineage_and_preserves_v2_fields(
    tmp_path, monkeypatch
):
    evidence = _run_materializer(
        monkeypatch, tmp_path, _full_materializer_fixture(tmp_path)
    )

    assert evidence["version"] == 2
    assert evidence["repair_query_descriptors"].shape[0] == 2
    assert evidence["repair_query_indices"].tolist() == [0, 1]
    assert evidence["repair_source_query_rows"].tolist() == [0, 0]
    assert evidence["protection_query_descriptors"].shape[0] == 2
    assert evidence["negative_action_anchor_rows"].tolist() == [0]
    assert evidence["repair_wrong_winner_clean_support_family_counts"].tolist() == [
        2,
        2,
    ]
    assert evidence["v9_v19_training_binding_counts"]["authorized"] == 2
    assert evidence["design_query_indices"] == [0, 1]
    assert evidence["design_pose_family_ids"] == [10, 11]
    assert evidence["design_source_record_sha256s"] == sorted(
        item["sha256"]
        for item in json.loads(
            Path(evidence["inputs"]["certified_batch"]).read_text()
        )["records"]
    )
    assert len(evidence["inputs"]["teacher_shards"]) == 2


def test_materializer_freezes_negative_action_clean_support_threshold(
    tmp_path, monkeypatch
):
    evidence = _run_materializer(
        monkeypatch,
        tmp_path,
        _full_materializer_fixture(tmp_path),
        extra_args=(
            "--minimum-negative-action-clean-pose-families",
            3,
        ),
    )

    assert evidence["minimum_negative_action_clean_pose_families"] == 3
    assert evidence["negative_action_anchor_rows"].numel() == 0


def test_materializer_rejects_incomplete_teacher_shards(tmp_path, monkeypatch):
    fixture = _full_materializer_fixture(tmp_path)
    with pytest.raises(ValueError, match="teacher shard registry is incomplete"):
        _run_materializer(monkeypatch, tmp_path, fixture, teacher_count=1)


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"duplicate_teacher": True}, "teacher query appears in multiple shards"),
        ({"pose_mismatch": True}, "teacher and observer source bindings differ"),
    ],
)
def test_materializer_rejects_duplicate_query_or_pose_mismatch(
    tmp_path, monkeypatch, fixture_kwargs, message
):
    fixture = _full_materializer_fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(ValueError, match=message):
        _run_materializer(monkeypatch, tmp_path, fixture)
