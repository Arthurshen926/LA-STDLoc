import json
import sys

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from scripts.train_v20_sparse_anchor_descriptors import main


def test_v20_training_cli_writes_json_safe_bound_artifacts(
    tmp_path, monkeypatch
) -> None:
    anchors = torch.zeros((3, 256), dtype=torch.float32)
    anchors[0, :2] = torch.tensor([1.0, 0.0])
    anchors[1, :2] = torch.tensor([0.9, 0.4358899])
    anchors[2, 2] = 1.0
    anchors = F.normalize(anchors, dim=1)
    baseline_path = tmp_path / "baseline.pt"
    torch.save(
        {
            "schema": "lafgs_materialized_anchor_map",
            "anchor_ids": torch.arange(3),
            "anchor_xyz": torch.eye(3),
            "anchor_features": anchors,
        },
        baseline_path,
    )
    repair_query = torch.zeros((1, 256))
    repair_query[0, :2] = torch.tensor([0.8, 0.6])
    protection_query = anchors[0].reshape(1, -1)
    evidence_path = tmp_path / "evidence.pt"
    torch.save(
        {
            "schema": "lafgs_v20_topk_competition_evidence",
            "version": 2,
            "uses_test_queries": False,
            "loo_used": False,
            "strong_feedback_authorized": False,
            "design_query_indices": [7],
            "design_pose_family_ids": [70],
            "design_source_record_sha256s": ["s" * 64],
            "inputs": {
                "anchor_map_sha256": sha256_file(baseline_path),
                "design_batch_sha256": "d" * 64,
            },
            "per_query": [
                {
                    "query_index": 7,
                    "pose_family_id": 70,
                    "repair_row_count": 1,
                    "protection_row_count": 1,
                }
            ],
            "repair_query_descriptors": repair_query,
            "repair_positive_offsets": torch.tensor([0, 1]),
            "repair_positive_anchor_rows": torch.tensor([0]),
            "repair_negative_offsets": torch.tensor([0, 1]),
            "repair_negative_anchor_rows": torch.tensor([1]),
            "repair_wrong_winner_anchor_rows": torch.tensor([1]),
            "repair_wrong_winner_clean_support_family_counts": torch.tensor([0]),
            "negative_action_anchor_rows": torch.empty(0, dtype=torch.long),
            "minimum_negative_action_clean_pose_families": 2,
            "repair_sample_weights": torch.ones(1),
            "protection_query_descriptors": protection_query,
            "protection_positive_offsets": torch.tensor([0, 1]),
            "protection_positive_anchor_rows": torch.tensor([0]),
            "protection_negative_offsets": torch.tensor([0, 1]),
            "protection_negative_anchor_rows": torch.tensor([1]),
            "protection_initial_margin": torch.tensor(
                [float(protection_query @ anchors[0] - protection_query @ anchors[1])]
            ),
        },
        evidence_path,
    )
    output_dir = tmp_path / "proposal"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_v20_sparse_anchor_descriptors.py",
            "--baseline-map",
            str(baseline_path),
            "--evidence",
            str(evidence_path),
            "--steps",
            "1",
            "--device",
            "cpu",
            "--output-dir",
            str(output_dir),
        ],
    )
    main()
    report = json.loads((output_dir / "report.json").read_text())
    assert isinstance(report["training"]["selected_anchor_rows"], list)
    assert report["design_split"]["pose_family_ids"] == [70]
    assert report["design_split"]["source_record_sha256s"] == ["s" * 64]
    candidate = torch.load(
        output_dir / "candidate_anchor_map.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert candidate["v20_sparse_descriptor_action"]["design_batch_sha256"] == "d" * 64
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    assert candidate["anchor_features"].dtype == baseline["anchor_features"].dtype
