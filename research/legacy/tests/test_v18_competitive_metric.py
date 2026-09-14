import torch

from map_learning.v18_competitive_metric import (
    build_truth_aligned_metric_evidence,
)
from map_learning.v18_provenance_truth import TRUTH_UNIQUE


def _record(query: int, family: int, gain: float) -> dict:
    return {
        "query_index": query,
        "pose_family_id": family,
        "candidate_anchor_rows": torch.tensor([[1, 0, 2], [0, 2, 1]]),
        "candidate_scores": torch.tensor([[0.9, 0.7, 0.2], [0.8, 0.3, 0.1]]),
        "truth": {
            "row_count": 2,
            "truth_status": torch.tensor([TRUTH_UNIQUE, TRUTH_UNIQUE]),
            "truth_offsets": torch.tensor([0, 1, 2]),
            "truth_anchor_rows": torch.tensor([0, 0]),
        },
        "responsibility": {
            "row_counterfactuals": [
                {
                    "query_row": 0,
                    "wrong_anchor_row": 1,
                    "truth_anchor_row": 0,
                    "replace_with_truth_task_gain": gain,
                }
            ]
        },
    }


def test_metric_evidence_requires_cross_family_and_balances_families() -> None:
    records = [_record(10, 100, 0.2), _record(11, 200, 0.4)]
    descriptors = {10: torch.eye(2, 4), 11: torch.eye(2, 4)}
    keypoints = {
        10: torch.tensor([[10.0, 10.0], [20.0, 20.0]]),
        11: torch.tensor([[10.0, 10.0], [20.0, 20.0]]),
    }
    image_hw = {10: torch.tensor([40, 40]), 11: torch.tensor([40, 40])}
    evidence = build_truth_aligned_metric_evidence(
        responsibility_records=records,
        query_descriptors=descriptors,
        query_keypoints=keypoints,
        query_image_hw=image_hw,
        active_anchor_mask=torch.ones(3, dtype=torch.bool),
    )

    assert evidence["repair_positive_anchor_rows"].tolist() == [0, 0]
    assert evidence["repair_negative_anchor_rows"].tolist() == [1, 1]
    assert torch.allclose(evidence["repair_sample_weights"], torch.tensor([0.5, 0.5]))
    assert evidence["repair_pose_family_count"] == 2
    assert evidence["weighting_policy"].startswith("equal_pose_family")


def test_metric_evidence_rejects_single_family_attractor() -> None:
    records = [_record(10, 100, 0.2), _record(11, 100, 0.4)]
    descriptors = {10: torch.eye(2, 4), 11: torch.eye(2, 4)}
    keypoints = {index: torch.zeros(2, 2) for index in descriptors}
    image_hw = {index: torch.tensor([40, 40]) for index in descriptors}
    try:
        build_truth_aligned_metric_evidence(
            responsibility_records=records,
            query_descriptors=descriptors,
            query_keypoints=keypoints,
            query_image_hw=image_hw,
            active_anchor_mask=torch.ones(3, dtype=torch.bool),
        )
    except RuntimeError as error:
        assert "no truth-aligned metric repair" in str(error)
    else:
        raise AssertionError("single-family repair should not train a shared metric")
