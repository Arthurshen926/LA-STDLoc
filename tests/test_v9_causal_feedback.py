import pytest
import torch

from map_learning.v9_causal_feedback import (
    aggregate_actual_removal_gain,
    first_correct_topk_replacement,
    require_no_loo_feedback_contract,
    topk_geometric_correctness,
)


def test_topk_geometry_emits_changed_wrong_to_right_rows() -> None:
    keypoints = torch.tensor([[50.0, 50.0], [70.0, 50.0]])
    xyz = torch.tensor([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.3, 0.0, 5.0]])
    candidates = torch.tensor([[2, 0], [0, 1]])
    intrinsic = torch.tensor([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    correct = topk_geometric_correctness(
        keypoints=keypoints,
        candidate_anchor_rows=candidates,
        anchor_xyz=xyz,
        pose_w2c=torch.eye(4),
        intrinsic=intrinsic,
        alpha=torch.ones(100, 100),
        depth=torch.full((100, 100), 5.0),
        row_valid=torch.ones(2, dtype=torch.bool),
    )
    evidence = first_correct_topk_replacement(
        candidates, torch.tensor([[0.9, 0.8], [0.9, 0.8]]), correct
    )
    assert evidence["changed_query_rows"].tolist() == [0, 1]
    assert evidence["positive_anchor_rows"].tolist() == [0, 1]


def test_no_loo_contract_is_fail_closed() -> None:
    require_no_loo_feedback_contract({"loo_used": False})
    with pytest.raises(ValueError, match="no-LOO"):
        require_no_loo_feedback_contract({"query_descriptor_loo": True})


def test_actual_removal_gain_needs_two_families_and_limits_harm() -> None:
    records = [
        {"anchor_row": 7, "pose_family_id": 1, "baseline_task_error": 2.0, "removed_task_error": 1.0},
        {"anchor_row": 7, "pose_family_id": 2, "baseline_task_error": 1.5, "removed_task_error": 1.0},
        {"anchor_row": 8, "pose_family_id": 1, "baseline_task_error": 1.0, "removed_task_error": 0.5},
    ]
    result = aggregate_actual_removal_gain(records)
    assert result["loo_used"] is False
    assert result["authorized_anchor_rows"].tolist() == [7]
