import numpy as np
import torch

from map_learning.pose_set_oracle import PoseSetAction
from map_learning.v6_pose_set_oracle import (
    apply_swaps,
    bounded_minimum_success_set,
    unique_anchor_rows,
)


def test_unique_anchor_rows_uses_lowest_gt_reprojection_residual():
    pairs = torch.tensor([[0, 0], [1, 0], [2, 1]])
    selected = unique_anchor_rows(
        pairs,
        keypoints=torch.tensor([[0.2, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        anchor_xyz=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        intrinsics=torch.eye(3),
        pose_w2c=torch.eye(4),
    )
    assert selected.tolist() == [[1, 0], [2, 1]]


def test_bounded_minimum_success_set_finds_joint_pose_flip():
    actions = [PoseSetAction("swap", index, 10 + index, 3 - index) for index in range(3)]

    def evaluate(selected):
        return {"success": len(selected) >= 2, "risk": 4.0 - len(selected)}

    selected, outcome, trace = bounded_minimum_success_set(
        actions, evaluate, maximum_depth=3, beam_width=2
    )
    assert len(selected) == 2
    assert outcome["success"] is True
    assert trace[-1]["depth"] == 2


def test_bounded_minimum_success_set_reports_unavailable():
    selected, outcome, trace = bounded_minimum_success_set(
        [PoseSetAction("swap", 0, 1)],
        lambda _: {"success": False, "risk": 2.0},
        maximum_depth=1,
        beam_width=1,
    )
    assert selected is None
    assert outcome is None
    assert trace[-1]["success_count"] == 0


def test_apply_swaps_rejects_two_edits_to_one_row():
    actions = (PoseSetAction("swap", 0, 2), PoseSetAction("swap", 0, 3))
    try:
        apply_swaps(np.asarray([1]), actions)
    except ValueError as error:
        assert "more than once" in str(error)
    else:
        raise AssertionError("contradictory swaps must fail")
