import numpy as np

from map_learning.pose_set_oracle import (
    PoseSetAction,
    apply_pose_set_actions,
    beam_search_pose_set,
    normalized_pose_risk,
)


def test_pose_set_actions_apply_swaps_and_rejections():
    revised, active = apply_pose_set_actions(
        np.asarray([1, 2, 3]),
        (PoseSetAction("swap", 0, 7), PoseSetAction("reject", 2)),
    )
    assert revised.tolist() == [7, 2, 3]
    assert active.tolist() == [True, True, False]


def test_beam_search_can_find_joint_non_additive_gain():
    actions = [PoseSetAction("swap", 0, 4), PoseSetAction("swap", 1, 5)]

    def evaluate(selected):
        count = len(selected)
        return {"risk": {0: 2.0, 1: 2.1, 2: 0.5}[count]}

    selected, outcome, _ = beam_search_pose_set(
        actions, evaluate, maximum_depth=2, beam_width=2
    )
    assert len(selected) == 2
    assert outcome["risk"] == 0.5


def test_normalized_pose_risk_is_scene_scaled():
    first = normalized_pose_risk(
        translation_cm=5.0,
        rotation_deg=5.0,
        translation_scale_m=0.05,
        rotation_scale_deg=5.0,
    )
    second = normalized_pose_risk(
        translation_cm=50.0,
        rotation_deg=5.0,
        translation_scale_m=0.5,
        rotation_scale_deg=5.0,
    )
    assert first == second
