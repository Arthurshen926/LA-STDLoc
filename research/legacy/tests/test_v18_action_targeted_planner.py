import torch

from evidence.v18_action_targeted_planner import select_action_targeted_queries


def test_action_targeted_planner_realizes_three_supervision_roles() -> None:
    poses = torch.eye(4).repeat(3, 1, 1).double()
    poses[1, 0, 3] = -0.5
    poses[2, 0, 3] = 0.5
    plan = {
        "query_count": 3,
        "uses_test_queries": False,
        "pose_w2c": poses,
        "intrinsics": torch.tensor(
            [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]
        ).repeat(3, 1, 1),
        "image_hw": torch.tensor([[100, 100]]).repeat(3, 1),
        "pose_family_ids": torch.arange(3),
        "visible_cell_count": torch.tensor([8, 10, 12]),
        "source_mapping_indices": [[0], [1], [2]],
    }
    result = select_action_targeted_queries(
        candidate_plan=plan,
        anchor_xyz=torch.tensor([[0.0, 0.0, 5.0], [0.2, 0.0, 5.0]]),
        harmful_anchor_rows=torch.tensor([0]),
        reactivated_anchor_rows=torch.empty(0, dtype=torch.long),
        backup_offsets=torch.tensor([0, 1]),
        backup_anchor_rows=torch.tensor([1]),
        anchor_observation_offsets=torch.tensor([0, 2, 3]),
        observation_query_indices=torch.tensor([0, 1, 2]),
        mapping_poses_w2c=poses,
        maximum_queries=3,
    )
    assert sorted(result["query_kinds"]) == [
        "global_collateral",
        "intervention",
        "necessity",
    ]
    assert result["action_targeted_planner"]["uses_test_queries"] is False
    assert result["action_targeted_planner"]["loo_used"] is False


def test_action_targeted_planner_targets_reactivation_without_backup() -> None:
    poses = torch.eye(4).repeat(2, 1, 1).double()
    plan = {
        "query_count": 2,
        "uses_test_queries": False,
        "pose_w2c": poses,
        "intrinsics": torch.tensor(
            [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]
        ).repeat(2, 1, 1),
        "image_hw": torch.tensor([[100, 100]]).repeat(2, 1),
        "pose_family_ids": torch.arange(2),
        "visible_cell_count": torch.tensor([4, 8]),
    }
    result = select_action_targeted_queries(
        candidate_plan=plan,
        anchor_xyz=torch.tensor([[0.0, 0.0, 5.0]]),
        harmful_anchor_rows=torch.empty(0, dtype=torch.long),
        reactivated_anchor_rows=torch.tensor([0]),
        backup_offsets=torch.tensor([0]),
        backup_anchor_rows=torch.empty(0, dtype=torch.long),
        anchor_observation_offsets=torch.tensor([0, 1]),
        observation_query_indices=torch.tensor([0]),
        mapping_poses_w2c=poses,
        maximum_queries=2,
    )
    assert sorted(result["query_kinds"]) == ["global_collateral", "intervention"]
    assert (
        result["action_targeted_planner"]["target_reactivated_anchor_count"] == 1
    )
    assert bool((result["target_action_direction_score"] >= 0.5).all())


def test_reactivation_outside_original_view_cone_is_not_intervention() -> None:
    mapping_pose = torch.eye(4).repeat(1, 1, 1).double()
    candidate_pose = torch.eye(4).repeat(1, 1, 1).double()
    # Mapping camera center is at the origin, while the candidate center is on
    # the opposite side of the Anchor; the 3D point is still in-frame after a
    # 180-degree camera rotation, but its appearance direction is unsupported.
    candidate_pose[0, :3, :3] = torch.diag(torch.tensor([-1.0, 1.0, -1.0]))
    candidate_pose[0, 2, 3] = 10.0
    plan = {
        "query_count": 1,
        "uses_test_queries": False,
        "pose_w2c": candidate_pose,
        "intrinsics": torch.tensor(
            [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]
        ),
        "image_hw": torch.tensor([[100, 100]]),
        "pose_family_ids": torch.arange(1),
        "visible_cell_count": torch.tensor([1]),
    }
    result = select_action_targeted_queries(
        candidate_plan=plan,
        anchor_xyz=torch.tensor([[0.0, 0.0, 5.0]]),
        harmful_anchor_rows=torch.empty(0, dtype=torch.long),
        reactivated_anchor_rows=torch.tensor([0]),
        backup_offsets=torch.tensor([0]),
        backup_anchor_rows=torch.empty(0, dtype=torch.long),
        anchor_observation_offsets=torch.tensor([0, 1]),
        observation_query_indices=torch.tensor([0]),
        mapping_poses_w2c=mapping_pose,
        maximum_queries=1,
    )
    assert result["query_kinds"] == ["global_collateral"]
    assert float(result["target_action_direction_score"][0]) < 0.5
