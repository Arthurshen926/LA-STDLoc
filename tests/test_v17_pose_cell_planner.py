import torch

from evidence.v17_pose_cell_planner import plan_v17_pose_cell_confirmation


def test_pose_cell_confirmation_is_fresh_but_keeps_parent_blocks() -> None:
    count = 48
    poses = torch.eye(4, dtype=torch.float64).repeat(count, 1, 1)
    poses[:, 0, 3] = -torch.arange(count, dtype=torch.float64) * 0.2
    intrinsics = torch.eye(3, dtype=torch.float64).repeat(count, 1, 1)
    intrinsics[:, 0, 0] = 500
    intrinsics[:, 1, 1] = 500
    intrinsics[:, 0, 2] = 320
    intrinsics[:, 1, 2] = 240
    image_hw = torch.tensor([[480, 640]]).repeat(count, 1)
    xyz = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 9, 20),
            torch.linspace(-2, 2, 8),
            torch.linspace(4, 12, 6),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    prior = poses[:16].clone()
    plan = plan_v17_pose_cell_confirmation(
        pose_w2c=poses,
        intrinsics=intrinsics,
        image_hw=image_hw,
        names=[f"seq/frame{index}.png" for index in range(count)],
        anchor_xyz=xyz,
        prior_pose_w2c=prior,
        prior_source_family_ids=torch.arange(16),
        seed=17,
        maximum_queries=8,
        anchor_projection_stride=1,
    )
    assert plan["query_count"] == 8
    assert torch.unique(plan["pose_family_ids"]).numel() == 8
    assert bool((plan["nearest_prior_combined_separation"] >= 1.0).all())
    assert bool((plan["novelty_baselines"] <= 2.35).all())
    assert plan["planner_contract"]["freshness_unit"] == "continuous_se3_pose_cell"
    assert plan["planner_contract"]["statistical_block_unit"] == "source_mapping_parent"
    assert len(set(plan["pose_cell_ids"])) == 8
