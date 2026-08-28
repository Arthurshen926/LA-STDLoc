import torch

from evidence.v10_confusion_query_planner import plan_v10_confusion_queries


def _inputs(count: int = 80):
    poses = torch.eye(4, dtype=torch.float64).repeat(count, 1, 1)
    poses[:, 0, 3] = -torch.arange(count, dtype=torch.float64) * 0.2
    intrinsics = torch.eye(3, dtype=torch.float64).repeat(count, 1, 1)
    intrinsics[:, 0, 0] = 500
    intrinsics[:, 1, 1] = 500
    intrinsics[:, 0, 2] = 320
    intrinsics[:, 1, 2] = 240
    hw = torch.tensor([[480, 640]]).repeat(count, 1)
    names = [f"seq1/frame{index:05d}.png" for index in range(count)]
    xyz = torch.stack(
        torch.meshgrid(
            torch.linspace(0, 15, 40),
            torch.linspace(-2, 2, 8),
            torch.linspace(5, 12, 6),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    pairs = torch.tensor([[index, index + 1] for index in range(0, 200, 2)])
    return poses, intrinsics, hw, names, xyz, pairs


def test_v10_planner_is_bounded_and_has_no_full_look_at() -> None:
    poses, intrinsics, hw, names, xyz, pairs = _inputs()
    plan = plan_v10_confusion_queries(
        pose_w2c=poses,
        intrinsics=intrinsics,
        image_hw=hw,
        names=names,
        anchor_xyz=xyz,
        confusion_pairs=pairs,
        role="feedback_query",
        feedback_stage="safety",
        seed=7,
        maximum_queries=8,
        priority_anchor_rows=[0, 2, 4],
        anchor_projection_stride=1,
    )
    assert plan["loo_used"] is False
    assert plan["trajectory_interpolation_candidate_count"] == 0
    assert plan["ambiguity_full_look_at"] is False
    assert bool((plan["novelty_baselines"] >= 0.9).all())
    assert bool((plan["novelty_baselines"] <= 2.0).all())
    assert bool((plan["rotation_from_parent_deg"] <= 40.0001).all())
    assert plan["priority_target_count"] > 0
