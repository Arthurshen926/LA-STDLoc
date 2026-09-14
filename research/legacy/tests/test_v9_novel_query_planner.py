import torch

from evidence.v7_query_planner import camera_centers
from evidence.v9_novel_query_planner import plan_v9_novel_queries


def _inputs(count: int = 48):
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
            torch.linspace(0, 9, 20),
            torch.linspace(-2, 2, 8),
            torch.linspace(4, 12, 6),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    return poses, intrinsics, hw, names, xyz


def test_v9_planner_forbids_interpolation_and_enforces_novelty() -> None:
    poses, intrinsics, hw, names, xyz = _inputs()
    plan = plan_v9_novel_queries(
        pose_w2c=poses,
        intrinsics=intrinsics,
        image_hw=hw,
        names=names,
        anchor_xyz=xyz,
        ambiguity_xyz=xyz[::20],
        role="feedback_query",
        seed=2026,
        maximum_queries=8,
        anchor_projection_stride=1,
    )
    assert plan["trajectory_interpolation_candidate_count"] == 0
    assert plan["loo_used"] is False
    assert set(plan["query_kinds"]) <= {"novel_se3", "ambiguity_directed"}
    assert bool((plan["novelty_baselines"] >= 0.65).all())
    assert bool((plan["nearest_view_angle_deg"] >= 8.0).all())
    distance = torch.cdist(camera_centers(plan["pose_w2c"]), camera_centers(poses))
    assert bool((distance.min(1).values > 0).all())


def test_v9_feedback_confirmation_are_disjoint_and_deterministic() -> None:
    poses, intrinsics, hw, names, xyz = _inputs()
    common = dict(
        pose_w2c=poses,
        intrinsics=intrinsics,
        image_hw=hw,
        names=names,
        anchor_xyz=xyz,
        ambiguity_xyz=xyz[::20],
        maximum_queries=8,
        anchor_projection_stride=1,
    )
    left = plan_v9_novel_queries(role="feedback_query", seed=2026, **common)
    replay = plan_v9_novel_queries(role="feedback_query", seed=2026, **common)
    right = plan_v9_novel_queries(
        role="confirmation_query",
        seed=2027,
        forbidden_pose_family_ids=left["pose_family_ids"].tolist(),
        **common,
    )
    assert left["plan_sha256"] == replay["plan_sha256"]
    delta = (left["pose_w2c"][:, None] - right["pose_w2c"][None]).abs().amax((-1, -2))
    assert bool((delta.min(1).values > 1e-9).all())
    assert not set(left["pose_family_ids"].tolist()) & set(
        right["pose_family_ids"].tolist()
    )
