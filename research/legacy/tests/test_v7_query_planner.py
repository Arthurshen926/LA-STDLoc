import torch

from common.v7_contracts import require_view_role
from evidence.v7_query_planner import (
    camera_centers,
    plan_v7_novel_queries,
    plan_v7_test_pose_render_diagnostic,
)


def _inputs(count: int = 12):
    poses = torch.eye(4, dtype=torch.float64).repeat(count, 1, 1)
    poses[:, 0, 3] = -torch.arange(count, dtype=torch.float64) * 0.2
    intrinsics = torch.eye(3, dtype=torch.float64).repeat(count, 1, 1)
    intrinsics[:, 0, 0] = 500
    intrinsics[:, 1, 1] = 500
    hw = torch.tensor([[480, 640]]).repeat(count, 1)
    names = [f"seq1/frame{index:05d}.png" for index in range(count)]
    return poses, intrinsics, hw, names


def test_v7_planner_is_deterministic_novel_and_clean_only() -> None:
    poses, intrinsics, hw, names = _inputs()
    arguments = dict(
        pose_w2c=poses, intrinsics=intrinsics, image_hw=hw, names=names,
        role="feedback_query", seed=2026, maximum_queries=8,
    )
    left = plan_v7_novel_queries(**arguments)
    right = plan_v7_novel_queries(**arguments)
    assert left["plan_sha256"] == right["plan_sha256"]
    assert torch.equal(left["pose_w2c"], right["pose_w2c"])
    assert left["render_protocol"] == "clean_once_per_pose"
    assert left["enters_track_registry"] is False
    assert left["enters_anchor_observation_csr"] is False
    assert left["enters_descriptor_bank"] is False
    require_view_role(left, "feedback_query")
    distance = torch.cdist(camera_centers(left["pose_w2c"]), camera_centers(poses))
    assert bool((distance.min(1).values > 1e-9).all())


def test_feedback_and_confirmation_batches_are_disjoint() -> None:
    poses, intrinsics, hw, names = _inputs()
    common = dict(
        pose_w2c=poses, intrinsics=intrinsics, image_hw=hw, names=names,
        seed=2026, maximum_queries=8,
    )
    feedback = plan_v7_novel_queries(role="feedback_query", **common)
    confirmation = plan_v7_novel_queries(role="confirmation_query", **common)
    pose_distance = (
        feedback["pose_w2c"][:, None] - confirmation["pose_w2c"][None]
    ).abs().amax(dim=(-1, -2))
    assert bool((pose_distance.min(1).values > 1e-9).all())
    require_view_role(confirmation, "confirmation_query")


def test_test_pose_diagnostic_is_metadata_only_and_nonformal() -> None:
    poses, intrinsics, hw, names = _inputs()
    plan = plan_v7_test_pose_render_diagnostic(
        mapping_pose_w2c=poses,
        mapping_names=names,
        test_pose_w2c=poses[:3],
        test_intrinsics=intrinsics[:3],
        test_image_hw=hw[:3],
        test_names=["test/a.png", "test/b.png", "test/c.png"],
        query_indices=torch.tensor([4, 7, 9]),
    )
    assert plan["formal_protocol_eligible"] is False
    assert plan["uses_test_pose_metadata"] is True
    assert plan["uses_test_rgb"] is False
    assert plan["query_indices"].tolist() == [4, 7, 9]
    assert not any("path" in key.lower() or "rgb" in key.lower() and plan[key] for key in plan)
    require_view_role(plan, "test_pose_render_diagnostic")
