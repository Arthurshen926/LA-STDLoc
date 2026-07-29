import torch

from localization_training.failure_atlas import (
    FailureAtlasConfig,
    interpolate_pose_w2c,
    plan_failure_conditioned_views,
)


def test_pose_interpolation_preserves_rigid_transform():
    first = torch.eye(4)
    second = torch.eye(4)
    second[0, 3] = -2.0
    pose = interpolate_pose_w2c(first, second, 0.5)
    rotation = pose[:3, :3]
    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-6)
    center = torch.linalg.inv(pose)[:3, 3]
    assert torch.allclose(center, torch.tensor([1.0, 0.0, 0.0]))


def test_view_planner_uses_only_render_eligible_high_risk_queries():
    cache = {
        f"seq1/frame{index:05d}.png": {
            "pose_w2c": torch.eye(4),
            "native_input_hw": [80, 120],
            "native_K": torch.tensor(
                [[100.0, 0.0, 60.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
            ),
        }
        for index in range(3)
    }
    cache["seq1/frame00002.png"]["pose_w2c"] = torch.tensor(
        [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    atlas = {
        "query_names": list(cache),
        "records": [
            {
                "query_name": name,
                "render_eligible": name.endswith("00001.png"),
                "risk": 3.0,
                "positive_topk_recall": 0.4,
                "legal_top1_recall": 0.1,
                "raw_gt_precision_coarse": 0.1,
                "view_bin": 2,
                "source_component": 7,
                "failure_class": "repeated_assignment_deficiency",
            }
            for name in cache
        ],
    }
    config = FailureAtlasConfig(
        maximum_planned_views=2,
        interpolation_alphas=(0.5,),
    )
    planned = plan_failure_conditioned_views(
        atlas=atlas, cache=cache, config=config
    )
    assert len(planned) == 2
    assert all(
        record["source_query"] == "seq1/frame00001.png"
        for record in planned
    )
    assert all(record["source_component"] == 7 for record in planned)


def test_view_planner_caps_correlated_views_per_source():
    cache = {
        f"seq1/frame{index:05d}.png": {
            "pose_w2c": torch.eye(4),
            "native_input_hw": [80, 120],
            "native_K": torch.eye(3),
        }
        for index in range(4)
    }
    atlas = {
        "query_names": list(cache),
        "records": [
            {
                "query_name": name,
                "render_eligible": index in (1, 2),
                "risk": 4.0 - index,
                "positive_topk_recall": 0.4,
                "legal_top1_recall": 0.1,
                "view_bin": 2,
                "source_component": index,
                "failure_class": "repeated_assignment_deficiency",
            }
            for index, name in enumerate(cache)
        ],
    }
    planned = plan_failure_conditioned_views(
        atlas=atlas,
        cache=cache,
        config=FailureAtlasConfig(
            maximum_planned_views=4,
            maximum_views_per_source=1,
            interpolation_alphas=(0.35, 0.65),
        ),
    )
    assert len(planned) == 2
    assert len({record["source_query"] for record in planned}) == 2
