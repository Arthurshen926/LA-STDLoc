import torch

from evidence.observation_provider import GaussianRenderObservationProvider
from map_learning.v6_feedback_evaluator import evaluate_query_local_feedback


def test_query_local_feedback_runs_one_top1_pose_with_geometry_loo() -> None:
    xyz = torch.tensor(
        [[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [-1.0, 1.0, 5.5], [1.0, 1.0, 6.0]]
    )
    K = torch.tensor([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0.0, 0.0, 1.0]])
    names = []
    queries = {}
    for query, center in enumerate(
        ((0.0, 0.0), (0.4, 0.0), (0.0, 0.4), (-0.4, 0.0), (0.0, -0.4))
    ):
        name = f"q{query}"
        names.append(name)
        pose = torch.eye(4)
        pose[0, 3] = -center[0]
        pose[1, 3] = -center[1]
        camera = xyz @ pose[:3, :3].T + pose[:3, 3]
        physical = (camera @ K.T)[:, :2] / camera[:, 2:]
        queries[name] = {
            "native_keypoints": physical - 0.5,
            "native_descriptors": torch.eye(4),
            "native_scores": torch.ones(4),
            "native_K": K,
            "pose_w2c": pose,
            "native_input_hw": torch.tensor([100, 100]),
            "native_alpha": torch.ones((100, 100)),
        }
    observations = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": queries},
        query_names=names,
    )
    offsets = torch.arange(0, 21, 5)
    state = {
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(4),
        "anchor_xyz": xyz,
        "anchor_features": torch.eye(4),
        "dependency_group_ids": torch.arange(4),
        "v6_mapping_query_names": names,
        "v6_mapping_query_bins": torch.arange(5),
        "projective_anchor_construction": {
            "final_xyz_source": "fixed_camera_robust_ray_triangulation"
        },
        "projective_anchor_observations": {
            "schema": "lafgs_projective_anchor_observations",
            "version": 1,
            "observation_offsets": offsets,
            "query_indices": torch.arange(5).repeat(4),
            "keypoint_indices": torch.arange(4).repeat_interleave(5),
        },
        "provenance": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        },
    }
    result = evaluate_query_local_feedback(
        state=state,
        observations=observations,
        source_map_sha256="a" * 64,
        query_cache_sha256="b" * 64,
        device=torch.device("cpu"),
        positive_radius_px=1.0,
        alpha_minimum=0.05,
        required_rank=4,
        ransac_reprojection_px=2.0,
        seed=2026,
    )
    assert result["summary"]["recall_5cm_5deg_percent"] == 100.0
    assert all(row["pose_solves"] == 1 for row in result["queries"])
    assert all(record["query_geometry_loo"] for record in result["feedback"]["records"])
    assert all(
        record["clean_inlier_pose_information"].shape[1:] == (6, 6)
        for record in result["feedback"]["records"]
    )
    assert all(
        record["pose_information_rank"] == 6 for record in result["feedback"]["records"]
    )
    assert result["version"] == 3
    assert result["feedback"]["version"] == 3
    assert result["feedback"]["identity_positive_count"] == 20
    assert result["feedback"]["geometry_compatible_ambiguous_count"] == 0
    assert result["feedback"]["pose_information_anchor_unique"] is True
    for record in result["feedback"]["records"]:
        assert record["identity_positive_count"] == 4
        assert torch.equal(
            record["winner_identity_correct_mask"], torch.ones(4, dtype=torch.bool)
        )
        assert (
            record["clean_inlier_pose_anchor_ids"].unique().numel()
            == record["clean_inlier_pose_anchor_ids"].numel()
        )
        assert record["descriptor_triplet_pose_weights"].shape == (
            record["descriptor_triplets"].shape[0],
        )
