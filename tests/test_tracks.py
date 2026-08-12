import torch

from evidence.tracks import (
    _selected_track_observation_lookup,
    _track_observation_lookup,
    fuse_track_descriptors,
    robust_fuse_track_descriptors,
)
from evidence.camera_pair_policy import (
    candidate_camera_pairs,
    mapping_scene_points_from_depth_samples,
)
from evidence.triangulation import (
    attach_pair_triangulation_statistics,
    build_cycle_consistent_tracks,
    robust_triangulate_associations,
)


def _look_at_pose(center, target):
    center = torch.as_tensor(center, dtype=torch.float64)
    target = torch.as_tensor(target, dtype=torch.float64)
    forward = torch.nn.functional.normalize(target - center, dim=0)
    right = torch.nn.functional.normalize(
        torch.cross(
            forward,
            torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            dim=0,
        ),
        dim=0,
    )
    down = torch.cross(forward, right, dim=0)
    rotation = torch.stack((right, down, forward))
    pose = torch.eye(4, dtype=torch.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = -(rotation @ center)
    return pose


def _project(point, K, pose):
    camera = (pose @ torch.cat((point, point.new_ones(1))))[:3]
    pixel = K @ camera
    return pixel[:2] / pixel[2]


def test_nearest_camera_pair_policy_remains_exactly_compatible():
    poses = torch.stack(
        [
            _look_at_pose([0.04 * index, 0.0, 0.0], [0.0, 0.0, 3.0])
            for index in range(8)
        ]
    )
    implicit = candidate_camera_pairs(
        poses, neighbors=3, minimum_baseline_m=0.03
    )
    explicit = candidate_camera_pairs(
        poses,
        neighbors=3,
        minimum_baseline_m=0.03,
        policy="nearest",
    )
    assert explicit == implicit


def test_mapping_scene_point_sample_unprojects_known_depth_without_test_data():
    K = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3)
    pose = torch.eye(4, dtype=torch.float64).reshape(1, 4, 4)
    points = mapping_scene_points_from_depth_samples(
        [torch.tensor([[0.0, 0.0], [1.0, 0.0]])],
        [torch.tensor([2.0, 3.0])],
        K,
        pose,
        points_per_camera=2,
        maximum_points=8,
        voxel_size_m=0.001,
    )
    torch.testing.assert_close(
        points, torch.tensor([[0.0, 0.0, 2.0], [3.0, 0.0, 3.0]])
    )


def test_parallax_diverse_policy_preserves_exact_global_budget_and_overlap():
    centers = [[-0.35 + 0.1 * index, 0.0, 0.0] for index in range(8)]
    poses = torch.stack(
        [_look_at_pose(center, [0.0, 0.0, 3.0]) for center in centers]
    )
    K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(8, 1, 1)
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-0.6, 0.6, 9),
        torch.linspace(-0.4, 0.4, 7),
        indexing="ij",
    )
    scene_points = torch.stack(
        (grid_x.flatten(), grid_y.flatten(), torch.full((63,), 3.0)), dim=1
    )
    nearest = candidate_camera_pairs(
        poses, neighbors=2, minimum_baseline_m=0.0
    )
    revised = candidate_camera_pairs(
        poses,
        neighbors=2,
        minimum_baseline_m=0.0,
        policy="parallax_diverse",
        camera_K=K,
        image_hw=torch.tensor([[480, 640]]).repeat(8, 1),
        scene_points_xyz=scene_points,
        minimum_overlap_jaccard=0.5,
        minimum_joint_visibility_points=8,
        candidate_pool_per_camera=7,
    )
    center_tensor = torch.as_tensor(centers)

    def mean_baseline(pairs):
        pair = torch.as_tensor(pairs, dtype=torch.long)
        return torch.linalg.norm(
            center_tensor[pair[:, 0]] - center_tensor[pair[:, 1]], dim=1
        ).mean()

    assert len(revised) == len(nearest)
    assert revised != nearest
    assert mean_baseline(revised) > mean_baseline(nearest)
    assert revised == candidate_camera_pairs(
        poses,
        neighbors=2,
        minimum_baseline_m=0.0,
        policy="parallax_diverse",
        camera_K=K,
        image_hw=torch.tensor([[480, 640]]).repeat(8, 1),
        scene_points_xyz=scene_points,
        minimum_overlap_jaccard=0.5,
        minimum_joint_visibility_points=8,
        candidate_pool_per_camera=7,
    )


def test_track_pair_sidecar_records_exact_match_and_triangulation_funnel():
    points = torch.tensor(
        [[-0.2, 0.0, 3.0], [0.0, 0.1, 3.2], [0.2, -0.1, 2.8]],
        dtype=torch.float64,
    )
    centers = [[-0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]
    poses = torch.stack(
        [_look_at_pose(center, [0.0, 0.0, 3.0]) for center in centers]
    )
    K = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(3, 1, 1)
    keypoints = [
        torch.stack([_project(point, K[query], poses[query]) for point in points])
        for query in range(3)
    ]
    descriptors = [torch.eye(3) for _ in range(3)]
    tracks, diagnostics, sidecar = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=poses,
        detector_scores=[torch.ones(3) for _ in range(3)],
        pair_neighbors=2,
        minimum_baseline_m=0.0,
        minimum_similarity=0.5,
        minimum_margin=0.1,
        maximum_epipolar_error_px=0.01,
        minimum_track_views=3,
        require_cycle=True,
        allow_chain_tracks=True,
        return_pair_sidecar=True,
        pair_image_hw=torch.tensor([[480, 640]]).repeat(3, 1),
        pair_scene_points_xyz=points,
        device="cpu",
    )
    assert diagnostics["track_camera_pair_candidate_count"] == 3
    assert diagnostics["track_count"] == 3
    pair = sidecar["pair"]
    assert pair["raw_top1_reciprocal_count"].tolist() == [3, 3, 3]
    assert pair["raw_match_count"].tolist() == [3, 3, 3]
    assert pair["final_reciprocal_epipolar_count"].tolist() == [3, 3, 3]
    assert pair["accepted_match_count"].tolist() == [3, 3, 3]
    assert pair["cycle_supported_edge_count"].tolist() == [3, 3, 3]
    assert pair["conflict_rejected_edge_count"].sum() == 0
    assert pair["final_component_edge_count"].tolist() == [3, 3, 3]

    observation_uv = torch.stack(
        [
            keypoints[int(query)][int(keypoint)]
            for query, keypoint in zip(
                tracks["query_index"].tolist(),
                tracks["keypoint_index"].tolist(),
            )
        ]
    )
    geometry = robust_triangulate_associations(
        landmark_count=3,
        landmark_index=tracks["track_index"],
        query_index=tracks["query_index"],
        uv=observation_uv,
        confidence=tracks["confidence"],
        camera_K=K,
        pose_w2c=poses,
        query_bin=torch.arange(3),
        minimum_views=3,
        minimum_view_bins=2,
        minimum_parallax_deg=0.0,
        maximum_reprojection_px=0.1,
    )
    attach_pair_triangulation_statistics(sidecar, tracks, geometry, poses)
    assert pair["triangulated_track_count"].tolist() == [3, 3, 3]
    assert torch.isfinite(
        pair["actual_triangulation_parallax_median_deg"]
    ).all()


def test_robust_triangulation_recovers_track_and_rejects_duplicate_view():
    point = torch.tensor([0.2, -0.1, 4.0], dtype=torch.float64)
    centers = [[-1.0, 0.0, 0.0], [0.0, 0.2, 0.0], [1.0, 0.0, 0.0], [0.3, -0.3, 0.0]]
    poses = torch.stack([_look_at_pose(center, point) for center in centers])
    K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(len(centers), 1, 1)
    observed = [_project(point, K[index], poses[index]) for index in range(4)]
    uv = torch.stack(
        [
            observed[0] + torch.tensor([20.0, 15.0]),
            observed[0],
            observed[1],
            observed[2],
            observed[3],
        ]
    )
    result = robust_triangulate_associations(
        landmark_count=2,
        landmark_index=torch.zeros(5, dtype=torch.long),
        query_index=torch.tensor([0, 0, 1, 2, 3]),
        uv=uv,
        confidence=torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]),
        camera_K=K,
        pose_w2c=poses,
        query_bin=torch.tensor([0, 1, 2, 3]),
        minimum_views=3,
        minimum_view_bins=2,
        maximum_reprojection_px=1.0,
    )
    assert result["triangulation_high_confidence"].tolist() == [True, False]
    torch.testing.assert_close(result["triangulated_xyz"][0], point.float(), atol=1e-4, rtol=0)


def test_surface_support_corrects_only_the_weak_track_axis():
    point = torch.tensor([0.1, -0.05, 3.0], dtype=torch.float64)
    centers = [[-0.035, 0.0, 0.0], [-0.01, 0.0, 0.0], [0.015, 0.0, 0.0], [0.04, 0.0, 0.0]]
    poses = torch.stack([_look_at_pose(center, point) for center in centers])
    K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(len(centers), 1, 1)
    noise = torch.tensor(
        [[0.15, -0.10], [-0.12, 0.08], [0.10, 0.05], [-0.08, -0.12]],
        dtype=torch.float64,
    )
    uv = torch.stack([_project(point, K[i], poses[i]) for i in range(4)]) + noise
    rendered_depth = torch.stack(
        [((poses[i] @ torch.cat((point, point.new_ones(1))))[2]) for i in range(4)]
    )
    common = dict(
        landmark_count=1,
        landmark_index=torch.zeros(4, dtype=torch.long),
        query_index=torch.arange(4),
        uv=uv,
        confidence=torch.ones(4),
        camera_K=K,
        pose_w2c=poses,
        query_bin=torch.arange(4),
        rendered_depth=rendered_depth,
        minimum_views=3,
        minimum_view_bins=2,
        minimum_parallax_deg=0.0,
        maximum_reprojection_px=2.0,
        maximum_rendered_depth_residual_m=0.1,
    )
    baseline = robust_triangulate_associations(**common)
    supported = robust_triangulate_associations(
        **common,
        surface_support_enabled=True,
        surface_support_huber_m=0.02,
        surface_support_maximum_correction_m=0.15,
        surface_support_maximum_weak_information_ratio=0.5,
        surface_support_covariance_sigma_m=0.02,
    )
    assert supported["triangulation_surface_supported"].tolist() == [True]
    assert supported["triangulation_covariance_trace"][0] < baseline[
        "triangulation_covariance_trace"
    ][0]
    assert torch.linalg.norm(supported["triangulated_xyz"][0] - point.float()) < torch.linalg.norm(
        baseline["triangulated_xyz"][0] - point.float()
    )


def test_surface_support_rejects_cross_fitted_inconsistent_depth():
    point = torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64)
    centers = [[-0.04, 0.0, 0.0], [-0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.05, 0.0, 0.0]]
    poses = torch.stack([_look_at_pose(center, point) for center in centers])
    K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(4, 1, 1)
    uv = torch.stack([_project(point, K[i], poses[i]) for i in range(4)])
    rendered_depth = torch.tensor([2.7, 3.3, 2.7, 3.3], dtype=torch.float64)
    result = robust_triangulate_associations(
        landmark_count=1,
        landmark_index=torch.zeros(4, dtype=torch.long),
        query_index=torch.arange(4),
        uv=uv,
        confidence=torch.ones(4),
        camera_K=K,
        pose_w2c=poses,
        query_bin=torch.arange(4),
        rendered_depth=rendered_depth,
        minimum_views=3,
        minimum_view_bins=2,
        minimum_parallax_deg=0.0,
        maximum_reprojection_px=2.0,
        surface_support_enabled=True,
        surface_support_maximum_weak_information_ratio=0.5,
    )
    assert result["triangulation_surface_supported"].tolist() == [False]
    torch.testing.assert_close(result["triangulated_xyz"][0], point.float(), atol=1e-4, rtol=0)


def test_selected_track_descriptor_fusion_preserves_legacy_observation_order():
    payload = {
        "query_names": ["q0", "q1"],
        "query_bins": torch.tensor([0, 1]),
        "tracks": {
            "track_index": torch.tensor([2, 0, 2, 1, 0, 2]),
            "query_index": torch.tensor([0, 0, 1, 1, 1, 0]),
            "keypoint_index": torch.tensor([0, 1, 0, 1, 2, 2]),
            "confidence": torch.tensor([1.0, 0.8, 0.9, 0.7, 0.6, 0.5]),
        },
    }
    query_cache = {
        "queries": {
            "q0": {
                "native_descriptors": torch.tensor(
                    [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]
                )
            },
            "q1": {
                "native_descriptors": torch.tensor(
                    [[0.0, 1.0], [0.2, 0.8], [0.1, 0.9]]
                )
            },
        }
    }
    selected = torch.tensor([2, 0])
    lookup = _selected_track_observation_lookup(payload, selected)
    assert lookup[2].tolist() == [0, 2, 5]
    assert lookup[0].tolist() == [1, 4]

    legacy = _track_observation_lookup(payload)
    expected = []
    for track in selected.tolist():
        observations = torch.tensor(legacy[track])
        queries = payload["tracks"]["query_index"][observations]
        keypoints = payload["tracks"]["keypoint_index"][observations]
        descriptors = torch.stack(
            [
                query_cache["queries"][payload["query_names"][int(query)]][
                    "native_descriptors"
                ][int(keypoint)]
                for query, keypoint in zip(queries, keypoints)
            ]
        )
        expected.append(
            robust_fuse_track_descriptors(
                descriptors,
                payload["query_bins"][queries],
                payload["tracks"]["confidence"][observations],
                trim_fraction=0.2,
            )
        )
    actual = fuse_track_descriptors(
        payload=payload,
        query_cache=query_cache,
        track_indices=selected,
        trim_fraction=0.2,
    )
    torch.testing.assert_close(actual, torch.stack(expected), rtol=0, atol=0)
