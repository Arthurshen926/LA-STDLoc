import torch

from localization_training.geometry_teacher import (
    build_cycle_consistent_tracks,
    camera_center_bins,
    camera_pose_bins,
    fundamental_from_known_poses,
    local_geometric_match_support,
    reciprocal_epipolar_matches,
    robust_triangulate_associations,
    symmetric_epipolar_distance,
    transfer_triangulated_track_groups_to_landmarks,
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
    return pixel[:2] / pixel[2], camera[2]


def test_robust_triangulation_recovers_point_and_rejects_duplicate_view():
    point = torch.tensor([0.2, -0.1, 4.0], dtype=torch.float64)
    centers = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [1.0, 0.0, 0.0],
        [0.3, -0.3, 0.0],
    ]
    poses = torch.stack([_look_at_pose(center, point) for center in centers])
    K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(len(centers), 1, 1)
    observations = [_project(point, K[i], poses[i])[0] for i in range(4)]
    uv = torch.stack(
        [
            observations[0] + torch.tensor([20.0, 15.0]),
            observations[0],
            observations[1],
            observations[2],
            observations[3],
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
    assert torch.allclose(
        result["triangulated_xyz"][0], point.float(), atol=1e-4
    )
    assert result["triangulation_distinct_view_count"][0] == 4
    assert result["triangulation_reprojection_median_px"][0] < 1e-4


def test_camera_center_bins_are_deterministic_and_nonempty():
    target = torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64)
    poses = torch.stack(
        [_look_at_pose([float(index), 0.0, 0.0], target) for index in range(6)]
    )
    first = camera_center_bins(poses, 3)
    second = camera_center_bins(poses, 3)
    assert torch.equal(first, second)
    assert torch.unique(first).numel() == 3


def test_epipolar_matching_and_cycle_tracks_are_map_independent():
    point = torch.tensor([0.2, -0.1, 4.0], dtype=torch.float64)
    centers = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [1.0, 0.0, 0.0],
    ]
    poses = torch.stack([_look_at_pose(center, point) for center in centers])
    K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(3, 1, 1)
    keypoints = []
    descriptors = []
    for index in range(3):
        pixel, _ = _project(point, K[index], poses[index])
        keypoints.append(
            torch.stack(
                (
                    pixel,
                    pixel
                    + torch.tensor(
                        [35.0 + 17.0 * index, 15.0 - 11.0 * index]
                    ),
                )
            )
        )
        descriptor = torch.eye(2, 4)
        descriptors.append(descriptor)
    fundamental = fundamental_from_known_poses(
        K[0], poses[0], K[1], poses[1]
    )
    epipolar = symmetric_epipolar_distance(
        keypoints[0][:1], keypoints[1][:1], fundamental
    )
    assert epipolar.item() < 1e-6
    match_device = "cuda" if torch.cuda.is_available() else "cpu"
    source, target, _ = reciprocal_epipolar_matches(
        descriptors[0].to(match_device),
        descriptors[1].to(match_device),
        keypoints[0],
        keypoints[1],
        K[0],
        poses[0],
        K[1],
        poses[1],
        minimum_similarity=0.5,
        minimum_margin=0.1,
        maximum_epipolar_error_px=1.0,
    )
    assert source.tolist() == [0]
    assert target.tolist() == [0]
    tracks, diagnostics = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=poses,
        pair_neighbors=2,
        minimum_baseline_m=0.01,
        maximum_baseline_m=3.0,
        minimum_similarity=0.5,
        minimum_margin=0.1,
        maximum_epipolar_error_px=1.0,
        minimum_track_views=3,
        device="cpu",
    )
    assert diagnostics["track_count"] == 1
    assert tracks["query_index"].tolist() == [0, 1, 2]
    assert tracks["track_level"].tolist() == [2]


def test_graded_tracks_admit_query_unique_chain_without_three_cycle():
    point = torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64)
    centers = [
        [-1.5, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.5, 0.0, 0.0],
    ]
    poses = torch.stack([_look_at_pose(center, point) for center in centers])
    K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(4, 1, 1)
    keypoints = []
    descriptors = []
    for index in range(4):
        projected = _project(point, K[index], poses[index])[0]
        keypoints.append(
            torch.stack(
                (
                    projected,
                    projected
                    + torch.tensor(
                        [13.0 + 7.0 * index, 19.0 - 3.0 * index],
                        dtype=projected.dtype,
                    ),
                )
            )
        )
        main_descriptor = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        distractor_descriptor = torch.zeros(5)
        distractor_descriptor[index + 1] = 1.0
        descriptors.append(
            torch.stack((main_descriptor, distractor_descriptor))
        )
    strict_tracks, strict_diagnostics = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=poses,
        pair_neighbors=2,
        minimum_baseline_m=0.01,
        maximum_baseline_m=1.1,
        minimum_similarity=0.5,
        minimum_margin=-1.0,
        maximum_epipolar_error_px=1.0,
        minimum_track_views=3,
        device="cpu",
    )
    graded_tracks, graded_diagnostics = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=poses,
        pair_neighbors=2,
        minimum_baseline_m=0.01,
        maximum_baseline_m=1.1,
        minimum_similarity=0.5,
        minimum_margin=-1.0,
        maximum_epipolar_error_px=1.0,
        minimum_track_views=3,
        require_cycle=True,
        allow_chain_tracks=True,
        device="cpu",
    )

    assert strict_diagnostics["track_count"] == 0
    assert strict_tracks["track_index"].numel() == 0
    assert graded_diagnostics["track_count"] == 1
    assert graded_diagnostics["track_level_a_count"] == 0
    assert graded_diagnostics["track_level_b_count"] == 1
    assert graded_diagnostics["track_graded_chain_edge_count"] == 3
    assert graded_tracks["query_index"].tolist() == [0, 1, 2, 3]
    assert graded_tracks["track_level"].tolist() == [1]


def test_epipolar_first_topk_recovers_valid_non_global_descriptor_match():
    target = torch.tensor([0.0, 0.0, 4.0], dtype=torch.float64)
    points = torch.tensor(
        [[0.0, -0.5, 4.0], [0.0, 0.5, 4.0]], dtype=torch.float64
    )
    poses = torch.stack(
        [
            _look_at_pose([-0.5, 0.0, 0.0], target),
            _look_at_pose([0.5, 0.0, 0.0], target),
        ]
    )
    K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    uv_a = torch.stack([_project(point, K, poses[0])[0] for point in points])
    uv_b = torch.stack([_project(point, K, poses[1])[0] for point in points])
    descriptors_a = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    descriptors_b = torch.tensor(
        [[0.8, 0.6, 0.0], [1.0, 0.0, 0.0]]
    )

    global_source, _, _ = reciprocal_epipolar_matches(
        descriptors_a,
        descriptors_b,
        uv_a,
        uv_b,
        K,
        poses[0],
        K,
        poses[1],
        minimum_similarity=0.5,
        minimum_margin=0.1,
        maximum_epipolar_error_px=0.5,
        epipolar_candidate_topk=1,
    )
    gated_source, gated_target, _ = reciprocal_epipolar_matches(
        descriptors_a,
        descriptors_b,
        uv_a,
        uv_b,
        K,
        poses[0],
        K,
        poses[1],
        minimum_similarity=0.5,
        minimum_margin=0.1,
        maximum_epipolar_error_px=0.5,
        epipolar_candidate_topk=2,
    )

    assert 0 not in global_source.tolist()
    assert gated_source.tolist() == [0]
    assert gated_target.tolist() == [0]


def test_local_geometric_match_support_is_translation_invariant_and_rejects_outlier():
    uv_a = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 0.0],
            [0.0, 20.0],
            [20.0, 10.0],
            [10.0, 20.0],
        ]
    )
    angle = torch.deg2rad(torch.tensor(12.0))
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ]
    )
    uv_b = 1.1 * (uv_a @ rotation.T) + torch.tensor([300.0, -120.0])
    coherent = local_geometric_match_support(uv_a, uv_b, neighbors=5)
    translated = local_geometric_match_support(
        uv_a + torch.tensor([-700.0, 400.0]),
        uv_b + torch.tensor([900.0, -200.0]),
        neighbors=5,
    )
    torch.testing.assert_close(coherent, translated)
    assert bool((coherent > 0).all())

    corrupted = uv_b.clone()
    corrupted[-1] += torch.tensor([35.0, -25.0])
    outlier_support = local_geometric_match_support(
        uv_a, corrupted, neighbors=5
    )
    assert outlier_support[-1] < coherent[-1]


def test_soft_local_geometry_weights_tracks_without_removing_cycle_edges():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [20.0, 0.0],
            [0.0, 20.0],
            [20.0, 10.0],
            [10.0, 20.0],
        ]
    )
    descriptors = [torch.eye(8) for _ in range(3)]
    keypoints = [
        points,
        points + torch.tensor([2.0, -1.0]),
        points + torch.tensor([4.0, -2.0]),
    ]
    pose = torch.eye(4, dtype=torch.float64).repeat(3, 1, 1)
    pose[1, 0, 3] = -0.1
    pose[2, 0, 3] = -0.2
    K = torch.tensor(
        [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(3, 1, 1)
    tracks, diagnostics = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        camera_K=K,
        pose_w2c=pose,
        pair_neighbors=2,
        minimum_baseline_m=0.01,
        maximum_baseline_m=1.0,
        maximum_epipolar_error_px=5.0,
        minimum_similarity=0.5,
        minimum_margin=0.1,
        local_geometry_filter=True,
        local_geometry_mode="soft",
        local_geometry_minimum_matches=3,
        minimum_track_views=3,
        device="cpu",
    )
    assert diagnostics["track_lgcv_mode"] == "soft"
    assert diagnostics["track_lgcv_rejected_edge_count"] >= 0
    assert diagnostics["track_cycle_supported_edge_count"] == 24
    assert tracks["track_index"].unique().numel() == 8


def test_high_confidence_uses_covariance_and_depth_corroboration():
    point = torch.tensor([0.2, -0.1, 4.0], dtype=torch.float64)
    centers = [
        [-1.0, 0.0, 0.0],
        [0.0, 0.2, 0.0],
        [1.0, 0.0, 0.0],
    ]
    poses = torch.stack([_look_at_pose(center, point) for center in centers])
    K = torch.tensor(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(3, 1, 1)
    projected = [_project(point, K[i], poses[i]) for i in range(3)]
    uv = torch.stack([item[0] for item in projected])
    depth = torch.stack([item[1] for item in projected])
    accepted = robust_triangulate_associations(
        landmark_count=1,
        landmark_index=torch.zeros(3, dtype=torch.long),
        query_index=torch.arange(3),
        uv=uv,
        confidence=torch.ones(3),
        camera_K=K,
        pose_w2c=poses,
        query_bin=camera_pose_bins(poses, 3),
        rendered_depth=depth,
        minimum_views=3,
        minimum_view_bins=2,
        maximum_reprojection_px=1.0,
        maximum_covariance_trace_m2=0.01,
        maximum_rendered_depth_residual_m=0.01,
        minimum_rendered_depth_observations=3,
    )
    assert accepted["triangulation_high_confidence"].item()
    rejected = robust_triangulate_associations(
        landmark_count=1,
        landmark_index=torch.zeros(3, dtype=torch.long),
        query_index=torch.arange(3),
        uv=uv,
        confidence=torch.ones(3),
        camera_K=K,
        pose_w2c=poses,
        query_bin=camera_pose_bins(poses, 3),
        rendered_depth=depth + 1.0,
        minimum_views=3,
        minimum_view_bins=2,
        maximum_reprojection_px=1.0,
        maximum_covariance_trace_m2=0.01,
        maximum_rendered_depth_residual_m=0.01,
        minimum_rendered_depth_observations=3,
    )
    assert not rejected["triangulation_high_confidence"].item()


def test_track_group_transfer_preserves_original_track_identity():
    track_geometry = {
        "triangulated_xyz": torch.tensor(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        ),
        "triangulated": torch.tensor([True, True]),
        "triangulation_high_confidence": torch.tensor([True, True]),
    }
    geometry, assignment = transfer_triangulated_track_groups_to_landmarks(
        track_geometry,
        edge_track_index=torch.tensor([0, 0, 1]),
        edge_landmark_index=torch.tensor([2, 3, 2]),
        landmark_count=5,
        edge_assignment_cost=torch.tensor([0.2, 0.3, 0.1]),
    )
    assert geometry["track_assigned"].tolist() == [
        False,
        False,
        True,
        True,
        False,
    ]
    assert geometry["triangulated_xyz"][2].tolist() == [2.0, 0.0, 0.0]
    assert geometry["triangulated_xyz"][3].tolist() == [1.0, 0.0, 0.0]
    assert assignment["landmark_best_track_index"].tolist() == [
        -1,
        -1,
        1,
        0,
        -1,
    ]
    assert assignment["landmark_track_offsets"].tolist() == [0, 0, 0, 2, 3, 3]
    assert assignment["landmark_track_indices"].tolist() == [0, 1, 0]
    assert geometry["landmark_track_count"].tolist() == [0, 0, 2, 1, 0]
    assert torch.allclose(
        geometry["landmark_effective_track_support"][2],
        torch.tensor(1.9931),
        atol=1e-4,
    )
    assert torch.allclose(
        geometry["landmark_track_xyz_mean"][2],
        torch.tensor([1.5294, 0.0, 0.0]),
        atol=1e-4,
    )
    assert geometry["landmark_track_xyz_max_residual_m"][2] > 0.5
