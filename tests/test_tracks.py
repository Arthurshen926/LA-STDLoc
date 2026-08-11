import torch

from evidence.tracks import (
    _selected_track_observation_lookup,
    _track_observation_lookup,
    fuse_track_descriptors,
    robust_fuse_track_descriptors,
)
from evidence.triangulation import robust_triangulate_associations


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
