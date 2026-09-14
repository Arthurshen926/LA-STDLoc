import math

import numpy as np
import torch

from features.multiview_fusion import (
    grid_index_to_physical,
    physical_to_grid_index,
    sample_dense_descriptors_at_image_uv,
)
from localization.localizer import _local_inliers_to_query_rows
from localization.matcher import Top1Matches
from localization.pose_solver import camera_intrinsics, solve_absolute_pose, solve_pose


def test_pixel_center_round_trip():
    points = torch.tensor([[0.0, 0.0], [7.25, 9.5]])
    assert torch.equal(physical_to_grid_index(grid_index_to_physical(points)), points)


def test_filtered_solver_inliers_map_back_to_full_query_rows() -> None:
    matches = Top1Matches(
        keypoint_indices=torch.tensor([1, 3, 6, 8]),
        anchor_indices=torch.tensor([10, 11, 12, 13]),
        scores=torch.ones(4),
    )
    rows = _local_inliers_to_query_rows(matches, np.array([0, 2]))
    assert rows.tolist() == [1, 6]


def test_dense_sampling_uses_native_stride_eight_centers():
    dense = torch.zeros(2, 2, 2)
    dense[0, 0, 0] = 1
    dense[1, 1, 1] = 1
    sampled = sample_dense_descriptors_at_image_uv(
        dense, torch.tensor([[4.0, 4.0], [12.0, 12.0]]), (16, 16)
    )
    assert sampled[0, 0] > sampled[0, 1]
    assert sampled[1, 1] > sampled[1, 0]


def test_resize_to_pnp_synthetic_regression():
    rng = np.random.default_rng(4)
    xyz = rng.uniform(-1, 1, size=(64, 3))
    xyz[:, 2] += 5
    intrinsic = camera_intrinsics(math.radians(60), math.radians(45), 640, 480)
    uvw = (intrinsic @ xyz.T).T
    uv = uvw[:, :2] / uvw[:, 2:]
    # Sparse keypoints are grid indices; deployment adds the physical-center offset.
    # The solver receives +0.5 in SparseLocalizer; emulate it explicitly here.
    estimate = solve_absolute_pose(
        (uv - 0.5) + 0.5, xyz, intrinsic,
        reprojection_error_px=0.01, max_iterations=1000, min_iterations=100,
    )
    assert np.max(np.abs(estimate.pose_w2c - np.eye(4))) < 1e-4
    assert estimate.inliers.size == xyz.shape[0]


def test_offline_pose_compatibility_accepts_but_does_not_sample_by_scores():
    rng = np.random.default_rng(8)
    xyz = rng.uniform(-1, 1, size=(32, 3))
    xyz[:, 2] += 5
    intrinsic = camera_intrinsics(math.radians(60), math.radians(45), 640, 480)
    uvw = (intrinsic @ xyz.T).T
    uv = uvw[:, :2] / uvw[:, 2:]
    pose, inliers = solve_pose(
        uv,
        xyz,
        intrinsic,
        scores=np.linspace(1, 0, xyz.shape[0]),
        reprojection_error=0.01,
        max_iterations=1000,
        min_iterations=100,
    )
    assert np.max(np.abs(pose - np.eye(4))) < 1e-4
    assert inliers.size == xyz.shape[0]
