import numpy as np

from localization_training.dependency_pose_sampler import (
    _diverse_set,
    solve_dependency_absolute_pose,
)


def test_diverse_set_requires_geometry_coverage():
    indices = np.arange(4)
    assert _diverse_set(
        indices,
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 2, 3]),
    )
    assert not _diverse_set(
        indices,
        np.array([0, 0, 0, 1]),
        np.array([0, 1, 2, 3]),
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 2, 3]),
    )


def test_dependency_pose_recovers_synthetic_camera():
    rng = np.random.default_rng(4)
    points3d = rng.uniform([-2, -1, 4], [2, 1, 9], size=(80, 3))
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    points2d = points3d[:, :2] / points3d[:, 2:3]
    points2d = points2d * np.array([500.0, 500.0]) + np.array([320.0, 240.0])
    points2d += rng.normal(scale=0.1, size=points2d.shape)
    outliers = rng.choice(points2d.shape[0], size=16, replace=False)
    points2d[outliers] = rng.uniform([0, 0], [640, 480], size=(outliers.size, 2))
    pose, inliers, diagnostics = solve_dependency_absolute_pose(
        points2d,
        points3d,
        K,
        dependency_groups=np.arange(80) % 8,
        image_cells=np.arange(80) % 16,
        depth_bins=np.arange(80) % 4,
        surface_groups=np.arange(80) % 10,
        reprojection_error=2.0,
        max_iterations=500,
        min_iterations=100,
        seed=9,
    )
    assert inliers.size >= 60
    np.testing.assert_allclose(pose, np.eye(4), atol=2e-2)
    assert diagnostics["diverse_samples"] > 0
