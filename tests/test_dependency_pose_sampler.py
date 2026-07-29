import numpy as np

from localization_training.dependency_pose_sampler import (
    _diverse_set,
    compiled_backend_available,
    solve_dependency_absolute_pose,
)


def test_diverse_set_requires_geometry_coverage():
    indices = np.arange(3)
    assert _diverse_set(
        indices,
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]]),
        1.0,
    )
    assert not _diverse_set(
        indices,
        np.array([0, 0, 0, 1]),
        np.array([0, 1, 2, 3]),
        np.array([0, 1, 2, 3]),
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.0]]),
        1.0,
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
        surface_groups=np.arange(80) % 10,
        reprojection_error=2.0,
        max_iterations=500,
        min_iterations=100,
        seed=9,
    )
    assert inliers.size >= 60
    np.testing.assert_allclose(pose, np.eye(4), atol=2e-2)
    assert diagnostics["diverse_samples"] > 0


def test_compiled_dependency_pose_recovers_synthetic_camera():
    if not compiled_backend_available():
        import pytest

        pytest.skip("compiled backend has not been built")
    rng = np.random.default_rng(7)
    points3d = rng.uniform([-2.0, -1.0, 4.0], [2.0, 1.0, 9.0], size=(80, 3))
    K = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    points2d = points3d[:, :2] / points3d[:, 2, None]
    points2d = points2d @ K[:2, :2].T + K[:2, 2]
    groups = np.arange(points3d.shape[0])
    pose, inliers, diagnostics = solve_dependency_absolute_pose(
        points2d,
        points3d,
        K,
        dependency_groups=groups,
        image_cells=groups % 16,
        surface_groups=groups % 7,
        sampling_scores=np.linspace(1.0, 0.0, points3d.shape[0]),
        guided_mixture=0.8,
        guided_rank_power=1.0,
        max_iterations=100,
        min_iterations=20,
        seed=9,
        backend="cpp",
    )
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-5)
    assert inliers.size == points3d.shape[0]
    assert diagnostics["backend"] == "cpp"


def test_dependency_pose_records_minimal_set_teacher_outcomes():
    rng = np.random.default_rng(7)
    points3d = rng.uniform([-2, -1, 4], [2, 1, 8], size=(40, 3))
    K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    points2d = points3d[:, :2] / points3d[:, 2:3]
    points2d = points2d * 400.0 + np.array([320.0, 240.0])
    _, _, diagnostics = solve_dependency_absolute_pose(
        points2d,
        points3d,
        K,
        dependency_groups=np.arange(40),
        image_cells=np.arange(40) % 16,
        surface_groups=np.arange(40) % 12,
        sampling_scores=np.linspace(0, 1, 40),
        guided_mixture=0.8,
        ground_truth_w2c=np.eye(4),
        minimal_set_record_limit=8,
        max_iterations=100,
        min_iterations=50,
        seed=3,
    )
    records = diagnostics["minimal_set_records"]
    assert 4 <= len(records) <= 8
    assert all(len(record["correspondence_indices"]) == 3 for record in records)
    assert any(record["correct_basin"] for record in records)
