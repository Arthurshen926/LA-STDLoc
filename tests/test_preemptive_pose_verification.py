import numpy as np
import poselib
import pytest

from localization_training.preemptive_pose_verification import (
    compiled_preemptive_backend_available,
    solve_preemptive_absolute_pose,
)
from utils.pose_utils import solve_pose


def _problem(seed=7):
    rng = np.random.default_rng(seed)
    points3d = rng.uniform(
        [-3.0, -2.0, 4.0], [3.0, 2.0, 12.0], size=(240, 3)
    )
    K = np.array(
        [[780.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]]
    )
    points2d = points3d[:, :2] / points3d[:, 2, None]
    points2d = points2d @ K[:2, :2].T + K[:2, 2]
    points2d += rng.normal(scale=0.25, size=points2d.shape)
    outliers = rng.choice(len(points2d), 72, replace=False)
    points2d[outliers] = rng.uniform(
        [0.0, 0.0], [640.0, 480.0], size=(len(outliers), 2)
    )
    priorities = np.zeros(len(points2d))
    priorities[outliers] = 1.0
    priorities += rng.normal(scale=0.01, size=len(priorities))
    return points2d, points3d, K, priorities


@pytest.mark.skipif(
    not compiled_preemptive_backend_available(),
    reason="compiled preemptive PoseLib backend has not been built",
)
def test_preemptive_pose_is_poselib_parity():
    points2d, points3d, K, priorities = _problem()
    options = {
        "max_iterations": 1000,
        "min_iterations": 200,
        "max_reproj_error": 3.0,
        "success_prob": 0.99999,
        "progressive_sampling": False,
        "max_prosac_iterations": 100000,
        "seed": 19,
    }
    camera = {
        "model": "PINHOLE",
        "width": 640,
        "height": 480,
        "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
    }
    expected_pose, expected_info = poselib.estimate_absolute_pose(
        points2d, points3d, camera, options, {"verbose": False}
    )
    pose, inliers, diagnostics = solve_preemptive_absolute_pose(
        points2d,
        points3d,
        K,
        verification_priorities=priorities,
        reprojection_error=3.0,
        confidence=0.99999,
        max_iterations=1000,
        min_iterations=200,
        seed=19,
    )
    np.testing.assert_allclose(pose[:3], expected_pose.Rt, atol=1e-11)
    np.testing.assert_array_equal(
        inliers, np.flatnonzero(expected_info["inliers"])
    )
    assert diagnostics["iterations"] == expected_info["iterations"]
    assert diagnostics["refinements"] == expected_info["refinements"]
    assert diagnostics["model_score"] == pytest.approx(
        expected_info["model_score"], abs=1e-14
    )
    assert diagnostics["pruned_score_calls"] > 0
    assert diagnostics["residual_evaluation_reduction"] > 0.0


@pytest.mark.skipif(
    not compiled_preemptive_backend_available(),
    reason="compiled preemptive PoseLib backend has not been built",
)
def test_solve_pose_preemptive_preserves_input_indices():
    points2d, points3d, K, priorities = _problem(seed=13)
    pose, inliers, diagnostics = solve_pose(
        points2d,
        points3d,
        K,
        solver="poselib_preemptive",
        reprojection_error=3.0,
        confidence=0.99999,
        max_iterations=500,
        min_iterations=100,
        ransac_seed=31,
        return_diagnostics=True,
        verification_priorities=priorities,
    )
    assert pose.shape == (4, 4)
    assert len(inliers) >= 150
    assert inliers.min() >= 0
    assert inliers.max() < len(points2d)
    assert diagnostics["ransac_preemptive_residual_reduction"] > 0.0


@pytest.mark.skipif(
    not compiled_preemptive_backend_available(),
    reason="compiled preemptive PoseLib backend has not been built",
)
def test_preemptive_pose_preserves_progressive_sampling_parity():
    points2d, points3d, K, priorities = _problem(seed=29)
    sampling_scores = -priorities
    common = {
        "reprojection_error": 3.0,
        "confidence": 0.99999,
        "max_iterations": 800,
        "min_iterations": 150,
        "scores": sampling_scores,
        "progressive_sampling": True,
        "max_prosac_iterations": 800,
        "ransac_seed": 43,
        "return_diagnostics": True,
    }
    expected_pose, expected_inliers, expected_diagnostics = solve_pose(
        points2d,
        points3d,
        K,
        solver="poselib",
        **common,
    )
    pose, inliers, diagnostics = solve_pose(
        points2d,
        points3d,
        K,
        solver="poselib_preemptive",
        verification_priorities=priorities,
        **common,
    )
    np.testing.assert_allclose(pose, expected_pose, atol=1e-11)
    np.testing.assert_array_equal(inliers, expected_inliers)
    assert (
        diagnostics["ransac_actual_hypotheses"]
        == expected_diagnostics["ransac_actual_hypotheses"]
    )
    assert (
        diagnostics["ransac_refinements"]
        == expected_diagnostics["ransac_refinements"]
    )
    assert diagnostics["ransac_preemptive_residual_reduction"] > 0.0
