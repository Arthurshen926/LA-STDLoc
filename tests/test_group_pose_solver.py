from __future__ import annotations

import numpy as np

import localization.group_consensus as group_consensus
import localization.pose_solver as pose_solver
from localization.pose_solver import PoseEstimate, solve_group_diverse_absolute_pose


def _scene(translation_x: float = 0.0):
    intrinsic = np.asarray([[120.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]])
    generator = np.random.default_rng(4)
    points_3d = np.column_stack(
        (
            generator.uniform(-1.0, 1.0, 24),
            generator.uniform(-0.6, 0.6, 24),
            generator.uniform(3.0, 6.0, 24),
        )
    )
    pose = np.eye(4)
    pose[0, 3] = float(translation_x)
    camera = points_3d @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera @ intrinsic.T
    points_2d = homogeneous[:, :2] / homogeneous[:, 2:]
    return points_2d, points_3d, intrinsic, pose


def test_group_diverse_pose_selects_and_refines_a_better_candidate(monkeypatch) -> None:
    points_2d, points_3d, intrinsic, expected = _scene(translation_x=0.25)
    monkeypatch.setattr(
        pose_solver,
        "solve_absolute_pose",
        lambda *args, **kwargs: PoseEstimate(
            np.eye(4, dtype=np.float32),
            np.empty(0, dtype=np.int64),
            {"iterations": 1, "num_inliers": 0},
        ),
    )
    result = solve_group_diverse_absolute_pose(
        points_2d,
        points_3d,
        intrinsic,
        np.arange(points_2d.shape[0]),
        reprojection_error_px=2.0,
        group_hypothesis_samples=16,
        seed=2,
    )
    assert result.diagnostics["group_diverse_selected"] is True
    assert result.inliers.size == points_2d.shape[0]
    np.testing.assert_allclose(result.pose_w2c, expected, atol=1e-5, rtol=0)


def test_group_diverse_pose_preserves_a_better_poselib_result(monkeypatch) -> None:
    points_2d, points_3d, intrinsic, expected = _scene()
    baseline = PoseEstimate(
        expected.astype(np.float32),
        np.arange(points_2d.shape[0]),
        {"iterations": 17, "num_inliers": points_2d.shape[0]},
    )
    wrong = np.eye(4)
    wrong[0, 3] = 1.0
    monkeypatch.setattr(pose_solver, "solve_absolute_pose", lambda *a, **k: baseline)
    monkeypatch.setattr(
        group_consensus,
        "build_group_diverse_hypotheses",
        lambda *a, **k: wrong[None],
    )
    result = solve_group_diverse_absolute_pose(
        points_2d,
        points_3d,
        intrinsic,
        np.arange(points_2d.shape[0]),
        reprojection_error_px=2.0,
        group_hypothesis_samples=1,
    )
    assert result.diagnostics["group_diverse_selected"] is False
    assert np.array_equal(result.pose_w2c, baseline.pose_w2c)
    assert np.array_equal(result.inliers, baseline.inliers)
