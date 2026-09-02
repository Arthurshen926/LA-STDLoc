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


def test_hypothesis_core_is_rescored_and_refined_on_all_matches(monkeypatch) -> None:
    points_2d, points_3d, intrinsic, expected = _scene()
    calls = []

    def solve(core_2d, core_3d, *_args, **_kwargs):
        calls.append(np.asarray(core_2d).copy())
        return PoseEstimate(
            expected.astype(np.float32),
            np.arange(core_2d.shape[0]),
            {"iterations": 23, "num_inliers": core_2d.shape[0]},
        )

    def refine(
        all_2d,
        all_3d,
        _intrinsic,
        _initial_pose,
        optimization_rows,
        **_kwargs,
    ):
        assert all_2d.shape[0] == points_2d.shape[0]
        assert optimization_rows.tolist() == list(range(points_2d.shape[0]))
        return PoseEstimate(
            expected.astype(np.float32),
            np.arange(all_2d.shape[0]),
            {"iterations": 2, "num_inliers": all_2d.shape[0]},
        )

    monkeypatch.setattr(pose_solver, "solve_absolute_pose", solve)
    monkeypatch.setattr(pose_solver, "refine_absolute_pose_from_initial", refine)
    quality = np.arange(points_2d.shape[0], dtype=np.float64)
    result = pose_solver.solve_absolute_pose_from_hypothesis_core(
        points_2d,
        points_3d,
        intrinsic,
        quality,
        hypothesis_core_size=8,
        reprojection_error_px=0.1,
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], points_2d[-8:][::-1])
    assert result.inliers.size == points_2d.shape[0]
    assert result.diagnostics["hypothesis_core_used"] is True
    assert result.diagnostics["hypothesis_core_fallback"] is False
    assert result.diagnostics["iterations"] == 23


def test_hypothesis_core_falls_back_when_it_cannot_solve(monkeypatch) -> None:
    points_2d, points_3d, intrinsic, expected = _scene()
    counts = []

    def solve(core_2d, *_args, **_kwargs):
        counts.append(core_2d.shape[0])
        if core_2d.shape[0] < points_2d.shape[0]:
            return PoseEstimate(
                np.eye(4, dtype=np.float32),
                np.empty(0, dtype=np.int64),
                {"iterations": 100, "num_inliers": 0},
            )
        return PoseEstimate(
            expected.astype(np.float32),
            np.arange(core_2d.shape[0]),
            {"iterations": 50, "num_inliers": core_2d.shape[0]},
        )

    monkeypatch.setattr(pose_solver, "solve_absolute_pose", solve)
    result = pose_solver.solve_absolute_pose_from_hypothesis_core(
        points_2d,
        points_3d,
        intrinsic,
        np.arange(points_2d.shape[0]),
        hypothesis_core_size=8,
    )
    assert counts == [8, points_2d.shape[0]]
    assert result.diagnostics["hypothesis_core_fallback"] is True
    assert result.inliers.size == points_2d.shape[0]
