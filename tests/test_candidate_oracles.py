import numpy as np


def _camera():
    return np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])


def test_candidate_oracle_projection_and_pnp_recover_pose():
    from scripts.eval_candidate_oracles import deterministic_pnp, project_points
    from utils.pose_utils import cal_pose_error

    points = np.array(
        [
            [-1.0, -1.0, 4.0],
            [1.0, -1.0, 4.5],
            [-1.0, 1.0, 5.0],
            [1.0, 1.0, 5.5],
            [-0.5, 0.2, 3.5],
            [0.3, -0.4, 6.0],
            [0.7, 0.6, 7.0],
            [-0.8, 0.5, 6.5],
        ]
    )
    K = _camera()
    pose_gt = np.eye(4)
    uv, _, valid = project_points(points, K, pose_gt)
    pose, success = deterministic_pnp(uv[valid], points[valid], K)
    ae, te = cal_pose_error(pose, pose_gt)

    assert success
    assert ae < 1e-3
    assert te < 1e-3


def test_candidate_oracle_assignment_and_translation_information_are_finite():
    from scripts.eval_candidate_oracles import (
        oracle_assign_detector_points,
        pose_information,
        project_points,
    )

    points = np.array(
        [[x, y, 5.0 + 0.2 * (x + y)] for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)]
    )
    K = _camera()
    pose = np.eye(4)
    uv, _, _ = project_points(points, K, pose)
    detector_points = np.floor(uv - 0.5)
    p2d, p3d, error = oracle_assign_detector_points(
        detector_points,
        points,
        K,
        pose,
        width=100,
        height=80,
        radius_px=2.0,
    )
    information = pose_information(p3d, K, pose)

    assert p2d.shape[0] == points.shape[0]
    assert np.max(error) < 2.0
    assert np.isfinite(information["translation_logdet"])
    assert information["translation_min_eig"] > 0


def test_candidate_oracle_balanced_subset_respects_budget():
    from scripts.eval_candidate_oracles import balanced_subset

    rng = np.random.default_rng(7)
    p2d = rng.uniform([0, 0], [99, 79], size=(100, 2))
    p3d = np.column_stack(
        [rng.normal(size=100), rng.normal(size=100), rng.uniform(3.0, 10.0, size=100)]
    )
    selected = balanced_subset(
        p2d,
        p3d,
        rng.uniform(size=100),
        _camera(),
        np.eye(4),
        width=100,
        height=80,
        max_count=32,
    )

    assert selected.shape == (32,)
    assert np.unique(selected).shape[0] == 32
