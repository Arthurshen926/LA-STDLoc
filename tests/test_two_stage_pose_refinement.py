import cv2
import numpy as np

from utils.pose_utils import cal_pose_error, two_stage_pose_refinement


def _project(points, K, pose):
    rvec = cv2.Rodrigues(pose[:3, :3])[0]
    projected, _ = cv2.projectPoints(
        points, rvec, pose[:3, 3], K, np.zeros((4, 1), dtype=np.float64)
    )
    return projected.reshape(-1, 2)


def test_two_stage_refinement_tightens_a_wide_ransac_seed():
    rng = np.random.default_rng(7)
    K = np.array([[640.0, 0.0, 320.0], [0.0, 640.0, 240.0], [0.0, 0.0, 1.0]])
    points = np.column_stack(
        [
            rng.uniform(-1.5, 1.5, size=48),
            rng.uniform(-1.0, 1.0, size=48),
            rng.uniform(4.0, 8.0, size=48),
        ]
    )
    gt = np.eye(4)
    measurements = _project(points, K, gt)
    measurements[:36] += rng.normal(scale=0.55, size=(36, 2))
    measurements[36:] += rng.uniform(16.0, 28.0, size=(12, 2))

    seed = np.eye(4)
    seed[:3, :3] = cv2.Rodrigues(np.array([0.004, -0.003, 0.002]))[0]
    seed[:3, 3] = np.array([0.006, -0.004, 0.003])
    refined, inliers, diagnostics = two_stage_pose_refinement(
        measurements,
        points,
        K,
        seed,
        np.arange(36),
        tight_reprojection_error=4.0,
        min_inliers=8,
        iterations=12,
        robust_delta=1.5,
    )

    _, seed_te = cal_pose_error(seed, gt)
    _, refined_te = cal_pose_error(refined, gt)
    assert diagnostics["two_stage_accepted"]
    assert diagnostics["two_stage_refined_cost"] <= diagnostics["two_stage_seed_cost"]
    assert inliers.size >= 8
    assert refined_te < seed_te


def test_two_stage_refinement_preserves_seed_without_tight_support():
    K = np.array([[400.0, 0.0, 200.0], [0.0, 400.0, 150.0], [0.0, 0.0, 1.0]])
    points = np.array(
        [[-1.0, -1.0, 5.0], [1.0, -1.0, 5.0], [-1.0, 1.0, 5.0], [1.0, 1.0, 5.0]]
    )
    seed = np.eye(4)
    measurements = _project(points, K, seed) + 20.0
    refined, inliers, diagnostics = two_stage_pose_refinement(
        measurements,
        points,
        K,
        seed,
        np.arange(4),
        tight_reprojection_error=1.0,
        min_inliers=4,
    )

    assert not diagnostics["two_stage_accepted"]
    assert diagnostics["two_stage_reason"] == "insufficient_tight_candidates"
    np.testing.assert_allclose(refined, seed)
    np.testing.assert_array_equal(inliers, np.arange(4))


def test_two_stage_refinement_uses_soft_matchability_without_dropping_pairs():
    rng = np.random.default_rng(17)
    K = np.array([[550.0, 0.0, 320.0], [0.0, 550.0, 240.0], [0.0, 0.0, 1.0]])
    points = np.column_stack(
        [
            rng.uniform(-1.2, 1.2, size=32),
            rng.uniform(-0.8, 0.8, size=32),
            rng.uniform(4.0, 7.0, size=32),
        ]
    )
    gt = np.eye(4)
    measurements = _project(points, K, gt)
    measurements[:24] += rng.normal(scale=0.35, size=(24, 2))
    measurements[24:] += rng.uniform(1.5, 2.1, size=(8, 2))
    weights = np.concatenate([np.ones(24), np.full(8, 0.2)])
    seed = np.eye(4)
    seed[:3, 3] = np.array([0.004, -0.003, 0.002])

    refined, inliers, diagnostics = two_stage_pose_refinement(
        measurements,
        points,
        K,
        seed,
        np.arange(points.shape[0]),
        tight_reprojection_error=4.0,
        min_inliers=8,
        iterations=10,
        matchability_weights=weights,
    )

    unweighted, _, _ = two_stage_pose_refinement(
        measurements,
        points,
        K,
        seed,
        np.arange(points.shape[0]),
        tight_reprojection_error=4.0,
        min_inliers=8,
        iterations=10,
    )
    assert diagnostics["two_stage_accepted"]
    assert diagnostics["two_stage_matchability_weighted"]
    assert diagnostics["two_stage_tight_matchability_p10"] < 1.0
    assert inliers.size >= 8
    assert np.isfinite(refined).all()
    assert not np.allclose(refined, unweighted)
