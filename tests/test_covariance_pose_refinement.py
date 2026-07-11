import numpy as np


def _project(points, K, pose):
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    camera = (pose @ points_h.T)[:3].T
    return np.column_stack(
        [
            K[0, 0] * camera[:, 0] / camera[:, 2] + K[0, 2],
            K[1, 1] * camera[:, 1] / camera[:, 2] + K[1, 2],
        ]
    )


def test_covariance_refinement_downweights_biased_measurements():
    from utils.pose_utils import cal_pose_error, covariance_weighted_pose_refinement

    rng = np.random.default_rng(3)
    points = np.column_stack(
        [
            rng.uniform(-2.0, 2.0, 40),
            rng.uniform(-1.5, 1.5, 40),
            rng.uniform(4.0, 9.0, 40),
        ]
    )
    K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]])
    gt = np.eye(4)
    observed = _project(points, K, gt)
    observed[20:] += np.array([4.0, -3.0])
    initial = np.eye(4)
    initial[:3, 3] = np.array([0.03, -0.02, 0.01])
    inliers = np.arange(points.shape[0])

    uniform_covariance = np.tile(np.eye(2)[None], (points.shape[0], 1, 1))
    heteroscedastic = uniform_covariance.copy()
    heteroscedastic[20:] *= 100.0
    uniform_pose, _ = covariance_weighted_pose_refinement(
        observed,
        points,
        K,
        initial,
        uniform_covariance,
        inliers,
        mahalanobis_threshold=100.0,
    )
    weighted_pose, _ = covariance_weighted_pose_refinement(
        observed,
        points,
        K,
        initial,
        heteroscedastic,
        inliers,
        mahalanobis_threshold=100.0,
    )
    _, uniform_te = cal_pose_error(uniform_pose, gt)
    _, weighted_te = cal_pose_error(weighted_pose, gt)

    assert weighted_te < uniform_te
    assert weighted_te < 1.0


def test_covariance_refinement_model_floor_preserves_ransac_inlier_set():
    from utils.pose_utils import covariance_weighted_pose_refinement

    points = np.array(
        [
            [-1.0, -0.5, 5.0],
            [1.0, -0.5, 5.5],
            [-1.0, 0.5, 6.0],
            [1.0, 0.5, 6.5],
            [0.0, 0.0, 7.0],
        ]
    )
    K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    observed = _project(points, K, pose) + np.array([2.0, -1.0])
    covariance = np.tile(np.eye(2)[None] * 0.01, (points.shape[0], 1, 1))
    inliers = np.arange(points.shape[0])

    _, selected = covariance_weighted_pose_refinement(
        observed,
        points,
        K,
        pose,
        covariance,
        inliers,
        iterations=0,
        mahalanobis_threshold=0.0,
        model_mismatch_floor_px=1.0,
    )

    np.testing.assert_array_equal(selected, inliers)


def test_progressive_pose_sampling_sorts_scores_and_restores_inlier_indices(monkeypatch):
    import utils.pose_utils as pose_utils

    captured = {}

    class FakePose:
        Rt = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)

    def fake_estimate(points2d, points3d, camera, ransac_options, bundle_options):
        captured["points2d"] = np.asarray(points2d).copy()
        captured["options"] = dict(ransac_options)
        return FakePose(), {
            "num_inliers": 2,
            "inliers": np.array([True, False, True, False, False]),
        }

    monkeypatch.setattr(
        pose_utils.poselib, "estimate_absolute_pose", fake_estimate
    )
    points2d = np.column_stack([np.arange(5), np.zeros(5)])
    points3d = np.column_stack([np.arange(5), np.zeros(5), np.ones(5) * 5])
    scores = np.array([0.2, 0.9, 0.5, 0.1, 0.3])
    K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 180.0], [0.0, 0.0, 1.0]])

    _, inliers = pose_utils.solve_pose(
        points2d,
        points3d,
        K,
        progressive_sampling=True,
        scores=scores,
        max_prosac_iterations=123,
    )

    np.testing.assert_array_equal(captured["points2d"][:, 0], [1, 2, 4, 0, 3])
    np.testing.assert_array_equal(inliers, [1, 4])
    assert captured["options"]["progressive_sampling"] is True
    assert captured["options"]["max_prosac_iterations"] == 123
