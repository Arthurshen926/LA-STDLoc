import numpy as np

from evaluation.evaluator import _gt_reprojection_diagnostics
from evaluation.metrics import summarize_pose_errors


def test_gt_reprojection_diagnostics_counts_invalid_depth_as_incorrect():
    intrinsic = np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )
    points_3d = np.asarray(
        [[0.0, 0.0, 1.0], [0.01, 0.0, 1.0], [0.0, 0.0, -1.0]]
    )
    points_2d = np.asarray([[50.0, 40.0], [54.0, 40.0], [50.0, 40.0]])
    result = _gt_reprojection_diagnostics(
        points_2d,
        points_3d,
        np.eye(4),
        intrinsic,
        np.asarray([0, 2]),
    )
    assert result == {
        "raw_count": 3,
        "raw_correct_2px": 1,
        "raw_correct_4px": 2,
        "inlier_count": 2,
        "inlier_correct_2px": 1,
        "inlier_correct_4px": 1,
    }


def test_pose_summary_reports_rotation_p90():
    summary = summarize_pose_errors([0.0, 1.0, 10.0], [1.0, 2.0, 3.0])
    assert summary["p90_ae_deg"] == np.percentile([0.0, 1.0, 10.0], 90)
