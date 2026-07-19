import unittest

import numpy as np


from scripts.lafgs_dense_diagnostics import (
    candidate_displacement_diagnostics,
    gt_local_basin_diagnostics,
    gt_reprojection_diagnostics,
)


class DenseRefinementDiagnosticsTest(unittest.TestCase):
    def test_gt_reprojection_diagnostics_uses_pixel_center_convention(self):
        query_xy = np.array([[-0.5, -0.5], [0.5, -0.5]], dtype=np.float64)
        points3d = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64)
        K = np.eye(3, dtype=np.float64)
        pose = np.eye(4, dtype=np.float64)

        result = gt_reprojection_diagnostics(
            query_xy, points3d, K, pose, np.array([1], dtype=np.int64)
        )

        self.assertEqual(result["dense_all_count"], 2)
        self.assertEqual(result["dense_ransac_inlier_count"], 1)
        self.assertAlmostEqual(result["dense_all_gt_reproj_px_median"], 0.0)
        self.assertEqual(result["dense_all_gt_precision_2px"], 1.0)
        self.assertEqual(result["dense_all_gt_precision_0p25px"], 1.0)
        self.assertEqual(result["dense_ransac_inlier_gt_precision_2px"], 1.0)

    def test_gt_reprojection_diagnostics_reports_score_ranked_cleanliness(self):
        query_xy = np.array([[-0.5, -0.5], [10.5, -0.5]], dtype=np.float64)
        points3d = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64)
        result = gt_reprojection_diagnostics(
            query_xy,
            points3d,
            np.eye(3, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            np.array([0], dtype=np.int64),
            scores=np.array([0.9, 0.1], dtype=np.float64),
        )
        self.assertEqual(result["dense_score_top_05pct_count"], 1)
        self.assertEqual(result["dense_score_top_05pct_gt_precision_2px"], 1.0)
        self.assertAlmostEqual(result["dense_all_gt_residual_x_mean_px"], 5.0)

    def test_gt_local_basin_diagnostics_uses_feature_cell_centers(self):
        # Both projections are exactly at their source cell centers.  The
        # second point moves by 1.6 feature cells and is outside radius-1.
        rendered_xy = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        points3d = np.array([[0.5, 0.5, 1.0], [4.1, 0.5, 1.0]], dtype=np.float64)
        result = gt_local_basin_diagnostics(
            rendered_xy,
            points3d,
            np.eye(3, dtype=np.float64),
            np.eye(4, dtype=np.float64),
            radius_px=1,
        )
        self.assertEqual(result["seed_local_basin_count"], 2)
        self.assertAlmostEqual(result["seed_local_basin_square_coverage"], 0.5)
        self.assertAlmostEqual(result["seed_local_basin_offset_x_mean_px"], 0.8)

    def test_candidate_displacement_reports_attenuated_response(self):
        result = candidate_displacement_diagnostics(
            query_xy=np.array([[0.25, 0.0], [0.0, 0.25]], dtype=np.float64),
            rendered_xy=np.zeros((2, 2), dtype=np.float64),
            points3d=np.array([[1.0, 0.5, 1.0], [0.5, 1.0, 1.0]], dtype=np.float64),
            intrinsic=np.eye(3, dtype=np.float64),
            gt_pose_w2c=np.eye(4, dtype=np.float64),
        )
        self.assertAlmostEqual(result["candidate_shift_response_gain"], 0.5)
        self.assertAlmostEqual(result["candidate_shift_response_gain_x"], 0.5)
        self.assertAlmostEqual(result["candidate_shift_response_gain_y"], 0.5)
        self.assertEqual(result["candidate_shift_precision_0p25px"], 1.0)


if __name__ == "__main__":
    unittest.main()
