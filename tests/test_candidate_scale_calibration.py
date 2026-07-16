import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_candidate_scales.py"
SPEC = importlib.util.spec_from_file_location("calibrate_candidate_scales", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CandidateScaleCalibrationTests(unittest.TestCase):
    def test_uses_train_validation_quantiles_and_bounds_outliers(self):
        records = [
            {
                "sparse_TE": value,
                "sparse": {"sparse_diag_inlier_gt_reproj_px_median": reproj},
            }
            for value, reproj in [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (1000.0, 50.0)]
        ]
        result = MODULE.calibrate(records, 0.02, 4.0)

        self.assertAlmostEqual(result["translation_scale_m"], 0.035, places=6)
        self.assertLessEqual(result["bias_clip"], 8.0)
        self.assertLessEqual(result["inlier_sigma_px"], 8.0)

    def test_falls_back_when_validation_has_no_finite_pose(self):
        result = MODULE.calibrate([], 0.03, 5.0)
        self.assertEqual(result["translation_scale_m"], 0.03)
        self.assertEqual(result["inlier_sigma_px"], 5.0)


if __name__ == "__main__":
    unittest.main()
