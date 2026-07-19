import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "select_lafgs_dense_refinement_gate.py"
SPEC = importlib.util.spec_from_file_location("dense_gate", MODULE_PATH)
dense_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dense_gate)


class DenseRefinementGateTest(unittest.TestCase):
    def _record(self, sparse_te, dense_te, inliers=32, ratio=0.1, delta_t=0.25, delta_r=2.0):
        return {
            "sparse_pose_w2c": [[1.0]],
            "raw_dense_pose_w2c": [[2.0]],
            "sparse_TE": sparse_te,
            "sparse_AE": 1.0,
            "raw_dense_TE": dense_te,
            "raw_dense_AE": 0.5,
            "raw_solver_success": True,
            "raw_dense_inliers": inliers,
            "raw_dense_inlier_ratio": ratio,
            "raw_pose_delta_translation_m": delta_t,
            "raw_pose_delta_rotation_deg": delta_r,
        }

    def test_apply_gate_keeps_sparse_pose_when_update_fails_guard(self):
        records = [self._record(20.0, 10.0, inliers=4)]
        applied = dense_gate.apply_gate(
            records,
            {
                "min_inliers": 16,
                "min_inlier_ratio": 0.05,
                "max_translation_delta_m": 0.5,
                "max_rotation_delta_deg": 5.0,
            },
        )
        self.assertFalse(applied[0]["gated_accepted"])
        self.assertEqual(applied[0]["gated_TE"], 20.0)

    def test_choose_gate_returns_fallback_without_validation_gain(self):
        records = [self._record(20.0, 30.0), self._record(40.0, 50.0)]
        selection = dense_gate.choose_gate(
            records,
            translation_gates=[0.5],
            rotation_gates=[5.0],
            min_inliers=[16],
            min_inlier_ratios=[0.05],
            mean_weight_cm=0.05,
            min_median_gain_cm=0.02,
            max_recall_2m_drop=0.01,
        )
        self.assertEqual(selection["selected"], "sparse_fallback")

    def test_choose_gate_selects_local_improving_candidate(self):
        records = [self._record(30.0, 10.0), self._record(50.0, 20.0)]
        selection = dense_gate.choose_gate(
            records,
            translation_gates=[0.5],
            rotation_gates=[5.0],
            min_inliers=[16],
            min_inlier_ratios=[0.05],
            mean_weight_cm=0.05,
            min_median_gain_cm=0.02,
            max_recall_2m_drop=0.01,
        )
        self.assertEqual(selection["selected"], "gated_dense")


if __name__ == "__main__":
    unittest.main()
