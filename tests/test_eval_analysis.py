import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path


class EvalAnalysisTest(unittest.TestCase):
    def test_paired_summary_matches_queries_by_image_name_and_reports_recall_changes(self):
        from localization_training.eval_analysis import paired_sparse_summary

        baseline = [
            {"image_name": "b.png", "sparse_TE": 6.0, "sparse_AE": 0.2},
            {"image_name": "a.png", "sparse_TE": 3.0, "sparse_AE": 0.1},
            {"image_name": "c.png", "sparse_TE": 1.5, "sparse_AE": 0.1},
        ]
        la = [
            {"image_name": "a.png", "sparse_TE": 2.0, "sparse_AE": 0.05},
            {"image_name": "b.png", "sparse_TE": 4.0, "sparse_AE": 0.1},
            {"image_name": "c.png", "sparse_TE": 2.5, "sparse_AE": 0.2},
        ]

        summary = paired_sparse_summary(baseline, la, bootstrap_samples=64, seed=7)

        self.assertEqual(summary["query_count"], 3)
        self.assertAlmostEqual(summary["translation_delta_mean"], -2.0 / 3.0)
        self.assertAlmostEqual(summary["improved_translation_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["degraded_translation_fraction"], 1.0 / 3.0)
        self.assertEqual(summary["recall_5cm_gain_count"], 1)
        self.assertEqual(summary["recall_5cm_loss_count"], 0)
        self.assertIn("translation_delta_bootstrap_ci95", summary)

    def test_threshold_curve_reports_shared_recall_points(self):
        from localization_training.eval_analysis import threshold_curve

        results = [
            {"sparse_TE": 1.0, "sparse_AE": 0.5},
            {"sparse_TE": 5.0, "sparse_AE": 0.5},
            {"sparse_TE": 9.0, "sparse_AE": 6.0},
        ]

        curve = threshold_curve(results, thresholds=[2, 6])

        self.assertEqual(curve[0]["threshold_px"], 2.0)
        self.assertAlmostEqual(curve[0]["recall_2cm_2deg"], 1.0 / 3.0)
        self.assertAlmostEqual(curve[1]["recall_5cm_5deg"], 2.0 / 3.0)

    def test_solver_threshold_sweep_uses_distinct_runs_and_reports_auc(self):
        from localization_training.eval_analysis import solver_threshold_sweep_summary

        baseline = {
            4: [
                {"image_name": "a.png", "sparse_TE": 4.0, "sparse_AE": 0.2, "sparse": {"inliers": 10}},
                {"image_name": "b.png", "sparse_TE": 7.0, "sparse_AE": 0.3, "sparse": {"inliers": 8}},
            ],
            8: [
                {"image_name": "a.png", "sparse_TE": 3.0, "sparse_AE": 0.2, "sparse": {"inliers": 12}},
                {"image_name": "b.png", "sparse_TE": 5.0, "sparse_AE": 0.3, "sparse": {"inliers": 9}},
            ],
        }
        la = {
            4: [
                {"image_name": "a.png", "sparse_TE": 3.0, "sparse_AE": 0.1, "sparse": {"inliers": 14}},
                {"image_name": "b.png", "sparse_TE": 5.0, "sparse_AE": 0.2, "sparse": {"inliers": 11}},
            ],
            8: [
                {"image_name": "a.png", "sparse_TE": 1.5, "sparse_AE": 0.1, "sparse": {"inliers": 16}},
                {"image_name": "b.png", "sparse_TE": 4.0, "sparse_AE": 0.2, "sparse": {"inliers": 12}},
            ],
        }

        summary = solver_threshold_sweep_summary(baseline, la, bootstrap_samples=0)

        self.assertEqual(summary["thresholds"], [4.0, 8.0])
        self.assertEqual(len(summary["curve"]), 2)
        self.assertAlmostEqual(summary["curve"][0]["baseline"]["median_te"], 5.5)
        self.assertAlmostEqual(summary["curve"][0]["la"]["median_te"], 4.0)
        self.assertAlmostEqual(summary["curve"][0]["delta"]["median_te"], -1.5)
        self.assertEqual(summary["curve"][0]["paired"]["query_count"], 2)
        self.assertAlmostEqual(summary["auc"]["baseline_recall_5cm_5deg"], 0.75)
        self.assertAlmostEqual(summary["auc"]["la_recall_5cm_5deg"], 1.0)
        self.assertAlmostEqual(summary["auc_delta"]["recall_5cm_5deg"], 0.25)

    def test_solver_threshold_sweep_script_writes_summary_json(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_solver_threshold_sweep.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base4 = tmp / "base4.json"
            la4 = tmp / "la4.json"
            out = tmp / "summary.json"
            base4.write_text(
                json.dumps(
                    [
                        {"image_name": "a.png", "sparse_TE": 6.0, "sparse_AE": 0.2},
                        {"image_name": "b.png", "sparse_TE": 4.0, "sparse_AE": 0.2},
                    ]
                )
            )
            la4.write_text(
                json.dumps(
                    [
                        {"image_name": "a.png", "sparse_TE": 3.0, "sparse_AE": 0.2},
                        {"image_name": "b.png", "sparse_TE": 2.0, "sparse_AE": 0.2},
                    ]
                )
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--baseline_run",
                    f"4:{base4}",
                    "--la_run",
                    f"4:{la4}",
                    "--output",
                    str(out),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary = json.loads(out.read_text())
            self.assertEqual(summary["thresholds"], [4.0])
            self.assertAlmostEqual(summary["curve"][0]["delta"]["recall_5cm_5deg"], 0.5)


if __name__ == "__main__":
    unittest.main()
