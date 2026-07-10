import unittest
import json
import os
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

    def test_paired_sparse_stage_summary_reports_inlier_and_sequence_failures(self):
        from localization_training.eval_analysis import paired_sparse_stage_summary

        baseline = [
            {
                "image_name": "seq4/a.png",
                "sparse_TE": 4.0,
                "sparse_AE": 0.2,
                "sparse": {
                    "inliers": 120,
                    "matches": 300,
                    "detected_keypoints": 900,
                    "sparse_diag_inlier_2d_entropy_norm": 0.8,
                    "sparse_diag_inlier_gt_precision_4px": 0.9,
                },
            },
            {
                "image_name": "seq4/b.png",
                "sparse_TE": 8.0,
                "sparse_AE": 0.3,
                "sparse": {
                    "inliers": 100,
                    "matches": 280,
                    "detected_keypoints": 850,
                    "sparse_diag_inlier_2d_entropy_norm": 0.6,
                    "sparse_diag_inlier_gt_precision_4px": 0.7,
                },
            },
            {
                "image_name": "seq8/c.png",
                "sparse_TE": 12.0,
                "sparse_AE": 0.4,
                "sparse": {
                    "inliers": 90,
                    "matches": 250,
                    "detected_keypoints": 800,
                    "sparse_diag_inlier_2d_entropy_norm": 0.9,
                    "sparse_diag_inlier_gt_precision_4px": 0.8,
                },
            },
        ]
        la = [
            {
                "image_name": "seq4/a.png",
                "sparse_TE": 7.0,
                "sparse_AE": 0.4,
                "sparse": {
                    "inliers": 55,
                    "matches": 240,
                    "detected_keypoints": 700,
                    "sparse_diag_inlier_2d_entropy_norm": 0.5,
                    "sparse_diag_inlier_gt_precision_4px": 0.6,
                },
            },
            {
                "image_name": "seq4/b.png",
                "sparse_TE": 5.0,
                "sparse_AE": 0.2,
                "sparse": {
                    "inliers": 110,
                    "matches": 300,
                    "detected_keypoints": 870,
                    "sparse_diag_inlier_2d_entropy_norm": 0.7,
                    "sparse_diag_inlier_gt_precision_4px": 0.75,
                },
            },
            {
                "image_name": "seq8/c.png",
                "sparse_TE": 20.0,
                "sparse_AE": 0.8,
                "sparse": {
                    "inliers": 40,
                    "matches": 220,
                    "detected_keypoints": 760,
                    "sparse_diag_inlier_2d_entropy_norm": 0.4,
                    "sparse_diag_inlier_gt_precision_4px": 0.5,
                },
            },
        ]

        summary = paired_sparse_stage_summary(baseline, la, inlier_drop_threshold=50, top_k=2)

        self.assertEqual(summary["query_count"], 3)
        self.assertEqual(summary["recall_5cm_loss_count"], 1)
        self.assertEqual(summary["inlier_drop_count"], 2)
        self.assertEqual(summary["pose_degraded_and_inlier_drop_count"], 2)
        self.assertEqual(summary["sequence_groups"][0]["sequence"], "seq4")
        self.assertEqual(summary["top_te_degraded"][0]["image_name"], "seq8/c.png")
        self.assertNotIn("baseline_sparse_diag_inlier_2d_entropy_norm", summary["top_te_degraded"][0])
        self.assertEqual(summary["top_inlier_drop"][0]["image_name"], "seq4/a.png")
        entropy = summary["diagnostics"]["sparse_diag_inlier_2d_entropy_norm"]
        self.assertAlmostEqual(entropy["delta_mean"], (-0.3 + 0.1 - 0.5) / 3.0)
        self.assertAlmostEqual(entropy["pose_degraded_delta_mean"], (-0.3 - 0.5) / 2.0)
        self.assertAlmostEqual(entropy["pose_improved_delta_mean"], 0.1)
        self.assertAlmostEqual(
            summary["diagnostics"]["sparse_diag_inlier_gt_precision_4px"]["delta_mean"],
            (-0.3 + 0.05 - 0.3) / 3.0,
        )

    def test_sparse_stage_delta_script_writes_json_and_csv(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_sparse_stage_delta.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            baseline_path = tmp / "baseline.json"
            la_path = tmp / "la.json"
            summary_path = tmp / "summary.json"
            csv_path = tmp / "frames.csv"
            baseline_path.write_text(
                json.dumps(
                    [
                        {
                            "image_name": "seq/a.png",
                            "sparse_TE": 2.0,
                            "sparse_AE": 0.1,
                            "sparse": {"inliers": 20},
                        }
                    ]
                )
            )
            la_path.write_text(
                json.dumps(
                    [
                        {
                            "image_name": "seq/a.png",
                            "sparse_TE": 8.0,
                            "sparse_AE": 0.2,
                            "sparse": {"inliers": 5},
                        }
                    ]
                )
            )

            proc = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--baseline_results",
                    str(baseline_path),
                    "--candidate_results",
                    str(la_path),
                    "--output_json",
                    str(summary_path),
                    "--output_csv",
                    str(csv_path),
                    "--inlier_drop_threshold",
                    "10",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).resolve().parents[1],
                env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(summary_path.exists())
            self.assertTrue(csv_path.exists())
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["query_count"], 1)
            self.assertEqual(summary["inlier_drop_count"], 1)


if __name__ == "__main__":
    unittest.main()
