import unittest

import torch


class DescriptorDiagnosticsTest(unittest.TestCase):
    def test_alignment_metrics_report_positive_margin_top1_mnn_and_drift(self):
        from localization_training.descriptor_diagnostics import descriptor_alignment_metrics

        gaussian = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        query = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        baseline = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        metrics = descriptor_alignment_metrics(gaussian, query, baseline_features=baseline)

        self.assertAlmostEqual(metrics["positive_cosine_mean"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["top1_recall"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["mnn_precision"], 2.0 / 3.0)
        self.assertGreater(metrics["margin_mean"], 0.0)
        self.assertAlmostEqual(metrics["feature_drift_mean"], 1.0 / 3.0)

    def test_empty_alignment_metrics_are_zero(self):
        from localization_training.descriptor_diagnostics import descriptor_alignment_metrics

        metrics = descriptor_alignment_metrics(torch.empty(0, 2), torch.empty(0, 2))

        self.assertEqual(metrics["pair_count"], 0)
        self.assertEqual(metrics["top1_recall"], 0.0)
        self.assertEqual(metrics["mnn_precision"], 0.0)

    def test_full_bank_metrics_compare_query_observations_against_complete_landmark_bank(self):
        from localization_training.descriptor_diagnostics import full_bank_descriptor_metrics

        query = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        bank = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.2, 0.98],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        positive_bank_indices = torch.tensor([0, 3, 2])

        metrics = full_bank_descriptor_metrics(
            query,
            bank,
            positive_bank_indices,
            topk=(1, 2, 4),
        )

        self.assertEqual(metrics["query_count"], 3)
        self.assertAlmostEqual(metrics["full_bank_recall_at_1"], 1.0)
        self.assertAlmostEqual(metrics["full_bank_recall_at_2"], 1.0)
        self.assertAlmostEqual(metrics["full_bank_mnn_precision"], 1.0)
        self.assertGreater(metrics["full_bank_margin_mean"], 0.0)

    def test_landmark_value_summary_correlates_utility_with_inlier_rate(self):
        from localization_training.descriptor_diagnostics import summarize_landmark_value

        summary = summarize_landmark_value(
            visible_count=torch.tensor([10, 10, 10, 10]),
            matched_count=torch.tensor([10, 10, 10, 10]),
            correct_count=torch.tensor([0, 2, 7, 10]),
            inlier_count=torch.tensor([0, 1, 8, 9]),
            utility=torch.tensor([0.1, 0.2, 0.8, 0.9]),
        )

        self.assertEqual(summary["landmark_count"], 4)
        self.assertEqual(summary["visible_total"], 40)
        self.assertEqual(summary["matched_total"], 40)
        self.assertEqual(summary["correct_total"], 19)
        self.assertEqual(summary["inlier_total"], 18)
        self.assertAlmostEqual(summary["spearman_utility_correct_rate"], 1.0)
        self.assertAlmostEqual(summary["spearman_utility_inlier_rate"], 1.0)
        self.assertAlmostEqual(summary["top_quartile_correct_rate"], 1.0)
        self.assertAlmostEqual(summary["bottom_quartile_correct_rate"], 0.0)
        self.assertAlmostEqual(summary["top_quartile_inlier_rate"], 0.9)
        self.assertAlmostEqual(summary["bottom_quartile_inlier_rate"], 0.0)

    def test_landmark_value_summary_ignores_unmatched_for_rate_correlation(self):
        from localization_training.descriptor_diagnostics import summarize_landmark_value

        summary = summarize_landmark_value(
            visible_count=torch.tensor([5, 5, 5]),
            matched_count=torch.tensor([0, 4, 4]),
            correct_count=torch.tensor([0, 1, 3]),
            inlier_count=torch.tensor([0, 1, 3]),
            utility=torch.tensor([0.9, 0.1, 0.8]),
        )

        self.assertEqual(summary["matched_landmark_count"], 2)
        self.assertAlmostEqual(summary["inlier_rate_mean"], 0.5)
        self.assertAlmostEqual(summary["spearman_utility_inlier_rate"], 1.0)

    def test_calibrated_landmark_quality_learns_from_sparse_inlier_labels(self):
        from localization_training.descriptor_diagnostics import calibrate_landmark_quality

        result = calibrate_landmark_quality(
            features={
                "repeatability": torch.tensor([0.05, 0.2, 0.8, 0.95]),
                "outlier": torch.tensor([0.95, 0.8, 0.2, 0.05]),
            },
            positive_count=torch.tensor([0, 1, 8, 10]),
            trial_count=torch.tensor([10, 10, 10, 10]),
            steps=300,
            lr=0.2,
        )

        score = result["score"]
        self.assertEqual(result["feature_names"], ["outlier", "repeatability"])
        self.assertGreater(score[3].item(), score[2].item())
        self.assertGreater(score[2].item(), score[1].item())
        self.assertGreater(score[1].item(), score[0].item())
        self.assertGreater(result["spearman_calibrated_inlier_rate"], 0.99)
        self.assertLess(result["calibrated_brier"], 0.03)

    def test_calibrated_landmark_quality_reports_heldout_label_metrics(self):
        from localization_training.descriptor_diagnostics import calibrate_landmark_quality

        result = calibrate_landmark_quality(
            features={
                "repeatability": torch.tensor([0.05, 0.2, 0.8, 0.95, 0.1, 0.9]),
                "outlier": torch.tensor([0.95, 0.8, 0.2, 0.05, 0.9, 0.1]),
            },
            positive_count=torch.tensor([0, 1, 8, 10, 0, 0]),
            trial_count=torch.tensor([10, 10, 10, 10, 0, 0]),
            mask=torch.tensor([True, True, True, True, False, False]),
            eval_positive_count=torch.tensor([0, 0, 0, 0, 1, 9]),
            eval_trial_count=torch.tensor([0, 0, 0, 0, 10, 10]),
            eval_mask=torch.tensor([False, False, False, False, True, True]),
            target_name="correct",
            steps=300,
            lr=0.2,
        )

        self.assertEqual(result["calibration_target_name"], "correct")
        self.assertEqual(result["calibration_train_landmark_count"], 4)
        self.assertEqual(result["calibration_eval_landmark_count"], 2)
        self.assertTrue(result["calibration_heldout"])
        self.assertGreater(result["score"][5].item(), result["score"][4].item())
        self.assertAlmostEqual(result["spearman_calibrated_label_rate"], 1.0)
        self.assertAlmostEqual(result["spearman_calibrated_correct_rate"], 1.0)
        self.assertAlmostEqual(result["bottom_quartile_calibrated_label_rate"], 0.1)
        self.assertAlmostEqual(result["top_quartile_calibrated_label_rate"], 0.9)


if __name__ == "__main__":
    unittest.main()
