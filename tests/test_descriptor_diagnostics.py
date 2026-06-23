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
        self.assertAlmostEqual(summary["spearman_utility_inlier_rate"], 1.0)
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


if __name__ == "__main__":
    unittest.main()
