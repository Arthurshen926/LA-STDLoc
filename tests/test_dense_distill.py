import unittest

import torch


class DenseDistillTest(unittest.TestCase):
    def test_responsibility_features_reconstruct_weighted_anchor_features(self):
        from localization_training.dense_distill import responsibility_weighted_features

        gaussian_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        contributor_ids = torch.tensor([[0, 1], [2, 1]])
        weights = torch.tensor([[0.75, 0.25], [0.5, 0.5]])

        reconstructed = responsibility_weighted_features(gaussian_features, contributor_ids, weights)

        expected = torch.nn.functional.normalize(
            torch.tensor([[0.75, 0.25], [0.5, 1.0]], dtype=torch.float32),
            dim=-1,
        )
        self.assertTrue(torch.allclose(reconstructed, expected, atol=1e-6))

    def test_responsibility_reconstruction_metrics_flag_bad_dense_attribution(self):
        from localization_training.dense_distill import responsibility_reconstruction_metrics

        rendered_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        gaussian_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        good_ids = torch.tensor([[0], [1]])
        bad_ids = torch.tensor([[1], [0]])
        weights = torch.ones(2, 1)

        good = responsibility_reconstruction_metrics(
            rendered_features,
            gaussian_features,
            good_ids,
            weights,
        )
        bad = responsibility_reconstruction_metrics(
            rendered_features,
            gaussian_features,
            bad_ids,
            weights,
        )

        self.assertAlmostEqual(good["mean_cosine"], 1.0, places=6)
        self.assertEqual(good["valid_anchor_count"], 2)
        self.assertLess(bad["mean_cosine"], 0.1)

    def test_dense_distribution_kl_moves_sparse_student_toward_teacher_gaussians(self):
        from localization_training.dense_distill import dense_to_sparse_kl, gaussian_teacher_distribution

        dense_probs = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
        contributor_ids = torch.tensor([[0, 1], [2, 1]])
        weights = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
        teacher = gaussian_teacher_distribution(dense_probs, contributor_ids, weights, bank_size=3)

        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        aligned_bank = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32)
        confused_bank = torch.tensor([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=torch.float32)

        aligned = dense_to_sparse_kl(query, aligned_bank, teacher, temperature=0.2)
        confused = dense_to_sparse_kl(query, confused_bank, teacher, temperature=0.2)

        self.assertTrue(torch.allclose(teacher, torch.tensor([[0.8, 0.15, 0.05]]), atol=1e-6))
        self.assertLess(aligned.item(), confused.item())


if __name__ == "__main__":
    unittest.main()
