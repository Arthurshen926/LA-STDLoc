import unittest

import torch


class DenseTeacherLossTest(unittest.TestCase):
    def test_symmetric_descriptor_loss_prefers_identity_matches(self):
        from localization_training.losses import symmetric_descriptor_loss

        rendered = torch.eye(4)
        query = torch.eye(4)
        good = symmetric_descriptor_loss(rendered, query, temperature=0.05)
        bad = symmetric_descriptor_loss(rendered, torch.roll(query, shifts=1, dims=0), temperature=0.05)

        self.assertLess(good.item(), 1e-3)
        self.assertGreater(bad.item(), good.item() + 5.0)

    def test_fine_reprojection_loss_recovers_local_peak(self):
        from localization_training.losses import fine_reprojection_loss

        feature_map = torch.zeros(3, 7, 7)
        rendered = torch.tensor([[1.0, 0.0, 0.0]])
        target_uv = torch.tensor([[4.0, 3.0]])
        feature_map[:, 3, 4] = rendered[0]

        loss, stats = fine_reprojection_loss(
            rendered,
            feature_map,
            target_uv,
            window_radius=1,
            temperature=0.01,
        )

        self.assertLess(loss.item(), 1e-4)
        self.assertTrue(torch.allclose(stats["pred_uv"], target_uv, atol=1e-3))
        self.assertGreater(stats["positive_prob"].item(), 0.99)

    def test_responsibility_stats_assign_dense_observations_to_contributing_gaussians(self):
        from localization_training.dense_teacher import aggregate_dense_anchor_stats

        visible_idx = torch.tensor([0, 1, 2])
        contributor_ids = torch.tensor([[0, 1], [1, 2]])
        responsibility = torch.tensor([[0.75, 0.25], [0.25, 0.75]])
        query_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        fine_stats = {
            "positive_prob": torch.tensor([0.8, 0.2]),
            "entropy": torch.tensor([0.1, 0.7]),
            "reproj_error": torch.tensor([0.5, 2.0]),
        }
        margin = torch.tensor([0.6, -0.4])
        information = torch.tensor([1.5, 0.5])

        stats = aggregate_dense_anchor_stats(
            visible_idx,
            contributor_ids,
            responsibility,
            query_features,
            fine_stats,
            margin,
            information,
        )

        self.assertTrue(torch.equal(stats["update_mask"], torch.tensor([True, True, True])))
        self.assertTrue(torch.allclose(stats["positive_prob"], torch.tensor([0.8, 0.5, 0.2]), atol=1e-6))
        self.assertTrue(torch.allclose(stats["margin"], torch.tensor([0.6, 0.1, -0.4]), atol=1e-6))
        expected_proto = torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
            p=2,
            dim=-1,
        )
        self.assertTrue(torch.allclose(stats["prototype"], expected_proto, atol=1e-6))

    def test_dense_responsibility_kl_prefers_sparse_bank_matching_dense_teacher_distribution(self):
        from localization_training.dense_teacher import dense_responsibility_kl_loss

        query_features = torch.eye(2)
        rendered_features = torch.eye(2)
        contributor_ids = torch.tensor([[0], [1]])
        responsibility = torch.ones(2, 1)
        aligned_bank = torch.eye(2)
        confused_bank = torch.flip(torch.eye(2), dims=[0])

        aligned = dense_responsibility_kl_loss(
            query_features,
            rendered_features,
            aligned_bank,
            contributor_ids,
            responsibility,
            dense_temperature=0.05,
            sparse_temperature=0.05,
        )
        confused = dense_responsibility_kl_loss(
            query_features,
            rendered_features,
            confused_bank,
            contributor_ids,
            responsibility,
            dense_temperature=0.05,
            sparse_temperature=0.05,
        )

        self.assertLess(aligned.item(), 1e-3)
        self.assertGreater(confused.item(), aligned.item() + 5.0)

    def test_dense_miss_hit_rank_loss_updates_only_sparse_misses(self):
        from localization_training.dense_distill import dense_sparse_miss_hit_rank_loss

        teacher = torch.tensor(
            [
                [0.9, 0.1, 0.0],
                [0.0, 0.1, 0.9],
                [0.34, 0.33, 0.33],
            ],
            dtype=torch.float32,
        )
        query_features = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        bank_features = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        loss, diagnostics = dense_sparse_miss_hit_rank_loss(
            query_features,
            bank_features,
            teacher,
            temperature=1.0,
            teacher_confidence_threshold=0.5,
            miss_topk=1,
            margin=0.2,
            return_diagnostics=True,
        )

        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(diagnostics["dense_rank_sparse_hit_count"], 1)
        self.assertEqual(diagnostics["dense_rank_sparse_miss_count"], 1)
        self.assertEqual(diagnostics["dense_rank_low_confidence_count"], 1)
        self.assertEqual(diagnostics["dense_rank_eligible_anchor_count"], 1)

        aligned_loss, aligned_diagnostics = dense_sparse_miss_hit_rank_loss(
            query_features,
            bank_features,
            teacher,
            temperature=1.0,
            teacher_confidence_threshold=0.5,
            miss_topk=3,
            margin=0.2,
            return_diagnostics=True,
        )

        self.assertEqual(aligned_loss.item(), 0.0)
        self.assertEqual(aligned_diagnostics["dense_rank_eligible_anchor_count"], 0)

    def test_dense_teacher_output_carries_responsibility_diagnostics(self):
        from localization_training.dense_teacher import DenseTeacherOutput

        zero = torch.tensor(0.0)
        out = DenseTeacherOutput(
            zero,
            zero,
            zero,
            zero,
            {},
            {},
            0,
            diagnostics={"responsibility_reconstruction_mean_cosine": 0.9},
        )

        self.assertEqual(out.diagnostics["responsibility_reconstruction_mean_cosine"], 0.9)

    def test_pixel_contributor_sampling_prefers_depth_consistent_gaussian(self):
        from localization_training.dense_teacher import _sample_pixel_contributors

        render_pkg = {
            "loc_visible_idx": torch.tensor([10, 20]),
            "loc_viewspace_points": torch.tensor([[5.0, 5.0], [5.4, 5.0]]),
            "loc_radii": torch.tensor([10.0, 10.0]),
            "loc_depths": torch.tensor([10.0, 4.0]),
        }

        contributor_ids, weights = _sample_pixel_contributors(
            render_pkg,
            anchor_idx=torch.tensor([0]),
            render_uv=torch.tensor([[5.0, 5.0]]),
            image_height=16,
            image_width=16,
            max_contributors=2,
            render_depth=torch.tensor([4.0]),
            depth_consistency_weight=1.0,
        )

        self.assertEqual(contributor_ids[0, 0].item(), 20)
        self.assertGreater(weights[0, 0].item(), weights[0, 1].item())

    def test_pixel_contributor_sampling_uses_conic_mahalanobis_distance(self):
        from localization_training.dense_teacher import _sample_pixel_contributors

        render_pkg = {
            "loc_visible_idx": torch.tensor([10, 20]),
            "loc_viewspace_points": torch.tensor([[5.0, 5.0], [6.0, 5.0]]),
            "loc_radii": torch.tensor([20.0, 20.0]),
            "loc_conics": torch.tensor(
                [
                    [20.0, 0.0, 20.0],
                    [0.1, 0.0, 20.0],
                ]
            ),
        }

        contributor_ids, weights = _sample_pixel_contributors(
            render_pkg,
            anchor_idx=torch.tensor([0]),
            render_uv=torch.tensor([[5.2, 5.0]]),
            image_height=16,
            image_width=16,
            max_contributors=2,
        )

        self.assertEqual(contributor_ids[0, 0].item(), 20)
        self.assertGreater(weights[0, 0].item(), weights[0, 1].item())


if __name__ == "__main__":
    unittest.main()
