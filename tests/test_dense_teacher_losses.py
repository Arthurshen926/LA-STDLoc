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

    def test_fine_reprojection_loss_excludes_invalid_local_cells(self):
        from localization_training.losses import fine_reprojection_loss

        feature_map = torch.zeros(2, 5, 5)
        rendered = torch.tensor([[1.0, 0.0]])
        target_uv = torch.tensor([[2.0, 2.0]])
        feature_map[:, 2, 2] = rendered[0]
        # An invalid adjacent cell has a stronger descriptor match.  Without
        # a per-window mask it incorrectly pulls the soft expectation away.
        feature_map[:, 2, 3] = torch.tensor([2.0, 0.0])
        valid_mask = torch.ones(5, 5, dtype=torch.bool)
        valid_mask[2, 3] = False

        loss, stats = fine_reprojection_loss(
            rendered,
            feature_map,
            target_uv,
            window_radius=1,
            temperature=0.01,
            valid_mask=valid_mask,
        )

        self.assertLess(loss.item(), 1e-4)
        self.assertTrue(torch.allclose(stats["pred_uv"], target_uv, atol=1e-3))

    def test_fine_reprojection_peak_term_rejects_symmetric_ambiguity(self):
        from localization_training.losses import fine_reprojection_loss

        target_uv = torch.tensor([[2.0, 2.0]])
        rendered = torch.tensor([[1.0, 0.0]])
        aligned = torch.zeros(2, 5, 5)
        aligned[:, 2, 2] = rendered[0]
        ambiguous = torch.zeros_like(aligned)
        ambiguous[:, 2, 1] = rendered[0]
        ambiguous[:, 2, 3] = rendered[0]

        aligned_loss, _ = fine_reprojection_loss(
            rendered,
            aligned,
            target_uv,
            window_radius=1,
            temperature=0.05,
            peak_weight=1.0,
            target_sigma=0.5,
        )
        ambiguous_loss, stats = fine_reprojection_loss(
            rendered,
            ambiguous,
            target_uv,
            window_radius=1,
            temperature=0.05,
            peak_weight=1.0,
            target_sigma=0.5,
        )

        self.assertGreater(ambiguous_loss.item(), aligned_loss.item())
        self.assertGreater(stats["target_nll"].item(), 0.0)

    def test_fine_reprojection_can_train_render_centered_candidate_window(self):
        from localization_training.losses import fine_reprojection_loss

        feature_map = torch.zeros(2, 7, 7)
        rendered = torch.tensor([[1.0, 0.0]])
        render_uv = torch.tensor([[2.0, 3.0]])
        target_uv = torch.tensor([[3.0, 3.0]])
        feature_map[:, 3, 3] = rendered[0]

        loss, stats = fine_reprojection_loss(
            rendered,
            feature_map,
            target_uv,
            window_radius=1,
            temperature=0.01,
            window_center_uv=render_uv,
        )

        self.assertLess(loss.item(), 1e-4)
        self.assertTrue(stats["target_in_window"].item())
        self.assertTrue(torch.allclose(stats["pred_uv"], target_uv, atol=1e-3))

    def test_fine_reprojection_ignores_targets_outside_render_candidate_window(self):
        from localization_training.losses import fine_reprojection_loss

        feature_map = torch.zeros(2, 9, 9)
        rendered = torch.tensor([[1.0, 0.0]])
        render_uv = torch.tensor([[2.0, 4.0]])
        target_uv = torch.tensor([[5.0, 4.0]])
        feature_map[:, 4, 5] = rendered[0]

        loss, stats = fine_reprojection_loss(
            rendered,
            feature_map,
            target_uv,
            window_radius=1,
            temperature=0.01,
            peak_weight=1.0,
            window_center_uv=render_uv,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertFalse(stats["target_in_window"].item())

    def test_target_correspondence_uses_feature_cell_center_convention(self):
        from localization_training.correspondence import build_target_correspondences

        pose = torch.eye(4)
        intrinsic = torch.tensor(
            [
                [80.0, 0.0, 8.0],
                [0.0, 80.0, 6.0],
                [0.0, 0.0, 1.0],
            ]
        )
        render_uv = torch.tensor([[3.0, 4.0], [7.0, 2.0]])
        depth = torch.tensor([4.0, 7.0])

        target = build_target_correspondences(
            render_uv,
            depth,
            intrinsic,
            pose,
            pose,
            pixel_center_offset=0.5,
        )

        self.assertTrue(target["valid"].all())
        self.assertTrue(torch.allclose(target["target_uv"], render_uv, atol=1e-6))
        expected_x = (render_uv[:, 0] + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0] * depth
        expected_y = (render_uv[:, 1] + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1] * depth
        self.assertTrue(torch.allclose(target["points_world"][:, 0], expected_x, atol=1e-6))
        self.assertTrue(torch.allclose(target["points_world"][:, 1], expected_y, atol=1e-6))

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

    def test_soft_pose_refinement_loss_backpropagates_to_local_predictions(self):
        from localization_training.dense_teacher import soft_pose_refinement_loss
        from localization_training.pose_refiner import project_points, se3_exp

        points = torch.tensor(
            [
                [-0.4, -0.3, 3.0],
                [0.5, -0.2, 3.5],
                [-0.3, 0.4, 4.0],
                [0.4, 0.5, 3.2],
                [0.1, -0.5, 4.5],
                [-0.5, 0.1, 3.8],
            ],
            dtype=torch.float32,
        )
        intrinsic = torch.tensor(
            [[120.0, 0.0, 40.0], [0.0, 120.0, 30.0], [0.0, 0.0, 1.0]]
        )
        pose_gt = torch.eye(4)
        pose_init = se3_exp(torch.tensor([0.08, 0.0, 0.0, 0.0, 0.0, 0.0])) @ pose_gt
        target_uv, valid = project_points(points, intrinsic, pose_gt)
        self.assertTrue(valid.all())
        predicted_uv = (target_uv - 0.5 + torch.tensor([0.25, -0.1])).detach()
        predicted_uv.requires_grad_(True)

        loss, diagnostics = soft_pose_refinement_loss(
            points,
            predicted_uv,
            intrinsic,
            pose_init,
            pose_gt,
            torch.ones(points.shape[0]),
            num_iterations=1,
            damping=1e-2,
            translation_scale_m=0.05,
            rotation_scale_deg=0.5,
            max_anchors=16,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(diagnostics["pose_refinement_used_anchor_count"], 4)
        loss.backward()
        self.assertTrue(torch.isfinite(predicted_uv.grad).all())
        self.assertGreater(float(predicted_uv.grad.abs().sum()), 0.0)

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
