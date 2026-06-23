import unittest

import torch


class LocalizationUtilityTest(unittest.TestCase):
    def _model_with_stats(self):
        from scene.gaussian_model import GaussianModel

        gaussians = GaussianModel(3)
        gaussians._xyz = torch.zeros(3, 3)
        gaussians._opacity = torch.zeros(3, 1)
        gaussians._loc_feature = torch.zeros(3, 1, 4)
        gaussians.init_localization_state(from_rgb_opacity=True)
        gaussians.loc_observation_count[:] = 10
        return gaussians

    def test_landmark_utility_does_not_use_localization_gradient_as_quality(self):
        gaussians = self._model_with_stats()
        gaussians.loc_grad_accum[:] = torch.tensor([[100.0], [0.0], [0.0]])
        gaussians.loc_grad_denom[:] = 1.0

        utility = gaussians.compute_localization_utility(min_observations=8)

        self.assertTrue(torch.allclose(utility, torch.zeros_like(utility)))

    def test_split_necessity_uses_gradient_ambiguity_repeatability_and_radius(self):
        gaussians = self._model_with_stats()
        gaussians.loc_grad_accum[:] = torch.tensor([[1.0], [8.0], [8.0]])
        gaussians.loc_grad_denom[:] = 1.0
        gaussians.loc_entropy_ema[:] = torch.tensor([0.2, 0.9, 0.9])
        gaussians.loc_repeatability_ema[:] = torch.tensor([0.9, 0.9, 0.1])
        gaussians.max_radii2D = torch.tensor([12.0, 12.0, 12.0])

        score = gaussians.compute_split_necessity(min_observations=8, min_radius=4.0)

        self.assertGreater(score[1].item(), score[0].item())
        self.assertEqual(score[2].item(), 0.0)

    def test_split_necessity_can_use_ambiguity_when_direct_teacher_has_no_viewspace_gradient(self):
        gaussians = self._model_with_stats()
        gaussians.loc_grad_accum.zero_()
        gaussians.loc_grad_denom.zero_()
        gaussians.loc_entropy_ema[:] = torch.tensor([0.1, 0.9, 0.8])
        gaussians.loc_repeatability_ema[:] = torch.tensor([0.9, 0.9, 0.1])
        gaussians.loc_positive_prob_ema[:] = torch.tensor([0.9, 0.9, 0.9])
        gaussians.max_radii2D = torch.tensor([12.0, 12.0, 12.0])

        score = gaussians.compute_split_necessity(min_observations=8, min_radius=4.0)

        self.assertGreater(score[1].item(), 0.0)
        self.assertEqual(score[2].item(), 0.0)

    def test_split_necessity_prefers_pose_effective_label_when_available(self):
        gaussians = self._model_with_stats()
        gaussians.loc_grad_accum.zero_()
        gaussians.loc_grad_denom.zero_()
        gaussians.loc_entropy_ema[:] = torch.tensor([0.8, 0.8, 0.8])
        gaussians.loc_repeatability_ema[:] = torch.tensor([0.8, 0.8, 0.8])
        gaussians.loc_positive_prob_ema[:] = torch.tensor([0.8, 0.8, 0.8])
        gaussians.loc_information_ema[:] = torch.tensor([0.1, 0.9, 0.0])
        gaussians.max_radii2D = torch.tensor([12.0, 12.0, 12.0])

        score = gaussians.compute_split_necessity(min_observations=8, min_radius=4.0)

        self.assertGreater(score[1].item(), score[0].item())
        self.assertEqual(score[2].item(), 0.0)

    def test_sparse_match_labels_update_real_utility_statistics(self):
        gaussians = self._model_with_stats()
        gaussians.loc_observation_count[:] = 0

        gaussians.add_sparse_match_label_stats(
            full_idx=torch.tensor([0, 1]),
            visible_count=torch.tensor([10, 10]),
            matched_count=torch.tensor([5, 5]),
            correct_count=torch.tensor([4, 1]),
            inlier_count=torch.tensor([3, 0]),
            ema_decay=0.0,
        )

        self.assertEqual(gaussians.loc_observation_count.tolist(), [10, 10, 0])
        self.assertAlmostEqual(gaussians.loc_repeatability_ema[0].item(), 0.4)
        self.assertAlmostEqual(gaussians.loc_positive_prob_ema[0].item(), 0.8)
        self.assertAlmostEqual(gaussians.loc_information_ema[0].item(), 0.6)
        self.assertAlmostEqual(gaussians.loc_outlier_ema[0].item(), 0.4)
        utility = gaussians.compute_landmark_reliability(min_observations=1)
        self.assertGreater(utility[0].item(), utility[1].item())

    def test_localization_stats_update_mask_only_updates_attributed_gaussians(self):
        gaussians = self._model_with_stats()
        gaussians.loc_observation_count[:] = 0

        gaussians.add_localization_stats(
            full_idx=torch.tensor([0, 1, 2]),
            episode_stats={
                "positive_prob": torch.tensor([0.9, 0.1, 0.8]),
                "prototype": torch.eye(3, 4),
                "update_mask": torch.tensor([True, False, True]),
            },
            ema_decay=0.0,
        )

        self.assertEqual(gaussians.loc_observation_count.tolist(), [1, 1, 1])
        self.assertAlmostEqual(gaussians.loc_positive_prob_ema[0].item(), 0.9)
        self.assertAlmostEqual(gaussians.loc_positive_prob_ema[1].item(), 0.0)
        self.assertAlmostEqual(gaussians.loc_positive_prob_ema[2].item(), 0.8)
        self.assertEqual(gaussians.loc_prototype_count.tolist(), [1, 0, 1])

    def test_localization_source_index_tracks_prune_and_split_children(self):
        gaussians = self._model_with_stats()

        gaussians._prune_localization_buffers(torch.tensor([False, True, True]))
        gaussians._cat_localization_buffers(torch.tensor([True, False]), repeat=2)

        self.assertEqual(gaussians.loc_source_index.tolist(), [1, 2, 1, 1])

    def test_localization_node_ids_are_stable_across_prune_and_unique_for_split_children(self):
        gaussians = self._model_with_stats()
        gaussians.loc_node_id[:] = torch.tensor([10, 11, 12])

        gaussians._prune_localization_buffers(torch.tensor([False, True, True]))
        gaussians._cat_localization_buffers(torch.tensor([True, False]), repeat=2)

        self.assertEqual(gaussians.loc_source_index.tolist(), [1, 2, 1, 1])
        self.assertEqual(gaussians.loc_node_id[:2].tolist(), [11, 12])
        self.assertEqual(gaussians.loc_parent_node_id.tolist(), [-1, -1, 11, 11])
        self.assertEqual(len(set(gaussians.loc_node_id.tolist())), 4)
        self.assertGreater(min(gaussians.loc_node_id[2:].tolist()), 12)

    def test_localization_source_xyz_tracks_original_parent_geometry(self):
        gaussians = self._model_with_stats()
        gaussians._xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        gaussians.init_localization_state(from_rgb_opacity=True)
        gaussians._xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )

        gaussians._prune_localization_buffers(torch.tensor([False, True, True]))
        gaussians._cat_localization_buffers(torch.tensor([True, False]), repeat=2)

        self.assertEqual(gaussians.loc_source_xyz.tolist(), [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_remap_sampled_indices_from_source_index_reports_missing_landmarks(self):
        from stdloc import remap_sampled_indices_from_source_index

        remapped, missing = remap_sampled_indices_from_source_index(
            sampled_idx=torch.tensor([7, 5, 8]),
            source_index=torch.tensor([5, 7, 5, 9]),
            return_missing=True,
        )

        self.assertEqual(remapped.tolist(), [1, 0])
        self.assertEqual(missing.tolist(), [8])

    def test_remap_sampled_indices_can_fill_missing_from_scored_current_points(self):
        from stdloc import remap_sampled_indices_from_source_index

        remapped, missing = remap_sampled_indices_from_source_index(
            sampled_idx=torch.tensor([7, 5, 8]),
            source_index=torch.tensor([5, 7, 5, 9]),
            return_missing=True,
            fill_missing=True,
            fill_scores=torch.tensor([0.1, 0.2, 0.9, 0.8]),
        )

        self.assertEqual(remapped.tolist(), [1, 0, 2])
        self.assertEqual(missing.tolist(), [8])

    def test_remap_sampled_indices_selects_best_duplicate_source_by_score(self):
        from stdloc import remap_sampled_indices_from_source_index

        remapped, missing = remap_sampled_indices_from_source_index(
            sampled_idx=torch.tensor([5, 7]),
            source_index=torch.tensor([5, 7, 5, 5]),
            return_missing=True,
            remap_scores=torch.tensor([0.1, 0.2, 0.9, 0.4]),
        )

        self.assertEqual(remapped.tolist(), [2, 1])
        self.assertEqual(missing.tolist(), [])

    def test_remap_sampled_indices_can_prefer_projection_consistent_duplicate(self):
        from stdloc import remap_sampled_indices_from_source_index

        remapped, missing = remap_sampled_indices_from_source_index(
            sampled_idx=torch.tensor([5]),
            source_index=torch.tensor([5, 5]),
            return_missing=True,
            remap_scores=torch.tensor([0.1, 100.0]),
            source_xyz=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            current_xyz=torch.tensor([[0.01, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            prefer_source_distance=True,
        )

        self.assertEqual(remapped.tolist(), [0])
        self.assertEqual(missing.tolist(), [])

    def test_remap_sampled_indices_can_drop_projection_inconsistent_duplicate(self):
        from stdloc import remap_sampled_indices_from_source_index

        remapped, missing = remap_sampled_indices_from_source_index(
            sampled_idx=torch.tensor([5]),
            source_index=torch.tensor([5]),
            return_missing=True,
            source_xyz=torch.tensor([[0.0, 0.0, 0.0]]),
            current_xyz=torch.tensor([[1.0, 0.0, 0.0]]),
            prefer_source_distance=True,
            max_source_distance=0.1,
        )

        self.assertEqual(remapped.tolist(), [])
        self.assertEqual(missing.tolist(), [5])

    def test_landmark_prior_from_meta_aligns_full_score_to_sampled_landmarks(self):
        from stdloc import landmark_prior_from_meta

        prior = landmark_prior_from_meta(
            {
                "full_score": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "landmark_indices": torch.tensor([2, 0]),
            },
            landmark_count=2,
        )

        self.assertTrue(torch.allclose(prior, torch.tensor([0.3, 0.1])))

    def test_landmark_prior_from_meta_rejects_misaligned_score(self):
        from stdloc import landmark_prior_from_meta

        with self.assertRaises(ValueError):
            landmark_prior_from_meta(
                {"score": torch.tensor([0.1, 0.2, 0.3])},
                landmark_count=2,
            )


if __name__ == "__main__":
    unittest.main()
