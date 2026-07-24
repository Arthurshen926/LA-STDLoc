import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch


class LandmarkDistillTest(unittest.TestCase):
    def test_hard_score_core_keeps_top_eligible_scores(self):
        from localization_training.landmark_distill import hard_score_core

        score = torch.tensor([0.1, 0.9, 0.4, 0.8, 0.7])
        eligible = torch.tensor([True, True, False, True, True])

        selected = hard_score_core(score, 3, eligible)

        self.assertEqual(selected.tolist(), [1, 3, 4])

    def test_top_score_reservoir_excludes_low_quality_observed_candidates(self):
        from localization_training.landmark_distill import top_score_reservoir

        score = torch.tensor([0.9, 0.2, 0.8, 0.1, 0.7, 0.6, 0.5])
        observed = torch.tensor([True, True, True, True, True, True, False])

        reservoir = top_score_reservoir(
            score, budget=3, multiplier=1.5, eligible=observed
        )

        # ceil(3 * 1.5) = 5: the two lowest observed candidates are not
        # allowed back into a later coverage fill.
        self.assertEqual(reservoir.tolist(), [0, 1, 2, 4, 5])

    def test_wilson_lower_confidence_penalizes_single_view_certainty(self):
        from localization_training.landmark_distill import wilson_lower_confidence

        confidence = wilson_lower_confidence(
            torch.tensor([1.0, 2.0, 8.0]),
            torch.tensor([1.0, 2.0, 8.0]),
        )

        self.assertLess(float(confidence[0]), float(confidence[1]))
        self.assertLess(float(confidence[1]), float(confidence[2]))

    def test_quality_reservoir_constrains_final_distillation_membership(self):
        from train_lafgs_map import _distill_final_landmark_bank

        count = 8
        statistics = {
            "observation_count": torch.tensor([3, 3, 1, 1, 1, 1, 1, 0]),
            "effective_observation_count": torch.tensor([3, 3, 1, 1, 1, 1, 1, 0]),
            "matchability": torch.tensor([0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.10, 0.0]),
            "false_top1_rate": torch.tensor([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.90, 1.0]),
            "margin": torch.ones(count),
            "entropy": torch.zeros(count),
            "reprojection_error": torch.ones(count),
            "translation_fim": torch.ones(count),
            "mean_uv": torch.stack((torch.linspace(0.1, 0.8, count), torch.full((count,), 0.5)), dim=1),
            "mean_depth": torch.linspace(2.0, 9.0, count),
            "source_identity_rate": torch.ones(count),
        }
        args = SimpleNamespace(
            distill_reprojection_scale_px=2.0,
            distill_voxel_size=1.0,
            distill_min_observations=2,
            distill_matchability_threshold=0.5,
            distill_false_top1_max=0.5,
            distill_proposal_weight=1.0,
            distill_budget=4,
            distill_rank_pool_multiplier=1.5,
            distill_require_exact_budget=True,
            distill_allow_coverage_fill=True,
            distill_quality_reservoir_multiplier=1.5,
            distill_quality_reservoir_score="posterior_mean",
            distill_quality_reservoir_wilson_z=1.96,
            distill_hard_matchability_core_ratio=0.5,
            distill_matchability_preserve_ratio=0.5,
            distill_utility_preserve_ratio=0.0,
            distill_high_confidence=0.75,
            distill_high_confidence_ratio=0.0,
            distill_max_per_voxel=16,
            distill_grid_size=0,
            distill_max_per_grid=0,
            distill_depth_bins=0,
            distill_max_per_depth_bin=0,
            steps=0,
        )
        with TemporaryDirectory() as temp_dir:
            result = _distill_final_landmark_bank(
                Path(temp_dir),
                torch.arange(count),
                torch.eye(count, 4)[:, :4],
                torch.stack((torch.arange(count, dtype=torch.float32), torch.zeros(count), torch.ones(count)), dim=1),
                torch.zeros(count, 3),
                statistics,
                args,
                {},
                torch.ones(count),
                None,
            )
            state = torch.load(result["state_path"], map_location="cpu", weights_only=False)

        selected = set(state["landmark_indices"].tolist())
        reservoir = set(state["selection_meta"]["quality_reservoir_indices"].tolist())
        self.assertEqual(len(selected), 4)
        self.assertEqual(reservoir, {0, 1, 2, 3, 4, 5})
        self.assertTrue(selected.issubset(reservoir))
        self.assertNotIn(6, selected)
        self.assertTrue(bool(state["config"]["distillation"]["quality_reservoir_active"]))

    def test_protected_core_preserves_stable_high_precision_landmark(self):
        from train_lafgs_map import _distill_final_landmark_bank

        count = 6
        observations = torch.full((count,), 10.0)
        statistics = {
            "observation_count": observations,
            "effective_observation_count": observations,
            "correct_count": torch.tensor([9.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
            "matchability": torch.tensor([0.80, 0.99, 0.98, 0.97, 0.96, 0.95]),
            "false_top1_rate": torch.tensor([0.10, 0.20, 0.20, 0.20, 0.20, 0.20]),
            "margin": torch.ones(count),
            "entropy": torch.zeros(count),
            "reprojection_error": torch.ones(count),
            "translation_fim": torch.ones(count),
            "mean_uv": torch.stack(
                (torch.linspace(0.1, 0.9, count), torch.full((count,), 0.5)),
                dim=1,
            ),
            "mean_depth": torch.linspace(2.0, 7.0, count),
            "source_identity_rate": torch.ones(count),
            "cross_view_top1_identity_switch_rate": torch.tensor(
                [0.0, 0.8, 0.8, 0.8, 0.8, 0.8]
            ),
        }
        args = SimpleNamespace(
            distill_reprojection_scale_px=2.0,
            distill_voxel_size=1.0,
            distill_min_observations=2,
            distill_matchability_threshold=0.0,
            distill_false_top1_max=1.0,
            distill_proposal_weight=1.0,
            distill_budget=3,
            distill_rank_pool_multiplier=2.0,
            distill_require_exact_budget=True,
            distill_allow_coverage_fill=False,
            distill_quality_reservoir_multiplier=0.0,
            distill_quality_reservoir_score="wilson_lower",
            distill_quality_reservoir_wilson_z=1.96,
            distill_hard_matchability_core_ratio=0.0,
            distill_protected_core_ratio=0.34,
            distill_protected_min_correct=3,
            distill_protected_matchability=0.75,
            distill_protected_identity_switch_max=0.25,
            distill_matchability_preserve_ratio=0.0,
            distill_utility_preserve_ratio=0.0,
            distill_high_confidence=0.75,
            distill_high_confidence_ratio=0.0,
            distill_max_per_voxel=16,
            distill_grid_size=0,
            distill_max_per_grid=0,
            distill_depth_bins=0,
            distill_max_per_depth_bin=0,
            steps=0,
        )
        with TemporaryDirectory() as temp_dir:
            result = _distill_final_landmark_bank(
                Path(temp_dir),
                torch.arange(count),
                torch.eye(count, 4)[:, :4],
                torch.stack(
                    (
                        torch.arange(count, dtype=torch.float32),
                        torch.zeros(count),
                        torch.ones(count),
                    ),
                    dim=1,
                ),
                torch.zeros(count, 3),
                statistics,
                args,
                {},
                torch.ones(count),
                None,
            )
            state = torch.load(
                result["state_path"], map_location="cpu", weights_only=False
            )

        self.assertIn(0, state["landmark_indices"].tolist())
        self.assertEqual(result["protected_core_count"], 1)
        self.assertEqual(
            state["selection_meta"]["protected_core_indices"].tolist(), [0]
        )

    def test_quality_reservoir_rejects_subunit_multiplier(self):
        from train_lafgs_map import _validate_distillation_semantics

        args = SimpleNamespace(
            distill_budget=16,
            distill_quality_reservoir_multiplier=0.5,
            distill_quality_reservoir_score="posterior_mean",
            distill_quality_reservoir_wilson_z=1.96,
            distill_hard_matchability_core_ratio=0.5,
        )

        with self.assertRaisesRegex(ValueError, "zero or at least one"):
            _validate_distillation_semantics(args)

    def test_global_attractor_prior_reorders_distillation_reservoir(self):
        from train_lafgs_map import _distill_final_landmark_bank

        count = 5
        statistics = {
            "observation_count": torch.full((count,), 4.0),
            "effective_observation_count": torch.full((count,), 4.0),
            "matchability": torch.tensor([0.99, 0.90, 0.80, 0.70, 0.60]),
            "false_top1_rate": torch.zeros(count),
            "margin": torch.ones(count),
            "entropy": torch.zeros(count),
            "reprojection_error": torch.ones(count),
            "translation_fim": torch.ones(count),
            "mean_uv": torch.stack(
                (torch.linspace(0.1, 0.9, count), torch.full((count,), 0.5)),
                dim=1,
            ),
            "mean_depth": torch.linspace(2.0, 6.0, count),
            "source_identity_rate": torch.ones(count),
        }
        args = SimpleNamespace(
            distill_reprojection_scale_px=2.0,
            distill_voxel_size=1.0,
            distill_min_observations=2,
            distill_matchability_threshold=0.0,
            distill_false_top1_max=1.0,
            distill_proposal_weight=1.0,
            distill_global_attractor_weight=1.0,
            distill_budget=2,
            distill_rank_pool_multiplier=1.0,
            distill_require_exact_budget=True,
            distill_allow_coverage_fill=False,
            distill_quality_reservoir_multiplier=1.0,
            distill_quality_reservoir_score="posterior_mean",
            distill_quality_reservoir_wilson_z=1.96,
            distill_hard_matchability_core_ratio=1.0,
            distill_matchability_preserve_ratio=0.0,
            distill_utility_preserve_ratio=0.0,
            distill_high_confidence=0.75,
            distill_high_confidence_ratio=0.0,
            distill_max_per_voxel=16,
            distill_grid_size=0,
            distill_max_per_grid=0,
            distill_depth_bins=0,
            distill_max_per_depth_bin=0,
            steps=0,
        )
        global_statistics = {
            "score": torch.tensor([4.0, 0.0, 0.0, 0.0, 0.0]),
            "false_rate": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
            "incoming_count": torch.tensor([100.0, 1.0, 1.0, 1.0, 1.0]),
        }
        with TemporaryDirectory() as temp_dir:
            result = _distill_final_landmark_bank(
                Path(temp_dir),
                torch.arange(count),
                torch.eye(count, 4)[:, :4],
                torch.stack(
                    (
                        torch.arange(count, dtype=torch.float32),
                        torch.zeros(count),
                        torch.ones(count),
                    ),
                    dim=1,
                ),
                torch.zeros(count, 3),
                statistics,
                args,
                {},
                torch.ones(count),
                None,
                global_attractor_statistics=global_statistics,
            )
            state = torch.load(
                result["state_path"], map_location="cpu", weights_only=False
            )

        self.assertEqual(set(state["landmark_indices"].tolist()), {1, 2})
        self.assertTrue(
            bool(state["selection_meta"]["global_attractor_selection_active"])
        )
        self.assertGreater(
            float(state["config"]["distillation"]["global_attractor_selection"]["weight"]),
            0.0,
        )

    def test_ulf_random_knn_vote_sample_matches_generator_seed_and_local_vote(self):
        from localization_training.landmark_distill import ulf_random_knn_vote_sample

        points = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [4.0, 0.0, 0.0],
             [4.1, 0.0, 0.0], [9.0, 0.0, 0.0], [9.1, 0.0, 0.0]],
            dtype=torch.float32,
        )
        vote = torch.tensor([1.0, 9.0, 2.0, 8.0, 3.0, 7.0])
        seed = 2026
        sampled_a = ulf_random_knn_vote_sample(points, 3, vote, k=2, seed=seed)
        sampled_b = ulf_random_knn_vote_sample(points, 3, vote, k=2, seed=seed)

        seed_indices = np.random.default_rng(seed).choice(6, size=3, replace=False)
        expected = []
        selected = set()
        for point_index in seed_indices:
            distances = torch.linalg.vector_norm(points - points[point_index], dim=1)
            neighbours = torch.topk(distances, 2, largest=False).indices.tolist()
            for index in sorted(neighbours, key=lambda index: -float(vote[index])):
                if index not in selected:
                    selected.add(index)
                    expected.append(index)
                    break
        self.assertEqual(sampled_a.tolist(), sorted(expected))
        self.assertEqual(sampled_a.tolist(), sampled_b.tolist())

    def test_localization_aware_sample_prefers_high_utility(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 3)
        base_score = torch.zeros(10)
        utility = torch.arange(10, dtype=torch.float32)
        with mock.patch("torch.randperm", return_value=torch.tensor([0, 4, 8, 1, 2, 3, 5, 6, 7, 9])):
            sampled, meta = localization_aware_sample(
                xyz,
                base_score,
                utility,
                num=3,
                k=4,
                min_observations=torch.ones(10, dtype=torch.bool),
            )

        self.assertEqual(sampled.numel(), 3)
        self.assertEqual(set(sampled.tolist()), {3, 6, 9})
        self.assertIn("utility", meta)
        self.assertEqual(meta["indices"].shape, sampled.shape)

    def test_localization_aware_sample_keeps_spatial_neighborhood_coverage(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.1, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        base_score = torch.zeros(6)
        utility = torch.tensor([100.0, 90.0, 80.0, 10.0, 9.0, 11.0])

        with mock.patch("torch.randperm", return_value=torch.tensor([0, 3, 5, 1, 2, 4])):
            sampled, _ = localization_aware_sample(
                xyz,
                base_score,
                utility,
                num=3,
                k=2,
                min_observations=torch.ones(6, dtype=torch.bool),
            )

        self.assertEqual(set(sampled.tolist()), {0, 3, 5})

    def test_localization_aware_sample_can_reproduce_global_topk_ablation(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.1, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        base_score = torch.zeros(6)
        utility = torch.tensor([100.0, 90.0, 80.0, 10.0, 9.0, 11.0])

        sampled, _ = localization_aware_sample(
            xyz,
            base_score,
            utility,
            num=3,
            k=2,
            min_observations=torch.ones(6, dtype=torch.bool),
            spatial=False,
        )

        self.assertEqual(set(sampled.tolist()), {0, 1, 2})

    def test_localization_aware_sample_pnp_balance_limits_voxel_collapse(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        base_score = torch.zeros(5)
        utility = torch.tensor([100.0, 99.0, 98.0, 10.0, 9.0])

        sampled, meta = localization_aware_sample(
            xyz,
            base_score,
            utility,
            num=3,
            k=2,
            min_observations=torch.ones(5, dtype=torch.bool),
            spatial=False,
            pnp_balance=True,
            pnp_voxel_size=1.0,
            pnp_max_per_voxel=1,
            pnp_preserve_ratio=0.0,
        )

        self.assertEqual(set(sampled.tolist()), {0, 3, 4})
        self.assertTrue(bool(meta["pnp_balance"].item()))

    def test_localization_aware_sample_pnp_balance_preserves_top_utility_core(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        base_score = torch.zeros(5)
        utility = torch.tensor([100.0, 99.0, 98.0, 10.0, 9.0])

        sampled, meta = localization_aware_sample(
            xyz,
            base_score,
            utility,
            num=3,
            k=2,
            min_observations=torch.ones(5, dtype=torch.bool),
            spatial=False,
            pnp_balance=True,
            pnp_voxel_size=1.0,
            pnp_max_per_voxel=1,
            pnp_preserve_ratio=2.0 / 3.0,
        )

        self.assertEqual(set(sampled.tolist()), {0, 1, 3})
        self.assertAlmostEqual(float(meta["pnp_preserve_ratio"].item()), 2.0 / 3.0, places=5)

    def test_localization_aware_sample_pnp_balance_keeps_spatial_candidate_count(self):
        from localization_training.landmark_distill import localization_aware_sample

        xyz = torch.arange(5, dtype=torch.float32)[:, None].repeat(1, 3)
        base_score = torch.zeros(5)
        utility = torch.tensor([100.0, 90.0, 80.0, 10.0, 9.0])

        with mock.patch(
            "localization_training.landmark_distill.spatial_knn_score",
            return_value=torch.tensor([0, 3, 4]),
        ) as spatial_mock:
            sampled, _ = localization_aware_sample(
                xyz,
                base_score,
                utility,
                num=5,
                k=1,
                min_observations=torch.ones(5, dtype=torch.bool),
                spatial=True,
                pnp_balance=True,
                pnp_voxel_size=1.0,
                pnp_max_per_voxel=1,
                pnp_preserve_ratio=0.0,
            )

        self.assertEqual(sampled.numel(), 3)
        spatial_mock.assert_called_once()

    def test_coverage_ranked_fill_never_replaces_or_leaks_outside_primary_shortfall_pool(self):
        from localization_training.landmark_distill import coverage_ranked_fill

        xyz = torch.tensor(
            [
                [0.0, 0.0, 4.0],
                [1.0, 0.0, 4.0],
                [2.0, 0.0, 4.0],
                [3.0, 0.0, 4.0],
                [4.0, 0.0, 4.0],
                [5.0, 0.0, 4.0],
            ],
            dtype=torch.float32,
        )
        score = torch.tensor([0.9, 0.8, 0.7, 0.6, 100.0, 99.0])
        strict_selected = torch.tensor([0, 1], dtype=torch.long)
        weak_observed = torch.tensor([False, False, True, True, False, False])

        filled = coverage_ranked_fill(
            xyz,
            score,
            2,
            weak_observed,
            selected=strict_selected,
            voxel_size=0.01,
            max_per_voxel=1,
        )

        self.assertEqual(set(filled.tolist()), {2, 3})
        self.assertTrue(set(filled.tolist()).isdisjoint(set(strict_selected.tolist())))


if __name__ == "__main__":
    unittest.main()
