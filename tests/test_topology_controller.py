import unittest
from contextlib import redirect_stdout
from io import StringIO

import torch
from torch import nn


class TopologyControllerTest(unittest.TestCase):
    def test_2dgs_fallback_score_does_not_require_viewspace_feature_gradients(self):
        from localization_training.topology_controller import (
            TopologyConfig,
            _localization_split_score,
        )

        class TwoDGSLike:
            get_xyz = torch.zeros(2, 3)
            loc_grad_accum = torch.zeros(2, 1)
            loc_grad_denom = torch.zeros(2, 1)
            loc_entropy_ema = torch.tensor([0.8, 0.2])
            loc_repeatability_ema = torch.tensor([0.9, 0.9])
            loc_positive_prob_ema = torch.tensor([0.0, 0.8])
            loc_reproj_error_ema = torch.tensor([0.0, 2.0])
            loc_information_ema = torch.tensor([0.0, 0.8])
            max_radii2D = torch.zeros(2)

        score = _localization_split_score(
            TwoDGSLike(),
            TopologyConfig(
                min_radius=0.0,
                pose_information_floor=0.05,
                residual_score_floor=0.05,
            ),
        )

        self.assertGreater(score[0].item(), 0.0)

    def test_split_child_features_use_observation_prototype_for_second_child(self):
        from scene.gaussian_model import split_localization_child_features

        parent_features = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ]
        )
        parent_prototypes = torch.tensor(
            [
                [0.0, 3.0],
                [4.0, 0.0],
            ]
        )
        prototype_counts = torch.tensor([2.0, 0.0])

        child_features = split_localization_child_features(
            parent_features,
            parent_prototypes,
            prototype_counts,
            repeat=2,
        )

        self.assertTrue(torch.allclose(child_features[0], parent_features[0]))
        self.assertTrue(torch.allclose(child_features[1], parent_features[1]))
        self.assertTrue(torch.allclose(child_features[2], torch.tensor([[0.0, 1.0]])))
        self.assertTrue(torch.allclose(child_features[3], parent_features[1]))

    def test_selects_splits_and_soft_prunes_low_utility_points(self):
        from localization_training.topology_controller import (
            TopologyConfig,
            apply_localization_soft_prune,
            select_localization_splits,
        )

        class FakeGaussians:
            def __init__(self):
                self._loc_opacity = nn.Parameter(torch.zeros(6, 1))
                self.loc_observation_count = torch.tensor([10, 10, 10, 2, 10, 10])
                self.loc_grad_accum = torch.tensor([[1.0], [8.0], [2.0], [9.0], [7.0], [0.1]])
                self.loc_grad_denom = torch.ones(6, 1)
                self.loc_entropy_ema = torch.tensor([0.1, 0.9, 0.2, 0.95, 0.8, 0.1])
                self.loc_repeatability_ema = torch.tensor([0.9, 0.95, 0.9, 0.95, 0.2, 0.9])
                self.loc_birth_iteration = torch.zeros(6, dtype=torch.long)
                self.last_topology_iteration = torch.zeros(6, dtype=torch.long)
                self.max_radii2D = torch.tensor([2.0, 12.0, 3.0, 13.0, 15.0, 1.0])
                self.utility = torch.tensor([0.5, 4.0, 0.2, 5.0, -2.0, -3.0])

            @property
            def get_xyz(self):
                return torch.zeros(6, 3)

            @property
            def get_opacity(self):
                return torch.tensor([[0.8], [0.8], [0.8], [0.8], [0.8], [0.8]])

            @property
            def get_loc_opacity(self):
                return torch.sigmoid(self._loc_opacity)

            def compute_localization_utility(self, min_observations=1):
                return self.utility

        gaussians = FakeGaussians()
        cfg = TopologyConfig(
            min_observations=8,
            split_quantile=0.7,
            ambiguity_quantile=0.7,
            min_repeatability=0.5,
            min_radius=4.0,
            growth_cap_per_event=0.5,
            cooldown_iterations=5,
        )

        split_mask = select_localization_splits(gaussians, cfg, iteration=10)
        self.assertTrue(split_mask[1].item())
        self.assertFalse(split_mask[3].item())
        self.assertFalse(split_mask[4].item())

        before = gaussians.get_loc_opacity.clone()
        prune_mask = apply_localization_soft_prune(
            gaussians,
            utility=gaussians.utility,
            threshold=-1.0,
            step=4.0,
        )
        after = gaussians.get_loc_opacity
        self.assertTrue(prune_mask[4].item())
        self.assertTrue(prune_mask[5].item())
        self.assertLess(after[5].item(), before[5].item())

    def test_ambiguity_quantile_filters_low_ambiguity_split_candidates(self):
        from localization_training.topology_controller import TopologyConfig, select_localization_splits

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.full((4,), 10)
                self.loc_grad_accum = torch.ones(4, 1)
                self.loc_grad_denom = torch.ones(4, 1)
                self.loc_entropy_ema = torch.tensor([0.1, 0.2, 0.9, 1.0])
                self.loc_repeatability_ema = torch.ones(4)
                self.last_topology_iteration = torch.zeros(4, dtype=torch.long)
                self.max_radii2D = torch.full((4,), 10.0)

            @property
            def get_xyz(self):
                return torch.zeros(4, 3)

            def compute_split_necessity(self, min_observations=1, min_radius=0.0, min_repeatability=0.0):
                return torch.tensor([100.0, 90.0, 10.0, 9.0])

        split = select_localization_splits(
            FakeGaussians(),
            TopologyConfig(
                min_observations=8,
                split_quantile=0.0,
                ambiguity_quantile=0.5,
                growth_cap_per_event=1.0,
                cooldown_iterations=5,
                min_repeatability=0.0,
                min_radius=1.0,
            ),
            iteration=10,
        )

        self.assertFalse(split[0].item())
        self.assertFalse(split[1].item())
        self.assertTrue(split[2].item())
        self.assertTrue(split[3].item())

    def test_split_scope_excludes_primitives_outside_final_landmark_bank(self):
        from localization_training.topology_controller import (
            TopologyConfig,
            select_localization_splits,
        )

        class FakeGaussians:
            loc_observation_count = torch.full((4,), 10)
            loc_entropy_ema = torch.ones(4)
            loc_repeatability_ema = torch.ones(4)
            last_topology_iteration = torch.zeros(4, dtype=torch.long)
            max_radii2D = torch.full((4,), 10.0)

            @property
            def get_xyz(self):
                return torch.zeros(4, 3)

            def compute_split_necessity(self, **_kwargs):
                return torch.tensor([4.0, 3.0, 2.0, 1.0])

        split = select_localization_splits(
            FakeGaussians(),
            TopologyConfig(
                min_observations=1,
                split_quantile=0.0,
                ambiguity_quantile=0.0,
                growth_cap_per_event=1.0,
                cooldown_iterations=0,
                min_repeatability=0.0,
                min_radius=0.0,
            ),
            iteration=10,
            candidate_scope_mask=torch.tensor([False, True, False, True]),
        )

        self.assertEqual(split.tolist(), [False, True, False, True])

    def test_topology_update_caps_split_count_to_remaining_total_budget(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.full((4,), 10)
                self.loc_grad_accum = torch.ones(4, 1)
                self.loc_grad_denom = torch.ones(4, 1)
                self.loc_entropy_ema = torch.ones(4)
                self.loc_repeatability_ema = torch.ones(4)
                self.last_topology_iteration = torch.zeros(4, dtype=torch.long)
                self.max_radii2D = torch.full((4,), 10.0)
                self._loc_feature = nn.Parameter(torch.zeros(4, 1, 2))
                self._loc_opacity = nn.Parameter(torch.zeros(4, 1))
                self.split_requests = []

            @property
            def get_xyz(self):
                return torch.zeros(self.loc_observation_count.shape[0], 3)

            def compute_localization_utility(self, min_observations=1):
                return torch.ones(self.get_xyz.shape[0])

            def compute_split_necessity(self, min_observations=1, min_radius=0.0, min_repeatability=0.0):
                return torch.tensor([4.0, 3.0, 2.0, 1.0])[: self.get_xyz.shape[0]]

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                split_count = int(selected_mask.sum().item())
                self.split_requests.append(split_count)
                keep = ~selected_mask.to(dtype=torch.bool)
                self.loc_observation_count = self.loc_observation_count[keep]
                self.loc_grad_accum = self.loc_grad_accum[keep]
                self.loc_grad_denom = self.loc_grad_denom[keep]
                self.loc_entropy_ema = self.loc_entropy_ema[keep]
                self.loc_repeatability_ema = self.loc_repeatability_ema[keep]
                self.last_topology_iteration = self.last_topology_iteration[keep]
                self.max_radii2D = self.max_radii2D[keep]
                self._loc_feature = nn.Parameter(self._loc_feature.detach()[keep])
                self._loc_opacity = nn.Parameter(self._loc_opacity.detach()[keep])
                self.loc_observation_count = torch.cat([self.loc_observation_count, torch.full((2 * split_count,), 10)])
                self.loc_grad_accum = torch.cat([self.loc_grad_accum, torch.ones(2 * split_count, 1)])
                self.loc_grad_denom = torch.cat([self.loc_grad_denom, torch.ones(2 * split_count, 1)])
                self.loc_entropy_ema = torch.cat([self.loc_entropy_ema, torch.ones(2 * split_count)])
                self.loc_repeatability_ema = torch.cat([self.loc_repeatability_ema, torch.ones(2 * split_count)])
                self.last_topology_iteration = torch.cat(
                    [self.last_topology_iteration, torch.zeros(2 * split_count, dtype=torch.long)]
                )
                self.max_radii2D = torch.cat([self.max_radii2D, torch.full((2 * split_count,), 10.0)])
                self._loc_feature = nn.Parameter(
                    torch.cat([self._loc_feature.detach(), torch.zeros(2 * split_count, 1, 2)])
                )
                self._loc_opacity = nn.Parameter(torch.cat([self._loc_opacity.detach(), torch.zeros(2 * split_count, 1)]))

        gaussians = FakeGaussians()
        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
                split_quantile=0.0,
                ambiguity_quantile=0.0,
                growth_cap_per_event=1.0,
                total_point_budget_ratio=1.25,
                cooldown_iterations=5,
                min_repeatability=0.0,
                min_radius=1.0,
            ),
            initial_points=4,
        )

        event = controller.update(gaussians, scene_extent=1.0, iteration=10)

        self.assertEqual(gaussians.split_requests, [1])
        self.assertEqual(event["requested_split_count"], 1)
        self.assertEqual(event["point_count_after"], 5)

    def test_risk_commit_rejects_split_proposal_before_mutation(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.full((4,), 10)
                self.loc_grad_accum = torch.ones(4, 1)
                self.loc_grad_denom = torch.ones(4, 1)
                self.loc_entropy_ema = torch.ones(4)
                self.loc_repeatability_ema = torch.ones(4)
                self.last_topology_iteration = torch.zeros(4, dtype=torch.long)
                self.max_radii2D = torch.full((4,), 10.0)
                self._loc_feature = nn.Parameter(torch.zeros(4, 1, 2))
                self._loc_opacity = nn.Parameter(torch.zeros(4, 1))
                self.split_requests = []

            @property
            def get_xyz(self):
                return torch.zeros(self.loc_observation_count.shape[0], 3)

            def compute_localization_utility(self, min_observations=1):
                return torch.ones(self.get_xyz.shape[0])

            def compute_split_necessity(self, min_observations=1, min_radius=0.0, min_repeatability=0.0):
                return torch.arange(self.get_xyz.shape[0], 0, -1, dtype=torch.float32)

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                self.split_requests.append(int(selected_mask.sum().item()))
                raise AssertionError("risk-rejected proposal should not mutate topology")

        decisions = []

        def reject_risk(proposal, gaussians):
            decisions.append(int(proposal.split_mask.sum().item()))
            return {"accepted": False, "reason": "holdout risk increased", "delta_risk": 0.25}

        gaussians = FakeGaussians()
        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
                split_quantile=0.0,
                ambiguity_quantile=0.0,
                growth_cap_per_event=1.0,
                total_point_budget_ratio=2.0,
                cooldown_iterations=0,
                min_repeatability=0.0,
                min_radius=1.0,
                risk_commit_policy="callback",
            ),
            initial_points=4,
            risk_evaluator=reject_risk,
        )

        event = controller.update(gaussians, scene_extent=1.0, iteration=10)

        self.assertEqual(decisions, [4])
        self.assertEqual(gaussians.split_requests, [])
        self.assertEqual(event["requested_split_count"], 4)
        self.assertEqual(event["actual_children_added"], 0)
        self.assertEqual(event["point_count_after"], 4)
        self.assertFalse(event["risk_commit"]["accepted"])
        self.assertEqual(event["risk_commit"]["reason"], "holdout risk increased")
        self.assertEqual(controller.mutation_event_count, 0)

    def test_risk_commit_logs_numeric_risk_details(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.full((2,), 10)
                self.loc_grad_accum = torch.ones(2, 1)
                self.loc_grad_denom = torch.ones(2, 1)
                self.loc_entropy_ema = torch.ones(2)
                self.loc_repeatability_ema = torch.ones(2)
                self.last_topology_iteration = torch.zeros(2, dtype=torch.long)
                self.max_radii2D = torch.full((2,), 10.0)
                self._loc_opacity = nn.Parameter(torch.zeros(2, 1))

            @property
            def get_xyz(self):
                return torch.zeros(2, 3)

            def compute_localization_utility(self, min_observations=1):
                return torch.ones(2)

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                raise AssertionError("rejected risk proposal must not mutate")

        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
                split_quantile=0.0,
                ambiguity_quantile=0.0,
                growth_cap_per_event=1.0,
                total_point_budget_ratio=2.0,
                cooldown_iterations=0,
                min_repeatability=0.0,
                min_radius=1.0,
                risk_commit_policy="callback",
            ),
            initial_points=2,
            risk_evaluator=lambda proposal, gaussians: {
                "accepted": False,
                "reason": "heldout_descriptor_not_decreased",
                "baseline_risk": 1.25,
                "trial_risk": 1.50,
                "delta_risk": 0.25,
                "epsilon": 0.01,
            },
        )

        buffer = StringIO()
        with redirect_stdout(buffer):
            controller.update(FakeGaussians(), scene_extent=1.0, iteration=10)

        text = buffer.getvalue()
        self.assertIn("risk_baseline=1.250000", text)
        self.assertIn("risk_trial=1.500000", text)
        self.assertIn("risk_delta=0.250000", text)
        self.assertIn("risk_epsilon=0.010000", text)

    def test_risk_commit_accepts_split_proposal_before_mutation(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.full((3,), 10)
                self.loc_grad_accum = torch.ones(3, 1)
                self.loc_grad_denom = torch.ones(3, 1)
                self.loc_entropy_ema = torch.ones(3)
                self.loc_repeatability_ema = torch.ones(3)
                self.last_topology_iteration = torch.zeros(3, dtype=torch.long)
                self.max_radii2D = torch.full((3,), 10.0)
                self._loc_feature = nn.Parameter(torch.zeros(3, 1, 2))
                self._loc_opacity = nn.Parameter(torch.zeros(3, 1))
                self.split_requests = []

            @property
            def get_xyz(self):
                return torch.zeros(self.loc_observation_count.shape[0], 3)

            def compute_localization_utility(self, min_observations=1):
                return torch.ones(self.get_xyz.shape[0])

            def compute_split_necessity(self, min_observations=1, min_radius=0.0, min_repeatability=0.0):
                return torch.arange(self.get_xyz.shape[0], 0, -1, dtype=torch.float32)

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                split = selected_mask.to(dtype=torch.bool)
                split_count = int(split.sum().item())
                self.split_requests.append(split_count)
                keep = ~split
                self.loc_observation_count = self.loc_observation_count[keep]
                self.loc_grad_accum = self.loc_grad_accum[keep]
                self.loc_grad_denom = self.loc_grad_denom[keep]
                self.loc_entropy_ema = self.loc_entropy_ema[keep]
                self.loc_repeatability_ema = self.loc_repeatability_ema[keep]
                self.last_topology_iteration = self.last_topology_iteration[keep]
                self.max_radii2D = self.max_radii2D[keep]
                self._loc_feature = nn.Parameter(self._loc_feature.detach()[keep])
                self._loc_opacity = nn.Parameter(self._loc_opacity.detach()[keep])
                self.loc_observation_count = torch.cat([self.loc_observation_count, torch.full((N * split_count,), 10)])
                self.loc_grad_accum = torch.cat([self.loc_grad_accum, torch.ones(N * split_count, 1)])
                self.loc_grad_denom = torch.cat([self.loc_grad_denom, torch.ones(N * split_count, 1)])
                self.loc_entropy_ema = torch.cat([self.loc_entropy_ema, torch.ones(N * split_count)])
                self.loc_repeatability_ema = torch.cat([self.loc_repeatability_ema, torch.ones(N * split_count)])
                self.last_topology_iteration = torch.cat(
                    [self.last_topology_iteration, torch.zeros(N * split_count, dtype=torch.long)]
                )
                self.max_radii2D = torch.cat([self.max_radii2D, torch.full((N * split_count,), 10.0)])
                self._loc_feature = nn.Parameter(
                    torch.cat([self._loc_feature.detach(), torch.zeros(N * split_count, 1, 2)])
                )
                self._loc_opacity = nn.Parameter(torch.cat([self._loc_opacity.detach(), torch.zeros(N * split_count, 1)]))

        def accept_risk(proposal, gaussians):
            return {"accepted": True, "reason": "holdout risk decreased", "delta_risk": -0.1}

        gaussians = FakeGaussians()
        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
                split_quantile=0.0,
                ambiguity_quantile=0.0,
                growth_cap_per_event=1.0,
                total_point_budget_ratio=2.0,
                cooldown_iterations=0,
                min_repeatability=0.0,
                min_radius=1.0,
                risk_commit_policy="callback",
            ),
            initial_points=3,
            risk_evaluator=accept_risk,
        )

        event = controller.update(gaussians, scene_extent=1.0, iteration=10)

        self.assertEqual(gaussians.split_requests, [3])
        self.assertTrue(event["risk_commit"]["accepted"])
        self.assertEqual(event["risk_commit"]["reason"], "holdout risk decreased")
        self.assertEqual(event["actual_parent_removed"], 3)
        self.assertEqual(event["actual_children_added"], 6)
        self.assertEqual(event["point_count_after"], 6)

    def test_risk_commit_rejects_unsupported_soft_prune_before_mutation(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self._loc_feature = nn.Parameter(torch.zeros(2, 1, 2))
                self._loc_opacity = nn.Parameter(torch.zeros(2, 1))
                self.utility = torch.tensor([-2.0, 1.0])

            @property
            def get_xyz(self):
                return torch.zeros(2, 3)

            def compute_localization_utility(self, min_observations=1):
                return self.utility

        gaussians = FakeGaussians()
        before = gaussians._loc_opacity.detach().clone()
        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                enable_split=False,
                enable_soft_prune=True,
                soft_prune_threshold=-1.0,
                risk_commit_policy="reject_all",
            ),
            initial_points=2,
        )

        with self.assertRaisesRegex(RuntimeError, "soft prune"):
            controller.update(gaussians, scene_extent=1.0, iteration=10)
        self.assertTrue(torch.equal(gaussians._loc_opacity.detach(), before))

    def test_topology_controller_stops_after_max_mutation_events(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.full((3,), 10)
                self.loc_grad_accum = torch.ones(3, 1)
                self.loc_grad_denom = torch.ones(3, 1)
                self.loc_entropy_ema = torch.ones(3)
                self.loc_repeatability_ema = torch.ones(3)
                self.last_topology_iteration = torch.zeros(3, dtype=torch.long)
                self.loc_birth_iteration = torch.zeros(3, dtype=torch.long)
                self.max_radii2D = torch.full((3,), 10.0)
                self.loc_node_id = torch.arange(3)
                self.loc_parent_node_id = torch.full((3,), -1)
                self.loc_source_index = torch.arange(3)
                self.loc_source_xyz = torch.zeros(3, 3)
                self.loc_prototype = torch.zeros(3, 2)
                self.loc_prototype_count = torch.zeros(3)
                self._loc_feature = nn.Parameter(torch.zeros(3, 1, 2))
                self._loc_opacity = nn.Parameter(torch.zeros(3, 1))

            @property
            def get_xyz(self):
                return torch.zeros(self.loc_observation_count.shape[0], 3)

            def compute_localization_utility(self, min_observations=1):
                return torch.ones(self.get_xyz.shape[0])

            def compute_split_necessity(self, min_observations=1, min_radius=0.0, min_repeatability=0.0):
                return torch.arange(self.get_xyz.shape[0], 0, -1, dtype=torch.float32)

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                split_count = int(selected_mask.sum().item())
                keep = ~selected_mask.to(dtype=torch.bool)
                for name in (
                    "loc_observation_count",
                    "loc_grad_accum",
                    "loc_grad_denom",
                    "loc_entropy_ema",
                    "loc_repeatability_ema",
                    "last_topology_iteration",
                    "loc_birth_iteration",
                    "max_radii2D",
                    "loc_node_id",
                    "loc_parent_node_id",
                    "loc_source_index",
                    "loc_source_xyz",
                    "loc_prototype",
                    "loc_prototype_count",
                ):
                    setattr(self, name, getattr(self, name)[keep])
                self._loc_feature = nn.Parameter(self._loc_feature.detach()[keep])
                self._loc_opacity = nn.Parameter(self._loc_opacity.detach()[keep])
                self.loc_observation_count = torch.cat([self.loc_observation_count, torch.full((2 * split_count,), 10)])
                self.loc_grad_accum = torch.cat([self.loc_grad_accum, torch.ones(2 * split_count, 1)])
                self.loc_grad_denom = torch.cat([self.loc_grad_denom, torch.ones(2 * split_count, 1)])
                self.loc_entropy_ema = torch.cat([self.loc_entropy_ema, torch.ones(2 * split_count)])
                self.loc_repeatability_ema = torch.cat([self.loc_repeatability_ema, torch.ones(2 * split_count)])
                self.last_topology_iteration = torch.cat(
                    [self.last_topology_iteration, torch.zeros(2 * split_count, dtype=torch.long)]
                )
                self.loc_birth_iteration = torch.cat(
                    [self.loc_birth_iteration, torch.zeros(2 * split_count, dtype=torch.long)]
                )
                self.max_radii2D = torch.cat([self.max_radii2D, torch.full((2 * split_count,), 10.0)])
                self.loc_node_id = torch.cat([self.loc_node_id, torch.arange(100, 100 + 2 * split_count)])
                self.loc_parent_node_id = torch.cat([self.loc_parent_node_id, torch.zeros(2 * split_count, dtype=torch.long)])
                self.loc_source_index = torch.cat([self.loc_source_index, torch.zeros(2 * split_count, dtype=torch.long)])
                self.loc_source_xyz = torch.cat([self.loc_source_xyz, torch.zeros(2 * split_count, 3)])
                self.loc_prototype = torch.cat([self.loc_prototype, torch.zeros(2 * split_count, 2)])
                self.loc_prototype_count = torch.cat([self.loc_prototype_count, torch.zeros(2 * split_count)])
                self._loc_feature = nn.Parameter(
                    torch.cat([self._loc_feature.detach(), torch.zeros(2 * split_count, 1, 2)])
                )
                self._loc_opacity = nn.Parameter(torch.cat([self._loc_opacity.detach(), torch.zeros(2 * split_count, 1)]))

        gaussians = FakeGaussians()
        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
                split_quantile=0.0,
                ambiguity_quantile=0.0,
                growth_cap_per_event=1.0,
                total_point_budget_ratio=2.0,
                cooldown_iterations=0,
                min_repeatability=0.0,
                min_radius=1.0,
                max_mutation_events=1,
            ),
            initial_points=3,
        )

        first = controller.update(gaussians, scene_extent=1.0, iteration=1)

        self.assertGreater(first["actual_children_added"], 0)
        self.assertEqual(controller.mutation_event_count, 1)
        self.assertFalse(controller.should_update(2))

    def test_topology_update_can_disable_split_for_prune_only_attribution(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self._loc_feature = nn.Parameter(torch.zeros(3, 1, 2))
                self._loc_opacity = nn.Parameter(torch.zeros(3, 1))
                self.utility = torch.tensor([-2.0, 0.5, -3.0])
                self.split_attempted = False

            @property
            def get_xyz(self):
                return torch.zeros(3, 3)

            def compute_localization_utility(self, min_observations=1):
                return self.utility

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                self.split_attempted = True
                raise AssertionError("prune-only attribution should not call split")

        gaussians = FakeGaussians()
        event = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                enable_split=False,
                enable_soft_prune=True,
                soft_prune_threshold=-1.0,
                soft_prune_step=4.0,
            ),
            initial_points=3,
        ).update(gaussians, scene_extent=1.0, iteration=10)

        self.assertFalse(gaussians.split_attempted)
        self.assertEqual(event["candidate_count"], 0)
        self.assertEqual(event["requested_split_count"], 0)
        self.assertLess(torch.sigmoid(gaussians._loc_opacity[0]).item(), 0.5)
        self.assertLess(torch.sigmoid(gaussians._loc_opacity[2]).item(), 0.5)

    def test_topology_update_logs_physical_prune_only_events_without_split(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self._opacity = torch.full((3, 1), -10.0)
                self._loc_opacity = nn.Parameter(torch.full((3, 1), -10.0))
                self._loc_feature = nn.Parameter(torch.zeros(3, 1, 2))
                self.utility = torch.full((3,), -2.0)
                self.loc_opacity_grad_seen = True

            @property
            def get_xyz(self):
                return torch.zeros(self._opacity.shape[0], 3)

            @property
            def get_opacity(self):
                return torch.sigmoid(self._opacity)

            @property
            def get_loc_opacity(self):
                return torch.sigmoid(self._loc_opacity)

            def compute_localization_utility(self, min_observations=1):
                return self.utility

            def prune_points(self, mask):
                keep = ~mask
                self._opacity = self._opacity[keep]
                self._loc_opacity = nn.Parameter(self._loc_opacity.detach()[keep])
                self._loc_feature = nn.Parameter(self._loc_feature.detach()[keep])
                self.utility = self.utility[keep]

        gaussians = FakeGaussians()
        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                enable_split=False,
                enable_physical_prune=True,
                physical_rgb_threshold=0.1,
                physical_loc_threshold=0.1,
                physical_utility_threshold=1.0,
            ),
            initial_points=3,
        )

        buffer = StringIO()
        with redirect_stdout(buffer):
            event = controller.update(gaussians, scene_extent=1.0, iteration=10)

        self.assertEqual(event["requested_split_count"], 0)
        self.assertEqual(event["physical_prune_count"], 3)
        self.assertIn("[Topology]", buffer.getvalue())
        self.assertIn("physical_prune=3", buffer.getvalue())

    def test_topology_update_marks_all_new_split_clones_on_cooldown(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.tensor([10, 10, 10, 10])
                self.loc_grad_accum = torch.tensor([[1.0], [10.0], [0.5], [0.2]])
                self.loc_grad_denom = torch.ones(4, 1)
                self.xyz_gradient_accum = torch.zeros(4, 1)
                self.loc_entropy_ema = torch.tensor([0.1, 1.0, 0.1, 0.1])
                self.loc_repeatability_ema = torch.ones(4)
                self.last_topology_iteration = torch.zeros(4, dtype=torch.long)
                self.max_radii2D = torch.full((4,), 10.0)
                self._loc_opacity = nn.Parameter(torch.zeros(4, 1))

            @property
            def get_xyz(self):
                return torch.zeros(self.last_topology_iteration.shape[0], 3)

            @property
            def get_opacity(self):
                return torch.full((self.last_topology_iteration.shape[0], 1), 0.8)

            @property
            def get_loc_opacity(self):
                return torch.sigmoid(self._loc_opacity)

            def compute_localization_utility(self, min_observations=1):
                return torch.ones(self.last_topology_iteration.shape[0])

            def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
                raise AssertionError("topology update should use explicit selected split masks")

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                split = selected_mask.to(dtype=torch.bool)
                parent_iters = self.last_topology_iteration[split].repeat(N)
                keep = ~split
                self.last_topology_iteration = torch.cat([self.last_topology_iteration[keep], parent_iters], dim=0)
                self.loc_observation_count = torch.cat([self.loc_observation_count[keep], self.loc_observation_count[split].repeat(N)], dim=0)
                self.loc_grad_accum = torch.cat([self.loc_grad_accum[keep], self.loc_grad_accum[split].repeat(N, 1)], dim=0)
                self.loc_grad_denom = torch.cat([self.loc_grad_denom[keep], self.loc_grad_denom[split].repeat(N, 1)], dim=0)
                self.loc_entropy_ema = torch.cat([self.loc_entropy_ema[keep], self.loc_entropy_ema[split].repeat(N)], dim=0)
                self.loc_repeatability_ema = torch.cat([self.loc_repeatability_ema[keep], self.loc_repeatability_ema[split].repeat(N)], dim=0)
                self.max_radii2D = torch.cat([self.max_radii2D[keep], self.max_radii2D[split].repeat(N)], dim=0)
                self._loc_opacity = nn.Parameter(torch.zeros(self.last_topology_iteration.shape[0], 1))

            def prune_points(self, mask):
                raise AssertionError("physical pruning should not run in this test")

        gaussians = FakeGaussians()
        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
                split_quantile=0.9,
                ambiguity_quantile=0.9,
                growth_cap_per_event=1.0,
                cooldown_iterations=5,
                min_repeatability=0.0,
                min_radius=1.0,
            ),
            initial_points=4,
        )

        event = controller.update(gaussians, scene_extent=1.0, iteration=10)

        self.assertEqual(gaussians.last_topology_iteration.shape[0], 5)
        self.assertTrue(torch.equal(gaussians.last_topology_iteration[-2:], torch.tensor([10, 10])))
        self.assertEqual(event["requested_split_count"], 1)
        self.assertEqual(event["actual_parent_removed"], 1)
        self.assertEqual(event["actual_children_added"], 2)
        self.assertEqual(event["point_count_before"], 4)
        self.assertEqual(event["point_count_after"], 5)

    def test_topology_event_candidate_count_reports_eligible_candidates(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.tensor([10, 2, 10, 10])
                self.loc_grad_accum = torch.tensor([[1.0], [10.0], [0.5], [0.2]])
                self.loc_grad_denom = torch.ones(4, 1)
                self.loc_entropy_ema = torch.tensor([1.0, 1.0, 0.1, 0.1])
                self.loc_repeatability_ema = torch.ones(4)
                self.last_topology_iteration = torch.zeros(4, dtype=torch.long)
                self.max_radii2D = torch.full((4,), 10.0)
                self._loc_opacity = nn.Parameter(torch.zeros(4, 1))

            @property
            def get_xyz(self):
                return torch.zeros(self.last_topology_iteration.shape[0], 3)

            @property
            def get_opacity(self):
                return torch.full((self.last_topology_iteration.shape[0], 1), 0.8)

            @property
            def get_loc_opacity(self):
                return torch.sigmoid(self._loc_opacity)

            def compute_localization_utility(self, min_observations=1):
                return torch.ones(self.last_topology_iteration.shape[0])

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                split = selected_mask.to(dtype=torch.bool)
                keep = ~split
                self.last_topology_iteration = torch.cat([self.last_topology_iteration[keep], self.last_topology_iteration[split].repeat(N)], dim=0)
                self.loc_observation_count = torch.cat([self.loc_observation_count[keep], self.loc_observation_count[split].repeat(N)], dim=0)
                self.loc_grad_accum = torch.cat([self.loc_grad_accum[keep], self.loc_grad_accum[split].repeat(N, 1)], dim=0)
                self.loc_grad_denom = torch.cat([self.loc_grad_denom[keep], self.loc_grad_denom[split].repeat(N, 1)], dim=0)
                self.loc_entropy_ema = torch.cat([self.loc_entropy_ema[keep], self.loc_entropy_ema[split].repeat(N)], dim=0)
                self.loc_repeatability_ema = torch.cat([self.loc_repeatability_ema[keep], self.loc_repeatability_ema[split].repeat(N)], dim=0)
                self.max_radii2D = torch.cat([self.max_radii2D[keep], self.max_radii2D[split].repeat(N)], dim=0)
                self._loc_opacity = nn.Parameter(torch.zeros(self.last_topology_iteration.shape[0], 1))

            def prune_points(self, mask):
                raise AssertionError("physical pruning should not run in this test")

        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
                split_quantile=0.9,
                growth_cap_per_event=1.0,
                cooldown_iterations=5,
                min_repeatability=0.0,
                min_radius=1.0,
            ),
            initial_points=4,
        )

        event = controller.update(FakeGaussians(), scene_extent=1.0, iteration=10)

        self.assertEqual(event["candidate_count"], 3)
        self.assertEqual(event["requested_split_count"], 1)

    def test_topology_update_rejects_split_count_mismatch(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.tensor([10, 10, 10, 10])
                self.loc_grad_accum = torch.tensor([[1.0], [10.0], [0.5], [0.2]])
                self.loc_grad_denom = torch.ones(4, 1)
                self.loc_entropy_ema = torch.tensor([0.1, 1.0, 0.1, 0.1])
                self.loc_repeatability_ema = torch.ones(4)
                self.last_topology_iteration = torch.zeros(4, dtype=torch.long)
                self.max_radii2D = torch.full((4,), 10.0)
                self._loc_opacity = nn.Parameter(torch.zeros(4, 1))

            @property
            def get_xyz(self):
                return torch.zeros(self.last_topology_iteration.shape[0], 3)

            @property
            def get_opacity(self):
                return torch.full((self.last_topology_iteration.shape[0], 1), 0.8)

            @property
            def get_loc_opacity(self):
                return torch.sigmoid(self._loc_opacity)

            def compute_localization_utility(self, min_observations=1):
                return torch.ones(self.last_topology_iteration.shape[0])

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                return None

            def prune_points(self, mask):
                raise AssertionError("physical pruning should not run in this test")

        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
                split_quantile=0.9,
                ambiguity_quantile=0.9,
                growth_cap_per_event=1.0,
                cooldown_iterations=5,
                min_repeatability=0.0,
                min_radius=1.0,
            ),
            initial_points=4,
        )

        with self.assertRaisesRegex(RuntimeError, "requested 1 splits"):
            controller.update(FakeGaussians(), scene_extent=1.0, iteration=10)

    def test_topology_update_is_split_only_by_default(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.zeros(2, dtype=torch.long)
                self.loc_grad_accum = torch.zeros(2, 1)
                self.loc_grad_denom = torch.ones(2, 1)
                self.loc_entropy_ema = torch.zeros(2)
                self.loc_repeatability_ema = torch.zeros(2)
                self.last_topology_iteration = torch.zeros(2, dtype=torch.long)
                self.max_radii2D = torch.zeros(2)
                self._loc_opacity = nn.Parameter(torch.full((2, 1), -10.0))
                self.prune_called = False

            @property
            def get_xyz(self):
                return torch.zeros(2, 3)

            @property
            def get_opacity(self):
                return torch.zeros(2, 1)

            @property
            def get_loc_opacity(self):
                return torch.sigmoid(self._loc_opacity)

            def compute_localization_utility(self, min_observations=1):
                return torch.full((2,), -5.0)

            def prune_points(self, mask):
                self.prune_called = True

        gaussians = FakeGaussians()
        before = gaussians._loc_opacity.detach().clone()
        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
            ),
            initial_points=2,
        )

        event = controller.update(gaussians, scene_extent=1.0, iteration=10)

        self.assertFalse(gaussians.prune_called)
        self.assertTrue(torch.equal(gaussians._loc_opacity, before))
        self.assertEqual(event["physical_prune_count"], 0)

    def test_topology_update_can_physical_prune_then_split_with_synced_buffers(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.tensor([10, 10, 10, 10])
                self.loc_grad_accum = torch.tensor([[0.2], [10.0], [0.1], [0.1]])
                self.loc_grad_denom = torch.ones(4, 1)
                self.loc_entropy_ema = torch.tensor([0.1, 1.0, 0.1, 0.1])
                self.loc_repeatability_ema = torch.ones(4)
                self.last_topology_iteration = torch.zeros(4, dtype=torch.long)
                self.max_radii2D = torch.full((4,), 10.0)
                self._loc_feature = nn.Parameter(torch.zeros(4, 1, 2))
                self._loc_opacity = nn.Parameter(torch.tensor([[0.0], [0.0], [-10.0], [0.0]]))
                self._opacity = torch.tensor([[2.0], [2.0], [-10.0], [2.0]])
                self.loc_opacity_grad_seen = True

            @property
            def get_xyz(self):
                return torch.zeros(self.loc_observation_count.shape[0], 3)

            @property
            def get_opacity(self):
                return torch.sigmoid(self._opacity)

            @property
            def get_loc_opacity(self):
                return torch.sigmoid(self._loc_opacity)

            def compute_landmark_reliability(self, min_observations=1):
                return torch.tensor([1.0, 1.0, -5.0, 0.5])[: self.get_xyz.shape[0]]

            def compute_pose_geometry_value(self, min_observations=1):
                return torch.zeros(self.get_xyz.shape[0])

            def compute_split_necessity(self, min_observations=1, min_radius=0.0, min_repeatability=0.0):
                score = torch.zeros(self.get_xyz.shape[0])
                score[1] = 10.0
                return score

            def _keep(self, keep):
                self.loc_observation_count = self.loc_observation_count[keep]
                self.loc_grad_accum = self.loc_grad_accum[keep]
                self.loc_grad_denom = self.loc_grad_denom[keep]
                self.loc_entropy_ema = self.loc_entropy_ema[keep]
                self.loc_repeatability_ema = self.loc_repeatability_ema[keep]
                self.last_topology_iteration = self.last_topology_iteration[keep]
                self.max_radii2D = self.max_radii2D[keep]
                self._loc_feature = nn.Parameter(self._loc_feature.detach()[keep])
                self._loc_opacity = nn.Parameter(self._loc_opacity.detach()[keep])
                self._opacity = self._opacity[keep]

            def prune_points(self, mask):
                self._keep(~mask.to(dtype=torch.bool))

            def densify_and_split_selected(self, selected_mask, scene_extent, N=2):
                split = selected_mask.to(dtype=torch.bool)
                keep = ~split
                self._keep(keep)
                self.loc_observation_count = torch.cat([self.loc_observation_count, torch.full((N,), 10)])
                self.loc_grad_accum = torch.cat([self.loc_grad_accum, torch.full((N, 1), 10.0)])
                self.loc_grad_denom = torch.cat([self.loc_grad_denom, torch.ones(N, 1)])
                self.loc_entropy_ema = torch.cat([self.loc_entropy_ema, torch.ones(N)])
                self.loc_repeatability_ema = torch.cat([self.loc_repeatability_ema, torch.ones(N)])
                self.last_topology_iteration = torch.cat([self.last_topology_iteration, torch.zeros(N, dtype=torch.long)])
                self.max_radii2D = torch.cat([self.max_radii2D, torch.full((N,), 10.0)])
                self._loc_feature = nn.Parameter(torch.cat([self._loc_feature.detach(), torch.zeros(N, 1, 2)]))
                self._loc_opacity = nn.Parameter(torch.cat([self._loc_opacity.detach(), torch.zeros(N, 1)]))
                self._opacity = torch.cat([self._opacity, torch.zeros(N, 1)])

        gaussians = FakeGaussians()
        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                min_observations=8,
                split_quantile=0.9,
                growth_cap_per_event=1.0,
                total_point_budget_ratio=2.0,
                cooldown_iterations=5,
                min_repeatability=0.0,
                min_radius=1.0,
                enable_physical_prune=True,
            ),
            initial_points=4,
        )

        buffer = StringIO()
        with redirect_stdout(buffer):
            event = controller.update(gaussians, scene_extent=1.0, iteration=10)

        self.assertEqual(event["physical_prune_count"], 1)
        self.assertEqual(event["actual_parent_removed"], 1)
        self.assertEqual(event["actual_children_added"], 2)
        self.assertEqual(event["point_count_after"], 4)
        self.assertEqual(gaussians.last_topology_iteration.shape[0], gaussians.get_xyz.shape[0])
        self.assertIn("physical_prune=1", buffer.getvalue())

    def test_physical_prune_can_protect_sparse_landmark_source_ids(self):
        from localization_training.topology_controller import joint_physical_prune_mask

        class FakeGaussians:
            def __init__(self):
                self._opacity = torch.full((4, 1), -10.0)
                self._loc_opacity = nn.Parameter(torch.full((4, 1), -10.0))
                self.loc_source_index = torch.tensor([10, 20, 20, 30])

            @property
            def get_xyz(self):
                return torch.zeros(4, 3)

            @property
            def get_opacity(self):
                return torch.sigmoid(self._opacity)

            @property
            def get_loc_opacity(self):
                return torch.sigmoid(self._loc_opacity)

        mask = joint_physical_prune_mask(
            FakeGaussians(),
            utility=torch.full((4,), -5.0),
            rgb_threshold=0.1,
            loc_threshold=0.1,
            utility_threshold=-1.0,
            protected_source_indices=torch.tensor([20]),
        )

        self.assertTrue(torch.equal(mask, torch.tensor([True, False, False, True])))

    def test_physical_prune_requires_loc_opacity_training_signal_by_default(self):
        from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig

        class FakeGaussians:
            def __init__(self):
                self.loc_observation_count = torch.full((2,), 10)
                self.loc_grad_accum = torch.ones(2, 1)
                self.loc_grad_denom = torch.ones(2, 1)
                self.loc_entropy_ema = torch.ones(2)
                self.loc_repeatability_ema = torch.ones(2)
                self.last_topology_iteration = torch.zeros(2, dtype=torch.long)
                self.max_radii2D = torch.ones(2)
                self._loc_opacity = nn.Parameter(torch.full((2, 1), -10.0))
                self.loc_opacity_grad_seen = False

            @property
            def get_xyz(self):
                return torch.zeros(2, 3)

            @property
            def get_opacity(self):
                return torch.zeros(2, 1)

            @property
            def get_loc_opacity(self):
                return torch.sigmoid(self._loc_opacity)

            def compute_localization_utility(self, min_observations=1):
                return torch.full((2,), -5.0)

        controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=0,
                update_interval=1,
                enable_physical_prune=True,
            ),
            initial_points=2,
        )

        with self.assertRaisesRegex(RuntimeError, "loc opacity"):
            controller.update(FakeGaussians(), scene_extent=1.0, iteration=10)


if __name__ == "__main__":
    unittest.main()
