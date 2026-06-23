import unittest
from contextlib import redirect_stdout
from io import StringIO

import torch
from torch import nn


class TopologyControllerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
