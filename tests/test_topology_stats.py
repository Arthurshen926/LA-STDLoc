import unittest

import torch
from torch import nn


@unittest.skipUnless(torch.cuda.is_available(), "GaussianModel uses CUDA tensors")
class GaussianLocalizationStatsTest(unittest.TestCase):
    def _make_model(self, n=5, feature_dim=4):
        from scene.gaussian_model import GaussianModel
        from utils.general_utils import inverse_sigmoid

        model = GaussianModel(3)
        device = torch.device("cuda")
        model._xyz = nn.Parameter(torch.zeros(n, 3, device=device))
        model._features_dc = nn.Parameter(torch.zeros(n, 1, 3, device=device))
        model._features_rest = nn.Parameter(torch.zeros(n, 15, 3, device=device))
        model._scaling = nn.Parameter(torch.zeros(n, 3, device=device))
        model._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1))
        model._opacity = nn.Parameter(inverse_sigmoid(torch.full((n, 1), 0.25, device=device)))
        model._loc_feature = nn.Parameter(torch.randn(n, 1, feature_dim, device=device))
        model.max_radii2D = torch.zeros(n, device=device)
        model.xyz_gradient_accum = torch.zeros(n, 1, device=device)
        model.denom = torch.zeros(n, 1, device=device)
        return model

    def test_localization_state_initializes_updates_and_round_trips(self):
        model = self._make_model()

        model.init_localization_state(from_rgb_opacity=True)
        self.assertEqual(model.get_loc_opacity.shape, (5, 1))
        self.assertTrue(torch.allclose(model.get_loc_opacity, model.get_opacity, atol=1e-6))

        full_idx = torch.tensor([0, 2, 4], device="cuda")
        model.add_localization_stats(
            full_idx=full_idx,
            means2d_grad=torch.tensor([[3.0, 4.0], [0.0, 2.0], [1.0, 2.0]], device="cuda"),
            radii=torch.tensor([2.0, 3.0, 4.0], device="cuda"),
            episode_stats={
                "positive_prob": torch.tensor([0.9, 0.8, 0.7], device="cuda"),
                "margin": torch.tensor([0.5, 0.4, 0.3], device="cuda"),
                "entropy": torch.tensor([0.1, 0.2, 0.3], device="cuda"),
                "reproj_error": torch.tensor([1.0, 2.0, 3.0], device="cuda"),
                "prototype": torch.ones(3, 4, device="cuda"),
            },
            ema_decay=0.5,
        )

        self.assertEqual(model.loc_observation_count[2].item(), 1)
        self.assertGreater(model.loc_grad_accum[0].item(), 0)
        utility = model.compute_localization_utility(min_observations=1)
        self.assertEqual(utility.shape, (5,))
        self.assertTrue(torch.isfinite(utility).all())

        state = model.capture_localization_state()
        restored = self._make_model()
        restored.restore_localization_state(state)
        self.assertTrue(torch.allclose(restored.loc_positive_prob_ema, model.loc_positive_prob_ema))
        self.assertTrue(torch.allclose(restored.get_loc_opacity, model.get_loc_opacity))

    def test_restore_backfills_legacy_localization_state_buffers(self):
        model = self._make_model()
        model.init_localization_state(from_rgb_opacity=True, birth_iteration=7)
        model.loc_observation_count[:] = torch.arange(5, device="cuda")
        model.loc_positive_prob_ema[:] = torch.linspace(0.1, 0.5, 5, device="cuda")

        legacy_state = model.capture_localization_state()
        legacy_state.pop("last_topology_iteration")
        legacy_state["loc_redundancy_ema"] = torch.empty(0, device="cuda")

        restored = self._make_model()
        restored.restore_localization_state(legacy_state)

        for name in restored._localization_buffer_names():
            self.assertEqual(getattr(restored, name).shape[0], restored.get_xyz.shape[0], name)
        self.assertTrue(torch.allclose(restored.loc_positive_prob_ema, model.loc_positive_prob_ema))
        self.assertTrue(torch.equal(restored.loc_observation_count, model.loc_observation_count))
        self.assertTrue(torch.equal(restored.last_topology_iteration, torch.zeros(5, dtype=torch.long, device="cuda")))
        self.assertTrue(torch.equal(restored.loc_redundancy_ema, torch.zeros(5, device="cuda")))

    def test_restore_localization_state_rebinds_loc_opacity_optimizer_param(self):
        class Opt:
            percent_dense = 0.01
            position_lr_init = 0.001
            position_lr_final = 0.0001
            position_lr_delay_mult = 0.01
            position_lr_max_steps = 100
            feature_lr = 0.001
            opacity_lr = 0.001
            scaling_lr = 0.001
            rotation_lr = 0.001
            loc_feature_lr = 0.001
            loc_opacity_lr = 0.001

        model = self._make_model(n=3)
        model.init_localization_state(from_rgb_opacity=True)
        model.training_setup(Opt())
        old_param = model._loc_opacity
        restored_opacity = torch.full_like(model._loc_opacity, -4.0)

        model.restore_localization_state({"loc_opacity": restored_opacity})

        loc_group = next(group for group in model.optimizer.param_groups if group["name"] == "loc_opacity")
        self.assertIs(loc_group["params"][0], model._loc_opacity)
        self.assertIsNot(model._loc_opacity, old_param)
        model.get_loc_opacity.mean().backward()
        self.assertIsNotNone(model._loc_opacity.grad)
        self.assertGreater(model._loc_opacity.grad.abs().max().item(), 0.0)

    def test_localization_stats_backfill_and_update_screen_radii(self):
        model = self._make_model()
        model.init_localization_state(from_rgb_opacity=True)
        model.max_radii2D = torch.empty(0, device="cuda")

        full_idx = torch.tensor([0, 2, 4], device="cuda")
        model.add_localization_stats(
            full_idx=full_idx,
            radii=torch.tensor([1.5, 5.0, 3.0], device="cuda"),
            episode_stats={"repeatability": torch.ones(3, device="cuda")},
        )

        self.assertEqual(model.max_radii2D.shape[0], model.get_xyz.shape[0])
        self.assertTrue(torch.allclose(model.max_radii2D[full_idx], torch.tensor([1.5, 5.0, 3.0], device="cuda")))
        model.update_screen_radii(
            torch.tensor([True, False, True, False, False], device="cuda"),
            torch.tensor([2.0, 0.0, 4.0, 0.0, 0.0], device="cuda"),
        )
        self.assertTrue(torch.allclose(model.max_radii2D[full_idx], torch.tensor([2.0, 5.0, 3.0], device="cuda")))

    def test_densification_and_prune_keep_localization_buffers_in_sync(self):
        class Opt:
            percent_dense = 0.01
            position_lr_init = 0.001
            position_lr_final = 0.0001
            position_lr_delay_mult = 0.01
            position_lr_max_steps = 100
            feature_lr = 0.001
            opacity_lr = 0.001
            scaling_lr = 0.001
            rotation_lr = 0.001
            loc_feature_lr = 0.001
            loc_opacity_lr = 0.001

        model = self._make_model(n=5)
        model.init_localization_state(from_rgb_opacity=True)
        model.loc_observation_count[:] = torch.arange(5, device="cuda")
        model.training_setup(Opt())

        parent_mask = torch.tensor([False, True, False, True, False], device="cuda")
        model.densification_postfix(
            model._xyz[parent_mask].detach().clone(),
            model._features_dc[parent_mask].detach().clone(),
            model._features_rest[parent_mask].detach().clone(),
            model._opacity[parent_mask].detach().clone(),
            model._scaling[parent_mask].detach().clone(),
            model._rotation[parent_mask].detach().clone(),
            model._loc_feature[parent_mask].detach().clone(),
            model._loc_opacity[parent_mask].detach().clone(),
            loc_parent_mask=parent_mask,
            loc_repeat=1,
        )

        self.assertEqual(model.get_xyz.shape[0], 7)
        self.assertEqual(model.loc_grad_accum.shape[0], 7)
        self.assertEqual(model.loc_observation_count[-2:].tolist(), [1, 3])

        prune_mask = torch.tensor([False, True, False, False, False, True, False], device="cuda")
        model.prune_points(prune_mask)
        self.assertEqual(model.get_xyz.shape[0], 5)
        self.assertEqual(model.get_loc_opacity.shape[0], 5)
        self.assertEqual(model.loc_grad_accum.shape[0], 5)
        self.assertEqual(model.loc_prototype.shape[0], 5)


if __name__ == "__main__":
    unittest.main()
