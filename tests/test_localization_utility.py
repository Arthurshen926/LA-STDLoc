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


if __name__ == "__main__":
    unittest.main()
