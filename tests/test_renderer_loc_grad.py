import math
import unittest

import torch
from torch import nn


@unittest.skipUnless(torch.cuda.is_available(), "renderer requires CUDA")
class RendererLocGradTest(unittest.TestCase):
    def test_render_gsplat_returns_feature_pass_metadata_with_grad(self):
        from gaussian_renderer import render_gsplat
        from scene.gaussian_model import GaussianModel
        from utils.general_utils import inverse_sigmoid

        class Camera:
            image_width = 32
            image_height = 32
            FoVx = math.radians(60)
            FoVy = math.radians(60)
            world_view_transform = torch.eye(4, device="cuda")

        model = GaussianModel(3)
        model.active_sh_degree = 0
        model._xyz = nn.Parameter(torch.tensor([[0.0, 0.0, 2.0], [0.15, 0.0, 2.2]], device="cuda"))
        model._features_dc = nn.Parameter(torch.ones(2, 1, 3, device="cuda") * 0.2)
        model._features_rest = nn.Parameter(torch.zeros(2, 15, 3, device="cuda"))
        model._opacity = nn.Parameter(inverse_sigmoid(torch.full((2, 1), 0.8, device="cuda")))
        model._scaling = nn.Parameter(torch.log(torch.full((2, 3), 0.08, device="cuda")))
        model._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").repeat(2, 1))
        model._loc_feature = nn.Parameter(torch.randn(2, 1, 8, device="cuda"))
        model.init_localization_state(from_rgb_opacity=True)

        pkg = render_gsplat(
            Camera(),
            model,
            torch.zeros(3, device="cuda"),
            rgb_only=False,
            return_loc_meta=True,
            use_loc_opacity=True,
        )

        self.assertIn("loc_viewspace_points", pkg)
        self.assertIn("loc_visible_idx", pkg)
        self.assertEqual(pkg["loc_visible_idx"].ndim, 1)
        self.assertTrue(pkg["loc_viewspace_points"].requires_grad)

        loc_loss = pkg["feature_map"].sum()
        loc_grad = torch.autograd.grad(
            loc_loss,
            pkg["loc_viewspace_points"],
            retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertIsNotNone(loc_grad)
        self.assertTrue(torch.isfinite(loc_grad).all())


if __name__ == "__main__":
    unittest.main()
