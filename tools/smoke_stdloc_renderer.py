"""Exercise the patched native STDLoc RGB+descriptor renderer and backward."""
import argparse
import sys
from types import SimpleNamespace
import torch

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True)
a = p.parse_args()
sys.path.insert(0, a.repo)
from gaussian_renderer import render_gsplat

torch.manual_seed(2026)
torch.set_num_threads(4)
n = 16
xyz = torch.randn(n, 3, device='cuda') * .2
xyz[:, 2] += 3
features = torch.randn(n, 1, 256, device='cuda', requires_grad=True)
pc = SimpleNamespace(get_xyz=xyz.requires_grad_(), get_opacity=torch.ones(n, 1, device='cuda') * .5,
    get_scaling=torch.ones(n, 3, device='cuda') * .1,
    get_rotation=torch.tensor([[1., 0., 0., 0.]], device='cuda').repeat(n, 1),
    get_features=torch.zeros(n, 1, 3, device='cuda'), active_sh_degree=0,
    get_loc_feature=features)
camera = SimpleNamespace(world_view_transform=torch.eye(4, device='cuda'),
                         image_width=32, image_height=32, FoVx=1., FoVy=1.)
out = render_gsplat(camera, pc, bg_color=None, rgb_only=False)
assert out['visibility_filter'].shape == (n,)
assert out['radii'].shape == (n,)
assert out['feature_map'].shape == (256, 32, 32)
loss = out['render'].sum() + out['feature_map'][0].sum()
loss.backward()
assert features.grad is not None and torch.isfinite(features.grad).all()
assert pc.get_xyz.grad is not None and torch.isfinite(pc.get_xyz.grad).all()
print('STDLoc native RGB/feature forward+backward passed')
