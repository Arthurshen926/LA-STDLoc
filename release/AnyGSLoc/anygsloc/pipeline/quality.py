"""Reference-view PSNR, with the exact paper rendering and pixel metric."""

import math

import numpy as np
import torch
from PIL import Image

from anygsloc.artifacts import sha256
from anygsloc.pipeline.io import complete, prepare_output, write_json


class Renderer:
    def __init__(self, source, device="cuda:0"):
        self.device = device
        import torch

        from anygsloc.priors.models import FrozenGaussianModel

        assert sha256(source["ply"]) == source["ply_sha256"]
        self.model = FrozenGaussianModel(source["sh_degree"], device=self.device)
        self.model.load_ply(source["ply"], loc_feature_dim=0)
        m = self.model
        self.two = m.get_scaling.shape[1] == 2
        scales = m.get_scaling
        if self.two:
            scales = torch.cat([scales, torch.ones_like(scales[:, :1])], 1)
        self.params = dict(
            means=m.get_xyz,
            quats=m.get_rotation,
            scales=scales,
            opacities=m.get_opacity[:, 0],
            colors=m.get_features,
            sh_degree=m.active_sh_degree,
            packed=False,
            near_plane=0.01,
            far_plane=10000.0,
            eps2d=0.3,
            radius_clip=0.0,
            backgrounds=torch.zeros((1, 3), device=self.device),
            render_mode="RGB",
        )
        for k in ["means", "quats", "scales", "opacities", "colors"]:
            assert torch.isfinite(self.params[k]).all().item(), (
                "nonfinite prior",
                k,
                source["ply"],
            )
        assert self.two == (source["gaussian_type"] == "2dgs")
        self.n_primitives = len(m.get_xyz)

    def render(self, view):
        import torch
        from gsplat import rasterization, rasterization_2dgs

        args = dict(
            self.params,
            viewmats=torch.tensor(
                view["pose_w2c"], dtype=torch.float32, device=self.device
            )[None],
            Ks=torch.tensor(view["K"], dtype=torch.float32, device=self.device)[None],
            width=view["width"],
            height=view["height"],
        )
        tile = 16
        if self.two:
            try:
                result = rasterization_2dgs(**args, tile_size=16)
            except torch.cuda.OutOfMemoryError:
                # Leave the failed-call traceback before retrying.
                result = None
            if result is None:
                torch.cuda.empty_cache()
                tile = 32
                result = rasterization_2dgs(**args, tile_size=32)
        else:
            result = rasterization(**args, rasterize_mode="antialiased", tile_size=16)
        pred = result[0][0].clamp(0, 1)
        assert tuple(pred.shape) == (view["height"], view["width"], 3)
        assert torch.isfinite(pred).all().item()
        del result, args
        return pred, tile


def evaluate_psnr(spec, output, device):
    from anygsloc.data.datasets import ColmapDataset
    from anygsloc.mapping.virtual_camera_registry import camera_intrinsic

    source = {
        "ply": spec["prior"],
        "ply_sha256": sha256(spec["prior"]),
        "sh_degree": spec.get("sh_degree", 3),
        "gaussian_type": spec["gaussian_type"],
    }
    out = prepare_output(
        output, {"prior": source, "quality_background": "black", "split": "mapping"}
    )
    data = ColmapDataset(
        spec["dataset"], images=spec.get("images", "processed"), load_masks=False
    )
    rows = []
    with torch.cuda.device(device), torch.inference_mode():
        renderer = Renderer(source, device)
        for camera in data.split("mapping"):
            view = {
                "pose_w2c": camera.pose_w2c,
                "K": camera_intrinsic(camera),
                "width": camera.width,
                "height": camera.height,
            }
            pred, tile = renderer.render(view)
            with Image.open(camera.image_path) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            gt = torch.from_numpy(rgb).to(device=device, dtype=torch.float64) / 255.0
            mse = float((pred.double() - gt).square().mean())
            if mse <= 0 or not math.isfinite(mse):
                raise ValueError("invalid PSNR MSE")
            rows.append(
                {
                    "image_name": camera.image_name,
                    "mse": mse,
                    "psnr_db": -10 * math.log10(mse),
                    "tile_size": tile,
                }
            )
    write_json(out / "records.json", rows)
    write_json(
        out / "summary.json",
        {
            "views": len(rows),
            "mean_psnr_db": float(np.mean([r["psnr_db"] for r in rows])),
            "pixel_population": "all RGB samples; no mask, crop, exposure fit or 8-bit prediction quantization",
        },
    )
    complete(out, ["records.json", "summary.json"])
