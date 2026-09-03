#!/usr/bin/env python3
"""Render registered views and audit an external Gaussian prior's appearance."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity
import torch

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rendering import render_from_pose_gsplat


SCHEMA = "anygsloc_gaussian_prior_view_audit"


def _sample_indices(count: int, requested: int) -> list[int]:
    if count < 1 or requested < 1:
        raise ValueError("camera and sample counts must be positive")
    sample_count = min(count, requested)
    if sample_count == 1:
        return [count // 2]
    return sorted({round(i * (count - 1) / (sample_count - 1)) for i in range(sample_count)})


def _uint8_image(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _error_image(reference: torch.Tensor, rendered: torch.Tensor) -> Image.Image:
    error = (reference - rendered).abs().mean(dim=0).clamp(0.0, 1.0)
    # A compact blue -> yellow -> red heatmap without an extra plotting dependency.
    red = (2.0 * error).clamp(0.0, 1.0)
    green = (2.0 - 2.0 * error).clamp(0.0, 1.0) * error.sqrt()
    blue = (1.0 - 2.0 * error).clamp(0.0, 1.0)
    return _uint8_image(torch.stack((red, green, blue)))


def _metrics(reference: torch.Tensor, rendered: torch.Tensor, alpha: torch.Tensor) -> dict:
    reference_cpu = reference.detach().float().clamp(0.0, 1.0).cpu()
    rendered_cpu = rendered.detach().float().clamp(0.0, 1.0).cpu()
    mse = float(torch.mean((reference_cpu - rendered_cpu) ** 2))
    mae = float(torch.mean(torch.abs(reference_cpu - rendered_cpu)))
    ref_np = reference_cpu.permute(1, 2, 0).numpy()
    render_np = rendered_cpu.permute(1, 2, 0).numpy()
    ssim = float(structural_similarity(ref_np, render_np, channel_axis=2, data_range=1.0))
    alpha_cpu = alpha.detach().float().squeeze().cpu()
    return {
        "psnr_db": float(-10.0 * math.log10(max(mse, 1e-12))),
        "ssim": ssim,
        "mae": mae,
        "alpha_mean": float(alpha_cpu.mean()),
        "alpha_covered_fraction_005": float((alpha_cpu >= 0.05).float().mean()),
        "alpha_covered_fraction_050": float((alpha_cpu >= 0.50).float().mean()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--split", choices=("mapping", "test", "all"), default="all")
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or not 0 <= args.sh_degree <= 3:
        parser.error("samples must be positive and SH degree must be in [0, 3]")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.mkdir(parents=True)
    started = time.perf_counter()
    try:
        dataset = ColmapDataset(args.dataset, images=args.images)
        cameras = dataset.split(args.split)
        indices = _sample_indices(len(cameras), args.samples)
        device = torch.device(args.device)
        model_class = GaussianModel3D if args.gaussian_type == "3dgs" else GaussianModel2D
        model = model_class(args.sh_degree, device=device)
        gaussian_ply = args.gaussian_ply.expanduser().resolve()
        model.load_ply(gaussian_ply, loc_feature_dim=0)
        model.eval()
        background = torch.ones(3, device=device) if args.white_background else torch.zeros(3, device=device)

        records = []
        previews = []
        for ordinal, camera_index in enumerate(indices):
            camera = cameras[camera_index]
            reference = dataset.load_image(camera).to(device)
            pose = torch.as_tensor(camera.pose_w2c, device=device, dtype=torch.float32)
            package = render_from_pose_gsplat(
                model,
                pose,
                camera.fov_x,
                camera.fov_y,
                camera.width,
                camera.height,
                bg_color=background,
                render_mode="RGB+ED",
                rgb_only=True,
                rasterize_mode="antialiased",
            )
            rendered = package["render"][:3].float().clamp(0.0, 1.0)
            alpha = package.get("alphas", package.get("rend_alpha"))
            if alpha is None:
                raise ValueError("renderer did not return alpha")
            metrics = _metrics(reference, rendered, alpha)
            records.append(
                {
                    "camera_index": camera_index,
                    "image_name": camera.image_name,
                    **metrics,
                }
            )
            reference_image = _uint8_image(reference)
            rendered_image = _uint8_image(rendered)
            error_image = _error_image(reference, rendered)
            triptych = Image.new("RGB", (camera.width * 3, camera.height))
            triptych.paste(reference_image, (0, 0))
            triptych.paste(rendered_image, (camera.width, 0))
            triptych.paste(error_image, (camera.width * 2, 0))
            preview_path = temporary / f"view_{ordinal:03d}.jpg"
            triptych.save(preview_path, quality=92)
            previews.append(triptych)
            print(
                f"{ordinal + 1}/{len(indices)} {camera.image_name}: "
                f"PSNR={metrics['psnr_db']:.2f} SSIM={metrics['ssim']:.3f} "
                f"alpha={metrics['alpha_covered_fraction_005']:.3f}",
                flush=True,
            )

        target_width = 1200
        rows = []
        for preview in previews:
            height = round(preview.height * target_width / preview.width)
            rows.append(preview.resize((target_width, height), Image.Resampling.LANCZOS))
        montage = Image.new("RGB", (target_width, sum(row.height for row in rows)), "black")
        y = 0
        for row in rows:
            montage.paste(ImageOps.exif_transpose(row), (0, y))
            y += row.height
        montage.save(temporary / "montage.jpg", quality=92)

        aggregate = {
            key: float(np.median([record[key] for record in records]))
            for key in (
                "psnr_db",
                "ssim",
                "mae",
                "alpha_mean",
                "alpha_covered_fraction_005",
                "alpha_covered_fraction_050",
            )
        }
        payload = {
            "schema": SCHEMA,
            "version": 1,
            "dataset": str(args.dataset.expanduser().resolve()),
            "images": args.images,
            "split": args.split,
            "gaussian_ply": str(gaussian_ply),
            "gaussian_ply_sha256": sha256_file(gaussian_ply),
            "gaussian_type": args.gaussian_type,
            "sh_degree": int(args.sh_degree),
            "background": "white" if args.white_background else "black",
            "sample_count": len(records),
            "median": aggregate,
            "records": records,
            "timing_seconds": time.perf_counter() - started,
            "preview_layout": "reference | render | absolute-error heatmap",
        }
        (temporary / "report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        os.rename(temporary, output)
        print(json.dumps(payload["median"], indent=2, sort_keys=True), flush=True)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
