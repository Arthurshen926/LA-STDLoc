#!/usr/bin/env python3
"""Evaluate an RGB-only Gaussian prior on held-out Cambridge test views."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

def _summary(rows: list[dict]) -> dict:
    output = {"query_count": len(rows)}
    for key in ("psnr_db", "ssim", "lpips"):
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        output[key] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90)),
        }
    return output


@torch.inference_mode()
def evaluate(args) -> dict:
    from gaussian_renderer import render_gsplat
    from scene import Scene
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
    from utils.loss_utils import ssim

    model_root = Path(args.model_root).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    manifest_path = model_root / "rgb_prior_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["gaussian_type"] != args.gaussian_type:
        raise ValueError("manifest Gaussian type does not match evaluation")
    if int(manifest["sh_degree"]) != int(args.sh_degree):
        raise ValueError("manifest SH degree does not match evaluation")
    if bool(manifest.get("prior_training_used_feature_loss", True)):
        raise ValueError("quality evaluation requires an RGB-only prior")
    scene_manifest = json.loads(
        (source / "evaluation_scene_manifest.json").read_text()
    )
    if scene_manifest.get("evaluation_only") is not True:
        raise ValueError("source is not an evaluation-only camera scene")
    if scene_manifest.get("used_for_prior_training") is not False:
        raise ValueError("evaluation source was used for prior training")

    dataset = Namespace(
        model_path=str(model_root),
        source_path=str(source),
        images="images",
        eval=True,
        feature_type="sp",
        gaussian_type=args.gaussian_type,
        sh_degree=int(args.sh_degree),
        resolution=1,
        data_device="cpu",
        longest_edge=0,
        white_background=bool(manifest.get("white_background", False)),
        speedup=False,
    )
    gaussians = (
        GaussianModel_2dgs(args.sh_degree)
        if args.gaussian_type == "2dgs"
        else GaussianModel(args.sh_degree)
    )
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=int(args.iteration),
        shuffle=False,
        preload_cameras=False,
        load_test_cameras=True,
    )
    background = torch.full(
        (3,),
        1.0 if dataset.white_background else 0.0,
        dtype=torch.float32,
        device="cuda",
    )
    perceptual = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).cuda().eval()
    rows = []
    cameras = scene.getTestCameras()
    for index, camera in enumerate(tqdm(cameras, desc="prior quality")):
        if args.max_views and index >= args.max_views:
            break
        gt = camera.original_image[:3].cuda(non_blocking=True).clamp(0, 1)
        longest_edge = max(int(camera.image_width), int(camera.image_height))
        rendered = render_gsplat(
            camera,
            gaussians,
            background,
            rgb_only=True,
            longest_edge=longest_edge,
        )["render"].clamp(0, 1)
        mse = torch.mean((rendered - gt) ** 2).clamp_min(1e-12)
        psnr_db = -10.0 * torch.log10(mse)
        ssim_value = ssim(rendered[None], gt[None])
        perceptual.reset()
        perceptual.update(rendered[None], gt[None])
        lpips_value = perceptual.compute()
        rows.append(
            {
                "image_name": str(camera.image_name),
                "psnr_db": float(psnr_db.item()),
                "ssim": float(ssim_value.item()),
                "lpips": float(lpips_value.item()),
            }
        )
    expected = int(scene_manifest["test_image_count"])
    if not args.max_views and len(rows) != expected:
        raise ValueError(
            f"expected {expected} test views, evaluated {len(rows)}"
        )
    report = {
        "schema": "lafgs_off_the_shelf_prior_quality",
        "version": 1,
        "model_root": str(model_root),
        "evaluation_source": str(source),
        "gaussian_type": args.gaussian_type,
        "iteration": int(args.iteration),
        "full_resolution": True,
        "semantic_mask_used": False,
        "background": "white" if dataset.white_background else "black",
        "summary": _summary(rows),
        "per_view": rows,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--gaussian-type", choices=("3dgs", "2dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-views", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(evaluate(args)["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
