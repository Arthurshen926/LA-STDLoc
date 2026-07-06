#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_gsplat
from la_artifacts.detector import ArtifactDetector
from la_artifacts.repair import ArtifactRepair
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import seed_everything
from utils.loss_utils import ssim


def _resize_like(image, target_hw):
    if tuple(image.shape[-2:]) == tuple(target_hw):
        return image
    return F.interpolate(image[None], size=target_hw, mode="bilinear", align_corners=False)[0]


def _psnr(rendered, target):
    mse = (rendered - target).pow(2).mean().clamp_min(1e-12)
    return float((20.0 * torch.log10(torch.ones_like(mse) / torch.sqrt(mse))).item())


def _metrics(rendered, target):
    target = _resize_like(target, rendered.shape[-2:])
    diff = (rendered - target).abs()
    return {
        "psnr": _psnr(rendered, target),
        "ssim": float(ssim(rendered[None], target[None]).mean().item()),
        "l1": float(diff.mean().item()),
        "residual_frac_025": float((diff > 0.25).float().mean().item()),
    }


def _save_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((array * 255.0).round().astype("uint8")).save(path)


def main():
    parser = argparse.ArgumentParser(description="Audit non-destructive artifact repair by opacity suppression.")
    model = ModelParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_images", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min_opacity_multiplier", type=float, default=0.15)
    args = parser.parse_args()
    args.eval = False
    seed_everything(args.seed)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    detector = ArtifactDetector()
    repair = ArtifactRepair()
    repair.config.min_opacity_multiplier = float(args.min_opacity_multiplier)

    rows = []
    out = Path(args.output_dir)
    cameras = scene.getTrainCameras()[: int(args.max_images)]
    for camera in tqdm(cameras, desc="Repair audit"):
        with torch.no_grad():
            initial = render_gsplat(
                camera,
                gaussians,
                background,
                rgb_only=False,
                return_loc_meta=True,
                norm_feat_bf_render=dataset.norm_before_render,
                longest_edge=dataset.longest_edge,
                rasterize_mode="antialiased",
            )
            gt_rgb = _resize_like(camera.original_image.cuda(), initial["render"].shape[-2:])
            gt_feature = feature_extractor(camera.original_image.cuda()[None])["feature_map"][0]
            gt_feature = F.interpolate(
                gt_feature[None],
                size=initial["feature_map"].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
            gt_feature = F.normalize(gt_feature, p=2, dim=0)
            evidence = detector.detect(
                rendered_rgb=initial["render"],
                target_rgb=gt_rgb,
                rendered_feature=initial["feature_map"],
                target_feature=gt_feature,
                alpha=initial.get("alphas"),
            )
            gaussian_scores = detector.gaussian_scores_from_projected_map(
                initial["loc_visible_idx"],
                initial["loc_viewspace_points"],
                evidence.score_map,
                gaussian_count=gaussians.get_xyz.shape[0],
            )
            opacity_multiplier = repair.gaussian_opacity_multiplier(gaussian_scores).to(device=gaussians.get_xyz.device)
            repaired = render_gsplat(
                camera,
                gaussians,
                background,
                rgb_only=False,
                norm_feat_bf_render=dataset.norm_before_render,
                longest_edge=dataset.longest_edge,
                rasterize_mode="antialiased",
                opacity_multiplier=opacity_multiplier,
                loc_opacity_multiplier=opacity_multiplier,
            )
        before = _metrics(initial["render"], gt_rgb)
        after = _metrics(repaired["render"], gt_rgb)
        name = str(camera.image_name).replace("/", "__")
        _save_image(initial["render"], out / "images" / f"{name}_before.png")
        _save_image(repaired["render"], out / "images" / f"{name}_after.png")
        row = {
            "image_name": camera.image_name,
            "artifact_score_mean": evidence.summary["artifact_score_mean"],
            "artifact_score_p95": evidence.summary["artifact_score_p95"],
            "suppressed_gaussians": int((opacity_multiplier < 0.999).sum().item()),
            "before_psnr": before["psnr"],
            "after_psnr": after["psnr"],
            "before_ssim": before["ssim"],
            "after_ssim": after["ssim"],
            "before_residual_frac_025": before["residual_frac_025"],
            "after_residual_frac_025": after["residual_frac_025"],
        }
        rows.append(row)

    out.mkdir(parents=True, exist_ok=True)
    with (out / "repair_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["image_name"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "count": len(rows),
        "mean_before_psnr": sum(row["before_psnr"] for row in rows) / len(rows) if rows else 0.0,
        "mean_after_psnr": sum(row["after_psnr"] for row in rows) / len(rows) if rows else 0.0,
        "improved_psnr_count": sum(1 for row in rows if row["after_psnr"] > row["before_psnr"]),
    }
    with (out / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
