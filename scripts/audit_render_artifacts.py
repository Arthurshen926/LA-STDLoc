#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

from localization_training.render_artifacts import (
    ArtifactThresholds,
    candidate_rows,
    classify_artifact_severity,
    comma_set,
    local_artifact_weight_map,
    normalize_image_name,
)


AUDIT_FIELDNAMES = [
    "scene",
    "variant",
    "iteration",
    "split",
    "sample_index",
    "image_name",
    "image_width",
    "image_height",
    "psnr",
    "psnr_mean_matched",
    "ssim",
    "l1",
    "residual_frac_025",
    "residual_frac_040",
    "alpha_mean",
    "alpha_cov_05",
    "alpha_cov_09",
    "render_mean",
    "gt_mean",
    "mean_abs_bias",
    "gate_severity",
]


CANDIDATE_FIELDNAMES = [
    "scene",
    "split",
    "image_name",
    "psnr",
    "psnr_mean_matched",
    "ssim",
    "alpha_cov_05",
    "residual_frac_025",
    "mean_abs_bias",
    "gate_severity",
]


REGION_WEIGHT_FIELDNAMES = CANDIDATE_FIELDNAMES + [
    "region_weight_path",
    "region_weight_min",
    "region_weight_mean",
    "region_weight_weighted_frac",
]


def select_audit_cameras(
    train_cameras,
    test_cameras,
    split,
    support_query_split=False,
    query_holdout_ratio=0.2,
    query_split_seed=2025,
    query_split_mode="sequence_block",
    support_query_sort_by_name=False,
):
    split = str(split)
    train_cameras = list(train_cameras)
    test_cameras = list(test_cameras)
    if support_query_sort_by_name:
        train_cameras = sorted(
            train_cameras,
            key=lambda camera: normalize_image_name(getattr(camera, "image_name", "")),
        )

    if split == "final_test_sample":
        return test_cameras
    if split == "train_sample":
        return train_cameras

    if split in {"heldout_query_sample", "support_train_sample"}:
        if support_query_split:
            from localization_training.episode_sampler import split_support_query_cameras

            support_cameras, query_cameras = split_support_query_cameras(
                train_cameras,
                query_ratio=query_holdout_ratio,
                seed=query_split_seed,
                mode=query_split_mode,
            )
        else:
            support_cameras, query_cameras = train_cameras, train_cameras
        return query_cameras if split == "heldout_query_sample" else support_cameras

    raise ValueError(f"Unknown audit split: {split}")


def subsample_cameras(cameras, max_images=0, sample_stride=1, sample_seed=0):
    cameras = list(cameras)
    stride = max(1, int(sample_stride))
    if stride > 1:
        cameras = cameras[::stride]
    max_images = int(max_images or 0)
    if max_images > 0 and len(cameras) > max_images:
        rng = random.Random(int(sample_seed))
        indices = sorted(rng.sample(range(len(cameras)), max_images))
        cameras = [cameras[idx] for idx in indices]
    return cameras


def _flatten_alpha(alpha):
    if alpha is None:
        return None
    while alpha.dim() > 2:
        if alpha.shape[0] == 1:
            alpha = alpha.squeeze(0)
        elif alpha.shape[-1] == 1:
            alpha = alpha.squeeze(-1)
        else:
            break
    if alpha.dim() == 3 and alpha.shape[0] == 1:
        alpha = alpha.squeeze(0)
    return alpha


def _resize_image_like(image, target_hw):
    if tuple(image.shape[-2:]) == tuple(target_hw):
        return image
    import torch.nn.functional as F

    return F.interpolate(
        image[None],
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )[0]


def psnr_value(rendered, target):
    import torch

    mse = ((rendered - target) ** 2).reshape(1, -1).mean(dim=1)
    if torch.any(mse <= 0):
        return float("inf")
    value = 20.0 * torch.log10(torch.ones_like(mse) / torch.sqrt(mse))
    return float(value.mean().item())


def render_camera_metrics(
    camera,
    gaussians,
    background,
    longest_edge=640,
    region_weight_size=0,
    region_weight_min=0.25,
    region_weight_power=1.0,
):
    import torch

    from gaussian_renderer import render_gsplat
    from utils.loss_utils import ssim

    with torch.no_grad():
        render_pkg = render_gsplat(
            camera,
            gaussians,
            background,
            rgb_only=True,
            longest_edge=longest_edge,
        )
    rendered = torch.clamp(render_pkg["render"], 0.0, 1.0).contiguous()
    gt = camera.original_image.to(device=rendered.device, dtype=rendered.dtype)
    gt = _resize_image_like(gt, rendered.shape[-2:]).contiguous()
    diff = (rendered - gt).abs()
    alpha = _flatten_alpha(render_pkg.get("alphas"))
    if alpha is not None:
        alpha = alpha.to(device=rendered.device, dtype=rendered.dtype)

    metrics = {
        "image_width": int(getattr(camera, "image_width", rendered.shape[-1])),
        "image_height": int(getattr(camera, "image_height", rendered.shape[-2])),
        "psnr": psnr_value(rendered, gt),
        "psnr_mean_matched": psnr_value(rendered, gt),
        "ssim": float(ssim(rendered[None], gt[None]).mean().item()),
        "l1": float(diff.mean().item()),
        "residual_frac_025": float((diff > 0.25).float().mean().item()),
        "residual_frac_040": float((diff > 0.40).float().mean().item()),
        "alpha_mean": float(alpha.mean().item()) if alpha is not None else float("nan"),
        "alpha_cov_05": float((alpha > 0.05).float().mean().item()) if alpha is not None else float("nan"),
        "alpha_cov_09": float((alpha > 0.90).float().mean().item()) if alpha is not None else float("nan"),
        "render_mean": float(rendered.mean().item()),
        "gt_mean": float(gt.mean().item()),
        "mean_abs_bias": float((rendered.mean(dim=(1, 2)) - gt.mean(dim=(1, 2))).abs().mean().item()),
    }
    if int(region_weight_size or 0) > 0:
        metrics["_region_weight_map"] = local_artifact_weight_map(
            rendered,
            gt,
            alpha=alpha,
            output_size=int(region_weight_size),
            min_weight=float(region_weight_min),
            power=float(region_weight_power),
        ).detach().cpu()
    return metrics


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _severity_counts(rows):
    counts = {"none": 0, "mild": 0, "severe": 0}
    for row in rows:
        severity = row.get("gate_severity", "none")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _region_weight_relative_path(scene_name, split, image_name):
    image_path = Path(normalize_image_name(image_name))
    return Path(scene_name) / split / image_path.with_suffix(".pt")


def _write_region_weight_map(root, scene_name, split, image_name, weight_map, row):
    import torch

    rel_path = _region_weight_relative_path(scene_name, split, image_name)
    out_path = Path(root) / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weight": weight_map.to(dtype=torch.float32).cpu(),
            "scene": scene_name,
            "split": split,
            "image_name": normalize_image_name(image_name),
            "image_width": int(row.get("image_width", 0)),
            "image_height": int(row.get("image_height", 0)),
        },
        out_path,
    )
    return str(rel_path)


def run_audit(args):
    import torch
    from tqdm import tqdm

    from arguments import ModelParams
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from utils.general_utils import seed_everything

    seed_everything(args.train_seed)

    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    model_args = argparse.Namespace(**vars(args))
    dataset = model_params.extract(model_args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration)

    cameras = select_audit_cameras(
        scene.getTrainCameras(),
        scene.getTestCameras(),
        split=args.split,
        support_query_split=args.support_query_split,
        query_holdout_ratio=args.query_holdout_ratio,
        query_split_seed=args.query_split_seed,
        query_split_mode=args.query_split_mode,
        support_query_sort_by_name=args.support_query_sort_by_name,
    )
    cameras = subsample_cameras(
        cameras,
        max_images=args.max_images,
        sample_stride=args.sample_stride,
        sample_seed=args.sample_seed,
    )
    scene_name = args.scene_name or os.path.basename(os.path.normpath(dataset.source_path))
    variant = args.variant or f"{Path(dataset.model_path).name}_{args.iteration}"
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    thresholds = ArtifactThresholds.from_args(args)

    rows = []
    region_rows = []
    region_severities = comma_set(args.region_weight_severities)
    for sample_index, camera in enumerate(tqdm(cameras, desc=f"Render audit {scene_name}:{args.split}")):
        metrics = render_camera_metrics(
            camera,
            gaussians,
            background,
            longest_edge=args.longest_edge,
            region_weight_size=args.region_weight_size,
            region_weight_min=args.region_weight_min,
            region_weight_power=args.region_weight_power,
        )
        region_weight_map = metrics.pop("_region_weight_map", None)
        row = {
            "scene": scene_name,
            "variant": variant,
            "iteration": int(args.iteration),
            "split": args.split,
            "sample_index": sample_index,
            "image_name": normalize_image_name(getattr(camera, "image_name", "")),
            **metrics,
        }
        row["gate_severity"] = classify_artifact_severity(row, thresholds)
        rows.append(row)
        if (
            region_weight_map is not None
            and args.region_weight_dir
            and row["gate_severity"] in region_severities
        ):
            rel_path = _write_region_weight_map(
                args.region_weight_dir,
                scene_name,
                args.split,
                row["image_name"],
                region_weight_map,
                row,
            )
            weighted_frac = float((region_weight_map < 0.999).float().mean().item())
            region_rows.append(
                {
                    **{key: row.get(key) for key in CANDIDATE_FIELDNAMES},
                    "region_weight_path": rel_path,
                    "region_weight_min": float(region_weight_map.min().item()),
                    "region_weight_mean": float(region_weight_map.mean().item()),
                    "region_weight_weighted_frac": weighted_frac,
                }
            )

    _write_csv(args.output_csv, rows, AUDIT_FIELDNAMES)
    candidates = candidate_rows(rows, severities=args.candidate_severities, thresholds=thresholds)
    if args.candidate_csv:
        _write_csv(args.candidate_csv, candidates, CANDIDATE_FIELDNAMES)
    if args.region_weight_manifest:
        _write_csv(args.region_weight_manifest, region_rows, REGION_WEIGHT_FIELDNAMES)

    summary = {
        "scene": scene_name,
        "variant": variant,
        "iteration": int(args.iteration),
        "split": args.split,
        "images": len(rows),
        "candidate_images": len(candidates),
        "severity_counts": _severity_counts(rows),
        "candidate_severities": sorted(comma_set(args.candidate_severities)),
        "thresholds": thresholds.to_dict(),
        "output_csv": args.output_csv,
        "candidate_csv": args.candidate_csv,
        "region_weight_dir": args.region_weight_dir,
        "region_weight_manifest": args.region_weight_manifest,
        "region_weight_maps": len(region_rows),
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, allow_nan=True) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Audit rendered RGB quality and generate artifact filter candidates.")
    from arguments import ModelParams

    ModelParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=["heldout_query_sample", "support_train_sample", "train_sample", "final_test_sample"],
        default="heldout_query_sample",
    )
    parser.add_argument("--support_query_split", action="store_true", default=False)
    parser.add_argument("--query_holdout_ratio", type=float, default=0.2)
    parser.add_argument("--query_split_seed", type=int, default=2025)
    parser.add_argument("--query_split_mode", choices=["random", "sequence_block", "temporal_block"], default="sequence_block")
    parser.add_argument("--support_query_sort_by_name", action="store_true", default=False)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--scene_name", default="")
    parser.add_argument("--variant", default="")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--candidate_csv", default="")
    parser.add_argument("--output_json", default="")
    parser.add_argument("--candidate_severities", default="mild,severe")
    parser.add_argument("--region_weight_dir", default="")
    parser.add_argument("--region_weight_manifest", default="")
    parser.add_argument("--region_weight_size", type=int, default=0)
    parser.add_argument("--region_weight_min", type=float, default=0.25)
    parser.add_argument("--region_weight_power", type=float, default=1.0)
    parser.add_argument("--region_weight_severities", default="mild,severe")
    parser.add_argument("--severe_psnr", type=float, default=13.5)
    parser.add_argument("--severe_ssim", type=float, default=0.42)
    parser.add_argument("--severe_residual", type=float, default=0.18)
    parser.add_argument("--mild_psnr", type=float, default=15.5)
    parser.add_argument("--mild_ssim", type=float, default=0.56)
    parser.add_argument("--mild_residual", type=float, default=0.10)
    parser.add_argument("--mild_alpha_cov", type=float, default=0.85)
    parser.add_argument("--mild_abs_bias", type=float, default=0.04)
    return parser


def main():
    args = build_parser().parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
