#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import torch

from localization_training.render_artifacts import (
    ArtifactThresholds,
    classify_artifact_severity,
    continuous_quality_weight,
    normalize_image_name,
)
from scripts.audit_render_artifacts import (
    _flatten_alpha,
    _resize_image_like,
    _severity_counts,
    psnr_value,
    render_camera_metrics,
    select_audit_cameras,
    subsample_cameras,
)


FIELDNAMES = [
    "scene",
    "variant",
    "iteration",
    "split",
    "sample_index",
    "image_name",
    "forward_m",
    "is_original_pose",
    "is_best_candidate",
    "quality_weight",
    "quality_gain_vs_original",
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


def parse_forward_offsets(value):
    offsets = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        offsets.append(max(0.0, float(item)))
    offsets.append(0.0)
    unique = sorted(set(offsets))
    return unique


def forward_pose_w2c(pose_w2c, distance):
    pose_w2c = torch.as_tensor(pose_w2c)
    c2w = torch.linalg.inv(pose_w2c)
    moved = c2w.clone()
    forward_world = c2w[:3, 2]
    moved[:3, 3] = moved[:3, 3] + forward_world * float(distance)
    return torch.linalg.inv(moved)


def _resolution_from_longest_edge(height, width, longest_edge):
    longest_edge = int(longest_edge or 0)
    if longest_edge <= 0:
        return int(height), int(width)
    longest = max(int(height), int(width), 1)
    if longest <= longest_edge:
        return int(height), int(width)
    scale = float(longest_edge) / float(longest)
    return max(1, int(round(int(height) * scale))), max(1, int(round(int(width) * scale)))


def _pose_render_metrics(camera, gaussians, background, pose_w2c, longest_edge=640):
    from gaussian_renderer import render_from_pose_gsplat
    from utils.loss_utils import ssim

    height, width = _resolution_from_longest_edge(
        getattr(camera, "image_height", camera.original_image.shape[-2]),
        getattr(camera, "image_width", camera.original_image.shape[-1]),
        longest_edge,
    )
    with torch.no_grad():
        render_pkg = render_from_pose_gsplat(
            gaussians,
            pose_w2c,
            camera.FoVx,
            camera.FoVy,
            width,
            height,
            bg_color=background,
            render_mode="RGB+ED",
            rgb_only=True,
            rasterize_mode="antialiased",
        )
    rendered = torch.clamp(render_pkg["render"], 0.0, 1.0).contiguous()
    gt = camera.original_image.to(device=rendered.device, dtype=rendered.dtype)
    gt = _resize_image_like(gt, rendered.shape[-2:]).contiguous()
    diff = (rendered - gt).abs()
    alpha = _flatten_alpha(render_pkg.get("alphas"))
    if alpha is not None:
        alpha = alpha.to(device=rendered.device, dtype=rendered.dtype)
    return {
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


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_audit(args):
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
    variant = args.variant or f"{Path(dataset.model_path).name}_{args.iteration}_pose_forward"
    background = torch.tensor(
        [1, 1, 1] if dataset.white_background else [0, 0, 0],
        dtype=torch.float32,
        device="cuda",
    )
    thresholds = ArtifactThresholds.from_args(args)
    offsets = parse_forward_offsets(args.forward_offsets)

    rows = []
    improved = 0
    for sample_index, camera in enumerate(tqdm(cameras, desc=f"Pose-forward audit {scene_name}:{args.split}")):
        base_pose = camera.world_view_transform.transpose(0, 1).cuda()
        candidate_rows = []
        original_quality = None
        for forward_m in offsets:
            if forward_m == 0.0:
                metrics = render_camera_metrics(
                    camera,
                    gaussians,
                    background,
                    longest_edge=args.longest_edge,
                )
                pose = base_pose
            else:
                pose = forward_pose_w2c(base_pose, forward_m)
                metrics = _pose_render_metrics(
                    camera,
                    gaussians,
                    background,
                    pose,
                    longest_edge=args.longest_edge,
                )
            row = {
                "scene": scene_name,
                "variant": variant,
                "iteration": int(args.iteration),
                "split": args.split,
                "sample_index": sample_index,
                "image_name": normalize_image_name(getattr(camera, "image_name", "")),
                "forward_m": float(forward_m),
                "is_original_pose": 1 if forward_m == 0.0 else 0,
                **metrics,
            }
            row["gate_severity"] = classify_artifact_severity(row, thresholds)
            row["quality_weight"] = continuous_quality_weight(
                row,
                thresholds=thresholds,
                min_weight=args.quality_min_weight,
                power=args.quality_power,
            )
            if forward_m == 0.0:
                original_quality = row["quality_weight"]
            candidate_rows.append(row)
            _ = pose
        if original_quality is None:
            original_quality = candidate_rows[0]["quality_weight"]
        best = max(
            candidate_rows,
            key=lambda row: (
                float(row["quality_weight"]),
                float(row["psnr_mean_matched"]),
                float(row["ssim"]),
            ),
        )
        if float(best["forward_m"]) > 0.0 and float(best["quality_weight"]) > float(original_quality):
            improved += 1
        for row in candidate_rows:
            row["is_best_candidate"] = 1 if row is best else 0
            row["quality_gain_vs_original"] = float(row["quality_weight"]) - float(original_quality)
            rows.append(row)

    _write_csv(args.output_csv, rows)
    best_rows = [row for row in rows if int(row.get("is_best_candidate", 0)) == 1]
    summary = {
        "scene": scene_name,
        "variant": variant,
        "iteration": int(args.iteration),
        "split": args.split,
        "images": len(cameras),
        "forward_offsets": offsets,
        "candidate_rows": len(rows),
        "improved_images": improved,
        "improved_fraction": float(improved) / float(len(cameras)) if cameras else 0.0,
        "best_severity_counts": _severity_counts(best_rows),
        "thresholds": thresholds.to_dict(),
        "output_csv": args.output_csv,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, allow_nan=True) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=True))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Audit small camera-forward render candidates for artifact mitigation.")
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
    parser.add_argument("--forward_offsets", default="0,0.05,0.10,0.20")
    parser.add_argument("--quality_min_weight", type=float, default=0.25)
    parser.add_argument("--quality_power", type=float, default=1.0)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", default="")
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
