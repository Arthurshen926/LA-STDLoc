#!/usr/bin/env python3
"""Audit Gaussian-depth agreement as a soft Track prior on mapping views."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from data.datasets import ColmapDataset
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rendering import render_from_pose_gsplat


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def run(args) -> dict:
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    statistics = torch.load(args.statistics, map_location="cpu", weights_only=False)
    if cache_payload.get("uses_source_mapping_rgb") is not False:
        raise ValueError("depth audit cache is not rendered-RGB-only")
    if cache_payload.get("uses_test_queries") is not False:
        raise ValueError("depth audit cache contains test queries")
    cache = cache_payload["queries"]
    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    camera_by_name = {camera.image_name: camera for camera in mapping}
    if set(cache) != set(camera_by_name):
        raise ValueError("rendered cache and mapping camera registry differ")
    model = (
        GaussianModel2D(args.sh_degree)
        if args.gaussian_type == "2dgs"
        else GaussianModel3D(args.sh_degree)
    )
    model.load_ply(args.gaussian_ply)
    model = model.cuda().eval()
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cuda()
    count = int(xyz.shape[0])
    residuals: list[list[float]] = [[] for _ in range(count)]
    valid_observations = 0
    for completed, (name, cached) in enumerate(cache.items(), start=1):
        camera = camera_by_name[name]
        pose = torch.as_tensor(cached["pose_w2c"]).cuda().float()
        package = render_from_pose_gsplat(
            model,
            pose,
            camera.fov_x,
            camera.fov_y,
            camera.width,
            camera.height,
            bg_color=torch.zeros(3, device="cuda"),
            render_mode="RGB+ED",
            rgb_only=True,
            rasterize_mode="antialiased",
        )
        rendered_depth = torch.as_tensor(package["depth"]).squeeze()
        rendered_alpha = torch.as_tensor(
            package.get("alphas", package.get("rend_alpha"))
        ).squeeze()
        intrinsic = torch.as_tensor(cached["native_K"]).cuda().float()
        camera_xyz = xyz @ pose[:3, :3].T + pose[:3, 3]
        depth = camera_xyz[:, 2]
        projected = camera_xyz @ intrinsic.T
        uv = projected[:, :2] / depth[:, None].clamp_min(1e-8)
        pixel = torch.round(uv - 0.5).long()
        valid = (
            torch.isfinite(uv).all(dim=1)
            & torch.isfinite(depth)
            & (depth > 1e-5)
            & (pixel[:, 0] >= 0)
            & (pixel[:, 0] < camera.width)
            & (pixel[:, 1] >= 0)
            & (pixel[:, 1] < camera.height)
        )
        rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if rows.numel():
            sampled_depth = rendered_depth[pixel[rows, 1], pixel[rows, 0]]
            sampled_alpha = rendered_alpha[pixel[rows, 1], pixel[rows, 0]]
            legal = (
                torch.isfinite(sampled_depth)
                & (sampled_depth > 0)
                & torch.isfinite(sampled_alpha)
                & (sampled_alpha >= float(args.alpha_minimum))
            )
            rows = rows[legal]
            value = (depth[rows] - sampled_depth[legal]).abs().cpu()
            for anchor, residual in zip(rows.tolist(), value.tolist()):
                residuals[int(anchor)].append(float(residual))
            valid_observations += int(rows.numel())
        if completed % 100 == 0 or completed == len(cache):
            print(
                json.dumps(
                    {
                        "completed_queries": completed,
                        "valid_depth_observations": valid_observations,
                    }
                ),
                flush=True,
            )
    observation_count = torch.as_tensor([len(values) for values in residuals]).long()
    median = torch.full((count,), float("nan"))
    p90 = torch.full((count,), float("nan"))
    for anchor, values in enumerate(residuals):
        if not values:
            continue
        tensor = torch.as_tensor(values)
        median[anchor] = tensor.median()
        p90[anchor] = torch.quantile(tensor, 0.9)
    counters = statistics["counters"]
    harmful = torch.as_tensor(counters["harmful_inlier_count"]).float()
    false = torch.as_tensor(counters["false_attractor_count"]).float()
    clean = torch.as_tensor(counters["clean_inlier_count"]).float()
    known = torch.isfinite(median) & (observation_count >= args.minimum_observations)
    thresholds = [0.02, 0.05, 0.10, 0.20]
    strata = []
    for threshold in thresholds:
        high = known & (median > threshold)
        low = known & ~high
        strata.append(
            {
                "median_residual_threshold_m": threshold,
                "high_anchor_count": int(high.sum()),
                "low_anchor_count": int(low.sum()),
                "high_false_per_anchor": float(false[high].mean())
                if bool(high.any())
                else None,
                "low_false_per_anchor": float(false[low].mean())
                if bool(low.any())
                else None,
                "high_harmful_per_anchor": float(harmful[high].mean())
                if bool(high.any())
                else None,
                "low_harmful_per_anchor": float(harmful[low].mean())
                if bool(low.any())
                else None,
                "high_clean_per_anchor": float(clean[high].mean())
                if bool(high.any())
                else None,
                "low_clean_per_anchor": float(clean[low].mean())
                if bool(low.any())
                else None,
            }
        )
    output = {
        "schema": "lafgs_rendered_track_gaussian_depth_soft_prior_audit",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_gaussian_depth_as_hard_filter": False,
        "observation_count": observation_count,
        "median_absolute_depth_residual_m": median,
        "p90_absolute_depth_residual_m": p90,
        "known_anchor_count": int(known.sum()),
        "valid_depth_observation_count": valid_observations,
        "strata": strata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    _atomic_save(output, args.output)
    return {
        key: value
        for key, value in output.items()
        if key
        not in {
            "observation_count",
            "median_absolute_depth_residual_m",
            "p90_absolute_depth_residual_m",
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), default="2dgs")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-minimum", type=float, default=0.01)
    parser.add_argument("--minimum-observations", type=int, default=3)
    args = parser.parse_args()
    for field in (
        "dataset",
        "gaussian_ply",
        "anchor_map",
        "query_cache",
        "statistics",
        "output",
    ):
        setattr(args, field, getattr(args, field).resolve())
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
