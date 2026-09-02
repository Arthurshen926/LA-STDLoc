#!/usr/bin/env python3
"""Render one shard of mapping cameras and audit cached rows with V7 V2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.v7_render_certificate import (
    CertificateThresholds,
    certify_v7_render,
    extreme_distortion_row_mask,
)
from evidence.virtual_camera_registry import resolve_virtual_camera_registry
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rendering import render_from_pose_gsplat


SCHEMA = "lafgs_v7_mapping_render_quality_audit_shard"


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument(
        "--gaussian-type", choices=("2dgs", "3dgs"), default="2dgs"
    )
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument(
        "--anchor-map",
        type=Path,
        help=(
            "Optional prior full-map registry cross-check. Fresh high-capacity "
            "scenes instead bind the cache registry directly to the dataset."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    if args.output.exists():
        raise FileExistsError(args.output)

    cache_path = args.observation_cache.resolve()
    map_path = args.anchor_map.resolve() if args.anchor_map is not None else None
    prior_path = args.gaussian_ply.resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if (
        cache.get("schema") != "render_observation_cache_v2"
        or cache.get("uses_source_mapping_rgb") is not False
        or cache.get("uses_test_queries") is not False
    ):
        raise ValueError("audit requires the frozen rendered mapping cache")
    names = list(cache["queries"])
    if map_path is not None:
        anchor_map = torch.load(map_path, map_location="cpu", weights_only=False)
        if list(anchor_map["v6_mapping_query_names"]) != names:
            raise ValueError("map and observation cache query registries differ")
        del anchor_map

    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    cameras = resolve_virtual_camera_registry(mapping, cache["virtual_camera_registry"])
    if [camera.image_name for camera in cameras] != names:
        raise ValueError("resolved camera registry differs from map lineage")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("mapping render-quality audit requires CUDA")
    model_class = GaussianModel2D if args.gaussian_type == "2dgs" else GaussianModel3D
    model = model_class(args.sh_degree, device=device)
    model.load_ply(prior_path, loc_feature_dim=0)
    model = model.to(device).eval()
    thresholds = CertificateThresholds()
    indices = list(range(args.shard_index, len(cameras), args.shard_count))
    records = []
    background = torch.ones(3, device=device) if args.white_background else torch.zeros(3, device=device)
    started = time.perf_counter()
    for local_row, query_index in enumerate(indices):
        camera = cameras[query_index]
        cached = cache["queries"][names[query_index]]
        pose = torch.from_numpy(camera.pose_w2c).to(device=device, dtype=torch.float32)
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
        rgb = package["render"].float()[:3].clamp(0.0, 1.0)
        alpha = package.get("alphas", package.get("rend_alpha"))
        depth = package.get("depth")
        if alpha is None or depth is None:
            raise ValueError("renderer did not return alpha/depth")
        keypoints = torch.as_tensor(
            cached["native_keypoints"], device=device, dtype=torch.float32
        )
        distortion = package.get("rend_dist")
        artifact_rows = (
            None
            if distortion is None
            else extreme_distortion_row_mask(
                distortion,
                keypoints,
                mad_multiplier=thresholds.distortion_mad_multiplier,
                tail_quantile=thresholds.distortion_tail_quantile,
            )
        )
        certificate = certify_v7_render(
            rgb=rgb,
            alpha=alpha,
            depth=depth,
            keypoints=keypoints,
            nearest_mapping_distance_m=0.0,
            median_adjacent_baseline_m=1.0,
            source_family_support=1,
            artifact_row_mask=artifact_rows,
            thresholds=thresholds,
        )
        reasons = {
            key: value.detach().cpu() for key, value in certificate["row_reasons"].items()
        }
        records.append(
            {
                "query_index": query_index,
                "query_name": names[query_index],
                "keypoint_count": int(keypoints.shape[0]),
                "row_valid": certificate["row_valid"].detach().cpu(),
                "row_structure_supported": (~reasons["low_rgb_structure_support"]),
                "row_reasons": reasons,
                "signals": certificate["signals"],
                "certificate_decision": certificate["decision"],
            }
        )
        if (local_row + 1) % args.progress_interval == 0 or local_row + 1 == len(indices):
            elapsed = time.perf_counter() - started
            print(
                f"shard {args.shard_index}: {local_row + 1}/{len(indices)} "
                f"views, {elapsed:.1f}s",
                flush=True,
            )

    payload = {
        "schema": SCHEMA,
        "version": 1,
        "status": "PASS",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "map_mutation_count": 0,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "mapping_query_count": len(cameras),
        "record_count": len(records),
        "thresholds": thresholds.__dict__,
        "input": {
            "dataset": str(args.dataset.resolve()),
            "gaussian_ply": str(prior_path),
            "gaussian_ply_sha256": sha256_file(prior_path),
            "gaussian_type": args.gaussian_type,
            "sh_degree": int(args.sh_degree),
            "background": "white" if args.white_background else "black",
            "observation_cache": str(cache_path),
            "observation_cache_sha256": sha256_file(cache_path),
            "anchor_map": str(map_path) if map_path is not None else None,
            "anchor_map_sha256": (
                sha256_file(map_path) if map_path is not None else None
            ),
            "query_registry_authority": (
                "prior_full_map_plus_cache_plus_dataset"
                if map_path is not None
                else "observation_cache_plus_dataset_mapping_split"
            ),
        },
        "timing_seconds": time.perf_counter() - started,
        "records": records,
    }
    _atomic_save(payload, args.output)
    manifest = {
        "schema": f"{SCHEMA}_manifest",
        "version": 1,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "record_count": len(records),
        "timing_seconds": payload["timing_seconds"],
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
