#!/usr/bin/env python3
"""Render leakage-safe tri-state training data for the V8 scene detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.scene_detector_supervision import (
    build_tri_state_heatmap,
    project_visible_clean_anchors,
    spatially_balance_points,
)
from evidence.v7_render_certificate import CertificateThresholds, render_quality_pixel_masks
from evidence.virtual_camera_registry import resolve_virtual_camera_registry
from priors.models import GaussianModel2D
from priors.rendering import render_from_pose_gsplat


SCHEMA = "lafgs_v8_scene_detector_dataset"


def _family(name: str) -> str:
    return str(name).split("/", 1)[0]


def _family_split(names: list[str]) -> dict[str, str]:
    families = sorted({_family(name) for name in names})
    if len(families) < 3:
        raise ValueError("at least three camera pose families are required")
    assignment = {}
    for index, family in enumerate(families):
        remainder = index % 5
        assignment[family] = "validation" if remainder == 3 else "confirmation" if remainder == 4 else "train"
    if set(assignment.values()) != {"train", "validation", "confirmation"}:
        # Deterministic fallback for scenes with three or four families.
        assignment = {family: "train" for family in families}
        assignment[families[-2]] = "validation"
        assignment[families[-1]] = "confirmation"
    return assignment


def _scaled_size(width: int, height: int, longest_edge: int) -> tuple[int, int]:
    scale = min(1.0, float(longest_edge) / max(width, height))
    return max(8, int(round(width * scale))), max(8, int(round(height * scale)))


def _intrinsic(width: int, height: int, fov_x: float, fov_y: float, device) -> torch.Tensor:
    import math
    return torch.tensor([
        [width / (2 * math.tan(fov_x / 2)), 0.0, width / 2],
        [0.0, height / (2 * math.tan(fov_y / 2)), height / 2],
        [0.0, 0.0, 1.0],
    ], device=device)


def _atomic_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--anchor-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "confirmation"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--maximum-views", type=int, default=0)
    parser.add_argument("--longest-edge", type=int, default=960)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")

    cache = torch.load(args.observation_cache, map_location="cpu", weights_only=False)
    anchor_map = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    evidence = torch.load(args.anchor_evidence, map_location="cpu", weights_only=False)
    names = list(anchor_map["v6_mapping_query_names"])
    if names != list(cache["queries"]):
        raise ValueError("map/cache camera registry mismatch")
    clean_metadata = anchor_map.get("v8_clean_anchor")
    source_rows = (
        torch.as_tensor(clean_metadata["source_anchor_rows"]).long()
        if isinstance(clean_metadata, dict)
        else torch.arange(torch.as_tensor(anchor_map["anchor_ids"]).numel())
    )
    if source_rows.numel() != torch.as_tensor(anchor_map["anchor_ids"]).numel():
        raise ValueError("clean-map source row lineage is invalid")
    assignment = _family_split(names)
    indices = [i for i, name in enumerate(names) if assignment[_family(name)] == args.split]
    if args.maximum_views > 0 and len(indices) > args.maximum_views:
        positions = torch.linspace(0, len(indices) - 1, steps=args.maximum_views).round().long().tolist()
        indices = [indices[i] for i in positions]
    indices = indices[args.shard_index::args.shard_count]

    dataset = ColmapDataset(args.dataset, images="processed")
    cameras = resolve_virtual_camera_registry(dataset.split("mapping"), cache["virtual_camera_registry"])
    device = torch.device(args.device)
    model = GaussianModel2D(3, device=device)
    model.load_ply(args.gaussian_ply, loc_feature_dim=0)
    model = model.to(device).eval()
    xyz = torch.as_tensor(anchor_map["anchor_xyz"], device=device).float()
    source_rows = source_rows.to(device)
    clean = (
        (torch.as_tensor(evidence["valid_observation_count"], device=device)[source_rows] >= 3)
        & (torch.as_tensor(evidence["valid_view_family_count"], device=device)[source_rows] >= 2)
    )
    reliability = torch.as_tensor(evidence["valid_observation_fraction"], device=device).float()[source_rows]
    thresholds = CertificateThresholds()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    for completed, query_index in enumerate(indices, 1):
        camera = cameras[query_index]
        width, height = _scaled_size(camera.width, camera.height, args.longest_edge)
        pose = torch.from_numpy(camera.pose_w2c).to(device=device, dtype=torch.float32)
        package = render_from_pose_gsplat(
            model, pose, camera.fov_x, camera.fov_y, width, height,
            bg_color=torch.zeros(3, device=device), render_mode="RGB+ED",
            rgb_only=True, rasterize_mode="antialiased",
        )
        rgb = package["render"][:3].float().clamp(0, 1)
        alpha = package.get("alphas", package.get("rend_alpha"))
        depth = package["depth"]
        quality = render_quality_pixel_masks(
            rgb=rgb, alpha=alpha, depth=depth, distortion=package.get("rend_dist"),
            thresholds=thresholds,
        )
        K = _intrinsic(width, height, camera.fov_x, camera.fov_y, device)
        uv, anchor_rows = project_visible_clean_anchors(
            anchor_xyz=xyz, clean_anchor_mask=clean, intrinsic=K, pose_w2c=pose,
            rendered_depth=depth, valid_pixel_mask=quality["valid"],
        )
        selected = spatially_balance_points(
            uv, reliability[anchor_rows], image_hw=(height, width), grid_hw=(24, 32)
        )
        labels = build_tri_state_heatmap(
            image_hw=(height, width), positive_uv=uv[selected],
            invalid_pixel_mask=quality["invalid"], uncertain_pixel_mask=quality["uncertain"],
        )
        name_hash = hashlib.sha256(names[query_index].encode()).hexdigest()[:16]
        record_path = args.output / f"{args.split}_{query_index:04d}_{name_hash}.pt"
        payload = {
            "schema": SCHEMA, "version": 1, "split": args.split,
            "uses_source_mapping_rgb": False, "uses_real_training_rgb": False,
            "uses_test_rgb": False, "query_index": query_index,
            "query_name": names[query_index], "pose_family": _family(names[query_index]),
            "rgb_u8": (rgb.cpu() * 255).round().to(torch.uint8),
            "labels": labels.cpu(), "intrinsic": K.cpu(), "pose_w2c": pose.cpu(),
            "rendered_depth": depth.float().cpu(),
            "positive_uv": uv[selected].cpu(), "positive_anchor_rows": anchor_rows[selected].cpu(),
        }
        _atomic_save(payload, record_path)
        records.append(str(record_path.resolve()))
        if completed % 10 == 0 or completed == len(indices):
            print(f"{args.split} shard {args.shard_index}: {completed}/{len(indices)}", flush=True)
    manifest = {
        "schema": f"{SCHEMA}_manifest", "version": 1, "split": args.split,
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "record_count": len(records), "records": records,
        "pose_family_assignment": assignment, "uses_test_rgb": False,
        "anchor_map": str(args.anchor_map.resolve()), "anchor_map_sha256": sha256_file(args.anchor_map),
        "anchor_evidence": str(args.anchor_evidence.resolve()), "anchor_evidence_sha256": sha256_file(args.anchor_evidence),
        "timing_seconds": time.perf_counter() - started,
    }
    manifest_path = args.output / f"manifest_{args.split}_shard{args.shard_index}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("split", "record_count", "timing_seconds")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
