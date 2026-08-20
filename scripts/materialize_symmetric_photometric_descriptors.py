#!/usr/bin/env python3
"""Re-describe a frozen Gaussian-rendered map with symmetric photometrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import torch

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from features.extractor import FeatureExtractor
from features.photometric import canonicalize_image, percentile_grayscale_contract
from features.superpoint import SUPERPOINT_WEIGHT_SHA256, resolve_superpoint_weights, sample_descriptors
from map_learning.metric import SharedLowRankMetric
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rendering import render_from_pose_gsplat
from scripts.materialize_mapping_rgb_descriptors import (
    TOPOLOGY_FIELDS,
    fuse_frozen_rows,
    validate_frozen_inputs,
)
from scripts.probe_rendered_rgb_track_map import _intrinsic


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def materialize(args) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(int(args.cpu_threads))
    source_map = torch.load(args.selected_map, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    source_cache = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    names, mapping_indices = validate_frozen_inputs(source_map, payload, source_cache)
    replay = fuse_frozen_rows(source_map, payload, source_cache, trim_fraction=args.descriptor_trim_fraction)
    if not torch.equal(replay, torch.as_tensor(source_map["anchor_features"]).float()):
        raise ValueError("source descriptor fusion does not reproduce the frozen map")
    config = source_cache.get("configuration", {})
    if config.get("gaussian_type") != args.gaussian_type or int(config.get("sh_degree", -1)) != args.sh_degree:
        raise ValueError("Gaussian rendering configuration differs from source cache")
    if int(config.get("nms_radius", -1)) != args.nms_radius:
        raise ValueError("SuperPoint NMS differs from source cache")

    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    cameras = [mapping[int(index)] for index in mapping_indices]
    if names != [camera.image_name for camera in cameras]:
        raise ValueError("mapping camera schedule differs from frozen evidence")
    model = GaussianModel2D(args.sh_degree) if args.gaussian_type == "2dgs" else GaussianModel3D(args.sh_degree)
    model.load_ply(args.gaussian_ply, loc_feature_dim=0)
    model = model.cuda().eval()
    extractor = FeatureExtractor("sp", nms_radius=args.nms_radius).cuda().eval()
    extractor.requires_grad_(False)
    contract = percentile_grayscale_contract()
    records = {}
    for index, camera in enumerate(cameras):
        source = source_cache["queries"][camera.image_name]
        pose = torch.from_numpy(camera.pose_w2c).float()
        if not torch.equal(pose, torch.as_tensor(source["pose_w2c"]).float()):
            raise ValueError(f"mapping pose differs for {camera.image_name}")
        if not torch.equal(_intrinsic(camera), torch.as_tensor(source["native_K"])):
            raise ValueError(f"mapping intrinsics differ for {camera.image_name}")
        package = render_from_pose_gsplat(
            model, pose.cuda(), camera.fov_x, camera.fov_y, camera.width, camera.height,
            bg_color=torch.zeros(3, device="cuda"), render_mode="RGB+ED",
            rgb_only=True, rasterize_mode="antialiased",
        )
        rendered = package["render"].float().clamp(0.0, 1.0)
        canonical = canonicalize_image(rendered, contract)
        dense, _ = extractor.detectAndComputeDense(canonical[None])
        keypoints = torch.as_tensor(source["native_keypoints"]).float().cuda()
        descriptors = sample_descriptors(keypoints[None], dense)[0].transpose(0, 1).cpu()
        records[camera.image_name] = {
            **source,
            "native_descriptors": descriptors,
            "source": "gaussian_render_symmetric_percentile_grayscale",
        }
        if (index + 1) % max(args.progress_interval, 1) == 0 or index + 1 == len(cameras):
            print(json.dumps({"completed_views": index + 1, "mapping_views": len(cameras)}), flush=True)
    canonical_cache = {
        **source_cache,
        "schema": "lafgs_symmetric_photometric_descriptor_cache",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": records,
        "photometric_canonicalization_contract": contract,
    }
    features = fuse_frozen_rows(source_map, payload, canonical_cache, trim_fraction=args.descriptor_trim_fraction)
    output_map = dict(source_map)
    output_map["anchor_features"] = features
    output_map["v7_metric_raw_features"] = features.clone()
    output_map["photometric_canonicalization_contract"] = contract
    output_map["provenance"] = {
        **source_map.get("provenance", {}),
        "symmetric_photometric_canonicalization": {
            "contract": contract,
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "descriptor_only": True,
            "fixed_anchor_rows_identity_xyz_selection": True,
            "source_map": str(args.selected_map),
            "source_cache": str(args.source_cache),
            "gaussian_ply": str(args.gaussian_ply),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    map_path = args.output_dir / "symmetric_photometric_anchor_map.pt"
    metric_path = args.output_dir / "symmetric_photometric_identity_metric.pt"
    _atomic_torch_save(output_map, map_path)
    metric = SharedLowRankMetric(descriptor_dim=features.shape[1], rank=1, max_residual_norm=0.0)
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    metric_state = {
        "schema": "lafgs_shared_metric_state", "version": 1,
        "landmark_indices": torch.arange(features.shape[0]).long(),
        "metric_config": metric.export_config(),
        "metric_state_dict": {name: value.detach().cpu().clone() for name, value in metric.state_dict().items()},
        "map_path": str(map_path), "step": 0,
        "protocol": "symmetric_percentile_grayscale_fixed_projective_anchor_map",
        "photometric_canonicalization_contract": contract,
    }
    _atomic_torch_save(metric_state, metric_path)
    for field in TOPOLOGY_FIELDS:
        if field in source_map and not torch.equal(torch.as_tensor(source_map[field]), torch.as_tensor(output_map[field])):
            raise AssertionError(f"topology field changed: {field}")
    report = {
        "schema": "lafgs_symmetric_photometric_descriptor_materialization",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "descriptor_only": True,
        "fixed_anchor_rows_identity_xyz_selection": True,
        "photometric_canonicalization_contract": contract,
        "mapping_query_count": len(cameras),
        "anchor_count": int(features.shape[0]),
        "inputs": {"dataset": str(args.dataset), "gaussian_ply": str(args.gaussian_ply), "selected_map": str(args.selected_map), "source_cache": str(args.source_cache), "track_payload": str(args.track_payload)},
        "input_sha256": {"gaussian_ply": sha256_file(args.gaussian_ply), "selected_map": sha256_file(args.selected_map), "source_cache": sha256_file(args.source_cache), "track_payload": sha256_file(args.track_payload)},
        "outputs": {"anchor_map": str(map_path), "identity_metric": str(metric_path)},
        "output_sha256": {"anchor_map": sha256_file(map_path), "identity_metric": sha256_file(metric_path)},
        "runtime_identity": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name()},
        "superpoint_weight_sha256": SUPERPOINT_WEIGHT_SHA256,
        "superpoint_weights_path": str(resolve_superpoint_weights()),
        "code_sha256": {"materializer": sha256_file(Path(__file__).resolve()), "photometric": sha256_file(Path(__file__).resolve().parents[1] / "features/photometric.py")},
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    _atomic_json(report, args.output_dir / "symmetric_photometric_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), default="2dgs")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--selected-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("symmetric photometric materialization requires CUDA")
    for field in ("dataset", "gaussian_ply", "source_cache", "track_payload", "selected_map", "output_dir"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    print(json.dumps(materialize(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
