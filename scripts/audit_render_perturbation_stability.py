#!/usr/bin/env python3
"""Test whether source-free render perturbation stability predicts real-RGB gap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from features.extractor import FeatureExtractor
from features.superpoint import sample_descriptors
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rendering import render_from_pose_gsplat


RECIPES = (
    ("identity", 1.0, 1.0, 1.0),
    ("exposure_0p9", 0.9, 1.0, 1.0),
    ("exposure_1p1", 1.1, 1.0, 1.0),
    ("contrast_0p9", 1.0, 0.9, 1.0),
    ("contrast_1p1", 1.0, 1.1, 1.0),
    ("gamma_0p9", 1.0, 1.0, 0.9),
    ("gamma_1p1", 1.0, 1.0, 1.1),
)
SUBPIXEL_OFFSETS = ((-0.25, 0.0), (0.25, 0.0), (0.0, -0.25), (0.0, 0.25))


def perturb_rgb(rgb: torch.Tensor, *, exposure: float, contrast: float, gamma: float) -> torch.Tensor:
    value = torch.as_tensor(rgb).float().clamp(0.0, 1.0) * float(exposure)
    value = (value - 0.5) * float(contrast) + 0.5
    return value.clamp(0.0, 1.0).pow(float(gamma))


def descriptor_stability(descriptors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return variance-to-centroid and worst cosine-to-identity for [V,N,D]."""
    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=2)
    if descriptors.ndim != 3 or descriptors.shape[0] < 2:
        raise ValueError("descriptor variants must have shape [V>=2,N,D]")
    centroid = F.normalize(descriptors.mean(dim=0), dim=1)
    cosine_centroid = torch.einsum("vnd,nd->vn", descriptors, centroid)
    identity_cosine = torch.einsum("vnd,nd->vn", descriptors, descriptors[0])
    return (1.0 - cosine_centroid).mean(dim=0), identity_cosine.min(dim=0).values


def decile_diagnostics(instability: torch.Tensor, gap: torch.Tensor) -> dict:
    instability = torch.as_tensor(instability).float()
    gap = torch.as_tensor(gap).float()
    finite = torch.isfinite(instability) & torch.isfinite(gap)
    instability, gap = instability[finite], gap[finite]
    rho = float(spearmanr(instability.numpy(), gap.numpy()).statistic)
    edges = torch.unique(torch.quantile(instability, torch.linspace(0, 1, 11)))
    rows = []
    for index in range(edges.numel() - 1):
        selected = (instability >= edges[index]) & (
            instability <= edges[index + 1] if index + 2 == edges.numel() else instability < edges[index + 1]
        )
        if bool(selected.any()):
            rows.append({
                "minimum": float(edges[index]), "maximum": float(edges[index + 1]),
                "count": int(selected.sum()), "real_gap_mean": float(gap[selected].mean()),
            })
    violations = sum(rows[index + 1]["real_gap_mean"] < rows[index]["real_gap_mean"] for index in range(len(rows) - 1))
    ratio = rows[-1]["real_gap_mean"] / max(rows[0]["real_gap_mean"], 1e-12)
    return {
        "spearman_rho": rho,
        "deciles": rows,
        "monotonic_violation_count": int(violations),
        "highest_over_lowest_decile_gap_ratio": float(ratio),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@torch.inference_mode()
def run(args) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(int(args.cpu_threads))
    cache = torch.load(args.render_cache, map_location="cpu", weights_only=False)
    paired = torch.load(args.paired_records, map_location="cpu", weights_only=False)
    if cache.get("uses_source_mapping_rgb") is not False or cache.get("uses_test_queries") is not False:
        raise ValueError("stability predictor requires mapping-only rendered cache")
    if paired.get("schema") != "lafgs_render_real_descriptor_gap_records":
        raise ValueError("paired descriptor-gap records schema differs")
    names = list(cache["queries"])
    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    indices = torch.as_tensor(cache["source_mapping_indices"]).long()
    cameras = [mapping[int(index)] for index in indices]
    if names != [camera.image_name for camera in cameras]:
        raise ValueError("render cache and mapping camera schedule differ")
    model = GaussianModel2D(args.sh_degree) if args.gaussian_type == "2dgs" else GaussianModel3D(args.sh_degree)
    model.load_ply(args.gaussian_ply, loc_feature_dim=0)
    model = model.cuda().eval()
    extractor = FeatureExtractor("sp", nms_radius=args.nms_radius).cuda().eval()
    extractor.requires_grad_(False)
    variance_by_query, worst_by_query = {}, {}
    for index, camera in enumerate(cameras):
        source = cache["queries"][camera.image_name]
        pose = torch.as_tensor(source["pose_w2c"]).float().cuda()
        package = render_from_pose_gsplat(
            model, pose, camera.fov_x, camera.fov_y, camera.width, camera.height,
            bg_color=torch.zeros(3, device="cuda"), render_mode="RGB+ED",
            rgb_only=True, rasterize_mode="antialiased",
        )
        rgb = package["render"].float().clamp(0.0, 1.0)
        keypoints = torch.as_tensor(source["native_keypoints"]).float().cuda()
        variants = []
        identity_dense = None
        for _, exposure, contrast, gamma in RECIPES:
            dense, _ = extractor.detectAndComputeDense(
                perturb_rgb(rgb, exposure=exposure, contrast=contrast, gamma=gamma)[None]
            )
            if identity_dense is None:
                identity_dense = dense
            variants.append(sample_descriptors(keypoints[None], dense)[0].transpose(0, 1).cpu())
        for dx, dy in SUBPIXEL_OFFSETS:
            offset = keypoints + keypoints.new_tensor([dx, dy])
            variants.append(sample_descriptors(offset[None], identity_dense)[0].transpose(0, 1).cpu())
        variance, worst = descriptor_stability(torch.stack(variants))
        variance_by_query[camera.image_name] = variance
        worst_by_query[camera.image_name] = worst
        if (index + 1) % max(1, args.progress_interval) == 0 or index + 1 == len(cameras):
            print(json.dumps({"completed_views": index + 1, "mapping_views": len(cameras)}), flush=True)
    query_indices = torch.as_tensor(paired["query_indices"]).long()
    keypoint_indices = torch.as_tensor(paired["keypoint_indices"]).long()
    variance = torch.empty(query_indices.numel())
    worst = torch.empty_like(variance)
    for query in torch.unique(query_indices, sorted=True).tolist():
        selected = query_indices == int(query)
        keypoints = keypoint_indices[selected]
        variance[selected] = variance_by_query[names[int(query)]][keypoints]
        worst[selected] = worst_by_query[names[int(query)]][keypoints]
    real_gap = 1.0 - torch.as_tensor(paired["factors"]["paired_cosine"]).float()
    variance_report = decile_diagnostics(variance, real_gap)
    worst_report = decile_diagnostics(1.0 - worst, real_gap)
    go = (
        variance_report["spearman_rho"] >= 0.2
        and worst_report["spearman_rho"] >= 0.2
        and variance_report["monotonic_violation_count"] <= 2
        and worst_report["monotonic_violation_count"] <= 2
        and variance_report["highest_over_lowest_decile_gap_ratio"] >= 1.25
        and worst_report["highest_over_lowest_decile_gap_ratio"] >= 1.25
    )
    records_path = args.output.with_suffix(".records.pt")
    records_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "lafgs_render_perturbation_stability_records", "version": 1,
        "query_indices": query_indices, "keypoint_indices": keypoint_indices,
        "descriptor_variance": variance, "worst_variant_cosine": worst,
        "paired_real_gap": real_gap,
    }, records_path)
    report = {
        "schema": "lafgs_render_perturbation_stability_audit", "version": 1,
        "uses_test_queries": False, "uses_source_mapping_rgb_for_predictor": False,
        "uses_source_mapping_rgb_as_offline_oracle_label": True,
        "audit_only": True, "map_mutated": False, "trains_feature_or_matcher": False,
        "configuration": {"recipes": RECIPES, "subpixel_offsets_xy": SUBPIXEL_OFFSETS},
        "descriptor_variance": variance_report,
        "worst_variant_cosine_instability": worst_report,
        "go_contract": {"rho_minimum_each": 0.2, "maximum_monotonic_violations_each": 2, "decile_gap_ratio_minimum_each": 1.25},
        "go": bool(go),
        "candidate_reliability_if_go": "clamp(worst_variant_cosine,0,1)",
        "records": str(records_path.resolve()), "records_sha256": sha256_file(records_path),
        "inputs": {"dataset": str(args.dataset), "gaussian_ply": str(args.gaussian_ply), "render_cache": str(args.render_cache), "paired_records": str(args.paired_records)},
        "input_sha256": {"gaussian_ply": sha256_file(args.gaussian_ply), "render_cache": sha256_file(args.render_cache), "paired_records": sha256_file(args.paired_records)},
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    _atomic_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), default="2dgs")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--render-cache", type=Path, required=True)
    parser.add_argument("--paired-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()
    for field in ("dataset", "gaussian_ply", "render_cache", "paired_records", "output"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    report = run(args)
    print(json.dumps({"go": report["go"], "descriptor_variance": report["descriptor_variance"], "worst_variant_cosine_instability": report["worst_variant_cosine_instability"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
