#!/usr/bin/env python3
"""Render an observer-only V6 virtual probe cache from a frozen Gaussian prior."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.v6_observer_probes import SCHEMA as PLAN_SCHEMA
from features.extractor import FeatureExtractor
from features.raster_sampling import sample_raster_at_grid_uv
from priors.models import GaussianModel2D
from priors.rendering import render_from_pose_gsplat


CACHE_SCHEMA = "lafgs_v6_fixed_map_observer_probe_cache"
CACHE_VERSION = 1


def _load(path: Path, expected: str, label: str) -> tuple[dict, str]:
    actual = sha256_file(path.resolve())
    if actual != str(expected).lower():
        raise ValueError(f"{label} SHA differs")
    return torch.load(path.resolve(), map_location="cpu", weights_only=False), actual


def _plane(value, label: str) -> torch.Tensor:
    result = torch.as_tensor(value).float().squeeze()
    if result.ndim != 2:
        raise ValueError(f"rendered {label} is not a plane")
    return result


def _sensor_variant(rgb: torch.Tensor, variant: str, *, seed: int) -> torch.Tensor:
    image = torch.as_tensor(rgb).float().clamp(0.0, 1.0)
    if variant == "clean":
        return image
    if variant == "exposure_down":
        return image * 0.75
    if variant == "exposure_up":
        return (image * 1.25).clamp(0.0, 1.0)
    if variant == "gamma_low":
        return image.clamp_min(1e-6).pow(0.8)
    if variant == "gamma_high":
        return image.clamp_min(1e-6).pow(1.25)
    if variant == "motion_blur_mild":
        kernel = image.new_ones((3, 1, 1, 7)) / 7.0
        return F.conv2d(image[None], kernel, padding=(0, 3), groups=3)[0]
    if variant == "sensor_noise_mild":
        generator = torch.Generator(device=image.device).manual_seed(int(seed))
        noise = torch.randn(
            image.shape, generator=generator, device=image.device, dtype=image.dtype
        )
        return (image + 0.01 * noise).clamp(0.0, 1.0)
    if variant == "resize_compression_mild":
        height, width = image.shape[-2:]
        small = F.interpolate(
            image[None],
            size=(max(height // 2, 1), max(width // 2, 1)),
            mode="bilinear",
            align_corners=False,
        )
        return F.interpolate(
            small, size=(height, width), mode="bilinear", align_corners=False
        )[0]
    if variant == "local_occlusion_mild":
        output = image.clone()
        height, width = output.shape[-2:]
        y0, y1 = height * 2 // 5, height * 3 // 5
        x0, x1 = width * 2 // 5, width * 3 // 5
        output[:, y0:y1, x0:x1] = 0.0
        return output
    raise ValueError(f"unsupported observer sensor variant: {variant}")


def _atomic_save(payload: dict, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--expected-gaussian-ply-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--keypoints", type=int, default=2048)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--detection-threshold", type=float, default=0.0)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--maximum-probes", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    plan, plan_sha = _load(args.plan, args.expected_plan_sha256, "probe plan")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("uses_source_mapping_rgb") is not False
        or plan.get("uses_test_queries") is not False
        or plan.get("virtual_probes_added_to_map") is not False
    ):
        raise ValueError("virtual observer probe plan contract differs")
    gaussian_sha = sha256_file(args.gaussian_ply.resolve())
    if gaussian_sha != str(args.expected_gaussian_ply_sha256).lower():
        raise ValueError("Gaussian prior SHA differs")
    selected = list(plan["selected_probes"])
    if args.maximum_probes is not None:
        if int(args.maximum_probes) < 1:
            raise ValueError("maximum probes must be positive")
        selected = selected[: int(args.maximum_probes)]
    model = GaussianModel2D(int(args.sh_degree))
    model.load_ply(args.gaussian_ply.resolve(), loc_feature_dim=0)
    model = model.to(args.device).eval()
    extractor = FeatureExtractor("sp", nms_radius=int(args.nms_radius)).to(
        args.device
    ).eval()
    extractor.requires_grad_(False)
    records = {}
    render_audits = []
    for probe_order, probe in enumerate(selected):
        pose = torch.as_tensor(probe["pose_w2c"]).float()
        intrinsics = torch.as_tensor(probe["native_K"]).float()
        height, width = map(int, probe["native_input_hw"])
        fov_x = 2.0 * math.atan(width / (2.0 * float(intrinsics[0, 0])))
        fov_y = 2.0 * math.atan(height / (2.0 * float(intrinsics[1, 1])))
        package = render_from_pose_gsplat(
            model,
            pose.to(args.device),
            fov_x,
            fov_y,
            width,
            height,
            bg_color=torch.zeros(3, device=args.device),
            render_mode="RGB+ED",
            rgb_only=True,
            rasterize_mode="antialiased",
        )
        rgb = torch.as_tensor(package["render"]).float().clamp(0.0, 1.0)
        alpha = _plane(
            package.get("alphas", package.get("rend_alpha")), "alpha"
        ).cpu()
        depth = _plane(package["depth"], "expected depth").cpu()
        for variant_index, variant in enumerate(probe["sensor_variants"]):
            variant_rgb = _sensor_variant(
                rgb, str(variant), seed=int(args.seed) + probe_order * 17 + variant_index
            )
            sparse = extractor.detectAndCompute(
                variant_rgb[None],
                top_k=int(args.keypoints),
                detection_threshold=float(args.detection_threshold),
            )[0]
            keypoints = sparse["keypoints"].detach().cpu().float()
            descriptors = F.normalize(
                sparse["descriptors"].detach().cpu().float(), dim=1
            )
            scores = sparse["keypoint_scores"].detach().cpu().float()
            alpha_at = sample_raster_at_grid_uv(alpha, keypoints).float()
            depth_at = sample_raster_at_grid_uv(depth, keypoints).float()
            valid = (
                torch.isfinite(alpha_at)
                & (alpha_at >= float(args.alpha_minimum))
                & torch.isfinite(depth_at)
                & (depth_at > 0.0)
            )
            name = f"virtual/{probe_order:04d}/{variant}"
            records[name] = {
                "native_keypoints": keypoints,
                "native_descriptors": descriptors.half(),
                "native_scores": scores.half(),
                "native_K": intrinsics,
                "pose_w2c": pose,
                "native_input_hw": torch.tensor([height, width]),
                "native_rendered_rgb": variant_rgb.detach().cpu().half(),
                "native_alpha": alpha.half(),
                "native_depth": depth.half(),
                "native_alpha_at_keypoints": alpha_at.half(),
                "native_depth_at_keypoints": depth_at.half(),
                "native_valid_keypoint_mask": valid,
                "sequence_id": f"virtual_probe/{probe_order:04d}",
                "probe_index": probe_order,
                "candidate_index": int(probe["candidate_index"]),
                "sensor_variant": str(variant),
                "clean_pose_probe": str(variant) == "clean",
                "pixel_center_offset": 0.5,
            }
            render_audits.append(
                {
                    "image_name": name,
                    "detector_row_count": int(keypoints.shape[0]),
                    "render_valid_row_count": int(valid.sum()),
                    "alpha_supported_image_fraction": float(
                        (alpha >= float(args.alpha_minimum)).float().mean()
                    ),
                }
            )
            print(
                f"[v6-probe] {len(render_audits)} views: {name}, "
                f"valid={int(valid.sum())}/{int(keypoints.shape[0])}",
                flush=True,
            )
    payload = {
        "schema": CACHE_SCHEMA,
        "version": CACHE_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "mapping_only": True,
        "inputs": {
            "probe_plan_sha256": plan_sha,
            "source_map_sha256": plan["inputs"]["map_sha256"],
            "source_observation_cache_sha256": plan["inputs"][
                "observation_cache_sha256"
            ],
            "source_feedback_sha256": plan["inputs"]["feedback_sha256"],
            "gaussian_ply_sha256": gaussian_sha,
        },
        "query_names": list(records),
        "queries": records,
        "render_audits": render_audits,
        "probe_count": len(selected),
        "rendered_variant_count": len(records),
        "virtual_probes_added_to_map": False,
        "virtual_probes_added_to_anchor_observations": False,
        "virtual_probes_increase_track_view_count": False,
        "depth_channels": {
            "expected_depth": True,
            "median_depth": False,
            "contribution_entropy": False,
            "depth_consistency": False,
            "unavailable_channels_are_not_fabricated": True,
        },
        "configuration": {
            "keypoints": int(args.keypoints),
            "nms_radius": int(args.nms_radius),
            "detection_threshold": float(args.detection_threshold),
            "alpha_minimum": float(args.alpha_minimum),
            "seed": int(args.seed),
        },
    }
    _atomic_save(payload, args.output)
    print(args.output.resolve())
    print(sha256_file(args.output.resolve()))


if __name__ == "__main__":
    main()
