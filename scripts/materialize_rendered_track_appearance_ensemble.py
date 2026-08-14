#!/usr/bin/env python3
"""Re-describe a frozen rendered-Track topology across camera-response renders.

The script never changes camera pairs, Track identity, triangulated xyz, or the
selected anchor set.  It renders the frozen Gaussian prior only at source
mapping poses, samples SuperPoint at the already-frozen keypoint rows, and
robustly fuses deterministic appearance variants.  Alpha/depth are persisted
only as observation validity and teacher-visibility evidence; primitive centers
remain absent from localization geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.tracks import fuse_track_descriptors
from features.extractor import FeatureExtractor
from features.superpoint import sample_descriptors
from map_learning.metric import SharedLowRankMetric
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rendering import render_from_pose_gsplat
from scripts.probe_rendered_rgb_track_map import _intrinsic


APPEARANCE_RECIPES = (
    {
        "name": "identity",
        "exposure": 1.0,
        "gamma": 1.0,
        "white_balance": (1.0, 1.0, 1.0),
        "contrast": 1.0,
    },
    {
        "name": "exposure_0p80",
        "exposure": 0.8,
        "gamma": 1.0,
        "white_balance": (1.0, 1.0, 1.0),
        "contrast": 1.0,
    },
    {
        "name": "exposure_1p20",
        "exposure": 1.2,
        "gamma": 1.0,
        "white_balance": (1.0, 1.0, 1.0),
        "contrast": 1.0,
    },
    {
        "name": "gamma_0p85",
        "exposure": 1.0,
        "gamma": 0.85,
        "white_balance": (1.0, 1.0, 1.0),
        "contrast": 1.0,
    },
    {
        "name": "gamma_1p15",
        "exposure": 1.0,
        "gamma": 1.15,
        "white_balance": (1.0, 1.0, 1.0),
        "contrast": 1.0,
    },
    {
        "name": "warm",
        "exposure": 1.0,
        "gamma": 1.0,
        "white_balance": (1.08, 1.0, 0.92),
        "contrast": 1.05,
    },
    {
        "name": "cool",
        "exposure": 1.0,
        "gamma": 1.0,
        "white_balance": (0.92, 1.0, 1.08),
        "contrast": 0.95,
    },
)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _plane(value: torch.Tensor, *, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).squeeze()
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError(f"rendered {name} must be a finite [H, W] plane")
    return value.float()


def _camera_response(rgb: torch.Tensor, recipe: dict) -> torch.Tensor:
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("rendered RGB must have shape [3, H, W]")
    white_balance = rgb.new_tensor(recipe["white_balance"]).reshape(3, 1, 1)
    value = rgb.float().clamp(0.0, 1.0) * float(recipe["exposure"])
    value = value * white_balance
    value = (value - 0.5) * float(recipe["contrast"]) + 0.5
    value = value.clamp(0.0, 1.0).pow(float(recipe["gamma"]))
    return value.clamp(0.0, 1.0)


def robust_appearance_fusion(
    descriptors: torch.Tensor, *, retain_fraction: float = 0.75
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse [V,N,D] descriptors after dropping variant-wise outliers."""

    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=2)
    if descriptors.ndim != 3 or descriptors.shape[0] < 2:
        raise ValueError("appearance descriptors must have shape [V>=2, N, D]")
    variant_count = int(descriptors.shape[0])
    retain = max(2, min(variant_count, math.ceil(variant_count * retain_fraction)))
    consensus = torch.einsum("vnd,wnd->nvw", descriptors, descriptors).mean(dim=2)
    order = torch.argsort(consensus, dim=1, descending=True, stable=True)
    selected = order[:, :retain]
    by_row = descriptors.permute(1, 0, 2)
    retained = torch.gather(
        by_row,
        1,
        selected[:, :, None].expand(-1, -1, descriptors.shape[2]),
    )
    fused = F.normalize(retained.mean(dim=1), dim=1)
    cosine = torch.einsum("nkd,nd->nk", retained, fused)
    dispersion = (1.0 - cosine).mean(dim=1).clamp(0.0, 2.0)
    reliability = (1.0 - dispersion).clamp(0.0, 1.0)
    return fused, dispersion, reliability


def _keypoint_validity(
    alpha: torch.Tensor,
    keypoints: torch.Tensor,
    *,
    alpha_minimum: float,
    erosion_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = _plane(alpha, name="alpha")
    if erosion_radius:
        kernel = 2 * int(erosion_radius) + 1
        local_min = -F.max_pool2d(
            -alpha[None, None], kernel_size=kernel, stride=1, padding=erosion_radius
        )[0, 0]
    else:
        local_min = alpha
    xy = torch.as_tensor(keypoints).round().long()
    xy[:, 0].clamp_(0, alpha.shape[1] - 1)
    xy[:, 1].clamp_(0, alpha.shape[0] - 1)
    sampled = alpha[xy[:, 1], xy[:, 0]]
    valid = local_min[xy[:, 1], xy[:, 0]] >= float(alpha_minimum)
    return valid.cpu(), sampled.cpu()


def _track_dispersion(
    payload: dict, cache: dict, track_indices: torch.Tensor
) -> torch.Tensor:
    tracks = payload["tracks"]
    names = payload["query_names"]
    selected = {int(track) for track in torch.as_tensor(track_indices).long().tolist()}
    observations_by_track = {track: [] for track in selected}
    for observation, track in enumerate(
        torch.as_tensor(tracks["track_index"]).long().tolist()
    ):
        if track in observations_by_track:
            observations_by_track[track].append(observation)
    output = []
    for track in torch.as_tensor(track_indices).long().tolist():
        observations = observations_by_track[int(track)]
        values = []
        for observation in observations:
            query = int(tracks["query_index"][observation])
            keypoint = int(tracks["keypoint_index"][observation])
            record = cache[names[query]]
            if bool(record["native_valid_keypoint_mask"][keypoint]):
                values.append(float(record["native_appearance_dispersion"][keypoint]))
        if not values:
            values = [
                float(
                    cache[names[int(tracks["query_index"][observation])]][
                        "native_appearance_dispersion"
                    ][int(tracks["keypoint_index"][observation])]
                )
                for observation in observations
            ]
        output.append(sum(values) / max(len(values), 1))
    return torch.as_tensor(output, dtype=torch.float32)


@torch.inference_mode()
def materialize(args) -> dict:
    started = time.perf_counter()
    source_cache = torch.load(args.source_cache, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    source_map = torch.load(args.selected_map, map_location="cpu", weights_only=False)
    if source_cache.get("uses_source_mapping_rgb") is not False:
        raise ValueError("source cache is not rendered-RGB-only")
    if source_cache.get("uses_test_queries") is not False:
        raise ValueError("source cache contains test queries")
    source_config = source_cache.get("configuration", {})
    if source_config.get("gaussian_type") != args.gaussian_type:
        raise ValueError("Gaussian type differs from source rendered cache")
    if int(source_config.get("sh_degree", -1)) != int(args.sh_degree):
        raise ValueError("Gaussian SH degree differs from source rendered cache")
    if int(source_config.get("nms_radius", -1)) != int(args.nms_radius):
        raise ValueError("SuperPoint NMS differs from source rendered cache")
    if payload.get("rendered_rgb_only") is not True:
        raise ValueError("Track payload is not rendered-RGB-only")
    if not bool((torch.as_tensor(source_map["anchor_type"]).long() == 1).all()):
        raise ValueError("appearance ensemble accepts only Track anchors")
    selected_tracks = torch.as_tensor(source_map["track_cluster_ids"]).long()
    if selected_tracks.unique().numel() != selected_tracks.numel():
        raise ValueError("selected Track identities are not unique")

    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    indices = torch.as_tensor(source_cache["source_mapping_indices"]).long()
    cameras = [mapping[int(index)] for index in indices]
    source_queries = source_cache["queries"]
    names = list(source_queries)
    if names != [camera.image_name for camera in cameras]:
        raise ValueError("source cache and mapping camera order differ")
    if names != list(payload["query_names"]):
        raise ValueError("source cache and Track payload query order differ")

    model = (
        GaussianModel2D(args.sh_degree)
        if args.gaussian_type == "2dgs"
        else GaussianModel3D(args.sh_degree)
    )
    model.load_ply(args.gaussian_ply, loc_feature_dim=0)
    model = model.cuda().eval()
    extractor = FeatureExtractor("sp", nms_radius=args.nms_radius).cuda().eval()
    extractor.requires_grad_(False)

    records = {}
    base_cosines = []
    valid_rows = total_rows = 0
    for camera_index, camera in enumerate(cameras):
        name = camera.image_name
        source = source_queries[name]
        pose = torch.from_numpy(camera.pose_w2c).float()
        if not torch.equal(pose, torch.as_tensor(source["pose_w2c"]).float()):
            raise ValueError(f"mapping pose differs for {name}")
        if not torch.equal(_intrinsic(camera), torch.as_tensor(source["native_K"])):
            raise ValueError(f"mapping intrinsics differ for {name}")
        if torch.as_tensor(source["native_input_hw"]).long().tolist() != [
            camera.height,
            camera.width,
        ]:
            raise ValueError(f"mapping image shape differs for {name}")
        package = render_from_pose_gsplat(
            model,
            pose.cuda(),
            camera.fov_x,
            camera.fov_y,
            camera.width,
            camera.height,
            bg_color=torch.zeros(3, device="cuda"),
            render_mode="RGB+ED",
            rgb_only=True,
            rasterize_mode="antialiased",
        )
        rendered = package["render"].float().clamp(0.0, 1.0)
        alpha = _plane(package.get("alphas", package.get("rend_alpha")), name="alpha")
        depth = _plane(package["depth"], name="depth")
        keypoints = torch.as_tensor(source["native_keypoints"]).float().cuda()
        variant_descriptors = []
        for recipe in APPEARANCE_RECIPES:
            image = _camera_response(rendered, recipe)
            dense, _ = extractor.detectAndComputeDense(image[None])
            variant_descriptors.append(
                sample_descriptors(keypoints[None], dense)[0].transpose(0, 1).cpu()
            )
        stacked = torch.stack(variant_descriptors)
        fused, dispersion, reliability = robust_appearance_fusion(stacked)
        original = F.normalize(
            torch.as_tensor(source["native_descriptors"]).float(), dim=1
        )
        cosine = (stacked[0] * original).sum(dim=1)
        if float(cosine.min()) < 0.999:
            raise ValueError(
                f"base rerender does not reproduce frozen descriptors for {name}"
            )
        base_cosines.append(cosine)
        valid, alpha_at_keypoints = _keypoint_validity(
            alpha,
            keypoints.cpu(),
            alpha_minimum=args.alpha_minimum,
            erosion_radius=args.alpha_erosion_radius,
        )
        xy = keypoints.cpu().round().long()
        xy[:, 0].clamp_(0, camera.width - 1)
        xy[:, 1].clamp_(0, camera.height - 1)
        depth_at_keypoints = depth.cpu()[xy[:, 1], xy[:, 0]]
        total_rows += int(valid.numel())
        valid_rows += int(valid.sum())
        records[name] = {
            **source,
            "native_descriptors": fused,
            "native_valid_keypoint_mask": valid,
            "native_alpha_at_keypoints": alpha_at_keypoints.half(),
            "native_depth_at_keypoints": depth_at_keypoints.half(),
            "native_rendered_alpha": alpha.cpu().half(),
            "native_rendered_depth": depth.cpu().half(),
            "native_appearance_dispersion": dispersion,
            "native_appearance_reliability": reliability,
            "source": "gaussian_rendered_rgb_appearance_ensemble",
        }
        if (camera_index + 1) % max(args.progress_interval, 1) == 0 or (
            camera_index + 1 == len(cameras)
        ):
            print(
                json.dumps(
                    {
                        "completed_views": camera_index + 1,
                        "valid_fraction": valid_rows / max(total_rows, 1),
                    }
                ),
                flush=True,
            )

    ensemble_cache = {
        **source_cache,
        "schema": "lafgs_rendered_rgb_appearance_ensemble_cache",
        "version": 1,
        "uses_rendered_depth": True,
        "uses_rendered_alpha": True,
        "uses_gaussian_geometry_for_triangulation": False,
        "queries": records,
        "appearance_ensemble": {
            "recipes": list(APPEARANCE_RECIPES),
            "fixed_source_keypoints": True,
            "fixed_track_identity": True,
            "alpha_minimum": float(args.alpha_minimum),
            "alpha_erosion_radius": int(args.alpha_erosion_radius),
            "robust_retain_fraction": 0.75,
        },
    }
    output_cache = args.output_dir / "appearance_ensemble_cache.pt"
    output_map = args.output_dir / "appearance_ensemble_anchor_map.pt"
    output_metric = args.output_dir / "appearance_ensemble_identity_metric.pt"
    for path in (output_cache, output_map, output_metric):
        if path.exists():
            raise FileExistsError(path)
    _atomic_torch_save(ensemble_cache, output_cache)

    fused_tracks = fuse_track_descriptors(
        payload=payload,
        query_cache=ensemble_cache,
        track_indices=selected_tracks,
        trim_fraction=args.descriptor_trim_fraction,
    )
    track_dispersion = _track_dispersion(payload, records, selected_tracks)
    output_state = dict(source_map)
    output_state["anchor_features"] = fused_tracks.float()
    output_state["v7_metric_raw_features"] = fused_tracks.float()
    output_state["anchor_appearance_dispersion"] = track_dispersion
    source_matchability = torch.as_tensor(
        source_map.get("anchor_matchability", torch.ones(selected_tracks.numel()))
    ).float()
    output_state["anchor_matchability"] = source_matchability * (
        1.0 - track_dispersion
    ).clamp(0.0, 1.0)
    output_state["provenance"] = {
        **source_map.get("provenance", {}),
        "rendered_appearance_ensemble": {
            "source_map": str(args.selected_map.resolve()),
            "source_cache": str(args.source_cache.resolve()),
            "track_payload": str(args.track_payload.resolve()),
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "fixed_track_identity_and_xyz": True,
        },
    }
    _atomic_torch_save(output_state, output_map)
    metric = SharedLowRankMetric(
        descriptor_dim=fused_tracks.shape[1], rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    metric_state = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(selected_tracks.numel()).long(),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in metric.state_dict().items()
        },
        "map_path": str(output_map.resolve()),
        "step": 0,
        "protocol": "rendered_track_fixed_geometry_appearance_ensemble",
    }
    _atomic_torch_save(metric_state, output_metric)

    base_cosine = torch.cat(base_cosines)
    report = {
        "schema": "lafgs_rendered_track_appearance_ensemble_materialization",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_rendered_alpha_for_observation_validity": True,
        "uses_rendered_depth_for_teacher_visibility_only": True,
        "uses_gaussian_geometry_for_triangulation": False,
        "fixed_track_identity_and_xyz": True,
        "mapping_query_count": len(cameras),
        "anchor_count": int(selected_tracks.numel()),
        "appearance_variant_count": len(APPEARANCE_RECIPES),
        "valid_keypoint_fraction": valid_rows / max(total_rows, 1),
        "base_rerender_descriptor_cosine_minimum": float(base_cosine.min()),
        "base_rerender_descriptor_cosine_median": float(base_cosine.median()),
        "keypoint_dispersion_median": float(
            torch.cat(
                [record["native_appearance_dispersion"] for record in records.values()]
            ).median()
        ),
        "track_dispersion_median": float(track_dispersion.median()),
        "configuration": ensemble_cache["appearance_ensemble"],
        "inputs": {
            "dataset": str(args.dataset.resolve()),
            "gaussian_ply": str(args.gaussian_ply.resolve()),
            "source_cache": str(args.source_cache.resolve()),
            "track_payload": str(args.track_payload.resolve()),
            "selected_map": str(args.selected_map.resolve()),
        },
        "input_sha256": {
            "gaussian_ply": sha256_file(args.gaussian_ply),
            "source_cache": sha256_file(args.source_cache),
            "track_payload": sha256_file(args.track_payload),
            "selected_map": sha256_file(args.selected_map),
        },
        "outputs": {
            "query_cache": str(output_cache.resolve()),
            "anchor_map": str(output_map.resolve()),
            "identity_metric": str(output_metric.resolve()),
        },
        "output_sha256": {
            "query_cache": sha256_file(output_cache),
            "anchor_map": sha256_file(output_map),
            "identity_metric": sha256_file(output_metric),
        },
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    _atomic_json(report, args.output_dir / "appearance_ensemble_report.json")
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
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--alpha-erosion-radius", type=int, default=4)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("appearance ensemble materialization requires CUDA")
    for field in (
        "dataset",
        "gaussian_ply",
        "source_cache",
        "track_payload",
        "selected_map",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(json.dumps(materialize(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
