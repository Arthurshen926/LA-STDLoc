#!/usr/bin/env python3
"""Build an experimental Track-only map from Gaussian-rendered mapping RGB.

This is deliberately a small ablation, not a production pipeline entrypoint.
The mapping branch never loads source RGB pixels and never consumes rendered
depth, Gaussian primitive positions, KCS, or GWFF.  The Gaussian prior is used
only to render RGB at the frozen COLMAP mapping poses.  Sparse SuperPoint
observations from those rendered images are matched and triangulated from
camera rays; the resulting Track descriptors form a one-vector localization
map with an identity shared metric.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.camera_pair_policy import (
    candidate_camera_pairs,
    trajectory_balanced_camera_pairs,
)
from evidence.tracks import fuse_track_descriptors
from evidence.triangulation import (
    build_cycle_consistent_tracks,
    camera_pose_bins,
    reciprocal_epipolar_matches,
    robust_triangulate_associations,
)
from features.extractor import FeatureExtractor
from features.multiview_fusion import PIXEL_CENTER_OFFSET
from map_learning.metric import SharedLowRankMetric
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rendering import render_from_pose_gsplat
from topology.track_core import _eligible_tracks, _track_quality


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _uniform_indices(count: int, limit: int) -> torch.Tensor:
    if limit <= 0 or limit >= count:
        return torch.arange(count, dtype=torch.long)
    return torch.div(
        torch.arange(limit, dtype=torch.long) * count,
        limit,
        rounding_mode="floor",
    )


def _intrinsic(camera) -> torch.Tensor:
    focal_x = camera.width / (2.0 * torch.tan(torch.tensor(camera.fov_x / 2.0)))
    focal_y = camera.height / (2.0 * torch.tan(torch.tensor(camera.fov_y / 2.0)))
    return torch.tensor(
        [
            [float(focal_x), 0.0, camera.width / 2.0],
            [0.0, float(focal_y), camera.height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


@torch.inference_mode()
def _render_feature_cache(args, output: Path) -> dict:
    started = time.perf_counter()
    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    indices = _uniform_indices(len(mapping), args.max_views)
    cameras = [mapping[int(index)] for index in indices]

    model = (
        GaussianModel2D(args.sh_degree)
        if args.gaussian_type == "2dgs"
        else GaussianModel3D(args.sh_degree)
    )
    # This experiment consumes only Gaussian RGB appearance.  Do not
    # materialize the fallback random localization-feature bank when the PLY
    # has no loc_* fields; it is unused by rgb_only rendering and can dominate
    # prior-loading time and memory for million-Gaussian scenes.
    model.load_ply(args.gaussian_ply, loc_feature_dim=0)
    model = model.cuda().eval()
    extractor = FeatureExtractor("sp", nms_radius=args.nms_radius).cuda().eval()
    extractor.requires_grad_(False)

    records: dict[str, dict] = {}
    render_seconds = 0.0
    feature_seconds = 0.0
    for row, camera in enumerate(cameras):
        pose = torch.from_numpy(camera.pose_w2c).cuda().float()
        render_started = time.perf_counter()
        package = render_from_pose_gsplat(
            model,
            pose,
            camera.fov_x,
            camera.fov_y,
            camera.width,
            camera.height,
            bg_color=torch.zeros(3, device="cuda"),
            render_mode="RGB",
            rgb_only=True,
            rasterize_mode="antialiased",
        )
        rendered = package["render"].float().clamp(0.0, 1.0)
        torch.cuda.synchronize()
        render_seconds += time.perf_counter() - render_started

        feature_started = time.perf_counter()
        sparse = extractor.detectAndCompute(
            rendered[None],
            top_k=args.keypoints,
            detection_threshold=args.detection_threshold,
        )[0]
        torch.cuda.synchronize()
        feature_seconds += time.perf_counter() - feature_started
        keypoints = sparse["keypoints"].detach().cpu().float()
        descriptors = F.normalize(sparse["descriptors"].detach().cpu().float(), dim=1)
        scores = sparse["keypoint_scores"].detach().cpu().float()
        records[camera.image_name] = {
            "native_keypoints": keypoints,
            "native_descriptors": descriptors,
            "native_scores": scores,
            "native_K": _intrinsic(camera),
            "pose_w2c": torch.from_numpy(camera.pose_w2c).float(),
            "native_input_hw": torch.tensor(
                [camera.height, camera.width], dtype=torch.long
            ),
            "source": "gaussian_rendered_rgb_only",
        }
        if (row + 1) % max(int(args.progress_interval), 1) == 0 or row + 1 == len(
            cameras
        ):
            print(
                f"rendered {row + 1}/{len(cameras)} views; rows={keypoints.shape[0]}",
                flush=True,
            )

    payload = {
        "schema": "lafgs_rendered_rgb_only_sparse_mapping_cache",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_rendered_depth": False,
        "uses_gaussian_geometry_for_triangulation": False,
        "mapping_query_count": len(cameras),
        "full_mapping_query_count": len(mapping),
        "source_mapping_indices": indices,
        "queries": records,
        "configuration": {
            "gaussian_type": args.gaussian_type,
            "sh_degree": args.sh_degree,
            "keypoints": args.keypoints,
            "nms_radius": args.nms_radius,
            "detection_threshold": args.detection_threshold,
            "background": "black",
            "render_mode": "RGB",
            "rasterize_mode": "antialiased",
        },
        "timing_seconds": {
            "render": render_seconds,
            "superpoint": feature_seconds,
            "total": time.perf_counter() - started,
        },
    }
    _atomic_torch_save(payload, output)
    print(f"wrote rendered feature cache {output}", flush=True)
    return payload


def _load_cache(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "lafgs_rendered_rgb_only_sparse_mapping_cache":
        raise ValueError("unexpected rendered-RGB cache schema")
    if payload.get("uses_source_mapping_rgb") is not False:
        raise ValueError("cache does not attest rendered-RGB-only mapping")
    return payload


def _query_trajectory(image_name: str) -> str:
    return str(image_name).split("/", maxsplit=1)[0]


def _trajectory_balanced_matches(
    *,
    args,
    names: list[str],
    descriptors: list[torch.Tensor],
    keypoints: list[torch.Tensor],
    scores: list[torch.Tensor],
    intrinsics: torch.Tensor,
    poses: torch.Tensor,
) -> tuple[
    list[tuple[int, int]],
    dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    dict[tuple[int, int], dict[str, int]],
]:
    nearest = candidate_camera_pairs(
        poses,
        neighbors=args.pair_neighbors,
        minimum_baseline_m=args.minimum_baseline_m,
        maximum_baseline_m=args.maximum_baseline_m,
        maximum_axis_angle_deg=args.maximum_axis_angle_deg,
        policy="nearest",
    )
    pairs = trajectory_balanced_camera_pairs(
        poses,
        [_query_trajectory(name) for name in names],
        local_neighbors=args.local_pair_neighbors,
        pair_budget=len(nearest),
        minimum_baseline_m=args.minimum_baseline_m,
        maximum_baseline_m=args.maximum_baseline_m,
        maximum_axis_angle_deg=args.maximum_axis_angle_deg,
    )
    device = torch.device(args.device)
    matches = {}
    diagnostics = {}
    for completed, (left, right) in enumerate(pairs, start=1):
        source, target, confidence, pair_diagnostics = reciprocal_epipolar_matches(
            descriptors[left].to(device),
            descriptors[right].to(device),
            keypoints[left],
            keypoints[right],
            intrinsics[left],
            poses[left],
            intrinsics[right],
            poses[right],
            minimum_similarity=args.minimum_similarity,
            minimum_margin=args.minimum_margin,
            maximum_epipolar_error_px=args.maximum_epipolar_error_px,
            epipolar_candidate_topk=args.epipolar_candidate_topk,
            recovered_minimum_similarity=-1.0,
            recovered_minimum_margin=-1.0,
            return_diagnostics=True,
        )
        confidence = confidence.cpu() * torch.sqrt(
            scores[left][source.cpu()].float().clamp_min(0.0)
            * scores[right][target.cpu()].float().clamp_min(0.0)
        )
        matches[(left, right)] = (
            source.cpu().long(),
            target.cpu().long(),
            confidence.float(),
        )
        diagnostics[(left, right)] = pair_diagnostics
        if completed % 250 == 0 or completed == len(pairs):
            print(
                f"matched {completed}/{len(pairs)} trajectory-balanced pairs",
                flush=True,
            )
    return pairs, matches, diagnostics


def _build_track_map(args, cache_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    payload = _load_cache(cache_path)
    cache = payload["queries"]
    names = list(cache)
    descriptors = [
        torch.as_tensor(cache[name]["native_descriptors"]).float() for name in names
    ]
    keypoints = [
        torch.as_tensor(cache[name]["native_keypoints"]).float()
        + float(PIXEL_CENTER_OFFSET)
        for name in names
    ]
    scores = [torch.as_tensor(cache[name]["native_scores"]).float() for name in names]
    intrinsics = torch.stack(
        [torch.as_tensor(cache[name]["native_K"]).float() for name in names]
    )
    poses = torch.stack(
        [torch.as_tensor(cache[name]["pose_w2c"]).float() for name in names]
    )
    image_hw = torch.stack(
        [torch.as_tensor(cache[name]["native_input_hw"]).long() for name in names]
    )

    track_started = time.perf_counter()
    precomputed_pairs = None
    precomputed_matches = None
    precomputed_diagnostics = None
    if args.pair_policy == "trajectory_balanced":
        precomputed_pairs, precomputed_matches, precomputed_diagnostics = (
            _trajectory_balanced_matches(
                args=args,
                names=names,
                descriptors=descriptors,
                keypoints=keypoints,
                scores=scores,
                intrinsics=intrinsics,
                poses=poses,
            )
        )
    tracks, diagnostics, sidecar = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        detector_scores=scores,
        camera_K=intrinsics,
        pose_w2c=poses,
        pair_neighbors=args.pair_neighbors,
        pair_policy=(
            "nearest" if args.pair_policy == "nearest" else "trajectory_balanced"
        ),
        pair_budget=(len(precomputed_pairs) if precomputed_pairs is not None else None),
        pair_image_hw=image_hw,
        minimum_baseline_m=args.minimum_baseline_m,
        maximum_baseline_m=args.maximum_baseline_m,
        maximum_axis_angle_deg=args.maximum_axis_angle_deg,
        minimum_similarity=args.minimum_similarity,
        minimum_margin=args.minimum_margin,
        maximum_epipolar_error_px=args.maximum_epipolar_error_px,
        epipolar_candidate_topk=args.epipolar_candidate_topk,
        epipolar_recovered_minimum_similarity=-1.0,
        epipolar_recovered_minimum_margin=-1.0,
        minimum_track_views=args.minimum_views,
        require_cycle=True,
        allow_chain_tracks=True,
        return_pair_sidecar=True,
        precomputed_pairs=precomputed_pairs,
        precomputed_pair_matches=precomputed_matches,
        precomputed_pair_match_diagnostics=precomputed_diagnostics,
        precomputed_confidence_includes_detector_scores=(precomputed_pairs is not None),
        device=args.device,
    )
    track_seconds = time.perf_counter() - track_started
    if int(diagnostics["track_count"]) == 0:
        raise RuntimeError("rendered RGB produced no cycle/chain Tracks")

    observation_query = tracks["query_index"].long()
    observation_keypoint = tracks["keypoint_index"].long()
    observation_uv = torch.stack(
        [
            keypoints[int(query)][int(keypoint)]
            for query, keypoint in zip(
                observation_query.tolist(), observation_keypoint.tolist()
            )
        ]
    )
    query_bins = camera_pose_bins(
        poses, args.view_bins, direction_weight=args.view_direction_weight
    )
    triangulation_started = time.perf_counter()
    geometry = robust_triangulate_associations(
        landmark_count=int(diagnostics["track_count"]),
        landmark_index=tracks["track_index"],
        query_index=observation_query,
        uv=observation_uv,
        confidence=tracks["confidence"],
        camera_K=intrinsics,
        pose_w2c=poses,
        query_bin=query_bins,
        rendered_depth=None,
        maximum_observations_per_landmark=args.maximum_observations,
        minimum_views=args.minimum_views,
        minimum_view_bins=args.minimum_view_bins,
        huber_delta_px=args.huber_delta_px,
        iterations=args.triangulation_iterations,
        minimum_parallax_deg=args.minimum_parallax_deg,
        parallax_quantile=args.parallax_quantile,
        maximum_reprojection_px=args.triangulation_maximum_reprojection_px,
        maximum_condition_number=args.maximum_condition_number,
        maximum_covariance_trace_m2=float("inf"),
        maximum_rendered_depth_residual_m=float("inf"),
        minimum_rendered_depth_observations=0,
        surface_support_enabled=False,
    )
    triangulation_seconds = time.perf_counter() - triangulation_started
    geometry["track_confidence_level"] = tracks["track_level"].clone()

    broad = _eligible_tracks(geometry, "broad")
    quality = _track_quality(geometry)
    quality_order = torch.argsort(quality, descending=True, stable=True)
    selected = quality_order[broad[quality_order]][: args.map_capacity]
    if selected.numel() == 0:
        raise RuntimeError("rendered RGB produced no broad triangulated Tracks")
    track_payload = {
        "schema": "lafgs_track_first_payload",
        "version": 1,
        "query_names": names,
        "tracks": tracks,
        "track_geometry": geometry,
        "query_bins": query_bins,
        "diagnostics": diagnostics,
        "pair_sidecar": sidecar,
        "rendered_rgb_only": True,
    }
    fused = fuse_track_descriptors(
        payload=track_payload,
        query_cache=payload,
        track_indices=selected,
        trim_fraction=args.descriptor_trim_fraction,
    )
    xyz = torch.as_tensor(geometry["triangulated_xyz"])[selected].float()
    covariance = torch.as_tensor(geometry["triangulation_covariance_matrix"])[
        selected
    ].float()
    anchor_count = int(selected.numel())
    anchor_ids = torch.arange(anchor_count, dtype=torch.long)
    anchor_map = {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": anchor_ids,
        "anchor_xyz": xyz,
        "anchor_features": fused.float(),
        "source_primitive_ids": torch.full((anchor_count,), -1, dtype=torch.long),
        "track_cluster_ids": selected.long(),
        "anchor_type": torch.ones(anchor_count, dtype=torch.long),
        "dependency_group_ids": torch.arange(anchor_count, dtype=torch.long),
        "coarse_dependency_group_ids": torch.arange(anchor_count, dtype=torch.long),
        "fine_identity_ids": selected.long(),
        "anchor_position_covariance": covariance,
        "anchor_matchability": torch.ones(anchor_count),
        "base_anchor_count": 0,
        "canonical_anchor_count": anchor_count,
        "micro_anchor_count": anchor_count,
        "provenance": {
            "mapping_rgb_source": "gaussian_render_only",
            "uses_source_mapping_rgb": False,
            "uses_rendered_depth": False,
            "uses_gaussian_geometry_for_triangulation": False,
            "feature_cache": str(cache_path.resolve()),
        },
    }
    metric = SharedLowRankMetric(
        descriptor_dim=fused.shape[1], rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    metric_state = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": anchor_ids,
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in metric.state_dict().items()
        },
        "map_path": str(output.resolve()),
        "step": 0,
        "protocol": "raw_superpoint_identity_rendered_rgb_track_probe",
    }
    metric_path = output.with_name("metric_state_identity.pt")
    track_path = output.with_name("rendered_rgb_track_payload.pt")
    _atomic_torch_save(track_payload, track_path)
    _atomic_torch_save(anchor_map, output)
    _atomic_torch_save(metric_state, metric_path)

    triangulated = torch.as_tensor(geometry["triangulated"]).bool()
    reprojection = torch.as_tensor(geometry["triangulation_reprojection_median_px"])[
        triangulated
    ]
    parallax = torch.as_tensor(geometry["triangulation_parallax_deg"])[triangulated]
    report = {
        "schema": "lafgs_rendered_rgb_only_track_probe_report",
        "version": 1,
        "scientific_scope": {
            "mapping_source_rgb_loaded": False,
            "mapping_source_rgb_used": False,
            "test_queries_used_for_map_construction": False,
            "gaussian_rendered_rgb_used": True,
            "rendered_depth_used": False,
            "gaussian_primitive_geometry_used_for_triangulation": False,
            "kcs_or_gwff_used": False,
        },
        "query_count": len(names),
        "pair_count": int(diagnostics["track_camera_pair_candidate_count"]),
        "pair_policy": str(args.pair_policy),
        "cross_trajectory_pair_count": int(
            sum(
                _query_trajectory(names[left]) != _query_trajectory(names[right])
                for left, right in zip(
                    torch.as_tensor(sidecar["pair"]["left_query_index"]).tolist(),
                    torch.as_tensor(sidecar["pair"]["right_query_index"]).tolist(),
                )
            )
        ),
        "raw_match_count": int(diagnostics["track_raw_reciprocal_epipolar_edge_count"]),
        "cycle_or_chain_supported_edge_count": int(
            diagnostics["track_cycle_supported_edge_count"]
        ),
        "track_count": int(diagnostics["track_count"]),
        "triangulated_track_count": int(triangulated.sum()),
        "broad_track_count": int(broad.sum()),
        "selected_map_track_count": anchor_count,
        "triangulated_reprojection_median_px": (
            float(reprojection.median()) if reprojection.numel() else None
        ),
        "triangulated_reprojection_p90_px": (
            float(torch.quantile(reprojection, 0.9)) if reprojection.numel() else None
        ),
        "triangulated_parallax_median_deg": (
            float(parallax.median()) if parallax.numel() else None
        ),
        "timing_seconds": {
            "matching_and_track_build": track_seconds,
            "pure_ray_triangulation": triangulation_seconds,
            "total": time.perf_counter() - started,
        },
        "artifacts": {
            "feature_cache": str(cache_path.resolve()),
            "track_payload": str(track_path.resolve()),
            "anchor_map": str(output.resolve()),
            "identity_metric": str(metric_path.resolve()),
        },
    }
    report_path = output.with_suffix(".json")
    _atomic_json(report, report_path)
    report["artifacts_sha256"] = {
        name: sha256_file(path)
        for name, path in {
            "feature_cache": cache_path,
            "track_payload": track_path,
            "anchor_map": output,
            "identity_metric": metric_path,
        }.items()
    }
    _atomic_json(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), default="2dgs")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--reuse-feature-cache", action="store_true")
    parser.add_argument("--keypoints", type=int, default=1024)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--detection-threshold", type=float, default=0.0)
    parser.add_argument("--pair-neighbors", type=int, default=6)
    parser.add_argument(
        "--pair-policy",
        choices=("nearest", "trajectory_balanced"),
        default="nearest",
    )
    parser.add_argument("--local-pair-neighbors", type=int, default=4)
    parser.add_argument("--minimum-baseline-m", type=float, default=0.03)
    parser.add_argument("--maximum-baseline-m", type=float, default=5.0)
    parser.add_argument("--maximum-axis-angle-deg", type=float, default=75.0)
    parser.add_argument("--minimum-similarity", type=float, default=0.65)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--maximum-epipolar-error-px", type=float, default=2.0)
    parser.add_argument("--epipolar-candidate-topk", type=int, default=4)
    parser.add_argument("--minimum-views", type=int, default=3)
    parser.add_argument("--minimum-view-bins", type=int, default=2)
    parser.add_argument("--view-bins", type=int, default=8)
    parser.add_argument("--view-direction-weight", type=float, default=0.5)
    parser.add_argument("--maximum-observations", type=int, default=32)
    parser.add_argument("--huber-delta-px", type=float, default=2.0)
    parser.add_argument("--triangulation-iterations", type=int, default=3)
    parser.add_argument("--minimum-parallax-deg", type=float, default=1.0)
    parser.add_argument("--parallax-quantile", type=float, default=0.75)
    parser.add_argument(
        "--triangulation-maximum-reprojection-px", type=float, default=2.0
    )
    parser.add_argument("--maximum-condition-number", type=float, default=1e6)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--map-capacity", type=int, default=16000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("rendered-RGB Track probe requires CUDA")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "rendered_rgb_feature_cache.pt"
    map_path = args.output_dir / "rendered_rgb_track_map.pt"
    if args.reuse_feature_cache:
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
    else:
        if cache_path.exists():
            raise FileExistsError(cache_path)
        _render_feature_cache(args, cache_path)
    if not args.render_only:
        if map_path.exists():
            raise FileExistsError(map_path)
        _build_track_map(args, cache_path, map_path)


if __name__ == "__main__":
    main()
