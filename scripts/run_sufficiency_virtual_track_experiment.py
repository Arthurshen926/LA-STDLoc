#!/usr/bin/env python3
"""Run the bounded mapping-only virtual-render Track closed-loop experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.triangulation import (
    build_cycle_consistent_tracks,
    camera_pose_bins,
    robust_triangulate_associations,
)
from evidence.virtual_render_planner import SCHEMA as PLAN_SCHEMA
from evidence.virtual_render_planner import camera_registry_sha256
from evidence.virtual_track_experiment import (
    DRY_RUN_THRESHOLDS,
    augment_formal_anchor_map,
    build_map_bound_identity_metric,
    dry_run_passes,
    enforce_one_observation_per_family,
    validate_augmented_mapping_guard,
)
from features.extractor import FeatureExtractor
from features.raster_sampling import sample_raster_at_grid_uv
from localization.pose_solver import pose_error, solve_absolute_pose
from priors.models import GaussianModel2D
from priors.rendering import render_from_pose_gsplat
from topology.anchor_construction import TrackAnchorProvider, UnifiedAnchorConstructor
from topology.track_core import _eligible_tracks, _track_quality


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _atomic_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _plane(value, name: str) -> torch.Tensor:
    value = torch.as_tensor(value).float().squeeze()
    if value.ndim != 2:
        raise ValueError(f"rendered {name} must be a plane")
    return value


def _gather(rows, query, keypoint):
    counts = torch.tensor([int(value.shape[0]) for value in rows], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return torch.cat(rows)[offsets[query.long()] + keypoint.long()]


def validate_runtime_plan_lineage(
    plan: dict, gaussian_ply: Path
) -> tuple[Path, Path, Path, dict, dict, str]:
    """Re-hash every mutable parent and the ordered geometry registry."""
    support = plan.get("candidate_render_support", {})
    if support.get("mode") != "real_low_resolution_alpha_depth_zbuffer":
        raise ValueError("closed loop requires real candidate alpha/depth support")
    selected_map_path = Path(plan.get("inputs", {}).get("selected_map", ""))
    if (
        not selected_map_path.is_file()
        or sha256_file(selected_map_path) != plan["inputs"].get("selected_map_sha256")
    ):
        raise ValueError("formal unified map identity changed after planning")
    if support.get("gaussian_ply_sha256") != sha256_file(gaussian_ply):
        raise ValueError("planner and closed loop use different Gaussian priors")
    query_path = Path(plan.get("inputs", {}).get("query_cache", ""))
    track_path = Path(plan.get("inputs", {}).get("track_payload", ""))
    if not query_path.is_file() or sha256_file(query_path) != plan["inputs"].get(
        "query_cache_sha256"
    ):
        raise ValueError("planner query cache identity changed at runtime")
    if not track_path.is_file() or sha256_file(track_path) != plan["inputs"].get(
        "track_payload_sha256"
    ):
        raise ValueError("planner Track payload identity changed at runtime")
    query_payload, track_payload = _load(query_path), _load(track_path)
    ordered_registry_sha = camera_registry_sha256(
        query_payload, list(track_payload["query_names"])
    )
    if ordered_registry_sha != plan["inputs"].get("canonical_camera_registry_sha256"):
        raise ValueError("ordered canonical camera registry changed at runtime")
    return (
        selected_map_path, query_path, track_path,
        query_payload, track_payload, ordered_registry_sha,
    )


@torch.inference_mode()
def render_observations(args, plan: dict, selected: torch.Tensor) -> dict:
    source_cache = _load(Path(plan["inputs"]["query_cache"]))
    source_records = source_cache.get("queries", source_cache)
    names = list(_load(Path(plan["inputs"]["track_payload"]))["query_names"])
    if set(names) != set(source_records):
        raise ValueError("planner source camera registry changed")
    candidates = plan["candidates"]
    families = torch.as_tensor(candidates["pose_family"])[selected].long()
    if families.unique().numel() != families.numel():
        raise ValueError("selected virtual cameras violate one-view-per-family plan")
    model = GaussianModel2D(args.sh_degree)
    model.load_ply(args.gaussian_ply, loc_feature_dim=0)
    model = model.to(args.device).eval()
    extractor = FeatureExtractor("sp", nms_radius=args.nms_radius).to(args.device).eval()
    extractor.requires_grad_(False)
    records = {}
    detector_rows = []
    supported_rows = []
    alpha_fraction = []
    for output_row, candidate_index in enumerate(selected.tolist()):
        parent = int(candidates["parent_camera_index"][candidate_index])
        source = source_records[names[parent]]
        pose = torch.as_tensor(candidates["pose_w2c"][candidate_index]).float()
        K = torch.as_tensor(source["native_K"]).float()
        height, width = map(int, torch.as_tensor(source["native_input_hw"]).tolist())
        fov_x = 2.0 * math.atan(width / (2.0 * float(K[0, 0])))
        fov_y = 2.0 * math.atan(height / (2.0 * float(K[1, 1])))
        package = render_from_pose_gsplat(
            model, pose.to(args.device), fov_x, fov_y, width, height,
            bg_color=torch.zeros(3, device=args.device), render_mode="RGB+ED",
            rgb_only=True, rasterize_mode="antialiased",
        )
        rgb = package["render"].float().clamp(0, 1)
        alpha = _plane(package.get("alphas", package.get("rend_alpha")), "alpha")
        depth = _plane(package["depth"], "depth")
        sparse = extractor.detectAndCompute(
            rgb[None], top_k=args.keypoints,
            detection_threshold=args.detection_threshold,
        )[0]
        keypoints = sparse["keypoints"].detach().cpu().float()
        descriptors = F.normalize(
            sparse["descriptors"].detach().cpu().float(), dim=1
        )
        scores = sparse["keypoint_scores"].detach().cpu().float()
        alpha_cpu, depth_cpu = alpha.cpu(), depth.cpu()
        alpha_at = sample_raster_at_grid_uv(alpha_cpu, keypoints).float()
        depth_at = sample_raster_at_grid_uv(depth_cpu, keypoints).float()
        valid = (alpha_at >= args.alpha_minimum) & torch.isfinite(depth_at) & (depth_at > 0)
        name = f"virtual/{output_row:04d}"
        records[name] = {
            "native_keypoints": keypoints,
            "native_descriptors": descriptors,
            "native_scores": scores,
            "native_K": K,
            "pose_w2c": pose,
            "native_input_hw": torch.tensor([height, width]),
            "native_rendered_alpha": alpha_cpu.half(),
            "native_rendered_depth": depth_cpu.half(),
            "native_alpha_at_keypoints": alpha_at.half(),
            "native_depth_at_keypoints": depth_at.half(),
            "native_valid_keypoint_mask": valid,
            "sequence_id": f"pose_family/{int(families[output_row])}",
            "pose_family": int(families[output_row]),
            "candidate_index": int(candidate_index),
            "parent_mapping_camera_name": names[parent],
            "pixel_center_offset": 0.5,
            "source": "sufficiency_guided_gaussian_render",
        }
        detector_rows.append(int(keypoints.shape[0]))
        supported_rows.append(int(valid.sum()))
        alpha_fraction.append(float((alpha_cpu >= args.alpha_minimum).float().mean()))
        print(json.dumps({"rendered": output_row + 1, "views": len(selected),
                          "detector_rows": detector_rows[-1],
                          "supported_rows": supported_rows[-1]}), flush=True)
    return {
        "schema": "lafgs_sufficiency_virtual_observations",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_gaussian_primitive_geometry_for_triangulation": False,
        "query_bins": torch.as_tensor(candidates["formal_pose_bin"])[selected].long(),
        "pose_family": families,
        "queries": records,
        "render_quality": {
            "detector_rows": detector_rows,
            "supported_detector_rows": supported_rows,
            "alpha_supported_image_fraction": alpha_fraction,
        },
        "configuration": vars(args),
    }


def build_tracks(args, cache: dict) -> tuple[dict, object, torch.Tensor]:
    provider = GaussianRenderObservationProvider(cache)
    inputs = provider.track_inputs()
    raw, diagnostics = build_cycle_consistent_tracks(
        descriptors=inputs["descriptors"], keypoints=inputs["keypoints"],
        detector_scores=inputs["detector_scores"], camera_K=inputs["camera_K"],
        pose_w2c=inputs["pose_w2c"], pair_neighbors=len(provider) - 1,
        pair_policy="nearest", pair_image_hw=inputs["image_hw"],
        minimum_baseline_m=args.minimum_baseline_m,
        maximum_baseline_m=args.maximum_baseline_m,
        maximum_axis_angle_deg=args.maximum_axis_angle_deg,
        minimum_similarity=args.minimum_similarity, minimum_margin=args.minimum_margin,
        maximum_epipolar_error_px=args.maximum_epipolar_error_px,
        epipolar_candidate_topk=args.epipolar_candidate_topk,
        epipolar_recovered_minimum_similarity=-1.0,
        epipolar_recovered_minimum_margin=-1.0,
        minimum_track_views=args.minimum_views, require_cycle=True,
        allow_chain_tracks=True, device=args.device,
    )
    tracks, family_audit = enforce_one_observation_per_family(
        raw, cache["pose_family"]
    )
    if not tracks["track_index"].numel():
        raise RuntimeError("virtual renders produced no family-independent Tracks")
    uv = _gather(inputs["keypoints"], tracks["query_index"], tracks["keypoint_index"])
    bins = torch.as_tensor(cache["query_bins"]).long()
    geometry = robust_triangulate_associations(
        landmark_count=int(tracks["track_level"].numel()),
        landmark_index=tracks["track_index"], query_index=tracks["query_index"],
        uv=uv, confidence=tracks["confidence"], camera_K=inputs["camera_K"],
        pose_w2c=inputs["pose_w2c"], query_bin=bins,
        minimum_views=args.minimum_views, minimum_view_bins=args.minimum_view_bins,
        minimum_parallax_deg=args.minimum_parallax_deg,
        maximum_reprojection_px=args.maximum_reprojection_px,
        maximum_condition_number=args.maximum_condition_number,
        surface_support_enabled=False,
    )
    geometry["track_confidence_level"] = tracks["track_level"].clone()
    payload = {
        "schema": "lafgs_sufficiency_virtual_track_payload", "version": 1,
        "query_names": inputs["query_names"], "tracks": tracks,
        "track_geometry": geometry, "query_bins": bins,
        "diagnostics": {**diagnostics, "family_contract": family_audit},
        "rendered_rgb_only": True,
    }
    return payload, provider, uv


def mapping_oracle(payload, provider, selected: torch.Tensor) -> dict:
    tracks, geometry = payload["tracks"], payload["track_geometry"]
    selected_set = torch.zeros(len(tracks["track_level"]), dtype=torch.bool)
    selected_set[selected] = True
    errors, successes = [], 0
    xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()
    inputs = provider.track_inputs()
    for query in range(len(provider)):
        rows = torch.nonzero(
            (tracks["query_index"] == query) & selected_set[tracks["track_index"]],
            as_tuple=False,
        ).reshape(-1)
        if rows.numel() < 4:
            errors.append(None)
            continue
        uv = inputs["keypoints"][query][tracks["keypoint_index"][rows]]
        points = xyz[tracks["track_index"][rows]]
        estimate = solve_absolute_pose(
            uv.numpy(), points.numpy(), inputs["camera_K"][query].numpy(),
            reprojection_error_px=4.0, min_iterations=100, max_iterations=10000,
        )
        if estimate.inliers.size:
            rotation, translation = pose_error(
                estimate.pose_w2c, inputs["pose_w2c"][query].numpy()
            )
            errors.append({"rotation_deg": rotation, "translation_cm": translation,
                           "inliers": int(estimate.inliers.size)})
            successes += 1
        else:
            errors.append(None)
    translation = torch.tensor([x["translation_cm"] for x in errors if x])
    rotation = torch.tensor([x["rotation_deg"] for x in errors if x])
    return {
        "name": "mapping_leave_one_query_observation_oracle",
        "warning": "diagnostic only; each query observation participates in Track geometry",
        "success_count": successes, "query_count": len(provider),
        "success_fraction": successes / max(len(provider), 1),
        "median_translation_cm": float(translation.median()) if translation.numel() else None,
        "median_rotation_deg": float(rotation.median()) if rotation.numel() else None,
        "per_query": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view-count", type=int, choices=(8, 32), required=True)
    parser.add_argument("--dry-run-decision", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--keypoints", type=int, default=1024)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--detection-threshold", type=float, default=0.0)
    parser.add_argument("--alpha-minimum", type=float, default=0.2)
    parser.add_argument("--view-bins", type=int, default=8)
    parser.add_argument("--minimum-baseline-m", type=float, default=0.01)
    parser.add_argument("--maximum-baseline-m", type=float, default=5.0)
    parser.add_argument("--maximum-axis-angle-deg", type=float, default=75.0)
    parser.add_argument("--minimum-similarity", type=float, default=0.65)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--maximum-epipolar-error-px", type=float, default=2.0)
    parser.add_argument("--epipolar-candidate-topk", type=int, default=4)
    parser.add_argument("--minimum-views", type=int, default=3)
    parser.add_argument("--minimum-view-bins", type=int, default=2)
    parser.add_argument("--minimum-parallax-deg", type=float, default=0.5)
    parser.add_argument("--maximum-reprojection-px", type=float, default=2.0)
    parser.add_argument("--maximum-condition-number", type=float, default=1e6)
    args = parser.parse_args()
    plan = _load(args.plan)
    if plan.get("schema") != PLAN_SCHEMA or plan.get("mapping_only") is not True:
        raise ValueError("invalid mapping-only virtual-render plan")
    if plan.get("gt_visible_diagnostic", "missing") is not None:
        raise ValueError("planner contains a GT-visible/test diagnostic")
    (
        selected_map_path, query_path, track_path,
        query_payload, track_payload, ordered_registry_sha,
    ) = validate_runtime_plan_lineage(plan, args.gaussian_ply)
    if plan.get("triangulation_family_contract", {}).get(
        "source_and_pose_proximity_components"
    ) is not True:
        raise ValueError("planner lacks source/pose-proximity family identity")
    selected = torch.as_tensor(plan["selected_candidate_indices"]).long()[:args.view_count]
    if selected.numel() != args.view_count:
        raise ValueError("plan does not contain the requested frozen view count")
    if args.view_count == 32:
        if args.dry_run_decision is None:
            raise ValueError("Top-32 requires the frozen Top-8 decision")
        decision = json.loads(args.dry_run_decision.read_text())
        if decision.get("passed") is not True or decision.get("plan_sha256") != sha256_file(args.plan):
            raise ValueError("Top-8 decision did not pass for this exact plan")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache = render_observations(args, plan, selected)
    payload, provider, _ = build_tracks(args, cache)
    geometry = payload["track_geometry"]
    broad = _eligible_tracks(geometry, "broad")
    stable = torch.as_tensor(geometry["track_confidence_level"]) >= 2
    eligible = broad & stable
    order = torch.argsort(_track_quality(geometry), descending=True, stable=True)
    chosen = order[eligible[order]]
    if chosen.numel():
        batch = UnifiedAnchorConstructor.materialize([
            TrackAnchorProvider(payload=payload, observations=provider,
                                track_indices=chosen, trim_fraction=0.2)
        ])
    else:
        raise RuntimeError("virtual renders produced no stable broad Anchor")
    oracle = mapping_oracle(payload, provider, chosen)
    quality = cache["render_quality"]
    metrics = {
        "selected_view_count": args.view_count,
        "deficit_voxel_count": int((torch.as_tensor(plan["coverage_field"]["deficit_demand"]) > 0).sum()),
        "detector_rows": int(sum(quality["detector_rows"])),
        "median_detector_rows": float(torch.tensor(quality["detector_rows"]).median()),
        "supported_detector_rows": int(sum(quality["supported_detector_rows"])),
        "median_supported_detector_rows": float(torch.tensor(quality["supported_detector_rows"]).median()),
        "median_alpha_supported_image_fraction": float(torch.tensor(quality["alpha_supported_image_fraction"]).median()),
        "raw_track_count": int(payload["diagnostics"]["track_count"]),
        "family_filtered_track_count": int(payload["diagnostics"]["family_contract"]["retained_track_count"]),
        "stable_broad_track_count": int(eligible.sum()),
        "new_anchor_count": int(batch.xyz.shape[0]),
        "distinct_view_bins": int(torch.unique(payload["query_bins"]).numel()),
        "view_bin_histogram": torch.bincount(payload["query_bins"]).tolist(),
        "family_contract_passed": True,
        "gt_visible_diagnostic": None,
        "mapping_oracle_loo": oracle,
    }
    passed, failures = dry_run_passes(metrics) if args.view_count == 8 else (True, [])
    artifact = {
        "schema": "lafgs_sufficiency_virtual_track_closed_loop", "version": 1,
        "mapping_only": True, "uses_test_queries": False,
        "uses_source_mapping_rgb": False,
        "uses_gaussian_primitive_geometry_for_triangulation": False,
        "plan_path": str(args.plan.resolve()), "plan_sha256": sha256_file(args.plan),
        "gaussian_ply": str(args.gaussian_ply.resolve()),
        "gaussian_ply_sha256": sha256_file(args.gaussian_ply),
        "selected_candidate_indices": selected,
        "observation_cache": cache, "track_payload": payload,
        "unified_anchor_batch": asdict(batch), "mapping_metrics": metrics,
        "frozen": True,
    }
    artifact_path = args.output_dir / "frozen_virtual_track_closed_loop.pt"
    _atomic_save(artifact, artifact_path)
    artifact_sha = sha256_file(artifact_path)
    formal_map = _load(selected_map_path)
    virtual_registry = {
        "schema": "lafgs_virtual_observation_registry",
        "version": 1,
        "formal_query_count": len(track_payload["query_names"]),
        "query_count": len(provider),
        "ordered_names": list(provider.names),
        "pose_w2c": torch.stack([
            provider.build_view(index).pose_w2c.float() for index in range(len(provider))
        ]),
        "camera_K": torch.stack([
            provider.build_view(index).intrinsics.float() for index in range(len(provider))
        ]),
        "image_hw": torch.tensor([
            provider.build_view(index).image_hw for index in range(len(provider))
        ], dtype=torch.long),
        "pose_family": torch.as_tensor(cache["pose_family"]).long(),
        "formal_pose_bin": torch.as_tensor(cache["query_bins"]).long(),
        "ordered_formal_camera_registry_sha256": ordered_registry_sha,
    }
    augmented = augment_formal_anchor_map(
        formal_map, batch,
        formal_query_count=len(track_payload["query_names"]),
        virtual_registry=virtual_registry,
        lineage={
            "schema": "lafgs_virtual_anchor_augmentation_lineage",
            "version": 1,
            "mapping_only": True,
            "uses_test_queries": False,
            "formal_map": str(selected_map_path.resolve()),
            "formal_map_sha256": sha256_file(selected_map_path),
            "plan": str(args.plan.resolve()),
            "plan_sha256": sha256_file(args.plan),
            "virtual_track_artifact": str(artifact_path.resolve()),
            "virtual_track_artifact_sha256": artifact_sha,
            "query_cache_sha256": sha256_file(query_path),
            "track_payload_sha256": sha256_file(track_path),
            "ordered_camera_registry_sha256": ordered_registry_sha,
            "augmentation_semantics": "formal_5794_prefix_plus_virtual_track_anchors",
        },
    )
    guard = validate_augmented_mapping_guard(
        augmented, formal_map, virtual_query_count=len(provider)
    )
    augmented_path = args.output_dir / "augmented_formal_anchor_map.pt"
    _atomic_save(augmented, augmented_path)
    augmented_sha = sha256_file(augmented_path)
    metric = build_map_bound_identity_metric(
        map_path=str(augmented_path.resolve()), map_sha256=augmented_sha,
        anchor_ids=augmented["anchor_ids"],
        descriptor_dim=int(torch.as_tensor(augmented["anchor_features"]).shape[1]),
        producer={
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "torch_version": str(torch.__version__),
            "source": "formal_prefix_plus_frozen_virtual_track_suffix",
        },
    )
    metric_path = args.output_dir / "identity_metric.pt"
    _atomic_save(metric, metric_path)
    metric_sha = sha256_file(metric_path)
    guard.update({
        "augmented_map": str(augmented_path.resolve()),
        "augmented_map_sha256": augmented_sha,
        "formal_map_sha256": sha256_file(selected_map_path),
        "virtual_track_artifact_sha256": artifact_sha,
        "identity_metric": str(metric_path.resolve()),
        "identity_metric_sha256": metric_sha,
        "identity_metric_landmark_count": int(metric["landmark_indices"].numel()),
        "identity_metric_protocol": metric["protocol"],
        "learned_descriptor_transform": False,
    })
    _atomic_json(guard, args.output_dir / "mapping_guard.json")
    decision = {
        "schema": "lafgs_sufficiency_virtual_track_dry_run_decision", "version": 1,
        "plan_sha256": sha256_file(args.plan), "view_count": args.view_count,
        "thresholds": DRY_RUN_THRESHOLDS if args.view_count == 8 else None,
        "metrics": metrics, "passed": passed, "failures": failures,
        "artifact": str(artifact_path.resolve()), "artifact_sha256": artifact_sha,
        "augmented_map": str(augmented_path.resolve()),
        "augmented_map_sha256": augmented_sha,
        "mapping_guard": guard,
        "identity_metric": str(metric_path.resolve()),
        "identity_metric_sha256": metric_sha,
    }
    _atomic_json(decision, args.output_dir / "decision.json")
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
