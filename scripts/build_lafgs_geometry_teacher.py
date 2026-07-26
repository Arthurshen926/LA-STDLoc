#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from train_lafgs_map import (
    _collect_independent_geometry_teacher,
    _collect_track_first_geometry_teacher,
)


def _arguments(parsed):
    return SimpleNamespace(
        geometry_teacher_identity_mode=parsed.mode,
        geometry_teacher_min_similarity=parsed.map_min_similarity,
        geometry_teacher_min_margin=parsed.map_min_margin,
        positive_radius_px=parsed.positive_radius_px,
        geometry_teacher_view_bins=parsed.view_bins,
        geometry_teacher_view_direction_weight=parsed.view_direction_weight,
        geometry_teacher_max_observations_per_landmark=parsed.max_observations,
        geometry_teacher_min_views=parsed.min_views,
        geometry_teacher_min_view_bins=parsed.min_view_bins,
        geometry_teacher_huber_delta_px=parsed.huber_delta_px,
        geometry_teacher_iterations=parsed.iterations,
        geometry_teacher_min_parallax_deg=parsed.min_parallax_deg,
        geometry_teacher_parallax_quantile=parsed.parallax_quantile,
        geometry_teacher_max_reprojection_px=parsed.max_reprojection_px,
        geometry_teacher_max_condition_number=parsed.max_condition_number,
        geometry_teacher_max_covariance_trace_m2=parsed.max_covariance_trace_m2,
        geometry_teacher_max_rendered_depth_residual_m=(
            parsed.max_rendered_depth_residual_m
        ),
        geometry_teacher_min_rendered_depth_observations=(
            parsed.min_rendered_depth_observations
        ),
        geometry_teacher_track_pair_neighbors=parsed.track_pair_neighbors,
        geometry_teacher_track_min_baseline_m=parsed.track_min_baseline_m,
        geometry_teacher_track_max_baseline_m=parsed.track_max_baseline_m,
        geometry_teacher_track_max_axis_angle_deg=(
            parsed.track_max_axis_angle_deg
        ),
        geometry_teacher_track_min_similarity=parsed.track_min_similarity,
        geometry_teacher_track_min_margin=parsed.track_min_margin,
        geometry_teacher_track_max_epipolar_error_px=(
            parsed.track_max_epipolar_error_px
        ),
        geometry_teacher_track_require_cycle=parsed.track_require_cycle,
        geometry_teacher_track_lgcv=parsed.track_lgcv,
        geometry_teacher_track_lgcv_neighbors=parsed.track_lgcv_neighbors,
        geometry_teacher_track_lgcv_support_threshold=(
            parsed.track_lgcv_support_threshold
        ),
        geometry_teacher_track_lgcv_angle_cosine=(
            parsed.track_lgcv_angle_cosine
        ),
        geometry_teacher_track_lgcv_scale_threshold=(
            parsed.track_lgcv_scale_threshold
        ),
        geometry_teacher_track_lgcv_scale_limit=(
            parsed.track_lgcv_scale_limit
        ),
        geometry_teacher_track_lgcv_maximum_edge_px=(
            parsed.track_lgcv_maximum_edge_px
        ),
        geometry_teacher_track_lgcv_minimum_matches=(
            parsed.track_lgcv_minimum_matches
        ),
        geometry_teacher_track_lgcv_mode=parsed.track_lgcv_mode,
        geometry_teacher_track_lgcv_confidence_floor=(
            parsed.track_lgcv_confidence_floor
        ),
        geometry_teacher_track_assignment_max_distance_m=(
            parsed.track_assignment_max_distance_m
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild only LaFGS geometry evidence against a frozen descriptor "
            "state and query cache"
        )
    )
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--base_statistics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=["map_top1", "gt_clean_map_top1", "track_first"],
        required=True,
    )
    parser.add_argument("--map_min_similarity", type=float, default=0.7)
    parser.add_argument("--map_min_margin", type=float, default=0.03)
    parser.add_argument("--positive_radius_px", type=float, default=2.0)
    parser.add_argument("--view_bins", type=int, default=8)
    parser.add_argument("--view_direction_weight", type=float, default=0.5)
    parser.add_argument("--max_observations", type=int, default=32)
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--min_view_bins", type=int, default=2)
    parser.add_argument("--huber_delta_px", type=float, default=2.0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--min_parallax_deg", type=float, default=1.0)
    parser.add_argument("--parallax_quantile", type=float, default=0.75)
    parser.add_argument("--max_reprojection_px", type=float, default=2.0)
    parser.add_argument("--max_condition_number", type=float, default=1e6)
    parser.add_argument("--max_covariance_trace_m2", type=float, default=0.01)
    parser.add_argument(
        "--max_rendered_depth_residual_m", type=float, default=0.15
    )
    parser.add_argument(
        "--min_rendered_depth_observations", type=int, default=2
    )
    parser.add_argument("--track_pair_neighbors", type=int, default=6)
    parser.add_argument("--track_min_baseline_m", type=float, default=0.03)
    parser.add_argument("--track_max_baseline_m", type=float, default=5.0)
    parser.add_argument("--track_max_axis_angle_deg", type=float, default=75.0)
    parser.add_argument("--track_min_similarity", type=float, default=0.65)
    parser.add_argument("--track_min_margin", type=float, default=0.01)
    parser.add_argument(
        "--track_max_epipolar_error_px", type=float, default=2.0
    )
    parser.add_argument(
        "--track_require_cycle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--track_lgcv",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--track_lgcv_neighbors", type=int, default=8)
    parser.add_argument(
        "--track_lgcv_support_threshold", type=float, default=4.0
    )
    parser.add_argument(
        "--track_lgcv_angle_cosine", type=float, default=0.9659
    )
    parser.add_argument(
        "--track_lgcv_scale_threshold", type=float, default=0.1
    )
    parser.add_argument(
        "--track_lgcv_scale_limit", type=float, default=3.0
    )
    parser.add_argument(
        "--track_lgcv_maximum_edge_px", type=float, default=50.0
    )
    parser.add_argument(
        "--track_lgcv_minimum_matches", type=int, default=8
    )
    parser.add_argument(
        "--track_lgcv_mode", choices=["hard", "soft"], default="hard"
    )
    parser.add_argument(
        "--track_lgcv_confidence_floor", type=float, default=0.25
    )
    parser.add_argument(
        "--track_assignment_max_distance_m", type=float, default=0.20
    )
    args = parser.parse_args()

    query_cache_path = Path(args.query_cache).expanduser().resolve()
    state_path = Path(args.state).expanduser().resolve()
    base_statistics_path = Path(args.base_statistics).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    cache_payload = torch.load(
        query_cache_path, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    query_names = sorted(
        name
        for name, value in cache.items()
        if isinstance(value, dict) and "native_descriptors" in value
    )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    base = torch.load(
        base_statistics_path, map_location="cpu", weights_only=False
    )
    state_indices = torch.as_tensor(state["landmark_indices"]).long()
    base_indices = torch.as_tensor(base["landmark_indices"]).long()
    if not torch.equal(state_indices, base_indices):
        raise ValueError("State and base statistics landmark IDs differ")
    features = torch.as_tensor(state["landmark_features"]).cuda().float()
    bank_xyz = torch.as_tensor(state["landmark_xyz"]).cuda().float()
    teacher_args = _arguments(args)
    if args.mode == "track_first":
        statistics, geometry, diagnostics = (
            _collect_track_first_geometry_teacher(
                query_names, cache, bank_xyz, teacher_args
            )
        )
    else:
        statistics, geometry, diagnostics = (
            _collect_independent_geometry_teacher(
                features,
                query_names,
                cache,
                bank_xyz,
                teacher_args,
            )
        )
    triangulated = torch.as_tensor(geometry["triangulated"]).bool()
    triangulated_xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()
    current_xyz = bank_xyz.detach().cpu()
    current_offset = torch.full(
        (current_xyz.shape[0],), float("inf"), dtype=torch.float32
    )
    current_offset[triangulated] = torch.linalg.norm(
        triangulated_xyz[triangulated] - current_xyz[triangulated], dim=1
    )
    geometry["triangulation_current_center_offset_m"] = current_offset
    output = dict(base)
    output["version"] = max(int(output.get("version", 0)), 4)
    output_statistics = dict(output.get("statistics", {}))
    output_statistics.update(
        {name: torch.as_tensor(value).cpu() for name, value in statistics.items()}
    )
    output["statistics"] = output_statistics
    output_geometry = dict(output.get("geometry_evidence", {}))
    for name in list(output_geometry):
        if name.startswith("triangulation_") or name.startswith("track_"):
            output_geometry.pop(name)
    output_geometry.update(
        {name: torch.as_tensor(value).cpu() for name, value in geometry.items()}
    )
    output["geometry_evidence"] = output_geometry
    output_diagnostics = dict(output.get("diagnostics", {}))
    output_diagnostics.update(diagnostics)
    output_diagnostics.update(
        {
            "geometry_teacher_query_cache": str(query_cache_path),
            "geometry_teacher_frozen_state": str(state_path),
            "geometry_teacher_base_statistics": str(base_statistics_path),
            "geometry_teacher_query_count": len(query_names),
        }
    )
    output["diagnostics"] = output_diagnostics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "mode": args.mode,
                "query_count": len(query_names),
                **diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
