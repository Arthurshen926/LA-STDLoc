#!/usr/bin/env python3
"""Refine rendered-RGB Track points with fixed mapping cameras.

The Track graph, observation rows, descriptors, mapping poses, and intrinsics
remain frozen.  Only each Track's 3D point is optimized against its existing
multi-view reprojection residuals.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.triangulation import robust_triangulate_associations


_TRACK_IDENTITY_FIELDS = (
    "track_index",
    "query_index",
    "keypoint_index",
    "confidence",
    "track_level",
    "source_track_index",
    "parent_source_track_ids",
    "repair_child_index",
    "repair_parent_child_count",
    "coverage_certified",
    "observation_reprojection_px",
    "identity_positive_certified",
)


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _atomic_save(payload: dict, path: Path, source: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        for field in _TRACK_IDENTITY_FIELDS:
            before = torch.as_tensor(source["tracks"][field])
            after = torch.as_tensor(reloaded["tracks"][field])
            if (
                before.dtype != after.dtype
                or before.shape != after.shape
                or not torch.equal(before, after)
            ):
                raise RuntimeError(f"Track identity field {field} changed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(
    *,
    track_payload_path: Path,
    expected_track_payload_sha256: str,
    query_cache_path: Path,
    expected_query_cache_sha256: str,
    output_path: Path,
    nonlinear_iterations: int,
    keypoint_offset_px: float = 0.0,
) -> dict:
    if int(nonlinear_iterations) <= 0:
        raise ValueError("nonlinear refinement iterations must be positive")
    if float(keypoint_offset_px) != 0.0:
        raise ValueError(
            "geometry refinement must preserve the frozen Track pixel convention"
        )
    track_payload_path = track_payload_path.resolve()
    query_cache_path = query_cache_path.resolve()
    output_path = output_path.resolve()
    if output_path in (track_payload_path, query_cache_path):
        raise ValueError("geometry-refinement output aliases an input")
    track_sha = _require_sha(
        track_payload_path, expected_track_payload_sha256, "Track payload"
    )
    cache_sha = _require_sha(
        query_cache_path, expected_query_cache_sha256, "query cache"
    )
    payload = torch.load(track_payload_path, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        query_cache_path, map_location="cpu", weights_only=False
    )
    if (
        payload.get("rendered_rgb_only") is not True
        or cache_payload.get("uses_source_mapping_rgb") is not False
        or cache_payload.get("uses_test_queries") is not False
    ):
        raise ValueError("geometry refinement requires rendered mapping-only evidence")
    names = list(payload["query_names"])
    cache = cache_payload["queries"]
    if names != list(cache):
        raise ValueError("Track and query-cache registries differ")
    repair = payload.get("support_repair", {})
    contract = repair.get("track_build_contract", {})
    required = {
        "minimum_track_views",
        "maximum_observations",
        "minimum_view_bins",
        "huber_delta_px",
        "triangulation_iterations",
        "minimum_parallax_deg",
        "parallax_quantile",
        "maximum_reprojection_px",
        "maximum_condition_number",
    }
    if repair.get("schema") != "lafgs_rendered_track_support_repair" or not (
        required <= set(contract)
    ):
        raise ValueError("Track payload lacks a complete frozen geometry contract")
    tracks = payload["tracks"]
    observation_query = torch.as_tensor(tracks["query_index"])
    observation_keypoint = torch.as_tensor(tracks["keypoint_index"])
    if observation_query.dtype != torch.long or observation_keypoint.dtype != torch.long:
        raise ValueError("Track observation identity must be exact int64")
    keypoints = [torch.as_tensor(cache[name]["native_keypoints"]).float() for name in names]
    uv = torch.stack(
        [
            keypoints[int(query)][int(keypoint)]
            for query, keypoint in zip(
                observation_query.tolist(), observation_keypoint.tolist()
            )
        ]
    )
    intrinsics = torch.stack(
        [torch.as_tensor(cache[name]["native_K"]).float() for name in names]
    )
    poses = torch.stack(
        [torch.as_tensor(cache[name]["pose_w2c"]).float() for name in names]
    )
    source_geometry = payload["track_geometry"]
    track_count = int(torch.as_tensor(source_geometry["triangulated"]).numel())
    geometry = robust_triangulate_associations(
        landmark_count=track_count,
        landmark_index=torch.as_tensor(tracks["track_index"]),
        query_index=observation_query,
        uv=uv,
        confidence=torch.as_tensor(tracks["confidence"]),
        camera_K=intrinsics,
        pose_w2c=poses,
        query_bin=torch.as_tensor(payload["query_bins"]),
        rendered_depth=None,
        maximum_observations_per_landmark=int(contract["maximum_observations"]),
        minimum_views=int(contract["minimum_track_views"]),
        minimum_view_bins=int(contract["minimum_view_bins"]),
        huber_delta_px=float(contract["huber_delta_px"]),
        iterations=int(contract["triangulation_iterations"]),
        minimum_parallax_deg=float(contract["minimum_parallax_deg"]),
        parallax_quantile=float(contract["parallax_quantile"]),
        maximum_reprojection_px=float(contract["maximum_reprojection_px"]),
        maximum_condition_number=float(contract["maximum_condition_number"]),
        maximum_covariance_trace_m2=float("inf"),
        maximum_rendered_depth_residual_m=float("inf"),
        minimum_rendered_depth_observations=0,
        surface_support_enabled=False,
        nonlinear_refinement_iterations=int(nonlinear_iterations),
    )
    geometry["track_confidence_level"] = torch.as_tensor(
        source_geometry["track_confidence_level"]
    ).clone()
    before_xyz = torch.as_tensor(source_geometry["triangulated_xyz"]).float()
    after_xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()
    common = (
        torch.as_tensor(source_geometry["triangulated"]).bool()
        & torch.as_tensor(geometry["triangulated"]).bool()
        & torch.isfinite(before_xyz).all(dim=1)
        & torch.isfinite(after_xyz).all(dim=1)
    )
    shift = torch.linalg.vector_norm(after_xyz[common] - before_xyz[common], dim=1)
    refined = dict(payload)
    refined["track_geometry"] = geometry
    refined["geometry_refinement"] = {
        "schema": "lafgs_fixed_camera_track_point_refinement",
        "version": 1,
        "track_payload": str(track_payload_path),
        "track_payload_sha256": track_sha,
        "query_cache": str(query_cache_path),
        "query_cache_sha256": cache_sha,
        "mapping_cameras_fixed": True,
        "track_graph_fixed": True,
        "descriptor_rows_fixed": True,
        "keypoint_offset_px": float(keypoint_offset_px),
        "nonlinear_iterations": int(nonlinear_iterations),
    }
    _require_sha(track_payload_path, track_sha, "Track payload")
    _require_sha(query_cache_path, cache_sha, "query cache")
    _atomic_save(refined, output_path, payload)
    applied = torch.as_tensor(
        geometry["triangulation_nonlinear_refinement_applied"]
    ).bool()
    report = {
        "schema": "lafgs_fixed_camera_track_point_refinement_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "input_track_payload": str(track_payload_path),
        "input_track_payload_sha256": track_sha,
        "query_cache": str(query_cache_path),
        "query_cache_sha256": cache_sha,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "track_count": track_count,
        "refinement_applied_count": int(applied.sum()),
        "common_triangulated_count": int(common.sum()),
        "median_point_shift_m": float(shift.median()) if shift.numel() else 0.0,
        "p90_point_shift_m": (
            float(torch.quantile(shift, 0.9)) if shift.numel() else 0.0
        ),
        "mapping_cameras_fixed": True,
        "track_graph_fixed": True,
        "descriptor_rows_fixed": True,
        "decision": "REFINED_PAYLOAD_REQUIRES_SHARED_SELECTOR_EVALUATION",
    }
    report_path = output_path.with_suffix(".json")
    if report_path.exists():
        raise FileExistsError(report_path)
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        json.loads(temporary.read_text())
        os.replace(temporary, report_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--expected-track-payload-sha256", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nonlinear-iterations", type=int, default=5)
    parser.add_argument("--keypoint-offset-px", type=float, default=0.0)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                track_payload_path=args.track_payload,
                expected_track_payload_sha256=args.expected_track_payload_sha256,
                query_cache_path=args.query_cache,
                expected_query_cache_sha256=args.expected_query_cache_sha256,
                output_path=args.output,
                nonlinear_iterations=args.nonlinear_iterations,
                keypoint_offset_px=args.keypoint_offset_px,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
