#!/usr/bin/env python3
"""Replay one mapping-only Track build with a frozen cache and pair policy."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch

from common.hashing import sha256_file
from evidence.camera_pair_policy import (
    mapping_scene_points_from_depth_samples,
)
from evidence.parallax_stratified_pair_policy import (
    representative_scene_depth_from_samples,
)
from evidence.triangulation import (
    attach_pair_triangulation_statistics,
    build_cycle_consistent_tracks,
    camera_pose_bins,
    robust_triangulate_associations,
)
from evidence.parallel_triangulation import (
    robust_triangulate_associations_fresh_cpu,
)
from features.multiview_fusion import PIXEL_CENTER_OFFSET
from topology.track_core import _eligible_tracks


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    value = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
    value = value[torch.isfinite(value)]
    report: dict[str, float | int | None] = {"count": int(value.numel())}
    for name, probability in (
        ("p10", 0.10),
        ("median", 0.50),
        ("p90", 0.90),
        ("p95", 0.95),
    ):
        report[name] = (
            None if value.numel() == 0 else float(torch.quantile(value, probability))
        )
    return report


def _sample_depth_at_keypoints(
    depth_source: torch.Tensor,
    keypoints: torch.Tensor,
    rows: torch.Tensor | None = None,
) -> torch.Tensor:
    source = torch.as_tensor(depth_source)
    selected = (
        torch.arange(keypoints.shape[0], dtype=torch.long)
        if rows is None
        else torch.as_tensor(rows, dtype=torch.long)
    )
    if source.ndim == 1:
        return source[selected]
    physical = keypoints[selected] - float(PIXEL_CENTER_OFFSET)
    x = physical[:, 0].round().long().clamp(0, int(source.shape[1]) - 1)
    y = physical[:, 1].round().long().clamp(0, int(source.shape[0]) - 1)
    return source[y, x]


def _image_hw(record: dict) -> torch.Tensor:
    value = record.get("native_input_hw")
    if value is not None:
        return torch.as_tensor(value, dtype=torch.long)
    depth = record.get("native_depth")
    if depth is None or torch.as_tensor(depth).ndim < 2:
        raise ValueError("Query cache lacks native_input_hw/native_depth")
    return torch.as_tensor(torch.as_tensor(depth).shape[-2:], dtype=torch.long)


def _track_report(tracks: dict, geometry: dict, *, query_count: int) -> dict:
    triangulated = torch.as_tensor(geometry["triangulated"]).bool()
    broad = _eligible_tracks(geometry, "broad")
    strict = _eligible_tracks(geometry, "strict")
    query_index = torch.as_tensor(tracks["query_index"]).long()
    track_index = torch.as_tensor(tracks["track_index"]).long()
    query_count = int(query_count)
    if query_count <= 0 or (
        query_index.numel() and int(query_index.max()) >= query_count
    ):
        raise ValueError("Track observations escape the mapping query registry")
    broad_observation = broad[track_index]
    broad_support = torch.bincount(
        query_index[broad_observation], minlength=query_count
    )
    triangulated_support = torch.bincount(
        query_index[triangulated[track_index]], minlength=query_count
    )
    covariance = torch.as_tensor(geometry["triangulation_covariance_trace"])
    parallax = torch.as_tensor(geometry["triangulation_parallax_deg"])
    reprojection = torch.as_tensor(geometry["triangulation_reprojection_median_px"])
    return {
        "track_count": int(torch.as_tensor(tracks["track_level"]).numel()),
        "observation_count": int(track_index.numel()),
        "triangulated_track_count": int(triangulated.sum()),
        "high_confidence_track_count": int(
            torch.as_tensor(geometry["triangulation_high_confidence"]).sum()
        ),
        "strict_eligible_track_count": int(strict.sum()),
        "broad_eligible_track_count": int(broad.sum()),
        "triangulated_covariance_trace_m2": _distribution(covariance[triangulated]),
        "broad_covariance_trace_m2": _distribution(covariance[broad]),
        "triangulated_parallax_deg": _distribution(parallax[triangulated]),
        "broad_parallax_deg": _distribution(parallax[broad]),
        "triangulated_reprojection_median_px": _distribution(
            reprojection[triangulated]
        ),
        "broad_track_support_per_mapping_query": _distribution(broad_support),
        "triangulated_track_support_per_mapping_query": _distribution(
            triangulated_support
        ),
        "mapping_query_with_broad_track_fraction": float(
            (broad_support > 0).float().mean() if broad_support.numel() else 0.0
        ),
    }


def _ordered_names_sha256(names: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(str(name) for name in names) + "\n").encode("utf-8")
    ).hexdigest()


def _pair_report(sidecar: dict) -> dict:
    pair = sidecar["pair"]
    selection_parallax = torch.as_tensor(pair["mapping_point_parallax_median_deg"])
    actual_parallax = torch.as_tensor(pair["actual_triangulation_parallax_median_deg"])
    overlap = torch.as_tensor(pair["mapping_point_overlap_jaccard"])
    report = {
        "pair_count": int(torch.as_tensor(pair["left_query_index"]).numel()),
        "baseline_m": _distribution(pair["baseline_m"]),
        "axis_angle_deg": _distribution(pair["axis_angle_deg"]),
        "mapping_point_overlap_jaccard": _distribution(overlap),
        "mapping_point_parallax_median_deg": _distribution(selection_parallax),
        "mapping_point_parallax_below_1deg_fraction": float(
            (selection_parallax[torch.isfinite(selection_parallax)] < 1.0)
            .float()
            .mean()
        ),
        "actual_triangulation_parallax_deg": _distribution(actual_parallax),
    }
    for name in (
        "raw_match_count",
        "accepted_match_count",
        "rejected_ambiguity_count",
        "rejected_epipolar_count",
        "raw_top1_reciprocal_count",
        "descriptor_accepted_before_epipolar_count",
        "epipolar_accepted_top1_count",
        "epipolar_rejected_after_descriptor_count",
        "ambiguity_rejected_count",
        "final_reciprocal_epipolar_count",
        "epipolar_recovered_final_count",
        "cycle_supported_edge_count",
        "graph_accepted_edge_count",
        "conflict_rejected_edge_count",
        "final_component_edge_count",
        "triangulated_track_count",
    ):
        if name in pair:
            value = torch.as_tensor(pair[name]).long()
            report[name] = {
                "total": int(value.sum()),
                "per_pair": _distribution(value),
            }
    return report


def _sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return normalized


def _validate_factor_input_lineage(
    *,
    manifest_payload: dict,
    manifest_path: Path,
    query_cache_path: Path,
    frozen_track_payload_path: Path,
    expected_manifest_sha256: str,
    expected_query_cache_sha256: str,
    expected_frozen_track_payload_sha256: str,
) -> dict:
    """Bind an arm to the exact manifest/cache/Track inputs it consumed."""
    manifest_path = manifest_path.resolve()
    query_cache_path = query_cache_path.resolve()
    frozen_track_payload_path = frozen_track_payload_path.resolve()
    expected_manifest_sha256 = _sha256(
        expected_manifest_sha256, label="Expected manifest SHA-256"
    )
    expected_query_cache_sha256 = _sha256(
        expected_query_cache_sha256, label="Expected query-cache SHA-256"
    )
    expected_frozen_track_payload_sha256 = _sha256(
        expected_frozen_track_payload_sha256,
        label="Expected frozen Track payload SHA-256",
    )
    actual_manifest_sha256 = sha256_file(manifest_path)
    actual_query_cache_sha256 = sha256_file(query_cache_path)
    actual_frozen_track_payload_sha256 = sha256_file(frozen_track_payload_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Manifest SHA-256 differs from the explicit factor contract")
    if actual_query_cache_sha256 != expected_query_cache_sha256:
        raise ValueError(
            "Query-cache SHA-256 differs from the explicit factor contract"
        )
    if actual_frozen_track_payload_sha256 != expected_frozen_track_payload_sha256:
        raise ValueError(
            "Frozen Track payload SHA-256 differs from the explicit factor contract"
        )
    arguments = manifest_payload.get("arguments")
    inputs = manifest_payload.get("inputs")
    if not isinstance(arguments, dict) or not isinstance(inputs, dict):
        raise ValueError("Factor manifest lacks arguments/inputs")
    if Path(str(arguments.get("query_cache_path", ""))).resolve() != query_cache_path:
        raise ValueError("Factor manifest arguments name a different query cache")
    query_input = inputs.get("query_cache_path")
    if not isinstance(query_input, dict):
        raise ValueError("Factor manifest lacks query-cache input lineage")
    if Path(str(query_input.get("path", ""))).resolve() != query_cache_path:
        raise ValueError("Factor manifest input lineage names a different query cache")
    if query_input.get("sha256") != expected_query_cache_sha256:
        raise ValueError(
            "Factor manifest does not bind the expected query-cache SHA-256"
        )
    rebind = manifest_payload.get("equivalent_query_cache_rebind")
    if rebind is not None:
        if not isinstance(rebind, dict):
            raise ValueError("Equivalent query-cache rebind must be a mapping")
        if rebind.get("schema") != "lafgs_equivalent_query_cache_rebind":
            raise ValueError("Unexpected equivalent query-cache rebind schema")
        if rebind.get("version") != 1 or rebind.get("uses_test_queries") is not False:
            raise ValueError("Equivalent query-cache rebind is not mapping-only V1")
        refreshed = rebind.get("refreshed_cache")
        track = rebind.get("source_track_payload")
        equivalence = rebind.get("equivalence_report")
        source = rebind.get("source_cache")
        parent = rebind.get("parent_manifest")
        if not all(
            isinstance(value, dict)
            for value in (refreshed, track, equivalence, source, parent)
        ):
            raise ValueError("Equivalent query-cache rebind is incomplete")
        if (
            Path(str(refreshed.get("path", ""))).resolve() != query_cache_path
            or refreshed.get("sha256") != expected_query_cache_sha256
        ):
            raise ValueError("Rebind names a different refreshed query cache")
        if (
            Path(str(track.get("path", ""))).resolve() != frozen_track_payload_path
            or track.get("sha256") != expected_frozen_track_payload_sha256
        ):
            raise ValueError("Rebind names a different frozen Track payload")
        equivalence_path = Path(str(equivalence.get("path", ""))).resolve()
        if not equivalence_path.is_file() or sha256_file(equivalence_path) != (
            equivalence.get("sha256")
        ):
            raise ValueError("Rebind equivalence report is missing or changed")
        source_path = Path(str(source.get("path", ""))).resolve()
        parent_path = Path(str(parent.get("path", ""))).resolve()
        if (
            not source_path.is_file()
            or sha256_file(source_path) != source.get("sha256")
            or not parent_path.is_file()
            or sha256_file(parent_path) != parent.get("sha256")
        ):
            raise ValueError(
                "Rebind source cache/parent manifest is missing or changed"
            )
        equivalence_payload = json.loads(equivalence_path.read_text())
        if (
            equivalence_payload.get("schema")
            != "lafgs_mapping_sparse_refresh_equivalence"
            or equivalence_payload.get("version") != 2
            or equivalence_payload.get("uses_test_queries") is not False
            or equivalence_payload.get("valid") is not True
            or not equivalence_payload.get("checks")
            or not all(equivalence_payload["checks"].values())
            or equivalence_payload.get("audit", {}).get(
                "content_equivalent_track_payload_reuse_authorized"
            )
            is not True
        ):
            raise ValueError("Rebind equivalence report does not authorize reuse")
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": actual_manifest_sha256,
        },
        "query_cache": {
            "path": str(query_cache_path),
            "sha256": actual_query_cache_sha256,
        },
        "frozen_track_payload": {
            "path": str(frozen_track_payload_path),
            "sha256": actual_frozen_track_payload_sha256,
        },
        "equivalent_query_cache_rebind": deepcopy(rebind),
    }


def _validate_expected_factor_contract(
    *,
    expected_mapping_keypoints: int,
    expected_nms_radius: int,
    expected_pair_budget: int,
    manifest: dict,
    query_cache_payload: dict,
    frozen_track_payload: dict,
) -> tuple[int, int, int]:
    """Bind the factor axes to explicit CLI and frozen-input contracts."""
    expected_mapping_keypoints = int(expected_mapping_keypoints)
    expected_pair_budget = int(expected_pair_budget)
    expected_nms_radius = int(expected_nms_radius)
    if expected_mapping_keypoints <= 0:
        raise ValueError("Expected mapping keypoints must be positive")
    if expected_pair_budget <= 0:
        raise ValueError("Expected pair budget must be positive")
    if expected_nms_radius <= 0:
        raise ValueError("Expected NMS radius must be positive")
    manifest_keypoints = int(manifest.get("native_keypoint_count", -1))
    cache_contract = dict(query_cache_payload.get("signature_payload", {}))
    cache_keypoints = int(cache_contract.get("native_sparse_keypoint_count", -1))
    cache_nms_radius = int(cache_contract.get("native_sparse_nms_radius", -1))
    if manifest_keypoints != expected_mapping_keypoints:
        raise ValueError(
            "Frozen bootstrap mapping K differs from the explicit factor contract"
        )
    if cache_keypoints != expected_mapping_keypoints:
        raise ValueError(
            "Query-cache mapping K differs from the explicit factor contract"
        )
    if cache_nms_radius != expected_nms_radius:
        raise ValueError("Query-cache NMS differs from the explicit factor contract")
    frozen_budget = int(
        frozen_track_payload.get("diagnostics", {}).get(
            "track_camera_pair_candidate_count", -1
        )
    )
    if frozen_budget != expected_pair_budget:
        raise ValueError(
            "Frozen nearest-pair budget differs from the explicit factor contract"
        )
    return expected_mapping_keypoints, expected_pair_budget, expected_nms_radius


def _factor_payload(
    *,
    mapping_keypoints: int,
    nms_radius: int,
    pair_policy: str,
    pair_policy_parameters: dict,
    query_names: list[str],
    query_bins: torch.Tensor,
    tracks: dict,
    track_geometry: dict,
    pair_sidecar: dict,
    diagnostics: dict,
    input_lineage: dict | None = None,
) -> dict:
    """Materialize Track evidence only; provenance assignment is a later stage."""
    return {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "uses_test_queries": False,
        "mapping_keypoint_factor": int(mapping_keypoints),
        "mapping_nms_radius": int(nms_radius),
        "descriptor_factor_mutated": False,
        "density_factor_mutated": False,
        "selector_factor_mutated": False,
        "pair_policy": str(pair_policy),
        "pair_policy_parameters": deepcopy(pair_policy_parameters),
        "query_names": list(query_names),
        "query_names_sha256": _ordered_names_sha256(query_names),
        "query_bins": query_bins,
        "tracks": tracks,
        "track_geometry": track_geometry,
        "pair_sidecar": pair_sidecar,
        "diagnostics": diagnostics,
        "input_lineage": deepcopy(input_lineage),
    }


def _build_report(
    *,
    result: dict,
    frozen: dict,
    sidecar: dict,
    keypoint_counts: torch.Tensor,
    scene_point_count: int,
    pair_budget: int,
    manifest_path: Path,
    query_cache_path: Path,
    frozen_track_payload_path: Path,
) -> dict:
    query_count = len(result["query_names"])
    track_report = _track_report(
        result["tracks"], result["track_geometry"], query_count=query_count
    )
    frozen_track_report = _track_report(
        frozen["tracks"], frozen["track_geometry"], query_count=query_count
    )
    frozen_count_contract = {
        name: {
            "frozen": int(frozen_track_report[name]),
            "replay": int(track_report[name]),
            "equal": int(frozen_track_report[name]) == int(track_report[name]),
        }
        for name in (
            "track_count",
            "observation_count",
            "triangulated_track_count",
            "high_confidence_track_count",
            "strict_eligible_track_count",
            "broad_eligible_track_count",
        )
    }
    pair_policy = str(result["pair_policy"])
    return {
        "schema": result["schema"],
        "version": result["version"],
        "uses_test_queries": False,
        "mapping_keypoint_factor": int(result["mapping_keypoint_factor"]),
        "mapping_nms_radius": int(result["mapping_nms_radius"]),
        "pair_policy": pair_policy,
        "pair_policy_parameters": deepcopy(result["pair_policy_parameters"]),
        "exact_pair_budget": int(pair_budget),
        "scene_point_count": int(scene_point_count),
        "mapping_query_count": len(result["query_names"]),
        "query_names_sha256": result["query_names_sha256"],
        "mapping_keypoints_per_query": _distribution(keypoint_counts),
        "track": track_report,
        "frozen_reference_track": frozen_track_report,
        "frozen_count_contract": frozen_count_contract,
        "nearest_reproduces_all_frozen_counts": (
            all(value["equal"] for value in frozen_count_contract.values())
            if pair_policy == "nearest"
            else None
        ),
        "pair": _pair_report(sidecar),
        "inputs": deepcopy(result.get("input_lineage")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--frozen-track-payload", type=Path, required=True)
    parser.add_argument("--expected-frozen-track-payload-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pair-policy",
        choices=["nearest", "parallax_diverse", "parallax_stratified"],
        required=True,
    )
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument(
        "--expected-pair-budget",
        "--pair-budget",
        dest="expected_pair_budget",
        type=int,
        required=True,
        help=(
            "Exact frozen nearest-pair cardinality. --pair-budget remains an "
            "alias for archived Stairs command compatibility."
        ),
    )
    parser.add_argument("--minimum-overlap-jaccard", type=float, default=0.15)
    parser.add_argument("--minimum-joint-visibility-points", type=int, default=8)
    parser.add_argument("--parallax-saturation-deg", type=float, default=2.0)
    parser.add_argument("--diversity-weight", type=float, default=0.20)
    parser.add_argument("--candidate-pool-per-camera", type=int, default=48)
    parser.add_argument("--scene-points-per-camera", type=int, default=8)
    parser.add_argument("--maximum-scene-points", type=int, default=4096)
    parser.add_argument("--scene-point-voxel-size-m", type=float, default=0.02)
    parser.add_argument("--minimum-expected-parallax-deg", type=float, default=1.0)
    parser.add_argument("--near-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--maximum-baseline-depth-ratio", type=float, default=0.5
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--triangulation-cpu-workers", type=int, default=2)
    parser.add_argument("--parallel-triangulation-min-tracks", type=int, default=5000)
    args = parser.parse_args()
    started = time.perf_counter()
    stage_seconds = {}

    manifest_payload = json.loads(args.manifest.read_text())
    input_lineage = _validate_factor_input_lineage(
        manifest_payload=manifest_payload,
        manifest_path=args.manifest,
        query_cache_path=args.query_cache,
        frozen_track_payload_path=args.frozen_track_payload,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_query_cache_sha256=args.expected_query_cache_sha256,
        expected_frozen_track_payload_sha256=(
            args.expected_frozen_track_payload_sha256
        ),
    )
    frozen = _load(args.frozen_track_payload)
    if frozen.get("schema") != "lafgs_track_first_payload":
        raise ValueError("Unexpected frozen Track payload schema")
    names = [str(value) for value in frozen["query_names"]]
    payload = _load(args.query_cache)
    stage_seconds["load_artifacts"] = time.perf_counter() - started
    cache = payload.get("queries", payload)
    manifest = dict(manifest_payload.get("arguments", {}))
    mapping_keypoints, pair_budget, nms_radius = _validate_expected_factor_contract(
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_pair_budget=args.expected_pair_budget,
        manifest=manifest,
        query_cache_payload=payload,
        frozen_track_payload=frozen,
    )
    if names != list(cache):
        raise ValueError("Query cache does not equal the frozen mapping order")
    for name in names:
        metadata = cache[name].get("native_sparse_metadata")
        if (
            not isinstance(metadata, dict)
            or int(metadata.get("nms_radius", -1)) != nms_radius
            or int(
                metadata.get("requested_keypoint_count", metadata.get("detect_num", -1))
            )
            != mapping_keypoints
        ):
            raise ValueError(
                f"Mapping query {name} does not attest the explicit K/NMS contract"
            )

    descriptors = []
    keypoints = []
    scores = []
    intrinsics = []
    poses = []
    image_hw = []
    depth_at_keypoints = []
    for name in names:
        record = cache[name]
        descriptor = torch.as_tensor(record["native_descriptors"]).float()
        uv = torch.as_tensor(record["native_keypoints"]).float() + float(
            PIXEL_CENTER_OFFSET
        )
        score = torch.as_tensor(record["native_scores"]).float()
        depth_source = record.get(
            "native_depth_at_keypoints", record.get("native_depth")
        )
        if depth_source is None:
            raise ValueError(f"Mapping query {name} lacks native depth")
        descriptors.append(descriptor)
        keypoints.append(uv)
        scores.append(score)
        intrinsics.append(torch.as_tensor(record["native_K"]).float())
        poses.append(torch.as_tensor(record["pose_w2c"]).float())
        image_hw.append(_image_hw(record))
        depth_at_keypoints.append(_sample_depth_at_keypoints(depth_source, uv).float())
    stage_seconds["materialize_camera_tables"] = (
        time.perf_counter() - started - sum(stage_seconds.values())
    )
    intrinsics = torch.stack(intrinsics)
    poses = torch.stack(poses)
    image_hw = torch.stack(image_hw)
    keypoint_counts = torch.as_tensor(
        [int(value.shape[0]) for value in keypoints], dtype=torch.long
    )
    if int(keypoint_counts.max()) > mapping_keypoints:
        raise ValueError("Observed mapping rows exceed the explicit mapping-K contract")
    scene_points = mapping_scene_points_from_depth_samples(
        keypoints,
        depth_at_keypoints,
        intrinsics,
        poses,
        points_per_camera=int(args.scene_points_per_camera),
        maximum_points=int(args.maximum_scene_points),
        voxel_size_m=float(args.scene_point_voxel_size_m),
    )
    scene_depth_m = representative_scene_depth_from_samples(
        depth_at_keypoints
    )
    stage_seconds["mapping_scene_points"] = (
        time.perf_counter() - started - sum(stage_seconds.values())
    )
    build_result = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        detector_scores=scores,
        camera_K=intrinsics,
        pose_w2c=poses,
        pair_neighbors=int(manifest.get("geometry_teacher_track_pair_neighbors", 6)),
        pair_policy=str(args.pair_policy),
        pair_budget=pair_budget,
        pair_image_hw=image_hw,
        pair_scene_points_xyz=scene_points,
        pair_minimum_overlap_jaccard=float(args.minimum_overlap_jaccard),
        pair_minimum_joint_visibility_points=int(args.minimum_joint_visibility_points),
        pair_parallax_saturation_deg=float(args.parallax_saturation_deg),
        pair_diversity_weight=float(args.diversity_weight),
        pair_candidate_pool_per_camera=int(args.candidate_pool_per_camera),
        pair_scene_depth_m=scene_depth_m,
        pair_minimum_expected_parallax_deg=float(
            args.minimum_expected_parallax_deg
        ),
        pair_near_fraction=float(args.near_fraction),
        pair_maximum_baseline_depth_ratio=float(
            args.maximum_baseline_depth_ratio
        ),
        minimum_baseline_m=float(
            manifest.get("geometry_teacher_track_min_baseline_m", 0.03)
        ),
        maximum_baseline_m=float(
            manifest.get("geometry_teacher_track_max_baseline_m", 5.0)
        ),
        maximum_axis_angle_deg=float(
            manifest.get("geometry_teacher_track_max_axis_angle_deg", 75.0)
        ),
        minimum_similarity=float(
            manifest.get("geometry_teacher_track_min_similarity", 0.65)
        ),
        minimum_margin=float(manifest.get("geometry_teacher_track_min_margin", 0.01)),
        maximum_epipolar_error_px=float(
            manifest.get("geometry_teacher_track_max_epipolar_error_px", 2.0)
        ),
        epipolar_candidate_topk=int(
            manifest.get("geometry_teacher_track_epipolar_candidate_topk", 1)
        ),
        epipolar_recovered_minimum_similarity=float(
            manifest.get(
                "geometry_teacher_track_epipolar_recovered_min_similarity", -1.0
            )
        ),
        epipolar_recovered_minimum_margin=float(
            manifest.get("geometry_teacher_track_epipolar_recovered_min_margin", -1.0)
        ),
        minimum_track_views=int(manifest.get("geometry_teacher_min_views", 3)),
        require_cycle=bool(manifest.get("geometry_teacher_track_require_cycle", True)),
        allow_chain_tracks=bool(
            manifest.get("geometry_teacher_track_allow_chain_tracks", False)
        ),
        return_pair_sidecar=True,
        device=args.device,
    )
    stage_seconds["pair_matching_and_track_build"] = (
        time.perf_counter() - started - sum(stage_seconds.values())
    )
    tracks, diagnostics, sidecar = build_result
    actual_pair_count = int(
        torch.as_tensor(sidecar["pair"]["left_query_index"]).numel()
    )
    if actual_pair_count != pair_budget:
        raise ValueError(
            "Pair policy did not fill the exact global pair budget: "
            f"expected={pair_budget} actual={actual_pair_count}"
        )

    observation_query = tracks["query_index"]
    observation_keypoint = tracks["keypoint_index"]
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), keypoint_counts.cumsum(0)))
    all_keypoints = torch.cat(keypoints)
    observation_uv = all_keypoints[offsets[observation_query] + observation_keypoint]
    rendered_depth = torch.empty(observation_query.numel(), dtype=torch.float32)
    order = torch.argsort(observation_query, stable=True)
    counts = torch.bincount(observation_query, minlength=len(names))
    observation_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), counts.cumsum(0))
    )
    for query in range(len(names)):
        begin = int(observation_offsets[query])
        end = int(observation_offsets[query + 1])
        if begin == end:
            continue
        rows = order[begin:end]
        rendered_depth[rows] = depth_at_keypoints[query][observation_keypoint[rows]]
    query_bins = camera_pose_bins(
        poses,
        int(manifest.get("geometry_teacher_view_bins", 8)),
        direction_weight=float(
            manifest.get("geometry_teacher_view_direction_weight", 0.5)
        ),
    )
    triangulate = robust_triangulate_associations
    triangulation_extra = {}
    if (
        int(args.triangulation_cpu_workers) > 1
        and int(diagnostics["track_count"])
        >= int(args.parallel_triangulation_min_tracks)
    ):
        triangulate = robust_triangulate_associations_fresh_cpu
        triangulation_extra["worker_count"] = int(args.triangulation_cpu_workers)
    geometry = triangulate(
        landmark_count=int(diagnostics["track_count"]),
        landmark_index=tracks["track_index"],
        query_index=observation_query,
        uv=observation_uv,
        confidence=tracks["confidence"],
        camera_K=intrinsics,
        pose_w2c=poses,
        query_bin=query_bins,
        rendered_depth=rendered_depth,
        maximum_observations_per_landmark=int(
            manifest.get("geometry_teacher_max_observations_per_landmark", 32)
        ),
        minimum_views=int(manifest.get("geometry_teacher_min_views", 3)),
        minimum_view_bins=int(manifest.get("geometry_teacher_min_view_bins", 2)),
        huber_delta_px=float(manifest.get("geometry_teacher_huber_delta_px", 2.0)),
        iterations=int(manifest.get("geometry_teacher_iterations", 3)),
        minimum_parallax_deg=float(
            manifest.get("geometry_teacher_min_parallax_deg", 1.0)
        ),
        parallax_quantile=float(
            manifest.get("geometry_teacher_parallax_quantile", 0.75)
        ),
        maximum_reprojection_px=float(
            manifest.get("geometry_teacher_max_reprojection_px", 2.0)
        ),
        maximum_condition_number=float(
            manifest.get("geometry_teacher_max_condition_number", 1e6)
        ),
        maximum_covariance_trace_m2=float(
            manifest.get("geometry_teacher_max_covariance_trace_m2", float("inf"))
        ),
        maximum_rendered_depth_residual_m=float(
            manifest.get("geometry_teacher_max_rendered_depth_residual_m", float("inf"))
        ),
        minimum_rendered_depth_observations=int(
            manifest.get("geometry_teacher_min_rendered_depth_observations", 0)
        ),
        surface_support_enabled=False,
        **triangulation_extra,
    )
    stage_seconds["robust_triangulation"] = (
        time.perf_counter() - started - sum(stage_seconds.values())
    )
    geometry["track_confidence_level"] = tracks["track_level"].clone()
    attach_pair_triangulation_statistics(sidecar, tracks, geometry, poses)
    stage_seconds["attach_pair_triangulation"] = (
        time.perf_counter() - started - sum(stage_seconds.values())
    )

    if args.pair_policy == "parallax_stratified":
        pair_policy_parameters = {
            "minimum_expected_parallax_deg": float(
                args.minimum_expected_parallax_deg
            ),
            "near_fraction": float(args.near_fraction),
            "maximum_baseline_depth_ratio": float(
                args.maximum_baseline_depth_ratio
            ),
            "scene_depth_estimator": "median_positive_mapping_keypoint_depth",
        }
    else:
        pair_policy_parameters = {
            "minimum_overlap_jaccard": float(args.minimum_overlap_jaccard),
            "minimum_joint_visibility_points": int(
                args.minimum_joint_visibility_points
            ),
            "parallax_saturation_deg": float(args.parallax_saturation_deg),
            "diversity_weight": float(args.diversity_weight),
            "candidate_pool_per_camera": int(args.candidate_pool_per_camera),
            "scene_points_per_camera": int(args.scene_points_per_camera),
            "maximum_scene_points": int(args.maximum_scene_points),
            "scene_point_voxel_size_m": float(args.scene_point_voxel_size_m),
        }
    result = _factor_payload(
        mapping_keypoints=mapping_keypoints,
        nms_radius=nms_radius,
        pair_policy=args.pair_policy,
        pair_policy_parameters=pair_policy_parameters,
        query_names=names,
        query_bins=query_bins,
        tracks=tracks,
        track_geometry=geometry,
        pair_sidecar=sidecar,
        diagnostics=diagnostics,
        input_lineage=input_lineage,
    )
    report = _build_report(
        result=result,
        frozen=frozen,
        sidecar=sidecar,
        keypoint_counts=keypoint_counts,
        scene_point_count=int(scene_points.shape[0]),
        pair_budget=pair_budget,
        manifest_path=args.manifest,
        query_cache_path=args.query_cache,
        frozen_track_payload_path=args.frozen_track_payload,
    )
    report["stage_seconds"] = {
        name: float(value) for name, value in stage_seconds.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output_dir / f"{args.pair_policy}_track_factor.pt"
    report_path = args.output_dir / f"{args.pair_policy}_track_factor.json"
    torch.save(result, artifact_path)
    report["artifact"] = str(artifact_path.resolve())
    report["artifact_sha256"] = sha256_file(artifact_path)
    report["stage_seconds"]["total"] = float(time.perf_counter() - started)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
