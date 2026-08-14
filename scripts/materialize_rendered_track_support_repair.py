#!/usr/bin/env python3
"""Repair frozen rendered Tracks with uncertain alpha/depth support evidence.

The camera-pair table, SuperPoint rows, original reciprocal matcher, and source
Track identities are frozen inputs.  Support may only remove/reweight edges
inside an existing Track component; it can never join different source Tracks.
Every retained child is re-triangulated from camera rays, with at most a
bounded number of children per source component.  Gaussian depth is used as
uncertain evidence, never as landmark xyz.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.rendered_track_support import local_depth_spread, pair_support_evidence
from evidence.tracks import fuse_track_descriptors
from evidence.triangulation import (
    build_cycle_consistent_tracks,
    reciprocal_epipolar_matches,
    robust_triangulate_associations,
)
from features.multiview_fusion import PIXEL_CENTER_OFFSET
from map_learning.metric import SharedLowRankMetric
from topology.track_core import _eligible_tracks, _track_quality


_PRODUCER_SOURCE_PATHS = (
    "scripts/materialize_rendered_track_support_repair.py",
    "evidence/rendered_track_support.py",
    "evidence/triangulation.py",
    "evidence/tracks.py",
    "topology/track_core.py",
    "features/multiview_fusion.py",
    "common/hashing.py",
)


def _producer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain=v1")
    if dirty:
        raise RuntimeError("support-repair producer worktree must be clean")
    source_sha256 = {}
    for relative in _PRODUCER_SOURCE_PATHS:
        path = repository / relative
        if not path.is_file():
            raise RuntimeError(f"support-repair producer source is missing: {relative}")
        source_sha256[relative] = sha256_file(path)
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": source_sha256,
        "torch_version": torch.__version__,
    }


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary support artifact did not reload")
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


def _require_sha(path: Path, expected: str, name: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{name} SHA differs: expected {expected}, got {actual}")
    return actual


def _source_track_lookup(
    payload: dict, keypoint_counts: list[int]
) -> list[torch.Tensor]:
    lookup = [torch.full((count,), -1, dtype=torch.long) for count in keypoint_counts]
    tracks = payload["tracks"]
    for track, query, keypoint in zip(
        torch.as_tensor(tracks["track_index"]).long().tolist(),
        torch.as_tensor(tracks["query_index"]).long().tolist(),
        torch.as_tensor(tracks["keypoint_index"]).long().tolist(),
    ):
        if int(keypoint) < 0 or int(keypoint) >= keypoint_counts[int(query)]:
            raise ValueError("source Track keypoint is outside the frozen cache")
        if int(lookup[int(query)][int(keypoint)]) >= 0:
            raise ValueError("source payload assigns one row to multiple Tracks")
        lookup[int(query)][int(keypoint)] = int(track)
    return lookup


def _limit_children(
    tracks: dict[str, torch.Tensor],
    source_lookup: list[torch.Tensor],
    *,
    maximum_children: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    if int(maximum_children) < 1:
        raise ValueError("maximum children per source Track must be positive")
    track = torch.as_tensor(tracks["track_index"]).long()
    query = torch.as_tensor(tracks["query_index"]).long()
    keypoint = torch.as_tensor(tracks["keypoint_index"]).long()
    confidence = torch.as_tensor(tracks["confidence"]).float()
    track_count = int(torch.as_tensor(tracks["track_level"]).numel())
    source_by_child = torch.full((track_count,), -1, dtype=torch.long)
    count = torch.bincount(track, minlength=track_count)
    confidence_sum = torch.zeros(track_count)
    confidence_sum.index_add_(0, track, confidence)
    for child, query_row, keypoint_row in zip(
        track.tolist(), query.tolist(), keypoint.tolist()
    ):
        source = int(source_lookup[int(query_row)][int(keypoint_row)])
        if source < 0:
            raise ValueError("repaired observation lacks a source Track identity")
        existing = int(source_by_child[int(child)])
        if existing >= 0 and existing != source:
            raise ValueError("support repair merged different source Tracks")
        source_by_child[int(child)] = source
    children: dict[int, list[int]] = {}
    for child, source in enumerate(source_by_child.tolist()):
        children.setdefault(int(source), []).append(child)
    retained = []
    split_source_count = 0
    dropped_child_count = 0
    for source in sorted(children):
        ordered = sorted(
            children[source],
            key=lambda child: (
                -int(count[child]),
                -float(confidence_sum[child]),
                child,
            ),
        )
        if len(ordered) > 1:
            split_source_count += 1
        retained.extend(ordered[: int(maximum_children)])
        dropped_child_count += max(0, len(ordered) - int(maximum_children))
    retained = sorted(retained)
    retain_lookup = torch.full((track_count,), -1, dtype=torch.long)
    retain_lookup[torch.as_tensor(retained)] = torch.arange(len(retained))
    observation_keep = retain_lookup[track] >= 0
    revised = {
        "track_index": retain_lookup[track[observation_keep]],
        "query_index": query[observation_keep],
        "keypoint_index": keypoint[observation_keep],
        "confidence": confidence[observation_keep],
        "track_level": torch.as_tensor(tracks["track_level"])[retained].clone(),
        "source_track_index": source_by_child[retained].clone(),
    }
    return revised, {
        "unbounded_child_track_count": track_count,
        "retained_child_track_count": len(retained),
        "split_source_track_count": split_source_count,
        "dropped_excess_child_count": dropped_child_count,
    }


def _coverage_certification(
    *,
    tracks: dict[str, torch.Tensor],
    geometry: dict,
    support_records: list[dict],
    keypoints: list[torch.Tensor],
    poses: torch.Tensor,
    depth_uncertainty: list[torch.Tensor],
    depth_abs_tolerance_m: float,
    depth_relative_tolerance: float,
    minimum_view_bins: int,
    minimum_parallax_deg: float,
    maximum_reprojection_px: float,
) -> torch.Tensor:
    track = torch.as_tensor(tracks["track_index"]).long()
    query = torch.as_tensor(tracks["query_index"]).long()
    keypoint_index = torch.as_tensor(tracks["keypoint_index"]).long()
    xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()[track]
    camera = (
        torch.bmm(poses[query, :3, :3].float(), xyz[:, :, None]).squeeze(2)
        + poses[query, :3, 3].float()
    )
    predicted_depth = camera[:, 2]
    reference_depth = torch.as_tensor(
        [
            support_records[int(q)]["depth"][int(k)]
            for q, k in zip(query.tolist(), keypoint_index.tolist())
        ]
    ).float()
    uncertainty = torch.as_tensor(
        [
            depth_uncertainty[int(q)][int(k)]
            for q, k in zip(query.tolist(), keypoint_index.tolist())
        ]
    ).float()
    alpha_valid = torch.as_tensor(
        [
            support_records[int(q)]["valid"][int(k)]
            for q, k in zip(query.tolist(), keypoint_index.tolist())
        ],
        dtype=torch.bool,
    )
    tolerance = (
        float(depth_abs_tolerance_m)
        + float(depth_relative_tolerance) * reference_depth.abs()
        + uncertainty.nan_to_num(posinf=1e6)
    )
    depth_consistent = (
        torch.isfinite(reference_depth)
        & torch.isfinite(predicted_depth)
        & ((predicted_depth - reference_depth).abs() <= 3.0 * tolerance)
    )
    strong_track = (
        torch.as_tensor(geometry["triangulated"]).bool()
        & (
            torch.as_tensor(geometry["triangulation_distinct_view_bin_count"])
            >= int(minimum_view_bins)
        )
        & (
            torch.as_tensor(geometry["triangulation_parallax_deg"])
            >= float(minimum_parallax_deg)
        )
        & (
            torch.as_tensor(geometry["triangulation_reprojection_p90_px"])
            <= float(maximum_reprojection_px)
        )
    )
    # Strong projective geometry may override uncertain expected depth, but an
    # alpha-invalid observation never certifies mapping coverage.
    return alpha_valid & (depth_consistent | strong_track[track])


@torch.no_grad()
def materialize(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    producer_identity = _producer_identity()
    for path in (args.source_cache, args.support_cache, args.source_track_payload):
        if not path.is_file():
            raise FileNotFoundError(path)
    input_sha = {
        "source_cache": _require_sha(
            args.source_cache, args.expected_source_cache_sha256, "source cache"
        ),
        "support_cache": _require_sha(
            args.support_cache, args.expected_support_cache_sha256, "support cache"
        ),
        "source_track_payload": _require_sha(
            args.source_track_payload,
            args.expected_source_track_payload_sha256,
            "source Track payload",
        ),
    }
    source_payload = torch.load(
        args.source_cache, map_location="cpu", weights_only=False
    )
    support_payload = torch.load(
        args.support_cache, map_location="cpu", weights_only=False
    )
    source_tracks = torch.load(
        args.source_track_payload, map_location="cpu", weights_only=False
    )
    if (
        source_payload.get("uses_source_mapping_rgb") is not False
        or support_payload.get("uses_source_mapping_rgb") is not False
        or source_payload.get("uses_test_queries") is not False
        or support_payload.get("uses_test_queries") is not False
    ):
        raise ValueError("support repair accepts mapping-render-only caches")
    if source_tracks.get("rendered_rgb_only") is not True:
        raise ValueError("source Track payload is not rendered-RGB-only")
    source_cache = source_payload["queries"]
    support_cache = support_payload["queries"]
    names = list(source_tracks["query_names"])
    if names != list(source_cache) or names != list(support_cache):
        raise ValueError("query order differs across support-repair inputs")

    descriptors = []
    keypoints = []
    scores = []
    intrinsics = []
    poses = []
    support_records = []
    uncertainties = []
    for name in names:
        source = source_cache[name]
        support = support_cache[name]
        source_keypoints = torch.as_tensor(source["native_keypoints"]).float()
        if not torch.equal(
            source_keypoints, torch.as_tensor(support["native_keypoints"]).float()
        ):
            raise ValueError(f"support cache changed frozen keypoint rows for {name}")
        descriptors.append(
            F.normalize(torch.as_tensor(source["native_descriptors"]).float(), dim=1)
        )
        keypoints.append(source_keypoints + float(PIXEL_CENTER_OFFSET))
        scores.append(torch.as_tensor(source["native_scores"]).float())
        intrinsics.append(torch.as_tensor(source["native_K"]).float())
        poses.append(torch.as_tensor(source["pose_w2c"]).float())
        alpha = torch.as_tensor(support["native_rendered_alpha"]).float()
        depth = torch.as_tensor(support["native_rendered_depth"]).float()
        valid = torch.as_tensor(support["native_valid_keypoint_mask"]).bool()
        alpha_at = torch.as_tensor(support["native_alpha_at_keypoints"]).float()
        depth_at = torch.as_tensor(support["native_depth_at_keypoints"]).float()
        reliability = torch.as_tensor(
            support.get("native_appearance_reliability", torch.ones(valid.numel()))
        ).float()
        uncertainty = local_depth_spread(
            depth,
            alpha,
            source_keypoints,
            alpha_minimum=float(args.alpha_minimum),
            radius=int(args.local_depth_radius),
        )
        support_records.append(
            {
                "valid": valid,
                "alpha": alpha_at,
                "depth": depth_at,
                "reliability": reliability,
            }
        )
        uncertainties.append(uncertainty)
    intrinsics_tensor = torch.stack(intrinsics)
    poses_tensor = torch.stack(poses)
    keypoint_counts = [int(rows.shape[0]) for rows in keypoints]
    source_lookup = _source_track_lookup(source_tracks, keypoint_counts)

    pair_table = source_tracks["pair_sidecar"]["pair"]
    pairs = list(
        zip(
            torch.as_tensor(pair_table["left_query_index"]).long().tolist(),
            torch.as_tensor(pair_table["right_query_index"]).long().tolist(),
        )
    )
    if pairs != sorted(set(pairs)):
        raise ValueError("source pair table is not canonical")
    precomputed_matches = {}
    precomputed_diagnostics = {}
    total_raw = total_within_source = total_hard_reject = 0
    total_high_confidence = 0
    weight_sum = cycle_sum = depth_sigma_sum = 0.0
    diagnostic_valid_edge_count = 0
    device = torch.device(args.device)
    for completed, (left, right) in enumerate(pairs, start=1):
        source, target, confidence, diagnostic = reciprocal_epipolar_matches(
            descriptors[left].to(device),
            descriptors[right].to(device),
            keypoints[left],
            keypoints[right],
            intrinsics_tensor[left],
            poses_tensor[left],
            intrinsics_tensor[right],
            poses_tensor[right],
            minimum_similarity=float(args.minimum_similarity),
            minimum_margin=float(args.minimum_margin),
            maximum_epipolar_error_px=float(args.maximum_epipolar_error_px),
            epipolar_candidate_topk=int(args.epipolar_candidate_topk),
            recovered_minimum_similarity=-1.0,
            recovered_minimum_margin=-1.0,
            return_diagnostics=True,
        )
        source = source.cpu().long()
        target = target.cpu().long()
        confidence = confidence.cpu().float() * torch.sqrt(
            scores[left][source].clamp_min(0) * scores[right][target].clamp_min(0)
        )
        total_raw += int(source.numel())
        same_source = (source_lookup[left][source] >= 0) & (
            source_lookup[left][source] == source_lookup[right][target]
        )
        total_within_source += int(same_source.sum())
        evidence = pair_support_evidence(
            left_uv=keypoints[left][source],
            right_uv=keypoints[right][target],
            left_depth=support_records[left]["depth"][source],
            right_depth=support_records[right]["depth"][target],
            left_alpha=support_records[left]["alpha"][source],
            right_alpha=support_records[right]["alpha"][target],
            left_valid=support_records[left]["valid"][source],
            right_valid=support_records[right]["valid"][target],
            left_uncertainty=uncertainties[left][source],
            right_uncertainty=uncertainties[right][target],
            left_intrinsic=intrinsics_tensor[left],
            right_intrinsic=intrinsics_tensor[right],
            left_pose_w2c=poses_tensor[left],
            right_pose_w2c=poses_tensor[right],
            left_reliability=support_records[left]["reliability"][source],
            right_reliability=support_records[right]["reliability"][target],
            depth_abs_tolerance_m=float(args.depth_abs_tolerance_m),
            depth_relative_tolerance=float(args.depth_relative_tolerance),
            hard_alpha_minimum=float(args.hard_alpha_minimum),
            soft_cycle_px=float(args.soft_cycle_px),
            hard_cycle_px=float(args.hard_cycle_px),
            hard_depth_sigma=float(args.hard_depth_sigma),
            uncertain_weight_floor=float(args.uncertain_weight_floor),
        )
        keep = same_source & ~evidence["hard_reject"]
        total_hard_reject += int((same_source & evidence["hard_reject"]).sum())
        total_high_confidence += int(
            (same_source & evidence["high_confidence_support"]).sum()
        )
        if bool(same_source.any()):
            weight_sum += float(evidence["soft_weight"][same_source].sum())
        diagnostic_valid = (
            same_source
            & evidence["valid_support_pair"]
            & torch.isfinite(evidence["cycle_error_px"])
            & torch.isfinite(evidence["depth_disagreement_sigma"])
        )
        if bool(diagnostic_valid.any()):
            diagnostic_valid_edge_count += int(diagnostic_valid.sum())
            cycle_sum += float(evidence["cycle_error_px"][diagnostic_valid].sum())
            depth_sigma_sum += float(
                evidence["depth_disagreement_sigma"][diagnostic_valid].sum()
            )
        precomputed_matches[(left, right)] = (
            source[keep],
            target[keep],
            confidence[keep] * evidence["soft_weight"][keep],
        )
        precomputed_diagnostics[(left, right)] = diagnostic
        if completed % max(int(args.progress_interval), 1) == 0 or completed == len(
            pairs
        ):
            print(
                json.dumps(
                    {
                        "completed_pairs": completed,
                        "within_source_edges": total_within_source,
                        "hard_rejected_edges": total_hard_reject,
                    }
                ),
                flush=True,
            )

    repaired, track_diagnostics, pair_sidecar = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        detector_scores=scores,
        camera_K=intrinsics_tensor,
        pose_w2c=poses_tensor,
        pair_neighbors=int(args.pair_neighbors),
        pair_policy=str(source_tracks["pair_sidecar"]["policy"]["name"]),
        pair_budget=len(pairs),
        minimum_baseline_m=float(args.minimum_baseline_m),
        maximum_baseline_m=float(args.maximum_baseline_m),
        maximum_axis_angle_deg=float(args.maximum_axis_angle_deg),
        minimum_similarity=float(args.minimum_similarity),
        minimum_margin=float(args.minimum_margin),
        maximum_epipolar_error_px=float(args.maximum_epipolar_error_px),
        epipolar_candidate_topk=int(args.epipolar_candidate_topk),
        minimum_track_views=int(args.minimum_views),
        require_cycle=True,
        allow_chain_tracks=True,
        return_pair_sidecar=True,
        precomputed_pairs=pairs,
        precomputed_pair_matches=precomputed_matches,
        precomputed_pair_match_diagnostics=precomputed_diagnostics,
        precomputed_confidence_includes_detector_scores=True,
        device=args.device,
    )
    repaired, split_diagnostics = _limit_children(
        repaired, source_lookup, maximum_children=int(args.maximum_children)
    )
    observation_query = repaired["query_index"].long()
    observation_keypoint = repaired["keypoint_index"].long()
    observation_uv = torch.stack(
        [
            keypoints[int(query)][int(keypoint)]
            for query, keypoint in zip(
                observation_query.tolist(), observation_keypoint.tolist()
            )
        ]
    )
    query_bins = torch.as_tensor(source_tracks["query_bins"]).long()
    geometry = robust_triangulate_associations(
        landmark_count=int(repaired["track_level"].numel()),
        landmark_index=repaired["track_index"],
        query_index=observation_query,
        uv=observation_uv,
        confidence=repaired["confidence"],
        camera_K=intrinsics_tensor,
        pose_w2c=poses_tensor,
        query_bin=query_bins,
        rendered_depth=None,
        maximum_observations_per_landmark=int(args.maximum_observations),
        minimum_views=int(args.minimum_views),
        minimum_view_bins=int(args.minimum_view_bins),
        huber_delta_px=float(args.huber_delta_px),
        iterations=int(args.triangulation_iterations),
        minimum_parallax_deg=float(args.minimum_parallax_deg),
        parallax_quantile=float(args.parallax_quantile),
        maximum_reprojection_px=float(args.maximum_reprojection_px),
        maximum_condition_number=float(args.maximum_condition_number),
        maximum_covariance_trace_m2=float("inf"),
        maximum_rendered_depth_residual_m=float("inf"),
        minimum_rendered_depth_observations=0,
        surface_support_enabled=False,
    )
    geometry["track_confidence_level"] = repaired["track_level"].clone()
    repaired["coverage_certified"] = _coverage_certification(
        tracks=repaired,
        geometry=geometry,
        support_records=support_records,
        keypoints=keypoints,
        poses=poses_tensor,
        depth_uncertainty=uncertainties,
        depth_abs_tolerance_m=float(args.depth_abs_tolerance_m),
        depth_relative_tolerance=float(args.depth_relative_tolerance),
        minimum_view_bins=int(args.minimum_view_bins),
        minimum_parallax_deg=float(args.minimum_parallax_deg),
        maximum_reprojection_px=float(args.maximum_reprojection_px),
    )
    payload = {
        "schema": "lafgs_track_first_payload",
        "version": 1,
        "query_names": names,
        "tracks": repaired,
        "track_geometry": geometry,
        "query_bins": query_bins,
        "diagnostics": {**track_diagnostics, **split_diagnostics},
        "pair_sidecar": pair_sidecar,
        "rendered_rgb_only": True,
        "support_repair": {
            "schema": "lafgs_rendered_track_support_repair",
            "version": 1,
            "source_track_payload": str(args.source_track_payload.resolve()),
            "source_track_payload_sha256": input_sha["source_track_payload"],
            "forbids_cross_source_track_merge": True,
            "maximum_children_per_source_track": int(args.maximum_children),
            "uses_gaussian_geometry_for_triangulation": False,
        },
    }
    broad = _eligible_tracks(geometry, "broad")
    broad_tracks = torch.nonzero(broad, as_tuple=False).reshape(-1)
    quality = _track_quality(geometry)
    broad_tracks = broad_tracks[
        torch.argsort(quality[broad_tracks], descending=True, stable=True)
    ]
    fused = fuse_track_descriptors(
        payload=payload,
        query_cache=support_payload,
        track_indices=broad_tracks,
        trim_fraction=float(args.descriptor_trim_fraction),
    )
    xyz = torch.as_tensor(geometry["triangulated_xyz"])[broad_tracks].float()
    covariance = torch.as_tensor(geometry["triangulation_covariance_matrix"])[
        broad_tracks
    ].float()
    anchor_count = int(broad_tracks.numel())
    anchor_map = {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": torch.arange(anchor_count),
        "anchor_xyz": xyz,
        "anchor_features": fused.float(),
        "source_primitive_ids": torch.full((anchor_count,), -1, dtype=torch.long),
        "track_cluster_ids": broad_tracks,
        "anchor_type": torch.ones(anchor_count, dtype=torch.long),
        "dependency_group_ids": torch.arange(anchor_count),
        "coarse_dependency_group_ids": torch.arange(anchor_count),
        "fine_identity_ids": broad_tracks.clone(),
        "anchor_position_covariance": covariance,
        "anchor_matchability": torch.ones(anchor_count),
        "base_anchor_count": 0,
        "canonical_anchor_count": anchor_count,
        "micro_anchor_count": anchor_count,
        "provenance": {
            "mapping_rgb_source": "gaussian_render_only",
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "uses_gaussian_geometry_for_triangulation": False,
            "support_repaired_track_candidates": True,
        },
    }
    descriptor_dim = int(fused.shape[1])
    empty_map = {
        "schema": "lafgs_materialized_anchor_map",
        "version": 1,
        "anchor_ids": torch.empty(0, dtype=torch.long),
        "anchor_xyz": torch.empty((0, 3)),
        "anchor_features": torch.empty((0, descriptor_dim)),
        "source_primitive_ids": torch.empty(0, dtype=torch.long),
        "track_cluster_ids": torch.empty(0, dtype=torch.long),
        "anchor_type": torch.empty(0, dtype=torch.long),
        "dependency_group_ids": torch.empty(0, dtype=torch.long),
        "coarse_dependency_group_ids": torch.empty(0, dtype=torch.long),
        "fine_identity_ids": torch.empty(0, dtype=torch.long),
        "base_anchor_count": 0,
        "canonical_anchor_count": 0,
        "micro_anchor_count": 0,
    }
    empty_records = []
    for query_index, name in enumerate(names):
        valid_rows = torch.nonzero(
            support_records[query_index]["valid"], as_tuple=False
        ).reshape(-1)
        empty_records.append(
            {
                "query_index": query_index,
                "query_name": name,
                "query_rows": valid_rows,
                "positive_offsets": torch.zeros(
                    valid_rows.numel() + 1, dtype=torch.long
                ),
                "positive_indices": torch.empty(0, dtype=torch.long),
                "ambiguous_offsets": torch.zeros(
                    valid_rows.numel() + 1, dtype=torch.long
                ),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        )
    selector_teacher = {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "anchor_count": 0,
        "query_names": names,
        "records": empty_records,
        "config": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
            "purpose": "support_repaired_track_only_selector_query_registry",
        },
    }
    empty_graph = {
        "schema": "lafgs_rendered_track_only_empty_base_graph",
        "version": 1,
        "query_names": names,
        "records": [],
        "uses_test_queries": False,
        "base_anchor_count": 0,
    }
    metric = SharedLowRankMetric(
        descriptor_dim=descriptor_dim, rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    identity_metric = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(anchor_count),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            name: value.cpu() for name, value in metric.state_dict().items()
        },
        "map_path": str(
            (args.output_dir / "support_repaired_candidate_map.pt").resolve()
        ),
        "step": 0,
        "protocol": "rendered_track_support_repaired_identity",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "track_payload": args.output_dir / "support_repaired_track_payload.pt",
        "candidate_map": args.output_dir / "support_repaired_candidate_map.pt",
        "identity_metric": args.output_dir / "support_repaired_identity_metric.pt",
        "empty_canonical_map": args.output_dir / "empty_canonical_map.pt",
        "empty_function_graph": args.output_dir / "empty_function_graph.pt",
        "selector_teacher": args.output_dir / "selector_teacher.pt",
    }
    for name, value in (
        ("track_payload", payload),
        ("candidate_map", anchor_map),
        ("identity_metric", identity_metric),
        ("empty_canonical_map", empty_map),
        ("empty_function_graph", empty_graph),
        ("selector_teacher", selector_teacher),
    ):
        _atomic_save(value, outputs[name])
    observation_count = int(repaired["track_index"].numel())
    report = {
        "schema": "lafgs_rendered_track_support_repair_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_gaussian_geometry_for_triangulation": False,
        "producer_identity": producer_identity,
        "pair_policy": source_tracks["pair_sidecar"]["policy"]["name"],
        "pair_count": len(pairs),
        "source_track_count": int(
            torch.as_tensor(source_tracks["track_geometry"]["triangulated"]).numel()
        ),
        "repaired_track_count": int(repaired["track_level"].numel()),
        "triangulated_track_count": int(
            torch.as_tensor(geometry["triangulated"]).sum()
        ),
        "broad_track_count": anchor_count,
        "observation_count": observation_count,
        "coverage_certified_observation_count": int(
            repaired["coverage_certified"].sum()
        ),
        "edge_diagnostics": {
            "recomputed_raw_match_count": total_raw,
            "within_source_component_edge_count": total_within_source,
            "high_confidence_support_edge_count": total_high_confidence,
            "hard_rejected_edge_count": total_hard_reject,
            "valid_support_diagnostic_edge_count": diagnostic_valid_edge_count,
            "mean_soft_weight": weight_sum / max(total_within_source, 1),
            "mean_cycle_error_px": cycle_sum / max(diagnostic_valid_edge_count, 1),
            "mean_depth_disagreement_sigma": depth_sigma_sum
            / max(diagnostic_valid_edge_count, 1),
        },
        "split_diagnostics": split_diagnostics,
        "configuration": {
            "alpha_minimum": float(args.alpha_minimum),
            "hard_alpha_minimum": float(args.hard_alpha_minimum),
            "depth_abs_tolerance_m": float(args.depth_abs_tolerance_m),
            "depth_relative_tolerance": float(args.depth_relative_tolerance),
            "soft_cycle_px": float(args.soft_cycle_px),
            "hard_cycle_px": float(args.hard_cycle_px),
            "hard_depth_sigma": float(args.hard_depth_sigma),
            "uncertain_weight_floor": float(args.uncertain_weight_floor),
            "maximum_children_per_source_track": int(args.maximum_children),
            "coverage_policy": "alpha_valid_and_depth_consistent_or_strong_projective_geometry",
        },
        "inputs": {
            "source_cache": str(args.source_cache.resolve()),
            "support_cache": str(args.support_cache.resolve()),
            "source_track_payload": str(args.source_track_payload.resolve()),
        },
        "input_sha256": input_sha,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
        "output_sha256": {name: sha256_file(path) for name, path in outputs.items()},
        "timing_seconds": {"total": time.perf_counter() - started},
    }
    if _producer_identity() != producer_identity:
        raise RuntimeError("support-repair producer identity changed during materialization")
    _atomic_json(report, args.output_dir / "support_repair_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--support-cache", type=Path, required=True)
    parser.add_argument("--expected-support-cache-sha256", required=True)
    parser.add_argument("--source-track-payload", type=Path, required=True)
    parser.add_argument("--expected-source-track-payload-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-similarity", type=float, default=0.65)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--maximum-epipolar-error-px", type=float, default=2.0)
    parser.add_argument("--epipolar-candidate-topk", type=int, default=4)
    parser.add_argument("--pair-neighbors", type=int, default=6)
    parser.add_argument("--minimum-baseline-m", type=float, default=0.03)
    parser.add_argument("--maximum-baseline-m", type=float, default=5.0)
    parser.add_argument("--maximum-axis-angle-deg", type=float, default=75.0)
    parser.add_argument("--minimum-views", type=int, default=3)
    parser.add_argument("--minimum-view-bins", type=int, default=2)
    parser.add_argument("--maximum-observations", type=int, default=32)
    parser.add_argument("--huber-delta-px", type=float, default=2.0)
    parser.add_argument("--triangulation-iterations", type=int, default=3)
    parser.add_argument("--minimum-parallax-deg", type=float, default=1.0)
    parser.add_argument("--parallax-quantile", type=float, default=0.75)
    parser.add_argument("--maximum-reprojection-px", type=float, default=2.0)
    parser.add_argument("--maximum-condition-number", type=float, default=1e6)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--hard-alpha-minimum", type=float, default=0.20)
    parser.add_argument("--local-depth-radius", type=int, default=1)
    parser.add_argument("--depth-abs-tolerance-m", type=float, required=True)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--soft-cycle-px", type=float, default=4.0)
    parser.add_argument("--hard-cycle-px", type=float, default=8.0)
    parser.add_argument("--hard-depth-sigma", type=float, default=3.0)
    parser.add_argument("--uncertain-weight-floor", type=float, default=0.25)
    parser.add_argument("--maximum-children", type=int, default=3)
    parser.add_argument("--progress-interval", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.source_cache = args.source_cache.resolve()
    args.support_cache = args.support_cache.resolve()
    args.source_track_payload = args.source_track_payload.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA support repair without CUDA")
    print(json.dumps(materialize(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
