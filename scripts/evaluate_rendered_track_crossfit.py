#!/usr/bin/env python3
"""Leave-one-mapping-sequence-out replay for rendered-RGB Track maps."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evidence.tracks import robust_fuse_track_descriptors
from evidence.triangulation import (
    build_cycle_consistent_tracks,
    robust_triangulate_associations,
)
from map_learning.metric import SharedLowRankMetric
from topology.deployment_revision import collect_deployment_statistics, subset_teacher
from topology.track_core import _eligible_tracks
from topology.track_core import _track_quality
from scripts.materialize_rendered_track_support_repair import (
    _limit_children_after_triangulation,
)


COUNTER_NAMES = (
    "winner_count",
    "correct_winner_count",
    "false_attractor_count",
    "ambiguous_winner_count",
    "clean_inlier_count",
    "harmful_inlier_count",
    "counterfactual_clean_gain",
    "information_deletion_loss",
)

_PRODUCER_SOURCE_PATHS = (
    "scripts/evaluate_rendered_track_crossfit.py",
    "common/hashing.py",
    "evidence/tracks.py",
    "evidence/triangulation.py",
    "map_learning/metric.py",
    "topology/deployment_revision.py",
    "topology/track_core.py",
    "scripts/materialize_rendered_track_support_repair.py",
)


def _producer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain=v1"):
        raise RuntimeError("rendered Track crossfit producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "torch_version": torch.__version__,
        "source_sha256": {
            relative: sha256_file(repository / relative)
            for relative in _PRODUCER_SOURCE_PATHS
        },
    }


def _combined_summary(query_rows: list[dict], counters: dict) -> dict:
    import numpy as np

    te = np.asarray([row["te_cm"] for row in query_rows], dtype=np.float64)
    ae = np.asarray([row["ae_deg"] for row in query_rows], dtype=np.float64)
    tail = max(int(np.ceil(0.05 * te.size)), 1)
    raw = int(counters["winner_count"].sum())
    correct = int(counters["correct_winner_count"].sum())
    clean = int(counters["clean_inlier_count"].sum())
    harmful = int(counters["harmful_inlier_count"].sum())
    return {
        "query_count": int(te.size),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "cvar95_te_cm": float(np.sort(te)[-tail:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "p90_ae_deg": float(np.percentile(ae, 90)),
        "recall_5cm_5deg_percent": float(100.0 * np.mean((te < 5.0) & (ae < 5.0))),
        "catastrophic_100cm_count": int(np.count_nonzero(te >= 100.0)),
        "raw_gt_precision_percent": 100.0 * correct / max(raw, 1),
        "inlier_gt_precision_percent": 100.0 * clean / max(clean + harmful, 1),
        "retained_matches_mean": float(
            np.mean([row["correspondences"] for row in query_rows])
        ),
        "mean_hypotheses": float(np.mean([row["hypotheses"] for row in query_rows])),
    }


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sequence_name(image_name: str) -> str:
    return str(image_name).split("/", maxsplit=1)[0]


def _crossfit_groups(
    names: list[str], blocked_folds: int
) -> tuple[list[str], list[str]]:
    sequences = [_sequence_name(name) for name in names]
    unique = sorted(set(sequences))
    if len(unique) >= 2:
        return sequences, unique
    fold_count = int(blocked_folds)
    if fold_count < 2:
        raise ValueError(
            "a single mapping trajectory requires at least two blocked folds"
        )
    # Camera names are already in the frozen dataset/cache order.  Contiguous
    # blocks keep neighboring Cambridge frames together and make every held
    # block disjoint from the descriptors used to build its Track bank.
    groups = [
        f"blocked_{min(index * fold_count // len(names), fold_count - 1):02d}"
        for index in range(len(names))
    ]
    return groups, sorted(set(groups))


def _fold_bank(
    *,
    state: dict,
    payload: dict,
    query_cache: dict,
    held_sequence: str,
    crossfit_groups: list[str],
    trim_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    names = list(payload["query_names"])
    cache = query_cache.get("queries", query_cache)
    tracks = payload["tracks"]
    query_bins = torch.as_tensor(
        payload.get("pose_view_bins", payload["query_bins"])
    ).long()
    selected_tracks = torch.as_tensor(state["track_cluster_ids"]).long()
    selected_lookup = {
        int(track): row for row, track in enumerate(selected_tracks.tolist())
    }
    track_index = torch.as_tensor(tracks["track_index"]).long()
    track_query = torch.as_tensor(tracks["query_index"]).long()
    track_keypoint = torch.as_tensor(tracks["keypoint_index"]).long()
    track_confidence = torch.as_tensor(tracks["confidence"]).float()
    observations: dict[int, list[int]] = defaultdict(list)
    for observation, (track, query) in enumerate(
        zip(track_index.tolist(), track_query.tolist())
    ):
        row = selected_lookup.get(int(track))
        if row is not None and crossfit_groups[int(query)] != held_sequence:
            observations[row].append(observation)

    landmark_rows = []
    observation_queries = []
    observation_uv = []
    observation_confidence = []
    for anchor, selected_observations in observations.items():
        for observation in selected_observations:
            query = int(track_query[observation])
            keypoint = int(track_keypoint[observation])
            cached = cache[names[query]]
            landmark_rows.append(int(anchor))
            observation_queries.append(query)
            observation_uv.append(
                torch.as_tensor(cached["native_keypoints"])[keypoint].float()
                + float(cached.get("pixel_center_offset", 0.5))
            )
            observation_confidence.append(float(track_confidence[observation]))
    if not landmark_rows:
        raise RuntimeError(f"held fold {held_sequence} has no support observations")
    camera_K = torch.stack(
        [torch.as_tensor(cache[name]["native_K"]).float() for name in names]
    )
    pose_w2c = torch.stack(
        [torch.as_tensor(cache[name]["pose_w2c"]).float() for name in names]
    )
    query_bin = torch.as_tensor(
        payload.get("pose_view_bins", payload["query_bins"])
    ).long()
    geometry = robust_triangulate_associations(
        landmark_count=int(selected_tracks.numel()),
        landmark_index=torch.as_tensor(landmark_rows).long(),
        query_index=torch.as_tensor(observation_queries).long(),
        uv=torch.stack(observation_uv),
        confidence=torch.as_tensor(observation_confidence).float(),
        camera_K=camera_K,
        pose_w2c=pose_w2c,
        query_bin=query_bin,
        rendered_depth=None,
        maximum_observations_per_landmark=32,
        minimum_views=3,
        minimum_view_bins=2,
        huber_delta_px=2.0,
        iterations=3,
        minimum_parallax_deg=1.0,
        parallax_quantile=0.75,
        maximum_reprojection_px=2.0,
        maximum_condition_number=1e6,
        maximum_covariance_trace_m2=float("inf"),
        maximum_rendered_depth_residual_m=float("inf"),
        minimum_rendered_depth_observations=0,
        surface_support_enabled=False,
    )
    eligible = _eligible_tracks(geometry, "broad")
    features = []
    for anchor in range(selected_tracks.numel()):
        if not bool(eligible[anchor]):
            continue
        selected_observations = observations.get(anchor, ())
        observation_rows = torch.as_tensor(selected_observations, dtype=torch.long)
        queries = track_query[observation_rows]
        keypoints = track_keypoint[observation_rows]
        descriptor = torch.stack(
            [
                torch.as_tensor(cache[names[int(query)]]["native_descriptors"])[
                    int(keypoint)
                ]
                for query, keypoint in zip(queries.tolist(), keypoints.tolist())
            ]
        )
        features.append(
            robust_fuse_track_descriptors(
                descriptor,
                query_bins[queries],
                track_confidence[observation_rows],
                trim_fraction=float(trim_fraction),
            )
        )
    if not features:
        raise RuntimeError(
            f"held fold {held_sequence} has no support-only broad anchors"
        )
    return eligible, torch.stack(features), geometry


def _fold_component_bank(
    *,
    state: dict,
    payload: dict,
    query_cache: dict,
    held_sequence: str,
    crossfit_groups: list[str],
    trim_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, dict, dict]:
    """Rebuild support components from frozen pair rows without held cameras."""
    names = list(payload["query_names"])
    cache = query_cache.get("queries", query_cache)
    repair = payload.get("support_repair", {})
    frozen = repair.get("frozen_support_pair_matches", {})
    contract = repair.get("track_build_contract", {})
    source_lookup = repair.get("source_track_index_at_keypoint")
    if (
        frozen.get("schema") != "lafgs_rendered_track_frozen_support_pair_matches"
        or source_lookup is None
        or not contract
    ):
        raise ValueError("support-repaired crossfit lacks frozen component inputs")
    pair_table = payload["pair_sidecar"]["pair"]
    all_pairs = list(
        zip(
            torch.as_tensor(pair_table["left_query_index"]).long().tolist(),
            torch.as_tensor(pair_table["right_query_index"]).long().tolist(),
        )
    )
    offsets = torch.as_tensor(frozen["offsets"]).long()
    source_rows = torch.as_tensor(frozen["source_keypoint_indices"]).long()
    target_rows = torch.as_tensor(frozen["target_keypoint_indices"]).long()
    confidence_rows = torch.as_tensor(frozen["confidence"]).float()
    if (
        offsets.shape != (len(all_pairs) + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != int(source_rows.numel())
        or source_rows.shape != target_rows.shape
        or source_rows.shape != confidence_rows.shape
    ):
        raise ValueError("frozen support pair rows are malformed")

    pairs = []
    matches = {}
    for pair_index, pair in enumerate(all_pairs):
        left, right = pair
        if (
            crossfit_groups[int(left)] == held_sequence
            or crossfit_groups[int(right)] == held_sequence
        ):
            continue
        begin, end = int(offsets[pair_index]), int(offsets[pair_index + 1])
        pairs.append(pair)
        matches[pair] = (
            source_rows[begin:end].clone(),
            target_rows[begin:end].clone(),
            confidence_rows[begin:end].clone(),
        )
    if not pairs:
        raise RuntimeError(f"held fold {held_sequence} leaves no camera pairs")

    descriptors = [
        F.normalize(torch.as_tensor(cache[name]["native_descriptors"]).float(), dim=1)
        for name in names
    ]
    keypoints = [
        torch.as_tensor(cache[name]["native_keypoints"]).float()
        + float(cache[name].get("pixel_center_offset", 0.5))
        for name in names
    ]
    scores = [torch.as_tensor(cache[name]["native_scores"]).float() for name in names]
    camera_K = torch.stack(
        [torch.as_tensor(cache[name]["native_K"]).float() for name in names]
    )
    pose_w2c = torch.stack(
        [torch.as_tensor(cache[name]["pose_w2c"]).float() for name in names]
    )
    rebuilt, diagnostics = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        detector_scores=scores,
        camera_K=camera_K,
        pose_w2c=pose_w2c,
        pair_neighbors=int(contract["pair_neighbors"]),
        pair_policy=str(contract["pair_policy"]),
        pair_budget=len(pairs),
        minimum_baseline_m=float(contract["minimum_baseline_m"]),
        maximum_baseline_m=float(contract["maximum_baseline_m"]),
        maximum_axis_angle_deg=float(contract["maximum_axis_angle_deg"]),
        minimum_similarity=float(contract["minimum_similarity"]),
        minimum_margin=float(contract["minimum_margin"]),
        maximum_epipolar_error_px=float(contract["maximum_epipolar_error_px"]),
        epipolar_candidate_topk=int(contract["epipolar_candidate_topk"]),
        minimum_track_views=int(contract["minimum_track_views"]),
        require_cycle=True,
        allow_chain_tracks=True,
        precomputed_pairs=pairs,
        precomputed_pair_matches=matches,
        precomputed_confidence_includes_detector_scores=True,
        device="cpu",
    )
    observation_query = torch.as_tensor(rebuilt["query_index"]).long()
    observation_keypoint = torch.as_tensor(rebuilt["keypoint_index"]).long()
    observation_uv = torch.stack(
        [
            keypoints[int(query)][int(keypoint)]
            for query, keypoint in zip(
                observation_query.tolist(), observation_keypoint.tolist()
            )
        ]
    )
    query_bins = torch.as_tensor(
        payload.get("pose_view_bins", payload["query_bins"])
    ).long()
    geometry = robust_triangulate_associations(
        landmark_count=int(torch.as_tensor(rebuilt["track_level"]).numel()),
        landmark_index=torch.as_tensor(rebuilt["track_index"]).long(),
        query_index=observation_query,
        uv=observation_uv,
        confidence=torch.as_tensor(rebuilt["confidence"]).float(),
        camera_K=camera_K,
        pose_w2c=pose_w2c,
        query_bin=query_bins,
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
    )
    geometry["track_confidence_level"] = rebuilt["track_level"].clone()
    rebuilt, geometry, _, cap = _limit_children_after_triangulation(
        rebuilt,
        geometry,
        [torch.as_tensor(rows).long() for rows in source_lookup],
        maximum_children=int(repair["maximum_children_per_source_track"]),
    )

    parent_by_child = torch.as_tensor(rebuilt["parent_source_track_ids"]).long()
    quality = _track_quality(geometry)
    children_by_parent: dict[int, list[int]] = defaultdict(list)
    for child, parent in enumerate(parent_by_child.tolist()):
        children_by_parent[int(parent)].append(child)
    for parent in children_by_parent:
        children_by_parent[parent].sort(
            key=lambda child: (-float(quality[child]), child)
        )
    state_parents = torch.as_tensor(state.get("parent_source_track_ids"))
    if state_parents.dtype != torch.long or state_parents.shape != (
        int(torch.as_tensor(state["anchor_ids"]).numel()),
    ):
        raise ValueError("fold component rebuild requires exact map parent lineage")
    chosen = []
    keep_rows = []
    for row, parent in enumerate(state_parents.tolist()):
        candidates = children_by_parent.get(int(parent), ())
        if candidates:
            keep_rows.append(row)
            chosen.append(int(candidates[0]))
    keep = torch.zeros(state_parents.numel(), dtype=torch.bool)
    keep[torch.as_tensor(keep_rows).long()] = True

    child_track = torch.as_tensor(rebuilt["track_index"]).long()
    child_query = torch.as_tensor(rebuilt["query_index"]).long()
    child_keypoint = torch.as_tensor(rebuilt["keypoint_index"]).long()
    child_confidence = torch.as_tensor(rebuilt["confidence"]).float()
    features = []
    for child in chosen:
        observations = torch.nonzero(child_track == child, as_tuple=False).reshape(-1)
        queries = child_query[observations]
        descriptor = torch.stack(
            [
                torch.as_tensor(cache[names[int(query)]]["native_descriptors"])[
                    int(keypoint)
                ]
                for query, keypoint in zip(
                    queries.tolist(), child_keypoint[observations].tolist()
                )
            ]
        )
        features.append(
            robust_fuse_track_descriptors(
                descriptor,
                query_bins[queries],
                child_confidence[observations],
                trim_fraction=float(trim_fraction),
            )
        )
    if not features:
        raise RuntimeError(f"held fold {held_sequence} rebuilds no selected parent")

    aligned_geometry = {}
    chosen_rows = torch.as_tensor(chosen).long()
    keep_rows_tensor = torch.as_tensor(keep_rows).long()
    for key, value in geometry.items():
        if (
            not torch.is_tensor(value)
            or not value.ndim
            or value.shape[0] != len(parent_by_child)
        ):
            continue
        shape = (int(state_parents.numel()), *value.shape[1:])
        if value.dtype == torch.bool:
            aligned = torch.zeros(shape, dtype=torch.bool)
        elif value.dtype.is_floating_point:
            aligned = torch.full(shape, float("nan"), dtype=value.dtype)
        else:
            aligned = torch.zeros(shape, dtype=value.dtype)
        aligned[keep_rows_tensor] = value[chosen_rows]
        aligned_geometry[key] = aligned
    diagnostics = {
        **diagnostics,
        **cap,
        "fold_specific_component_rebuild": True,
        "frozen_support_pair_count": len(pairs),
        "selected_parent_count": int(keep.sum()),
    }
    return keep, torch.stack(features), aligned_geometry, diagnostics


def _subset_state(
    state: dict, keep: torch.Tensor, features: torch.Tensor, geometry: dict
) -> dict:
    keep = torch.as_tensor(keep).bool()
    count = int(keep.numel())
    output = dict(state)
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
            output[key] = value[keep]
    output["anchor_ids"] = torch.arange(int(keep.sum()), dtype=torch.long)
    output["anchor_features"] = features.float()
    output["v7_metric_raw_features"] = features.float()
    output["anchor_xyz"] = torch.as_tensor(geometry["triangulated_xyz"])[keep].float()
    output["anchor_position_covariance"] = torch.as_tensor(
        geometry["triangulation_covariance_matrix"]
    )[keep].float()
    selected_tracks = torch.as_tensor(state["track_cluster_ids"]).long()[keep]
    output["track_cluster_ids"] = selected_tracks
    output["track_centric_reconstruction"] = {
        **state.get("track_centric_reconstruction", {}),
        "track_indices": selected_tracks.clone(),
        "base_canonical_rows": torch.empty(0, dtype=torch.long),
        "track_anchor_count": int(keep.sum()),
        "base_reserve_count": 0,
    }
    output["base_anchor_count"] = 0
    output["micro_anchor_count"] = int(keep.sum())
    output["canonical_anchor_count"] = int(keep.sum())
    output["rendered_track_crossfit_geometry"] = {
        "schema": "lafgs_rendered_track_support_only_retriangulation",
        "version": 1,
        "support_only": True,
        "minimum_views": 3,
        "minimum_view_bins": 2,
        "minimum_parallax_deg": 1.0,
        "maximum_reprojection_px": 2.0,
        "maximum_condition_number": 1e6,
        "uses_gaussian_geometry": False,
    }
    return output


def _identity_metric(anchor_count: int, descriptor_dim: int, map_path: Path) -> dict:
    metric = SharedLowRankMetric(
        descriptor_dim=descriptor_dim, rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    return {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(anchor_count, dtype=torch.long),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            name: value.detach().cpu() for name, value in metric.state_dict().items()
        },
        "map_path": str(map_path.resolve()),
        "step": 0,
        "protocol": "rendered_track_leave_one_mapping_sequence_out_identity",
    }


def run(args) -> dict:
    producer_identity = _producer_identity()
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
    cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    if cache.get("uses_source_mapping_rgb") is not False:
        raise ValueError("crossfit cache is not rendered-RGB-only")
    if cache.get("uses_test_queries") is not False:
        raise ValueError("crossfit cache contains test queries")
    names = list(payload["query_names"])
    if names != list(teacher["query_names"]):
        raise ValueError("teacher and Track query order differs")
    crossfit_groups, sequences = _crossfit_groups(names, args.blocked_folds)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    full_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    aggregate = {
        name: torch.zeros(full_count, dtype=torch.float64) for name in COUNTER_NAMES
    }
    all_rows = []
    folds = []
    component_rebuild = (
        payload.get("support_repair", {})
        .get("frozen_support_pair_matches", {})
        .get("schema")
        == "lafgs_rendered_track_frozen_support_pair_matches"
    )
    for fold_index, held_sequence in enumerate(sequences):
        fold_dir = args.output_dir / held_sequence
        fold_dir.mkdir()
        if component_rebuild:
            keep, features, geometry, rebuild_diagnostics = _fold_component_bank(
                state=state,
                payload=payload,
                query_cache=cache,
                held_sequence=held_sequence,
                crossfit_groups=crossfit_groups,
                trim_fraction=args.descriptor_trim_fraction,
            )
        else:
            keep, features, geometry = _fold_bank(
                state=state,
                payload=payload,
                query_cache=cache,
                held_sequence=held_sequence,
                crossfit_groups=crossfit_groups,
                trim_fraction=args.descriptor_trim_fraction,
            )
            rebuild_diagnostics = {"fold_specific_component_rebuild": False}
        fold_map = _subset_state(state, keep, features, geometry)
        map_path = fold_dir / "anchor_map.pt"
        metric_path = fold_dir / "metric_state.pt"
        teacher_path = fold_dir / "positive_teacher.pt"
        fold_teacher = subset_teacher(teacher, keep, map_path)
        metric = _identity_metric(int(keep.sum()), int(features.shape[1]), map_path)
        _atomic_save(fold_map, map_path)
        _atomic_save(metric, metric_path)
        _atomic_save(fold_teacher, teacher_path)
        query_indices = [
            index
            for index, name in enumerate(names)
            if crossfit_groups[index] == held_sequence
        ]
        statistics = collect_deployment_statistics(
            state=fold_map,
            metric_state_path=metric_path,
            teacher=fold_teacher,
            query_cache=cache,
            device=torch.device(args.device),
            ransac_reprojection_px=args.ransac_reprojection_px,
            clean_reprojection_px=args.clean_reprojection_px,
            task_translation_m=args.task_translation_m,
            task_rotation_deg=args.task_rotation_deg,
            seed=args.seed,
            query_indices=query_indices,
            progress_label=f"held_{held_sequence}",
        )
        _atomic_save(statistics, fold_dir / "statistics.pt")
        for name in COUNTER_NAMES:
            aggregate[name][keep] += torch.as_tensor(
                statistics["counters"][name]
            ).double()
        all_rows.extend(statistics["queries"])
        folds.append(
            {
                "held_sequence": held_sequence,
                "query_count": len(query_indices),
                "train_supported_anchor_count": int(keep.sum()),
                "support_only_triangulated_anchor_count": int(
                    torch.as_tensor(geometry["triangulated"]).sum()
                ),
                "summary": statistics["summary"],
                "statistics": str((fold_dir / "statistics.pt").resolve()),
                "component_rebuild": rebuild_diagnostics,
            }
        )
        print(json.dumps(folds[-1], sort_keys=True), flush=True)

    combined = {
        "schema": "lafgs_rendered_track_mapping_sequence_crossfit_statistics",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "support_only_retriangulation": True,
        "fold_specific_component_rebuild": component_rebuild,
        "frozen_support_pair_matches_reused": component_rebuild,
        "producer_identity": producer_identity,
        "query_rows": all_rows,
        "counters": aggregate,
        "summary": _combined_summary(all_rows, aggregate),
        "folds": folds,
    }
    statistics_path = args.output_dir / "crossfit_statistics.pt"
    _atomic_save(combined, statistics_path)
    report = {
        "schema": "lafgs_rendered_track_mapping_sequence_crossfit_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "support_only_retriangulation": True,
        "fold_specific_component_rebuild": component_rebuild,
        "frozen_support_pair_matches_reused": component_rebuild,
        "producer_identity": producer_identity,
        "sequences": sequences,
        "grouping": (
            "mapping_trajectory"
            if len(set(_sequence_name(name) for name in names)) >= 2
            else "contiguous_mapping_blocks"
        ),
        "folds": folds,
        "combined_summary": combined["summary"],
        "inputs": {
            "anchor_map": str(args.anchor_map.resolve()),
            "track_payload": str(args.track_payload.resolve()),
            "teacher": str(args.teacher.resolve()),
            "query_cache": str(args.query_cache.resolve()),
        },
        "input_sha256": {
            "anchor_map": sha256_file(args.anchor_map),
            "track_payload": sha256_file(args.track_payload),
            "teacher": sha256_file(args.teacher),
            "query_cache": sha256_file(args.query_cache),
        },
        "statistics": str(statistics_path.resolve()),
        "statistics_sha256": sha256_file(statistics_path),
    }
    if _producer_identity() != producer_identity:
        raise RuntimeError("rendered Track crossfit producer identity changed")
    (args.output_dir / "crossfit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocked-folds", type=int, default=3)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--ransac-reprojection-px", type=float, default=12.0)
    parser.add_argument("--clean-reprojection-px", type=float, default=4.0)
    parser.add_argument("--task-translation-m", type=float, default=0.05)
    parser.add_argument("--task-rotation-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for field in ("anchor_map", "track_payload", "teacher", "query_cache"):
        setattr(args, field, getattr(args, field).resolve())
    args.output_dir = args.output_dir.resolve()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
