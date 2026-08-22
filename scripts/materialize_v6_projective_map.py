#!/usr/bin/env python3
"""Materialize the V6 render-valid, pure-ray Projective Anchor map."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from common.hashing import sha256_file
from common.v6_contracts import RENDER_OBSERVATION_SCHEMA, require_schema
from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.projective_association import build_projective_association_graph
from evidence.projective_completion import build_projective_completion
from evidence.projective_reconstruction import reconstruct_projective_anchors
from topology.v6_anchor_map import (
    identity_metric_state,
    materialize_projective_anchor_map,
    merge_projective_candidates,
)


_SOURCE_PATHS = (
    "scripts/materialize_v6_projective_map.py",
    "common/v6_contracts.py",
    "evidence/observation_provider.py",
    "evidence/projective_association.py",
    "evidence/projective_completion.py",
    "evidence/projective_reconstruction.py",
    "evidence/parallel_triangulation.py",
    "evidence/triangulation.py",
    "evidence/tracks.py",
    "topology/v6_anchor_map.py",
)


def _identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=root, check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("V6 materializer requires a clean producer worktree")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {name: sha256_file(root / name) for name in _SOURCE_PATHS},
        "torch_version": torch.__version__,
    }


def _save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    if int(args.cpu_threads) < 1:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(int(args.cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(int(args.cpu_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(args.cpu_threads))
    identity = _identity()
    cache_path = args.observation_cache.resolve()
    actual_cache_sha = sha256_file(cache_path)
    if actual_cache_sha != args.expected_observation_cache_sha256:
        raise ValueError("observation cache SHA differs")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    require_schema(cache, RENDER_OBSERVATION_SCHEMA, label="V6 observations")
    if cache.get("uses_rendered_depth") is not True:
        raise ValueError("V6 observations must carry proposal-only rendered depth")
    provider = GaussianRenderObservationProvider(cache)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    association_started = time.perf_counter()
    association = build_projective_association_graph(
        provider,
        pair_neighbors=args.pair_neighbors,
        minimum_similarity=args.minimum_similarity,
        minimum_margin=args.minimum_margin,
        maximum_epipolar_error_px=args.maximum_epipolar_error_px,
        minimum_track_views=args.minimum_views,
        device=args.device,
    )
    association_seconds = time.perf_counter() - association_started
    reconstruction_started = time.perf_counter()
    base = reconstruct_projective_anchors(
        provider,
        association,
        minimum_views=args.minimum_views,
        minimum_view_bins=args.minimum_camera_families,
        minimum_parallax_deg=args.minimum_parallax_deg,
        maximum_reprojection_px=args.maximum_reprojection_px,
        parallel_workers=args.triangulation_workers,
        parallel_minimum_tracks=args.parallel_triangulation_minimum_tracks,
    )
    base["candidate_kind"] = "projective_track"
    parts = [base]
    completion = None
    if args.enable_projective_completion:
        completion = build_projective_completion(
            provider,
            association,
            voxel_size_m=args.completion_voxel_size_m,
            alpha_minimum=args.alpha_minimum,
            minimum_similarity=args.completion_minimum_similarity,
            minimum_margin=args.minimum_margin,
            maximum_epipolar_error_px=args.maximum_epipolar_error_px,
            minimum_observations=args.minimum_views,
            minimum_camera_families=args.minimum_camera_families,
            maximum_rows_per_view=args.completion_maximum_rows_per_view,
            safety_maximum_components=args.completion_safety_maximum_components,
            device=args.device,
        )
        parts.append(completion)
    candidates = merge_projective_candidates(parts)
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    lineage = {
        "v6_observation_cache": str(cache_path),
        "v6_observation_cache_sha256": actual_cache_sha,
        "v6_producer": identity,
        "v6_round": 0,
        "projective_completion_enabled": bool(args.enable_projective_completion),
    }
    state = materialize_projective_anchor_map(candidates, lineage=lineage)
    association_path = args.output_dir / "association_graph.pt"
    candidates_path = args.output_dir / "projective_anchor_candidates.pt"
    map_path = args.output_dir / "projective_anchor_map.pt"
    metric_path = args.output_dir / "identity_metric.pt"
    _save(association, association_path)
    _save(candidates, candidates_path)
    _save(state, map_path)
    map_sha = sha256_file(map_path)
    metric = identity_metric_state(
        state, map_path=str(map_path.resolve()), map_sha256=map_sha
    )
    _save(metric, metric_path)
    report = {
        "schema": "lafgs_v6_projective_map_materialization_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "producer": identity,
        "input": {"observation_cache": str(cache_path), "sha256": actual_cache_sha},
        "output": {
            "association_graph": str(association_path.resolve()),
            "association_graph_sha256": sha256_file(association_path),
            "candidates": str(candidates_path.resolve()),
            "candidates_sha256": sha256_file(candidates_path),
            "map": str(map_path.resolve()),
            "map_sha256": map_sha,
            "metric": str(metric_path.resolve()),
            "metric_sha256": sha256_file(metric_path),
        },
        "counts": {
            "mapping_views": len(provider),
            "association_components": int(association["diagnostics"]["track_count"]),
            "base_projective_anchors": int(base["anchor_xyz"].shape[0]),
            "completion_anchors": 0 if completion is None else int(completion["anchor_xyz"].shape[0]),
            "total_anchors": int(state["anchor_xyz"].shape[0]),
        },
        "association_diagnostics": dict(association["diagnostics"]),
        "contracts": {
            "render_valid_before_nms": True,
            "unified_association_once": True,
            "posthoc_support_repair": False,
            "parent_child_semantics": False,
            "direct_gaussian_surface_anchor": False,
            "final_xyz_pure_ray": True,
            "online_protocol_unchanged": True,
        },
        "timing_seconds": {
            "association": association_seconds,
            "reconstruction_and_completion": reconstruction_seconds,
            "total": time.perf_counter() - started,
        },
        "cpu_threads": int(args.cpu_threads),
    }
    _json(report, args.output_dir / "report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--expected-observation-cache-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--pair-neighbors", type=int, default=6)
    parser.add_argument("--minimum-similarity", type=float, default=0.65)
    parser.add_argument("--minimum-margin", type=float, default=0.01)
    parser.add_argument("--maximum-epipolar-error-px", type=float, default=2.0)
    parser.add_argument("--minimum-views", type=int, default=3)
    parser.add_argument("--minimum-camera-families", type=int, default=2)
    parser.add_argument("--minimum-parallax-deg", type=float, default=1.0)
    parser.add_argument("--maximum-reprojection-px", type=float, default=2.0)
    parser.add_argument("--triangulation-workers", type=int, default=2)
    parser.add_argument("--parallel-triangulation-minimum-tracks", type=int, default=5000)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--enable-projective-completion", action="store_true")
    parser.add_argument("--completion-voxel-size-m", type=float, default=0.05)
    parser.add_argument("--completion-minimum-similarity", type=float, default=0.7)
    parser.add_argument("--completion-maximum-rows-per-view", type=int, default=256)
    parser.add_argument("--completion-safety-maximum-components", type=int, default=100000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
