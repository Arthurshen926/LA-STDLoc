#!/usr/bin/env python3
"""Materialize bounded non-Track Anchors from Gaussian rendered surface evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.observation_provider import GaussianRenderObservationProvider
from topology.anchor_construction import SurfaceCompletionProvider
from topology.surface_completion import materialize_gaussian_surface_completion


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _atomic_save(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        rows = torch.arange(int(reloaded["anchor_ids"].numel()), dtype=torch.long)
        SurfaceCompletionProvider(
            reloaded, rows, maximum_candidates=int(rows.numel())
        ).materialize()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        json.loads(temporary.read_text())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict:
    cache_path = args.query_cache.resolve()
    track_path = args.track_payload.resolve()
    output = args.output.resolve()
    cache_sha = _require_sha(
        cache_path, args.expected_query_cache_sha256, "query cache"
    )
    track_sha = _require_sha(
        track_path, args.expected_track_payload_sha256, "Track payload"
    )
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    track = torch.load(track_path, map_location="cpu", weights_only=False)
    if (
        cache.get("uses_source_mapping_rgb") is not False
        or cache.get("uses_test_queries") is not False
        or track.get("rendered_rgb_only") is not True
    ):
        raise ValueError(
            "surface completion inputs must be mapping-only rendered evidence"
        )
    provider = GaussianRenderObservationProvider(
        cache,
        query_names=list(track["query_names"]),
        query_bins=track["query_bins"],
    )
    result = materialize_gaussian_surface_completion(
        provider,
        voxel_size_m=float(args.voxel_size_m),
        maximum_candidates=int(args.maximum_candidates),
        maximum_rows_per_view=int(args.maximum_rows_per_view),
        alpha_minimum=float(args.alpha_minimum),
        minimum_observations=int(args.minimum_observations),
        minimum_views=int(args.minimum_views),
        minimum_pose_bins=int(args.minimum_pose_bins),
        descriptor_trim_fraction=float(args.descriptor_trim_fraction),
    )
    result["provenance"] = {
        "query_cache": str(cache_path),
        "query_cache_sha256": cache_sha,
        "track_payload": str(track_path),
        "track_payload_sha256": track_sha,
        "mapping_source": "gaussian_render",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
    }
    _require_sha(cache_path, cache_sha, "query cache")
    _require_sha(track_path, track_sha, "Track payload")
    _atomic_save(result, output)
    report = {
        "schema": "lafgs_gaussian_render_surface_completion_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "map": str(output),
        "map_sha256": sha256_file(output),
        "inputs": result["provenance"],
        "surface_completion": result["surface_completion"],
        "completion_candidate_provider": "always_enabled",
        "authorizes_default_map_change": False,
        "decision": "CANDIDATE_PROVIDER_READY_REQUIRES_UNIFIED_SELECTOR_EVALUATION",
    }
    _atomic_json(report, output.with_suffix(".json"))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--expected-track-payload-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, required=True)
    parser.add_argument("--maximum-candidates", type=int, default=1024)
    parser.add_argument("--maximum-rows-per-view", type=int, default=256)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--minimum-observations", type=int, default=3)
    parser.add_argument("--minimum-views", type=int, default=2)
    parser.add_argument("--minimum-pose-bins", type=int, default=2)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
