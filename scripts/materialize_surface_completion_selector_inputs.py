#!/usr/bin/env python3
"""Compile a rendered surface-completion map into unified Selector evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.observation_provider import GaussianRenderObservationProvider
from topology.surface_completion import surface_completion_selector_inputs


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary Selector input did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface-map", type=Path, required=True)
    parser.add_argument("--expected-surface-map-sha256", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--expected-track-payload-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "surface_map": args.surface_map.resolve(),
        "query_cache": args.query_cache.resolve(),
        "track_payload": args.track_payload.resolve(),
    }
    expected = {
        "surface_map": args.expected_surface_map_sha256,
        "query_cache": args.expected_query_cache_sha256,
        "track_payload": args.expected_track_payload_sha256,
    }
    hashes = {
        name: _require_sha(path, expected[name], name) for name, path in paths.items()
    }
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    surface_map = torch.load(
        paths["surface_map"], map_location="cpu", weights_only=False
    )
    cache = torch.load(paths["query_cache"], map_location="cpu", weights_only=False)
    track = torch.load(paths["track_payload"], map_location="cpu", weights_only=False)
    provider = GaussianRenderObservationProvider(
        cache,
        query_names=list(track["query_names"]),
        query_bins=track["query_bins"],
    )
    teacher, graph = surface_completion_selector_inputs(surface_map, provider)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    teacher_path = args.output_dir / "surface_completion_teacher.pt"
    graph_path = args.output_dir / "surface_completion_function_graph.pt"
    _atomic_save(teacher, teacher_path)
    _atomic_save(graph, graph_path)
    for name, path in paths.items():
        _require_sha(path, hashes[name], name)
    report = {
        "schema": "lafgs_surface_completion_selector_inputs",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "inputs": {name: str(path) for name, path in paths.items()},
        "input_sha256": hashes,
        "outputs": {
            "teacher": str(teacher_path),
            "function_graph": str(graph_path),
        },
        "output_sha256": {
            "teacher": sha256_file(teacher_path),
            "function_graph": sha256_file(graph_path),
        },
        "candidate_count": int(surface_map["anchor_ids"].numel()),
        "identity_positive_count": int(
            sum(record["positive_indices"].numel() for record in teacher["records"])
        ),
        "authorizes_default_map_change": False,
    }
    report_path = args.output_dir / "surface_completion_selector_inputs.json"
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
