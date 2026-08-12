#!/usr/bin/env python3
"""Audit a frozen mapping Track pair graph without changing construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evidence.track_pair_audit import audit_track_pair_graph


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _arguments_from_manifest(path: Path | None) -> dict:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    return dict(payload.get("arguments", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproducibility-manifest", type=Path)
    parser.add_argument("--pair-neighbors", type=int)
    parser.add_argument("--minimum-baseline-m", type=float)
    parser.add_argument("--maximum-baseline-m", type=float)
    parser.add_argument("--maximum-axis-angle-deg", type=float)
    parser.add_argument("--minimum-effective-parallax-deg", type=float)
    parser.add_argument("--temporal-adjacency-gap", type=int, default=1)
    parser.add_argument("--maximum-visibility-points", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if int(args.cpu_threads) < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(int(args.cpu_threads))

    frozen = _arguments_from_manifest(args.reproducibility_manifest)

    def resolved(value, key: str, default):
        return frozen.get(key, default) if value is None else value

    result = audit_track_pair_graph(
        _load(args.track_payload),
        _load(args.query_cache),
        pair_neighbors=int(
            resolved(args.pair_neighbors, "geometry_teacher_track_pair_neighbors", 6)
        ),
        minimum_baseline_m=float(
            resolved(
                args.minimum_baseline_m,
                "geometry_teacher_track_min_baseline_m",
                0.03,
            )
        ),
        maximum_baseline_m=float(
            resolved(
                args.maximum_baseline_m,
                "geometry_teacher_track_max_baseline_m",
                5.0,
            )
        ),
        maximum_axis_angle_deg=float(
            resolved(
                args.maximum_axis_angle_deg,
                "geometry_teacher_track_max_axis_angle_deg",
                75.0,
            )
        ),
        minimum_effective_parallax_deg=float(
            resolved(
                args.minimum_effective_parallax_deg,
                "geometry_teacher_min_parallax_deg",
                1.0,
            )
        ),
        temporal_adjacency_gap=int(args.temporal_adjacency_gap),
        maximum_visibility_points=int(args.maximum_visibility_points),
    )
    result["inputs"] = {
        "track_payload": str(args.track_payload.resolve()),
        "query_cache": str(args.query_cache.resolve()),
        "reproducibility_manifest": (
            None
            if args.reproducibility_manifest is None
            else str(args.reproducibility_manifest.resolve())
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {
                "schema": result["schema"],
                "version": result["version"],
                "uses_test_queries": result["uses_test_queries"],
                "audit_only": result["audit_only"],
                "pair_selection_mutated": result["pair_selection_mutated"],
                "deployment_mutated": result["deployment_mutated"],
                "inputs": result["inputs"],
                "policy": result["policy"],
                "report": result["report"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(result["report"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
