#!/usr/bin/env python3
"""Replay one map on mapping descriptors and report translation/rotation pose error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from topology.deployment_revision import collect_deployment_statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument(
        "--query-count",
        type=int,
        default=0,
        help="Deterministic uniformly spaced mapping gate; zero evaluates all queries.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    calibration = json.loads(args.scene_calibration.read_text())
    parameters = calibration["parameters"]
    total_queries = len(teacher["records"])
    if int(args.query_count) < 0:
        raise ValueError("query count must be non-negative")
    query_indices = None
    if 0 < int(args.query_count) < total_queries:
        query_indices = torch.linspace(
            0, total_queries - 1, steps=int(args.query_count)
        ).round().long().unique(sorted=True)
    statistics = collect_deployment_statistics(
        state=state,
        metric_state_path=args.metric_state,
        teacher=teacher,
        query_cache=cache,
        device=torch.device(args.device),
        ransac_reprojection_px=float(parameters["ransac_reprojection_px"]),
        clean_reprojection_px=float(parameters["clean_radius_px"]),
        task_translation_m=float(parameters["task_translation_m"]),
        task_rotation_deg=float(parameters["task_rotation_deg"]),
        seed=args.seed,
        query_indices=query_indices,
        deployment_row_limit=args.deployment_row_limit,
        collect_anchor_statistics=False,
        progress_label="mapping_cache_evaluation",
    )
    report = {
        "schema": "lafgs_mapping_cache_evaluation",
        "version": 1,
        "uses_test_queries": False,
        "map": str(args.map.resolve()),
        "metric_state": str(args.metric_state.resolve()),
        "deployment_row_limit": int(args.deployment_row_limit),
        "pose_error_units": {"translation": "cm", "rotation": "deg"},
        "query_count": int(
            total_queries if query_indices is None else query_indices.numel()
        ),
        "query_selection": "all" if query_indices is None else "uniform_mapping_gate",
        "summary": statistics["summary"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "mapping_cache_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
