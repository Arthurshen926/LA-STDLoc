#!/usr/bin/env python3
"""Cross-fit observation descriptors across disjoint mapping trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization.localizer import load_shared_metric
from topology.observation_descriptor_crossfit import (
    audit_crossfit_observation_descriptors,
)


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-chunk", type=int, default=256)
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-support-queries", type=int, default=2)
    parser.add_argument("--minimum-support-strata", type=int, default=2)
    parser.add_argument("--minimum-direction-cosine", type=float, default=0.65)
    parser.add_argument(
        "--fold-a-trajectories",
        default="",
        help="Optional comma-separated explicit trajectory labels for fold A.",
    )
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(int(args.cpu_threads))
    registry = _load(args.registry)
    anchor_ids = torch.as_tensor(registry["anchor_ids"]).long()
    metric = load_shared_metric(
        args.metric_state, anchor_ids=anchor_ids, device=torch.device(args.device)
    )
    result = audit_crossfit_observation_descriptors(
        registry,
        _load(args.query_cache),
        metric,
        trim_fraction=float(args.trim_fraction),
        minimum_support_queries=int(args.minimum_support_queries),
        minimum_support_strata=int(args.minimum_support_strata),
        minimum_direction_cosine=float(args.minimum_direction_cosine),
        fold_a_trajectories=(
            tuple(value for value in args.fold_a_trajectories.split(",") if value)
            if args.fold_a_trajectories
            else None
        ),
        score_chunk=int(args.score_chunk),
        device=args.device,
    )
    result["inputs"] = {
        "registry": str(args.registry.resolve()),
        "query_cache": str(args.query_cache.resolve()),
        "metric_state": str(args.metric_state.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    report = {
        key: result[key]
        for key in (
            "schema",
            "version",
            "uses_test_queries",
            "audit_only",
            "deployment_descriptor_mutated",
            "replacement_scope",
            "descriptor_space",
            "crossfit_available",
            "blocker",
            "trajectory_labels",
            "fold_trajectory_labels",
            "minimum_support_queries",
            "minimum_support_strata",
            "minimum_direction_cosine",
            "trim_fraction",
            "inputs",
            "report",
        )
        if key in result
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
