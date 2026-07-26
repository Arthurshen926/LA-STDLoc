#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch

from localization_training.micro_anchors import (
    build_add_only_materialized_anchor_map,
    compute_track_coverage_gain,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build a shared-evidence add-only micro-anchor capacity sweep"
    )
    parser.add_argument("--base_state", required=True)
    parser.add_argument("--track_payload", required=True)
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[0, 256, 512, 1024, 2048, 4096]
    )
    parser.add_argument("--minimum_coverage_gain", type=int, default=1)
    parser.add_argument("--minimum_distinct_view_bins", type=int, default=2)
    parser.add_argument("--minimum_separation_m", type=float, default=0.005)
    parser.add_argument("--descriptor_trim_fraction", type=float, default=0.2)
    parser.add_argument("--coverage_radius_px", type=float, default=2.0)
    args = parser.parse_args()

    base_state = torch.load(
        args.base_state, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    query_cache = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    coverage = compute_track_coverage_gain(
        payload=payload,
        query_cache=query_cache.get("queries", query_cache),
        base_xyz=base_state["landmark_xyz"],
        radius_px=args.coverage_radius_px,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for budget in args.budgets:
        state, diagnostics = build_add_only_materialized_anchor_map(
            base_state=base_state,
            payload=payload,
            query_cache=query_cache,
            budget=budget,
            minimum_coverage_gain=args.minimum_coverage_gain,
            minimum_distinct_view_bins=args.minimum_distinct_view_bins,
            minimum_separation_m=args.minimum_separation_m,
            descriptor_trim_fraction=args.descriptor_trim_fraction,
            radius_px=args.coverage_radius_px,
            coverage=coverage,
        )
        state["provenance"] = {
            "base_state": str(Path(args.base_state).resolve()),
            "track_payload": str(Path(args.track_payload).resolve()),
            "query_cache": str(Path(args.query_cache).resolve()),
        }
        path = output_dir / f"add_only_{int(budget):04d}.pt"
        torch.save(state, path)
        summaries[str(int(budget))] = {"state": str(path), **diagnostics}
    summary_path = output_dir / "capacity_sweep_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
