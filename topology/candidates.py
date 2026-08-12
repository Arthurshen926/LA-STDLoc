#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch

from evidence.tracks import (
    build_add_only_materialized_anchor_map,
    build_canonical_base_anchor_map,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build add-only Track-First localization micro-anchors"
    )
    parser.add_argument("--base_state", required=True)
    parser.add_argument("--track_payload", required=True)
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--minimum_coverage_gain", type=int, default=1)
    parser.add_argument("--minimum_distinct_view_bins", type=int, default=2)
    parser.add_argument("--minimum_separation_m", type=float, default=0.005)
    parser.add_argument("--descriptor_trim_fraction", type=float, default=0.2)
    parser.add_argument("--coverage_radius_px", type=float, default=2.0)
    parser.add_argument(
        "--evaluate_zero_budget_eligibility",
        action="store_true",
        help=(
            "run the legacy full Track/query coverage audit even when budget "
            "is zero"
        ),
    )
    args = parser.parse_args()

    base_state = torch.load(
        args.base_state, map_location="cpu", weights_only=False
    )
    if int(args.budget) == 0 and not args.evaluate_zero_budget_eligibility:
        state, diagnostics = build_canonical_base_anchor_map(
            base_state=base_state,
            minimum_coverage_gain=args.minimum_coverage_gain,
            minimum_distinct_view_bins=args.minimum_distinct_view_bins,
            minimum_separation_m=args.minimum_separation_m,
            descriptor_trim_fraction=args.descriptor_trim_fraction,
            radius_px=args.coverage_radius_px,
        )
    else:
        track_payload = torch.load(
            args.track_payload, map_location="cpu", weights_only=False
        )
        query_cache = torch.load(
            args.query_cache, map_location="cpu", weights_only=False
        )
        state, diagnostics = build_add_only_materialized_anchor_map(
            base_state=base_state,
            payload=track_payload,
            query_cache=query_cache,
            budget=args.budget,
            minimum_coverage_gain=args.minimum_coverage_gain,
            minimum_distinct_view_bins=args.minimum_distinct_view_bins,
            minimum_separation_m=args.minimum_separation_m,
            descriptor_trim_fraction=args.descriptor_trim_fraction,
            radius_px=args.coverage_radius_px,
        )
    state["provenance"] = {
        "base_state": str(Path(args.base_state).resolve()),
        "track_payload": str(Path(args.track_payload).resolve()),
        "query_cache": str(Path(args.query_cache).resolve()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    report = output.with_suffix(".json")
    report.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), **diagnostics}, indent=2))


if __name__ == "__main__":
    main()
