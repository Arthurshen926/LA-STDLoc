#!/usr/bin/env python3
"""Distill Track core and Gaussian coverage reserve into a compact map."""

from __future__ import annotations

import argparse
from pathlib import Path

from map_learning.pipeline import distill_compact_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--positive-teacher", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/paper_mainline.yaml")
    parser.add_argument(
        "--scene-calibration",
        type=Path,
        help="Exact variant-bound calibration for a frozen single-factor run.",
    )
    args = parser.parse_args()
    print(
        distill_compact_map(
            canonical_map=args.canonical_map,
            function_graph=args.function_graph,
            positive_teacher=args.positive_teacher,
            track_payload=args.track_payload,
            query_cache=args.query_cache,
            output=args.output,
            config=args.config,
            scene_calibration=args.scene_calibration,
        )
    )


if __name__ == "__main__":
    main()
