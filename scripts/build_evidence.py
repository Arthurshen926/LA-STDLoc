#!/usr/bin/env python3
"""Build the canonical anchor universe and localization evidence graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from map_learning.pipeline import build_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-state", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--valid-masks", default="")
    parser.add_argument("--visibility-cache", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/paper_mainline.yaml")
    parser.add_argument("--function-graph-shards", type=int, default=1)
    parser.add_argument("--provenance-shards", type=int, default=1)
    parser.add_argument("--observation-shards", type=int, default=1)
    args = parser.parse_args()
    build_evidence(
        base_state=args.base_state,
        track_payload=args.track_payload,
        query_cache=args.query_cache,
        prior_ply=args.gaussian_ply,
        gaussian_type=args.gaussian_type,
        sh_degree=args.sh_degree,
        visibility_cache=args.visibility_cache,
        output=args.output,
        config=args.config,
        function_graph_shards=args.function_graph_shards,
        provenance_shards=args.provenance_shards,
        observation_shards=args.observation_shards,
        valid_masks=args.valid_masks or None,
    )


if __name__ == "__main__":
    main()
