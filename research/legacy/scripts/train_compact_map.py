#!/usr/bin/env python3
"""Rebuild compact evidence and train one lineage-aligned metric from scratch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from map_learning.pipeline import train_compact_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-map", type=Path, required=True)
    parser.add_argument("--canonical-function-graph", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_mainline.yaml"))
    parser.add_argument("--valid-masks", type=Path)
    parser.add_argument("--function-graph-shards", type=int, default=1)
    parser.add_argument("--provenance-shards", type=int, default=1)
    parser.add_argument("--observation-shards", type=int, default=1)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    args = parser.parse_args()
    result = train_compact_map(
        compact_map=args.compact_map,
        function_graph=args.canonical_function_graph,
        track_payload=args.track_payload,
        query_cache=args.query_cache,
        prior_ply=args.gaussian_ply,
        gaussian_type=args.gaussian_type,
        sh_degree=args.sh_degree,
        output=args.output,
        config=args.config,
        valid_masks=args.valid_masks,
        rebuild_function_graph=True,
        function_graph_shards=args.function_graph_shards,
        provenance_shards=args.provenance_shards,
        observation_shards=args.observation_shards,
        scene_calibration=args.scene_calibration,
    )
    print(json.dumps({name: str(path) for name, path in result.items()}, indent=2))


if __name__ == "__main__":
    main()
