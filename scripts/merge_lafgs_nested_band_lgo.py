#!/usr/bin/env python3
"""Merge query-identical, group-disjoint LGO screening shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    influences = sorted(
        [
            influence
            for shard in shards
            for influence in shard["influences"]
        ],
        key=lambda influence: int(influence["group_id"]),
    )
    ids = [int(influence["group_id"]) for influence in influences]
    if len(ids) != len(set(ids)):
        raise ValueError("LGO shards overlap")
    output = {
        "schema": "lafgs_nested_band_lgo",
        "version": 1,
        "groups_screened_total": shards[0]["groups_screened_total"],
        "influences": influences,
        "shard_configs": [shard["config"] for shard in shards],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    summary = {
        "evaluated_group_count": len(influences),
        "band_counts": {
            str(band): sum(
                int(influence["band"]) == band
                for influence in influences
            )
            for band in (1, 2, 3)
        },
        "remove_helpful_groups": sum(
            influence["operations"]["remove"]["delta"]["mean_te_m"] < 0
            for influence in influences
        ),
        "add_helpful_groups": sum(
            influence["operations"]["add_to_30k"]["delta"]["mean_te_m"] < 0
            for influence in influences
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
