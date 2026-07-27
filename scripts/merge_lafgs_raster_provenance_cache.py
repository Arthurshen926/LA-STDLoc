#!/usr/bin/env python3
"""Merge disjoint native-keypoint raster provenance shards."""

from __future__ import annotations

import argparse
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
    reference = shards[0]
    immutable = (
        "anchor_map",
        "query_cache",
        "gaussian_ply",
        "query_names",
        "primitive_count",
    )
    for shard in shards[1:]:
        for key in immutable:
            if shard[key] != reference[key]:
                raise ValueError(f"provenance shard mismatch: {key}")
        for key in (
            "source_universe",
            "anchor_source_offsets",
            "anchor_source_primitive_ids",
            "anchor_source_weights",
        ):
            if not torch.equal(
                torch.as_tensor(shard[key]),
                torch.as_tensor(reference[key]),
            ):
                raise ValueError(f"provenance shard mismatch: {key}")
    records = sorted(
        [
            record
            for shard in shards
            for record in shard["records"]
        ],
        key=lambda record: int(record["query_index"]),
    )
    indices = [int(record["query_index"]) for record in records]
    if indices != list(range(len(reference["query_names"]))):
        raise ValueError("provenance shards do not cover every query once")
    output = {
        **reference,
        "records": records,
        "config": {
            "merged_inputs": [
                str(Path(path).resolve()) for path in args.inputs
            ],
            "shard_configs": [shard["config"] for shard in shards],
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"merged {len(records)} provenance records -> {output_path}")


if __name__ == "__main__":
    main()
