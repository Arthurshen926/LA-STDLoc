#!/usr/bin/env python3
"""Merge deterministic raster-provenance shards by global query index."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge_provenance_shards(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("At least one raster-provenance shard is required")
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in paths
    ]
    first = shards[0]
    invariant_keys = (
        "schema",
        "version",
        "anchor_map",
        "query_cache",
        "gaussian_ply",
        "query_names",
        "primitive_count",
    )
    tensor_invariant_keys = (
        "source_universe",
        "anchor_source_offsets",
        "anchor_source_primitive_ids",
        "anchor_source_weights",
    )
    for shard in shards[1:]:
        for key in invariant_keys:
            if shard[key] != first[key]:
                raise ValueError(f"Raster-provenance shard mismatch: {key}")
        for key in tensor_invariant_keys:
            if not torch.equal(shard[key], first[key]):
                raise ValueError(f"Raster-provenance shard mismatch: {key}")

    query_total = len(first["query_names"])
    records: dict[int, dict] = {}
    for shard in shards:
        for record in shard["records"]:
            query_index = int(record["query_index"])
            if query_index in records:
                raise ValueError(f"Duplicate query index {query_index}")
            records[query_index] = record
    expected = list(range(query_total))
    if sorted(records) != expected:
        raise ValueError(
            "Raster-provenance shards do not cover every query once"
        )

    output = {
        key: first[key]
        for key in (*invariant_keys, *tensor_invariant_keys)
    }
    output["records"] = [records[index] for index in expected]
    output["config"] = {
        **first["config"],
        "num_shards": len(shards),
        "shard_index": -1,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_provenance_shards(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(
        f"Merged {len(args.inputs)} raster-provenance shards: "
        f"queries={len(merged['records'])} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
