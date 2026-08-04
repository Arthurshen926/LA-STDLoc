#!/usr/bin/env python3
"""Merge deterministic complete-positive-teacher query shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


DIAGNOSTIC_KEYS = (
    "query_count",
    "positive_rows",
    "strong_pair_count",
    "ambiguous_pair_count",
    "exact_track_positive_count",
)


def merge_observation_shards(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("At least one observation shard is required")
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in paths
    ]
    first = shards[0]
    invariant_keys = (
        "schema",
        "version",
        "anchor_count",
        "query_names",
        "anchor_map",
        "query_cache",
        "raster_provenance",
        "track_payload",
    )
    method_config = {
        key: value
        for key, value in first["config"].items()
        if key not in {"num_shards", "shard_index"}
    }
    for shard in shards[1:]:
        for key in invariant_keys:
            if shard[key] != first[key]:
                raise ValueError(f"Observation shard mismatch: {key}")
        shard_method_config = {
            key: value
            for key, value in shard["config"].items()
            if key not in {"num_shards", "shard_index"}
        }
        if shard_method_config != method_config:
            raise ValueError("Observation shard method config mismatch")

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
        raise ValueError("Observation shards do not cover every query once")

    output = {key: first[key] for key in invariant_keys}
    output["records"] = [records[index] for index in expected]
    output["diagnostics"] = {
        key: sum(int(shard["diagnostics"][key]) for shard in shards)
        for key in DIAGNOSTIC_KEYS
    }
    output["config"] = {
        **method_config,
        "num_shards": len(shards),
        "shard_index": -1,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_observation_shards(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(
        f"Merged {len(args.inputs)} observation shards: "
        f"queries={len(merged['records'])} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
