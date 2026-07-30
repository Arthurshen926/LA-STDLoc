#!/usr/bin/env python3
"""Merge disjoint dynamic self-localization shards in reference query order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def merge_dynamic_outcomes(
    shards: list[dict],
    *,
    reference_query_names: list[str],
) -> dict:
    if not shards:
        raise ValueError("at least one dynamic outcome shard is required")
    identity = {
        key: shards[0].get(key)
        for key in ("schema", "version", "anchor_count", "map", "metric_state")
    }
    by_name = {}
    for shard in shards:
        current = {key: shard.get(key) for key in identity}
        if current != identity:
            raise ValueError("dynamic outcome shard identity differs")
        if len(shard["query_names"]) != len(shard["records"]):
            raise ValueError("dynamic outcome shard record count differs")
        for name, record in zip(shard["query_names"], shard["records"]):
            name = str(name)
            if name in by_name:
                raise ValueError(f"duplicate dynamic query: {name}")
            if str(record["query_name"]) != name:
                raise ValueError("dynamic outcome record registry differs")
            by_name[name] = record
    expected = set(reference_query_names)
    if set(by_name) != expected:
        missing = sorted(expected - set(by_name))
        extra = sorted(set(by_name) - expected)
        raise ValueError(
            f"dynamic shard coverage differs: missing={missing[:3]}, extra={extra[:3]}"
        )
    return {
        **shards[0],
        "query_names": list(reference_query_names),
        "records": [by_name[name] for name in reference_query_names],
        "summary": {
            "query_count": len(reference_query_names),
            "source_shard_count": len(shards),
        },
        "source_shards": [
            shard.get("_source_path") for shard in shards
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--reference-topk", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shards = []
    for path in args.input:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        shard["_source_path"] = str(Path(path).resolve())
        shards.append(shard)
    reference = torch.load(
        args.reference_topk, map_location="cpu", weights_only=False
    )
    merged = merge_dynamic_outcomes(
        shards,
        reference_query_names=list(reference["query_names"]),
    )
    merged.pop("_source_path", None)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(merged, temporary)
    os.replace(temporary, output)
    print(
        {
            "output": str(output),
            "query_count": len(merged["query_names"]),
            "source_shard_count": len(shards),
        }
    )


if __name__ == "__main__":
    main()
