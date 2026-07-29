#!/usr/bin/env python3
"""Merge contiguous Basin Teacher V2 shards with identity checks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    shards.sort(key=lambda shard: int(shard["query_start"]))
    first = shards[0]
    records = []
    names = []
    expected = int(first["query_start"])
    totals = {
        "good": 0,
        "harmful": 0,
        "near_miss": 0,
        "queries_with_good": 0,
        "queries_with_harmful": 0,
        "queries_with_near_miss": 0,
    }
    for shard in shards:
        if shard.get("schema") != "lafgs_basin_teacher":
            raise ValueError("unsupported basin teacher shard")
        if int(shard["anchor_count"]) != int(first["anchor_count"]):
            raise ValueError("basin teacher anchor registries differ")
        if shard["artifacts"] != first["artifacts"]:
            raise ValueError("basin teacher source artifact identities differ")
        if int(shard["query_start"]) != expected:
            raise ValueError("basin teacher shards are not contiguous")
        expected = int(shard["query_stop"])
        records.extend(shard["records"])
        names.extend(shard["query_names"])
        for key in totals:
            totals[key] += int(shard["summary"].get(key, 0))
    summary = {
        **totals,
        "query_count": len(records),
        "good_sets_per_query": totals["good"] / max(len(records), 1),
        "harmful_sets_per_query": totals["harmful"] / max(len(records), 1),
        "near_miss_per_query": totals["near_miss"] / max(len(records), 1),
    }
    output = {
        **first,
        "query_names": names,
        "query_start": int(first["query_start"]),
        "query_stop": expected,
        "records": records,
        "summary": summary,
        "shards": [str(Path(path).resolve()) for path in args.inputs],
    }
    output["config"] = {
        key: value
        for key, value in first["config"].items()
        if key not in {"output", "query_start", "query_limit"}
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(output, temporary)
    os.replace(temporary, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": output["schema"],
                "version": output["version"],
                "summary": summary,
                "config": output["config"],
                "artifacts": output["artifacts"],
                "shards": output["shards"],
            },
            indent=2,
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
