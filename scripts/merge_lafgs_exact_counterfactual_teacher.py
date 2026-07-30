#!/usr/bin/env python3
"""Merge contiguous exact-counterfactual PoseLib teacher shards."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch

from localization_training.artifact_contract import sha256_file


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(value).resolve() for value in args.input]
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in paths
    ]
    shards.sort(key=lambda value: int(value["query_start"]))
    reference = shards[0]
    records = []
    summary = Counter()
    expected = int(shards[0]["query_start"])
    for shard in shards:
        if shard["schema"] != reference["schema"]:
            raise ValueError("exact-teacher shard schemas differ")
        for key in ("query_names", "anchor_count", "config", "provenance"):
            if shard[key] != reference[key]:
                raise ValueError(f"exact-teacher shard {key} differs")
        if int(shard["query_start"]) != expected:
            raise ValueError("exact-teacher shards are not contiguous")
        records.extend(shard["records"])
        summary.update(shard["summary"])
        expected = int(shard["query_stop"])
    payload = {
        **reference,
        "query_start": int(shards[0]["query_start"]),
        "query_stop": int(shards[-1]["query_stop"]),
        "records": records,
        "summary": dict(summary),
        "shards": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in paths
        ],
    }
    output = Path(args.output)
    _atomic_torch(output, payload)
    summary_path = output.with_suffix(".json")
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema": payload["schema"],
                "version": payload["version"],
                "query_start": payload["query_start"],
                "query_stop": payload["query_stop"],
                "summary": payload["summary"],
                "config": payload["config"],
                "shards": payload["shards"],
            },
            indent=2,
        )
        + "\n"
    )
    os.replace(temporary, summary_path)


if __name__ == "__main__":
    main()
