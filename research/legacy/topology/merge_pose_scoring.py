#!/usr/bin/env python3
"""Merge pose-reserve scoring shards before one global greedy selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def merge_pose_scoring_shards(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("At least one pose-scoring shard is required")
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in paths
    ]
    identity = shards[0]["identity"]
    candidates: dict[int, list[tuple[int, float]]] = {}
    diagnostics: dict[int, dict] = {}
    for shard in shards:
        if shard["identity"] != identity:
            raise ValueError("Pose-scoring shard identity mismatch")
        indices = [int(index) for index in shard["query_indices"]]
        shard_candidates = shard["query_candidates"]
        shard_diagnostics = shard["query_diagnostics"]
        if (
            len(indices) != len(shard_candidates)
            or len(indices) != len(shard_diagnostics)
        ):
            raise ValueError("Pose-scoring shard lengths are inconsistent")
        for query_index, query_candidates, query_diagnostic in zip(
            indices, shard_candidates, shard_diagnostics
        ):
            if query_index in candidates:
                raise ValueError(f"Duplicate query index {query_index}")
            candidates[query_index] = query_candidates
            diagnostics[query_index] = query_diagnostic
    expected = list(range(len(candidates)))
    if sorted(candidates) != expected:
        raise ValueError("Pose-scoring shards are not contiguous and complete")
    return {
        "identity": identity,
        "query_indices": expected,
        "query_candidates": [candidates[index] for index in expected],
        "query_diagnostics": [diagnostics[index] for index in expected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_pose_scoring_shards(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(
        f"Merged {len(args.inputs)} pose-scoring shards: "
        f"queries={len(merged['query_indices'])} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
