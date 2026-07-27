#!/usr/bin/env python3
"""Merge query-disjoint Active Map V4 function-graph shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


COUNTERS = (
    "candidate_opportunity_count",
    "winner_count",
    "legal_hit_2px_count",
    "legal_hit_4px_count",
    "legal_hit_8px_count",
    "legal_winner_2px_count",
    "legal_winner_4px_count",
    "solver_inlier_count",
    "solver_inlier_gtclean_2px_count",
    "solver_inlier_gtclean_4px_count",
    "harmful_solver_inlier_count",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    first = shards[0]
    for shard in shards:
        if shard["schema"] != "lafgs_keypoint_function_graph_shard":
            raise ValueError("unexpected graph shard schema")
        if shard["anchor_count"] != first["anchor_count"]:
            raise ValueError("anchor count mismatch")
        if shard["query_names"] != first["query_names"]:
            raise ValueError("query ordering mismatch")
        for key in (
            "source_primitive_ids",
            "track_cluster_ids",
            "anchor_type",
        ):
            if not torch.equal(shard[key], first[key]):
                raise ValueError(f"{key} mismatch")
    records = sorted(
        [record for shard in shards for record in shard["records"]],
        key=lambda record: int(record["query_index"]),
    )
    query_indices = [int(record["query_index"]) for record in records]
    if len(query_indices) != len(set(query_indices)):
        raise ValueError("graph shards overlap")
    output = {
        "schema": "lafgs_keypoint_function_graph",
        "version": 2,
        "anchor_map": first["anchor_map"],
        "query_cache": first["query_cache"],
        "anchor_count": first["anchor_count"],
        "query_count_total": first["query_count_total"],
        "query_names": first["query_names"],
        "query_indices": torch.as_tensor(query_indices, dtype=torch.int32),
        "source_primitive_ids": first["source_primitive_ids"],
        "track_cluster_ids": first["track_cluster_ids"],
        "anchor_type": first["anchor_type"],
        "records": records,
        "query_diagnostics": sorted(
            [
                diagnostic
                for shard in shards
                for diagnostic in shard["query_diagnostics"]
            ],
            key=lambda diagnostic: int(diagnostic["query_index"]),
        ),
        "raster_visibility_enabled": all(
            bool(shard["raster_visibility_enabled"])
            for shard in shards
        ),
        "shard_configs": [shard["config"] for shard in shards],
        **{
            key: sum(
                (torch.as_tensor(shard[key]) for shard in shards),
                torch.zeros_like(torch.as_tensor(first[key])),
            )
            for key in COUNTERS
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    opportunity = output["candidate_opportunity_count"].clamp_min(1)
    summary = {
        "anchor_count": output["anchor_count"],
        "query_count": len(records),
        "raster_visibility_enabled": output["raster_visibility_enabled"],
        "anchors_with_legal_2px": int(
            (output["legal_hit_2px_count"] > 0).sum()
        ),
        "anchors_with_legal_4px": int(
            (output["legal_hit_4px_count"] > 0).sum()
        ),
        "anchors_with_gtclean_solver_inlier": int(
            (output["solver_inlier_gtclean_4px_count"] > 0).sum()
        ),
        "anchors_with_harmful_solver_inlier": int(
            (output["harmful_solver_inlier_count"] > 0).sum()
        ),
        "harmful_consensus_rate_mean": float(
            (
                output["harmful_solver_inlier_count"].float()
                / opportunity
            ).mean()
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
