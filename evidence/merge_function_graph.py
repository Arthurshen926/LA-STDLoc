#!/usr/bin/env python3
"""Merge deterministic function-graph shards into the canonical artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


COUNTER_KEYS = (
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
    "legal_hit_strong_count",
    "legal_hit_clean_count",
    "legal_hit_ambiguous_count",
    "legal_winner_strong_count",
    "legal_winner_clean_count",
    "solver_inlier_gtclean_strong_count",
    "solver_inlier_gtclean_clean_count",
)


def merge_function_graph_shards(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("At least one function-graph shard is required")
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
        "anchor_count",
        "query_count_total",
        "query_names",
        "raster_visibility_enabled",
    )
    if "resolved_thresholds" in first:
        invariant_keys = (*invariant_keys, "resolved_thresholds")
    tensor_invariant_keys = (
        "source_primitive_ids",
        "track_cluster_ids",
        "anchor_type",
    )
    for shard in shards[1:]:
        for key in invariant_keys:
            if shard[key] != first[key]:
                raise ValueError(f"Function-graph shard mismatch: {key}")
        for key in tensor_invariant_keys:
            if not torch.equal(shard[key], first[key]):
                raise ValueError(f"Function-graph shard mismatch: {key}")

    query_total = int(first["query_count_total"])
    indexed_records: dict[int, dict] = {}
    indexed_diagnostics: dict[int, dict] = {}
    covered: list[int] = []
    for shard in shards:
        indices = torch.as_tensor(shard["query_indices"]).long().tolist()
        records = shard["records"]
        diagnostics = shard["query_diagnostics"]
        if len(indices) != len(records) or len(indices) != len(diagnostics):
            raise ValueError("Shard query metadata has inconsistent lengths")
        for query_index, record, diagnostic in zip(
            indices, records, diagnostics
        ):
            query_index = int(query_index)
            if query_index in indexed_records:
                raise ValueError(f"Duplicate query index {query_index}")
            if int(record["query_index"]) != query_index:
                raise ValueError("Record query index does not match shard index")
            if int(diagnostic["query_index"]) != query_index:
                raise ValueError(
                    "Diagnostic query index does not match shard index"
                )
            indexed_records[query_index] = record
            indexed_diagnostics[query_index] = diagnostic
            covered.append(query_index)
    expected = list(range(query_total))
    if sorted(covered) != expected:
        raise ValueError("Function-graph shards do not cover every query once")

    output = {
        key: first[key]
        for key in (*invariant_keys, *tensor_invariant_keys)
    }
    output["query_indices"] = torch.arange(query_total, dtype=torch.int32)
    output["records"] = [indexed_records[index] for index in expected]
    output["query_diagnostics"] = [
        indexed_diagnostics[index] for index in expected
    ]
    output["config"] = {
        **first["config"],
        "num_shards": len(shards),
        "shard_index": -1,
    }
    for key in COUNTER_KEYS:
        if key not in first:
            continue
        output[key] = torch.stack(
            [torch.as_tensor(shard[key]).long() for shard in shards]
        ).sum(dim=0)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merged = merge_function_graph_shards(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(
        f"Merged {len(args.inputs)} function-graph shards: "
        f"queries={merged['query_count_total']} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
