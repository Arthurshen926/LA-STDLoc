#!/usr/bin/env python3
"""Merge independently extracted synthetic-evidence shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.synthetic_evidence import (
    synthetic_function_graph_payload,
    synthetic_positive_teacher_payload,
    synthetic_query_cache_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shards = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    records = [record for shard in shards for record in shard["records"]]
    names = [str(record["query_name"]) for record in records]
    if len(names) != len(set(names)):
        raise ValueError("synthetic evidence shards contain duplicate query names")
    rejected = [
        record for shard in shards for record in shard.get("rejected_records", [])
    ]
    evidence = {
        "schema": "lafgs_artifact_filtered_synthetic_appearance_evidence",
        "version": 1,
        "query_names": names,
        "records": records,
        "rejected_records": rejected,
        "summary": {
            "candidate_view_count": sum(
                int(shard["summary"]["candidate_view_count"]) for shard in shards
            ),
            "accepted_view_count": len(records),
            "rejected_view_count": len(rejected),
            "positive_pair_count": sum(
                int(record["positive_pair_count"]) for record in records
            ),
            "matchable_rate_mean": (
                sum(float(record["matchable_rate"]) for record in records)
                / max(len(records), 1)
            ),
        },
        "provenance": {
            "schema": "lafgs_merged_synthetic_evidence_shards",
            "inputs": [str(Path(path).resolve()) for path in args.inputs],
            "geometry_policy": (
                "existing Track-First anchors only; rendered evidence cannot "
                "create or move geometry"
            ),
        },
    }
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(evidence, path)
    torch.save(
        synthetic_query_cache_payload(evidence),
        path.with_name(path.stem + "_query_cache.pt"),
    )
    torch.save(
        synthetic_positive_teacher_payload(evidence, anchor_count=anchor_count),
        path.with_name(path.stem + "_positive_teacher.pt"),
    )
    torch.save(
        synthetic_function_graph_payload(evidence, anchor_count=anchor_count),
        path.with_name(path.stem + "_function_graph.pt"),
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": evidence["schema"],
                "summary": evidence["summary"],
                "provenance": evidence["provenance"],
            },
            indent=2,
        )
        + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
