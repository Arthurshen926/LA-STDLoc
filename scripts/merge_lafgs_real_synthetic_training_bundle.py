#!/usr/bin/env python3
"""Merge real A1 evidence with filtered existing-anchor rendered evidence."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch


def _load(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _renumber(records: list[dict]) -> list[dict]:
    output = []
    for index, record in enumerate(records):
        value = dict(record)
        value["query_index"] = index
        output.append(value)
    return output


def _validate_critical_teacher(critical: dict, real_names: list[str]) -> None:
    """Reject replay/outcome payloads that are not pair-aligned teachers."""
    if real_names != [str(value) for value in critical.get("query_names", [])]:
        raise ValueError("real critical teacher query order differs")
    records = critical.get("records", [])
    if len(records) != len(real_names):
        raise ValueError("real critical teacher query count differs")
    required = {"query_rows", "positive_weights", "row_weights"}
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ValueError(
                "real critical teacher is not pair-aligned: "
                f"record {index} is missing {sorted(missing)}. "
                "Do not pass a dynamic replay/outcome payload as a teacher."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-function-graph", required=True)
    parser.add_argument("--real-query-cache", required=True)
    parser.add_argument("--real-positive-teacher", required=True)
    parser.add_argument("--real-track-payload", required=True)
    parser.add_argument("--real-critical-teacher", default="")
    parser.add_argument("--synthetic-evidence", required=True)
    parser.add_argument("--synthetic-function-graph", required=True)
    parser.add_argument("--synthetic-query-cache", required=True)
    parser.add_argument("--synthetic-positive-teacher", required=True)
    parser.add_argument("--synthetic-row-weight", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    real_graph = _load(args.real_function_graph)
    real_cache_payload = _load(args.real_query_cache)
    real_teacher = _load(args.real_positive_teacher)
    track = _load(args.real_track_payload)
    critical = _load(args.real_critical_teacher) if args.real_critical_teacher else None
    evidence = _load(args.synthetic_evidence)
    synthetic_graph = _load(args.synthetic_function_graph)
    synthetic_cache_payload = _load(args.synthetic_query_cache)
    synthetic_teacher = _load(args.synthetic_positive_teacher)

    real_names = [str(value) for value in real_graph["query_names"]]
    synthetic_names = [str(value) for value in synthetic_graph["query_names"]]
    if real_names != [str(value) for value in real_teacher["query_names"]]:
        raise ValueError("real graph and positive teacher query order differ")
    if real_names != [str(value) for value in track["query_names"]]:
        raise ValueError("real graph and track payload query order differ")
    if synthetic_names != [str(value) for value in synthetic_teacher["query_names"]]:
        raise ValueError("synthetic graph and positive teacher query order differ")
    evidence_names = [str(value) for value in evidence["query_names"]]
    if synthetic_names != evidence_names:
        raise ValueError("synthetic evidence and graph query order differ")
    if set(real_names) & set(synthetic_names):
        raise ValueError("real and synthetic query names must be disjoint")
    if int(real_teacher["anchor_count"]) != int(synthetic_teacher["anchor_count"]):
        raise ValueError("real and synthetic positive teachers use different maps")

    names = real_names + synthetic_names
    graph = dict(real_graph)
    graph["schema"] = "lafgs_real_viewpoint_complete_function_graph"
    graph["version"] = 1
    graph["query_names"] = names
    graph["query_count_total"] = len(names)
    real_graph_records = [
        {**record, "evidence_source": "real_mapping"}
        for record in real_graph["records"]
    ]
    synthetic_graph_records = [
        {
            **record,
            "evidence_source": "rendered_viewpoint_completion",
            "ambiguous_training_policy": "ignore",
        }
        for record in synthetic_graph["records"]
    ]
    graph["records"] = _renumber(real_graph_records + synthetic_graph_records)
    graph["viewpoint_completion"] = evidence["provenance"]

    real_queries = real_cache_payload.get("queries", real_cache_payload)
    synthetic_queries = synthetic_cache_payload.get("queries", synthetic_cache_payload)
    if set(real_queries) != set(real_names):
        raise ValueError("real query cache registry differs")
    if set(synthetic_queries) != set(synthetic_names):
        raise ValueError("synthetic query cache registry differs")
    cache = {
        "schema": "lafgs_real_viewpoint_complete_query_cache",
        "version": 1,
        "queries": {**real_queries, **synthetic_queries},
    }

    teacher = dict(real_teacher)
    teacher["schema"] = "lafgs_real_viewpoint_complete_positive_teacher"
    teacher["version"] = 1
    teacher["query_names"] = names
    teacher["records"] = _renumber(
        list(real_teacher["records"]) + list(synthetic_teacher["records"])
    )
    diagnostics = dict(real_teacher.get("diagnostics", {}))
    diagnostics["strong_pair_count"] = sum(
        int(torch.as_tensor(record["positive_indices"]).numel())
        for record in teacher["records"]
    )
    diagnostics["real_query_count"] = len(real_names)
    diagnostics["synthetic_query_count"] = len(synthetic_names)
    diagnostics["synthetic_fraction"] = len(synthetic_names) / max(len(names), 1)
    teacher["diagnostics"] = diagnostics

    track_output = deepcopy(track)
    track_output["query_names"] = names
    synthetic_bins = [
        int(record.get("view_bin", 0)) for record in evidence["records"]
    ]
    track_output["query_bins"] = torch.cat(
        (torch.as_tensor(track["query_bins"]).long(), torch.as_tensor(synthetic_bins).long())
    )
    track_output["viewpoint_completion"] = {
        "real_query_count": len(real_names),
        "synthetic_query_count": len(synthetic_names),
    }

    critical_records = []
    if critical is not None:
        _validate_critical_teacher(critical, real_names)
        critical_records.extend(critical["records"])
    else:
        for record in real_teacher["records"]:
            row_count = int(torch.as_tensor(record["query_rows"]).numel())
            pair_count = int(torch.as_tensor(record["positive_indices"]).numel())
            critical_records.append(
                {
                    "query_rows": torch.as_tensor(record["query_rows"]).long(),
                    "positive_weights": torch.ones(pair_count),
                    "row_weights": torch.ones(row_count),
                }
            )
    for record in synthetic_teacher["records"]:
        row_count = int(torch.as_tensor(record["query_rows"]).numel())
        pair_count = int(torch.as_tensor(record["positive_indices"]).numel())
        critical_records.append(
            {
                "query_rows": torch.as_tensor(record["query_rows"]).long(),
                "positive_weights": torch.ones(pair_count),
                "row_weights": torch.full(
                    (row_count,), float(args.synthetic_row_weight)
                ),
                "promotion_positive_anchor": torch.full(
                    (row_count,), -1, dtype=torch.long
                ),
                "promotion_negative_anchor": torch.full(
                    (row_count,), -1, dtype=torch.long
                ),
                "promotion_weights": torch.zeros(row_count),
                "promotion_types": torch.zeros(row_count, dtype=torch.uint8),
            }
        )
    critical_output = {
        "schema": "lafgs_trajectory_stable_viewpoint_complete_teacher",
        "version": 1,
        "anchor_count": int(real_teacher["anchor_count"]),
        "query_names": names,
        "records": _renumber(critical_records),
        "summary": {
            "real_query_count": len(real_names),
            "synthetic_query_count": len(synthetic_names),
            "synthetic_row_weight": float(args.synthetic_row_weight),
        },
    }

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "function_graph.pt": graph,
        "query_cache.pt": cache,
        "complete_positive_teacher.pt": teacher,
        "track_payload.pt": track_output,
        "critical_teacher.pt": critical_output,
    }
    for name, payload in artifacts.items():
        torch.save(payload, output / name)
    report = {
        "schema": "lafgs_viewpoint_complete_training_bundle",
        "version": 1,
        "real_query_count": len(real_names),
        "synthetic_query_count": len(synthetic_names),
        "synthetic_fraction": len(synthetic_names) / max(len(names), 1),
        "anchor_count": int(real_teacher["anchor_count"]),
        "artifacts": {name: str((output / name).resolve()) for name in artifacts},
    }
    (output / "bundle.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
