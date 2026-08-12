#!/usr/bin/env python3
"""Audit cross-group alias risk over every eligible selector candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from topology.alias_risk import (
    aggregate_group_alias_evidence,
    alias_risk_from_counters,
    crossfit_alias_separability,
)


def _distribution(values: torch.Tensor) -> dict:
    values = torch.as_tensor(values).float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "p50": float(torch.quantile(values, 0.5)),
        "p90": float(torch.quantile(values, 0.9)),
        "p95": float(torch.quantile(values, 0.95)),
        "p99": float(torch.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function-graph", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--candidate-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    graph = torch.load(args.function_graph, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    candidate_map = torch.load(args.candidate_map, map_location="cpu", weights_only=False)
    if candidate_map.get("audit_contract", {}).get("uses_test_queries") is not False:
        raise ValueError("alias audit requires a mapping-only candidate universe")
    graph_names = list(graph["query_names"])
    payload_names = list(payload["query_names"])
    name_to_payload = {name: index for index, name in enumerate(payload_names)}
    if set(graph_names) - set(name_to_payload):
        raise ValueError("track payload does not cover function-graph queries")
    payload_groups = torch.as_tensor(payload["query_bins"]).long()
    query_groups = torch.as_tensor(
        [int(payload_groups[name_to_payload[name]]) for name in graph_names]
    )
    counters = aggregate_group_alias_evidence(graph, query_groups)
    risk = alias_risk_from_counters(counters)
    crossfit = crossfit_alias_separability(counters)
    anchor_type = torch.as_tensor(candidate_map["anchor_type"]).long()
    if anchor_type.numel() != int(graph["anchor_count"]):
        raise ValueError("candidate map and function graph do not align")
    by_evidence = {}
    for label, mask in (
        ("track", anchor_type != 0),
        ("surface", anchor_type == 0),
    ):
        by_evidence[label] = {
            "candidate_count": int(mask.sum()),
            "risk": _distribution(risk["alias_risk"][mask]),
            "recurrent_alias_count": int(risk["recurrent_alias"][mask].sum()),
            "harmful_event_count": int(
                torch.as_tensor(counters["harmful"])[:, mask].sum()
            ),
            "false_winner_count": int(
                torch.as_tensor(counters["false"])[:, mask].sum()
            ),
        }
    report = {
        "schema": "lafgs_all_candidate_alias_risk_audit",
        "version": 1,
        "uses_test_queries": False,
        "selector_change_authorized": False,
        "candidate_map_sha256": sha256_file(args.candidate_map),
        "function_graph_sha256": sha256_file(args.function_graph),
        "candidate_count": int(graph["anchor_count"]),
        "query_count": len(graph_names),
        "raster_visibility_enabled": bool(graph["raster_visibility_enabled"]),
        "legality_model": (
            "rendered_depth_alpha_and_ground_truth_reprojection"
            if not graph["raster_visibility_enabled"]
            else "rendered_depth_alpha_reprojection_and_raster_visibility"
        ),
        "query_group_semantics": "track_payload_query_bins",
        "risk_definition": "max_wilson95_lower_false_rate_harmful_inlier_rate",
        "uncertainty_definition": "wilson95_upper_minus_lower_reported_separately",
        "recurrence_definition": "false_or_harmful_in_at_least_two_query_groups",
        "by_evidence": by_evidence,
        "crossfit": crossfit,
    }
    artifact = {
        **report,
        "candidate_universe_ids": torch.as_tensor(
            candidate_map["candidate_universe_ids"]
        ).long(),
        "query_group_ids": query_groups,
        "counters": counters,
        "risk": risk,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
