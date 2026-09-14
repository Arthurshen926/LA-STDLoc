#!/usr/bin/env python3
"""Evaluate audited identity folding on mapping queries without deleting Anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from topology.anchor_equivalence import SCHEMA as AUDIT_SCHEMA
from topology.equivalence_counterfactual import evaluate_identity_folding


def _load_mmap(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--equivalence-audit", type=Path, required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=96)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    state = _load_mmap(args.map)
    teacher = _load_mmap(args.complete_positive_teacher)
    cache = _load_mmap(args.query_cache)
    audit = _load_mmap(args.equivalence_audit)
    if audit.get("schema") != AUDIT_SCHEMA:
        raise ValueError("unsupported equivalence audit schema")
    if audit.get("map_sha256") != sha256_file(args.map):
        raise ValueError("equivalence audit was built from another compact map")
    calibration = json.loads(args.scene_calibration.read_text())
    if calibration.get("schema") != "lafgs_mapping_only_scene_calibration":
        raise ValueError("identity folding requires mapping-only calibration")
    if calibration.get("uses_test_queries") is True or calibration.get(
        "sources", {}
    ).get("uses_test_queries") is True:
        raise ValueError("identity folding must not use test queries")
    parameters = calibration["parameters"]
    total = len(teacher["records"])
    if int(args.query_count) < 0:
        raise ValueError("query count must be non-negative")
    query_indices = None
    if 0 < int(args.query_count) < total:
        query_indices = (
            torch.linspace(0, total - 1, steps=int(args.query_count))
            .round()
            .long()
            .unique(sorted=True)
        )
    result = evaluate_identity_folding(
        state=state,
        metric_state_path=str(args.metric_state),
        teacher=teacher,
        query_cache=cache,
        component_ids=audit["independent_support_component_ids"],
        device=torch.device(args.device),
        reprojection_error_px=float(parameters["ransac_reprojection_px"]),
        clean_reprojection_px=float(parameters["clean_radius_px"]),
        seed=int(args.seed),
        query_indices=query_indices,
        deployment_row_limit=int(args.deployment_row_limit),
    )
    report = {
        "schema": "lafgs_identity_folding_counterfactual",
        "version": 1,
        "uses_test_queries": False,
        "physical_map_mutated": False,
        "map": str(args.map.resolve()),
        "map_sha256": sha256_file(args.map),
        "metric_state": str(args.metric_state.resolve()),
        "metric_state_sha256": sha256_file(args.metric_state),
        "equivalence_audit": str(args.equivalence_audit.resolve()),
        "equivalence_audit_sha256": sha256_file(args.equivalence_audit),
        "query_count": int(
            total if query_indices is None else query_indices.numel()
        ),
        "query_selection": "all" if query_indices is None else "uniform_mapping_gate",
        "seed": int(args.seed),
        **result,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(report["queries"], indent=2) + "\n"
    )
    (args.output / "summary.json").write_text(
        json.dumps({key: value for key, value in report.items() if key != "queries"}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
