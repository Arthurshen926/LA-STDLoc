#!/usr/bin/env python3
"""Aggregate V10 group gains and materialize authorized descriptor groups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v10_anchor_descriptor_controller import (
    aggregate_group_descriptor_action_gain,
)
from topology.v6_anchor_map import identity_metric_state


def _save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    map_path = args.map.resolve()
    map_sha = sha256_file(map_path)
    proposal_path = args.proposal.resolve()
    proposal = torch.load(proposal_path, map_location="cpu", weights_only=False)
    groups_path = args.groups.resolve()
    group_state = torch.load(groups_path, map_location="cpu", weights_only=False)
    payloads = [json.loads(path.read_text()) for path in args.replay_shards]
    if any(
        payload.get("schema") != "lafgs_v10_group_descriptor_counterfactual_shard"
        or payload.get("loo_used") is not False
        or payload.get("map_sha256") != map_sha
        or payload.get("proposal_sha256") != sha256_file(proposal_path)
        or payload.get("groups_sha256") != sha256_file(groups_path)
        for payload in payloads
    ):
        raise ValueError("invalid V10 group replay shard")
    records = [record for payload in payloads for record in payload["records"]]
    audit = aggregate_group_descriptor_action_gain(records)
    authorized_groups = set(audit["authorized_group_indices"].tolist())
    authorized_rows = sorted(
        anchor
        for index, group in enumerate(group_state["groups"])
        if index in authorized_groups
        for anchor in group
    )
    descriptor_by_row = {
        int(row): torch.as_tensor(proposal["candidate_descriptors"])[index].float()
        for index, row in enumerate(
            torch.as_tensor(proposal["candidate_anchor_rows"]).long().tolist()
        )
    }
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    candidate = dict(state)
    features = torch.as_tensor(state["anchor_features"]).clone().float()
    for row in authorized_rows:
        features[row] = descriptor_by_row[row]
    candidate["anchor_features"] = features
    candidate["provenance"] = {
        **dict(state.get("provenance", {})),
        "v10_bounded_confusion_group_descriptor_action": True,
        "v10_updated_anchor_count": len(authorized_rows),
        "v10_authorized_group_count": len(authorized_groups),
        "feedback_descriptor_exact_copy": False,
        "feedback_descriptor_robust_aggregate": True,
        "loo_used": False,
        "uses_test_queries": False,
    }
    args.output_dir.mkdir(parents=True)
    audit_path = args.output_dir / "group_action_audit.pt"
    _save(audit, audit_path)
    output_map = args.output_dir / "projective_anchor_map.pt"
    _save(candidate, output_map)
    output_metric = args.output_dir / "identity_metric.pt"
    _save(
        identity_metric_state(
            candidate,
            map_path=str(output_map.resolve()),
            map_sha256=sha256_file(output_map),
        ),
        output_metric,
    )
    report = {
        "schema": "lafgs_v10_group_descriptor_map_report",
        "version": 1,
        "status": "PROPOSAL",
        "loo_used": False,
        "uses_test_queries": False,
        "authorized_group_count": len(authorized_groups),
        "updated_anchor_count": len(authorized_rows),
        "geometry_mutation_count": 0,
        "descriptor_count_per_anchor": 1,
        "output": {
            "audit": str(audit_path.resolve()),
            "audit_sha256": sha256_file(audit_path),
            "map": str(output_map.resolve()),
            "map_sha256": sha256_file(output_map),
            "metric": str(output_metric.resolve()),
            "metric_sha256": sha256_file(output_metric),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
