#!/usr/bin/env python3
"""Aggregate V10 single-action gains and materialize the authorized map."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v10_anchor_descriptor_controller import (
    aggregate_descriptor_action_gain,
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
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    map_path = args.map.resolve()
    map_sha = sha256_file(map_path)
    proposal_path = args.proposal.resolve()
    proposal_sha = sha256_file(proposal_path)
    proposal = torch.load(proposal_path, map_location="cpu", weights_only=False)
    payloads = [json.loads(path.read_text()) for path in args.replay_shards]
    if any(
        payload.get("schema") != "lafgs_v10_single_descriptor_counterfactual_shard"
        or payload.get("loo_used") is not False
        or payload.get("uses_test_queries") is not False
        or payload.get("map_sha256") != map_sha
        or payload.get("proposal_sha256") != proposal_sha
        for payload in payloads
    ):
        raise ValueError("invalid V10 single-descriptor replay shard")
    shard_count = int(payloads[0]["shard_count"])
    if sorted(int(payload["shard_index"]) for payload in payloads) != list(
        range(shard_count)
    ):
        raise ValueError("V10 descriptor replay shard registry is incomplete")
    candidate_rows = [
        int(row) for payload in payloads for row in payload["candidate_anchor_rows"]
    ]
    if len(candidate_rows) != len(set(candidate_rows)) or set(candidate_rows) != set(
        torch.as_tensor(proposal["candidate_anchor_rows"]).tolist()
    ):
        raise ValueError("V10 descriptor replay candidate registry differs")
    records = [record for payload in payloads for record in payload["records"]]
    audit = aggregate_descriptor_action_gain(records)
    authorized = audit["authorized_anchor_rows"]
    proposal_rows = torch.as_tensor(proposal["candidate_anchor_rows"]).long()
    proposal_descriptors = torch.as_tensor(proposal["candidate_descriptors"]).float()
    descriptor_by_row = {
        int(row): proposal_descriptors[index]
        for index, row in enumerate(proposal_rows.tolist())
    }
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    candidate = dict(state)
    features = torch.as_tensor(state["anchor_features"]).clone().float()
    for row in authorized.tolist():
        features[int(row)] = descriptor_by_row[int(row)]
    candidate["anchor_features"] = features
    candidate["provenance"] = {
        **dict(state.get("provenance", {})),
        "v10_actionable_anchor_descriptor_feedback": True,
        "v10_single_action_counterfactual_required": True,
        "v10_updated_anchor_count": int(authorized.numel()),
        "feedback_descriptor_exact_copy": False,
        "feedback_descriptor_robust_aggregate": True,
        "feedback_queries_enter_mapping_csr": False,
        "loo_used": False,
        "uses_test_queries": False,
    }
    args.output_dir.mkdir(parents=True)
    audit_path = args.output_dir / "single_action_audit.pt"
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
        "schema": "lafgs_v10_counterfactual_descriptor_map_report",
        "version": 1,
        "status": "PROPOSAL",
        "loo_used": False,
        "uses_test_queries": False,
        "candidate_count": proposal["candidate_count"],
        "authorized_update_count": int(authorized.numel()),
        "geometry_mutation_count": 0,
        "anchor_addition_count": 0,
        "anchor_deletion_count": 0,
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
