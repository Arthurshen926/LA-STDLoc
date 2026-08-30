#!/usr/bin/env python3
"""Select intervention/necessity/global views for a frozen V18 action."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.v18_action_targeted_planner import select_action_targeted_queries
from map_learning.v18_provenance_truth import TRUTH_EQUIVALENT, TRUTH_UNIQUE


def _save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--mapping-provenance", type=Path, required=True)
    parser.add_argument("--responsibility", type=Path, required=True)
    parser.add_argument("--controller-audit", type=Path, required=True)
    parser.add_argument("--maximum-queries", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate = torch.load(args.candidate_plan, map_location="cpu", weights_only=False)
    baseline = torch.load(args.baseline_map, map_location="cpu", weights_only=False)
    mapping = torch.load(
        args.mapping_provenance, map_location="cpu", weights_only=False
    )
    responsibility = torch.load(
        args.responsibility, map_location="cpu", weights_only=False
    )
    controller = torch.load(
        args.controller_audit, map_location="cpu", weights_only=False
    )
    if not (
        candidate.get("uses_test_queries") is False
        and candidate.get("loo_used") is False
        and mapping.get("schema")
        == "lafgs_v18_mapping_observation_gaussian_provenance"
        and responsibility.get("schema")
        == "lafgs_v18_operation_responsibility_batch"
        and controller.get("schema") == "lafgs_v18_truth_aware_controller_audit"
        and controller.get("uses_test_queries") is False
        and controller.get("loo_used") is False
    ):
        raise ValueError("V18 targeted planner input contract differs")
    harmful = sorted(
        {
            int(action["anchor_row"])
            for action in controller["actions"]
            if action.get("accepted")
            and action.get("kind") in {"harmful_removal", "dominated_removal"}
        }
    )
    reactivated = sorted(
        {
            int(action["anchor_row"])
            for action in controller["actions"]
            if action.get("accepted")
            and action.get("kind") == "truth_reactivation"
        }
    )
    backups: dict[int, set[int]] = {anchor: set() for anchor in harmful}
    for record in responsibility["records"]:
        candidates = torch.as_tensor(record["candidate_anchor_rows"]).long()
        truth = record["truth"]
        status = torch.as_tensor(truth["truth_status"]).long()
        offsets = torch.as_tensor(truth["truth_offsets"]).long()
        truth_rows = torch.as_tensor(truth["truth_anchor_rows"]).long()
        decisive = (status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT)
        for row in torch.nonzero(decisive, as_tuple=False).reshape(-1).tolist():
            winner = int(candidates[row, 0])
            if winner not in backups:
                continue
            start, stop = int(offsets[row]), int(offsets[row + 1])
            backups[winner].update(
                int(anchor)
                for anchor in truth_rows[start:stop].tolist()
                if int(anchor) != winner
            )
    # A deletion without a known positive backup cannot produce a valid
    # intervention view; retain it for necessity scoring with an empty edge.
    offsets = [0]
    backup_rows = []
    for anchor in harmful:
        backup_rows.extend(sorted(backups[anchor]))
        offsets.append(len(backup_rows))
    observations = baseline["projective_anchor_observations"]
    plan = select_action_targeted_queries(
        candidate_plan=candidate,
        anchor_xyz=baseline["anchor_xyz"],
        harmful_anchor_rows=torch.tensor(harmful, dtype=torch.long),
        reactivated_anchor_rows=torch.tensor(reactivated, dtype=torch.long),
        backup_offsets=torch.tensor(offsets, dtype=torch.long),
        backup_anchor_rows=torch.tensor(backup_rows, dtype=torch.long),
        anchor_observation_offsets=observations["observation_offsets"],
        observation_query_indices=observations["query_indices"],
        mapping_poses_w2c=mapping["mapping_poses_w2c"],
        maximum_queries=int(args.maximum_queries),
    )
    plan["target_action_anchor_rows"] = torch.tensor(harmful, dtype=torch.long)
    plan["target_reactivated_anchor_rows"] = torch.tensor(
        reactivated, dtype=torch.long
    )
    plan["target_backup_offsets"] = torch.tensor(offsets, dtype=torch.long)
    plan["target_backup_anchor_rows"] = torch.tensor(backup_rows, dtype=torch.long)
    plan["inputs"] = {
        "candidate_plan": str(args.candidate_plan.resolve()),
        "candidate_plan_sha256": sha256_file(args.candidate_plan),
        "baseline_map": str(args.baseline_map.resolve()),
        "baseline_map_sha256": sha256_file(args.baseline_map),
        "mapping_provenance": str(args.mapping_provenance.resolve()),
        "mapping_provenance_sha256": sha256_file(args.mapping_provenance),
        "responsibility": str(args.responsibility.resolve()),
        "responsibility_sha256": sha256_file(args.responsibility),
        "controller_audit": str(args.controller_audit.resolve()),
        "controller_audit_sha256": sha256_file(args.controller_audit),
    }
    _save(plan, args.output.resolve())
    report = {
        **plan["action_targeted_planner"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "harmful_anchor_rows": harmful,
        "reactivated_anchor_rows": reactivated,
        "backup_anchor_count": len(backup_rows),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
