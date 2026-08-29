#!/usr/bin/env python3
"""Aggregate exact gain-curve replay and apply the frozen V13 supervisor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from map_learning.v13_risk_supervisor import supervise_candidate, validate_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--phase", choices=("control", "confirmation"), required=True)
    parser.add_argument("--selected-arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = [json.loads(path.read_text()) for path in args.shards]
    if any(
        item.get("schema") != "lafgs_v13_metric_gain_curve_shard"
        or item.get("uses_test_queries") is not False
        or item.get("loo_used") is not False
        for item in payloads
    ):
        raise ValueError("invalid V13 gain-curve shard")
    expected = int(payloads[0]["shard_count"])
    if sorted(int(item["shard_index"]) for item in payloads) != list(range(expected)):
        raise ValueError("gain-curve shard registry is incomplete")
    identity = payloads[0]["input"]
    arms = payloads[0]["candidate_arms"]
    if any(item["input"] != identity or item["candidate_arms"] != arms for item in payloads[1:]):
        raise ValueError("gain-curve shards have different frozen inputs")
    records = validate_records(
        [row for item in payloads for row in item["records"]], arms
    )
    decisions = {
        arm: supervise_candidate(records, arm, seed=1320260828 + index)
        for index, arm in enumerate(arms)
    }
    influence = {
        "query_indices": [int(row["query_index"]) for row in records],
        "pose_family_ids": [int(row["pose_family_id"]) for row in records],
        "task_gain_by_action": {
            arm: [
                float(row["baseline"]["task_error"] - row[arm]["task_error"])
                for row in records
            ]
            for arm in arms
        },
    }
    selected = args.selected_arm
    if args.phase == "control":
        if selected is not None:
            parser.error("control phase selects the arm; do not pass --selected-arm")
        eligible = [
            arm
            for arm in arms
            if decisions[arm]["classification"]
            in {"DEFAULT_CANDIDATE", "PARETO_CANDIDATE"}
        ]
        selected = min(
            eligible,
            key=lambda arm: decisions[arm]["candidate"]["total_risk"],
            default=None,
        )
    else:
        if selected not in arms:
            parser.error("confirmation requires the single control-selected arm")
        if payloads[0]["evaluation_role"] != "confirmation_query":
            raise ValueError("confirmation decision requires confirmation_query records")
        decisions = {selected: decisions[selected]}
    output = {
        "schema": "lafgs_v13_risk_supervised_gain_curve",
        "version": 1,
        "phase": args.phase,
        "uses_test_queries": False,
        "confirmation_can_train_or_select": False,
        "accepted_query_count": len(records),
        "input": identity,
        "decisions": decisions,
        "action_query_influence": influence,
        "selected_arm": selected,
        "decision": (
            "NO_ACTION"
            if selected is None
            else (
                "ADVANCE_TO_CONFIRMATION"
                if args.phase == "control"
                else decisions[selected]["classification"]
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    concise = {key: value for key, value in output.items() if key != "action_query_influence"}
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
