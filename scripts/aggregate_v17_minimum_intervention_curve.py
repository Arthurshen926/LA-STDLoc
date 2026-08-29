#!/usr/bin/env python3
"""Supervise V17 gain curves with minimum-effective-intervention control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from map_learning.v13_risk_supervisor import supervise_candidate, validate_records
from map_learning.v17_competitive_metric import select_minimum_effective_gain


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
        raise ValueError("invalid V17 gain-curve shard")
    expected = int(payloads[0]["shard_count"])
    if sorted(int(item["shard_index"]) for item in payloads) != list(range(expected)):
        raise ValueError("V17 gain-curve shard registry is incomplete")
    identity = payloads[0]["input"]
    arms = payloads[0]["candidate_arms"]
    if any(
        item["input"] != identity or item["candidate_arms"] != arms
        for item in payloads[1:]
    ):
        raise ValueError("V17 gain-curve shards have different frozen inputs")
    records = validate_records(
        [row for item in payloads for row in item["records"]], arms
    )
    decisions = {
        arm: supervise_candidate(records, arm, seed=1720260829 + index)
        for index, arm in enumerate(arms)
    }
    selected = args.selected_arm
    if args.phase == "control":
        if selected is not None:
            parser.error("control phase selects the arm")
        selected = select_minimum_effective_gain(decisions)
    else:
        if selected not in arms:
            parser.error("confirmation requires the control-selected arm")
        if payloads[0]["evaluation_role"] != "confirmation_query":
            raise ValueError("confirmation requires confirmation_query records")
        decisions = {selected: decisions[selected]}
    output = {
        "schema": "lafgs_v17_minimum_intervention_gain_curve",
        "version": 1,
        "phase": args.phase,
        "uses_test_queries": False,
        "confirmation_can_train_or_select": False,
        "selection_policy": (
            "minimum_gain_default_then_pareto_after_safety_positive_net_and_r5"
        ),
        "accepted_query_count": len(records),
        "input": identity,
        "decisions": decisions,
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
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
