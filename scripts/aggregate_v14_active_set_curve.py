#!/usr/bin/env python3
"""Select a simple active-set action from exact held-out plant replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from map_learning.v14_task_supervisor import supervise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--phase", choices=("control", "confirmation"), default="control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = [json.loads(path.read_text()) for path in args.shards]
    schemas = {item.get("schema") for item in payloads}
    if len(schemas) != 1 or next(iter(schemas)) not in {
        "lafgs_v14_active_set_curve_shard",
        "lafgs_v9_paired_confirmation_shard",
        "lafgs_v13_metric_gain_curve_shard",
    } or any(
        item.get("uses_test_queries") is not False
        or item.get("loo_used") is not False
        for item in payloads
    ):
        raise ValueError("invalid V14 active-set replay shard")
    expected = int(payloads[0]["shard_count"])
    if sorted(int(item["shard_index"]) for item in payloads) != list(range(expected)):
        raise ValueError("active-set replay shards are incomplete")
    curve_schema = next(iter(schemas))
    arms = (
        payloads[0]["candidate_arms"]
        if curve_schema
        in {"lafgs_v14_active_set_curve_shard", "lafgs_v13_metric_gain_curve_shard"}
        else ["proposal"]
    )
    identity = payloads[0]["input"]
    if any(
        (item.get("candidate_arms", ["proposal"]) != arms)
        or item["input"] != identity
        for item in payloads[1:]
    ):
        raise ValueError("active-set shards do not share frozen inputs")
    records = [row for payload in payloads for row in payload["records"]]
    if len({int(row["query_index"]) for row in records}) != len(records):
        raise ValueError("active-set shards overlap")
    decisions = {
        arm: supervise(records, arm, seed=1420260828 + index)
        for index, arm in enumerate(arms)
    }
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
    output = {
        "schema": "lafgs_v14_active_set_control_decision",
        "version": 1,
        "uses_test_queries": False,
        "accepted_query_count": len(records),
        "phase": args.phase,
        "candidate_arms": arms,
        "decisions": decisions,
        "selected_arm": selected,
        "decision": (
            (
                decisions[selected]["classification"].replace("CANDIDATE", "CONFIRMED")
                if args.phase == "confirmation"
                else "ADVANCE_TO_CONFIRMATION"
            )
            if selected
            else ("NOT_CONFIRMED" if args.phase == "confirmation" else "NO_ACTION")
        ),
        "input": identity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
