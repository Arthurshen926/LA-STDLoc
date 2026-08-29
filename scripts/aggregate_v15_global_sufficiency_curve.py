#!/usr/bin/env python3
"""Select the smallest safe global-map operating point from held-out replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.hashing import sha256_file
from map_learning.v14_task_supervisor import supervise
from map_learning.v15_global_sufficiency import size_aware_supervision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--arm-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--phase", choices=("control", "confirmation"), default="control")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = [json.loads(path.read_text()) for path in args.shards]
    if any(
        item.get("schema") != "lafgs_v14_active_set_curve_shard"
        or item.get("uses_test_queries") is not False
        or item.get("loo_used") is not False
        for item in payloads
    ):
        raise ValueError("V15 requires exact no-test V14-compatible plant replay")
    shard_count = int(payloads[0]["shard_count"])
    if sorted(int(item["shard_index"]) for item in payloads) != list(range(shard_count)):
        raise ValueError("V15 replay shard registry is incomplete")
    arms = payloads[0]["candidate_arms"]
    identity = payloads[0]["input"]
    if any(item["candidate_arms"] != arms or item["input"] != identity for item in payloads):
        raise ValueError("V15 replay shards do not share frozen candidates")
    records = [row for payload in payloads for row in payload["records"]]
    if len({int(row["query_index"]) for row in records}) != len(records):
        raise ValueError("V15 replay shards overlap")

    reports = {item["arm"]: item for item in (
        json.loads(path.read_text()) for path in args.arm_reports
    )}
    if set(reports) != set(arms):
        raise ValueError("V15 arm reports and replay candidates differ")
    for arm, report in reports.items():
        replay_map = identity["candidates"][arm]
        if (
            report["map"] != replay_map["path"]
            or report["map_sha256"] != replay_map["sha256"]
            or sha256_file(Path(report["map"])) != report["map_sha256"]
        ):
            raise ValueError("V15 replay map differs from its sufficiency report")

    decisions = {}
    for index, arm in enumerate(arms):
        report = reports[arm]
        compression = float(report["compression_fraction"])
        if not 0.0 <= compression < 1.0:
            raise ValueError("V15 sufficiency report has an invalid compression fraction")
        decisions[arm] = size_aware_supervision(
            supervise(records, arm, seed=1420260828 + index),
            compression_fraction=compression,
        )
    eligible = [
        arm
        for arm in arms
        if reports[arm]["budget_feasible"]
        and decisions[arm]["classification"]
        in {"DEFAULT_CANDIDATE", "PARETO_CANDIDATE"}
    ]
    # Map size is the controlled resource.  Once task safety and a measurable
    # effect are established, choose the smallest state, then lower task risk.
    selected = min(
        eligible,
        key=lambda arm: (
            int(reports[arm]["selected_anchor_count"]),
            decisions[arm]["candidate"]["total_risk"],
        ),
        default=None,
    )

    paired_feedback = {}
    for profile in sorted({report["profile"] for report in reports.values()}):
        mapping = f"{profile}_mapping_only"
        feedback = f"{profile}_feedback_conditioned"
        if mapping not in reports or feedback not in reports:
            continue
        transformed = []
        for row in records:
            transformed.append(
                {
                    "query_index": row["query_index"],
                    "pose_family_id": row["pose_family_id"],
                    "baseline": row[mapping],
                    "feedback": row[feedback],
                }
            )
        paired_feedback[profile] = supervise(
            transformed, "feedback", seed=1420261828 + len(paired_feedback)
        )

    output = {
        "schema": "lafgs_v15_global_sufficiency_control_decision",
        "version": 1,
        "uses_test_queries": False,
        "phase": args.phase,
        "accepted_query_count": len(records),
        "selection_objective": (
            "minimum_map_size_subject_to_hard_task_safety_lower_risk_and_"
            "substantial_compression"
        ),
        "candidate_arms": arms,
        "arm_reports": reports,
        "decisions": decisions,
        "same_budget_feedback_effect": paired_feedback,
        "selected_arm": selected,
        "decision": (
            (
                decisions[selected]["classification"].replace(
                    "CANDIDATE", "CONFIRMED"
                )
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
