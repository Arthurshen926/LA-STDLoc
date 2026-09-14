#!/usr/bin/env python3
"""Pool two frozen V17 confirmations with source-parent block bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from map_learning.v13_risk_supervisor import supervise_candidate, validate_records


def _load_group(paths: list[Path], arm: str, batch_index: int) -> tuple[list[dict], dict]:
    payloads = [json.loads(path.read_text()) for path in paths]
    expected = int(payloads[0]["shard_count"])
    if sorted(int(item["shard_index"]) for item in payloads) != list(range(expected)):
        raise ValueError("repeated-confirmation shard group is incomplete")
    identity = payloads[0]["input"]
    if any(
        item.get("schema") != "lafgs_v13_metric_gain_curve_shard"
        or item.get("evaluation_role") != "confirmation_query"
        or arm not in item.get("candidate_arms", ())
        or item.get("input") != identity
        for item in payloads
    ):
        raise ValueError("repeated-confirmation shard contract differs")
    records = validate_records(
        [row for item in payloads for row in item["records"]], [arm]
    )
    tagged = []
    for row in records:
        tagged.append(
            {
                **row,
                "source_query_index": int(row["query_index"]),
                "confirmation_batch_index": int(batch_index),
                "query_index": int(batch_index) * 1_000_000
                + int(row["query_index"]),
            }
        )
    return tagged, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--second-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--selected-arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    first, first_input = _load_group(args.first_shards, args.selected_arm, 0)
    second, second_input = _load_group(args.second_shards, args.selected_arm, 1)
    if (
        first_input["baseline_map_sha256"] != second_input["baseline_map_sha256"]
        or first_input["metric_sha256"] != second_input["metric_sha256"]
        or first_input["active_set_map_sha256"]
        != second_input["active_set_map_sha256"]
    ):
        raise ValueError("repeated confirmations do not evaluate one frozen action")
    records = first + second
    decision = supervise_candidate(
        records, args.selected_arm, seed=1720260832
    )
    output = {
        "schema": "lafgs_v17_repeated_confirmation_decision",
        "version": 1,
        "uses_test_queries": False,
        "confirmation_can_train_or_select": False,
        "selected_arm": args.selected_arm,
        "accepted_query_count": len(records),
        "confirmation_batch_count": 2,
        "bootstrap_block_unit": "source_mapping_parent_across_batches",
        "inputs": [first_input, second_input],
        "decision_detail": decision,
        "decision": decision["classification"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
