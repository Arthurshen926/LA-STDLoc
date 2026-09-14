#!/usr/bin/env python3
"""Aggregate sealed V9 paired-confirmation shards and apply the fixed gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _summary(records: list[dict], key: str) -> dict:
    translation = np.asarray([row[key]["translation_error_cm"] for row in records])
    rotation = np.asarray([row[key]["rotation_error_deg"] for row in records])
    task = np.asarray([row[key]["task_error"] for row in records])
    return {
        "median_translation_cm": float(np.median(translation)),
        "p90_translation_cm": float(np.percentile(translation, 90)),
        "median_rotation_deg": float(np.median(rotation)),
        "p90_rotation_deg": float(np.percentile(rotation, 90)),
        "median_task_error": float(np.median(task)),
        "r5_percent": float(100.0 * np.mean((translation < 5) & (rotation < 5))),
        "catastrophic_count": int(np.sum((translation >= 100) | (rotation >= 30))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--sparse-action", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payloads = [json.loads(path.read_text()) for path in args.shards]
    if any(
        payload.get("schema") != "lafgs_v9_paired_confirmation_shard"
        or payload.get("loo_used") is not False
        or payload.get("uses_test_queries") is not False
        for payload in payloads
    ):
        raise ValueError("invalid V9 paired-confirmation shard")
    identities = [payload["input"] for payload in payloads]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("paired-confirmation shard inputs differ")
    expected_shards = int(payloads[0]["shard_count"])
    if sorted(int(payload["shard_index"]) for payload in payloads) != list(
        range(expected_shards)
    ):
        raise ValueError("paired-confirmation shard registry is incomplete")
    records = [record for payload in payloads for record in payload["records"]]
    if len({int(row["query_index"]) for row in records}) != len(records):
        raise ValueError("paired-confirmation queries overlap")
    baseline = _summary(records, "baseline")
    proposal = _summary(records, "proposal")
    gains = np.asarray([float(row["task_gain"]) for row in records])
    median_gain = float(np.median(gains))
    p90_regression = float(np.percentile(-gains, 90))
    r5_delta = proposal["r5_percent"] - baseline["r5_percent"]
    changed = np.abs(gains) > 1e-12
    changed_gains = gains[changed]
    changed_families = {
        int(records[index]["pose_family_id"]) for index in np.flatnonzero(changed)
    }
    sparse = {
        "changed_query_count": int(changed.sum()),
        "changed_pose_family_count": len(changed_families),
        "conditional_median_task_gain": (
            math.nan if not bool(changed.any()) else float(np.median(changed_gains))
        ),
        "conditional_cumulative_task_gain": float(changed_gains.sum()),
        "conditional_worsening_fraction": (
            math.nan if not bool(changed.any()) else float(np.mean(changed_gains < 0))
        ),
    }
    if args.sparse_action:
        accepted = bool(
            sparse["changed_query_count"] >= 8
            and sparse["changed_pose_family_count"] >= 8
            and sparse["conditional_median_task_gain"] >= 0.001
            and sparse["conditional_cumulative_task_gain"] > 0
            and sparse["conditional_worsening_fraction"] <= 0.25
            and proposal["median_task_error"] <= baseline["median_task_error"]
            and p90_regression <= 0.20
            and r5_delta >= -0.01
            and proposal["catastrophic_count"] <= baseline["catastrophic_count"]
        )
    else:
        accepted = bool(
            math.isfinite(median_gain)
            and median_gain >= 0.001
            and p90_regression <= 0.20
            and r5_delta >= -0.01
            and proposal["catastrophic_count"] <= baseline["catastrophic_count"]
        )
    output = {
        "schema": "lafgs_v9_paired_confirmation_decision",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "accepted_query_count": len(records),
        "baseline": baseline,
        "proposal": proposal,
        "paired": {
            "median_task_gain": median_gain,
            "p90_task_regression": p90_regression,
            "r5_delta_percent": r5_delta,
            "improving_fraction": float(np.mean(gains > 0)),
        },
        "sparse_action_diagnostic": sparse,
        "decision": "ACCEPT" if accepted else "ROLLBACK",
        "gate": (
            {
                "type": "sparse_changed_query_and_global_safety",
                "minimum_changed_queries": 8,
                "minimum_changed_pose_families": 8,
                "minimum_conditional_median_task_gain": 0.001,
                "positive_conditional_cumulative_task_gain": True,
                "maximum_conditional_worsening_fraction": 0.25,
                "global_median_task_non_regression": True,
                "maximum_p90_task_regression": 0.20,
                "minimum_r5_delta_percent": -0.01,
                "no_catastrophic_increase": True,
            }
            if args.sparse_action
            else {
                "type": "global_paired",
                "minimum_median_task_gain": 0.001,
                "maximum_p90_task_regression": 0.20,
                "minimum_r5_delta_percent": -0.01,
                "no_catastrophic_increase": True,
            }
        ),
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
