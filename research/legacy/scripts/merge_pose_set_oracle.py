#!/usr/bin/env python3
"""Merge deterministic mapping-query shards from pose-set oracle audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _summary(rows: list[dict], prefix: str) -> dict:
    te = np.asarray([row[f"{prefix}_te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row[f"{prefix}_ae_deg"] for row in rows], dtype=np.float64)
    risk = np.asarray([row[f"{prefix}_risk"] for row in rows], dtype=np.float64)
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "p90_ae_deg": float(np.percentile(ae, 90)),
        "median_risk": float(np.median(risk)),
        "mean_risk": float(np.mean(risk)),
        "p90_risk": float(np.percentile(risk, 90)),
        "failure_count": int(sum(row[f"{prefix}_failed"] for row in rows)),
        "mean_hypotheses": float(
            np.mean([row[f"{prefix}_hypotheses"] for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    if any(payload.get("uses_test_queries") for payload in payloads):
        raise ValueError("pose-set oracle merge accepts mapping-only shards")
    shared = {
        key: payloads[0][key]
        for key in (
            "map",
            "metric_state",
            "deployment_row_limit",
            "retrieval_topk",
            "maximum_actions",
            "joint_depth",
            "beam_width",
            "high_cost_hypotheses",
        )
    }
    for payload in payloads[1:]:
        for key, value in shared.items():
            if payload[key] != value:
                raise ValueError(f"oracle shard configuration differs for {key}")
    rows = sorted(
        [row for payload in payloads for row in payload["queries"]],
        key=lambda row: int(row["query_index"]),
    )
    if len({int(row["query_index"]) for row in rows}) != len(rows):
        raise ValueError("oracle shards overlap")
    report = {
        "schema": "lafgs_pose_set_oracle_audit",
        "version": 1,
        "uses_test_queries": False,
        **shared,
        "query_count": len(rows),
        "query_selection": "uniform_mapping_only_sharded",
        "source_shards": [str(path.resolve()) for path in args.inputs],
        "summaries": {
            label: _summary(rows, label)
            for label in ("current", "topk", "positive", "single", "joint")
        },
        "headroom": {
            f"{label}_risk_gain_mean": float(
                np.mean([row["current_risk"] - row[f"{label}_risk"] for row in rows])
            )
            for label in ("topk", "positive", "single", "joint")
        },
        "queries": rows,
    }
    report["headroom"].update(
        {
            "single_stable_positive_query_fraction": float(
                np.mean([row["single_stability"]["risk_gain_median"] > 0 for row in rows])
            ),
            "joint_stable_positive_query_fraction": float(
                np.mean([row["joint_stability"]["risk_gain_median"] > 0 for row in rows])
            ),
            "recoverable_false_top1_fraction": float(
                sum(row["recoverable_false_top1_count"] for row in rows)
                / max(sum(row["false_top1_count"] for row in rows), 1)
            ),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["headroom"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
