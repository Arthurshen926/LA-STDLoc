#!/usr/bin/env python3
"""Freeze V8 feedback action decisions from paired non-test evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.hashing import sha256_file


def _task_metrics(result: dict) -> dict:
    rows = result["rows"]
    translation = np.asarray([row["translation_error_cm"] for row in rows])
    rotation = np.asarray([row["rotation_error_deg"] for row in rows])
    task = np.hypot(translation / 5.0, rotation / 5.0)
    return {
        "median_task_error": float(np.median(task)),
        "p90_task_error": float(np.quantile(task, 0.9)),
        **result["metrics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--real-baseline", type=Path, required=True)
    parser.add_argument("--real-proposals", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--baseline-map-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.real_proposals) != len(args.names):
        raise ValueError("real proposal registry differs")
    if args.output.exists():
        raise FileExistsError(args.output)
    control = json.loads(args.control.read_text())
    confirmation = json.loads(args.confirmation.read_text())
    real_baseline = json.loads(args.real_baseline.read_text())
    real = {
        name: json.loads(path.read_text())
        for name, path in zip(args.names, args.real_proposals)
    }
    baseline_control = _task_metrics(control["results"]["baseline"])
    baseline_confirmation = _task_metrics(confirmation["results"]["baseline"])
    actions = {}
    for name in args.names:
        control_metrics = _task_metrics(control["results"][name])
        confirmation_metrics = _task_metrics(confirmation["results"][name])
        gates = {
            "control_median_task_improvement": (
                baseline_control["median_task_error"]
                - control_metrics["median_task_error"]
                >= 0.001
            ),
            "fresh_confirmation_median_task_improvement": (
                baseline_confirmation["median_task_error"]
                - confirmation_metrics["median_task_error"]
                >= 0.001
            ),
            "fresh_confirmation_p90_nonregression": (
                confirmation_metrics["p90_task_error"]
                <= baseline_confirmation["p90_task_error"] + 0.02
            ),
            "fresh_confirmation_r5_nonregression": (
                confirmation_metrics["recall_5cm_5deg_percent"]
                >= baseline_confirmation["recall_5cm_5deg_percent"] - 0.01
            ),
            "fresh_confirmation_catastrophic_nonincrease": (
                confirmation_metrics["catastrophic_50cm_count"]
                <= baseline_confirmation["catastrophic_50cm_count"]
            ),
        }
        accepted = all(gates.values())
        actions[name] = {
            "decision": "ACCEPT" if accepted else "ROLLBACK",
            "gates": gates,
            "control": control_metrics,
            "fresh_confirmation": confirmation_metrics,
            "paired_control": control["results"][name]["paired_vs_baseline"],
            "paired_fresh_confirmation": confirmation["results"][name][
                "paired_vs_baseline"
            ],
            "real_mapping_rgb_evaluation_only": real[name]["metrics"],
        }
    payload = {
        "schema": "lafgs_v8_map_feedback_decision",
        "version": 1,
        "status": "PASS",
        "uses_test_queries": False,
        "threshold_tuning_from_results": False,
        "baseline_map_sha256": args.baseline_map_sha256,
        "chosen_map_sha256": args.baseline_map_sha256,
        "all_actions_rolled_back": all(
            item["decision"] == "ROLLBACK" for item in actions.values()
        ),
        "baseline": {
            "control": baseline_control,
            "fresh_confirmation": baseline_confirmation,
            "real_mapping_rgb_evaluation_only": real_baseline["metrics"],
        },
        "actions": actions,
        "inputs": {
            "control": {"path": str(args.control), "sha256": sha256_file(args.control)},
            "confirmation": {
                "path": str(args.confirmation),
                "sha256": sha256_file(args.confirmation),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
