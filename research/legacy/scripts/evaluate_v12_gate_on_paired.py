#!/usr/bin/env python3
"""Evaluate a frozen V12 action gate on paired render-only validation rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.hashing import sha256_file
from features.scene_action_gate import feature_tensor, load_scene_action_gate


def _summary(rows: list[dict]) -> dict:
    te = np.asarray([row["translation_error_cm"] for row in rows])
    ae = np.asarray([row["rotation_error_deg"] for row in rows])
    task = np.asarray([row["task_error"] for row in rows])
    return {
        "median_task_error": float(np.median(task)),
        "p90_task_error": float(np.percentile(task, 90)),
        "median_translation_cm": float(np.median(te)),
        "p90_translation_cm": float(np.percentile(te, 90)),
        "r5_percent": float(100 * np.mean((te < 5) & (ae < 5))),
        "catastrophic_count": int(np.sum((te >= 100) | (ae >= 30))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paired = json.loads(args.paired.read_text())
    gate_state = torch.load(args.gate, map_location="cpu", weights_only=False)
    gate = load_scene_action_gate(
        gate_state, map_sha256=paired["input"]["anchor_map_sha256"],
        detector_sha256=paired["input"]["checkpoint_sha256"],
    )
    baseline, proposal, output_rows = [], [], []
    for row in paired["paired_rows"]:
        logit = float(gate(feature_tensor(row)))
        activate = logit >= 0
        chosen = row["strength_1.0000"] if activate else row["native"]
        baseline.append(row["native"])
        proposal.append(chosen)
        output_rows.append({
            "query_index": row["query_index"], "gate_logit": logit,
            "activated": activate,
            "task_gain": row["native"]["task_error"] - chosen["task_error"],
        })
    base_summary, proposal_summary = _summary(baseline), _summary(proposal)
    gains = np.asarray([row["task_gain"] for row in output_rows])
    changed = np.asarray([row["activated"] for row in output_rows])
    result = {
        "schema": "lafgs_v12_scene_action_gate_validation",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "input": {
            "paired": str(args.paired.resolve()), "paired_sha256": sha256_file(args.paired),
            "gate": str(args.gate.resolve()), "gate_sha256": sha256_file(args.gate),
        },
        "query_count": len(output_rows),
        "activated_query_count": int(changed.sum()),
        "baseline": base_summary,
        "proposal": proposal_summary,
        "paired_median_task_gain": float(np.median(gains)),
        "conditional_median_task_gain": float(np.median(gains[changed])) if bool(changed.any()) else None,
        "conditional_worsening_fraction": float(np.mean(gains[changed] < 0)) if bool(changed.any()) else None,
        "r5_delta_percent": proposal_summary["r5_percent"] - base_summary["r5_percent"],
        "decision": "PROPOSE" if (
            bool(changed.any())
            and float(np.median(gains[changed])) >= 0.001
            and float(np.mean(gains[changed] < 0)) <= 0.25
            and proposal_summary["r5_percent"] >= base_summary["r5_percent"] - 0.01
            and proposal_summary["catastrophic_count"] <= base_summary["catastrophic_count"]
        ) else "ROLLBACK",
        "rows": output_rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
