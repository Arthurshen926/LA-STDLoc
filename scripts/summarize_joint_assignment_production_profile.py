#!/usr/bin/env python3
"""Summarize one full-test, diagnostics-free joint-assignment profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RUNTIME_KEYS = {
    "frontend": "sparse_diag_runtime_frontend_ms",
    "matching": "sparse_diag_runtime_matching_ms",
    "ransac": "sparse_diag_runtime_ransac_ms",
    "total": "sparse_diag_runtime_total_ms",
}


def _load(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text())
    indexed = {str(row["image_name"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate query names in {path}")
    return indexed


def _summary(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("production runtime metrics must be finite and non-empty")
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
    }


def summarize(scene: str, selector: str, baseline_path: Path, candidate_path: Path) -> dict:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("production baseline/candidate query registries differ")
    names = sorted(baseline)
    stages = {}
    paired_delta = {}
    for label, key in RUNTIME_KEYS.items():
        base = np.asarray([baseline[name]["sparse"][key] for name in names])
        value = np.asarray([candidate[name]["sparse"][key] for name in names])
        stages[label] = {
            "baseline_ms": _summary(base),
            "candidate_ms": _summary(value),
        }
        paired_delta[label] = _summary(value - base)

    explicit = np.asarray(
        [
            float(candidate[name]["sparse"]["sparse_diag_native_rerank_runtime_ms"])
            + float(
                candidate[name]["sparse"].get(
                    "sparse_diag_joint_assignment_fixed_selector_runtime_ms", 0.0
                )
            )
            for name in names
        ],
        dtype=np.float64,
    )
    baseline_hypotheses = np.asarray(
        [baseline[name]["sparse"]["sparse_diag_ransac_actual_hypotheses"] for name in names],
        dtype=np.float64,
    )
    candidate_hypotheses = np.asarray(
        [candidate[name]["sparse"]["sparse_diag_ransac_actual_hypotheses"] for name in names],
        dtype=np.float64,
    )
    available = (baseline_hypotheses >= 0) & (candidate_hypotheses >= 0)
    if not available.any():
        raise ValueError("production profile has no actual RANSAC hypothesis counts")
    reduction = 1.0 - float(candidate_hypotheses[available].mean()) / max(
        float(baseline_hypotheses[available].mean()), 1e-12
    )
    baseline_te = np.asarray([baseline[name]["sparse_TE"] for name in names])
    candidate_te = np.asarray([candidate[name]["sparse_TE"] for name in names])
    report = {
        "schema": "lafgs_joint_assignment_production_profile_v1",
        "scene": scene,
        "selector": selector,
        "query_count": len(names),
        "diagnostic_retrieval": False,
        "stages": stages,
        "paired_candidate_minus_baseline_ms": paired_delta,
        "explicit_assignment_selection_ms": _summary(explicit),
        "hypotheses": {
            "baseline_mean": float(baseline_hypotheses[available].mean()),
            "candidate_mean": float(candidate_hypotheses[available].mean()),
            "reduction_fraction": reduction,
        },
        "accuracy_replay": {
            "baseline_median_te_cm": float(np.median(baseline_te)),
            "candidate_median_te_cm": float(np.median(candidate_te)),
            "baseline_p90_te_cm": float(np.quantile(baseline_te, 0.9)),
            "candidate_p90_te_cm": float(np.quantile(candidate_te, 0.9)),
        },
    }
    report["checks"] = {
        "explicit_assignment_selection_p90_below_10ms": report[
            "explicit_assignment_selection_ms"
        ]["p90"] < 10.0,
        "paired_matching_overhead_p90_below_10ms": paired_delta["matching"][
            "p90"
        ] < 10.0,
        "hypotheses_reduce_at_least_50_percent": reduction >= 0.5,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(
        args.scene, args.selector, Path(args.baseline), Path(args.candidate)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
