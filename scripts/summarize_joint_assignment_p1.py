#!/usr/bin/env python3
"""Apply the frozen five-scene gate to one joint-assignment protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCENES = (
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
)


def _load(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    rows = payload if isinstance(payload, list) else payload["results"]
    indexed = {str(row["image_name"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate query names in {path}")
    return indexed


def _finite(values):
    values = np.asarray(list(values), dtype=np.float64)
    return values[np.isfinite(values)]


def _metrics(rows: list[dict]) -> dict:
    te = np.asarray([row["sparse_TE"] for row in rows], dtype=np.float64)
    ae = np.asarray([row["sparse_AE"] for row in rows], dtype=np.float64)
    sparse = [row["sparse"] for row in rows]
    hypotheses = _finite(
        row.get("sparse_diag_ransac_actual_hypotheses", np.nan) for row in sparse
    )
    hypotheses = hypotheses[hypotheses >= 0.0]
    assignment_ms = _finite(
        row.get("sparse_diag_native_rerank_runtime_ms", np.nan) for row in sparse
    )
    selector_ms = _finite(
        row.get(
            "sparse_diag_joint_assignment_fixed_selector_runtime_ms", np.nan
        )
        for row in sparse
    )
    overhead = (
        np.asarray(
            [
                float(row.get("sparse_diag_native_rerank_runtime_ms", 0.0))
                + float(
                    row.get(
                        "sparse_diag_joint_assignment_fixed_selector_runtime_ms",
                        0.0,
                    )
                )
                for row in sparse
            ],
            dtype=np.float64,
        )
    )
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.quantile(te, 0.9)),
        "r5_percent": float(100.0 * np.mean((te <= 5.0) & (ae <= 5.0))),
        "catastrophic_count": int(np.sum((te > 100.0) | (ae > 10.0))),
        "hypotheses_mean": float(np.mean(hypotheses)) if len(hypotheses) else None,
        "assignment_ms_mean": (
            float(np.mean(assignment_ms)) if len(assignment_ms) else None
        ),
        "selector_ms_mean": float(np.mean(selector_ms)) if len(selector_ms) else None,
        "assignment_selection_ms_mean": float(np.mean(overhead)),
        "assignment_selection_ms_p90": float(np.quantile(overhead, 0.9)),
    }


def summarize(baseline_paths, candidate_paths, protocol):
    scenes = {}
    for scene in SCENES:
        baseline = _load(baseline_paths[scene])
        candidate = _load(candidate_paths[scene])
        if set(baseline) != set(candidate):
            raise ValueError(f"{scene}: baseline/candidate query registries differ")
        names = sorted(baseline)
        base_rows = [baseline[name] for name in names]
        candidate_rows = [candidate[name] for name in names]
        required_candidate_diagnostics = {
            "sparse_diag_native_rerank_runtime_ms",
            "sparse_diag_joint_assignment_fixed_selector_runtime_ms",
            "sparse_diag_ransac_actual_hypotheses",
        }
        for row in candidate_rows:
            missing = required_candidate_diagnostics - set(row["sparse"])
            if missing:
                raise ValueError(
                    f"{scene}: candidate misses diagnostics: {sorted(missing)}"
                )
        base = _metrics(base_rows)
        value = _metrics(candidate_rows)
        base_cat = np.asarray(
            [
                row["sparse_TE"] > 100.0 or row["sparse_AE"] > 10.0
                for row in base_rows
            ],
            dtype=bool,
        )
        candidate_cat = np.asarray(
            [
                row["sparse_TE"] > 100.0 or row["sparse_AE"] > 10.0
                for row in candidate_rows
            ],
            dtype=bool,
        )
        strict_non_worse = bool(
            value["median_te_cm"] <= base["median_te_cm"]
            and value["p90_te_cm"] <= base["p90_te_cm"]
            and value["r5_percent"] >= base["r5_percent"]
        )
        scenes[scene] = {
            "baseline": base,
            "candidate": value,
            "delta_candidate_minus_baseline": {
                key: value[key] - base[key]
                for key in ("median_te_cm", "mean_te_cm", "p90_te_cm", "r5_percent")
            },
            "strict_median_p90_r5_non_worse": strict_non_worse,
            "catastrophic_introduced_count": int((~base_cat & candidate_cat).sum()),
            "catastrophic_rescued_count": int((base_cat & ~candidate_cat).sum()),
        }

    non_worse_count = sum(
        row["strict_median_p90_r5_non_worse"] for row in scenes.values()
    )
    catastrophic_introduced = sum(
        row["catastrophic_introduced_count"] for row in scenes.values()
    )
    base_mean = np.mean([row["baseline"]["mean_te_cm"] for row in scenes.values()])
    value_mean = np.mean([row["candidate"]["mean_te_cm"] for row in scenes.values()])
    base_p90 = np.mean([row["baseline"]["p90_te_cm"] for row in scenes.values()])
    value_p90 = np.mean([row["candidate"]["p90_te_cm"] for row in scenes.values()])
    base_hypotheses = sum(
        row["baseline"]["hypotheses_mean"] for row in scenes.values()
    )
    value_hypotheses = sum(
        row["candidate"]["hypotheses_mean"] for row in scenes.values()
    )
    hypothesis_reduction = 1.0 - value_hypotheses / max(base_hypotheses, 1e-12)
    maximum_overhead_p90 = max(
        row["candidate"]["assignment_selection_ms_p90"] for row in scenes.values()
    )
    checks = {
        "median_p90_r5_non_worse_at_least_4_of_5": bool(non_worse_count >= 4),
        "no_new_catastrophic_query_in_5_of_5": bool(catastrophic_introduced == 0),
        "macro_mean_or_p90_improves": bool(
            value_mean < base_mean or value_p90 < base_p90
        ),
        "hypotheses_reduce_at_least_50_percent": bool(
            hypothesis_reduction >= 0.5
        ),
        "assignment_selection_p90_below_10ms_in_5_of_5": bool(
            maximum_overhead_p90 < 10.0
        ),
    }
    return {
        "schema": "lafgs_joint_assignment_p1_gate_v1",
        "protocol": protocol,
        "status": "PASS" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "non_worse_scene_count": non_worse_count,
        "catastrophic_introduced_count": catastrophic_introduced,
        "macro": {
            "baseline_mean_te_cm": float(base_mean),
            "candidate_mean_te_cm": float(value_mean),
            "baseline_p90_te_cm": float(base_p90),
            "candidate_p90_te_cm": float(value_p90),
            "hypothesis_reduction_fraction": float(hypothesis_reduction),
            "maximum_assignment_selection_p90_ms": float(maximum_overhead_p90),
        },
        "scenes": scenes,
        "recommendation": (
            "RETAIN_JOINT_ASSIGNMENT"
            if all(checks.values())
            else "CLOSE_SELECTOR_BRANCH"
        ),
    }


def _parse(values):
    parsed = {}
    for value in values:
        scene, separator, path = value.partition("=")
        if not separator or scene not in SCENES:
            raise ValueError(f"invalid scene path: {value}")
        parsed[scene] = Path(path).resolve()
    missing = sorted(set(SCENES) - set(parsed))
    if missing:
        raise ValueError("missing scenes: " + ", ".join(missing))
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(
        _parse(args.baseline), _parse(args.candidate), args.protocol
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
