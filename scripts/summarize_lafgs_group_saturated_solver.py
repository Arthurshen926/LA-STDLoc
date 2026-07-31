#!/usr/bin/env python
"""Apply the formal cross-scene gate to group-saturated pose solving."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _pose_summary(rows, te_key, ae_key):
    te = np.asarray([row[te_key] for row in rows], dtype=np.float64)
    ae = np.asarray([row[ae_key] for row in rows], dtype=np.float64)
    return {
        "query_count": int(te.size),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "recall_5cm_5deg_percent": float(
            100.0 * np.mean((te <= 5.0) & (ae <= 5.0))
        ),
    }


def _non_regressing(candidate, baseline, tolerance=1e-9):
    lower_is_better = ("median_te_cm", "mean_te_cm", "p90_te_cm")
    higher_is_better = ("recall_5cm_5deg_percent",)
    no_regression = all(
        candidate[key] <= baseline[key] + tolerance
        for key in lower_is_better
    ) and all(
        candidate[key] + tolerance >= baseline[key]
        for key in higher_is_better
    )
    strict_improvement = any(
        candidate[key] < baseline[key] - tolerance
        for key in lower_is_better
    ) or any(
        candidate[key] > baseline[key] + tolerance
        for key in higher_is_better
    )
    return bool(no_regression and strict_improvement)


def summarize(reports):
    solver_versions = sorted(
        {str(report.get("solver_version", "unknown")) for report in reports}
    )
    scenes = {}
    for report in reports:
        primary_seed = str(report["primary_seed_comparison"]["seed"])
        primary_rows = [
            row for row in report["queries"] if str(row["seed"]) == primary_seed
        ]
        baseline = _pose_summary(
            primary_rows, "baseline_te_cm", "baseline_ae_deg"
        )
        candidate = report["summary_by_seed"][primary_seed]["all"]
        delta = {
            key: float(candidate[key]) - float(baseline[key])
            for key in (
                "median_te_cm",
                "mean_te_cm",
                "p90_te_cm",
                "recall_5cm_5deg_percent",
            )
        }
        comparison = report["primary_seed_comparison"]
        scenes[report["scene"]] = {
            "primary_seed": int(primary_seed),
            "baseline": baseline,
            "group_saturated": candidate,
            "delta": delta,
            "paired_lower_te_count": int(comparison["lower_te_count"]),
            "paired_higher_te_count": int(comparison["higher_te_count"]),
            "catastrophic_recovered_count": int(
                comparison["catastrophic_recovered_count"]
            ),
            "catastrophic_regressed_count": int(
                comparison["catastrophic_regressed_count"]
            ),
            "pareto_improved": _non_regressing(candidate, baseline),
        }

    improved = [
        scene for scene, row in scenes.items() if row["pareto_improved"]
    ]
    great = scenes.get("GreatCourt")
    greatcourt_non_regression = bool(
        great
        and great["group_saturated"]["median_te_cm"]
        <= great["baseline"]["median_te_cm"] + 1e-9
        and great["group_saturated"]["p90_te_cm"]
        <= great["baseline"]["p90_te_cm"] + 1e-9
        and great["group_saturated"]["recall_5cm_5deg_percent"] + 1e-9
        >= great["baseline"]["recall_5cm_5deg_percent"]
        and great["catastrophic_regressed_count"] == 0
    )
    gate_pass = len(improved) >= 3 and greatcourt_non_regression
    return {
        "schema": "lafgs_group_saturated_formal_cross_scene_gate_v1",
        "scene_count": len(scenes),
        "required_improved_scenes": 3,
        "requires_greatcourt_non_regression": True,
        "solver_versions": solver_versions,
        "solver_version_consistent": len(solver_versions) == 1,
        "improved_scenes": improved,
        "improved_scene_count": len(improved),
        "greatcourt_non_regression": greatcourt_non_regression,
        "gate_pass": gate_pass and len(solver_versions) == 1,
        "decision": (
            "retain"
            if gate_pass and len(solver_versions) == 1
            else "reject"
        ),
        "scenes": scenes,
    }


def _markdown(payload):
    lines = [
        "# Formal Group-Saturated Solver Gate",
        "",
        f"Decision: **{payload['decision'].upper()}**",
        "",
        (
            f"Pareto-improved scenes: {payload['improved_scene_count']}/"
            f"{payload['scene_count']}; GreatCourt non-regression: "
            f"{'yes' if payload['greatcourt_non_regression'] else 'no'}."
        ),
        "",
        (
            "Protocol: deterministic 28-query audit per scene "
            "(16 A2 tail failures + 12 stratified controls); this is a "
            "formal pre-full-test gate, not a full-test result."
        ),
        "",
        "| Scene | Baseline med. | Group med. | Delta | Baseline P90 | "
        "Group P90 | Delta | R5 delta | Lower / higher | Cat. rec. / reg. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scene, row in sorted(payload["scenes"].items()):
        baseline = row["baseline"]
        current = row["group_saturated"]
        delta = row["delta"]
        lines.append(
            f"| {scene} | {baseline['median_te_cm']:.3f} | "
            f"{current['median_te_cm']:.3f} | "
            f"{delta['median_te_cm']:+.3f} | "
            f"{baseline['p90_te_cm']:.3f} | "
            f"{current['p90_te_cm']:.3f} | "
            f"{delta['p90_te_cm']:+.3f} | "
            f"{delta['recall_5cm_5deg_percent']:+.2f} | "
            f"{row['paired_lower_te_count']} / "
            f"{row['paired_higher_te_count']} | "
            f"{row['catastrophic_recovered_count']} / "
            f"{row['catastrophic_regressed_count']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in args.reports]
    payload = summarize(reports)
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    Path(args.output_markdown).write_text(_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
