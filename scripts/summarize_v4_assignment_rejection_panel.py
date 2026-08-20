#!/usr/bin/env python3
"""Apply the frozen early-rejection gate to the eight-scene V4 panel."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def summary(rows: list[dict]) -> dict:
    te = np.asarray([row["te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row["ae_deg"] for row in rows], dtype=np.float64)
    tail = max(int(math.ceil(0.05 * te.size)), 1)
    return {
        "query_count": int(te.size),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "cvar95_te_cm": float(np.sort(te)[-tail:].mean()),
        "recall_5cm_5deg_percent": float(100.0 * np.mean((te < 5.0) & (ae < 5.0))),
        "catastrophic_100cm_count": int(np.count_nonzero(te >= 100.0)),
        "mean_hypotheses": float(np.mean([row["hypotheses"] for row in rows])),
        "retained_matches_mean": float(
            np.mean([row["correspondences"] for row in rows])
        ),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_queries(report_path: Path) -> list[dict]:
    report = json.loads(report_path.read_text())
    if report.get("uses_test_queries") is not False:
        raise ValueError(f"panel report uses test queries: {report_path}")
    statistics = torch.load(
        report["statistics"], map_location="cpu", weights_only=False
    )
    if statistics.get("uses_test_queries") is not False:
        raise ValueError(f"panel statistics use test queries: {report_path}")
    return list(statistics["queries"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path(
            "docs/evidence/v4_assignment_rejection_panel_preregistration.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.resolve().read_text())
    if (
        prereg.get("schema") != "lafgs_v4_assignment_rejection_panel_preregistration"
        or prereg.get("uses_test_queries") is not False
    ):
        raise ValueError("invalid assignment rejection preregistration")
    scenes = list(prereg["hard_scenes"])
    candidates = dict(prereg["candidates"])
    families = ("7Scenes", "12Scenes", "Cambridge")
    baseline_rows = {family: [] for family in families}
    candidate_rows = {
        candidate: {family: [] for family in families} for candidate in candidates
    }
    for scene in scenes:
        family, name = scene.split("/", 1)
        baseline_rows[family].extend(
            load_queries(
                args.baseline_root.resolve()
                / family
                / name
                / "top1"
                / "full_mapping_loo_report.json"
            )
        )
        for candidate in candidates:
            candidate_rows[candidate][family].extend(
                load_queries(
                    args.panel_root.resolve()
                    / family
                    / name
                    / "candidate_batch"
                    / candidate
                    / "mapping_topk_replay_report.json"
                )
            )
    baseline = {family: summary(baseline_rows[family]) for family in families}
    pooled_baseline = summary(
        [row for family in families for row in baseline_rows[family]]
    )
    results = {}
    limits = prereg["pooled_hard_scene_rejection_gate"]
    useful_limit = prereg["minimum_useful_change"][
        "at_least_one_family_catastrophic_count_strictly_lower_or_cvar95_ratio_at_most"
    ]
    for candidate in candidates:
        candidate_summary = {
            family: summary(candidate_rows[candidate][family]) for family in families
        }
        pooled = summary(
            [row for family in families for row in candidate_rows[candidate][family]]
        )
        gates = {
            "catastrophic_not_increased": pooled["catastrophic_100cm_count"]
            <= pooled_baseline["catastrophic_100cm_count"]
            + int(limits["catastrophic_100cm_count_maximum_baseline_delta"]),
            "recall_within_tolerance": pooled["recall_5cm_5deg_percent"]
            >= pooled_baseline["recall_5cm_5deg_percent"]
            - float(limits["recall_5cm_5deg_maximum_drop_percentage_points"]),
            "mean_within_tolerance": pooled["mean_te_cm"]
            <= pooled_baseline["mean_te_cm"] * float(limits["mean_te_maximum_ratio"]),
            "cvar95_within_tolerance": pooled["cvar95_te_cm"]
            <= pooled_baseline["cvar95_te_cm"]
            * float(limits["cvar95_te_maximum_ratio"]),
            "median_within_tolerance": pooled["median_te_cm"]
            <= pooled_baseline["median_te_cm"]
            * float(limits["median_te_maximum_ratio"]),
            "pose_hypotheses_within_tolerance": pooled["mean_hypotheses"]
            <= pooled_baseline["mean_hypotheses"]
            * float(limits["mean_hypotheses_maximum_ratio"]),
        }
        useful = pooled["catastrophic_100cm_count"] < pooled_baseline[
            "catastrophic_100cm_count"
        ] or pooled["cvar95_te_cm"] <= pooled_baseline["cvar95_te_cm"] * float(
            useful_limit
        )
        results[candidate] = {
            "survives": bool(useful and all(gates.values())),
            "has_minimum_useful_change": bool(useful),
            "pooled_gates": gates,
            "pooled_summary": pooled,
            "family_diagnostics": candidate_summary,
        }
    survivors = [name for name, row in results.items() if row["survives"]]
    selected = (
        min(
            survivors,
            key=lambda name: (
                results[name]["pooled_summary"]["catastrophic_100cm_count"],
                results[name]["pooled_summary"]["cvar95_te_cm"],
                results[name]["pooled_summary"]["mean_te_cm"],
                -results[name]["pooled_summary"]["recall_5cm_5deg_percent"],
                results[name]["pooled_summary"]["median_te_cm"],
                name,
            ),
        )
        if survivors
        else None
    )
    output = {
        "schema": "lafgs_v4_assignment_rejection_panel_summary",
        "version": 1,
        "uses_test_queries": False,
        "baseline": baseline,
        "pooled_hard_scene_baseline": pooled_baseline,
        "candidates": results,
        "hard_scene_survivors": survivors,
        "selected_shared_hard_scene_candidate": selected,
        "authorizes_official_test": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output.resolve(), output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
