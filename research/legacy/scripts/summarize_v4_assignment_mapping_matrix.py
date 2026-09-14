#!/usr/bin/env python3
"""Summarize the all-24 mapping-LOO capacity-assignment experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def _summary(rows: list[dict]) -> dict:
    te = np.asarray([row["te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row["ae_deg"] for row in rows], dtype=np.float64)
    if not te.size:
        raise ValueError("cannot summarize an empty query set")
    tail = max(int(math.ceil(0.05 * te.size)), 1)
    return {
        "query_count": int(te.size),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "cvar95_te_cm": float(np.sort(te)[-tail:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "recall_5cm_5deg_percent": float(100.0 * np.mean((te < 5.0) & (ae < 5.0))),
        "catastrophic_100cm_count": int(np.count_nonzero(te >= 100.0)),
        "mean_correspondences": float(
            np.mean([row["correspondences"] for row in rows])
        ),
        "assignment_unmatched_query_rows": int(
            sum(row.get("assignment_unmatched_queries", 0) for row in rows)
        ),
        "assignment_reassigned_query_rows": int(
            sum(row.get("assignment_reassigned_queries", 0) for row in rows)
        ),
        "assignment_top1_collisions": int(
            sum(row.get("assignment_top1_collisions", 0) for row in rows)
        ),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix_state.resolve().read_text())
    if matrix.get("schema") != "lafgs_v4_assignment_mapping_loo_matrix":
        raise ValueError("unsupported assignment matrix schema")
    if matrix.get("uses_test_queries") is not False:
        raise ValueError("assignment matrix must be mapping-only")
    jobs = matrix.get("jobs", {})
    if len(jobs) != 72 or any(row.get("status") != "done" for row in jobs.values()):
        raise ValueError("summary requires all 24 scenes and three completed variants")

    rows_by_family_variant: dict[tuple[str, str], list[dict]] = {}
    scene_summaries = {}
    for key, job in sorted(jobs.items()):
        report_path = Path(job["output"]) / "full_mapping_loo_report.json"
        report = json.loads(report_path.read_text())
        if (
            report.get("schema") != "lafgs_rendered_track_full_mapping_loo_report"
            or report.get("uses_test_queries") is not False
            or report.get("uses_source_mapping_rgb") is not False
        ):
            raise ValueError(f"invalid mapping-only report: {key}")
        configuration = report["configuration"]
        expected_topk = (
            0 if job["variant"] == "top1" else int(job["variant"].split("k")[-1])
        )
        if int(configuration.get("assignment_topk", -1)) != expected_topk:
            raise ValueError(f"assignment configuration differs for {key}")
        statistics = torch.load(
            report["statistics"], map_location="cpu", weights_only=False
        )
        if statistics.get("uses_test_queries") is not False:
            raise ValueError(f"statistics unexpectedly use test queries: {key}")
        query_rows = list(statistics["queries"])
        rows_by_family_variant.setdefault((job["family"], job["variant"]), []).extend(
            query_rows
        )
        scene_summaries[key] = _summary(query_rows)

    families = sorted({family for family, _ in rows_by_family_variant})
    variants = tuple(matrix["variants"])
    family_summaries = {
        family: {
            variant: _summary(rows_by_family_variant[(family, variant)])
            for variant in variants
        }
        for family in families
    }
    gate_fields = (
        ("mean_te_cm", "lower"),
        ("p90_te_cm", "lower"),
        ("cvar95_te_cm", "lower"),
        ("recall_5cm_5deg_percent", "higher"),
    )
    gates = {}
    for variant in variants:
        if variant == "top1":
            continue
        family_gates = {}
        for family in families:
            baseline = family_summaries[family]["top1"]
            candidate = family_summaries[family][variant]
            field_gates = {
                field: (
                    candidate[field] <= baseline[field]
                    if direction == "lower"
                    else candidate[field] >= baseline[field]
                )
                for field, direction in gate_fields
            }
            family_gates[family] = {
                "passed": bool(all(field_gates.values())),
                "gates": field_gates,
            }
        gates[variant] = {
            "passed_all_three_datasets": bool(
                all(row["passed"] for row in family_gates.values())
            ),
            "families": family_gates,
        }
    eligible = [
        variant for variant, gate in gates.items() if gate["passed_all_three_datasets"]
    ]

    def selection_key(variant: str) -> tuple[float, float, int]:
        cvar_ratios = []
        mean_ratios = []
        for family in families:
            baseline = family_summaries[family]["top1"]
            candidate = family_summaries[family][variant]
            cvar_ratios.append(
                candidate["cvar95_te_cm"] / max(baseline["cvar95_te_cm"], 1e-12)
            )
            mean_ratios.append(
                candidate["mean_te_cm"] / max(baseline["mean_te_cm"], 1e-12)
            )
        return (
            max(cvar_ratios),
            max(mean_ratios),
            int(variant.rsplit("k", 1)[-1]),
        )

    selected_variant = min(eligible, key=selection_key) if eligible else None
    output = {
        "schema": "lafgs_v4_assignment_mapping_loo_summary",
        "version": 1,
        "uses_test_queries": False,
        "protocol": {
            "scene_count": 24,
            "parameters_shared_across_scenes": True,
            "selection": "full_mapping_leave_one_query_descriptor_out",
            "pose": "one_standard_poselib_call",
            "dataset_gate": "mean_p90_cvar95_recall_all_nonregression",
            "selection_rule": (
                "eligible_all_three_datasets_then_minimize_worst_family_cvar_ratio_"
                "then_worst_family_mean_ratio_then_smaller_k"
            ),
        },
        "matrix_state": str(args.matrix_state.resolve()),
        "family_summaries": family_summaries,
        "scene_summaries": scene_summaries,
        "gates": gates,
        "selected_variant": selected_variant,
        "authorizes_one_frozen_24_scene_test_matrix": selected_variant is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output.resolve(), output)
    print(json.dumps({"output": str(args.output.resolve()), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
