#!/usr/bin/env python
"""Validation-only gate selection for experimental dense pose refinements."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(records, pose_key):
    # STDLoc's ``*_TE`` result fields are centimeters, not meters.
    te = np.asarray([float(record[f"{pose_key}_TE"]) for record in records], dtype=np.float64)
    ae = np.asarray([float(record[f"{pose_key}_AE"]) for record in records], dtype=np.float64)
    return {
        "count": int(te.size),
        "median_te_cm": float(np.median(te)) if te.size else None,
        "mean_te_cm": float(np.mean(te)) if te.size else None,
        "median_ae_deg": float(np.median(ae)) if ae.size else None,
        "mean_ae_deg": float(np.mean(ae)) if ae.size else None,
        "recall_5cm_5deg": float(np.mean((te <= 5.0) & (ae <= 5.0))) if te.size else None,
        "recall_2m_5deg": float(np.mean((te <= 200.0) & (ae <= 5.0))) if te.size else None,
    }


def accepts(record, gate):
    return bool(
        record.get("raw_solver_success", False)
        and int(record.get("raw_dense_inliers", 0)) >= int(gate["min_inliers"])
        and float(record.get("raw_dense_inlier_ratio", 0.0)) >= float(gate["min_inlier_ratio"])
        and float(record.get("raw_pose_delta_translation_m", np.inf))
        <= float(gate["max_translation_delta_m"])
        and float(record.get("raw_pose_delta_rotation_deg", np.inf))
        <= float(gate["max_rotation_delta_deg"])
    )


def apply_gate(records, gate):
    applied = []
    for source in records:
        record = dict(source)
        accepted = accepts(record, gate)
        prefix = "raw_dense" if accepted else "sparse"
        record["gated_accepted"] = accepted
        record["gated_pose_w2c"] = record[f"{prefix}_pose_w2c"]
        record["gated_AE"] = float(record[f"{prefix}_AE"])
        record["gated_TE"] = float(record[f"{prefix}_TE"])
        applied.append(record)
    return applied


def score(summary, mean_weight_cm):
    return float(summary["median_te_cm"] + float(mean_weight_cm) * summary["mean_te_cm"])


def choose_gate(records, *, translation_gates, rotation_gates, min_inliers, min_inlier_ratios,
                mean_weight_cm, min_median_gain_cm, max_recall_2m_drop):
    baseline = summarize(records, "sparse")
    rows = []
    for translation in translation_gates:
        for rotation in rotation_gates:
            for inliers in min_inliers:
                for ratio in min_inlier_ratios:
                    gate = {
                        "max_translation_delta_m": float(translation),
                        "max_rotation_delta_deg": float(rotation),
                        "min_inliers": int(inliers),
                        "min_inlier_ratio": float(ratio),
                    }
                    applied = apply_gate(records, gate)
                    summary = summarize(applied, "gated")
                    rows.append(
                        {
                            "gate": gate,
                            "summary": summary,
                            "score": score(summary, mean_weight_cm),
                            "acceptance_rate": float(np.mean([r["gated_accepted"] for r in applied])),
                        }
                    )
    valid = [
        row
        for row in rows
        if row["summary"]["median_te_cm"] <= baseline["median_te_cm"] - float(min_median_gain_cm)
        and row["summary"]["recall_2m_5deg"]
        >= baseline["recall_2m_5deg"] - float(max_recall_2m_drop)
    ]
    if not valid:
        return {
            "selected": "sparse_fallback",
            "gate": None,
            "baseline": baseline,
            "rows": rows,
            "reason": "No predeclared local gate improved validation median TE under recall guard.",
        }
    selected = min(valid, key=lambda row: (row["score"], -row["acceptance_rate"]))
    return {
        "selected": "gated_dense",
        "gate": selected["gate"],
        "baseline": baseline,
        "selected_row": selected,
        "rows": rows,
        "reason": "Selected on validation only by median TE + weighted mean TE with recall guard.",
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Select/apply a dense refinement update gate")
    parser.add_argument("--validation_results", required=True)
    parser.add_argument("--output", required=True)
    # Dense PnP is a local update to an already localized sparse pose.  Include
    # centimeter-scale update guards before considering broad fallbacks.
    parser.add_argument(
        "--translation_gates",
        type=float,
        nargs="+",
        default=[0.025, 0.05, 0.10, 0.15, 0.25, 0.5, 1.0],
    )
    parser.add_argument(
        "--rotation_gates",
        type=float,
        nargs="+",
        default=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    )
    parser.add_argument("--min_inliers", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--min_inlier_ratios", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--mean_weight_cm", type=float, default=0.05)
    parser.add_argument("--min_median_gain_cm", type=float, default=0.02)
    parser.add_argument("--max_recall_2m_drop", type=float, default=0.01)
    parser.add_argument("--apply_results", default=None)
    parser.add_argument("--applied_output", default=None)
    args = parser.parse_args()

    validation_records = json.loads(Path(args.validation_results).read_text())
    if not isinstance(validation_records, list):
        raise ValueError("validation_results must be a list")
    selection = choose_gate(
        validation_records,
        translation_gates=args.translation_gates,
        rotation_gates=args.rotation_gates,
        min_inliers=args.min_inliers,
        min_inlier_ratios=args.min_inlier_ratios,
        mean_weight_cm=args.mean_weight_cm,
        min_median_gain_cm=args.min_median_gain_cm,
        max_recall_2m_drop=args.max_recall_2m_drop,
    )
    selection["selection_protocol"] = {
        "validation_results": str(Path(args.validation_results).resolve()),
        "test_metrics_used": False,
        "objective": "median_te_cm + 0.05 * mean_te_cm",
        "recall_guard": "recall_2m_5deg",
    }
    write_json(args.output, selection)
    print(json.dumps(selection, indent=2, sort_keys=True))

    if args.apply_results:
        if not args.applied_output:
            raise ValueError("--apply_results requires --applied_output")
        records = json.loads(Path(args.apply_results).read_text())
        if selection["gate"] is None:
            applied = apply_gate(
                records,
                {
                    "max_translation_delta_m": -1.0,
                    "max_rotation_delta_deg": -1.0,
                    "min_inliers": 10**9,
                    "min_inlier_ratio": 1.0,
                },
            )
        else:
            applied = apply_gate(records, selection["gate"])
        report = {
            "schema_version": 1,
            "selection_path": str(Path(args.output).resolve()),
            "selection": selection["selected"],
            "gate": selection["gate"],
            "count": len(applied),
            "sparse": summarize(applied, "sparse"),
            "raw_dense": summarize(applied, "raw_dense"),
            "gated": summarize(applied, "gated"),
            "acceptance_rate": float(np.mean([record["gated_accepted"] for record in applied])) if applied else 0.0,
        }
        output = Path(args.applied_output)
        write_json(output / "gated_results.json", applied)
        write_json(output / "gated_summary.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
