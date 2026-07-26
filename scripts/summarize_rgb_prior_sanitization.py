#!/usr/bin/env python3
"""Summarize RGB-prior sanitization evaluations under one fixed protocol."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DIAGNOSTIC_KEYS = {
    "raw_gt_p2": "sparse_diag_all_gt_precision_2px_mean",
    "inlier_gt_p2": "sparse_diag_inlier_gt_precision_2px_mean",
    "solver_inlier_ratio": "sparse_diag_ransac_inlier_ratio_solver_mean",
    "pose_info_t_logdet": "sparse_diag_inlier_pose_info_translation_logdet_mean",
    "ransac_hypotheses": "sparse_diag_ransac_actual_hypotheses_mean",
    "runtime_ms": "sparse_diag_runtime_total_ms_mean",
}


def _load_evaluation(pointer: Path) -> dict:
    result_dir = Path(pointer.read_text().strip())
    summary = json.loads((result_dir / "results_summary.json").read_text())
    results = json.loads((result_dir / "results.json").read_text())
    translation_error = np.asarray(
        [float(result["sparse_TE"]) for result in results], dtype=np.float64
    )
    rotation_error = np.asarray(
        [float(result["sparse_AE"]) for result in results], dtype=np.float64
    )
    sparse = summary["sparse"]
    diagnostics = summary.get("sparse_diagnostics", {})
    row = {
        "label": pointer.stem,
        "result_dir": str(result_dir),
        "count": int(translation_error.size),
        "median_te_cm": float(np.median(translation_error)),
        "mean_te_cm": float(translation_error.mean()),
        "p90_te_cm": float(np.percentile(translation_error, 90)),
        "p95_te_cm": float(np.percentile(translation_error, 95)),
        "max_te_cm": float(translation_error.max()),
        "median_ae_deg": float(np.median(rotation_error)),
        "mean_ae_deg": float(rotation_error.mean()),
        "recall_2cm": float(sparse["recall_2cm_2d"]),
        "recall_5cm": float(sparse["recall_5cm_5d"]),
        "avg_inliers": float(sparse["avg_inliers"]),
    }
    for output_key, input_key in DIAGNOSTIC_KEYS.items():
        value = diagnostics.get(input_key)
        row[output_key] = None if value is None else float(value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="Run directory containing results/*.path; repeat for multiple priors.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for run_dir in args.run_dir:
        for pointer in sorted((run_dir / "results").glob("*.path")):
            row = _load_evaluation(pointer)
            row["prior"] = run_dir.name
            rows.append(row)
    if not rows:
        raise SystemExit("No completed result pointers were found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(args.output), "csv": str(csv_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
