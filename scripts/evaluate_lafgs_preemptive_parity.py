#!/usr/bin/env python3
"""Compare a parity-preserving pose solver against frozen PoseLib results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_results(path: Path) -> dict[str, dict]:
    with path.open() as handle:
        payload = json.load(handle)
    rows = payload if isinstance(payload, list) else payload["results"]
    indexed = {str(row["image_name"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate query names in {path}")
    return indexed


def _pose_metrics(rows: list[dict]) -> dict[str, float]:
    translation = np.asarray([row["sparse_TE"] for row in rows], dtype=float)
    rotation = np.asarray([row["sparse_AE"] for row in rows], dtype=float)
    return {
        "median_te_cm": float(np.median(translation)),
        "mean_te_cm": float(np.mean(translation)),
        "p90_te_cm": float(np.quantile(translation, 0.9)),
        "r2_percent": float(
            100.0 * np.mean((translation <= 2.0) & (rotation <= 2.0))
        ),
        "r5_percent": float(
            100.0 * np.mean((translation <= 5.0) & (rotation <= 5.0))
        ),
    }


def _runtime_metrics(rows: list[dict]) -> dict[str, float]:
    sparse = [row["sparse"] for row in rows]

    def values(key: str) -> np.ndarray:
        return np.asarray([row.get(key, np.nan) for row in sparse], dtype=float)

    ransac = values("sparse_diag_runtime_ransac_ms")
    total = values("sparse_diag_runtime_total_ms")
    reduction = values("sparse_diag_preemptive_residual_reduction")
    return {
        "ransac_ms_mean": float(np.nanmean(ransac)),
        "ransac_ms_median": float(np.nanmedian(ransac)),
        "ransac_ms_p90": float(np.nanquantile(ransac, 0.9)),
        "total_ms_mean": float(np.nanmean(total)),
        "total_ms_median": float(np.nanmedian(total)),
        "total_ms_p90": float(np.nanquantile(total, 0.9)),
        "residual_evaluation_reduction_mean": (
            float(np.nanmean(reduction))
            if bool(np.isfinite(reduction).any())
            else 0.0
        ),
    }


def compare_results(baseline_path: Path, candidate_path: Path) -> dict:
    baseline = _load_results(baseline_path)
    candidate = _load_results(candidate_path)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(
            f"paired query contract differs: missing={missing}, extra={extra}"
        )
    names = sorted(baseline)
    baseline_rows = [baseline[name] for name in names]
    candidate_rows = [candidate[name] for name in names]

    def sparse_equal(name: str, key: str) -> bool:
        return baseline[name]["sparse"].get(key) == candidate[name][
            "sparse"
        ].get(key)

    pose_max_abs_difference = max(
        float(
            np.max(
                np.abs(
                    np.asarray(baseline[name]["sparse"]["pose_w2c"])
                    - np.asarray(candidate[name]["sparse"]["pose_w2c"])
                )
            )
        )
        for name in names
    )
    te_max_abs_difference = max(
        abs(
            float(baseline[name]["sparse_TE"])
            - float(candidate[name]["sparse_TE"])
        )
        for name in names
    )
    ae_max_abs_difference = max(
        abs(
            float(baseline[name]["sparse_AE"])
            - float(candidate[name]["sparse_AE"])
        )
        for name in names
    )
    baseline_runtime = _runtime_metrics(baseline_rows)
    candidate_runtime = _runtime_metrics(candidate_rows)
    ransac_delta = (
        candidate_runtime["ransac_ms_mean"]
        / baseline_runtime["ransac_ms_mean"]
        - 1.0
    )
    total_delta = (
        candidate_runtime["total_ms_mean"]
        / baseline_runtime["total_ms_mean"]
        - 1.0
    )
    parity = {
        "pose_matrix_max_abs_difference": pose_max_abs_difference,
        "te_max_abs_difference_cm": te_max_abs_difference,
        "ae_max_abs_difference_degrees": ae_max_abs_difference,
        "inlier_count_equal_queries": sum(
            sparse_equal(name, "inliers") for name in names
        ),
        "iterations_equal_queries": sum(
            sparse_equal(
                name, "sparse_diag_ransac_actual_hypotheses"
            )
            for name in names
        ),
        "refinements_equal_queries": sum(
            sparse_equal(name, "sparse_diag_ransac_refinements")
            for name in names
        ),
    }
    exact_parity = bool(
        pose_max_abs_difference == 0.0
        and te_max_abs_difference == 0.0
        and ae_max_abs_difference == 0.0
        and parity["inlier_count_equal_queries"] == len(names)
        and parity["iterations_equal_queries"] == len(names)
        and parity["refinements_equal_queries"] == len(names)
    )
    runtime_improved = bool(ransac_delta < 0.0 and total_delta <= 0.0)
    return {
        "schema": "lafgs_preemptive_parity_evaluation_v1",
        "baseline_path": str(baseline_path.resolve()),
        "candidate_path": str(candidate_path.resolve()),
        "query_count": len(names),
        "baseline": {
            **_pose_metrics(baseline_rows),
            **baseline_runtime,
        },
        "candidate": {
            **_pose_metrics(candidate_rows),
            **candidate_runtime,
        },
        "parity": parity,
        "ransac_runtime_delta_percent": float(100.0 * ransac_delta),
        "total_runtime_delta_percent": float(100.0 * total_delta),
        "success": {
            "exact_pose_solver_parity": exact_parity,
            "runtime_improved": runtime_improved,
            "deployable": bool(exact_parity and runtime_improved),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = compare_results(
        Path(args.baseline), Path(args.candidate)
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), **payload["success"]}))


if __name__ == "__main__":
    main()
