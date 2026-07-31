#!/usr/bin/env python3
"""Summarize scene-specific SLPS mapping gates and held-out Cambridge tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _pose_metrics(rows: list[dict], *, sparse_keys: bool) -> dict[str, float]:
    te_key = "sparse_TE" if sparse_keys else "te_cm"
    ae_key = "sparse_AE" if sparse_keys else "re_deg"
    te = np.asarray([float(row[te_key]) for row in rows])
    ae = np.asarray([float(row[ae_key]) for row in rows])
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(ae)),
        "r5_percent": float(100.0 * np.mean((te <= 5.0) & (ae <= 5.0))),
        "catastrophic_count": int(np.sum((te > 100.0) | (ae > 10.0))),
    }


def _test_result(pointer: Path) -> tuple[dict, list[dict]]:
    result = Path(pointer.read_text().strip())
    return (
        json.loads((result / "results_summary.json").read_text()),
        json.loads((result / "results.json").read_text()),
    )


def _mean(values):
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="/mnt/pool/sqy/stdloc_lafgs_slps_20260731",
    )
    parser.add_argument(
        "--frozen-root",
        default="/mnt/pool/sqy/stdloc_lafgs_v1_frozen_multiscene_20260731",
    )
    parser.add_argument(
        "--scenes",
        default="GreatCourt,KingsCollege,ShopFacade,StMarysChurch",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.root)
    frozen_root = Path(args.frozen_root)
    scenes = [
        value.strip()
        for value in str(args.scenes).split(",")
        if value.strip()
    ]
    summary = {
        "schema": "lafgs_slps_multiscene_summary",
        "root": str(root.resolve()),
        "scenes": {},
    }
    for scene in scenes:
        scene_root = root / scene
        scene_summary: dict = {"mapping": {}, "test": {}}
        for variant in (
            "adaptive",
            "fixed256",
            "fixed384",
            "fixed512",
            "fixed768",
        ):
            path = scene_root / f"mapping_{variant}_seed2026.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            scene_summary["mapping"][variant] = {
                key: payload.get(key)
                for key in (
                    "query_count",
                    "median_te_cm",
                    "mean_te_cm",
                    "p90_te_cm",
                    "recall_5cm_5deg_percent",
                    "raw_gt_precision_2px_percent",
                    "inlier_gt_precision_2px_percent",
                    "mean_solver_inlier_ratio_percent",
                    "mean_hypotheses",
                    "ransac_ms_per_query",
                )
            }
        baseline_payload = json.loads(
            (frozen_root / scene / "frozen_results.json").read_text()
        )
        baseline_seeds = baseline_payload["results"][
            "A1_reconstructed"
        ]
        paired = []
        slps_metrics = []
        baseline_metrics = []
        for seed in (2026, 2027, 2028):
            pointer = scene_root / "test" / "adaptive" / f"seed{seed}" / "result.path"
            if not pointer.is_file() or str(seed) not in baseline_seeds:
                continue
            _, slps_rows = _test_result(pointer)
            baseline_result = Path(
                baseline_seeds[str(seed)]["result_path"]
            )
            baseline_rows = json.loads(
                (baseline_result / "results.json").read_text()
            )
            if len(slps_rows) != len(baseline_rows):
                raise ValueError(f"{scene} test query counts differ")
            slps_value = _pose_metrics(slps_rows, sparse_keys=True)
            baseline_value = _pose_metrics(
                baseline_rows, sparse_keys=True
            )
            slps_metrics.append(slps_value)
            baseline_metrics.append(baseline_value)
            paired.append(
                {
                    "seed": seed,
                    "slps": slps_value,
                    "a1": baseline_value,
                    "median_delta_cm": (
                        slps_value["median_te_cm"]
                        - baseline_value["median_te_cm"]
                    ),
                    "mean_delta_cm": (
                        slps_value["mean_te_cm"]
                        - baseline_value["mean_te_cm"]
                    ),
                    "p90_delta_cm": (
                        slps_value["p90_te_cm"]
                        - baseline_value["p90_te_cm"]
                    ),
                    "r5_delta_percent": (
                        slps_value["r5_percent"]
                        - baseline_value["r5_percent"]
                    ),
                    "catastrophic_delta": (
                        slps_value["catastrophic_count"]
                        - baseline_value["catastrophic_count"]
                    ),
                }
            )
        if paired:
            scene_summary["test"] = {
                "seeds": paired,
                "aggregate": {
                    key: _mean(
                        [row["slps"][key] for row in paired]
                    )
                    for key in (
                        "median_te_cm",
                        "mean_te_cm",
                        "p90_te_cm",
                        "r5_percent",
                        "catastrophic_count",
                    )
                },
                "a1_aggregate": {
                    key: _mean([row["a1"][key] for row in paired])
                    for key in (
                        "median_te_cm",
                        "mean_te_cm",
                        "p90_te_cm",
                        "r5_percent",
                        "catastrophic_count",
                    )
                },
                "delta": {
                    key: _mean([row[key] for row in paired])
                    for key in (
                        "median_delta_cm",
                        "mean_delta_cm",
                        "p90_delta_cm",
                        "r5_delta_percent",
                        "catastrophic_delta",
                    )
                },
            }
        summary["scenes"][scene] = scene_summary

    complete = [
        value["test"]
        for value in summary["scenes"].values()
        if value.get("aggregate")
    ]
    if complete:
        non_degraded = sum(
            value["delta"]["median_delta_cm"] <= 0.0
            and value["delta"]["p90_delta_cm"] <= 0.0
            and value["delta"]["r5_delta_percent"] >= 0.0
            and value["delta"]["catastrophic_delta"] <= 0.0
            for value in complete
        )
        summary["gate"] = {
            "complete_scene_count": len(complete),
            "joint_non_degraded_scene_count": non_degraded,
            "passed": bool(
                len(complete) >= 4 and non_degraded >= 4
            ),
        }
    output = (
        Path(args.output)
        if args.output
        else root / "slps_multiscene_summary.json"
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
