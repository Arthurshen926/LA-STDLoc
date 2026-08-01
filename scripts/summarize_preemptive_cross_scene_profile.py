#!/usr/bin/env python3
"""Summarize the one-time exact-preemptive cross-scene crossover profile."""

import argparse
import json
from pathlib import Path


SCENES = (
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
)
DIFFICULT_SCENES = {"GreatCourt", "KingsCollege", "OldHospital", "StMarysChurch"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = {}
    missing = []
    for scene in SCENES:
        path = root / scene / "parity.json"
        if not path.is_file():
            missing.append(scene)
            continue
        report = json.loads(path.read_text())
        rows[scene] = {
            "query_count": report["query_count"],
            "exact_parity": report["success"]["exact_pose_solver_parity"],
            "native_ransac_ms_mean": report["baseline"]["ransac_ms_mean"],
            "preemptive_ransac_ms_mean": report["candidate"]["ransac_ms_mean"],
            "ransac_delta_percent": report["ransac_runtime_delta_percent"],
            "native_total_ms_mean": report["baseline"]["total_ms_mean"],
            "preemptive_total_ms_mean": report["candidate"]["total_ms_mean"],
            "total_delta_percent": report["total_runtime_delta_percent"],
            "residual_reduction": report["candidate"][
                "residual_evaluation_reduction_mean"
            ],
        }
    if missing:
        summary = {
            "schema": "lafgs_preemptive_cross_scene_profile_v1",
            "status": "INCOMPLETE",
            "missing_scenes": missing,
            "scenes": rows,
        }
    else:
        exact = all(row["exact_parity"] for row in rows.values())
        difficult_speedups = [
            scene
            for scene in DIFFICULT_SCENES
            if rows[scene]["ransac_delta_percent"] <= -5.0
            and rows[scene]["total_delta_percent"] <= 0.0
        ]
        deployable = bool(exact and difficult_speedups)
        summary = {
            "schema": "lafgs_preemptive_cross_scene_profile_v1",
            "status": "RETAIN" if deployable else "CLOSE",
            "checks": {
                "five_scene_exact_parity": exact,
                "difficult_scene_at_least_5pct_ransac_speedup_without_total_regression": bool(
                    difficult_speedups
                ),
            },
            "difficult_scene_speedups": difficult_speedups,
            "scenes": rows,
            "recommendation": (
                "RETAIN_EXACT_PREEMPTIVE_FOR_CROSSOVER"
                if deployable
                else "PERMANENTLY_CLOSE_EXACT_PREEMPTIVE_DEPLOYMENT"
            ),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
