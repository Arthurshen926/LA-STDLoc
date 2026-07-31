#!/usr/bin/env python
"""Evaluate the formal group-saturated solver on frozen A2 query dumps."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from localization_training.dependency_pose_sampler import (
    solve_group_saturated_absolute_pose,
)
from localization_training.group_saturated_consensus import (
    build_surface_component_groups,
)
from scripts.evaluate_lafgs_group_consensus_oracle import (
    _load_source_normals,
)
from utils.pose_utils import cal_pose_error


def _summary(rows):
    te = np.asarray([row["te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row["ae_deg"] for row in rows], dtype=np.float64)
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "recall_5cm_5deg_percent": float(
            100.0 * np.mean((te <= 5.0) & (ae <= 5.0))
        ),
        "mean_iterations": float(
            np.mean([row["iterations"] for row in rows])
        ),
        "mean_runtime_ms": float(
            np.mean([row["runtime_ms"] for row in rows])
        ),
    }


def evaluate(args):
    dump_dir = Path(args.dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    bank = np.load(dump_dir / manifest["landmark_bank"])
    bank_xyz = np.asarray(bank["landmark_xyz"], dtype=np.float64)
    bank_source = np.asarray(bank["source_gaussian_idx"], dtype=np.int64)
    normals = _load_source_normals(args.gaussian_ply, bank_source)
    surface_groups, surface_diagnostics = build_surface_component_groups(
        bank_xyz,
        normals,
        voxel_scale_ratio=args.surface_voxel_scale_ratio,
        minimum_voxel_size=args.surface_minimum_voxel_size,
        maximum_normal_angle_degrees=args.surface_normal_angle,
    )
    selection = json.loads(Path(args.selection_report).read_text())
    selection_by_name = {
        str(row["image_name"]): row for row in selection["queries"]
    }

    rows = []
    for query_file in manifest["query_files"]:
        query = np.load(dump_dir / query_file)
        image_name = str(np.asarray(query["image_name"]).item())
        landmark_idx = np.asarray(
            query["hard_post_landmark_idx"], dtype=np.int64
        )
        keypoint_idx = np.asarray(
            query["hard_post_keypoint_idx"], dtype=np.int64
        )
        points2d = np.asarray(
            query["keypoint_xy"], dtype=np.float64
        )[keypoint_idx] + 0.5
        points3d = bank_xyz[landmark_idx]
        groups = surface_groups[landmark_idx]
        K = np.asarray(query["K"], dtype=np.float64)
        gt_pose = np.asarray(query["gt_pose_w2c"], dtype=np.float64)
        threshold = float(np.asarray(query["reprojection_error"]).item())
        baseline_pose = np.asarray(
            query["pred_pose_w2c"], dtype=np.float64
        )
        baseline_ae, baseline_te = cal_pose_error(baseline_pose, gt_pose)
        role = selection_by_name[image_name]["role"]
        for seed in args.seeds:
            start = time.perf_counter()
            pose, inliers, diagnostics = (
                solve_group_saturated_absolute_pose(
                    points2d,
                    points3d,
                    K,
                    surface_groups=groups,
                    group_cap=args.group_cap,
                    reprojection_error=threshold,
                    confidence=args.confidence,
                    max_iterations=args.max_iterations,
                    min_iterations=args.min_iterations,
                    seed=seed,
                )
            )
            runtime_ms = 1000.0 * (time.perf_counter() - start)
            ae_deg, te_cm = cal_pose_error(pose, gt_pose)
            rows.append(
                {
                    "image_name": image_name,
                    "role": role,
                    "seed": int(seed),
                    "te_cm": float(te_cm),
                    "ae_deg": float(ae_deg),
                    "baseline_te_cm": float(baseline_te),
                    "baseline_ae_deg": float(baseline_ae),
                    "inliers": int(inliers.size),
                    "iterations": int(diagnostics["iterations"]),
                    "runtime_ms": float(runtime_ms),
                    "group_score": float(diagnostics["group_score"]),
                    "group_capacity": float(
                        diagnostics["group_capacity"]
                    ),
                    "supported_groups": int(
                        diagnostics["supported_groups"]
                    ),
                    "maximum_group_fraction": float(
                        diagnostics["maximum_group_fraction"]
                    ),
                    "group_effective_sample_size": float(
                        diagnostics["group_effective_sample_size"]
                    ),
                    "solver_version": str(
                        diagnostics["implementation_version"]
                    ),
                }
            )

    by_seed = {}
    for seed in args.seeds:
        seed_rows = [row for row in rows if row["seed"] == seed]
        by_seed[str(seed)] = {
            role: _summary(
                seed_rows
                if role == "all"
                else [row for row in seed_rows if row["role"] == role]
            )
            for role in ("all", "failure", "control")
        }
    primary_seed = int(args.seeds[0])
    primary = [row for row in rows if row["seed"] == primary_seed]
    recovered = sum(
        row["te_cm"] < row["baseline_te_cm"] for row in primary
    )
    regressed = sum(
        row["te_cm"] > row["baseline_te_cm"] for row in primary
    )
    large_recovered = sum(
        row["baseline_te_cm"] > 50.0 and row["te_cm"] <= 50.0
        for row in primary
    )
    large_regressed = sum(
        row["baseline_te_cm"] <= 50.0 and row["te_cm"] > 50.0
        for row in primary
    )
    payload = {
        "schema": "lafgs_group_saturated_formal_solver_v1",
        "scene": args.scene,
        "group_cap": float(args.group_cap),
        "solver_version": (
            rows[0]["solver_version"] if rows else "unknown"
        ),
        "surface_diagnostics": surface_diagnostics,
        "seeds": [int(seed) for seed in args.seeds],
        "summary_by_seed": by_seed,
        "primary_seed_comparison": {
            "seed": primary_seed,
            "lower_te_count": int(recovered),
            "higher_te_count": int(regressed),
            "catastrophic_recovered_count": int(large_recovered),
            "catastrophic_regressed_count": int(large_regressed),
        },
        "queries": rows,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        f"# {args.scene} Formal Group-Saturated Solver",
        "",
        (
            f"Primary seed: lower TE {recovered}, higher TE {regressed}; "
            f"catastrophic recovered {large_recovered}, "
            f"regressed {large_regressed}."
        ),
        "",
        "| Seed | Role | Median TE | Mean TE | P90 TE | R5 | Iterations | Solver ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in args.seeds:
        for role in ("all", "failure", "control"):
            row = by_seed[str(seed)][role]
            lines.append(
                f"| {seed} | {role} | {row['median_te_cm']:.3f} | "
                f"{row['mean_te_cm']:.3f} | {row['p90_te_cm']:.3f} | "
                f"{row['recall_5cm_5deg_percent']:.2f} | "
                f"{row['mean_iterations']:.1f} | "
                f"{row['mean_runtime_ms']:.2f} |"
            )
    Path(args.output_markdown).write_text("\n".join(lines) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--group-cap", type=float, default=8.0)
    parser.add_argument("--surface-voxel-scale-ratio", type=float, default=0.02)
    parser.add_argument("--surface-minimum-voxel-size", type=float, default=0.5)
    parser.add_argument("--surface-normal-angle", type=float, default=25.0)
    parser.add_argument("--confidence", type=float, default=0.99999)
    parser.add_argument("--max-iterations", type=int, default=100000)
    parser.add_argument("--min-iterations", type=int, default=1000)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
