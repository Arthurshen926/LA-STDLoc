#!/usr/bin/env python3
"""Select sparse detector density from mapping-only deployment replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.config import load_mainline_config
from topology.crossfit_swap_revision import temporal_crossfit_split
from topology.deployment_revision import collect_deployment_statistics


def summarize_candidate(
    *, keypoints: int, anchor_count: int, folds: list[dict], coverage: dict
) -> dict:
    summaries = [value["summary"] for value in folds]
    return {
        "keypoints": int(keypoints),
        "anchor_count": int(anchor_count),
        "worst_median_te_cm": max(value["median_te_cm"] for value in summaries),
        "worst_mean_te_cm": max(value["mean_te_cm"] for value in summaries),
        "worst_p90_te_cm": max(value["p90_te_cm"] for value in summaries),
        "catastrophic_100cm_count": sum(
            value["catastrophic_100cm_count"] for value in summaries
        ),
        "minimum_raw_gt_precision_percent": min(
            value["raw_gt_precision_percent"] for value in summaries
        ),
        "minimum_inlier_gt_precision_percent": min(
            value["inlier_gt_precision_percent"] for value in summaries
        ),
        "worst_mean_hypotheses": max(
            value["mean_hypotheses"] for value in summaries
        ),
        "matching_rank_p10": float(coverage["final_rank_p10"]),
        "normalized_coverage_p10": float(coverage["normalized_coverage_p10"]),
    }


def select_smallest_stable(candidates: list[dict]) -> tuple[dict, dict]:
    """Use pose/cleanliness constraints first, then minimize online density."""
    best = {
        "median": min(value["worst_median_te_cm"] for value in candidates),
        "mean": min(value["worst_mean_te_cm"] for value in candidates),
        "p90": min(value["worst_p90_te_cm"] for value in candidates),
        "catastrophic": min(value["catastrophic_100cm_count"] for value in candidates),
        "raw": max(value["minimum_raw_gt_precision_percent"] for value in candidates),
        "inlier": max(value["minimum_inlier_gt_precision_percent"] for value in candidates),
    }
    eligible = []
    checks = {}
    for value in candidates:
        candidate_checks = {
            "coverage": value["normalized_coverage_p10"] >= 0.95,
            "median": value["worst_median_te_cm"] <= 1.05 * best["median"],
            "mean": value["worst_mean_te_cm"] <= 1.05 * best["mean"],
            "p90": value["worst_p90_te_cm"] <= 1.05 * best["p90"],
            "catastrophic": value["catastrophic_100cm_count"] <= best["catastrophic"],
            "raw_precision": value["minimum_raw_gt_precision_percent"] + 1.0 >= best["raw"],
            "inlier_precision": value["minimum_inlier_gt_precision_percent"] + 2.0 >= best["inlier"],
        }
        checks[str(value["keypoints"])] = candidate_checks
        if all(candidate_checks.values()):
            eligible.append(value)
    if eligible:
        selected = min(
            eligible,
            key=lambda value: (
                value["keypoints"],
                value["anchor_count"],
                value["worst_mean_hypotheses"],
            ),
        )
        policy = "minimum_density_within_mapping_pose_cleanliness_tolerances"
    else:
        selected = min(
            candidates,
            key=lambda value: (
                value["catastrophic_100cm_count"],
                value["worst_mean_te_cm"],
                value["worst_p90_te_cm"],
                value["keypoints"],
            ),
        )
        policy = "fallback_lexicographic_mapping_pose_risk"
    return selected, {"policy": policy, "best": best, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        help="KEYPOINTS=PIPELINE_ROOT; provide at least two candidates",
    )
    parser.add_argument(
        "--fixed-pipeline-root",
        type=Path,
        help="Hold map, metric, evidence, and cache fixed while varying row prefixes.",
    )
    parser.add_argument(
        "--keypoints",
        help="Comma-separated deployment prefixes for --fixed-pipeline-root.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--crossfit-blocks", type=int, default=8)
    args = parser.parse_args()
    if args.fixed_pipeline_root is not None:
        if args.candidate:
            raise ValueError("use either fixed-pipeline-root or candidate, not both")
        keypoints = [int(value) for value in str(args.keypoints or "").split(",") if value]
        specifications = [
            f"{value}={args.fixed_pipeline_root}" for value in keypoints
        ]
        fixed_map_metric = True
    else:
        specifications = list(args.candidate or [])
        fixed_map_metric = False
    if len(specifications) < 2:
        raise ValueError("density selection needs at least two deployment prefixes")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    fold_reports = {}
    for specification in specifications:
        keypoint_text, root_text = specification.split("=", 1)
        keypoints = int(keypoint_text)
        root = Path(root_text).resolve()
        manifest = {
            key: Path(value).resolve()
            for key, value in json.loads(
                (root / "pipeline_manifest.json").read_text()
            ).items()
        }
        state = torch.load(manifest["trained_map"], map_location="cpu", weights_only=False)
        teacher = torch.load(
            manifest["compact_positive_teacher"], map_location="cpu", weights_only=False
        )
        cache = torch.load(manifest["query_cache"], map_location="cpu", weights_only=False)
        calibration = json.loads(manifest["scene_calibration"].read_text())
        parameters = calibration["parameters"]
        selection, gate, split = temporal_crossfit_split(
            list(teacher["query_names"]), args.crossfit_blocks
        )
        common = {
            "state": state,
            "metric_state_path": manifest["metric_state"],
            "teacher": teacher,
            "query_cache": cache,
            "device": torch.device(args.device),
            "ransac_reprojection_px": float(parameters["ransac_reprojection_px"]),
            "clean_reprojection_px": float(parameters["clean_radius_px"]),
            "task_translation_m": float(parameters["task_translation_m"]),
            "task_rotation_deg": float(parameters["task_rotation_deg"]),
            "seed": args.seed,
        }
        folds = [
            collect_deployment_statistics(
                query_indices=indices,
                progress_label=f"density_k{keypoints}_{name}",
                deployment_row_limit=keypoints,
                collect_anchor_statistics=False,
                **common,
            )
            for name, indices in (("fold_a", selection), ("fold_b", gate))
        ]
        topology = json.loads(
            (root / "topology" / "adaptive_distillation_build.json").read_text()
        )
        result = summarize_candidate(
            keypoints=keypoints,
            anchor_count=int(torch.as_tensor(state["anchor_ids"]).numel()),
            folds=folds,
            coverage=topology["coverage"],
        )
        results.append(result)
        fold_reports[str(keypoints)] = {
            "pipeline": str(root),
            "split": split,
            "fold_a": folds[0]["summary"],
            "fold_b": folds[1]["summary"],
        }
    selected, decision = select_smallest_stable(results)
    report = {
        "schema": "lafgs_mapping_only_detector_density_selection",
        "version": 1,
        "changes_default_mainline": False,
        "uses_test_queries": False,
        "fixed_map_metric_and_evidence": fixed_map_metric,
        "candidates": results,
        "folds": fold_reports,
        "selected_keypoints": int(selected["keypoints"]),
        "decision": decision,
    }
    (output / "density_selection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
