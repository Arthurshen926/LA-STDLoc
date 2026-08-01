#!/usr/bin/env python3
"""Focused P0 oracle for top-K identity assignment plus fixed set selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_discrete_decision_oracles import (
    CandidateSet,
    fixed_protocol_candidates,
    oracle_topk_candidates,
    paired_summary,
    pose_error,
    project_points,
    run_pose,
    summarize_pose_errors,
)


ASSIGNMENT_KS = (1, 2, 4, 8, 16)
FIXED_ASSIGNMENT_KS = (4, 8, 16)
FIXED_PROTOCOLS = ("S512-PoseSufficient", "S1024-Block8")


def _hard_candidates(query: dict) -> CandidateSet:
    rows = np.asarray(query["hard_post_keypoint_idx"], dtype=np.int64)
    landmarks = np.asarray(query["hard_post_landmark_idx"], dtype=np.int64)
    scores = np.asarray(query["hard_post_scores"], dtype=np.float64)
    return CandidateSet(rows, landmarks, scores, np.arange(len(rows)))


def _rescue_summary(errors: dict[str, dict[str, list[float]]]) -> dict:
    actual_ae = np.asarray(errors["actual"]["ae"], dtype=np.float64)
    actual_te = np.asarray(errors["actual"]["te"], dtype=np.float64)
    catastrophic = (actual_te > 100.0) | (actual_ae > 10.0)
    r5_failure = (actual_te > 5.0) | (actual_ae > 5.0)
    tail = actual_te >= np.percentile(actual_te, 90)
    result = {}
    for method, values in errors.items():
        if method == "actual":
            continue
        method_ae = np.asarray(values["ae"], dtype=np.float64)
        method_te = np.asarray(values["te"], dtype=np.float64)
        method_catastrophic = (method_te > 100.0) | (method_ae > 10.0)
        method_r5 = (method_te <= 5.0) & (method_ae <= 5.0)
        result[method] = {
            "baseline_catastrophic_count": int(catastrophic.sum()),
            "catastrophic_rescued_count": int(
                (catastrophic & ~method_catastrophic).sum()
            ),
            "catastrophic_introduced_count": int(
                (~catastrophic & method_catastrophic).sum()
            ),
            "baseline_r5_failure_count": int(r5_failure.sum()),
            "r5_failure_rescued_count": int((r5_failure & method_r5).sum()),
            "p90_tail_improved_count": int(
                (tail & (method_te < actual_te - 1e-9)).sum()
            ),
            "p90_tail_query_count": int(tail.sum()),
        }
    return result


def evaluate(dump_dir: Path, radius: float, seed: int, bootstrap_samples: int) -> dict:
    dump_dir = dump_dir.resolve()
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    with np.load(dump_dir / manifest["landmark_bank"]) as bank:
        landmark_xyz = np.asarray(bank["landmark_xyz"], dtype=np.float64)
        source_groups = np.asarray(
            bank["source_gaussian_idx"], dtype=np.int64
        )
        dependency_groups = np.asarray(
            bank["dependency_group_id"]
            if "dependency_group_id" in bank.files
            else np.arange(len(landmark_xyz)),
            dtype=np.int64,
        )

    methods = ["actual", "replay"]
    methods += [f"A1_{protocol}" for protocol in FIXED_PROTOCOLS]
    methods += [f"OK{k}_one_of_k" for k in ASSIGNMENT_KS]
    methods += [
        f"OK{k}_{protocol}"
        for k in FIXED_ASSIGNMENT_KS
        for protocol in FIXED_PROTOCOLS
    ]
    errors = {method: {"ae": [], "te": []} for method in methods}
    solver_statistics = {
        method: {"matches": [], "inliers": [], "hypotheses": []}
        for method in methods
        if method != "actual"
    }
    coverage_counts = {
        k: {"positive": 0, "matchable_positive": 0, "all": 0, "matchable": 0}
        for k in ASSIGNMENT_KS
    }
    first_rank_counts = np.zeros(max(ASSIGNMENT_KS) + 1, dtype=np.int64)
    no_positive_count = 0
    pose_replay_failures = []
    query_records = []

    for query_index, filename in enumerate(manifest["query_files"]):
        with np.load(dump_dir / filename, allow_pickle=False) as loaded:
            query = {key: np.asarray(loaded[key]) for key in loaded.files}
        image_name = str(query["image_name"].item())
        keypoints = np.asarray(query["keypoint_xy"], dtype=np.float64) + 0.5
        topk_landmarks = np.asarray(
            query["topk_landmark_idx"], dtype=np.int64
        )
        topk_scores = np.asarray(query["topk_scores"], dtype=np.float64)
        gt_pose = np.asarray(query["gt_pose_w2c"], dtype=np.float64)
        K = np.asarray(query["K"], dtype=np.float64)
        height, width = int(query["height"]), int(query["width"])
        visible = np.asarray(query["render_visible_bank"], dtype=bool)
        projected, _, projection_valid = project_points(landmark_xyz, K, gt_pose)
        valid = (
            visible
            & projection_valid
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        distance = np.linalg.norm(
            keypoints[:, None] - projected[topk_landmarks], axis=2
        )
        correct = valid[topk_landmarks] & (distance <= float(radius))
        matchable = np.zeros(len(keypoints), dtype=bool)
        if bool(valid.any()):
            # A row is matchable if any visible map anchor projects inside the
            # strict radius; this denominator is independent of retrieval K.
            from scipy.spatial import cKDTree

            tree = cKDTree(projected[valid])
            nearest, _ = tree.query(keypoints, k=1)
            matchable = np.isfinite(nearest) & (nearest <= float(radius))
        for k in ASSIGNMENT_KS:
            width_k = min(k, correct.shape[1])
            positive = correct[:, :width_k].any(axis=1)
            counts = coverage_counts[k]
            counts["positive"] += int(positive.sum())
            counts["matchable_positive"] += int(positive[matchable].sum())
            counts["all"] += int(len(positive))
            counts["matchable"] += int(matchable.sum())
        any_positive = correct.any(axis=1)
        ranks = correct[any_positive].argmax(axis=1) + 1
        first_rank_counts += np.bincount(
            ranks, minlength=len(first_rank_counts)
        )[: len(first_rank_counts)]
        no_positive_count += int((~any_positive).sum())

        baseline = _hard_candidates(query)
        pre_rows = np.asarray(query["hard_pre_keypoint_idx"], dtype=np.int64)
        pre_landmarks = np.asarray(
            query["hard_pre_landmark_idx"], dtype=np.int64
        )
        if not (
            np.array_equal(pre_rows, baseline.keypoint_idx)
            and np.array_equal(pre_landmarks, baseline.landmark_idx)
            and not bool(query["geometry_selector_enabled"].item())
        ):
            raise ValueError(f"{image_name}: dump is not an A1-All graph")

        poses = {"actual": np.asarray(query["pred_pose_w2c"], dtype=np.float64)}
        replay_pose, replay_inliers, replay_diag = run_pose(
            baseline,
            keypoints,
            landmark_xyz,
            K,
            query,
            seed + query_index,
            return_diagnostics=True,
        )
        poses["replay"] = replay_pose
        solver_statistics["replay"]["matches"].append(len(baseline.scores))
        solver_statistics["replay"]["inliers"].append(len(replay_inliers))
        solver_statistics["replay"]["hypotheses"].append(
            replay_diag.get("ransac_actual_hypotheses")
        )
        dumped_inliers = np.asarray(query["hard_post_inliers"], dtype=np.int64)
        dumped_inliers = dumped_inliers[
            (dumped_inliers >= 0) & (dumped_inliers < len(baseline.scores))
        ]
        if not (
            np.allclose(poses["actual"], replay_pose, atol=1e-5, rtol=0.0)
            and np.array_equal(dumped_inliers, replay_inliers)
        ):
            pose_replay_failures.append(image_name)

        for protocol in FIXED_PROTOCOLS:
            selected = fixed_protocol_candidates(
                baseline,
                protocol,
                keypoints,
                landmark_xyz,
                dependency_groups,
                source_groups,
                (height, width),
            )
            pose, inliers, diagnostics = run_pose(
                selected,
                keypoints,
                landmark_xyz,
                K,
                query,
                seed + query_index,
                return_diagnostics=True,
            )
            method = f"A1_{protocol}"
            poses[method] = pose
            solver_statistics[method]["matches"].append(len(selected.scores))
            solver_statistics[method]["inliers"].append(len(inliers))
            solver_statistics[method]["hypotheses"].append(
                diagnostics.get("ransac_actual_hypotheses")
            )

        oracle_sets = {}
        for k in ASSIGNMENT_KS:
            width_k = min(k, topk_landmarks.shape[1])
            selected = oracle_topk_candidates(
                topk_landmarks[:, :width_k],
                topk_scores[:, :width_k],
                correct[:, :width_k],
            )
            oracle_sets[k] = selected
            pose, inliers, diagnostics = run_pose(
                selected,
                keypoints,
                landmark_xyz,
                K,
                query,
                seed + query_index,
                return_diagnostics=True,
            )
            method = f"OK{k}_one_of_k"
            poses[method] = pose
            solver_statistics[method]["matches"].append(len(selected.scores))
            solver_statistics[method]["inliers"].append(len(inliers))
            solver_statistics[method]["hypotheses"].append(
                diagnostics.get("ransac_actual_hypotheses")
            )
            if k not in FIXED_ASSIGNMENT_KS:
                continue
            for protocol in FIXED_PROTOCOLS:
                fixed = fixed_protocol_candidates(
                    selected,
                    protocol,
                    keypoints,
                    landmark_xyz,
                    dependency_groups,
                    source_groups,
                    (height, width),
                )
                unchanged = (
                    np.array_equal(fixed.keypoint_idx, selected.keypoint_idx)
                    and np.array_equal(fixed.landmark_idx, selected.landmark_idx)
                    and np.array_equal(fixed.scores, selected.scores)
                )
                if unchanged:
                    fixed_pose, fixed_inliers, fixed_diag = (
                        pose,
                        inliers,
                        diagnostics,
                    )
                else:
                    fixed_pose, fixed_inliers, fixed_diag = run_pose(
                        fixed,
                        keypoints,
                        landmark_xyz,
                        K,
                        query,
                        seed + query_index,
                        return_diagnostics=True,
                    )
                fixed_method = f"OK{k}_{protocol}"
                poses[fixed_method] = fixed_pose
                solver_statistics[fixed_method]["matches"].append(
                    len(fixed.scores)
                )
                solver_statistics[fixed_method]["inliers"].append(
                    len(fixed_inliers)
                )
                solver_statistics[fixed_method]["hypotheses"].append(
                    fixed_diag.get("ransac_actual_hypotheses")
                )

        query_pose = {}
        for method, pose in poses.items():
            ae, te = pose_error(pose, gt_pose)
            errors[method]["ae"].append(ae)
            errors[method]["te"].append(te)
            query_pose[method] = {"ae_deg": ae, "te_cm": te}
        query_records.append(
            {
                "image_name": image_name,
                "matchable_rows": int(matchable.sum()),
                "positive_rows_top16": int(any_positive.sum()),
                "pose": query_pose,
            }
        )

    pose_summary = {
        method: summarize_pose_errors(values["ae"], values["te"])
        for method, values in errors.items()
    }
    solver_summary = {}
    for method, values in solver_statistics.items():
        hypotheses = [value for value in values["hypotheses"] if value is not None]
        solver_summary[method] = {
            "matches_mean": float(np.mean(values["matches"])),
            "inliers_mean": float(np.mean(values["inliers"])),
            "hypotheses_mean": float(np.mean(hypotheses)) if hypotheses else None,
        }
    coverage_summary = {
        f"top_{k}": {
            "positive_row_rate_all": values["positive"] / max(values["all"], 1),
            "positive_recall_matchable": values["matchable_positive"]
            / max(values["matchable"], 1),
            **{name: int(value) for name, value in values.items()},
        }
        for k, values in coverage_counts.items()
    }
    paired = {
        method: paired_summary(
            errors["actual"]["te"],
            values["te"],
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        for method, values in errors.items()
        if method != "actual"
    }
    return {
        "schema": "lafgs_joint_assignment_p0_scene_v1",
        "dump_dir": str(dump_dir),
        "query_count": len(query_records),
        "strict_gt_radius_px": float(radius),
        "pose_replay_failures": pose_replay_failures,
        "topk_identity_coverage": {"radius_2px": coverage_summary},
        "first_strict_positive_rank": {
            "radius_2px": {
                "rank_counts": {
                    str(rank): int(first_rank_counts[rank])
                    for rank in range(1, len(first_rank_counts))
                },
                "no_positive_in_topk": int(no_positive_count),
                "total_rows": int(
                    coverage_counts[ASSIGNMENT_KS[-1]]["all"]
                ),
            }
        },
        "P0_oracles": {
            "one_of_k": {
                str(k): pose_summary[f"OK{k}_one_of_k"]
                for k in ASSIGNMENT_KS
            },
            "fixed_top1": {
                protocol: pose_summary[f"A1_{protocol}"]
                for protocol in FIXED_PROTOCOLS
            },
            "one_of_k_fixed_set": {
                str(k): {
                    protocol: pose_summary[f"OK{k}_{protocol}"]
                    for protocol in FIXED_PROTOCOLS
                }
                for k in FIXED_ASSIGNMENT_KS
            },
        },
        "pose": pose_summary,
        "solver": solver_summary,
        "paired_vs_actual": paired,
        "tail_rescue": _rescue_summary(errors),
        "queries": query_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    report = evaluate(
        Path(args.dump_dir),
        radius=args.radius,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "query_count": report["query_count"],
                "pose_replay_failure_count": len(report["pose_replay_failures"]),
                "topk_identity_coverage": report["topk_identity_coverage"],
                "P0_oracles": report["P0_oracles"],
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
