#!/usr/bin/env python3
"""Measure exact PoseLib headroom before training a pose-set objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.pose_set_oracle import (
    PoseSetAction,
    apply_pose_set_actions,
    beam_search_pose_set,
    normalized_pose_risk,
)
from map_learning.trainer import _project_errors


def _csr_values(record: dict, prefix: str, row: int) -> torch.Tensor:
    offsets = torch.as_tensor(record[f"{prefix}_offsets"]).long()
    indices = torch.as_tensor(record[f"{prefix}_indices"]).long()
    return indices[int(offsets[row]) : int(offsets[row + 1])]


def _summary(rows: list[dict], prefix: str) -> dict:
    te = np.asarray([row[f"{prefix}_te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row[f"{prefix}_ae_deg"] for row in rows], dtype=np.float64)
    risk = np.asarray([row[f"{prefix}_risk"] for row in rows], dtype=np.float64)
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "p90_ae_deg": float(np.percentile(ae, 90)),
        "median_risk": float(np.median(risk)),
        "mean_risk": float(np.mean(risk)),
        "p90_risk": float(np.percentile(risk, 90)),
        "failure_count": int(sum(row[f"{prefix}_failed"] for row in rows)),
        "mean_hypotheses": float(
            np.mean([row[f"{prefix}_hypotheses"] for row in rows])
        ),
    }


def _select_queries(total: int, requested: int) -> list[int]:
    if requested <= 0 or requested >= total:
        return list(range(total))
    return (
        torch.linspace(0, total - 1, steps=requested)
        .round()
        .long()
        .unique(sorted=True)
        .tolist()
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--query-count", type=int, default=96)
    parser.add_argument("--query-shard-count", type=int, default=1)
    parser.add_argument("--query-shard-index", type=int, default=0)
    parser.add_argument("--retrieval-topk", type=int, default=8)
    parser.add_argument("--maximum-actions", type=int, default=6)
    parser.add_argument("--joint-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--high-cost-hypotheses", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--stability-seeds", default="2026,2027,2028")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    calibration = json.loads(args.scene_calibration.read_text())
    parameters = calibration["parameters"]
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    if int(teacher["anchor_count"]) != int(xyz.shape[0]):
        raise ValueError("teacher and map anchor counts differ")
    metric = load_shared_metric(
        args.metric_state, anchor_ids=anchor_ids, device=device
    )
    names = list(teacher["query_names"])
    selected_queries = _select_queries(len(names), int(args.query_count))
    if int(args.query_shard_count) < 1:
        raise ValueError("query shard count must be positive")
    if not 0 <= int(args.query_shard_index) < int(args.query_shard_count):
        raise ValueError("query shard index is out of range")
    selected_queries = selected_queries[
        int(args.query_shard_index) :: int(args.query_shard_count)
    ]
    if not selected_queries:
        raise ValueError("query shard is empty")
    stability_seeds = [
        int(value) for value in args.stability_seeds.split(",") if value
    ]
    if int(args.seed) not in stability_seeds:
        stability_seeds.insert(0, int(args.seed))
    ransac_px = float(parameters["ransac_reprojection_px"])
    clean_px = float(parameters["clean_radius_px"])
    task_translation_m = float(parameters["task_translation_m"])
    task_rotation_deg = float(parameters["task_rotation_deg"])

    output_rows = []
    for completed, query_index in enumerate(selected_queries, start=1):
        record = teacher["records"][query_index]
        cached = cache[names[query_index]]
        all_rows = torch.as_tensor(record["query_rows"]).long()
        selected_local = torch.arange(all_rows.numel())
        if int(args.deployment_row_limit) > 0:
            selected_local = selected_local[
                all_rows < int(args.deployment_row_limit)
            ]
        rows = all_rows[selected_local]
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptors)
        scores, top_indices = torch.topk(
            adapted @ bank.T,
            k=min(int(args.retrieval_topk), bank.shape[0]),
            dim=1,
        )
        top_indices_cpu = top_indices.cpu()
        winners = top_indices_cpu[:, 0].numpy().astype(np.int64)
        keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        intrinsic = torch.as_tensor(cached["native_K"]).float()
        gt_pose = torch.as_tensor(cached["pose_w2c"]).float()

        positive_rows = [
            _csr_values(record, "positive", int(local))
            for local in selected_local.tolist()
        ]
        has_positive = torch.as_tensor(
            [positive.numel() > 0 for positive in positive_rows]
        )
        current_correct = torch.as_tensor(
            [bool((positive == int(winner)).any()) for positive, winner in zip(positive_rows, winners)]
        )
        legal_topk = []
        all_positive_best = winners.copy()
        topk_oracle = winners.copy()
        for local, positives in enumerate(positive_rows):
            legal = [
                int(anchor)
                for anchor in top_indices_cpu[local].tolist()
                if bool((positives == int(anchor)).any())
            ]
            legal_topk.append(legal)
            if not bool(current_correct[local]) and legal:
                topk_oracle[local] = legal[0]
            if positives.numel():
                positive_scores = adapted[local] @ bank[positives.to(device)].T
                all_positive_best[local] = int(
                    positives[int(positive_scores.argmax())]
                )

        solve_cache: dict[tuple[bytes, bytes, int], dict] = {}

        def solve(assignments: np.ndarray, active: np.ndarray, seed: int) -> dict:
            cache_key = (
                np.asarray(assignments, dtype=np.int64).tobytes(),
                np.asarray(active, dtype=bool).tobytes(),
                int(seed),
            )
            if cache_key in solve_cache:
                return solve_cache[cache_key]
            estimate = solve_absolute_pose(
                keypoints[active].numpy(),
                xyz[torch.from_numpy(assignments[active])].numpy(),
                intrinsic.numpy(),
                reprojection_error_px=ransac_px,
                confidence=0.99999,
                max_iterations=100000,
                min_iterations=1000,
                seed=int(seed),
            )
            ae_deg, te_cm = pose_error(estimate.pose_w2c, gt_pose.numpy())
            failed = int(estimate.inliers.size) < 4
            result = {
                "te_cm": float(te_cm),
                "ae_deg": float(ae_deg),
                "risk": normalized_pose_risk(
                    translation_cm=te_cm,
                    rotation_deg=ae_deg,
                    translation_scale_m=task_translation_m,
                    rotation_scale_deg=task_rotation_deg,
                    failed=failed,
                ),
                "failed": bool(failed),
                "inliers": int(estimate.inliers.size),
                "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
                "inlier_indices": np.asarray(estimate.inliers, dtype=np.int64),
            }
            solve_cache[cache_key] = result
            return result

        all_active = np.ones(winners.shape[0], dtype=bool)
        current = solve(winners, all_active, int(args.seed))
        topk_result = solve(topk_oracle, all_active, int(args.seed))
        positive_result = solve(all_positive_best, all_active, int(args.seed))
        current_errors = _project_errors(
            xyz[torch.from_numpy(winners)], keypoints, intrinsic, gt_pose
        )
        harmful_rows = set(
            int(value)
            for value in current["inlier_indices"]
            if float(current_errors[int(value)]) > clean_px
        )

        actions = []
        for local, legal in enumerate(legal_topk):
            if not bool(current_correct[local]) and legal:
                replacement = int(legal[0])
                replacement_error = float(
                    _project_errors(
                        xyz[replacement : replacement + 1],
                        keypoints[local : local + 1],
                        intrinsic,
                        gt_pose,
                    )[0]
                )
                priority = 100.0 * float(local in harmful_rows)
                priority += min(float(current_errors[local]), 100.0) - replacement_error
                actions.append(
                    PoseSetAction("swap", local, replacement, priority)
                )
            if local in harmful_rows:
                actions.append(
                    PoseSetAction(
                        "reject", local, -1, 50.0 + min(float(current_errors[local]), 100.0)
                    )
                )
        high_cost = int(current["hypotheses"]) >= int(args.high_cost_hypotheses)
        action_limit = min(
            int(args.maximum_actions), 3 if high_cost else int(args.maximum_actions)
        )
        joint_depth = min(int(args.joint_depth), 2 if high_cost else int(args.joint_depth))
        beam_width = min(int(args.beam_width), 1 if high_cost else int(args.beam_width))
        actions = sorted(actions, key=lambda action: -action.priority)[:action_limit]

        evaluation_cache: dict[tuple[PoseSetAction, ...], dict] = {}
        def evaluate(selected: tuple[PoseSetAction, ...]) -> dict:
            selected = tuple(sorted(selected))
            if selected not in evaluation_cache:
                revised, active = apply_pose_set_actions(winners, selected)
                evaluation_cache[selected] = solve(revised, active, int(args.seed))
            return evaluation_cache[selected]

        joint_actions, joint_result, trace = beam_search_pose_set(
            actions,
            evaluate,
            maximum_depth=joint_depth,
            beam_width=beam_width,
        )
        single_candidates = [()]
        single_candidates.extend((action,) for action in actions)
        single_actions = min(
            single_candidates,
            key=lambda selected: float(evaluate(selected)["risk"]),
        )
        single_result = evaluate(single_actions)

        stable = {}
        for label, selected in (("single", single_actions), ("joint", joint_actions)):
            revised, active = apply_pose_set_actions(winners, selected)
            risks = [solve(revised, active, seed)["risk"] for seed in stability_seeds]
            current_risks = [
                solve(winners, all_active, seed)["risk"] for seed in stability_seeds
            ]
            stable[label] = {
                "risk_gain_median": float(
                    np.median(np.asarray(current_risks) - np.asarray(risks))
                ),
                "positive_seed_count": int(
                    np.count_nonzero(np.asarray(current_risks) > np.asarray(risks))
                ),
                "seed_count": len(stability_seeds),
            }

        row = {
            "query_index": int(query_index),
            "image_name": names[query_index],
            "row_count": int(rows.numel()),
            "false_top1_count": int((has_positive & ~current_correct).sum()),
            "recoverable_false_top1_count": int(
                sum(not bool(current_correct[i]) and bool(legal_topk[i]) for i in range(len(legal_topk)))
            ),
            "candidate_action_count": len(actions),
            "single_action_count": len(single_actions),
            "joint_action_count": len(joint_actions),
            "high_cost_query": bool(high_cost),
            "effective_action_limit": int(action_limit),
            "effective_joint_depth": int(joint_depth),
            "effective_beam_width": int(beam_width),
            "single_stability": stable["single"],
            "joint_stability": stable["joint"],
            "joint_trace": trace,
        }
        for label, result in (
            ("current", current),
            ("topk", topk_result),
            ("positive", positive_result),
            ("single", single_result),
            ("joint", joint_result),
        ):
            for key in ("te_cm", "ae_deg", "risk", "failed", "inliers", "hypotheses"):
                row[f"{label}_{key}"] = result[key]
        output_rows.append(row)
        if completed % 8 == 0 or completed == len(selected_queries):
            partial = args.output.with_name(f"{args.output.stem}.partial.json")
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(
                json.dumps(
                    {
                        "schema": "lafgs_pose_set_oracle_partial",
                        "uses_test_queries": False,
                        "queries_complete": completed,
                        "query_count": len(selected_queries),
                        "queries": output_rows,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(
                json.dumps(
                    {
                        "event": "pose_set_oracle",
                        "queries_complete": completed,
                        "query_count": len(selected_queries),
                    }
                ),
                flush=True,
            )

    report = {
        "schema": "lafgs_pose_set_oracle_audit",
        "version": 1,
        "uses_test_queries": False,
        "map": str(args.map.resolve()),
        "metric_state": str(args.metric_state.resolve()),
        "query_count": len(output_rows),
        "query_selection": "uniform_mapping_only",
        "query_shard_count": int(args.query_shard_count),
        "query_shard_index": int(args.query_shard_index),
        "deployment_row_limit": int(args.deployment_row_limit),
        "retrieval_topk": int(args.retrieval_topk),
        "maximum_actions": int(args.maximum_actions),
        "joint_depth": int(args.joint_depth),
        "beam_width": int(args.beam_width),
        "high_cost_hypotheses": int(args.high_cost_hypotheses),
        "summaries": {
            label: _summary(output_rows, label)
            for label in ("current", "topk", "positive", "single", "joint")
        },
        "headroom": {
            "topk_risk_gain_mean": float(
                np.mean([row["current_risk"] - row["topk_risk"] for row in output_rows])
            ),
            "positive_risk_gain_mean": float(
                np.mean([row["current_risk"] - row["positive_risk"] for row in output_rows])
            ),
            "single_risk_gain_mean": float(
                np.mean([row["current_risk"] - row["single_risk"] for row in output_rows])
            ),
            "joint_risk_gain_mean": float(
                np.mean([row["current_risk"] - row["joint_risk"] for row in output_rows])
            ),
            "single_stable_positive_query_fraction": float(
                np.mean([row["single_stability"]["risk_gain_median"] > 0 for row in output_rows])
            ),
            "joint_stable_positive_query_fraction": float(
                np.mean([row["joint_stability"]["risk_gain_median"] > 0 for row in output_rows])
            ),
            "recoverable_false_top1_fraction": float(
                sum(row["recoverable_false_top1_count"] for row in output_rows)
                / max(sum(row["false_top1_count"] for row in output_rows), 1)
            ),
        },
        "queries": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **report["headroom"]}, indent=2))


if __name__ == "__main__":
    main()
