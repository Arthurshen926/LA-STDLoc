#!/usr/bin/env python3
"""Replay SLPS selections on a frozen native sparse candidate graph.

The query dump is produced once by ``stdloc.py`` with
``diagnostics.dump_discrete_oracle``. Learned modes only consume deployment
features; ground truth is used for reporting and the explicitly named oracle
upper bound.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import torch

from localization_training.pose_sufficient_selector import (
    build_pose_sufficient_features,
)
from localization_training.slps_selector import (
    SLPS_BIAS_AWARE_FEATURE_NAMES,
    build_relation_groups,
    build_slps_features,
    slps_from_state,
)
from localization_training.slps_residual_signatures import (
    residual_signature_features,
)
from utils.pose_utils import cal_pose_error, solve_pose


_QUERIES: list[dict] = []


def _project_errors(
    keypoints: np.ndarray,
    xyz: np.ndarray,
    K: np.ndarray,
    pose_w2c: np.ndarray,
) -> np.ndarray:
    camera = xyz @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    depth = camera[:, 2]
    projected = np.empty_like(keypoints, dtype=np.float64)
    projected[:, 0] = K[0, 0] * camera[:, 0] / np.maximum(depth, 1e-8) + K[0, 2]
    projected[:, 1] = K[1, 1] * camera[:, 1] / np.maximum(depth, 1e-8) + K[1, 2]
    errors = np.linalg.norm(projected - keypoints, axis=1)
    errors[depth <= 1e-8] = np.inf
    return errors


def _solve(task: tuple[int, str, int, int]) -> tuple[int, str, int, dict]:
    query_index, method, budget, seed = task
    query = _QUERIES[query_index]
    count = len(query["keypoints"])
    target = count if int(budget) <= 0 else min(max(int(budget), 4), count)
    if method in query.get("sets", {}):
        indices = np.asarray(query["sets"][method], dtype=np.int64)
    elif method == "all":
        indices = np.arange(count, dtype=np.int64)
    else:
        indices = np.asarray(query["orders"][method][:target], dtype=np.int64)
    start = time.perf_counter()
    pose, inliers, diagnostics = solve_pose(
        query["keypoints"][indices] + 0.5,
        query["xyz"][indices],
        query["K"],
        solver="poselib",
        reprojection_error=float(query["reprojection_error"]),
        confidence=float(query["confidence"]),
        max_iterations=int(query["max_iterations"]),
        min_iterations=int(query["min_iterations"]),
        scores=query["scores"][indices],
        ransac_seed=int(seed),
        return_diagnostics=True,
    )
    runtime_ms = 1000.0 * (time.perf_counter() - start)
    re_deg, te_cm = cal_pose_error(pose, query["gt_pose_w2c"])
    inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    selected_errors = query["gt_errors"][indices]
    return query_index, method, int(budget), {
        "query_name": query["query_name"],
        "seed": int(seed),
        "selected_count": int(len(indices)),
        "te_cm": float(te_cm),
        "re_deg": float(re_deg),
        "r5_success": bool(te_cm <= 5.0 and re_deg <= 5.0),
        "catastrophic": bool(te_cm > 100.0 or re_deg > 10.0),
        "inlier_count": int(len(inliers)),
        "inlier_ratio": float(len(inliers) / max(len(indices), 1)),
        "raw_gt_precision_2px": float(np.mean(selected_errors <= 2.0)),
        "inlier_gt_precision_2px": (
            float(np.mean(selected_errors[inliers] <= 2.0))
            if len(inliers)
            else 0.0
        ),
        "hypotheses": int(
            diagnostics.get("ransac_actual_hypotheses") or 100000
        ),
        "runtime_ms": runtime_ms,
        "selector_runtime_ms": float(query["selector_runtime_ms"]),
    }


def _summary(rows: list[dict]) -> dict[str, float]:
    te = np.asarray([row["te_cm"] for row in rows])
    re = np.asarray([row["re_deg"] for row in rows])
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.quantile(te, 0.9)),
        "median_re_deg": float(np.median(re)),
        "recall_5cm_percent": float(
            100.0 * np.mean((te <= 5.0) & (re <= 5.0))
        ),
        "recall_2cm_percent": float(
            100.0 * np.mean((te <= 2.0) & (re <= 2.0))
        ),
        "catastrophic_count": int(sum(row["catastrophic"] for row in rows)),
        "mean_selected_count": float(
            np.mean([row["selected_count"] for row in rows])
        ),
        "mean_inlier_ratio_percent": float(
            100.0 * np.mean([row["inlier_ratio"] for row in rows])
        ),
        "raw_gt_precision_2px_percent": float(
            100.0 * np.mean([row["raw_gt_precision_2px"] for row in rows])
        ),
        "inlier_gt_precision_2px_percent": float(
            100.0 * np.mean(
                [row["inlier_gt_precision_2px"] for row in rows]
            )
        ),
        "mean_hypotheses": float(
            np.mean([row["hypotheses"] for row in rows])
        ),
        "mean_ransac_ms": float(
            np.mean([row["runtime_ms"] for row in rows])
        ),
        "mean_selector_ms": float(
            np.mean([row["selector_runtime_ms"] for row in rows])
        ),
        "mean_selector_plus_ransac_ms": float(
            np.mean(
                [
                    row["selector_runtime_ms"] + row["runtime_ms"]
                    for row in rows
                ]
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="256,384,512,768,1024,1536,0")
    parser.add_argument(
        "--methods",
        default="learned,strict_probability,solver_probability,score,oracle_clean,all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--greedy-block-size",
        type=int,
        default=0,
        help="Override selector greedy block size; zero keeps the checkpoint.",
    )
    parser.add_argument(
        "--maximum-queries",
        type=int,
        default=0,
        help="Limit replay queries for a deterministic smoke test; zero uses all.",
    )
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir).resolve()
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    bank = np.load(dump_dir / manifest["landmark_bank"])
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    selector_state = torch.load(
        args.selector, map_location="cpu", weights_only=False
    )
    if selector_state.get("schema") != "lafgs_slps_selector":
        raise ValueError("replay requires an SLPS selector")
    anchor_count = len(torch.as_tensor(state["anchor_ids"]))
    if len(bank["anchor_id"]) != anchor_count:
        raise ValueError("dump bank and materialized map differ")

    methods = tuple(
        value.strip() for value in str(args.methods).split(",") if value.strip()
    )
    supported = {
        "learned",
        "adaptive",
        "strict_probability",
        "solver_probability",
        "score",
        "oracle_clean",
        "all",
    }
    if set(methods) - supported:
        raise ValueError(f"unsupported replay methods: {set(methods) - supported}")
    budgets = tuple(
        sorted(
            {int(value) for value in str(args.budgets).split(",") if value.strip()},
            key=lambda value: (value <= 0, value),
        )
    )
    device = torch.device(args.device)
    model = slps_from_state(selector_state, device=device)
    if int(args.greedy_block_size) > 0:
        model.config = type(model.config)(
            **{
                **model.export_config(),
                "greedy_block_size": int(args.greedy_block_size),
            }
        )
    source = torch.as_tensor(state["source_primitive_ids"]).long()
    dependency = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    track = torch.as_tensor(
        state.get("track_cluster_ids", state["dependency_group_ids"])
    ).long()
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    anchor_type = torch.as_tensor(
        state.get("anchor_type", torch.zeros(anchor_count))
    ).long()
    statistics = {
        name: torch.as_tensor(value).float()
        for name, value in selector_state["anchor_statistics"].items()
    }
    stability = torch.as_tensor(
        selector_state["anchor_track_stability"]
    ).float()
    residual_state = selector_state.get("residual_signature_state")
    needs_residual_signatures = list(
        selector_state.get("feature_names", ())
    ) == list(SLPS_BIAS_AWARE_FEATURE_NAMES)
    if needs_residual_signatures and residual_state is None:
        raise ValueError("bias-aware selector misses residual signature state")

    maximum_budget = max(
        [value for value in budgets if value > 0] or [0]
    )
    queries = []
    with torch.inference_mode():
        query_files = list(manifest["query_files"])
        if int(args.maximum_queries) > 0:
            query_files = query_files[: int(args.maximum_queries)]
        for filename in query_files:
            payload = np.load(dump_dir / filename)
            selector_start = time.perf_counter()
            query_rows = np.asarray(payload["hard_pre_keypoint_idx"], dtype=np.int64)
            topk_indices = torch.from_numpy(
                np.asarray(payload["topk_landmark_idx"], dtype=np.int64)[query_rows]
            ).long()
            topk_scores = torch.from_numpy(
                np.asarray(payload["topk_scores"], dtype=np.float32)[query_rows]
            ).float()
            top1 = topk_indices[:, 0]
            hard_landmarks = torch.from_numpy(
                np.asarray(payload["hard_pre_landmark_idx"], dtype=np.int64)
            ).long()
            if not torch.equal(top1, hard_landmarks):
                raise ValueError(f"{filename} is not an unchanged top-1 graph")
            keypoints = torch.from_numpy(
                np.asarray(payload["keypoint_xy"], dtype=np.float32)[query_rows]
            ).float()
            keypoint_scores = torch.from_numpy(
                np.asarray(payload["keypoint_detector_score"], dtype=np.float32)[
                    query_rows
                ]
            ).float()
            image_hw = (int(payload["height"]), int(payload["width"]))
            base = build_pose_sufficient_features(
                topk_scores,
                topk_indices,
                keypoints=keypoints,
                keypoint_scores=keypoint_scores,
                image_hw=image_hw,
                source_groups=source,
                dependency_groups=dependency,
                anchor_statistics=statistics,
                entropy_temperature=float(
                    selector_state.get("entropy_temperature", 0.05)
                ),
                prior_strength=float(selector_state.get("prior_strength", 12.0)),
            )
            residual_features = None
            if residual_state is not None:
                residual_config = residual_state["config"]
                residual_features = residual_signature_features(
                    residual_state["statistics"],
                    anchor_indices=top1,
                    keypoints=keypoints,
                    image_hw=image_hw,
                    grid_size=int(residual_config["grid_size"]),
                    clip_px=float(residual_config["clip_px"]),
                    anchor_prior=float(residual_config["anchor_prior"]),
                    cell_prior=float(residual_config["cell_prior"]),
                    rate_prior=float(residual_config["rate_prior"]),
                )
            features = build_slps_features(
                base,
                xyz=xyz[top1],
                anchor_type=anchor_type[top1],
                track_groups=track[top1],
                track_stability=stability[top1],
                anchor_map_support=statistics["attempts"][top1],
                residual_signature_features=residual_features,
            )
            groups = build_relation_groups(
                keypoints=keypoints,
                image_hw=image_hw,
                xyz=xyz[top1],
                dependency_groups=dependency[top1],
                source_groups=source[top1],
                track_groups=track[top1],
            )
            encoded = model.encode(features, groups)
            count = len(features)
            target = min(maximum_budget, count)
            orders: dict[str, np.ndarray] = {}
            explicit_sets: dict[str, np.ndarray] = {}
            if "learned" in methods:
                orders["learned"] = model.greedy_order(
                    encoded, groups, maximum_count=target
                ).cpu().numpy()
            if "adaptive" in methods:
                selection = model.select(
                    features,
                    groups,
                    anchor_indices=top1,
                    query_name=str(payload["image_name"]),
                    encoded=encoded,
                    **selector_state["selector_config"],
                )
                explicit_sets["adaptive"] = torch.where(
                    selection.selected_mask
                )[0].numpy()
            for name in ("strict_probability", "solver_probability"):
                if name in methods:
                    orders[name] = torch.argsort(
                        encoded[name], descending=True, stable=True
                    ).cpu().numpy()
            if "score" in methods:
                orders["score"] = torch.argsort(
                    topk_scores[:, 0], descending=True, stable=True
                ).numpy()
            keypoints_np = keypoints.numpy()
            xyz_np = xyz[top1].numpy()
            K = np.asarray(payload["K"], dtype=np.float64)
            gt_pose = np.asarray(payload["gt_pose_w2c"], dtype=np.float64)
            gt_errors = _project_errors(keypoints_np + 0.5, xyz_np, K, gt_pose)
            if "oracle_clean" in methods:
                orders["oracle_clean"] = np.argsort(
                    gt_errors, kind="stable"
                )
            selector_runtime_ms = 1000.0 * (
                time.perf_counter() - selector_start
            )
            queries.append(
                {
                    "query_name": str(payload["image_name"]),
                    "keypoints": keypoints_np,
                    "xyz": xyz_np,
                    "scores": topk_scores[:, 0].numpy(),
                    "K": K,
                    "gt_pose_w2c": gt_pose,
                    "gt_errors": gt_errors,
                    "orders": orders,
                    "sets": explicit_sets,
                    "selector_runtime_ms": selector_runtime_ms,
                    "reprojection_error": float(payload["reprojection_error"]),
                    "confidence": float(payload["confidence"]),
                    "max_iterations": int(payload["max_iterations"]),
                    "min_iterations": int(payload["min_iterations"]),
                }
            )

    global _QUERIES
    _QUERIES = queries
    tasks = []
    for query_index in range(len(queries)):
        for method in methods:
            method_budgets = (
                (0,) if method in {"all", "adaptive"} else budgets
            )
            for budget in method_budgets:
                if budget <= 0 and method not in {"all", "adaptive"}:
                    continue
                tasks.append((query_index, method, int(budget), int(args.seed)))
    torch.set_num_threads(1)
    context = mp.get_context("fork")
    results = []
    with context.Pool(processes=max(int(args.workers), 1)) as pool:
        for result in pool.imap_unordered(_solve, tasks, chunksize=1):
            results.append(result)
    grouped: dict[str, dict[int, list[dict]]] = {}
    for _, method, budget, row in results:
        grouped.setdefault(method, {}).setdefault(int(budget), []).append(row)
    for method in grouped:
        for budget in grouped[method]:
            grouped[method][budget].sort(key=lambda row: row["query_name"])
    summaries = {
        method: {
            str(budget): _summary(rows)
            for budget, rows in sorted(by_budget.items())
        }
        for method, by_budget in grouped.items()
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "lafgs_slps_frozen_candidate_replay",
        "dump_dir": str(dump_dir),
        "map": str(Path(args.map).resolve()),
        "selector": str(Path(args.selector).resolve()),
        "seed": int(args.seed),
        "methods": list(methods),
        "budgets": list(budgets),
        "summary": summaries,
        "results": {
            method: {str(budget): rows for budget, rows in by_budget.items()}
            for method, by_budget in grouped.items()
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
