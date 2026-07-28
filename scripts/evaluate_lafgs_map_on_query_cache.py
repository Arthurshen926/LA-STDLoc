#!/usr/bin/env python3
"""Replay the deployment matcher/PnP on cached mapping queries."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.shared_metric import SharedLowRankMetric
from utils.pose_utils import cal_pose_error, solve_pose


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _project_errors(xyz, keypoints, K, pose):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = torch.empty_like(keypoints)
    projected[:, 0] = K[0, 0] * camera[:, 0] / depth.clamp_min(1e-8) + K[0, 2]
    projected[:, 1] = K[1, 1] * camera[:, 1] / depth.clamp_min(1e-8) + K[1, 2]
    return torch.linalg.norm(projected - keypoints, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric-state", default="")
    parser.add_argument("--reprojection-error", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--dynamic-outcomes-output", default="")
    parser.add_argument("--dependency-aware-sampler", action="store_true")
    parser.add_argument("--dependency-max-iterations", type=int, default=8000)
    parser.add_argument("--dependency-min-iterations", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda")
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    graph = torch.load(args.function_graph, map_location="cpu", weights_only=False)
    cache = cache_payload.get("queries", cache_payload)
    names = list(graph["query_names"])
    if args.query_limit > 0:
        names = names[: args.query_limit]
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    dependency_groups = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    surface_groups = torch.as_tensor(state["source_primitive_ids"]).long()
    scene_center = xyz.median(dim=0).values
    radial_distance = torch.linalg.norm(xyz - scene_center, dim=1)
    radial_boundaries = torch.quantile(
        radial_distance, torch.tensor([0.25, 0.5, 0.75])
    )
    radial_bins = torch.bucketize(radial_distance, radial_boundaries)
    bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1).to(
        device
    )
    metric = None
    if args.metric_state:
        metric_payload = torch.load(
            args.metric_state, map_location="cpu", weights_only=False
        )
        if int(torch.as_tensor(metric_payload["landmark_indices"]).numel()) != len(
            bank
        ):
            raise ValueError("metric state does not align with active map")
        metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
        metric.load_state_dict(metric_payload["metric_state_dict"])
        metric.eval()

    output = Path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    run_identity = {
        "map": str(Path(args.map).resolve()),
        "metric_state": str(Path(args.metric_state).resolve())
        if args.metric_state
        else None,
        "seed": int(args.seed),
        "reprojection_error": float(args.reprojection_error),
        "query_count_requested": len(names),
        "dependency_aware_sampler": bool(args.dependency_aware_sampler),
        "dependency_max_iterations": int(args.dependency_max_iterations),
        "dependency_min_iterations": int(args.dependency_min_iterations),
    }
    results = []
    dynamic_records = []
    if partial.is_file():
        saved = json.loads(partial.read_text())
        saved_identity = dict(saved["run_identity"])
        saved_identity.setdefault("dependency_aware_sampler", False)
        saved_identity.setdefault("dependency_max_iterations", 8000)
        saved_identity.setdefault("dependency_min_iterations", 500)
        if saved_identity != run_identity:
            raise ValueError("partial replay identity does not match current run")
        results = list(saved["results"])
        dynamic_records = list(saved.get("dynamic_records", []))
        if args.dynamic_outcomes_output and len(dynamic_records) != len(results):
            raise ValueError(
                "partial replay predates dynamic-outcome checkpointing"
            )
    completed_names = {row["query"] for row in results}
    if len(completed_names) != len(results):
        raise ValueError("partial replay contains duplicate queries")
    matching_seconds = 0.0
    ransac_seconds = 0.0
    for query_index, name in enumerate(names):
        if name in completed_names:
            continue
        cached = cache[name]
        rows = torch.as_tensor(graph["records"][query_index]["query_rows"]).long()
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows], dim=1
        ).to(device)
        with torch.no_grad():
            if metric is not None:
                descriptors, _ = metric(descriptors)
            torch.cuda.synchronize()
            start = time.perf_counter()
            scores, indices = (descriptors @ bank.T).max(dim=1)
            torch.cuda.synchronize()
            matching_seconds += time.perf_counter() - start
        keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        K = torch.as_tensor(cached["native_K"]).float()
        height, width = cached["native_input_hw"]
        cells = (
            (keypoints[:, 1] * 4 / max(float(height), 1.0)).floor().long().clamp(0, 3)
            * 4
            + (keypoints[:, 0] * 4 / max(float(width), 1.0)).floor().long().clamp(0, 3)
        )
        matched_indices = indices.cpu()
        start = time.perf_counter()
        pose, inliers, diagnostics = solve_pose(
            keypoints.numpy(),
            xyz[matched_indices].numpy(),
            K.numpy(),
            solver=(
                "poselib_dependency"
                if args.dependency_aware_sampler
                else "poselib"
            ),
            reprojection_error=float(args.reprojection_error),
            confidence=0.99999,
            max_iterations=(
                int(args.dependency_max_iterations)
                if args.dependency_aware_sampler
                else 100000
            ),
            min_iterations=(
                int(args.dependency_min_iterations)
                if args.dependency_aware_sampler
                else 1000
            ),
            scores=scores.cpu().numpy(),
            ransac_seed=int(args.seed),
            return_diagnostics=True,
            dependency_groups=dependency_groups[matched_indices].numpy(),
            image_cells=cells.numpy(),
            depth_bins=radial_bins[matched_indices].numpy(),
            surface_groups=surface_groups[matched_indices].numpy(),
        )
        ransac_seconds += time.perf_counter() - start
        re, te = cal_pose_error(pose, torch.as_tensor(cached["pose_w2c"]).numpy())
        gt_errors = _project_errors(
            xyz[indices.cpu()],
            keypoints,
            K,
            torch.as_tensor(cached["pose_w2c"]).float(),
        )
        inliers = torch.as_tensor(inliers).long().reshape(-1)
        results.append(
            {
                "query": name,
                "te_cm": float(te),
                "re_deg": float(re),
                "match_count": int(rows.numel()),
                "inlier_count": int(inliers.numel()),
                "raw_gt_precision_2px": float((gt_errors <= 2).float().mean()),
                "inlier_gt_precision_2px": float(
                    (gt_errors[inliers] <= 2).float().mean()
                    if inliers.numel()
                    else 0.0
                ),
                "hypotheses": diagnostics.get("ransac_actual_hypotheses"),
                "diverse_minimal_sets": int(
                    diagnostics.get("ransac_diverse_samples", 0)
                ),
                "fallback_minimal_sets": int(
                    diagnostics.get("ransac_fallback_samples", 0)
                ),
            }
        )
        if args.dynamic_outcomes_output:
            inlier_mask = torch.zeros(rows.numel(), dtype=torch.bool)
            inlier_mask[inliers] = True
            dynamic_records.append(
                {
                    "query_name": name,
                    "query_rows": rows,
                    "top1_anchor_indices": indices.cpu(),
                    "top1_scores": scores.cpu(),
                    "gt_reprojection_errors_px": gt_errors,
                    "ransac_inlier_mask": inlier_mask,
                    "clean_inlier_mask": inlier_mask & (gt_errors <= 4),
                    "harmful_inlier_mask": inlier_mask & (gt_errors > 4),
                    "te_cm": float(te),
                    "re_deg": float(re),
                    "hypotheses": diagnostics.get("ransac_actual_hypotheses"),
                }
            )
        if len(results) % 50 == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                partial,
                {
                    "run_identity": run_identity,
                    "results": results,
                    "dynamic_records": dynamic_records,
                },
            )
            print(f"{len(results)}/{len(names)}", flush=True)

    if len(results) != len(names):
        raise RuntimeError("cached replay did not cover every requested query")
    te = np.asarray([row["te_cm"] for row in results])
    hypotheses = [
        row["hypotheses"] for row in results if row["hypotheses"] is not None
    ]
    summary = {
        "schema": "lafgs_cached_deployment_replay",
        "map": str(Path(args.map).resolve()),
        "metric_state": str(Path(args.metric_state).resolve())
        if args.metric_state
        else None,
        "query_count": len(results),
        "anchor_count": int(xyz.shape[0]),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "recall_5cm_percent": float(100 * np.mean(te <= 5)),
        "raw_gt_precision_2px_percent": float(
            100 * np.mean([row["raw_gt_precision_2px"] for row in results])
        ),
        "inlier_gt_precision_2px_percent": float(
            100 * np.mean([row["inlier_gt_precision_2px"] for row in results])
        ),
        "mean_hypotheses": float(np.mean(hypotheses)) if hypotheses else None,
        "matching_ms_per_query": float(1000 * matching_seconds / len(results)),
        "ransac_ms_per_query": float(1000 * ransac_seconds / len(results)),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, summary)
    if args.dynamic_outcomes_output:
        if len(dynamic_records) != len(names):
            raise RuntimeError("dynamic outcomes do not cover every query")
        _atomic_torch(
            Path(args.dynamic_outcomes_output),
            {
                "schema": "lafgs_dynamic_self_localization_outcomes",
                "version": 1,
                "query_names": names,
                "anchor_count": int(xyz.shape[0]),
                "map": str(Path(args.map).resolve()),
                "metric_state": run_identity["metric_state"],
                "seed": int(args.seed),
                "records": dynamic_records,
                "summary": {
                    key: value
                    for key, value in summary.items()
                    if key != "results"
                },
            },
        )
    partial.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
