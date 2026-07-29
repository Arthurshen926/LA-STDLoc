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
from localization_training.full_primitive_retrieval import (
    chunked_exact_topk_family_prototype,
)
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
    parser.add_argument("--family-prototype-state", default="")
    parser.add_argument("--reprojection-error", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-name-file", default="")
    parser.add_argument("--dynamic-outcomes-output", default="")
    parser.add_argument(
        "--split",
        choices=("mapping_replay", "crossfold_mapping", "heldout_test"),
        default="mapping_replay",
    )
    parser.add_argument("--dependency-aware-sampler", action="store_true")
    parser.add_argument("--dependency-max-iterations", type=int, default=8000)
    parser.add_argument("--dependency-min-iterations", type=int, default=500)
    parser.add_argument("--dependency-rescue-max-iterations", type=int, default=0)
    parser.add_argument("--dependency-rescue-inlier-ratio", type=float, default=0.0)
    parser.add_argument("--dependency-guided-mixture", type=float, default=0.0)
    parser.add_argument("--dependency-guided-rank-power", type=float, default=0.5)
    parser.add_argument(
        "--dependency-sampling-model",
        default="",
        help="Optional OOF sampling-logit model; uses only deployment-visible match features.",
    )
    parser.add_argument(
        "--dependency-risk-score-threshold",
        type=float,
        default=0.0,
        help="Use dependency sampling only when the query median top-1 score is at or below this value.",
    )
    parser.add_argument("--minimal-set-record-limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    graph = torch.load(args.function_graph, map_location="cpu", weights_only=False)
    sampling_model = (
        torch.load(
            args.dependency_sampling_model,
            map_location="cpu",
            weights_only=False,
        )
        if args.dependency_sampling_model
        else None
    )
    cache = cache_payload.get("queries", cache_payload)
    all_names = list(graph["query_names"])
    if args.query_name_file:
        if args.query_start:
            raise ValueError("query_start cannot be combined with query_name_file")
        requested = [
            value.strip()
            for value in Path(args.query_name_file).read_text().splitlines()
            if value.strip()
        ]
        missing = sorted(set(requested) - set(all_names))
        if missing:
            raise ValueError(f"query_name_file contains unknown queries: {missing[:3]}")
        query_start = 0
        names = requested
    else:
        if args.query_start < 0 or args.query_start >= len(all_names):
            raise ValueError("query_start is outside the function graph")
        query_start = int(args.query_start)
        names = all_names[query_start:]
    if args.query_limit > 0:
        names = names[: args.query_limit]
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    dependency_groups = torch.as_tensor(
        state.get("coarse_dependency_group_ids", state["dependency_group_ids"])
    ).long()
    surface_groups = torch.as_tensor(state["source_primitive_ids"]).long()
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
    family_state = None
    if args.family_prototype_state:
        family_state = torch.load(
            args.family_prototype_state, map_location="cpu", weights_only=False
        )
        if family_state.get("schema") != "lafgs_basin_family_prototypes":
            raise ValueError("unsupported family prototype state")
        family_indices = torch.as_tensor(
            family_state["landmark_indices"]
        ).long().reshape(-1)
        if not torch.equal(family_indices, torch.arange(len(bank))):
            raise ValueError("family prototype state does not align with map rows")
        family_features = F.normalize(
            torch.as_tensor(family_state["prototype_features"]).float(), dim=1
        ).to(device)
        family_parents = torch.as_tensor(
            family_state["prototype_anchor_indices"]
        ).long().to(device)
        family_bias = torch.as_tensor(
            family_state.get(
                "prototype_bias", torch.zeros(len(family_features))
            )
        ).float().to(device)
        family_temperature = torch.as_tensor(
            family_state.get(
                "prototype_temperature", torch.ones(len(family_features))
            )
        ).float().to(device)

    output = Path(args.output)
    partial = output.with_suffix(output.suffix + ".partial")
    dynamic_partial = output.with_suffix(output.suffix + ".dynamic.partial.pt")
    run_identity = {
        "map": str(Path(args.map).resolve()),
        "metric_state": str(Path(args.metric_state).resolve())
        if args.metric_state
        else None,
        "family_prototype_state": str(
            Path(args.family_prototype_state).resolve()
        )
        if args.family_prototype_state
        else None,
        "seed": int(args.seed),
        "reprojection_error": float(args.reprojection_error),
        "query_count_requested": len(names),
        "query_start": query_start,
        "query_name_file": str(Path(args.query_name_file).resolve())
        if args.query_name_file
        else None,
        "split": args.split,
        "dependency_sampler_version": 5
        if args.dependency_aware_sampler
        else None,
        "dependency_aware_sampler": bool(args.dependency_aware_sampler),
        "dependency_max_iterations": int(args.dependency_max_iterations),
        "dependency_min_iterations": int(args.dependency_min_iterations),
        "dependency_rescue_max_iterations": int(
            args.dependency_rescue_max_iterations
        ),
        "dependency_rescue_inlier_ratio": float(
            args.dependency_rescue_inlier_ratio
        ),
        "dependency_guided_mixture": float(args.dependency_guided_mixture),
        "dependency_guided_rank_power": float(
            args.dependency_guided_rank_power
        ),
        "dependency_sampling_model": str(
            Path(args.dependency_sampling_model).resolve()
        )
        if args.dependency_sampling_model
        else None,
        "dependency_risk_score_threshold": float(
            args.dependency_risk_score_threshold
        ),
        "minimal_set_record_limit": int(args.minimal_set_record_limit),
    }
    results = []
    dynamic_records = []
    if partial.is_file():
        saved = json.loads(partial.read_text())
        saved_identity = dict(saved["run_identity"])
        saved_identity.setdefault("split", "mapping_replay")
        saved_identity.setdefault("query_start", 0)
        saved_identity.setdefault("query_name_file", None)
        saved_identity.setdefault("dependency_aware_sampler", False)
        saved_identity.setdefault("dependency_max_iterations", 8000)
        saved_identity.setdefault("dependency_min_iterations", 500)
        saved_identity.setdefault("dependency_rescue_max_iterations", 0)
        saved_identity.setdefault("dependency_rescue_inlier_ratio", 0.0)
        saved_identity.setdefault("dependency_guided_mixture", 0.0)
        saved_identity.setdefault("dependency_guided_rank_power", 0.5)
        saved_identity.setdefault("dependency_sampling_model", None)
        saved_identity.setdefault("dependency_risk_score_threshold", 0.0)
        saved_identity.setdefault("minimal_set_record_limit", 0)
        if not saved_identity["dependency_aware_sampler"]:
            saved_identity.setdefault("dependency_sampler_version", None)
        if saved_identity != run_identity:
            raise ValueError("partial replay identity does not match current run")
        results = list(saved["results"])
        if args.dynamic_outcomes_output:
            if not dynamic_partial.is_file():
                raise ValueError("dynamic replay partial sidecar is missing")
            saved_dynamic = torch.load(
                dynamic_partial, map_location="cpu", weights_only=False
            )
            if saved_dynamic["run_identity"] != run_identity:
                raise ValueError("dynamic partial identity mismatch")
            dynamic_records = list(saved_dynamic["records"])
            if len(dynamic_records) != len(results):
                raise ValueError("dynamic partial does not align with replay")
    completed_names = {row["query"] for row in results}
    if len(completed_names) != len(results):
        raise ValueError("partial replay contains duplicate queries")
    matching_seconds = 0.0
    ransac_seconds = 0.0
    for local_query_index, name in enumerate(names):
        if name in completed_names:
            continue
        query_index = all_names.index(name)
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
            if family_state is None:
                top_values, top_indices = torch.topk(
                    descriptors @ bank.T, k=min(2, bank.shape[0]), dim=1
                )
            else:
                retrieval = chunked_exact_topk_family_prototype(
                    descriptors,
                    bank,
                    family_features,
                    family_parents,
                    prototype_bias=family_bias,
                    prototype_temperature=family_temperature,
                    topk=min(2, bank.shape[0]),
                )
                top_values, top_indices = retrieval.scores, retrieval.indices
            scores = top_values[:, 0]
            indices = top_indices[:, 0]
            score_margins = (
                top_values[:, 0] - top_values[:, 1]
                if top_values.shape[1] > 1
                else torch.zeros_like(scores)
            )
            torch.cuda.synchronize()
            matching_duration = time.perf_counter() - start
            matching_seconds += matching_duration
        keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float()[rows]
            + float(cached.get("pixel_center_offset", 0.5))
        )
        keypoint_scores = torch.as_tensor(
            cached.get(
                "native_scores",
                torch.ones(
                    torch.as_tensor(cached["native_keypoints"]).shape[0]
                ),
            )
        ).float()[rows]
        pose_sampling_scores = scores
        if sampling_model is not None:
            fold = str(sampling_model["query_to_fold"].get(name, "all"))
            model = sampling_model["models"].get(
                fold, sampling_model["models"].get("all")
            )
            if model is None:
                raise ValueError(f"sampling model has no coefficients for fold {fold}")
            features = torch.stack(
                (scores.cpu(), score_margins.cpu(), keypoint_scores), dim=1
            ).float()
            mean = torch.as_tensor(model["feature_mean"]).float()
            scale = torch.as_tensor(model["feature_scale"]).float().clamp_min(1e-6)
            coefficients = torch.as_tensor(model["coefficients"]).float()
            pose_sampling_scores = (
                ((features - mean) / scale) @ coefficients
                + float(model.get("intercept", 0.0))
            ).to(scores.device)
        K = torch.as_tensor(cached["native_K"]).float()
        height, width = cached["native_input_hw"]
        cells = (
            (keypoints[:, 1] * 4 / max(float(height), 1.0)).floor().long().clamp(0, 3)
            * 4
            + (keypoints[:, 0] * 4 / max(float(width), 1.0)).floor().long().clamp(0, 3)
        )
        matched_indices = indices.cpu()
        use_dependency_sampler = bool(args.dependency_aware_sampler) and (
            args.dependency_risk_score_threshold <= 0
            or float(scores.median())
            <= float(args.dependency_risk_score_threshold)
        )
        query_risk_probability = None
        if (
            bool(args.dependency_aware_sampler)
            and sampling_model is not None
            and sampling_model.get("risk_models")
        ):
            fold = str(sampling_model["query_to_fold"].get(name, "all"))
            risk_model = sampling_model["risk_models"].get(
                fold, sampling_model["risk_models"].get("all")
            )
            query_features = torch.stack(
                (
                    scores.median().cpu(),
                    score_margins.median().cpu(),
                    keypoint_scores.median(),
                )
            ).float()
            risk_mean = torch.as_tensor(risk_model["feature_mean"]).float()
            risk_scale = (
                torch.as_tensor(risk_model["feature_scale"]).float().clamp_min(1e-6)
            )
            risk_coefficients = torch.as_tensor(
                risk_model["coefficients"]
            ).float()
            risk_logit = (
                ((query_features - risk_mean) / risk_scale) @ risk_coefficients
                + float(risk_model.get("intercept", 0.0))
            )
            query_risk_probability = float(torch.sigmoid(risk_logit))
            use_dependency_sampler = query_risk_probability >= float(
                sampling_model.get("risk_threshold", 0.5)
            )
        start = time.perf_counter()
        pose, inliers, diagnostics = solve_pose(
            keypoints.numpy(),
            xyz[matched_indices].numpy(),
            K.numpy(),
            solver=(
                "poselib_dependency"
                if use_dependency_sampler
                else "poselib"
            ),
            reprojection_error=float(args.reprojection_error),
            confidence=0.99999,
            max_iterations=(
                int(args.dependency_max_iterations)
                if use_dependency_sampler
                else 100000
            ),
            min_iterations=(
                int(args.dependency_min_iterations)
                if use_dependency_sampler
                else 1000
            ),
            scores=pose_sampling_scores.cpu().numpy(),
            ransac_seed=int(args.seed),
            return_diagnostics=True,
            dependency_groups=dependency_groups[matched_indices].numpy(),
            image_cells=cells.numpy(),
            surface_groups=surface_groups[matched_indices].numpy(),
            dependency_guided_mixture=float(args.dependency_guided_mixture),
            dependency_guided_rank_power=float(
                args.dependency_guided_rank_power
            ),
            dependency_rescue_max_iterations=int(
                args.dependency_rescue_max_iterations
            ),
            dependency_rescue_inlier_ratio=float(
                args.dependency_rescue_inlier_ratio
            ),
            ground_truth_w2c=torch.as_tensor(cached["pose_w2c"]).numpy()
            if args.minimal_set_record_limit > 0
            else None,
            minimal_set_record_limit=int(args.minimal_set_record_limit),
            sampling_margins=score_margins.cpu().numpy(),
            sampling_keypoint_scores=keypoint_scores.numpy(),
        )
        ransac_duration = time.perf_counter() - start
        ransac_seconds += ransac_duration
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
                "median_top1_margin": float(score_margins.median()),
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
                "local_refinements": int(
                    diagnostics.get("ransac_local_refinements", 0)
                ),
                "dependency_sampler_used": use_dependency_sampler,
                "dependency_sampler_backend": diagnostics.get(
                    "ransac_backend", "poselib"
                ),
                "query_risk_probability": query_risk_probability,
                "dependency_rescue_used": bool(
                    diagnostics.get("ransac_rescue_used", False)
                ),
                "matching_ms": float(1000 * matching_duration),
                "ransac_ms": float(1000 * ransac_duration),
                "minimal_set_records": diagnostics.get(
                    "minimal_set_records", []
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
                    "top1_margins": score_margins.cpu(),
                    "keypoint_scores": keypoint_scores.cpu(),
                    "gt_reprojection_errors_px": gt_errors,
                    "ransac_inlier_mask": inlier_mask,
                    "clean_inlier_mask": inlier_mask & (gt_errors <= 4),
                    "harmful_inlier_mask": inlier_mask & (gt_errors > 4),
                    "te_cm": float(te),
                    "re_deg": float(re),
                    "hypotheses": diagnostics.get("ransac_actual_hypotheses"),
                    "minimal_set_records": diagnostics.get(
                        "minimal_set_records", []
                    ),
                }
            )
        if len(results) % 50 == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            if args.dynamic_outcomes_output:
                _atomic_torch(
                    dynamic_partial,
                    {
                        "run_identity": run_identity,
                        "records": dynamic_records,
                    },
                )
            _atomic_json(
                partial,
                {
                    "run_identity": run_identity,
                    "results": results,
                },
            )
            print(f"{len(results)}/{len(names)}", flush=True)

    if len(results) != len(names):
        raise RuntimeError("cached replay did not cover every requested query")
    te = np.asarray([row["te_cm"] for row in results])
    re = np.asarray([row["re_deg"] for row in results])
    inlier_ratios = np.asarray(
        [
            row["inlier_count"] / max(row["match_count"], 1)
            for row in results
        ]
    )
    dependency_query_count = sum(
        bool(row.get("dependency_sampler_used", False)) for row in results
    )
    rescue_query_count = sum(
        bool(row.get("dependency_rescue_used", False)) for row in results
    )
    runtime_rows_complete = all(
        "matching_ms" in row and "ransac_ms" in row for row in results
    )
    if runtime_rows_complete:
        matching_ms = float(np.mean([row["matching_ms"] for row in results]))
        ransac_ms = float(np.mean([row["ransac_ms"] for row in results]))
    else:
        matching_ms = float(1000 * matching_seconds / len(results))
        ransac_ms = float(1000 * ransac_seconds / len(results))
    hypotheses = [
        row["hypotheses"] for row in results if row["hypotheses"] is not None
    ]
    summary = {
        "schema": "lafgs_cached_deployment_replay",
        "split": args.split,
        "runtime_scope": "cached_descriptor_matching_and_pnp",
        "feature_extraction_included": False,
        "runtime_rows_complete": runtime_rows_complete,
        "map": str(Path(args.map).resolve()),
        "metric_state": str(Path(args.metric_state).resolve())
        if args.metric_state
        else None,
        "query_count": len(results),
        "anchor_count": int(xyz.shape[0]),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(re)),
        "mean_ae_deg": float(np.mean(re)),
        "recall_5cm_percent": float(100 * np.mean(te <= 5)),
        "recall_5cm_5deg_percent": float(
            100 * np.mean((te <= 5) & (re <= 5))
        ),
        "mean_solver_inlier_ratio_percent": float(100 * np.mean(inlier_ratios)),
        "dependency_sampler_query_count": int(dependency_query_count),
        "dependency_sampler_query_fraction_percent": float(
            100 * dependency_query_count / len(results)
        ),
        "dependency_rescue_query_count": int(rescue_query_count),
        "raw_gt_precision_2px_percent": float(
            100 * np.mean([row["raw_gt_precision_2px"] for row in results])
        ),
        "inlier_gt_precision_2px_percent": float(
            100 * np.mean([row["inlier_gt_precision_2px"] for row in results])
        ),
        "mean_hypotheses": float(np.mean(hypotheses)) if hypotheses else None,
        "median_hypotheses": float(np.median(hypotheses)) if hypotheses else None,
        "p90_hypotheses": float(np.percentile(hypotheses, 90))
        if hypotheses
        else None,
        "matching_ms_per_query": matching_ms,
        "ransac_ms_per_query": ransac_ms,
        "total_ms_per_query": matching_ms + ransac_ms,
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
    dynamic_partial.unlink(missing_ok=True)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
