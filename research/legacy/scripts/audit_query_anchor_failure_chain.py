#!/usr/bin/env python3
"""Fail-closed per-query L1-L5 Anchor failure-chain audit."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from evaluation.query_anchor_failure_chain import (
    classify_failure_chain,
    descriptor_recall_diagnostics,
    geometry_diagnostics,
    grid_coverage,
    maximum_cardinality_minimum_distance_matching,
    nearby_anchor_edges,
    project_gt_visible_anchors,
)
from evidence.tracks import (
    LeaveOneQueryOutProjectiveAnchorDescriptorBank,
    LeaveOneQueryOutTrackDescriptorBank,
)
from localization.localizer import load_shared_metric
from localization.matcher import global_cosine_topk
from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.trainer import bounded_anchor_bank, track_descriptor_payload_for_loo
from scripts.evaluate_rendered_track_fullmap import _DeviceBankUpdater


_SOURCE_PATHS = (
    "scripts/audit_query_anchor_failure_chain.py",
    "evaluation/query_anchor_failure_chain.py",
    "evidence/tracks.py",
    "localization/localizer.py",
    "localization/matcher.py",
    "localization/pose_solver.py",
    "map_learning/trainer.py",
    "scripts/evaluate_rendered_track_fullmap.py",
    "topology/pose_information.py",
)


def _producer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("failure-chain producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in _SOURCE_PATHS
        },
        "torch_version": torch.__version__,
    }


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _atomic_json(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if json.loads(temporary.read_text()).get("schema") != payload.get("schema"):
            raise RuntimeError("temporary failure-chain report did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch(payload: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary failure-chain sidecar did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _declared_sha_matches(container: dict, key: str, actual: str, label: str) -> None:
    declared = container.get(key)
    if declared is not None and str(declared) != actual:
        raise ValueError(f"{label} declared SHA differs from the selected input")


def _matching_for_edge_subset(
    query_edges: np.ndarray,
    anchor_edges: np.ndarray,
    distance_edges: np.ndarray,
    *,
    query_count: int,
    anchor_mask: np.ndarray | None = None,
):
    keep = np.ones(anchor_edges.size, dtype=bool)
    if anchor_mask is not None:
        keep &= np.asarray(anchor_mask, dtype=bool)[anchor_edges]
    return maximum_cardinality_minimum_distance_matching(
        query_edges[keep], anchor_edges[keep], distance_edges[keep],
        query_count=query_count,
    )


def _topk_positive_matching(
    top_indices: np.ndarray,
    positive_query_rows: np.ndarray,
    positive_anchor_rows: np.ndarray,
    *,
    k: int,
):
    positive_codes = set(
        zip(positive_query_rows.tolist(), positive_anchor_rows.tolist())
    )
    query = []
    anchor = []
    for row in range(top_indices.shape[0]):
        for candidate in top_indices[row, : int(k)].tolist():
            if (row, int(candidate)) in positive_codes:
                query.append(row)
                anchor.append(int(candidate))
    return maximum_cardinality_minimum_distance_matching(
        np.asarray(query, dtype=np.int64), np.asarray(anchor, dtype=np.int64),
        np.zeros(len(query), dtype=np.float64), query_count=top_indices.shape[0],
    )


def _covariance_summary(state: dict, payload: dict, selected: np.ndarray) -> dict:
    track_ids = torch.as_tensor(state["track_cluster_ids"]).long().numpy()[selected]
    track = track_ids >= 0
    covariance = payload.get("track_geometry", {}).get("triangulation_covariance_matrix")
    if covariance is None:
        return {
            "track_anchor_count": int(track.sum()),
            "track_covariance_available": False,
            "surface_completion_covariance_available": False,
        }
    covariance = torch.as_tensor(covariance).double().numpy()
    if track.any() and int(track_ids[track].max()) >= covariance.shape[0]:
        raise ValueError("selected Track Anchor references covariance outside payload")
    matrices = covariance[track_ids[track]] if track.any() else np.empty((0, 3, 3))
    traces = np.trace(matrices, axis1=1, axis2=2) if matrices.size else np.empty(0)
    finite = np.isfinite(matrices).all(axis=(1, 2)) if matrices.size else np.empty(0, bool)
    return {
        "track_anchor_count": int(track.sum()),
        "track_covariance_available": True,
        "track_covariance_finite_count": int(finite.sum()),
        "track_covariance_trace_median": (
            float(np.median(traces[finite])) if finite.any() else None
        ),
        "surface_completion_anchor_count": int((~track).sum()),
        "surface_completion_covariance_available": False,
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    if int(args.cpu_threads) <= 0:
        raise ValueError("CPU thread count must be positive")
    if int(args.topk) != 32:
        raise ValueError("failure-chain descriptor audit fixes Top-32")
    torch.set_num_threads(int(args.cpu_threads))
    output = args.output.resolve()
    sidecar_path = output.with_suffix(".sidecar.pt")
    if output.exists() or sidecar_path.exists():
        raise FileExistsError("failure-chain output already exists")
    identity = _producer_identity()
    paths = {
        "map": args.map.resolve(),
        "metric": args.metric_state.resolve(),
        "track_payload": args.track_payload.resolve(),
        "teacher": args.teacher.resolve(),
        "query_cache": args.query_cache.resolve(),
        "scene_calibration": args.scene_calibration.resolve(),
    }
    expected = {
        "map": args.expected_map_sha256,
        "metric": args.expected_metric_sha256,
        "track_payload": args.expected_track_payload_sha256,
        "teacher": args.expected_teacher_sha256,
        "query_cache": args.expected_query_cache_sha256,
        "scene_calibration": args.expected_scene_calibration_sha256,
    }
    input_sha = {
        name: _require_sha(path, expected[name], name) for name, path in paths.items()
    }
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    metric_state = torch.load(paths["metric"], map_location="cpu", weights_only=False)
    payload = torch.load(paths["track_payload"], map_location="cpu", weights_only=False)
    teacher = torch.load(paths["teacher"], map_location="cpu", weights_only=False)
    cache_payload = torch.load(
        paths["query_cache"], map_location="cpu", weights_only=False
    )
    calibration = json.loads(paths["scene_calibration"].read_text())
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("failure chain requires the unified materialized Anchor map")
    if payload.get("rendered_rgb_only") is not True:
        raise ValueError("failure chain requires rendered source-image-free Track payload")
    uses_test = bool(cache_payload.get("uses_test_queries", False))
    if uses_test and not bool(args.allow_test_diagnostic):
        raise ValueError("test query cache requires --allow-test-diagnostic")
    if uses_test and cache_payload.get("frozen_after_mapping") is not True:
        raise ValueError("test diagnostics require an explicitly frozen mapping artifact")
    if cache_payload.get("uses_source_mapping_rgb") is not False:
        raise ValueError("this audit requires the source-image-free query contract")
    names = list(payload["query_names"])
    cache = cache_payload.get("queries", cache_payload)
    if names != list(teacher.get("query_names", ())) or names != list(cache):
        raise ValueError("payload, teacher, and cache query registries differ")
    anchor_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    if int(teacher.get("anchor_count", -1)) != anchor_count:
        raise ValueError("teacher and map Anchor registries differ")
    if (
        metric_state.get("map_path") != str(paths["map"])
        or metric_state.get("map_sha256") != input_sha["map"]
    ):
        raise ValueError("metric is not SHA-bound to the selected map")
    _declared_sha_matches(teacher, "query_cache_sha256", input_sha["query_cache"], "teacher")
    _declared_sha_matches(teacher, "scene_calibration_sha256", input_sha["scene_calibration"], "teacher")
    provenance = state.get("provenance", {})
    _declared_sha_matches(provenance, "query_cache_sha256", input_sha["query_cache"], "map")
    _declared_sha_matches(provenance, "track_payload_sha256", input_sha["track_payload"], "map")
    parameters = calibration.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("scene calibration lacks parameters")
    positive_radius = float(parameters["positive_radius_px"])
    detector_radius = (
        positive_radius if args.detector_radius_px is None
        else float(args.detector_radius_px)
    )
    if detector_radius <= 0 or positive_radius <= 0:
        raise ValueError("detector and positive radii must be positive")

    selected_names = list(args.query_name)
    if not selected_names:
        raise ValueError("select at least one explicit frozen query name")
    if len(set(selected_names)) != len(selected_names):
        raise ValueError("query selection contains duplicates")
    missing = sorted(set(selected_names) - set(names))
    if missing:
        raise ValueError(f"selected query is absent from registry: {missing[:3]}")
    name_to_index = {name: index for index, name in enumerate(names)}

    loo_payload = track_descriptor_payload_for_loo(payload)
    raw_reference = torch.as_tensor(
        state.get("v7_metric_raw_features", state["anchor_features"])
    ).float()
    if bool((torch.as_tensor(state["track_cluster_ids"]) < 0).any()):
        replay = LeaveOneQueryOutProjectiveAnchorDescriptorBank(
            state=state, payload=loo_payload, query_cache=cache_payload,
            reference_features=raw_reference,
            trim_fraction=float(args.descriptor_trim_fraction),
        )
    else:
        replay = LeaveOneQueryOutTrackDescriptorBank(
            payload=loo_payload, query_cache=cache_payload,
            track_indices=state["track_cluster_ids"],
            reference_features=raw_reference,
            trim_fraction=float(args.descriptor_trim_fraction),
        )
    device = torch.device(args.device)
    online_config = state.get("v7_online_metric", {}).get("config", {})
    updater = _DeviceBankUpdater(
        replay, device, metric_state=metric_state,
        adapted_reference_features=state["anchor_features"],
        anchor_residual_parameter=state.get("v7_anchor_residual_parameter"),
        anchor_residual_max_norm=float(
            online_config.get("anchor_feature_residual_max_norm", 0.0)
        ),
    )
    metric = load_shared_metric(
        paths["metric"], anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    bank, _, _ = bounded_anchor_bank(
        metric, F.normalize(raw_reference, dim=1).to(device),
        (
            None if state.get("v7_anchor_residual_parameter") is None
            else torch.as_tensor(state["v7_anchor_residual_parameter"]).float().to(device)
        ),
        float(online_config.get("anchor_feature_residual_max_norm", 0.0)),
    )
    expected_bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    if not torch.allclose(bank, expected_bank, atol=1e-6, rtol=1e-6):
        raise ValueError("deployment bank replay differs from frozen map")
    bank.copy_(expected_bank)

    xyz = torch.as_tensor(state["anchor_xyz"]).float().numpy()
    anchor_type = torch.as_tensor(state["anchor_type"]).long().numpy()
    support_weight = torch.as_tensor(state["anchor_surface_support_weight"]).float().numpy()
    component_ids = torch.as_tensor(state["coarse_dependency_group_ids"]).long().numpy()
    query_reports = []
    sidecar_queries = []
    for completed, name in enumerate(selected_names, start=1):
        query_index = name_to_index[name]
        updater(query_index, bank)
        cached = cache[name]
        record = teacher["records"][query_index]
        intrinsic = torch.as_tensor(cached["native_K"]).float().numpy()
        gt_pose = torch.as_tensor(cached["pose_w2c"]).float().numpy()
        height, width = torch.as_tensor(cached["native_input_hw"]).long().tolist()
        if float(cached.get("pixel_center_offset", 0.5)) != 0.5:
            raise ValueError("failure chain supports only the frozen +0.5 pixel contract")
        projected = project_gt_visible_anchors(
            xyz, intrinsic, gt_pose, image_size=(width, height),
            rendered_alpha=cached.get("native_rendered_alpha"),
            rendered_depth=cached.get("native_rendered_depth"),
            alpha_minimum=float(teacher["config"]["alpha_minimum"]),
            depth_abs_tolerance_m=float(teacher["config"]["depth_abs_tolerance_m"]),
            depth_relative_tolerance=float(teacher["config"]["depth_relative_tolerance"]),
            depth_policy=(
                "audit_only"
                if teacher["config"].get("exact_depth_policy")
                == "audit_only_never_hard_reject"
                else "hard"
            ),
        )
        visible_rows = np.flatnonzero(projected.visible)
        visible_geometry = geometry_diagnostics(
            projected.uv[visible_rows], xyz[visible_rows], intrinsic, gt_pose,
            image_size=(width, height), seed=int(args.seed) + query_index,
        )
        all_keypoints = (
            torch.as_tensor(cached["native_keypoints"]).float() + 0.5
        ).numpy()
        valid_keypoints = torch.as_tensor(
            cached.get("native_valid_keypoint_mask", torch.ones(len(all_keypoints)))
        ).bool().numpy()
        detector_rows = np.flatnonzero(valid_keypoints)
        det_q, det_a, det_d = nearby_anchor_edges(
            all_keypoints[detector_rows], projected.uv, projected.visible,
            radius_px=detector_radius,
        )
        detector_matching = _matching_for_edge_subset(
            det_q, det_a, det_d, query_count=detector_rows.size,
        )
        detector_accessible = np.unique(det_a)

        deployment_rows = torch.as_tensor(record["query_rows"]).long()
        keypoints = all_keypoints[deployment_rows.numpy()]
        pos_q, pos_a, pos_d = nearby_anchor_edges(
            keypoints, projected.uv, projected.visible, radius_px=positive_radius,
        )
        oracle_matching = _matching_for_edge_subset(
            pos_q, pos_a, pos_d, query_count=len(keypoints),
        )
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[deployment_rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptors)
        topk = global_cosine_topk(
            adapted, bank, topk=32, anchor_descriptors_normalized=True
        )
        dense_scores = (adapted @ bank.T).cpu().numpy()
        top_indices = topk.anchor_indices.cpu().numpy()
        descriptor = descriptor_recall_diagnostics(
            dense_scores, top_indices, pos_q, pos_a, anchor_type,
        )
        top32_matching = _topk_positive_matching(
            top_indices, pos_q, pos_a, k=32,
        )

        oracle_geometry = geometry_diagnostics(
            keypoints[oracle_matching.query_rows], xyz[oracle_matching.anchor_rows],
            intrinsic, gt_pose, image_size=(width, height),
            seed=int(args.seed) + query_index,
        )
        solve_kwargs = {
            "reprojection_error_px": float(parameters["ransac_reprojection_px"]),
            "confidence": 0.99999, "max_iterations": 100000,
            "min_iterations": 1000, "seed": int(args.seed),
        }
        oracle_estimate = solve_absolute_pose(
            keypoints[oracle_matching.query_rows], xyz[oracle_matching.anchor_rows],
            intrinsic, **solve_kwargs,
        )
        oracle_ae, oracle_te = pose_error(oracle_estimate.pose_w2c, gt_pose)
        top1 = top_indices[:, 0]
        deployed_estimate = solve_absolute_pose(
            keypoints, xyz[top1], intrinsic, **solve_kwargs,
        )
        deployed_ae, deployed_te = pose_error(deployed_estimate.pose_w2c, gt_pose)
        task_correct = lambda ae, te: (
            te <= 100.0 * float(parameters["task_translation_m"])
            and ae <= float(parameters["task_rotation_deg"])
        )
        oracle_correct = task_correct(oracle_ae, oracle_te)
        deployed_correct = task_correct(deployed_ae, deployed_te)
        category = classify_failure_chain(
            visible_geometry_solvable=bool(visible_geometry["pnp_geometry_solvable"]),
            detector_matching_rank=detector_matching.rank,
            oracle_matching_rank=oracle_matching.rank,
            top32_positive_matching_rank=top32_matching.rank,
            oracle_pose_correct=oracle_correct,
            deployed_pose_correct=deployed_correct,
        )
        visible_track = visible_rows[anchor_type[visible_rows] == 1]
        visible_surface = visible_rows[anchor_type[visible_rows] == 0]
        query_reports.append(
            {
                "query_index": query_index,
                "image_name": name,
                "failure_category": category,
                "l1_map_coverage": {
                    "in_frame_anchor_count": int(projected.in_frame.sum()),
                    "gt_visible_anchor_count": int(visible_rows.size),
                    "gt_visible_track_count": int(visible_track.size),
                    "gt_visible_surface_completion_count": int(visible_surface.size),
                    "alpha_rejected_in_frame_count": int(
                        (projected.in_frame & ~projected.alpha_supported).sum()
                    ),
                    "depth_rejected_in_frame_count": int(
                        (projected.in_frame & ~projected.depth_supported).sum()
                    ),
                    "depth_visibility_policy": teacher["config"].get(
                        "exact_depth_policy", "hard"
                    ),
                    "grid_coverage_4x4": grid_coverage(
                        projected.uv[visible_rows], (width, height)
                    ),
                    "support_component_count": int(
                        np.unique(component_ids[visible_rows]).size
                    ),
                    "support_weight": {
                        "mean": float(support_weight[visible_rows].mean())
                        if visible_rows.size else None,
                        "minimum": float(support_weight[visible_rows].min())
                        if visible_rows.size else None,
                    },
                    "covariance": _covariance_summary(state, payload, visible_rows),
                    "geometry": visible_geometry,
                },
                "l2_detector_access": {
                    "valid_detector_row_count": int(detector_rows.size),
                    "detector_accessible_anchor_count": int(detector_accessible.size),
                    "detector_accessible_track_count": int(
                        (anchor_type[detector_accessible] == 1).sum()
                    ),
                    "detector_accessible_surface_completion_count": int(
                        (anchor_type[detector_accessible] == 0).sum()
                    ),
                    "detector_recall_percent": (
                        100.0 * detector_accessible.size / visible_rows.size
                        if visible_rows.size else 0.0
                    ),
                    "maximum_matching_rank": detector_matching.rank,
                    "maximum_matching_grid_coverage_4x4": grid_coverage(
                        all_keypoints[detector_rows[detector_matching.query_rows]],
                        (width, height),
                    ),
                },
                "l3_descriptor_recall": {
                    **descriptor,
                    "geometric_positive_edge_count": int(pos_q.size),
                    "top32_positive_maximum_matching_rank": top32_matching.rank,
                },
                "l4_oracle_geometry": {
                    "gt_positive_maximum_matching_rank": oracle_matching.rank,
                    "gt_positive_edge_count": int(pos_q.size),
                    "matched_track_count": int(
                        (anchor_type[oracle_matching.anchor_rows] == 1).sum()
                    ),
                    "matched_surface_completion_count": int(
                        (anchor_type[oracle_matching.anchor_rows] == 0).sum()
                    ),
                    "matched_support_component_count": int(
                        np.unique(component_ids[oracle_matching.anchor_rows]).size
                    ),
                    "geometry": oracle_geometry,
                    "oracle_pose": {
                        "te_cm": oracle_te, "ae_deg": oracle_ae,
                        "correct_5cm_5deg": oracle_correct,
                        "inlier_count": int(oracle_estimate.inliers.size),
                        "hypotheses": int(
                            oracle_estimate.diagnostics.get("iterations", 0)
                        ),
                    },
                },
                "l5_solver_gap": {
                    "deployed_top1_pose": {
                        "te_cm": deployed_te, "ae_deg": deployed_ae,
                        "correct_5cm_5deg": deployed_correct,
                        "inlier_count": int(deployed_estimate.inliers.size),
                        "hypotheses": int(
                            deployed_estimate.diagnostics.get("iterations", 0)
                        ),
                    },
                    "oracle_correspondences_solvable_but_poselib_failed": bool(
                        oracle_geometry["pnp_geometry_solvable"] and not oracle_correct
                    ),
                    "solver_parameters_changed": False,
                },
            }
        )
        sidecar_queries.append(
            {
                "query_index": query_index, "image_name": name,
                "deployment_rows": deployment_rows.clone(),
                "top32_anchor_rows": topk.anchor_indices.cpu().clone(),
                "top32_scores": topk.scores.cpu().clone(),
                "gt_positive_query_rows": torch.from_numpy(pos_q),
                "gt_positive_anchor_rows": torch.from_numpy(pos_a),
                "gt_positive_distances_px": torch.from_numpy(pos_d),
                "oracle_matched_query_rows": torch.from_numpy(oracle_matching.query_rows),
                "oracle_matched_anchor_rows": torch.from_numpy(oracle_matching.anchor_rows),
            }
        )
        print(json.dumps({"event": "query_failure_chain", "complete": completed,
                          "count": len(selected_names), "image_name": name,
                          "category": category}), flush=True)

    sidecar = {
        "schema": "lafgs_query_anchor_failure_chain_sidecar",
        "version": 1,
        "uses_test_queries": uses_test,
        "input_sha256": input_sha,
        "queries": sidecar_queries,
    }
    _atomic_torch(sidecar, sidecar_path)
    categories = sorted({row["failure_category"] for row in query_reports})
    report = {
        "schema": "lafgs_query_anchor_failure_chain_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": uses_test,
        "mapping_only_development": not uses_test,
        "test_diagnostic_is_read_only": bool(uses_test),
        "results_may_not_update_map_or_hyperparameters": True,
        "producer": identity,
        "inputs": {name: str(path) for name, path in paths.items()},
        "input_sha256": input_sha,
        "configuration": {
            "query_names": selected_names,
            "pixel_center_offset": 0.5,
            "visibility_sampling": "nearest_pixel_round_matching_frozen_teacher",
            "detector_radius_px": detector_radius,
            "positive_radius_px": positive_radius,
            "descriptor_ks": [1, 2, 4, 8, 16, 32],
            "leave_one_mapping_query_descriptor_out": not uses_test,
            "solver": "unchanged_standard_poselib",
            "seed": int(args.seed),
        },
        "summary": {
            "query_count": len(query_reports),
            "failure_category_counts": {
                category: sum(row["failure_category"] == category for row in query_reports)
                for category in categories
            },
        },
        "queries": query_reports,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "decision": "DIAGNOSTIC_ONLY_NO_MAP_OR_SOLVER_CHANGE",
    }
    for name, path in paths.items():
        _require_sha(path, input_sha[name], name)
    if _producer_identity() != identity:
        raise RuntimeError("failure-chain producer identity changed")
    _atomic_json(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--expected-metric-sha256", required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--expected-track-payload-sha256", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--expected-scene-calibration-sha256", required=True)
    parser.add_argument("--query-name", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-radius-px", type=float)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--allow-test-diagnostic", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args)["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
