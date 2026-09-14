#!/usr/bin/env python3
"""Cross-fitted surface-bounded refinement of Gaussian reserve geometry."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from localization.pose_solver import solve_absolute_pose
from map_learning.observations import build_teacher
from priors.models import gaussian_model
from topology.deployment_revision import collect_deployment_statistics


def temporal_threeway_split(
    query_names: list[str], block_count: int = 9
) -> tuple[list[int], list[int], list[int], dict]:
    """Create disjoint fit, point-validation, and pose-gate trajectory blocks."""
    if int(block_count) < 3:
        raise ValueError("three-way geometry split needs at least three blocks")
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(query_names):
        sequence = str(name).rsplit("/", 1)[0] if "/" in str(name) else ""
        by_sequence[sequence].append(index)
    partitions = ([], [], [])
    assignments = {}
    for sequence, indices in sorted(by_sequence.items()):
        ordered = sorted(indices, key=lambda index: str(query_names[index]))
        for rank, query in enumerate(ordered):
            block = min(
                int(rank * int(block_count) / max(len(ordered), 1)),
                int(block_count) - 1,
            )
            partition = block % 3
            partitions[partition].append(query)
            assignments[str(query_names[query])] = {
                "block": block,
                "partition": ("fit", "point_validation", "pose_gate")[partition],
            }
    if any(not values for values in partitions):
        raise ValueError("three-way geometry split produced an empty partition")
    fit, validation, gate = (sorted(values) for values in partitions)
    return (
        fit,
        validation,
        gate,
        {
            "policy": "per_sequence_alternating_contiguous_threeway_blocks",
            "block_count": int(block_count),
            "fit_query_count": len(fit),
            "point_validation_query_count": len(validation),
            "pose_gate_query_count": len(gate),
            "assignments": assignments,
            "uses_test_queries": False,
        },
    )


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    value /= max(float(np.linalg.norm(value)), 1e-12)
    w, x, y, z = value
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _project_point(
    point: np.ndarray, poses_w2c: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    camera = np.einsum("nij,j->ni", poses_w2c[:, :3, :3], point) + poses_w2c[:, :3, 3]
    projected = np.einsum("nij,nj->ni", intrinsics, camera)
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)


def reprojection_errors(
    point: np.ndarray,
    poses_w2c: np.ndarray,
    intrinsics: np.ndarray,
    observations_uv: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm(
        _project_point(point, poses_w2c, intrinsics) - observations_uv, axis=1
    )


def fit_surface_bounded_point(
    *,
    initial_xyz: np.ndarray,
    surface_center: np.ndarray,
    surface_basis: np.ndarray,
    local_bounds: np.ndarray,
    poses_w2c: np.ndarray,
    intrinsics: np.ndarray,
    observations_uv: np.ndarray,
    prior_sigma: np.ndarray,
    huber_px: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit one point in Gaussian-local coordinates with a trust-region prior."""
    center = np.asarray(surface_center, dtype=np.float64).reshape(3)
    basis = np.asarray(surface_basis, dtype=np.float64).reshape(3, 3)
    bounds = np.asarray(local_bounds, dtype=np.float64).reshape(3)
    sigma = np.asarray(prior_sigma, dtype=np.float64).reshape(3)
    local_initial = basis.T @ (np.asarray(initial_xyz, dtype=np.float64) - center)
    lower, upper = -bounds, bounds
    local_initial = np.clip(local_initial, lower + 1e-9, upper - 1e-9)

    def residual(local: np.ndarray) -> np.ndarray:
        point = center + basis @ local
        pixel = (
            _project_point(point, poses_w2c, intrinsics) - observations_uv
        ).reshape(-1)
        prior = (local - local_initial) / np.maximum(sigma, 1e-8)
        return np.concatenate((pixel, prior))

    result = least_squares(
        residual,
        local_initial,
        bounds=(lower, upper),
        loss="huber",
        f_scale=max(float(huber_px), 0.5),
        max_nfev=100,
        x_scale="jac",
    )
    point = center + basis @ result.x
    pixel_residual = reprojection_errors(point, poses_w2c, intrinsics, observations_uv)
    pixel_variance = max(float(np.median(pixel_residual**2)), 0.25)
    normal = result.jac.T @ result.jac
    covariance_local = np.linalg.pinv(normal + np.eye(3) * 1e-8) * pixel_variance
    covariance_world = basis @ covariance_local @ basis.T
    return (
        point,
        covariance_world,
        {
            "success": bool(result.success),
            "cost": float(result.cost),
            "function_evaluations": int(result.nfev),
            "displacement_m": float(np.linalg.norm(point - initial_xyz)),
        },
    )


def _positive_observations(
    *,
    state: dict,
    teacher: dict,
    query_cache: dict,
    query_indices: list[int],
) -> dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    """Select one closest legal positive per anchor and mapping view."""
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    xyz = torch.as_tensor(state["anchor_xyz"]).float().numpy()
    reserve = torch.as_tensor(state["anchor_type"]).long() == 0
    output: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    for query in query_indices:
        record = teacher["records"][query]
        cached = cache[names[query]]
        rows = torch.as_tensor(record["query_rows"]).long()
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()[rows]
        keypoints += float(cached.get("pixel_center_offset", 0.5))
        pose = np.asarray(cached["pose_w2c"], dtype=np.float64)
        intrinsic = np.asarray(cached["native_K"], dtype=np.float64)
        by_anchor: dict[int, list[int]] = defaultdict(list)
        offsets = torch.as_tensor(record["positive_offsets"]).long()
        indices = torch.as_tensor(record["positive_indices"]).long()
        for local in range(rows.numel()):
            for anchor in indices[offsets[local] : offsets[local + 1]].tolist():
                if bool(reserve[int(anchor)]):
                    by_anchor[int(anchor)].append(local)
        for anchor, local_rows in by_anchor.items():
            candidates = keypoints[torch.as_tensor(local_rows)].numpy()
            projected = _project_point(xyz[anchor], pose[None], intrinsic[None])[0]
            best = int(np.argmin(np.linalg.norm(candidates - projected, axis=1)))
            output[anchor].append(
                (query, pose, intrinsic, candidates[best].astype(np.float64))
            )
    return output


@torch.inference_mode()
def _clean_ransac_observations(
    *,
    state: dict,
    metric_state_path: str | Path,
    teacher: dict,
    query_cache: dict,
    query_indices: list[int],
    device: torch.device,
    ransac_reprojection_px: float,
    clean_reprojection_px: float,
    seed: int,
) -> dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    """Collect reserve observations that survive the deployed final inlier set."""
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    reserve = torch.as_tensor(state["anchor_type"]).long() == 0
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    metric = load_shared_metric(
        metric_state_path,
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    output: dict[int, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    for query in query_indices:
        record = teacher["records"][query]
        cached = cache[names[query]]
        rows = torch.as_tensor(record["query_rows"]).long()
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptors)
        winners = torch.argmax(adapted @ bank.T, dim=1).cpu()
        keypoints = torch.as_tensor(cached["native_keypoints"]).float()[rows]
        keypoints += float(cached.get("pixel_center_offset", 0.5))
        pose = np.asarray(cached["pose_w2c"], dtype=np.float64)
        intrinsic = np.asarray(cached["native_K"], dtype=np.float64)
        estimate = solve_absolute_pose(
            keypoints.numpy(),
            xyz[winners].numpy(),
            intrinsic,
            reprojection_error_px=float(ransac_reprojection_px),
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            seed=int(seed),
        )
        inliers = torch.as_tensor(estimate.inliers).long().reshape(-1)
        if not inliers.numel():
            continue
        winner_xyz = xyz[winners[inliers]].numpy()
        projected = np.stack(
            [
                _project_point(point, pose[None], intrinsic[None])[0]
                for point in winner_xyz
            ]
        )
        errors = np.linalg.norm(projected - keypoints[inliers].numpy(), axis=1)
        by_anchor: dict[int, tuple[float, int]] = {}
        for local, error in zip(inliers.tolist(), errors.tolist()):
            anchor = int(winners[local])
            if not bool(reserve[anchor]) or float(error) > float(clean_reprojection_px):
                continue
            positives = torch.as_tensor(record["positive_indices"])[
                int(torch.as_tensor(record["positive_offsets"])[local]) :
                int(torch.as_tensor(record["positive_offsets"])[local + 1])
            ]
            if not bool((positives == anchor).any()):
                continue
            if anchor not in by_anchor or float(error) < by_anchor[anchor][0]:
                by_anchor[anchor] = (float(error), local)
        for anchor, (_, local) in by_anchor.items():
            output[anchor].append(
                (
                    query,
                    pose,
                    intrinsic,
                    keypoints[local].numpy().astype(np.float64),
                )
            )
    return output


def _observation_arrays(values):
    return (
        np.stack([value[1] for value in values]),
        np.stack([value[2] for value in values]),
        np.stack([value[3] for value in values]),
    )


def refine_reserve_geometry(
    *,
    state: dict,
    teacher: dict,
    query_cache: dict,
    track_payload: dict,
    gaussian_ply: str | Path,
    gaussian_type: str,
    sh_degree: int,
    fit_queries: list[int],
    validation_queries: list[int],
    minimum_fit_views: int,
    minimum_validation_views: int,
    minimum_fit_improvement: float,
    maximum_validation_degradation: float,
    evidence_mode: str = "legal_positive",
    metric_state_path: str | Path | None = None,
    device: torch.device | None = None,
    ransac_reprojection_px: float = 12.0,
    clean_reprojection_px: float = 4.0,
    seed: int = 2026,
) -> tuple[dict, dict]:
    evidence_mode = str(evidence_mode)
    if evidence_mode == "legal_positive":
        builder = lambda indices: _positive_observations(
            state=state,
            teacher=teacher,
            query_cache=query_cache,
            query_indices=indices,
        )
    elif evidence_mode == "clean_ransac_inlier":
        if metric_state_path is None or device is None:
            raise ValueError("clean RANSAC geometry needs metric state and device")
        builder = lambda indices: _clean_ransac_observations(
            state=state,
            metric_state_path=metric_state_path,
            teacher=teacher,
            query_cache=query_cache,
            query_indices=indices,
            device=device,
            ransac_reprojection_px=ransac_reprojection_px,
            clean_reprojection_px=clean_reprojection_px,
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported reserve geometry evidence mode: {evidence_mode}")
    fit = builder(fit_queries)
    validation = builder(validation_queries)
    model = gaussian_model(gaussian_type, sh_degree)
    model.target_device = torch.device("cpu")
    model.load_ply(gaussian_ply)
    centers = model.get_xyz.detach().cpu().double().numpy()
    scales = model.get_scaling.detach().cpu().double().numpy()
    rotations = model.get_rotation.detach().cpu().double().numpy()

    output = dict(state)
    xyz = torch.as_tensor(state["anchor_xyz"]).float().clone()
    anchor_type = torch.as_tensor(state["anchor_type"]).long()
    count = int(xyz.shape[0])
    covariance = torch.zeros((count, 3, 3), dtype=torch.float32)
    payload_geometry = track_payload["track_geometry"]
    track_rows = torch.nonzero(anchor_type != 0, as_tuple=False).reshape(-1)
    track_ids = torch.as_tensor(state["track_cluster_ids"])[track_rows].long()
    if "triangulation_covariance_matrix" in payload_geometry:
        covariance[track_rows] = torch.as_tensor(
            payload_geometry["triangulation_covariance_matrix"]
        ).float()[track_ids]
    else:
        trace = torch.as_tensor(
            payload_geometry["triangulation_covariance_trace"]
        ).float()[track_ids]
        covariance[track_rows] = torch.diag_embed((trace / 3.0)[:, None].expand(-1, 3))

    parameters = state["track_centric_reconstruction"]["calibration"]["parameters"]
    tangent_cap = float(parameters["surface_max_distance_m"])
    normal_cap = float(parameters["surface_point_plane_m"])
    clean_px = float(parameters["clean_radius_px"])
    source = torch.as_tensor(state["source_primitive_ids"]).long().numpy()
    accepted = []
    diagnostics = []
    reserve_rows = torch.nonzero(anchor_type == 0, as_tuple=False).reshape(-1).tolist()
    for anchor in reserve_rows:
        primitive = int(source[anchor])
        basis = quaternion_wxyz_to_matrix(rotations[primitive])
        primitive_scale = scales[primitive]
        if gaussian_type == "2dgs":
            bounds = np.asarray(
                [
                    min(tangent_cap, 2.0 * float(primitive_scale[0])),
                    min(tangent_cap, 2.0 * float(primitive_scale[1])),
                    normal_cap,
                ]
            )
        else:
            bounds = np.minimum(tangent_cap, 2.0 * primitive_scale[:3])
        bounds = np.maximum(bounds, 1e-4)
        prior_sigma = np.maximum(bounds / 3.0, 1e-4)
        covariance[anchor] = torch.from_numpy(
            basis @ np.diag(prior_sigma**2) @ basis.T
        ).float()
        fit_values = fit.get(anchor, [])
        validation_values = validation.get(anchor, [])
        if len(fit_values) < int(minimum_fit_views) or len(validation_values) < int(
            minimum_validation_views
        ):
            continue
        fit_pose, fit_K, fit_uv = _observation_arrays(fit_values)
        val_pose, val_K, val_uv = _observation_arrays(validation_values)
        initial = xyz[anchor].double().numpy()
        before_fit = reprojection_errors(initial, fit_pose, fit_K, fit_uv)
        before_val = reprojection_errors(initial, val_pose, val_K, val_uv)
        point, point_covariance, solver = fit_surface_bounded_point(
            initial_xyz=initial,
            surface_center=centers[primitive],
            surface_basis=basis,
            local_bounds=bounds,
            poses_w2c=fit_pose,
            intrinsics=fit_K,
            observations_uv=fit_uv,
            prior_sigma=prior_sigma,
            huber_px=clean_px,
        )
        after_fit = reprojection_errors(point, fit_pose, fit_K, fit_uv)
        after_val = reprojection_errors(point, val_pose, val_K, val_uv)
        fit_ratio = float(after_fit.mean() / max(before_fit.mean(), 1e-8))
        validation_ratio = float(after_val.mean() / max(before_val.mean(), 1e-8))
        accept = bool(
            solver["success"]
            and fit_ratio <= 1.0 - float(minimum_fit_improvement)
            and validation_ratio <= 1.0 + float(maximum_validation_degradation)
        )
        diagnostics.append(
            {
                "anchor_row": anchor,
                "source_primitive_id": primitive,
                "fit_views": len(fit_values),
                "validation_views": len(validation_values),
                "fit_error_before_px": float(before_fit.mean()),
                "fit_error_after_px": float(after_fit.mean()),
                "validation_error_before_px": float(before_val.mean()),
                "validation_error_after_px": float(after_val.mean()),
                "fit_error_ratio": fit_ratio,
                "validation_error_ratio": validation_ratio,
                "accepted": accept,
                **solver,
            }
        )
        if accept:
            xyz[anchor] = torch.from_numpy(point).float()
            covariance[anchor] = torch.from_numpy(point_covariance).float()
            accepted.append(anchor)
    output["anchor_xyz"] = xyz
    output["anchor_position_covariance"] = covariance
    output["track_centric_reconstruction"] = {
        **state["track_centric_reconstruction"],
        "reserve_geometry_refinement": {
            "fit_split": "mapping_temporal_fit_blocks",
            "point_validation_split": "mapping_temporal_validation_blocks",
            "accepted_anchor_count": len(accepted),
            "candidate_anchor_count": len(diagnostics),
            "surface_constraint": "source_gaussian_local_frame",
            "evidence_mode": evidence_mode,
            "uses_test_queries": False,
        },
    }
    output["provenance"] = {
        **state.get("provenance", {}),
        "reserve_geometry_refinement": {
            "gaussian_ply": str(Path(gaussian_ply).resolve()),
            "uses_test_queries": False,
        },
    }
    return output, {
        "accepted_anchor_rows": accepted,
        "candidate_anchor_count": len(diagnostics),
        "diagnostics": diagnostics,
    }


def _gate(before: dict, after: dict) -> dict:
    source, revised = before["summary"], after["summary"]
    checks = {
        "median_non_degraded": revised["median_te_cm"] <= 1.02 * source["median_te_cm"],
        "mean_non_degraded": revised["mean_te_cm"] <= 1.02 * source["mean_te_cm"],
        "p90_non_degraded": revised["p90_te_cm"] <= 1.02 * source["p90_te_cm"],
        "cvar95_non_degraded": revised["cvar95_te_cm"] <= 1.02 * source["cvar95_te_cm"],
        "catastrophic_non_degraded": revised["catastrophic_100cm_count"]
        <= source["catastrophic_100cm_count"],
        "raw_precision_non_degraded": revised["raw_gt_precision_percent"] + 0.05
        >= source["raw_gt_precision_percent"],
    }
    checks["meaningful_improvement"] = bool(
        revised["mean_te_cm"] <= 0.995 * source["mean_te_cm"]
        or revised["p90_te_cm"] <= 0.995 * source["p90_te_cm"]
        or revised["cvar95_te_cm"] <= 0.995 * source["cvar95_te_cm"]
        or revised["raw_gt_precision_percent"]
        >= source["raw_gt_precision_percent"] + 0.01
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--raster-provenance", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ransac-reprojection-px", type=float, required=True)
    parser.add_argument("--clean-reprojection-px", type=float, required=True)
    parser.add_argument("--crossfit-blocks", type=int, default=9)
    parser.add_argument("--minimum-fit-views", type=int, default=3)
    parser.add_argument("--minimum-validation-views", type=int, default=2)
    parser.add_argument("--minimum-fit-improvement", type=float, default=0.01)
    parser.add_argument("--maximum-validation-degradation", type=float, default=0.01)
    parser.add_argument(
        "--evidence-mode",
        choices=("legal_positive", "clean_ransac_inlier"),
        default="legal_positive",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_state = torch.load(args.metric_state, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    query_cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    track_payload = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    provenance = torch.load(
        args.raster_provenance, map_location="cpu", weights_only=False
    )
    fit, validation, gate_queries, split_report = temporal_threeway_split(
        list(teacher["query_names"]), args.crossfit_blocks
    )
    revised, refinement = refine_reserve_geometry(
        state=state,
        teacher=teacher,
        query_cache=query_cache,
        track_payload=track_payload,
        gaussian_ply=args.gaussian_ply,
        gaussian_type=args.gaussian_type,
        sh_degree=args.sh_degree,
        fit_queries=fit,
        validation_queries=validation,
        minimum_fit_views=args.minimum_fit_views,
        minimum_validation_views=args.minimum_validation_views,
        minimum_fit_improvement=args.minimum_fit_improvement,
        maximum_validation_degradation=args.maximum_validation_degradation,
        evidence_mode=args.evidence_mode,
        metric_state_path=args.metric_state,
        device=torch.device(args.device),
        ransac_reprojection_px=args.ransac_reprojection_px,
        clean_reprojection_px=args.clean_reprojection_px,
        seed=args.seed,
    )
    revised_map_path = output / "refined_anchor_map.pt"
    revised_metric_path = output / "refined_metric_state.pt"
    revised_teacher_path = output / "refined_complete_positive_teacher.pt"
    revised_metric = dict(metric_state)
    revised_metric["map_path"] = str(revised_map_path)
    torch.save(revised, revised_map_path)
    torch.save(revised_metric, revised_metric_path)
    config = teacher["config"]
    revised_teacher = build_teacher(
        anchor_map=revised,
        query_cache=query_cache,
        provenance=provenance,
        track_payload=track_payload,
        device=torch.device(args.device),
        strong_radius_px=float(config["strong_radius_px"]),
        ambiguous_radius_px=float(config["ambiguous_radius_px"]),
        depth_abs_tolerance_m=float(config["depth_abs_tolerance_m"]),
        depth_rel_tolerance=float(config["depth_rel_tolerance"]),
        alpha_minimum=float(config["alpha_minimum"]),
        contribution_minimum=float(config["contribution_minimum"]),
    )
    revised_teacher.update(
        {
            "anchor_map": str(revised_map_path),
            "query_cache": str(Path(args.query_cache).resolve()),
            "raster_provenance": str(Path(args.raster_provenance).resolve()),
            "track_payload": str(Path(args.track_payload).resolve()),
        }
    )
    torch.save(revised_teacher, revised_teacher_path)
    parameters = state["track_centric_reconstruction"]["calibration"]["parameters"]
    common = {
        "query_cache": query_cache,
        "device": torch.device(args.device),
        "ransac_reprojection_px": args.ransac_reprojection_px,
        "clean_reprojection_px": args.clean_reprojection_px,
        "task_translation_m": float(parameters["task_translation_m"]),
        "task_rotation_deg": float(parameters["task_rotation_deg"]),
        "seed": args.seed,
        "query_indices": gate_queries,
    }
    before = collect_deployment_statistics(
        state=state,
        metric_state_path=args.metric_state,
        teacher=teacher,
        progress_label="geometry_gate_before",
        **common,
    )
    after = collect_deployment_statistics(
        state=revised,
        metric_state_path=revised_metric_path,
        teacher=revised_teacher,
        progress_label="geometry_gate_after",
        **common,
    )
    checks = _gate(before, after)
    accepted = bool(refinement["accepted_anchor_rows"]) and all(checks.values())
    report = {
        "schema": "lafgs_crossfit_reserve_geometry_refinement",
        "version": 1,
        "uses_test_queries": False,
        "source_map": str(Path(args.map).resolve()),
        "refined_map": str(revised_map_path),
        "refined_metric_state": str(revised_metric_path),
        "refined_teacher": str(revised_teacher_path),
        "split": split_report,
        "refinement": refinement,
        "gate_before": before["summary"],
        "gate_after": after["summary"],
        "gate": checks,
        "accepted": accepted,
    }
    torch.save(
        {"before": before["counters"], "after": after["counters"]},
        output / "geometry_gate_statistics.pt",
    )
    (output / "reserve_geometry_refinement_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "accepted": accepted,
                "accepted_anchor_count": len(refinement["accepted_anchor_rows"]),
                "candidate_anchor_count": refinement["candidate_anchor_count"],
                "gate_before": before["summary"],
                "gate_after": after["summary"],
                "gate": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
