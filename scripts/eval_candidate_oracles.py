import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
import torch

from localization_training.pose_information import (
    compute_pose_information,
    pose_jacobian_analytic,
)
from utils.pose_utils import cal_pose_error, solve_pose


def project_points(points, K, pose_w2c):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    camera = (np.asarray(pose_w2c, dtype=np.float64) @ points_h.T)[:3].T
    depth = camera[:, 2]
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.isfinite(camera).all(axis=1) & (depth > 1e-8)
    uv[valid, 0] = K[0, 0] * camera[valid, 0] / depth[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * camera[valid, 1] / depth[valid] + K[1, 2]
    return uv, depth, valid


def deterministic_pnp(p2d, p3d, K):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if p2d.shape[0] < 4:
        return np.eye(4, dtype=np.float64), False
    success, rvec, tvec = cv2.solvePnP(
        p3d,
        p2d,
        K,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return np.eye(4, dtype=np.float64), False
    success, rvec, tvec = cv2.solvePnP(
        p3d,
        p2d,
        K,
        np.zeros((4, 1), dtype=np.float64),
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return np.eye(4, dtype=np.float64), False
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec.reshape(3)
    return pose, True


def robust_pnp(
    p2d,
    p3d,
    K,
    solver="poselib",
    reprojection_error=8.0,
    confidence=0.9999,
    max_iterations=100000,
    min_iterations=1000,
):
    pose, inliers = solve_pose(
        np.asarray(p2d, dtype=np.float64).reshape(-1, 2),
        np.asarray(p3d, dtype=np.float64).reshape(-1, 3),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        solver,
        float(reprojection_error),
        float(confidence),
        int(max_iterations),
        int(min_iterations),
    )
    return np.asarray(pose, dtype=np.float64), np.asarray(inliers).reshape(-1)


def covariance_weighted_refine(
    initial_pose,
    p2d,
    p3d,
    K,
    gt_residual,
    sigma_floor=0.25,
    iterations=10,
):
    """Oracle refinement using GT residual magnitude as diagonal covariance."""
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    gt_residual = np.asarray(gt_residual, dtype=np.float64).reshape(-1, 2)
    pose = np.asarray(initial_pose, dtype=np.float64).reshape(4, 4)
    if p2d.shape[0] < 4:
        return pose, False
    rvec = cv2.Rodrigues(pose[:3, :3])[0].reshape(3)
    tvec = pose[:3, 3].copy()
    sigma = np.maximum(np.abs(gt_residual), float(sigma_floor))
    inv_sigma = 1.0 / sigma
    for _ in range(int(iterations)):
        projected, jacobian = cv2.projectPoints(
            p3d,
            rvec,
            tvec,
            np.asarray(K, dtype=np.float64),
            np.zeros((4, 1), dtype=np.float64),
        )
        projected = projected.reshape(-1, 2)
        residual = p2d - projected
        jacobian = jacobian[:, :6].reshape(-1, 2, 6)
        radial = np.linalg.norm(residual * inv_sigma, axis=1)
        robust = np.minimum(1.0, 2.5 / np.maximum(radial, 1e-8))
        whiten = inv_sigma * np.sqrt(robust[:, None])
        design = (jacobian * whiten[:, :, None]).reshape(-1, 6)
        target = (residual * whiten).reshape(-1)
        H = design.T @ design + np.eye(6, dtype=np.float64) * 1e-6
        step = np.linalg.solve(H, design.T @ target)
        rvec += step[:3]
        tvec += step[3:]
        if np.linalg.norm(step) < 1e-9:
            break
    refined = np.eye(4, dtype=np.float64)
    refined[:3, :3] = cv2.Rodrigues(rvec)[0]
    refined[:3, 3] = tvec
    return refined, bool(np.isfinite(refined).all())


def linearized_pose_bias(p2d, p3d, K, pose_w2c, weights=None, damping=1e-6):
    """Estimate the physical pose bias induced by signed 2D residuals at GT."""
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    projected, _, valid = project_points(p3d, K, pose_w2c)
    valid &= np.isfinite(p2d).all(axis=1)
    p2d = p2d[valid]
    p3d = p3d[valid]
    projected = projected[valid]
    if weights is None:
        weights = np.ones(p2d.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)[valid]
    if p2d.shape[0] < 4:
        return {
            "success": False,
            "translation_cm": float("nan"),
            "rotation_deg": float("nan"),
            "condition": float("nan"),
        }

    points_h = np.concatenate([p3d, np.ones((p3d.shape[0], 1))], axis=1)
    camera = (pose_w2c @ points_h.T)[:3].T
    residual = p2d - projected
    H = np.eye(6, dtype=np.float64) * float(damping)
    b = np.zeros(6, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    for (x, y, z), r, weight in zip(camera, residual, weights):
        if z <= 1e-8 or not np.isfinite([x, y, z, weight]).all():
            continue
        dproj = np.array(
            [[fx / z, 0.0, -fx * x / (z * z)], [0.0, fy / z, -fy * y / (z * z)]],
            dtype=np.float64,
        )
        skew = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
        )
        jacobian = dproj @ np.concatenate([np.eye(3), -skew], axis=1)
        H += float(weight) * jacobian.T @ jacobian
        b += float(weight) * jacobian.T @ r
    try:
        delta = np.linalg.solve(H, b)
    except np.linalg.LinAlgError:
        delta = np.linalg.pinv(H) @ b
    update = np.eye(4, dtype=np.float64)
    update[:3, :3] = cv2.Rodrigues(delta[3:])[0]
    update[:3, 3] = delta[:3]
    biased_pose = update @ pose_w2c
    ae, te = cal_pose_error(biased_pose, pose_w2c)
    eig = np.linalg.eigvalsh(0.5 * (H + H.T)).clip(1e-12, None)
    return {
        "success": bool(np.isfinite(delta).all()),
        "translation_cm": float(te),
        "rotation_deg": float(ae),
        "condition": float(eig[-1] / eig[0]),
        "delta": delta.tolist(),
    }


def pose_information(points, K, pose_w2c, damping=1e-6):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 6:
        return {
            "full_logdet": float("nan"),
            "full_condition": float("nan"),
            "translation_logdet": float("nan"),
            "translation_condition": float("nan"),
            "translation_min_eig": float("nan"),
        }
    _, _, valid = project_points(points, K, pose_w2c)
    points = points[valid]
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    camera = (np.asarray(pose_w2c, dtype=np.float64) @ points_h.T)[:3].T
    fx, fy = float(K[0, 0]), float(K[1, 1])
    H = np.eye(6, dtype=np.float64) * float(damping)
    for x, y, z in camera:
        dproj = np.array(
            [[fx / z, 0.0, -fx * x / (z * z)], [0.0, fy / z, -fy * y / (z * z)]],
            dtype=np.float64,
        )
        skew = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
        )
        jacobian = dproj @ np.concatenate([np.eye(3), -skew], axis=1)
        H += jacobian.T @ jacobian
    H_tt = H[:3, :3]
    H_tr = H[:3, 3:]
    H_rr = H[3:, 3:]
    translation = H_tt - H_tr @ np.linalg.pinv(
        H_rr + np.eye(3, dtype=np.float64) * float(damping)
    ) @ H_tr.T
    translation = 0.5 * (translation + translation.T)
    full_eig = np.linalg.eigvalsh(H).clip(1e-12, None)
    trans_eig = np.linalg.eigvalsh(translation).clip(1e-12, None)
    return {
        "full_logdet": float(np.log(full_eig).sum()),
        "full_condition": float(full_eig[-1] / full_eig[0]),
        "translation_logdet": float(np.log(trans_eig).sum()),
        "translation_condition": float(trans_eig[-1] / trans_eig[0]),
        "translation_min_eig": float(trans_eig[0]),
    }


def fisher_candidate_scores(
    points,
    match_scores,
    K,
    pose_w2c,
    *,
    match_threshold=0.525,
    match_temperature=0.05,
    translation_scale=0.02,
    rotation_scale_degrees=2.0,
    measurement_sigma_px=1.0,
    damping=1e-4,
):
    """Return controlled per-point geometry scores for one fixed candidate set."""
    points_tensor = torch.as_tensor(points, dtype=torch.float64)
    K_tensor = torch.as_tensor(K, dtype=torch.float64)
    pose_tensor = torch.as_tensor(pose_w2c, dtype=torch.float64)
    scores_tensor = torch.as_tensor(match_scores, dtype=torch.float64).reshape(-1)
    if points_tensor.shape[0] != scores_tensor.shape[0]:
        raise ValueError("Fisher points and match scores must have equal length")

    jacobian = pose_jacobian_analytic(points_tensor, K_tensor, pose_tensor)
    point_jacobian = jacobian.square().sum(dim=(1, 2))
    common = {
        "damping": float(damping),
        "measurement_covariance": torch.full(
            (points_tensor.shape[0],),
            max(float(measurement_sigma_px), 1e-4) ** 2,
            dtype=torch.float64,
        ),
        "translation_scale": float(translation_scale),
        "rotation_scale": math.radians(float(rotation_scale_degrees)),
        "use_analytic_jacobian": True,
    }
    unit_information = compute_pose_information(
        points_tensor,
        K_tensor,
        pose_tensor,
        weights=torch.ones(points_tensor.shape[0], dtype=torch.float64),
        **common,
    )
    temperature = max(float(match_temperature), 1e-6)
    match_probability = torch.sigmoid(
        (scores_tensor - float(match_threshold)) / temperature
    )
    weighted_information = compute_pose_information(
        points_tensor,
        K_tensor,
        pose_tensor,
        weights=match_probability,
        **common,
    )
    return {
        "point_jacobian": point_jacobian.detach().cpu().numpy(),
        "full_set_leverage": (
            unit_information.full_set_leverage_scores.detach().cpu().numpy()
        ),
        "conditional_full": unit_information.scores.detach().cpu().numpy(),
        "conditional_translation": (
            unit_information.translation_scores.detach().cpu().numpy()
        ),
        "score_weighted_translation": (
            weighted_information.translation_scores.detach().cpu().numpy()
        ),
        "set_translation_logdet": float(unit_information.translation_logdet.item()),
        "set_translation_min_eig": float(
            unit_information.translation_min_eigenvalue.item()
        ),
        "set_translation_worst_std_task": float(
            unit_information.translation_worst_std.item()
        ),
        "set_effective_match_count": float(weighted_information.effective_count.item()),
    }


def fisher_retention_diagnostics(
    p2d,
    p3d,
    match_scores,
    K,
    pose_gt,
    args,
):
    """Compare equal-budget retention by legacy and conditional Fisher scores."""
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    match_scores = np.asarray(match_scores, dtype=np.float64).reshape(-1)
    count = int(p2d.shape[0])
    if count < 8:
        return {"fisher_inlier_count": count}
    retain_count = max(6, min(count, int(round(count * args.fisher_retain_fraction))))
    scores = fisher_candidate_scores(
        p3d,
        match_scores,
        K,
        pose_gt,
        match_threshold=args.fisher_match_threshold,
        match_temperature=args.fisher_match_temperature,
        translation_scale=args.fisher_translation_scale,
        rotation_scale_degrees=args.fisher_rotation_scale_degrees,
        measurement_sigma_px=args.fisher_measurement_sigma_px,
    )
    output = {
        "fisher_inlier_count": count,
        "fisher_retain_count": retain_count,
        "fisher_set_translation_logdet": scores.pop("set_translation_logdet"),
        "fisher_set_translation_min_eig": scores.pop("set_translation_min_eig"),
        "fisher_set_translation_worst_std_task": scores.pop(
            "set_translation_worst_std_task"
        ),
        "fisher_set_effective_match_count": scores.pop(
            "set_effective_match_count"
        ),
    }
    for name, values in scores.items():
        values = np.asarray(values, dtype=np.float64)
        descending = np.argsort(-values, kind="stable")
        keep_high = descending[:retain_count]
        keep_low = descending[-retain_count:]
        high = pose_result(
            f"fisher_{name}_keep_high", p2d[keep_high], p3d[keep_high], K, pose_gt
        )
        low = pose_result(
            f"fisher_{name}_keep_low", p2d[keep_low], p3d[keep_low], K, pose_gt
        )
        output.update(high)
        output.update(low)
        output[f"fisher_{name}_high_vs_low_te_gain_cm"] = (
            low[f"fisher_{name}_keep_low_te_cm"]
            - high[f"fisher_{name}_keep_high_te_cm"]
        )
        output[f"fisher_{name}_high_vs_low_ae_gain_deg"] = (
            low[f"fisher_{name}_keep_low_ae_deg"]
            - high[f"fisher_{name}_keep_high_ae_deg"]
        )
    return output


def visibility_filter(projected, depth, valid, width, height, abs_tol=0.25, rel_tol=0.02):
    in_image = (
        valid
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    candidate = np.flatnonzero(in_image)
    if candidate.size == 0:
        return in_image
    x = np.clip(np.floor(projected[candidate, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.floor(projected[candidate, 1]).astype(np.int64), 0, height - 1)
    cell = y * width + x
    min_depth = np.full(width * height, np.inf, dtype=np.float64)
    np.minimum.at(min_depth, cell, depth[candidate])
    tolerance = np.maximum(float(abs_tol), float(rel_tol) * min_depth[cell])
    visible_candidate = depth[candidate] <= min_depth[cell] + tolerance
    visible = np.zeros_like(in_image)
    visible[candidate[visible_candidate]] = True
    return visible


def oracle_assign_detector_points(p2d, bank_xyz, K, pose_w2c, width, height, radius_px):
    projected, depth, valid = project_points(bank_xyz, K, pose_w2c)
    visible = visibility_filter(projected, depth, valid, width, height)
    visible_idx = np.flatnonzero(visible)
    if visible_idx.size == 0:
        return np.empty((0, 2)), np.empty((0, 3)), np.empty(0)
    tree = cKDTree(projected[visible_idx])
    distance, local_idx = tree.query(np.asarray(p2d, dtype=np.float64) + 0.5, k=1)
    keep = np.isfinite(distance) & (distance <= float(radius_px))
    return (
        np.asarray(p2d, dtype=np.float64)[keep],
        bank_xyz[visible_idx[local_idx[keep]]],
        distance[keep],
    )


def balanced_subset(p2d, p3d, scores, K, pose_w2c, width, height, max_count=512):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if p2d.shape[0] <= int(max_count):
        return np.arange(p2d.shape[0], dtype=np.int64)
    _, depth, _ = project_points(p3d, K, pose_w2c)
    boundaries = np.quantile(depth, [0.25, 0.5, 0.75])
    depth_bin = np.digitize(depth, boundaries)
    grid_x = np.clip(np.floor(p2d[:, 0] / max(width, 1) * 4).astype(np.int64), 0, 3)
    grid_y = np.clip(np.floor(p2d[:, 1] / max(height, 1) * 4).astype(np.int64), 0, 3)
    grid_bin = grid_y * 4 + grid_x
    voxel = np.floor(p3d / 0.25).astype(np.int64)
    order = np.argsort(-scores, kind="stable")
    grid_cap = max(1, int(np.ceil(max_count / 16.0 * 1.5)))
    depth_cap = max(1, int(np.ceil(max_count / 4.0 * 1.5)))
    grid_count = np.zeros(16, dtype=np.int64)
    depth_count = np.zeros(4, dtype=np.int64)
    voxel_count = {}
    selected = []
    for idx in order:
        voxel_key = tuple(voxel[idx].tolist())
        if grid_count[grid_bin[idx]] >= grid_cap:
            continue
        if depth_count[depth_bin[idx]] >= depth_cap:
            continue
        if voxel_count.get(voxel_key, 0) >= 2:
            continue
        selected.append(int(idx))
        grid_count[grid_bin[idx]] += 1
        depth_count[depth_bin[idx]] += 1
        voxel_count[voxel_key] = voxel_count.get(voxel_key, 0) + 1
        if len(selected) >= int(max_count):
            break
    if len(selected) < int(max_count):
        selected_set = set(selected)
        selected.extend(int(idx) for idx in order if int(idx) not in selected_set)
    return np.asarray(selected[: int(max_count)], dtype=np.int64)


def pose_result(prefix, p2d, p3d, K, pose_gt):
    pose, success = deterministic_pnp(np.asarray(p2d) + 0.5, p3d, K)
    ae, te = cal_pose_error(pose, pose_gt) if success else (float("inf"), float("inf"))
    result = {
        f"{prefix}_count": int(np.asarray(p2d).shape[0]),
        f"{prefix}_success": bool(success),
        f"{prefix}_ae_deg": float(ae),
        f"{prefix}_te_cm": float(te),
    }
    result.update({f"{prefix}_{key}": value for key, value in pose_information(p3d, K, pose_gt).items()})
    return result


def robust_pose_result(prefix, p2d, p3d, K, pose_gt, args):
    pose, inliers = robust_pnp(
        np.asarray(p2d) + 0.5,
        p3d,
        K,
        solver=args.solver,
        reprojection_error=args.reprojection_error,
        confidence=args.confidence,
        max_iterations=args.max_iterations,
        min_iterations=args.min_iterations,
    )
    success = inliers.shape[0] >= 4
    ae, te = cal_pose_error(pose, pose_gt) if success else (float("inf"), float("inf"))
    result = {
        f"{prefix}_count": int(np.asarray(p2d).shape[0]),
        f"{prefix}_inliers": int(inliers.shape[0]),
        f"{prefix}_success": bool(success),
        f"{prefix}_ae_deg": float(ae),
        f"{prefix}_te_cm": float(te),
    }
    if inliers.shape[0] >= 4:
        result.update(
            {
                f"{prefix}_{key}": value
                for key, value in pose_information(
                    np.asarray(p3d)[inliers], K, pose_gt
                ).items()
            }
        )
    return result


def summarize(per_query):
    summary = {"query_count": len(per_query)}
    keys = sorted({key for item in per_query for key in item if key != "image_name"})
    for key in keys:
        values = [item[key] for item in per_query if isinstance(item.get(key), (int, float))]
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            summary[f"{key}_mean"] = float(finite.mean())
            summary[f"{key}_median"] = float(np.median(finite))
    pose_prefixes = sorted(
        {
            key[: -len("_te_cm")]
            for item in per_query
            for key in item
            if key.endswith("_te_cm")
        }
    )
    for prefix in pose_prefixes:
        te = np.asarray([item.get(f"{prefix}_te_cm", np.inf) for item in per_query])
        ae = np.asarray([item.get(f"{prefix}_ae_deg", np.inf) for item in per_query])
        summary[f"{prefix}_r2"] = float(np.mean((te <= 2.0) & (ae <= 2.0)))
        summary[f"{prefix}_r5"] = float(np.mean((te <= 5.0) & (ae <= 5.0)))
    bias = np.asarray(
        [item.get("o6_linearized_bias_te_cm", np.nan) for item in per_query],
        dtype=np.float64,
    )
    actual = np.asarray(
        [item.get("actual_te_cm", np.nan) for item in per_query], dtype=np.float64
    )
    valid = np.isfinite(bias) & np.isfinite(actual)
    if valid.sum() >= 3:
        correlation = spearmanr(bias[valid], actual[valid])
        summary["o6_bias_actual_te_spearman"] = float(correlation.statistic)
        summary["o6_bias_actual_te_spearman_pvalue"] = float(correlation.pvalue)
    for name in (
        "point_jacobian",
        "full_set_leverage",
        "conditional_full",
        "conditional_translation",
        "score_weighted_translation",
    ):
        key = f"fisher_{name}_high_vs_low_te_gain_cm"
        gains = np.asarray([item.get(key, np.nan) for item in per_query], dtype=np.float64)
        gains = gains[np.isfinite(gains)]
        if gains.size:
            summary[f"{key}_positive_rate"] = float(np.mean(gains > 0.0))
    for key in (
        "fisher_set_translation_logdet",
        "fisher_set_translation_min_eig",
        "fisher_set_translation_worst_std_task",
        "fisher_set_effective_match_count",
    ):
        values = np.asarray(
            [item.get(key, np.nan) for item in per_query], dtype=np.float64
        )
        valid = np.isfinite(values) & np.isfinite(actual)
        if valid.sum() >= 3:
            correlation = spearmanr(values[valid], actual[valid])
            summary[f"{key}_actual_te_spearman"] = float(correlation.statistic)
            summary[f"{key}_actual_te_spearman_pvalue"] = float(
                correlation.pvalue
            )
    return summary


def evaluate(args):
    records = [json.loads(line) for line in Path(args.correspondences).read_text().splitlines() if line]
    if not records:
        raise ValueError("correspondence dump is empty")
    bank_xyz = np.unique(
        np.concatenate([np.asarray(record["p3d"], dtype=np.float32) for record in records], axis=0),
        axis=0,
    ).astype(np.float64)
    result_by_name = {}
    if args.results:
        for item in json.loads(Path(args.results).read_text()):
            result_by_name[item["image_name"]] = item
    fixed_budgets = sorted(
        {int(value) for value in str(args.fixed_budgets).split(",") if value.strip()}
    )
    per_query = []
    for record in records:
        p2d = np.asarray(record["p2d"], dtype=np.float64)
        p3d = np.asarray(record["p3d"], dtype=np.float64)
        scores = np.asarray(record["scores"], dtype=np.float64)
        K = np.asarray(record["K"], dtype=np.float64)
        pose_gt = np.asarray(record["gt_pose_w2c"], dtype=np.float64)
        projected, depth, valid = project_points(p3d, K, pose_gt)
        error = np.linalg.norm(projected - (p2d + 0.5), axis=1)
        valid &= np.isfinite(error) & (depth > 0)
        positive = valid & (error < float(args.positive_radius_px))
        ambiguous = valid & (error >= float(args.positive_radius_px)) & (
            error <= float(args.negative_radius_px)
        )
        query = {
            "image_name": record["image_name"],
            "candidate_count": int(p2d.shape[0]),
            "candidate_positive_count": int(positive.sum()),
            "candidate_ambiguous_count": int(ambiguous.sum()),
            "candidate_precision_2px": float(positive.mean()) if positive.size else 0.0,
            "candidate_duplicate_landmark_rate": float(
                1.0 - np.unique(p3d, axis=0).shape[0] / max(p3d.shape[0], 1)
            ),
        }
        actual = result_by_name.get(record["image_name"])
        if actual is not None:
            query["actual_te_cm"] = float(actual.get("sparse_TE", np.nan))
            query["actual_ae_deg"] = float(actual.get("sparse_AE", np.nan))

        clean_p2d, clean_p3d, clean_scores = p2d[positive], p3d[positive], scores[positive]
        query.update(pose_result("o1_delete_wrong", clean_p2d, clean_p3d, K, pose_gt))
        query.update(pose_result("o3_clean", clean_p2d, clean_p3d, K, pose_gt))

        signed_corrected_p2d = p2d.copy()
        signed_corrected_p2d[positive] = projected[positive] - 0.5
        query.update(
            robust_pose_result(
                "o2_signed_residual",
                signed_corrected_p2d,
                p3d,
                K,
                pose_gt,
                args,
            )
        )

        initial_pose, initial_inliers = robust_pnp(
            p2d + 0.5,
            p3d,
            K,
            solver=args.solver,
            reprojection_error=args.reprojection_error,
            confidence=args.confidence,
            max_iterations=args.max_iterations,
            min_iterations=args.min_iterations,
        )
        recorded_inliers = np.asarray(record.get("inliers", []), dtype=np.int64)
        recorded_inliers = recorded_inliers[
            (recorded_inliers >= 0) & (recorded_inliers < p2d.shape[0])
        ]
        query.update(
            fisher_retention_diagnostics(
                p2d[recorded_inliers],
                p3d[recorded_inliers],
                scores[recorded_inliers],
                K,
                pose_gt,
                args,
            )
        )
        covariance_pose, covariance_success = covariance_weighted_refine(
            initial_pose,
            clean_p2d + 0.5,
            clean_p3d,
            K,
            (clean_p2d + 0.5) - projected[positive],
            sigma_floor=args.covariance_floor_px,
        )
        covariance_ae, covariance_te = (
            cal_pose_error(covariance_pose, pose_gt)
            if covariance_success and initial_inliers.shape[0] >= 4
            else (float("inf"), float("inf"))
        )
        query.update(
            {
                "o3_covariance_count": int(clean_p2d.shape[0]),
                "o3_covariance_success": bool(covariance_success),
                "o3_covariance_ae_deg": float(covariance_ae),
                "o3_covariance_te_cm": float(covariance_te),
            }
        )

        clean_residual = (clean_p2d + 0.5) - projected[positive]
        if clean_residual.shape[0] > 0:
            query.update(
                {
                    "o6_signed_residual_x_mean_px": float(clean_residual[:, 0].mean()),
                    "o6_signed_residual_y_mean_px": float(clean_residual[:, 1].mean()),
                    "o6_signed_residual_mean_norm_px": float(
                        np.linalg.norm(clean_residual.mean(axis=0))
                    ),
                    "o6_residual_radial_mean_px": float(
                        np.linalg.norm(clean_residual, axis=1).mean()
                    ),
                }
            )
        score_weights = 1.0 / (1.0 + np.exp(-np.clip(clean_scores, -20.0, 20.0)))
        bias = linearized_pose_bias(
            clean_p2d + 0.5,
            clean_p3d,
            K,
            pose_gt,
            weights=score_weights,
        )
        query.update(
            {
                "o6_linearized_bias_te_cm": bias["translation_cm"],
                "o6_linearized_bias_ae_deg": bias["rotation_deg"],
                "o6_linearized_bias_condition": bias["condition"],
            }
        )

        score_order = np.argsort(-scores, kind="stable")
        for budget in fixed_budgets:
            fixed = score_order[: min(int(budget), score_order.shape[0])]
            query.update(
                robust_pose_result(
                    f"o4_fixed_k{budget}",
                    p2d[fixed],
                    p3d[fixed],
                    K,
                    pose_gt,
                    args,
                )
            )
        selected = balanced_subset(
            clean_p2d,
            clean_p3d,
            clean_scores,
            K,
            pose_gt,
            int(record["width"]),
            int(record["height"]),
            max_count=args.balanced_count,
        )
        query.update(
            pose_result(
                "o4_balanced",
                clean_p2d[selected],
                clean_p3d[selected],
                K,
                pose_gt,
            )
        )
        oracle_p2d, oracle_p3d, oracle_error = oracle_assign_detector_points(
            p2d,
            bank_xyz,
            K,
            pose_gt,
            int(record["width"]),
            int(record["height"]),
            args.positive_radius_px,
        )
        query["o2_detector_matchable_rate"] = float(
            oracle_p2d.shape[0] / max(p2d.shape[0], 1)
        )
        query["o2_detector_oracle_reproj_mean_px"] = float(
            oracle_error.mean() if oracle_error.size else np.nan
        )
        query.update(pose_result("o2_detector_oracle", oracle_p2d, oracle_p3d, K, pose_gt))
        per_query.append(query)
    payload = {
        "config": {
            "correspondences": str(args.correspondences),
            "positive_radius_px": float(args.positive_radius_px),
            "negative_radius_px": float(args.negative_radius_px),
            "balanced_count": int(args.balanced_count),
            "fixed_budgets": fixed_budgets,
            "oracle_bank_count": int(bank_xyz.shape[0]),
            "solver": args.solver,
            "reprojection_error": float(args.reprojection_error),
            "covariance_floor_px": float(args.covariance_floor_px),
            "fisher_retain_fraction": float(args.fisher_retain_fraction),
            "fisher_match_threshold": float(args.fisher_match_threshold),
            "fisher_match_temperature": float(args.fisher_match_temperature),
            "fisher_translation_scale": float(args.fisher_translation_scale),
            "fisher_rotation_scale_degrees": float(
                args.fisher_rotation_scale_degrees
            ),
            "fisher_measurement_sigma_px": float(
                args.fisher_measurement_sigma_px
            ),
        },
        "summary": summarize(per_query),
        "queries": per_query,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--correspondences", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive_radius_px", type=float, default=2.0)
    parser.add_argument("--negative_radius_px", type=float, default=6.0)
    parser.add_argument("--balanced_count", type=int, default=512)
    parser.add_argument("--fixed_budgets", default="256,512,1024,2048,4096")
    parser.add_argument("--results", default="")
    parser.add_argument("--solver", choices=["poselib", "opencv"], default="poselib")
    parser.add_argument("--reprojection_error", type=float, default=8.0)
    parser.add_argument("--confidence", type=float, default=0.9999)
    parser.add_argument("--max_iterations", type=int, default=100000)
    parser.add_argument("--min_iterations", type=int, default=1000)
    parser.add_argument("--covariance_floor_px", type=float, default=0.25)
    parser.add_argument("--fisher_retain_fraction", type=float, default=0.75)
    parser.add_argument("--fisher_match_threshold", type=float, default=0.525)
    parser.add_argument("--fisher_match_temperature", type=float, default=0.05)
    parser.add_argument("--fisher_translation_scale", type=float, default=0.02)
    parser.add_argument(
        "--fisher_rotation_scale_degrees", type=float, default=2.0
    )
    parser.add_argument(
        "--fisher_measurement_sigma_px", type=float, default=1.0
    )
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
