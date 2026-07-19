import datetime
import hashlib
import json
import os
import pickle
import warnings
from collections import Counter
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import get_render_visible_mask, render_from_pose_gsplat
from localization_training.direct_landmark_teacher import gaussian_localization_xyz
from localization_training.episode_sampler import split_support_query_cameras
from localization_training.geometry_selector import GeometryBalancedSelector
from localization_training.full_primitive_retrieval import (
    chunked_exact_topk,
    suppress_redundant_hypotheses,
)
from localization_training.pair_scorer import SparsePairScorer
from localization_training.pair_measurement import (
    PairMeasurementHead,
    build_pair_geometry_features,
    sample_local_correlation_patch,
)
from localization_training.sparse_frontend import (
    SparseMatchResult,
    build_pair_context_features,
    build_score_matrix,
    dual_softmax as shared_dual_softmax,
    gather_aligned_pair_values,
    match_score_matrix,
    rank_keypoint_proposals,
    select_match_candidates,
    select_offset_only_candidates,
    select_match_candidates_with_geometry_refill,
)
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from scene.kpdetector import KpDetector, simple_nms
from utils.graphics_utils import fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.pose_utils import (
    cal_pose_error,
    covariance_weighted_pose_refinement,
    solve_pose,
)


def select_candidate_validation_cameras(
    cameras,
    *,
    query_ratio=0.2,
    validation_ratio=0.25,
    split_mode="temporal_block",
    split_seed=2026,
    direct_holdout=False,
):
    if bool(direct_holdout):
        _, validation_cameras = split_support_query_cameras(
            list(cameras),
            query_ratio=validation_ratio,
            seed=split_seed + 1,
            mode=split_mode,
        )
        return validation_cameras
    _, query_cameras = split_support_query_cameras(
        list(cameras),
        query_ratio=query_ratio,
        seed=split_seed,
        mode=split_mode,
    )
    _, validation_cameras = split_support_query_cameras(
        query_cameras,
        query_ratio=validation_ratio,
        seed=split_seed + 1,
        mode=split_mode,
    )
    return validation_cameras


def load_evaluation_camera_list(cameras, path):
    """Select an ordered, explicit evaluation subset by image name.

    Candidate-validation splits are useful for standard reporting, but dense
    refinement training also needs sparse seed poses for the complementary
    real-training views.  An explicit list makes that protocol reproducible
    without changing the scene split or relying on list ordering.
    """
    with open(path) as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("image_names", payload.get("images"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            "evaluation_camera_list must be a non-empty JSON list, or an object "
            "with an image_names/images list"
        )
    names = [str(name).replace("\\", "/") for name in payload]
    if len(set(names)) != len(names):
        raise ValueError("evaluation_camera_list contains duplicate image names")
    by_name = {
        str(camera.image_name).replace("\\", "/"): camera
        for camera in cameras
    }
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(
            "evaluation_camera_list contains images absent from this scene: "
            f"{missing[:3]}"
        )
    return [by_name[name] for name in names]


def candidate_frontend_mismatches(state_config, sparse_config):
    """Report train/eval sparse-frontend settings that change candidate context."""
    if not isinstance(state_config, dict):
        return []
    eval_mode = "mnn" if bool(sparse_config.get("mnn_match", False)) else "topk"
    checks = (
        ("detect_num", state_config.get("detect_num"), sparse_config.get("detect_num")),
        ("nms_radius", state_config.get("nms_radius"), sparse_config.get("nms")),
        ("match_mode", state_config.get("match_mode"), eval_mode),
        ("match_topk", state_config.get("match_topk"), sparse_config.get("topk")),
        (
            "match_threshold",
            state_config.get("match_threshold"),
            sparse_config.get("threshold"),
        ),
        (
            "dual_softmax",
            state_config.get("dual_softmax"),
            sparse_config.get("dual_softmax"),
        ),
        (
            "dual_softmax_temperature",
            state_config.get("dual_softmax_temperature"),
            sparse_config.get("dual_softmax_temp"),
        ),
        (
            "pair_context_topk",
            state_config.get("pair_context_topk"),
            sparse_config.get("pair_context_topk", 8),
        ),
        (
            "map_max_matches_per_landmark",
            state_config.get("map_max_matches_per_landmark"),
            sparse_config.get("max_matches_per_landmark"),
        ),
    )
    mismatches = []
    for name, trained, evaluated in checks:
        if trained is None or evaluated is None:
            continue
        if isinstance(trained, (float, int)) and isinstance(evaluated, (float, int)):
            equal = abs(float(trained) - float(evaluated)) <= 1e-8
        else:
            equal = trained == evaluated
        if not equal:
            mismatches.append((name, trained, evaluated))
    return mismatches


def validate_candidate_frontend_compatibility(state_config, sparse_config):
    policy = str(sparse_config.get("candidate_frontend_match_policy", "warn")).lower()
    if policy not in {"error", "warn", "ignore"}:
        raise ValueError(
            "candidate_frontend_match_policy must be one of: error, warn, ignore"
        )
    mismatches = candidate_frontend_mismatches(state_config, sparse_config)
    if not mismatches or policy == "ignore":
        return mismatches
    details = ", ".join(
        f"{name}: trained={trained!r} eval={evaluated!r}"
        for name, trained, evaluated in mismatches
    )
    message = f"candidate scorer train/eval frontend mismatch: {details}"
    if policy == "error":
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return mismatches


def candidate_direct_holdout_mismatches(
    state_config,
    *,
    validation_ratio,
    split_mode,
    split_seed,
):
    """Compare a direct validation request with candidate-teacher training metadata."""
    if not isinstance(state_config, dict):
        return [("candidate_teacher_state_config", None, "required")]
    expected = (
        ("validation_ratio", float(validation_ratio)),
        ("split_mode", str(split_mode)),
        ("split_seed", int(split_seed)),
    )
    mismatches = []
    for name, requested in expected:
        trained = state_config.get(name)
        if isinstance(requested, float):
            matches = trained is not None and abs(float(trained) - requested) <= 1e-8
        else:
            matches = trained == requested
        if not matches:
            mismatches.append((name, trained, requested))
    return mismatches


def validate_candidate_direct_holdout_compatibility(
    state_config,
    *,
    validation_ratio,
    split_mode,
    split_seed,
    policy="error",
):
    """Reject a claimed direct holdout unless its training partition matches."""
    policy = str(policy).lower()
    if policy not in {"ignore", "warn", "error"}:
        raise ValueError(
            "candidate direct holdout policy must be one of: ignore, warn, error"
        )
    mismatches = candidate_direct_holdout_mismatches(
        state_config,
        validation_ratio=validation_ratio,
        split_mode=split_mode,
        split_seed=split_seed,
    )
    if not mismatches or policy == "ignore":
        return mismatches
    details = ", ".join(
        f"{name}: trained={trained!r}, requested={requested!r}"
        for name, trained, requested in mismatches
    )
    message = (
        "candidate_direct_validation_holdout requires a candidate-teacher state "
        f"trained with the same held-out partition ({details})"
    )
    if policy == "error":
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    return mismatches


def apply_sparse_artifact_overrides(
    config,
    *,
    detector_path=None,
    landmark_feature_override_path=None,
):
    """Apply explicit evaluation artifact overrides without changing the source YAML."""
    sparse_config = config.setdefault("sparse", {})
    if detector_path is not None:
        sparse_config["detector_path"] = detector_path
    if landmark_feature_override_path is not None:
        sparse_config["landmark_feature_override_path"] = (
            landmark_feature_override_path
        )


# TODO use interpolate
def lift_2d_to_3d(points2d, intrinsic, Twc, depth_map):
    """
    points2d: tensor [N, 2]
    intrinsic: tensor [3, 3]
    Twc: tensor [4, 4]
    depth_map: tensor [H, W]
    """
    device = points2d.device
    depth_idx = points2d.long()
    points2d = points2d + 0.5
    points2d_homo = torch.cat(
        [points2d, torch.ones((points2d.shape[0], 1), device=device)], dim=1
    )
    points3d_camera = (
        torch.inverse(intrinsic)
        @ points2d_homo.T
        * depth_map[depth_idx[:, 1], depth_idx[:, 0]]
    )  # [3, N]
    points3d_camera_homo = torch.cat(
        [
            points3d_camera,
            torch.ones((1, points3d_camera.shape[-1]), device=device),
        ],
        dim=0,
    )  # [4, N]
    points3d_world = Twc @ points3d_camera_homo  # [4, N]
    points3d = points3d_world.T[:, :3]
    return points3d


def sample_gaussians(gaussians: GaussianModel, idx_sampled):
    loc_xyz = gaussian_localization_xyz(gaussians)
    idx_sampled = validate_sampled_indices(
        idx_sampled,
        loc_xyz.shape[0],
    ).to(device=loc_xyz.device)
    sampled_gaussians = GaussianModel(3)
    sampled_gaussians._xyz = loc_xyz[idx_sampled]
    sampled_gaussians._loc_feature = gaussians.materialized_loc_feature(idx_sampled)
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    return sampled_gaussians


def mnn_match(corr_matrix, thr=-1):
    """
    corr_matrix: torch.Tensor, shape (B, N, M)
    """
    mask = corr_matrix > thr
    mask = (
        mask
        * (corr_matrix == corr_matrix.max(dim=-1, keepdim=True)[0])
        * (corr_matrix == corr_matrix.max(dim=-2, keepdim=True)[0])
    )
    b_ids, i_ids, j_ids = torch.where(mask)
    return b_ids.squeeze(), i_ids.squeeze(), j_ids.squeeze()


def topk_match(corr_matrix, topk, thr=-1):
    """
    corr_matrix: torch.Tensor, shape (B, N, M)
    """
    N_im = corr_matrix.shape[-2]
    val, idx = torch.topk(corr_matrix, topk, dim=-1)
    val_flattened = val.flatten(1)
    idx_flattened = idx.flatten(1)
    mask = val_flattened > thr
    arange_tensor = torch.arange(N_im, device=corr_matrix.device)
    image_ids = arange_tensor.view(1, N_im, 1).expand(
        corr_matrix.shape[0], N_im, topk
    )
    idx_im = image_ids.reshape(corr_matrix.shape[0], -1)[mask]
    idx_gs = idx_flattened[mask]
    val = val_flattened[mask]

    return idx_im, idx_gs, val


def dual_softmax(corr_matrix, temp=1):
    return shared_dual_softmax(corr_matrix, temp)


def get_intrinsic(fovx, fovy, width, height):
    focalX = fov2focal(fovx, width)
    focalY = fov2focal(fovy, height)
    K = np.array(
        [
            [focalX, 0.0, width / 2],
            [0.0, focalY, height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return K


def _project_points_np(p3d, K, pose_w2c, eps=1e-8):
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    if p3d.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    p3d_h = np.concatenate([p3d, np.ones((p3d.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (pose_w2c @ p3d_h.T)[:3].T
    depth = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (depth > eps)
    uv = np.full((p3d.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = K[0, 0] * cam[valid, 0] / depth[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * cam[valid, 1] / depth[valid] + K[1, 2]
    return uv, depth


def _residual_stats(prefix, residual):
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(residual)),
        f"{prefix}_median": float(np.median(residual)),
        f"{prefix}_p95": float(np.percentile(residual, 95)),
        f"{prefix}_max": float(np.max(residual)),
    }


def _occupancy_stats_2d(prefix, p2d, width, height, grid_rows=4, grid_cols=4):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    if p2d.shape[0] == 0:
        return {}
    width = max(float(width), 1.0)
    height = max(float(height), 1.0)
    grid_rows = max(int(grid_rows), 1)
    grid_cols = max(int(grid_cols), 1)
    x = np.clip(np.floor(p2d[:, 0] / width * grid_cols).astype(np.int64), 0, grid_cols - 1)
    y = np.clip(np.floor(p2d[:, 1] / height * grid_rows).astype(np.int64), 0, grid_rows - 1)
    cell = y * grid_cols + x
    counts = np.bincount(cell, minlength=grid_rows * grid_cols).astype(np.float64)
    occupied = int(np.count_nonzero(counts))
    prob = counts[counts > 0] / max(float(p2d.shape[0]), 1.0)
    entropy = -float(np.sum(prob * np.log(prob + 1e-12))) if prob.size else 0.0
    entropy_norm = entropy / np.log(grid_rows * grid_cols) if grid_rows * grid_cols > 1 else 0.0
    return {
        f"{prefix}_2d_occupied_cells": occupied,
        f"{prefix}_2d_occupancy_frac": float(occupied / float(grid_rows * grid_cols)),
        f"{prefix}_2d_max_cell_frac": float(np.max(counts) / max(float(p2d.shape[0]), 1.0)),
        f"{prefix}_2d_entropy_norm": float(entropy_norm),
    }


def _occupancy_stats_3d(prefix, p3d, voxel_size=0.25):
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    if p3d.shape[0] == 0:
        return {}
    voxel_size = float(voxel_size or 0.0)
    if voxel_size <= 0.0:
        return {}
    voxels = np.floor(p3d / voxel_size).astype(np.int64)
    _, counts = np.unique(voxels, axis=0, return_counts=True)
    counts = counts.astype(np.float64)
    return {
        f"{prefix}_3d_voxels": int(counts.shape[0]),
        f"{prefix}_3d_voxel_per_match": float(counts.shape[0] / max(float(p3d.shape[0]), 1.0)),
        f"{prefix}_3d_max_voxel_frac": float(np.max(counts) / max(float(p3d.shape[0]), 1.0)),
    }


def _pose_information_stats(
    prefix,
    p3d,
    K,
    pose_w2c,
    regularization=1e-6,
    translation_task_scale_m=0.02,
    rotation_task_scale_degrees=2.0,
):
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    if p3d.shape[0] < 6:
        return {}
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    p3d_h = np.concatenate([p3d, np.ones((p3d.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (pose_w2c @ p3d_h.T)[:3].T
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > 1e-8)
    cam = cam[valid]
    if cam.shape[0] < 6:
        return {}
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    H = np.zeros((6, 6), dtype=np.float64)
    jacobians = []
    for x, y, z in cam:
        dproj = np.array(
            [
                [fx / z, 0.0, -fx * x / (z * z)],
                [0.0, fy / z, -fy * y / (z * z)],
            ],
            dtype=np.float64,
        )
        skew = np.array(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=np.float64,
        )
        jac = dproj @ np.concatenate([np.eye(3, dtype=np.float64), -skew], axis=1)
        jacobians.append(jac)
        H += jac.T @ jac
    H_reg = H + float(regularization) * np.eye(6, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(H_reg)
    eigvals = eigvals[np.isfinite(eigvals)]
    if eigvals.size == 0:
        return {}
    eigvals = np.clip(eigvals, 1e-12, None)
    sign, logdet = np.linalg.slogdet(H_reg)
    translation_scale = max(float(translation_task_scale_m), 1e-12)
    rotation_scale = max(
        float(np.deg2rad(rotation_task_scale_degrees)),
        1e-12,
    )
    task_scale = np.diag(
        [translation_scale] * 3 + [rotation_scale] * 3
    ).astype(np.float64)
    jacobians = np.stack(jacobians, axis=0)
    task_jacobians = jacobians @ task_scale
    contributions = np.einsum(
        "nki,nkj->nij",
        task_jacobians,
        task_jacobians,
    )
    task_H = contributions.sum(axis=0)
    task_H += float(regularization) * np.eye(6, dtype=np.float64)
    H_tt = task_H[:3, :3]
    H_tr = task_H[:3, 3:]
    H_rr = task_H[3:, 3:]
    schur = H_tt - H_tr @ np.linalg.solve(
        H_rr + 1e-12 * np.eye(3, dtype=np.float64),
        H_tr.T,
    )
    schur = 0.5 * (schur + schur.T)
    task_eigvals = np.clip(np.linalg.eigvalsh(task_H), 1e-12, None)
    translation_eigvals = np.clip(np.linalg.eigvalsh(schur), 1e-12, None)
    task_sign, task_logdet = np.linalg.slogdet(task_H)
    translation_sign, translation_logdet = np.linalg.slogdet(schur)
    translation_covariance = np.linalg.inv(schur)
    translation_covariance_eigvals = np.clip(
        np.linalg.eigvalsh(translation_covariance),
        0.0,
        None,
    )
    translation_worst_std_task = float(
        np.sqrt(translation_covariance_eigvals.max())
    )
    full_inverse = np.linalg.inv(task_H)
    leverage_matrix = np.eye(2, dtype=np.float64)[None] + (
        task_jacobians @ full_inverse @ task_jacobians.transpose(0, 2, 1)
    )
    leverage_sign, leverage_scores = np.linalg.slogdet(leverage_matrix)
    leverage_scores = np.where(leverage_sign > 0, leverage_scores, np.nan)
    without = task_H[None] - contributions
    without = 0.5 * (without + without.transpose(0, 2, 1))
    without_sign, without_logdet = np.linalg.slogdet(without)
    full_delete_gain = np.where(
        without_sign > 0,
        task_logdet - without_logdet,
        np.nan,
    )
    without_tt = without[:, :3, :3]
    without_tr = without[:, :3, 3:]
    without_rr = without[:, 3:, 3:]
    without_schur = without_tt - without_tr @ np.linalg.solve(
        without_rr + 1e-12 * np.eye(3, dtype=np.float64)[None],
        without_tr.transpose(0, 2, 1),
    )
    without_schur = 0.5 * (
        without_schur + without_schur.transpose(0, 2, 1)
    )
    without_translation_sign, without_translation_logdet = np.linalg.slogdet(
        without_schur
    )
    translation_delete_gain = np.where(
        without_translation_sign > 0,
        translation_logdet - without_translation_logdet,
        np.nan,
    )

    def distribution_stats(name, values):
        values = np.asarray(values, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return {}
        values = np.clip(values, 0.0, None)
        return {
            f"{prefix}_pose_info_{name}_mean": float(np.mean(values)),
            f"{prefix}_pose_info_{name}_p95": float(np.percentile(values, 95)),
            f"{prefix}_pose_info_{name}_max": float(np.max(values)),
        }

    result = {
        f"{prefix}_pose_info_condition": float(eigvals.max() / eigvals.min()),
        f"{prefix}_pose_info_logdet": float(logdet if sign > 0 else np.nan),
        f"{prefix}_pose_info_min_eig": float(eigvals.min()),
        f"{prefix}_pose_info_task_condition": float(
            task_eigvals.max() / task_eigvals.min()
        ),
        f"{prefix}_pose_info_task_logdet": float(
            task_logdet if task_sign > 0 else np.nan
        ),
        f"{prefix}_pose_info_translation_condition": float(
            translation_eigvals.max() / translation_eigvals.min()
        ),
        f"{prefix}_pose_info_translation_logdet": float(
            translation_logdet if translation_sign > 0 else np.nan
        ),
        f"{prefix}_pose_info_translation_min_eig": float(
            translation_eigvals.min()
        ),
        f"{prefix}_pose_info_translation_trace_covariance": float(
            np.trace(translation_covariance)
        ),
        f"{prefix}_pose_info_translation_worst_std_task": (
            translation_worst_std_task
        ),
        f"{prefix}_pose_info_translation_worst_std_m": float(
            translation_worst_std_task * translation_scale
        ),
        f"{prefix}_pose_info_effective_count": float(cam.shape[0]),
        f"{prefix}_pose_info_translation_task_scale_m": translation_scale,
        f"{prefix}_pose_info_rotation_task_scale_degrees": float(
            rotation_task_scale_degrees
        ),
    }
    result.update(
        distribution_stats(
            "point_jacobian",
            np.square(jacobians).sum(axis=(1, 2)),
        )
    )
    result.update(distribution_stats("full_set_leverage", leverage_scores))
    result.update(distribution_stats("full_delete_gain", full_delete_gain))
    result.update(
        distribution_stats("translation_delete_gain", translation_delete_gain)
    )
    return result


def _pose_bias_stats(
    prefix,
    p2d,
    p3d,
    K,
    pose_w2c,
    *,
    translation_task_scale_m=0.02,
    rotation_task_scale_degrees=2.0,
    measurement_sigma_px=1.0,
    inlier_sigma_px=4.0,
    residual_clip_px=12.0,
    regularization=1e-4,
):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    if p2d.shape[0] != p3d.shape[0] or p3d.shape[0] < 6:
        return {}
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    projected, depth = _project_points_np(p3d, K, pose_w2c)
    residual = projected - (p2d + 0.5)
    valid = (
        np.isfinite(p3d).all(axis=1)
        & np.isfinite(residual).all(axis=1)
        & np.isfinite(depth)
        & (depth > 1e-8)
    )
    if np.count_nonzero(valid) < 6:
        return {}
    p3d = p3d[valid]
    residual = residual[valid]
    p3d_h = np.concatenate(
        [p3d, np.ones((p3d.shape[0], 1), dtype=np.float64)], axis=1
    )
    cam = (pose_w2c @ p3d_h.T)[:3].T
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    jacobians = []
    for x, y, z in cam:
        dproj = np.array(
            [[fx / z, 0.0, -fx * x / (z * z)], [0.0, fy / z, -fy * y / (z * z)]],
            dtype=np.float64,
        )
        skew = np.array(
            [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
            dtype=np.float64,
        )
        jacobians.append(
            dproj @ np.concatenate([np.eye(3, dtype=np.float64), -skew], axis=1)
        )
    jacobians = np.stack(jacobians, axis=0)
    translation_scale = max(float(translation_task_scale_m), 1e-12)
    rotation_scale = max(float(np.deg2rad(rotation_task_scale_degrees)), 1e-12)
    task_scale = np.diag(
        [translation_scale] * 3 + [rotation_scale] * 3
    ).astype(np.float64)
    task_jacobians = jacobians @ task_scale
    residual_norm = np.linalg.norm(residual, axis=1)
    inlier_sigma = max(float(inlier_sigma_px), 1e-8)
    weights = np.exp(-0.5 * np.square(residual_norm / inlier_sigma))
    measurement_sigma = max(float(measurement_sigma_px), 1e-8)
    information = float(regularization) * np.eye(6, dtype=np.float64)
    information += np.einsum(
        "n,nai,naj->ij",
        weights / (measurement_sigma * measurement_sigma),
        task_jacobians,
        task_jacobians,
    )
    clip = max(float(residual_clip_px), 0.0)
    residual_scale = np.minimum(
        1.0,
        clip / np.maximum(residual_norm, 1e-12),
    )
    clipped_residual = residual * residual_scale[:, None]
    bias_gradient = np.einsum(
        "n,nai,na->i",
        weights / (measurement_sigma * measurement_sigma),
        task_jacobians,
        clipped_residual,
    )
    try:
        delta_task = -np.linalg.solve(information, bias_gradient)
    except np.linalg.LinAlgError:
        return {}
    translation_bias = delta_task[:3] * translation_scale
    rotation_bias_degrees = np.rad2deg(
        np.linalg.norm(delta_task[3:] * rotation_scale)
    )
    weight_sum = float(np.sum(weights))
    effective_count = float(
        weight_sum * weight_sum / max(float(np.sum(np.square(weights))), 1e-12)
    )
    return {
        f"{prefix}_pose_bias_translation_x_m": float(translation_bias[0]),
        f"{prefix}_pose_bias_translation_y_m": float(translation_bias[1]),
        f"{prefix}_pose_bias_translation_z_m": float(translation_bias[2]),
        f"{prefix}_pose_bias_translation_norm_m": float(
            np.linalg.norm(translation_bias)
        ),
        f"{prefix}_pose_bias_rotation_norm_degrees": float(rotation_bias_degrees),
        f"{prefix}_pose_bias_soft_inlier_count": weight_sum,
        f"{prefix}_pose_bias_effective_count": effective_count,
    }


def sparse_correspondence_diagnostics(
    p2d,
    p3d,
    K,
    pose_w2c,
    inliers,
    width,
    height,
    *,
    gt_pose_w2c=None,
    grid_rows=4,
    grid_cols=4,
    voxel_size=0.25,
    translation_task_scale_m=0.02,
    rotation_task_scale_degrees=2.0,
):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    inliers = inliers[(inliers >= 0) & (inliers < p2d.shape[0])]
    diagnostics = {
        "sparse_diag_match_count": int(p2d.shape[0]),
        "sparse_diag_inlier_count": int(inliers.shape[0]),
        "sparse_diag_inlier_ratio": float(inliers.shape[0] / max(float(p2d.shape[0]), 1.0)),
    }
    if p2d.shape[0] == 0:
        return diagnostics
    observed = p2d + 0.5
    projected, depth = _project_points_np(p3d, K, pose_w2c)
    residual = np.linalg.norm(projected - observed, axis=1)
    valid_projection = np.isfinite(projected).all(axis=1)
    diagnostics["sparse_diag_all_est_projected_ratio"] = float(
        np.mean(valid_projection)
    )
    diagnostics.update(_residual_stats("sparse_diag_all_est_reproj_px", residual))
    diagnostics.update(_occupancy_stats_2d("sparse_diag_all", p2d, width, height, grid_rows, grid_cols))
    diagnostics.update(_occupancy_stats_3d("sparse_diag_all", p3d, voxel_size))
    finite_depth = depth[np.isfinite(depth) & (depth > 1e-8)]
    if finite_depth.size:
        diagnostics.update(
            {
                "sparse_diag_all_depth_mean": float(np.mean(finite_depth)),
                "sparse_diag_all_depth_std": float(np.std(finite_depth)),
                "sparse_diag_all_depth_range": float(np.max(finite_depth) - np.min(finite_depth)),
            }
        )
    if inliers.shape[0] > 0:
        inlier_p2d = p2d[inliers]
        inlier_p3d = p3d[inliers]
        inlier_depth = depth[inliers]
        diagnostics["sparse_diag_inlier_est_projected_ratio"] = float(
            np.mean(valid_projection[inliers])
        )
        diagnostics.update(_residual_stats("sparse_diag_inlier_est_reproj_px", residual[inliers]))
        diagnostics.update(_occupancy_stats_2d("sparse_diag_inlier", inlier_p2d, width, height, grid_rows, grid_cols))
        diagnostics.update(_occupancy_stats_3d("sparse_diag_inlier", inlier_p3d, voxel_size))
        diagnostics.update(
            _pose_information_stats(
                "sparse_diag_inlier",
                inlier_p3d,
                K,
                pose_w2c,
                translation_task_scale_m=translation_task_scale_m,
                rotation_task_scale_degrees=rotation_task_scale_degrees,
            )
        )
        finite_inlier_depth = inlier_depth[
            np.isfinite(inlier_depth) & (inlier_depth > 1e-8)
        ]
        if finite_inlier_depth.size:
            diagnostics.update(
                {
                    "sparse_diag_inlier_depth_mean": float(np.mean(finite_inlier_depth)),
                    "sparse_diag_inlier_depth_std": float(np.std(finite_inlier_depth)),
                    "sparse_diag_inlier_depth_range": float(
                        np.max(finite_inlier_depth) - np.min(finite_inlier_depth)
                    ),
                }
            )
    if gt_pose_w2c is not None:
        gt_projected, _ = _project_points_np(p3d, K, gt_pose_w2c)
        gt_residual = np.linalg.norm(gt_projected - observed, axis=1)
        gt_valid_projection = np.isfinite(gt_projected).all(axis=1)
        diagnostics["sparse_diag_all_gt_projected_ratio"] = float(
            np.mean(gt_valid_projection)
        )
        diagnostics.update(_residual_stats("sparse_diag_all_gt_reproj_px", gt_residual))
        for threshold in (2.0, 4.0, 6.0):
            diagnostics[f"sparse_diag_all_gt_precision_{int(threshold)}px"] = float(
                np.mean(gt_residual <= threshold)
            )
        if inliers.shape[0] > 0:
            diagnostics["sparse_diag_inlier_gt_projected_ratio"] = float(
                np.mean(gt_valid_projection[inliers])
            )
            diagnostics.update(_residual_stats("sparse_diag_inlier_gt_reproj_px", gt_residual[inliers]))
            for threshold in (2.0, 4.0, 6.0):
                diagnostics[f"sparse_diag_inlier_gt_precision_{int(threshold)}px"] = float(
                    np.mean(gt_residual[inliers] <= threshold)
                )
        diagnostics.update(
            _pose_bias_stats(
                "sparse_diag_all_gt",
                p2d,
                p3d,
                K,
                gt_pose_w2c,
                translation_task_scale_m=translation_task_scale_m,
                rotation_task_scale_degrees=rotation_task_scale_degrees,
            )
        )
        if inliers.shape[0] > 0:
            diagnostics.update(
                _pose_bias_stats(
                    "sparse_diag_inlier_gt",
                    p2d[inliers],
                    p3d[inliers],
                    K,
                    gt_pose_w2c,
                    translation_task_scale_m=translation_task_scale_m,
                    rotation_task_scale_degrees=rotation_task_scale_degrees,
                )
            )
        clean_threshold = 4.0
        gt_clean = np.isfinite(gt_residual) & (gt_residual <= clean_threshold)
        gt_clean_inliers = inliers[gt_clean[inliers]] if inliers.size else inliers
        diagnostics.update(
            {
                "sparse_diag_gt_clean4_count": int(gt_clean.sum()),
                "sparse_diag_inlier_gt_clean4_count": int(gt_clean_inliers.size),
                "sparse_diag_inlier_gt_clean4_ratio": float(
                    gt_clean_inliers.size / max(float(inliers.size), 1.0)
                ),
            }
        )
        diagnostics.update(
            _pose_information_stats(
                "sparse_diag_gt_clean4",
                p3d[gt_clean],
                K,
                gt_pose_w2c,
                translation_task_scale_m=translation_task_scale_m,
                rotation_task_scale_degrees=rotation_task_scale_degrees,
            )
        )
        diagnostics.update(
            _pose_information_stats(
                "sparse_diag_inlier_gt_clean4",
                p3d[gt_clean_inliers],
                K,
                gt_pose_w2c,
                translation_task_scale_m=translation_task_scale_m,
                rotation_task_scale_degrees=rotation_task_scale_degrees,
            )
        )
        diagnostics.update(
            _pose_bias_stats(
                "sparse_diag_gt_clean4",
                p2d[gt_clean],
                p3d[gt_clean],
                K,
                gt_pose_w2c,
                translation_task_scale_m=translation_task_scale_m,
                rotation_task_scale_degrees=rotation_task_scale_degrees,
            )
        )
        diagnostics.update(
            _pose_bias_stats(
                "sparse_diag_inlier_gt_clean4",
                p2d[gt_clean_inliers],
                p3d[gt_clean_inliers],
                K,
                gt_pose_w2c,
                translation_task_scale_m=translation_task_scale_m,
                rotation_task_scale_degrees=rotation_task_scale_degrees,
            )
        )
    return diagnostics


def resize_sparse_valid_mask_to_feature_grid(valid_mask, height, width, min_fraction=0.5):
    if valid_mask is None:
        return None
    mask = torch.as_tensor(valid_mask, dtype=torch.float32)
    if mask.dim() == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask.mean(dim=0)
    if mask.dim() != 2:
        raise ValueError(f"Expected sparse valid mask to be 2D, got shape {tuple(mask.shape)}")
    if tuple(mask.shape) != (int(height), int(width)):
        mask = F.interpolate(
            mask[None, None],
            size=(int(height), int(width)),
            mode="area",
        )[0, 0]
    return mask >= float(min_fraction)


def resize_sparse_score_to_feature_grid(score_map, height, width):
    if score_map is None:
        return None
    score = torch.as_tensor(score_map, dtype=torch.float32).detach()
    if score.dim() == 3:
        if score.shape[0] == 1:
            score = score[0]
        elif score.shape[-1] == 1:
            score = score[..., 0]
        else:
            score = score.mean(dim=0)
    if score.dim() != 2:
        raise ValueError(f"Expected sparse support score to be 2D, got shape {tuple(score.shape)}")
    score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    if tuple(score.shape) != (int(height), int(width)):
        score = F.interpolate(
            score[None, None],
            size=(int(height), int(width)),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return score.clamp(0.0, 1.0)


def apply_sparse_support_score_prior(kp_scores, support_score, weight=0.0, min_multiplier=1.0):
    scores = torch.as_tensor(kp_scores, dtype=torch.float32)
    support = torch.as_tensor(support_score, dtype=torch.float32, device=scores.device)
    if scores.dim() == 3 and scores.shape[0] == 1 and support.dim() == 2:
        support = support[None]
    if support.shape != scores.shape:
        raise ValueError(
            "support_score must match keypoint score grid shape after resizing: "
            f"{tuple(support.shape)} vs {tuple(scores.shape)}"
        )
    weight = float(weight or 0.0)
    min_multiplier = float(min_multiplier)
    if weight == 0.0 and abs(min_multiplier - 1.0) < 1e-8:
        return scores, {
            "sparse_support_score_prior_weight": weight,
            "sparse_support_score_prior_min_multiplier": min_multiplier,
            "sparse_support_score_prior_multiplier_mean": 1.0,
            "sparse_support_score_prior_score_mean": float(support.float().mean().item()) if support.numel() else 0.0,
        }
    support = torch.nan_to_num(support, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    multiplier = min_multiplier + (1.0 + weight - min_multiplier) * support
    adjusted = scores * multiplier.to(dtype=scores.dtype, device=scores.device)
    return adjusted, {
        "sparse_support_score_prior_weight": weight,
        "sparse_support_score_prior_min_multiplier": min_multiplier,
        "sparse_support_score_prior_multiplier_mean": float(multiplier.float().mean().item()) if multiplier.numel() else 1.0,
        "sparse_support_score_prior_score_mean": float(support.float().mean().item()) if support.numel() else 0.0,
    }


def apply_dense_query_valid_mask_to_corr(
    corr_matrix,
    valid_mask,
    query_height,
    query_width,
    min_fraction=0.5,
    fill_value=-1e9,
):
    corr = torch.as_tensor(corr_matrix)
    if valid_mask is None:
        return corr, {
            "dense_valid_mask_enabled": False,
            "dense_valid_mask_valid_cells": int(corr.shape[-2]) if corr.dim() >= 2 else 0,
            "dense_valid_mask_valid_frac": 1.0,
        }
    if corr.dim() != 3:
        raise ValueError(f"Expected dense correlation matrix as BxNxM, got shape {tuple(corr.shape)}")
    query_height = int(query_height)
    query_width = int(query_width)
    expected = query_height * query_width
    if int(corr.shape[-2]) != expected:
        raise ValueError(
            "dense valid mask grid does not match correlation query cells: "
            f"{query_height}x{query_width}={expected} vs {int(corr.shape[-2])}"
        )
    mask = resize_sparse_valid_mask_to_feature_grid(
        valid_mask,
        query_height,
        query_width,
        min_fraction=min_fraction,
    ).reshape(-1)
    mask = mask.to(device=corr.device)
    masked = corr.clone()
    invalid = ~mask
    if invalid.any():
        masked[:, invalid, :] = torch.as_tensor(fill_value, dtype=masked.dtype, device=masked.device)
    return masked, {
        "dense_valid_mask_enabled": True,
        "dense_valid_mask_valid_cells": int(mask.sum().item()),
        "dense_valid_mask_valid_frac": float(mask.float().mean().item()) if mask.numel() else 0.0,
    }


def filter_sparse_keypoints_by_valid_mask(kp_ids, valid_mask, height, width, min_fraction=0.5):
    kp_ids = torch.as_tensor(kp_ids, dtype=torch.long)
    raw_count = int(kp_ids.numel())
    if valid_mask is None:
        return kp_ids, {
            "detected_keypoints_raw": raw_count,
            "detected_keypoints": raw_count,
            "sparse_valid_mask_filtered_keypoints": 0,
            "sparse_valid_mask_valid_frac": 1.0,
        }
    mask = resize_sparse_valid_mask_to_feature_grid(
        valid_mask,
        int(height),
        int(width),
        min_fraction=min_fraction,
    ).reshape(-1)
    if kp_ids.device != mask.device:
        mask = mask.to(kp_ids.device)
    keep = mask[kp_ids] if raw_count else torch.zeros(0, dtype=torch.bool, device=kp_ids.device)
    filtered = kp_ids[keep]
    filtered_count = int(raw_count - filtered.numel())
    return filtered, {
        "detected_keypoints_raw": raw_count,
        "detected_keypoints": int(filtered.numel()),
        "sparse_valid_mask_filtered_keypoints": filtered_count,
        "sparse_valid_mask_valid_frac": float(mask.float().mean().item()) if mask.numel() else 0.0,
    }


def select_sparse_keypoints_by_valid_mask(
    kp_ids,
    valid_mask,
    height,
    width,
    target_count=0,
    min_fraction=0.5,
    refill_invalid=True,
):
    kp_ids = torch.as_tensor(kp_ids, dtype=torch.long)
    raw_count = int(kp_ids.numel())
    target_count = int(target_count or 0)
    if valid_mask is None:
        if target_count > 0:
            kp_ids = kp_ids[:target_count]
        count = int(kp_ids.numel())
        return kp_ids, {
            "detected_keypoints_raw": raw_count,
            "detected_keypoints": count,
            "sparse_valid_mask_filtered_keypoints": raw_count - count,
            "sparse_valid_mask_valid_frac": 1.0,
            "sparse_valid_mask_invalid_candidates": 0,
            "sparse_valid_mask_selected_valid_keypoints": 0,
            "sparse_valid_mask_refill_keypoints": 0,
        }

    mask = resize_sparse_valid_mask_to_feature_grid(
        valid_mask,
        int(height),
        int(width),
        min_fraction=min_fraction,
    ).reshape(-1)
    if kp_ids.device != mask.device:
        mask = mask.to(kp_ids.device)
    keep = mask[kp_ids] if raw_count else torch.zeros(0, dtype=torch.bool, device=kp_ids.device)
    valid_kp_ids = kp_ids[keep]
    invalid_kp_ids = kp_ids[~keep]
    if target_count <= 0:
        selected = valid_kp_ids
        refill_count = 0
    else:
        selected_valid = valid_kp_ids[:target_count]
        if bool(refill_invalid) and selected_valid.numel() < target_count:
            refill = invalid_kp_ids[: target_count - selected_valid.numel()]
        else:
            refill = invalid_kp_ids[:0]
        selected = torch.cat([selected_valid, refill], dim=0)
        refill_count = int(refill.numel())
    selected_valid_count = int((mask[selected].sum().item() if selected.numel() else 0))
    selected_count = int(selected.numel())
    return selected, {
        "detected_keypoints_raw": raw_count,
        "detected_keypoints": selected_count,
        "sparse_valid_mask_filtered_keypoints": raw_count - selected_count,
        "sparse_valid_mask_valid_frac": float(mask.float().mean().item()) if mask.numel() else 0.0,
        "sparse_valid_mask_invalid_candidates": int((~keep).sum().item()) if keep.numel() else 0,
        "sparse_valid_mask_selected_valid_keypoints": selected_valid_count,
        "sparse_valid_mask_refill_keypoints": refill_count,
    }


def _geometry_selector_from_config(sparse_config, width, height):
    cfg = sparse_config.get("geometry_balance", None)
    if not cfg or not cfg.get("enabled", False):
        return None
    post_cfg = cfg.get("post", {})
    post_enabled = bool(post_cfg.get("enabled", False))
    return GeometryBalancedSelector(
        image_width=width,
        image_height=height,
        grid_rows=cfg.get("grid_rows", 4),
        grid_cols=cfg.get("grid_cols", 4),
        max_per_cell=cfg.get("max_per_cell", 64),
        voxel_size=cfg.get("voxel_size", 0.25),
        max_per_voxel=cfg.get("max_per_voxel", 64),
        max_matches=cfg.get("max_matches", 0),
        post_max_matches=post_cfg.get("max_matches", 0) if post_enabled else 0,
        post_candidate_pool=post_cfg.get("candidate_pool", 1024),
        post_regularization=post_cfg.get("regularization", 1e-4),
        post_score_weight=post_cfg.get("score_weight", 1e-3),
    )


def resolve_artifact_path(model_path, artifact_path, artifact_model_path=None):
    if os.path.isabs(artifact_path):
        return artifact_path
    root = artifact_model_path or model_path
    return os.path.join(root, artifact_path)


def file_sha256(path, chunk_size=1024 * 1024):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_file_manifest(root, relative_paths):
    """Hash a named input set so image/preprocessing changes are reproducible."""
    root = os.path.realpath(os.fspath(root))
    digest = hashlib.sha256()
    found = 0
    missing = []
    normalized_paths = sorted(
        {os.path.normpath(str(path)) for path in relative_paths}
    )
    for relative_path in normalized_paths:
        file_path = os.path.realpath(os.path.join(root, relative_path))
        file_digest = file_sha256(file_path)
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        if file_digest is None:
            digest.update(b"MISSING")
            missing.append(relative_path)
        else:
            digest.update(file_digest.encode("ascii"))
            found += 1
        digest.update(b"\n")
    return {
        "root": root,
        "requested_count": len(normalized_paths),
        "found_count": found,
        "missing_count": len(missing),
        "missing_paths": missing,
        "sha256": digest.hexdigest(),
    }


def _protocol_camera_records(cameras):
    """Read DataLoader camera metadata without consuming its image iterator."""
    loader_dataset = getattr(cameras, "dataset", None)
    scene = getattr(loader_dataset, "scene", None)
    split = getattr(loader_dataset, "split", None)
    scene_info = getattr(scene, "scene_info", None)
    if scene_info is not None and split in {"train", "test"}:
        if split == "train":
            return list(scene_info.train_cameras)
        return list(scene_info.test_cameras)
    return list(cameras)


def _protocol_loaded_shape(camera, resolution):
    image = getattr(camera, "original_image", None)
    if image is not None and len(image.shape) >= 2:
        return int(image.shape[-2]), int(image.shape[-1])

    width = getattr(camera, "width", None)
    height = getattr(camera, "height", None)
    if width is None or height is None:
        width = getattr(camera, "image_width", None)
        height = getattr(camera, "image_height", None)
    if width is None or height is None:
        raise ValueError(
            f"Camera {getattr(camera, 'image_name', '<unnamed>')} has no image shape metadata"
        )

    width = int(width)
    height = int(height)
    if resolution in {1, 2, 3, 4, 8}:
        return round(height / float(resolution)), round(width / float(resolution))
    if resolution == -2:
        return 320, 480
    if resolution == -1:
        scale = width / 1600.0 if width > 1600 else 1.0
    else:
        scale = width / float(resolution)
    return int(height / scale), int(width / scale)


def build_evaluation_protocol(dataset, args, cameras):
    """Materialize every image-side setting that can change sparse candidates."""
    camera_records = _protocol_camera_records(cameras)
    camera_names = [str(camera.image_name) for camera in camera_records]
    image_root = os.path.join(dataset.source_path, dataset.images)
    image_manifest = named_file_manifest(image_root, camera_names)
    if image_manifest["missing_count"]:
        raise FileNotFoundError(
            "evaluation image manifest is incomplete under "
            f"{image_manifest['root']}: {image_manifest['missing_paths'][:3]}"
        )

    shape_counts = Counter()
    for camera in camera_records:
        shape_counts[_protocol_loaded_shape(camera, dataset.resolution)] += 1
    loaded_shapes = [
        {"height": height, "width": width, "count": count}
        for (height, width), count in sorted(shape_counts.items())
    ]
    camera_names_digest = hashlib.sha256(
        ("\n".join(camera_names) + "\n").encode("utf-8")
    ).hexdigest()
    protocol = {
        "schema_version": 1,
        "source_path": os.path.realpath(dataset.source_path),
        "images": str(dataset.images),
        "resolution": dataset.resolution,
        "longest_edge": int(dataset.longest_edge),
        "feature_type": str(dataset.feature_type),
        "gaussian_type": str(dataset.gaussian_type),
        "evaluation_camera_subset": str(args.evaluation_camera_subset),
        "evaluation_camera_list": (
            os.path.realpath(getattr(args, "evaluation_camera_list", ""))
            if getattr(args, "evaluation_camera_list", "")
            else ""
        ),
        "evaluation_camera_list_sha256": (
            file_sha256(getattr(args, "evaluation_camera_list", ""))
            if getattr(args, "evaluation_camera_list", "")
            else None
        ),
        "evaluation_camera_count": len(camera_records),
        "camera_names_sha256": camera_names_digest,
        "loaded_image_shapes": loaded_shapes,
        "query_image_manifest": image_manifest,
        "candidate_split": {
            "query_ratio": float(args.candidate_query_ratio),
            "validation_ratio": float(args.candidate_validation_ratio),
            "mode": str(args.candidate_split_mode),
            "seed": int(args.candidate_split_seed),
            "direct_holdout": bool(args.candidate_direct_validation_holdout),
        },
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return protocol


def tensor_sha256(value):
    tensor = torch.as_tensor(value).detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def landmark_feature_delta(reference, candidate):
    reference = F.normalize(torch.as_tensor(reference).detach().float().reshape(reference.shape[0], -1), dim=1)
    candidate = F.normalize(torch.as_tensor(candidate).detach().float().reshape(candidate.shape[0], -1), dim=1)
    if reference.shape != candidate.shape:
        raise ValueError(
            "landmark feature tensors must have identical shapes: "
            f"reference={tuple(reference.shape)} candidate={tuple(candidate.shape)}"
        )
    delta = torch.linalg.norm(candidate - reference, dim=1)
    cosine = torch.sum(candidate * reference, dim=1)
    return {
        "l2_mean": float(delta.mean().item()) if delta.numel() else 0.0,
        "l2_p95": float(torch.quantile(delta, 0.95).item()) if delta.numel() else 0.0,
        "l2_max": float(delta.max().item()) if delta.numel() else 0.0,
        "cosine_mean": float(cosine.mean().item()) if cosine.numel() else 1.0,
    }


def load_sparse_candidate_state(path):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Invalid sparse candidate state: {path}")
    return state


def validate_sampled_indices(sampled_idx, point_count):
    if isinstance(sampled_idx, torch.Tensor):
        idx = sampled_idx.detach().reshape(-1).to(dtype=torch.long)
        idx_cpu = idx.cpu()
    else:
        idx_cpu = torch.tensor([int(idx) for idx in sampled_idx], dtype=torch.long)
        idx = idx_cpu
    if idx_cpu.numel() == 0:
        raise ValueError(f"Landmark indices are empty for point_count={point_count}.")
    min_idx = int(idx_cpu.min().item())
    max_idx = int(idx_cpu.max().item())
    if min_idx < 0 or max_idx >= int(point_count):
        raise ValueError(
            "Landmark indices out of bounds: "
            f"point_count={int(point_count)}, min={min_idx}, max={max_idx}."
        )
    return idx


def load_candidate_teacher_landmark_features(
    path,
    landmark_indices,
    *,
    expected_feature_dim=None,
    device=None,
    dtype=torch.float32,
):
    state = load_sparse_candidate_state(path)
    if not isinstance(state, dict) or "landmark_features" not in state:
        raise ValueError(f"Invalid sparse candidate teacher state: {path}")
    expected_indices = torch.as_tensor(landmark_indices, dtype=torch.long).reshape(-1).cpu()
    state_indices = torch.as_tensor(
        state.get("landmark_indices", []), dtype=torch.long
    ).reshape(-1).cpu()
    if not torch.equal(state_indices, expected_indices):
        raise ValueError(
            "sparse candidate teacher state is not aligned with detector landmarks: "
            f"state_count={state_indices.numel()} expected_count={expected_indices.numel()}"
        )
    features = torch.as_tensor(state["landmark_features"], dtype=dtype)
    if features.ndim < 2 or features.shape[0] != expected_indices.numel():
        raise ValueError(
            "sparse candidate teacher feature count does not match detector landmarks: "
            f"features={features.shape[0] if features.ndim else 0} "
            f"expected={expected_indices.numel()}"
        )
    features = features.reshape(expected_indices.numel(), -1)
    if expected_feature_dim is not None and features.shape[1] != int(expected_feature_dim):
        raise ValueError(
            "sparse candidate teacher feature dimension does not match map features: "
            f"state_dim={features.shape[1]} expected_dim={int(expected_feature_dim)}"
        )
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("sparse candidate teacher features contain non-finite values")
    features = F.normalize(features, dim=1)
    if device is not None:
        features = features.to(device=device, dtype=dtype)
    return features, state


def load_candidate_teacher_landmark_geometry(
    state,
    landmark_indices,
    *,
    device=None,
    dtype=torch.float32,
):
    """Load materialized localization anchors with exact sparse-ID alignment."""
    if not isinstance(state, dict) or "landmark_xyz" not in state:
        return None
    expected_indices = torch.as_tensor(
        landmark_indices, dtype=torch.long
    ).reshape(-1).cpu()
    state_indices = torch.as_tensor(
        state.get("landmark_indices", []), dtype=torch.long
    ).reshape(-1).cpu()
    if not torch.equal(state_indices, expected_indices):
        raise ValueError(
            "sparse localization geometry is not aligned with detector landmarks: "
            f"state_count={state_indices.numel()} expected_count={expected_indices.numel()}"
        )
    xyz = torch.as_tensor(state["landmark_xyz"], dtype=dtype).reshape(-1, 3)
    if xyz.shape[0] != expected_indices.numel():
        raise ValueError(
            "sparse localization geometry count does not match detector landmarks"
        )
    if not bool(torch.isfinite(xyz).all().item()):
        raise ValueError("sparse localization geometry contains non-finite values")
    if device is not None:
        xyz = xyz.to(device=device, dtype=dtype)
    return xyz


def remap_sampled_indices_from_source_index(
    sampled_idx,
    source_index,
    return_missing=False,
    fill_missing=False,
    fill_scores=None,
    remap_scores=None,
    source_xyz=None,
    current_xyz=None,
    prefer_source_distance=False,
    max_source_distance=None,
):
    sampled_idx = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).cpu()
    source_index = torch.as_tensor(source_index, dtype=torch.long).reshape(-1).cpu()
    if remap_scores is not None:
        remap_scores = torch.as_tensor(remap_scores, dtype=torch.float32).reshape(-1).cpu()
        if remap_scores.numel() != source_index.numel():
            raise ValueError(
                "remap_scores must contain one score per current point: "
                f"got {remap_scores.numel()} scores for {source_index.numel()} points."
            )
    if source_xyz is not None or current_xyz is not None:
        if source_xyz is None or current_xyz is None:
            raise ValueError("source_xyz and current_xyz must be provided together.")
        source_xyz = torch.as_tensor(source_xyz, dtype=torch.float32).reshape(source_index.numel(), -1).cpu()
        current_xyz = torch.as_tensor(current_xyz, dtype=torch.float32).reshape(source_index.numel(), -1).cpu()
        if source_xyz.shape[0] != source_index.numel() or current_xyz.shape[0] != source_index.numel():
            raise ValueError(
                "source_xyz/current_xyz must contain one row per current point: "
                f"got {source_xyz.shape[0]} and {current_xyz.shape[0]} rows for {source_index.numel()} points."
            )
        source_distance = torch.linalg.norm(current_xyz[:, :3] - source_xyz[:, :3], dim=-1)
    else:
        source_distance = None

    candidates_for_source = {}
    for current_idx, source_id in enumerate(source_index.tolist()):
        source_id = int(source_id)
        candidates_for_source.setdefault(source_id, []).append(int(current_idx))

    def choose_candidate(candidates):
        if not candidates:
            return None
        if prefer_source_distance and source_distance is not None:
            order = sorted(
                candidates,
                key=lambda idx: (
                    float(source_distance[idx].item()),
                    -float(remap_scores[idx].item()) if remap_scores is not None else 0.0,
                    idx,
                ),
            )
            best = order[0]
            if max_source_distance is not None and float(source_distance[best].item()) > float(max_source_distance):
                return None
            return best
        if remap_scores is None:
            return candidates[0]
        return max(candidates, key=lambda idx: (float(remap_scores[idx].item()), -idx))

    remapped = []
    missing = []
    for source_id in sampled_idx.tolist():
        current_idx = choose_candidate(candidates_for_source.get(int(source_id), []))
        if current_idx is None:
            missing.append(int(source_id))
        else:
            remapped.append(current_idx)

    if fill_missing and len(missing) > 0:
        selected = set(remapped)
        if fill_scores is None:
            fill_scores = torch.zeros(source_index.numel(), dtype=torch.float32)
        fill_scores = torch.as_tensor(fill_scores, dtype=torch.float32).reshape(-1).cpu()
        if fill_scores.numel() != source_index.numel():
            raise ValueError(
                "fill_scores must contain one score per current point: "
                f"got {fill_scores.numel()} scores for {source_index.numel()} points."
            )
        order = torch.argsort(fill_scores, descending=True).tolist()
        for current_idx in order:
            if current_idx in selected:
                continue
            remapped.append(int(current_idx))
            selected.add(int(current_idx))
            if len(remapped) >= sampled_idx.numel():
                break

    remapped = torch.tensor(remapped, dtype=torch.long)
    missing = torch.tensor(missing, dtype=torch.long)
    if return_missing:
        return remapped, missing
    return remapped


def landmark_prior_from_meta(meta, landmark_count, sampled_indices=None):
    if meta is None:
        return None
    landmark_count = int(landmark_count)
    score = meta.get("candidate_quality", meta.get("score", meta.get("utility", None)))
    full_score = meta.get("full_candidate_quality", meta.get("full_score", None))
    meta_indices = meta.get("landmark_indices", None)
    if sampled_indices is not None:
        sampled_indices = torch.as_tensor(sampled_indices, dtype=torch.long).reshape(-1).cpu()
        if sampled_indices.numel() != landmark_count:
            raise ValueError(
                "sampled_indices must contain one index per sparse landmark: "
                f"got {sampled_indices.numel()} for landmark_count={landmark_count}."
            )
    if meta_indices is not None:
        meta_indices = torch.as_tensor(meta_indices, dtype=torch.long).reshape(-1).cpu()

    if full_score is not None and sampled_indices is not None:
        full_score = torch.as_tensor(full_score, dtype=torch.float32).reshape(-1)
        if sampled_indices.numel() > 0 and int(sampled_indices.max().item()) >= full_score.numel():
            raise ValueError(
                "landmark prior sampled index is outside full_score: "
                f"max_index={int(sampled_indices.max().item())}, full_score={full_score.numel()}."
            )
        return full_score[sampled_indices].clone()

    if score is not None:
        score = torch.as_tensor(score, dtype=torch.float32).reshape(-1)
        if score.numel() == landmark_count:
            if sampled_indices is not None and meta_indices is not None and not torch.equal(meta_indices, sampled_indices):
                if full_score is not None:
                    full_score = torch.as_tensor(full_score, dtype=torch.float32).reshape(-1)
                    return full_score[sampled_indices].clone()
                raise ValueError("landmark prior score indices do not match current sampled landmarks.")
            return score.clone()
        if full_score is None:
            raise ValueError(
                "landmark prior score length must match sparse landmark count: "
                f"score={score.numel()}, landmark_count={landmark_count}."
            )

    if full_score is not None and meta_indices is not None:
        if meta_indices.numel() != landmark_count:
            raise ValueError(
                "landmark prior meta indices must match sparse landmark count: "
                f"indices={meta_indices.numel()}, landmark_count={landmark_count}."
            )
        full_score = torch.as_tensor(full_score, dtype=torch.float32).reshape(-1)
        if meta_indices.numel() > 0 and int(meta_indices.max().item()) >= full_score.numel():
            raise ValueError(
                "landmark prior meta index is outside full_score: "
                f"max_index={int(meta_indices.max().item())}, full_score={full_score.numel()}."
            )
        return full_score[meta_indices].clone()

    return None


class STDLoc:
    def __init__(self, gaussians, config):
        self.gaussians = gaussians
        self.config = config
        sparse_config = config["sparse"]
        sampled_idx_path = resolve_artifact_path(
            config["model_path"],
            sparse_config["landmark_path"],
            sparse_config.get("landmark_model_path"),
        )
        with open(sampled_idx_path, "rb") as handle:
            sampled_idx = pickle.load(handle)
        sampled_idx = validate_sampled_indices(sampled_idx, gaussians.get_xyz.shape[0]).detach().cpu()
        self.full_primitive_retrieval = bool(
            sparse_config.get("full_primitive_retrieval", False)
        )
        if self.full_primitive_retrieval:
            self.landmark_indices = torch.arange(
                gaussians.get_xyz.shape[0], dtype=torch.long
            )
            self.landmarks = gaussians
        else:
            self.landmark_indices = sampled_idx
            self.landmarks = sample_gaussians(gaussians, self.landmark_indices)
        map_features = self.landmarks.get_loc_feature.detach().clone()
        map_features_flat = map_features.reshape(map_features.shape[0], -1)
        legacy_state_path = sparse_config.get("candidate_teacher_state_path", "")
        feature_override_path = sparse_config.get(
            "landmark_feature_override_path",
            sparse_config.get("landmark_feature_path", ""),
        )
        override_landmark_features = bool(
            sparse_config.get("override_landmark_features", False)
        )
        if override_landmark_features and not feature_override_path:
            if legacy_state_path:
                feature_override_path = legacy_state_path
                warnings.warn(
                    "Using legacy candidate_teacher_state_path as an explicit landmark "
                    "feature override. Set landmark_feature_override_path instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "override_landmark_features=true requires "
                    "landmark_feature_override_path"
                )
        elif feature_override_path and not override_landmark_features:
            warnings.warn(
                "landmark feature override path is configured but "
                "override_landmark_features=false; map checkpoint features remain active",
                RuntimeWarning,
                stacklevel=2,
            )

        self.candidate_teacher_state = None
        self.landmark_feature_override_state = None
        full_feature_override_path = None
        if override_landmark_features:
            full_feature_override_path = resolve_artifact_path(
                config["model_path"],
                feature_override_path,
                sparse_config.get(
                    "landmark_feature_override_model_path",
                    sparse_config.get(
                        "candidate_teacher_state_model_path",
                        sparse_config.get("landmark_model_path"),
                    ),
                ),
            )
            current_features = self.landmarks.get_loc_feature
            current_flat = current_features.reshape(current_features.shape[0], -1)
            if self.full_primitive_retrieval:
                state = load_sparse_candidate_state(full_feature_override_path)
                state_indices = validate_sampled_indices(
                    state.get("landmark_indices", []), current_flat.shape[0]
                ).to(current_flat.device)
                override = torch.as_tensor(
                    state["landmark_features"],
                    device=current_flat.device,
                    dtype=current_flat.dtype,
                ).reshape(state_indices.numel(), -1)
                if override.shape[1] != current_flat.shape[1]:
                    raise ValueError("full-map feature override dimension mismatch")
                merged = current_flat.clone()
                merged[state_indices] = F.normalize(override, dim=1)
                self.landmarks._loc_feature = torch.nn.Parameter(
                    merged.reshape_as(current_features), requires_grad=False
                )
                if "landmark_xyz" in state:
                    override_xyz = load_candidate_teacher_landmark_geometry(
                        {
                            **state,
                            "landmark_indices": state_indices.detach().cpu(),
                        },
                        state_indices.detach().cpu(),
                        device=self.landmarks.get_xyz.device,
                        dtype=self.landmarks.get_xyz.dtype,
                    )
                    merged_xyz = self.landmarks.get_xyz.detach().clone()
                    merged_xyz[state_indices] = override_xyz
                    self.landmarks._xyz = torch.nn.Parameter(
                        merged_xyz, requires_grad=False
                    )
                self.landmark_feature_override_state = state
            else:
                override, self.landmark_feature_override_state = load_candidate_teacher_landmark_features(
                    full_feature_override_path,
                    self.landmark_indices,
                    expected_feature_dim=current_flat.shape[1],
                    device=current_features.device,
                    dtype=current_features.dtype,
                )
                self.landmarks._loc_feature = override.reshape_as(current_features)
                override_xyz = load_candidate_teacher_landmark_geometry(
                    self.landmark_feature_override_state,
                    self.landmark_indices,
                    device=self.landmarks.get_xyz.device,
                    dtype=self.landmarks.get_xyz.dtype,
                )
                if override_xyz is not None:
                    self.landmarks._xyz = torch.nn.Parameter(
                        override_xyz, requires_grad=False
                    )

        landmark_meta_path = sparse_config.get("landmark_meta_path", "detector/landmark_meta.pt")
        full_meta_path = resolve_artifact_path(
            config["model_path"],
            landmark_meta_path,
            sparse_config.get("landmark_meta_model_path", sparse_config.get("landmark_model_path")),
        )
        self.landmark_meta = torch.load(full_meta_path) if os.path.exists(full_meta_path) else None

        self.pair_scorer = None
        self.pair_scorer_threshold = float(
            sparse_config.get("pair_scorer_threshold", 0.0)
        )
        pair_scorer_state_path = sparse_config.get("pair_scorer_state_path", "")
        need_candidate_state = bool(
            sparse_config.get("use_candidate_pair_scorer", False)
            or sparse_config.get("use_candidate_dustbin", False)
        )
        if need_candidate_state:
            if not pair_scorer_state_path:
                pair_scorer_state_path = legacy_state_path
                if pair_scorer_state_path:
                    warnings.warn(
                        "candidate_teacher_state_path is deprecated for scorer loading; "
                        "set pair_scorer_state_path explicitly",
                        DeprecationWarning,
                        stacklevel=2,
                    )
            if not pair_scorer_state_path:
                raise ValueError(
                    "candidate scorer/dustbin requires pair_scorer_state_path"
                )
            full_pair_scorer_state_path = resolve_artifact_path(
                config["model_path"],
                pair_scorer_state_path,
                sparse_config.get(
                    "pair_scorer_state_model_path",
                    sparse_config.get(
                        "candidate_teacher_state_model_path",
                        sparse_config.get("landmark_model_path"),
                    ),
                ),
            )
            self.candidate_teacher_state = load_sparse_candidate_state(
                full_pair_scorer_state_path
            )
        else:
            full_pair_scorer_state_path = None

        if sparse_config.get("use_candidate_pair_scorer", False):
            if not isinstance(self.candidate_teacher_state, dict):
                raise ValueError("candidate pair scorer requires a pair scorer state")
            validate_candidate_frontend_compatibility(
                self.candidate_teacher_state.get("config", {}),
                sparse_config,
            )
            scorer_config = self.candidate_teacher_state.get("pair_scorer_config")
            scorer_state = self.candidate_teacher_state.get("pair_scorer_state_dict")
            if not isinstance(scorer_config, dict) or not isinstance(scorer_state, dict):
                raise ValueError("candidate teacher state does not contain a trained pair scorer")
            architecture = scorer_config.get("architecture", "cosine_residual_v1")
            self.pair_scorer = SparsePairScorer(
                input_dim=int(scorer_config.get("input_dim", 6)),
                hidden_dim=int(scorer_config.get("hidden_dim", 16)),
                architecture=architecture,
                descriptor_dim=int(scorer_config.get("descriptor_dim", 0)),
            )
            self.pair_scorer.load_state_dict(scorer_state)
            self.pair_scorer.eval().cuda()
            if sparse_config.get(
                "use_candidate_pair_scorer_calibrated_threshold", False
            ):
                if "pair_scorer_threshold" not in self.candidate_teacher_state:
                    raise ValueError(
                        "candidate state does not contain a held-out calibrated pair scorer threshold"
                    )
                self.pair_scorer_threshold = float(
                    self.candidate_teacher_state["pair_scorer_threshold"]
                )

        self.pair_measurement_head = None
        self.pair_measurement_state = None
        self.pair_measurement_threshold = float(
            sparse_config.get("pair_measurement_threshold", 0.0)
        )
        full_pair_measurement_state_path = None
        if sparse_config.get("use_pair_measurement", False):
            if self.full_primitive_retrieval:
                raise ValueError(
                    "full primitive retrieval currently requires PairMeasurement disabled; "
                    "the existing head was trained on the fixed bank"
                )
            if self.pair_scorer is not None:
                raise ValueError(
                    "pair measurement and legacy pair scorer cannot be enabled together"
                )
            pair_measurement_state_path = sparse_config.get(
                "pair_measurement_state_path", ""
            )
            if not pair_measurement_state_path:
                raise ValueError(
                    "use_pair_measurement=true requires pair_measurement_state_path"
                )
            full_pair_measurement_state_path = resolve_artifact_path(
                config["model_path"],
                pair_measurement_state_path,
                sparse_config.get(
                    "pair_measurement_state_model_path",
                    sparse_config.get("landmark_model_path"),
                ),
            )
            self.pair_measurement_state = load_sparse_candidate_state(
                full_pair_measurement_state_path
            )
            validate_candidate_frontend_compatibility(
                self.pair_measurement_state.get("config", {}),
                sparse_config,
            )
            measurement_config = self.pair_measurement_state.get(
                "pair_measurement_config"
            )
            measurement_state = self.pair_measurement_state.get(
                "pair_measurement_state_dict"
            )
            if not isinstance(measurement_config, dict) or not isinstance(
                measurement_state, dict
            ):
                raise ValueError(
                    "pair measurement state does not contain a trained measurement head"
                )
            self.pair_measurement_head = PairMeasurementHead(
                descriptor_dim=int(measurement_config["descriptor_dim"]),
                pair_feature_dim=int(
                    measurement_config.get("pair_feature_dim", 6)
                ),
                patch_radius=int(measurement_config.get("patch_radius", 2)),
                hidden_dim=int(measurement_config.get("hidden_dim", 64)),
                max_offset=float(measurement_config.get("max_offset", 2.0)),
                covariance_floor=float(
                    measurement_config.get("covariance_floor", 0.1)
                ),
                use_set_context=bool(
                    measurement_config.get("use_set_context", False)
                ),
                use_geometry_context=bool(
                    measurement_config.get("use_geometry_context", False)
                ),
            )
            current_feature_dim = self.landmarks.get_loc_feature.reshape(
                self.landmarks.get_loc_feature.shape[0], -1
            ).shape[1]
            if self.pair_measurement_head.descriptor_dim != current_feature_dim:
                raise ValueError(
                    "pair measurement descriptor dimension does not match map: "
                    f"state={self.pair_measurement_head.descriptor_dim} "
                    f"map={current_feature_dim}"
                )
            self.pair_measurement_head.load_state_dict(measurement_state)
            self.pair_measurement_head.eval().cuda()
            if sparse_config.get(
                "use_pair_measurement_calibrated_threshold", False
            ):
                if "pair_measurement_threshold" not in self.pair_measurement_state:
                    raise ValueError(
                        "measurement state does not contain a held-out calibrated threshold"
                    )
                self.pair_measurement_threshold = float(
                    self.pair_measurement_state["pair_measurement_threshold"]
                )

        active_features_flat = self.landmarks.get_loc_feature.detach().reshape(
            self.landmarks.get_loc_feature.shape[0], -1
        )
        scorer_training_features = None
        if isinstance(self.candidate_teacher_state, dict):
            candidate_features = self.candidate_teacher_state.get("landmark_features")
            if candidate_features is not None:
                candidate_features = torch.as_tensor(candidate_features)
                if candidate_features.numel() == active_features_flat.numel():
                    scorer_training_features = candidate_features.reshape_as(
                        active_features_flat.cpu()
                    )
        measurement_training_features = None
        if isinstance(self.pair_measurement_state, dict):
            candidate_features = self.pair_measurement_state.get(
                "landmark_features"
            )
            if candidate_features is not None:
                candidate_features = torch.as_tensor(candidate_features)
                if candidate_features.numel() == active_features_flat.numel():
                    measurement_training_features = candidate_features.reshape_as(
                        active_features_flat.cpu()
                    )
        detector_path = resolve_artifact_path(
            config["model_path"],
            sparse_config["detector_path"],
            sparse_config.get("detector_model_path"),
        )
        self.artifact_provenance = {
            "map_model_path": str(config["model_path"]),
            "map_checkpoint_path": config.get("_map_checkpoint_path"),
            "map_checkpoint_sha256": file_sha256(config.get("_map_checkpoint_path")),
            "sampled_idx_path": sampled_idx_path,
            "sampled_idx_sha256": file_sha256(sampled_idx_path),
            "landmark_indices_sha256": tensor_sha256(self.landmark_indices),
            "map_landmark_features_sha256": tensor_sha256(map_features_flat),
            "active_landmark_features_sha256": tensor_sha256(active_features_flat),
            "override_landmark_features": override_landmark_features,
            "landmark_feature_override_path": full_feature_override_path,
            "landmark_feature_override_file_sha256": file_sha256(full_feature_override_path),
            "pair_scorer_state_path": full_pair_scorer_state_path,
            "pair_scorer_state_file_sha256": file_sha256(full_pair_scorer_state_path),
            "pair_measurement_state_path": full_pair_measurement_state_path,
            "pair_measurement_state_file_sha256": file_sha256(
                full_pair_measurement_state_path
            ),
            "detector_path": detector_path,
            "detector_file_sha256": file_sha256(detector_path),
            "active_vs_map_feature_delta": landmark_feature_delta(
                map_features_flat, active_features_flat
            ),
        }
        if self.landmark_feature_override_state is not None:
            override_state_features = self.landmark_feature_override_state.get(
                "landmark_features"
            )
            if override_state_features is not None:
                self.artifact_provenance["override_landmark_features_sha256"] = tensor_sha256(
                    override_state_features
                )
        if scorer_training_features is not None:
            self.artifact_provenance.update(
                {
                    "scorer_training_landmark_features_sha256": tensor_sha256(
                        scorer_training_features
                    ),
                    "active_vs_scorer_training_feature_delta": landmark_feature_delta(
                        scorer_training_features, active_features_flat.cpu()
                    ),
                }
            )
        if measurement_training_features is not None:
            self.artifact_provenance.update(
                {
                    "measurement_training_landmark_features_sha256": tensor_sha256(
                        measurement_training_features
                    ),
                    "active_vs_measurement_training_feature_delta": landmark_feature_delta(
                        measurement_training_features,
                        active_features_flat.cpu(),
                    ),
                }
            )

        self.feature_extractor = FeatureExtractor(config["feature_type"]).cuda().eval()
        self.longest_edge = config["longest_edge"]

        self.detector = KpDetector(
            self.feature_extractor.feature_dim,
            matchability_head=config["sparse"].get("use_detector_matchability", False),
            offset_head=config["sparse"].get("use_detector_offset", False),
            max_offset=config["sparse"].get("detector_max_offset", 2.0),
        )
        self.detector.load_state_dict(
            torch.load(
                resolve_artifact_path(
                    config["model_path"],
                    sparse_config["detector_path"],
                    sparse_config.get("detector_model_path"),
                )
            )
        )
        self.detector.eval().cuda()

    @torch.no_grad()
    def localize(self, query_image, fovx, fovy, sparse_valid_mask=None, sparse_support_score=None):
        """
        image: torch.Tensor, shape (3, H, W)
        """
        # Get feature
        query_fine_feature_map, query_coarse_feature_map = self.get_feature_map(
            query_image
        )

        # Sparse stage
        sparse_result = self.loc_sparse(
            query_fine_feature_map,
            fovx,
            fovy,
            valid_mask=sparse_valid_mask,
            support_score=sparse_support_score,
        )
        if self.config.get("sparse_only", self.config["sparse"].get("sparse_only", False)):
            return {"sparse": sparse_result, "dense": []}

        # Dense stage
        pose_w2c = sparse_result["pose_w2c"]
        dense_results = []
        for iter in range(self.config["dense"]["iters"]):
            dense_result = self.loc_dense(
                query_coarse_feature_map,
                query_fine_feature_map,
                pose_w2c,
                fovx,
                fovy,
                valid_mask=sparse_valid_mask,
                support_score=sparse_support_score,
            )
            pose_w2c = dense_result["pose_w2c"]
            
            dense_results.append(dense_result)

        return {"sparse": sparse_result, "dense": dense_results}

    @torch.no_grad()
    def loc_sparse(self, query_feature_map, fovx, fovy, valid_mask=None, support_score=None):
        """
        feature_map: torch.Tensor, shape (C, H, W)
        """
        # detect
        H, W = query_feature_map.shape[-2:]
        diag_cfg = self.config["sparse"].get("diagnostics", {})
        dump_discrete_oracle = bool(diag_cfg.get("dump_discrete_oracle", False))

        nms_radius = self.config["sparse"].get("nms", 4)
        offset_heatmap = None
        if self.config["sparse"].get("use_detector_matchability", False):
            matchability_mode = self.config["sparse"].get(
                "detector_matchability_mode", "combined_nms"
            )
            if matchability_mode == "proposal_rerank":
                (
                    keypoint_heatmap,
                    matchability_heatmap,
                    offset_heatmap,
                ) = self.detector.forward_all(query_feature_map)
                kp_scores_after_nms = rank_keypoint_proposals(
                    keypoint_heatmap,
                    matchability_heatmap,
                    nms_radius,
                )
            elif matchability_mode == "combined_nms":
                if self.config["sparse"].get("use_detector_offset", False):
                    _, _, offset_heatmap = self.detector.forward_all(query_feature_map)
                kp_scores_after_nms = simple_nms(
                    self.detector.forward_combined(query_feature_map),
                    nms_radius,
                )
            else:
                raise ValueError(
                    "detector_matchability_mode must be 'combined_nms' or "
                    f"'proposal_rerank', got {matchability_mode!r}"
                )
        else:
            if self.config["sparse"].get("use_detector_offset", False):
                keypoint_heatmap, _, offset_heatmap = self.detector.forward_all(
                    query_feature_map
                )
            else:
                keypoint_heatmap = self.detector(query_feature_map)
            kp_scores_after_nms = simple_nms(
                keypoint_heatmap,
                nms_radius,
            )
        support_diagnostics = {}
        if support_score is not None:
            support_grid = resize_sparse_score_to_feature_grid(support_score, H, W)
            if support_grid.device != kp_scores_after_nms.device:
                support_grid = support_grid.to(kp_scores_after_nms.device)
            kp_scores_after_nms, support_diagnostics = apply_sparse_support_score_prior(
                kp_scores_after_nms,
                support_grid,
                weight=self.config["sparse"].get("support_score_weight", 0.0),
                min_multiplier=self.config["sparse"].get("support_score_min_multiplier", 1.0),
            )
        kp_scores_after_nms = kp_scores_after_nms.flatten()
        detect_num = int(self.config["sparse"].get("detect_num", 2048))
        valid_mask_refill = bool(self.config["sparse"].get("valid_mask_refill", True))
        candidate_num = detect_num
        if valid_mask is not None and valid_mask_refill:
            candidate_multiplier = float(self.config["sparse"].get("valid_mask_candidate_multiplier", 2.0))
            candidate_num = max(detect_num, int(np.ceil(detect_num * candidate_multiplier)))
        candidate_num = min(candidate_num, int(kp_scores_after_nms.numel()))
        _, kp_ids = torch.topk(
            kp_scores_after_nms,
            candidate_num,
        )
        pos_mask = kp_scores_after_nms > 0
        kp_ids = kp_ids[pos_mask[kp_ids]]
        kp_ids, mask_diagnostics = select_sparse_keypoints_by_valid_mask(
            kp_ids,
            valid_mask,
            H,
            W,
            target_count=detect_num,
            min_fraction=self.config["sparse"].get("valid_mask_min_fraction", 0.5),
            refill_invalid=valid_mask_refill,
        )
        if kp_ids.numel() == 0:
            result = {
                "pose_w2c": np.eye(4, dtype=np.float32),
                "inliers": 0,
                "matches": 0,
            }
            result.update(support_diagnostics)
            result.update(mask_diagnostics)
            return result

        kp_mask = torch.zeros_like(kp_scores_after_nms, dtype=torch.bool)
        kp_mask[kp_ids] = True

        # sparse query features
        sampled_features = query_feature_map.reshape(query_feature_map.shape[0], -1)[
            :, kp_mask
        ]

        # sparse landmark features
        landmark_features = F.normalize(
            self.landmarks.get_loc_feature.squeeze(), dim=-1
        )

        retrieval_diagnostics = {}
        retrieval_matches = None
        if self.full_primitive_retrieval:
            if self.config["sparse"]["dual_softmax"]:
                raise ValueError("full primitive retrieval does not support dual-softmax")
            retrieval_topk = int(
                self.config["sparse"].get(
                    "full_primitive_retrieval_topk",
                    max(self.config["sparse"].get("topk", 1), 1),
                )
            )
            result = chunked_exact_topk(
                sampled_features.T,
                landmark_features,
                topk=retrieval_topk,
                chunk_size=self.config["sparse"].get(
                    "full_primitive_chunk_size", 8192
                ),
            )
            retrieval_scores = result.scores
            retrieval_indices = result.indices
            suppression_ratio = 1.0
            if self.config["sparse"].get(
                "full_primitive_surface_suppression", False
            ):
                retrieval_scores, retrieval_indices, suppression_ratio = (
                    suppress_redundant_hypotheses(
                        retrieval_scores,
                        retrieval_indices,
                        self.landmarks.get_xyz,
                        output_topk=self.config["sparse"].get("topk", 1),
                        voxel_size=self.config["sparse"].get(
                            "full_primitive_voxel_size", 0.05
                        ),
                        source_indices=getattr(
                            self.landmarks, "loc_source_index", None
                        ),
                        max_per_group=self.config["sparse"].get(
                            "full_primitive_max_per_surface", 1
                        ),
                    )
                )
            else:
                keep = min(
                    int(self.config["sparse"].get("topk", 1)),
                    retrieval_scores.shape[1],
                )
                retrieval_scores = retrieval_scores[:, :keep]
                retrieval_indices = retrieval_indices[:, :keep]
            keypoint_idx = torch.arange(
                retrieval_scores.shape[0], device=retrieval_scores.device
            )[:, None].expand_as(retrieval_indices)
            finite = torch.isfinite(retrieval_scores)
            retrieval_matches = SparseMatchResult(
                keypoint_idx[finite],
                retrieval_indices[finite],
                retrieval_scores[finite],
            )
            similarity = None
            corr_matrix = None
            retrieval_diagnostics = {
                "sparse_diag_full_primitive_retrieval": 1.0,
                "sparse_diag_full_primitive_count": int(landmark_features.shape[0]),
                "sparse_diag_full_primitive_retrieval_ms": float(result.elapsed_ms),
                "sparse_diag_full_primitive_chunks": int(result.chunks),
                "sparse_diag_full_primitive_suppression_keep_ratio": float(suppression_ratio),
            }
        else:
            similarity, corr_matrix = build_score_matrix(
                sampled_features.T,
                landmark_features,
                normalize=True,
                use_dual_softmax=self.config["sparse"]["dual_softmax"],
                dual_softmax_temperature=self.config["sparse"]["dual_softmax_temp"],
            )

        if corr_matrix is not None and self.landmark_meta is not None and self.config["sparse"].get("use_landmark_prior", False):
            prior = landmark_prior_from_meta(
                self.landmark_meta,
                landmark_features.shape[0],
                sampled_indices=self.landmark_indices,
            )
            if prior is not None:
                prior = prior.to(corr_matrix.device, dtype=corr_matrix.dtype)
                prior = (prior - prior.mean()) / prior.std().clamp_min(1e-6)
                corr_matrix = corr_matrix + self.config["sparse"].get("landmark_prior_weight", 0.05) * prior[None]

        oracle_topk_landmark_idx = None
        oracle_topk_scores = None
        if dump_discrete_oracle and corr_matrix is not None:
            oracle_topk = min(
                max(int(diag_cfg.get("oracle_topk", 32)), 1),
                int(corr_matrix.shape[1]),
            )
            oracle_topk_scores, oracle_topk_landmark_idx = torch.topk(
                corr_matrix,
                oracle_topk,
                dim=1,
            )

        match_threshold = float(self.config["sparse"]["threshold"])
        candidate_dustbin_threshold = None
        if self.config["sparse"].get("use_candidate_dustbin", False):
            if not isinstance(self.candidate_teacher_state, dict) or "dustbin_score" not in self.candidate_teacher_state:
                raise ValueError(
                    "use_candidate_dustbin requires a candidate teacher state with dustbin_score"
                )
            state_config = self.candidate_teacher_state.get("config", {})
            trained_dual_softmax = bool(state_config.get("dual_softmax", False))
            eval_dual_softmax = bool(self.config["sparse"]["dual_softmax"])
            if trained_dual_softmax != eval_dual_softmax:
                raise ValueError(
                    "candidate dustbin score space does not match eval matcher: "
                    f"trained_dual_softmax={trained_dual_softmax} "
                    f"eval_dual_softmax={eval_dual_softmax}"
                )
            candidate_dustbin_threshold = float(self.candidate_teacher_state["dustbin_score"])
            if not np.isfinite(candidate_dustbin_threshold):
                raise ValueError("candidate dustbin score must be finite")
            match_threshold = max(match_threshold, candidate_dustbin_threshold)

        matches = retrieval_matches or match_score_matrix(
            corr_matrix,
            mode="mnn" if self.config["sparse"]["mnn_match"] else "topk",
            topk=self.config["sparse"]["topk"],
            threshold=-float("inf"),
        )
        matcher_raw_matches = SparseMatchResult(
            matches.keypoint_idx.clone(),
            matches.landmark_idx.clone(),
            matches.scores.clone(),
        )
        raw_match_count = int(matches.keypoint_idx.numel())
        max_matches_per_landmark = int(
            self.config["sparse"].get("max_matches_per_landmark", 0) or 0
        )
        max_matches_per_keypoint = int(
            self.config["sparse"].get("max_matches_per_keypoint", 0) or 0
        )
        if self.config["sparse"].get("unique_landmark_matches", False):
            max_matches_per_landmark = 1
        min_candidate_matches = int(
            self.config["sparse"].get("min_candidate_matches", 0) or 0
        )
        candidate_refill_trigger_count = int(
            self.config["sparse"].get("candidate_refill_trigger_count", 0) or 0
        )
        pair_measurement_fixed_candidate_count = int(
            self.config["sparse"].get(
                "pair_measurement_fixed_candidate_count", 0
            )
            or 0
        )
        pair_measurement_preserve_candidates = bool(
            self.config["sparse"].get(
                "pair_measurement_preserve_candidates", False
            )
        )
        if pair_measurement_preserve_candidates:
            if self.pair_measurement_head is None:
                raise ValueError(
                    "pair_measurement_preserve_candidates requires "
                    "use_pair_measurement=true"
                )
            if pair_measurement_fixed_candidate_count > 0:
                raise ValueError(
                    "pair_measurement_preserve_candidates is incompatible with "
                    "pair_measurement_fixed_candidate_count"
                )
            if self.config["sparse"].get(
                "use_pair_measurement_progressive_sampling", False
            ):
                raise ValueError(
                    "pair_measurement_preserve_candidates is incompatible with "
                    "progressive sampling"
                )
            if self.config["sparse"].get(
                "use_pair_measurement_covariance_refinement", False
            ):
                raise ValueError(
                    "pair_measurement_preserve_candidates is incompatible with "
                    "covariance refinement"
                )
        selected_measurement_offsets = None
        selected_measurement_covariance = None
        pair_measurement_match_count_before = raw_match_count
        if self.pair_measurement_head is not None:
            if pair_measurement_preserve_candidates:
                matches = select_offset_only_candidates(
                    matches,
                    threshold=match_threshold,
                    max_matches_per_keypoint=max_matches_per_keypoint,
                    max_matches_per_landmark=max_matches_per_landmark,
                    min_match_count=min_candidate_matches,
                    refill_trigger_count=candidate_refill_trigger_count,
                )
            else:
                matches = select_match_candidates(matches, threshold=match_threshold)
            pair_measurement_match_count_before = int(matches.keypoint_idx.numel())
            pair_features = build_pair_context_features(
                similarity,
                kp_scores_after_nms[kp_mask],
                matches,
                context_topk=self.config["sparse"].get("pair_context_topk", 8),
                entropy_temperature=self.config["sparse"].get(
                    "pair_context_entropy_temperature", 0.1
                ),
            )
            sampled_query_features = sampled_features.T
            sampled_flat_ids = torch.nonzero(kp_mask, as_tuple=False).reshape(-1)
            sampled_keypoint_xy = torch.stack(
                [sampled_flat_ids % W, sampled_flat_ids // W], dim=1
            ).to(device=query_feature_map.device, dtype=query_feature_map.dtype)
            local_patch = sample_local_correlation_patch(
                query_feature_map,
                sampled_keypoint_xy[matches.keypoint_idx],
                landmark_features[matches.landmark_idx],
                radius=self.pair_measurement_head.patch_radius,
            )
            measurement_output = self.pair_measurement_head(
                pair_features,
                local_patch,
                sampled_query_features[matches.keypoint_idx],
                landmark_features[matches.landmark_idx],
                geometry_features=(
                    build_pair_geometry_features(
                        sampled_keypoint_xy[matches.keypoint_idx] + 0.5,
                        self.landmarks.get_xyz[matches.landmark_idx],
                        self.landmarks.get_xyz,
                        (H, W),
                    )
                    if self.pair_measurement_head.use_geometry_context
                    else None
                ),
            )
            if pair_measurement_preserve_candidates:
                selected_measurement_offsets = measurement_output.offset
                selected_measurement_covariance = measurement_output.covariance
            else:
                measurement_source_matches = SparseMatchResult(
                    matches.keypoint_idx,
                    matches.landmark_idx,
                    measurement_output.inlier_logits,
                )
                pair_measurement_refill_mode = self.config["sparse"].get(
                    "pair_measurement_refill_mode", "score"
                )
                selection_threshold = (
                    -float("inf")
                    if pair_measurement_fixed_candidate_count > 0
                    else self.pair_measurement_threshold
                )
                if pair_measurement_refill_mode == "geometry":
                    matches = select_match_candidates_with_geometry_refill(
                        measurement_source_matches,
                        sampled_keypoint_xy,
                        self.landmarks.get_xyz,
                        (H, W),
                        threshold=selection_threshold,
                        max_matches_per_keypoint=max_matches_per_keypoint,
                        max_matches_per_landmark=max_matches_per_landmark,
                        min_match_count=min_candidate_matches,
                        refill_trigger_count=candidate_refill_trigger_count,
                        max_match_count=pair_measurement_fixed_candidate_count,
                        grid_rows=self.config["sparse"].get(
                            "pair_measurement_refill_grid_rows", 4
                        ),
                        grid_cols=self.config["sparse"].get(
                            "pair_measurement_refill_grid_cols", 4
                        ),
                        voxel_size=self.config["sparse"].get(
                            "pair_measurement_refill_voxel_size", 0.25
                        ),
                        spatial_weight=self.config["sparse"].get(
                            "pair_measurement_refill_spatial_weight", 0.25
                        ),
                        voxel_weight=self.config["sparse"].get(
                            "pair_measurement_refill_voxel_weight", 0.25
                        ),
                    )
                elif pair_measurement_refill_mode == "score":
                    matches = select_match_candidates(
                        measurement_source_matches,
                        threshold=selection_threshold,
                        max_matches_per_keypoint=max_matches_per_keypoint,
                        max_matches_per_landmark=max_matches_per_landmark,
                        min_match_count=min_candidate_matches,
                        refill_trigger_count=candidate_refill_trigger_count,
                        max_match_count=pair_measurement_fixed_candidate_count,
                    )
                else:
                    raise ValueError(
                        "pair_measurement_refill_mode must be 'score' or "
                        f"'geometry', got {pair_measurement_refill_mode!r}"
                    )
                selected_measurement_offsets = gather_aligned_pair_values(
                    measurement_source_matches,
                    matches,
                    measurement_output.offset,
                    landmark_features.shape[0],
                )
                selected_measurement_covariance = gather_aligned_pair_values(
                    measurement_source_matches,
                    matches,
                    measurement_output.covariance,
                    landmark_features.shape[0],
                )
            pair_scorer_match_count_before = raw_match_count
        elif self.pair_scorer is not None:
            matches = select_match_candidates(matches, threshold=match_threshold)
            pair_scorer_match_count_before = int(matches.keypoint_idx.numel())
            pair_features = build_pair_context_features(
                similarity,
                kp_scores_after_nms[kp_mask],
                matches,
                context_topk=self.config["sparse"].get("pair_context_topk", 8),
                entropy_temperature=self.config["sparse"].get(
                    "pair_context_entropy_temperature",
                    0.1,
                ),
            )
            sampled_query_features = sampled_features.T
            global_query_descriptor = F.normalize(
                sampled_query_features.mean(dim=0),
                dim=0,
            )
            pair_logits = self.pair_scorer(
                pair_features,
                sampled_query_features[matches.keypoint_idx],
                landmark_features[matches.landmark_idx],
                global_query_descriptor,
            )
            matches = select_match_candidates(
                SparseMatchResult(
                    matches.keypoint_idx,
                    matches.landmark_idx,
                    pair_logits,
                ),
                threshold=self.pair_scorer_threshold,
                max_matches_per_keypoint=max_matches_per_keypoint,
                max_matches_per_landmark=max_matches_per_landmark,
                min_match_count=min_candidate_matches,
                refill_trigger_count=candidate_refill_trigger_count,
            )
        else:
            pair_scorer_match_count_before = raw_match_count
            matches = select_match_candidates(
                matches,
                threshold=match_threshold,
                max_matches_per_keypoint=max_matches_per_keypoint,
                max_matches_per_landmark=max_matches_per_landmark,
                min_match_count=min_candidate_matches,
                refill_trigger_count=candidate_refill_trigger_count,
            )
        pair_scorer_match_count_after = int(matches.keypoint_idx.numel())
        pair_measurement_match_count_after = int(matches.keypoint_idx.numel())
        match_count_before_landmark_dedup = raw_match_count
        match_count_after_landmark_dedup = int(matches.landmark_idx.numel())
        im_idx = matches.keypoint_idx
        gs_ids = matches.landmark_idx
        val = matches.scores
        if im_idx.numel() == 0:
            result = {
                "pose_w2c": np.eye(4, dtype=np.float32),
                "inliers": 0,
                "matches": 0,
            }
            result.update(support_diagnostics)
            result.update(mask_diagnostics)
            return result

        p2d_grid = torch.stack(
            [torch.arange(H * W) % W, torch.arange(H * W) // W], dim=1
        ).float()
        sampled_p2d_grid = p2d_grid[kp_mask.cpu()]
        sampled_offsets = torch.zeros_like(sampled_p2d_grid)
        if offset_heatmap is not None:
            sampled_offsets = (
                offset_heatmap.reshape(2, -1)[:, kp_mask].T.detach().cpu().float()
            )
            sampled_p2d_grid = sampled_p2d_grid + sampled_offsets
        p2d_matcher_raw = sampled_p2d_grid[
            matcher_raw_matches.keypoint_idx.cpu()
        ].float()
        p3d_matcher_raw = self.landmarks.get_xyz[
            matcher_raw_matches.landmark_idx
        ].detach().cpu().float()
        scores_matcher_raw = matcher_raw_matches.scores.detach().cpu().float()

        p2d = sampled_p2d_grid[im_idx.cpu()].float()
        measurement_offset_norm_mean = 0.0
        if selected_measurement_offsets is not None:
            selected_measurement_offsets = (
                selected_measurement_offsets.detach().cpu().float()
            )
            if selected_measurement_offsets.numel() > 0:
                measurement_offset_norm_mean = float(
                    torch.linalg.norm(selected_measurement_offsets, dim=1)
                    .mean()
                    .item()
                )
            if self.config["sparse"].get(
                "use_pair_measurement_offset", True
            ):
                p2d = p2d + selected_measurement_offsets
        if selected_measurement_covariance is not None:
            selected_measurement_covariance = (
                selected_measurement_covariance.detach().cpu().float()
            )
        p3d = self.landmarks.get_xyz[gs_ids].detach().cpu().float()
        pre_selector_keypoint_idx = im_idx.detach().cpu().long().clone()
        pre_selector_landmark_idx = gs_ids.detach().cpu().long().clone()
        p2d_pre_selector = p2d.clone()
        p3d_pre_selector = p3d.clone()
        scores_pre_selector = val.detach().cpu().float().clone()
        measurement_covariance_pre_selector = (
            selected_measurement_covariance.clone()
            if selected_measurement_covariance is not None
            else None
        )
        selector_indices = torch.arange(p2d.shape[0], dtype=torch.long)
        match_count_before_selector = int(p2d.shape[0])
        selector = _geometry_selector_from_config(self.config["sparse"], W, H)
        if selector is not None:
            selected = selector.select(p2d, p3d, val.detach().cpu().float())
            selector_indices = selected.detach().cpu().long()
            p2d = p2d[selected]
            p3d = p3d[selected]
            val = val.detach().cpu().float()[selected]
            if selected_measurement_covariance is not None:
                selected_measurement_covariance = selected_measurement_covariance[
                    selected
                ]
        post_selector_keypoint_idx = pre_selector_keypoint_idx[selector_indices]
        post_selector_landmark_idx = pre_selector_landmark_idx[selector_indices]
        match_count = int(p2d.shape[0])

        p2d = p2d.numpy()
        p3d = p3d.numpy()
        scores = val.detach().cpu().float().numpy() if torch.is_tensor(val) else np.asarray(val, dtype=np.float32)
        measurement_covariance = (
            selected_measurement_covariance.numpy()
            if selected_measurement_covariance is not None
            else None
        )

        K = get_intrinsic(fovx, fovy, W, H)

        pose_w2c, inliers = solve_pose(
            p2d + 0.5,
            p3d,
            K,
            self.config["sparse"]["solver"],
            self.config["sparse"]["reprojection_error"],
            self.config["sparse"]["confidence"],
            self.config["sparse"]["max_iterations"],
            self.config["sparse"]["min_iterations"],
            scores=scores,
            progressive_sampling=(
                self.pair_measurement_head is not None
                and self.config["sparse"].get(
                    "use_pair_measurement_progressive_sampling", False
                )
            ),
            max_prosac_iterations=self.config["sparse"].get(
                "pair_measurement_max_prosac_iterations", 100000
            ),
            ransac_seed=self.config["sparse"].get("ransac_seed", 0),
        )
        covariance_refinement_inliers = 0
        if (
            measurement_covariance is not None
            and self.config["sparse"].get(
                "use_pair_measurement_covariance_refinement", False
            )
            and inliers.shape[0] >= 4
        ):
            pose_w2c, covariance_inliers = covariance_weighted_pose_refinement(
                p2d + 0.5,
                p3d,
                K,
                pose_w2c,
                measurement_covariance,
                inliers,
                iterations=self.config["sparse"].get(
                    "pair_measurement_refinement_iterations", 10
                ),
                mahalanobis_threshold=self.config["sparse"].get(
                    "pair_measurement_mahalanobis_threshold", 3.0
                ),
                robust_delta=self.config["sparse"].get(
                    "pair_measurement_robust_delta", 2.5
                ),
                model_mismatch_floor_px=self.config["sparse"].get(
                    "pair_measurement_covariance_model_floor_px", 1.0
                ),
            )
            inliers = np.asarray(covariance_inliers, dtype=np.int64)
            covariance_refinement_inliers = int(inliers.shape[0])

        if selector is not None and inliers.shape[0] >= 4:
            selected_inliers = selector.select_pose_informative_inliers(
                torch.from_numpy(p3d),
                torch.from_numpy(pose_w2c),
                torch.from_numpy(K),
                torch.from_numpy(inliers.reshape(-1)),
                scores=val,
            )
            selected_inliers_np = selected_inliers.cpu().numpy()
            if selected_inliers_np.shape[0] >= 4 and selected_inliers_np.shape[0] < inliers.shape[0]:
                refined_pose_w2c, refined_inliers = solve_pose(
                    p2d[selected_inliers_np] + 0.5,
                    p3d[selected_inliers_np],
                    K,
                    self.config["sparse"]["solver"],
                    self.config["sparse"]["reprojection_error"],
                    self.config["sparse"]["confidence"],
                    self.config["sparse"]["max_iterations"],
                    self.config["sparse"]["min_iterations"],
                    ransac_seed=self.config["sparse"].get("ransac_seed", 0),
                )
                if refined_inliers.shape[0] > 0:
                    pose_w2c = refined_pose_w2c
                    inliers = selected_inliers_np[refined_inliers.reshape(-1)]

        result = {
            "pose_w2c": pose_w2c,
            "inliers": inliers.shape[0],
            "matches": match_count,
            "matches_before_selector": match_count_before_selector,
            "sparse_diag_matches_before_landmark_limit": match_count_before_landmark_dedup,
            "sparse_diag_matches_after_landmark_limit": match_count_after_landmark_dedup,
            "sparse_diag_max_matches_per_keypoint": max_matches_per_keypoint,
            "sparse_diag_landmark_limit_removed_ratio": (
                1.0 - match_count_after_landmark_dedup / max(match_count_before_landmark_dedup, 1)
            ),
            "sparse_diag_candidate_match_threshold": match_threshold,
            "sparse_diag_candidate_dustbin_enabled": float(
                candidate_dustbin_threshold is not None
            ),
            "sparse_diag_pair_scorer_enabled": float(self.pair_scorer is not None),
            "sparse_diag_pair_scorer_matches_before": pair_scorer_match_count_before,
            "sparse_diag_pair_scorer_matches_after": pair_scorer_match_count_after,
            "sparse_diag_pair_measurement_enabled": float(
                self.pair_measurement_head is not None
            ),
            "sparse_diag_pair_measurement_matches_before": (
                pair_measurement_match_count_before
            ),
            "sparse_diag_pair_measurement_matches_after": (
                pair_measurement_match_count_after
            ),
            "sparse_diag_pair_measurement_fixed_candidate_count": (
                pair_measurement_fixed_candidate_count
            ),
            "sparse_diag_pair_measurement_geometry_refill_enabled": float(
                self.config["sparse"].get(
                    "pair_measurement_refill_mode", "score"
                )
                == "geometry"
            ),
            "sparse_diag_pair_measurement_preserve_candidates": float(
                pair_measurement_preserve_candidates
            ),
            "sparse_diag_pair_measurement_offset_enabled": float(
                self.pair_measurement_head is not None
                and self.config["sparse"].get(
                    "use_pair_measurement_offset", True
                )
            ),
            "sparse_diag_pair_measurement_offset_norm_mean": (
                measurement_offset_norm_mean
            ),
            "sparse_diag_pair_measurement_covariance_refinement_enabled": float(
                measurement_covariance is not None
                and self.config["sparse"].get(
                    "use_pair_measurement_covariance_refinement", False
                )
            ),
            "sparse_diag_pair_measurement_covariance_refinement_inliers": (
                covariance_refinement_inliers
            ),
            "sparse_diag_pair_measurement_covariance_model_floor_px": float(
                self.config["sparse"].get(
                    "pair_measurement_covariance_model_floor_px", 1.0
                )
            ),
            "sparse_diag_pair_measurement_progressive_sampling_enabled": float(
                self.pair_measurement_head is not None
                and self.config["sparse"].get(
                    "use_pair_measurement_progressive_sampling", False
                )
            ),
            "sparse_diag_min_candidate_matches": min_candidate_matches,
            "sparse_diag_candidate_refill_trigger_count": candidate_refill_trigger_count,
            "sparse_diag_detector_offset_enabled": float(offset_heatmap is not None),
            "sparse_diag_detector_offset_norm_mean": float(
                torch.linalg.norm(sampled_offsets, dim=1).mean().item()
                if sampled_offsets.numel() > 0
                else 0.0
            ),
        }
        result.update(retrieval_diagnostics)
        if bool(diag_cfg.get("enabled", True)):
            result.update(
                sparse_correspondence_diagnostics(
                    p2d,
                    p3d,
                    K,
                    pose_w2c,
                    inliers.reshape(-1),
                    W,
                    H,
                    grid_rows=diag_cfg.get("grid_rows", 4),
                    grid_cols=diag_cfg.get("grid_cols", 4),
                    voxel_size=diag_cfg.get("voxel_size", 0.25),
                    translation_task_scale_m=diag_cfg.get(
                        "task_translation_scale_m", 0.02
                    ),
                    rotation_task_scale_degrees=diag_cfg.get(
                        "task_rotation_scale_degrees", 2.0
                    ),
                )
            )
        if (
            bool(diag_cfg.get("gt_metrics", True))
            or bool(diag_cfg.get("dump_correspondences", False))
            or dump_discrete_oracle
        ):
            inliers_flat = inliers.reshape(-1).copy()
            valid_post_inliers = inliers_flat[
                (inliers_flat >= 0) & (inliers_flat < selector_indices.numel())
            ]
            pre_selector_inliers = (
                selector_indices[torch.from_numpy(valid_post_inliers).long()].numpy()
                if valid_post_inliers.size > 0
                else np.empty(0, dtype=np.int64)
            )
            result["_debug_sparse_matches"] = {
                "p2d": p2d,
                "p3d": p3d,
                "scores": scores,
                "inliers": inliers_flat,
                "p2d_pre_selector": p2d_pre_selector.numpy(),
                "p3d_pre_selector": p3d_pre_selector.numpy(),
                "scores_pre_selector": scores_pre_selector.numpy(),
                "inliers_pre_selector": pre_selector_inliers,
                "p2d_matcher_raw": p2d_matcher_raw.numpy(),
                "p3d_matcher_raw": p3d_matcher_raw.numpy(),
                "scores_matcher_raw": scores_matcher_raw.numpy(),
                "measurement_covariance": measurement_covariance,
                "measurement_covariance_pre_selector": (
                    measurement_covariance_pre_selector.numpy()
                    if measurement_covariance_pre_selector is not None
                    else None
                ),
                "K": K,
                "width": int(W),
                "height": int(H),
            }
            if dump_discrete_oracle:
                sampled_flat_ids = torch.nonzero(
                    kp_mask, as_tuple=False
                ).reshape(-1)
                result["_debug_sparse_matches"]["discrete_oracle"] = {
                    "keypoint_xy": sampled_p2d_grid.numpy(),
                    "keypoint_flat_idx": sampled_flat_ids.detach().cpu().numpy(),
                    "keypoint_detector_score": (
                        kp_scores_after_nms[kp_mask].detach().cpu().float().numpy()
                    ),
                    "topk_landmark_idx": (
                        oracle_topk_landmark_idx.detach().cpu().numpy()
                    ),
                    "topk_scores": oracle_topk_scores.detach().cpu().float().numpy(),
                    "matcher_raw_keypoint_idx": (
                        matcher_raw_matches.keypoint_idx.detach().cpu().numpy()
                    ),
                    "matcher_raw_landmark_idx": (
                        matcher_raw_matches.landmark_idx.detach().cpu().numpy()
                    ),
                    "matcher_raw_scores": (
                        matcher_raw_matches.scores.detach().cpu().float().numpy()
                    ),
                    "hard_pre_keypoint_idx": pre_selector_keypoint_idx.numpy(),
                    "hard_pre_landmark_idx": pre_selector_landmark_idx.numpy(),
                    "hard_pre_scores": scores_pre_selector.numpy(),
                    "hard_post_keypoint_idx": post_selector_keypoint_idx.numpy(),
                    "hard_post_landmark_idx": post_selector_landmark_idx.numpy(),
                    "hard_post_scores": np.asarray(scores),
                    "hard_post_inliers": inliers_flat,
                    "selector_indices": selector_indices.numpy(),
                    "candidate_threshold": np.asarray(match_threshold),
                    "candidate_dustbin_threshold": np.asarray(
                        candidate_dustbin_threshold
                        if candidate_dustbin_threshold is not None
                        else np.nan
                    ),
                    "match_topk": np.asarray(self.config["sparse"]["topk"]),
                    "max_matches_per_keypoint": np.asarray(
                        max_matches_per_keypoint
                    ),
                    "max_matches_per_landmark": np.asarray(
                        max_matches_per_landmark
                    ),
                    "min_candidate_matches": np.asarray(min_candidate_matches),
                    "candidate_refill_trigger_count": np.asarray(
                        candidate_refill_trigger_count
                    ),
                    "geometry_selector_enabled": np.asarray(selector is not None),
                }
        result.update(support_diagnostics)
        result.update(mask_diagnostics)
        return result

    @torch.no_grad()
    def loc_dense(
        self,
        coarse_query_feature_map,
        fine_query_feature_map,
        pose_w2c,
        fovx,
        fovy,
        valid_mask=None,
        support_score=None,
    ):
        """
        coarse_feature_map: torch.Tensor, shape (C, H, W)
        fine_feature_map: torch.Tensor, shape (C, H, W)
        """
        Hf, Wf = fine_query_feature_map.shape[-2:]
        Hc, Wc = coarse_query_feature_map.shape[-2:]
        W = Hf // Hc  # window size
        C = self.feature_extractor.feature_dim
        WW = W * W
        overlap_size = 0  
        K = get_intrinsic(fovx, fovy, Wf, Hf)
        dense_guidance_diagnostics = {}

        render_pkg = render_from_pose_gsplat(
            self.gaussians,
            torch.tensor(pose_w2c, device="cuda"),
            fovx,
            fovy,
            Wf,
            Hf,
            render_mode="RGB+ED",
            norm_feat_bf_render=self.config["dense"]["norm_before_render"],
            rasterize_mode="antialiased",
        )

        depth = render_pkg["depth"].squeeze()

        fine_rendered_feature_map = render_pkg["feature_map"]
        if (fine_rendered_feature_map == 0).all():
            print("[skip] Rendered feature map is all zero")
            result = {"pose_w2c": pose_w2c, "inliers": 0}
            result.update(dense_guidance_diagnostics)
            return result
        
        coarse_rendered_feature_map = F.interpolate(
            fine_rendered_feature_map[None],
            size=(Hc, Wc),
            mode="bilinear",
            align_corners=False,
        )[0]
        coarse_rendered_feature_map = F.normalize(coarse_rendered_feature_map, dim=0)

        # coarse match
        coarse_corr_matrix = torch.matmul(
            coarse_query_feature_map.permute(1, 2, 0).reshape(1, -1, C),
            coarse_rendered_feature_map.reshape(1, C, -1),
        )  # 1, N, M

        coarse_corr_matrix = dual_softmax(
            coarse_corr_matrix, temp=self.config["dense"]["coarse_dual_softmax_temp"]
        )
        coarse_corr_matrix, dense_guidance_diagnostics = apply_dense_query_valid_mask_to_corr(
            coarse_corr_matrix,
            valid_mask,
            Hc,
            Wc,
            min_fraction=self.config["dense"].get(
                "valid_mask_min_fraction",
                self.config["sparse"].get("valid_mask_min_fraction", 0.5),
            ),
        )

        c_b_ids, c_i_ids, c_j_ids = mnn_match(
            coarse_corr_matrix, thr=self.config["dense"]["coarse_threshold"]
        )

        if c_i_ids.dim() == 0:
            print("[skip] Failed in coarse match")
            result = {"pose_w2c": pose_w2c, "inliers": 0}
            result.update(dense_guidance_diagnostics)
            return result
        elif c_i_ids.shape[0] < 3:
            print("[skip] Failed in coarse match")
            result = {"pose_w2c": pose_w2c, "inliers": 0}
            result.update(dense_guidance_diagnostics)
            return result
        
        # fine match
        query_feature_windows = (
            F.unfold(
                fine_query_feature_map, (W, W), stride=W, padding=overlap_size // 2
            )
            .reshape(1, C, WW, -1)[c_b_ids, :, :, c_i_ids]
            .permute(0, 2, 1)
        )  # B, N, C
        rendered_feature_windows = (
            F.unfold(
                fine_rendered_feature_map, (W, W), stride=W, padding=overlap_size // 2
            )
            .reshape(1, C, WW, -1)[c_b_ids, :, :, c_j_ids]
            .permute(0, 2, 1)
        )  # B, M, C

        fine_corr_matrix = torch.matmul(
            query_feature_windows, rendered_feature_windows.transpose(-2, -1)
        )  # B, N, M

        fine_corr_matrix = dual_softmax(
            fine_corr_matrix, temp=self.config["dense"]["fine_dual_softmax_temp"]
        )

        f_b_ids, f_i_ids, f_j_ids = mnn_match(
            fine_corr_matrix, thr=self.config["dense"]["fine_threshold"]
        )

        if f_i_ids.dim() == 0:
            print("[skip] Failed in fine match")
            result = {"pose_w2c": pose_w2c, "inliers": 0}
            result.update(dense_guidance_diagnostics)
            return result
        elif f_i_ids.shape[0] < 3:
            print("[skip] Failed in fine match")
            result = {"pose_w2c": pose_w2c, "inliers": 0}
            result.update(dense_guidance_diagnostics)
            return result

        query_p2d = torch.stack(
            [
                c_i_ids[f_b_ids] % Wc * W + f_i_ids % W,
                c_i_ids[f_b_ids] // Wc * W + f_i_ids // W,
            ],
            dim=1,
        ).float()
        rendered_p2d = torch.stack(
            [
                c_j_ids[f_b_ids] % Wc * W + f_j_ids % W,
                c_j_ids[f_b_ids] // Wc * W + f_j_ids // W,
            ],
            dim=1,
        ).float()

        pose_c2w = np.linalg.inv(pose_w2c)
        p3d = lift_2d_to_3d(rendered_p2d, torch.tensor(K, device="cuda"), torch.tensor(pose_c2w, device="cuda"), depth)

        # Solve pose
        query_p2d = query_p2d.cpu().numpy()
        p3d = p3d.cpu().numpy()

        pose_w2c, inliers = solve_pose(
            query_p2d + 0.5,
            p3d,
            K,
            self.config["dense"]["solver"],
            self.config["dense"]["reprojection_error"],
            self.config["dense"]["confidence"],
            self.config["dense"]["max_iterations"],
            self.config["dense"]["min_iterations"],
            ransac_seed=self.config["dense"].get("ransac_seed", 0),
        )

        result = {
            "pose_w2c": pose_w2c,
            "inliers": inliers.shape[0],
        }
        result.update(dense_guidance_diagnostics)
        return result

    def get_feature_map(self, image):
        """
        image: torch.Tensor, shape (3, H, W)
        """
        fine_resolution = get_resolution_from_longest_edge(
            image.shape[-2], image.shape[-1], self.longest_edge
        )
        coarse_resolution = (fine_resolution[0] // 8, fine_resolution[1] // 8)

        # Get feature
        feature_map = self.feature_extractor(image[None])["feature_map"]  # 1, C, H, W

        coarse_feature_map = F.interpolate(
            feature_map, size=coarse_resolution, mode="bilinear", align_corners=False
        )[0]
        coarse_feature_map = F.normalize(coarse_feature_map, p=2, dim=0)
        fine_feature_map = F.interpolate(
            feature_map, size=fine_resolution, mode="bilinear", align_corners=False
        )[0]
        fine_feature_map = F.normalize(fine_feature_map, p=2, dim=0)

        return fine_feature_map, coarse_feature_map


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--cfg", default=None, type=str)
    parser.add_argument("--prefix", default=None, type=str)
    parser.add_argument(
        "--detector_path",
        default=None,
        type=str,
        help="Override sparse.detector_path from the config.",
    )
    parser.add_argument(
        "--landmark_feature_override_path",
        default=None,
        type=str,
        help="Override sparse.landmark_feature_override_path from the config.",
    )
    parser.add_argument("--sparse_only", action="store_true")
    parser.add_argument(
        "--evaluation_camera_subset",
        choices=["test", "candidate_validation"],
        default="test",
    )
    parser.add_argument(
        "--evaluation_camera_list",
        default="",
        help=(
            "Optional JSON list of image names that overrides the standard "
            "test/candidate-validation camera subset."
        ),
    )
    parser.add_argument("--candidate_query_ratio", type=float, default=0.2)
    parser.add_argument("--candidate_validation_ratio", type=float, default=0.25)
    parser.add_argument(
        "--candidate_split_mode",
        choices=["random", "sequence_block", "temporal_block"],
        default="temporal_block",
    )
    parser.add_argument("--candidate_split_seed", type=int, default=2026)
    parser.add_argument(
        "--candidate_direct_validation_holdout",
        action="store_true",
    )
    parser.add_argument(
        "--candidate_direct_validation_holdout_policy",
        choices=["ignore", "warn", "error"],
        default="error",
    )
    args = get_combined_args(parser)
    # The dataset reader interprets ``eval=True`` as "read only the official
    # test-image list".  An explicit camera list may intentionally target
    # train images (for example, to collect sparse priors for dense-field
    # training), so it must load the normal train/test split first.
    args.eval = args.evaluation_camera_subset == "test" and not bool(
        args.evaluation_camera_list
    )

    results_root = os.environ.get("STDLOC_RESULTS_ROOT", "results")
    if hasattr(args, "prefix"):
        output_name = (
            f"{args.prefix}-{args.model_path.replace('/', '_')}-"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    else:
        output_name = (
            f"{args.model_path.replace('/', '_')}-"
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output_path = os.path.join(results_root, output_name)
    print("Output path:", output_path)
    os.makedirs(output_path, exist_ok=True)

    # Load feature gaussian scene
    dataset = model.extract(args)
    if dataset.gaussian_type == "3dgs":
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError("Gaussian type not supported")

    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        preload_cameras=False,
        # Explicit lists can contain either partition, while ``args.eval``
        # above only controls how the COLMAP reader builds those partitions.
        load_test_cameras=args.eval or bool(args.evaluation_camera_list),
    )

    # Set up config
    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    apply_sparse_artifact_overrides(
        config,
        detector_path=getattr(args, "detector_path", None),
        landmark_feature_override_path=getattr(
            args, "landmark_feature_override_path", None
        ),
    )
    if args.sparse_only:
        config.setdefault("sparse", {})["sparse_only"] = True
        
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    if scene.loaded_iter:
        config["_map_checkpoint_path"] = os.path.join(
            dataset.model_path,
            "point_cloud",
            f"iteration_{scene.loaded_iter}",
            "point_cloud.ply",
        )

    yaml.dump(config, open(os.path.join(output_path, os.path.basename(args.cfg)), "w"))

    # loc main
    stdloc = STDLoc(gaussians, config)

    if (
        args.evaluation_camera_subset == "candidate_validation"
        and not args.evaluation_camera_list
        and args.candidate_direct_validation_holdout
    ):
        override_state = stdloc.landmark_feature_override_state
        state_config = (
            override_state.get("config", {})
            if isinstance(override_state, dict)
            else None
        )
        validate_candidate_direct_holdout_compatibility(
            state_config,
            validation_ratio=args.candidate_validation_ratio,
            split_mode=args.candidate_split_mode,
            split_seed=args.candidate_split_seed,
            policy=args.candidate_direct_validation_holdout_policy,
        )

    discrete_oracle_dump_dir = None
    discrete_oracle_query_files = []
    sparse_diag_cfg = config.get("sparse", {}).get("diagnostics", {})
    if bool(sparse_diag_cfg.get("dump_discrete_oracle", False)):
        discrete_oracle_dump_dir = os.path.join(
            output_path, "discrete_oracle_dump"
        )
        os.makedirs(discrete_oracle_dump_dir, exist_ok=True)
        bank_indices = torch.as_tensor(stdloc.landmark_indices).detach().cpu().long()
        bank_loc_xyz = stdloc.landmarks.get_xyz.detach().cpu().float()
        bank_render_xyz = gaussians.get_xyz[
            bank_indices.to(device=gaussians.get_xyz.device)
        ].detach().cpu().float()
        np.savez_compressed(
            os.path.join(discrete_oracle_dump_dir, "landmark_bank.npz"),
            landmark_xyz=bank_loc_xyz.numpy(),
            render_xyz=bank_render_xyz.numpy(),
            source_gaussian_idx=bank_indices.numpy(),
        )

    if args.evaluation_camera_list:
        all_cameras = list(scene.getTrainCameras()) + list(scene.getTestCameras())
        test_cameras = load_evaluation_camera_list(
            all_cameras, args.evaluation_camera_list
        )
        print(
            "Explicit evaluation cameras: "
            f"{len(test_cameras)} list={args.evaluation_camera_list}"
        )
    elif args.evaluation_camera_subset == "candidate_validation":
        test_cameras = select_candidate_validation_cameras(
            scene.getTrainCameras(),
            query_ratio=args.candidate_query_ratio,
            validation_ratio=args.candidate_validation_ratio,
            split_mode=args.candidate_split_mode,
            split_seed=args.candidate_split_seed,
            direct_holdout=args.candidate_direct_validation_holdout,
        )
        print(
            "Candidate validation cameras: "
            f"{len(test_cameras)} mode={args.candidate_split_mode} "
            f"seed={args.candidate_split_seed}"
        )
    else:
        test_cameras = scene.getTestCameras()

    evaluation_protocol = build_evaluation_protocol(dataset, args, test_cameras)
    with open(
        os.path.join(output_path, "evaluation_protocol.json"), "w"
    ) as protocol_file:
        json.dump(evaluation_protocol, protocol_file, indent=2)
        protocol_file.write("\n")

    results = []
    sparse_aes = []
    sparse_tes = []
    sparse_inliers = []
    dense_aes = []
    dense_tes = []
    dense_inliers = []
    sparse_diag_values = {}
    corr_dump_file = None
    if bool(sparse_diag_cfg.get("dump_correspondences", False)):
        corr_dump_file = open(os.path.join(output_path, "sparse_correspondences.jsonl"), "w")

    for idx, camera_info in enumerate(tqdm(test_cameras, desc="STDLoc")):
        print("\nLocalize image:", camera_info.image_name)
        gt_w2c = camera_info.world_view_transform.transpose(0, 1).cpu().numpy()
        query_image = camera_info.original_image.to("cuda")
        fovx = camera_info.FoVx
        fovy = camera_info.FoVy

        # localization
        loc_res = stdloc.localize(query_image, fovx, fovy)
        sparse_debug = loc_res["sparse"].pop("_debug_sparse_matches", None)
        if sparse_debug is not None:
            post_selector_diagnostics = sparse_correspondence_diagnostics(
                sparse_debug["p2d"],
                sparse_debug["p3d"],
                sparse_debug["K"],
                loc_res["sparse"]["pose_w2c"],
                sparse_debug["inliers"],
                sparse_debug["width"],
                sparse_debug["height"],
                gt_pose_w2c=gt_w2c,
                grid_rows=sparse_diag_cfg.get("grid_rows", 4),
                grid_cols=sparse_diag_cfg.get("grid_cols", 4),
                voxel_size=sparse_diag_cfg.get("voxel_size", 0.25),
                translation_task_scale_m=sparse_diag_cfg.get(
                    "task_translation_scale_m", 0.02
                ),
                rotation_task_scale_degrees=sparse_diag_cfg.get(
                    "task_rotation_scale_degrees", 2.0
                ),
            )
            loc_res["sparse"].update(post_selector_diagnostics)
            for key, value in post_selector_diagnostics.items():
                if key.startswith("sparse_diag_"):
                    loc_res["sparse"][
                        "sparse_diag_post_selector_" + key[len("sparse_diag_") :]
                    ] = value

            if "p2d_pre_selector" in sparse_debug:
                pre_selector_diagnostics = sparse_correspondence_diagnostics(
                    sparse_debug["p2d_pre_selector"],
                    sparse_debug["p3d_pre_selector"],
                    sparse_debug["K"],
                    loc_res["sparse"]["pose_w2c"],
                    sparse_debug.get("inliers_pre_selector", []),
                    sparse_debug["width"],
                    sparse_debug["height"],
                    gt_pose_w2c=gt_w2c,
                    grid_rows=sparse_diag_cfg.get("grid_rows", 4),
                    grid_cols=sparse_diag_cfg.get("grid_cols", 4),
                    voxel_size=sparse_diag_cfg.get("voxel_size", 0.25),
                    translation_task_scale_m=sparse_diag_cfg.get(
                        "task_translation_scale_m", 0.02
                    ),
                    rotation_task_scale_degrees=sparse_diag_cfg.get(
                        "task_rotation_scale_degrees", 2.0
                    ),
                )
                for key, value in pre_selector_diagnostics.items():
                    if key.startswith("sparse_diag_"):
                        loc_res["sparse"][
                            "sparse_diag_pre_selector_" + key[len("sparse_diag_") :]
                        ] = value
                loc_res["sparse"]["raw_pre_selector_gt_precision_2px"] = pre_selector_diagnostics.get(
                    "sparse_diag_all_gt_precision_2px", 0.0
                )
                loc_res["sparse"]["raw_post_selector_gt_precision_2px"] = post_selector_diagnostics.get(
                    "sparse_diag_all_gt_precision_2px", 0.0
                )
            if "p2d_matcher_raw" in sparse_debug:
                matcher_raw_diagnostics = sparse_correspondence_diagnostics(
                    sparse_debug["p2d_matcher_raw"],
                    sparse_debug["p3d_matcher_raw"],
                    sparse_debug["K"],
                    loc_res["sparse"]["pose_w2c"],
                    [],
                    sparse_debug["width"],
                    sparse_debug["height"],
                    gt_pose_w2c=gt_w2c,
                    grid_rows=sparse_diag_cfg.get("grid_rows", 4),
                    grid_cols=sparse_diag_cfg.get("grid_cols", 4),
                    voxel_size=sparse_diag_cfg.get("voxel_size", 0.25),
                    translation_task_scale_m=sparse_diag_cfg.get(
                        "task_translation_scale_m", 0.02
                    ),
                    rotation_task_scale_degrees=sparse_diag_cfg.get(
                        "task_rotation_scale_degrees", 2.0
                    ),
                )
                for key, value in matcher_raw_diagnostics.items():
                    if key.startswith("sparse_diag_all_") or key == "sparse_diag_match_count":
                        loc_res["sparse"][
                            "sparse_diag_matcher_raw_" + key[len("sparse_diag_") :]
                        ] = value
                loc_res["sparse"]["matcher_raw_gt_precision_2px"] = (
                    matcher_raw_diagnostics.get("sparse_diag_all_gt_precision_2px", 0.0)
                )
            if corr_dump_file is not None:
                inliers_only = bool(sparse_diag_cfg.get("dump_inliers_only", True))
                max_dump = int(sparse_diag_cfg.get("dump_max_correspondences", 0) or 0)
                dump_pre_selector = bool(sparse_diag_cfg.get("dump_pre_selector", True))
                if dump_pre_selector and "p2d_pre_selector" in sparse_debug:
                    dump_p2d = np.asarray(sparse_debug["p2d_pre_selector"])
                    dump_p3d = np.asarray(sparse_debug["p3d_pre_selector"])
                    dump_scores = np.asarray(sparse_debug["scores_pre_selector"])
                    dump_covariance = sparse_debug.get(
                        "measurement_covariance_pre_selector"
                    )
                    dump_inliers = np.asarray(
                        sparse_debug.get("inliers_pre_selector", []), dtype=np.int64
                    ).reshape(-1)
                    candidate_stage = "pre_selector"
                else:
                    dump_p2d = np.asarray(sparse_debug["p2d"])
                    dump_p3d = np.asarray(sparse_debug["p3d"])
                    dump_scores = np.asarray(sparse_debug["scores"])
                    dump_covariance = sparse_debug.get("measurement_covariance")
                    dump_inliers = np.asarray(
                        sparse_debug["inliers"], dtype=np.int64
                    ).reshape(-1)
                    candidate_stage = "post_selector"
                if inliers_only:
                    dump_idx = dump_inliers
                else:
                    dump_idx = np.arange(dump_p2d.shape[0], dtype=np.int64)
                dump_idx = dump_idx[(dump_idx >= 0) & (dump_idx < dump_p2d.shape[0])]
                if max_dump > 0:
                    dump_idx = dump_idx[:max_dump]
                corr_dump_file.write(
                    json.dumps(
                        {
                            "image_name": camera_info.image_name,
                            "candidate_stage": candidate_stage,
                            "indices": dump_idx.tolist(),
                            "p2d": dump_p2d[dump_idx].tolist(),
                            "p3d": dump_p3d[dump_idx].tolist(),
                            "scores": dump_scores[dump_idx].tolist(),
                            "measurement_covariance": (
                                np.asarray(dump_covariance)[dump_idx].tolist()
                                if dump_covariance is not None
                                else None
                            ),
                            "inliers": dump_inliers.tolist(),
                            "K": np.asarray(sparse_debug["K"]).tolist(),
                            "width": int(sparse_debug["width"]),
                            "height": int(sparse_debug["height"]),
                            "gt_pose_w2c": np.asarray(gt_w2c).tolist(),
                        }
                    )
                    + "\n"
                )
            if discrete_oracle_dump_dir is not None:
                oracle_debug = sparse_debug.get("discrete_oracle")
                if oracle_debug is None:
                    raise RuntimeError(
                        "discrete oracle dump requested but sparse debug payload is missing"
                    )
                if gaussians._xyz.grad is not None:
                    gaussians._xyz.grad.zero_()
                with torch.enable_grad():
                    render_visible_mask = get_render_visible_mask(
                        gaussians,
                        camera_info,
                        int(sparse_debug["width"]),
                        int(sparse_debug["height"]),
                    )
                bank_indices_device = torch.as_tensor(
                    stdloc.landmark_indices,
                    device=render_visible_mask.device,
                    dtype=torch.long,
                )
                bank_visible = render_visible_mask[bank_indices_device]
                oracle_file = (
                    f"query_{idx:04d}_"
                    f"{hashlib.sha1(camera_info.image_name.encode()).hexdigest()[:10]}.npz"
                )
                oracle_payload = {
                    key: np.asarray(value) for key, value in oracle_debug.items()
                }
                oracle_payload.update(
                    {
                        "image_name": np.asarray(camera_info.image_name),
                        "gt_pose_w2c": np.asarray(gt_w2c, dtype=np.float64),
                        "pred_pose_w2c": np.asarray(
                            loc_res["sparse"]["pose_w2c"], dtype=np.float64
                        ),
                        "K": np.asarray(sparse_debug["K"], dtype=np.float64),
                        "width": np.asarray(sparse_debug["width"]),
                        "height": np.asarray(sparse_debug["height"]),
                        "render_visible_bank": bank_visible.detach()
                        .cpu()
                        .numpy()
                        .astype(np.uint8),
                        "solver": np.asarray(config["sparse"]["solver"]),
                        "reprojection_error": np.asarray(
                            config["sparse"]["reprojection_error"]
                        ),
                        "confidence": np.asarray(config["sparse"]["confidence"]),
                        "max_iterations": np.asarray(
                            config["sparse"]["max_iterations"]
                        ),
                        "min_iterations": np.asarray(
                            config["sparse"]["min_iterations"]
                        ),
                        "ransac_seed": np.asarray(
                            config["sparse"].get("ransac_seed", 0)
                        ),
                    }
                )
                np.savez_compressed(
                    os.path.join(discrete_oracle_dump_dir, oracle_file),
                    **oracle_payload,
                )
                discrete_oracle_query_files.append(oracle_file)

        # evaluation
        sparse_ae, sparse_te = cal_pose_error(loc_res["sparse"]["pose_w2c"], gt_w2c)
        sparse_aes.append(sparse_ae)
        sparse_tes.append(sparse_te)
        sparse_inliers.append(loc_res["sparse"]["inliers"])
        loc_res["sparse_AE"] = sparse_ae
        loc_res["sparse_TE"] = sparse_te

        dense_final = loc_res["dense"][-1] if len(loc_res["dense"]) > 0 else loc_res["sparse"]
        dense_ae, dense_te = cal_pose_error(dense_final["pose_w2c"], gt_w2c) # degree, cm
        dense_aes.append(dense_ae)
        dense_tes.append(dense_te)
        dense_inliers.append(dense_final["inliers"])
        print(f"AE: {dense_ae:.3f}deg, TE: {dense_te:.3f}cm, inliers: {dense_final['inliers']}")

        loc_res["gt_pose_w2c"] = gt_w2c.tolist()
        loc_res["dense_AE"] = dense_ae
        loc_res["dense_TE"] = dense_te
        loc_res["image_name"] = camera_info.image_name
        for key, value in loc_res["sparse"].items():
            if key.startswith("sparse_diag_") and isinstance(value, (int, float, np.integer, np.floating)):
                if np.isfinite(float(value)):
                    sparse_diag_values.setdefault(key, []).append(float(value))

        results.append(loc_res)

    if corr_dump_file is not None:
        corr_dump_file.close()
    if discrete_oracle_dump_dir is not None:
        with open(
            os.path.join(discrete_oracle_dump_dir, "manifest.json"), "w"
        ) as f:
            json.dump(
                {
                    "schema_version": 1,
                    "landmark_bank": "landmark_bank.npz",
                    "query_files": discrete_oracle_query_files,
                    "query_count": len(discrete_oracle_query_files),
                    "oracle_topk": int(sparse_diag_cfg.get("oracle_topk", 32)),
                    "task_translation_scale_m": float(
                        sparse_diag_cfg.get("task_translation_scale_m", 0.02)
                    ),
                    "task_rotation_scale_degrees": float(
                        sparse_diag_cfg.get("task_rotation_scale_degrees", 2.0)
                    ),
                    "evaluation_camera_subset": args.evaluation_camera_subset,
                    "model_path": dataset.model_path,
                },
                f,
                indent=2,
            )
            f.write("\n")

    # get summary
    sparse_aes = np.array(sparse_aes)
    sparse_tes = np.array(sparse_tes)
    dense_aes = np.array(dense_aes)
    dense_tes = np.array(dense_tes)

    results_summary = {
        "model_path": dataset.model_path,
        "evaluation_camera_subset": args.evaluation_camera_subset,
        "evaluation_camera_count": len(test_cameras),
        "evaluation_protocol": evaluation_protocol,
        "artifact_provenance": stdloc.artifact_provenance,
        "sparse": {
            "median_ae": np.median(sparse_aes),
            "median_te": np.median(sparse_tes),
            "recall_5m_10d": ((sparse_aes <= 10) & (sparse_tes <= 500)).sum()
            / len(sparse_aes),
            "recall_2m_5d": ((sparse_aes <= 5) & (sparse_tes <= 200)).sum()
            / len(sparse_aes),
            "recall_5cm_5d": ((sparse_aes <= 5) & (sparse_tes <= 5)).sum()
            / len(sparse_aes),
            "recall_2cm_2d": ((sparse_aes <= 2) & (sparse_tes <= 2)).sum()
            / len(sparse_aes),
            "avg_inliers": np.array(sparse_inliers).mean(),
        },
        "dense": {
            "median_ae": np.median(dense_aes),
            "median_te": np.median(dense_tes),
            "recall_5m_10d": ((dense_aes <= 10) & (dense_tes <= 500)).sum()
            / len(dense_aes),
            "recall_2m_5d": ((dense_aes <= 5) & (dense_tes <= 200)).sum()
            / len(dense_aes),
            "recall_5cm_5d": ((dense_aes <= 5) & (dense_tes <= 5)).sum()
            / len(dense_aes),
            "recall_2cm_2d": ((dense_aes <= 2) & (dense_tes <= 2)).sum()
            / len(dense_aes),
            "avg_inliers": np.array(dense_inliers).mean(),
        },
    }
    if sparse_diag_values:
        results_summary["sparse_diagnostics"] = {}
        for key, values in sorted(sparse_diag_values.items()):
            arr = np.asarray(values, dtype=np.float64)
            results_summary["sparse_diagnostics"][f"{key}_mean"] = float(np.mean(arr))
            results_summary["sparse_diagnostics"][f"{key}_median"] = float(np.median(arr))

    print("Result Summary:")
    print(json.dumps(results_summary, indent=4))

    for summary_name in ("summary.json", "results_summary.json"):
        with open(os.path.join(output_path, summary_name), "w") as summary_file:
            json.dump(results_summary, summary_file, indent=4)

    for item in results:
        item["sparse"]["pose_w2c"] = item["sparse"]["pose_w2c"].tolist()
        for dense_item in item["dense"]:
            dense_item["pose_w2c"] = dense_item["pose_w2c"].tolist()
    json.dump(results, open(os.path.join(output_path, "results.json"), "w"), indent=4)


    print("Result are saved in", output_path)
