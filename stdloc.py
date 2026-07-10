import datetime
import json
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_from_pose_gsplat
from localization_training.direct_landmark_teacher import gaussian_localization_xyz
from localization_training.geometry_selector import GeometryBalancedSelector
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from scene.kpdetector import KpDetector, simple_nms
from utils.graphics_utils import fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.pose_utils import cal_pose_error, solve_pose

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
    corr_matrix = corr_matrix / temp
    corr_matrix = F.softmax(corr_matrix, dim=-2) * F.softmax(corr_matrix, dim=-1)
    return corr_matrix


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
    safe_depth = np.maximum(depth, eps)
    uv = np.empty((p3d.shape[0], 2), dtype=np.float64)
    uv[:, 0] = K[0, 0] * cam[:, 0] / safe_depth + K[0, 2]
    uv[:, 1] = K[1, 1] * cam[:, 1] / safe_depth + K[1, 2]
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


def _pose_information_stats(prefix, p3d, K, pose_w2c, regularization=1e-6):
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
        H += jac.T @ jac
    H_reg = H + float(regularization) * np.eye(6, dtype=np.float64)
    eigvals = np.linalg.eigvalsh(H_reg)
    eigvals = eigvals[np.isfinite(eigvals)]
    if eigvals.size == 0:
        return {}
    eigvals = np.clip(eigvals, 1e-12, None)
    sign, logdet = np.linalg.slogdet(H_reg)
    return {
        f"{prefix}_pose_info_condition": float(eigvals.max() / eigvals.min()),
        f"{prefix}_pose_info_logdet": float(logdet if sign > 0 else np.nan),
        f"{prefix}_pose_info_min_eig": float(eigvals.min()),
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
    diagnostics.update(_residual_stats("sparse_diag_all_est_reproj_px", residual))
    diagnostics.update(_occupancy_stats_2d("sparse_diag_all", p2d, width, height, grid_rows, grid_cols))
    diagnostics.update(_occupancy_stats_3d("sparse_diag_all", p3d, voxel_size))
    finite_depth = depth[np.isfinite(depth)]
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
        diagnostics.update(_residual_stats("sparse_diag_inlier_est_reproj_px", residual[inliers]))
        diagnostics.update(_occupancy_stats_2d("sparse_diag_inlier", inlier_p2d, width, height, grid_rows, grid_cols))
        diagnostics.update(_occupancy_stats_3d("sparse_diag_inlier", inlier_p3d, voxel_size))
        diagnostics.update(_pose_information_stats("sparse_diag_inlier", inlier_p3d, K, pose_w2c))
        finite_inlier_depth = inlier_depth[np.isfinite(inlier_depth)]
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
        diagnostics.update(_residual_stats("sparse_diag_all_gt_reproj_px", gt_residual))
        for threshold in (2.0, 4.0, 6.0):
            diagnostics[f"sparse_diag_all_gt_precision_{int(threshold)}px"] = float(
                np.mean(gt_residual <= threshold)
            )
        if inliers.shape[0] > 0:
            diagnostics.update(_residual_stats("sparse_diag_inlier_gt_reproj_px", gt_residual[inliers]))
            for threshold in (2.0, 4.0, 6.0):
                diagnostics[f"sparse_diag_inlier_gt_precision_{int(threshold)}px"] = float(
                    np.mean(gt_residual[inliers] <= threshold)
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
    state = torch.load(path, map_location="cpu")
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

        sampled_idx = pickle.load(
            open(
                resolve_artifact_path(
                    config["model_path"],
                    config["sparse"]["landmark_path"],
                    config["sparse"].get("landmark_model_path"),
                ),
                "rb",
            )
        )
        self.landmark_indices = validate_sampled_indices(sampled_idx, gaussians.get_xyz.shape[0]).detach().cpu()
        self.landmarks = sample_gaussians(gaussians, self.landmark_indices)
        candidate_state_path = config["sparse"].get(
            "candidate_teacher_state_path",
            config["sparse"].get("landmark_feature_path", ""),
        )
        self.candidate_teacher_state = None
        if candidate_state_path:
            full_candidate_state_path = resolve_artifact_path(
                config["model_path"],
                candidate_state_path,
                config["sparse"].get(
                    "candidate_teacher_state_model_path",
                    config["sparse"].get("landmark_model_path"),
                ),
            )
            current_features = self.landmarks.get_loc_feature
            current_flat = current_features.reshape(current_features.shape[0], -1)
            override, self.candidate_teacher_state = load_candidate_teacher_landmark_features(
                full_candidate_state_path,
                self.landmark_indices,
                expected_feature_dim=current_flat.shape[1],
                device=current_features.device,
                dtype=current_features.dtype,
            )
            self.landmarks._loc_feature = override.reshape_as(current_features)
        landmark_meta_path = config["sparse"].get("landmark_meta_path", "detector/landmark_meta.pt")
        full_meta_path = resolve_artifact_path(
            config["model_path"],
            landmark_meta_path,
            config["sparse"].get("landmark_meta_model_path", config["sparse"].get("landmark_model_path")),
        )
        self.landmark_meta = torch.load(full_meta_path) if os.path.exists(full_meta_path) else None

        self.feature_extractor = FeatureExtractor(config["feature_type"]).cuda().eval()
        self.longest_edge = config["longest_edge"]

        self.detector = KpDetector(self.feature_extractor.feature_dim)
        self.detector.load_state_dict(
            torch.load(
                resolve_artifact_path(
                    config["model_path"],
                    config["sparse"]["detector_path"],
                    config["sparse"].get("detector_model_path"),
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

        heat_map = self.detector(query_feature_map)

        kp_scores_after_nms = simple_nms(
            heat_map, self.config["sparse"].get("nms", 4)
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

        # sparse match
        corr_matrix = torch.matmul(sampled_features.T, landmark_features.T)

        # dual softmax
        if self.config["sparse"]["dual_softmax"] is True:
            corr_matrix = dual_softmax(
                corr_matrix=corr_matrix, temp=self.config["sparse"]["dual_softmax_temp"]
            )

        if self.landmark_meta is not None and self.config["sparse"].get("use_landmark_prior", False):
            prior = landmark_prior_from_meta(
                self.landmark_meta,
                landmark_features.shape[0],
                sampled_indices=self.landmark_indices,
            )
            if prior is not None:
                prior = prior.to(corr_matrix.device, dtype=corr_matrix.dtype)
                prior = (prior - prior.mean()) / prior.std().clamp_min(1e-6)
                corr_matrix = corr_matrix + self.config["sparse"].get("landmark_prior_weight", 0.05) * prior[None]

        if self.config["sparse"]["mnn_match"] is True:
            # mnn match
            b_ids, im_idx, gs_ids = mnn_match(
                corr_matrix[None], thr=self.config["sparse"]["threshold"]
            )
            val = corr_matrix[im_idx, gs_ids] if im_idx.numel() > 0 else corr_matrix.new_empty(0)
        else:
            # topk match
            im_idx, gs_ids, val = topk_match(
                corr_matrix[None],
                self.config["sparse"]["topk"],
                thr=self.config["sparse"]["threshold"],
            )
        if im_idx.numel() == 0:
            result = {
                "pose_w2c": np.eye(4, dtype=np.float32),
                "inliers": 0,
                "matches": 0,
            }
            result.update(support_diagnostics)
            result.update(mask_diagnostics)
            return result

        p2d = torch.stack([torch.arange(H * W) % W, torch.arange(H * W) // W], dim=1)

        p2d = p2d[kp_mask.cpu()][im_idx.cpu()].float()
        p3d = self.landmarks.get_xyz[gs_ids].detach().cpu().float()
        p2d_pre_selector = p2d.clone()
        p3d_pre_selector = p3d.clone()
        scores_pre_selector = val.detach().cpu().float().clone()
        selector_indices = torch.arange(p2d.shape[0], dtype=torch.long)
        match_count_before_selector = int(p2d.shape[0])
        selector = _geometry_selector_from_config(self.config["sparse"], W, H)
        if selector is not None:
            selected = selector.select(p2d, p3d, val.detach().cpu().float())
            selector_indices = selected.detach().cpu().long()
            p2d = p2d[selected]
            p3d = p3d[selected]
            val = val.detach().cpu().float()[selected]
        match_count = int(p2d.shape[0])

        p2d = p2d.numpy()
        p3d = p3d.numpy()
        scores = val.detach().cpu().float().numpy() if torch.is_tensor(val) else np.asarray(val, dtype=np.float32)

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
        )

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
                )
                if refined_inliers.shape[0] > 0:
                    pose_w2c = refined_pose_w2c
                    inliers = selected_inliers_np[refined_inliers.reshape(-1)]

        result = {
            "pose_w2c": pose_w2c,
            "inliers": inliers.shape[0],
            "matches": match_count,
            "matches_before_selector": match_count_before_selector,
        }
        diag_cfg = self.config["sparse"].get("diagnostics", {})
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
                )
            )
        if bool(diag_cfg.get("gt_metrics", True)) or bool(diag_cfg.get("dump_correspondences", False)):
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
                "K": K,
                "width": int(W),
                "height": int(H),
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
    parser.add_argument("--sparse_only", action="store_true")
    args = get_combined_args(parser)
    args.eval = True

    if hasattr(args, "prefix"):
        output_path = f"results/{args.prefix}-{args.model_path.replace('/', '_')}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_path = f"results/{args.model_path.replace('/', '_')}-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    )

    # Set up config
    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    if args.sparse_only:
        config.setdefault("sparse", {})["sparse_only"] = True
        
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path

    yaml.dump(config, open(os.path.join(output_path, os.path.basename(args.cfg)), "w"))

    # loc main
    stdloc = STDLoc(gaussians, config)

    test_cameras = scene.getTestCameras()

    results = []
    sparse_aes = []
    sparse_tes = []
    sparse_inliers = []
    dense_aes = []
    dense_tes = []
    dense_inliers = []
    sparse_diag_values = {}
    sparse_diag_cfg = config.get("sparse", {}).get("diagnostics", {})
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
            if corr_dump_file is not None:
                inliers_only = bool(sparse_diag_cfg.get("dump_inliers_only", True))
                max_dump = int(sparse_diag_cfg.get("dump_max_correspondences", 0) or 0)
                if inliers_only:
                    dump_idx = np.asarray(sparse_debug["inliers"], dtype=np.int64).reshape(-1)
                else:
                    dump_idx = np.arange(np.asarray(sparse_debug["p2d"]).shape[0], dtype=np.int64)
                dump_idx = dump_idx[(dump_idx >= 0) & (dump_idx < np.asarray(sparse_debug["p2d"]).shape[0])]
                if max_dump > 0:
                    dump_idx = dump_idx[:max_dump]
                corr_dump_file.write(
                    json.dumps(
                        {
                            "image_name": camera_info.image_name,
                            "indices": dump_idx.tolist(),
                            "p2d": np.asarray(sparse_debug["p2d"])[dump_idx].tolist(),
                            "p3d": np.asarray(sparse_debug["p3d"])[dump_idx].tolist(),
                            "scores": np.asarray(sparse_debug["scores"])[dump_idx].tolist(),
                            "inliers": np.asarray(sparse_debug["inliers"], dtype=np.int64).reshape(-1).tolist(),
                        }
                    )
                    + "\n"
                )

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

    # get summary
    sparse_aes = np.array(sparse_aes)
    sparse_tes = np.array(sparse_tes)
    dense_aes = np.array(dense_aes)
    dense_tes = np.array(dense_tes)

    results_summary = {
        "model_path": dataset.model_path,
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

    json.dump(
        results_summary, open(os.path.join(output_path, "summary.json"), "w"), indent=4
    )

    for item in results:
        item["sparse"]["pose_w2c"] = item["sparse"]["pose_w2c"].tolist()
        for dense_item in item["dense"]:
            dense_item["pose_w2c"] = dense_item["pose_w2c"].tolist()
    json.dump(results, open(os.path.join(output_path, "results.json"), "w"), indent=4)


    print("Result are saved in", output_path)
