#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer import get_render_visible_mask, render_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state, seed_everything
from utils.graphics_utils import focal2fov, fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.loss_utils import *

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import pickle

import torch.nn.functional as F

from encoders.feature_extractor import FeatureExtractor
from localization_training.direct_landmark_teacher import gaussian_localization_xyz
from localization_training.episode_sampler import split_support_query_cameras
from localization_training.landmark_distill import (
    coverage_preserving_sample,
    localization_aware_sample,
    save_landmark_meta,
)
from localization_training.pair_scorer import SparsePairScorer
from localization_training.sparse_candidate_teacher import (
    build_sparse_candidate_batch,
    calibrate_binary_threshold,
    sparse_candidate_losses,
)
from localization_training.sparse_frontend import (
    SparseMatchResult,
    limit_matches_per_keypoint,
    rank_keypoint_proposals,
)
from scene.kpdetector import KpDetector


def extract_normalized_feature_map(feature_extractor, image, size):
    """Run the fixed image encoder without keeping gradients for detector targets."""
    with torch.no_grad():
        gt_feature_map = feature_extractor(image[None])["feature_map"]
        gt_feature_map = F.interpolate(
            gt_feature_map,
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
    return gt_feature_map.detach()


def store_render_visible_mask(render_visible_masks, image_name, visible_mask):
    render_visible_masks[image_name] = visible_mask.detach().to(device="cpu", dtype=torch.bool)


def render_visible_mask_from_cache(render_visible_masks, image_name, device):
    visible_mask = render_visible_masks.get(image_name, None)
    if visible_mask is None:
        return None
    return visible_mask.to(device=device, dtype=torch.bool, non_blocking=True)


def get_sampled_gaussian(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = gaussians.__class__(gaussians.max_sh_degree)
    sampled_gaussians.active_sh_degree = gaussians.active_sh_degree
    sampled_gaussians.spatial_lr_scale = gaussians.spatial_lr_scale
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._loc_feature = gaussians.materialized_loc_feature(idx_sampled)
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    if torch.is_tensor(getattr(gaussians, "_loc_opacity", None)) and gaussians._loc_opacity.shape[0] == gaussians.get_xyz.shape[0]:
        sampled_gaussians._loc_opacity = gaussians._loc_opacity[idx_sampled]
    if torch.is_tensor(getattr(gaussians, "_loc_anchor_offset", None)) and gaussians._loc_anchor_offset.shape[0] == gaussians.get_xyz.shape[0]:
        sampled_gaussians._loc_anchor_offset = gaussians._loc_anchor_offset[idx_sampled]
    sampled_gaussians.surfel_loc_tangent_bound = float(getattr(gaussians, "surfel_loc_tangent_bound", 0.0) or 0.0)
    sampled_gaussians.surfel_loc_normal_bound = float(getattr(gaussians, "surfel_loc_normal_bound", 0.0) or 0.0)
    sampled_gaussians.max_radii2D = torch.zeros(
        sampled_gaussians.get_xyz.shape[0],
        dtype=torch.float32,
        device=sampled_gaussians.get_xyz.device,
    )
    return sampled_gaussians


@torch.no_grad()
def calculate_match_score(
    gaussians: GaussianModel,
    gt_feature_map,
    pose,
    K,
    render_visible_mask=None,
    img_mask=None,
):
    xyz = gaussian_localization_xyz(gaussians)
    feat = gaussians.get_loc_feature.squeeze()

    # project gaussians to image space
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=-1)
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_homo = xyz_cam / depths

    xy = (K @ xyz_cam_homo)[:2].long()

    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_feature_map.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_feature_map.shape[1])
    )

    if render_visible_mask is not None:
        visible_mask = in_mask & render_visible_mask
    else:
        visible_mask = in_mask

    if img_mask is not None:
        visible_xy = xy[:, in_mask]
        img_mask_expand = torch.zeros_like(visible_mask, dtype=torch.bool)
        img_mask_expand[in_mask] = img_mask[0, visible_xy[1], visible_xy[0]]
        visible_mask = visible_mask & img_mask_expand

    xy = xy[:, visible_mask]
    depths = depths[visible_mask]
    feat = feat[visible_mask]

    gs_feats = F.normalize(feat, p=2, dim=1)
    im_feats = gt_feature_map[:, xy[1], xy[0]].T
    score = (gs_feats * im_feats).sum(-1)
    return score, visible_mask


def generate_gt_map(
    gaussians: GaussianModel,
    gt_feature_map,
    idx_sampled,
    pose,
    K,
    render_visible_mask=None,
):
    if render_visible_mask is not None:
        render_visible_mask = render_visible_mask[idx_sampled]
        idx_sampled = idx_sampled[render_visible_mask]
    sampled_xyz = gaussian_localization_xyz(gaussians)[idx_sampled]

    gt_map = torch.zeros(
        (1, gt_feature_map.shape[1], gt_feature_map.shape[2]),
        device=gt_feature_map.device,
    )
    
    xyz_homo = torch.cat(
        [sampled_xyz, torch.ones(sampled_xyz.shape[0], 1, device=sampled_xyz.device)],
        dim=-1,
    )
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths

    xy = (K @ xyz_cam_norm)[:2].long()

    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_feature_map.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_feature_map.shape[1])
    )

    xy_pos = xy[:, in_mask]

    gt_map[:, xy_pos[1], xy_pos[0]] = 1

    return gt_map


def _project_xyz_to_feature(xyz, pose, K, height, width):
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)], dim=-1)
    xyz_cam = (pose.to(device=xyz.device, dtype=xyz.dtype) @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths.clamp_min(1e-8)
    xy = (K.to(device=xyz.device, dtype=xyz.dtype) @ xyz_cam_norm)[:2]
    valid = (
        (depths > 1e-8)
        & (xy[0] >= 0)
        & (xy[0] <= width - 1)
        & (xy[1] >= 0)
        & (xy[1] <= height - 1)
    )
    return xy, valid


def _project_xyz_to_feature_with_depth(xyz, pose, K, height, width):
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)], dim=-1)
    xyz_cam = (pose.to(device=xyz.device, dtype=xyz.dtype) @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths.clamp_min(1e-8)
    xy = (K.to(device=xyz.device, dtype=xyz.dtype) @ xyz_cam_norm)[:2]
    valid = (
        (depths > 1e-8)
        & (xy[0] >= 0)
        & (xy[0] <= width - 1)
        & (xy[1] >= 0)
        & (xy[1] <= height - 1)
    )
    return xy, valid, depths


def _calibrated_utility_weights(utility, min_weight=1.0, max_weight=2.0):
    if utility is None:
        return None
    utility = utility.float().reshape(-1)
    if utility.numel() == 0:
        return utility
    center = utility.median()
    scale = (utility - center).abs().median().clamp_min(1e-6)
    normalized = (utility - center) / scale
    return min_weight + (max_weight - min_weight) * torch.sigmoid(normalized)


def _meta_vector(landmark_meta, key, count):
    if landmark_meta is None or key not in landmark_meta:
        return None
    value = torch.as_tensor(landmark_meta[key], dtype=torch.float32).reshape(-1)
    if value.numel() == 1:
        value = value.expand(count)
    if value.numel() < count:
        pad = value[-1].expand(count - value.numel()) if value.numel() else torch.zeros(count)
        value = torch.cat([value, pad], dim=0)
    return value[:count]


def _first_meta_vector(landmark_meta, keys, count):
    for key in keys:
        value = _meta_vector(landmark_meta, key, count)
        if value is not None:
            return value
    return None


def _quality_factor(values, floor=0.25):
    values = values.float()
    finite = torch.isfinite(values)
    factor = torch.zeros_like(values)
    if finite.any():
        finite_values = values[finite].clamp_min(0.0)
        if finite_values.numel() > 0 and float(finite_values.max().item()) > 1.0:
            finite_values = finite_values / finite_values.max().clamp_min(1e-6)
        factor[finite] = finite_values.clamp(0.0, 1.0)
    floor = max(0.0, min(float(floor), 1.0))
    return floor + (1.0 - floor) * factor


def _has_any_meta_key(landmark_meta, keys):
    return landmark_meta is not None and any(key in landmark_meta for key in keys)


def _error_cleanliness(values, scale=4.0):
    values = values.float()
    scale = max(float(scale), 1e-6)
    clean = torch.exp(-values.clamp_min(0.0) / scale).clamp(0.0, 1.0)
    clean[~torch.isfinite(clean)] = 0.0
    return clean


def _inverse_count_balance(ids, count):
    ids = torch.as_tensor(ids, dtype=torch.long).reshape(-1)
    device = ids.device
    balance = torch.ones(count, dtype=torch.float32, device=device)
    if ids.numel() < count:
        pad = torch.full((count - ids.numel(),), -1, dtype=torch.long, device=device)
        ids = torch.cat([ids, pad], dim=0)
    ids = ids[:count]
    valid = ids >= 0
    if not bool(valid.any().item()):
        return balance
    unique, counts = torch.unique(ids[valid], return_counts=True)
    count_map = torch.zeros(int(unique.max().item()) + 1, dtype=torch.float32, device=device)
    count_map[unique] = counts.to(dtype=torch.float32)
    balance[valid] = torch.rsqrt(count_map[ids[valid]].clamp_min(1.0))
    return balance / balance.mean().clamp_min(1e-6)


def _coverage_spatial_balance_from_meta(landmark_meta, count):
    uv = landmark_meta.get("coverage_uv") if landmark_meta is not None else None
    if uv is None:
        return None
    uv = torch.as_tensor(uv, dtype=torch.float32).reshape(-1, 2)
    if uv.numel() == 0:
        return None
    if uv.shape[0] < count:
        pad = uv[-1:].expand(count - uv.shape[0], -1) if uv.shape[0] else torch.zeros(count, 2)
        uv = torch.cat([uv, pad], dim=0)
    uv = uv[:count]
    grid_size_value = landmark_meta.get("coverage_grid_size", 4)
    grid_size = int(torch.as_tensor(grid_size_value).reshape(-1)[0].item()) if grid_size_value is not None else 4
    if grid_size <= 1:
        return None
    image_size = landmark_meta.get("coverage_image_size")
    if image_size is not None:
        image_size = torch.as_tensor(image_size, dtype=torch.float32).reshape(-1)
    if image_size is not None and image_size.numel() >= 2 and float(image_size[0]) > 0 and float(image_size[1]) > 0:
        height = float(image_size[0])
        width = float(image_size[1])
        x = torch.floor(uv[:, 0].clamp(0, width - 1) / max(width, 1.0) * grid_size)
        y = torch.floor(uv[:, 1].clamp(0, height - 1) / max(height, 1.0) * grid_size)
    else:
        finite = torch.isfinite(uv).all(dim=1)
        if not bool(finite.any().item()):
            return None
        min_xy = uv[finite].min(dim=0).values
        span = (uv[finite].max(dim=0).values - min_xy).clamp_min(1e-6)
        normalized = (uv - min_xy) / span
        x = torch.floor(normalized[:, 0].clamp(0, 1) * grid_size)
        y = torch.floor(normalized[:, 1].clamp(0, 1) * grid_size)
    x = x.to(dtype=torch.long).clamp(0, grid_size - 1)
    y = y.to(dtype=torch.long).clamp(0, grid_size - 1)
    return _inverse_count_balance(y * grid_size + x, count)


def _coverage_depth_balance_from_meta(landmark_meta, count):
    depth = landmark_meta.get("coverage_depth") if landmark_meta is not None else None
    if depth is None:
        return None
    depth = torch.as_tensor(depth, dtype=torch.float32).reshape(-1)
    if depth.numel() == 0:
        return None
    if depth.numel() < count:
        pad = depth[-1].expand(count - depth.numel())
        depth = torch.cat([depth, pad], dim=0)
    depth = depth[:count]
    bins_value = landmark_meta.get("coverage_depth_bins", 4)
    bins = int(torch.as_tensor(bins_value).reshape(-1)[0].item()) if bins_value is not None else 4
    if bins <= 1:
        return None
    finite = torch.isfinite(depth)
    if not bool(finite.any().item()):
        return None
    selected = depth[finite]
    span = (selected.max() - selected.min()).clamp_min(1e-6)
    ids = torch.full((count,), -1, dtype=torch.long, device=depth.device)
    ids[finite] = torch.floor((selected - selected.min()) / span * bins).to(dtype=torch.long).clamp(0, bins - 1)
    return _inverse_count_balance(ids, count)


def final_candidate_quality_from_meta(
    landmark_meta,
    count,
    reprojection_error_scale=4.0,
    cleanliness_weight=1.0,
    pose_info_weight=1.0,
    balance_weight=1.0,
    reliability_weight=0.25,
    utility_weight=0.0,
):
    """Compose detector supervision from final-candidate localization quality signals."""
    device = None
    if landmark_meta is not None:
        for value in landmark_meta.values():
            if torch.is_tensor(value):
                device = value.device
                break
    if device is None:
        device = torch.device("cpu")
    count = int(count)
    ones = torch.ones(count, dtype=torch.float32, device=device)
    eps = 1e-6

    raw_precision = _first_meta_vector(
        landmark_meta,
        ("raw_gt_precision_2px", "all_gt_precision_2px", "gt_precision_2px"),
        count,
    )
    inlier_precision = _first_meta_vector(
        landmark_meta,
        ("inlier_gt_precision_2px", "pnp_inlier_precision_2px"),
        count,
    )
    explicit_cleanliness = _first_meta_vector(
        landmark_meta,
        ("candidate_cleanliness", "gt_cleanliness", "gt_precision_6px", "raw_gt_precision_4px", "all_gt_precision_4px"),
        count,
    )
    clean_parts = []
    if raw_precision is not None:
        clean_parts.append(_quality_factor(raw_precision.to(device), floor=0.0))
    if inlier_precision is not None:
        clean_parts.append(_quality_factor(inlier_precision.to(device), floor=0.0))
    if explicit_cleanliness is not None:
        clean_parts.append(_quality_factor(explicit_cleanliness.to(device), floor=0.0))
    reproj_error = _first_meta_vector(
        landmark_meta,
        ("reproj_error", "gt_reproj_error", "gt_reproj_px"),
        count,
    )
    if reproj_error is not None:
        clean_parts.append(_error_cleanliness(reproj_error.to(device), scale=reprojection_error_scale))
    if clean_parts:
        cleanliness = torch.stack(clean_parts, dim=0).clamp_min(eps).log().mean(dim=0).exp()
    else:
        cleanliness = ones

    pose = _first_meta_vector(
        landmark_meta,
        ("pose_info_contribution", "pose_min_eig", "information", "pose_information"),
        count,
    )
    if pose is not None:
        pose_info = _quality_factor(pose.to(device), floor=0.05)
    else:
        pose_info = ones

    spatial_balance = _first_meta_vector(
        landmark_meta,
        ("spatial_balance", "geometry_balance"),
        count,
    )
    if spatial_balance is None and landmark_meta is not None:
        spatial_balance = _coverage_spatial_balance_from_meta(landmark_meta, count)
    if spatial_balance is not None:
        spatial_balance = _quality_factor(spatial_balance.to(device), floor=0.05)
    else:
        spatial_balance = ones

    depth_balance = _meta_vector(landmark_meta, "depth_balance", count)
    if depth_balance is None and landmark_meta is not None:
        depth_balance = _coverage_depth_balance_from_meta(landmark_meta, count)
    if depth_balance is not None:
        depth_balance = _quality_factor(depth_balance.to(device), floor=0.05)
    else:
        depth_balance = ones
    balance = (spatial_balance.clamp_min(eps) * depth_balance.clamp_min(eps)).sqrt()

    reliability_parts = []
    repeatability = _meta_vector(landmark_meta, "repeatability", count)
    if repeatability is not None:
        reliability_parts.append(_quality_factor(repeatability.to(device), floor=0.05))
    positive_prob = _meta_vector(landmark_meta, "positive_prob", count)
    if positive_prob is not None:
        reliability_parts.append(_quality_factor(positive_prob.to(device), floor=0.05))
    margin = _meta_vector(landmark_meta, "margin", count)
    if margin is not None:
        reliability_parts.append(_quality_factor(margin.to(device), floor=0.05))
    outlier = _meta_vector(landmark_meta, "outlier", count)
    if outlier is not None:
        reliability_parts.append((1.0 - _quality_factor(outlier.to(device), floor=0.0)).clamp(0.0, 1.0))
    if reliability_parts:
        reliability = torch.stack(reliability_parts, dim=0).clamp_min(eps).log().mean(dim=0).exp()
    else:
        reliability = ones

    utility = _meta_vector(landmark_meta, "utility", count)
    if utility is not None:
        utility_quality = _quality_factor(utility.to(device), floor=0.05)
    else:
        utility_quality = ones

    weighted_logs = []
    weight_sum = 0.0
    for value, weight in (
        (cleanliness, cleanliness_weight),
        (pose_info, pose_info_weight),
        (balance, balance_weight),
        (reliability, reliability_weight),
        (utility_quality, utility_weight),
    ):
        weight = max(0.0, float(weight))
        if weight <= 0.0:
            continue
        weighted_logs.append(weight * value.clamp_min(eps).log())
        weight_sum += weight

    if not weighted_logs:
        quality = ones
    else:
        quality = (torch.stack(weighted_logs, dim=0).sum(dim=0) / max(weight_sum, eps)).exp()
    quality[~torch.isfinite(quality)] = 0.0
    components = {
        "candidate_quality": quality.clamp(0.0, 1.0),
        "candidate_cleanliness": cleanliness.clamp(0.0, 1.0),
        "pose_info_contribution": pose_info.clamp(0.0, 1.0),
        "spatial_balance": spatial_balance.clamp(0.0, 1.0),
        "depth_balance": depth_balance.clamp(0.0, 1.0),
        "candidate_balance": balance.clamp(0.0, 1.0),
        "candidate_reliability": reliability.clamp(0.0, 1.0),
    }
    return components["candidate_quality"], components


def detector_landmark_quality_from_meta(landmark_meta, count, reprojection_error_scale=8.0):
    if landmark_meta is None:
        return None
    candidate_quality = _meta_vector(landmark_meta, "candidate_quality", count)
    if candidate_quality is not None:
        candidate_quality[~torch.isfinite(candidate_quality)] = 0.0
        return candidate_quality.clamp_min(0.0)
    final_signal_keys = (
        "candidate_cleanliness",
        "gt_cleanliness",
        "raw_gt_precision_2px",
        "all_gt_precision_2px",
        "gt_precision_2px",
        "inlier_gt_precision_2px",
        "pnp_inlier_precision_2px",
        "pose_info_contribution",
        "pose_min_eig",
        "reproj_error",
        "gt_reproj_error",
        "gt_reproj_px",
        "depth_balance",
        "spatial_balance",
        "geometry_balance",
        "coverage_uv",
        "coverage_depth",
    )
    if _has_any_meta_key(landmark_meta, final_signal_keys):
        quality, _ = final_candidate_quality_from_meta(
            landmark_meta,
            count,
            reprojection_error_scale=reprojection_error_scale,
        )
        return quality
    quality = _meta_vector(landmark_meta, "utility", count)
    if quality is None:
        quality = torch.ones(count, dtype=torch.float32)
    else:
        quality = quality.float().clamp_min(0.0)

    information = _first_meta_vector(landmark_meta, ("pose_min_eig", "information", "pose_information"), count)
    if information is not None:
        quality = quality * _quality_factor(information, floor=0.25)

    raw_precision = _first_meta_vector(
        landmark_meta,
        ("raw_gt_precision_2px", "all_gt_precision_2px", "gt_precision_2px"),
        count,
    )
    if raw_precision is not None:
        quality = quality * _quality_factor(raw_precision, floor=0.1)

    inlier_precision = _first_meta_vector(
        landmark_meta,
        ("inlier_gt_precision_2px", "pnp_inlier_precision_2px"),
        count,
    )
    if inlier_precision is not None:
        quality = quality * _quality_factor(inlier_precision, floor=0.1)

    precision = _first_meta_vector(
        landmark_meta,
        ("gt_precision_6px", "raw_gt_precision_4px", "all_gt_precision_4px"),
        count,
    )
    if precision is not None:
        quality = quality * _quality_factor(precision, floor=0.25)

    for balance_key in ("depth_balance", "spatial_balance", "geometry_balance"):
        balance = _meta_vector(landmark_meta, balance_key, count)
        if balance is not None:
            quality = quality * _quality_factor(balance, floor=0.25)
    if not any(key in landmark_meta for key in ("spatial_balance", "geometry_balance")):
        spatial_balance = _coverage_spatial_balance_from_meta(landmark_meta, count)
        if spatial_balance is not None:
            quality = quality * _quality_factor(spatial_balance, floor=0.25)
    if "depth_balance" not in landmark_meta:
        depth_balance = _coverage_depth_balance_from_meta(landmark_meta, count)
        if depth_balance is not None:
            quality = quality * _quality_factor(depth_balance, floor=0.25)

    reproj_error = _meta_vector(landmark_meta, "reproj_error", count)
    if reproj_error is None:
        reproj_error = _meta_vector(landmark_meta, "gt_reproj_error", count)
    if reproj_error is None:
        reproj_error = _meta_vector(landmark_meta, "gt_reproj_px", count)
    if reproj_error is not None:
        scale = max(float(reprojection_error_scale), 1e-6)
        reproj_quality = torch.exp(-reproj_error.float().clamp_min(0.0) / scale).clamp(0.0, 1.0)
        quality = quality * reproj_quality

    quality[~torch.isfinite(quality)] = 0.0
    return quality.clamp_min(0.0)


def generate_weighted_hard_gt_map(
    xyz,
    gt_feature_map,
    pose,
    K,
    utility=None,
    render_visible_mask=None,
):
    """Generate hard detector peaks and a calibrated utility loss-weight map."""
    height, width = gt_feature_map.shape[1], gt_feature_map.shape[2]
    device = gt_feature_map.device
    dtype = gt_feature_map.dtype
    xyz = xyz.to(device=device, dtype=dtype)
    xy, valid = _project_xyz_to_feature(xyz, pose, K, height, width)
    if render_visible_mask is not None:
        valid = valid & render_visible_mask.to(device=device, dtype=torch.bool)

    gt_flat = torch.zeros(height * width, device=device, dtype=dtype)
    weight_flat = torch.ones(height * width, device=device, dtype=dtype)
    if valid.sum() == 0:
        return gt_flat.view(height, width)[None], weight_flat.view(height, width)[None]

    xy_int = xy[:, valid].to(dtype=torch.long)
    flat_idx = xy_int[1] * width + xy_int[0]
    gt_flat[flat_idx] = 1.0

    utility_weights = _calibrated_utility_weights(utility)
    if utility_weights is not None:
        utility_weights = utility_weights.to(device=device, dtype=dtype)[valid]
        weight_flat.scatter_reduce_(
            0,
            flat_idx,
            utility_weights,
            reduce="amax",
            include_self=True,
        )
    return gt_flat.view(height, width)[None].detach(), weight_flat.view(height, width)[None].detach()


def generate_soft_gt_map(
    xyz,
    gt_feature_map,
    pose,
    K,
    utility=None,
    render_visible_mask=None,
    soft_sigma=1.5,
):
    """Generate local Gaussian detector targets without allocating a full image meshgrid per landmark."""
    height, width = gt_feature_map.shape[1], gt_feature_map.shape[2]
    device = gt_feature_map.device
    dtype = gt_feature_map.dtype
    xyz = xyz.to(device=device, dtype=dtype)
    xy, valid = _project_xyz_to_feature(xyz, pose, K, height, width)
    if render_visible_mask is not None:
        valid = valid & render_visible_mask.to(device=device, dtype=torch.bool)

    gt_flat = torch.zeros(height * width, device=device, dtype=dtype)
    weight_flat = torch.ones(height * width, device=device, dtype=dtype)
    if valid.sum() == 0:
        return gt_flat.view(height, width)[None], weight_flat.view(height, width)[None]

    sigma = max(float(soft_sigma), 1e-6)
    radius = max(1, int(torch.ceil(torch.tensor(3.0 * sigma)).item()))
    offsets = torch.arange(-radius, radius + 1, device=device)
    off_y, off_x = torch.meshgrid(offsets, offsets, indexing="ij")
    off_x = off_x.reshape(1, -1)
    off_y = off_y.reshape(1, -1)

    centers = xy[:, valid].T
    centers_int = centers.round().to(dtype=torch.long)
    px = centers_int[:, 0:1] + off_x
    py = centers_int[:, 1:2] + off_y
    in_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    dist2 = (px.to(dtype=dtype) - centers[:, 0:1]).pow(2) + (py.to(dtype=dtype) - centers[:, 1:2]).pow(2)
    values = torch.exp(-0.5 * dist2 / (sigma * sigma)).to(dtype=dtype)
    flat_idx = py * width + px

    gt_flat.scatter_reduce_(
        0,
        flat_idx[in_image].to(dtype=torch.long),
        values[in_image],
        reduce="amax",
        include_self=True,
    )

    utility_weights = _calibrated_utility_weights(utility)
    if utility_weights is not None:
        utility_weights = utility_weights.to(device=device, dtype=dtype)[valid]
        weight_values = 1.0 + (utility_weights[:, None] - 1.0) * values
        weight_flat.scatter_reduce_(
            0,
            flat_idx[in_image].to(dtype=torch.long),
            weight_values[in_image],
            reduce="amax",
            include_self=True,
        )
    return gt_flat.view(height, width)[None].detach(), weight_flat.view(height, width)[None].detach()


def utility_weighted_detector_loss(pred, target, weight_map=None, gamma=2.0, alpha=0.25):
    """Focal BCE for detector targets with optional calibrated utility weights."""
    target = target.float()
    pred = pred.float()
    if pred.min() < 0 or pred.max() > 1:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
    else:
        prob = pred.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(prob, target, reduction="none")
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = alpha_t * (1.0 - pt).pow(gamma) * bce
    weights = weight_map.float() if weight_map is not None else 1.0 + target
    return (focal * weights).sum() / weights.sum().clamp_min(1e-6)


def build_detector_target_map(
    gaussians,
    gt_feature_map,
    sampled_idx,
    pose,
    K,
    render_visible_mask=None,
    detector_target_mode="hard",
    landmark_meta=None,
    soft_sigma=1.5,
):
    if detector_target_mode == "soft":
        utility = detector_landmark_quality_from_meta(landmark_meta, int(sampled_idx.numel()))
        sampled_visible = None
        if render_visible_mask is not None:
            sampled_visible = render_visible_mask[sampled_idx]
        gt_map, weight_map = generate_soft_gt_map(
            gaussian_localization_xyz(gaussians)[sampled_idx],
            gt_feature_map,
            pose,
            K,
            utility=utility,
            render_visible_mask=sampled_visible,
            soft_sigma=soft_sigma,
        )
        return gt_map, True, weight_map
    if detector_target_mode == "weighted_hard":
        utility = detector_landmark_quality_from_meta(landmark_meta, int(sampled_idx.numel()))
        sampled_visible = None
        if render_visible_mask is not None:
            sampled_visible = render_visible_mask[sampled_idx]
        gt_map, weight_map = generate_weighted_hard_gt_map(
            gaussian_localization_xyz(gaussians)[sampled_idx],
            gt_feature_map,
            pose,
            K,
            utility=utility,
            render_visible_mask=sampled_visible,
        )
        return gt_map, True, weight_map
    if detector_target_mode != "hard":
        raise ValueError(f"Unknown detector_target_mode: {detector_target_mode}")
    return generate_gt_map(gaussians, gt_feature_map, sampled_idx, pose, K, render_visible_mask), False, None


def detector_target_loss(heat_map, gt_map, soft_target=False, weight_map=None):
    if soft_target:
        return utility_weighted_detector_loss(heat_map, gt_map, weight_map=weight_map)
    return score_map_bce_loss(heat_map, gt_map)


@torch.no_grad()
def random_knn_score(points, npoints, score, k=32, query_chunk=512, point_chunk=65536):
    points = points.detach()
    device = points.device
    dtype = torch.float32 if not points.is_floating_point() else points.dtype
    points = points.to(device=device, dtype=dtype)
    score = score.to(device=device, dtype=torch.float32).reshape(-1)
    total = int(points.shape[0])
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    npoints = min(int(npoints), total)
    k = max(1, min(int(k), total))
    sampled_idx = torch.randperm(total, device=device)[:npoints]
    selected = []
    selected_set = set()

    for q_start in range(0, npoints, int(query_chunk)):
        q_end = min(q_start + int(query_chunk), npoints)
        query = points[sampled_idx[q_start:q_end]]
        q_count = query.shape[0]
        best_dist = torch.full((q_count, k), float("inf"), dtype=dtype, device=device)
        best_idx = torch.full((q_count, k), -1, dtype=torch.long, device=device)

        for p_start in range(0, total, int(point_chunk)):
            p_end = min(p_start + int(point_chunk), total)
            dist = torch.cdist(query, points[p_start:p_end])
            local_k = min(k, p_end - p_start)
            local_dist, local_idx = torch.topk(dist, local_k, largest=False, dim=-1)
            local_idx = local_idx + p_start

            merged_dist = torch.cat([best_dist, local_dist], dim=1)
            merged_idx = torch.cat([best_idx, local_idx], dim=1)
            best_dist, order = torch.topk(merged_dist, k, largest=False, dim=-1)
            best_idx = torch.gather(merged_idx, 1, order)
            del dist, local_dist, local_idx, merged_dist, merged_idx, order

        knn_score = score[best_idx.clamp_min(0)]
        score_order = torch.argsort(knn_score, descending=True, dim=-1)
        best_idx_cpu = best_idx.detach().cpu()
        score_order_cpu = score_order.detach().cpu()
        fallback_cpu = sampled_idx[q_start:q_end].detach().cpu()
        for row in range(q_count):
            chosen = None
            for col in score_order_cpu[row].tolist():
                idx = int(best_idx_cpu[row, col].item())
                if idx >= 0 and idx not in selected_set:
                    chosen = idx
                    break
            if chosen is None:
                chosen = int(fallback_cpu[row].item())
            if chosen not in selected_set:
                selected_set.add(chosen)
                selected.append(chosen)
        del best_dist, best_idx, knn_score, score_order

    return torch.tensor(selected, dtype=torch.long, device=device)


def matching_oriented_sample(
    scene,
    gaussians,
    feature_extractor,
    render_visible_masks,
    masks=None,
    num=16384,
    k=32,
    return_coverage_stats=False,
):
    viewpoint_stack = scene.getTrainCameras().copy()
    loc_xyz = gaussian_localization_xyz(gaussians)
    score_sum = torch.zeros(
        loc_xyz.shape[0], dtype=torch.float32, device="cuda"
    )
    score_num = torch.zeros(loc_xyz.shape[0], dtype=torch.int, device="cuda")
    uv_sum = torch.zeros((loc_xyz.shape[0], 2), dtype=torch.float32, device="cuda")
    depth_sum = torch.zeros(loc_xyz.shape[0], dtype=torch.float32, device="cuda")
    fine_resolution = (
        viewpoint_stack[0].original_image.shape[1],
        viewpoint_stack[0].original_image.shape[2],
    )

    for viewpoint_cam in tqdm(viewpoint_stack, desc="Match Score"):
        gt_image = viewpoint_cam.original_image.cuda()
        gt_feature_map = extract_normalized_feature_map(
            feature_extractor,
            gt_image,
            size=(fine_resolution[0], fine_resolution[1]),
        )

        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
        focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
        # print("focal:", focalX, focalY)
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        render_visible_mask = render_visible_mask_from_cache(
            render_visible_masks,
            viewpoint_cam.image_name,
            gt_feature_map.device,
        )
        if render_visible_mask is None:
            render_visible_mask = get_render_visible_mask(
                gaussians,
                viewpoint_cam,
                gt_feature_map.shape[2],
                gt_feature_map.shape[1],
            )
            store_render_visible_mask(
                render_visible_masks,
                viewpoint_cam.image_name,
                render_visible_mask,
            )
        if masks is not None:
            object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]
            mask = object_mask & distort_mask
            img_mask = (
                F.interpolate(
                    mask[None].float(),
                    size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                > 0.5
            )
        else:
            img_mask = None

        score, mask = calculate_match_score(
            gaussians,
            gt_feature_map,
            viewmat,
            K,
            render_visible_mask=render_visible_mask,
            img_mask=img_mask,
        )
        score_num[mask] += 1
        score_sum[mask] += score
        if return_coverage_stats and bool(mask.any().item()):
            xy, project_valid, depths = _project_xyz_to_feature_with_depth(
                loc_xyz,
                viewmat,
                K,
                gt_feature_map.shape[1],
                gt_feature_map.shape[2],
            )
            stat_mask = mask & project_valid
            uv_sum[stat_mask] += xy[:, stat_mask].T.float()
            depth_sum[stat_mask] += depths[stat_mask].float()

    observation_count = score_num.clone()
    score_num[score_num == 0] = 1  # avoid divide by zero
    score_avg = score_sum / score_num

    sampled_idx = random_knn_score(loc_xyz, num, score_avg, k=k)
    sampled_idx = torch.unique(sampled_idx)
    if return_coverage_stats:
        denom = observation_count.clamp_min(1).to(dtype=torch.float32)
        coverage_stats = {
            "uv": uv_sum / denom[:, None],
            "depth": depth_sum / denom,
            "observed": observation_count > 0,
            "image_size": torch.tensor(
                [fine_resolution[0], fine_resolution[1]],
                dtype=torch.long,
                device=loc_xyz.device,
            ),
        }
        return sampled_idx, score_avg, score_num, coverage_stats
    return sampled_idx, score_avg, score_num


def validate_detector_sampled_indices(
    sampled_idx,
    sampling_mode="baseline",
    min_loc_observations=0,
    point_count=None,
):
    sampled_idx = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1)
    if sampled_idx.numel() == 0:
        raise ValueError(
            "sampled 0 detector landmarks; "
            f"sampling_mode={sampling_mode}, min_loc_observations={min_loc_observations}. "
            "Check that the loaded Gaussian map has localization observations for localization-aware sampling, "
            "or lower min_loc_observations/use baseline sampling for detector-only ablations."
        )
    if point_count is not None:
        min_idx = int(sampled_idx.min().item())
        max_idx = int(sampled_idx.max().item())
        if min_idx < 0 or max_idx >= int(point_count):
            raise ValueError(
                "detector landmark indices are outside the Gaussian point cloud; "
                f"point_count={int(point_count)}, min_idx={min_idx}, max_idx={max_idx}"
            )
    return sampled_idx


def detector_sampling_observed_mask(loc_observation_count, min_loc_observations=1, coverage_stats=None):
    observed = torch.as_tensor(loc_observation_count) >= int(min_loc_observations)
    if coverage_stats is None or "observed" not in coverage_stats:
        return observed
    coverage_observed = torch.as_tensor(
        coverage_stats["observed"],
        device=observed.device,
        dtype=torch.bool,
    )
    if coverage_observed.shape != observed.shape:
        raise ValueError(
            "coverage_stats['observed'] shape does not match localization observation count: "
            f"{tuple(coverage_observed.shape)} vs {tuple(observed.shape)}"
        )
    return observed & coverage_observed


def load_precomputed_detector_landmarks(path, point_count=None, device=None):
    with open(path, "rb") as handle:
        sampled_idx = pickle.load(handle)
    sampled_idx = validate_detector_sampled_indices(
        sampled_idx,
        sampling_mode="precomputed",
        point_count=point_count,
    )
    if device is not None:
        sampled_idx = sampled_idx.to(device=device, dtype=torch.long)
    return sampled_idx


def load_precomputed_landmark_meta(path, device="cuda"):
    meta_path = os.path.join(os.path.dirname(path), "landmark_meta.pt")
    if not os.path.exists(meta_path):
        return None
    return torch.load(meta_path, map_location=device)


def _gaussian_localization_vector(gaussians, name, count, default=0.0):
    value = getattr(gaussians, name, None)
    if torch.is_tensor(value):
        value = value.detach().float().reshape(-1)
        if value.numel() >= count:
            return value[:count]
        if value.numel() > 0:
            return torch.cat([value, value[-1].expand(count - value.numel())], dim=0)
    device = gaussians.get_xyz.device
    return torch.full((count,), float(default), dtype=torch.float32, device=device)


def final_candidate_quality_from_gaussians(
    gaussians,
    min_observations=4,
    coverage_stats=None,
    reprojection_error_scale=4.0,
    cleanliness_weight=1.0,
    pose_info_weight=1.0,
    balance_weight=1.0,
    reliability_weight=0.25,
    utility_weight=0.0,
):
    count = int(gaussians.get_xyz.shape[0])
    legacy_utility = gaussians.compute_localization_utility(min_observations=min_observations)
    meta = {
        "utility": legacy_utility.detach(),
        "repeatability": _gaussian_localization_vector(gaussians, "loc_repeatability_ema", count),
        "positive_prob": _gaussian_localization_vector(gaussians, "loc_positive_prob_ema", count),
        "margin": _gaussian_localization_vector(gaussians, "loc_margin_ema", count),
        "outlier": _gaussian_localization_vector(gaussians, "loc_outlier_ema", count),
        "reproj_error": _gaussian_localization_vector(gaussians, "loc_reproj_error_ema", count),
        "information": _gaussian_localization_vector(gaussians, "loc_information_ema", count),
    }
    if coverage_stats is not None:
        if coverage_stats.get("uv") is not None:
            meta["coverage_uv"] = coverage_stats["uv"]
        if coverage_stats.get("depth") is not None:
            meta["coverage_depth"] = coverage_stats["depth"]
        if coverage_stats.get("image_size") is not None:
            meta["coverage_image_size"] = coverage_stats["image_size"]
        if "coverage_grid_size" in coverage_stats:
            meta["coverage_grid_size"] = coverage_stats["coverage_grid_size"]
        if "coverage_depth_bins" in coverage_stats:
            meta["coverage_depth_bins"] = coverage_stats["coverage_depth_bins"]
    quality, components = final_candidate_quality_from_meta(
        meta,
        count,
        reprojection_error_scale=reprojection_error_scale,
        cleanliness_weight=cleanliness_weight,
        pose_info_weight=pose_info_weight,
        balance_weight=balance_weight,
        reliability_weight=reliability_weight,
        utility_weight=utility_weight,
    )
    observed = detector_sampling_observed_mask(
        gaussians.loc_observation_count,
        min_loc_observations=min_observations,
        coverage_stats=coverage_stats,
    )
    quality = quality.to(device=gaussians.get_xyz.device, dtype=torch.float32)
    quality = quality.masked_fill(~observed.to(device=quality.device, dtype=torch.bool), 0.0)
    components["candidate_quality"] = quality
    components["legacy_utility"] = legacy_utility.detach()
    return quality, components


def evaluate_detector(
    detector,
    feature_extractor,
    gaussians,
    sampled_idx,
    scene,
    masks=None,
    render_visible_masks=None,
    tb_writer=None,
    iteration=0,
):
    torch.cuda.empty_cache()

    landmarks = get_sampled_gaussian(gaussians, sampled_idx)

    bg_color = [1, 1, 1] if scene.args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)
            ],
        },
    )

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            fine_resolution = get_resolution_from_longest_edge(
                config["cameras"][0].original_image.shape[1],
                config["cameras"][0].original_image.shape[2],
                scene.longest_edge,
            )
            loss_sum = 0.0
            for idx, viewpoint_cam in enumerate(config["cameras"]):
                gt_image = viewpoint_cam.original_image.cuda()
                gt_feature_map = extract_normalized_feature_map(
                    feature_extractor,
                    gt_image,
                    size=(fine_resolution[0], fine_resolution[1]),
                )

                viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
                focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
                focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
                # print("focal:", focalX, focalY)
                K = torch.tensor(
                    [
                        [focalX, 0.0, gt_feature_map.shape[2] / 2],
                        [0.0, focalY, gt_feature_map.shape[1] / 2],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                    device="cuda",
                )
                visible_mask = render_visible_mask_from_cache(
                    render_visible_masks,
                    viewpoint_cam.image_name,
                    gt_feature_map.device,
                )
                if visible_mask is None:
                    visible_mask = get_render_visible_mask(
                        gaussians,
                        viewpoint_cam,
                        gt_feature_map.shape[2],
                        gt_feature_map.shape[1],
                    )
                    store_render_visible_mask(
                        render_visible_masks,
                        viewpoint_cam.image_name,
                        visible_mask,
                    )

                gt_map = generate_gt_map(
                    gaussians,
                    gt_feature_map,
                    sampled_idx,
                    viewmat,
                    K,
                    visible_mask,
                )

                if masks is not None:
                    object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
                    # sky_mask = masks[viewpoint_cam.image_name][1].cuda()[None]
                    distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]

                    mask = object_mask & distort_mask
                    gt_map_mask = (
                        F.interpolate(
                            mask[None].float(),
                            size=(gt_map.shape[1], gt_map.shape[2]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                        > 0.5
                    )
                    gt_map = gt_map * gt_map_mask

                # Loss
                heat_map = detector(gt_feature_map)
                loss = score_map_bce_loss(heat_map, gt_map)

                loss_sum += loss.item()
                if tb_writer and idx < 5:
                    render = render_gsplat(
                        viewpoint_cam, gaussians, background, rgb_only=True
                    )["render"]
                    sampled_render = render_gsplat(
                        viewpoint_cam, landmarks, background, rgb_only=True
                    )["render"]
                    heat_map = (heat_map - heat_map.min()) / (
                        heat_map.max() - heat_map.min()
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/gt_map_{idx}",
                        gt_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/heat_map{idx}",
                        heat_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/render_{idx}",
                        render[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/sampled_render_{idx}",
                        sampled_render[None],
                        iteration,
                    )

            loss_sum /= len(config["cameras"])
            print(
                f"\n[ITER {iteration}] Evaluating detector: {config['name']} loss {loss_sum}"
            )
            if tb_writer:
                tb_writer.add_scalar(
                f"detector_loss_patches/{config['name']}_loss",
                loss_sum,
                iteration,
            )


def _resolve_detector_artifact_path(scene_model_path, path):
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    candidate = os.path.join(scene_model_path, path)
    return candidate if os.path.exists(candidate) else path


def save_sparse_candidate_teacher_state(
    path,
    sampled_idx,
    landmark_features,
    iteration,
    config,
    diagnostics=None,
    dustbin_score=None,
    pair_scorer=None,
    pair_scorer_threshold=None,
):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    state = {
        "version": 3,
        "iteration": int(iteration),
        "landmark_indices": torch.as_tensor(sampled_idx, dtype=torch.long).detach().cpu(),
        "landmark_features": F.normalize(
            landmark_features.detach().reshape(landmark_features.shape[0], -1).float(),
            dim=1,
        ).cpu(),
        "config": dict(config),
        "diagnostics": dict(diagnostics or {}),
    }
    if dustbin_score is not None:
        state["dustbin_score"] = float(torch.as_tensor(dustbin_score).detach().item())
    if pair_scorer is not None:
        state["pair_scorer_config"] = pair_scorer.export_config()
        state["pair_scorer_state_dict"] = {
            key: value.detach().cpu() for key, value in pair_scorer.state_dict().items()
        }
    if pair_scorer_threshold is not None:
        state["pair_scorer_threshold"] = float(pair_scorer_threshold)
    torch.save(state, path)
    return state


def load_sparse_candidate_teacher_features(path, sampled_idx, device="cuda"):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or "landmark_features" not in state:
        raise ValueError(f"Invalid sparse candidate teacher state: {path}")
    expected = torch.as_tensor(sampled_idx, dtype=torch.long).reshape(-1).cpu()
    actual = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long).reshape(-1).cpu()
    if not torch.equal(actual, expected):
        raise ValueError(
            "sparse candidate teacher landmark indices do not match sampled_idx: "
            f"state_count={actual.numel()} expected_count={expected.numel()}"
        )
    features = torch.as_tensor(state["landmark_features"], dtype=torch.float32)
    if features.ndim < 2 or features.shape[0] != expected.numel():
        raise ValueError(
            "sparse candidate teacher feature count does not match sampled_idx: "
            f"features={features.shape[0] if features.ndim else 0} expected={expected.numel()}"
        )
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError("sparse candidate teacher features contain non-finite values")
    return features.to(device=device)


def _numeric_teacher_diagnostics(diagnostics):
    result = {}
    for key, value in diagnostics.items():
        if isinstance(value, bool):
            result[key] = bool(value)
        elif isinstance(value, (int, float)):
            result[key] = float(value)
        elif torch.is_tensor(value) and value.numel() == 1:
            result[key] = float(value.detach().item())
    return result


@torch.no_grad()
def evaluate_sparse_candidate_teacher(
    detector,
    feature_extractor,
    gaussians,
    sampled_idx,
    landmark_features,
    landmark_xyz,
    dustbin_score,
    pair_scorer,
    cameras,
    render_visible_masks,
    masks,
    scene,
    candidate_kwargs,
    assignment_temperature,
    assignment_margin,
    reprojection_sigma_px=1.0,
    scorer_min_recall=0.75,
    scorer_max_matches_per_keypoint=1,
):
    if not cameras:
        return {}
    was_training = detector.training
    detector.eval()
    records = []
    scorer_logits = []
    scorer_labels = []
    scorer_valid = []
    reranked_correct_count = 0
    reranked_valid_count = 0
    for camera in cameras:
        fine_resolution = get_resolution_from_longest_edge(
            camera.original_image.shape[1],
            camera.original_image.shape[2],
            scene.longest_edge,
        )
        feature_map = extract_normalized_feature_map(
            feature_extractor,
            camera.original_image.cuda(),
            size=(fine_resolution[0], fine_resolution[1]),
        )
        pose_w2c = camera.world_view_transform.transpose(0, 1).cuda()
        focal_x = fov2focal(camera.FoVx, feature_map.shape[2])
        focal_y = fov2focal(camera.FoVy, feature_map.shape[1])
        K = torch.tensor(
            [
                [focal_x, 0.0, feature_map.shape[2] / 2],
                [0.0, focal_y, feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=feature_map.device,
        )
        visible_mask = render_visible_mask_from_cache(
            render_visible_masks,
            camera.image_name,
            feature_map.device,
        )
        if visible_mask is None:
            with torch.enable_grad():
                visible_mask = get_render_visible_mask(
                    gaussians,
                    camera,
                    feature_map.shape[2],
                    feature_map.shape[1],
                )
            store_render_visible_mask(
                render_visible_masks,
                camera.image_name,
                visible_mask,
            )
        keypoint_heatmap, matchability_heatmap, offset_heatmap = detector.forward_all(
            feature_map
        )
        heatmap = rank_keypoint_proposals(
            keypoint_heatmap,
            matchability_heatmap,
            candidate_kwargs["nms_radius"],
        )
        if masks is not None:
            valid_mask = masks[camera.image_name][0].cuda()[None] & masks[camera.image_name][2].cuda()[None]
            valid_mask = F.interpolate(
                valid_mask[None].float(),
                size=feature_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0) > 0.5
            heatmap = heatmap * valid_mask
            matchability_heatmap = matchability_heatmap * valid_mask
        validation_candidate_kwargs = dict(candidate_kwargs)
        validation_candidate_kwargs["nms_radius"] = 0
        batch = build_sparse_candidate_batch(
            feature_map,
            heatmap,
            landmark_features,
            landmark_xyz,
            K,
            pose_w2c,
            visible_mask=visible_mask[sampled_idx],
            dustbin_score=dustbin_score,
            pair_scorer=pair_scorer,
            detector_supervision_heatmap=matchability_heatmap,
            keypoint_offset_map=offset_heatmap,
            **validation_candidate_kwargs,
        )
        losses = sparse_candidate_losses(
            batch,
            assignment_temperature=assignment_temperature,
            assignment_margin=assignment_margin,
            reprojection_sigma_px=reprojection_sigma_px,
        )
        record = _numeric_teacher_diagnostics(batch.diagnostics)
        record.update(
            {
                "loss_pair": float(losses.pair.item()),
                "loss_hard_negative": float(losses.hard_negative.item()),
                "loss_assignment": float(losses.assignment.item()),
                "loss_dustbin_assignment": float(losses.dustbin_assignment.item()),
                "loss_matcher_assignment": float(losses.matcher_assignment.item()),
                "loss_matcher_reprojection_assignment": float(
                    losses.matcher_reprojection_assignment.item()
                ),
                "loss_pair_scorer": float(losses.pair_scorer.item()),
                "loss_pair_scorer_assignment": float(
                    losses.pair_scorer_assignment.item()
                ),
                "loss_matcher_translation_info": float(
                    losses.matcher_translation_info.item()
                ),
                "loss_translation_info": float(losses.translation_info.item()),
                "loss_detector_match": float(losses.detector_match.item()),
                "loss_detector_offset": float(losses.detector_offset.item()),
                "loss_geometry_set": float(losses.geometry_set.item()),
                "loss_coverage": float(losses.coverage.item()),
            }
        )
        records.append(record)
        if batch.pair_scorer_logits.numel() > 0:
            hypothesis_ids = torch.arange(
                batch.pair_scorer_logits.numel(),
                device=batch.pair_scorer_logits.device,
            )
            calibrated_matches = limit_matches_per_keypoint(
                SparseMatchResult(
                    batch.pair_scorer_keypoint_idx,
                    hypothesis_ids,
                    batch.pair_scorer_logits,
                ),
                scorer_max_matches_per_keypoint,
            )
            selected = calibrated_matches.landmark_idx
            selected_valid = batch.pair_scorer_valid_mask[selected]
            selected_correct = (
                (batch.pair_scorer_labels[selected] > 0.5) & selected_valid
            )
            reranked_correct_count += int(selected_correct.sum().item())
            reranked_valid_count += int(selected_valid.sum().item())
            scorer_logits.append(batch.pair_scorer_logits[selected].detach().cpu())
            scorer_labels.append(batch.pair_scorer_labels[selected].detach().cpu())
            scorer_valid.append(batch.pair_scorer_valid_mask[selected].detach().cpu())
    if was_training:
        detector.train()
    keys = sorted({key for record in records for key in record})
    result = {"camera_count": float(len(records))}
    for key in keys:
        values = [record[key] for record in records if key in record]
        if values:
            tensor = torch.as_tensor(values, dtype=torch.float64)
            result[f"{key}_mean"] = float(tensor.mean().item())
            result[f"{key}_median"] = float(tensor.median().item())
    if scorer_logits:
        result["pair_scorer_reranked_correct_count_mean"] = float(
            reranked_correct_count / len(records)
        )
        result["pair_scorer_reranked_valid_count_mean"] = float(
            reranked_valid_count / len(records)
        )
        result["pair_scorer_reranked_gt_precision"] = float(
            reranked_correct_count / max(reranked_valid_count, 1)
        )
        calibrated = calibrate_binary_threshold(
            torch.cat(scorer_logits),
            torch.cat(scorer_labels),
            torch.cat(scorer_valid),
            min_recall=scorer_min_recall,
        )
        result.update(
            {
                "pair_scorer_calibrated_threshold": calibrated["threshold"],
                "pair_scorer_calibrated_precision": calibrated["precision"],
                "pair_scorer_calibrated_recall": calibrated["recall"],
                "pair_scorer_calibrated_accepted_count": float(
                    calibrated["accepted_count"]
                ),
                "pair_scorer_calibrated_correct_count": float(
                    calibrated["correct_count"]
                ),
            }
        )
    return result


def training_detector(
    gaussians,
    scene: Scene,
    masks,
    testing_iterations,
    saving_iterations,
    tb_writer,
    train_iteration=30000,
    detector_folder="",
    landmark_num=16384,
    landmark_k=32,
    sampling_mode="baseline",
    utility_weight=1.0,
    pnp_voxel_size=0.25,
    pnp_max_per_voxel=8,
    pnp_preserve_ratio=0.5,
    min_loc_observations=1,
    detector_target_mode="hard",
    soft_sigma=1.5,
    coverage_preserve_ratio=0.5,
    coverage_utility_ratio=0.25,
    coverage_high_confidence_ratio=0.0,
    coverage_grid_size=0,
    coverage_max_per_grid=0,
    coverage_depth_bins=0,
    coverage_max_per_depth_bin=0,
    coverage_allow_unbalanced_fallback=False,
    candidate_reprojection_error_scale=4.0,
    candidate_cleanliness_weight=1.0,
    candidate_pose_info_weight=1.0,
    candidate_balance_weight=1.0,
    candidate_reliability_weight=0.25,
    candidate_utility_weight=0.0,
    landmark_only=False,
    precomputed_landmark_path="",
    sparse_candidate_teacher=False,
    candidate_teacher_detector_init_path="",
    candidate_teacher_state_init_path="",
    candidate_teacher_pair_scorer_init_path="",
    candidate_teacher_optimize_features=False,
    candidate_teacher_freeze_detector=False,
    candidate_teacher_detector_lr=1e-4,
    candidate_teacher_feature_lr=5e-5,
    candidate_teacher_dustbin_lr=0.0,
    candidate_teacher_pair_scorer_lr=1e-3,
    candidate_teacher_pair_scorer_architecture="auto",
    candidate_teacher_detect_num=2048,
    candidate_teacher_nms_radius=2,
    candidate_teacher_match_mode="topk",
    candidate_teacher_match_topk=1,
    candidate_teacher_match_threshold=0.0,
    candidate_teacher_dual_softmax=False,
    candidate_teacher_dual_softmax_temperature=0.1,
    candidate_teacher_positive_radius_px=2.0,
    candidate_teacher_negative_radius_px=2.0,
    candidate_teacher_max_positives=1,
    candidate_teacher_hard_negatives=8,
    candidate_teacher_match_temperature=0.1,
    candidate_teacher_match_margin=0.5,
    candidate_teacher_assignment_temperature=0.05,
    candidate_teacher_assignment_margin=0.05,
    candidate_teacher_grid_rows=4,
    candidate_teacher_grid_cols=4,
    candidate_teacher_depth_bins=4,
    candidate_teacher_pair_weight=1.0,
    candidate_teacher_hard_negative_weight=0.5,
    candidate_teacher_assignment_weight=1.0,
    candidate_teacher_dustbin_weight=0.0,
    candidate_teacher_matcher_assignment_weight=0.0,
    candidate_teacher_matcher_reprojection_weight=0.0,
    candidate_teacher_reprojection_sigma_px=1.0,
    candidate_teacher_dustbin_init=0.5,
    candidate_teacher_pair_scorer_weight=0.0,
    candidate_teacher_pair_scorer_assignment_weight=0.0,
    candidate_teacher_matcher_translation_info_weight=0.0,
    candidate_teacher_translation_info_weight=0.0,
    candidate_teacher_pair_scorer_hidden_dim=16,
    candidate_teacher_pair_context_topk=8,
    candidate_teacher_scorer_min_recall=0.75,
    candidate_teacher_scorer_max_matches_per_keypoint=1,
    candidate_teacher_matchability_head=False,
    candidate_teacher_matchability_only=False,
    candidate_teacher_offset_head=False,
    candidate_teacher_offset_only=False,
    candidate_teacher_max_offset=2.0,
    candidate_teacher_offset_target_source="geometric_nearest",
    candidate_teacher_selection_source="combined",
    candidate_teacher_detector_target_source="geometric",
    candidate_teacher_detector_binary_target=False,
    candidate_teacher_detector_match_weight=1.0,
    candidate_teacher_detector_offset_weight=0.0,
    candidate_teacher_geometry_weight=0.1,
    candidate_teacher_coverage_weight=0.1,
    candidate_teacher_base_detector_weight=0.1,
    candidate_teacher_feature_anchor_weight=0.01,
    candidate_teacher_support_query_split=False,
    candidate_teacher_query_ratio=0.2,
    candidate_teacher_validation_ratio=0.0,
    candidate_teacher_split_mode="temporal_block",
    candidate_teacher_split_seed=2026,
):
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    feature_extractor = FeatureExtractor(scene.feature_type).cuda().eval()

    render_visible_masks = {}

    save_path = os.path.join(scene.model_path, detector_folder)
    os.makedirs(save_path, exist_ok=True)
    landmark_meta = None
    if precomputed_landmark_path:
        precomputed_landmark_path = _resolve_detector_artifact_path(
            scene.model_path,
            precomputed_landmark_path,
        )
        print(f"Loading precomputed detector landmarks from {precomputed_landmark_path}")
        sampled_idx = validate_detector_sampled_indices(
            load_precomputed_detector_landmarks(
                precomputed_landmark_path,
                point_count=gaussians.get_xyz.shape[0],
                device=gaussians.get_xyz.device,
            ),
            sampling_mode="precomputed",
            min_loc_observations=min_loc_observations,
            point_count=gaussians.get_xyz.shape[0],
        )
        landmark_meta = load_precomputed_landmark_meta(precomputed_landmark_path)
        if landmark_meta is not None:
            save_landmark_meta(os.path.join(save_path, "landmark_meta.pt"), landmark_meta)
    else:
        # M.O. sampling
        print("Matching oriented sampling...")
        sample_result = matching_oriented_sample(
            scene,
            gaussians,
            feature_extractor,
            render_visible_masks,
            masks=masks,
            num=landmark_num,
            k=landmark_k,
            return_coverage_stats=sampling_mode == "coverage_preserving",
        )
        if sampling_mode == "coverage_preserving":
            sampled_idx, score_avg, score_num, coverage_stats = sample_result
        else:
            sampled_idx, score_avg, score_num = sample_result
            coverage_stats = None
        if sampling_mode in {
            "localization_aware",
            "localization_aware_spatial",
            "localization_aware_global",
            "localization_aware_pnp",
            "coverage_preserving",
        }:
            if not hasattr(gaussians, "compute_localization_utility"):
                raise ValueError("localization_aware sampling requires Gaussian localization state")
            observed = detector_sampling_observed_mask(
                gaussians.loc_observation_count,
                min_loc_observations=min_loc_observations,
                coverage_stats=coverage_stats if sampling_mode == "coverage_preserving" else None,
            )
            candidate_quality, candidate_components = final_candidate_quality_from_gaussians(
                gaussians,
                min_observations=min_loc_observations,
                coverage_stats=coverage_stats if sampling_mode == "coverage_preserving" else None,
                reprojection_error_scale=candidate_reprojection_error_scale,
                cleanliness_weight=candidate_cleanliness_weight,
                pose_info_weight=candidate_pose_info_weight,
                balance_weight=candidate_balance_weight,
                reliability_weight=candidate_reliability_weight,
                utility_weight=candidate_utility_weight,
            )
            utility = candidate_quality
            if sampling_mode == "coverage_preserving":
                high_confidence = candidate_components.get("candidate_cleanliness", utility) * candidate_components.get(
                    "pose_info_contribution", utility.new_ones(utility.shape)
                )
                coverage_uv = coverage_stats.get("uv") if coverage_stats is not None else None
                coverage_depth = coverage_stats.get("depth") if coverage_stats is not None else None
                image_size_tensor = coverage_stats.get("image_size") if coverage_stats is not None else None
                image_size = (
                    tuple(int(v) for v in image_size_tensor.detach().cpu().tolist())
                    if image_size_tensor is not None
                    else None
                )
                sampled_idx, landmark_meta = coverage_preserving_sample(
                    gaussian_localization_xyz(gaussians),
                    score_avg,
                    utility,
                    num=landmark_num,
                    k=landmark_k,
                    min_observations=observed,
                    utility_weight=utility_weight,
                    base_preserve_ratio=coverage_preserve_ratio,
                    utility_preserve_ratio=coverage_utility_ratio,
                    high_confidence=high_confidence,
                    high_confidence_ratio=coverage_high_confidence_ratio,
                    voxel_size=pnp_voxel_size,
                    max_per_voxel=pnp_max_per_voxel,
                    uv=coverage_uv,
                    image_size=image_size,
                    grid_size=coverage_grid_size,
                    max_per_grid=coverage_max_per_grid,
                    depth=coverage_depth,
                    depth_bins=coverage_depth_bins,
                    max_per_depth_bin=coverage_max_per_depth_bin,
                    allow_unbalanced_fallback=coverage_allow_unbalanced_fallback,
                )
            else:
                pnp_balance = sampling_mode == "localization_aware_pnp"
                use_spatial_sampling = sampling_mode != "localization_aware_global"
                sampled_idx, landmark_meta = localization_aware_sample(
                    gaussian_localization_xyz(gaussians),
                    score_avg,
                    utility,
                    num=landmark_num,
                    k=landmark_k,
                    min_observations=observed,
                    utility_weight=utility_weight,
                    spatial=use_spatial_sampling,
                    pnp_balance=pnp_balance,
                    pnp_voxel_size=pnp_voxel_size,
                    pnp_max_per_voxel=pnp_max_per_voxel,
                    pnp_preserve_ratio=pnp_preserve_ratio,
                )
            sampled_idx = validate_detector_sampled_indices(
                sampled_idx,
                sampling_mode=sampling_mode,
                min_loc_observations=min_loc_observations,
                point_count=gaussians.get_xyz.shape[0],
            )
            landmark_meta["repeatability"] = gaussians.loc_repeatability_ema[sampled_idx]
            landmark_meta["margin"] = gaussians.loc_margin_ema[sampled_idx]
            landmark_meta["information"] = gaussians.loc_information_ema[sampled_idx]
            landmark_meta["reproj_error"] = gaussians.loc_reproj_error_ema[sampled_idx]
            landmark_meta["prototype"] = gaussians.loc_prototype[sampled_idx]
            landmark_meta["legacy_utility"] = candidate_components["legacy_utility"][sampled_idx]
            for key, value in candidate_components.items():
                if key == "legacy_utility":
                    continue
                landmark_meta[key] = value[sampled_idx]
            landmark_meta["full_candidate_quality"] = candidate_quality.detach().clone()
            landmark_meta["landmark_indices"] = sampled_idx.detach().clone()
            save_landmark_meta(os.path.join(save_path, "landmark_meta.pt"), landmark_meta)
        elif sampling_mode != "baseline":
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
        else:
            sampled_idx = validate_detector_sampled_indices(
                sampled_idx,
                sampling_mode=sampling_mode,
                min_loc_observations=min_loc_observations,
                point_count=gaussians.get_xyz.shape[0],
            )
    pickle.dump(sampled_idx, open(os.path.join(save_path, "sampled_idx.pkl"), "wb"))
    if sparse_candidate_teacher:
        requested_landmarks = int(landmark_num)
        unique_landmarks = int(torch.unique(sampled_idx).numel())
        if sampled_idx.numel() != requested_landmarks or unique_landmarks != requested_landmarks:
            raise ValueError(
                "sparse candidate teacher requires an exact, duplicate-free landmark bank: "
                f"requested={requested_landmarks} actual={sampled_idx.numel()} unique={unique_landmarks}"
            )
    if landmark_only:
        print(
            "Detector landmark-only bootstrap complete: "
            f"path={save_path} landmarks={sampled_idx.numel()} sampling_mode={sampling_mode}"
        )
        return
    if "score_avg" in locals():
        del score_avg, score_num
    if "utility" in locals():
        del utility
    if "observed" in locals():
        del observed
    torch.cuda.empty_cache()

    # training scene-specific detector
    print("Training detector...")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    training_cameras = scene.getTrainCameras().copy()
    validation_cameras = []
    support_camera_count = len(training_cameras)
    if sparse_candidate_teacher and candidate_teacher_support_query_split:
        support_cameras, query_cameras = split_support_query_cameras(
            training_cameras,
            query_ratio=candidate_teacher_query_ratio,
            seed=candidate_teacher_split_seed,
            mode=candidate_teacher_split_mode,
        )
        training_cameras = query_cameras
        if float(candidate_teacher_validation_ratio) > 0.0:
            training_cameras, validation_cameras = split_support_query_cameras(
                training_cameras,
                query_ratio=candidate_teacher_validation_ratio,
                seed=candidate_teacher_split_seed + 1,
                mode=candidate_teacher_split_mode,
            )
        support_camera_count = len(support_cameras)
        print(
            "Sparse candidate teacher support/query split: "
            f"support={len(support_cameras)} candidate_train={len(training_cameras)} "
            f"candidate_val={len(validation_cameras)} "
            f"mode={candidate_teacher_split_mode}"
        )

    viewpoint_stack = None
    progress_bar = tqdm(range(0, train_iteration), desc="Scene-Specific Detector")
    first_iter = 1

    detector = KpDetector(
        feature_extractor.feature_dim,
        matchability_head=candidate_teacher_matchability_head,
        offset_head=candidate_teacher_offset_head,
        max_offset=candidate_teacher_max_offset,
    ).cuda().train()
    detector_init_path = _resolve_detector_artifact_path(
        scene.model_path,
        candidate_teacher_detector_init_path,
    )
    if detector_init_path:
        print(f"Loading detector initialization from {detector_init_path}")
        detector_state = torch.load(detector_init_path, map_location="cuda")
        has_optional_head = bool(
            candidate_teacher_matchability_head or candidate_teacher_offset_head
        )
        incompatible = detector.load_state_dict(
            detector_state,
            strict=not has_optional_head,
        )
        if has_optional_head:
            allowed_missing = set()
            if candidate_teacher_matchability_head:
                allowed_missing.update(
                    {"matchability_head.weight", "matchability_head.bias"}
                )
            if candidate_teacher_offset_head:
                allowed_missing.update({"offset_head.weight", "offset_head.bias"})
            unexpected = set(incompatible.unexpected_keys)
            missing = set(incompatible.missing_keys)
            if unexpected or not missing.issubset(allowed_missing):
                raise ValueError(
                    "detector initialization is incompatible with optional heads: "
                    f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
                )
            if missing & {"matchability_head.weight", "matchability_head.bias"}:
                detector.initialize_matchability_from_keypoint()
            if missing & {"offset_head.weight", "offset_head.bias"}:
                detector.initialize_offset_to_zero()

    teacher_landmark_features = None
    teacher_initial_features = None
    teacher_landmark_xyz = None
    teacher_history = []
    teacher_validation_history = []
    teacher_last_diagnostics = {}
    calibrated_pair_scorer_threshold = None
    grad_accum = 8
    grad_clip_norm = 10.0
    teacher_config = {
        "enabled": bool(sparse_candidate_teacher),
        "optimize_features": bool(candidate_teacher_optimize_features),
        "freeze_detector": bool(candidate_teacher_freeze_detector),
        "detector_init_path": detector_init_path,
        "state_init_path": candidate_teacher_state_init_path,
        "pair_scorer_init_path": candidate_teacher_pair_scorer_init_path,
        "detector_lr": float(candidate_teacher_detector_lr),
        "feature_lr": float(candidate_teacher_feature_lr),
        "dustbin_lr": float(
            candidate_teacher_dustbin_lr
            if float(candidate_teacher_dustbin_lr) > 0.0
            else candidate_teacher_feature_lr
        ),
        "optimizer": "AdamW",
        "optimizer_weight_decay": 1e-4,
        "gradient_accumulation": int(grad_accum),
        "gradient_clip_norm": float(grad_clip_norm),
        "landmark_num": int(landmark_num),
        "sampling_mode": str(sampling_mode),
        "precomputed_landmark_path": str(precomputed_landmark_path),
        "detector_target_mode": str(detector_target_mode),
        "soft_sigma": float(soft_sigma),
        "detect_num": int(candidate_teacher_detect_num),
        "nms_radius": int(candidate_teacher_nms_radius),
        "match_mode": str(candidate_teacher_match_mode),
        "match_topk": int(candidate_teacher_match_topk),
        "match_threshold": float(candidate_teacher_match_threshold),
        "dual_softmax": bool(candidate_teacher_dual_softmax),
        "dual_softmax_temperature": float(candidate_teacher_dual_softmax_temperature),
        "positive_radius_px": float(candidate_teacher_positive_radius_px),
        "negative_radius_px": float(candidate_teacher_negative_radius_px),
        "max_positives": int(candidate_teacher_max_positives),
        "hard_negatives": int(candidate_teacher_hard_negatives),
        "match_temperature": float(candidate_teacher_match_temperature),
        "match_margin": float(candidate_teacher_match_margin),
        "assignment_temperature": float(candidate_teacher_assignment_temperature),
        "assignment_margin": float(candidate_teacher_assignment_margin),
        "grid_rows": int(candidate_teacher_grid_rows),
        "grid_cols": int(candidate_teacher_grid_cols),
        "depth_bins": int(candidate_teacher_depth_bins),
        "pair_weight": float(candidate_teacher_pair_weight),
        "hard_negative_weight": float(candidate_teacher_hard_negative_weight),
        "assignment_weight": float(candidate_teacher_assignment_weight),
        "dustbin_weight": float(candidate_teacher_dustbin_weight),
        "matcher_assignment_weight": float(
            candidate_teacher_matcher_assignment_weight
        ),
        "matcher_reprojection_weight": float(
            candidate_teacher_matcher_reprojection_weight
        ),
        "reprojection_sigma_px": float(candidate_teacher_reprojection_sigma_px),
        "dustbin_init": float(candidate_teacher_dustbin_init),
        "pair_scorer_weight": float(candidate_teacher_pair_scorer_weight),
        "pair_scorer_assignment_weight": float(
            candidate_teacher_pair_scorer_assignment_weight
        ),
        "matcher_translation_info_weight": float(
            candidate_teacher_matcher_translation_info_weight
        ),
        "translation_info_weight": float(candidate_teacher_translation_info_weight),
        "pair_scorer_lr": float(candidate_teacher_pair_scorer_lr),
        "pair_scorer_architecture": str(candidate_teacher_pair_scorer_architecture),
        "pair_scorer_hidden_dim": int(candidate_teacher_pair_scorer_hidden_dim),
        "pair_context_topk": int(candidate_teacher_pair_context_topk),
        "scorer_min_recall": float(candidate_teacher_scorer_min_recall),
        "scorer_max_matches_per_keypoint": int(
            candidate_teacher_scorer_max_matches_per_keypoint
        ),
        "matchability_head": bool(candidate_teacher_matchability_head),
        "matchability_only": bool(candidate_teacher_matchability_only),
        "offset_head": bool(candidate_teacher_offset_head),
        "offset_only": bool(candidate_teacher_offset_only),
        "max_offset": float(candidate_teacher_max_offset),
        "offset_target_source": str(candidate_teacher_offset_target_source),
        "selection_source": str(candidate_teacher_selection_source),
        "detector_target_source": str(candidate_teacher_detector_target_source),
        "detector_binary_target": bool(candidate_teacher_detector_binary_target),
        "detector_match_weight": float(candidate_teacher_detector_match_weight),
        "detector_offset_weight": float(candidate_teacher_detector_offset_weight),
        "geometry_weight": float(candidate_teacher_geometry_weight),
        "coverage_weight": float(candidate_teacher_coverage_weight),
        "base_detector_weight": float(candidate_teacher_base_detector_weight),
        "feature_anchor_weight": float(candidate_teacher_feature_anchor_weight),
        "support_query_split": bool(candidate_teacher_support_query_split),
        "support_camera_count": int(support_camera_count),
        "query_camera_count": int(len(training_cameras)),
        "validation_camera_count": int(len(validation_cameras)),
        "query_ratio": float(candidate_teacher_query_ratio),
        "validation_ratio": float(candidate_teacher_validation_ratio),
        "split_mode": str(candidate_teacher_split_mode),
        "split_seed": int(candidate_teacher_split_seed),
    }

    if sparse_candidate_teacher:
        teacher_initial_features = gaussians.materialized_loc_feature(sampled_idx).reshape(
            sampled_idx.numel(), -1
        ).detach().float().clone()
        state_init_path = _resolve_detector_artifact_path(
            scene.model_path,
            candidate_teacher_state_init_path,
        )
        teacher_init_state = None
        if state_init_path:
            print(f"Loading sparse candidate teacher feature initialization from {state_init_path}")
            teacher_init_state = torch.load(state_init_path, map_location="cpu")
            teacher_initial_features = load_sparse_candidate_teacher_features(
                state_init_path,
                sampled_idx,
                device=teacher_initial_features.device,
            )
        teacher_initial_features = F.normalize(teacher_initial_features, dim=1)
        teacher_landmark_features = torch.nn.Parameter(
            teacher_initial_features.clone(),
            requires_grad=bool(candidate_teacher_optimize_features),
        )
        teacher_landmark_xyz = gaussian_localization_xyz(gaussians)[sampled_idx].detach().float()
        initial_dustbin_score = float(candidate_teacher_dustbin_init)
        if isinstance(teacher_init_state, dict) and "dustbin_score" in teacher_init_state:
            initial_dustbin_score = float(teacher_init_state["dustbin_score"])
        teacher_dustbin_score = torch.nn.Parameter(
            teacher_initial_features.new_tensor(initial_dustbin_score),
            requires_grad=float(candidate_teacher_dustbin_weight) > 0.0,
        )
        scorer_init_state = teacher_init_state
        scorer_init_path = _resolve_detector_artifact_path(
            scene.model_path,
            candidate_teacher_pair_scorer_init_path,
        )
        if scorer_init_path:
            print(f"Loading pair scorer initialization from {scorer_init_path}")
            scorer_init_state = torch.load(scorer_init_path, map_location="cpu")
        scorer_state = (
            scorer_init_state.get("pair_scorer_state_dict")
            if isinstance(scorer_init_state, dict)
            else None
        )
        scorer_config = (
            scorer_init_state.get("pair_scorer_config", {})
            if isinstance(scorer_init_state, dict)
            else {}
        )
        optimize_pair_scorer = (
            float(candidate_teacher_pair_scorer_weight) > 0.0
            or float(candidate_teacher_pair_scorer_assignment_weight) > 0.0
        )
        if optimize_pair_scorer or scorer_state is not None:
            source_architecture = scorer_config.get(
                "architecture", "cosine_residual_v1"
            )
            scorer_architecture = str(candidate_teacher_pair_scorer_architecture)
            if scorer_architecture == "auto":
                scorer_architecture = source_architecture
            descriptor_dim = (
                int(teacher_initial_features.shape[1])
                if scorer_architecture == "descriptor_set_residual_v2"
                else 0
            )
            teacher_pair_scorer = SparsePairScorer(
                input_dim=int(scorer_config.get("input_dim", 6)),
                hidden_dim=int(
                    scorer_config.get(
                        "hidden_dim",
                        candidate_teacher_pair_scorer_hidden_dim,
                    )
                ),
                cosine_bias=float(candidate_teacher_dustbin_init),
                architecture=scorer_architecture,
                descriptor_dim=descriptor_dim,
            ).to(device=teacher_initial_features.device)
            if scorer_state is not None:
                upgrading_to_descriptor = (
                    source_architecture == "cosine_residual_v1"
                    and scorer_architecture == "descriptor_set_residual_v2"
                )
                incompatible = teacher_pair_scorer.load_state_dict(
                    scorer_state,
                    strict=not upgrading_to_descriptor,
                )
                if upgrading_to_descriptor:
                    allowed_missing = {
                        "descriptor_network.0.weight",
                        "descriptor_network.0.bias",
                        "descriptor_network.2.weight",
                        "descriptor_network.2.bias",
                    }
                    if (
                        set(incompatible.missing_keys) != allowed_missing
                        or incompatible.unexpected_keys
                    ):
                        raise ValueError(
                            "incompatible v1-to-v2 pair scorer upgrade: "
                            f"missing={incompatible.missing_keys} "
                            f"unexpected={incompatible.unexpected_keys}"
                        )
            teacher_pair_scorer.requires_grad_(
                optimize_pair_scorer
            )
        else:
            teacher_pair_scorer = None
    else:
        teacher_dustbin_score = None
        teacher_pair_scorer = None

    if candidate_teacher_matchability_only and candidate_teacher_offset_only:
        raise ValueError("matchability-only and offset-only training are mutually exclusive")
    if sparse_candidate_teacher and candidate_teacher_freeze_detector:
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
    elif sparse_candidate_teacher and candidate_teacher_offset_only:
        if detector.offset_head is None:
            raise ValueError("offset-only training requires --candidate_teacher_offset_head")
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        for parameter in detector.offset_head.parameters():
            parameter.requires_grad_(True)
    elif sparse_candidate_teacher and candidate_teacher_matchability_only:
        if detector.matchability_head is None:
            raise ValueError("matchability-only training requires --candidate_teacher_matchability_head")
        for parameter in detector.parameters():
            parameter.requires_grad_(False)
        for parameter in detector.matchability_head.parameters():
            parameter.requires_grad_(True)

    if sparse_candidate_teacher:
        parameter_groups = []
        detector_parameters = [parameter for parameter in detector.parameters() if parameter.requires_grad]
        if detector_parameters:
            parameter_groups.append(
                {"params": detector_parameters, "lr": float(candidate_teacher_detector_lr), "name": "detector"}
            )
        if teacher_landmark_features is not None and teacher_landmark_features.requires_grad:
            parameter_groups.append(
                {
                    "params": [teacher_landmark_features],
                    "lr": float(candidate_teacher_feature_lr),
                    "weight_decay": 0.0,
                    "name": "landmark_features",
                }
            )
        if teacher_dustbin_score is not None and teacher_dustbin_score.requires_grad:
            parameter_groups.append(
                {
                    "params": [teacher_dustbin_score],
                    "lr": float(
                        candidate_teacher_dustbin_lr
                        if float(candidate_teacher_dustbin_lr) > 0.0
                        else candidate_teacher_feature_lr
                    ),
                    "weight_decay": 0.0,
                    "name": "dustbin_score",
                }
            )
        if teacher_pair_scorer is not None:
            scorer_parameters = [
                parameter for parameter in teacher_pair_scorer.parameters() if parameter.requires_grad
            ]
            if scorer_parameters:
                parameter_groups.append(
                    {
                        "params": scorer_parameters,
                        "lr": float(candidate_teacher_pair_scorer_lr),
                        "weight_decay": 1e-4,
                        "name": "pair_scorer",
                    }
                )
        if not parameter_groups:
            raise ValueError(
                "sparse candidate teacher has no trainable parameters; enable detector training or "
                "--candidate_teacher_optimize_features"
            )
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(detector.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_iteration // grad_accum),
        eta_min=0.0 if sparse_candidate_teacher else 0.0005,
    )
    optimizer.zero_grad()

    for iteration in range(first_iter, train_iteration + 1):
        iter_start.record()
        if not viewpoint_stack:
            viewpoint_stack = training_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        fine_resolution = get_resolution_from_longest_edge(
            viewpoint_cam.original_image.shape[1],
            viewpoint_cam.original_image.shape[2],
            scene.longest_edge,
        )

        # generate gt_feature_map
        gt_image = viewpoint_cam.original_image.cuda()
        gt_feature_map = extract_normalized_feature_map(
            feature_extractor,
            gt_image,
            size=(fine_resolution[0], fine_resolution[1]),
        )

        # get viewmat and K
        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
        focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        # get render visible mask
        render_visible_mask = render_visible_mask_from_cache(
            render_visible_masks,
            viewpoint_cam.image_name,
            gt_feature_map.device,
        )
        if render_visible_mask is None:
            render_visible_mask = get_render_visible_mask(
                gaussians,
                viewpoint_cam,
                gt_feature_map.shape[2],
                gt_feature_map.shape[1],
            )
            store_render_visible_mask(
                render_visible_masks,
                viewpoint_cam.image_name,
                render_visible_mask,
            )

        need_base_target = (not sparse_candidate_teacher) or float(candidate_teacher_base_detector_weight) > 0.0
        gt_map = None
        soft_target = False
        weight_map = None
        if need_base_target:
            gt_map, soft_target, weight_map = build_detector_target_map(
                gaussians,
                gt_feature_map,
                sampled_idx,
                viewmat,
                K,
                render_visible_mask=render_visible_mask,
                detector_target_mode=detector_target_mode,
                landmark_meta=landmark_meta,
                soft_sigma=soft_sigma,
            )

        # use mask to filter out object
        gt_map_mask = None
        if masks is not None:
            object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]
            mask = object_mask & distort_mask
            gt_map_mask = (
                F.interpolate(
                    mask[None].float(),
                    size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                > 0.5
            )
            if gt_map is not None:
                gt_map = gt_map * gt_map_mask
                if weight_map is not None:
                    weight_map = torch.where(gt_map_mask, weight_map, torch.ones_like(weight_map))

        # Loss
        keypoint_heat_map, matchability_heat_map, offset_heat_map = detector.forward_all(
            gt_feature_map
        )
        heat_map = keypoint_heat_map
        candidate_heat_map = torch.sqrt(
            (keypoint_heat_map * matchability_heat_map).clamp_min(0.0)
        )
        base_detector_loss = (
            detector_target_loss(
                heat_map,
                gt_map,
                soft_target=soft_target,
                weight_map=weight_map,
            )
            if gt_map is not None
            else heat_map.sum() * 0.0
        )
        teacher_losses = None
        feature_anchor_loss = heat_map.sum() * 0.0
        if sparse_candidate_teacher:
            if candidate_teacher_selection_source == "keypoint_teacher":
                # Phase-D labels must come from a fixed proposal distribution.  The
                # keypoint head is frozen in matchability-only training, so detach it
                # here while retaining gradients through the sampled matchability
                # scores below.
                candidate_selection_heat_map = keypoint_heat_map.detach()
            elif candidate_teacher_selection_source == "combined":
                candidate_selection_heat_map = candidate_heat_map
            else:
                raise ValueError(
                    "candidate_teacher_selection_source must be 'combined' or "
                    f"'keypoint_teacher', got {candidate_teacher_selection_source!r}"
                )
            teacher_heat_map = (
                candidate_selection_heat_map
                if gt_map_mask is None
                else candidate_selection_heat_map * gt_map_mask
            )
            detector_supervision_heatmap = (
                matchability_heat_map
                if gt_map_mask is None
                else matchability_heat_map * gt_map_mask
            )
            candidate_batch = build_sparse_candidate_batch(
                gt_feature_map,
                teacher_heat_map,
                teacher_landmark_features,
                teacher_landmark_xyz,
                K,
                viewmat,
                visible_mask=render_visible_mask[sampled_idx],
                detect_num=candidate_teacher_detect_num,
                nms_radius=candidate_teacher_nms_radius,
                match_mode=candidate_teacher_match_mode,
                match_topk=candidate_teacher_match_topk,
                match_threshold=candidate_teacher_match_threshold,
                dual_softmax=candidate_teacher_dual_softmax,
                dual_softmax_temperature=candidate_teacher_dual_softmax_temperature,
                positive_radius_px=candidate_teacher_positive_radius_px,
                negative_radius_px=candidate_teacher_negative_radius_px,
                max_positives=candidate_teacher_max_positives,
                hard_negatives=candidate_teacher_hard_negatives,
                match_temperature=candidate_teacher_match_temperature,
                match_margin=candidate_teacher_match_margin,
                grid_rows=candidate_teacher_grid_rows,
                grid_cols=candidate_teacher_grid_cols,
                depth_bins=candidate_teacher_depth_bins,
                dustbin_score=teacher_dustbin_score,
                pair_scorer=teacher_pair_scorer,
                pair_context_topk=candidate_teacher_pair_context_topk,
                detector_supervision_heatmap=detector_supervision_heatmap,
                keypoint_offset_map=offset_heat_map,
                detector_offset_target_source=(
                    candidate_teacher_offset_target_source
                ),
                detector_target_source=candidate_teacher_detector_target_source,
                detector_binary_target=candidate_teacher_detector_binary_target,
            )
            teacher_losses = sparse_candidate_losses(
                candidate_batch,
                assignment_temperature=candidate_teacher_assignment_temperature,
                assignment_margin=candidate_teacher_assignment_margin,
                reprojection_sigma_px=candidate_teacher_reprojection_sigma_px,
            )
            if candidate_teacher_optimize_features:
                feature_anchor_loss = (
                    1.0
                    - (
                        F.normalize(teacher_landmark_features, dim=1)
                        * teacher_initial_features
                    ).sum(dim=1)
                ).clamp_min(0.0).mean()
            loss = (
                float(candidate_teacher_pair_weight) * teacher_losses.pair
                + float(candidate_teacher_hard_negative_weight) * teacher_losses.hard_negative
                + float(candidate_teacher_assignment_weight) * teacher_losses.assignment
                + float(candidate_teacher_dustbin_weight) * teacher_losses.dustbin_assignment
                + float(candidate_teacher_matcher_assignment_weight)
                * teacher_losses.matcher_assignment
                + float(candidate_teacher_matcher_reprojection_weight)
                * teacher_losses.matcher_reprojection_assignment
                + float(candidate_teacher_pair_scorer_weight) * teacher_losses.pair_scorer
                + float(candidate_teacher_pair_scorer_assignment_weight)
                * teacher_losses.pair_scorer_assignment
                + float(candidate_teacher_matcher_translation_info_weight)
                * teacher_losses.matcher_translation_info
                + float(candidate_teacher_translation_info_weight) * teacher_losses.translation_info
                + float(candidate_teacher_detector_match_weight) * teacher_losses.detector_match
                + float(candidate_teacher_detector_offset_weight)
                * teacher_losses.detector_offset
                + float(candidate_teacher_geometry_weight) * teacher_losses.geometry_set
                + float(candidate_teacher_coverage_weight) * teacher_losses.coverage
                + float(candidate_teacher_base_detector_weight) * base_detector_loss
                + float(candidate_teacher_feature_anchor_weight) * feature_anchor_loss
            )
            teacher_last_diagnostics = _numeric_teacher_diagnostics(candidate_batch.diagnostics)
        else:
            loss = base_detector_loss

        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                f"non-finite detector loss at iteration {iteration}: {float(loss.detach().item())}"
            )

        loss.backward()
        if iteration % grad_accum == 0 or iteration == train_iteration:
            if sparse_candidate_teacher:
                trainable_parameters = [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                if trainable_parameters:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        max_norm=grad_clip_norm,
                    )
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            loss_val = loss.item()
            if iteration % 10 == 0:
                postfix = {"Loss": f"{loss_val:.7f}"}
                if sparse_candidate_teacher:
                    postfix.update(
                        {
                            "Pair": f"{float(teacher_losses.pair.detach().item()):.4f}",
                            "Rank": f"{float(teacher_losses.assignment.detach().item()):.4f}",
                            "Reproj": f"{float(teacher_losses.matcher_reprojection_assignment.detach().item()):.4f}",
                            "Dust": f"{float(teacher_losses.dustbin_assignment.detach().item()):.4f}",
                            "Score": f"{float(teacher_losses.pair_scorer.detach().item()):.4f}",
                            "Trans": f"{float(teacher_losses.translation_info.detach().item()):.4f}",
                            "Offset": f"{float(teacher_losses.detector_offset.detach().item()):.4f}",
                            "Prec": f"{teacher_last_diagnostics.get('predicted_gt_precision', 0.0):.3f}",
                            "FN": f"{teacher_last_diagnostics.get('false_negative_rate', 0.0):.3f}",
                        }
                    )
                progress_bar.set_postfix(
                    postfix
                )
                progress_bar.update(10)
            if iteration == train_iteration:
                progress_bar.close()
            if tb_writer:
                tb_writer.add_scalar(
                    "detector_loss_patches/training_loss", loss_val, iteration
                )
                tb_writer.add_scalar(
                    "detector_loss_patches/lr",
                    optimizer.param_groups[0]["lr"],
                    iteration,
                )
                if sparse_candidate_teacher:
                    component_values = {
                        "pair": teacher_losses.pair,
                        "hard_negative": teacher_losses.hard_negative,
                        "assignment": teacher_losses.assignment,
                        "dustbin_assignment": teacher_losses.dustbin_assignment,
                        "matcher_assignment": teacher_losses.matcher_assignment,
                        "matcher_reprojection_assignment": (
                            teacher_losses.matcher_reprojection_assignment
                        ),
                        "pair_scorer": teacher_losses.pair_scorer,
                        "pair_scorer_assignment": teacher_losses.pair_scorer_assignment,
                        "matcher_translation_info": teacher_losses.matcher_translation_info,
                        "translation_info": teacher_losses.translation_info,
                        "detector_match": teacher_losses.detector_match,
                        "detector_offset": teacher_losses.detector_offset,
                        "geometry_set": teacher_losses.geometry_set,
                        "coverage": teacher_losses.coverage,
                        "base_detector": base_detector_loss,
                        "feature_anchor": feature_anchor_loss,
                    }
                    for name, value in component_values.items():
                        tb_writer.add_scalar(
                            f"sparse_candidate_teacher/loss_{name}",
                            float(value.detach().item()),
                            iteration,
                        )
                    for name, value in teacher_last_diagnostics.items():
                        tb_writer.add_scalar(
                            f"sparse_candidate_teacher/{name}",
                            value,
                            iteration,
                        )

            if sparse_candidate_teacher and (
                iteration == 1 or iteration % 50 == 0 or iteration == train_iteration
            ):
                history_item = {
                    "iteration": int(iteration),
                    "loss_total": float(loss.detach().item()),
                    "loss_pair": float(teacher_losses.pair.detach().item()),
                    "loss_hard_negative": float(teacher_losses.hard_negative.detach().item()),
                    "loss_assignment": float(teacher_losses.assignment.detach().item()),
                    "loss_dustbin_assignment": float(
                        teacher_losses.dustbin_assignment.detach().item()
                    ),
                    "loss_matcher_assignment": float(
                        teacher_losses.matcher_assignment.detach().item()
                    ),
                    "loss_matcher_reprojection_assignment": float(
                        teacher_losses.matcher_reprojection_assignment.detach().item()
                    ),
                    "loss_pair_scorer": float(teacher_losses.pair_scorer.detach().item()),
                    "loss_pair_scorer_assignment": float(
                        teacher_losses.pair_scorer_assignment.detach().item()
                    ),
                    "loss_matcher_translation_info": float(
                        teacher_losses.matcher_translation_info.detach().item()
                    ),
                    "loss_translation_info": float(
                        teacher_losses.translation_info.detach().item()
                    ),
                    "loss_detector_match": float(teacher_losses.detector_match.detach().item()),
                    "loss_detector_offset": float(
                        teacher_losses.detector_offset.detach().item()
                    ),
                    "loss_geometry_set": float(teacher_losses.geometry_set.detach().item()),
                    "loss_coverage": float(teacher_losses.coverage.detach().item()),
                    "loss_base_detector": float(base_detector_loss.detach().item()),
                    "loss_feature_anchor": float(feature_anchor_loss.detach().item()),
                }
                history_item.update(teacher_last_diagnostics)
                teacher_history.append(history_item)

        if iteration in testing_iterations:
            print("\n[ITER {}] Evaluating detector".format(iteration))
            detector.eval()
            evaluate_detector(
                detector,
                feature_extractor,
                gaussians,
                sampled_idx,
                scene,
                masks,
                render_visible_masks,
                tb_writer,
                iteration,
            )
            detector.train()

        if iteration in saving_iterations:
            print("\n[ITER {}] Saving detector".format(iteration))
            if sparse_candidate_teacher and validation_cameras:
                validation_metrics = evaluate_sparse_candidate_teacher(
                    detector,
                    feature_extractor,
                    gaussians,
                    sampled_idx,
                    teacher_landmark_features,
                    teacher_landmark_xyz,
                    teacher_dustbin_score,
                    teacher_pair_scorer,
                    validation_cameras,
                    render_visible_masks,
                    masks,
                    scene,
                    candidate_kwargs={
                        "detect_num": candidate_teacher_detect_num,
                        "nms_radius": candidate_teacher_nms_radius,
                        "match_mode": candidate_teacher_match_mode,
                        "match_topk": candidate_teacher_match_topk,
                        "match_threshold": candidate_teacher_match_threshold,
                        "dual_softmax": candidate_teacher_dual_softmax,
                        "dual_softmax_temperature": candidate_teacher_dual_softmax_temperature,
                        "positive_radius_px": candidate_teacher_positive_radius_px,
                        "negative_radius_px": candidate_teacher_negative_radius_px,
                        "max_positives": candidate_teacher_max_positives,
                        "hard_negatives": candidate_teacher_hard_negatives,
                        "match_temperature": candidate_teacher_match_temperature,
                        "match_margin": candidate_teacher_match_margin,
                        "grid_rows": candidate_teacher_grid_rows,
                        "grid_cols": candidate_teacher_grid_cols,
                        "depth_bins": candidate_teacher_depth_bins,
                        "pair_context_topk": candidate_teacher_pair_context_topk,
                        "detector_offset_target_source": (
                            candidate_teacher_offset_target_source
                        ),
                        "detector_target_source": candidate_teacher_detector_target_source,
                        "detector_binary_target": candidate_teacher_detector_binary_target,
                    },
                    assignment_temperature=candidate_teacher_assignment_temperature,
                    assignment_margin=candidate_teacher_assignment_margin,
                    reprojection_sigma_px=candidate_teacher_reprojection_sigma_px,
                    scorer_min_recall=candidate_teacher_scorer_min_recall,
                    scorer_max_matches_per_keypoint=(
                        candidate_teacher_scorer_max_matches_per_keypoint
                    ),
                )
                calibrated_pair_scorer_threshold = validation_metrics.get(
                    "pair_scorer_calibrated_threshold"
                )
                validation_item = {"iteration": int(iteration)}
                validation_item.update(validation_metrics)
                teacher_validation_history.append(validation_item)
                print(
                    "Sparse candidate validation: "
                    f"AP={validation_metrics.get('pair_ap_mean', 0.0):.4f} "
                    f"scorer_AP={validation_metrics.get('pair_scorer_ap_mean', 0.0):.4f} "
                    f"scorer_thr={validation_metrics.get('pair_scorer_calibrated_threshold', 0.0):.4f} "
                    f"accepted_precision={validation_metrics.get('dustbin_accepted_gt_precision_mean', 0.0):.4f} "
                    f"reject={validation_metrics.get('dustbin_unmatched_reject_accuracy_mean', 0.0):.4f}"
                )
            torch.save(detector.state_dict(), save_path + f"/{iteration}_detector.pth")
            if sparse_candidate_teacher:
                state_path = os.path.join(
                    save_path,
                    f"{iteration}_candidate_teacher_state.pt",
                )
                save_sparse_candidate_teacher_state(
                    state_path,
                    sampled_idx,
                    teacher_landmark_features,
                    iteration,
                    teacher_config,
                    teacher_last_diagnostics,
                    teacher_dustbin_score,
                    teacher_pair_scorer,
                    calibrated_pair_scorer_threshold,
                )
                save_sparse_candidate_teacher_state(
                    os.path.join(save_path, "candidate_teacher_state.pt"),
                    sampled_idx,
                    teacher_landmark_features,
                    iteration,
                    teacher_config,
                    teacher_last_diagnostics,
                    teacher_dustbin_score,
                    teacher_pair_scorer,
                    calibrated_pair_scorer_threshold,
                )

    if sparse_candidate_teacher:
        summary = {
            "version": 3,
            "iterations": int(train_iteration),
            "landmark_count": int(sampled_idx.numel()),
            "config": teacher_config,
            "final": teacher_history[-1] if teacher_history else teacher_last_diagnostics,
            "history": teacher_history,
            "validation_history": teacher_validation_history,
        }
        summary_path = os.path.join(save_path, "candidate_teacher_training_summary.json")
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(f"Saved sparse candidate teacher summary: {summary_path}")


def prepare_output_and_logger(args, folder=None):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    if folder:
        output_path = os.path.join(args.model_path, folder)
    else:
        output_path = args.model_path
    print("Output folder: {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def fill_missing_model_defaults(args):
    defaults = {
        "sh_degree": 3,
        "feature_type": "",
        "gaussian_type": "3dgs",
        "images": "images",
        "resolution": -1,
        "white_background": True,
        "longest_edge": 640,
        "data_device": "cuda",
        "eval": False,
        "speedup": False,
        "norm_before_render": True,
        "render_items": ["RGB", "Depth", "Edge", "Normal", "Curvature", "Feature Map"],
    }
    for key, value in defaults.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_arg_parser(with_components=False):
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--detector_folder", type=str, default="detector")
    parser.add_argument("--landmark_num", type=int, default=16384)
    parser.add_argument("--landmark_k", type=int, default=32)
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="baseline",
        choices=[
            "baseline",
            "localization_aware",
            "localization_aware_spatial",
            "localization_aware_global",
            "localization_aware_pnp",
            "coverage_preserving",
        ],
    )
    parser.add_argument("--utility_weight", type=float, default=1.0)
    parser.add_argument("--pnp_voxel_size", type=float, default=0.25)
    parser.add_argument("--pnp_max_per_voxel", type=int, default=8)
    parser.add_argument("--pnp_preserve_ratio", type=float, default=0.5)
    parser.add_argument("--min_loc_observations", type=int, default=1)
    parser.add_argument("--detector_target_mode", type=str, default="hard", choices=["hard", "soft", "weighted_hard"])
    parser.add_argument("--soft_sigma", type=float, default=1.5)
    parser.add_argument("--coverage_preserve_ratio", type=float, default=0.5)
    parser.add_argument("--coverage_utility_ratio", type=float, default=0.25)
    parser.add_argument("--coverage_high_confidence_ratio", type=float, default=0.0)
    parser.add_argument("--coverage_grid_size", type=int, default=0)
    parser.add_argument("--coverage_max_per_grid", type=int, default=0)
    parser.add_argument("--coverage_depth_bins", type=int, default=0)
    parser.add_argument("--coverage_max_per_depth_bin", type=int, default=0)
    parser.add_argument("--coverage_allow_unbalanced_fallback", action="store_true")
    parser.add_argument("--candidate_reprojection_error_scale", type=float, default=4.0)
    parser.add_argument("--candidate_cleanliness_weight", type=float, default=1.0)
    parser.add_argument("--candidate_pose_info_weight", type=float, default=1.0)
    parser.add_argument("--candidate_balance_weight", type=float, default=1.0)
    parser.add_argument("--candidate_reliability_weight", type=float, default=0.25)
    parser.add_argument("--candidate_utility_weight", type=float, default=0.0)
    parser.add_argument("--landmark_only", action="store_true", default=False)
    parser.add_argument("--precomputed_landmark_path", type=str, default="")
    parser.add_argument("--sparse_candidate_teacher", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_detector_init_path", type=str, default="")
    parser.add_argument("--candidate_teacher_state_init_path", type=str, default="")
    parser.add_argument("--candidate_teacher_pair_scorer_init_path", type=str, default="")
    parser.add_argument("--candidate_teacher_optimize_features", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_freeze_detector", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_detector_lr", type=float, default=1e-4)
    parser.add_argument("--candidate_teacher_feature_lr", type=float, default=5e-5)
    parser.add_argument("--candidate_teacher_dustbin_lr", type=float, default=0.0)
    parser.add_argument("--candidate_teacher_pair_scorer_lr", type=float, default=1e-3)
    parser.add_argument(
        "--candidate_teacher_pair_scorer_architecture",
        choices=["auto", "cosine_residual_v1", "descriptor_set_residual_v2"],
        default="auto",
    )
    parser.add_argument("--candidate_teacher_detect_num", type=int, default=2048)
    parser.add_argument("--candidate_teacher_nms_radius", type=int, default=2)
    parser.add_argument(
        "--candidate_teacher_match_mode",
        choices=["topk", "mnn"],
        default="topk",
    )
    parser.add_argument("--candidate_teacher_match_topk", type=int, default=1)
    parser.add_argument("--candidate_teacher_match_threshold", type=float, default=0.0)
    parser.add_argument("--candidate_teacher_dual_softmax", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_dual_softmax_temperature", type=float, default=0.1)
    parser.add_argument("--candidate_teacher_positive_radius_px", type=float, default=2.0)
    parser.add_argument("--candidate_teacher_negative_radius_px", type=float, default=2.0)
    parser.add_argument("--candidate_teacher_max_positives", type=int, default=1)
    parser.add_argument("--candidate_teacher_hard_negatives", type=int, default=8)
    parser.add_argument("--candidate_teacher_match_temperature", type=float, default=0.1)
    parser.add_argument("--candidate_teacher_match_margin", type=float, default=0.5)
    parser.add_argument("--candidate_teacher_assignment_temperature", type=float, default=0.05)
    parser.add_argument("--candidate_teacher_assignment_margin", type=float, default=0.05)
    parser.add_argument("--candidate_teacher_grid_rows", type=int, default=4)
    parser.add_argument("--candidate_teacher_grid_cols", type=int, default=4)
    parser.add_argument("--candidate_teacher_depth_bins", type=int, default=4)
    parser.add_argument("--candidate_teacher_pair_weight", type=float, default=1.0)
    parser.add_argument("--candidate_teacher_hard_negative_weight", type=float, default=0.5)
    parser.add_argument("--candidate_teacher_assignment_weight", type=float, default=1.0)
    parser.add_argument("--candidate_teacher_dustbin_weight", type=float, default=0.0)
    parser.add_argument(
        "--candidate_teacher_matcher_assignment_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_matcher_reprojection_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_reprojection_sigma_px", type=float, default=1.0
    )
    parser.add_argument("--candidate_teacher_dustbin_init", type=float, default=0.5)
    parser.add_argument("--candidate_teacher_pair_scorer_weight", type=float, default=0.0)
    parser.add_argument(
        "--candidate_teacher_pair_scorer_assignment_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--candidate_teacher_matcher_translation_info_weight", type=float, default=0.0
    )
    parser.add_argument("--candidate_teacher_translation_info_weight", type=float, default=0.0)
    parser.add_argument("--candidate_teacher_pair_scorer_hidden_dim", type=int, default=16)
    parser.add_argument("--candidate_teacher_pair_context_topk", type=int, default=8)
    parser.add_argument("--candidate_teacher_scorer_min_recall", type=float, default=0.75)
    parser.add_argument(
        "--candidate_teacher_scorer_max_matches_per_keypoint", type=int, default=1
    )
    parser.add_argument("--candidate_teacher_matchability_head", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_matchability_only", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_offset_head", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_offset_only", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_max_offset", type=float, default=2.0)
    parser.add_argument(
        "--candidate_teacher_offset_target_source",
        choices=["geometric_nearest", "matched_top1"],
        default="geometric_nearest",
    )
    parser.add_argument(
        "--candidate_teacher_selection_source",
        choices=["combined", "keypoint_teacher"],
        default="combined",
    )
    parser.add_argument(
        "--candidate_teacher_detector_target_source",
        choices=["geometric", "predicted_correct", "scorer_accepted_correct"],
        default="geometric",
    )
    parser.add_argument("--candidate_teacher_detector_binary_target", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_detector_match_weight", type=float, default=1.0)
    parser.add_argument("--candidate_teacher_detector_offset_weight", type=float, default=0.0)
    parser.add_argument("--candidate_teacher_geometry_weight", type=float, default=0.1)
    parser.add_argument("--candidate_teacher_coverage_weight", type=float, default=0.1)
    parser.add_argument("--candidate_teacher_base_detector_weight", type=float, default=0.1)
    parser.add_argument("--candidate_teacher_feature_anchor_weight", type=float, default=0.01)
    parser.add_argument("--candidate_teacher_support_query_split", action="store_true", default=False)
    parser.add_argument("--candidate_teacher_query_ratio", type=float, default=0.2)
    parser.add_argument("--candidate_teacher_validation_ratio", type=float, default=0.0)
    parser.add_argument(
        "--candidate_teacher_split_mode",
        choices=["random", "sequence_block", "temporal_block"],
        default="temporal_block",
    )
    parser.add_argument("--candidate_teacher_split_seed", type=int, default=2026)
    if with_components:
        return parser, lp, op
    return parser


if __name__ == "__main__":
    seed_everything(2025)
    # Set up command line argument parser
    parser, lp, op = build_arg_parser(with_components=True)
    args = get_combined_args(parser)
    fill_missing_model_defaults(args)
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    if dataset.gaussian_type == "3dgs":
        from scene.gaussian_model import GaussianModel
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        from scene.gaussian_model import GaussianModel_2dgs
        gaussians = GaussianModel_2dgs(dataset.sh_degree)

    masks = None
    for mask_path in (
        os.path.join(dataset.source_path, dataset.images, "masks.pkl"),
        os.path.join(dataset.source_path, "masks.pkl"),
    ):
        if os.path.exists(mask_path):
            import pickle
            masks = pickle.load(open(mask_path, "rb"))
            break

    scene = Scene(dataset, gaussians, load_iteration=args.iteration)

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(
            os.path.join(dataset.model_path, args.detector_folder)
        )
    else:
        tb_writer = None

    training_detector(
        gaussians,
        scene,
        masks,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        tb_writer=tb_writer,
        train_iteration=args.iterations,
        detector_folder=args.detector_folder,
        landmark_num=args.landmark_num,
        landmark_k=args.landmark_k,
        sampling_mode=args.sampling_mode,
        utility_weight=args.utility_weight,
        pnp_voxel_size=args.pnp_voxel_size,
        pnp_max_per_voxel=args.pnp_max_per_voxel,
        pnp_preserve_ratio=args.pnp_preserve_ratio,
        min_loc_observations=args.min_loc_observations,
        detector_target_mode=args.detector_target_mode,
        soft_sigma=args.soft_sigma,
        coverage_preserve_ratio=args.coverage_preserve_ratio,
        coverage_utility_ratio=args.coverage_utility_ratio,
        coverage_high_confidence_ratio=args.coverage_high_confidence_ratio,
        coverage_grid_size=args.coverage_grid_size,
        coverage_max_per_grid=args.coverage_max_per_grid,
        coverage_depth_bins=args.coverage_depth_bins,
        coverage_max_per_depth_bin=args.coverage_max_per_depth_bin,
        coverage_allow_unbalanced_fallback=args.coverage_allow_unbalanced_fallback,
        candidate_reprojection_error_scale=args.candidate_reprojection_error_scale,
        candidate_cleanliness_weight=args.candidate_cleanliness_weight,
        candidate_pose_info_weight=args.candidate_pose_info_weight,
        candidate_balance_weight=args.candidate_balance_weight,
        candidate_reliability_weight=args.candidate_reliability_weight,
        candidate_utility_weight=args.candidate_utility_weight,
        landmark_only=args.landmark_only,
        precomputed_landmark_path=args.precomputed_landmark_path,
        sparse_candidate_teacher=args.sparse_candidate_teacher,
        candidate_teacher_detector_init_path=args.candidate_teacher_detector_init_path,
        candidate_teacher_state_init_path=args.candidate_teacher_state_init_path,
        candidate_teacher_pair_scorer_init_path=(
            args.candidate_teacher_pair_scorer_init_path
        ),
        candidate_teacher_optimize_features=args.candidate_teacher_optimize_features,
        candidate_teacher_freeze_detector=args.candidate_teacher_freeze_detector,
        candidate_teacher_detector_lr=args.candidate_teacher_detector_lr,
        candidate_teacher_feature_lr=args.candidate_teacher_feature_lr,
        candidate_teacher_dustbin_lr=args.candidate_teacher_dustbin_lr,
        candidate_teacher_pair_scorer_lr=args.candidate_teacher_pair_scorer_lr,
        candidate_teacher_pair_scorer_architecture=(
            args.candidate_teacher_pair_scorer_architecture
        ),
        candidate_teacher_detect_num=args.candidate_teacher_detect_num,
        candidate_teacher_nms_radius=args.candidate_teacher_nms_radius,
        candidate_teacher_match_mode=args.candidate_teacher_match_mode,
        candidate_teacher_match_topk=args.candidate_teacher_match_topk,
        candidate_teacher_match_threshold=args.candidate_teacher_match_threshold,
        candidate_teacher_dual_softmax=args.candidate_teacher_dual_softmax,
        candidate_teacher_dual_softmax_temperature=args.candidate_teacher_dual_softmax_temperature,
        candidate_teacher_positive_radius_px=args.candidate_teacher_positive_radius_px,
        candidate_teacher_negative_radius_px=args.candidate_teacher_negative_radius_px,
        candidate_teacher_max_positives=args.candidate_teacher_max_positives,
        candidate_teacher_hard_negatives=args.candidate_teacher_hard_negatives,
        candidate_teacher_match_temperature=args.candidate_teacher_match_temperature,
        candidate_teacher_match_margin=args.candidate_teacher_match_margin,
        candidate_teacher_assignment_temperature=args.candidate_teacher_assignment_temperature,
        candidate_teacher_assignment_margin=args.candidate_teacher_assignment_margin,
        candidate_teacher_grid_rows=args.candidate_teacher_grid_rows,
        candidate_teacher_grid_cols=args.candidate_teacher_grid_cols,
        candidate_teacher_depth_bins=args.candidate_teacher_depth_bins,
        candidate_teacher_pair_weight=args.candidate_teacher_pair_weight,
        candidate_teacher_hard_negative_weight=args.candidate_teacher_hard_negative_weight,
        candidate_teacher_assignment_weight=args.candidate_teacher_assignment_weight,
        candidate_teacher_dustbin_weight=args.candidate_teacher_dustbin_weight,
        candidate_teacher_matcher_assignment_weight=(
            args.candidate_teacher_matcher_assignment_weight
        ),
        candidate_teacher_matcher_reprojection_weight=(
            args.candidate_teacher_matcher_reprojection_weight
        ),
        candidate_teacher_reprojection_sigma_px=(
            args.candidate_teacher_reprojection_sigma_px
        ),
        candidate_teacher_dustbin_init=args.candidate_teacher_dustbin_init,
        candidate_teacher_pair_scorer_weight=args.candidate_teacher_pair_scorer_weight,
        candidate_teacher_pair_scorer_assignment_weight=(
            args.candidate_teacher_pair_scorer_assignment_weight
        ),
        candidate_teacher_matcher_translation_info_weight=(
            args.candidate_teacher_matcher_translation_info_weight
        ),
        candidate_teacher_translation_info_weight=args.candidate_teacher_translation_info_weight,
        candidate_teacher_pair_scorer_hidden_dim=args.candidate_teacher_pair_scorer_hidden_dim,
        candidate_teacher_pair_context_topk=args.candidate_teacher_pair_context_topk,
        candidate_teacher_scorer_min_recall=args.candidate_teacher_scorer_min_recall,
        candidate_teacher_scorer_max_matches_per_keypoint=(
            args.candidate_teacher_scorer_max_matches_per_keypoint
        ),
        candidate_teacher_matchability_head=args.candidate_teacher_matchability_head,
        candidate_teacher_matchability_only=args.candidate_teacher_matchability_only,
        candidate_teacher_offset_head=args.candidate_teacher_offset_head,
        candidate_teacher_offset_only=args.candidate_teacher_offset_only,
        candidate_teacher_max_offset=args.candidate_teacher_max_offset,
        candidate_teacher_offset_target_source=(
            args.candidate_teacher_offset_target_source
        ),
        candidate_teacher_selection_source=args.candidate_teacher_selection_source,
        candidate_teacher_detector_target_source=args.candidate_teacher_detector_target_source,
        candidate_teacher_detector_binary_target=args.candidate_teacher_detector_binary_target,
        candidate_teacher_detector_match_weight=args.candidate_teacher_detector_match_weight,
        candidate_teacher_detector_offset_weight=(
            args.candidate_teacher_detector_offset_weight
        ),
        candidate_teacher_geometry_weight=args.candidate_teacher_geometry_weight,
        candidate_teacher_coverage_weight=args.candidate_teacher_coverage_weight,
        candidate_teacher_base_detector_weight=args.candidate_teacher_base_detector_weight,
        candidate_teacher_feature_anchor_weight=args.candidate_teacher_feature_anchor_weight,
        candidate_teacher_support_query_split=args.candidate_teacher_support_query_split,
        candidate_teacher_query_ratio=args.candidate_teacher_query_ratio,
        candidate_teacher_validation_ratio=args.candidate_teacher_validation_ratio,
        candidate_teacher_split_mode=args.candidate_teacher_split_mode,
        candidate_teacher_split_seed=args.candidate_teacher_split_seed,
    )

    # All done
    print("\n Scene-specific detector training complete.")
