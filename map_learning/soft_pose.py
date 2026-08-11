"""Differentiable local pose-bias surrogate around a known mapping pose."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from common.geometry import project_points
from topology.pose_information import pose_jacobian_analytic


def soft_pose_bias_loss(
    *,
    query_features: torch.Tensor,
    anchor_features: torch.Tensor,
    anchor_xyz: torch.Tensor,
    keypoint_xy: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_gt_w2c: torch.Tensor,
    topk: int = 8,
    temperature: float = 0.05,
    inlier_threshold_px: float = 4.0,
    inlier_softness_px: float = 1.0,
    damping: float = 1e-3,
    task_translation_m: float = 0.05,
    task_rotation_deg: float = 5.0,
    minimum_depth: float = 1e-4,
    miss_weight: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Penalize descriptor-induced one-step SE(3) bias at the GT pose.

    Candidate top-k membership is discrete, matching deployed retrieval. Within
    that set, descriptor probabilities, soft inlier weights, and the damped
    Gauss-Newton update are differentiable.
    """
    query_features = F.normalize(torch.as_tensor(query_features), dim=1)
    anchor_features = F.normalize(torch.as_tensor(anchor_features), dim=1)
    anchor_xyz = torch.as_tensor(
        anchor_xyz, device=query_features.device, dtype=query_features.dtype
    )
    keypoint_xy = torch.as_tensor(
        keypoint_xy, device=query_features.device, dtype=query_features.dtype
    )
    intrinsic = torch.as_tensor(
        intrinsic, device=query_features.device, dtype=query_features.dtype
    )
    pose_gt_w2c = torch.as_tensor(
        pose_gt_w2c, device=query_features.device, dtype=query_features.dtype
    )
    count = int(query_features.shape[0])
    if count == 0 or anchor_xyz.shape[0] < 4:
        zero = query_features.sum() * 0.0
        return zero, {"soft_pose_active": 0.0}
    candidate_count = min(max(int(topk), 1), int(anchor_xyz.shape[0]))
    scores, indices = torch.topk(
        query_features @ anchor_features.T, k=candidate_count, dim=1
    )
    probabilities = torch.softmax(scores / float(temperature), dim=1)
    points = anchor_xyz[indices].reshape(-1, 3)
    projected, positive_depth = project_points(points, intrinsic, pose_gt_w2c)
    points_h = torch.cat(
        [points, torch.ones_like(points[:, :1])], dim=1
    )
    camera_depth = (pose_gt_w2c @ points_h.T)[2]
    observations = keypoint_xy[:, None, :].expand(-1, candidate_count, -1)
    residual_flat = observations.reshape(-1, 2) - projected
    valid = (
        positive_depth
        & (camera_depth > float(minimum_depth))
        & torch.isfinite(projected).all(dim=1)
        & torch.isfinite(residual_flat).all(dim=1)
    )
    residual = torch.where(
        valid[:, None], residual_flat, torch.zeros_like(residual_flat)
    ).reshape(
        count, candidate_count, 2
    )
    residual_norm = torch.linalg.norm(residual, dim=2)
    soft_inlier = torch.sigmoid(
        (float(inlier_threshold_px) - residual_norm)
        / max(float(inlier_softness_px), 1e-4)
    )
    soft_inlier *= valid.reshape(count, candidate_count).to(
        dtype=soft_inlier.dtype
    )
    unnormalized_weights = probabilities * soft_inlier
    weight_sum = unnormalized_weights.sum(dim=1, keepdim=True)
    active_rows = weight_sum[:, 0] > 1e-8
    weights = torch.where(
        active_rows[:, None],
        unnormalized_weights / weight_sum.clamp_min(1e-8),
        torch.zeros_like(unnormalized_weights),
    )
    jacobian = pose_jacobian_analytic(points, intrinsic, pose_gt_w2c).reshape(
        count, candidate_count, 2, 6
    )
    jacobian = torch.where(
        valid.reshape(count, candidate_count, 1, 1),
        jacobian,
        torch.zeros_like(jacobian),
    )
    jacobian = torch.nan_to_num(jacobian, nan=0.0, posinf=0.0, neginf=0.0)
    contribution = jacobian.transpose(2, 3) @ jacobian
    rhs = (jacobian.transpose(2, 3) @ residual[..., None])[..., 0]
    active_count = active_rows.sum().clamp_min(1).to(dtype=weights.dtype)
    normal = (weights[..., None, None] * contribution).sum(dim=(0, 1)) / active_count
    gradient = (weights[..., None] * rhs).sum(dim=(0, 1)) / active_count
    normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0).double()
    gradient = torch.nan_to_num(
        gradient, nan=0.0, posinf=0.0, neginf=0.0
    ).double()
    eye = torch.eye(6, device=normal.device, dtype=normal.dtype)
    normal_scale = (normal.diagonal().mean().detach()).clamp_min(1.0)
    update = torch.linalg.solve(
        normal + float(damping) * normal_scale * eye, gradient
    )
    scale = torch.as_tensor(
        [
            task_translation_m,
            task_translation_m,
            task_translation_m,
            math.radians(task_rotation_deg),
            math.radians(task_rotation_deg),
            math.radians(task_rotation_deg),
        ],
        device=update.device,
        dtype=update.dtype,
    ).clamp_min(1e-6)
    normalized_update = update / scale
    pose_loss = F.smooth_l1_loss(
        normalized_update, torch.zeros_like(normalized_update), reduction="mean"
    )
    miss_loss = -torch.log(weight_sum[:, 0].clamp_min(1e-6)).mean()
    loss = pose_loss + float(miss_weight) * miss_loss.to(dtype=pose_loss.dtype)
    diagnostics = {
        "soft_pose_active": 1.0,
        "soft_pose_loss": float(loss.detach()),
        "soft_pose_update_loss": float(pose_loss.detach()),
        "soft_pose_miss_loss": float(miss_loss.detach()),
        "soft_pose_translation_update_m": float(
            torch.linalg.norm(update[:3]).detach()
        ),
        "soft_pose_rotation_update_deg": float(
            torch.linalg.norm(update[3:]).detach() * 180.0 / math.pi
        ),
        "soft_pose_effective_inlier_probability": float(
            (probabilities * soft_inlier).sum(dim=1).mean().detach()
        ),
        "soft_pose_active_row_fraction": float(active_rows.float().mean().detach()),
        "soft_pose_valid_candidate_fraction": float(valid.float().mean().detach()),
    }
    return loss, diagnostics
