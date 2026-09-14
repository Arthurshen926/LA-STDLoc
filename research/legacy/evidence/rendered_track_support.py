"""Gaussian-render support evidence for source-image-free Track repair.

The helpers in this module never create landmark geometry.  They evaluate
whether two already matched rendered observations are mutually compatible
with RGB-render support.  Expected depth is deliberately treated as uncertain:
only a simultaneous high-confidence depth and reprojection contradiction may
hard-reject an edge; every other case produces a soft confidence multiplier.
"""

from __future__ import annotations

import torch


def local_depth_spread(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    keypoints: torch.Tensor,
    *,
    alpha_minimum: float,
    radius: int = 1,
) -> torch.Tensor:
    """Return conservative local expected-depth range at sparse rows."""
    depth = torch.as_tensor(depth).float()
    alpha = torch.as_tensor(alpha).float()
    keypoints = torch.as_tensor(keypoints).float()
    if depth.ndim != 2 or alpha.shape != depth.shape:
        raise ValueError("rendered depth and alpha must be aligned image planes")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("keypoints must have shape [N, 2]")
    if int(radius) < 0:
        raise ValueError("local depth radius must be non-negative")
    height, width = depth.shape
    xy = keypoints.round().long()
    values = []
    valids = []
    for dy in range(-int(radius), int(radius) + 1):
        for dx in range(-int(radius), int(radius) + 1):
            x = (xy[:, 0] + dx).clamp(0, width - 1)
            y = (xy[:, 1] + dy).clamp(0, height - 1)
            sample = depth[y, x]
            valid = (
                torch.isfinite(sample)
                & (sample > 1e-5)
                & (alpha[y, x] >= float(alpha_minimum))
            )
            values.append(sample)
            valids.append(valid)
    stacked = torch.stack(values, dim=1)
    valid = torch.stack(valids, dim=1)
    lower = (
        torch.where(valid, stacked, torch.full_like(stacked, float("inf")))
        .min(1)
        .values
    )
    upper = (
        torch.where(valid, stacked, torch.full_like(stacked, -float("inf")))
        .max(1)
        .values
    )
    count = valid.sum(1)
    spread = upper - lower
    return torch.where(
        count >= 3, spread.clamp_min(0), torch.full_like(spread, float("inf"))
    )


def _backproject_world(
    uv: torch.Tensor,
    depth: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    ones = torch.ones((uv.shape[0], 1), dtype=torch.float64)
    ray = torch.cat((uv.double(), ones), dim=1) @ torch.linalg.inv(intrinsic.double()).T
    camera = ray * depth.double()[:, None]
    rotation = pose_w2c[:3, :3].double()
    translation = pose_w2c[:3, 3].double()
    return (camera - translation) @ rotation


def _project(
    world: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = world @ pose_w2c[:3, :3].double().T + pose_w2c[:3, 3].double()
    depth = camera[:, 2]
    homogeneous = camera @ intrinsic.double().T
    uv = homogeneous[:, :2] / depth[:, None].clamp_min(1e-8)
    return uv, depth


def pair_support_evidence(
    *,
    left_uv: torch.Tensor,
    right_uv: torch.Tensor,
    left_depth: torch.Tensor,
    right_depth: torch.Tensor,
    left_alpha: torch.Tensor,
    right_alpha: torch.Tensor,
    left_valid: torch.Tensor,
    right_valid: torch.Tensor,
    left_uncertainty: torch.Tensor,
    right_uncertainty: torch.Tensor,
    left_intrinsic: torch.Tensor,
    right_intrinsic: torch.Tensor,
    left_pose_w2c: torch.Tensor,
    right_pose_w2c: torch.Tensor,
    left_reliability: torch.Tensor | None = None,
    right_reliability: torch.Tensor | None = None,
    depth_abs_tolerance_m: float = 0.05,
    depth_relative_tolerance: float = 0.02,
    uncertainty_scale: float = 1.0,
    hard_alpha_minimum: float = 0.20,
    soft_cycle_px: float = 4.0,
    hard_cycle_px: float = 8.0,
    hard_depth_sigma: float = 3.0,
    uncertain_weight_floor: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Score aligned pair matches without treating rendered depth as truth."""
    left_uv = torch.as_tensor(left_uv).float()
    right_uv = torch.as_tensor(right_uv).float()
    count = int(left_uv.shape[0])
    if left_uv.shape != (count, 2) or right_uv.shape != (count, 2):
        raise ValueError("pair coordinates must be aligned [N, 2] tensors")
    columns = {
        "left_depth": left_depth,
        "right_depth": right_depth,
        "left_alpha": left_alpha,
        "right_alpha": right_alpha,
        "left_valid": left_valid,
        "right_valid": right_valid,
        "left_uncertainty": left_uncertainty,
        "right_uncertainty": right_uncertainty,
    }
    values = {
        name: torch.as_tensor(value).reshape(-1) for name, value in columns.items()
    }
    if any(value.numel() != count for value in values.values()):
        raise ValueError("support columns must align with pair matches")
    ld = values["left_depth"].float()
    rd = values["right_depth"].float()
    lu = values["left_uncertainty"].float()
    ru = values["right_uncertainty"].float()
    finite_depth = torch.isfinite(ld) & torch.isfinite(rd) & (ld > 1e-5) & (rd > 1e-5)
    valid_pair = (
        values["left_valid"].bool() & values["right_valid"].bool() & finite_depth
    )

    left_world = _backproject_world(left_uv, ld, left_intrinsic, left_pose_w2c)
    right_world = _backproject_world(right_uv, rd, right_intrinsic, right_pose_w2c)
    projected_right, predicted_right_depth = _project(
        left_world, right_intrinsic, right_pose_w2c
    )
    projected_left, predicted_left_depth = _project(
        right_world, left_intrinsic, left_pose_w2c
    )
    cycle = torch.maximum(
        torch.linalg.norm(projected_right.float() - right_uv, dim=1),
        torch.linalg.norm(projected_left.float() - left_uv, dim=1),
    )
    right_tolerance = (
        float(depth_abs_tolerance_m)
        + float(depth_relative_tolerance) * rd.abs()
        + float(uncertainty_scale) * ru.nan_to_num(posinf=1e6)
    ).clamp_min(1e-6)
    left_tolerance = (
        float(depth_abs_tolerance_m)
        + float(depth_relative_tolerance) * ld.abs()
        + float(uncertainty_scale) * lu.nan_to_num(posinf=1e6)
    ).clamp_min(1e-6)
    depth_sigma = torch.maximum(
        (predicted_right_depth.float() - rd).abs() / right_tolerance,
        (predicted_left_depth.float() - ld).abs() / left_tolerance,
    )
    finite = valid_pair & torch.isfinite(cycle) & torch.isfinite(depth_sigma)
    left_smooth = lu <= (
        float(depth_abs_tolerance_m) + float(depth_relative_tolerance) * ld.abs()
    )
    right_smooth = ru <= (
        float(depth_abs_tolerance_m) + float(depth_relative_tolerance) * rd.abs()
    )
    high_confidence = (
        finite
        & (values["left_alpha"].float() >= float(hard_alpha_minimum))
        & (values["right_alpha"].float() >= float(hard_alpha_minimum))
        & left_smooth
        & right_smooth
    )
    # A single expected-depth disagreement is never enough to erase a
    # projectively supported edge.  Hard rejection requires simultaneous,
    # high-confidence two-view geometric and depth contradiction.
    hard_reject = (
        high_confidence
        & (cycle > float(hard_cycle_px))
        & (depth_sigma > float(hard_depth_sigma))
    )
    cycle_score = torch.exp(-0.5 * (cycle / max(float(soft_cycle_px), 1e-6)).square())
    depth_score = torch.exp(-0.5 * depth_sigma.square())
    score = (cycle_score * depth_score).nan_to_num(0.0).clamp(0.0, 1.0)
    if left_reliability is not None and right_reliability is not None:
        reliability = torch.sqrt(
            torch.as_tensor(left_reliability).float().reshape(-1).clamp(0, 1)
            * torch.as_tensor(right_reliability).float().reshape(-1).clamp(0, 1)
        )
        if reliability.numel() != count:
            raise ValueError("appearance reliability and pair matches differ")
        score *= reliability
    floor = float(uncertain_weight_floor)
    weight = torch.where(
        finite,
        floor + (1.0 - floor) * score,
        torch.full_like(score, floor),
    ).clamp(0.0, 1.0)
    return {
        "soft_weight": weight,
        "hard_reject": hard_reject,
        "high_confidence_support": high_confidence,
        "cycle_error_px": cycle,
        "depth_disagreement_sigma": depth_sigma,
        "valid_support_pair": valid_pair,
    }
