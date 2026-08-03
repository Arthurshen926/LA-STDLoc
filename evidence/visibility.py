"""Projection and depth-consistency utilities for localization evidence."""

from __future__ import annotations

import math

import torch

from features.sampling import bilinear_sample_features


def fov2focal(fov: float, pixels: int) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def make_intrinsics_from_fov(
    fovx,
    fovy,
    width,
    height,
    device=None,
    dtype=torch.float32,
):
    """Build a centered pinhole intrinsic matrix."""
    return torch.tensor(
        [
            [fov2focal(fovx, width), 0.0, width / 2.0],
            [0.0, fov2focal(fovy, height), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def project_landmarks_to_query(
    xyz,
    K,
    pose_w2c,
    height,
    width,
    eps=1e-8,
    pixel_center_offset=0.0,
):
    """Project landmarks to grid coordinates under an explicit pixel offset."""
    if xyz.numel() == 0:
        empty_uv = xyz.new_zeros((0, 2))
        empty_depth = xyz.new_zeros((0,))
        empty_valid = torch.zeros(0, dtype=torch.bool, device=xyz.device)
        return empty_uv, empty_depth, empty_valid
    xyz = xyz.to(device=K.device, dtype=K.dtype)
    pose_w2c = pose_w2c.to(device=K.device, dtype=K.dtype)
    xyz_h = torch.cat(
        [xyz, torch.ones(xyz.shape[0], 1, dtype=K.dtype, device=K.device)],
        dim=1,
    )
    xyz_cam = (pose_w2c @ xyz_h.T)[:3].T
    depth = xyz_cam[:, 2]
    physical_uv = torch.empty(xyz.shape[0], 2, dtype=K.dtype, device=K.device)
    physical_uv[:, 0] = (
        K[0, 0] * xyz_cam[:, 0] / depth.clamp_min(eps) + K[0, 2]
    )
    physical_uv[:, 1] = (
        K[1, 1] * xyz_cam[:, 1] / depth.clamp_min(eps) + K[1, 2]
    )
    uv = physical_uv - torch.as_tensor(
        pixel_center_offset, dtype=K.dtype, device=K.device
    )
    valid = (
        (depth > eps)
        & torch.isfinite(uv).all(dim=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= height - 1)
    )
    return uv, depth, valid


def _sample_scalar_map(scalar_map, uv):
    if scalar_map is None:
        return None
    while scalar_map.dim() > 2:
        scalar_map = scalar_map.squeeze(0)
    return bilinear_sample_features(scalar_map[None], uv)[:, 0]


def filter_depth_consistent_landmarks(
    uv,
    projected_depth,
    valid,
    target_depth=None,
    target_alpha=None,
    alpha_threshold=0.2,
    abs_tolerance=1e-3,
    rel_tolerance=0.01,
):
    """Reject projections that disagree with rendered depth or alpha."""
    valid = valid.clone()
    if target_depth is not None:
        sampled_depth = _sample_scalar_map(
            target_depth.to(device=uv.device, dtype=uv.dtype), uv
        )
        tolerance = torch.maximum(
            uv.new_full(sampled_depth.shape, float(abs_tolerance)),
            sampled_depth.abs() * float(rel_tolerance),
        )
        valid &= (
            torch.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & ((projected_depth - sampled_depth).abs() <= tolerance)
        )
    if target_alpha is not None:
        sampled_alpha = _sample_scalar_map(
            target_alpha.to(device=uv.device, dtype=uv.dtype), uv
        )
        valid &= torch.isfinite(sampled_alpha) & (
            sampled_alpha >= float(alpha_threshold)
        )
    return valid
