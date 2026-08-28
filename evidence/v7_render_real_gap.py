"""Pure helpers for the non-formal V7 render--real causal diagnostic."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from evidence.v7_render_certificate import (
    CertificateThresholds,
    _depth_discontinuity,
    rgb_structure_support_mask,
)


def shared_support_mask(
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    *,
    thresholds: CertificateThresholds,
) -> torch.Tensor:
    """Return the pixel support proxy frozen for the P0.5 diagnostic.

    The renderer's distortion raster was not persisted.  Consequently this
    pixel mask intentionally omits the V2 rank-extreme distortion row veto and
    is named a proxy rather than silently claiming exact V2 row equivalence.
    """

    image = torch.as_tensor(rgb).float()
    alpha_value = torch.as_tensor(alpha, device=image.device).float().squeeze()
    depth_value = torch.as_tensor(depth, device=image.device).float().squeeze()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("shared-support RGB must have shape [3,H,W]")
    if alpha_value.shape != image.shape[1:] or depth_value.shape != image.shape[1:]:
        raise ValueError("shared-support rasters must align")
    geometry = (
        torch.isfinite(alpha_value)
        & (alpha_value >= float(thresholds.alpha_minimum))
        & torch.isfinite(depth_value)
        & (depth_value > 0)
    )
    discontinuity = _depth_discontinuity(
        depth_value, relative=float(thresholds.depth_discontinuity_relative)
    )
    height, width = depth_value.shape
    border = torch.zeros_like(geometry)
    margin = max(1, round(min(height, width) * float(thresholds.border_fraction)))
    border[:margin] = True
    border[-margin:] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    structure, _ = rgb_structure_support_mask(
        image,
        support_threshold=float(thresholds.rgb_structure_support_threshold),
        window_reference_px=int(thresholds.rgb_structure_window_reference_px),
        dilate_reference_px=int(thresholds.rgb_structure_dilate_reference_px),
        reference_short_side=int(thresholds.rgb_structure_reference_short_side),
    )
    return geometry & (~discontinuity) & (~border) & structure


def sample_pixel_mask(mask: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(mask).bool()
    points = torch.as_tensor(keypoints, device=value.device).float()
    if value.ndim != 2 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pixel mask/keypoints must be [H,W] and [N,2]")
    xy = torch.round(points).long()
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < value.shape[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < value.shape[0])
    )
    result = torch.zeros(points.shape[0], dtype=torch.bool, device=value.device)
    result[inside] = value[xy[inside, 1], xy[inside, 0]]
    return result


def feather_support(mask: torch.Tensor, *, reference_px: int = 15) -> torch.Tensor:
    value = torch.as_tensor(mask).float()
    if value.ndim != 2 or int(reference_px) < 1:
        raise ValueError("feather input must be [H,W] with a positive radius")
    scale = min(value.shape) / 1080.0
    radius = max(1, int(round(int(reference_px) * scale)))
    kernel = 2 * radius + 1
    # Two passes yield a smooth, deterministic tent kernel without requiring
    # an image-processing dependency or creating a hard hybrid seam.
    soft = value[None, None]
    for _ in range(2):
        soft = F.avg_pool2d(soft, kernel, stride=1, padding=radius)
    return soft[0, 0].clamp(0.0, 1.0)


def mutual_spatial_pairs(
    left_xy: torch.Tensor,
    right_xy: torch.Tensor,
    *,
    maximum_distance_px: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = torch.as_tensor(left_xy).float()
    right = torch.as_tensor(right_xy, device=left.device).float()
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != (2,) or right.shape[1:] != (2,):
        raise ValueError("mutual pairing requires [N,2] keypoint matrices")
    if float(maximum_distance_px) <= 0:
        raise ValueError("maximum pair distance must be positive")
    if left.shape[0] == 0 or right.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.long, device=left.device)
        return empty, empty.clone(), torch.empty(0, device=left.device)
    distances = torch.cdist(left, right)
    left_distance, left_to_right = distances.min(dim=1)
    right_to_left = distances.min(dim=0).indices
    left_rows = torch.arange(left.shape[0], device=left.device)
    keep = (
        (left_distance <= float(maximum_distance_px))
        & (right_to_left[left_to_right] == left_rows)
    )
    return left_rows[keep], left_to_right[keep], left_distance[keep]


def projected_match_correctness(
    keypoints: torch.Tensor,
    anchor_xyz: torch.Tensor,
    pose_w2c: torch.Tensor | np.ndarray,
    intrinsic: torch.Tensor | np.ndarray,
    *,
    maximum_reprojection_px: float,
) -> torch.Tensor:
    points = torch.as_tensor(anchor_xyz).double()
    xy = torch.as_tensor(keypoints, device=points.device).double() + 0.5
    pose = torch.as_tensor(pose_w2c, device=points.device).double()
    calibration = torch.as_tensor(intrinsic, device=points.device).double()
    camera = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    homogeneous = (calibration @ camera.T).T
    projected = homogeneous[:, :2] / homogeneous[:, 2:3].clamp_min(1e-12)
    return (camera[:, 2] > 0) & (
        torch.linalg.norm(projected - xy, dim=1) <= float(maximum_reprojection_px)
    )


def oracle_visible_correspondences(
    keypoints: torch.Tensor,
    anchor_xyz: torch.Tensor,
    pose_w2c: torch.Tensor | np.ndarray,
    intrinsic: torch.Tensor | np.ndarray,
    depth: torch.Tensor,
    row_support: torch.Tensor,
    *,
    maximum_reprojection_px: float,
    search_neighbors: int,
    absolute_depth_tolerance_m: float,
    relative_depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build GT-projection/depth-visible, Anchor-unique oracle pairs."""

    xyz = torch.as_tensor(anchor_xyz).double().cpu()
    pose = torch.as_tensor(pose_w2c).double().cpu()
    calibration = torch.as_tensor(intrinsic).double().cpu()
    query = torch.as_tensor(keypoints).double().cpu() + 0.5
    support = torch.as_tensor(row_support).bool().cpu()
    raster = torch.as_tensor(depth).float().squeeze().cpu()
    if support.shape != (query.shape[0],):
        raise ValueError("oracle row support does not align with keypoints")
    camera = (pose[:3, :3] @ xyz.T + pose[:3, 3:4]).T
    positive = torch.isfinite(camera).all(1) & (camera[:, 2] > 1e-8)
    projected = torch.empty(camera.shape[0], 2, dtype=torch.float64)
    projected[:] = torch.nan
    homogeneous = (calibration @ camera[positive].T).T
    projected[positive] = homogeneous[:, :2] / homogeneous[:, 2:3]
    active_rows = torch.nonzero(
        positive & torch.isfinite(projected).all(1), as_tuple=False
    ).flatten().numpy()
    query_rows = torch.nonzero(support, as_tuple=False).flatten().numpy()
    if active_rows.size == 0 or query_rows.size == 0:
        return np.empty((0, 2), np.float32), np.empty((0, 3), np.float32)
    tree = cKDTree(projected[active_rows].numpy())
    neighbors = min(max(1, int(search_neighbors)), active_rows.size)
    distances, local_rows = tree.query(query[query_rows].numpy(), k=neighbors)
    distances = np.asarray(distances).reshape(-1, neighbors)
    local_rows = np.asarray(local_rows).reshape(-1, neighbors)
    xy_floor = torch.floor(query[query_rows]).long()
    x = xy_floor[:, 0].clamp(0, raster.shape[1] - 1)
    y = xy_floor[:, 1].clamp(0, raster.shape[0] - 1)
    query_depth = raster[y, x].numpy()
    candidates: list[tuple[float, int, int]] = []
    camera_depth = camera[:, 2].numpy()
    for local_query, query_row in enumerate(query_rows):
        observed = float(query_depth[local_query])
        if not math.isfinite(observed) or observed <= 0:
            continue
        tolerance = max(
            float(absolute_depth_tolerance_m),
            float(relative_depth_tolerance) * observed,
        )
        for neighbor in range(neighbors):
            distance = float(distances[local_query, neighbor])
            if distance > float(maximum_reprojection_px):
                break
            anchor_row = int(active_rows[local_rows[local_query, neighbor]])
            if abs(float(camera_depth[anchor_row]) - observed) <= tolerance:
                candidates.append((distance, int(query_row), anchor_row))
    candidates.sort()
    used_query: set[int] = set()
    used_anchor: set[int] = set()
    selected: list[tuple[int, int]] = []
    for _, query_row, anchor_row in candidates:
        if query_row in used_query or anchor_row in used_anchor:
            continue
        used_query.add(query_row)
        used_anchor.add(anchor_row)
        selected.append((query_row, anchor_row))
    if not selected:
        return np.empty((0, 2), np.float32), np.empty((0, 3), np.float32)
    query_index = np.asarray([item[0] for item in selected], dtype=np.int64)
    anchor_index = np.asarray([item[1] for item in selected], dtype=np.int64)
    return query[query_index].float().numpy(), xyz[anchor_index].float().numpy()


def summarize_pose_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        return {"query_count": 0}
    translation = np.asarray([float(item["translation_error_cm"]) for item in rows])
    rotation = np.asarray([float(item["rotation_error_deg"]) for item in rows])
    success = (translation < 5.0) & (rotation < 5.0)
    catastrophic = (translation >= 100.0) | (rotation >= 30.0)
    return {
        "query_count": len(rows),
        "median_translation_cm": float(np.median(translation)),
        "p90_translation_cm": float(np.quantile(translation, 0.9)),
        "median_rotation_deg": float(np.median(rotation)),
        "p90_rotation_deg": float(np.quantile(rotation, 0.9)),
        "recall_5cm_5deg_percent": float(success.mean() * 100.0),
        "catastrophic_count": int(catastrophic.sum()),
    }


def signed_translation_bias(
    estimated_pose_w2c: Sequence[Sequence[float]],
    gt_pose_w2c: Sequence[Sequence[float]],
) -> np.ndarray:
    predicted = np.asarray(estimated_pose_w2c, dtype=np.float64)
    ground_truth = np.asarray(gt_pose_w2c, dtype=np.float64)
    predicted_center = -predicted[:3, :3].T @ predicted[:3, 3]
    gt_center = -ground_truth[:3, :3].T @ ground_truth[:3, 3]
    return ground_truth[:3, :3] @ (predicted_center - gt_center) * 100.0


def summarize_fixed_bias(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    vectors = np.asarray(
        [
            signed_translation_bias(item["estimated_pose_w2c"], item["gt_pose_w2c"])
            for item in rows
        ],
        dtype=np.float64,
    )
    norms = np.linalg.norm(vectors, axis=1)
    stable = norms < 5.0
    selected = vectors[stable] if np.any(stable) else vectors
    selected_norms = np.linalg.norm(selected, axis=1)
    units = selected / np.maximum(selected_norms[:, None], 1e-12)
    median_vector = np.median(selected, axis=0)
    median_norm = float(np.median(selected_norms))
    return {
        "query_count": int(selected.shape[0]),
        "camera_frame_median_vector_cm": median_vector.tolist(),
        "unit_direction_resultant": float(np.linalg.norm(units.mean(axis=0))),
        "robust_bias_ratio": float(np.linalg.norm(median_vector) / max(median_norm, 1e-12)),
    }
