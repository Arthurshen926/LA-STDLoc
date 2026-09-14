"""Leakage-safe supervision for a render-only scene-specific detector."""

from __future__ import annotations

import torch
import torch.nn.functional as F


IGNORE_LABEL = -1
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


def build_feedback_match_heatmap(
    *,
    image_hw: tuple[int, int],
    keypoints: torch.Tensor,
    reprojection_error_px: torch.Tensor,
    row_valid: torch.Tensor,
    row_uncertain: torch.Tensor,
    output_stride: int = 8,
    positive_threshold_px: float = 4.0,
    negative_threshold_px: float = 12.0,
) -> torch.Tensor:
    """Turn closed-loop Top-1 outcomes into detector-only supervision.

    A point is positive only when the *deployed* global Top-1 match is
    geometrically correct.  A confidently wrong Top-1 point, or a point on a
    V2-invalid render region, is negative.  The margin between the two
    reprojection thresholds and all V2-uncertain rows are ignored.  Thus the
    detector learns allocation reliability; it is never asked to repair a
    descriptor ranking that it cannot change.
    """

    height, width = map(int, image_hw)
    stride = int(output_stride)
    points = torch.as_tensor(keypoints).float().reshape(-1, 2)
    error = torch.as_tensor(reprojection_error_px, device=points.device).float().reshape(-1)
    valid = torch.as_tensor(row_valid, device=points.device).bool().reshape(-1)
    uncertain = torch.as_tensor(row_uncertain, device=points.device).bool().reshape(-1)
    if not (points.shape[0] == error.numel() == valid.numel() == uncertain.numel()):
        raise ValueError("feedback keypoint evidence must align")
    if stride < 1 or not 0 <= float(positive_threshold_px) < float(negative_threshold_px):
        raise ValueError("feedback detector thresholds are invalid")
    out_h, out_w = height // stride, width // stride
    labels = torch.full(
        (out_h, out_w), IGNORE_LABEL, dtype=torch.int8, device=points.device
    )
    xy = torch.floor(points / stride).long()
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < out_w)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < out_h)
        & torch.isfinite(error)
    )
    positive = inside & valid & (~uncertain) & (error <= float(positive_threshold_px))
    negative = inside & (~uncertain) & ((~valid) | (error >= float(negative_threshold_px)))
    # Resolve the rare cell collision conservatively: a demonstrated correct
    # correspondence is useful and wins over a wrong point in the same cell.
    for index in torch.nonzero(negative, as_tuple=False).reshape(-1).tolist():
        labels[xy[index, 1], xy[index, 0]] = NEGATIVE_LABEL
    for index in torch.nonzero(positive, as_tuple=False).reshape(-1).tolist():
        labels[xy[index, 1], xy[index, 0]] = POSITIVE_LABEL
    return labels


def build_pose_contribution_weights(
    *,
    labels: torch.Tensor,
    keypoints: torch.Tensor,
    image_hw: tuple[int, int],
    reprojection_error_px: torch.Tensor,
    camera_depth: torch.Tensor,
    match_margin: torch.Tensor,
    output_stride: int = 8,
    spatial_grid_hw: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    """Weight detector labels by non-LOO pose utility and harm proxies.

    Correct points receive more weight when they cover a rare spatial cell and
    provide useful parallax (off-axis or near-camera support).  Confidently
    wrong matches receive more negative weight.  This uses no point-removal or
    leave-one-out solve; it is a bounded analytic proxy for contribution to a
    well-conditioned correspondence set.
    """

    target = torch.as_tensor(labels)
    points = torch.as_tensor(keypoints, device=target.device).float().reshape(-1, 2)
    error = torch.as_tensor(reprojection_error_px, device=target.device).float().reshape(-1)
    depth = torch.as_tensor(camera_depth, device=target.device).float().reshape(-1)
    margin = torch.as_tensor(match_margin, device=target.device).float().reshape(-1)
    if not (points.shape[0] == error.numel() == depth.numel() == margin.numel()):
        raise ValueError("pose-contribution evidence must align")
    stride = int(output_stride)
    height, width = map(int, image_hw)
    if target.shape != (height // stride, width // stride):
        raise ValueError("pose-contribution labels have the wrong resolution")
    weights = torch.zeros_like(target, dtype=torch.float32)
    xy = torch.floor(points / stride).long()
    inside = (
        (xy[:, 0] >= 0) & (xy[:, 0] < target.shape[1])
        & (xy[:, 1] >= 0) & (xy[:, 1] < target.shape[0])
    )
    point_labels = torch.full((points.shape[0],), IGNORE_LABEL, device=target.device)
    point_labels[inside] = target[xy[inside, 1], xy[inside, 0]].long()

    positive = (point_labels == POSITIVE_LABEL) & torch.isfinite(depth) & (depth > 0)
    if bool(positive.any()):
        rows, cols = map(int, spatial_grid_hw)
        coarse_x = (points[:, 0] * cols / max(width, 1)).long().clamp(0, cols - 1)
        coarse_y = (points[:, 1] * rows / max(height, 1)).long().clamp(0, rows - 1)
        cell = coarse_y * cols + coarse_x
        count = torch.bincount(cell[positive], minlength=rows * cols).float().clamp_min(1)
        rarity = count[cell[positive]].rsqrt()
        centered_x = (points[positive, 0] - width / 2) / max(width / 2, 1)
        centered_y = (points[positive, 1] - height / 2) / max(height / 2, 1)
        off_axis = torch.sqrt(centered_x.square() + centered_y.square()).clamp_max(1.5)
        inverse_depth = depth[positive].reciprocal()
        inverse_depth = inverse_depth / inverse_depth.median().clamp_min(1e-6)
        leverage = rarity * (1.0 + 0.35 * off_axis + 0.25 * inverse_depth.clamp_max(3.0))
        leverage = leverage / leverage.mean().clamp_min(1e-6)
        for index, value in zip(torch.nonzero(positive).reshape(-1).tolist(), leverage.tolist()):
            y, x = int(xy[index, 1]), int(xy[index, 0])
            weights[y, x] = max(float(weights[y, x]), float(value))

    negative = point_labels == NEGATIVE_LABEL
    if bool(negative.any()):
        harm = 1.0 + (margin[negative].clamp_min(0) / 0.10).clamp_max(2.0)
        # Very large reprojection errors are harmful, but cap their influence.
        harm = harm * (1.0 + 0.25 * (error[negative] / 24.0).clamp(0, 2))
        harm = harm / harm.mean().clamp_min(1e-6)
        for index, value in zip(torch.nonzero(negative).reshape(-1).tolist(), harm.tolist()):
            y, x = int(xy[index, 1]), int(xy[index, 0])
            if target[y, x] == NEGATIVE_LABEL:
                weights[y, x] = max(float(weights[y, x]), float(value))
    weights[(target >= 0) & (weights == 0)] = 1.0
    return weights


def project_visible_clean_anchors(
    *,
    anchor_xyz: torch.Tensor,
    clean_anchor_mask: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
    rendered_depth: torch.Tensor,
    valid_pixel_mask: torch.Tensor,
    relative_depth_tolerance: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project clean Anchors and retain per-view depth/quality-consistent rows."""

    xyz = torch.as_tensor(anchor_xyz).float()
    clean = torch.as_tensor(clean_anchor_mask, device=xyz.device).bool().reshape(-1)
    K = torch.as_tensor(intrinsic, device=xyz.device).float()
    pose = torch.as_tensor(pose_w2c, device=xyz.device).float()
    depth = torch.as_tensor(rendered_depth, device=xyz.device).float().squeeze()
    support = torch.as_tensor(valid_pixel_mask, device=xyz.device).bool().squeeze()
    if xyz.ndim != 2 or xyz.shape[1] != 3 or clean.numel() != xyz.shape[0]:
        raise ValueError("Anchor xyz and clean mask must align")
    if depth.ndim != 2 or support.shape != depth.shape:
        raise ValueError("rendered depth and valid pixel mask must align")
    if float(relative_depth_tolerance) <= 0:
        raise ValueError("relative_depth_tolerance must be positive")

    rows = torch.nonzero(clean, as_tuple=False).reshape(-1)
    points = xyz[rows]
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    projected = camera @ K.T
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    xy = uv.round().long()
    height, width = depth.shape
    inside = (
        (camera[:, 2] > 1e-6)
        & (xy[:, 0] >= 0) & (xy[:, 0] < width)
        & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    )
    selected = torch.nonzero(inside, as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        return uv[:0], rows[:0]
    sampled_depth = depth[xy[selected, 1], xy[selected, 0]]
    sampled_support = support[xy[selected, 1], xy[selected, 0]]
    expected = camera[selected, 2]
    consistent = (
        sampled_support
        & torch.isfinite(sampled_depth)
        & (sampled_depth > 0)
        & ((sampled_depth - expected).abs()
           <= float(relative_depth_tolerance) * torch.maximum(sampled_depth, expected))
    )
    selected = selected[consistent]
    return uv[selected], rows[selected]


def spatially_balance_points(
    uv: torch.Tensor,
    reliability: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    grid_hw: tuple[int, int] = (24, 32),
    per_cell: int = 1,
) -> torch.Tensor:
    """Pick the highest-reliability projected targets in each image cell."""

    points = torch.as_tensor(uv).float()
    score = torch.as_tensor(reliability, device=points.device).float().reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2 or score.numel() != points.shape[0]:
        raise ValueError("projected points and reliability must align")
    if int(per_cell) < 1:
        raise ValueError("per_cell must be positive")
    height, width = map(int, image_hw)
    rows, cols = map(int, grid_hw)
    x = (points[:, 0] * cols / max(width, 1)).long().clamp(0, cols - 1)
    y = (points[:, 1] * rows / max(height, 1)).long().clamp(0, rows - 1)
    cell = y * cols + x
    chosen = []
    for value in torch.unique(cell).tolist():
        indices = torch.nonzero(cell == int(value), as_tuple=False).reshape(-1)
        count = min(int(per_cell), int(indices.numel()))
        chosen.append(indices[torch.topk(score[indices], count).indices])
    return torch.cat(chosen) if chosen else torch.empty(0, dtype=torch.long, device=points.device)


def build_tri_state_heatmap(
    *,
    image_hw: tuple[int, int],
    positive_uv: torch.Tensor,
    invalid_pixel_mask: torch.Tensor,
    uncertain_pixel_mask: torch.Tensor | None = None,
    output_stride: int = 8,
    positive_radius_px: float = 6.0,
) -> torch.Tensor:
    """Create positive/negative/ignore supervision at detector-head resolution.

    Only explicitly invalid renderer pixels are negative.  Uncovered clean
    content and uncertain geometry remain ignored rather than becoming false
    negatives.
    """

    height, width = map(int, image_hw)
    stride = int(output_stride)
    if stride < 1 or float(positive_radius_px) <= 0:
        raise ValueError("stride and positive radius must be positive")
    invalid = torch.as_tensor(invalid_pixel_mask).bool().squeeze()
    if invalid.shape != (height, width):
        raise ValueError("invalid pixel mask must match image size")
    uncertain = (
        torch.zeros_like(invalid)
        if uncertain_pixel_mask is None
        else torch.as_tensor(uncertain_pixel_mask, device=invalid.device).bool().squeeze()
    )
    if uncertain.shape != invalid.shape:
        raise ValueError("uncertain pixel mask must match image size")
    # SuperPoint applies three stride-2 pools, each with floor rounding.
    out_h = height // stride
    out_w = width // stride
    negative = F.adaptive_max_pool2d(invalid.float()[None, None], (out_h, out_w))[0, 0] > 0
    ignored_uncertain = F.adaptive_max_pool2d(uncertain.float()[None, None], (out_h, out_w))[0, 0] > 0
    labels = torch.full((out_h, out_w), IGNORE_LABEL, dtype=torch.int8, device=invalid.device)
    labels[negative & (~ignored_uncertain)] = NEGATIVE_LABEL

    points = torch.as_tensor(positive_uv, device=invalid.device).float().reshape(-1, 2)
    if points.numel():
        yy, xx = torch.meshgrid(
            torch.arange(out_h, device=invalid.device),
            torch.arange(out_w, device=invalid.device), indexing="ij",
        )
        centers = torch.stack(((xx + 0.5) * stride, (yy + 0.5) * stride), dim=-1)
        # Chunking keeps target construction bounded for dense Full maps.
        hit = torch.zeros((out_h, out_w), dtype=torch.bool, device=invalid.device)
        radius2 = float(positive_radius_px) ** 2
        for start in range(0, points.shape[0], 2048):
            distance2 = (centers[:, :, None] - points[None, None, start:start + 2048]).square().sum(-1)
            hit |= (distance2 <= radius2).any(-1)
        labels[hit & (~ignored_uncertain)] = POSITIVE_LABEL
    return labels
