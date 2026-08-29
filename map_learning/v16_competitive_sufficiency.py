"""Competition-aware localization sufficiency for delete-only map control.

Unlike additive mapping coverage, this module treats the deployed Top-1 winner
as part of the map state.  Certified positives must beat every active
non-certified competitor by a fixed margin before they contribute to the
Anchor-unique correspondence and pose reserve.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from topology.pose_information import compute_pose_information


def _raster_values(raster: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(raster).float().squeeze()
    return values[pixels[:, 1], pixels[:, 0]]


def certify_topl_relations(
    *,
    keypoints: torch.Tensor,
    candidate_anchor_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    anchor_covariance: torch.Tensor,
    observation_count: torch.Tensor,
    view_family_count: torch.Tensor,
    pose_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    surface_median_depth: torch.Tensor | None,
    row_valid: torch.Tensor,
    strict_reprojection_px: float = 4.0,
    broad_reprojection_px: float = 8.0,
    minimum_alpha: float = 0.05,
    strict_depth_absolute_m: float = 0.25,
    strict_depth_relative: float = 0.05,
    broad_depth_absolute_m: float = 0.50,
    broad_depth_relative: float = 0.10,
    maximum_projection_std_px: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Partition Top-L edges into certified positive/ambiguous/negative/invalid.

    Ambiguous edges are conservatively treated as competitors by the reserve
    calculation.  They can never become supervision positives.
    """

    xy = torch.as_tensor(keypoints).float().reshape(-1, 2)
    candidates = torch.as_tensor(candidate_anchor_rows).long()
    xyz = torch.as_tensor(anchor_xyz).float()
    covariance = torch.as_tensor(anchor_covariance).float()
    pose = torch.as_tensor(pose_w2c).float()
    calibration = torch.as_tensor(intrinsic).float()
    valid_rows = torch.as_tensor(row_valid).bool().reshape(-1)
    if candidates.ndim != 2 or candidates.shape[0] != xy.shape[0]:
        raise ValueError("Top-L candidates do not align with query rows")
    if valid_rows.numel() != xy.shape[0]:
        raise ValueError("valid-row mask does not align with query rows")
    if covariance.shape != (xyz.shape[0], 3, 3):
        raise ValueError("Anchor covariance does not align with Anchor xyz")
    if candidates.numel() and (
        int(candidates.min()) < 0 or int(candidates.max()) >= xyz.shape[0]
    ):
        raise ValueError("Top-L candidate is outside the frozen map")

    depth_raster = torch.as_tensor(depth).float().squeeze()
    height, width = map(int, depth_raster.shape)
    pixels = xy.round().long()
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    safe_pixels = pixels.clone()
    safe_pixels[:, 0].clamp_(0, max(width - 1, 0))
    safe_pixels[:, 1].clamp_(0, max(height - 1, 0))
    observed_alpha = _raster_values(alpha, safe_pixels)
    observed_depth = _raster_values(depth, safe_pixels)

    world = xyz[candidates]
    rotation = pose[:3, :3]
    camera = world @ rotation.T + pose[:3, 3]
    projected = camera @ calibration.T
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    reprojection = torch.linalg.norm(uv - xy[:, None], dim=2)

    anchor_covariance = covariance[candidates]
    camera_covariance = torch.einsum(
        "ab,rlbc,dc->rlad", rotation, anchor_covariance, rotation
    )
    x, y, z = camera.unbind(2)
    dproj = camera.new_zeros((*camera.shape[:2], 2, 3))
    dproj[..., 0, 0] = calibration[0, 0] / z.clamp_min(1e-8)
    dproj[..., 0, 2] = -calibration[0, 0] * x / z.square().clamp_min(1e-8)
    dproj[..., 1, 1] = calibration[1, 1] / z.clamp_min(1e-8)
    dproj[..., 1, 2] = -calibration[1, 1] * y / z.square().clamp_min(1e-8)
    pixel_covariance = dproj @ camera_covariance @ dproj.transpose(-1, -2)
    projection_std = pixel_covariance.diagonal(dim1=-2, dim2=-1).sum(2).clamp_min(0).sqrt()
    depth_std = camera_covariance[..., 2, 2].clamp_min(0).sqrt()

    strict_depth = torch.maximum(
        torch.full_like(observed_depth, float(strict_depth_absolute_m)),
        observed_depth.abs() * float(strict_depth_relative),
    )
    broad_depth = torch.maximum(
        torch.full_like(observed_depth, float(broad_depth_absolute_m)),
        observed_depth.abs() * float(broad_depth_relative),
    )
    surface_stable = torch.ones_like(valid_rows)
    if surface_median_depth is not None:
        surface = _raster_values(surface_median_depth, safe_pixels)
        surface_stable = (
            torch.isfinite(surface)
            & (surface > 0)
            & ((surface - observed_depth).abs() <= strict_depth)
        )
    supported_row = (
        valid_rows
        & inside
        & torch.isfinite(observed_alpha)
        & torch.isfinite(observed_depth)
        & (observed_alpha >= float(minimum_alpha))
        & (observed_depth > 0)
    )
    finite_edge = (
        torch.isfinite(uv).all(2)
        & torch.isfinite(projection_std)
        & torch.isfinite(depth_std)
        & (camera[..., 2] > 0)
    )
    depth_error = (camera[..., 2] - observed_depth[:, None]).abs()
    broad = (
        supported_row[:, None]
        & finite_edge
        & (reprojection <= float(broad_reprojection_px))
        & (depth_error <= broad_depth[:, None] + 3.0 * depth_std)
    )
    support = (
        (torch.as_tensor(observation_count).long()[candidates] >= 3)
        & (torch.as_tensor(view_family_count).long()[candidates] >= 1)
    )
    positive = (
        broad
        & surface_stable[:, None]
        & support
        & (reprojection <= float(strict_reprojection_px))
        & (projection_std <= float(maximum_projection_std_px))
        & (depth_std <= strict_depth[:, None])
        & (depth_error <= strict_depth[:, None] + 2.0 * depth_std)
    )
    ambiguous = broad & ~positive
    negative = supported_row[:, None] & finite_edge & ~broad
    invalid = ~(positive | ambiguous | negative)
    return {
        "positive": positive,
        "ambiguous": ambiguous,
        "negative": negative,
        "invalid": invalid,
        "reprojection_error_px": reprojection,
        "projection_std_px": projection_std,
        "depth_error_m": depth_error,
        "depth_std_m": depth_std,
    }


def _maximum_anchor_unique_matching(
    safe: torch.Tensor,
    candidates: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an exact maximum-cardinality row↔Anchor matching."""

    edge_rows = []
    for row in range(safe.shape[0]):
        columns = torch.nonzero(safe[row], as_tuple=False).reshape(-1)
        ordered = sorted(
            columns.tolist(),
            key=lambda column: (-float(scores[row, column]), int(candidates[row, column])),
        )
        edge_rows.append([int(candidates[row, column]) for column in ordered])
    row_order = sorted(
        range(len(edge_rows)),
        key=lambda row: (
            -max((float(scores[row, column]) for column in torch.nonzero(safe[row]).reshape(-1).tolist()), default=-torch.inf),
            row,
        ),
    )
    anchor_to_row: dict[int, int] = {}

    def augment(row: int, seen: set[int]) -> bool:
        for anchor in edge_rows[row]:
            if anchor in seen:
                continue
            seen.add(anchor)
            previous = anchor_to_row.get(anchor)
            if previous is None or augment(previous, seen):
                anchor_to_row[anchor] = row
                return True
        return False

    for row in row_order:
        augment(row, set())
    pairs = sorted((row, anchor) for anchor, row in anchor_to_row.items())
    if not pairs:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return (
        torch.tensor([row for row, _ in pairs], dtype=torch.long),
        torch.tensor([anchor for _, anchor in pairs], dtype=torch.long),
    )


def competitive_reserve_state(
    *,
    candidate_anchor_rows: torch.Tensor,
    candidate_scores: torch.Tensor,
    certified_positive: torch.Tensor,
    active_anchor_mask: torch.Tensor,
    keypoints: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
    image_hw: torch.Tensor,
    margin_delta: float = 0.005,
    image_grid: tuple[int, int] = (4, 4),
    pose_damping: float = 1e-6,
) -> dict:
    """Compute deployed winners and the certified novel-view reserve state."""

    candidates = torch.as_tensor(candidate_anchor_rows).long()
    scores = torch.as_tensor(candidate_scores).float()
    positive = torch.as_tensor(certified_positive).bool()
    active_mask = torch.as_tensor(active_anchor_mask).bool().reshape(-1)
    if candidates.shape != scores.shape or candidates.shape != positive.shape:
        raise ValueError("Top-L rows, scores, and positive labels must align")
    active = active_mask[candidates]
    available = active.any(1)
    masked_scores = scores.masked_fill(~active, -torch.inf)
    winner_column = masked_scores.argmax(1)
    row_index = torch.arange(candidates.shape[0])
    winner_anchor = candidates[row_index, winner_column]
    winner_score = scores[row_index, winner_column]
    winner_positive = positive[row_index, winner_column] & available

    noncertified = active & ~positive
    best_noncertified = scores.masked_fill(~noncertified, -torch.inf).max(1).values
    safe = (
        active
        & positive
        & (scores >= best_noncertified[:, None] + float(margin_delta))
    )
    safe_count = safe.sum(1)
    matched_rows, matched_anchors = _maximum_anchor_unique_matching(
        safe, candidates, scores
    )
    height, width = map(int, torch.as_tensor(image_hw).tolist())
    grid_h, grid_w = map(int, image_grid)
    matched_keypoints = torch.as_tensor(keypoints).float()[matched_rows]
    if matched_rows.numel():
        cell_x = (matched_keypoints[:, 0] * grid_w / max(width, 1)).long().clamp(0, grid_w - 1)
        cell_y = (matched_keypoints[:, 1] * grid_h / max(height, 1)).long().clamp(0, grid_h - 1)
        spatial_cells = int(torch.unique(cell_y * grid_w + cell_x).numel())
        information = compute_pose_information(
            torch.as_tensor(anchor_xyz).double()[matched_anchors],
            torch.as_tensor(intrinsic).double(),
            torch.as_tensor(pose_w2c).double(),
            damping=float(pose_damping),
        )
        eigenvalues = torch.linalg.eigvalsh(information.matrix)
        pose_logdet = float(information.logdet)
        pose_minimum_eigenvalue = float(eigenvalues[0])
    else:
        spatial_cells = 0
        pose_logdet = float(6.0 * torch.log(torch.tensor(float(pose_damping))))
        pose_minimum_eigenvalue = float(pose_damping)
    return {
        "topl_exhausted": ~available,
        "winner_anchor_rows": winner_anchor,
        "winner_scores": winner_score,
        "winner_certified_positive": winner_positive,
        "best_noncertified_scores": best_noncertified,
        "safe_positive_count_per_row": safe_count,
        "safe_edge_mask": safe,
        "anchor_unique_safe_query_rows": matched_rows,
        "anchor_unique_safe_anchor_rows": matched_anchors,
        "anchor_unique_safe_count": int(matched_rows.numel()),
        "spatial_cell_count": spatial_cells,
        "pose_logdet": pose_logdet,
        "pose_minimum_eigenvalue": pose_minimum_eigenvalue,
        "margin_delta": float(margin_delta),
    }


def reserve_transition_is_safe(
    before: Mapping,
    after: Mapping,
    *,
    pose_relative_tolerance: float = 1e-6,
) -> tuple[bool, list[str]]:
    """Require a deletion not to consume any current novel-view reserve."""

    reasons = []
    if bool(torch.as_tensor(after["topl_exhausted"]).any()):
        reasons.append("topl_exhausted")
    before_correct = torch.as_tensor(before["winner_certified_positive"]).bool()
    after_correct = torch.as_tensor(after["winner_certified_positive"]).bool()
    if bool((before_correct & ~after_correct).any()):
        reasons.append("correct_winner_lost")
    if int(after["anchor_unique_safe_count"]) < int(before["anchor_unique_safe_count"]):
        reasons.append("anchor_unique_reserve_decreased")
    if int(after["spatial_cell_count"]) < int(before["spatial_cell_count"]):
        reasons.append("spatial_reserve_decreased")
    tolerance = float(pose_relative_tolerance)
    if float(after["pose_logdet"]) + tolerance < float(before["pose_logdet"]):
        reasons.append("pose_logdet_decreased")
    minimum_floor = float(before["pose_minimum_eigenvalue"]) * (1.0 - tolerance)
    if float(after["pose_minimum_eigenvalue"]) < minimum_floor:
        reasons.append("pose_minimum_eigenvalue_decreased")
    return not reasons, reasons
