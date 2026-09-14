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
    query_keypoint_variance_px2: torch.Tensor | float | None = None,
    surface_depth_std_m: torch.Tensor | float | None = None,
    calibration_variance_px2: float = 0.25,
    strict_reprojection_chi2: float = 9.210340371976184,
    broad_reprojection_chi2: float = 13.815510557964274,
    strict_depth_chi2: float = 6.6348966010212145,
    broad_depth_chi2: float = 10.827566170662733,
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

    query_variance = torch.as_tensor(
        1.0 if query_keypoint_variance_px2 is None else query_keypoint_variance_px2,
        dtype=pixel_covariance.dtype,
        device=pixel_covariance.device,
    )
    if query_variance.ndim == 0:
        query_variance = query_variance.expand(xy.shape[0])
    query_variance = query_variance.reshape(-1)
    if query_variance.numel() != xy.shape[0]:
        raise ValueError("query keypoint variance does not align with query rows")
    identity2 = torch.eye(2, dtype=pixel_covariance.dtype, device=pixel_covariance.device)
    measurement_covariance = pixel_covariance + (
        query_variance[:, None, None, None] + float(calibration_variance_px2)
    ) * identity2
    inverse_measurement_covariance = torch.linalg.pinv(
        measurement_covariance + 1e-8 * identity2
    )
    residual = uv - xy[:, None]
    reprojection_mahalanobis_squared = torch.einsum(
        "rli,rlij,rlj->rl",
        residual,
        inverse_measurement_covariance,
        residual,
    )

    strict_depth = torch.maximum(
        torch.full_like(observed_depth, float(strict_depth_absolute_m)),
        observed_depth.abs() * float(strict_depth_relative),
    )
    broad_depth = torch.maximum(
        torch.full_like(observed_depth, float(broad_depth_absolute_m)),
        observed_depth.abs() * float(broad_depth_relative),
    )
    surface_stable = torch.ones_like(valid_rows)
    inferred_surface_std = torch.maximum(
        torch.full_like(observed_depth, 0.025),
        observed_depth.abs() * 0.005,
    )
    if surface_median_depth is not None:
        surface = _raster_values(surface_median_depth, safe_pixels)
        inferred_surface_std = torch.maximum(
            inferred_surface_std,
            (surface - observed_depth).abs(),
        )
        surface_stable = (
            torch.isfinite(surface)
            & (surface > 0)
            & ((surface - observed_depth).abs() <= strict_depth)
        )
    if surface_depth_std_m is None:
        surface_std = inferred_surface_std
    else:
        surface_std = torch.as_tensor(
            surface_depth_std_m,
            dtype=observed_depth.dtype,
            device=observed_depth.device,
        )
        if surface_std.ndim == 0:
            surface_std = surface_std.expand(xy.shape[0])
        surface_std = surface_std.reshape(-1)
        if surface_std.numel() != xy.shape[0]:
            raise ValueError("surface depth uncertainty does not align with query rows")
        surface_std = torch.maximum(surface_std, inferred_surface_std)
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
    combined_depth_variance = depth_std.square() + surface_std[:, None].square()
    depth_mahalanobis_squared = depth_error.square() / combined_depth_variance.clamp_min(
        1e-8
    )
    broad = (
        supported_row[:, None]
        & finite_edge
        & (reprojection <= float(broad_reprojection_px))
        & (reprojection_mahalanobis_squared <= float(broad_reprojection_chi2))
        & (depth_error <= broad_depth[:, None] + 3.0 * depth_std)
        & (depth_mahalanobis_squared <= float(broad_depth_chi2))
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
        & (reprojection_mahalanobis_squared <= float(strict_reprojection_chi2))
        & (projection_std <= float(maximum_projection_std_px))
        & (depth_std <= strict_depth[:, None])
        & (depth_error <= strict_depth[:, None] + 2.0 * depth_std)
        & (depth_mahalanobis_squared <= float(strict_depth_chi2))
    )
    ambiguous = broad & ~positive
    negative = supported_row[:, None] & finite_edge & ~broad
    invalid = ~(positive | ambiguous | negative)
    certification_confidence = torch.exp(
        -0.5
        * (
            reprojection_mahalanobis_squared / float(strict_reprojection_chi2)
            + depth_mahalanobis_squared / float(strict_depth_chi2)
        )
    ).clamp(0.0, 1.0)
    certification_confidence = certification_confidence.masked_fill(~positive, 0.0)
    return {
        "positive": positive,
        "ambiguous": ambiguous,
        "negative": negative,
        "invalid": invalid,
        "reprojection_error_px": reprojection,
        "projection_std_px": projection_std,
        "depth_error_m": depth_error,
        "depth_std_m": depth_std,
        "measurement_covariance_px2": measurement_covariance,
        "reprojection_mahalanobis_squared": reprojection_mahalanobis_squared,
        "depth_mahalanobis_squared": depth_mahalanobis_squared,
        "surface_depth_std_m": surface_std,
        "certification_confidence": certification_confidence,
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
    certification_confidence: torch.Tensor | None = None,
    measurement_covariance_px2: torch.Tensor | None = None,
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
        matched_columns = []
        for row, anchor in zip(matched_rows.tolist(), matched_anchors.tolist()):
            columns = torch.nonzero(
                safe[row] & (candidates[row] == int(anchor)), as_tuple=False
            ).reshape(-1)
            if columns.numel() == 0:
                raise RuntimeError("Anchor-unique matching edge disappeared")
            column = columns[scores[row, columns].argmax()]
            matched_columns.append(int(column))
        matched_columns = torch.tensor(matched_columns, dtype=torch.long)
        if certification_confidence is None:
            matched_confidence = torch.ones(matched_rows.numel(), dtype=torch.double)
        else:
            confidence = torch.as_tensor(certification_confidence).float()
            if confidence.shape != candidates.shape:
                raise ValueError("certification confidence does not align with Top-L")
            matched_confidence = confidence[
                matched_rows, matched_columns
            ].double().clamp(0.0, 1.0)
        matched_covariance = None
        if measurement_covariance_px2 is not None:
            covariance = torch.as_tensor(measurement_covariance_px2).float()
            if covariance.shape != (*candidates.shape, 2, 2):
                raise ValueError("measurement covariance does not align with Top-L")
            matched_covariance = covariance[
                matched_rows, matched_columns
            ].double()
        cell_x = (matched_keypoints[:, 0] * grid_w / max(width, 1)).long().clamp(0, grid_w - 1)
        cell_y = (matched_keypoints[:, 1] * grid_h / max(height, 1)).long().clamp(0, grid_h - 1)
        spatial_cells = int(torch.unique(cell_y * grid_w + cell_x).numel())
        information = compute_pose_information(
            torch.as_tensor(anchor_xyz).double()[matched_anchors],
            torch.as_tensor(intrinsic).double(),
            torch.as_tensor(pose_w2c).double(),
            weights=matched_confidence,
            measurement_covariance=matched_covariance,
            damping=float(pose_damping),
        )
        eigenvalues = torch.linalg.eigvalsh(information.matrix)
        pose_logdet = float(information.logdet)
        pose_minimum_eigenvalue = float(eigenvalues[0])
        pose_effective_correspondence_count = float(information.effective_count)
    else:
        spatial_cells = 0
        pose_logdet = float(6.0 * torch.log(torch.tensor(float(pose_damping))))
        pose_minimum_eigenvalue = float(pose_damping)
        pose_effective_correspondence_count = 0.0
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
        "pose_effective_correspondence_count": pose_effective_correspondence_count,
        "margin_delta": float(margin_delta),
    }


def reserve_transition_is_safe(
    before: Mapping,
    after: Mapping,
    *,
    pose_relative_tolerance: float = 1e-6,
    minimum_anchor_unique_safe_count: int | None = None,
    minimum_spatial_cell_count: int | None = None,
    maximum_pose_logdet_drop: float | None = None,
    minimum_pose_eigenvalue_retention: float | None = None,
    minimum_effective_correspondence_count: float | None = None,
) -> tuple[bool, list[str]]:
    """Require a deletion to remain above registered novel-view safety floors.

    Omitting every floor retains the historical zero-reserve-loss contract for
    compatibility.  The V18 controller supplies explicit floors and may then
    consume redundant reserve without violating PnP/coverage safety.
    """

    reasons = []
    before_exhausted = torch.as_tensor(before["topl_exhausted"]).bool()
    after_exhausted = torch.as_tensor(after["topl_exhausted"]).bool()
    if before_exhausted.shape != after_exhausted.shape:
        raise ValueError("Top-L exhaustion state does not align across transition")
    if bool((after_exhausted & ~before_exhausted).any()):
        reasons.append("new_topl_exhaustion")
    before_correct = torch.as_tensor(before["winner_certified_positive"]).bool()
    after_correct = torch.as_tensor(after["winner_certified_positive"]).bool()
    if bool((before_correct & ~after_correct).any()):
        reasons.append("correct_winner_lost")
    count_floor = (
        int(before["anchor_unique_safe_count"])
        if minimum_anchor_unique_safe_count is None
        else min(
            int(before["anchor_unique_safe_count"]),
            int(minimum_anchor_unique_safe_count),
        )
    )
    if int(after["anchor_unique_safe_count"]) < count_floor:
        reasons.append("anchor_unique_reserve_below_floor")
    cell_floor = (
        int(before["spatial_cell_count"])
        if minimum_spatial_cell_count is None
        else min(int(before["spatial_cell_count"]), int(minimum_spatial_cell_count))
    )
    if int(after["spatial_cell_count"]) < cell_floor:
        reasons.append("spatial_reserve_below_floor")
    tolerance = float(pose_relative_tolerance)
    logdet_floor = (
        float(before["pose_logdet"])
        if maximum_pose_logdet_drop is None
        else float(before["pose_logdet"]) - max(float(maximum_pose_logdet_drop), 0.0)
    )
    if float(after["pose_logdet"]) + tolerance < logdet_floor:
        reasons.append("pose_logdet_below_floor")
    eigenvalue_retention = (
        1.0 - tolerance
        if minimum_pose_eigenvalue_retention is None
        else float(minimum_pose_eigenvalue_retention)
    )
    if not 0.0 <= eigenvalue_retention <= 1.0:
        raise ValueError("pose eigenvalue retention must lie in [0, 1]")
    minimum_floor = float(before["pose_minimum_eigenvalue"]) * eigenvalue_retention
    if float(after["pose_minimum_eigenvalue"]) < minimum_floor:
        reasons.append("pose_minimum_eigenvalue_below_floor")
    effective_floor = (
        float(before.get("pose_effective_correspondence_count", 0.0))
        if minimum_effective_correspondence_count is None
        else min(
            float(before.get("pose_effective_correspondence_count", 0.0)),
            float(minimum_effective_correspondence_count),
        )
    )
    if float(after.get("pose_effective_correspondence_count", 0.0)) + tolerance < effective_floor:
        reasons.append("effective_correspondence_count_below_floor")
    return not reasons, reasons
