"""No-LOO Top-K causal evidence for ranking and delete-only map feedback."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.v8_feedback_controller import task_error


def require_no_loo_feedback_contract(payload: Mapping) -> None:
    """Fail closed if feedback has mapping membership or any LOO provenance."""

    forbidden = (
        "enters_track_registry",
        "enters_anchor_observation_csr",
        "enters_descriptor_bank",
        "loo_used",
        "query_geometry_loo",
        "query_descriptor_loo",
    )
    enabled = [name for name in forbidden if payload.get(name) is True]
    if enabled:
        raise ValueError(f"V9 feedback violates no-LOO/non-mapping contract: {enabled}")


def topk_geometric_correctness(
    *,
    keypoints: torch.Tensor,
    candidate_anchor_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    row_valid: torch.Tensor,
    reprojection_px: float = 4.0,
    minimum_alpha: float = 0.05,
    depth_absolute_m: float = 0.5,
    depth_relative: float = 0.10,
) -> torch.Tensor:
    """Label Top-K candidates by independent GT geometry and V2 row support."""

    xy = torch.as_tensor(keypoints).float().reshape(-1, 2)
    candidates = torch.as_tensor(candidate_anchor_rows).long()
    xyz = torch.as_tensor(anchor_xyz).float()
    pose = torch.as_tensor(pose_w2c).float()
    calibration = torch.as_tensor(intrinsic).float()
    valid_rows = torch.as_tensor(row_valid).bool().reshape(-1)
    if candidates.ndim != 2 or candidates.shape[0] != xy.shape[0]:
        raise ValueError("Top-K candidate rows do not align with query keypoints")
    if valid_rows.shape[0] != xy.shape[0]:
        raise ValueError("V2 row mask does not align with query keypoints")
    if candidates.numel() and (
        int(candidates.min()) < 0 or int(candidates.max()) >= xyz.shape[0]
    ):
        raise ValueError("Top-K candidate is outside the fixed map")
    height, width = map(int, torch.as_tensor(depth).squeeze().shape)
    pixel = xy.round().long()
    inside = (
        (pixel[:, 0] >= 0)
        & (pixel[:, 0] < width)
        & (pixel[:, 1] >= 0)
        & (pixel[:, 1] < height)
    )
    safe = pixel.clone()
    safe[:, 0].clamp_(0, max(width - 1, 0))
    safe[:, 1].clamp_(0, max(height - 1, 0))
    alpha_raster = torch.as_tensor(alpha).float().squeeze()
    depth_raster = torch.as_tensor(depth).float().squeeze()
    observed_alpha = alpha_raster[safe[:, 1], safe[:, 0]]
    observed_depth = depth_raster[safe[:, 1], safe[:, 0]]

    world = xyz[candidates]
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    projected = camera @ calibration.T
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    reprojection = torch.linalg.norm(uv - xy[:, None], dim=2)
    depth_tolerance = torch.maximum(
        torch.full_like(observed_depth, float(depth_absolute_m)),
        observed_depth.abs() * float(depth_relative),
    )
    supported_row = (
        valid_rows
        & inside
        & torch.isfinite(observed_alpha)
        & torch.isfinite(observed_depth)
        & (observed_alpha >= float(minimum_alpha))
        & (observed_depth > 0)
    )
    return (
        supported_row[:, None]
        & torch.isfinite(uv).all(2)
        & (camera[..., 2] > 0)
        & (reprojection <= float(reprojection_px))
        & ((camera[..., 2] - observed_depth[:, None]).abs() <= depth_tolerance[:, None])
    )


def first_correct_topk_replacement(
    candidate_anchor_rows: torch.Tensor,
    candidate_scores: torch.Tensor,
    correct: torch.Tensor,
) -> dict:
    """Return the first correct candidate and causal wrong→right training rows."""

    candidates = torch.as_tensor(candidate_anchor_rows).long()
    scores = torch.as_tensor(candidate_scores).float()
    correctness = torch.as_tensor(correct).bool()
    if candidates.shape != scores.shape or candidates.shape != correctness.shape:
        raise ValueError("Top-K candidates, scores, and labels must align")
    has_correct = correctness.any(1)
    first_rank = torch.where(
        has_correct,
        correctness.float().argmax(1),
        torch.full((candidates.shape[0],), -1, dtype=torch.long),
    ).long()
    rows = torch.nonzero(has_correct, as_tuple=False).reshape(-1)
    positive = candidates[rows, first_rank[rows]]
    negative = candidates[rows, 0]
    changed = positive != negative
    rows = rows[changed]
    return {
        "supported_query_rows": torch.nonzero(has_correct, as_tuple=False).reshape(-1),
        "changed_query_rows": rows,
        "positive_anchor_rows": candidates[rows, first_rank[rows]],
        "negative_anchor_rows": candidates[rows, 0],
        "positive_rank": first_rank[rows],
        "positive_scores": scores[rows, first_rank[rows]],
        "negative_scores": scores[rows, 0],
    }


def anchor_unique_spatial_correspondences(
    *,
    keypoints: torch.Tensor,
    anchor_rows: torch.Tensor,
    scores: torch.Tensor,
    image_hw: torch.Tensor,
    grid_shape: tuple[int, int] = (4, 4),
) -> dict:
    """Keep one highest-score row per Anchor and report spatial support."""

    xy = torch.as_tensor(keypoints).float().reshape(-1, 2)
    anchors = torch.as_tensor(anchor_rows).long().reshape(-1)
    values = torch.as_tensor(scores).float().reshape(-1)
    if not (xy.shape[0] == anchors.numel() == values.numel()):
        raise ValueError("correspondence rows do not align")
    order = torch.argsort(values, descending=True, stable=True)
    retained = []
    seen = set()
    for row in order.tolist():
        anchor = int(anchors[row])
        if anchor not in seen:
            retained.append(row)
            seen.add(anchor)
    selected = torch.tensor(sorted(retained), dtype=torch.long)
    height, width = map(int, torch.as_tensor(image_hw).tolist())
    grid_h, grid_w = map(int, grid_shape)
    pixels = xy[selected]
    cell_x = (pixels[:, 0] * grid_w / max(width, 1)).long().clamp(0, grid_w - 1)
    cell_y = (pixels[:, 1] * grid_h / max(height, 1)).long().clamp(0, grid_h - 1)
    return {
        "selected_rows": selected,
        "anchor_rows": anchors[selected],
        "spatial_cell_count": int(torch.unique(cell_y * grid_w + cell_x).numel()),
    }


def standard_pose_replay(
    *,
    keypoints: torch.Tensor,
    anchor_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
    seed: int = 2026,
    reprojection_error_px: float = 11.954343111400277,
) -> dict:
    """Run exactly one standard PoseLib solve and return frozen task error."""

    xy = torch.as_tensor(keypoints).float().cpu().numpy()
    rows = torch.as_tensor(anchor_rows).long()
    xyz = torch.as_tensor(anchor_xyz).float()[rows].cpu().numpy()
    estimate = solve_absolute_pose(
        xy,
        xyz,
        torch.as_tensor(intrinsic).float().cpu().numpy(),
        reprojection_error_px=float(reprojection_error_px),
        seed=int(seed),
    )
    rotation_deg, translation_cm = pose_error(
        estimate.pose_w2c, torch.as_tensor(ground_truth_w2c).cpu().numpy()
    )
    return {
        "pose_w2c": torch.from_numpy(estimate.pose_w2c),
        "inlier_count": int(estimate.inliers.size),
        "rotation_error_deg": rotation_deg,
        "translation_error_cm": translation_cm,
        "task_error": task_error(translation_cm, rotation_deg),
    }


def aggregate_actual_removal_gain(
    records: Sequence[Mapping],
    *,
    minimum_pose_families: int = 2,
    minimum_improving_queries: int = 2,
    minimum_median_gain: float = 0.0,
    maximum_worsening_fraction: float = 0.25,
) -> dict:
    """Authorize deletion only from paired standard-PoseLib removal replays."""

    grouped: dict[int, list[Mapping]] = {}
    for record in records:
        if record.get("loo_used") is True:
            raise ValueError("LOO evidence is forbidden in V9")
        grouped.setdefault(int(record["anchor_row"]), []).append(record)
    audit = []
    for anchor, rows in grouped.items():
        gains = torch.tensor(
            [float(row["baseline_task_error"]) - float(row["removed_task_error"]) for row in rows]
        )
        families = {int(row["pose_family_id"]) for row in rows}
        improving = gains > 0
        worsening_fraction = float((gains < 0).float().mean())
        accepted = (
            len(families) >= int(minimum_pose_families)
            and int(improving.sum()) >= int(minimum_improving_queries)
            and float(gains.median()) > float(minimum_median_gain)
            and float(gains.sum()) > 0.0
            and worsening_fraction <= float(maximum_worsening_fraction)
        )
        audit.append(
            {
                "anchor_row": anchor,
                "query_count": len(rows),
                "pose_family_count": len(families),
                "median_actual_task_gain": float(gains.median()),
                "cumulative_actual_task_gain": float(gains.sum()),
                "worsening_fraction": worsening_fraction,
                "authorized": accepted,
            }
        )
    audit.sort(key=lambda row: (-row["cumulative_actual_task_gain"], row["anchor_row"]))
    authorized = torch.tensor(
        [row["anchor_row"] for row in audit if row["authorized"]],
        dtype=torch.long,
    )
    return {
        "schema": "lafgs_v9_actual_removal_gain_audit",
        "version": 1,
        "loo_used": False,
        "authorized_anchor_rows": authorized,
        "authorized_anchor_count": int(authorized.numel()),
        "candidate_audit": audit,
    }
