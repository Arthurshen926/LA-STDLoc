"""Self-localization residual evidence for pose-sufficient set selection.

The statistics are built only from mapping-view self-localization.  At
deployment they provide a pose-free lookup keyed by the matched anchor and a
coarse image cell.  Raw sufficient statistics are retained so mapping-query
training can subtract the current query and avoid target leakage.
"""

from __future__ import annotations

import torch

from localization_training.pose_sufficient_selector import image_grid_cells


RESIDUAL_SIGNATURE_FEATURE_NAMES = (
    "residual_signature_x",
    "residual_signature_y",
    "residual_signature_norm",
    "residual_signature_std",
    "residual_signature_consistency",
    "residual_signature_solver_rate",
    "residual_signature_strict_rate",
    "residual_signature_log_support",
)

_STATISTIC_NAMES = (
    "attempts",
    "soft_weight",
    "weighted_sum",
    "weighted_square_norm",
    "strict_count",
    "solver_count",
)


def signed_reprojection_residual(
    xyz: torch.Tensor,
    keypoints: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return projected-minus-observed pixel residual and positive depth."""

    xyz = torch.as_tensor(xyz).double().reshape(-1, 3)
    keypoints = torch.as_tensor(keypoints).double().reshape(-1, 2)
    K = torch.as_tensor(K).double().reshape(3, 3)
    pose = torch.as_tensor(pose_w2c).double().reshape(4, 4)
    if len(xyz) != len(keypoints):
        raise ValueError("residual xyz and keypoints must align")
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = torch.empty_like(keypoints)
    projected[:, 0] = K[0, 0] * camera[:, 0] / depth.clamp_min(1e-8) + K[0, 2]
    projected[:, 1] = K[1, 1] * camera[:, 1] / depth.clamp_min(1e-8) + K[1, 2]
    residual = projected - keypoints
    valid = (depth > 1e-8) & torch.isfinite(residual).all(dim=1)
    residual[~valid] = 0.0
    return residual.float(), valid


def empty_residual_statistics(
    anchor_count: int,
    *,
    grid_size: int = 4,
) -> dict[str, torch.Tensor]:
    cells = max(int(grid_size), 1) ** 2
    shape = (int(anchor_count), cells)
    return {
        "attempts": torch.zeros(shape),
        "soft_weight": torch.zeros(shape),
        "weighted_sum": torch.zeros((*shape, 2)),
        "weighted_square_norm": torch.zeros(shape),
        "strict_count": torch.zeros(shape),
        "solver_count": torch.zeros(shape),
    }


def residual_statistics_contribution(
    *,
    anchor_indices: torch.Tensor,
    keypoints: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    signed_residual: torch.Tensor,
    valid: torch.Tensor,
    anchor_count: int,
    grid_size: int = 4,
    clip_px: float = 12.0,
    strict_px: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Accumulate one query into anchor-by-image-cell sufficient statistics."""

    anchors = torch.as_tensor(anchor_indices).long().reshape(-1)
    keypoints = torch.as_tensor(keypoints).float().reshape(-1, 2)
    residual = torch.as_tensor(signed_residual).float().reshape(-1, 2)
    valid = torch.as_tensor(valid).bool().reshape(-1)
    if not all(len(value) == len(anchors) for value in (keypoints, residual, valid)):
        raise ValueError("residual contribution rows must align")
    if len(anchors) and (int(anchors.min()) < 0 or int(anchors.max()) >= anchor_count):
        raise ValueError("residual contribution anchor is out of bounds")
    grid_size = max(int(grid_size), 1)
    cells = image_grid_cells(
        keypoints, image_hw, rows=grid_size, cols=grid_size
    )
    flat = anchors * (grid_size * grid_size) + cells
    count = int(anchor_count) * grid_size * grid_size
    norm = torch.linalg.norm(residual, dim=1)
    finite = valid & torch.isfinite(norm)
    clip = max(float(clip_px), 1e-6)
    scale = torch.minimum(
        torch.ones_like(norm), clip / norm.clamp_min(1e-8)
    )
    clipped = residual * scale[:, None]
    clipped[~finite] = 0.0
    soft_weight = torch.exp(-0.5 * (norm / clip).square())
    soft_weight[~finite] = 0.0
    attempts = finite.float()
    strict = (finite & (norm <= float(strict_px))).float()
    solver = (finite & (norm <= clip)).float()

    def fold(values: torch.Tensor) -> torch.Tensor:
        tail = tuple(values.shape[1:])
        output = values.new_zeros((count, *tail))
        output.index_add_(0, flat, values)
        return output.reshape(int(anchor_count), grid_size * grid_size, *tail)

    return {
        "attempts": fold(attempts),
        "soft_weight": fold(soft_weight),
        "weighted_sum": fold(soft_weight[:, None] * clipped),
        "weighted_square_norm": fold(
            soft_weight * clipped.square().sum(dim=1)
        ),
        "strict_count": fold(strict),
        "solver_count": fold(solver),
    }


def add_residual_statistics(
    target: dict[str, torch.Tensor],
    contribution: dict[str, torch.Tensor],
) -> None:
    for name in _STATISTIC_NAMES:
        if name not in target or target[name].shape != contribution[name].shape:
            raise ValueError(f"residual statistic {name} does not align")
        target[name].add_(contribution[name])


def subtract_residual_statistics(
    total: dict[str, torch.Tensor],
    contribution: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output = {}
    for name in _STATISTIC_NAMES:
        if name not in total or total[name].shape != contribution[name].shape:
            raise ValueError(f"residual statistic {name} does not align")
        value = torch.as_tensor(total[name]).float() - torch.as_tensor(
            contribution[name]
        ).float()
        if name not in {"weighted_sum"}:
            value = value.clamp_min(0.0)
        output[name] = value
    return output


def residual_signature_features(
    statistics: dict[str, torch.Tensor],
    *,
    anchor_indices: torch.Tensor,
    keypoints: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    grid_size: int = 4,
    clip_px: float = 12.0,
    anchor_prior: float = 4.0,
    cell_prior: float = 8.0,
    rate_prior: float = 12.0,
) -> torch.Tensor:
    """Lookup shrinkage-stabilized signed residual evidence for query rows."""

    anchors = torch.as_tensor(anchor_indices).long().reshape(-1)
    keypoints = torch.as_tensor(keypoints).float().reshape(-1, 2)
    if len(anchors) != len(keypoints):
        raise ValueError("residual signature lookup rows must align")
    grid_size = max(int(grid_size), 1)
    expected_cells = grid_size * grid_size
    for name in _STATISTIC_NAMES:
        value = torch.as_tensor(statistics[name]).float()
        if value.shape[:2] != (
            torch.as_tensor(statistics["attempts"]).shape[0],
            expected_cells,
        ):
            raise ValueError(f"residual statistic {name} has an invalid grid")
    anchor_count = int(torch.as_tensor(statistics["attempts"]).shape[0])
    if len(anchors) and (int(anchors.min()) < 0 or int(anchors.max()) >= anchor_count):
        raise ValueError("residual signature anchor is out of bounds")
    cells = image_grid_cells(
        keypoints, image_hw, rows=grid_size, cols=grid_size
    )
    attempts = torch.as_tensor(statistics["attempts"]).float()
    weight = torch.as_tensor(statistics["soft_weight"]).float()
    vector_sum = torch.as_tensor(statistics["weighted_sum"]).float()
    square_sum = torch.as_tensor(
        statistics["weighted_square_norm"]
    ).float()
    strict_count = torch.as_tensor(statistics["strict_count"]).float()
    solver_count = torch.as_tensor(statistics["solver_count"]).float()

    # Aggregate only anchors present in this query. Recomputing the complete
    # map for every deployment frame dominates CPU time on compact queries.
    anchor_weight = weight[anchors].sum(dim=1)
    anchor_vector_sum = vector_sum[anchors].sum(dim=1)
    anchor_square_sum = square_sum[anchors].sum(dim=1)
    anchor_denom = anchor_weight + max(float(anchor_prior), 0.0)
    anchor_mean = anchor_vector_sum / anchor_denom.clamp_min(1e-6)[:, None]
    anchor_second = anchor_square_sum / anchor_denom.clamp_min(1e-6)

    row_weight = weight[anchors, cells]
    row_sum = vector_sum[anchors, cells]
    row_square = square_sum[anchors, cells]
    prior = max(float(cell_prior), 0.0)
    denominator = row_weight + prior
    mean = (row_sum + prior * anchor_mean) / denominator.clamp_min(
        1e-6
    )[:, None]
    second = (
        row_square + prior * anchor_second
    ) / denominator.clamp_min(1e-6)
    mean_norm = torch.linalg.norm(mean, dim=1)
    standard_deviation = (
        second - mean_norm.square()
    ).clamp_min(0.0).sqrt()
    consistency = mean_norm / second.clamp_min(1e-8).sqrt()

    global_attempts = attempts.sum().clamp_min(1.0)
    global_solver = solver_count.sum() / global_attempts
    global_strict = strict_count.sum() / global_attempts
    row_attempts = attempts[anchors, cells]
    rate_prior = max(float(rate_prior), 0.0)
    solver_rate = (
        solver_count[anchors, cells] + rate_prior * global_solver
    ) / (row_attempts + rate_prior).clamp_min(1e-6)
    strict_rate = (
        strict_count[anchors, cells] + rate_prior * global_strict
    ) / (row_attempts + rate_prior).clamp_min(1e-6)
    clip = max(float(clip_px), 1e-6)
    return torch.stack(
        (
            mean[:, 0] / clip,
            mean[:, 1] / clip,
            mean_norm / clip,
            standard_deviation / clip,
            consistency.clamp(0.0, 1.0),
            solver_rate.clamp(0.0, 1.0),
            strict_rate.clamp(0.0, 1.0),
            torch.log1p(row_weight) / 4.0,
        ),
        dim=1,
    )


def normalized_bias_target(
    signed_residual: torch.Tensor,
    valid: torch.Tensor,
    *,
    clip_px: float = 12.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded direction targets and solver-relevance weights."""

    residual = torch.as_tensor(signed_residual).float().reshape(-1, 2)
    valid = torch.as_tensor(valid).bool().reshape(-1)
    norm = torch.linalg.norm(residual, dim=1)
    clip = max(float(clip_px), 1e-6)
    scale = torch.minimum(
        torch.ones_like(norm), clip / norm.clamp_min(1e-8)
    )
    target = residual * scale[:, None] / clip
    weight = torch.exp(-0.5 * (norm / clip).square())
    weight[~valid] = 0.0
    target[~valid] = 0.0
    return target, weight
