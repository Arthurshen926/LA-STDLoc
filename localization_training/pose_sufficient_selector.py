"""Constrained one-pass correspondence selection for sparse PnP."""

from __future__ import annotations

from collections import Counter
from math import ceil

import torch


FEATURE_NAMES = (
    "top1_score",
    "top1_margin",
    "top16_entropy",
    "query_score_z",
    "query_margin_z",
    "keypoint_score",
    "keypoint_x",
    "keypoint_y",
    "keypoint_radius",
    "anchor_log_attempts",
    "anchor_clean_rate",
    "anchor_clean_inlier_rate",
    "anchor_harmful_inlier_rate",
    "query_anchor_multiplicity",
    "query_dependency_multiplicity",
    "query_source_multiplicity",
)


def _multiplicity(values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values).long().reshape(-1)
    _, inverse, counts = torch.unique(
        values, sorted=False, return_inverse=True, return_counts=True
    )
    return counts[inverse].float()


def image_grid_cells(
    keypoints: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    *,
    rows: int = 4,
    cols: int = 4,
) -> torch.Tensor:
    """Assign sparse keypoints to a fixed image grid."""

    keypoints = torch.as_tensor(keypoints).float().reshape(-1, 2)
    height, width = map(int, image_hw)
    row = (
        keypoints[:, 1] * max(int(rows), 1) / max(float(height), 1.0)
    ).floor().long().clamp(0, max(int(rows), 1) - 1)
    col = (
        keypoints[:, 0] * max(int(cols), 1) / max(float(width), 1.0)
    ).floor().long().clamp(0, max(int(cols), 1) - 1)
    return row * max(int(cols), 1) + col


def build_pose_sufficient_features(
    topk_scores: torch.Tensor,
    topk_anchor_indices: torch.Tensor,
    *,
    keypoints: torch.Tensor,
    keypoint_scores: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    source_groups: torch.Tensor,
    dependency_groups: torch.Tensor,
    anchor_statistics: dict[str, torch.Tensor],
    entropy_temperature: float = 0.05,
    prior_strength: float = 12.0,
) -> torch.Tensor:
    """Build the exact train/deploy feature contract for set selection."""

    scores = torch.as_tensor(topk_scores).float()
    indices = torch.as_tensor(topk_anchor_indices).long()
    keypoints = torch.as_tensor(keypoints).float().reshape(-1, 2)
    keypoint_scores = torch.as_tensor(keypoint_scores).float().reshape(-1)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("pose-sufficient features require at least top-2 scores")
    if indices.shape != scores.shape:
        raise ValueError("top-K scores and anchor indices must align")
    if len(keypoints) != len(scores) or len(keypoint_scores) != len(scores):
        raise ValueError("keypoint inputs must align with top-K rows")
    anchors = indices[:, 0]
    anchor_count = len(torch.as_tensor(source_groups).reshape(-1))
    anchors_in_bounds = not len(anchors) or (
        int(anchors.min()) >= 0 and int(anchors.max()) < anchor_count
    )
    if (
        len(torch.as_tensor(dependency_groups).reshape(-1)) != anchor_count
        or not anchors_in_bounds
    ):
        raise ValueError("pose-sufficient anchor groups do not align with retrieval")
    required_statistics = (
        "attempts",
        "clean",
        "clean_inlier",
        "harmful_inlier",
    )
    folded = {}
    for name in required_statistics:
        if name not in anchor_statistics:
            raise ValueError(f"pose-sufficient statistics miss {name}")
        value = torch.as_tensor(anchor_statistics[name]).float().reshape(-1)
        if len(value) != anchor_count:
            raise ValueError(f"pose-sufficient statistic {name} does not align")
        folded[name] = value

    margin = scores[:, 0] - scores[:, 1]
    distribution = torch.softmax(
        (scores - scores[:, :1]) / max(float(entropy_temperature), 1e-6),
        dim=1,
    )
    entropy = -(
        distribution * distribution.clamp_min(1e-12).log()
    ).sum(dim=1) / torch.log(scores.new_tensor(float(scores.shape[1])))
    score_scale = scores[:, 0].std(unbiased=False).clamp_min(1e-6)
    margin_scale = margin.std(unbiased=False).clamp_min(1e-6)
    score_z = (scores[:, 0] - scores[:, 0].mean()) / score_scale
    margin_z = (margin - margin.mean()) / margin_scale
    height, width = map(int, image_hw)
    keypoint_x = keypoints[:, 0] / max(float(width), 1.0)
    keypoint_y = keypoints[:, 1] / max(float(height), 1.0)
    radius = torch.sqrt(
        (keypoint_x - 0.5).square() + (keypoint_y - 0.5).square()
    )

    attempts = folded["attempts"][anchors]
    prior_strength = max(float(prior_strength), 0.0)
    global_clean_rate = (
        folded["clean"].sum() / folded["attempts"].sum().clamp_min(1.0)
    )
    denominator = attempts + prior_strength
    clean_rate = (
        folded["clean"][anchors] + prior_strength * global_clean_rate
    ) / denominator.clamp_min(1e-6)
    clean_inlier_rate = (
        folded["clean_inlier"][anchors] / denominator.clamp_min(1e-6)
    )
    harmful_inlier_rate = (
        folded["harmful_inlier"][anchors] / denominator.clamp_min(1e-6)
    )
    source = torch.as_tensor(source_groups).long().reshape(-1)[anchors]
    dependency = (
        torch.as_tensor(dependency_groups).long().reshape(-1)[anchors]
    )
    return torch.stack(
        (
            scores[:, 0],
            margin,
            entropy,
            score_z,
            margin_z,
            keypoint_scores,
            keypoint_x,
            keypoint_y,
            radius,
            torch.log1p(attempts),
            clean_rate,
            clean_inlier_rate,
            harmful_inlier_rate,
            torch.log1p(_multiplicity(anchors)),
            torch.log1p(_multiplicity(dependency)),
            torch.log1p(_multiplicity(source)),
        ),
        dim=1,
    )


def predict_pose_sufficient_probability(
    features: torch.Tensor,
    model_state: dict,
) -> torch.Tensor:
    """Apply a serialized logistic selector without a sklearn dependency."""

    if list(model_state.get("feature_names", ())) != list(FEATURE_NAMES):
        raise ValueError("pose-sufficient model feature contract differs")
    features = torch.as_tensor(features).float()
    mean = torch.as_tensor(model_state["feature_mean"]).float()
    scale = torch.as_tensor(model_state["feature_scale"]).float().clamp_min(1e-6)
    coefficients = torch.as_tensor(model_state["coefficients"]).float()
    if (
        features.ndim != 2
        or features.shape[1] != len(FEATURE_NAMES)
        or mean.numel() != len(FEATURE_NAMES)
        or scale.numel() != len(FEATURE_NAMES)
        or coefficients.numel() != len(FEATURE_NAMES)
    ):
        raise ValueError("pose-sufficient model dimensions differ")
    logits = (
        (features - mean.reshape(1, -1)) / scale.reshape(1, -1)
    ) @ coefficients.reshape(-1)
    logits = logits + float(model_state.get("intercept", 0.0))
    return torch.sigmoid(logits)


def spatial_octants(xyz: torch.Tensor) -> torch.Tensor:
    """Assign query-relative 3D candidates to eight robust spatial bins."""

    xyz = torch.as_tensor(xyz).float().reshape(-1, 3)
    if not len(xyz):
        return torch.empty(0, dtype=torch.long)
    center = torch.quantile(xyz, 0.5, dim=0)
    high = xyz >= center
    return (
        high[:, 0].long()
        + 2 * high[:, 1].long()
        + 4 * high[:, 2].long()
    )


def constrained_pose_sufficient_mask(
    probabilities: torch.Tensor,
    *,
    image_cells: torch.Tensor,
    dependency_groups: torch.Tensor,
    source_groups: torch.Tensor,
    xyz: torch.Tensor,
    budget: int,
    minimum_per_image_cell: int = 4,
    minimum_per_spatial_bin: int = 2,
    maximum_per_dependency: int = 4,
    maximum_per_source: int = 2,
) -> torch.Tensor:
    """Select one correspondence set while preserving geometric diversity.

    The selector has no pose branch. It first reserves high-confidence rows
    across image and 3D bins, then fills by confidence under dependency/source
    caps, and finally relaxes only those caps to satisfy the requested budget.
    """

    probabilities = torch.as_tensor(probabilities).float().reshape(-1)
    image_cells = torch.as_tensor(image_cells).long().reshape(-1)
    dependency_groups = torch.as_tensor(dependency_groups).long().reshape(-1)
    source_groups = torch.as_tensor(source_groups).long().reshape(-1)
    xyz = torch.as_tensor(xyz).float().reshape(-1, 3)
    length = len(probabilities)
    if not all(
        len(value) == length
        for value in (image_cells, dependency_groups, source_groups, xyz)
    ):
        raise ValueError("pose-sufficient selector inputs must align")
    if not torch.isfinite(probabilities).all():
        raise ValueError("selection probabilities must be finite")
    target = min(max(int(budget), 4), length)
    if target == length:
        return torch.ones(length, dtype=torch.bool)

    order = torch.argsort(probabilities, descending=True, stable=True).tolist()
    image_values = image_cells.tolist()
    octant_values = spatial_octants(xyz).tolist()
    dependency_values = dependency_groups.tolist()
    source_values = source_groups.tolist()
    selected_indices: list[int] = []
    selected_set: set[int] = set()
    dependency_count: Counter[int] = Counter()
    source_count: Counter[int] = Counter()

    def add(index: int, *, enforce_caps: bool) -> bool:
        if index in selected_set:
            return False
        dependency = dependency_values[index]
        source = source_values[index]
        if enforce_caps and (
            dependency_count[dependency] >= int(maximum_per_dependency)
            or source_count[source] >= int(maximum_per_source)
        ):
            return False
        selected_set.add(index)
        selected_indices.append(index)
        dependency_count[dependency] += 1
        source_count[source] += 1
        return True

    def reserve(group_ids: list[int], minimum: int) -> None:
        for group in sorted(set(group_ids)):
            candidates = [
                index
                for index in order
                if group_ids[index] == group
            ]
            count = 0
            for index in candidates:
                if len(selected_indices) >= target:
                    return
                if add(index, enforce_caps=True):
                    count += 1
                    if count >= int(minimum):
                        break

    reserve(image_values, max(int(minimum_per_image_cell), 0))
    reserve(octant_values, max(int(minimum_per_spatial_bin), 0))
    for index in order:
        if len(selected_indices) >= target:
            break
        add(index, enforce_caps=True)
    for index in order:
        if len(selected_indices) >= target:
            break
        add(index, enforce_caps=False)
    if len(selected_indices) != target:
        raise AssertionError("selector did not satisfy the requested budget")
    selected = torch.zeros(length, dtype=torch.bool)
    selected[torch.as_tensor(selected_indices).long()] = True
    return selected


def balanced_group_cap(budget: int, group_count: int, multiplier: float) -> int:
    """Return a stable cap for callers that use scene-dependent group counts."""

    return max(int(ceil(int(budget) / max(int(group_count), 1) * multiplier)), 1)
