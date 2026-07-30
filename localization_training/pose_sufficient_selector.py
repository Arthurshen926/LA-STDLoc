"""Constrained one-pass correspondence selection for sparse PnP."""

from __future__ import annotations

from collections import Counter
from math import ceil, comb

import torch

from localization_training.basis_utility import (
    deterministic_triplets,
    group_independent_triplets,
    image_triangle_area_fraction,
    triangle_shape_quality,
)


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


def predict_multitask_quality(
    features: torch.Tensor,
    model_states: dict[str, dict],
) -> dict[str, torch.Tensor]:
    """Predict strict-clean, solver-clean, and harmful probabilities."""

    required = ("strict_clean", "solver_clean", "harmful")
    if any(name not in model_states for name in required):
        raise ValueError("multi-task selector requires all three quality heads")
    return {
        name: predict_pose_sufficient_probability(
            features, model_states[name]
        )
        for name in required
    }


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


def _expected_basis_count(
    selected: torch.Tensor,
    *,
    usable_probability: torch.Tensor,
    image_points: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    dependency_groups: torch.Tensor,
    source_groups: torch.Tensor,
    xyz: torch.Tensor,
    sample_count: int = 256,
) -> float:
    indices = torch.where(torch.as_tensor(selected).bool())[0]
    if len(indices) < 3:
        return 0.0
    triplets = deterministic_triplets(
        indices,
        count=min(int(sample_count), comb(len(indices), 3)),
        seed=0,
        query_name="basis-sufficiency",
    )
    independent = group_independent_triplets(
        triplets, dependency_groups, source_groups
    )
    image_points = torch.as_tensor(image_points).float().reshape(-1, 2)
    xyz = torch.as_tensor(xyz).float().reshape(-1, 3)
    non_degenerate = (
        (
            image_triangle_area_fraction(
                image_points[triplets], image_hw
            )
            >= 1e-4
        )
        & (triangle_shape_quality(image_points[triplets]) >= 0.01)
        & (triangle_shape_quality(xyz[triplets]) >= 0.01)
    )
    probability = torch.as_tensor(
        usable_probability
    ).float().reshape(-1)[triplets].prod(dim=1)
    expected_rate = (
        probability * independent.float() * non_degenerate.float()
    ).mean()
    return float(expected_rate) * comb(len(indices), 3)


def basis_aware_core_reserve_mask(
    strict_probability: torch.Tensor,
    solver_probability: torch.Tensor,
    harmful_probability: torch.Tensor,
    *,
    image_points: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    image_cells: torch.Tensor,
    dependency_groups: torch.Tensor,
    source_groups: torch.Tensor,
    xyz: torch.Tensor,
    core_budget: int = 384,
    minimum_budget: int = 512,
    maximum_budget: int = 768,
    harmful_power: float = 2.0,
    strict_lcb_z: float = 1.64,
    minimum_strict_lcb: float = 80.0,
    minimum_dependency_groups: int = 96,
    minimum_image_cells: int = 16,
    minimum_log_expected_basis: float = 11.0,
    representative_count: int = 64,
    pair_count: int = 256,
    maximum_per_dependency: int = 6,
    maximum_per_source: int = 3,
    return_details: bool = False,
) -> (
    tuple[torch.Tensor, dict[str, float]]
    | tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]
):
    """Build a strict core and add only geometrically useful reserve rows."""

    strict = torch.as_tensor(strict_probability).float().reshape(-1)
    solver = torch.as_tensor(solver_probability).float().reshape(-1)
    harmful = torch.as_tensor(harmful_probability).float().reshape(-1)
    image_points = torch.as_tensor(image_points).float().reshape(-1, 2)
    image_cells = torch.as_tensor(image_cells).long().reshape(-1)
    dependency = torch.as_tensor(dependency_groups).long().reshape(-1)
    source = torch.as_tensor(source_groups).long().reshape(-1)
    xyz = torch.as_tensor(xyz).float().reshape(-1, 3)
    length = len(strict)
    if not all(
        len(value) == length
        for value in (
            solver,
            harmful,
            image_points,
            image_cells,
            dependency,
            source,
            xyz,
        )
    ):
        raise ValueError("basis-aware selector inputs must align")
    probabilities = torch.stack((strict, solver, harmful), dim=1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("basis-aware probabilities must be finite")
    if bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("basis-aware probabilities must be in [0, 1]")

    maximum = min(max(int(maximum_budget), 4), length)
    minimum = min(max(int(minimum_budget), 4), maximum)
    core_count = min(max(int(core_budget), 4), minimum)
    usable = solver * (1.0 - harmful).clamp_min(0).pow(
        float(harmful_power)
    )
    core_score = strict * (1.0 - harmful).clamp_min(0).pow(
        float(harmful_power)
    )
    core = constrained_pose_sufficient_mask(
        core_score,
        image_cells=image_cells,
        dependency_groups=dependency,
        source_groups=source,
        xyz=xyz,
        budget=core_count,
        minimum_per_image_cell=4,
        minimum_per_spatial_bin=2,
        maximum_per_dependency=4,
        maximum_per_source=2,
    )

    core_order = torch.argsort(
        core_score.masked_fill(~core, float("-inf")),
        descending=True,
        stable=True,
    )
    representatives = core_order[: min(int(representative_count), int(core.sum()))]
    pairs = torch.combinations(representatives, r=2)
    pair_valid = (
        (dependency[pairs[:, 0]] != dependency[pairs[:, 1]])
        & (source[pairs[:, 0]] != source[pairs[:, 1]])
        & (image_cells[pairs[:, 0]] != image_cells[pairs[:, 1]])
    )
    pairs = pairs[pair_valid]
    pair_weight = usable[pairs].prod(dim=1)
    if len(pairs) > int(pair_count):
        keep = torch.argsort(
            pair_weight, descending=True, stable=True
        )[: int(pair_count)]
        pairs = pairs[keep]
        pair_weight = pair_weight[keep]
    reserve_gain = torch.zeros(length)
    if len(pairs):
        candidate = torch.arange(length)[:, None]
        first = pairs[:, 0][None]
        second = pairs[:, 1][None]
        group_valid = (
            (candidate != first)
            & (candidate != second)
            & (dependency[:, None] != dependency[first])
            & (dependency[:, None] != dependency[second])
            & (source[:, None] != source[first])
            & (source[:, None] != source[second])
            & (image_cells[:, None] != image_cells[first])
            & (image_cells[:, None] != image_cells[second])
        )
        candidate_points = image_points[:, None, :].expand(-1, len(pairs), -1)
        image_triangles = torch.stack(
            (
                candidate_points,
                image_points[pairs[:, 0]][None].expand(length, -1, -1),
                image_points[pairs[:, 1]][None].expand(length, -1, -1),
            ),
            dim=2,
        ).reshape(-1, 3, 2)
        candidate_xyz = xyz[:, None, :].expand(-1, len(pairs), -1)
        world_triangles = torch.stack(
            (
                candidate_xyz,
                xyz[pairs[:, 0]][None].expand(length, -1, -1),
                xyz[pairs[:, 1]][None].expand(length, -1, -1),
            ),
            dim=2,
        ).reshape(-1, 3, 3)
        geometry_valid = (
            (
                image_triangle_area_fraction(
                    image_triangles, image_hw
                )
                >= 1e-4
            )
            & (triangle_shape_quality(image_triangles) >= 0.01)
            & (triangle_shape_quality(world_triangles) >= 0.01)
        ).reshape(length, -1)
        weighted = (
            group_valid.float()
            * geometry_valid.float()
            * pair_weight[None]
        )
        reserve_gain = usable * (
            weighted.sum(dim=1) / pair_weight.sum().clamp_min(1e-8)
        )
    reserve_score = reserve_gain + 0.05 * core_score + 0.02 * usable
    reserve_score[core] = float("-inf")
    reserve_order = torch.argsort(
        reserve_score, descending=True, stable=True
    ).tolist()

    selected = core.clone()
    dependency_count = Counter(dependency[selected].tolist())
    source_count = Counter(source[selected].tolist())
    expected_basis = _expected_basis_count(
        selected,
        usable_probability=usable,
        image_points=image_points,
        image_hw=image_hw,
        dependency_groups=dependency,
        source_groups=source,
        xyz=xyz,
    )

    def sufficient() -> bool:
        chosen = torch.where(selected)[0]
        strict_mean = strict[chosen].sum()
        strict_variance = (
            strict[chosen] * (1.0 - strict[chosen])
        ).sum()
        strict_lcb = strict_mean - float(strict_lcb_z) * torch.sqrt(
            strict_variance.clamp_min(0)
        )
        return (
            len(chosen) >= minimum
            and float(strict_lcb) >= float(minimum_strict_lcb)
            and len(torch.unique(dependency[chosen]))
            >= int(minimum_dependency_groups)
            and len(torch.unique(image_cells[chosen]))
            >= int(minimum_image_cells)
            and torch.log1p(
                torch.tensor(expected_basis)
            ).item()
            >= float(minimum_log_expected_basis)
        )

    for index in reserve_order:
        if int(selected.sum()) >= maximum or sufficient():
            break
        group = int(dependency[index])
        primitive = int(source[index])
        enforce_caps = int(selected.sum()) < minimum
        if enforce_caps and (
            dependency_count[group] >= int(maximum_per_dependency)
            or source_count[primitive] >= int(maximum_per_source)
        ):
            continue
        selected[index] = True
        dependency_count[group] += 1
        source_count[primitive] += 1
        expected_basis += float(reserve_gain[index]) * comb(
            int(selected.sum()) - 1, 2
        )

    if int(selected.sum()) < minimum:
        for index in reserve_order:
            if int(selected.sum()) >= minimum:
                break
            if not selected[index]:
                selected[index] = True
                expected_basis += float(reserve_gain[index]) * comb(
                    int(selected.sum()) - 1, 2
                )
    chosen = torch.where(selected)[0]
    strict_mean = strict[chosen].sum()
    strict_variance = (
        strict[chosen] * (1.0 - strict[chosen])
    ).sum()
    strict_lcb = strict_mean - float(strict_lcb_z) * torch.sqrt(
        strict_variance.clamp_min(0)
    )
    diagnostics = {
        "selected_count": float(len(chosen)),
        "strict_lcb": float(strict_lcb),
        "dependency_group_count": float(len(torch.unique(dependency[chosen]))),
        "image_cell_count": float(len(torch.unique(image_cells[chosen]))),
        "log_expected_basis": float(
            torch.log1p(torch.tensor(expected_basis))
        ),
        "mean_core_score": float(core_score[core].mean()),
        "mean_reserve_gain": float(
            reserve_gain[selected & ~core].mean()
            if bool((selected & ~core).any())
            else 0.0
        ),
    }
    if not return_details:
        return selected, diagnostics
    return (
        selected,
        diagnostics,
        {
            "core_mask": core,
            "usable_probability": usable,
            "core_score": core_score,
            "reserve_gain": reserve_gain,
            "reserve_score": reserve_score,
        },
    )


def balanced_group_cap(budget: int, group_count: int, multiplier: float) -> int:
    """Return a stable cap for callers that use scene-dependent group counts."""

    return max(int(ceil(int(budget) / max(int(group_count), 1) * multiplier)), 1)
