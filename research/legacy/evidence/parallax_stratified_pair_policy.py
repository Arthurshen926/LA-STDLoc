"""Archived 7Scenes parallax-stratified camera-pair proposal policy.

This module intentionally contains only mapping-pose/depth proposal logic.  It
does not read descriptors, matches, Track identities, test images, or deployed
Map rows.  The result remains an experimental proposal table and is not an
authorization to replace the current pair policy.
"""

from __future__ import annotations

import torch


def representative_scene_depth_from_samples(
    depth_at_keypoints: list[torch.Tensor],
) -> torch.Tensor:
    """Return one positive finite median mapping depth per camera.

    Cameras without a usable sample retain a NaN sentinel.  The selector
    replaces those sentinels by the scene median after proving at least one
    valid camera depth exists.
    """
    result = []
    for value in depth_at_keypoints:
        depth = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
        valid = depth[torch.isfinite(depth) & (depth > 0.0)]
        result.append(
            valid.median()
            if valid.numel()
            else torch.tensor(float("nan"), dtype=torch.float64)
        )
    return torch.stack(result) if result else torch.zeros(0, dtype=torch.float64)


def parallax_stratified_pairs(
    *,
    centers: torch.Tensor,
    axes: torch.Tensor,
    distance: torch.Tensor,
    axis_cosine: torch.Tensor,
    valid: torch.Tensor,
    cost: torch.Tensor,
    legacy_pairs: set[tuple[int, int]],
    neighbors: int,
    scene_depth_m: torch.Tensor | None,
    minimum_expected_parallax_deg: float = 1.0,
    near_fraction: float = 1.0 / 3.0,
    maximum_baseline_depth_ratio: float = 0.5,
) -> list[tuple[int, int]]:
    """Replay the archived exact-budget proposal selection rule."""
    count = int(centers.shape[0])
    if scene_depth_m is None:
        raise ValueError("parallax_stratified policy requires mapping scene depth")
    if not 0.0 <= float(near_fraction) <= 1.0:
        raise ValueError("near_fraction must lie in [0, 1]")
    if float(minimum_expected_parallax_deg) <= 0.0:
        raise ValueError("minimum_expected_parallax_deg must be positive")
    if float(maximum_baseline_depth_ratio) <= 0.0:
        raise ValueError("maximum_baseline_depth_ratio must be positive")

    depth = torch.as_tensor(
        scene_depth_m, dtype=centers.dtype, device=centers.device
    ).reshape(-1)
    if int(depth.numel()) != count:
        raise ValueError("scene_depth_m must have one value per camera")
    finite_depth = torch.isfinite(depth) & (depth > 0.0)
    if not bool(finite_depth.any()):
        raise ValueError("scene_depth_m has no positive finite values")
    depth = torch.where(finite_depth, depth, depth[finite_depth].median()).clamp_min(
        1e-4
    )

    longitudinal_left = axes @ centers.T - (axes * centers).sum(dim=1)[:, None]
    longitudinal_right = longitudinal_left.T
    transverse_left = (
        (distance.square() - longitudinal_left.square()).clamp_min(0.0).sqrt()
    )
    transverse_right = (
        (distance.square() - longitudinal_right.square()).clamp_min(0.0).sqrt()
    )
    transverse = 0.5 * (transverse_left + transverse_right)
    pair_depth = torch.sqrt(depth[:, None] * depth[None, :])
    expected_parallax = torch.rad2deg(torch.atan2(transverse, pair_depth))
    depth_ratio = distance / pair_depth.clamp_min(1e-6)
    parallax_valid = valid & (depth_ratio <= float(maximum_baseline_depth_ratio))

    width = min(max(int(neighbors), 1), max(count - 1, 1))
    pairs: set[tuple[int, int]] = set()
    near_reserve_pairs: set[tuple[int, int]] = set()
    for query in range(count):
        order = torch.argsort(cost[query], stable=True)
        legacy = order[torch.isfinite(cost[query, order])][:width]
        if legacy.numel() <= 1:
            candidates = legacy
        else:
            near_count = min(
                width,
                max(1, int(round(width * float(near_fraction)))),
            )
            selected = legacy[:near_count].tolist()
            near_reserve_pairs.update(
                (min(query, int(other)), max(query, int(other))) for other in selected
            )
            remaining = width - len(selected)
            valid_candidates = torch.nonzero(
                parallax_valid[query], as_tuple=False
            ).flatten()
            if remaining > 0 and valid_candidates.numel() > 0:
                target_min = max(float(minimum_expected_parallax_deg), 1e-3)
                targets = torch.logspace(
                    torch.log10(centers.new_tensor(target_min)),
                    torch.log10(centers.new_tensor(max(6.0, target_min))),
                    steps=remaining,
                    dtype=centers.dtype,
                    device=centers.device,
                )
                for target in targets:
                    selected_tensor = torch.as_tensor(
                        selected,
                        dtype=torch.long,
                        device=centers.device,
                    )
                    available = valid_candidates[
                        ~torch.isin(valid_candidates, selected_tensor)
                    ]
                    if available.numel() == 0:
                        break
                    values = expected_parallax[query, available].clamp_min(1e-4)
                    target_cost = torch.log(values / target).abs() + 0.25 * (
                        1.0 - axis_cosine[query, available]
                    )
                    rank = torch.argsort(target_cost, stable=True)
                    selected.append(int(available[rank[0]]))
            if len(selected) < width:
                for other in legacy.tolist():
                    if other not in selected:
                        selected.append(int(other))
                    if len(selected) == width:
                        break
            candidates = torch.as_tensor(selected, dtype=torch.long)
        for other in candidates.tolist():
            if bool(torch.isfinite(cost[query, other])):
                pairs.add((min(query, other), max(query, other)))

    target_budget = len(legacy_pairs)
    if len(pairs) < target_budget:
        upper = torch.triu_indices(count, count, offset=1, device=centers.device)
        eligible = parallax_valid[upper[0], upper[1]]
        pair_rows = upper[:, eligible].T
        pair_parallax = expected_parallax[pair_rows[:, 0], pair_rows[:, 1]].clamp_min(
            1e-4
        )
        pair_axis = axis_cosine[pair_rows[:, 0], pair_rows[:, 1]]
        score = (
            torch.minimum(pair_parallax, pair_parallax.new_tensor(6.0))
            + 0.5 * pair_axis
        )
        for row in torch.argsort(score, descending=True, stable=True).tolist():
            pairs.add(tuple(int(value) for value in pair_rows[row].tolist()))
            if len(pairs) == target_budget:
                break
    if len(pairs) < target_budget:
        for pair in sorted(legacy_pairs - pairs):
            pairs.add(pair)
            if len(pairs) == target_budget:
                break
    if len(pairs) > target_budget:
        removable = sorted(
            pairs - near_reserve_pairs,
            key=lambda pair: (
                float(expected_parallax[pair[0], pair[1]]),
                float(axis_cosine[pair[0], pair[1]]),
                pair,
            ),
        )
        for pair in removable:
            if len(pairs) == target_budget:
                break
            pairs.remove(pair)
    if len(pairs) != target_budget:
        raise RuntimeError(
            "parallax_stratified policy could not preserve the nearest budget"
        )
    return sorted(pairs)


__all__ = [
    "parallax_stratified_pairs",
    "representative_scene_depth_from_samples",
]
