"""Fixed set selection for one-of-K correspondence assignment.

The assignment stage may replace a query row's top-1 landmark, but each row
still contributes at most one 2D--3D correspondence.  These selectors are
deliberately deterministic: they provide a stable set-selection control for
measuring whether identity correction has value without learning another
global correspondence ordering.
"""

from __future__ import annotations

import torch

from localization_training.pose_sufficient_selector import (
    constrained_pose_sufficient_mask,
    image_grid_cells,
)


FIXED_SET_PROTOCOLS = (
    "A1-All",
    "S512-PoseSufficient",
    "S1024-Block8",
)


def block_balanced_mask(
    scores: torch.Tensor,
    keypoints: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    *,
    budget: int,
    blocks: int = 8,
) -> torch.Tensor:
    """Round-robin score ordering over an image block grid.

    Each occupied block contributes its highest-score row before any block
    contributes its second row.  A final score-ordered fill is only needed for
    defensive completeness.  The procedure has no learned state or GT input.
    """

    scores = torch.as_tensor(scores).float().reshape(-1)
    keypoints = torch.as_tensor(
        keypoints, device=scores.device
    ).float().reshape(-1, 2)
    if len(scores) != len(keypoints):
        raise ValueError("block selector scores and keypoints must align")
    if not torch.isfinite(scores).all():
        raise ValueError("block selector scores must be finite")
    target = min(max(int(budget), 4), len(scores))
    if target == len(scores):
        return torch.ones(len(scores), dtype=torch.bool, device=scores.device)

    cells = image_grid_cells(
        keypoints,
        image_hw,
        rows=max(int(blocks), 1),
        cols=max(int(blocks), 1),
    )
    score_order = torch.argsort(scores, descending=True, stable=True)
    # A stable sort by cell preserves descending score within each cell.
    grouped = score_order[
        torch.argsort(cells[score_order], stable=True)
    ]
    grouped_cells = cells[grouped]
    group_start_mask = torch.ones(
        len(grouped), dtype=torch.bool, device=scores.device
    )
    group_start_mask[1:] = grouped_cells[1:] != grouped_cells[:-1]
    starts = torch.where(group_start_mask)[0]
    group_ids = group_start_mask.cumsum(0) - 1
    within_group_rank = torch.arange(
        len(grouped), device=scores.device
    ) - starts[group_ids]
    occupied_count = int(starts.numel())
    round_robin_priority = within_group_rank * occupied_count + group_ids
    selected = grouped[
        torch.argsort(round_robin_priority, stable=True)[:target]
    ]
    selected_mask = torch.zeros(
        len(scores), dtype=torch.bool, device=scores.device
    )
    selected_mask[selected] = True
    if int(selected_mask.sum()) != target:
        raise AssertionError("block selector did not satisfy its budget")
    return selected_mask


def pose_sufficient_mask(
    scores: torch.Tensor,
    keypoints: torch.Tensor,
    landmark_xyz: torch.Tensor,
    dependency_groups: torch.Tensor,
    source_groups: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
    *,
    budget: int = 512,
) -> torch.Tensor:
    """Apply the frozen geometric coverage contract used by the S512 control."""

    scores = torch.as_tensor(scores).float().reshape(-1).cpu()
    keypoints = torch.as_tensor(keypoints).float().reshape(-1, 2).cpu()
    landmark_xyz = torch.as_tensor(landmark_xyz).float().reshape(-1, 3).cpu()
    dependency_groups = torch.as_tensor(dependency_groups).long().reshape(-1).cpu()
    source_groups = torch.as_tensor(source_groups).long().reshape(-1).cpu()
    if not all(
        len(value) == len(scores)
        for value in (
            keypoints,
            landmark_xyz,
            dependency_groups,
            source_groups,
        )
    ):
        raise ValueError("pose-sufficient selector inputs must align")
    return constrained_pose_sufficient_mask(
        scores,
        image_cells=image_grid_cells(keypoints, image_hw, rows=4, cols=4),
        dependency_groups=dependency_groups,
        source_groups=source_groups,
        xyz=landmark_xyz,
        budget=budget,
        minimum_per_image_cell=4,
        minimum_per_spatial_bin=2,
        maximum_per_dependency=4,
        maximum_per_source=2,
    )


def fixed_set_mask(
    protocol: str,
    scores: torch.Tensor,
    keypoints: torch.Tensor,
    landmark_xyz: torch.Tensor,
    dependency_groups: torch.Tensor,
    source_groups: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
) -> torch.Tensor:
    """Return the selected rows for one named frozen protocol."""

    if protocol == "A1-All":
        return torch.ones(len(torch.as_tensor(scores).reshape(-1)), dtype=torch.bool)
    if protocol == "S512-PoseSufficient":
        return pose_sufficient_mask(
            scores,
            keypoints,
            landmark_xyz,
            dependency_groups,
            source_groups,
            image_hw,
            budget=512,
        )
    if protocol == "S1024-Block8":
        return block_balanced_mask(
            scores,
            keypoints,
            image_hw,
            budget=1024,
            blocks=8,
        )
    raise ValueError(f"unknown fixed set protocol: {protocol}")
