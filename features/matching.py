"""Minimal matching helpers used by offline self-localization teachers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseMatchResult:
    keypoint_idx: torch.Tensor
    landmark_idx: torch.Tensor
    scores: torch.Tensor


def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Apply SuperPoint's deterministic two-pass non-maximum suppression."""
    nms_radius = int(nms_radius)
    if nms_radius < 0:
        raise ValueError("nms_radius must be non-negative")

    def max_pool(value: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(
            value,
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        suppression = max_pool(max_mask.float()) > 0
        suppressed_scores = torch.where(suppression, zeros, scores)
        new_max_mask = suppressed_scores == max_pool(suppressed_scores)
        max_mask = max_mask | (new_max_mask & ~suppression)
    return torch.where(max_mask, scores, zeros)


def _matches_per_group_mask(
    scores: torch.Tensor,
    group_ids: torch.Tensor,
    maximum: int,
    group_name: str,
) -> torch.Tensor:
    maximum = int(maximum)
    if maximum <= 0:
        return torch.ones(group_ids.numel(), dtype=torch.bool, device=group_ids.device)
    if group_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=group_ids.device)
    if bool((group_ids < 0).any()):
        raise ValueError(f"{group_name} indices must be non-negative")
    count = int(group_ids.numel())
    group_count = int(group_ids.max()) + 1
    positions = torch.arange(count, device=group_ids.device)
    active = torch.ones(count, dtype=torch.bool, device=group_ids.device)
    keep = torch.zeros_like(active)
    for _ in range(min(maximum, count)):
        if not bool(active.any()):
            break
        best_scores = scores.new_full((group_count,), -torch.inf)
        best_scores.scatter_reduce_(
            0,
            group_ids,
            scores.masked_fill(~active, -torch.inf),
            reduce="amax",
            include_self=True,
        )
        has_best = active & (scores == best_scores[group_ids])
        first = positions.new_full((group_count,), count)
        first.scatter_reduce_(
            0,
            group_ids,
            torch.where(has_best, positions, positions.new_full((), count)),
            reduce="amin",
            include_self=True,
        )
        selected = has_best & (positions == first[group_ids])
        keep |= selected
        active &= ~selected
    return keep


def match_candidate_selection_mask(
    matches: SparseMatchResult,
    *,
    threshold: float = -float("inf"),
    max_matches_per_keypoint: int = 0,
    max_matches_per_landmark: int = 0,
    min_match_count: int = 0,
    refill_trigger_count: int = 0,
    max_match_count: int = 0,
) -> torch.Tensor:
    """Replay the exact sparse candidate acceptance mask for a teacher."""
    count = int(matches.scores.numel())
    keep_after_quota = torch.ones(
        count, dtype=torch.bool, device=matches.scores.device
    )
    keep_after_quota &= _matches_per_group_mask(
        matches.scores,
        matches.keypoint_idx,
        max_matches_per_keypoint,
        "keypoint",
    )
    keypoint_limited = torch.nonzero(
        keep_after_quota, as_tuple=False
    ).reshape(-1)
    landmark_keep = _matches_per_group_mask(
        matches.scores[keypoint_limited],
        matches.landmark_idx[keypoint_limited],
        max_matches_per_landmark,
        "landmark",
    )
    quota_mask = torch.zeros_like(keep_after_quota)
    quota_mask[keypoint_limited[landmark_keep]] = True
    limited = torch.nonzero(quota_mask, as_tuple=False).reshape(-1)
    if limited.numel() == 0:
        return quota_mask
    scores = matches.scores[limited]
    accepted = scores > float(threshold)
    target = min(max(int(min_match_count), 0), int(scores.numel()))
    trigger = max(int(refill_trigger_count), 0) or target
    if int(accepted.sum()) < trigger and int(accepted.sum()) < target:
        accepted = accepted.clone()
        accepted[torch.topk(scores, target).indices] = True
    cap = max(int(max_match_count), 0)
    if cap > 0 and int(accepted.sum()) > cap:
        indices = torch.nonzero(accepted, as_tuple=False).reshape(-1)
        capped = torch.zeros_like(accepted)
        capped[indices[torch.topk(scores[indices], cap).indices]] = True
        accepted = capped
    result = torch.zeros_like(quota_mask)
    result[limited[accepted]] = True
    return result
