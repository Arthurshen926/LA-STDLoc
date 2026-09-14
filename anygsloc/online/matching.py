"""Frozen numerical kernels extracted without algorithm changes."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Top1Matches:
    keypoint_indices: torch.Tensor
    anchor_indices: torch.Tensor
    scores: torch.Tensor


@torch.inference_mode()
def retain_high_score_matches(
    matches: Top1Matches,
    *,
    retention_fraction: float,
    minimum_count: int = 256,
) -> Top1Matches:
    """Keep the strongest cosine matches but preserve native query order."""

    keypoints = torch.as_tensor(matches.keypoint_indices)
    anchors = torch.as_tensor(matches.anchor_indices)
    scores = torch.as_tensor(matches.scores)
    count = int(scores.numel())
    fraction = float(retention_fraction)
    minimum = int(minimum_count)
    if not (
        keypoints.shape == anchors.shape == scores.shape == (count,)
        and bool(torch.isfinite(scores).all())
        and 0.0 < fraction <= 1.0
        and minimum >= 4
    ):
        raise ValueError("match retention inputs are invalid")
    retained_count = min(count, max(minimum, int(np.ceil(fraction * count))))
    if retained_count == count:
        return matches
    strongest = torch.topk(scores, retained_count, largest=True, sorted=False).indices
    # Filtering must not silently become a progressive-sampling ablation.
    # Restore the native query registry order before the standard PoseLib call.
    retained = torch.sort(strongest).values
    return Top1Matches(
        keypoint_indices=keypoints[retained],
        anchor_indices=anchors[retained],
        scores=scores[retained],
    )


@torch.inference_mode()
def global_cosine_top1(
    query_descriptors: torch.Tensor,
    anchor_descriptors: torch.Tensor,
    *,
    chunk_size: int = 8192,
    anchor_descriptors_normalized: bool = False,
) -> Top1Matches:
    if query_descriptors.ndim != 2 or anchor_descriptors.ndim != 2:
        raise ValueError("query and anchor descriptors must be matrices")
    if query_descriptors.shape[1] != anchor_descriptors.shape[1]:
        raise ValueError("query and anchor descriptor dimensions differ")
    count = int(anchor_descriptors.shape[0])
    if count == 0:
        raise ValueError("anchor map is empty")
    chunk_size = max(int(chunk_size), 1)
    query = F.normalize(query_descriptors.float(), dim=1)
    best_scores = query.new_full((query.shape[0],), -torch.inf)
    best_indices = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        anchors = anchor_descriptors[start:stop].float()
        if not anchor_descriptors_normalized:
            anchors = F.normalize(anchors, dim=1)
        scores = query @ anchors.T
        local_scores, local_positions = scores.max(dim=1)
        local_indices = local_positions + start
        # torch.max returns the first occurrence.  Keeping the previous winner
        # on equality therefore defines one chunk-independent tie contract:
        # the lower Anchor row wins equal cosine scores.
        update = local_scores > best_scores
        best_scores = torch.where(update, local_scores, best_scores)
        best_indices = torch.where(update, local_indices, best_indices)
    keypoints = torch.arange(query.shape[0], device=query.device)
    return Top1Matches(
        keypoint_indices=keypoints,
        anchor_indices=best_indices,
        scores=best_scores,
    )
