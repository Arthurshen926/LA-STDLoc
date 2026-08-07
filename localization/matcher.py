"""Exact global cosine top-1 matching for a compact anchor map."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Top1Matches:
    keypoint_indices: torch.Tensor
    anchor_indices: torch.Tensor
    scores: torch.Tensor


@torch.inference_mode()
def suppress_duplicate_anchor_matches(matches: Top1Matches) -> Top1Matches:
    """Keep the highest-scoring query correspondence for each anchor.

    The retained rows are returned in their original query-keypoint order so
    the option changes only correspondence multiplicity, not solver ordering.
    """
    count = int(matches.scores.numel())
    if not (
        matches.keypoint_indices.numel()
        == matches.anchor_indices.numel()
        == count
    ):
        raise ValueError("top-1 match rows do not align")
    if count < 2:
        return matches
    score_order = torch.argsort(matches.scores, descending=True, stable=True)
    anchor_order = torch.argsort(
        matches.anchor_indices[score_order], stable=True
    )
    grouped = score_order[anchor_order]
    grouped_anchors = matches.anchor_indices[grouped]
    first = torch.ones(count, dtype=torch.bool, device=grouped.device)
    first[1:] = grouped_anchors[1:] != grouped_anchors[:-1]
    retained = torch.sort(grouped[first]).values
    return Top1Matches(
        keypoint_indices=matches.keypoint_indices[retained],
        anchor_indices=matches.anchor_indices[retained],
        scores=matches.scores[retained],
    )


@torch.inference_mode()
def global_cosine_top1(
    query_descriptors: torch.Tensor,
    anchor_descriptors: torch.Tensor,
    *,
    chunk_size: int = 8192,
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
    best_scores = query.new_full((query.shape[0], 1), -torch.inf)
    best_indices = torch.zeros(
        (query.shape[0], 1), dtype=torch.long, device=query.device
    )
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        anchors = F.normalize(anchor_descriptors[start:stop].float(), dim=1)
        scores = query @ anchors.T
        indices = torch.arange(start, stop, device=query.device)[None].expand(
            query.shape[0], -1
        )
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_indices = torch.cat((best_indices, indices), dim=1)
        best_scores, positions = torch.topk(merged_scores, 1, dim=1)
        best_indices = torch.gather(merged_indices, 1, positions)
    keypoints = torch.arange(query.shape[0], device=query.device)
    return Top1Matches(
        keypoint_indices=keypoints,
        anchor_indices=best_indices[:, 0],
        scores=best_scores[:, 0],
    )
