"""Exact global cosine top-1 matching for a compact anchor map."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching


@dataclass(frozen=True)
class Top1Matches:
    keypoint_indices: torch.Tensor
    anchor_indices: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class Top2Matches:
    keypoint_indices: torch.Tensor
    anchor_indices: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class TopKMatches:
    keypoint_indices: torch.Tensor
    anchor_indices: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class AnchorAssignment:
    matches: Top1Matches
    candidate_edge_count: int
    eligible_edge_count: int
    unmatched_query_count: int
    reassigned_query_count: int
    top1_collision_count: int


@torch.inference_mode()
def suppress_duplicate_anchor_matches(matches: Top1Matches) -> Top1Matches:
    """Keep the highest-scoring query correspondence for each anchor.

    The retained rows are returned in their original query-keypoint order so
    the option changes only correspondence multiplicity, not solver ordering.
    """
    count = int(matches.scores.numel())
    if not (
        matches.keypoint_indices.numel() == matches.anchor_indices.numel() == count
    ):
        raise ValueError("top-1 match rows do not align")
    if count < 2:
        return matches
    score_order = torch.argsort(matches.scores, descending=True, stable=True)
    anchor_order = torch.argsort(matches.anchor_indices[score_order], stable=True)
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
def suppress_duplicate_entity_matches(
    matches: Top1Matches,
    anchor_component_ids: torch.Tensor,
) -> Top1Matches:
    """Keep one best correspondence per audited identity component.

    ``-1`` denotes an isolated anchor and is treated as its own entity.  This
    post-top-1 operation changes neither the map nor the winning Anchor of any
    query keypoint, making it suitable for a no-delete counterfactual audit.
    """
    count = int(matches.scores.numel())
    if not (
        matches.keypoint_indices.numel() == matches.anchor_indices.numel() == count
    ):
        raise ValueError("top-1 match rows do not align")
    components = (
        torch.as_tensor(anchor_component_ids, device=matches.anchor_indices.device)
        .long()
        .reshape(-1)
    )
    if components.numel() == 0:
        raise ValueError("anchor component registry is empty")
    if bool((components < -1).any()):
        raise ValueError("anchor component IDs must be -1 or non-negative")
    if matches.anchor_indices.numel() and (
        int(matches.anchor_indices.min()) < 0
        or int(matches.anchor_indices.max()) >= components.numel()
    ):
        raise ValueError("match references an anchor outside the component registry")
    if count < 2:
        return matches
    component_count = int(components.max()) + 1 if bool((components >= 0).any()) else 0
    matched_components = components[matches.anchor_indices]
    entity_ids = torch.where(
        matched_components >= 0,
        matched_components,
        component_count + matches.anchor_indices,
    )
    score_order = torch.argsort(matches.scores, descending=True, stable=True)
    entity_order = torch.argsort(entity_ids[score_order], stable=True)
    grouped = score_order[entity_order]
    grouped_entities = entity_ids[grouped]
    first = torch.ones(count, dtype=torch.bool, device=grouped.device)
    first[1:] = grouped_entities[1:] != grouped_entities[:-1]
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


@torch.inference_mode()
def global_cosine_top2(
    query_descriptors: torch.Tensor,
    anchor_descriptors: torch.Tensor,
    *,
    chunk_size: int = 8192,
    anchor_descriptors_normalized: bool = False,
) -> Top2Matches:
    """Return exact global top-2 rows for margin-aware one-shot sampling."""
    if query_descriptors.ndim != 2 or anchor_descriptors.ndim != 2:
        raise ValueError("query and anchor descriptors must be matrices")
    if query_descriptors.shape[1] != anchor_descriptors.shape[1]:
        raise ValueError("query and anchor descriptor dimensions differ")
    count = int(anchor_descriptors.shape[0])
    if count < 2:
        raise ValueError("top-2 matching needs at least two anchors")
    chunk_size = max(int(chunk_size), 1)
    query = F.normalize(query_descriptors.float(), dim=1)
    best_scores = query.new_full((query.shape[0], 2), -torch.inf)
    best_indices = torch.zeros(
        (query.shape[0], 2), dtype=torch.long, device=query.device
    )
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        anchors = anchor_descriptors[start:stop].float()
        if not anchor_descriptors_normalized:
            anchors = F.normalize(anchors, dim=1)
        scores = query @ anchors.T
        local_first_scores, local_first_positions = scores.max(dim=1)
        local_first_indices = local_first_positions + start
        if stop - start > 1:
            scores.scatter_(1, local_first_positions[:, None], -torch.inf)
            local_second_scores, local_second_positions = scores.max(dim=1)
            local_second_indices = local_second_positions + start
        else:
            local_second_scores = local_first_scores.new_full(
                local_first_scores.shape, -torch.inf
            )
            local_second_indices = local_first_indices
        local_scores = torch.stack((local_first_scores, local_second_scores), dim=1)
        local_indices = torch.stack((local_first_indices, local_second_indices), dim=1)
        candidate_scores = torch.cat((best_scores, local_scores), dim=1)
        candidate_indices = torch.cat((best_indices, local_indices), dim=1)
        # Canonicalize equal scores by Anchor row before the stable score sort.
        # Only four candidates per row are sorted; the old implementation
        # sorted the complete score chunk at every iteration.
        index_order = torch.argsort(candidate_indices, dim=1, stable=True)
        candidate_scores = torch.gather(candidate_scores, 1, index_order)
        candidate_indices = torch.gather(candidate_indices, 1, index_order)
        score_order = torch.argsort(
            candidate_scores, dim=1, descending=True, stable=True
        )[:, :2]
        best_scores = torch.gather(candidate_scores, 1, score_order)
        best_indices = torch.gather(candidate_indices, 1, score_order)
    keypoints = torch.arange(query.shape[0], device=query.device)
    return Top2Matches(
        keypoint_indices=keypoints,
        anchor_indices=best_indices,
        scores=best_scores,
    )


@torch.inference_mode()
def global_cosine_topk(
    query_descriptors: torch.Tensor,
    anchor_descriptors: torch.Tensor,
    *,
    topk: int,
    chunk_size: int = 8192,
    anchor_descriptors_normalized: bool = False,
) -> TopKMatches:
    """Return exact global cosine top-K candidates without a dense score bank."""
    if query_descriptors.ndim != 2 or anchor_descriptors.ndim != 2:
        raise ValueError("query and anchor descriptors must be matrices")
    if query_descriptors.shape[1] != anchor_descriptors.shape[1]:
        raise ValueError("query and anchor descriptor dimensions differ")
    count = int(anchor_descriptors.shape[0])
    topk = int(topk)
    if topk < 1 or topk > count:
        raise ValueError("top-K must be between one and the anchor count")
    chunk_size = max(int(chunk_size), 1)
    query = F.normalize(query_descriptors.float(), dim=1)
    best_scores = query.new_full((query.shape[0], topk), -torch.inf)
    best_indices = torch.zeros(
        (query.shape[0], topk), dtype=torch.long, device=query.device
    )
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        anchors = anchor_descriptors[start:stop].float()
        if not anchor_descriptors_normalized:
            anchors = F.normalize(anchors, dim=1)
        scores = query @ anchors.T
        indices = torch.arange(start, stop, device=query.device)[None].expand(
            query.shape[0], -1
        )
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_indices = torch.cat((best_indices, indices), dim=1)
        best_scores, positions = torch.topk(merged_scores, topk, dim=1)
        best_indices = torch.gather(merged_indices, 1, positions)
    return TopKMatches(
        keypoint_indices=torch.arange(query.shape[0], device=query.device),
        anchor_indices=best_indices,
        scores=best_scores,
    )


@torch.inference_mode()
def maximum_weight_anchor_assignment(
    candidates: TopKMatches,
    *,
    dustbin_score: float,
) -> AnchorAssignment:
    """Extract an Anchor-unique correspondence set from sparse top-K edges.

    Each query row and each real Anchor has capacity one.  Every query also has
    a private dustbin edge, so rows whose best feasible utility is not strictly
    above ``dustbin_score`` remain unmatched.  The sparse bipartite optimum is
    solved on CPU; returned tensors are restored to the candidate device and
    remain ordered by query row for the single standard PoseLib call.
    """
    keypoints = torch.as_tensor(candidates.keypoint_indices)
    anchors = torch.as_tensor(candidates.anchor_indices)
    scores = torch.as_tensor(candidates.scores)
    if keypoints.ndim != 1 or anchors.ndim != 2 or scores.ndim != 2:
        raise ValueError("top-K assignment inputs have invalid ranks")
    if anchors.shape != scores.shape or anchors.shape[0] != keypoints.numel():
        raise ValueError("top-K assignment rows do not align")
    if anchors.shape[1] < 1:
        raise ValueError("top-K assignment needs at least one candidate per row")
    if torch.unique(keypoints).numel() != keypoints.numel():
        raise ValueError("top-K assignment query rows must be unique")
    if anchors.numel() and bool((anchors < 0).any()):
        raise ValueError("top-K assignment contains a negative Anchor index")
    if anchors.shape[1] > 1:
        ordered = torch.sort(anchors, dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("top-K assignment candidates must be unique per row")
    if not torch.isfinite(scores).all():
        raise ValueError("top-K assignment scores must be finite")
    if scores.shape[1] > 1 and bool((scores[:, 1:] > scores[:, :-1]).any()):
        raise ValueError("top-K assignment scores must be rank-sorted")
    if not np.isfinite(float(dustbin_score)):
        raise ValueError("dustbin score must be finite")
    query_count, candidate_count = anchors.shape
    if query_count == 0:
        empty_long = keypoints.new_empty((0,), dtype=torch.long)
        empty_score = scores.new_empty((0,))
        return AnchorAssignment(
            Top1Matches(empty_long, empty_long.clone(), empty_score),
            0,
            0,
            0,
            0,
            0,
        )

    anchors_cpu = anchors.detach().cpu().numpy().astype(np.int64, copy=False)
    scores_cpu = scores.detach().cpu().numpy().astype(np.float64, copy=False)
    # Compare in the descriptor score dtype.  This makes an exactly represented
    # serialized threshold a strict boundary instead of promoting float32 ties
    # through a later float64 conversion.
    eligible = (scores > scores.new_tensor(float(dustbin_score))).detach().cpu().numpy()
    eligible_rows, eligible_ranks = np.nonzero(eligible)
    eligible_anchors = anchors_cpu[eligible_rows, eligible_ranks]
    unique_anchors = np.unique(eligible_anchors)
    real_column_count = int(unique_anchors.size)

    # Every row has a private dustbin.  Positive weights are required because
    # scipy sparse matrices remove explicit zero-weight edges.
    rows = [np.arange(query_count, dtype=np.int64)]
    columns = [real_column_count + np.arange(query_count, dtype=np.int64)]
    weights = [np.full(query_count, 2.0, dtype=np.float64)]
    if eligible_rows.size:
        real_columns = np.searchsorted(unique_anchors, eligible_anchors)
        rows.append(eligible_rows.astype(np.int64, copy=False))
        columns.append(real_columns.astype(np.int64, copy=False))
        weights.append(
            2.0 + scores_cpu[eligible_rows, eligible_ranks] - float(dustbin_score)
        )
    graph = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(query_count, real_column_count + query_count),
    ).tocsr()
    row_indices, column_indices = min_weight_full_bipartite_matching(
        graph, maximize=True
    )
    order = np.argsort(row_indices, kind="stable")
    row_indices = row_indices[order]
    column_indices = column_indices[order]
    real = column_indices < real_column_count
    matched_rows = row_indices[real]
    matched_anchors = unique_anchors[column_indices[real]]

    matched_scores = np.empty(matched_rows.size, dtype=np.float64)
    matched_ranks = np.empty(matched_rows.size, dtype=np.int64)
    for position, (row, anchor) in enumerate(zip(matched_rows, matched_anchors)):
        ranks = np.flatnonzero(anchors_cpu[row] == anchor)
        if ranks.size != 1:
            raise RuntimeError("assignment result does not map to one candidate edge")
        rank = int(ranks[0])
        matched_ranks[position] = rank
        matched_scores[position] = scores_cpu[row, rank]

    device = keypoints.device
    retained_rows = torch.as_tensor(matched_rows, dtype=torch.long, device=device)
    matches = Top1Matches(
        keypoint_indices=keypoints[retained_rows],
        anchor_indices=torch.as_tensor(
            matched_anchors, dtype=torch.long, device=anchors.device
        ),
        scores=torch.as_tensor(
            matched_scores, dtype=scores.dtype, device=scores.device
        ),
    )
    top1_unique = np.unique(anchors_cpu[:, 0]).size
    return AnchorAssignment(
        matches=matches,
        candidate_edge_count=int(query_count * candidate_count),
        eligible_edge_count=int(eligible_rows.size),
        unmatched_query_count=int(query_count - matched_rows.size),
        reassigned_query_count=int(np.count_nonzero(matched_ranks > 0)),
        top1_collision_count=int(query_count - top1_unique),
    )
