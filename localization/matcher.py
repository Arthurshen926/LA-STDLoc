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
    strongest = torch.topk(
        scores, retained_count, largest=True, sorted=False
    ).indices
    # Filtering must not silently become a progressive-sampling ablation.
    # Restore the native query registry order before the standard PoseLib call.
    retained = torch.sort(strongest).values
    return Top1Matches(
        keypoint_indices=keypoints[retained],
        anchor_indices=anchors[retained],
        scores=scores[retained],
    )


@torch.inference_mode()
def retain_diverse_confidence_matches(
    matches: Top1Matches,
    *,
    keypoints: torch.Tensor,
    second_best_scores: torch.Tensor,
    anchor_matchability: torch.Tensor,
    anchor_uncertainty: torch.Tensor,
    anchor_xyz: torch.Tensor,
    image_hw: tuple[int, int],
    retention_fraction: float,
    minimum_count: int = 256,
    image_grid_shape: tuple[int, int] = (6, 8),
    spatial_bins_per_axis: int = 4,
    diversity_weight: float = 0.15,
) -> Top1Matches:
    """Select a confidence core with mapping-only 2D/3D diversity priors."""

    rows = torch.as_tensor(matches.keypoint_indices).long()
    owners = torch.as_tensor(matches.anchor_indices).long()
    score = torch.as_tensor(matches.scores).float()
    second = torch.as_tensor(second_best_scores, device=score.device).float()
    xy = torch.as_tensor(keypoints, device=score.device).float()[rows]
    reliability_bank = torch.as_tensor(anchor_matchability, device=score.device).float()
    uncertainty_bank = torch.as_tensor(anchor_uncertainty, device=score.device).float()
    xyz_bank = torch.as_tensor(anchor_xyz, device=score.device).float()
    count = score.numel()
    height, width = int(image_hw[0]), int(image_hw[1])
    grid_y, grid_x = int(image_grid_shape[0]), int(image_grid_shape[1])
    target = min(count, max(int(minimum_count), int(np.ceil(float(retention_fraction) * count))))
    if not (
        rows.shape == owners.shape == score.shape == second.shape == (count,)
        and 0.0 < float(retention_fraction) <= 1.0
        and int(minimum_count) >= 4
        and height > 0
        and width > 0
        and grid_y > 1
        and grid_x > 1
        and int(spatial_bins_per_axis) >= 2
        and 0.0 <= float(diversity_weight) <= 0.5
        and (not owners.numel() or int(owners.min()) >= 0)
        and (not owners.numel() or int(owners.max()) < reliability_bank.numel())
        and reliability_bank.shape == uncertainty_bank.shape == (xyz_bank.shape[0],)
        and xyz_bank.shape[1:] == (3,)
        and bool(torch.isfinite(score).all())
        and bool(torch.isfinite(second).all())
    ):
        raise ValueError("diverse confidence-core inputs are invalid")
    if target == count:
        return matches

    def percentile(value: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(value, stable=True)
        output = torch.empty_like(value)
        output[order] = torch.linspace(0.0, 1.0, value.numel(), device=value.device)
        return output

    reliability = reliability_bank[owners]
    certainty = (1.0 + uncertainty_bank[owners]).reciprocal()
    margin = score - second
    confidence = (
        0.45 * percentile(score)
        + 0.30 * percentile(margin)
        + 0.15 * percentile(reliability)
        + 0.10 * percentile(certainty)
    )
    cell_x = (xy[:, 0] * grid_x / width).long().clamp(0, grid_x - 1)
    cell_y = (xy[:, 1] * grid_y / height).long().clamp(0, grid_y - 1)
    cells = cell_y * grid_x + cell_x
    cell_population = torch.bincount(cells, minlength=grid_y * grid_x).float()
    cell_bonus = cell_population[cells].clamp_min(1).rsqrt()

    matched_xyz = xyz_bank[owners]
    low = torch.quantile(matched_xyz, 0.02, dim=0)
    high = torch.quantile(matched_xyz, 0.98, dim=0)
    normalized = ((matched_xyz - low) / (high - low).clamp_min(1e-8)).clamp(0, 0.999999)
    bins = (normalized * int(spatial_bins_per_axis)).long()
    spatial_id = (
        bins[:, 0] * int(spatial_bins_per_axis) ** 2
        + bins[:, 1] * int(spatial_bins_per_axis)
        + bins[:, 2]
    )
    spatial_population = torch.bincount(
        spatial_id, minlength=int(spatial_bins_per_axis) ** 3
    ).float()
    spatial_bonus = spatial_population[spatial_id].clamp_min(1).rsqrt()
    quality = confidence + float(diversity_weight) * (cell_bonus + spatial_bonus)

    # One query row per Anchor inside the core.  The highest-quality row owns
    # the Anchor; if uniqueness leaves too few rows, fill from remaining rows.
    ranked = torch.argsort(quality, descending=True, stable=True)
    owner_order = torch.argsort(owners[ranked], stable=True)
    grouped = ranked[owner_order]
    grouped_owners = owners[grouped]
    first = torch.ones(count, dtype=torch.bool, device=score.device)
    first[1:] = grouped_owners[1:] != grouped_owners[:-1]
    unique_rows = grouped[first]
    unique_rows = unique_rows[torch.argsort(quality[unique_rows], descending=True, stable=True)]
    selected = unique_rows[:target]
    if selected.numel() < target:
        used = torch.zeros(count, dtype=torch.bool, device=score.device)
        used[selected] = True
        selected = torch.cat((selected, ranked[~used[ranked]][: target - selected.numel()]))
    selected = torch.sort(selected).values
    return Top1Matches(
        keypoint_indices=rows[selected],
        anchor_indices=owners[selected],
        scores=score[selected],
    )


@torch.inference_mode()
def global_owner_prototype_top1(
    query_descriptors: torch.Tensor,
    anchor_descriptors: torch.Tensor,
    extra_prototypes: torch.Tensor,
    prototype_owner_rows: torch.Tensor,
    *,
    chunk_size: int = 8192,
    anchor_descriptors_normalized: bool = False,
    prototype_activation_threshold: float | None = None,
) -> Top1Matches:
    """Match a sparse prototype extension while emitting base Anchor rows.

    Base descriptors remain first in the flat bank, so an empty extension is
    bit-for-bit the historical matcher and an exact score tie keeps the base
    Anchor.  Extra prototypes change appearance capacity only: every winning
    prototype is collapsed to its owning Anchor before PnP.
    """

    anchors = torch.as_tensor(anchor_descriptors)
    prototypes = torch.as_tensor(extra_prototypes)
    owners = torch.as_tensor(prototype_owner_rows).long().reshape(-1)
    if anchors.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("base Anchors and extra prototypes must be matrices")
    if prototypes.shape[0] != owners.numel():
        raise ValueError("prototype owners do not align with prototype rows")
    if prototypes.shape[1] != anchors.shape[1]:
        raise ValueError("prototype and base descriptor dimensions differ")
    if owners.numel() and (
        int(owners.min()) < 0 or int(owners.max()) >= anchors.shape[0]
    ):
        raise ValueError("prototype owner is outside the base Anchor registry")
    if prototype_activation_threshold is not None and not (
        -1.0 <= float(prototype_activation_threshold) <= 1.0
    ):
        raise ValueError("prototype activation threshold must be in [-1, 1]")
    if prototypes.shape[0] == 0:
        return global_cosine_top1(
            query_descriptors,
            anchors,
            chunk_size=chunk_size,
            anchor_descriptors_normalized=anchor_descriptors_normalized,
        )
    base = global_cosine_top1(
        query_descriptors,
        anchors,
        chunk_size=chunk_size,
        anchor_descriptors_normalized=anchor_descriptors_normalized,
    )
    query = F.normalize(torch.as_tensor(query_descriptors).float(), dim=1)
    normalized_prototypes = prototypes.float()
    if not anchor_descriptors_normalized:
        normalized_prototypes = F.normalize(normalized_prototypes, dim=1)
    prototype_scores = query @ normalized_prototypes.T
    best_prototype_scores, best_prototype_rows = prototype_scores.max(dim=1)
    # Strict improvement preserves the historical base winner on exact ties.
    use_prototype = best_prototype_scores > base.scores
    if prototype_activation_threshold is not None:
        use_prototype &= best_prototype_scores >= float(
            prototype_activation_threshold
        )
    owner_rows = base.anchor_indices.clone()
    owner_rows[use_prototype] = owners[best_prototype_rows[use_prototype]]
    scores = base.scores.clone()
    scores[use_prototype] = best_prototype_scores[use_prototype]
    return Top1Matches(
        keypoint_indices=base.keypoint_indices,
        anchor_indices=owner_rows,
        scores=scores,
    )


@torch.inference_mode()
def global_view_mixture_topk(
    query_descriptors: torch.Tensor,
    anchor_prototypes: torch.Tensor,
    prototype_priors: torch.Tensor,
    *,
    topk: int = 1,
    temperature: float = 0.05,
) -> TopKMatches:
    """Match one or two prototypes while emitting one score per Anchor."""
    # Deployment already supplies the exact normalized query bank.  A second
    # normalization would move all-K1 scores at the last float32 bit.
    query = torch.as_tensor(query_descriptors).float()
    prototypes = torch.as_tensor(anchor_prototypes).float()
    priors = torch.as_tensor(prototype_priors).float()
    if prototypes.ndim != 3 or prototypes.shape[1] != 2:
        raise ValueError("view-mixture prototypes must have shape [N,2,D]")
    if priors.shape != prototypes.shape[:2]:
        raise ValueError("view-mixture priors do not align")
    if query.shape[1] != prototypes.shape[2]:
        raise ValueError("query and prototype descriptor dimensions differ")
    if bool((priors < 0).any()) or not torch.allclose(
        priors.sum(dim=1), torch.ones(priors.shape[0], device=priors.device)
    ):
        raise ValueError("each Anchor prototype prior must sum to one")
    if float(temperature) <= 0:
        raise ValueError("mixture temperature must be positive")
    count = int(prototypes.shape[0])
    topk = int(topk)
    if topk < 1 or topk > count:
        raise ValueError("top-K must be between one and the Anchor count")
    prototype_scores = (
        query @ prototypes.reshape(-1, prototypes.shape[2]).T
    ).reshape(query.shape[0], count, 2)
    log_prior = torch.where(
        priors > 0, priors.log(), torch.full_like(priors, -torch.inf)
    )
    scores = float(temperature) * torch.logsumexp(
        prototype_scores / float(temperature) + log_prior[None], dim=2
    )
    single = priors[:, 1] == 0
    scores[:, single] = prototype_scores[:, single, 0]
    best_scores, best_indices = torch.topk(scores, k=topk, dim=1)
    return TopKMatches(
        keypoint_indices=torch.arange(query.shape[0], device=query.device),
        anchor_indices=best_indices,
        scores=best_scores,
    )


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
        local_k = min(topk, stop - start)
        local_scores, local_positions = torch.topk(
            scores, local_k, dim=1, largest=True, sorted=True
        )

        # ``torch.topk`` is fast but may choose arbitrary members of an exact
        # tie at its K boundary.  The common path has no boundary tie and only
        # sorts K candidates below.  On the rare tied rows, explicitly add the
        # lowest-row boundary members before the same lexicographic reduction.
        boundary = local_scores[:, -1]
        boundary_tie_count = (scores == boundary[:, None]).sum(dim=1)
        selected_boundary_count = (local_scores == boundary[:, None]).sum(dim=1)
        if bool((boundary_tie_count != selected_boundary_count).any()):
            relative_rows = torch.arange(
                stop - start, device=query.device, dtype=torch.long
            )[None].expand(query.shape[0], -1)
            sentinel = stop - start
            boundary_rows = torch.where(
                scores == boundary[:, None],
                relative_rows,
                torch.full_like(relative_rows, sentinel),
            )
            lowest_boundary_rows = torch.topk(
                boundary_rows,
                local_k,
                dim=1,
                largest=False,
                sorted=True,
            ).values
            valid_boundary = lowest_boundary_rows < sentinel
            safe_boundary_rows = lowest_boundary_rows.clamp_max(sentinel - 1)
            lowest_boundary_scores = torch.gather(
                scores, 1, safe_boundary_rows
            )
            lowest_boundary_scores = torch.where(
                valid_boundary,
                lowest_boundary_scores,
                torch.full_like(lowest_boundary_scores, -torch.inf),
            )
            local_positions = torch.cat(
                (local_positions, safe_boundary_rows), dim=1
            )
            local_scores = torch.cat(
                (local_scores, lowest_boundary_scores), dim=1
            )
            # The arbitrary Top-K boundary members and the explicit lowest
            # boundary rows may overlap.  Keep one occurrence per local row.
            local_index_order = torch.argsort(
                local_positions, dim=1, stable=True
            )
            local_positions = torch.gather(
                local_positions, 1, local_index_order
            )
            local_scores = torch.gather(local_scores, 1, local_index_order)
            duplicate = torch.zeros_like(local_positions, dtype=torch.bool)
            duplicate[:, 1:] = local_positions[:, 1:] == local_positions[:, :-1]
            local_scores = local_scores.masked_fill(duplicate, -torch.inf)

        local_indices = local_positions + start
        # Canonicalize all retained exact ties by global Anchor row.  Only K
        # (or 2K on the rare boundary-tie fallback) values are sorted here;
        # the previous kernel stably sorted every full map chunk for every
        # query, which dominated exact Top-K runtime.
        local_index_order = torch.argsort(local_indices, dim=1, stable=True)
        local_scores = torch.gather(local_scores, 1, local_index_order)
        local_indices = torch.gather(local_indices, 1, local_index_order)
        local_score_order = torch.argsort(
            local_scores, dim=1, descending=True, stable=True
        )[:, :local_k]
        local_scores = torch.gather(local_scores, 1, local_score_order)
        local_indices = torch.gather(local_indices, 1, local_score_order)

        merged_scores = torch.cat((best_scores, local_scores), dim=1)
        merged_indices = torch.cat((best_indices, local_indices), dim=1)
        index_order = torch.argsort(merged_indices, dim=1, stable=True)
        merged_scores = torch.gather(merged_scores, 1, index_order)
        merged_indices = torch.gather(merged_indices, 1, index_order)
        score_order = torch.argsort(merged_scores, dim=1, descending=True, stable=True)[
            :, :topk
        ]
        best_scores = torch.gather(merged_scores, 1, score_order)
        best_indices = torch.gather(merged_indices, 1, score_order)
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
