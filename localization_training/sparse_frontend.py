from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SparseMatchResult:
    keypoint_idx: torch.Tensor
    landmark_idx: torch.Tensor
    scores: torch.Tensor


def simple_nms(scores, nms_radius):
    nms_radius = int(nms_radius)
    if nms_radius < 0:
        raise ValueError("nms_radius must be non-negative")

    def max_pool(value):
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


def rank_keypoint_proposals(keypoint_scores, matchability_scores, nms_radius):
    """Keep keypoint-head NMS locations and rank them by matchability."""
    if keypoint_scores.shape != matchability_scores.shape:
        raise ValueError(
            "keypoint and matchability heatmaps must have the same shape, got "
            f"{tuple(keypoint_scores.shape)} and {tuple(matchability_scores.shape)}"
        )
    proposals = simple_nms(keypoint_scores, nms_radius)
    combined = torch.sqrt(
        (keypoint_scores * matchability_scores).clamp_min(0.0)
    )
    return torch.where(proposals > 0, combined, torch.zeros_like(combined))


def select_keypoints(heatmap, count, nms_radius):
    scores = simple_nms(heatmap, nms_radius).reshape(-1)
    count = min(max(int(count), 0), int(scores.numel()))
    if count == 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return empty, scores.new_empty(0)
    values, keypoint_ids = torch.topk(scores, count)
    keypoint_ids = keypoint_ids[values > 0].sort().values
    return keypoint_ids, scores[keypoint_ids]


def dual_softmax(score_matrix, temperature=1.0):
    scaled = score_matrix / max(float(temperature), 1e-6)
    return torch.softmax(scaled, dim=-2) * torch.softmax(scaled, dim=-1)


def build_score_matrix(
    query_features,
    landmark_features,
    *,
    normalize=True,
    use_dual_softmax=False,
    dual_softmax_temperature=0.1,
):
    if normalize:
        query_features = F.normalize(query_features, dim=-1)
        landmark_features = F.normalize(landmark_features, dim=-1)
    similarity = torch.matmul(query_features, landmark_features.transpose(-1, -2))
    score_matrix = (
        dual_softmax(similarity, dual_softmax_temperature)
        if bool(use_dual_softmax)
        else similarity
    )
    return similarity, score_matrix


def match_score_matrix(score_matrix, mode="topk", topk=1, threshold=0.0):
    if score_matrix.dim() != 2:
        raise ValueError(f"score_matrix must be 2D, got {tuple(score_matrix.shape)}")
    if score_matrix.numel() == 0:
        empty_idx = torch.empty(0, dtype=torch.long, device=score_matrix.device)
        return SparseMatchResult(empty_idx, empty_idx, score_matrix.new_empty(0))
    if mode == "mnn":
        keep = (
            (score_matrix > float(threshold))
            & (score_matrix == score_matrix.max(dim=1, keepdim=True).values)
            & (score_matrix == score_matrix.max(dim=0, keepdim=True).values)
        )
        keypoint_idx, landmark_idx = torch.where(keep)
        return SparseMatchResult(
            keypoint_idx,
            landmark_idx,
            score_matrix[keypoint_idx, landmark_idx],
        )
    if mode != "topk":
        raise ValueError(f"Unknown sparse match mode: {mode}")
    topk = min(max(int(topk), 1), int(score_matrix.shape[1]))
    values, landmark_idx = torch.topk(score_matrix, topk, dim=1)
    keypoint_idx = torch.arange(score_matrix.shape[0], device=score_matrix.device)
    keypoint_idx = keypoint_idx[:, None].expand_as(landmark_idx)
    keep = values > float(threshold)
    return SparseMatchResult(keypoint_idx[keep], landmark_idx[keep], values[keep])


def _matches_per_group_mask(scores, group_ids, max_matches, group_name):
    max_matches = int(max_matches)
    if max_matches <= 0:
        return torch.ones(group_ids.numel(), dtype=torch.bool, device=group_ids.device)
    if group_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=group_ids.device)
    if bool((group_ids < 0).any()):
        raise ValueError(f"{group_name} indices must be non-negative")

    count = int(group_ids.numel())
    group_count = int(group_ids.max().item()) + 1
    positions = torch.arange(count, device=group_ids.device)

    active = torch.ones(count, dtype=torch.bool, device=group_ids.device)
    keep = torch.zeros_like(active)
    for _ in range(min(max_matches, count)):
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
        has_best_score = active & (scores == best_scores[group_ids])
        first_best = positions.new_full((group_count,), count)
        first_best.scatter_reduce_(
            0,
            group_ids,
            torch.where(has_best_score, positions, positions.new_full((), count)),
            reduce="amin",
            include_self=True,
        )
        selected = has_best_score & (positions == first_best[group_ids])
        keep |= selected
        active &= ~selected
    return keep


def _limit_matches_per_group(matches, group_ids, max_matches, group_name):
    keep = _matches_per_group_mask(
        matches.scores,
        group_ids,
        max_matches,
        group_name,
    )
    return SparseMatchResult(
        matches.keypoint_idx[keep],
        matches.landmark_idx[keep],
        matches.scores[keep],
    )


def limit_matches_per_landmark(matches, max_matches):
    """Keep at most ``max_matches`` highest-scoring query pairs per landmark."""
    return _limit_matches_per_group(
        matches,
        matches.landmark_idx,
        max_matches,
        "landmark",
    )


def matches_per_landmark_mask(landmark_idx, scores, max_matches):
    """Return the exact quota mask used by ``limit_matches_per_landmark``."""
    scores = torch.as_tensor(scores)
    landmark_idx = torch.as_tensor(landmark_idx, device=scores.device, dtype=torch.long)
    if landmark_idx.numel() != scores.numel():
        raise ValueError("landmark indices and scores must have equal lengths")
    return _matches_per_group_mask(
        scores,
        landmark_idx,
        max_matches,
        "landmark",
    )


def limit_matches_per_keypoint(matches, max_matches):
    """Keep at most ``max_matches`` highest-scoring landmark hypotheses per query point."""
    return _limit_matches_per_group(
        matches,
        matches.keypoint_idx,
        max_matches,
        "keypoint",
    )


def deduplicate_landmark_matches(matches):
    """Keep only the highest-scoring query pair for each 3D landmark."""
    return limit_matches_per_landmark(matches, 1)


def match_candidate_selection_mask(
    matches,
    *,
    threshold=-float("inf"),
    max_matches_per_keypoint=0,
    max_matches_per_landmark=0,
    min_match_count=0,
    refill_trigger_count=0,
    max_match_count=0,
):
    """Return the exact keep mask used by the final sparse frontend.

    The mask indexes the input ``matches``. Quotas are applied sequentially in
    the same order as evaluation, before score rejection and optional refill.
    """
    count = int(matches.scores.numel())
    keep_after_quota = torch.ones(
        count, dtype=torch.bool, device=matches.scores.device
    )
    keypoint_keep = _matches_per_group_mask(
        matches.scores,
        matches.keypoint_idx,
        max_matches_per_keypoint,
        "keypoint",
    )
    keep_after_quota &= keypoint_keep
    keypoint_limited_idx = torch.nonzero(
        keep_after_quota, as_tuple=False
    ).reshape(-1)
    landmark_keep = _matches_per_group_mask(
        matches.scores[keypoint_limited_idx],
        matches.landmark_idx[keypoint_limited_idx],
        max_matches_per_landmark,
        "landmark",
    )
    quota_mask = torch.zeros_like(keep_after_quota)
    quota_mask[keypoint_limited_idx[landmark_keep]] = True

    limited_idx = torch.nonzero(quota_mask, as_tuple=False).reshape(-1)
    if limited_idx.numel() == 0:
        return quota_mask
    limited_scores = matches.scores[limited_idx]
    limited_keep = limited_scores > float(threshold)
    target_count = min(max(int(min_match_count), 0), int(limited_scores.numel()))
    accepted_count = int(limited_keep.sum().item())
    trigger_count = max(int(refill_trigger_count), 0) or target_count
    if accepted_count < trigger_count and accepted_count < target_count:
        refill = torch.topk(limited_scores, target_count).indices
        limited_keep = limited_keep.clone()
        limited_keep[refill] = True
    max_match_count = max(int(max_match_count), 0)
    accepted_count = int(limited_keep.sum().item())
    if max_match_count > 0 and accepted_count > max_match_count:
        accepted_idx = torch.nonzero(
            limited_keep, as_tuple=False
        ).reshape(-1)
        top_idx = torch.topk(
            limited_scores[accepted_idx], max_match_count
        ).indices
        capped_keep = torch.zeros_like(limited_keep)
        capped_keep[accepted_idx[top_idx]] = True
        limited_keep = capped_keep
    result = torch.zeros_like(quota_mask)
    result[limited_idx[limited_keep]] = True
    return result


def select_match_candidates(
    matches,
    *,
    threshold=-float("inf"),
    max_matches_per_keypoint=0,
    max_matches_per_landmark=0,
    min_match_count=0,
    refill_trigger_count=0,
    max_match_count=0,
):
    """Apply score rejection and a landmark quota with an optional count floor."""
    keep = match_candidate_selection_mask(
        matches,
        threshold=threshold,
        max_matches_per_keypoint=max_matches_per_keypoint,
        max_matches_per_landmark=max_matches_per_landmark,
        min_match_count=min_match_count,
        refill_trigger_count=refill_trigger_count,
        max_match_count=max_match_count,
    )
    return SparseMatchResult(
        matches.keypoint_idx[keep],
        matches.landmark_idx[keep],
        matches.scores[keep],
    )


def select_offset_only_candidates(
    matches,
    *,
    threshold=-float("inf"),
    max_matches_per_keypoint=0,
    max_matches_per_landmark=0,
    min_match_count=0,
    refill_trigger_count=0,
):
    """Select the deployed candidate graph before applying a side offset head.

    The offset-only path is intentionally unable to replace cosine scores with
    learned pair scores: doing so changes the landmark quota and turns a
    measurement refinement into an untracked candidate selector.
    """
    return select_match_candidates(
        matches,
        threshold=threshold,
        max_matches_per_keypoint=max_matches_per_keypoint,
        max_matches_per_landmark=max_matches_per_landmark,
        min_match_count=min_match_count,
        refill_trigger_count=refill_trigger_count,
    )


def select_match_candidates_with_geometry_refill(
    matches,
    keypoint_xy,
    landmark_xyz,
    image_size,
    *,
    threshold=-float("inf"),
    max_matches_per_keypoint=0,
    max_matches_per_landmark=0,
    min_match_count=0,
    refill_trigger_count=0,
    max_match_count=0,
    grid_rows=4,
    grid_cols=4,
    voxel_size=0.25,
    spatial_weight=0.25,
    voxel_weight=0.25,
):
    """Refill a confidence-filtered set while preserving 2D/3D diversity."""
    limited = limit_matches_per_keypoint(matches, max_matches_per_keypoint)
    limited = limit_matches_per_landmark(limited, max_matches_per_landmark)
    if limited.scores.numel() == 0:
        return limited

    finite_scores = torch.isfinite(limited.scores)
    keep = finite_scores & (limited.scores > float(threshold))
    fixed_count = min(max(int(max_match_count), 0), int(finite_scores.sum().item()))
    target_count = min(max(int(min_match_count), 0), int(limited.scores.numel()))
    if fixed_count > 0:
        target_count = fixed_count
        # Fixed-budget Pair selection is a fresh geometry-aware ranking over all
        # finite candidates. Keeping every candidate at threshold=-inf would
        # otherwise bypass both the budget and the diversity refill.
        keep = torch.zeros_like(keep)
    accepted_count = int(keep.sum().item())
    trigger_count = max(int(refill_trigger_count), 0) or target_count
    if fixed_count <= 0 and (
        accepted_count >= trigger_count or accepted_count >= target_count
    ):
        return SparseMatchResult(
            limited.keypoint_idx[keep],
            limited.landmark_idx[keep],
            limited.scores[keep],
        )

    keypoint_xy = torch.as_tensor(
        keypoint_xy, device=limited.scores.device, dtype=limited.scores.dtype
    ).reshape(-1, 2)
    landmark_xyz = torch.as_tensor(
        landmark_xyz, device=limited.scores.device, dtype=limited.scores.dtype
    ).reshape(-1, 3)
    if int(limited.keypoint_idx.max().item()) >= keypoint_xy.shape[0]:
        raise ValueError("geometry refill keypoint index is out of range")
    if int(limited.landmark_idx.max().item()) >= landmark_xyz.shape[0]:
        raise ValueError("geometry refill landmark index is out of range")

    height, width = (int(image_size[0]), int(image_size[1]))
    grid_rows = max(int(grid_rows), 1)
    grid_cols = max(int(grid_cols), 1)
    pair_xy = keypoint_xy[limited.keypoint_idx]
    grid_x = torch.floor(pair_xy[:, 0] * grid_cols / max(width, 1)).long()
    grid_y = torch.floor(pair_xy[:, 1] * grid_rows / max(height, 1)).long()
    grid_x = grid_x.clamp(0, grid_cols - 1)
    grid_y = grid_y.clamp(0, grid_rows - 1)
    grid_ids = grid_y * grid_cols + grid_x

    voxel_size = max(float(voxel_size), 1e-6)
    voxel_coordinates = torch.floor(
        landmark_xyz[limited.landmark_idx] / voxel_size
    ).long()
    _, voxel_ids = torch.unique(
        voxel_coordinates, dim=0, return_inverse=True
    )
    grid_counts = torch.bincount(
        grid_ids[keep], minlength=grid_rows * grid_cols
    ).to(dtype=limited.scores.dtype)
    voxel_counts = torch.bincount(
        voxel_ids[keep], minlength=int(voxel_ids.max().item()) + 1
    ).to(dtype=limited.scores.dtype)

    normalized_scores = torch.zeros_like(limited.scores)
    if bool(finite_scores.any()):
        finite_values = limited.scores[finite_scores]
        score_min = finite_values.min()
        score_range = (finite_values.max() - score_min).clamp_min(1e-6)
        normalized_scores[finite_scores] = (
            limited.scores[finite_scores] - score_min
        ) / score_range

    refill_count = target_count - accepted_count
    for _ in range(refill_count):
        priority = (
            normalized_scores
            + float(spatial_weight)
            / torch.sqrt(1.0 + grid_counts[grid_ids])
            + float(voxel_weight)
            / torch.sqrt(1.0 + voxel_counts[voxel_ids])
        )
        priority = priority.masked_fill(keep | ~finite_scores, -torch.inf)
        selected = int(torch.argmax(priority).item())
        if not bool(torch.isfinite(priority[selected]).item()):
            break
        keep[selected] = True
        grid_counts[grid_ids[selected]] += 1.0
        voxel_counts[voxel_ids[selected]] += 1.0

    return SparseMatchResult(
        limited.keypoint_idx[keep],
        limited.landmark_idx[keep],
        limited.scores[keep],
    )


def gather_aligned_pair_values(source_matches, target_matches, values, landmark_count):
    """Gather per-pair values after score filtering/quota selection."""
    if values.shape[0] != source_matches.keypoint_idx.numel():
        raise ValueError("pair value count must match source matches")
    if target_matches.keypoint_idx.numel() == 0:
        return values[:0]
    if source_matches.keypoint_idx.numel() == 0:
        raise ValueError("target matches are not a subset of empty source matches")
    landmark_count = int(landmark_count)
    if landmark_count <= 0:
        raise ValueError("landmark_count must be positive")
    source_ids = source_matches.keypoint_idx.long() * landmark_count
    source_ids = source_ids + source_matches.landmark_idx.long()
    if torch.unique(source_ids).numel() != source_ids.numel():
        raise ValueError("source matches contain duplicate query-landmark pairs")
    target_ids = target_matches.keypoint_idx.long() * landmark_count
    target_ids = target_ids + target_matches.landmark_idx.long()
    sorted_ids, order = torch.sort(source_ids)
    positions = torch.searchsorted(sorted_ids, target_ids)
    valid = positions < sorted_ids.numel()
    safe_positions = positions.clamp_max(max(sorted_ids.numel() - 1, 0))
    valid = valid & (sorted_ids[safe_positions] == target_ids)
    if not bool(valid.all()):
        raise ValueError("target matches are not a subset of source matches")
    return values[order[safe_positions]]


def build_pair_context_features(
    similarity,
    detector_scores,
    matches,
    *,
    context_topk=8,
    entropy_temperature=0.1,
):
    """Build inference-available context for query-conditioned pair scoring."""
    if similarity.dim() != 2:
        raise ValueError(f"similarity must be 2D, got {tuple(similarity.shape)}")
    if detector_scores.reshape(-1).shape[0] != similarity.shape[0]:
        raise ValueError("detector score count must match similarity rows")
    pair_count = int(matches.keypoint_idx.numel())
    if pair_count == 0:
        return similarity.new_empty((0, 6))

    keypoint_idx = matches.keypoint_idx
    landmark_idx = matches.landmark_idx
    pair_similarity = similarity[keypoint_idx, landmark_idx]
    detached = similarity.detach()
    neighbor_count = min(max(int(context_topk), 1), int(similarity.shape[1]))
    neighbor_values, neighbor_idx = torch.topk(
        detached,
        neighbor_count,
        dim=1,
    )
    if neighbor_count > 1:
        best_is_pair = neighbor_idx[:, 0][keypoint_idx] == landmark_idx
        competitor = torch.where(
            best_is_pair,
            neighbor_values[:, 1][keypoint_idx],
            neighbor_values[:, 0][keypoint_idx],
        )
        row_margin = pair_similarity.detach() - competitor
    else:
        row_margin = torch.zeros_like(pair_similarity)

    temperature = max(float(entropy_temperature), 1e-6)
    neighbor_probability = torch.softmax(neighbor_values / temperature, dim=1)
    if neighbor_count > 1:
        row_entropy = -torch.sum(
            neighbor_probability * torch.log(neighbor_probability.clamp_min(1e-8)),
            dim=1,
        ) / torch.log(similarity.new_tensor(float(neighbor_count)))
    else:
        row_entropy = similarity.new_zeros(similarity.shape[0])

    column_best = detached.max(dim=0).values
    column_gap = pair_similarity.detach() - column_best[landmark_idx]
    duplicate_count = torch.bincount(
        landmark_idx,
        minlength=similarity.shape[1],
    ).to(dtype=similarity.dtype)
    duplicate_pressure = (
        torch.log1p((duplicate_count[landmark_idx] - 1.0).clamp_min(0.0))
        / torch.log(similarity.new_tensor(16.0))
    ).clamp(0.0, 1.0)

    return torch.stack(
        [
            pair_similarity,
            detector_scores.reshape(-1)[keypoint_idx].detach(),
            row_margin,
            row_entropy[keypoint_idx],
            column_gap,
            duplicate_pressure,
        ],
        dim=1,
    )
