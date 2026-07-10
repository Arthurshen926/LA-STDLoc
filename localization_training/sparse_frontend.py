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


def _limit_matches_per_group(matches, group_ids, max_matches, group_name):
    max_matches = int(max_matches)
    if max_matches <= 0:
        return matches
    if group_ids.numel() == 0:
        return matches
    if bool((group_ids < 0).any()):
        raise ValueError(f"{group_name} indices must be non-negative")

    count = int(group_ids.numel())
    group_count = int(group_ids.max().item()) + 1
    positions = torch.arange(count, device=group_ids.device)

    active = torch.ones(count, dtype=torch.bool, device=matches.landmark_idx.device)
    keep = torch.zeros_like(active)
    for _ in range(min(max_matches, count)):
        if not bool(active.any()):
            break
        best_scores = matches.scores.new_full((group_count,), -torch.inf)
        best_scores.scatter_reduce_(
            0,
            group_ids,
            matches.scores.masked_fill(~active, -torch.inf),
            reduce="amax",
            include_self=True,
        )
        has_best_score = active & (matches.scores == best_scores[group_ids])
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


def select_match_candidates(
    matches,
    *,
    threshold=-float("inf"),
    max_matches_per_keypoint=0,
    max_matches_per_landmark=0,
    min_match_count=0,
    refill_trigger_count=0,
):
    """Apply score rejection and a landmark quota with an optional count floor."""
    limited = limit_matches_per_keypoint(matches, max_matches_per_keypoint)
    limited = limit_matches_per_landmark(limited, max_matches_per_landmark)
    if limited.scores.numel() == 0:
        return limited
    keep = limited.scores > float(threshold)
    target_count = min(max(int(min_match_count), 0), int(limited.scores.numel()))
    accepted_count = int(keep.sum().item())
    trigger_count = max(int(refill_trigger_count), 0) or target_count
    if accepted_count < trigger_count and accepted_count < target_count:
        refill = torch.topk(limited.scores, target_count).indices
        keep = keep.clone()
        keep[refill] = True
    return SparseMatchResult(
        limited.keypoint_idx[keep],
        limited.landmark_idx[keep],
        limited.scores[keep],
    )


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
