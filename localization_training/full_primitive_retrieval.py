from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F


@dataclass
class RetrievalResult:
    scores: torch.Tensor
    indices: torch.Tensor
    elapsed_ms: float
    chunks: int


def chunked_exact_topk(query_features, map_features, topk=1, chunk_size=8192):
    """Exact cosine top-k without materializing the full query-map matrix."""
    if query_features.ndim != 2 or map_features.ndim != 2:
        raise ValueError("query_features and map_features must be 2D")
    if query_features.shape[1] != map_features.shape[1]:
        raise ValueError("query and map feature dimensions must match")
    count = int(map_features.shape[0])
    topk = min(max(int(topk), 1), count)
    chunk_size = max(int(chunk_size), topk)
    query = F.normalize(query_features.float(), dim=1)
    best_scores = query.new_full((query.shape[0], topk), -torch.inf)
    best_indices = torch.zeros(
        (query.shape[0], topk), dtype=torch.long, device=query.device
    )
    start_time = time.perf_counter()
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        features = F.normalize(map_features[start:stop].float(), dim=1)
        scores = query @ features.T
        indices = torch.arange(start, stop, device=query.device, dtype=torch.long)
        indices = indices[None].expand(query.shape[0], -1)
        merged_scores = torch.cat([best_scores, scores], dim=1)
        merged_indices = torch.cat([best_indices, indices], dim=1)
        best_scores, positions = torch.topk(merged_scores, topk, dim=1)
        best_indices = torch.gather(merged_indices, 1, positions)
    return RetrievalResult(
        best_scores,
        best_indices,
        (time.perf_counter() - start_time) * 1000.0,
        int(math.ceil(count / chunk_size)),
    )


def chunked_exact_topk_dual_prototype(
    query_features,
    map_features,
    secondary_features,
    secondary_mask,
    topk=1,
    chunk_size=8192,
):
    """Exact anchor top-k using the best score from either view prototype."""
    if map_features.ndim != 2 or secondary_features.shape != map_features.shape:
        raise ValueError("primary and secondary map features must align")
    secondary_mask = torch.as_tensor(
        secondary_mask, device=map_features.device, dtype=torch.bool
    ).reshape(-1)
    if secondary_mask.numel() != map_features.shape[0]:
        raise ValueError("secondary_mask must have one value per map anchor")
    if query_features.ndim != 2 or query_features.shape[1] != map_features.shape[1]:
        raise ValueError("query and map feature dimensions must match")

    count = int(map_features.shape[0])
    topk = min(max(int(topk), 1), count)
    chunk_size = max(int(chunk_size), topk)
    query = F.normalize(query_features.float(), dim=1)
    best_scores = query.new_full((query.shape[0], topk), -torch.inf)
    best_indices = torch.zeros(
        (query.shape[0], topk), dtype=torch.long, device=query.device
    )
    start_time = time.perf_counter()
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        primary = F.normalize(map_features[start:stop].float(), dim=1)
        scores = query @ primary.T
        active_secondary = secondary_mask[start:stop]
        if bool(active_secondary.any().item()):
            secondary = F.normalize(
                secondary_features[start:stop][active_secondary].float(), dim=1
            )
            secondary_scores = query @ secondary.T
            scores[:, active_secondary] = torch.maximum(
                scores[:, active_secondary], secondary_scores
            )
        indices = torch.arange(start, stop, device=query.device, dtype=torch.long)
        indices = indices[None].expand(query.shape[0], -1)
        merged_scores = torch.cat([best_scores, scores], dim=1)
        merged_indices = torch.cat([best_indices, indices], dim=1)
        best_scores, positions = torch.topk(merged_scores, topk, dim=1)
        best_indices = torch.gather(merged_indices, 1, positions)
    return RetrievalResult(
        best_scores,
        best_indices,
        (time.perf_counter() - start_time) * 1000.0,
        int(math.ceil(count / chunk_size)),
    )


def _unique_anchor_topk(candidate_scores, candidate_indices, topk):
    """Select score-sorted unique anchor IDs from a small candidate table."""
    order = torch.argsort(candidate_scores, dim=1, descending=True)
    sorted_scores = torch.gather(candidate_scores, 1, order)
    sorted_indices = torch.gather(candidate_indices, 1, order)
    output_scores = candidate_scores.new_full(
        (candidate_scores.shape[0], topk), -torch.inf
    )
    output_indices = torch.zeros(
        (candidate_scores.shape[0], topk),
        dtype=torch.long,
        device=candidate_scores.device,
    )
    filled = torch.zeros(
        candidate_scores.shape[0], dtype=torch.long, device=candidate_scores.device
    )
    selected = torch.zeros(
        (candidate_scores.shape[0], topk),
        dtype=torch.long,
        device=candidate_scores.device,
    )
    for column in range(sorted_scores.shape[1]):
        anchor = sorted_indices[:, column]
        duplicate = (
            (selected == anchor[:, None])
            & (
                torch.arange(topk, device=filled.device)[None]
                < filled[:, None]
            )
        ).any(dim=1)
        accept = (~duplicate) & (filled < topk)
        rows = torch.nonzero(accept, as_tuple=False).reshape(-1)
        if rows.numel():
            positions = filled[rows]
            output_scores[rows, positions] = sorted_scores[rows, column]
            output_indices[rows, positions] = anchor[rows]
            selected[rows, positions] = anchor[rows]
            filled[rows] += 1
        if bool((filled == topk).all()):
            break
    if not bool((filled == topk).all()):
        raise RuntimeError("family prototype retrieval did not fill unique top-k")
    return output_scores, output_indices


def chunked_exact_topk_family_prototype(
    query_features,
    map_features,
    prototype_features,
    prototype_anchor_indices,
    prototype_bias=None,
    prototype_temperature=None,
    topk=1,
    chunk_size=8192,
):
    """Exact anchor top-k after calibrated max-pooling of descriptor families."""
    if query_features.ndim != 2 or map_features.ndim != 2:
        raise ValueError("query_features and map_features must be 2D")
    if prototype_features.ndim != 2:
        raise ValueError("prototype_features must be 2D")
    if (
        query_features.shape[1] != map_features.shape[1]
        or prototype_features.shape[1] != map_features.shape[1]
    ):
        raise ValueError("query, map, and prototype dimensions must match")
    parents = torch.as_tensor(
        prototype_anchor_indices,
        device=map_features.device,
        dtype=torch.long,
    ).reshape(-1)
    if parents.numel() != prototype_features.shape[0]:
        raise ValueError("prototype rows and anchor indices must align")
    if parents.numel() and (
        int(parents.min()) < 0 or int(parents.max()) >= map_features.shape[0]
    ):
        raise ValueError("prototype anchor index is outside the map")
    bias = (
        torch.zeros(
            parents.numel(), device=map_features.device, dtype=torch.float32
        )
        if prototype_bias is None
        else torch.as_tensor(
            prototype_bias, device=map_features.device, dtype=torch.float32
        ).reshape(-1)
    )
    temperature = (
        torch.ones(
            parents.numel(), device=map_features.device, dtype=torch.float32
        )
        if prototype_temperature is None
        else torch.as_tensor(
            prototype_temperature, device=map_features.device, dtype=torch.float32
        ).reshape(-1)
    )
    if bias.numel() != parents.numel():
        raise ValueError("prototype_bias must have one value per prototype")
    if bias.numel() and bool((bias > 1e-8).any().item()):
        raise ValueError("prototype_bias must be non-positive")
    if temperature.numel() != parents.numel():
        raise ValueError("prototype_temperature must have one value per prototype")
    if temperature.numel() and bool((temperature <= 0).any().item()):
        raise ValueError("prototype_temperature must be positive")
    count = int(map_features.shape[0])
    topk = min(max(int(topk), 1), count)
    start_time = time.perf_counter()
    primary = chunked_exact_topk(
        query_features, map_features, topk=topk, chunk_size=chunk_size
    )
    if not parents.numel():
        return primary
    query = F.normalize(query_features.float(), dim=1)
    prototypes = F.normalize(prototype_features.float(), dim=1)
    prototype_scores = (
        (query @ prototypes.T) / temperature[None] + bias[None]
    )
    prototype_indices = parents[None].expand(query.shape[0], -1)
    scores, indices = _unique_anchor_topk(
        torch.cat((primary.scores, prototype_scores), dim=1),
        torch.cat((primary.indices, prototype_indices), dim=1),
        topk,
    )
    return RetrievalResult(
        scores,
        indices,
        (time.perf_counter() - start_time) * 1000.0,
        primary.chunks + 1,
    )


def chunked_exact_topk_with_bias(
    query_features,
    map_features,
    map_bias,
    topk=1,
    chunk_size=8192,
):
    """Exact cosine top-k with one query-global additive map bias."""
    map_bias = torch.as_tensor(
        map_bias, device=map_features.device, dtype=torch.float32
    ).reshape(-1)
    if map_bias.numel() != map_features.shape[0]:
        raise ValueError("map_bias must have one value per map anchor")
    if query_features.ndim != 2 or map_features.ndim != 2:
        raise ValueError("query_features and map_features must be 2D")
    count = int(map_features.shape[0])
    topk = min(max(int(topk), 1), count)
    chunk_size = max(int(chunk_size), topk)
    query = F.normalize(query_features.float(), dim=1)
    best_scores = query.new_full((query.shape[0], topk), -torch.inf)
    best_indices = torch.zeros(
        (query.shape[0], topk), dtype=torch.long, device=query.device
    )
    start_time = time.perf_counter()
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        features = F.normalize(map_features[start:stop].float(), dim=1)
        scores = query @ features.T
        scores = scores + map_bias[start:stop][None]
        indices = torch.arange(
            start, stop, device=query.device, dtype=torch.long
        )[None].expand(query.shape[0], -1)
        merged_scores = torch.cat([best_scores, scores], dim=1)
        merged_indices = torch.cat([best_indices, indices], dim=1)
        best_scores, positions = torch.topk(merged_scores, topk, dim=1)
        best_indices = torch.gather(merged_indices, 1, positions)
    return RetrievalResult(
        best_scores,
        best_indices,
        (time.perf_counter() - start_time) * 1000.0,
        int(math.ceil(count / chunk_size)),
    )


def ambiguity_gated_context_topk(
    query_features,
    map_features,
    map_bias,
    *,
    margin_threshold,
    topk=1,
    chunk_size=8192,
):
    """Apply query context only to rows ambiguous under raw cosine."""
    baseline = chunked_exact_topk(
        query_features,
        map_features,
        topk=max(int(topk), 2),
        chunk_size=chunk_size,
    )
    margin = baseline.scores[:, 0] - baseline.scores[:, 1]
    ambiguous = margin < float(margin_threshold)
    output_topk = min(max(int(topk), 1), int(map_features.shape[0]))
    scores = baseline.scores[:, :output_topk].clone()
    indices = baseline.indices[:, :output_topk].clone()
    elapsed_ms = baseline.elapsed_ms
    chunks = baseline.chunks
    if bool(ambiguous.any().item()):
        contextual = chunked_exact_topk_with_bias(
            query_features[ambiguous],
            map_features,
            map_bias,
            topk=output_topk,
            chunk_size=chunk_size,
        )
        scores[ambiguous] = contextual.scores
        indices[ambiguous] = contextual.indices
        elapsed_ms += contextual.elapsed_ms
        chunks += contextual.chunks
    return RetrievalResult(scores, indices, elapsed_ms, chunks), ambiguous, margin


def conditional_core_reserve_topk(
    query_features,
    map_features,
    core_mask,
    *,
    margin_threshold,
    topk=1,
    chunk_size=8192,
):
    """Query reserve anchors only for rows ambiguous within the core bank."""
    core_mask = torch.as_tensor(
        core_mask, device=map_features.device, dtype=torch.bool
    ).reshape(-1)
    if core_mask.numel() != map_features.shape[0]:
        raise ValueError("core_mask must have one value per map anchor")
    core_rows = torch.nonzero(core_mask, as_tuple=False).reshape(-1)
    reserve_rows = torch.nonzero(~core_mask, as_tuple=False).reshape(-1)
    if core_rows.numel() < 2:
        raise ValueError("conditional retrieval requires at least two core anchors")
    core = chunked_exact_topk(
        query_features,
        map_features[core_rows],
        topk=max(int(topk), 2),
        chunk_size=chunk_size,
    )
    core_indices = core_rows[core.indices]
    core_margin = core.scores[:, 0] - core.scores[:, 1]
    ambiguous = core_margin < float(margin_threshold)
    output_topk = min(max(int(topk), 1), int(map_features.shape[0]))
    scores = core.scores[:, :output_topk].clone()
    indices = core_indices[:, :output_topk].clone()
    elapsed_ms = core.elapsed_ms
    chunks = core.chunks
    if bool(ambiguous.any().item()) and reserve_rows.numel() > 0:
        reserve = chunked_exact_topk(
            query_features[ambiguous],
            map_features[reserve_rows],
            topk=min(output_topk, int(reserve_rows.numel())),
            chunk_size=chunk_size,
        )
        reserve_indices = reserve_rows[reserve.indices]
        merged_scores = torch.cat([scores[ambiguous], reserve.scores], dim=1)
        merged_indices = torch.cat(
            [indices[ambiguous], reserve_indices], dim=1
        )
        selected_scores, positions = torch.topk(
            merged_scores, output_topk, dim=1
        )
        scores[ambiguous] = selected_scores
        indices[ambiguous] = torch.gather(
            merged_indices, 1, positions
        )
        elapsed_ms += reserve.elapsed_ms
        chunks += reserve.chunks
    return (
        RetrievalResult(scores, indices, elapsed_ms, chunks),
        ambiguous,
        core_margin,
    )


def suppress_redundant_hypotheses(
    scores,
    indices,
    xyz,
    *,
    output_topk=1,
    voxel_size=0.05,
    source_indices=None,
    max_per_group=1,
):
    """Greedily retain descriptor-ranked hypotheses from distinct surface groups."""
    if scores.shape != indices.shape or scores.ndim != 2:
        raise ValueError("scores and indices must be equal 2D tensors")
    output_topk = min(max(int(output_topk), 1), int(scores.shape[1]))
    max_per_group = max(int(max_per_group), 1)
    xyz = torch.as_tensor(xyz, device=indices.device)
    source = None
    if source_indices is not None:
        source = torch.as_tensor(source_indices, device=indices.device, dtype=torch.long)
    out_scores = scores.new_full((scores.shape[0], output_topk), -torch.inf)
    out_indices = indices.new_zeros((indices.shape[0], output_topk))
    kept_total = 0
    for row in range(scores.shape[0]):
        voxel_counts = {}
        source_counts = {}
        kept = []
        for col in range(scores.shape[1]):
            idx = int(indices[row, col].item())
            point = xyz[idx]
            voxel = tuple(torch.floor(point / max(float(voxel_size), 1e-8)).long().tolist())
            source_id = int(source[idx].item()) if source is not None else -1
            if voxel_counts.get(voxel, 0) >= max_per_group:
                continue
            if source is not None and source_counts.get(source_id, 0) >= max_per_group:
                continue
            voxel_counts[voxel] = voxel_counts.get(voxel, 0) + 1
            if source is not None:
                source_counts[source_id] = source_counts.get(source_id, 0) + 1
            kept.append(col)
            if len(kept) == output_topk:
                break
        if kept:
            cols = torch.as_tensor(kept, device=indices.device, dtype=torch.long)
            out_scores[row, : len(kept)] = scores[row, cols]
            out_indices[row, : len(kept)] = indices[row, cols]
            kept_total += len(kept)
    return out_scores, out_indices, kept_total / max(int(scores.numel()), 1)
