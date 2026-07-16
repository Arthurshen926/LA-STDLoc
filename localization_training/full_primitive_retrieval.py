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
