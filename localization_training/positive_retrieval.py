"""Vectorized helpers for strict-positive retrieval audits."""

from __future__ import annotations

import torch


RANK_BUCKET_NAMES = (
    "rank_1_16",
    "rank_17_64",
    "rank_gt_64",
    "no_active_anchor",
    "no_canonical_anchor_or_invalid",
)


def csr_candidate_positive_mask(
    candidates: torch.Tensor,
    offsets: torch.Tensor,
    positive_indices: torch.Tensor,
) -> torch.Tensor:
    """Test candidate membership in one strict-positive CSR row at a time."""

    candidates = torch.as_tensor(candidates).long()
    offsets = torch.as_tensor(offsets).long()
    positive_indices = torch.as_tensor(positive_indices).long()
    if candidates.ndim != 2 or offsets.numel() != len(candidates) + 1:
        raise ValueError("positive CSR offsets do not align with candidates")
    if not positive_indices.numel():
        return torch.zeros_like(candidates, dtype=torch.bool)
    stride = int(max(int(candidates.max()), int(positive_indices.max())) + 1)
    positive_rows = torch.repeat_interleave(
        torch.arange(len(candidates), dtype=torch.long),
        offsets[1:] - offsets[:-1],
    )
    positive_keys = torch.sort(
        positive_rows * stride + positive_indices
    ).values
    candidate_keys = (
        torch.arange(len(candidates), dtype=torch.long)[:, None] * stride
        + candidates
    )
    flat = candidate_keys.reshape(-1)
    slot = torch.searchsorted(positive_keys, flat)
    valid = slot < positive_keys.numel()
    matched = torch.zeros_like(valid)
    matched[valid] = positive_keys[slot[valid]] == flat[valid]
    return matched.reshape_as(candidates)


def strict_positive_rank_buckets(
    *,
    active_offsets: torch.Tensor,
    canonical_offsets: torch.Tensor,
    top16_positive_mask: torch.Tensor,
    top64_positive_mask: torch.Tensor,
) -> torch.Tensor:
    """Separate compact-map coverage, ranking, and no-canonical-positive rows."""

    active_offsets = torch.as_tensor(active_offsets).long()
    canonical_offsets = torch.as_tensor(canonical_offsets).long()
    top16_positive_mask = torch.as_tensor(top16_positive_mask).bool()
    top64_positive_mask = torch.as_tensor(top64_positive_mask).bool()
    row_count = top16_positive_mask.shape[0]
    if (
        active_offsets.numel() != row_count + 1
        or canonical_offsets.numel() != row_count + 1
        or top64_positive_mask.shape[0] != row_count
    ):
        raise ValueError("rank-bucket inputs do not align")
    active = active_offsets[1:] > active_offsets[:-1]
    canonical = canonical_offsets[1:] > canonical_offsets[:-1]
    in16 = top16_positive_mask.any(dim=1)
    in64 = top64_positive_mask.any(dim=1)
    output = torch.full((row_count,), 4, dtype=torch.long)
    output[~active & canonical] = 3
    output[active & ~in64] = 2
    output[active & in64 & ~in16] = 1
    output[active & in16] = 0
    return output
