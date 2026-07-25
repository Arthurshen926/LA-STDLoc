from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class RankBudgetOutput:
    loss: torch.Tensor
    ranks: torch.Tensor
    valid_rows: torch.Tensor
    best_positive_scores: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


def _csr_positive_rows(offsets: torch.Tensor) -> torch.Tensor:
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=offsets.device),
        counts,
    )


def csr_best_positive_scores(
    scores: torch.Tensor,
    positive_offsets: torch.Tensor,
    positive_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the best score over every row's full CSR positive set."""
    row_count = int(scores.shape[0])
    offsets = torch.as_tensor(
        positive_offsets, device=scores.device, dtype=torch.long
    ).reshape(-1)
    indices = torch.as_tensor(
        positive_indices, device=scores.device, dtype=torch.long
    ).reshape(-1)
    if offsets.numel() != row_count + 1:
        raise ValueError("positive_offsets must have one entry per score row plus one")
    if int(offsets[-1].item()) != indices.numel():
        raise ValueError("positive_offsets and positive_indices disagree")
    counts = offsets[1:] - offsets[:-1]
    valid_rows = counts > 0
    best = scores.new_full((row_count,), -torch.inf)
    if indices.numel() == 0:
        return best, valid_rows
    rows = _csr_positive_rows(offsets)
    edge_scores = scores[rows, indices]
    best.scatter_reduce_(0, rows, edge_scores, reduce="amax", include_self=True)
    return best, valid_rows


def _full_valid_negative_mask(
    query_uv: torch.Tensor,
    bank_uv: torch.Tensor,
    bank_projected: torch.Tensor,
    positive_offsets: torch.Tensor,
    positive_indices: torch.Tensor,
    *,
    negative_radius_px: float,
) -> torch.Tensor:
    """Build exact negatives while ignoring the measurement-limited near band."""
    distance_squared = (
        query_uv.float().square().sum(dim=1, keepdim=True)
        + bank_uv.float().square().sum(dim=1)[None]
        - 2.0 * (query_uv.float() @ bank_uv.float().T)
    ).clamp_min_(0.0)
    ambiguous = (
        bank_projected.bool()[None]
        & torch.isfinite(distance_squared)
        & (distance_squared < float(negative_radius_px) ** 2)
    )
    negative = ~ambiguous
    if positive_indices.numel() > 0:
        positive_rows = _csr_positive_rows(positive_offsets)
        negative[positive_rows, positive_indices] = False
    return negative


def _band_balanced_weights(
    bands: torch.Tensor,
    valid_rows: torch.Tensor,
    proportions: Sequence[float],
) -> torch.Tensor:
    proportions_tensor = torch.as_tensor(
        proportions, device=bands.device, dtype=torch.float32
    )
    if proportions_tensor.numel() != 4 or bool((proportions_tensor < 0).any()):
        raise ValueError("rank band proportions must contain four non-negative values")
    if float(proportions_tensor.sum().item()) <= 0.0:
        raise ValueError("at least one rank band proportion must be positive")
    proportions_tensor = proportions_tensor / proportions_tensor.sum()
    weights = torch.zeros_like(bands, dtype=torch.float32)
    active = []
    for band in range(4):
        rows = valid_rows & (bands == band)
        count = int(rows.sum().item())
        if count > 0 and float(proportions_tensor[band].item()) > 0.0:
            weights[rows] = proportions_tensor[band] / count
            active.append(band)
    if not active:
        return weights
    # Redistribute unavailable quota rather than shrinking the entire update.
    return weights / weights.sum().clamp_min(1e-8)


def multi_positive_rank_budget_loss(
    full_scores: torch.Tensor,
    *,
    positive_offsets: torch.Tensor,
    positive_indices: torch.Tensor,
    query_uv: torch.Tensor,
    bank_uv: torch.Tensor,
    bank_projected: torch.Tensor,
    negative_radius_px: float = 6.0,
    budgets: Sequence[int] = (1, 4, 8, 32),
    margins: Sequence[float] = (0.02, 0.02, 0.02, 0.02),
    temperature: float = 0.03,
    top1_weight: float = 0.25,
    keep_weight: float = 1.0,
    band_proportions: Sequence[float] = (0.25, 0.25, 0.30, 0.20),
    landmark_opportunities: Optional[torch.Tensor] = None,
    reference_scores: Optional[torch.Tensor] = None,
    reference_clean_weight: float = 0.0,
    reference_clean_margin: float = 0.02,
) -> RankBudgetOutput:
    """Optimize each matchable row toward its nearest deployment rank budget.

    Bands are encoded as 0=rank1, 1=rank2--4, 2=rank5--32, 3=rank>32.
    Within band 2, ranks 5--8 target @4 and ranks 9--32 target @8.
    """
    if full_scores.ndim != 2:
        raise ValueError("full_scores must be a row-by-landmark matrix")
    if tuple(int(value) for value in budgets) != (1, 4, 8, 32):
        raise ValueError("the supported rank curriculum budgets are (1, 4, 8, 32)")
    if len(margins) != 4 or any(float(value) < 0.0 for value in margins):
        raise ValueError("rank margins must contain four non-negative values")
    if float(temperature) <= 0.0:
        raise ValueError("rank temperature must be positive")
    if float(reference_clean_weight) < 0.0:
        raise ValueError("reference clean weight must be non-negative")
    if float(reference_clean_margin) < 0.0:
        raise ValueError("reference clean margin must be non-negative")
    if reference_scores is not None and reference_scores.shape != full_scores.shape:
        raise ValueError("reference_scores must match full_scores")

    offsets = torch.as_tensor(
        positive_offsets, device=full_scores.device, dtype=torch.long
    )
    indices = torch.as_tensor(
        positive_indices, device=full_scores.device, dtype=torch.long
    )
    best_positive, valid_rows = csr_best_positive_scores(
        full_scores, offsets, indices
    )
    zero = full_scores.sum() * 0.0
    if not bool(valid_rows.any().item()):
        return RankBudgetOutput(
            zero,
            torch.zeros(full_scores.shape[0], dtype=torch.long, device=full_scores.device),
            valid_rows,
            best_positive,
            {"rank_budget_matchable_count": 0},
        )

    negative = _full_valid_negative_mask(
        query_uv,
        bank_uv,
        bank_projected,
        offsets,
        indices,
        negative_radius_px=negative_radius_px,
    )
    max_budget = min(32, int(full_scores.shape[1]))
    negative_scores = full_scores.masked_fill(~negative, -torch.inf)
    top_negative = torch.topk(
        negative_scores, k=max_budget, dim=1, largest=True, sorted=True
    ).values
    ranks = 1 + (
        negative & (full_scores > best_positive[:, None].detach())
    ).sum(dim=1)
    ranks = torch.where(valid_rows, ranks, torch.zeros_like(ranks))

    # 0: clean top1; 1: rank2--4; 2: rank5--32; 3: rank>32.
    bands = torch.full_like(ranks, -1)
    bands[valid_rows & (ranks == 1)] = 0
    bands[valid_rows & (ranks >= 2) & (ranks <= 4)] = 1
    bands[valid_rows & (ranks >= 5) & (ranks <= 32)] = 2
    bands[valid_rows & (ranks > 32)] = 3

    target_budget = torch.zeros_like(ranks)
    target_budget[valid_rows & (ranks == 1)] = 1
    target_budget[valid_rows & (ranks >= 2) & (ranks <= 4)] = 1
    target_budget[valid_rows & (ranks >= 5) & (ranks <= 8)] = 4
    target_budget[valid_rows & (ranks >= 9) & (ranks <= 32)] = 8
    target_budget[valid_rows & (ranks > 32)] = 32

    per_row = full_scores.new_zeros((full_scores.shape[0],))
    margin_by_budget = {
        1: float(margins[0]),
        4: float(margins[1]),
        8: float(margins[2]),
        32: float(margins[3]),
    }
    for budget in (1, 4, 8, 32):
        rows = valid_rows & (target_budget == budget)
        boundary_index = min(budget, max_budget) - 1
        finite = rows & torch.isfinite(top_negative[:, boundary_index])
        if bool(finite.any().item()):
            per_row[finite] = F.softplus(
                (
                    margin_by_budget[budget]
                    + top_negative[finite, boundary_index]
                    - best_positive[finite]
                )
                / float(temperature)
            )
    per_row[valid_rows & (ranks == 1)] *= float(keep_weight)
    per_row[valid_rows & (ranks >= 2) & (ranks <= 4)] *= float(top1_weight)

    row_weights = _band_balanced_weights(
        bands, valid_rows, band_proportions
    ).to(dtype=per_row.dtype)
    if landmark_opportunities is not None and indices.numel() > 0:
        opportunities = torch.as_tensor(
            landmark_opportunities,
            device=full_scores.device,
            dtype=per_row.dtype,
        ).reshape(-1)
        positive_rows = _csr_positive_rows(offsets)
        edge_weights = torch.rsqrt(1.0 + opportunities[indices].clamp_min(0.0))
        row_opportunity = per_row.new_zeros(per_row.shape)
        row_opportunity.scatter_reduce_(
            0, positive_rows, edge_weights, reduce="amax", include_self=False
        )
        row_weights = row_weights * row_opportunity.clamp_min(1e-8)
        row_weights = row_weights / row_weights.sum().clamp_min(1e-8)

    curriculum_loss = (row_weights * per_row).sum()
    reference_clean_loss = zero
    reference_clean_count = 0
    reference_clean_retained_count = 0
    if reference_scores is not None and float(reference_clean_weight) > 0.0:
        reference_scores = reference_scores.detach()
        reference_positive, reference_valid = csr_best_positive_scores(
            reference_scores, offsets, indices
        )
        reference_clean = reference_valid & ~(
            negative & (reference_scores > reference_positive[:, None])
        ).any(dim=1)
        finite_reference_clean = reference_clean & torch.isfinite(
            top_negative[:, 0]
        )
        reference_clean_count = int(reference_clean.sum().item())
        reference_clean_retained_count = int(
            (reference_clean & (ranks == 1)).sum().item()
        )
        if bool(finite_reference_clean.any().item()):
            reference_clean_loss = F.softplus(
                (
                    float(reference_clean_margin)
                    + top_negative[finite_reference_clean, 0]
                    - best_positive[finite_reference_clean]
                )
                / float(temperature)
            ).mean()
    loss = curriculum_loss + float(reference_clean_weight) * reference_clean_loss
    valid_ranks = ranks[valid_rows].float()
    diagnostics = {
        "rank_budget_active": 1.0,
        "rank_budget_matchable_count": int(valid_rows.sum().item()),
        "rank_budget_loss": float(loss.detach().item()),
        "rank_budget_curriculum_loss": float(curriculum_loss.detach().item()),
        "rank_budget_reference_clean_loss": float(
            reference_clean_loss.detach().item()
        ),
        "rank_budget_reference_clean_count": reference_clean_count,
        "rank_budget_reference_clean_retained_count": (
            reference_clean_retained_count
        ),
        "rank_budget_reference_clean_retention": (
            float(reference_clean_retained_count / reference_clean_count)
            if reference_clean_count > 0
            else 0.0
        ),
        "rank_budget_mrr": float((1.0 / valid_ranks).mean().detach().item()),
        "rank_budget_rank_mean": float(valid_ranks.mean().detach().item()),
        "rank_budget_rank_median": float(valid_ranks.median().detach().item()),
        "rank_budget_band_rank1_count": int((bands == 0).sum().item()),
        "rank_budget_band_rank2_4_count": int((bands == 1).sum().item()),
        "rank_budget_band_rank5_32_count": int((bands == 2).sum().item()),
        "rank_budget_band_rank33_plus_count": int((bands == 3).sum().item()),
    }
    for budget in (1, 2, 4, 8, 16, 32):
        diagnostics[f"rank_budget_recall_at_{budget}"] = float(
            (valid_ranks <= budget).float().mean().detach().item()
        )
        diagnostics[f"rank_budget_recall_at_{budget}_count"] = int(
            (valid_ranks <= budget).sum().item()
        )
    return RankBudgetOutput(
        loss,
        ranks.detach(),
        valid_rows.detach(),
        best_positive.detach(),
        diagnostics,
    )
