"""Confusion-directed objectives for bounded shared descriptor adaptation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def candidate_margin_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Rank one or more legal candidates above every illegal candidate."""

    scores = torch.as_tensor(scores)
    positive_mask = torch.as_tensor(
        positive_mask, device=scores.device, dtype=torch.bool
    )
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must have the same 2D shape")
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("every row must contain a positive candidate")
    if bool(positive_mask.all(dim=1).any()):
        raise ValueError("every row must contain a negative candidate")
    positive = scores.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
    negative = scores.masked_fill(positive_mask, -torch.inf).max(dim=1).values
    return F.softplus(
        (negative - positive + float(margin)) / float(temperature)
    ).mean()


def topk_distribution_distillation(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Preserve the complete local top-K distribution for neutral rows."""

    student_scores = torch.as_tensor(student_scores)
    teacher_scores = torch.as_tensor(
        teacher_scores,
        device=student_scores.device,
        dtype=student_scores.dtype,
    )
    if student_scores.ndim != 2 or teacher_scores.shape != student_scores.shape:
        raise ValueError("student and teacher top-K scores must align")
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    teacher_probability = torch.softmax(teacher_scores / temperature, dim=1)
    return (
        F.kl_div(
            torch.log_softmax(student_scores / temperature, dim=1),
            teacher_probability,
            reduction="batchmean",
        )
        * temperature**2
    )


def protected_top1_mask(
    row_count: int, candidate_count: int, *, device=None
) -> torch.Tensor:
    """Build the identity-preserving target used for already-clean top-1 rows."""

    if row_count < 0 or candidate_count < 2:
        raise ValueError("protected targets require rows and at least two candidates")
    output = torch.zeros(
        (int(row_count), int(candidate_count)),
        dtype=torch.bool,
        device=device,
    )
    output[:, 0] = True
    return output


def select_stratified_rows(
    mask: torch.Tensor,
    *,
    maximum: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Deterministically subsample one query without favoring cache order."""

    rows = torch.nonzero(torch.as_tensor(mask).bool(), as_tuple=False).reshape(-1)
    if maximum <= 0 or rows.numel() <= int(maximum):
        return rows
    order = torch.randperm(rows.numel(), generator=generator)[: int(maximum)]
    return rows[order]
