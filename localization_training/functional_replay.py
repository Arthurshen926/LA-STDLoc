from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class FunctionalReplayOutput:
    loss: torch.Tensor
    margin_loss: torch.Tensor
    distribution_loss: torch.Tensor
    retained: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


def protected_functional_replay_loss(
    bank_features: torch.Tensor,
    query_features: torch.Tensor,
    protected_landmark_indices: torch.Tensor,
    reference_candidate_indices: torch.Tensor,
    reference_candidate_logits: torch.Tensor,
    reference_margins: torch.Tensor,
    *,
    importance: Optional[torch.Tensor] = None,
    temperature: float = 0.05,
    margin_slack: float = 0.005,
    distribution_weight: float = 1.0,
) -> FunctionalReplayOutput:
    """Preserve frozen query-to-map behavior instead of descriptor distance."""
    if bank_features.ndim != 2 or query_features.ndim != 2:
        raise ValueError("bank_features and query_features must be matrices")
    if int(bank_features.shape[1]) != int(query_features.shape[1]):
        raise ValueError("bank and query feature dimensions must agree")
    row_count = int(query_features.shape[0])
    protected = torch.as_tensor(
        protected_landmark_indices,
        device=bank_features.device,
        dtype=torch.long,
    ).reshape(-1)
    candidates = torch.as_tensor(
        reference_candidate_indices,
        device=bank_features.device,
        dtype=torch.long,
    )
    reference_logits = torch.as_tensor(
        reference_candidate_logits,
        device=bank_features.device,
        dtype=bank_features.dtype,
    )
    reference_margins = torch.as_tensor(
        reference_margins,
        device=bank_features.device,
        dtype=bank_features.dtype,
    ).reshape(-1)
    if protected.numel() != row_count or reference_margins.numel() != row_count:
        raise ValueError("protected IDs and margins must have one value per row")
    if candidates.ndim != 2 or candidates.shape[0] != row_count:
        raise ValueError("reference candidates must have one row per query")
    if reference_logits.shape != candidates.shape:
        raise ValueError("reference candidate logits must match candidate IDs")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if float(margin_slack) < 0.0:
        raise ValueError("margin_slack must be non-negative")
    if row_count == 0:
        zero = bank_features.sum() * 0.0
        return FunctionalReplayOutput(
            zero, zero, zero, protected.new_empty((0,), dtype=torch.bool)
        )

    query = F.normalize(query_features.detach(), dim=1)
    bank = F.normalize(bank_features, dim=1)
    scores = query @ bank.T
    rows = torch.arange(row_count, device=scores.device)
    protected_scores = scores[rows, protected]
    competitor_scores = scores.clone()
    competitor_scores[rows, protected] = -torch.inf
    strongest_competitor = competitor_scores.max(dim=1).values
    current_margins = protected_scores - strongest_competitor
    margin_floor = reference_margins - float(margin_slack)
    # This is a functional floor, not another margin-maximization objective.
    # The frozen reference state must be feasible with zero protection
    # gradient; otherwise replay itself drifts already-correct candidates.
    per_row_margin = F.relu(
        (margin_floor - current_margins) / float(temperature)
    )

    current_candidate_logits = scores.gather(1, candidates)
    reference_probability = F.softmax(
        reference_logits.detach() / float(temperature), dim=1
    )
    per_row_distribution = F.kl_div(
        F.log_softmax(current_candidate_logits / float(temperature), dim=1),
        reference_probability,
        reduction="none",
    ).sum(dim=1)

    if importance is None:
        weights = torch.ones_like(per_row_margin)
    else:
        weights = torch.as_tensor(
            importance, device=scores.device, dtype=scores.dtype
        ).reshape(-1)
        if weights.numel() != row_count:
            raise ValueError("importance must have one value per replay row")
        weights = weights.clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-8)
    margin_loss = (weights * per_row_margin).sum()
    distribution_loss = (weights * per_row_distribution).sum()
    loss = margin_loss + float(distribution_weight) * distribution_loss
    deployed_top1 = scores.argmax(dim=1)
    retained = deployed_top1 == protected
    diagnostics = {
        "functional_replay_row_count": row_count,
        "functional_replay_loss": float(loss.detach().item()),
        "functional_replay_margin_loss": float(margin_loss.detach().item()),
        "functional_replay_distribution_loss": float(
            distribution_loss.detach().item()
        ),
        "functional_replay_retained_count": int(retained.sum().item()),
        "functional_replay_retention": float(
            retained.float().mean().detach().item()
        ),
        "functional_replay_margin_mean": float(
            current_margins.mean().detach().item()
        ),
        "functional_replay_margin_floor_mean": float(
            margin_floor.mean().detach().item()
        ),
    }
    return FunctionalReplayOutput(
        loss,
        margin_loss,
        distribution_loss,
        retained.detach(),
        diagnostics,
    )


def per_landmark_gradient_conflict(
    promotion_gradient: torch.Tensor,
    protection_gradient: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project promotion gradients away from conflicting protected directions."""
    if promotion_gradient.shape != protection_gradient.shape:
        raise ValueError("promotion and protection gradients must match")
    if promotion_gradient.ndim < 2:
        raise ValueError("gradients must retain a landmark dimension")
    promotion = promotion_gradient.reshape(promotion_gradient.shape[0], -1)
    protection = protection_gradient.reshape(protection_gradient.shape[0], -1)
    dot = (promotion * protection).sum(dim=1)
    protection_norm_squared = protection.square().sum(dim=1)
    conflict = (dot < 0.0) & (protection_norm_squared > float(epsilon))
    coefficient = torch.zeros_like(dot)
    coefficient[conflict] = (
        dot[conflict] / protection_norm_squared[conflict].clamp_min(epsilon)
    )
    projected = promotion - coefficient[:, None] * protection
    return projected.reshape_as(promotion_gradient), conflict
