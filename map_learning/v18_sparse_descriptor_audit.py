"""Optional per-Anchor descriptor controllability audit.

This is deliberately not a deployment controller.  It asks whether a single
descriptor constrained to the convex hull of original mapping observations can
repair multiple pose families without breaking a protected correct winner.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


def audit_observation_convex_descriptor(
    *,
    anchor_row: int,
    native_descriptor: torch.Tensor,
    observation_descriptors: torch.Tensor,
    repair_query_descriptors: torch.Tensor,
    repair_competitor_scores: torch.Tensor,
    repair_pose_family_ids: torch.Tensor,
    repair_weights: torch.Tensor,
    protection_query_descriptors: torch.Tensor,
    protection_competitor_scores: torch.Tensor,
    maximum_angle_deg: float = 5.0,
    desired_margin: float = 0.01,
    steps: int = 300,
    learning_rate: float = 0.05,
) -> dict:
    """Optimize and audit one observation-convex map descriptor proposal."""

    native = functional.normalize(
        torch.as_tensor(native_descriptor).float().reshape(1, -1), dim=1
    )[0]
    observations = functional.normalize(
        torch.as_tensor(observation_descriptors).float().reshape(-1, native.numel()),
        dim=1,
    )
    if observations.shape[0] < 2:
        raise ValueError("descriptor controllability requires two observations")
    repair_query = functional.normalize(
        torch.as_tensor(repair_query_descriptors).float().reshape(-1, native.numel()),
        dim=1,
    )
    repair_competitor = torch.as_tensor(repair_competitor_scores).float().reshape(-1)
    families = torch.as_tensor(repair_pose_family_ids).long().reshape(-1)
    weights = torch.as_tensor(repair_weights).float().reshape(-1)
    protect_query = functional.normalize(
        torch.as_tensor(protection_query_descriptors).float().reshape(-1, native.numel()),
        dim=1,
    )
    protect_competitor = torch.as_tensor(protection_competitor_scores).float().reshape(-1)
    if not (
        repair_query.shape[0]
        == repair_competitor.numel()
        == families.numel()
        == weights.numel()
        and protect_query.shape[0] == protect_competitor.numel()
    ):
        raise ValueError("sparse descriptor audit rows do not align")
    if repair_query.shape[0] == 0:
        raise ValueError("sparse descriptor audit requires repair evidence")
    # Native is included as an additional legal observation aggregate, keeping
    # every proposal inside a bounded convex hull tied to mapping evidence.
    basis = torch.cat((observations, native[None]), dim=0)
    logits = torch.zeros(basis.shape[0], requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=float(learning_rate))
    normalized_weights = weights.clamp_min(1e-6)
    normalized_weights = normalized_weights / normalized_weights.sum()
    for _ in range(int(steps)):
        mixture = torch.softmax(logits, dim=0) @ basis
        candidate = functional.normalize(mixture[None], dim=1)[0]
        repair_similarity = repair_query @ candidate
        protection_similarity = protect_query @ candidate
        repair_loss = (
            normalized_weights
            * functional.softplus(
                (float(desired_margin) + repair_competitor - repair_similarity) / 0.02
            )
        ).sum()
        protection_loss = (
            functional.relu(
                float(desired_margin)
                + protect_competitor
                - protection_similarity
            ).mean()
            if protect_query.shape[0]
            else repair_loss.new_zeros(())
        )
        angle_excess = functional.relu(
            math.cos(math.radians(float(maximum_angle_deg)))
            - torch.dot(candidate, native)
        )
        loss = repair_loss + 2.0 * protection_loss + 100.0 * angle_excess.square()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        convex_weights = torch.softmax(logits, dim=0)
        candidate = functional.normalize((convex_weights @ basis)[None], dim=1)[0]
        cosine = float(torch.dot(candidate, native).clamp(-1.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        native_repair = repair_query @ native
        candidate_repair = repair_query @ candidate
        repaired = (native_repair <= repair_competitor) & (
            candidate_repair >= repair_competitor + float(desired_margin)
        )
        native_protection = protect_query @ native
        candidate_protection = protect_query @ candidate
        protected_before = native_protection >= protect_competitor
        broken = protected_before & (candidate_protection < protect_competitor)
        improving_families = torch.unique(families[repaired])
        authorized = bool(
            angle <= float(maximum_angle_deg) + 1e-5
            and improving_families.numel() >= 2
            and not bool(broken.any())
        )
    return {
        "schema": "lafgs_v18_sparse_descriptor_controllability_audit",
        "version": 1,
        "anchor_row": int(anchor_row),
        "proposal_only": True,
        "authorized": authorized,
        "convex_weights": convex_weights.detach(),
        "candidate_descriptor": candidate.detach(),
        "descriptor_angle_deg": angle,
        "maximum_angle_deg": float(maximum_angle_deg),
        "repair_row_count": int(repair_query.shape[0]),
        "repaired_row_count": int(repaired.sum()),
        "improving_pose_family_count": int(improving_families.numel()),
        "protection_row_count": int(protect_query.shape[0]),
        "broken_protection_row_count": int(broken.sum()),
        "requires_intervention_necessity_global_confirmation": True,
    }


__all__ = ["audit_observation_convex_descriptor"]
