"""Reversible safety actions for the clean-Anchor detector mainline.

These helpers only produce proposals.  They never delete Gaussians, copy query
descriptors into the map, or mutate the input state in place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CounterfactualGate:
    minimum_improving_queries: int = 2
    minimum_pose_families: int = 2
    minimum_task_improvement: float = 0.05
    maximum_worsening_fraction: float = 0.25


def certified_feedback_row_mask(certificate: dict) -> torch.Tensor:
    """Only ACCEPT/V2-valid detector rows may create feedback evidence."""

    valid = torch.as_tensor(certificate.get("row_valid", ())).bool().reshape(-1)
    if certificate.get("decision") != "ACCEPT":
        return torch.zeros_like(valid)
    if certificate.get("can_drive_map_update") is not True:
        raise ValueError("ACCEPT certificate has inconsistent update permission")
    return valid.clone()


def propose_anchor_quarantine(
    *,
    anchor_count: int,
    harmful_anchor_rows: torch.Tensor,
    pose_family_ids: torch.Tensor,
    minimum_pose_families: int = 2,
) -> dict:
    """Propose, but do not apply, reversible Anchor deactivation."""

    rows = torch.as_tensor(harmful_anchor_rows).long().reshape(-1)
    families = torch.as_tensor(pose_family_ids).long().reshape(-1)
    if rows.shape != families.shape:
        raise ValueError("harmful Anchor evidence and pose families must align")
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= int(anchor_count)):
        raise ValueError("harmful Anchor row is out of range")
    support = torch.zeros(int(anchor_count), dtype=torch.long)
    for row in torch.unique(rows).tolist():
        support[row] = torch.unique(families[rows == row]).numel()
    proposed = support >= int(minimum_pose_families)
    return {
        "schema": "lafgs_v8_anchor_quarantine_proposal", "version": 1,
        "proposed_quarantine": proposed,
        "supporting_pose_family_count": support,
        "reversible": True, "map_mutation_count": 0,
        "feedback_descriptors_copied": False,
    }


def evaluate_counterfactual_gaussian_action(
    *,
    baseline_task_error: torch.Tensor,
    cleaned_task_error: torch.Tensor,
    pose_family_ids: torch.Tensor,
    gate: CounterfactualGate = CounterfactualGate(),
) -> dict:
    """Gate temporary Gaussian suppression using paired pose evidence.

    The renderer is expected to restore opacity after producing the cleaned
    render.  A PASS authorizes a reversible quarantine proposal only; it never
    authorizes permanent Gaussian deletion.
    """

    baseline = torch.as_tensor(baseline_task_error).float().reshape(-1)
    cleaned = torch.as_tensor(cleaned_task_error).float().reshape(-1)
    families = torch.as_tensor(pose_family_ids).long().reshape(-1)
    if baseline.shape != cleaned.shape or baseline.shape != families.shape or baseline.numel() == 0:
        raise ValueError("paired counterfactual rows must be non-empty and aligned")
    if not bool(torch.isfinite(baseline).all() & torch.isfinite(cleaned).all()):
        raise ValueError("counterfactual task errors must be finite")
    improvement = baseline - cleaned
    improving = improvement >= float(gate.minimum_task_improvement)
    worsening = improvement < 0
    improving_families = torch.unique(families[improving]).numel()
    worsening_fraction = float(worsening.float().mean())
    accepted = (
        int(improving.sum()) >= int(gate.minimum_improving_queries)
        and int(improving_families) >= int(gate.minimum_pose_families)
        and worsening_fraction <= float(gate.maximum_worsening_fraction)
        and float(torch.median(improvement)) >= 0.0
    )
    return {
        "schema": "lafgs_v8_counterfactual_gaussian_action", "version": 1,
        "decision": "PASS" if accepted else "ROLLBACK",
        "paired_query_count": int(baseline.numel()),
        "improving_query_count": int(improving.sum()),
        "improving_pose_family_count": int(improving_families),
        "worsening_fraction": worsening_fraction,
        "median_task_improvement": float(torch.median(improvement)),
        "reversible_quarantine_only": True,
        "permanent_gaussian_deletion_authorized": False,
        "map_mutation_count": 0,
    }
