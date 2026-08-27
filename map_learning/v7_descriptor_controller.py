"""Bounded observation-weight reconstruction for V7's single descriptors."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


def _angle_degrees(left: torch.Tensor, right: torch.Tensor) -> float:
    cosine = float(torch.dot(F.normalize(left, dim=0), F.normalize(right, dim=0)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _bounded_direction(
    previous: torch.Tensor, proposed: torch.Tensor, maximum_angle_deg: float
) -> torch.Tensor:
    previous = F.normalize(previous.float(), dim=0)
    proposed = F.normalize(proposed.float(), dim=0)
    angle = math.radians(_angle_degrees(previous, proposed))
    limit = math.radians(float(maximum_angle_deg))
    if angle <= limit or angle <= 1e-12:
        return proposed
    tangent = proposed - torch.dot(previous, proposed) * previous
    if float(tangent.norm()) <= 1e-12:
        return previous
    tangent = F.normalize(tangent, dim=0)
    return F.normalize(previous * math.cos(limit) + tangent * math.sin(limit), dim=0)


def reconstruct_v7_descriptors(
    *,
    anchor_ids: torch.Tensor,
    current_descriptors: torch.Tensor,
    feedback_evidence: Sequence[Mapping[str, Any]],
    observation_banks: Mapping[int, Mapping[str, torch.Tensor]],
    minimum_pose_families: int = 2,
    learning_rate: float = 0.35,
    harmful_weight: float = 1.0,
    minimum_relative_weight: float = 0.25,
    maximum_relative_weight: float = 4.0,
    trim_fraction: float = 0.10,
    maximum_descriptor_angle_deg: float = 5.0,
) -> dict[str, Any]:
    """Reconstruct descriptors only from original mapping observations.

    Feedback descriptors affect scalar observation weights and are never
    inserted into the output descriptor bank.
    """

    ids = torch.as_tensor(anchor_ids).long().cpu().reshape(-1)
    source = torch.as_tensor(current_descriptors).float().cpu()
    current = F.normalize(source, dim=1)
    if current.shape[0] != ids.numel() or current.ndim != 2:
        raise ValueError("V7 descriptor registry is not aligned")
    if torch.unique(ids).numel() != ids.numel():
        raise ValueError("V7 Anchor IDs must be unique")
    id_to_row = {int(anchor_id): row for row, anchor_id in enumerate(ids.tolist())}
    positive: dict[int, dict[int, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    harmful: dict[int, dict[int, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in feedback_evidence:
        family = int(item["pose_family_id"])
        descriptors = F.normalize(
            torch.as_tensor(item["query_descriptors"]).float().cpu(), dim=1
        )
        positives = torch.as_tensor(item["positive_anchor_ids"]).long().tolist()
        false_attractors = (
            torch.as_tensor(item["false_attractor_anchor_ids"]).long().tolist()
        )
        if not (len(positives) == len(false_attractors) == descriptors.shape[0]):
            raise ValueError("feedback evidence rows do not align")
        for descriptor, positive_id, harmful_id in zip(
            descriptors, positives, false_attractors
        ):
            positive[int(positive_id)][family].append(descriptor)
            harmful[int(harmful_id)][family].append(descriptor)

    output = source.clone()
    audits = []
    candidate_ids = sorted(set(positive) | set(harmful))
    for anchor_id in candidate_ids:
        positive_families = set(positive.get(anchor_id, {}))
        harmful_families = set(harmful.get(anchor_id, {}))
        evidence_families = positive_families | harmful_families
        conflict = bool(positive_families & harmful_families)
        if (
            anchor_id not in id_to_row
            or len(evidence_families) < int(minimum_pose_families)
            or conflict
        ):
            continue
        bank = observation_banks.get(anchor_id)
        if bank is None:
            raise ValueError("actionable Anchor lacks its mapping observation bank")
        observations = F.normalize(
            torch.as_tensor(bank["descriptors"]).float().cpu(), dim=1
        )
        view_families = torch.as_tensor(bank["view_families"]).long().cpu()
        if observations.ndim != 2 or observations.shape[1] != current.shape[1]:
            raise ValueError("mapping observation descriptor dimensions differ")
        if view_families.shape != (observations.shape[0],):
            raise ValueError("mapping observation view families do not align")
        support = torch.zeros(observations.shape[0])
        harm = torch.zeros_like(support)
        for family_records in positive.get(anchor_id, {}).values():
            family_query = F.normalize(torch.stack(family_records).mean(0), dim=0)
            score = observations @ family_query
            support += score - score.mean()
        for family_records in harmful.get(anchor_id, {}).values():
            family_query = F.normalize(torch.stack(family_records).mean(0), dim=0)
            score = observations @ family_query
            harm += torch.relu(score - score.mean())
        log_weights = float(learning_rate) * (support - float(harmful_weight) * harm)
        relative = log_weights.exp().clamp(
            float(minimum_relative_weight), float(maximum_relative_weight)
        )
        weights = torch.zeros_like(relative)
        unique_views = torch.unique(view_families)
        for family in unique_views.tolist():
            mask = view_families == int(family)
            weights[mask] = relative[mask] / relative[mask].sum()
        weights /= max(int(unique_views.numel()), 1)
        preliminary = F.normalize((weights[:, None] * observations).sum(0), dim=0)
        keep = torch.ones(observations.shape[0], dtype=torch.bool)
        trim_count = min(
            int(math.floor(observations.shape[0] * float(trim_fraction))),
            max(observations.shape[0] - 1, 0),
        )
        if trim_count:
            keep[torch.argsort(observations @ preliminary)[:trim_count]] = False
        trimmed_weights = weights * keep
        trimmed_weights /= trimmed_weights.sum()
        reconstructed = F.normalize(
            (trimmed_weights[:, None] * observations).sum(0), dim=0
        )
        row = id_to_row[anchor_id]
        bounded = _bounded_direction(
            current[row], reconstructed, float(maximum_descriptor_angle_deg)
        )
        output[row] = bounded
        entropy = float(
            -(
                trimmed_weights[trimmed_weights > 0]
                * trimmed_weights[trimmed_weights > 0].log()
            ).sum()
        )
        audits.append(
            {
                "anchor_id": anchor_id,
                "anchor_row": row,
                "pose_family_count": len(evidence_families),
                "positive_pose_family_count": len(positive_families),
                "harmful_pose_family_count": len(harmful_families),
                "observation_count": int(observations.shape[0]),
                "descriptor_angle_deg": _angle_degrees(current[row], bounded),
                "weight_entropy": entropy,
                "minimum_weight": float(trimmed_weights.min()),
                "maximum_weight": float(trimmed_weights.max()),
            }
        )
    changed_rows = torch.tensor(
        [item["anchor_row"] for item in audits], dtype=torch.long
    )
    return {
        "schema": "lafgs_v7_descriptor_reconstruction",
        "version": 1,
        "anchor_ids": ids,
        "anchor_features": output,
        "changed_anchor_rows": changed_rows,
        "changed_anchor_ids": ids[changed_rows],
        "changed_anchor_count": int(changed_rows.numel()),
        "audits": audits,
        "feedback_descriptors_copied_into_map": False,
        "minimum_pose_families": int(minimum_pose_families),
    }
