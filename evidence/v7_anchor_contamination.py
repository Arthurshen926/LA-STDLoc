"""Pure helpers for the V7 map-observation contamination audit.

The audit deliberately keeps render-quality evidence separate from pose
feedback.  A mapping observation is either accepted by the already frozen V2
row certificate or it is not; no localization result is used to tune that
decision.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


def pack_query_rows(
    rows_by_query: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ragged per-query Boolean rows and return their CSR offsets."""

    rows = [torch.as_tensor(value).bool().reshape(-1).cpu() for value in rows_by_query]
    counts = torch.tensor([value.numel() for value in rows], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    packed = torch.cat(rows) if rows else torch.empty(0, dtype=torch.bool)
    return packed, offsets


def gather_observation_rows(
    rows_by_query: Sequence[torch.Tensor],
    query_indices: torch.Tensor,
    keypoint_indices: torch.Tensor,
) -> torch.Tensor:
    """Resolve an Anchor observation CSR into its per-observation row flag."""

    packed, offsets = pack_query_rows(rows_by_query)
    queries = torch.as_tensor(query_indices).long().reshape(-1).cpu()
    keypoints = torch.as_tensor(keypoint_indices).long().reshape(-1).cpu()
    if queries.shape != keypoints.shape:
        raise ValueError("observation query/keypoint columns must align")
    if queries.numel() == 0:
        return torch.empty(0, dtype=torch.bool)
    if int(queries.min()) < 0 or int(queries.max()) >= len(rows_by_query):
        raise ValueError("observation query index is out of range")
    query_counts = offsets[1:] - offsets[:-1]
    if bool((keypoints < 0).any()) or bool((keypoints >= query_counts[queries]).any()):
        raise ValueError("observation keypoint index is out of range")
    return packed[offsets[queries] + keypoints]


def segment_counts(flags: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Count Boolean observation flags for each Anchor CSR segment."""

    value = torch.as_tensor(flags).long().reshape(-1).cpu()
    csr = torch.as_tensor(offsets).long().reshape(-1).cpu()
    if csr.numel() == 0 or int(csr[0]) != 0 or int(csr[-1]) != value.numel():
        raise ValueError("invalid observation CSR offsets")
    prefix = torch.cat((torch.zeros(1, dtype=torch.long), value.cumsum(0)))
    return prefix[csr[1:]] - prefix[csr[:-1]]


def aggregate_anchor_reliability(
    *,
    observation_valid: torch.Tensor,
    observation_structure_supported: torch.Tensor,
    observation_offsets: torch.Tensor,
    observation_query_indices: torch.Tensor,
    query_family_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Lift frozen per-observation V2 decisions to reversible Anchor evidence."""

    valid = torch.as_tensor(observation_valid).bool().reshape(-1).cpu()
    structure = (
        torch.as_tensor(observation_structure_supported).bool().reshape(-1).cpu()
    )
    offsets = torch.as_tensor(observation_offsets).long().reshape(-1).cpu()
    queries = torch.as_tensor(observation_query_indices).long().reshape(-1).cpu()
    families = torch.as_tensor(query_family_ids).long().reshape(-1).cpu()
    if valid.shape != structure.shape or valid.shape != queries.shape:
        raise ValueError("observation reliability columns must align")
    if queries.numel() and (int(queries.min()) < 0 or int(queries.max()) >= families.numel()):
        raise ValueError("observation family lookup is out of range")

    observation_count = offsets[1:] - offsets[:-1]
    valid_count = segment_counts(valid, offsets)
    structure_count = segment_counts(structure, offsets)
    family_values = families[queries] if queries.numel() else queries
    unique_families = torch.unique(families)
    family_count = torch.zeros_like(observation_count)
    valid_family_count = torch.zeros_like(observation_count)
    invalid_family_count = torch.zeros_like(observation_count)
    for family in unique_families.tolist():
        in_family = family_values == int(family)
        family_count += segment_counts(in_family, offsets) > 0
        valid_family_count += segment_counts(in_family & valid, offsets) > 0
        invalid_family_count += segment_counts(in_family & (~valid), offsets) > 0

    denominator = observation_count.clamp_min(1).float()
    valid_fraction = valid_count.float() / denominator
    structure_fraction = structure_count.float() / denominator
    # These are intentionally threshold-free except for the map's existing
    # minimum multiview construction contract.
    pure_contamination = (family_count >= 2) & (valid_count == 0)
    descriptor_reconstructable = (valid_count >= 3) & (valid_family_count >= 2)
    mixed_contamination = (
        (valid_count > 0)
        & (valid_count < observation_count)
        & descriptor_reconstructable
    )
    return {
        "observation_count": observation_count,
        "valid_observation_count": valid_count,
        "valid_observation_fraction": valid_fraction,
        "structure_supported_observation_count": structure_count,
        "structure_supported_observation_fraction": structure_fraction,
        "view_family_count": family_count,
        "valid_view_family_count": valid_family_count,
        "invalid_view_family_count": invalid_family_count,
        "pure_contamination": pure_contamination,
        "descriptor_reconstructable": descriptor_reconstructable,
        "mixed_contamination": mixed_contamination,
    }


def bounded_descriptor_reconstruction(
    current: torch.Tensor,
    proposed: torch.Tensor,
    eligible: torch.Tensor,
    *,
    maximum_angle_deg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move toward valid-observation means within a fixed angular trust region."""

    current_value = torch.as_tensor(current).float()
    baseline = F.normalize(current_value, dim=1)
    target = F.normalize(torch.as_tensor(proposed).float(), dim=1)
    active = torch.as_tensor(eligible).bool().reshape(-1)
    if baseline.shape != target.shape or active.shape != baseline.shape[:1]:
        raise ValueError("descriptor reconstruction tensors must align")
    maximum = math.radians(float(maximum_angle_deg))
    cosine = (baseline * target).sum(1).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    fraction = torch.minimum(
        torch.ones_like(angle),
        torch.full_like(angle, maximum) / angle.clamp_min(1e-8),
    )
    fraction = torch.where(active, fraction, torch.zeros_like(fraction))
    sine = torch.sin(angle).clamp_min(1e-8)
    left_weight = torch.sin((1.0 - fraction) * angle) / sine
    right_weight = torch.sin(fraction * angle) / sine
    output = F.normalize(
        baseline * left_weight[:, None] + target * right_weight[:, None], dim=1
    )
    coincident = angle < 1e-6
    output[coincident] = baseline[coincident]
    # Preserve every ineligible map vector bit-for-bit.  The fixed plant will
    # normalize on load, but the audit must not mislabel normalization noise as
    # a descriptor intervention.
    output[~active] = current_value[~active]
    changed_angle = torch.rad2deg(
        torch.acos(
            (baseline * F.normalize(output, dim=1)).sum(1).clamp(-1.0, 1.0)
        )
    )
    changed_angle[~active] = 0.0
    return output, changed_angle


def enrichment_table(
    *,
    anchor_rows: torch.Tensor,
    anchor_positive: torch.Tensor,
    reference_rows: torch.Tensor | None = None,
) -> Mapping[str, float | int]:
    """Return event prevalence and enrichment against a fixed Anchor universe."""

    positive = torch.as_tensor(anchor_positive).bool().reshape(-1)
    rows = torch.as_tensor(anchor_rows).long().reshape(-1)
    reference = (
        torch.arange(positive.numel(), dtype=torch.long)
        if reference_rows is None
        else torch.as_tensor(reference_rows).long().reshape(-1)
    )
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= positive.numel()):
        raise ValueError("winner Anchor row is out of range")
    event_rate = float(positive[rows].float().mean()) if rows.numel() else math.nan
    reference_rate = (
        float(positive[reference].float().mean()) if reference.numel() else math.nan
    )
    enrichment = (
        event_rate / reference_rate
        if math.isfinite(event_rate) and reference_rate > 0
        else math.nan
    )
    return {
        "event_count": int(rows.numel()),
        "event_positive_count": int(positive[rows].sum()) if rows.numel() else 0,
        "event_positive_fraction": event_rate,
        "reference_count": int(reference.numel()),
        "reference_positive_fraction": reference_rate,
        "enrichment_ratio": enrichment,
    }
