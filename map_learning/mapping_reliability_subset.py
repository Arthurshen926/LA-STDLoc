"""Mapping-only reliability gates for weak Gaussian-prior Anchor maps."""

from __future__ import annotations

import torch


def select_mapping_reliable_anchors(
    candidates: dict,
    *,
    minimum_identity_reliability: float,
    minimum_geometry_reliability: float,
    minimum_observations: int,
    maximum_covariance_trace_m2: float,
) -> tuple[torch.Tensor, dict]:
    """Select Anchors using only evidence available during mapping.

    The gate deliberately does not inspect localization queries or pose outcomes.
    All four conditions are absolute, so a uniformly poor prior cannot authorize
    a fixed top percentile of otherwise unreliable Tracks.
    """

    if candidates.get("schema") != "lafgs_projective_anchor_candidates":
        raise ValueError("reliability selection requires projective candidates")
    if candidates.get("uses_source_mapping_rgb") is not False:
        raise ValueError("source RGB candidates are outside this mapping-only arm")
    if candidates.get("uses_test_queries") is not False:
        raise ValueError("test-query candidates are outside this mapping-only arm")
    if not 0.0 <= float(minimum_identity_reliability) <= 1.0:
        raise ValueError("minimum identity reliability must be in [0, 1]")
    if not 0.0 <= float(minimum_geometry_reliability) <= 1.0:
        raise ValueError("minimum geometry reliability must be in [0, 1]")
    if int(minimum_observations) < 3:
        raise ValueError("minimum observations must be at least three")
    if float(maximum_covariance_trace_m2) <= 0.0:
        raise ValueError("maximum covariance trace must be positive")

    xyz = torch.as_tensor(candidates["anchor_xyz"])
    count = int(xyz.shape[0])
    identity = torch.as_tensor(candidates["identity_reliability"]).float().reshape(-1)
    geometry = torch.as_tensor(candidates["geometry_reliability"]).float().reshape(-1)
    covariance = torch.as_tensor(candidates["anchor_position_covariance"]).float()
    offsets = torch.as_tensor(
        candidates["projective_anchor_observations"]["observation_offsets"]
    ).long()
    if (
        count <= 0
        or identity.numel() != count
        or geometry.numel() != count
        or covariance.shape != (count, 3, 3)
        or offsets.shape != (count + 1,)
        or int(offsets[0]) != 0
        or bool((offsets[1:] < offsets[:-1]).any())
    ):
        raise ValueError("candidate reliability tensors are not aligned")
    if not (
        torch.isfinite(identity).all()
        and torch.isfinite(geometry).all()
        and torch.isfinite(covariance).all()
    ):
        raise ValueError("candidate reliability tensors must be finite")

    observation_count = offsets[1:] - offsets[:-1]
    covariance_trace = covariance.diagonal(dim1=-2, dim2=-1).sum(1)
    conditions = {
        "identity": identity >= float(minimum_identity_reliability),
        "geometry": geometry >= float(minimum_geometry_reliability),
        "observations": observation_count >= int(minimum_observations),
        "covariance": covariance_trace <= float(maximum_covariance_trace_m2),
    }
    keep = conditions["identity"]
    for condition in tuple(conditions.values())[1:]:
        keep = keep & condition
    selected = torch.nonzero(keep, as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        raise ValueError("mapping reliability gate retained no Anchor")

    report = {
        "schema": "anygsloc_mapping_reliability_selection",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "localization_outcomes_consumed": False,
        "selection_policy": "absolute_conjunctive_mapping_evidence",
        "input_anchor_count": count,
        "selected_anchor_count": int(selected.numel()),
        "selected_fraction": float(selected.numel() / count),
        "thresholds": {
            "minimum_identity_reliability": float(minimum_identity_reliability),
            "minimum_geometry_reliability": float(minimum_geometry_reliability),
            "minimum_observations": int(minimum_observations),
            "maximum_covariance_trace_m2": float(maximum_covariance_trace_m2),
        },
        "individual_pass_counts": {
            name: int(value.sum()) for name, value in conditions.items()
        },
    }
    return selected, report
