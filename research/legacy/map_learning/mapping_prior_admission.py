"""Test-unaware scene-level admission for a rendered sparse Anchor map."""

from __future__ import annotations

import torch

from common.v6_contracts import ANCHOR_CANDIDATE_SCHEMA


def audit_mapping_prior_admission(
    candidates: dict,
    *,
    high_identity_reliability: float,
    high_geometry_reliability: float,
    minimum_high_reliability_count: int,
    minimum_high_reliability_fraction: float,
) -> dict:
    if candidates.get("schema") != ANCHOR_CANDIDATE_SCHEMA:
        raise ValueError("prior admission requires projective candidates")
    if candidates.get("uses_source_mapping_rgb") is not False:
        raise ValueError("source RGB is outside rendered-prior admission")
    if candidates.get("uses_test_queries") is not False:
        raise ValueError("test queries are outside rendered-prior admission")
    if not 0.0 <= float(high_identity_reliability) <= 1.0:
        raise ValueError("identity threshold must be in [0, 1]")
    if not 0.0 <= float(high_geometry_reliability) <= 1.0:
        raise ValueError("geometry threshold must be in [0, 1]")
    if int(minimum_high_reliability_count) < 1:
        raise ValueError("minimum high-reliability count must be positive")
    if not 0.0 < float(minimum_high_reliability_fraction) <= 1.0:
        raise ValueError("minimum high-reliability fraction must be in (0, 1]")

    identity = torch.as_tensor(candidates["identity_reliability"]).float().reshape(-1)
    geometry = torch.as_tensor(candidates["geometry_reliability"]).float().reshape(-1)
    if (
        identity.numel() == 0
        or identity.shape != geometry.shape
        or not torch.isfinite(identity).all()
        or not torch.isfinite(geometry).all()
    ):
        raise ValueError("candidate reliability arrays are empty or invalid")
    high = (identity >= float(high_identity_reliability)) & (
        geometry >= float(high_geometry_reliability)
    )
    count = int(identity.numel())
    high_count = int(high.sum())
    high_fraction = float(high_count / count)
    count_pass = high_count >= int(minimum_high_reliability_count)
    fraction_pass = high_fraction >= float(minimum_high_reliability_fraction)
    admitted = count_pass and fraction_pass
    quantiles = torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float32)
    return {
        "schema": "anygsloc_mapping_prior_admission_audit",
        "version": 1,
        "status": "PASS" if admitted else "REJECT_UNSAFE_PRIOR",
        "admitted": admitted,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "localization_outcomes_consumed": False,
        "map_mutated": False,
        "calibration_scope": "cambridge_prior_quality_development",
        "required_external_validation": True,
        "anchor_count": count,
        "high_reliability_anchor_count": high_count,
        "high_reliability_anchor_fraction": high_fraction,
        "checks": {
            "minimum_count": count_pass,
            "minimum_fraction": fraction_pass,
        },
        "thresholds": {
            "high_identity_reliability": float(high_identity_reliability),
            "high_geometry_reliability": float(high_geometry_reliability),
            "minimum_high_reliability_count": int(minimum_high_reliability_count),
            "minimum_high_reliability_fraction": float(
                minimum_high_reliability_fraction
            ),
        },
        "diagnostics": {
            "quantile_levels": quantiles.tolist(),
            "identity_reliability_quantiles": torch.quantile(
                identity, quantiles
            ).tolist(),
            "geometry_reliability_quantiles": torch.quantile(
                geometry, quantiles
            ).tolist(),
        },
    }
