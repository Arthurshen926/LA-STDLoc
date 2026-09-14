"""Vanilla 3DGS prior adapter."""

from __future__ import annotations

from pathlib import Path

from priors.base import GaussianPrior, load_prior


def load_gaussian_3d(path: str | Path, *, manifest=None) -> GaussianPrior:
    prior = load_prior(path, manifest_path=manifest, source_method="vanilla_3dgs")
    if prior.prior_type != "3dgs":
        raise ValueError("vanilla 3DGS adapter received a non-3DGS PLY")
    return prior
