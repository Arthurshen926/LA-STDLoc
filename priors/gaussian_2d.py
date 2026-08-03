"""Vanilla or enhanced 2DGS prior adapter."""

from __future__ import annotations

from pathlib import Path

from priors.base import GaussianPrior, load_prior


def load_gaussian_2d(
    path: str | Path, *, manifest=None, source_method: str = "vanilla_2dgs"
) -> GaussianPrior:
    prior = load_prior(path, manifest_path=manifest, source_method=source_method)
    if prior.prior_type != "2dgs":
        raise ValueError("2DGS adapter received a non-2DGS PLY")
    return prior
