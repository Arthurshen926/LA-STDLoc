"""AnySplat feed-forward 3DGS export adapter."""

from __future__ import annotations

from pathlib import Path

from priors.base import GaussianPrior, load_prior


def load_anysplat(path: str | Path, *, manifest=None) -> GaussianPrior:
    prior = load_prior(path, manifest_path=manifest, source_method="anysplat")
    if prior.prior_type != "3dgs":
        raise ValueError("AnySplat adapter requires a 3DGS-compatible PLY")
    return prior
