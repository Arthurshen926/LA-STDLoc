"""Explicit dense-raster sampling contract for sparse image-grid rows."""

from __future__ import annotations

import torch


def sample_raster_at_grid_uv(raster: torch.Tensor, grid_uv: torch.Tensor) -> torch.Tensor:
    """Nearest-cell lookup with integer grid coordinates naming raster cells."""
    raster = torch.as_tensor(raster, device=grid_uv.device)
    grid_uv = torch.as_tensor(grid_uv, device=raster.device)
    if raster.ndim != 2 or grid_uv.ndim != 2 or grid_uv.shape[1] != 2:
        raise ValueError("expected raster [H,W] and grid_uv [N,2]")
    index = grid_uv.round().long()
    index[:, 0].clamp_(0, raster.shape[1] - 1)
    index[:, 1].clamp_(0, raster.shape[0] - 1)
    return raster[index[:, 1], index[:, 0]]
