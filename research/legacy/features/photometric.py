"""Frozen symmetric photometric preprocessing for descriptor extraction."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import cv2
import numpy as np


SCHEMA = "lafgs_symmetric_photometric_canonicalization"
VERSION = 1
LUMINANCE_WEIGHTS = (0.299, 0.587, 0.114)


def percentile_grayscale_contract() -> dict:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "mode": "luminance_percentile_v1",
        "luminance_weights_rgb": list(LUMINANCE_WEIGHTS),
        "lower_percentile": 0.01,
        "upper_percentile": 0.99,
        "constant_epsilon": 1e-6,
        "constant_image_output": "zeros",
        "channel_output": "replicate_grayscale_to_rgb",
        "quantile_scope": "all_image_pixels_per_image",
    }


def clahe_grayscale_contract() -> dict:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "mode": "clahe_luminance_v1",
        "luminance_weights_rgb": list(LUMINANCE_WEIGHTS),
        "clip_limit": 1.5,
        "tile_grid_size": [8, 8],
        "input_quantization": "clamp_round_uint8",
        "channel_output": "replicate_grayscale_to_rgb",
    }


def validate_photometric_contract(value: Mapping) -> dict:
    for expected in (percentile_grayscale_contract(), clahe_grayscale_contract()):
        if dict(value) == expected:
            return expected
    raise ValueError("unsupported or modified photometric canonicalization contract")


def canonicalize_image(image: torch.Tensor, contract: Mapping) -> torch.Tensor:
    """Apply the exact per-image grayscale percentile contract to [B,3,H,W]."""
    validate_photometric_contract(contract)
    value = torch.as_tensor(image).float()
    squeeze = value.ndim == 3
    if squeeze:
        value = value[None]
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("photometric input must have shape [3,H,W] or [B,3,H,W]")
    weights = value.new_tensor(LUMINANCE_WEIGHTS).reshape(1, 3, 1, 1)
    gray = (value * weights).sum(dim=1, keepdim=True)
    if contract["mode"] == "luminance_percentile_v1":
        flat = gray.flatten(1)
        low = torch.quantile(flat, 0.01, dim=1, keepdim=True).reshape(-1, 1, 1, 1)
        high = torch.quantile(flat, 0.99, dim=1, keepdim=True).reshape(-1, 1, 1, 1)
        scale = high - low
        normalized = ((gray - low) / scale.clamp_min(1e-6)).clamp(0.0, 1.0)
        normalized = torch.where(scale > 1e-6, normalized, torch.zeros_like(normalized))
    else:
        device = gray.device
        dtype = gray.dtype
        uint8 = gray.detach().clamp(0.0, 1.0).mul(255).round().byte().cpu().numpy()
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        rows = [clahe.apply(np.ascontiguousarray(row[0])) for row in uint8]
        normalized = torch.from_numpy(np.stack(rows))[:, None].to(device=device, dtype=dtype).div(255.0)
    output = normalized.expand(-1, 3, -1, -1).contiguous()
    return output[0] if squeeze else output
