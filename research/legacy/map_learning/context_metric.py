"""Map-consistent bounded context descriptor adapter.

The adapter consumes only frozen single-image SuperPoint outputs.  It never
uses a camera pose, map candidate, or query--map pairwise interaction online.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from features.multiview_fusion import (
    grid_index_to_physical,
    sample_dense_descriptors_at_image_uv,
)


DEFAULT_CONTEXT_KERNELS = (3, 7, 15)
CONTEXT_MODES = (
    "multi_scale_global",
    "local_only",
    "global_only",
    "zero",
)
RESIDUAL_PARAMETERIZATIONS = (
    "hard_clip_v1",
    "smooth_radial_rational_v1",
)


def _masked_average_pool(
    feature_map: torch.Tensor,
    valid_mask: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    kernel_size = int(kernel_size)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("context pooling kernels must be positive and odd")
    padding = kernel_size // 2
    mask = valid_mask[None, None].to(feature_map.dtype)
    numerator = F.avg_pool2d(
        feature_map[None] * mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    denominator = F.avg_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )
    return (numerator / denominator.clamp_min(1e-6))[0]


def dense_context_tokens(
    feature_map: torch.Tensor,
    keypoints: torch.Tensor,
    image_hw: Sequence[int],
    *,
    valid_mask: torch.Tensor | None = None,
    kernels: Sequence[int] = DEFAULT_CONTEXT_KERNELS,
) -> torch.Tensor:
    """Sample normalized multi-scale local tokens plus one global token.

    Keypoints use the native detector grid-index convention.  Sampling adds
    the repository-wide half-pixel center exactly once.
    """
    feature_map = torch.as_tensor(feature_map).float()
    keypoints = torch.as_tensor(
        keypoints, device=feature_map.device, dtype=feature_map.dtype
    ).reshape(-1, 2)
    if feature_map.ndim != 3:
        raise ValueError("feature_map must have shape [C, H, W]")
    if len(image_hw) != 2:
        raise ValueError("image_hw must contain height and width")
    if valid_mask is None:
        valid = torch.ones(
            feature_map.shape[-2:], dtype=torch.bool, device=feature_map.device
        )
    else:
        valid = torch.as_tensor(valid_mask, device=feature_map.device).bool()
        if valid.shape != feature_map.shape[-2:]:
            raise ValueError("valid_mask must align with the dense feature grid")
    physical = grid_index_to_physical(keypoints)
    tokens = []
    for kernel in kernels:
        pooled = _masked_average_pool(feature_map, valid, int(kernel))
        tokens.append(
            sample_dense_descriptors_at_image_uv(pooled, physical, image_hw)
        )
    valid_values = feature_map[:, valid]
    if valid_values.numel():
        global_token = F.normalize(valid_values.mean(dim=1), dim=0)
    else:
        global_token = feature_map.new_zeros((feature_map.shape[0],))
    tokens.append(global_token[None].expand(keypoints.shape[0], -1))
    return torch.stack(tokens, dim=1)


class MapConsistentContextAdapter(nn.Module):
    """Identity-initialized 256D descriptor adapter with a hard residual bound."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 256,
        hidden_dim: int = 256,
        context_kernels: Sequence[int] = DEFAULT_CONTEXT_KERNELS,
        context_mode: str = "multi_scale_global",
        maximum_residual_norm: float = 0.10,
        residual_parameterization: str = "smooth_radial_rational_v1",
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_kernels = tuple(int(value) for value in context_kernels)
        self.context_mode = str(context_mode)
        self.maximum_residual_norm = float(maximum_residual_norm)
        self.residual_parameterization = str(residual_parameterization)
        if self.descriptor_dim < 1 or self.hidden_dim < 1:
            raise ValueError("descriptor and hidden dimensions must be positive")
        if self.maximum_residual_norm < 0.0:
            raise ValueError("maximum residual norm must be non-negative")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"unsupported context mode: {self.context_mode}")
        if self.residual_parameterization not in RESIDUAL_PARAMETERIZATIONS:
            raise ValueError(
                "unsupported residual parameterization: "
                f"{self.residual_parameterization}"
            )
        token_count = len(self.context_kernels) + 1
        input_dim = self.descriptor_dim * (token_count + 1)
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.context_head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.descriptor_dim),
        )
        final = self.context_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        descriptors: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=-1)
        context_tokens = torch.as_tensor(
            context_tokens, device=descriptors.device, dtype=descriptors.dtype
        )
        expected = (descriptors.shape[0], len(self.context_kernels) + 1)
        if context_tokens.ndim != 3 or context_tokens.shape[:2] != expected:
            raise ValueError(
                "context_tokens must have shape "
                f"[N, {len(self.context_kernels) + 1}, D]"
            )
        if context_tokens.shape[2] != self.descriptor_dim:
            raise ValueError("context token dimension must match the descriptor")
        inputs = torch.cat(
            (descriptors, context_tokens.flatten(start_dim=1)), dim=1
        )
        raw_residual = self.context_head(self.input_norm(inputs))
        if self.maximum_residual_norm == 0.0:
            residual = torch.zeros_like(raw_residual)
        elif self.residual_parameterization == "hard_clip_v1":
            residual = torch.tanh(raw_residual)
            norm = torch.linalg.norm(residual, dim=1, keepdim=True)
            residual = residual * torch.clamp(
                residual.new_tensor(self.maximum_residual_norm)
                / norm.clamp_min(1e-8),
                max=1.0,
            )
        else:
            # Smooth radial squashing retains the identity derivative at zero,
            # stays strictly inside the trust region, and keeps a non-zero
            # radial gradient near the boundary. A hard clamp would make the
            # residual-norm trust loss constant once an update reaches the cap.
            norm = torch.linalg.norm(raw_residual, dim=1, keepdim=True)
            maximum = raw_residual.new_tensor(self.maximum_residual_norm)
            residual = raw_residual * (
                maximum / (maximum + norm)
            )
        return F.normalize(descriptors + residual, dim=1), residual

    def export_config(self) -> dict:
        return {
            "descriptor_dim": self.descriptor_dim,
            "hidden_dim": self.hidden_dim,
            "context_kernels": list(self.context_kernels),
            "context_mode": self.context_mode,
            "maximum_residual_norm": self.maximum_residual_norm,
            "residual_parameterization": self.residual_parameterization,
            "identity_initialization": "zero_final_projection",
            "output_dim": self.descriptor_dim,
        }


def context_from_cached_query(
    cached: dict,
    native_rows: torch.Tensor,
    *,
    device: torch.device,
    kernels: Sequence[int] = DEFAULT_CONTEXT_KERNELS,
    context_mode: str = "multi_scale_global",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load raw descriptors and dense context for selected native cache rows."""
    rows = torch.as_tensor(native_rows).long().cpu()
    raw = F.normalize(
        torch.as_tensor(cached["native_descriptors"]).float()[rows].to(device),
        dim=1,
    )
    context_mode = str(context_mode)
    if context_mode not in CONTEXT_MODES:
        raise ValueError(f"unsupported context mode: {context_mode}")
    if context_mode == "zero":
        tokens = raw.new_zeros(
            (raw.shape[0], len(tuple(kernels)) + 1, raw.shape[1])
        )
    else:
        feature_map = torch.as_tensor(cached["feature_map"]).float().to(device)
        valid_mask = cached.get("valid_mask")
        tokens = dense_context_tokens(
            feature_map,
            torch.as_tensor(cached["native_keypoints"]).float()[rows].to(device),
            cached["native_input_hw"],
            valid_mask=(
                None
                if valid_mask is None
                else torch.as_tensor(valid_mask).to(device)
            ),
            kernels=kernels,
        )
        if context_mode == "local_only":
            tokens[:, -1] = 0.0
        elif context_mode == "global_only":
            tokens[:, :-1] = 0.0
    return raw, tokens
