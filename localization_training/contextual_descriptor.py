"""Deployment-aligned 2D/3D context descriptors for sparse localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ContextDescriptorConfig:
    sparse_radii_px: tuple[float, ...] = (48.0, 96.0, 192.0)
    dense_radii_cells: tuple[int, ...] = (1, 3, 7)
    map_neighbor_counts: tuple[int, ...] = (8, 24, 64)
    maximum_sparse_neighbors: int = 48
    distance_chunk_size: int = 256


def _normalized_rows(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.as_tensor(value).float(), dim=-1)


def multiscale_sparse_query_context(
    descriptors: torch.Tensor,
    keypoints_xy: torch.Tensor,
    scores: torch.Tensor,
    *,
    radii_px: Iterable[float] = (48.0, 96.0, 192.0),
    maximum_neighbors: int = 48,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Pool neighboring native sparse descriptors at several image scales.

    The output is ``N x S x D``. The center descriptor is excluded, so the
    context cannot trivially copy the local descriptor used by the baseline
    matcher.
    """

    descriptors = _normalized_rows(descriptors)
    keypoints_xy = torch.as_tensor(
        keypoints_xy, device=descriptors.device, dtype=descriptors.dtype
    ).reshape(-1, 2)
    scores = torch.as_tensor(
        scores, device=descriptors.device, dtype=descriptors.dtype
    ).reshape(-1)
    if not (
        descriptors.shape[0] == keypoints_xy.shape[0] == scores.numel()
    ):
        raise ValueError("sparse context inputs must have aligned rows")
    radii = tuple(float(radius) for radius in radii_px)
    if not radii or any(radius <= 0.0 for radius in radii):
        raise ValueError("sparse context radii must be positive")
    if descriptors.shape[0] == 0:
        return descriptors.new_zeros(
            (0, len(radii), descriptors.shape[1])
        )

    count = descriptors.shape[0]
    neighbor_limit = min(max(int(maximum_neighbors), 1), max(count - 1, 1))
    output = descriptors.new_zeros((count, len(radii), descriptors.shape[1]))
    all_indices = torch.arange(count, device=descriptors.device)
    for start in range(0, count, max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), count)
        distance = torch.cdist(
            keypoints_xy[start:end].float(), keypoints_xy.float()
        )
        local_rows = torch.arange(end - start, device=distance.device)
        distance[local_rows, all_indices[start:end]] = torch.inf
        nearest_distance, nearest_indices = torch.topk(
            distance,
            k=neighbor_limit,
            dim=1,
            largest=False,
            sorted=True,
        )
        nearest_descriptors = descriptors[nearest_indices]
        nearest_scores = scores[nearest_indices].clamp_min(0.0)
        for scale_index, radius in enumerate(radii):
            valid = nearest_distance <= float(radius)
            weights = torch.exp(
                -0.5 * (nearest_distance / float(radius)).square()
            )
            weights = weights * nearest_scores * valid
            pooled = (
                nearest_descriptors * weights[..., None]
            ).sum(dim=1)
            pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(
                1e-8
            )
            nonempty = valid.any(dim=1)
            pooled[~nonempty] = 0.0
            output[start:end, scale_index] = F.normalize(
                pooled, dim=1
            )
    return output


def _sample_stride8_map(
    feature_map: torch.Tensor,
    keypoints_xy: torch.Tensor,
) -> torch.Tensor:
    feature_map = torch.as_tensor(feature_map)
    if feature_map.ndim == 3:
        feature_map = feature_map[None]
    if feature_map.ndim != 4 or feature_map.shape[0] != 1:
        raise ValueError("dense feature map must have shape 1xCxHxW")
    keypoints_xy = torch.as_tensor(
        keypoints_xy,
        device=feature_map.device,
        dtype=feature_map.dtype,
    ).reshape(-1, 2)
    height, width = feature_map.shape[-2:]
    normalized = (keypoints_xy + 0.5) / keypoints_xy.new_tensor(
        [float(width * 8), float(height * 8)]
    )
    grid = normalized.mul(2.0).sub(1.0).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, :, 0].T
    return F.normalize(sampled, dim=1)


def multiscale_dense_query_context(
    feature_map: torch.Tensor,
    keypoints_xy: torch.Tensor,
    *,
    radii_cells: Iterable[int] = (1, 3, 7),
) -> torch.Tensor:
    """Pool the native stride-8 SuperPoint map around sparse keypoints."""

    feature_map = torch.as_tensor(feature_map).float()
    if feature_map.ndim == 3:
        feature_map = feature_map[None]
    radii = tuple(int(radius) for radius in radii_cells)
    if not radii or any(radius < 0 for radius in radii):
        raise ValueError("dense context radii must be non-negative")
    outputs = []
    for radius in radii:
        if radius == 0:
            pooled = feature_map
        else:
            kernel = 2 * radius + 1
            pooled = F.avg_pool2d(
                feature_map,
                kernel_size=kernel,
                stride=1,
                padding=radius,
                count_include_pad=False,
            )
        outputs.append(_sample_stride8_map(pooled, keypoints_xy))
    return torch.stack(outputs, dim=1)


def multiscale_map_3d_context(
    anchor_features: torch.Tensor,
    anchor_xyz: torch.Tensor,
    *,
    query_indices: torch.Tensor | None = None,
    neighbor_counts: Iterable[int] = (8, 24, 64),
    chunk_size: int = 256,
) -> torch.Tensor:
    """Pool fixed 3D KNN neighborhoods without using matcher assignments.

    Coordinates only select a local neighborhood. The returned representation
    contains normalized descriptor moments and does not encode absolute world
    position.
    """

    anchor_features = _normalized_rows(anchor_features)
    anchor_xyz = torch.as_tensor(
        anchor_xyz,
        device=anchor_features.device,
        dtype=anchor_features.dtype,
    ).reshape(-1, 3)
    if anchor_features.shape[0] != anchor_xyz.shape[0]:
        raise ValueError("3D context features and coordinates must align")
    counts = tuple(int(value) for value in neighbor_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("3D context neighbor counts must be positive")
    if anchor_xyz.shape[0] < 2:
        raise ValueError("3D context requires at least two anchors")
    maximum = min(max(counts), anchor_xyz.shape[0] - 1)
    indices = (
        torch.arange(anchor_xyz.shape[0], device=anchor_xyz.device)
        if query_indices is None
        else torch.as_tensor(
            query_indices, device=anchor_xyz.device, dtype=torch.long
        ).reshape(-1)
    )
    output = anchor_features.new_zeros(
        (indices.numel(), len(counts), anchor_features.shape[1])
    )
    all_indices = torch.arange(anchor_xyz.shape[0], device=anchor_xyz.device)
    for start in range(0, indices.numel(), max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), indices.numel())
        rows = indices[start:end]
        distance = torch.cdist(
            anchor_xyz[rows].float(), anchor_xyz.float()
        )
        local_rows = torch.arange(end - start, device=distance.device)
        distance[local_rows, rows] = torch.inf
        nearest_distance, nearest = torch.topk(
            distance,
            k=maximum,
            dim=1,
            largest=False,
            sorted=True,
        )
        nearest_features = anchor_features[nearest]
        for scale_index, requested_count in enumerate(counts):
            count = min(requested_count, maximum)
            selected_distance = nearest_distance[:, :count]
            local_scale = selected_distance[:, -1:].clamp_min(1e-6)
            weights = torch.exp(
                -0.5 * (selected_distance / local_scale).square()
            )
            pooled = (
                nearest_features[:, :count] * weights[..., None]
            ).sum(dim=1)
            pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(
                1e-8
            )
            output[start:end, scale_index] = F.normalize(
                pooled, dim=1
            )
    return output


def flatten_context(context: torch.Tensor) -> torch.Tensor:
    context = torch.as_tensor(context).float()
    if context.ndim != 3:
        raise ValueError("context must have shape NxSxD")
    return F.normalize(context.reshape(context.shape[0], -1), dim=1)


def fuse_local_and_context(
    local_descriptor: torch.Tensor,
    context_descriptor: torch.Tensor,
    *,
    context_weight: float,
) -> torch.Tensor:
    """Materialize one descriptor for one exact global matrix match."""

    local_descriptor = _normalized_rows(local_descriptor)
    context_descriptor = _normalized_rows(context_descriptor)
    if local_descriptor.shape[0] != context_descriptor.shape[0]:
        raise ValueError("local and context descriptor rows must align")
    weight = max(float(context_weight), 0.0)
    return F.normalize(
        torch.cat(
            (
                local_descriptor,
                context_descriptor * weight**0.5,
            ),
            dim=1,
        ),
        dim=1,
    )


def joint_local_context_similarity(
    query_local: torch.Tensor,
    map_local: torch.Tensor,
    query_context: torch.Tensor,
    map_context: torch.Tensor,
    *,
    context_weight: float,
) -> torch.Tensor:
    """Compute the exact score induced by ``fuse_local_and_context``."""

    query_local = _normalized_rows(query_local)
    map_local = _normalized_rows(map_local)
    query_context = _normalized_rows(query_context)
    map_context = _normalized_rows(map_context)
    if query_local.shape[0] != query_context.shape[0]:
        raise ValueError("query local and context rows must align")
    if map_local.shape[0] != map_context.shape[0]:
        raise ValueError("map local and context rows must align")
    weight = max(float(context_weight), 0.0)
    return (
        query_local @ map_local.T
        + weight * (query_context @ map_context.T)
    ) / (1.0 + weight)


class BoundedContextProjector(nn.Module):
    """Low-rank residual around a fixed shared random projection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        rank: int = 32,
        maximum_residual: float = 0.35,
        *,
        base_projection: torch.Tensor | None = None,
        seed: int = 2026,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        self.maximum_residual = float(maximum_residual)
        if base_projection is None:
            generator = torch.Generator().manual_seed(int(seed))
            base_projection = torch.randn(
                self.output_dim,
                self.input_dim,
                generator=generator,
            )
            base_projection = F.normalize(base_projection, dim=1)
        base_projection = torch.as_tensor(base_projection).float()
        if tuple(base_projection.shape) != (
            self.output_dim,
            self.input_dim,
        ):
            raise ValueError("base context projection has the wrong shape")
        self.register_buffer("base_projection", base_projection.clone())
        self.down = nn.Linear(self.input_dim, self.rank)
        self.up = nn.Linear(self.rank, self.output_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = F.normalize(torch.as_tensor(value).float(), dim=1)
        base = F.linear(value, self.base_projection)
        residual = self.up(F.silu(self.down(value)))
        norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
        scale = torch.clamp(
            self.maximum_residual / norm.clamp_min(1e-8), max=1.0
        )
        residual = residual * scale
        return F.normalize(base + residual, dim=1), residual

    def export_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "rank": self.rank,
            "maximum_residual": self.maximum_residual,
        }


class BoundedDualContextEncoder(nn.Module):
    """Separate bounded query/map adapters with one shared base projection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        rank: int = 32,
        maximum_residual: float = 0.35,
        seed: int = 2026,
    ):
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        base = F.normalize(
            torch.randn(
                int(output_dim), int(input_dim), generator=generator
            ),
            dim=1,
        )
        kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "rank": rank,
            "maximum_residual": maximum_residual,
            "base_projection": base,
            "seed": seed,
        }
        self.query = BoundedContextProjector(**kwargs)
        self.map = BoundedContextProjector(**kwargs)

    def export_config(self) -> dict:
        return {
            **self.query.export_config(),
            "seed_contract": "shared_fixed_projection",
        }
