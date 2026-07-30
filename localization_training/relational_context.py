"""Relational 2D/3D context features for one-pass sparse localization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from localization_training.contextual_descriptor import (
    BoundedContextProjector,
)


def _relation_basis(offset_xy: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(offset_xy, dim=-1).clamp_min(1e-6)
    x = offset_xy[..., 0]
    y = offset_xy[..., 1]
    return torch.stack(
        (
            torch.ones_like(x),
            x,
            y,
            x * y,
            x.square() - y.square(),
            torch.log1p(radius),
        ),
        dim=-1,
    )


def _weighted_descriptor_moments(
    neighbor_descriptors: torch.Tensor,
    basis: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    moments = torch.einsum(
        "nk,nkb,nkd->nbd", weights, basis, neighbor_descriptors
    )
    return moments.reshape(moments.shape[0], -1)


def relational_sparse_query_context(
    descriptors: torch.Tensor,
    keypoints_xy: torch.Tensor,
    scores: torch.Tensor,
    *,
    neighbor_count: int = 16,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Encode ordered 2D neighbor layout without absolute image position."""

    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    keypoints_xy = torch.as_tensor(
        keypoints_xy, device=descriptors.device, dtype=descriptors.dtype
    ).reshape(-1, 2)
    scores = torch.as_tensor(
        scores, device=descriptors.device, dtype=descriptors.dtype
    ).reshape(-1)
    if not (
        len(descriptors) == len(keypoints_xy) == scores.numel()
    ):
        raise ValueError("relational query inputs must have aligned rows")
    if len(descriptors) < 2:
        return descriptors.new_zeros((len(descriptors), 6 * descriptors.shape[1] + 4))
    count = min(max(int(neighbor_count), 1), len(descriptors) - 1)
    output = descriptors.new_empty(
        (len(descriptors), 6 * descriptors.shape[1] + 4)
    )
    all_rows = torch.arange(len(descriptors), device=descriptors.device)
    for start in range(0, len(descriptors), max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), len(descriptors))
        distance = torch.cdist(
            keypoints_xy[start:end].float(), keypoints_xy.float()
        )
        local_rows = torch.arange(end - start, device=distance.device)
        distance[local_rows, all_rows[start:end]] = torch.inf
        nearest_distance, nearest = torch.topk(
            distance, k=count, dim=1, largest=False, sorted=True
        )
        scale = nearest_distance.median(dim=1).values.clamp_min(4.0)
        offset = (
            keypoints_xy[nearest] - keypoints_xy[start:end, None]
        ) / scale[:, None, None]
        basis = _relation_basis(offset)
        weights = (
            torch.exp(-0.5 * (nearest_distance / scale[:, None]).square())
            * scores[nearest].clamp_min(0.0)
        )
        moments = _weighted_descriptor_moments(
            descriptors[nearest], basis, weights
        )
        normalized_weights = weights / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        radius = torch.linalg.vector_norm(offset, dim=-1)
        geometry = torch.stack(
            (
                scale.log(),
                (normalized_weights * radius).sum(dim=1),
                (
                    normalized_weights
                    * (radius - (normalized_weights * radius).sum(dim=1)[:, None]).square()
                ).sum(dim=1),
                (nearest_distance < scale[:, None]).float().mean(dim=1),
            ),
            dim=1,
        )
        output[start:end] = torch.cat((moments, geometry), dim=1)
    return F.normalize(output, dim=1)


def relational_map_3d_context(
    anchor_features: torch.Tensor,
    anchor_xyz: torch.Tensor,
    tangent_x: torch.Tensor,
    tangent_y: torch.Tensor,
    normals: torch.Tensor,
    *,
    source_ids: torch.Tensor,
    track_ids: torch.Tensor,
    surface_scale: torch.Tensor,
    neighbor_count: int = 16,
    candidate_multiplier: int = 4,
    minimum_normal_cosine: float = 0.25,
    same_source_weight: float = 0.25,
    same_track_weight: float = 1.5,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Encode true relative geometry in each 2DGS tangent-normal frame."""

    features = F.normalize(torch.as_tensor(anchor_features).float(), dim=1)
    xyz = torch.as_tensor(
        anchor_xyz, device=features.device, dtype=features.dtype
    ).reshape(-1, 3)
    tangent_x = F.normalize(
        torch.as_tensor(tangent_x, device=features.device).float(), dim=1
    )
    tangent_y = F.normalize(
        torch.as_tensor(tangent_y, device=features.device).float(), dim=1
    )
    normals = F.normalize(
        torch.as_tensor(normals, device=features.device).float(), dim=1
    )
    source_ids = torch.as_tensor(
        source_ids, device=features.device
    ).long().reshape(-1)
    track_ids = torch.as_tensor(
        track_ids, device=features.device
    ).long().reshape(-1)
    surface_scale = torch.as_tensor(
        surface_scale, device=features.device
    ).float().reshape(-1).clamp_min(1e-5)
    length = len(features)
    if not all(
        len(value) == length
        for value in (
            xyz,
            tangent_x,
            tangent_y,
            normals,
            source_ids,
            track_ids,
            surface_scale,
        )
    ):
        raise ValueError("relational map inputs must have aligned rows")
    if length < 2:
        raise ValueError("relational map context requires at least two anchors")
    count = min(max(int(neighbor_count), 1), length - 1)
    pool = min(
        max(count * max(int(candidate_multiplier), 1), count),
        length - 1,
    )
    output = features.new_empty((length, 6 * features.shape[1] + 8))
    all_rows = torch.arange(length, device=features.device)
    for start in range(0, length, max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), length)
        rows = all_rows[start:end]
        distance = torch.cdist(xyz[rows].float(), xyz.float())
        local_rows = torch.arange(end - start, device=distance.device)
        distance[local_rows, rows] = torch.inf
        nearest_distance, nearest = torch.topk(
            distance, k=pool, dim=1, largest=False, sorted=True
        )
        normal_cosine = (
            normals[nearest] * normals[rows, None]
        ).sum(dim=-1)
        valid = normal_cosine >= float(minimum_normal_cosine)
        penalized = nearest_distance.masked_fill(~valid, torch.inf)
        selected_distance, order = torch.topk(
            penalized, k=count, dim=1, largest=False, sorted=True
        )
        selected_valid = torch.isfinite(selected_distance)
        selected = torch.gather(nearest, 1, order)
        selected_normal_cosine = torch.gather(normal_cosine, 1, order)
        delta = xyz[selected] - xyz[rows, None]
        valid_count = selected_valid.sum(dim=1)
        median_slot = ((valid_count - 1).clamp_min(0) // 2).reshape(-1, 1)
        compatible_scale = torch.gather(
            selected_distance.masked_fill(~selected_valid, 0.0),
            1,
            median_slot,
        ).reshape(-1)
        compatible_scale = torch.where(
            valid_count > 0,
            compatible_scale,
            surface_scale[rows],
        )
        adaptive_scale = torch.maximum(
            compatible_scale,
            surface_scale[rows],
        ).clamp_min(1e-5)
        local_x = torch.einsum(
            "nkj,nj->nk", delta, tangent_x[rows]
        ) / adaptive_scale[:, None]
        local_y = torch.einsum(
            "nkj,nj->nk", delta, tangent_y[rows]
        ) / adaptive_scale[:, None]
        local_n = torch.einsum(
            "nkj,nj->nk", delta, normals[rows]
        ) / adaptive_scale[:, None]
        offset = torch.stack((local_x, local_y), dim=-1)
        basis = _relation_basis(offset)
        same_source = source_ids[selected] == source_ids[rows, None]
        same_track = (
            (track_ids[rows, None] >= 0)
            & (track_ids[selected] >= 0)
            & (track_ids[selected] == track_ids[rows, None])
        )
        weights = torch.exp(
            -0.5
            * (
                selected_distance.masked_fill(~selected_valid, 0.0)
                / adaptive_scale[:, None]
            ).square()
        )
        weights = weights * selected_valid
        weights = weights * selected_normal_cosine.clamp_min(0.0)
        weights = weights * torch.where(
            same_source,
            weights.new_full((), float(same_source_weight)),
            weights.new_ones(()),
        )
        weights = weights * torch.where(
            same_track,
            weights.new_full((), float(same_track_weight)),
            weights.new_ones(()),
        )
        moments = _weighted_descriptor_moments(
            features[selected], basis, weights
        )
        normalized_weights = weights / weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        geometry = torch.stack(
            (
                adaptive_scale.log(),
                (normalized_weights * local_n).sum(dim=1),
                (normalized_weights * local_n.abs()).sum(dim=1),
                (normalized_weights * selected_normal_cosine).sum(dim=1),
                (
                    normalized_weights
                    * (selected_normal_cosine - (
                        normalized_weights * selected_normal_cosine
                    ).sum(dim=1)[:, None]).square()
                ).sum(dim=1),
                (normalized_weights * same_source.float()).sum(dim=1),
                (normalized_weights * same_track.float()).sum(dim=1),
                selected_valid.float().mean(dim=1),
            ),
            dim=1,
        )
        output[start:end] = torch.cat((moments, geometry), dim=1)
    return F.normalize(output, dim=1)


class AsymmetricBoundedDualContextEncoder(nn.Module):
    """Bounded adapters sharing the descriptor-moment projection."""

    def __init__(
        self,
        query_input_dim: int,
        map_input_dim: int,
        output_dim: int = 64,
        rank: int = 32,
        maximum_residual: float = 0.35,
        seed: int = 2026,
    ):
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        shared = min(int(query_input_dim), int(map_input_dim))
        common = torch.randn(
            int(output_dim), shared, generator=generator
        )
        query_base = torch.zeros(int(output_dim), int(query_input_dim))
        map_base = torch.zeros(int(output_dim), int(map_input_dim))
        query_base[:, :shared] = common
        map_base[:, :shared] = common
        query_base = F.normalize(query_base, dim=1)
        map_base = F.normalize(map_base, dim=1)
        common_kwargs = {
            "output_dim": output_dim,
            "rank": rank,
            "maximum_residual": maximum_residual,
            "seed": seed,
        }
        self.query = BoundedContextProjector(
            input_dim=query_input_dim,
            base_projection=query_base,
            **common_kwargs,
        )
        self.map = BoundedContextProjector(
            input_dim=map_input_dim,
            base_projection=map_base,
            **common_kwargs,
        )


class QueryAmbiguityGate(nn.Module):
    """Predict whether relational context may alter a local assignment."""

    def __init__(self, context_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(self.context_dim + 1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self, context: torch.Tensor, keypoint_score: torch.Tensor
    ) -> torch.Tensor:
        context = F.normalize(torch.as_tensor(context).float(), dim=1)
        keypoint_score = torch.as_tensor(
            keypoint_score,
            device=context.device,
            dtype=context.dtype,
        ).reshape(-1, 1)
        if len(context) != len(keypoint_score):
            raise ValueError("gate context and keypoint scores must align")
        return torch.sigmoid(
            self.network(torch.cat((context, keypoint_score), dim=1))
        ).reshape(-1)

    def export_config(self) -> dict:
        return {
            "context_dim": self.context_dim,
            "hidden_dim": self.hidden_dim,
        }
