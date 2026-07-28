from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SharedLowRankMetric(nn.Module):
    """Bounded low-rank residual metric shared by query and map descriptors."""

    def __init__(
        self,
        descriptor_dim: int = 256,
        rank: int = 16,
        max_residual_norm: float = 0.10,
    ):
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.rank = int(rank)
        self.max_residual_norm = float(max_residual_norm)
        self.down = nn.Linear(self.descriptor_dim, self.rank)
        self.up = nn.Linear(self.rank, self.descriptor_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, descriptor: torch.Tensor):
        descriptor = F.normalize(torch.as_tensor(descriptor), dim=-1)
        residual = self.up(F.silu(self.down(descriptor)))
        norm = torch.linalg.norm(residual, dim=-1, keepdim=True)
        residual = residual * torch.clamp(
            self.max_residual_norm / norm.clamp_min(1e-8), max=1.0
        )
        return F.normalize(descriptor + residual, dim=-1), residual

    def export_config(self) -> dict:
        return {
            "descriptor_dim": self.descriptor_dim,
            "rank": self.rank,
            "max_residual_norm": self.max_residual_norm,
        }


class NativeNullHead(nn.Module):
    """Calibrate whether a native top-1 correspondence should enter PnP."""

    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.linear = nn.Linear(self.feature_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = torch.as_tensor(features)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"null features must have shape [N, {self.feature_dim}]"
            )
        return self.linear(features).squeeze(1)


def build_native_null_features(
    top_scores: torch.Tensor,
    keypoint_scores: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Build scene-normalized score, margin, entropy, and detector features."""
    scores = torch.as_tensor(top_scores)
    keypoint = torch.as_tensor(
        keypoint_scores, device=scores.device, dtype=scores.dtype
    ).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] != keypoint.numel():
        raise ValueError("top scores and keypoint scores must align")
    if scores.shape[1] < 2:
        raise ValueError("null calibration requires at least top-2 scores")
    probability = torch.softmax(
        (scores - scores[:, :1]) / float(temperature), dim=1
    )
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1)
    entropy = entropy / torch.log(
        torch.as_tensor(
            float(scores.shape[1]), device=scores.device, dtype=scores.dtype
        )
    )
    return torch.stack(
        (
            scores[:, 0],
            scores[:, 0] - scores[:, 1],
            entropy,
            keypoint,
        ),
        dim=1,
    )


def select_native_matchable_rows(
    probability: torch.Tensor,
    keypoints: torch.Tensor,
    *,
    width: int,
    height: int,
    threshold: float,
    minimum_total: int,
    grid_rows: int = 4,
    grid_cols: int = 4,
    minimum_per_cell: int = 8,
) -> torch.Tensor:
    """Threshold null probabilities while preserving global and spatial floors."""
    probability = torch.as_tensor(probability).reshape(-1)
    keypoints = torch.as_tensor(keypoints, device=probability.device)
    if keypoints.shape != (probability.numel(), 2):
        raise ValueError("keypoints must align with null probabilities")
    keep = probability >= float(threshold)
    ranking = torch.argsort(probability, descending=True, stable=True)
    minimum_total = min(max(int(minimum_total), 0), probability.numel())
    keep[ranking[:minimum_total]] = True
    if int(grid_rows) > 0 and int(grid_cols) > 0 and int(minimum_per_cell) > 0:
        col = torch.clamp(
            (keypoints[:, 0] * int(grid_cols) / max(int(width), 1)).long(),
            0,
            int(grid_cols) - 1,
        )
        row = torch.clamp(
            (keypoints[:, 1] * int(grid_rows) / max(int(height), 1)).long(),
            0,
            int(grid_rows) - 1,
        )
        cell = row * int(grid_cols) + col
        for cell_id in range(int(grid_rows) * int(grid_cols)):
            local = ranking[cell[ranking] == cell_id]
            keep[local[: int(minimum_per_cell)]] = True
    return torch.nonzero(keep, as_tuple=False).reshape(-1)
