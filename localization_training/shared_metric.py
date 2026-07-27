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
