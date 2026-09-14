"""Bounded shared descriptor metric used by the frozen A1 method."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SharedLowRankMetric(nn.Module):
    """A bounded low-rank residual shared by query and map descriptors."""

    def __init__(
        self,
        descriptor_dim: int = 256,
        rank: int = 16,
        max_residual_norm: float = 0.10,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.rank = int(rank)
        self.max_residual_norm = float(max_residual_norm)
        self.down = nn.Linear(self.descriptor_dim, self.rank)
        self.up = nn.Linear(self.rank, self.descriptor_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, descriptor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        descriptor = F.normalize(torch.as_tensor(descriptor), dim=-1)
        residual = self.up(F.silu(self.down(descriptor)))
        norm = torch.linalg.norm(residual, dim=-1, keepdim=True)
        residual = residual * torch.clamp(
            self.max_residual_norm / norm.clamp_min(1e-8), max=1.0
        )
        return F.normalize(descriptor + residual, dim=-1), residual

    def export_config(self) -> dict[str, int | float]:
        return {
            "descriptor_dim": self.descriptor_dim,
            "rank": self.rank,
            "max_residual_norm": self.max_residual_norm,
        }


def validate_map_bound_identity_metric(
    payload: dict,
    *,
    descriptor_dim: int,
    anchor_count: int,
    map_path: str,
    map_sha256: str,
) -> None:
    """Fail closed unless ``payload`` is the V4 zero-transform metric shim.

    The generic low-rank metric remains available to reproduce older and mixed
    experiments.  The render-only V4 method deliberately has no learned
    descriptor transform; this validator prevents the legacy interface from
    silently reintroducing one.
    """
    if (
        payload.get("schema") != "lafgs_shared_metric_state"
        or payload.get("version") != 1
        or payload.get("protocol") != "rendered_track_map_bound_identity"
        or payload.get("map_path") != str(map_path)
        or payload.get("map_sha256") != str(map_sha256)
        or payload.get("step") != 0
    ):
        raise ValueError("render-only V4 requires an exact map-bound identity metric")
    config = payload.get("metric_config")
    if config != {
        "descriptor_dim": int(descriptor_dim),
        "rank": 1,
        "max_residual_norm": 0.0,
    }:
        raise ValueError("render-only V4 metric configuration is not identity-only")
    landmark_indices = torch.as_tensor(payload.get("landmark_indices"))
    expected_indices = torch.arange(int(anchor_count), dtype=torch.long)
    if (
        landmark_indices.dtype != torch.long
        or landmark_indices.shape != expected_indices.shape
        or not torch.equal(landmark_indices, expected_indices)
    ):
        raise ValueError("identity metric landmark rows do not match the map")
    metric = SharedLowRankMetric(**config)
    state = payload.get("metric_state_dict")
    if not isinstance(state, dict):
        raise ValueError("identity metric state dict is missing")
    try:
        metric.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("identity metric state dict is not exact") from error
    if any(bool(torch.count_nonzero(parameter)) for parameter in metric.parameters()):
        raise ValueError("render-only V4 forbids learned descriptor parameters")


def validate_zero_identity_metric(
    payload: dict,
    *,
    descriptor_dim: int,
    landmark_indices: torch.Tensor,
    map_path: str,
    map_sha256: str,
    allowed_protocols: set[str],
) -> None:
    """Validate the behavior and map lineage of a generic zero metric shim."""

    if (
        payload.get("schema") != "lafgs_shared_metric_state"
        or payload.get("version") != 1
        or payload.get("protocol") not in allowed_protocols
        or payload.get("map_path") != str(map_path)
        or payload.get("map_sha256") != str(map_sha256)
        or payload.get("step") != 0
    ):
        raise ValueError("identity metric lineage or protocol differs")
    config = payload.get("metric_config")
    if config != {
        "descriptor_dim": int(descriptor_dim),
        "rank": 1,
        "max_residual_norm": 0.0,
    }:
        raise ValueError("identity metric configuration is not zero-transform")
    expected = torch.as_tensor(landmark_indices).long().reshape(-1)
    actual = torch.as_tensor(payload.get("landmark_indices"))
    if (
        actual.dtype != torch.long
        or actual.shape != expected.shape
        or not torch.equal(actual, expected)
    ):
        raise ValueError("identity metric landmark registry differs from its map")
    metric = SharedLowRankMetric(**config)
    state = payload.get("metric_state_dict")
    if not isinstance(state, dict):
        raise ValueError("identity metric state dict is missing")
    try:
        metric.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("identity metric state dict is not exact") from error
    if any(bool(torch.count_nonzero(parameter)) for parameter in metric.parameters()):
        raise ValueError("identity metric contains a learned descriptor transform")
