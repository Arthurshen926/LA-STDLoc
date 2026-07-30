"""Geometry and sampling utilities for pose-sufficient P3P basis audits."""

from __future__ import annotations

import hashlib
from math import comb

import numpy as np
import torch


def triangle_shape_quality(points: torch.Tensor) -> torch.Tensor:
    """Return a scale-invariant triangle area divided by its longest edge squared."""

    points = torch.as_tensor(points).float()
    if points.ndim != 3 or points.shape[1] != 3 or points.shape[2] not in (2, 3):
        raise ValueError("triangle points must have shape [N, 3, 2|3]")
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    if points.shape[2] == 2:
        double_area = torch.abs(
            first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        )
    else:
        double_area = torch.linalg.norm(torch.cross(first, second, dim=1), dim=1)
    edge_squared = torch.stack(
        (
            (points[:, 1] - points[:, 0]).square().sum(dim=1),
            (points[:, 2] - points[:, 0]).square().sum(dim=1),
            (points[:, 2] - points[:, 1]).square().sum(dim=1),
        ),
        dim=1,
    ).max(dim=1).values
    return double_area / edge_squared.clamp_min(1e-12)


def image_triangle_area_fraction(
    points: torch.Tensor,
    image_hw: tuple[int, int] | list[int],
) -> torch.Tensor:
    """Return triangle area as a fraction of image area."""

    points = torch.as_tensor(points).float()
    if points.ndim != 3 or points.shape[1:] != (3, 2):
        raise ValueError("image triangle points must have shape [N, 3, 2]")
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    double_area = torch.abs(
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    )
    height, width = map(int, image_hw)
    return 0.5 * double_area / max(float(height * width), 1.0)


def deterministic_triplets(
    selected_indices: torch.Tensor,
    *,
    count: int,
    seed: int,
    query_name: str,
) -> torch.Tensor:
    """Sample uniform unique-row triplets reproducibly for one query."""

    selected = torch.as_tensor(selected_indices).long().reshape(-1).cpu()
    if len(selected) < 3 or int(count) <= 0:
        return torch.empty(0, 3, dtype=torch.long)
    query_seed = int.from_bytes(
        hashlib.sha256(str(query_name).encode("utf-8")).digest()[:8],
        "little",
    )
    rng = np.random.default_rng((int(seed) ^ query_seed) & ((1 << 63) - 1))
    maximum = comb(len(selected), 3)
    target = min(int(count), maximum)
    triplets: set[tuple[int, int, int]] = set()
    maximum_attempts = max(target * 20, 100)
    for _ in range(maximum_attempts):
        local = np.sort(rng.choice(len(selected), 3, replace=False))
        triplets.add(tuple(int(selected[index]) for index in local))
        if len(triplets) >= target:
            break
    if len(triplets) < target:
        raise RuntimeError("could not draw the requested unique triplets")
    return torch.as_tensor(sorted(triplets), dtype=torch.long)


def group_independent_triplets(
    triplets: torch.Tensor,
    dependency_groups: torch.Tensor,
    source_groups: torch.Tensor,
) -> torch.Tensor:
    """Require all three rows to come from distinct dependency and source groups."""

    triplets = torch.as_tensor(triplets).long().reshape(-1, 3)
    dependency = torch.as_tensor(dependency_groups).long().reshape(-1)[triplets]
    source = torch.as_tensor(source_groups).long().reshape(-1)[triplets]
    dependency_unique = (
        (dependency[:, 0] != dependency[:, 1])
        & (dependency[:, 0] != dependency[:, 2])
        & (dependency[:, 1] != dependency[:, 2])
    )
    source_unique = (
        (source[:, 0] != source[:, 1])
        & (source[:, 0] != source[:, 2])
        & (source[:, 1] != source[:, 2])
    )
    return dependency_unique & source_unique
