"""Dynamic full-SE(3) pose-information reserve with additive diversity."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import torch

from topology.pose_information import conditional_add_gain


@dataclass(frozen=True)
class PoseEvidence:
    query: int
    rows: tuple[int, ...]
    information: torch.Tensor
    image_cell: int
    depth_bin: int
    spatial_voxel: int


def spatial_voxel_ids(xyz: torch.Tensor, voxel_size: float) -> torch.Tensor:
    """Return pure spatial groups; primitive lineage is deliberately excluded."""
    xyz = torch.as_tensor(xyz).float()
    coordinates = torch.floor(xyz / float(voxel_size)).long()
    return torch.unique(coordinates, dim=0, return_inverse=True)[1]


def greedy_dynamic_pose_reserve(
    evidence_by_candidate: Sequence[Sequence[PoseEvidence]],
    initial_information: torch.Tensor,
    initial_used_rows: Sequence[set[int]],
    initial_image_cells: Sequence[set[int]],
    initial_depth_bins: Sequence[set[int]],
    initial_spatial_voxels: Sequence[set[int]],
    candidates: Sequence[int],
    source_ids: torch.Tensor,
    voxel_ids: torch.Tensor,
    *,
    maximum_additions: int,
    query_weights: torch.Tensor | None = None,
    maximum_per_source: int = 1,
    maximum_per_voxel: int = 3,
    minimum_relative_gain: float = 0.02,
    image_diversity_weight: float = 0.05,
    depth_diversity_weight: float = 0.05,
    voxel_diversity_weight: float = 0.05,
) -> tuple[torch.Tensor, dict]:
    """Lazy-greedy D-opt selection, updating every query Hessian after each add."""
    information = torch.as_tensor(initial_information).double().clone()
    query_count = int(information.shape[0])
    if information.shape != (query_count, 6, 6):
        raise ValueError("initial information must have shape [Q, 6, 6]")
    query_weights = (
        torch.ones(query_count, dtype=torch.float64)
        if query_weights is None
        else torch.as_tensor(query_weights).double().reshape(-1)
    )
    if query_weights.numel() != query_count:
        raise ValueError("query weights must align with information matrices")
    used_rows = [set(values) for values in initial_used_rows]
    image_cells = [set(values) for values in initial_image_cells]
    depth_bins = [set(values) for values in initial_depth_bins]
    spatial_voxels = [set(values) for values in initial_spatial_voxels]
    source_ids = torch.as_tensor(source_ids).long()
    voxel_ids = torch.as_tensor(voxel_ids).long()

    def available_row(item: PoseEvidence) -> int | None:
        return next(
            (int(row) for row in item.rows if int(row) not in used_rows[item.query]),
            None,
        )

    def score(candidate: int) -> float:
        value = 0.0
        for item in evidence_by_candidate[candidate]:
            if available_row(item) is None:
                continue
            query = int(item.query)
            gain = conditional_add_gain(
                information[query], item.information, objective="full"
            )
            if torch.isfinite(gain):
                value += float(query_weights[query] * gain.clamp_min(0))
            value += float(query_weights[query]) * (
                float(image_diversity_weight)
                * (item.image_cell not in image_cells[query])
                + float(depth_diversity_weight)
                * (item.depth_bin not in depth_bins[query])
                + float(voxel_diversity_weight)
                * (item.spatial_voxel not in spatial_voxels[query])
            )
        return value

    heap: list[tuple[float, int, int]] = []
    for candidate in candidates:
        candidate = int(candidate)
        value = score(candidate)
        if value > 0:
            heapq.heappush(heap, (-value, 0, candidate))
    selected: list[int] = []
    selected_scores: list[float] = []
    selected_set: set[int] = set()
    source_count: dict[int, int] = {}
    voxel_count: dict[int, int] = {}
    first_gain = None
    iteration = 0
    stop_reason = "candidate_exhaustion"
    while heap and len(selected) < int(maximum_additions):
        negative_upper, stamp, candidate = heapq.heappop(heap)
        if candidate in selected_set:
            continue
        source = int(source_ids[candidate])
        voxel = int(voxel_ids[candidate])
        if source_count.get(source, 0) >= int(maximum_per_source):
            continue
        if voxel_count.get(voxel, 0) >= int(maximum_per_voxel):
            continue
        current = score(candidate)
        if stamp != iteration:
            if current > 0:
                heapq.heappush(heap, (-current, iteration, candidate))
            continue
        if current <= 0 or not math.isfinite(current):
            stop_reason = "nonpositive_gain"
            break
        if first_gain is None:
            first_gain = current
        elif current < first_gain * float(minimum_relative_gain):
            stop_reason = "relative_marginal_gain"
            break
        selected.append(candidate)
        selected_scores.append(current)
        selected_set.add(candidate)
        source_count[source] = source_count.get(source, 0) + 1
        voxel_count[voxel] = voxel_count.get(voxel, 0) + 1
        for item in evidence_by_candidate[candidate]:
            row = available_row(item)
            if row is None:
                continue
            query = int(item.query)
            information[query] += item.information
            used_rows[query].add(row)
            image_cells[query].add(int(item.image_cell))
            depth_bins[query].add(int(item.depth_bin))
            spatial_voxels[query].add(int(item.spatial_voxel))
        iteration += 1
    if len(selected) >= int(maximum_additions):
        stop_reason = "maximum_additions"
    report = {
        "objective": "task_scaled_full_se3_logdet_plus_additive_capped_diversity",
        "duplicate_query_rows_counted": False,
        "selection_is_dynamic": True,
        "force_fill": False,
        "selected_count": len(selected),
        "maximum_additions": int(maximum_additions),
        "minimum_relative_gain": float(minimum_relative_gain),
        "first_gain": float(first_gain or 0.0),
        "last_gain": float(selected_scores[-1] if selected_scores else 0.0),
        "stop_reason": stop_reason,
        "source_capacity": int(maximum_per_source),
        "spatial_voxel_capacity": int(maximum_per_voxel),
    }
    return torch.as_tensor(selected, dtype=torch.long), report
