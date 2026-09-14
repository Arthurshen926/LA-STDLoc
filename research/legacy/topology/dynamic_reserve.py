"""Dynamic full-SE(3) pose-information reserve with additive diversity."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence

import torch

from topology.pose_information import conditional_add_gain, translation_schur_complement


@dataclass(frozen=True)
class PoseEvidence:
    query: int
    rows: tuple[int, ...]
    information: torch.Tensor
    image_cell: int
    depth_bin: int
    spatial_voxel: int
    matchability: float = 1.0


class IncrementalPoseRowAssignment:
    """Maintain feasible candidate-to-row assignments with augmenting paths."""

    def __init__(
        self,
        evidence_by_candidate: Sequence[Sequence[PoseEvidence]],
        initial_used_rows: Sequence[set[int]],
        initial_assignments: Sequence[dict[int, int]] | None = None,
    ) -> None:
        self.evidence = evidence_by_candidate
        self.query_count = len(initial_used_rows)
        self.row_to_candidate = [dict() for _ in range(self.query_count)]
        self.candidate_to_row = [dict() for _ in range(self.query_count)]
        self.fixed_rows = [set(values) for values in initial_used_rows]
        self.reassignment_count = 0
        if initial_assignments is not None:
            for query, assignments in enumerate(initial_assignments):
                for candidate, row in assignments.items():
                    candidate, row = int(candidate), int(row)
                    self.row_to_candidate[query][row] = candidate
                    self.candidate_to_row[query][candidate] = row
                    self.fixed_rows[query].discard(row)

    def _rows(self, candidate: int, query: int) -> tuple[int, ...]:
        for item in self.evidence[int(candidate)]:
            if int(item.query) == int(query):
                return item.rows
        return ()

    def _augment(
        self,
        candidate: int,
        query: int,
        row_to_candidate: dict[int, int],
        candidate_to_row: dict[int, int],
        seen_candidates: set[int],
        seen_rows: set[int],
    ) -> bool:
        if candidate in seen_candidates:
            return False
        seen_candidates.add(candidate)
        for row in self._rows(candidate, query):
            row = int(row)
            if row in self.fixed_rows[query] or row in seen_rows:
                continue
            seen_rows.add(row)
            previous = row_to_candidate.get(row)
            if previous is None or self._augment(
                previous,
                query,
                row_to_candidate,
                candidate_to_row,
                seen_candidates,
                seen_rows,
            ):
                old_row = candidate_to_row.get(candidate)
                if old_row is not None and old_row != row:
                    row_to_candidate.pop(old_row, None)
                row_to_candidate[row] = candidate
                candidate_to_row[candidate] = row
                return True
        return False

    def would_augment(self, candidate: int, query: int) -> bool:
        return self._augment(
            int(candidate),
            int(query),
            dict(self.row_to_candidate[int(query)]),
            dict(self.candidate_to_row[int(query)]),
            set(),
            set(),
        )

    def add(self, candidate: int) -> set[int]:
        augmented_queries: set[int] = set()
        for item in self.evidence[int(candidate)]:
            query = int(item.query)
            before = dict(self.candidate_to_row[query])
            if self._augment(
                int(candidate),
                query,
                self.row_to_candidate[query],
                self.candidate_to_row[query],
                set(),
                set(),
            ):
                augmented_queries.add(query)
                self.reassignment_count += sum(
                    before.get(anchor) != row
                    for anchor, row in self.candidate_to_row[query].items()
                    if anchor in before
                )
        return augmented_queries

    def used_rows(self, query: int) -> set[int]:
        return self.fixed_rows[int(query)] | set(self.row_to_candidate[int(query)])


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
    minimum_objective_relative_gain: float = 0.0,
    minimum_additions: int = 0,
    image_diversity_weight: float = 0.05,
    depth_diversity_weight: float = 0.05,
    voxel_diversity_weight: float = 0.05,
    initial_assignments: Sequence[dict[int, int]] | None = None,
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
    assignment = IncrementalPoseRowAssignment(
        evidence_by_candidate, initial_used_rows, initial_assignments
    )
    image_cells = [set(values) for values in initial_image_cells]
    depth_bins = [set(values) for values in initial_depth_bins]
    spatial_voxels = [set(values) for values in initial_spatial_voxels]
    source_ids = torch.as_tensor(source_ids).long()
    voxel_ids = torch.as_tensor(voxel_ids).long()

    def score(candidate: int) -> float:
        value = 0.0
        for item in evidence_by_candidate[candidate]:
            if not assignment.would_augment(candidate, int(item.query)):
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
    cumulative_gain = 0.0
    iteration = 0
    stop_reason = "candidate_exhaustion"
    history: list[dict] = []
    diagnostics: list[dict] = []

    def append_diagnostics(additions: int) -> None:
        matrix = 0.5 * (information + information.transpose(1, 2))
        logdet = torch.linalg.slogdet(matrix)[1]
        translation = translation_schur_complement(matrix)
        covariance = torch.linalg.pinv(translation)
        worst_std = torch.linalg.eigvalsh(covariance).clamp_min(0)[:, -1].sqrt()
        diagnostics.append(
            {
                "additions": int(additions),
                "full_logdet_p10": float(torch.quantile(logdet, 0.1)),
                "full_logdet_median": float(torch.quantile(logdet, 0.5)),
                "translation_worst_std_p90": float(
                    torch.quantile(worst_std, 0.9)
                ),
            }
        )

    append_diagnostics(0)
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
        elif (
            len(selected) >= int(minimum_additions)
            and current < first_gain * float(minimum_relative_gain)
        ):
            stop_reason = "relative_marginal_gain"
            break
        objective_relative = current / max(cumulative_gain + current, 1e-12)
        if (
            len(selected) >= int(minimum_additions)
            and float(minimum_objective_relative_gain) > 0
            and objective_relative < float(minimum_objective_relative_gain)
        ):
            stop_reason = "objective_relative_marginal_gain"
            break
        selected.append(candidate)
        selected_scores.append(current)
        selected_set.add(candidate)
        source_count[source] = source_count.get(source, 0) + 1
        voxel_count[voxel] = voxel_count.get(voxel, 0) + 1
        augmented_queries = assignment.add(candidate)
        for item in evidence_by_candidate[candidate]:
            query = int(item.query)
            if query not in augmented_queries:
                continue
            information[query] += item.information
            image_cells[query].add(int(item.image_cell))
            depth_bins[query].add(int(item.depth_bin))
            spatial_voxels[query].add(int(item.spatial_voxel))
        cumulative_gain += current
        history.append(
            {
                "additions": len(selected),
                "marginal_gain": float(current),
                "cumulative_utility_gain": float(cumulative_gain),
                "objective_relative_marginal_gain": float(objective_relative),
            }
        )
        if len(selected) % 64 == 0:
            append_diagnostics(len(selected))
        iteration += 1
    if len(selected) >= int(maximum_additions):
        stop_reason = "maximum_additions"
    if not diagnostics or diagnostics[-1]["additions"] != len(selected):
        append_diagnostics(len(selected))
    report = {
        "objective": "task_scaled_full_se3_logdet_plus_additive_capped_diversity",
        "duplicate_query_rows_counted": False,
        "selection_is_dynamic": True,
        "force_fill": False,
        "selected_count": len(selected),
        "maximum_additions": int(maximum_additions),
        "minimum_relative_gain": float(minimum_relative_gain),
        "minimum_objective_relative_gain": float(
            minimum_objective_relative_gain
        ),
        "minimum_additions": int(minimum_additions),
        "first_gain": float(first_gain or 0.0),
        "last_gain": float(selected_scores[-1] if selected_scores else 0.0),
        "stop_reason": stop_reason,
        "augmenting_row_assignment": True,
        "row_reassignment_count": int(assignment.reassignment_count),
        "marginal_history": history,
        "information_diagnostics": diagnostics,
        "theoretical_scope": (
            "D-opt logdet is monotone submodular under a cardinality constraint; "
            "source and spatial capacities are practical safeguards without a "
            "claimed 1-1/e guarantee"
        ),
        "source_capacity": int(maximum_per_source),
        "spatial_voxel_capacity": int(maximum_per_voxel),
    }
    return torch.as_tensor(selected, dtype=torch.long), report
