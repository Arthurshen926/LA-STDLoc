"""Matching-feasible query coverage for localization-map distillation."""

from __future__ import annotations

from collections import defaultdict
import heapq
from typing import Mapping, Sequence

import numpy as np
import torch


CandidateEdges = Sequence[Mapping[int, Sequence[int]]]


class IncrementalBipartiteCoverage:
    """Maintain one maximum anchor-to-keypoint matching per mapping query."""

    def __init__(self, query_count: int, edges: CandidateEdges) -> None:
        self.query_count = int(query_count)
        self.edges = edges
        self.selected: set[int] = set()
        self._row_to_anchor: list[dict[int, int]] = [
            {} for _ in range(self.query_count)
        ]
        self._anchor_to_row: list[dict[int, int]] = [
            {} for _ in range(self.query_count)
        ]

    def _augment(
        self,
        query: int,
        anchor: int,
        row_to_anchor: dict[int, int],
        anchor_to_row: dict[int, int],
        seen_anchors: set[int],
        seen_rows: set[int],
    ) -> bool:
        if anchor in seen_anchors:
            return False
        seen_anchors.add(anchor)
        for row in self.edges[anchor].get(query, ()):
            row = int(row)
            if row in seen_rows:
                continue
            seen_rows.add(row)
            previous = row_to_anchor.get(row)
            if previous is None or self._augment(
                query,
                previous,
                row_to_anchor,
                anchor_to_row,
                seen_anchors,
                seen_rows,
            ):
                row_to_anchor[row] = anchor
                anchor_to_row[anchor] = row
                return True
        return False

    def would_augment(self, anchor: int, query: int) -> bool:
        if anchor in self.selected or query not in self.edges[anchor]:
            return False
        return self._augment(
            query,
            anchor,
            dict(self._row_to_anchor[query]),
            dict(self._anchor_to_row[query]),
            set(),
            set(),
        )

    def add(self, anchor: int) -> int:
        anchor = int(anchor)
        if anchor in self.selected:
            return 0
        gained = 0
        for query in self.edges[anchor]:
            query = int(query)
            if self._augment(
                query,
                anchor,
                self._row_to_anchor[query],
                self._anchor_to_row[query],
                set(),
                set(),
            ):
                gained += 1
        self.selected.add(anchor)
        return gained

    @property
    def counts(self) -> np.ndarray:
        return np.asarray(
            [len(matches) for matches in self._row_to_anchor], dtype=np.int64
        )

    def assignments(self, query: int) -> dict[int, int]:
        """Return selected anchor -> matched query row for one query."""
        return dict(self._anchor_to_row[int(query)])


def track_candidate_edges(
    payload: Mapping,
    *,
    query_index_remap: torch.Tensor | None = None,
) -> list[dict[int, tuple[int, ...]]]:
    track_count = int(
        torch.as_tensor(payload["track_geometry"]["triangulated"]).numel()
    )
    pending: list[dict[int, set[int]]] = [defaultdict(set) for _ in range(track_count)]
    observations = payload["tracks"]
    track = torch.as_tensor(observations["track_index"]).long()
    query = torch.as_tensor(observations["query_index"]).long()
    row = torch.as_tensor(observations["keypoint_index"]).long()
    certified = torch.as_tensor(
        observations.get(
            "coverage_certified", torch.ones(track.shape[0], dtype=torch.bool)
        )
    ).bool()
    if certified.shape != track.shape:
        raise ValueError("Track coverage certification and observations differ")
    if query_index_remap is not None:
        query = torch.as_tensor(query_index_remap).long()[query]
    for track_index, query_index, keypoint_index in zip(
        track[certified].tolist(),
        query[certified].tolist(),
        row[certified].tolist(),
    ):
        pending[track_index][query_index].add(keypoint_index)
    return [
        {query: tuple(sorted(rows)) for query, rows in candidate.items()}
        for candidate in pending
    ]


def base_candidate_edges(
    teacher: Mapping, candidate_count: int
) -> list[dict[int, tuple[int, ...]]]:
    pending: list[dict[int, set[int]]] = [
        defaultdict(set) for _ in range(int(candidate_count))
    ]
    for record in teacher["records"]:
        query = int(record["query_index"])
        rows = torch.as_tensor(record["query_rows"]).long()
        offsets = torch.as_tensor(record["positive_offsets"]).long()
        indices = torch.as_tensor(record["positive_indices"]).long()
        for local_row, keypoint in enumerate(rows.tolist()):
            start, end = int(offsets[local_row]), int(offsets[local_row + 1])
            for anchor in indices[start:end].tolist():
                if 0 <= int(anchor) < int(candidate_count):
                    pending[int(anchor)][query].add(int(keypoint))
    return [
        {query: tuple(sorted(rows)) for query, rows in candidate.items()}
        for candidate in pending
    ]


def query_weights_from_groups(query_groups: torch.Tensor) -> np.ndarray:
    groups = torch.as_tensor(query_groups).long().reshape(-1)
    sizes = torch.bincount(groups).float().clamp_min(1)
    return torch.sqrt(sizes.float().mean() / sizes[groups]).numpy()


def greedy_matching_reserve(
    edges: CandidateEdges,
    initial: Sequence[int],
    candidates: Sequence[int],
    utility: torch.Tensor,
    query_groups: torch.Tensor,
    *,
    requested_rows_per_query: int | Sequence[int],
    maximum_reserve: int,
    alias_risk: torch.Tensor | None = None,
) -> tuple[torch.Tensor, IncrementalBipartiteCoverage, dict]:
    """Greedily meet capacity-constrained matching rank, not row union."""
    query_count = int(torch.as_tensor(query_groups).numel())
    state = IncrementalBipartiteCoverage(query_count, edges)
    for anchor in initial:
        state.add(int(anchor))
    core_counts = state.counts.copy()

    candidates = [int(value) for value in candidates]
    feasible = IncrementalBipartiteCoverage(query_count, edges)
    for anchor in initial:
        feasible.add(int(anchor))
    for anchor in candidates:
        feasible.add(anchor)
    requested = np.broadcast_to(
        np.asarray(requested_rows_per_query, dtype=np.int64), (query_count,)
    ).copy()
    targets = np.minimum(requested, feasible.counts)
    weights = query_weights_from_groups(query_groups)
    utility = torch.as_tensor(utility).float().reshape(-1)
    risk = None
    if alias_risk is not None:
        risk = torch.as_tensor(alias_risk).float().reshape(-1)
        if risk.numel() != utility.numel():
            raise ValueError("alias risk must align with candidate utility")

    def gain(anchor: int) -> float:
        counts = state.counts
        value = 0.0
        for query in edges[anchor]:
            if counts[query] < targets[query] and state.would_augment(anchor, query):
                value += float(weights[query])
        return value

    heap: list[tuple[float, float, float, int]] = []
    for anchor in candidates:
        candidate_gain = gain(anchor)
        if candidate_gain > 0:
            candidate_risk = (
                0.0 if risk is None else float(torch.nan_to_num(risk[anchor], nan=1.0))
            )
            heapq.heappush(
                heap,
                (-candidate_gain, candidate_risk, -float(utility[anchor]), anchor),
            )
    selected: list[int] = []
    while heap and len(selected) < int(maximum_reserve):
        negative_gain, candidate_risk, negative_utility, anchor = heapq.heappop(heap)
        candidate_gain = gain(anchor)
        if not np.isclose(candidate_gain, -negative_gain, atol=1e-9, rtol=0):
            if candidate_gain > 0:
                heapq.heappush(
                    heap,
                    (-candidate_gain, candidate_risk, negative_utility, anchor),
                )
            continue
        if candidate_gain <= 0:
            break
        state.add(anchor)
        selected.append(anchor)
        if bool((state.counts >= targets).all()):
            break

    final_counts = state.counts
    deficits = np.maximum(targets - final_counts, 0)
    normalized = final_counts / np.maximum(targets, 1)
    report = {
        "coverage_definition": "query_anchor_bipartite_matching_rank",
        "requested_rows_per_query": (
            int(requested[0])
            if bool((requested == requested[0]).all())
            else requested.tolist()
        ),
        "reserve_count": len(selected),
        "feasible_target_count": int(targets.sum()),
        "achieved_matching_rank": int(np.minimum(final_counts, targets).sum()),
        "unmet_query_count": int((deficits > 0).sum()),
        "unmet_rank": int(deficits.sum()),
        "core_rank_p10": float(np.percentile(core_counts, 10)),
        "core_rank_median": float(np.median(core_counts)),
        "final_rank_p10": float(np.percentile(final_counts, 10)),
        "final_rank_median": float(np.median(final_counts)),
        "normalized_coverage_p10": float(np.percentile(normalized, 10)),
        "feasibility_limited_query_count": int((targets < requested).sum()),
        "alias_risk_tiebreak_enabled": risk is not None,
        "alias_risk_unknown_policy": (
            None if risk is None else "unknown_after_supported_risk"
        ),
    }
    return torch.as_tensor(selected, dtype=torch.long), state, report
