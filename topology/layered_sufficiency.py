"""Layered visibility, detectability, matching, and pose sufficiency for V6."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from topology.matching_coverage import CandidateEdges, IncrementalBipartiteCoverage


LAYER_NAMES = ("visibility", "detectability", "matching")


def _targets(value: int | Sequence[int], query_count: int) -> np.ndarray:
    result = np.broadcast_to(np.asarray(value, dtype=np.int64), (query_count,)).copy()
    if bool((result < 0).any()):
        raise ValueError("sufficiency targets must be non-negative")
    return result


def select_layered_sufficiency(
    *,
    layer_edges: dict[str, CandidateEdges],
    reliability: torch.Tensor,
    pose_information: torch.Tensor,
    matching_target: int | Sequence[int],
    pose_logdet_target: float,
    maximum_anchors: int,
) -> dict:
    """Deterministic hierarchical selection without a weighted heuristic sum."""

    if tuple(layer_edges) != LAYER_NAMES:
        raise ValueError(f"layer order must be exactly {LAYER_NAMES}")
    candidate_count = len(layer_edges["visibility"])
    if any(len(edges) != candidate_count for edges in layer_edges.values()):
        raise ValueError("layer candidate registries differ")
    reliability = torch.as_tensor(reliability).float().reshape(-1)
    information = torch.as_tensor(pose_information).double()
    if reliability.numel() != candidate_count:
        raise ValueError("reliability does not align with candidates")
    if information.ndim != 4 or information.shape[1] != candidate_count:
        raise ValueError("pose_information must have shape [Q,N,6,6]")
    query_count = int(information.shape[0])
    if information.shape[2:] != (6, 6):
        raise ValueError("pose information matrices must be 6x6")
    target = _targets(matching_target, query_count)
    states = {
        name: IncrementalBipartiteCoverage(query_count, layer_edges[name])
        for name in LAYER_NAMES
    }
    selected: list[int] = []
    selected_set: set[int] = set()
    trace = []
    order = torch.argsort(reliability, descending=True, stable=True).tolist()

    def add(candidate: int, reason: str) -> None:
        selected.append(candidate)
        selected_set.add(candidate)
        for state in states.values():
            state.add(candidate)
        trace.append({"anchor": candidate, "reason": reason})

    for layer in LAYER_NAMES:
        state = states[layer]
        while bool((state.counts < target).any()):
            feasible = [
                candidate
                for candidate in order
                if candidate not in selected_set
                and any(
                    state.counts[query] < target[query]
                    and state.would_augment(candidate, query)
                    for query in layer_edges[layer][candidate]
                )
            ]
            if not feasible:
                break
            add(feasible[0], f"{layer}_sufficiency")
            if len(selected) >= int(maximum_anchors):
                break
        if len(selected) >= int(maximum_anchors):
            break

    base = torch.eye(6, dtype=torch.float64).repeat(query_count, 1, 1) * 1e-9
    for candidate in selected:
        base += information[:, candidate]

    def pose_score(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.slogdet(matrix)[1]

    while len(selected) < int(maximum_anchors):
        current = pose_score(base)
        if bool((current >= float(pose_logdet_target)).all()):
            break
        best = None
        best_gain = float("-inf")
        for candidate in order:
            if candidate in selected_set:
                continue
            gain = float(
                torch.clamp(
                    pose_score(base + information[:, candidate]) - current,
                    min=0,
                ).sum()
            )
            if gain > best_gain:
                best, best_gain = candidate, gain
        if best is None or not best_gain > 0:
            break
        add(best, "pose_sufficiency")
        base += information[:, best]
    return {
        "selected_anchor_rows": torch.tensor(selected, dtype=torch.long),
        "trace": trace,
        "layer_counts": {name: states[name].counts.tolist() for name in LAYER_NAMES},
        "pose_logdet": pose_score(base).tolist(),
        "unmet": {
            name: int(np.maximum(target - states[name].counts, 0).sum())
            for name in LAYER_NAMES
        },
        "contract": {
            "hierarchical_not_weighted_sum": True,
            "selection_order": [*LAYER_NAMES, "pose"],
            "fixed_edge_graph_required": True,
        },
    }
