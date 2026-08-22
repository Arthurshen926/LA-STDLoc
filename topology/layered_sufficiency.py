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
    dense_information = isinstance(pose_information, torch.Tensor)
    information = (
        torch.as_tensor(pose_information).double()
        if dense_information
        else pose_information
    )
    if reliability.numel() != candidate_count:
        raise ValueError("reliability does not align with candidates")
    if dense_information:
        if information.ndim != 4 or information.shape[1] != candidate_count:
            raise ValueError("pose_information must have shape [Q,N,6,6]")
        query_count = int(information.shape[0])
        if information.shape[2:] != (6, 6):
            raise ValueError("pose information matrices must be 6x6")
    else:
        if not isinstance(information, Sequence) or len(information) != candidate_count:
            raise ValueError("sparse pose information must have one mapping per candidate")
        information_query_count = max(
            (int(query) for candidate in information for query in candidate),
            default=-1,
        ) + 1
        edge_query_count = max(
            (
                int(query)
                for edges in layer_edges.values()
                for candidate in edges
                for query in candidate
            ),
            default=-1,
        ) + 1
        query_count = max(information_query_count, edge_query_count)
        if query_count <= 0:
            raise ValueError("cannot infer query count from sparse pose information")
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
            # The policy chooses the first feasible row in reliability order.
            # Stop at that row instead of materializing the full feasible list;
            # this is selection-exact and avoids thousands of unused matching
            # replays on every addition.
            chosen = None
            for candidate in order:
                if candidate in selected_set:
                    continue
                if any(
                    state.counts[query] < target[query]
                    and state.would_augment(candidate, query)
                    for query in layer_edges[layer][candidate]
                ):
                    chosen = candidate
                    break
            if chosen is None:
                break
            add(chosen, f"{layer}_sufficiency")
            if len(selected) >= int(maximum_anchors):
                break
        if len(selected) >= int(maximum_anchors):
            break

    base = torch.eye(6, dtype=torch.float64).repeat(query_count, 1, 1) * 1e-9
    for candidate in selected:
        if dense_information:
            base += information[:, candidate]
        else:
            for query, matrix in information[candidate].items():
                value = torch.as_tensor(matrix).double()
                if value.shape != (6, 6):
                    raise ValueError("sparse pose information matrices must be 6x6")
                base[int(query)] += value

    def pose_score(matrix: torch.Tensor) -> torch.Tensor:
        return torch.linalg.slogdet(matrix)[1]

    while len(selected) < int(maximum_anchors):
        current = pose_score(base)
        deficient = current < float(pose_logdet_target)
        if not bool(deficient.any()):
            break
        best = None
        best_gain = float("-inf")
        if dense_information:
            # Bounded batches avoid materializing [N,Q,6,6] on full scenes.
            for start in range(0, len(order), 256):
                chunk = torch.tensor(order[start : start + 256], dtype=torch.long)
                gains = torch.clamp(
                    torch.linalg.slogdet(
                        base[None] + information.transpose(0, 1)[chunk]
                    )[1]
                    - current[None],
                    min=0,
                )[:, deficient].sum(1)
                for candidate, gain in zip(chunk.tolist(), gains.tolist()):
                    if candidate not in selected_set and float(gain) > best_gain:
                        best, best_gain = candidate, float(gain)
        else:
            for candidate in order:
                if candidate in selected_set or not information[candidate]:
                    continue
                gain = 0.0
                for query, matrix in information[candidate].items():
                    query = int(query)
                    if not bool(deficient[query]):
                        continue
                    value = torch.as_tensor(matrix).double()
                    gain += float(
                        torch.clamp(
                            pose_score(base[query] + value) - current[query], min=0
                        )
                    )
                if gain > best_gain:
                    best, best_gain = candidate, gain
        if best is None or not best_gain > 0:
            break
        add(best, "pose_sufficiency")
        if dense_information:
            base += information[:, best]
        else:
            for query, matrix in information[best].items():
                base[int(query)] += torch.as_tensor(matrix).double()
    return {
        "selected_anchor_rows": torch.tensor(selected, dtype=torch.long),
        "trace": trace,
        "layer_counts": {name: states[name].counts.tolist() for name in LAYER_NAMES},
        "pose_logdet": pose_score(base).tolist(),
        "unmet": {
            **{
                name: int(np.maximum(target - states[name].counts, 0).sum())
                for name in LAYER_NAMES
            },
            "pose": int(
                (pose_score(base) < float(pose_logdet_target)).sum().item()
            ),
        },
        "contract": {
            "hierarchical_not_weighted_sum": True,
            "selection_order": [*LAYER_NAMES, "pose"],
            "fixed_edge_graph_required": True,
        },
    }
