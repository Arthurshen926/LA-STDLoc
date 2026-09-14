"""Layered visibility, detectability, matching, and pose sufficiency for V6."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from topology.matching_coverage import CandidateEdges, IncrementalBipartiteCoverage


LAYER_NAMES = ("visibility", "detectability", "matching")
DEFAULT_VISIBILITY_GRID = (4, 4)


def visibility_image_cells(
    projected_xy: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    grid_shape: tuple[int, int] = DEFAULT_VISIBILITY_GRID,
) -> torch.Tensor:
    """Map visible projections to deterministic image-grid cell identities."""

    xy = torch.as_tensor(projected_xy).float()
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("projected visibility coordinates must have shape [N,2]")
    height, width = (int(value) for value in image_hw)
    grid_rows, grid_cols = (int(value) for value in grid_shape)
    if height <= 0 or width <= 0 or grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("image and visibility-grid dimensions must be positive")
    if xy.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=xy.device)
    if not bool(torch.isfinite(xy).all()):
        raise ValueError("visible projection coordinates must be finite")
    columns = torch.floor(xy[:, 0] / float(width) * grid_cols).long()
    rows = torch.floor(xy[:, 1] / float(height) * grid_rows).long()
    columns = columns.clamp(0, grid_cols - 1)
    rows = rows.clamp(0, grid_rows - 1)
    return rows * grid_cols + columns


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
    visibility_target: int | Sequence[int] | None = None,
    detectability_target: int | Sequence[int] | None = None,
    pose_min_eigenvalue_target: float | None = None,
    query_count: int | None = None,
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
        inferred_query_count = int(information.shape[0])
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
        inferred_query_count = max(information_query_count, edge_query_count)
        if inferred_query_count <= 0 and query_count is None:
            raise ValueError("cannot infer query count from sparse pose information")
    if query_count is None:
        query_count = inferred_query_count
    else:
        query_count = int(query_count)
        if query_count <= 0 or inferred_query_count > query_count:
            raise ValueError("explicit query count does not cover the evidence graph")
        if dense_information and inferred_query_count != query_count:
            raise ValueError("explicit query count differs from dense pose information")
    layer_targets = {
        "visibility": _targets(
            matching_target if visibility_target is None else visibility_target,
            query_count,
        ),
        "detectability": _targets(
            matching_target if detectability_target is None else detectability_target,
            query_count,
        ),
        "matching": _targets(matching_target, query_count),
    }
    if not np.isfinite(float(pose_logdet_target)):
        raise ValueError("pose logdet target must be finite")
    if pose_min_eigenvalue_target is not None and (
        not np.isfinite(float(pose_min_eigenvalue_target))
        or float(pose_min_eigenvalue_target) < 0.0
    ):
        raise ValueError("pose minimum-eigenvalue target must be finite and non-negative")
    states = {
        name: IncrementalBipartiteCoverage(query_count, layer_edges[name])
        for name in LAYER_NAMES
    }
    selected: list[int] = []
    selected_set: set[int] = set()
    trace = []
    order = torch.argsort(reliability, descending=True, stable=True).tolist()
    layer_candidate_examination_count = {name: 0 for name in LAYER_NAMES}

    def add(candidate: int, reason: str) -> None:
        selected.append(candidate)
        selected_set.add(candidate)
        for state in states.values():
            state.add(candidate)
        trace.append({"anchor": candidate, "reason": reason})

    for layer in LAYER_NAMES:
        state = states[layer]
        target = layer_targets[layer]
        cursor = 0
        while bool((state.counts < target).any()):
            # The policy chooses the first feasible row in reliability order.
            # Coverage and bipartite-matching rank are monotone submodular, so
            # an Anchor with zero marginal gain cannot regain positive marginal
            # gain after the selected set grows.  Advancing the cursor is thus
            # selection-exact and avoids rechecking the same rejected prefix on
            # every addition.
            chosen = None
            while cursor < len(order):
                candidate = order[cursor]
                cursor += 1
                if candidate in selected_set:
                    continue
                layer_candidate_examination_count[layer] += 1
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

    def pose_scores(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        symmetric = (matrix + matrix.transpose(-1, -2)) * 0.5
        return (
            torch.linalg.slogdet(symmetric)[1],
            torch.linalg.eigvalsh(symmetric)[..., 0],
        )

    def pose_deficiency(
        logdet: torch.Tensor, minimum_eigenvalue: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logdet_deficient = logdet < float(pose_logdet_target)
        minimum_deficient = (
            torch.zeros_like(logdet_deficient)
            if pose_min_eigenvalue_target is None
            else minimum_eigenvalue < float(pose_min_eigenvalue_target)
        )
        return logdet_deficient | minimum_deficient, logdet_deficient, minimum_deficient

    def deficit_reduction(
        current_logdet: torch.Tensor,
        current_minimum: torch.Tensor,
        proposed: torch.Tensor,
        logdet_deficient: torch.Tensor,
        minimum_deficient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proposed_logdet, proposed_minimum = pose_scores(proposed)
        capped_logdet_gain = (
            torch.minimum(
                proposed_logdet,
                torch.full_like(proposed_logdet, float(pose_logdet_target)),
            )
            - torch.minimum(
                current_logdet,
                torch.full_like(current_logdet, float(pose_logdet_target)),
            )
        ).clamp_min(0)
        logdet_gain = capped_logdet_gain[..., logdet_deficient].sum(-1)
        if pose_min_eigenvalue_target is None:
            minimum_gain = torch.zeros_like(logdet_gain)
        else:
            target_value = float(pose_min_eigenvalue_target)
            capped_minimum_gain = (
                torch.minimum(
                    proposed_minimum,
                    torch.full_like(proposed_minimum, target_value),
                )
                - torch.minimum(
                    current_minimum,
                    torch.full_like(current_minimum, target_value),
                )
            ).clamp_min(0)
            minimum_gain = capped_minimum_gain[..., minimum_deficient].sum(-1)
        return minimum_gain, logdet_gain

    def better_gain(
        minimum_gain: float,
        logdet_gain: float,
        best_minimum_gain: float,
        best_logdet_gain: float,
    ) -> bool:
        if minimum_gain > best_minimum_gain:
            return True
        return bool(
            np.isclose(minimum_gain, best_minimum_gain, atol=1e-12, rtol=0.0)
            and logdet_gain > best_logdet_gain
        )

    while len(selected) < int(maximum_anchors):
        current_logdet, current_minimum = pose_scores(base)
        deficient, logdet_deficient, minimum_deficient = pose_deficiency(
            current_logdet, current_minimum
        )
        if not bool(deficient.any()):
            break
        best = None
        best_minimum_gain = float("-inf")
        best_logdet_gain = float("-inf")
        if dense_information:
            # Bounded batches avoid materializing [N,Q,6,6] on full scenes.
            for start in range(0, len(order), 256):
                chunk = torch.tensor(order[start : start + 256], dtype=torch.long)
                minimum_gains, logdet_gains = deficit_reduction(
                    current_logdet,
                    current_minimum,
                    base[None] + information.transpose(0, 1)[chunk],
                    logdet_deficient,
                    minimum_deficient,
                )
                for candidate, minimum_gain, logdet_gain in zip(
                    chunk.tolist(), minimum_gains.tolist(), logdet_gains.tolist()
                ):
                    if candidate in selected_set:
                        continue
                    if better_gain(
                        float(minimum_gain),
                        float(logdet_gain),
                        best_minimum_gain,
                        best_logdet_gain,
                    ):
                        best = candidate
                        best_minimum_gain = float(minimum_gain)
                        best_logdet_gain = float(logdet_gain)
        else:
            for candidate in order:
                if candidate in selected_set or not information[candidate]:
                    continue
                minimum_gain = 0.0
                logdet_gain = 0.0
                for query, matrix in information[candidate].items():
                    query = int(query)
                    if not bool(deficient[query]):
                        continue
                    value = torch.as_tensor(matrix).double()
                    query_minimum_gain, query_logdet_gain = deficit_reduction(
                        current_logdet[query : query + 1],
                        current_minimum[query : query + 1],
                        (base[query] + value)[None],
                        logdet_deficient[query : query + 1],
                        minimum_deficient[query : query + 1],
                    )
                    minimum_gain += float(query_minimum_gain)
                    logdet_gain += float(query_logdet_gain)
                if better_gain(
                    minimum_gain,
                    logdet_gain,
                    best_minimum_gain,
                    best_logdet_gain,
                ):
                    best = candidate
                    best_minimum_gain = minimum_gain
                    best_logdet_gain = logdet_gain
        if best is None or not (
            best_minimum_gain > 0.0 or best_logdet_gain > 0.0
        ):
            break
        add(best, "pose_sufficiency")
        if dense_information:
            base += information[:, best]
        else:
            for query, matrix in information[best].items():
                base[int(query)] += torch.as_tensor(matrix).double()
    final_logdet, final_minimum = pose_scores(base)
    pose_unmet, logdet_unmet, minimum_unmet = pose_deficiency(
        final_logdet, final_minimum
    )
    return {
        "selected_anchor_rows": torch.tensor(selected, dtype=torch.long),
        "trace": trace,
        "layer_counts": {name: states[name].counts.tolist() for name in LAYER_NAMES},
        "layer_targets": {
            name: target.tolist() for name, target in layer_targets.items()
        },
        "pose_logdet": final_logdet.tolist(),
        "pose_min_eigenvalue": final_minimum.tolist(),
        "unmet": {
            **{
                name: int(
                    np.maximum(layer_targets[name] - states[name].counts, 0).sum()
                )
                for name in LAYER_NAMES
            },
            "pose": int(pose_unmet.sum().item()),
            "pose_logdet": int(logdet_unmet.sum().item()),
            "pose_min_eigenvalue": int(minimum_unmet.sum().item()),
        },
        "contract": {
            "hierarchical_not_weighted_sum": True,
            "selection_order": [*LAYER_NAMES, "pose"],
            "fixed_edge_graph_required": True,
            "zero_marginal_candidates_examined_once_per_layer": True,
            "layer_candidate_examination_count": layer_candidate_examination_count,
            "separate_layer_targets": True,
            "pose_evidence_unit": "unique_anchor_per_query",
            "pose_logdet_target": float(pose_logdet_target),
            "pose_min_eigenvalue_target": (
                None
                if pose_min_eigenvalue_target is None
                else float(pose_min_eigenvalue_target)
            ),
        },
    }
