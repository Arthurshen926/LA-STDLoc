"""Candidate-conditioned relational rescue for one-pass sparse localization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DirectedConfusionEdge:
    correct_anchor: int
    confusing_anchor: int
    weight: float = 1.0


def directed_edges_by_confusing_anchor(
    edges: Iterable[dict],
) -> dict[int, tuple[DirectedConfusionEdge, ...]]:
    """Index retained directed edges without changing their label direction."""

    indexed: dict[int, list[DirectedConfusionEdge]] = defaultdict(list)
    for edge in edges:
        correct = int(edge["correct_anchor"])
        confusing = int(edge["confusing_anchor"])
        if correct == confusing:
            continue
        indexed[confusing].append(
            DirectedConfusionEdge(
                correct_anchor=correct,
                confusing_anchor=confusing,
                weight=float(edge.get("weight", 1.0)),
            )
        )
    return {
        anchor: tuple(
            sorted(
                values,
                key=lambda item: (-item.weight, item.correct_anchor),
            )
        )
        for anchor, values in indexed.items()
    }


def events_by_query_row(
    events: Iterable[dict],
) -> dict[tuple[str, int], tuple[DirectedConfusionEdge, ...]]:
    """Index query-specific events for causal analysis, not deployment."""

    indexed: dict[tuple[str, int], list[DirectedConfusionEdge]] = defaultdict(
        list
    )
    for event in events:
        correct = int(event["correct_anchor"])
        confusing = int(event["confusing_anchor"])
        if correct == confusing:
            continue
        indexed[(str(event["query_name"]), int(event["query_row"]))].append(
            DirectedConfusionEdge(
                correct_anchor=correct,
                confusing_anchor=confusing,
                weight=float(event.get("pose_blame", 1.0)),
            )
        )
    return {
        key: tuple(
            sorted(
                values,
                key=lambda item: (-item.weight, item.correct_anchor),
            )
        )
        for key, values in indexed.items()
    }


def directed_edge_keys(
    edges: Iterable[dict],
    *,
    anchor_count: int,
) -> torch.Tensor:
    """Encode directed (confusing, correct) pairs as sorted integer keys."""

    keys = {
        int(edge["confusing_anchor"]) * int(anchor_count)
        + int(edge["correct_anchor"])
        for edge in edges
        if int(edge["correct_anchor"]) != int(edge["confusing_anchor"])
    }
    return torch.as_tensor(sorted(keys), dtype=torch.long)


def candidate_conditioned_rescue_from_edge_keys(
    candidate_indices: torch.Tensor,
    local_scores: torch.Tensor,
    context_scores: torch.Tensor,
    edge_keys: torch.Tensor,
    *,
    anchor_count: int,
    context_weight: float,
    maximum_score_delta: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Vectorized deployment form of directed candidate rescue."""

    candidates = torch.as_tensor(candidate_indices).long()
    local = torch.as_tensor(
        local_scores, device=candidates.device
    ).float()
    context = torch.as_tensor(
        context_scores, device=candidates.device
    ).float()
    edges = torch.as_tensor(
        edge_keys, device=candidates.device
    ).long().reshape(-1)
    if candidates.ndim != 2 or local.shape != candidates.shape:
        raise ValueError("candidate indices and local scores must be NxK")
    if context.shape != candidates.shape:
        raise ValueError("context scores must align with local candidates")
    if edges.numel() and not bool((edges[1:] >= edges[:-1]).all()):
        raise ValueError("directed edge keys must be sorted")

    row_keys = (
        candidates[:, :1] * int(anchor_count) + candidates
    )
    if edges.numel():
        positions = torch.searchsorted(edges, row_keys)
        clamped = positions.clamp_max(edges.numel() - 1)
        eligible = (positions < edges.numel()) & (
            edges[clamped] == row_keys
        )
    else:
        eligible = torch.zeros_like(candidates, dtype=torch.bool)
    eligible[:, 0] = False
    correction = (
        max(float(context_weight), 0.0)
        * (context - context[:, :1])
    ).clamp(
        min=-max(float(maximum_score_delta), 0.0),
        max=max(float(maximum_score_delta), 0.0),
    )
    proposal_scores = (local + correction).masked_fill(
        ~eligible, -torch.inf
    )
    best_score, best_slot = proposal_scores.max(dim=1)
    changed = best_score > local[:, 0]
    best_slot = torch.where(
        changed, best_slot, torch.zeros_like(best_slot)
    )

    width = candidates.shape[1]
    rest = torch.arange(
        width - 1, device=candidates.device
    ).reshape(1, -1).expand(len(candidates), -1)
    rest = rest + (rest >= best_slot[:, None]).long()
    order = torch.cat((best_slot[:, None], rest), dim=1)
    output_indices = torch.gather(candidates, 1, order)
    output_scores = torch.gather(local, 1, order)
    output_scores[:, 0] = torch.where(
        changed, best_score, local[:, 0]
    )
    if not torch.equal(
        output_indices[~changed], candidates[~changed]
    ):
        raise AssertionError("vectorized rescue changed a non-rescued identity")
    if not torch.equal(output_scores[~changed], local[~changed]):
        raise AssertionError("vectorized rescue changed a non-rescued score")
    return output_indices, output_scores, {
        "row_count": int(len(candidates)),
        "eligible_row_count": int(eligible.any(dim=1).sum()),
        "eligible_pair_count": int(eligible.sum()),
        "changed_row_count": int(changed.sum()),
        "changed_mask": changed,
    }


def candidate_conditioned_rescue(
    candidate_indices: torch.Tensor,
    local_scores: torch.Tensor,
    context_scores: torch.Tensor,
    eligible_edges: list[tuple[DirectedConfusionEdge, ...]],
    *,
    context_weight: float,
    maximum_score_delta: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Rescue a local top-K winner through directed pair comparisons.

    The local winner and all candidates outside a directed confusion pair are
    immutable. Context may only promote a graph-labelled correct challenger
    over a graph-labelled confusing local winner.
    """

    candidates = torch.as_tensor(candidate_indices).long()
    local = torch.as_tensor(local_scores).float()
    context = torch.as_tensor(context_scores).float()
    if candidates.ndim != 2 or local.shape != candidates.shape:
        raise ValueError("candidate indices and local scores must be NxK")
    if context.shape != candidates.shape:
        raise ValueError("context scores must align with local candidates")
    if len(eligible_edges) != len(candidates):
        raise ValueError("eligible edge rows must align with candidates")

    output_indices = candidates.clone()
    output_scores = local.clone()
    changed = torch.zeros(len(candidates), dtype=torch.bool)
    eligible_rows = 0
    eligible_pairs = 0
    weight = max(float(context_weight), 0.0)
    maximum = max(float(maximum_score_delta), 0.0)

    for row_index, row_edges in enumerate(eligible_edges):
        baseline_anchor = int(candidates[row_index, 0])
        positions = {
            int(anchor): slot
            for slot, anchor in enumerate(candidates[row_index].tolist())
        }
        proposals: list[tuple[float, int]] = []
        for edge in row_edges:
            if edge.confusing_anchor != baseline_anchor:
                continue
            slot = positions.get(edge.correct_anchor)
            if slot is None or slot == 0:
                continue
            eligible_pairs += 1
            context_delta = weight * float(
                context[row_index, slot] - context[row_index, 0]
            )
            context_delta = min(max(context_delta, -maximum), maximum)
            proposals.append(
                (float(local[row_index, slot]) + context_delta, slot)
            )
        if not proposals:
            continue
        eligible_rows += 1
        proposal_score, proposal_slot = max(
            proposals, key=lambda value: (value[0], -value[1])
        )
        if proposal_score <= float(local[row_index, 0]):
            continue
        changed[row_index] = True
        reordered = torch.cat(
            (
                candidates[row_index, proposal_slot : proposal_slot + 1],
                candidates[row_index, :proposal_slot],
                candidates[row_index, proposal_slot + 1 :],
            )
        )
        reordered_scores = torch.cat(
            (
                local.new_tensor([proposal_score]),
                local[row_index, :proposal_slot],
                local[row_index, proposal_slot + 1 :],
            )
        )
        output_indices[row_index] = reordered
        output_scores[row_index] = reordered_scores

    unchanged = ~changed
    if not torch.equal(output_indices[unchanged], candidates[unchanged]):
        raise AssertionError("candidate rescue changed a non-rescued identity")
    if not torch.equal(output_scores[unchanged], local[unchanged]):
        raise AssertionError("candidate rescue changed a non-rescued score")
    return output_indices, output_scores, {
        "row_count": int(len(candidates)),
        "eligible_row_count": int(eligible_rows),
        "eligible_pair_count": int(eligible_pairs),
        "changed_row_count": int(changed.sum()),
        "changed_mask": changed,
    }


def oracle_acceptance_mask(
    baseline_errors_px: torch.Tensor,
    rescued_errors_px: torch.Tensor,
    changed_mask: torch.Tensor,
    *,
    strict_threshold_px: float = 2.0,
) -> torch.Tensor:
    """Select context-on only when it yields a strictly cleaner measurement."""

    baseline = torch.as_tensor(baseline_errors_px).float().reshape(-1)
    rescued = torch.as_tensor(rescued_errors_px).float().reshape(-1)
    changed = torch.as_tensor(changed_mask).bool().reshape(-1)
    if not (len(baseline) == len(rescued) == len(changed)):
        raise ValueError("oracle error rows must align")
    return (
        changed
        & torch.isfinite(baseline)
        & torch.isfinite(rescued)
        & (baseline > float(strict_threshold_px))
        & (rescued <= float(strict_threshold_px))
        & (rescued < baseline)
    )


class CandidateConditionedContextScorer(nn.Module):
    """Score context-on for one local-winner/confusion-edge competition."""

    def __init__(
        self,
        context_dim: int,
        scalar_dim: int,
        hidden_dim: int = 96,
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.scalar_dim = int(scalar_dim)
        self.hidden_dim = int(hidden_dim)
        input_dim = 4 * self.context_dim + self.scalar_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def forward(
        self,
        query_context: torch.Tensor,
        correct_context: torch.Tensor,
        confusing_context: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> torch.Tensor:
        query = F.normalize(torch.as_tensor(query_context).float(), dim=1)
        correct = F.normalize(
            torch.as_tensor(correct_context).float(), dim=1
        )
        confusing = F.normalize(
            torch.as_tensor(confusing_context).float(), dim=1
        )
        scalar = torch.as_tensor(
            scalar_features, device=query.device, dtype=query.dtype
        )
        if not (
            query.shape == correct.shape == confusing.shape
            and query.shape[1] == self.context_dim
            and scalar.shape == (len(query), self.scalar_dim)
        ):
            raise ValueError("candidate scorer inputs have incompatible shapes")
        features = torch.cat(
            (
                query,
                query * correct,
                query * confusing,
                correct - confusing,
                scalar,
            ),
            dim=1,
        )
        return self.network(features).reshape(-1)

    def export_config(self) -> dict:
        return {
            "context_dim": self.context_dim,
            "scalar_dim": self.scalar_dim,
            "hidden_dim": self.hidden_dim,
        }
