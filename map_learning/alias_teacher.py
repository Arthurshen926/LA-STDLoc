"""Hard-deployment evidence for recurrent query-specific alias groups.

The teacher is deliberately non-differentiable: current top-1 deployment
assignments define the E-step, while descriptor reconstruction remains the
only optimized state in the M-step.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RecurrentAliasTeacher:
    row_anchors: dict[int, dict[int, int]]
    row_weights: dict[int, dict[int, float]]
    active_anchors: torch.Tensor
    diagnostics: dict[str, float | int]


def replace_query_assignments(
    destination: dict[int, dict[int, int]],
    query_indices: list[int],
    refreshed: dict[int, dict[int, int]],
) -> None:
    """Replace one refresh shard without retaining stale assignments."""
    for query in query_indices:
        destination.pop(int(query), None)
    destination.update(
        {int(query): dict(assignments) for query, assignments in refreshed.items()}
    )


def build_recurrent_alias_teacher(
    false_assignments: dict[int, dict[int, int]],
    query_groups: torch.Tensor,
    *,
    anchor_count: int,
    minimum_distinct_groups: int,
    minimum_queries: int,
    minimum_occurrences: int,
    minimum_rows_per_query: int,
    solver_harmful_assignments: dict[int, dict[int, int]] | None = None,
) -> RecurrentAliasTeacher:
    """Keep only false winners that recur across independent pose groups.

    A globally recurrent anchor is supervised only in queries where it also
    attracts multiple rows. This distinguishes a repeated-assignment failure
    from an isolated hard negative.
    """
    query_groups = torch.as_tensor(query_groups).long().reshape(-1)
    if int(anchor_count) <= 0:
        raise ValueError("anchor_count must be positive")
    thresholds = (
        minimum_distinct_groups,
        minimum_queries,
        minimum_occurrences,
        minimum_rows_per_query,
    )
    if any(int(value) < 1 for value in thresholds):
        raise ValueError("alias recurrence thresholds must be positive")

    observed_assignment_count = sum(
        len(assignments) for assignments in false_assignments.values()
    )
    observed_anchors = {
        int(anchor)
        for assignments in false_assignments.values()
        for anchor in assignments.values()
    }
    if solver_harmful_assignments is None:
        eligible_assignments = false_assignments
    else:
        eligible_assignments = {}
        for query, assignments in false_assignments.items():
            harmful = solver_harmful_assignments.get(int(query), {})
            accepted = {
                int(row): int(anchor)
                for row, anchor in assignments.items()
                if int(harmful.get(int(row), -1)) == int(anchor)
            }
            if accepted:
                eligible_assignments[int(query)] = accepted

    occurrences = Counter()
    queries: dict[int, set[int]] = defaultdict(set)
    groups: dict[int, set[int]] = defaultdict(set)
    per_query: dict[int, Counter] = {}
    for query, assignments in eligible_assignments.items():
        query = int(query)
        if query < 0 or query >= query_groups.numel():
            raise ValueError("false-assignment query index is out of range")
        counts = Counter(int(anchor) for anchor in assignments.values())
        per_query[query] = counts
        for anchor, count in counts.items():
            if anchor < 0 or anchor >= int(anchor_count):
                raise ValueError("false-assignment anchor index is out of range")
            occurrences[anchor] += int(count)
            queries[anchor].add(query)
            groups[anchor].add(int(query_groups[query]))

    active = {
        anchor
        for anchor, count in occurrences.items()
        if count >= int(minimum_occurrences)
        and len(queries[anchor]) >= int(minimum_queries)
        and len(groups[anchor]) >= int(minimum_distinct_groups)
    }
    row_anchors: dict[int, dict[int, int]] = defaultdict(dict)
    row_weights: dict[int, dict[int, float]] = defaultdict(dict)
    active_rows = 0
    active_query_groups = 0
    supervised_anchors: set[int] = set()
    for query, assignments in eligible_assignments.items():
        local_counts = per_query[int(query)]
        accepted = {
            anchor
            for anchor, count in local_counts.items()
            if anchor in active and count >= int(minimum_rows_per_query)
        }
        if accepted:
            active_query_groups += len(accepted)
            supervised_anchors.update(accepted)
        for row, anchor in assignments.items():
            anchor = int(anchor)
            if anchor not in accepted:
                continue
            weight = float(
                torch.log1p(torch.tensor(float(occurrences[anchor]))).item()
                * torch.log1p(torch.tensor(float(local_counts[anchor]))).item()
            )
            row_anchors[int(query)][int(row)] = anchor
            row_weights[int(query)][int(row)] = weight
            active_rows += 1

    active_tensor = torch.as_tensor(sorted(supervised_anchors), dtype=torch.long)
    return RecurrentAliasTeacher(
        row_anchors=dict(row_anchors),
        row_weights=dict(row_weights),
        active_anchors=active_tensor,
        diagnostics={
            "observed_false_assignment_count": int(observed_assignment_count),
            "observed_false_anchor_count": int(len(observed_anchors)),
            "solver_conditioned_alias": bool(
                solver_harmful_assignments is not None
            ),
            "solver_conditioned_false_assignment_count": int(
                sum(occurrences.values())
            ),
            "solver_conditioned_false_anchor_count": int(len(occurrences)),
            "recurrent_alias_anchor_count": int(len(active)),
            "active_alias_anchor_count": int(len(supervised_anchors)),
            "active_alias_query_group_count": int(active_query_groups),
            "active_alias_row_count": int(active_rows),
            "active_alias_anchor_fraction": float(
                len(supervised_anchors) / int(anchor_count)
            ),
        },
    )


def alias_group_ranking_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    positives: torch.Tensor,
    alias_anchors: torch.Tensor,
    alias_weights: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Optimize the worst ranking violation in each query-specific alias group."""
    alias_anchors = torch.as_tensor(alias_anchors, device=query.device).long()
    alias_weights = torch.as_tensor(
        alias_weights, device=query.device, dtype=query.dtype
    )
    positive_valid = positives >= 0
    valid = (alias_anchors >= 0) & positive_valid.any(dim=1)
    zero = query.sum() * 0.0
    if not bool(valid.any().item()):
        return zero, {"alias_rows": 0, "alias_groups": 0}

    selected_query = query[valid]
    selected_positive = positives[valid]
    selected_alias = alias_anchors[valid]
    selected_weight = alias_weights[valid].clamp_min(1e-8)
    positive_scores = torch.einsum(
        "bd,bpd->bp", selected_query, bank[selected_positive.clamp_min(0)]
    ).masked_fill(selected_positive < 0, -torch.inf)
    positive_best = positive_scores.max(dim=1).values
    alias_score = torch.einsum("bd,bd->b", selected_query, bank[selected_alias])
    violation = F.softplus(
        (float(margin) + alias_score - positive_best) / float(temperature)
    ) * float(temperature)

    group_losses = []
    group_weights = []
    for anchor in torch.unique(selected_alias, sorted=True):
        members = selected_alias == anchor
        # Smooth maximum: every collision receives gradient, while the worst
        # member controls the group-level repair signal.
        values = violation[members]
        group_losses.append(
            torch.logsumexp(values / float(temperature), dim=0)
            * float(temperature)
            - float(temperature) * torch.log(
                values.new_tensor(float(values.numel()))
            )
        )
        group_weights.append(selected_weight[members].mean())
    losses = torch.stack(group_losses)
    weights = torch.stack(group_weights)
    loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss, {
        "alias_rows": int(valid.sum().item()),
        "alias_groups": int(len(group_losses)),
        "alias_violation_mean": float(violation.detach().mean().item()),
    }


def protected_clean_margin_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    clean_anchors: torch.Tensor,
    margin_floors: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Prevent other repairs from crossing a previously clean top-1 margin."""
    clean_anchors = torch.as_tensor(clean_anchors, device=query.device).long()
    margin_floors = torch.as_tensor(
        margin_floors, device=query.device, dtype=query.dtype
    )
    valid = (clean_anchors >= 0) & torch.isfinite(margin_floors)
    zero = query.sum() * 0.0
    if not bool(valid.any().item()):
        return zero, {"protected_clean_rows": 0, "protected_clean_violations": 0}
    selected_query = query[valid]
    selected_anchor = clean_anchors[valid]
    scores = selected_query @ bank.T
    clean_score = scores.gather(1, selected_anchor[:, None]).squeeze(1)
    scores = scores.scatter(1, selected_anchor[:, None], -torch.inf)
    current_margin = clean_score - scores.max(dim=1).values
    violations = F.relu(margin_floors[valid] - current_margin)
    return violations.mean(), {
        "protected_clean_rows": int(valid.sum().item()),
        "protected_clean_violations": int((violations > 0).sum().item()),
        "protected_clean_margin_mean": float(current_margin.detach().mean().item()),
    }
