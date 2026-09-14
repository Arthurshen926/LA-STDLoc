"""Mapping-only, cross-group alias-risk evidence for a unified candidate pool."""

from __future__ import annotations

from collections.abc import Mapping

import torch


COUNTER_NAMES = ("winner", "clean", "false", "solver_inlier", "harmful")


def aggregate_group_alias_evidence(
    function_graph: Mapping, query_group_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Aggregate top-1 outcomes by independent mapping trajectory group."""
    anchor_count = int(function_graph["anchor_count"])
    groups = torch.as_tensor(query_group_ids).long().reshape(-1)
    records = function_graph["records"]
    if groups.numel() != len(records):
        raise ValueError("query groups do not align with function-graph records")
    if groups.numel() == 0 or int(groups.min()) < 0:
        raise ValueError("query groups must be non-empty and non-negative")
    _, compact_groups = torch.unique(groups, sorted=True, return_inverse=True)
    group_count = int(compact_groups.max()) + 1
    counters = {
        name: torch.zeros((group_count, anchor_count), dtype=torch.long)
        for name in COUNTER_NAMES
    }
    for query, record in enumerate(records):
        top = torch.as_tensor(record["top_indices"]).long()
        flags = torch.as_tensor(record["legal_flags"]).to(torch.uint8)
        if top.ndim != 2 or flags.shape != top.shape or top.shape[1] < 1:
            raise ValueError("function-graph top rows are malformed")
        winner = top[:, 0]
        if winner.numel() and (
            int(winner.min()) < 0 or int(winner.max()) >= anchor_count
        ):
            raise ValueError("function-graph winner is outside the candidate pool")
        clean = (flags[:, 0] & 4) != 0
        ambiguous = (flags[:, 0] & 8) != 0
        solver = torch.as_tensor(record["solver_inlier"]).bool().reshape(-1)
        harmful = torch.as_tensor(record["harmful_solver_inlier"]).bool().reshape(-1)
        if not (clean.numel() == solver.numel() == harmful.numel() == winner.numel()):
            raise ValueError("function-graph outcome masks do not align")
        group = int(compact_groups[query])
        masks = {
            "winner": torch.ones_like(clean),
            "clean": clean,
            "false": ~ambiguous,
            "solver_inlier": solver,
            "harmful": harmful,
        }
        for name, mask in masks.items():
            if bool(mask.any()):
                counters[name][group] += torch.bincount(
                    winner[mask], minlength=anchor_count
                )
    counters["query_group_count"] = torch.tensor(group_count)
    return counters


def wilson_upper(
    events: torch.Tensor, opportunities: torch.Tensor, *, z: float = 1.959963984540054
) -> torch.Tensor:
    """Return a conservative binomial upper bound; no evidence remains NaN."""
    events = torch.as_tensor(events).double()
    opportunities = torch.as_tensor(opportunities).double()
    if events.shape != opportunities.shape:
        raise ValueError("events and opportunities must align")
    if bool((events < 0).any()) or bool((opportunities < events).any()):
        raise ValueError("binomial counters are invalid")
    valid = opportunities > 0
    n = opportunities.clamp_min(1)
    probability = events / n
    z2 = float(z) ** 2
    center = probability + z2 / (2.0 * n)
    radius = float(z) * torch.sqrt(
        probability * (1.0 - probability) / n + z2 / (4.0 * n.square())
    )
    upper = (center + radius) / (1.0 + z2 / n)
    return torch.where(valid, upper.clamp(0, 1), torch.full_like(upper, float("nan")))


def wilson_lower(
    events: torch.Tensor, opportunities: torch.Tensor, *, z: float = 1.959963984540054
) -> torch.Tensor:
    """Return the evidence-supported risk floor; no evidence remains NaN."""
    events = torch.as_tensor(events).double()
    opportunities = torch.as_tensor(opportunities).double()
    if events.shape != opportunities.shape:
        raise ValueError("events and opportunities must align")
    if bool((events < 0).any()) or bool((opportunities < events).any()):
        raise ValueError("binomial counters are invalid")
    valid = opportunities > 0
    n = opportunities.clamp_min(1)
    probability = events / n
    z2 = float(z) ** 2
    center = probability + z2 / (2.0 * n)
    radius = float(z) * torch.sqrt(
        probability * (1.0 - probability) / n + z2 / (4.0 * n.square())
    )
    lower = (center - radius) / (1.0 + z2 / n)
    return torch.where(valid, lower.clamp(0, 1), torch.full_like(lower, float("nan")))


def alias_risk_from_counters(counters: Mapping[str, torch.Tensor]) -> dict:
    winner = torch.as_tensor(counters["winner"]).sum(dim=0)
    false = torch.as_tensor(counters["false"]).sum(dim=0)
    solver = torch.as_tensor(counters["solver_inlier"]).sum(dim=0)
    harmful = torch.as_tensor(counters["harmful"]).sum(dim=0)
    false_upper = wilson_upper(false, winner)
    harmful_upper = wilson_upper(harmful, solver)
    false_lower = wilson_lower(false, winner)
    harmful_lower = wilson_lower(harmful, solver)
    stacked = torch.stack(
        (
            torch.nan_to_num(false_lower, nan=-1.0),
            torch.nan_to_num(harmful_lower, nan=-1.0),
        )
    )
    risk = stacked.max(dim=0).values
    risk[risk < 0] = float("nan")
    upper_combined = torch.stack(
        (
            torch.nan_to_num(false_upper, nan=-1.0),
            torch.nan_to_num(harmful_upper, nan=-1.0),
        )
    ).max(dim=0).values
    lower_combined = torch.stack(
        (
            torch.nan_to_num(false_lower, nan=-1.0),
            torch.nan_to_num(harmful_lower, nan=-1.0),
        )
    ).max(dim=0).values
    uncertainty = upper_combined - lower_combined
    uncertainty[upper_combined < 0] = float("nan")
    false_groups = (torch.as_tensor(counters["false"]) > 0).sum(dim=0)
    harmful_groups = (torch.as_tensor(counters["harmful"]) > 0).sum(dim=0)
    recurrent = (false_groups >= 2) | (harmful_groups >= 2)
    return {
        "alias_risk": risk.float(),
        "false_winner_upper": false_upper.float(),
        "harmful_inlier_upper": harmful_upper.float(),
        "false_winner_lower": false_lower.float(),
        "harmful_inlier_lower": harmful_lower.float(),
        "risk_uncertainty_width": uncertainty.float(),
        "false_group_count": false_groups.long(),
        "harmful_group_count": harmful_groups.long(),
        "recurrent_alias": recurrent,
    }


def _weighted_auc(
    scores: torch.Tensor, positive_weight: torch.Tensor, negative_weight: torch.Tensor
) -> float:
    scores = torch.as_tensor(scores).double()
    positive = torch.as_tensor(positive_weight).double()
    negative = torch.as_tensor(negative_weight).double()
    valid = torch.isfinite(scores) & ((positive > 0) | (negative > 0))
    scores, positive, negative = scores[valid], positive[valid], negative[valid]
    positive_total = float(positive.sum())
    negative_total = float(negative.sum())
    if positive_total <= 0 or negative_total <= 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    scores, positive, negative = scores[order], positive[order], negative[order]
    numerator = 0.0
    cumulative_negative = 0.0
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and bool(scores[end] == scores[start]):
            end += 1
        group_positive = float(positive[start:end].sum())
        group_negative = float(negative[start:end].sum())
        numerator += group_positive * (
            cumulative_negative + 0.5 * group_negative
        )
        cumulative_negative += group_negative
        start = end
    return numerator / (positive_total * negative_total)


def crossfit_alias_separability(counters: Mapping[str, torch.Tensor]) -> dict:
    """Predict each held-out query group using evidence from other groups."""
    group_count = int(torch.as_tensor(counters["winner"]).shape[0])
    false_scores = []
    harmful_scores = []
    false_events = []
    harmful_events = []
    clean_events = []
    per_group = []
    for held_out in range(group_count):
        train = {
            name: torch.cat(
                (
                    torch.as_tensor(counters[name])[:held_out],
                    torch.as_tensor(counters[name])[held_out + 1 :],
                )
            )
            for name in COUNTER_NAMES
        }
        risk = alias_risk_from_counters(train)
        clean = torch.as_tensor(counters["clean"])[held_out]
        false = torch.as_tensor(counters["false"])[held_out]
        harmful = torch.as_tensor(counters["harmful"])[held_out]
        false_score = risk["false_winner_lower"]
        harmful_score = risk["harmful_inlier_lower"]
        false_auc = _weighted_auc(false_score, false, clean)
        harmful_auc = _weighted_auc(harmful_score, harmful, clean)
        per_group.append(
            {
                "held_out_group": held_out,
                "false_vs_clean_auc": false_auc,
                "harmful_vs_clean_auc": harmful_auc,
                "training_supported_candidate_count": int(
                    torch.isfinite(risk["alias_risk"]).sum()
                ),
            }
        )
        false_scores.append(false_score)
        harmful_scores.append(harmful_score)
        false_events.append(false)
        harmful_events.append(harmful)
        clean_events.append(clean)
    return {
        "group_count": group_count,
        "false_vs_clean_auc": _weighted_auc(
            torch.cat(false_scores),
            torch.cat(false_events),
            torch.cat(clean_events),
        ),
        "harmful_vs_clean_auc": _weighted_auc(
            torch.cat(harmful_scores),
            torch.cat(harmful_events),
            torch.cat(clean_events),
        ),
        "per_group": per_group,
    }
