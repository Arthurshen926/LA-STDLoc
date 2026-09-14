"""Bounded descriptor M-step for exact query-level pose-set teachers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric


@dataclass(frozen=True)
class PoseSetConstraint:
    query_index: int
    query: torch.Tensor
    bad_anchor: torch.Tensor
    good_anchor: torch.Tensor
    weight: float
    pose_gain: float


@dataclass(frozen=True)
class CleanMarginConstraint:
    query: torch.Tensor
    clean_anchor: int
    competitor_anchor: int
    margin_floor: float


def _csr_values(record: dict, prefix: str, row: int) -> torch.Tensor:
    offsets = torch.as_tensor(record[f"{prefix}_offsets"]).long()
    indices = torch.as_tensor(record[f"{prefix}_indices"]).long()
    return indices[int(offsets[row]) : int(offsets[row + 1])]


def _selected_joint_actions(row: dict) -> list[dict]:
    matches = [
        trace
        for trace in row["joint_trace"]
        if len(trace["actions"]) == int(row["joint_action_count"])
        and abs(float(trace["risk"]) - float(row["joint_risk"])) < 1e-7
    ]
    if len(matches) != 1:
        raise ValueError("oracle report does not identify one joint action set")
    return list(matches[0]["actions"])


def _oracle_target_gains(oracle: dict) -> dict[int, float]:
    """Aggregate exact joint-pose gains by the legal target identity."""
    values: dict[int, list[float]] = defaultdict(list)
    for row in oracle["queries"]:
        gain = max(float(row["current_risk"]) - float(row["joint_risk"]), 0.0)
        for action in _selected_joint_actions(row):
            if action["kind"] == "swap":
                values[int(action["anchor"])].append(gain)
    return {
        anchor: float(np.median(np.asarray(gains, dtype=np.float64)))
        for anchor, gains in values.items()
    }


def _uniform_cap(values: list, maximum: int) -> list:
    if maximum <= 0 or len(values) <= maximum:
        return values
    indices = (
        torch.linspace(0, len(values) - 1, steps=int(maximum))
        .round()
        .long()
        .unique(sorted=True)
        .tolist()
    )
    return [values[index] for index in indices]


@torch.inference_mode()
def build_expanded_pose_set_constraints(
    *,
    state: dict,
    metric_state_path: str | Path,
    teacher: dict,
    query_cache: dict,
    oracle: dict,
    device: torch.device,
    clean_minimum_margin: float,
    clean_margin_slack: float,
    maximum_clean_constraints: int,
    minimum_target_views: int,
    maximum_constraints_per_target: int,
) -> tuple[list[PoseSetConstraint], list[CleanMarginConstraint], torch.Tensor, dict]:
    """Expand pose-critical identities to their complete mapping support.

    Exact PoseLib replay discovers which legal landmark identities can improve a
    pose. The replayed row itself is not a generalizable training target. This
    builder therefore uses it only to seed identities, then recovers every
    mapping-only observation for those identities from the complete-positive
    teacher. A row is trained only when its current top-1 is not already any
    legal positive.
    """
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    base_bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    metric = load_shared_metric(
        metric_state_path, anchor_ids=anchor_ids, device=device
    )
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    target_gains = _oracle_target_gains(oracle)
    if not target_gains:
        raise ValueError("pose-set oracle contains no swap target identities")
    target_lookup = torch.zeros(base_bank.shape[0], dtype=torch.bool)
    target_lookup[torch.as_tensor(sorted(target_gains), dtype=torch.long)] = True

    raw: list[dict] = []
    clean_constraints: list[CleanMarginConstraint] = []
    target_views: dict[int, set[int]] = defaultdict(set)
    target_observations: dict[int, int] = defaultdict(int)
    clean_rows_per_query = max(
        int(np.ceil(maximum_clean_constraints / max(len(names), 1))), 1
    )
    for query_index, record in enumerate(teacher["records"]):
        cached = cache[names[query_index]]
        record_rows = torch.as_tensor(record["query_rows"]).long()
        offsets = torch.as_tensor(record["positive_offsets"]).long()
        positive_indices = torch.as_tensor(record["positive_indices"]).long()
        lengths = offsets[1:] - offsets[:-1]
        positive_rows = torch.repeat_interleave(
            torch.arange(record_rows.numel()), lengths
        )
        selected_occurrence = target_lookup[positive_indices]
        if bool(selected_occurrence.any()):
            target_pairs = torch.stack(
                [
                    positive_rows[selected_occurrence],
                    positive_indices[selected_occurrence],
                ],
                dim=1,
            ).unique(dim=0)
        else:
            target_pairs = torch.empty((0, 2), dtype=torch.long)

        sample_count = min(clean_rows_per_query, int(record_rows.numel()))
        sampled_local = (
            torch.linspace(0, record_rows.numel() - 1, steps=sample_count)
            .round()
            .long()
            .unique(sorted=True)
            if sample_count
            else torch.empty((0,), dtype=torch.long)
        )
        relevant_local = torch.unique(
            torch.cat([target_pairs[:, 0], sampled_local]), sorted=True
        )
        if relevant_local.numel() == 0:
            continue
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[
                record_rows[relevant_local]
            ],
            dim=1,
        ).to(device)
        adapted, _ = metric(descriptors)
        top_scores, top_indices = torch.topk(adapted @ base_bank.T, k=2, dim=1)
        local_to_batch = {
            int(local): batch for batch, local in enumerate(relevant_local.tolist())
        }

        targets_by_row: dict[int, list[int]] = defaultdict(list)
        for local, anchor in target_pairs.tolist():
            targets_by_row[int(local)].append(int(anchor))
            target_views[int(anchor)].add(query_index)
            target_observations[int(anchor)] += 1
        for local, targets in targets_by_row.items():
            batch = local_to_batch[local]
            positives = _csr_values(record, "positive", local)
            winner = int(top_indices[batch, 0])
            if bool((positives == winner).any()):
                continue
            candidates = torch.as_tensor(sorted(set(targets)), device=device)
            target_scores = adapted[batch] @ base_bank[candidates].T
            target = int(candidates[int(target_scores.argmax())])
            raw.append(
                {
                    "query_index": query_index,
                    "query": adapted[batch].detach().cpu(),
                    "bad": winner,
                    "good": target,
                    "gain": float(target_gains[target]),
                }
            )

        for local in sampled_local.tolist():
            batch = local_to_batch[int(local)]
            winner = int(top_indices[batch, 0])
            positives = _csr_values(record, "positive", int(local))
            if not bool((positives == winner).any()):
                continue
            margin = float(top_scores[batch, 0] - top_scores[batch, 1])
            if margin < float(clean_minimum_margin):
                continue
            clean_constraints.append(
                CleanMarginConstraint(
                    query=adapted[batch].detach().cpu(),
                    clean_anchor=winner,
                    competitor_anchor=int(top_indices[batch, 1]),
                    margin_floor=margin - float(clean_margin_slack),
                )
            )

    eligible_targets = {
        anchor
        for anchor, views in target_views.items()
        if len(views) >= int(minimum_target_views)
    }
    grouped: dict[int, list[dict]] = defaultdict(list)
    for value in raw:
        if int(value["good"]) in eligible_targets:
            grouped[int(value["good"])].append(value)
    capped = []
    for target in sorted(grouped):
        ordered = sorted(
            grouped[target], key=lambda value: (value["query_index"], value["bad"])
        )
        capped.extend(_uniform_cap(ordered, int(maximum_constraints_per_target)))
    if not capped:
        raise ValueError("pose-critical identities have no cross-view repair constraints")

    gains = torch.as_tensor([value["gain"] for value in capped]).float()
    positive_gains = gains[gains > 0]
    gain_scale = float(positive_gains.median()) if positive_gains.numel() else 1.0
    gain_scale = max(gain_scale, 1e-6)
    target_counts = {target: len(values) for target, values in grouped.items()}
    count_scale = float(np.median(list(target_counts.values())))
    constraints = [
        PoseSetConstraint(
            query_index=int(value["query_index"]),
            query=torch.as_tensor(value["query"]).float(),
            bad_anchor=torch.as_tensor(value["bad"]).long(),
            good_anchor=torch.as_tensor(value["good"]).long(),
            weight=float(
                np.clip(value["gain"] / gain_scale, 0.25, 4.0)
                * np.clip(
                    count_scale / max(target_counts[int(value["good"])], 1),
                    0.25,
                    4.0,
                )
            ),
            pose_gain=float(value["gain"]),
        )
        for value in capped
    ]
    trainable = torch.as_tensor(
        sorted({int(value.good_anchor) for value in constraints}), dtype=torch.long
    )
    clean_constraints = _uniform_cap(
        clean_constraints, int(maximum_clean_constraints)
    )
    view_counts = [len(target_views[anchor]) for anchor in eligible_targets]
    observation_counts = [target_observations[anchor] for anchor in eligible_targets]
    return constraints, clean_constraints, trainable, {
        "constraint_mode": "oracle_seeded_complete_positive_expansion",
        "oracle_target_count": len(target_gains),
        "eligible_target_count": len(eligible_targets),
        "constraint_count_before_cap": sum(len(values) for values in grouped.values()),
        "constraint_count": len(constraints),
        "constraint_query_count": len({value.query_index for value in constraints}),
        "trainable_anchor_count": int(trainable.numel()),
        "clean_margin_constraint_count": len(clean_constraints),
        "minimum_target_views": int(minimum_target_views),
        "maximum_constraints_per_target": int(maximum_constraints_per_target),
        "target_view_count_median": float(np.median(view_counts)),
        "target_view_count_mean": float(np.mean(view_counts)),
        "target_observation_count_median": float(np.median(observation_counts)),
        "target_observation_count_mean": float(np.mean(observation_counts)),
        "pose_gain_median": float(gains.median()),
        "pose_gain_mean": float(gains.mean()),
    }


@torch.inference_mode()
def build_pose_set_constraints(
    *,
    state: dict,
    metric_state_path: str | Path,
    teacher: dict,
    query_cache: dict,
    oracle: dict,
    device: torch.device,
    clean_minimum_margin: float,
    clean_margin_slack: float,
    maximum_clean_constraints: int,
) -> tuple[list[PoseSetConstraint], list[CleanMarginConstraint], torch.Tensor, dict]:
    """Recover source winners and legal positives for exact oracle swaps."""
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    base_bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    metric = load_shared_metric(
        metric_state_path, anchor_ids=anchor_ids, device=device
    )
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    raw = []
    query_payloads = []
    skipped_reject = 0
    for oracle_row in oracle["queries"]:
        query_index = int(oracle_row["query_index"])
        record = teacher["records"][query_index]
        cached = cache[names[query_index]]
        all_rows = torch.as_tensor(record["query_rows"]).long()
        selected_local = torch.arange(all_rows.numel())
        row_limit = int(oracle.get("deployment_row_limit", 0))
        if row_limit > 0:
            selected_local = selected_local[all_rows < row_limit]
        deployment_rows = all_rows[selected_local]
        actions = _selected_joint_actions(oracle_row)
        if not actions:
            continue
        action_rows = torch.as_tensor(
            [int(action["row"]) for action in actions if action["kind"] == "swap"]
        ).long()
        skipped_reject += sum(action["kind"] == "reject" for action in actions)
        if action_rows.numel() == 0:
            continue
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[
                deployment_rows
            ],
            dim=1,
        ).to(device)
        adapted, _ = metric(descriptors)
        top_scores, top_indices = torch.topk(adapted @ base_bank.T, k=2, dim=1)
        winners = top_indices[action_rows, 0].cpu()
        good = torch.as_tensor(
            [
                int(action["anchor"])
                for action in actions
                if action["kind"] == "swap"
            ],
            dtype=torch.long,
        )
        for local, source_local, winner, target in zip(
            range(action_rows.numel()), action_rows.tolist(), winners.tolist(), good.tolist()
        ):
            positives = _csr_values(record, "positive", int(selected_local[source_local]))
            if not bool((positives == int(target)).any()):
                raise ValueError("pose-set swap target is not a legal positive")
            if int(winner) == int(target):
                continue
            raw.append(
                {
                    "query_index": query_index,
                    "query": adapted[action_rows[local]].detach().cpu(),
                    "bad": int(winner),
                    "good": int(target),
                    "gain": max(
                        float(oracle_row["current_risk"])
                        - float(oracle_row["joint_risk"]),
                        0.0,
                    ),
                }
            )
        query_payloads.append(
            (
                record,
                selected_local,
                adapted.detach().cpu(),
                top_scores.detach().cpu(),
                top_indices.detach().cpu(),
            )
        )
    if not raw:
        raise ValueError("pose-set oracle contains no trainable swap constraints")
    gains = torch.as_tensor([value["gain"] for value in raw]).float()
    positive_gains = gains[gains > 0]
    scale = float(positive_gains.median()) if positive_gains.numel() else 1.0
    scale = max(scale, 1e-6)
    constraints = [
        PoseSetConstraint(
            query_index=int(value["query_index"]),
            query=torch.as_tensor(value["query"]).float(),
            bad_anchor=torch.as_tensor(value["bad"]).long(),
            good_anchor=torch.as_tensor(value["good"]).long(),
            weight=float(np.clip(value["gain"] / scale, 0.25, 4.0)),
            pose_gain=float(value["gain"]),
        )
        for value in raw
    ]
    trainable = torch.unique(
        torch.cat(
            [
                torch.stack([constraint.bad_anchor for constraint in constraints]),
                torch.stack([constraint.good_anchor for constraint in constraints]),
            ]
        ),
        sorted=True,
    )
    clean_constraints = []
    for record, selected_local, adapted, top_scores, top_indices in query_payloads:
        for local, record_local in enumerate(selected_local.tolist()):
            winner = int(top_indices[local, 0])
            competitor = int(top_indices[local, 1])
            positives = _csr_values(record, "positive", int(record_local))
            if not bool((positives == winner).any()):
                continue
            margin = float(top_scores[local, 0] - top_scores[local, 1])
            if margin < float(clean_minimum_margin):
                continue
            clean_constraints.append(
                CleanMarginConstraint(
                    query=adapted[local],
                    clean_anchor=winner,
                    competitor_anchor=competitor,
                    margin_floor=margin - float(clean_margin_slack),
                )
            )
    if len(clean_constraints) > int(maximum_clean_constraints):
        indices = (
            torch.linspace(
                0,
                len(clean_constraints) - 1,
                steps=int(maximum_clean_constraints),
            )
            .round()
            .long()
            .unique(sorted=True)
            .tolist()
        )
        clean_constraints = [clean_constraints[index] for index in indices]
    return constraints, clean_constraints, trainable, {
        "constraint_count": len(constraints),
        "constraint_query_count": len({value.query_index for value in constraints}),
        "trainable_anchor_count": int(trainable.numel()),
        "clean_margin_constraint_count": len(clean_constraints),
        "skipped_reject_action_count": int(skipped_reject),
        "pose_gain_median": float(gains.median()),
        "pose_gain_mean": float(gains.mean()),
    }


def _bounded_local_bank(
    base: torch.Tensor, residual: torch.Tensor, maximum_norm: float
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.norm(residual, dim=1, keepdim=True)
    bounded = residual * torch.clamp(
        float(maximum_norm) / norm.clamp_min(1e-8), max=1.0
    )
    return F.normalize(base + bounded, dim=1), bounded


def train_pose_set_residual(
    *,
    state: dict,
    constraints: list[PoseSetConstraint],
    clean_constraints: list[CleanMarginConstraint],
    trainable_anchors: torch.Tensor,
    maximum_norm: float,
    steps: int,
    learning_rate: float,
    margin: float,
    temperature: float,
    trust_weight: float,
    clean_weight: float,
    holdout_modulus: int,
    holdout_remainder: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict]:
    """Cross-fit the bounded residual and select steps by held-out ranking."""
    base_bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    trainable_anchors = torch.as_tensor(trainable_anchors).long().to(device)
    local_base = base_bank[trainable_anchors]
    lookup = torch.full((base_bank.shape[0],), -1, dtype=torch.long, device=device)
    lookup[trainable_anchors] = torch.arange(trainable_anchors.numel(), device=device)
    query = torch.stack([value.query for value in constraints]).to(device)
    bad = torch.stack([value.bad_anchor for value in constraints]).long().to(device)
    good = torch.stack([value.good_anchor for value in constraints]).long().to(device)
    weights = torch.as_tensor(
        [value.weight for value in constraints], device=device
    ).float()
    query_indices = torch.as_tensor(
        [value.query_index for value in constraints], device=device
    ).long()
    holdout = query_indices.remainder(int(holdout_modulus)) == int(holdout_remainder)
    if not bool(holdout.any()) or not bool((~holdout).any()):
        holdout = torch.arange(len(constraints), device=device).remainder(5) == 0
    train = ~holdout
    if clean_constraints:
        clean_query = torch.stack([value.query for value in clean_constraints]).to(device)
        clean_anchor = torch.as_tensor(
            [value.clean_anchor for value in clean_constraints], device=device
        ).long()
        competitor_anchor = torch.as_tensor(
            [value.competitor_anchor for value in clean_constraints], device=device
        ).long()
        margin_floor = torch.as_tensor(
            [value.margin_floor for value in clean_constraints], device=device
        ).float()
    else:
        clean_query = query.new_empty((0, query.shape[1]))
        clean_anchor = bad.new_empty((0,))
        competitor_anchor = bad.new_empty((0,))
        margin_floor = query.new_empty((0,))

    with torch.no_grad():
        initial_bad_score = torch.einsum("bd,bd->b", query, base_bank[bad])
        initial_good_score = torch.einsum("bd,bd->b", query, base_bank[good])
        target_score = (initial_bad_score + float(margin)).clamp(-1.0, 1.0)
        good_angle = torch.acos(initial_good_score.clamp(-1.0, 1.0))
        target_angle = torch.acos(target_score)
        required_angle = (good_angle - target_angle).clamp_min(0.0)
        required_bound = torch.where(
            required_angle < (torch.pi / 2.0),
            torch.sin(required_angle),
            torch.ones_like(required_angle),
        )
        maximum_rotation = torch.asin(
            torch.as_tensor(min(max(float(maximum_norm), 0.0), 1.0), device=device)
        )
        independently_reachable = required_angle <= maximum_rotation + 1e-7

        def percentile(values: torch.Tensor, quantile: float) -> float:
            return float(torch.quantile(values.float(), float(quantile)))

        realizability = {
            "initial_pair_accuracy": float(
                (initial_good_score > initial_bad_score).float().mean()
            ),
            "initial_score_gap_bad_minus_good_median": percentile(
                initial_bad_score - initial_good_score, 0.5
            ),
            "initial_score_gap_bad_minus_good_p90": percentile(
                initial_bad_score - initial_good_score, 0.9
            ),
            "required_residual_bound_p10": percentile(required_bound, 0.1),
            "required_residual_bound_median": percentile(required_bound, 0.5),
            "required_residual_bound_p90": percentile(required_bound, 0.9),
            "independently_reachable_fraction_at_bound": float(
                independently_reachable.float().mean()
            ),
        }

    def effective_features(indices: torch.Tensor, local_bank: torch.Tensor) -> torch.Tensor:
        result = base_bank[indices].clone()
        local = lookup[indices]
        selected = local >= 0
        if bool(selected.any()):
            result[selected] = local_bank[local[selected]]
        return result

    def clean_barrier(local_bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if clean_query.shape[0] == 0:
            zero = local_bank.sum() * 0.0
            return zero, torch.zeros((), device=device)
        clean_feature = effective_features(clean_anchor, local_bank)
        competitor_feature = effective_features(competitor_anchor, local_bank)
        clean_score = torch.einsum("bd,bd->b", clean_query, clean_feature)
        competitor_score = torch.einsum(
            "bd,bd->b", clean_query, competitor_feature
        )
        local_scores = clean_query @ local_bank.T
        clean_local = lookup[clean_anchor]
        local_rows = torch.nonzero(clean_local >= 0, as_tuple=False).reshape(-1)
        if local_rows.numel():
            local_scores[local_rows, clean_local[local_rows]] = -torch.inf
        competitor_score = torch.maximum(competitor_score, local_scores.max(dim=1).values)
        violation = F.relu(margin_floor - (clean_score - competitor_score))
        return violation.mean(), (violation > 0).float().mean()

    def fit(
        mask: torch.Tensor, fit_steps: int, *, select_by_holdout: bool
    ) -> tuple[torch.Tensor, list[dict], int]:
        residual = torch.nn.Parameter(torch.zeros_like(local_base))
        optimizer = torch.optim.AdamW([residual], lr=float(learning_rate), weight_decay=1e-4)
        history = []
        best_step = 0
        best_score = float("inf")
        best_residual = residual.detach().clone()
        for step in range(1, int(fit_steps) + 1):
            local_bank, bounded = _bounded_local_bank(
                local_base, residual, maximum_norm
            )
            bad_feature = effective_features(bad, local_bank)
            good_feature = effective_features(good, local_bank)
            bad_score = torch.einsum("bd,bd->b", query, bad_feature)
            good_score = torch.einsum("bd,bd->b", query, good_feature)
            violation = F.softplus(
                (float(margin) + bad_score - good_score) / float(temperature)
            ) * float(temperature)
            loss = (violation[mask] * weights[mask]).sum() / weights[mask].sum()
            clean_loss, _ = clean_barrier(local_bank)
            trust = bounded.square().sum(dim=1).mean()
            objective = (
                loss
                + float(clean_weight) * clean_loss
                + float(trust_weight) * trust
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_([residual], 1.0)
            optimizer.step()
            with torch.no_grad():
                norm = torch.linalg.norm(residual, dim=1, keepdim=True)
                residual.mul_(
                    torch.clamp(
                        float(maximum_norm) / norm.clamp_min(1e-8), max=1.0
                    )
                )
            if step == 1 or step % 25 == 0 or step == int(fit_steps):
                with torch.no_grad():
                    local_eval, bounded_eval = _bounded_local_bank(
                        local_base, residual, maximum_norm
                    )
                    held_bad = torch.einsum(
                        "bd,bd->b", query, effective_features(bad, local_eval)
                    )
                    held_good = torch.einsum(
                        "bd,bd->b", query, effective_features(good, local_eval)
                    )
                    held_violation = F.relu(
                        float(margin) + held_bad - held_good
                    )
                    held_score = float(held_violation[holdout].mean())
                    held_accuracy = float(
                        (held_good[holdout] > held_bad[holdout]).float().mean()
                    )
                    clean_eval, clean_violation_rate = clean_barrier(local_eval)
                    selection_score = held_score + float(clean_weight) * float(clean_eval)
                    row = {
                        "step": step,
                        "train_loss": float(loss),
                        "holdout_margin_violation": held_score,
                        "holdout_pair_accuracy": held_accuracy,
                        "clean_margin_loss": float(clean_eval),
                        "clean_margin_violation_rate": float(clean_violation_rate),
                        "residual_mean_norm": float(
                            torch.linalg.norm(bounded_eval, dim=1).mean()
                        ),
                        "residual_max_norm": float(
                            torch.linalg.norm(bounded_eval, dim=1).max()
                        ),
                    }
                    history.append(row)
                    if selection_score < best_score:
                        best_score = selection_score
                        best_step = step
                        best_residual = residual.detach().clone()
        if select_by_holdout:
            return best_residual, history, best_step
        return residual.detach().clone(), history, int(fit_steps)

    _, calibration_history, selected_steps = fit(
        train, int(steps), select_by_holdout=True
    )
    selected_steps = max(int(selected_steps), 1)
    all_mask = torch.ones_like(train)
    final_residual, refit_history, _ = fit(
        all_mask, selected_steps, select_by_holdout=False
    )
    _, bounded = _bounded_local_bank(local_base, final_residual, maximum_norm)
    return bounded.detach().cpu(), {
        "selected_steps": selected_steps,
        "holdout_modulus": int(holdout_modulus),
        "holdout_remainder": int(holdout_remainder),
        "calibration_history": calibration_history,
        "refit_history": refit_history,
        "realizability": realizability,
    }


def materialize_pose_set_map(
    *,
    state: dict,
    trainable_anchors: torch.Tensor,
    residual: torch.Tensor,
    report: dict,
) -> dict:
    base = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    trainable_anchors = torch.as_tensor(trainable_anchors).long()
    residual = torch.as_tensor(residual).float()
    revised = base.clone()
    revised[trainable_anchors] = F.normalize(
        base[trainable_anchors] + residual, dim=1
    )
    output = dict(state)
    output["anchor_features"] = revised
    output["pose_set_trainable_anchors"] = trainable_anchors
    output["pose_set_anchor_residual"] = residual
    output["pose_set_refinement_report"] = json.loads(json.dumps(report))
    return output
