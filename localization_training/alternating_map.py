"""Utilities for inference-aligned alternating localization-map updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PoseRiskConfig:
    translation_scale_m: float = 0.10
    cvar_fraction: float = 0.20
    cvar_weight: float = 0.35
    worst_group_weight: float = 0.0
    complexity_weight: float = 0.01
    reference_anchor_count: int = 50_000
    hypotheses_weight: float = 0.0
    reference_hypotheses: float = 25_000.0
    runtime_weight: float = 0.0
    reference_runtime_seconds: float = 0.4
    soft_regression_weight: float = 0.0
    regression_scale_m: float = 0.05
    protected_regression_multiplier: float = 4.0
    churn_weight: float = 0.0
    reference_churn_count: int = 5_000
    hard_gate_mode: bool = True
    max_median_regression_m: float = 0.0005
    max_r5_regression: float = 0.002
    protected_threshold_m: float = 0.05
    protected_failure_threshold_m: float = 0.10
    max_protected_regressions: int = 0
    minimum_objective_gain: float = 1e-5


def _cvar(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return float("inf")
    count = max(1, int(np.ceil(values.size * float(fraction))))
    return float(np.partition(values, values.size - count)[-count:].mean())


def summarize_pose_risk(
    translation_errors_m,
    *,
    anchor_count: int,
    config: PoseRiskConfig,
    group_ids=None,
    hypotheses=None,
    runtime_seconds=None,
) -> dict:
    """Summarize robust pose risk without hiding catastrophic tail queries."""
    errors = np.asarray(translation_errors_m, dtype=np.float64).reshape(-1)
    if errors.size == 0 or not np.isfinite(errors).all():
        raise ValueError("translation errors must be a non-empty finite vector")
    scale = max(float(config.translation_scale_m), 1e-8)
    normalized = np.log1p(errors / scale)
    data_risk = float(normalized.mean())
    tail_risk = _cvar(normalized, config.cvar_fraction)
    complexity = max(int(anchor_count), 0) / max(
        int(config.reference_anchor_count), 1
    )
    objective = (
        data_risk
        + float(config.cvar_weight) * tail_risk
        + float(config.complexity_weight) * complexity
    )
    summary = {
        "query_count": int(errors.size),
        "anchor_count": int(anchor_count),
        "median_te_m": float(np.median(errors)),
        "mean_te_m": float(errors.mean()),
        "p90_te_m": float(np.quantile(errors, 0.90)),
        "r5": float((errors <= 0.05).mean()),
        "data_risk": data_risk,
        "tail_cvar": tail_risk,
        "complexity_cost": float(config.complexity_weight) * complexity,
    }
    if group_ids is not None:
        groups = np.asarray(group_ids).reshape(-1)
        if groups.shape != errors.shape:
            raise ValueError("group ids must align with translation errors")
        group_risks = {
            str(group): float(normalized[groups == group].mean())
            for group in np.unique(groups)
        }
        worst_group_risk = max(group_risks.values())
        worst_group_cost = (
            float(config.worst_group_weight) * worst_group_risk
        )
        summary["group_risks"] = group_risks
        summary["worst_group_risk"] = worst_group_risk
        summary["worst_group_cost"] = worst_group_cost
        objective += worst_group_cost
    if hypotheses is not None:
        hypotheses = np.asarray(hypotheses, dtype=np.float64).reshape(-1)
        finite = hypotheses[np.isfinite(hypotheses)]
        summary["ransac_hypotheses_mean"] = (
            float(finite.mean()) if finite.size else None
        )
        summary["ransac_hypotheses_p90"] = (
            float(np.quantile(finite, 0.90)) if finite.size else None
        )
        hypotheses_cost = (
            float(config.hypotheses_weight)
            * float(finite.mean())
            / max(float(config.reference_hypotheses), 1.0)
            if finite.size
            else 0.0
        )
        summary["hypotheses_cost"] = hypotheses_cost
        objective += hypotheses_cost
    if runtime_seconds is not None:
        runtime = np.asarray(runtime_seconds, dtype=np.float64).reshape(-1)
        finite_runtime = runtime[np.isfinite(runtime)]
        runtime_mean = (
            float(finite_runtime.mean()) if finite_runtime.size else 0.0
        )
        summary["pnp_runtime_ms_mean"] = float(runtime_mean * 1000.0)
        summary["pnp_runtime_ms_p90"] = float(
            np.quantile(finite_runtime, 0.90) * 1000.0
            if finite_runtime.size
            else 0.0
        )
        runtime_cost = (
            float(config.runtime_weight)
            * runtime_mean
            / max(float(config.reference_runtime_seconds), 1e-6)
        )
        summary["runtime_cost"] = runtime_cost
        objective += runtime_cost
    summary["objective"] = objective
    return summary


def evaluate_structure_proposal(
    current_errors_m,
    proposal_errors_m,
    *,
    current_anchor_count: int,
    proposal_anchor_count: int,
    config: PoseRiskConfig,
    group_ids=None,
    current_hypotheses=None,
    proposal_hypotheses=None,
    current_runtime_seconds=None,
    proposal_runtime_seconds=None,
    proposal_churn_count: int = 0,
) -> dict:
    """Apply the protected query-level acceptance gate for one operation batch."""
    current = np.asarray(current_errors_m, dtype=np.float64).reshape(-1)
    proposal = np.asarray(proposal_errors_m, dtype=np.float64).reshape(-1)
    if current.shape != proposal.shape:
        raise ValueError("current and proposal errors must align by query")
    current_summary = summarize_pose_risk(
        current,
        anchor_count=current_anchor_count,
        config=config,
        group_ids=group_ids,
        hypotheses=current_hypotheses,
        runtime_seconds=current_runtime_seconds,
    )
    proposal_summary = summarize_pose_risk(
        proposal,
        anchor_count=proposal_anchor_count,
        config=config,
        group_ids=group_ids,
        hypotheses=proposal_hypotheses,
        runtime_seconds=proposal_runtime_seconds,
    )
    protected = current <= float(config.protected_threshold_m)
    positive_regression = np.maximum(proposal - current, 0.0)
    regression_weights = np.where(
        protected,
        float(config.protected_regression_multiplier),
        1.0,
    )
    regression_scale = max(float(config.regression_scale_m), 1e-8)
    soft_regression = float(
        np.mean(
            regression_weights
            * np.square(positive_regression / regression_scale)
        )
    )
    regression_cost = (
        float(config.soft_regression_weight) * soft_regression
    )
    churn_cost = (
        float(config.churn_weight)
        * max(int(proposal_churn_count), 0)
        / max(int(config.reference_churn_count), 1)
    )
    proposal_summary["soft_regression_risk"] = soft_regression
    proposal_summary["soft_regression_cost"] = regression_cost
    proposal_summary["churn_cost"] = churn_cost
    proposal_summary["objective"] += regression_cost + churn_cost
    protected_regression = protected & (
        proposal > float(config.protected_failure_threshold_m)
    )
    objective_gain = (
        current_summary["objective"] - proposal_summary["objective"]
    )
    gates = {
        "objective": objective_gain
        > float(config.minimum_objective_gain),
        "median": proposal_summary["median_te_m"]
        <= current_summary["median_te_m"]
        + float(config.max_median_regression_m),
        "r5": proposal_summary["r5"]
        >= current_summary["r5"] - float(config.max_r5_regression),
        "protected": int(protected_regression.sum())
        <= int(config.max_protected_regressions),
    }
    accepted = gates["objective"]
    if bool(config.hard_gate_mode):
        accepted = bool(all(gates.values()))
    return {
        "accepted": accepted,
        "gates": gates,
        "objective_gain": float(objective_gain),
        "protected_query_count": int(protected.sum()),
        "protected_regression_count": int(protected_regression.sum()),
        "current": current_summary,
        "proposal": proposal_summary,
        "config": asdict(config),
    }


def affected_query_mask(
    current_best_scores: list[torch.Tensor],
    proposal_best_scores: list[torch.Tensor],
) -> torch.Tensor:
    """Return queries whose deployed top-1 assignment changes."""
    if len(current_best_scores) != len(proposal_best_scores):
        raise ValueError("score lists must have the same query count")
    return torch.as_tensor(
        [
            bool(torch.any(proposal > current).item())
            for current, proposal in zip(
                current_best_scores, proposal_best_scores
            )
        ],
        dtype=torch.bool,
    )
