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
    complexity_weight: float = 0.01
    reference_anchor_count: int = 50_000
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
        "objective": objective,
    }
    if hypotheses is not None:
        hypotheses = np.asarray(hypotheses, dtype=np.float64).reshape(-1)
        finite = hypotheses[np.isfinite(hypotheses)]
        summary["ransac_hypotheses_mean"] = (
            float(finite.mean()) if finite.size else None
        )
        summary["ransac_hypotheses_p90"] = (
            float(np.quantile(finite, 0.90)) if finite.size else None
        )
    if runtime_seconds is not None:
        runtime = np.asarray(runtime_seconds, dtype=np.float64).reshape(-1)
        summary["pnp_runtime_ms_mean"] = float(runtime.mean() * 1000.0)
        summary["pnp_runtime_ms_p90"] = float(
            np.quantile(runtime, 0.90) * 1000.0
        )
    return summary


def evaluate_structure_proposal(
    current_errors_m,
    proposal_errors_m,
    *,
    current_anchor_count: int,
    proposal_anchor_count: int,
    config: PoseRiskConfig,
    current_hypotheses=None,
    proposal_hypotheses=None,
    current_runtime_seconds=None,
    proposal_runtime_seconds=None,
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
        hypotheses=current_hypotheses,
        runtime_seconds=current_runtime_seconds,
    )
    proposal_summary = summarize_pose_risk(
        proposal,
        anchor_count=proposal_anchor_count,
        config=config,
        hypotheses=proposal_hypotheses,
        runtime_seconds=proposal_runtime_seconds,
    )
    protected = current <= float(config.protected_threshold_m)
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
    return {
        "accepted": bool(all(gates.values())),
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

