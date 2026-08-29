"""Risk-sensitive supervision for bounded closed-loop map actions.

The supervisor deliberately separates hard deployment safety, scalar task risk,
and Pareto value.  It consumes *paired* exact PoseLib replays; ranking losses and
observer labels are not accepted as substitutes for plant output.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskWeights:
    q50: float = 0.35
    q75: float = 0.20
    q90: float = 0.25
    cvar95: float = 0.20
    r5_failure: float = 2.0
    catastrophic: float = 5.0
    latency_ratio: float = 0.02


@dataclass(frozen=True)
class SafetyLimits:
    maximum_r5_drop_percent: float = 1.0
    maximum_p90_task_increase: float = 0.20
    maximum_single_query_task_regression: float = 2.0
    maximum_catastrophic_increase: int = 1


def _array(records: Sequence[Mapping], arm: str, key: str) -> np.ndarray:
    return np.asarray([float(row[arm][key]) for row in records], dtype=np.float64)


def _cvar(values: np.ndarray, percentile: float = 95.0) -> float:
    threshold = float(np.percentile(values, percentile))
    tail = values[values >= threshold]
    return float(tail.mean()) if tail.size else threshold


def summarize_arm(
    records: Sequence[Mapping],
    arm: str,
    *,
    weights: RiskWeights = RiskWeights(),
    latency_ratio: float = 1.0,
) -> dict:
    if not records:
        raise ValueError("risk supervision requires at least one paired replay")
    task = _array(records, arm, "task_error")
    translation = _array(records, arm, "translation_error_cm")
    rotation = _array(records, arm, "rotation_error_deg")
    r5 = (translation < 5.0) & (rotation < 5.0)
    catastrophic = (translation >= 100.0) | (rotation >= 30.0)
    quantiles = {
        "q50_task": float(np.percentile(task, 50)),
        "q75_task": float(np.percentile(task, 75)),
        "q90_task": float(np.percentile(task, 90)),
        "cvar95_task": _cvar(task),
    }
    pose_risk = (
        weights.q50 * quantiles["q50_task"]
        + weights.q75 * quantiles["q75_task"]
        + weights.q90 * quantiles["q90_task"]
        + weights.cvar95 * quantiles["cvar95_task"]
    )
    failure_rate = float(1.0 - r5.mean())
    catastrophic_rate = float(catastrophic.mean())
    total_risk = (
        pose_risk
        + weights.r5_failure * failure_rate
        + weights.catastrophic * catastrophic_rate
        + weights.latency_ratio * float(latency_ratio)
    )
    return {
        **quantiles,
        "pose_risk": float(pose_risk),
        "r5_percent": float(100.0 * r5.mean()),
        "catastrophic_count": int(catastrophic.sum()),
        "catastrophic_rate": catastrophic_rate,
        "latency_ratio": float(latency_ratio),
        "total_risk": float(total_risk),
    }


def paired_effect(records: Sequence[Mapping], candidate_arm: str) -> dict:
    baseline = _array(records, "baseline", "task_error")
    candidate = _array(records, candidate_arm, "task_error")
    gain = baseline - candidate
    benefit = float(np.maximum(gain, 0.0).sum())
    harm = float(np.maximum(-gain, 0.0).sum())
    return {
        "benefit": benefit,
        "harm": harm,
        "net_gain": benefit - harm,
        "harm_ratio": harm / max(benefit, 1e-12),
        "median_task_gain": float(np.median(gain)),
        "improving_fraction": float(np.mean(gain > 0.0)),
        "worsening_fraction": float(np.mean(gain < 0.0)),
        "maximum_task_regression": float(np.maximum(-gain, 0.0).max()),
    }


def hard_safety(
    baseline: Mapping,
    candidate: Mapping,
    effect: Mapping,
    *,
    limits: SafetyLimits = SafetyLimits(),
) -> dict:
    checks = {
        "catastrophic": int(candidate["catastrophic_count"])
        <= int(baseline["catastrophic_count"])
        + int(limits.maximum_catastrophic_increase),
        "r5": float(candidate["r5_percent"])
        >= float(baseline["r5_percent"]) - limits.maximum_r5_drop_percent,
        "p90_task": float(candidate["q90_task"])
        <= float(baseline["q90_task"]) + limits.maximum_p90_task_increase,
        "single_query": float(effect["maximum_task_regression"])
        <= limits.maximum_single_query_task_regression,
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def _block_indices(records: Sequence[Mapping]) -> list[np.ndarray]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[int(row["pose_family_id"])].append(index)
    return [np.asarray(rows, dtype=np.int64) for rows in groups.values()]


def block_bootstrap_improvement_probability(
    records: Sequence[Mapping],
    candidate_arm: str,
    *,
    weights: RiskWeights = RiskWeights(),
    baseline_latency_ratio: float = 1.0,
    candidate_latency_ratio: float = 1.0,
    samples: int = 2000,
    seed: int = 1320260828,
) -> dict:
    blocks = _block_indices(records)
    if len(blocks) < 2:
        raise ValueError("pose-family block bootstrap requires at least two families")
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(samples), dtype=np.float64)
    for draw in range(int(samples)):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[index] for index in chosen])
        sampled = [records[int(index)] for index in indices]
        base = summarize_arm(
            sampled, "baseline", weights=weights, latency_ratio=baseline_latency_ratio
        )
        cand = summarize_arm(
            sampled,
            candidate_arm,
            weights=weights,
            latency_ratio=candidate_latency_ratio,
        )
        deltas[draw] = cand["total_risk"] - base["total_risk"]
    return {
        "samples": int(samples),
        "pose_family_count": len(blocks),
        "probability_candidate_lower_risk": float(np.mean(deltas < 0.0)),
        "median_delta_risk": float(np.median(deltas)),
        "delta_risk_ci90": [
            float(np.percentile(deltas, 5)),
            float(np.percentile(deltas, 95)),
        ],
    }


def pareto_relation(baseline: Mapping, candidate: Mapping) -> str:
    keys = ("q50_task", "q75_task", "q90_task", "cvar95_task")
    lower = [float(candidate[key]) <= float(baseline[key]) for key in keys]
    strict_lower = [float(candidate[key]) < float(baseline[key]) for key in keys]
    higher_r5 = float(candidate["r5_percent"]) >= float(baseline["r5_percent"])
    lower_cat = int(candidate["catastrophic_count"]) <= int(
        baseline["catastrophic_count"]
    )
    if all(lower) and any(strict_lower) and higher_r5 and lower_cat:
        return "DOMINATES_BASELINE"
    improves_any = any(strict_lower) or float(candidate["r5_percent"]) > float(
        baseline["r5_percent"]
    ) or int(candidate["catastrophic_count"]) < int(baseline["catastrophic_count"])
    return "TRADEOFF" if improves_any else "DOMINATED_OR_EQUAL"


def supervise_candidate(
    records: Sequence[Mapping],
    candidate_arm: str,
    *,
    weights: RiskWeights = RiskWeights(),
    limits: SafetyLimits = SafetyLimits(),
    baseline_latency_ratio: float = 1.0,
    candidate_latency_ratio: float = 1.0,
    bootstrap_samples: int = 2000,
    seed: int = 1320260828,
) -> dict:
    baseline = summarize_arm(
        records, "baseline", weights=weights, latency_ratio=baseline_latency_ratio
    )
    candidate = summarize_arm(
        records,
        candidate_arm,
        weights=weights,
        latency_ratio=candidate_latency_ratio,
    )
    effect = paired_effect(records, candidate_arm)
    safety = hard_safety(baseline, candidate, effect, limits=limits)
    bootstrap = block_bootstrap_improvement_probability(
        records,
        candidate_arm,
        weights=weights,
        baseline_latency_ratio=baseline_latency_ratio,
        candidate_latency_ratio=candidate_latency_ratio,
        samples=bootstrap_samples,
        seed=seed,
    )
    probability = bootstrap["probability_candidate_lower_risk"]
    if not safety["passed"]:
        classification = "REJECTED"
    elif candidate["total_risk"] < baseline["total_risk"] and probability >= 0.95:
        classification = "DEFAULT_CANDIDATE"
    elif pareto_relation(baseline, candidate) != "DOMINATED_OR_EQUAL" and probability >= 0.80:
        classification = "PARETO_CANDIDATE"
    else:
        classification = "ANALYSIS_ONLY"
    return {
        "baseline": baseline,
        "candidate": candidate,
        "paired_effect": effect,
        "hard_safety": safety,
        "bootstrap": bootstrap,
        "pareto_relation": pareto_relation(baseline, candidate),
        "classification": classification,
    }


def validate_records(records: Iterable[Mapping], candidate_arms: Sequence[str]) -> list[dict]:
    materialized = list(records)
    required = {"query_index", "pose_family_id", "baseline", *candidate_arms}
    if any(not required.issubset(row) for row in materialized):
        raise ValueError("paired replay rows do not contain every requested arm")
    query_ids = [int(row["query_index"]) for row in materialized]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("paired replay query IDs must be unique")
    return materialized
