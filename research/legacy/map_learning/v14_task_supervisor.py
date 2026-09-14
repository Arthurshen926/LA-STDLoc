"""Bounded, task-space supervision for simple closed-loop map actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


TASK_CAP = 4.0


def _values(records: Sequence[Mapping], arm: str, key: str) -> np.ndarray:
    return np.asarray([float(row[arm][key]) for row in records], dtype=np.float64)


def summarize(records: Sequence[Mapping], arm: str) -> dict:
    """Summarize the actual PoseLib output without unbounded outlier leverage."""
    if not records:
        raise ValueError("task supervisor requires paired replay records")
    raw = _values(records, arm, "task_error")
    task = np.minimum(raw, TASK_CAP)
    te = _values(records, arm, "translation_error_cm")
    ae = _values(records, arm, "rotation_error_deg")
    success = (te < 5.0) & (ae < 5.0)
    catastrophic = (te >= 100.0) | (ae >= 30.0)
    q50, q75, q90, q95 = np.percentile(task, [50, 75, 90, 95])
    tail = task[task >= q95]
    cvar95 = float(tail.mean()) if tail.size else float(q95)
    pose_risk = 0.30 * q50 + 0.20 * q75 + 0.30 * q90 + 0.20 * cvar95
    failure_rate = float(np.mean(~success))
    catastrophic_rate = float(np.mean(catastrophic))
    return {
        "task_cap": TASK_CAP,
        "q50_task": float(q50),
        "q75_task": float(q75),
        "q90_task": float(q90),
        "cvar95_task": cvar95,
        "r5_percent": float(100.0 * success.mean()),
        "failure_count": int((~success).sum()),
        "catastrophic_count": int(catastrophic.sum()),
        "pose_risk": float(pose_risk),
        "total_risk": float(pose_risk + 2.0 * failure_rate + 4.0 * catastrophic_rate),
    }


def paired_effect(records: Sequence[Mapping], arm: str) -> dict:
    before = np.minimum(_values(records, "baseline", "task_error"), TASK_CAP)
    after = np.minimum(_values(records, arm, "task_error"), TASK_CAP)
    before_te = _values(records, "baseline", "translation_error_cm")
    before_ae = _values(records, "baseline", "rotation_error_deg")
    after_te = _values(records, arm, "translation_error_cm")
    after_ae = _values(records, arm, "rotation_error_deg")
    before_ok = (before_te < 5.0) & (before_ae < 5.0)
    after_ok = (after_te < 5.0) & (after_ae < 5.0)
    gain = before - after
    return {
        "mean_capped_task_gain": float(gain.mean()),
        "median_capped_task_gain": float(np.median(gain)),
        "improving_count": int((gain > 0).sum()),
        "worsening_count": int((gain < 0).sum()),
        "recovered_success_count": int((~before_ok & after_ok).sum()),
        "lost_success_count": int((before_ok & ~after_ok).sum()),
        "maximum_capped_regression": float(np.maximum(-gain, 0.0).max()),
    }


def _bootstrap_probability(
    records: Sequence[Mapping], arm: str, *, samples: int, seed: int
) -> float:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[int(row["pose_family_id"])].append(index)
    blocks = list(groups.values())
    if len(blocks) < 2:
        raise ValueError("pose-family bootstrap requires two families")
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(samples):
        selected = rng.integers(0, len(blocks), len(blocks))
        sample = [records[index] for block in selected for index in blocks[int(block)]]
        wins += int(summarize(sample, arm)["total_risk"] < summarize(sample, "baseline")["total_risk"])
    return float(wins / samples)


def supervise(
    records: Sequence[Mapping], arm: str, *, samples: int = 2000, seed: int = 1420260828
) -> dict:
    """Apply hard safety first, then require a measurable task-space effect."""
    baseline = summarize(records, "baseline")
    candidate = summarize(records, arm)
    effect = paired_effect(records, arm)
    hard_checks = {
        "r5_non_decreasing": candidate["r5_percent"] >= baseline["r5_percent"],
        "catastrophic_non_increasing": candidate["catastrophic_count"]
        <= baseline["catastrophic_count"],
        "q90_bounded": candidate["q90_task"] <= baseline["q90_task"] + 0.05,
        "no_large_single_regression": effect["maximum_capped_regression"] <= 2.0,
    }
    probability = _bootstrap_probability(records, arm, samples=samples, seed=seed)
    relative_risk_gain = (baseline["total_risk"] - candidate["total_risk"]) / max(
        baseline["total_risk"], 1e-12
    )
    substantial = bool(
        relative_risk_gain >= 0.01
        or candidate["r5_percent"] >= baseline["r5_percent"] + 1.0
        or (
            candidate["q50_task"] <= 0.99 * baseline["q50_task"]
            and candidate["q90_task"] <= 0.99 * baseline["q90_task"]
        )
    )
    safe_and_better = bool(
        all(hard_checks.values())
        and substantial
        and candidate["total_risk"] < baseline["total_risk"]
    )
    if safe_and_better and probability >= 0.95:
        classification = "DEFAULT_CANDIDATE"
    elif safe_and_better and probability >= 0.80:
        classification = "PARETO_CANDIDATE"
    else:
        classification = "NO_ACTION"
    return {
        "baseline": baseline,
        "candidate": candidate,
        "paired_effect": effect,
        "hard_checks": hard_checks,
        "relative_risk_gain": float(relative_risk_gain),
        "substantial_effect": substantial,
        "bootstrap_probability_lower_risk": probability,
        "classification": classification,
    }
