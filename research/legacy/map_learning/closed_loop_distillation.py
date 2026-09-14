"""Safeguarded finite-round acceptance for V6 policy improvement."""

from __future__ import annotations

from collections.abc import Mapping


RISK_FIELDS = (
    "catastrophic_100cm_count",
    "cvar95_te_cm",
    "one_minus_recall_5cm_5deg_percent",
    "mean_te_cm",
    "median_te_cm",
    "anchor_count",
    "online_latency_ms",
)


def deployment_risk(summary: Mapping) -> tuple[float, ...]:
    return (
        float(summary["catastrophic_100cm_count"]),
        float(summary["cvar95_te_cm"]),
        100.0 - float(summary["recall_5cm_5deg_percent"]),
        float(summary["mean_te_cm"]),
        float(summary["median_te_cm"]),
        float(summary["anchor_count"]),
        float(summary["online_latency_ms"]),
    )


def accept_candidate(
    baseline: Mapping,
    candidate: Mapping,
    *,
    seen_state_hashes: set[str],
    candidate_state_hash: str,
    maximum_anchor_count: int,
    maximum_online_latency_ms: float,
) -> dict:
    if candidate_state_hash in seen_state_hashes:
        return {"accepted": False, "reason": "repeated_state_hash"}
    hard_guards = {
        "catastrophic_nonincrease": int(candidate["catastrophic_100cm_count"])
        <= int(baseline["catastrophic_100cm_count"]),
        "recall_nonregression": float(candidate["recall_5cm_5deg_percent"])
        >= float(baseline["recall_5cm_5deg_percent"]),
        "anchor_budget": int(candidate["anchor_count"]) <= int(maximum_anchor_count),
        "latency_budget": float(candidate["online_latency_ms"])
        <= float(maximum_online_latency_ms),
    }
    if not all(hard_guards.values()):
        return {"accepted": False, "reason": "hard_guard", "guards": hard_guards}
    baseline_risk = deployment_risk(baseline)
    candidate_risk = deployment_risk(candidate)
    accepted = candidate_risk < baseline_risk
    return {
        "accepted": accepted,
        "reason": "strict_lexicographic_improvement" if accepted else "no_improvement",
        "guards": hard_guards,
        "risk_fields": list(RISK_FIELDS),
        "baseline_risk": baseline_risk,
        "candidate_risk": candidate_risk,
    }
