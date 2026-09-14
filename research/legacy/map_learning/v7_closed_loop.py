"""Fresh-batch confirmation, atomic rollback, and bounded-round supervision."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def _paired_records(
    baseline: Sequence[Mapping[str, Any]],
    proposal: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    left = {int(item["query_index"]): item for item in baseline}
    right = {int(item["query_index"]): item for item in proposal}
    if (
        len(left) != len(baseline)
        or len(right) != len(proposal)
        or set(left) != set(right)
    ):
        raise ValueError("P6 requires one paired result per confirmation query")
    pairs = []
    for query in sorted(left):
        a, b = left[query], right[query]
        if a.get("rgb_sha256") != b.get("rgb_sha256"):
            raise ValueError("P6 baseline/proposal did not consume the same RGB")
        if a.get("certificate_decision") != b.get("certificate_decision"):
            raise ValueError("P6 paired certificate decisions differ")
        if (
            a.get("uses_test_queries") is not False
            or b.get("uses_test_queries") is not False
        ):
            raise ValueError("test queries cannot enter P6 confirmation")
        pairs.append((a, b))
    return pairs


def _metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    translation_scale_cm: float,
    rotation_scale_deg: float,
    catastrophic_translation_cm: float,
    catastrophic_rotation_deg: float,
) -> dict[str, float | int]:
    accepted = [item for item in records if item["certificate_decision"] == "ACCEPT"]
    if not accepted:
        raise ValueError("P6 has no ACCEPT confirmation query")
    translation = np.asarray([float(item["translation_error_cm"]) for item in accepted])
    rotation = np.asarray([float(item["rotation_error_deg"]) for item in accepted])
    task_error = np.sqrt(
        (translation / float(translation_scale_cm)) ** 2
        + (rotation / float(rotation_scale_deg)) ** 2
    )
    r5 = (translation < 5.0) & (rotation < 5.0)
    catastrophic = (translation >= float(catastrophic_translation_cm)) | (
        rotation >= float(catastrophic_rotation_deg)
    )
    return {
        "query_count": len(accepted),
        "median_task_error": float(np.median(task_error)),
        "p90_task_error": float(np.quantile(task_error, 0.9)),
        "recall_5cm_5deg_percent": float(r5.mean() * 100.0),
        "catastrophic_count": int(catastrophic.sum()),
        "median_translation_cm": float(np.median(translation)),
        "median_rotation_deg": float(np.median(rotation)),
        "mean_runtime_ms": float(
            np.mean([float(item["runtime_ms"]) for item in accepted])
        ),
    }


def confirm_or_rollback_v7_round(
    *,
    round_index: int,
    baseline_map_sha256: str,
    proposal_map_sha256: str,
    control_query_ids: Sequence[str],
    confirmation_query_ids: Sequence[str],
    baseline_results: Sequence[Mapping[str, Any]],
    proposal_results: Sequence[Mapping[str, Any]],
    minimum_median_task_improvement: float = 0.001,
    maximum_p90_task_regression: float = 0.20,
    minimum_r5_regression_percent: float = -0.01,
    translation_scale_cm: float = 5.0,
    rotation_scale_deg: float = 5.0,
    catastrophic_translation_cm: float = 100.0,
    catastrophic_rotation_deg: float = 30.0,
) -> dict[str, Any]:
    """Accept a proposal only on a disjoint, identically rendered fresh batch."""

    if int(round_index) not in {0, 1}:
        raise ValueError("V7 permits at most two confirmation rounds")
    if baseline_map_sha256 == proposal_map_sha256:
        raise ValueError("P6 confirmation requires a changed proposal")
    if set(control_query_ids) & set(confirmation_query_ids):
        raise ValueError("P6 confirmation queries must be fresh for this action")
    pairs = _paired_records(baseline_results, proposal_results)
    baseline = _metrics(
        [item[0] for item in pairs],
        translation_scale_cm=translation_scale_cm,
        rotation_scale_deg=rotation_scale_deg,
        catastrophic_translation_cm=catastrophic_translation_cm,
        catastrophic_rotation_deg=catastrophic_rotation_deg,
    )
    proposal = _metrics(
        [item[1] for item in pairs],
        translation_scale_cm=translation_scale_cm,
        rotation_scale_deg=rotation_scale_deg,
        catastrophic_translation_cm=catastrophic_translation_cm,
        catastrophic_rotation_deg=catastrophic_rotation_deg,
    )
    median_improvement = float(
        baseline["median_task_error"] - proposal["median_task_error"]
    )
    p90_regression = float(proposal["p90_task_error"] - baseline["p90_task_error"])
    r5_delta = float(
        proposal["recall_5cm_5deg_percent"] - baseline["recall_5cm_5deg_percent"]
    )
    gates = {
        "median_improved": median_improvement >= float(minimum_median_task_improvement),
        "p90_soft_protection": p90_regression <= float(maximum_p90_task_regression),
        "r5_soft_protection": r5_delta >= float(minimum_r5_regression_percent),
        "catastrophic_no_increase": proposal["catastrophic_count"]
        <= baseline["catastrophic_count"],
    }
    accepted = all(gates.values())
    chosen_sha = proposal_map_sha256 if accepted else baseline_map_sha256
    return {
        "schema": "lafgs_v7_fresh_batch_confirmation",
        "version": 1,
        "round_index": int(round_index),
        "decision": "ACCEPT" if accepted else "ROLLBACK",
        "baseline_map_sha256": baseline_map_sha256,
        "proposal_map_sha256": proposal_map_sha256,
        "chosen_map_sha256": chosen_sha,
        "atomic_rollback_exact": not accepted and chosen_sha == baseline_map_sha256,
        "baseline": baseline,
        "proposal": proposal,
        "median_task_improvement": median_improvement,
        "p90_task_regression": p90_regression,
        "r5_delta_percent": r5_delta,
        "gates": gates,
        "uses_test_queries": False,
        "same_rgb_paired": True,
        "fresh_confirmation": True,
    }


def next_v7_round_action(
    *,
    completed_rounds: int,
    previous_proposal_accepted: bool,
    executable_descriptor_deficit_count: int,
    median_task_improvement: float,
    minimum_median_task_improvement: float = 0.001,
) -> str:
    """Return CONTINUE or a preregistered P7 stop reason."""

    if int(completed_rounds) >= 2:
        return "maximum_two_rounds_reached"
    if not previous_proposal_accepted:
        return "proposal_not_accepted"
    if int(executable_descriptor_deficit_count) <= 0:
        return "no_executable_descriptor_deficit"
    if not math.isfinite(float(median_task_improvement)) or float(
        median_task_improvement
    ) < float(minimum_median_task_improvement):
        return "median_improvement_below_threshold"
    return "CONTINUE"
