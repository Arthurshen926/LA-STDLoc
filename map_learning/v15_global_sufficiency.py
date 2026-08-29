"""Feedback-conditioned utilities for budgeted global map sufficiency.

The feedback signal only changes the order in which otherwise eligible Anchors
are admitted.  Mapping coverage and pose-information constraints remain hard
constraints owned by the global sufficiency selector.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from copy import deepcopy
import math

import torch


def size_aware_supervision(
    supervision: Mapping,
    *,
    compression_fraction: float,
    minimum_substantial_compression: float = 0.01,
) -> dict:
    """Extend task supervision with the controlled map-size objective.

    Localization safety remains non-negotiable.  A smaller map is considered a
    measurable closed-loop effect when it lowers paired task risk, passes every
    task-space hard check, and removes a non-trivial fraction of the state.  It
    therefore cannot excuse a localization regression; it only prevents the
    legacy accuracy-only gate from rejecting safe compression.
    """

    if not 0.0 <= compression_fraction < 1.0:
        raise ValueError("compression fraction must be in [0, 1)")
    if not 0.0 < minimum_substantial_compression < 1.0:
        raise ValueError("minimum substantial compression must be in (0, 1)")
    result = deepcopy(dict(supervision))
    hard_safe = all(bool(value) for value in result["hard_checks"].values())
    lower_risk = (
        float(result["candidate"]["total_risk"])
        < float(result["baseline"]["total_risk"])
    )
    size_effect = compression_fraction >= minimum_substantial_compression
    size_safe_and_better = bool(hard_safe and lower_risk and size_effect)
    probability = float(result["bootstrap_probability_lower_risk"])
    classification = str(result["classification"])
    if classification == "NO_ACTION" and size_safe_and_better:
        if probability >= 0.95:
            classification = "DEFAULT_CANDIDATE"
        elif probability >= 0.80:
            classification = "PARETO_CANDIDATE"
    result.update(
        {
            "compression_fraction": float(compression_fraction),
            "global_compression_substantial": bool(size_effect),
            "size_safe_and_better": size_safe_and_better,
            "task_only_classification": result["classification"],
            "classification": classification,
        }
    )
    return result


def _unique_rows(value: object, anchor_count: int) -> set[int]:
    rows = torch.as_tensor(value, dtype=torch.long).reshape(-1)
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= anchor_count):
        raise ValueError("feedback Anchor row is outside the frozen map")
    return set(rows.tolist())


def feedback_utility_components(
    records: Iterable[Mapping],
    *,
    anchor_count: int,
    harmful_anchor_rows: torch.Tensor,
    task_gain_cap: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Aggregate bounded, pose-family-unique positive and harmful evidence.

    Each pose family can credit an Anchor at most once for clean protection and
    once for task improvement.  A query's bounded task gain is divided over its
    unique positive alternatives, so dense feedback rows cannot dominate the
    controller merely by producing more keypoints.
    """

    if anchor_count <= 0 or task_gain_cap <= 0:
        raise ValueError("Anchor count and task gain cap must be positive")
    clean_families: dict[int, set[int]] = defaultdict(set)
    task_family_gain: dict[int, dict[int, float]] = defaultdict(dict)
    accepted_family_count = 0
    seen_families: set[int] = set()
    for record in records:
        if record.get("certificate_decision") != "ACCEPT":
            continue
        family = int(record["pose_family_id"])
        seen_families.add(family)
        clean = record.get("clean_protection_evidence", {})
        for row in _unique_rows(clean.get("positive_anchor_rows", ()), anchor_count):
            clean_families[row].add(family)
        if not bool(record.get("can_train_metric", False)):
            continue
        evidence = record.get("training_evidence", {})
        positives = _unique_rows(evidence.get("positive_anchor_rows", ()), anchor_count)
        if not positives:
            continue
        gain = max(
            0.0,
            min(float(evidence.get("actual_query_task_gain", 0.0)), task_gain_cap),
        ) / len(positives)
        for row in positives:
            # Multiple queries in one pose family still contribute at most the
            # strongest bounded evidence for this Anchor.
            task_family_gain[row][family] = max(
                task_family_gain[row].get(family, 0.0), gain
            )
    accepted_family_count = len(seen_families)

    clean_count = torch.zeros(anchor_count, dtype=torch.float32)
    task_gain = torch.zeros(anchor_count, dtype=torch.float32)
    task_count = torch.zeros(anchor_count, dtype=torch.float32)
    for row, families in clean_families.items():
        clean_count[row] = len(families)
    for row, family_gain in task_family_gain.items():
        task_count[row] = len(family_gain)
        task_gain[row] = sum(family_gain.values())

    harmful = torch.zeros(anchor_count, dtype=torch.bool)
    harmful_rows = torch.as_tensor(harmful_anchor_rows, dtype=torch.long).reshape(-1)
    if harmful_rows.numel():
        if int(harmful_rows.min()) < 0 or int(harmful_rows.max()) >= anchor_count:
            raise ValueError("harmful Anchor row is outside the frozen map")
        harmful[torch.unique(harmful_rows)] = True
    return {
        "clean_pose_family_count": clean_count,
        "task_pose_family_count": task_count,
        "bounded_task_gain": task_gain,
        "causally_harmful": harmful,
        "accepted_pose_family_count": torch.tensor(accepted_family_count),
    }


def feedback_conditioned_reliability(
    mapping_reliability: torch.Tensor,
    components: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return a bounded priority score without weakening mapping constraints."""

    base = torch.as_tensor(mapping_reliability).float().reshape(-1)
    if not bool(torch.isfinite(base).all()):
        raise ValueError("mapping reliability must be finite")
    clean = torch.as_tensor(components["clean_pose_family_count"]).float().reshape(-1)
    task_count = torch.as_tensor(components["task_pose_family_count"]).float().reshape(-1)
    task_gain = torch.as_tensor(components["bounded_task_gain"]).float().reshape(-1)
    harmful = torch.as_tensor(components["causally_harmful"]).bool().reshape(-1)
    if any(value.numel() != base.numel() for value in (clean, task_count, task_gain, harmful)):
        raise ValueError("feedback utility does not align with the frozen map")

    def saturating(value: torch.Tensor) -> torch.Tensor:
        return torch.log1p(value.clamp_min(0)) / math.log(2.0)

    positive = (
        0.20 * saturating(clean).clamp_max(3.0)
        + 0.30 * saturating(task_count).clamp_max(3.0)
        + 0.25 * saturating(task_gain).clamp_max(3.0)
    )
    # Exact delete-one PoseLib replay is a stronger causal signal than the
    # positive ranking hints.  Harmful Anchors remain eligible as a last resort
    # when a hard mapping constraint cannot be met without them.
    return base + positive - harmful.float() * 4.0
