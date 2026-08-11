"""Exact mapping-only pose-set oracle for compact localization maps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


@dataclass(frozen=True, order=True)
class PoseSetAction:
    kind: str
    row: int
    anchor: int = -1
    priority: float = 0.0


def normalized_pose_risk(
    *,
    translation_cm: float,
    rotation_deg: float,
    translation_scale_m: float,
    rotation_scale_deg: float,
    failed: bool = False,
) -> float:
    """Scene-normalized robust risk used only to compare paired replays."""
    translation_scale_cm = max(float(translation_scale_m) * 100.0, 1e-6)
    rotation_scale_deg = max(float(rotation_scale_deg), 1e-6)
    risk = math.log1p(max(float(translation_cm), 0.0) / translation_scale_cm)
    risk += math.log1p(max(float(rotation_deg), 0.0) / rotation_scale_deg)
    if failed:
        risk += 20.0
    return float(risk)


def apply_pose_set_actions(
    assignments: np.ndarray,
    actions: tuple[PoseSetAction, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply swap/reject actions while rejecting contradictory row edits."""
    revised = np.asarray(assignments, dtype=np.int64).copy()
    active = np.ones(revised.shape[0], dtype=bool)
    touched: set[int] = set()
    for action in actions:
        row = int(action.row)
        if row < 0 or row >= revised.shape[0]:
            raise ValueError("pose-set action row is out of range")
        if row in touched:
            raise ValueError("pose-set actions edit the same row more than once")
        touched.add(row)
        if action.kind == "swap":
            if int(action.anchor) < 0:
                raise ValueError("swap action requires an anchor")
            revised[row] = int(action.anchor)
        elif action.kind == "reject":
            active[row] = False
        else:
            raise ValueError(f"unsupported pose-set action: {action.kind}")
    return revised, active


def beam_search_pose_set(
    actions: list[PoseSetAction],
    evaluate: Callable[[tuple[PoseSetAction, ...]], dict],
    *,
    maximum_depth: int,
    beam_width: int,
) -> tuple[tuple[PoseSetAction, ...], dict, list[dict]]:
    """Search non-additive correspondence edits using exact solver outcomes."""
    maximum_depth = max(int(maximum_depth), 0)
    beam_width = max(int(beam_width), 1)
    ordered = sorted(actions, key=lambda action: (-action.priority, action))
    cache: dict[tuple[PoseSetAction, ...], dict] = {}

    def outcome(selected: tuple[PoseSetAction, ...]) -> dict:
        key = tuple(sorted(selected))
        if key not in cache:
            cache[key] = evaluate(key)
        return cache[key]

    empty: tuple[PoseSetAction, ...] = ()
    best = empty
    best_outcome = outcome(empty)
    beam = [empty]
    trace = [{"depth": 0, "risk": float(best_outcome["risk"]), "actions": []}]
    for depth in range(1, maximum_depth + 1):
        expanded: set[tuple[PoseSetAction, ...]] = set()
        for selected in beam:
            used_rows = {int(action.row) for action in selected}
            for action in ordered:
                if int(action.row) in used_rows:
                    continue
                expanded.add(tuple(sorted((*selected, action))))
        if not expanded:
            break
        ranked = sorted(
            expanded,
            key=lambda selected: (
                float(outcome(selected)["risk"]),
                len(selected),
                selected,
            ),
        )
        beam = ranked[:beam_width]
        candidate = beam[0]
        candidate_outcome = outcome(candidate)
        if float(candidate_outcome["risk"]) < float(best_outcome["risk"]):
            best, best_outcome = candidate, candidate_outcome
        trace.append(
            {
                "depth": depth,
                "risk": float(candidate_outcome["risk"]),
                "actions": [
                    {"kind": action.kind, "row": action.row, "anchor": action.anchor}
                    for action in candidate
                ],
            }
        )
    return best, best_outcome, trace
