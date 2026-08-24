"""Exact-identity pose-set diagnostics for the V6 feedback core."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable

import numpy as np
import torch

from map_learning.pose_set_oracle import PoseSetAction


def unique_anchor_rows(
    pairs: torch.Tensor,
    *,
    keypoints: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    """Keep one GT-consistent query row per exact Anchor identity.

    The oracle may use ground truth, but it may not give one landmark multiple
    votes.  Lowest reprojection residual wins and the row index breaks ties.
    """

    pairs = torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)
    if pairs.numel() == 0:
        return pairs
    rows, anchors = pairs.T
    if bool((rows < 0).any()) or bool((rows >= keypoints.shape[0]).any()):
        raise ValueError("exact-identity query row is out of range")
    if bool((anchors < 0).any()) or bool((anchors >= anchor_xyz.shape[0]).any()):
        raise ValueError("exact-identity Anchor row is out of range")
    camera = anchor_xyz[anchors].float() @ pose_w2c[:3, :3].float().T
    camera = camera + pose_w2c[:3, 3].float()
    homogeneous = camera @ intrinsics.float().T
    projected = homogeneous[:, :2] / homogeneous[:, 2:].clamp_min(1e-8)
    residual = torch.linalg.norm(projected - keypoints[rows].float(), dim=1)
    residual = torch.where(
        torch.isfinite(residual) & (camera[:, 2] > 0),
        residual,
        torch.full_like(residual, float("inf")),
    )
    selected = []
    for anchor in torch.unique(anchors, sorted=True).tolist():
        local = torch.nonzero(anchors == int(anchor), as_tuple=False).reshape(-1)
        minimum = residual[local].min()
        tied = local[residual[local] == minimum]
        chosen = tied[rows[tied] == rows[tied].min()].min()
        selected.append(int(chosen))
    return pairs[torch.as_tensor(selected, dtype=torch.long)]


def apply_swaps(
    assignments: np.ndarray,
    actions: tuple[PoseSetAction, ...],
) -> np.ndarray:
    """Apply a contradiction-free exact-identity correction set."""

    revised = np.asarray(assignments, dtype=np.int64).copy()
    touched: set[int] = set()
    for action in actions:
        if action.kind != "swap":
            raise ValueError("V6 correction-set oracle accepts swap actions only")
        row = int(action.row)
        if row < 0 or row >= revised.shape[0]:
            raise ValueError("correction-set query row is out of range")
        if row in touched:
            raise ValueError("correction set edits one row more than once")
        if int(action.anchor) < 0:
            raise ValueError("correction-set target Anchor is invalid")
        touched.add(row)
        revised[row] = int(action.anchor)
    return revised


def bounded_minimum_success_set(
    actions: list[PoseSetAction],
    evaluate: Callable[[tuple[PoseSetAction, ...]], dict],
    *,
    maximum_depth: int,
    beam_width: int,
) -> tuple[tuple[PoseSetAction, ...] | None, dict | None, list[dict]]:
    """Find the shallowest successful set retained by a bounded pose-risk beam.

    This is deliberately reported as a bounded oracle: candidate truncation or
    beam pruning can miss the global minimum.  Within each retained depth, a
    successful set is preferred by pose risk and then deterministically.
    """

    maximum_depth = max(int(maximum_depth), 0)
    beam_width = max(int(beam_width), 1)
    ordered = sorted(actions, key=lambda action: (-action.priority, action))
    cache: dict[tuple[PoseSetAction, ...], dict] = {}

    def outcome(selected: tuple[PoseSetAction, ...]) -> dict:
        selected = tuple(sorted(selected))
        if selected not in cache:
            cache[selected] = evaluate(selected)
        return cache[selected]

    empty: tuple[PoseSetAction, ...] = ()
    initial = outcome(empty)
    trace = [
        {
            "depth": 0,
            "evaluated_set_count": 1,
            "best_risk": float(initial["risk"]),
            "success_count": int(bool(initial["success"])),
        }
    ]
    if bool(initial["success"]):
        return empty, initial, trace
    beam = [empty]
    for depth in range(1, maximum_depth + 1):
        expanded: set[tuple[PoseSetAction, ...]] = set()
        for selected in beam:
            used_rows = {int(action.row) for action in selected}
            last = max((ordered.index(action) for action in selected), default=-1)
            for action in ordered[last + 1 :]:
                if int(action.row) in used_rows:
                    continue
                expanded.add(tuple(sorted((*selected, action))))
        if not expanded:
            break
        ranked = sorted(
            expanded,
            key=lambda selected: (
                not bool(outcome(selected)["success"]),
                float(outcome(selected)["risk"]),
                selected,
            ),
        )
        successful = [selected for selected in ranked if outcome(selected)["success"]]
        trace.append(
            {
                "depth": depth,
                "evaluated_set_count": len(expanded),
                "best_risk": float(outcome(ranked[0])["risk"]),
                "success_count": len(successful),
            }
        )
        if successful:
            selected = successful[0]
            return selected, outcome(selected), trace
        beam = ranked[:beam_width]
    return None, None, trace


def serialize_actions(actions: tuple[PoseSetAction, ...] | None) -> list[dict]:
    return [] if actions is None else [asdict(action) for action in actions]
