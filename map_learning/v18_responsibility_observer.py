"""Counterfactual responsibility decomposition for closed-loop feedback.

The observer reports which bounded operation changes the standard localization
plant.  It never declares that a Query or Anchor is intrinsically "bad" and it
never uses leave-one-out map construction.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

import torch

from map_learning.v18_provenance_truth import (
    TRUTH_EQUIVALENT,
    TRUTH_NONE,
    TRUTH_UNIQUE,
    truth_membership_mask,
)
from map_learning.v9_causal_feedback import standard_pose_replay


def _success(pose: Mapping) -> bool:
    return bool(
        float(pose["translation_error_cm"]) < 5.0
        and float(pose["rotation_error_deg"]) < 5.0
    )


def _pose(
    keypoints: torch.Tensor,
    anchor_rows: torch.Tensor,
    *,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> dict:
    return standard_pose_replay(
        keypoints=keypoints,
        anchor_rows=anchor_rows,
        anchor_xyz=anchor_xyz,
        intrinsic=intrinsic,
        ground_truth_w2c=pose_w2c,
    )


def _next_winners_without_anchor(
    candidate_anchor_rows: torch.Tensor,
    removed_anchor: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.as_tensor(candidate_anchor_rows).long()
    available = candidates != int(removed_anchor)
    valid = available.any(1)
    first = available.float().argmax(1).long()
    rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
    return rows, candidates[rows, first[rows]]


def decompose_correspondence_responsibility(
    *,
    keypoints: torch.Tensor,
    candidate_anchor_rows: torch.Tensor,
    candidate_scores: torch.Tensor,
    truth: Mapping,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
    maximum_individual_rows: int = 32,
    minimum_task_gain: float = 0.01,
) -> dict:
    """Replay remove-row/remove-Anchor/replace/oracle counterfactuals."""

    xy = torch.as_tensor(keypoints).float().reshape(-1, 2)
    candidates = torch.as_tensor(candidate_anchor_rows).long()
    scores = torch.as_tensor(candidate_scores).float()
    if candidates.shape != scores.shape or candidates.shape[0] != xy.shape[0]:
        raise ValueError("responsibility competition rows do not align")
    membership = truth_membership_mask(truth, candidates)
    status = torch.as_tensor(truth["truth_status"]).long()
    truth_offsets = torch.as_tensor(truth["truth_offsets"]).long()
    truth_anchors = torch.as_tensor(truth["truth_anchor_rows"]).long()
    decisive = (status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT)
    wrong = decisive & ~membership[:, 0]
    baseline = _pose(
        xy,
        candidates[:, 0],
        anchor_xyz=anchor_xyz,
        intrinsic=intrinsic,
        pose_w2c=pose_w2c,
    )
    wrong_rows = torch.nonzero(wrong, as_tuple=False).reshape(-1)
    if wrong_rows.numel() > int(maximum_individual_rows):
        # Rows with the strongest wrong-over-truth margin are the important
        # correspondences. Missing truth in Top-L receives infinite priority.
        priority = torch.full((wrong_rows.numel(),), float("inf"))
        for local, row in enumerate(wrong_rows.tolist()):
            positive = torch.nonzero(membership[row], as_tuple=False).reshape(-1)
            if positive.numel():
                priority[local] = scores[row, 0] - scores[row, positive[0]]
        selected = torch.argsort(priority, descending=True, stable=True)[
            : int(maximum_individual_rows)
        ]
        wrong_rows = wrong_rows[selected]

    row_audit = []
    anchors_to_rows: dict[int, list[int]] = defaultdict(list)
    for row in wrong_rows.tolist():
        wrong_anchor = int(candidates[row, 0])
        anchors_to_rows[wrong_anchor].append(row)
        retained = torch.cat((torch.arange(row), torch.arange(row + 1, xy.shape[0])))
        removed_row_pose = _pose(
            xy[retained],
            candidates[retained, 0],
            anchor_xyz=anchor_xyz,
            intrinsic=intrinsic,
            pose_w2c=pose_w2c,
        )
        start, stop = int(truth_offsets[row]), int(truth_offsets[row + 1])
        replacement_anchor = int(truth_anchors[start])
        replacement = candidates[:, 0].clone()
        replacement[row] = replacement_anchor
        replaced_pose = _pose(
            xy,
            replacement,
            anchor_xyz=anchor_xyz,
            intrinsic=intrinsic,
            pose_w2c=pose_w2c,
        )
        row_audit.append(
            {
                "query_row": row,
                "wrong_anchor_row": wrong_anchor,
                "truth_anchor_row": replacement_anchor,
                "remove_row_task_gain": float(
                    baseline["task_error"] - removed_row_pose["task_error"]
                ),
                "replace_with_truth_task_gain": float(
                    baseline["task_error"] - replaced_pose["task_error"]
                ),
                "remove_row_pose": removed_row_pose,
                "replace_with_truth_pose": replaced_pose,
            }
        )

    anchor_audit = []
    for anchor, causal_rows in sorted(anchors_to_rows.items()):
        retained_rows, replacement = _next_winners_without_anchor(candidates, anchor)
        removed_pose = _pose(
            xy[retained_rows],
            replacement,
            anchor_xyz=anchor_xyz,
            intrinsic=intrinsic,
            pose_w2c=pose_w2c,
        )
        anchor_audit.append(
            {
                "anchor_row": anchor,
                "causal_query_rows": torch.tensor(causal_rows, dtype=torch.long),
                "remove_anchor_task_gain": float(
                    baseline["task_error"] - removed_pose["task_error"]
                ),
                "remove_anchor_pose": removed_pose,
            }
        )

    oracle_rows = torch.nonzero(decisive, as_tuple=False).reshape(-1)
    oracle_anchors = candidates[oracle_rows, 0].clone()
    for local, row in enumerate(oracle_rows.tolist()):
        start, stop = int(truth_offsets[row]), int(truth_offsets[row + 1])
        if stop > start:
            oracle_anchors[local] = truth_anchors[start]
    oracle_pose = (
        None
        if oracle_rows.numel() < 4
        else _pose(
            xy[oracle_rows],
            oracle_anchors,
            anchor_xyz=anchor_xyz,
            intrinsic=intrinsic,
            pose_w2c=pose_w2c,
        )
    )
    minimum = float(minimum_task_gain)
    row_suppressible = [
        item["query_row"]
        for item in row_audit
        if item["remove_row_task_gain"] >= minimum
    ]
    metric_controllable = [
        item["query_row"]
        for item in row_audit
        if item["replace_with_truth_task_gain"] >= minimum
    ]
    anchor_suppressible = [
        item["anchor_row"]
        for item in anchor_audit
        if item["remove_anchor_task_gain"] >= minimum
    ]
    return {
        "schema": "lafgs_v18_counterfactual_responsibility",
        "version": 1,
        "loo_used": False,
        "baseline_pose": baseline,
        "truth_status_counts": dict(truth["status_counts"]),
        "coverage_limited_row_count": int((status == TRUTH_NONE).sum()),
        "wrong_decisive_row_count": int(wrong.sum()),
        "audited_wrong_row_count": int(wrong_rows.numel()),
        "row_counterfactuals": row_audit,
        "anchor_counterfactuals": anchor_audit,
        "full_truth_oracle_pose": oracle_pose,
        "full_truth_oracle_task_gain": (
            float("nan")
            if oracle_pose is None
            else float(baseline["task_error"] - oracle_pose["task_error"])
        ),
        "row_suppressible_query_rows": torch.tensor(row_suppressible, dtype=torch.long),
        "anchor_suppressible_anchor_rows": torch.tensor(
            anchor_suppressible, dtype=torch.long
        ),
        "metric_controllable_query_rows": torch.tensor(
            metric_controllable, dtype=torch.long
        ),
        "geometry_limited": bool(
            oracle_pose is not None
            and not _success(oracle_pose)
            and float(baseline["task_error"] - oracle_pose["task_error"]) < minimum
        ),
    }


__all__ = ["decompose_correspondence_responsibility"]
