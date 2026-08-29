"""Competition-aligned evidence for a bounded shared descriptor metric.

The controller and metric action must describe the same deployed plant.  This
module therefore derives both repair pairs and protection pairs from the active
Top-L winner/runner-up state rather than reusing an older observer's cached
pair choice.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import re

import torch


def select_minimum_effective_gain(
    decisions: Mapping[str, Mapping],
) -> str | None:
    """Choose the least metric energy that already clears the task gates."""

    pattern = re.compile(r"^alpha_(\d+(?:p\d+)?)_active$")
    eligible = []
    for arm, decision in decisions.items():
        match = pattern.match(str(arm))
        if match is None:
            raise ValueError(f"unsupported V17 gain arm: {arm}")
        classification = str(decision["classification"])
        safe = bool(decision["hard_safety"]["passed"])
        net_gain = float(decision["paired_effect"]["net_gain"])
        baseline_r5 = float(decision["baseline"]["r5_percent"])
        candidate_r5 = float(decision["candidate"]["r5_percent"])
        if (
            classification in {"DEFAULT_CANDIDATE", "PARETO_CANDIDATE"}
            and safe
            and net_gain > 0.0
            and candidate_r5 >= baseline_r5
        ):
            alpha = float(match.group(1).replace("p", "."))
            priority = 0 if classification == "DEFAULT_CANDIDATE" else 1
            eligible.append((priority, alpha, str(arm)))
    if not eligible:
        return None
    return min(eligible)[2]


def _balanced_subset(
    *,
    keypoints: torch.Tensor,
    eligible: torch.Tensor,
    priority: torch.Tensor,
    image_hw: torch.Tensor,
    limit: int,
    grid: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    """Select hard rows without allowing one image region to dominate."""

    rows = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if rows.numel() <= int(limit):
        return rows
    height, width = map(int, torch.as_tensor(image_hw).tolist())
    gx, gy = map(int, grid)
    xy = torch.as_tensor(keypoints).float()[rows]
    cell_x = (xy[:, 0] * gx / max(width, 1)).floor().long().clamp(0, gx - 1)
    cell_y = (xy[:, 1] * gy / max(height, 1)).floor().long().clamp(0, gy - 1)
    cells = cell_y * gx + cell_x
    queues: list[list[int]] = []
    for cell in range(gx * gy):
        local = rows[cells == cell].tolist()
        local.sort(key=lambda row: (-float(priority[row]), int(row)))
        if local:
            queues.append(local)
    selected: list[int] = []
    offset = 0
    while len(selected) < int(limit):
        advanced = False
        for queue in queues:
            if offset < len(queue):
                selected.append(queue[offset])
                advanced = True
                if len(selected) == int(limit):
                    break
        if not advanced:
            break
        offset += 1
    return torch.tensor(selected, dtype=torch.long)


def _current_competition(
    query: Mapping,
    active_anchor_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    candidates = torch.as_tensor(query["candidate_anchor_rows"]).long()
    scores = torch.as_tensor(query["candidate_scores"]).float()
    positive = torch.as_tensor(query["certified_positive"]).bool()
    active = torch.as_tensor(active_anchor_mask).bool()[candidates]
    available = active.any(1)
    winner_column = scores.masked_fill(~active, -torch.inf).argmax(1)
    rows = torch.arange(candidates.shape[0])
    winner_anchor = candidates[rows, winner_column]
    winner_score = scores[rows, winner_column]
    winner_positive = positive[rows, winner_column] & available

    active_positive = active & positive
    best_positive_column = scores.masked_fill(~active_positive, -torch.inf).argmax(1)
    best_positive_anchor = candidates[rows, best_positive_column]
    best_positive_score = scores[rows, best_positive_column]
    has_positive = active_positive.any(1)

    active_nonpositive = active & ~positive
    best_nonpositive_column = scores.masked_fill(
        ~active_nonpositive, -torch.inf
    ).argmax(1)
    best_nonpositive_anchor = candidates[rows, best_nonpositive_column]
    best_nonpositive_score = scores[rows, best_nonpositive_column]
    has_nonpositive = active_nonpositive.any(1)
    return {
        "available": available,
        "winner_anchor": winner_anchor,
        "winner_score": winner_score,
        "winner_positive": winner_positive,
        "best_positive_anchor": best_positive_anchor,
        "best_positive_score": best_positive_score,
        "has_positive": has_positive,
        "best_nonpositive_anchor": best_nonpositive_anchor,
        "best_nonpositive_score": best_nonpositive_score,
        "has_nonpositive": has_nonpositive,
    }


def build_competitive_metric_evidence(
    *,
    competition_queries: Sequence[Mapping],
    query_descriptors: Mapping[int, torch.Tensor],
    action_metadata: Mapping[int, Mapping],
    active_anchor_mask: torch.Tensor,
    minimum_negative_pose_families: int = 2,
    maximum_repair_rows_per_query: int = 256,
    maximum_protection_rows_per_query: int = 256,
) -> dict:
    """Build causal repair and correct-winner protection pairs.

    Repair is authorized only for a query with positive measured task response
    and for a wrong winning Anchor repeated in at least two pose families.
    Geometry-certified alternatives are never copied into the map: they only
    define the desired ordering for one bounded query/map-shared transform.
    """

    if minimum_negative_pose_families < 2:
        raise ValueError("competitive repair requires at least two pose families")
    if maximum_repair_rows_per_query < 1 or maximum_protection_rows_per_query < 1:
        raise ValueError("per-query metric evidence limits must be positive")
    active = torch.as_tensor(active_anchor_mask).bool().reshape(-1)
    states: dict[int, dict[str, torch.Tensor]] = {}
    negative_families: dict[int, set[int]] = defaultdict(set)
    for query in competition_queries:
        query_index = int(query["query_index"])
        state = _current_competition(query, active)
        states[query_index] = state
        metadata = action_metadata[query_index]
        causal = bool(metadata["can_train_metric"]) and float(
            metadata["actual_task_gain"]
        ) > 0.0
        if not causal:
            continue
        repair = (
            state["available"]
            & ~state["winner_positive"]
            & state["has_positive"]
        )
        family = int(query["pose_family_id"])
        for anchor in torch.unique(state["winner_anchor"][repair]).tolist():
            negative_families[int(anchor)].add(family)

    repair_queries = []
    repair_positive = []
    repair_negative = []
    repair_weights = []
    protection_queries = []
    protection_positive = []
    protection_negative = []
    protection_margin = []
    protection_weights = []
    repair_families: set[int] = set()
    protection_families: set[int] = set()
    per_query = []
    for query in competition_queries:
        query_index = int(query["query_index"])
        family = int(query["pose_family_id"])
        descriptors = torch.as_tensor(query_descriptors[query_index]).float()
        if descriptors.shape[0] != torch.as_tensor(
            query["candidate_anchor_rows"]
        ).shape[0]:
            raise ValueError("query descriptors do not align with competitive rows")
        state = states[query_index]
        metadata = action_metadata[query_index]

        causal = bool(metadata["can_train_metric"]) and float(
            metadata["actual_task_gain"]
        ) > 0.0
        authorized_negative = torch.tensor(
            [
                len(negative_families[int(anchor)])
                >= int(minimum_negative_pose_families)
                for anchor in state["winner_anchor"].tolist()
            ],
            dtype=torch.bool,
        )
        repair = (
            state["available"]
            & ~state["winner_positive"]
            & state["has_positive"]
            & authorized_negative
            & causal
        )
        repair_deficit = state["winner_score"] - state["best_positive_score"]
        selected_repair = _balanced_subset(
            keypoints=query["keypoints"],
            eligible=repair,
            priority=-repair_deficit,
            image_hw=query["image_hw"],
            limit=maximum_repair_rows_per_query,
        )
        if selected_repair.numel():
            gain = min(max(float(metadata["actual_task_gain"]), 0.01), 4.0)
            repair_queries.append(descriptors[selected_repair])
            repair_positive.append(state["best_positive_anchor"][selected_repair])
            repair_negative.append(state["winner_anchor"][selected_repair])
            repair_weights.append(
                torch.full(
                    (selected_repair.numel(),), gain / selected_repair.numel()
                )
            )
            repair_families.add(family)

        protect = (
            state["available"]
            & state["winner_positive"]
            & state["has_nonpositive"]
        )
        initial_margin = state["winner_score"] - state["best_nonpositive_score"]
        selected_protect = _balanced_subset(
            keypoints=query["keypoints"],
            eligible=protect,
            priority=-initial_margin,
            image_hw=query["image_hw"],
            limit=maximum_protection_rows_per_query,
        )
        if selected_protect.numel():
            protection_queries.append(descriptors[selected_protect])
            protection_positive.append(state["winner_anchor"][selected_protect])
            protection_negative.append(
                state["best_nonpositive_anchor"][selected_protect]
            )
            protection_margin.append(initial_margin[selected_protect])
            protection_weights.append(
                torch.full(
                    (selected_protect.numel(),), 1.0 / selected_protect.numel()
                )
            )
            protection_families.add(family)
        per_query.append(
            {
                "query_index": query_index,
                "pose_family_id": family,
                "repair_row_count": int(selected_repair.numel()),
                "protection_row_count": int(selected_protect.numel()),
            }
        )

    if not repair_queries:
        raise RuntimeError("no active competitive repair pair passed causal gates")
    descriptor_dim = int(repair_queries[0].shape[1])

    def cat(parts: list[torch.Tensor], *, dtype: torch.dtype) -> torch.Tensor:
        if parts:
            return torch.cat(parts).to(dtype=dtype)
        shape = (0, descriptor_dim) if dtype.is_floating_point else (0,)
        return torch.empty(shape, dtype=dtype)

    return {
        "schema": "lafgs_v17_active_competitive_metric_evidence",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
        "repair_query_descriptors": cat(repair_queries, dtype=torch.float32),
        "repair_positive_anchor_rows": cat(repair_positive, dtype=torch.long),
        "repair_negative_anchor_rows": cat(repair_negative, dtype=torch.long),
        "repair_sample_weights": cat(repair_weights, dtype=torch.float32),
        "protection_query_descriptors": cat(
            protection_queries, dtype=torch.float32
        ),
        "protection_positive_anchor_rows": cat(
            protection_positive, dtype=torch.long
        ),
        "protection_negative_anchor_rows": cat(
            protection_negative, dtype=torch.long
        ),
        "protection_initial_margin": cat(protection_margin, dtype=torch.float32),
        "protection_sample_weights": cat(
            protection_weights, dtype=torch.float32
        ),
        "minimum_negative_pose_families": int(minimum_negative_pose_families),
        "repair_pose_family_count": len(repair_families),
        "protection_pose_family_count": len(protection_families),
        "authorized_negative_anchor_count": sum(
            len(families) >= int(minimum_negative_pose_families)
            for families in negative_families.values()
        ),
        "per_query": per_query,
    }
