"""Truth-aligned evidence for the bounded shared descriptor controller."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

from map_learning.v18_provenance_truth import (
    TRUTH_EQUIVALENT,
    TRUTH_UNIQUE,
    truth_membership_mask,
)


def _spatially_balanced(
    keypoints: torch.Tensor,
    rows: torch.Tensor,
    priority: torch.Tensor,
    image_hw: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    rows = torch.as_tensor(rows).long().reshape(-1)
    if rows.numel() <= int(limit):
        return rows
    height, width = map(int, torch.as_tensor(image_hw).tolist())
    xy = torch.as_tensor(keypoints).float()[rows]
    x = (xy[:, 0] * 4 / max(width, 1)).floor().long().clamp(0, 3)
    y = (xy[:, 1] * 4 / max(height, 1)).floor().long().clamp(0, 3)
    cells = y * 4 + x
    queues = []
    for cell in range(16):
        local = rows[cells == cell].tolist()
        local.sort(key=lambda row: (-float(priority[row]), int(row)))
        if local:
            queues.append(local)
    selected = []
    cursor = 0
    while len(selected) < int(limit):
        advanced = False
        for queue in queues:
            if cursor < len(queue):
                selected.append(queue[cursor])
                advanced = True
                if len(selected) >= int(limit):
                    break
        if not advanced:
            break
        cursor += 1
    return torch.tensor(selected, dtype=torch.long)


def _family_equal_weights(
    sample_counts: list[int],
    sample_gains: list[torch.Tensor],
    family_ids: list[int],
) -> list[torch.Tensor]:
    """Give each Query unit mass within a family and each family equal mass."""

    if not sample_counts:
        return []
    family_queries: dict[int, int] = defaultdict(int)
    for count, family in zip(sample_counts, family_ids):
        if count:
            family_queries[int(family)] += 1
    family_count = len(family_queries)
    output = []
    for count, gain, family in zip(sample_counts, sample_gains, family_ids):
        if not count:
            output.append(torch.empty(0))
            continue
        local = torch.as_tensor(gain).float().clamp_min(1e-6)
        local = local / local.sum().clamp_min(1e-12)
        local = local / max(family_queries[int(family)], 1) / max(family_count, 1)
        output.append(local)
    return output


def build_truth_aligned_metric_evidence(
    *,
    responsibility_records: Sequence[Mapping],
    query_descriptors: Mapping[int, torch.Tensor],
    query_keypoints: Mapping[int, torch.Tensor],
    query_image_hw: Mapping[int, torch.Tensor],
    active_anchor_mask: torch.Tensor,
    minimum_repair_pose_families: int = 2,
    maximum_repair_rows_per_query: int = 256,
    maximum_protection_rows_per_query: int = 256,
    minimum_replace_task_gain: float = 0.01,
) -> dict:
    """Build repair/protection triples from provenance truth and exact response.

    Repair requires a positive single-row replace counterfactual and the same
    wrong Anchor recurring in at least two pose families.  Sample weights are
    normalized first within Query and then within pose family, preventing one
    dense render or trajectory segment from dominating the shared transform.
    """

    if int(minimum_repair_pose_families) < 2:
        raise ValueError("metric repair requires at least two pose families")
    active = torch.as_tensor(active_anchor_mask).bool().reshape(-1)
    negative_families: dict[int, set[int]] = defaultdict(set)
    counterfactuals: dict[int, dict[int, Mapping]] = {}
    for record in responsibility_records:
        query = int(record["query_index"])
        family = int(record["pose_family_id"])
        per_row = {
            int(item["query_row"]): item
            for item in record["responsibility"]["row_counterfactuals"]
        }
        counterfactuals[query] = per_row
        for item in per_row.values():
            if float(item["replace_with_truth_task_gain"]) >= float(
                minimum_replace_task_gain
            ):
                negative_families[int(item["wrong_anchor_row"])].add(family)

    repair_query = []
    repair_positive = []
    repair_negative = []
    repair_gain = []
    protection_query = []
    protection_positive = []
    protection_negative = []
    protection_margin = []
    repair_counts = []
    repair_families = []
    protection_counts = []
    protection_families = []
    per_query = []
    for record in responsibility_records:
        query = int(record["query_index"])
        family = int(record["pose_family_id"])
        descriptors = torch.as_tensor(query_descriptors[query]).float()
        keypoints = torch.as_tensor(query_keypoints[query]).float()
        candidates = torch.as_tensor(record["candidate_anchor_rows"]).long()
        scores = torch.as_tensor(record["candidate_scores"]).float()
        truth = record["truth"]
        if descriptors.shape[0] != candidates.shape[0] or scores.shape != candidates.shape:
            raise ValueError("V18 metric competition rows do not align")
        membership = truth_membership_mask(truth, candidates)
        status = torch.as_tensor(truth["truth_status"]).long()
        decisive = (status == TRUTH_UNIQUE) | (status == TRUTH_EQUIVALENT)
        candidate_active = active[candidates]
        winner_active = candidate_active[:, 0]
        winner_correct = decisive & winner_active & membership[:, 0]
        per_row = counterfactuals[query]
        eligible_repair = []
        row_gain = torch.zeros(candidates.shape[0])
        for row, item in per_row.items():
            wrong_anchor = int(item["wrong_anchor_row"])
            truth_anchor = int(item["truth_anchor_row"])
            gain = float(item["replace_with_truth_task_gain"])
            if (
                gain >= float(minimum_replace_task_gain)
                and len(negative_families[wrong_anchor])
                >= int(minimum_repair_pose_families)
                and bool(active[wrong_anchor])
                and bool(active[truth_anchor])
            ):
                eligible_repair.append(row)
                row_gain[row] = gain
        repair_rows = _spatially_balanced(
            keypoints,
            torch.tensor(eligible_repair, dtype=torch.long),
            row_gain,
            query_image_hw[query],
            int(maximum_repair_rows_per_query),
        )
        if repair_rows.numel():
            items = [per_row[int(row)] for row in repair_rows.tolist()]
            repair_query.append(descriptors[repair_rows])
            repair_positive.append(
                torch.tensor([item["truth_anchor_row"] for item in items]).long()
            )
            repair_negative.append(
                torch.tensor([item["wrong_anchor_row"] for item in items]).long()
            )
            repair_gain.append(
                torch.tensor(
                    [item["replace_with_truth_task_gain"] for item in items]
                ).float()
            )
        else:
            repair_gain.append(torch.empty(0))
        repair_counts.append(int(repair_rows.numel()))
        repair_families.append(family)

        protection_rows = torch.nonzero(winner_correct, as_tuple=False).reshape(-1)
        nontruth = candidate_active & ~membership
        strongest_negative_column = scores.masked_fill(~nontruth, -torch.inf).argmax(1)
        row_index = torch.arange(candidates.shape[0])
        negative_score = scores[row_index, strongest_negative_column]
        initial_margin = scores[:, 0] - negative_score
        protection_rows = protection_rows[nontruth[protection_rows].any(1)]
        protection_rows = _spatially_balanced(
            keypoints,
            protection_rows,
            -initial_margin,
            query_image_hw[query],
            int(maximum_protection_rows_per_query),
        )
        if protection_rows.numel():
            protection_query.append(descriptors[protection_rows])
            protection_positive.append(candidates[protection_rows, 0])
            protection_negative.append(
                candidates[
                    protection_rows,
                    strongest_negative_column[protection_rows],
                ]
            )
            protection_margin.append(initial_margin[protection_rows])
        protection_counts.append(int(protection_rows.numel()))
        protection_families.append(family)
        per_query.append(
            {
                "query_index": query,
                "pose_family_id": family,
                "repair_row_count": int(repair_rows.numel()),
                "protection_row_count": int(protection_rows.numel()),
            }
        )

    if not repair_query:
        raise RuntimeError("no truth-aligned metric repair survived causal gates")
    repair_weights = _family_equal_weights(
        repair_counts, repair_gain, repair_families
    )
    protection_gains = [torch.ones(count) for count in protection_counts]
    protection_weights = _family_equal_weights(
        protection_counts, protection_gains, protection_families
    )
    descriptor_dim = int(repair_query[0].shape[1])

    def cat(parts: list[torch.Tensor], shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.cat(parts).to(dtype) if parts else torch.empty(shape, dtype=dtype)

    return {
        "schema": "lafgs_v18_truth_aligned_competitive_metric_evidence",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "feedback_descriptors_copied_into_map": False,
        "repair_query_descriptors": cat(repair_query, (0, descriptor_dim), torch.float32),
        "repair_positive_anchor_rows": cat(repair_positive, (0,), torch.long),
        "repair_negative_anchor_rows": cat(repair_negative, (0,), torch.long),
        "repair_sample_weights": cat(repair_weights, (0,), torch.float32),
        "protection_query_descriptors": cat(
            protection_query, (0, descriptor_dim), torch.float32
        ),
        "protection_positive_anchor_rows": cat(protection_positive, (0,), torch.long),
        "protection_negative_anchor_rows": cat(protection_negative, (0,), torch.long),
        "protection_initial_margin": cat(protection_margin, (0,), torch.float32),
        "protection_sample_weights": cat(protection_weights, (0,), torch.float32),
        "repair_pose_family_count": len(
            {family for count, family in zip(repair_counts, repair_families) if count}
        ),
        "protection_pose_family_count": len(
            {
                family
                for count, family in zip(protection_counts, protection_families)
                if count
            }
        ),
        "authorized_negative_anchor_count": sum(
            len(families) >= int(minimum_repair_pose_families)
            for families in negative_families.values()
        ),
        "weighting_policy": "equal_pose_family_equal_query_then_counterfactual_gain",
        "per_query": per_query,
    }


__all__ = ["build_truth_aligned_metric_evidence"]
