"""Actionable one-descriptor-per-Anchor feedback controller for V10."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math

import torch
import torch.nn.functional as F


def aggregate_descriptor_action_gain(
    records: Sequence[Mapping],
    *,
    minimum_affected_pose_families: int = 2,
    minimum_improving_queries: int = 2,
    minimum_median_actual_task_gain: float = 0.001,
    maximum_worsening_fraction: float = 0.20,
) -> dict:
    """Authorize descriptor updates only from exact matching+PoseLib replays."""

    grouped: dict[int, list[Mapping]] = defaultdict(list)
    for record in records:
        if record.get("loo_used") is not False:
            raise ValueError("V10 descriptor replay cannot use LOO evidence")
        grouped[int(record["anchor_row"])].append(record)
    audit = []
    for anchor, rows in grouped.items():
        gains = torch.tensor(
            [
                float(row["baseline_task_error"])
                - float(row["updated_task_error"])
                for row in rows
            ]
        )
        families = {int(row["pose_family_id"]) for row in rows}
        improving = gains > 0
        worsening_fraction = float((gains < 0).float().mean())
        median_gain = float(gains.median())
        cumulative_gain = float(gains.sum())
        authorized = bool(
            len(families) >= int(minimum_affected_pose_families)
            and int(improving.sum()) >= int(minimum_improving_queries)
            and median_gain >= float(minimum_median_actual_task_gain)
            and cumulative_gain > 0
            and worsening_fraction <= float(maximum_worsening_fraction)
        )
        audit.append(
            {
                "anchor_row": anchor,
                "affected_query_count": len(rows),
                "affected_pose_family_count": len(families),
                "improving_query_count": int(improving.sum()),
                "median_actual_task_gain": median_gain,
                "cumulative_actual_task_gain": cumulative_gain,
                "worsening_fraction": worsening_fraction,
                "authorized": authorized,
            }
        )
    audit.sort(
        key=lambda row: (-row["cumulative_actual_task_gain"], row["anchor_row"])
    )
    authorized = torch.tensor(
        [row["anchor_row"] for row in audit if row["authorized"]], dtype=torch.long
    )
    return {
        "schema": "lafgs_v10_single_descriptor_action_gain_audit",
        "version": 1,
        "loo_used": False,
        "authorized_anchor_rows": authorized,
        "authorized_anchor_count": int(authorized.numel()),
        "candidate_audit": audit,
    }


def build_confusion_component_groups(
    *,
    candidate_anchor_rows: torch.Tensor,
    feedback_records: Sequence[Mapping],
    maximum_group_size: int = 8,
) -> list[list[int]]:
    """Build disjoint bounded groups from query-level candidate co-occurrence."""

    candidates = set(torch.as_tensor(candidate_anchor_rows).long().tolist())
    adjacency: dict[int, dict[int, int]] = {
        anchor: defaultdict(int) for anchor in candidates
    }
    for record in feedback_records:
        if record.get("loo_used") is not False:
            raise ValueError("V10 confusion groups cannot use LOO evidence")
        if not record.get("can_train_metric", False):
            continue
        rows = sorted(
            set(
                torch.as_tensor(
                    record["training_evidence"]["positive_anchor_rows"]
                ).long().tolist()
            )
            & candidates
        )
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1 :]:
                adjacency[left][right] += 1
                adjacency[right][left] += 1
    remaining = set(candidates)
    groups = []
    while remaining:
        seed = max(
            remaining,
            key=lambda anchor: (
                sum(adjacency[anchor].get(other, 0) for other in remaining),
                -anchor,
            ),
        )
        group = [seed]
        remaining.remove(seed)
        while remaining and len(group) < int(maximum_group_size):
            scored = [
                (
                    sum(adjacency[candidate].get(member, 0) for member in group),
                    candidate,
                )
                for candidate in remaining
            ]
            connection, candidate = max(scored, key=lambda item: (item[0], -item[1]))
            if connection <= 0:
                break
            group.append(candidate)
            remaining.remove(candidate)
        groups.append(sorted(group))
    groups.sort(key=lambda group: (-len(group), group))
    return groups


def aggregate_group_descriptor_action_gain(
    records: Sequence[Mapping],
    *,
    minimum_affected_pose_families: int = 2,
    minimum_improving_queries: int = 2,
    minimum_median_actual_task_gain: float = 0.001,
    maximum_worsening_fraction: float = 0.20,
) -> dict:
    """Authorize disjoint descriptor groups using exact group-level pose gain."""

    grouped: dict[int, list[Mapping]] = defaultdict(list)
    for record in records:
        if record.get("loo_used") is not False:
            raise ValueError("V10 group replay cannot use LOO evidence")
        grouped[int(record["group_index"])].append(record)
    audit = []
    for group_index, rows in grouped.items():
        gains = torch.tensor(
            [
                float(row["baseline_task_error"])
                - float(row["updated_task_error"])
                for row in rows
            ]
        )
        families = {int(row["pose_family_id"]) for row in rows}
        worsening_fraction = float((gains < 0).float().mean())
        item = {
            "group_index": group_index,
            "affected_query_count": len(rows),
            "affected_pose_family_count": len(families),
            "improving_query_count": int((gains > 0).sum()),
            "median_actual_task_gain": float(gains.median()),
            "cumulative_actual_task_gain": float(gains.sum()),
            "worsening_fraction": worsening_fraction,
        }
        item["authorized"] = bool(
            item["affected_pose_family_count"]
            >= int(minimum_affected_pose_families)
            and item["improving_query_count"] >= int(minimum_improving_queries)
            and item["median_actual_task_gain"]
            >= float(minimum_median_actual_task_gain)
            and item["cumulative_actual_task_gain"] > 0
            and worsening_fraction <= float(maximum_worsening_fraction)
        )
        audit.append(item)
    audit.sort(key=lambda row: row["group_index"])
    authorized = torch.tensor(
        [row["group_index"] for row in audit if row["authorized"]], dtype=torch.long
    )
    return {
        "schema": "lafgs_v10_group_descriptor_action_gain_audit",
        "version": 1,
        "loo_used": False,
        "authorized_group_indices": authorized,
        "authorized_group_count": int(authorized.numel()),
        "group_audit": audit,
    }


def _spherical_step(
    source: torch.Tensor, target: torch.Tensor, maximum_angle_deg: float
) -> tuple[torch.Tensor, float]:
    source = F.normalize(torch.as_tensor(source).float(), dim=0)
    target = F.normalize(torch.as_tensor(target).float(), dim=0)
    cosine = torch.dot(source, target).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    angle_deg = math.degrees(float(angle))
    if angle_deg <= float(maximum_angle_deg) or angle_deg < 1e-8:
        return target, angle_deg
    fraction = math.radians(float(maximum_angle_deg)) / float(angle)
    sine = torch.sin(angle).clamp_min(1e-8)
    output = (
        torch.sin((1.0 - fraction) * angle) / sine * source
        + torch.sin(fraction * angle) / sine * target
    )
    return F.normalize(output, dim=0), float(maximum_angle_deg)


def _robust_family_mean(descriptors: torch.Tensor) -> torch.Tensor:
    values = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    if values.shape[0] <= 2:
        return F.normalize(values.mean(0), dim=0)
    similarity = values @ values.T
    medoid = int(torch.argmax(similarity.median(1).values))
    order = torch.argsort(similarity[medoid], descending=True, stable=True)
    keep = max(2, math.ceil(values.shape[0] * 0.8))
    return F.normalize(values[order[:keep]].mean(0), dim=0)


def propose_actionable_anchor_descriptors(
    *,
    anchor_features: torch.Tensor,
    feedback_records: Sequence[Mapping],
    maximum_positive_rank: int = 8,
    minimum_pose_families: int = 3,
    maximum_median_dispersion_deg: float = 8.0,
    maximum_p90_dispersion_deg: float = 12.0,
    minimum_predicted_flip_families: int = 2,
    minimum_predicted_flip_rows: int = 2,
    maximum_update_angle_deg: float = 5.0,
) -> dict:
    """Propose bounded per-Anchor updates from family-balanced feedback evidence."""

    anchors = F.normalize(torch.as_tensor(anchor_features).float(), dim=1)
    evidence: dict[int, dict[int, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in feedback_records:
        if record.get("loo_used") is not False:
            raise ValueError("V10 forbids LOO feedback evidence")
        if not record.get("can_train_metric", False):
            continue
        family = int(record["pose_family_id"])
        query = int(record["query_index"])
        item = record["training_evidence"]
        descriptors = torch.as_tensor(item["query_descriptors"]).float()
        positive = torch.as_tensor(item["positive_anchor_rows"]).long()
        rank = torch.as_tensor(item["positive_rank"]).long() + 1
        negative_scores = torch.as_tensor(item["negative_scores"]).float()
        if not (
            descriptors.shape[0]
            == positive.numel()
            == rank.numel()
            == negative_scores.numel()
        ):
            raise ValueError("V10 feedback training rows do not align")
        for row in torch.nonzero(
            rank <= int(maximum_positive_rank), as_tuple=False
        ).reshape(-1).tolist():
            anchor = int(positive[row])
            if not 0 <= anchor < anchors.shape[0]:
                raise ValueError("V10 positive Anchor is outside M0")
            evidence[anchor][family].append(
                {
                    "query": query,
                    "descriptor": descriptors[row],
                    "negative_score": float(negative_scores[row]),
                    "positive_rank": int(rank[row]),
                }
            )

    candidates = []
    rejected = defaultdict(int)
    for anchor, families in evidence.items():
        if len(families) < int(minimum_pose_families):
            rejected["insufficient_pose_families"] += 1
            continue
        family_descriptors = []
        for rows in families.values():
            family_descriptors.append(
                _robust_family_mean(
                    torch.stack([torch.as_tensor(row["descriptor"]) for row in rows])
                )
            )
        family_bank = torch.stack(family_descriptors)
        target = _robust_family_mean(family_bank)
        dispersion = torch.rad2deg(
            torch.acos((family_bank @ target).clamp(-1.0, 1.0))
        )
        median_dispersion = float(dispersion.median())
        p90_dispersion = float(torch.quantile(dispersion, 0.9))
        if (
            median_dispersion > float(maximum_median_dispersion_deg)
            or p90_dispersion > float(maximum_p90_dispersion_deg)
        ):
            rejected["descriptor_inconsistent"] += 1
            continue
        descriptor, update_angle = _spherical_step(
            anchors[anchor], target, maximum_update_angle_deg
        )
        flipped_families = set()
        flipped_rows = 0
        supporting_rows = 0
        original_scores = []
        updated_scores = []
        for family, rows in families.items():
            for row in rows:
                query_descriptor = F.normalize(
                    torch.as_tensor(row["descriptor"]).float(), dim=0
                )
                old_score = float(torch.dot(query_descriptor, anchors[anchor]))
                new_score = float(torch.dot(query_descriptor, descriptor))
                original_scores.append(old_score)
                updated_scores.append(new_score)
                supporting_rows += 1
                if new_score > float(row["negative_score"]):
                    flipped_rows += 1
                    flipped_families.add(family)
        if (
            len(flipped_families) < int(minimum_predicted_flip_families)
            or flipped_rows < int(minimum_predicted_flip_rows)
        ):
            rejected["bounded_update_not_actionable"] += 1
            continue
        candidates.append(
            {
                "anchor_row": anchor,
                "descriptor": descriptor,
                "pose_family_count": len(families),
                "supporting_row_count": supporting_rows,
                "predicted_flip_family_count": len(flipped_families),
                "predicted_flip_row_count": flipped_rows,
                "median_family_dispersion_deg": median_dispersion,
                "p90_family_dispersion_deg": p90_dispersion,
                "update_angle_deg": update_angle,
                "mean_original_support_score": float(
                    torch.tensor(original_scores).mean()
                ),
                "mean_updated_support_score": float(
                    torch.tensor(updated_scores).mean()
                ),
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["predicted_flip_family_count"],
            -row["predicted_flip_row_count"],
            row["median_family_dispersion_deg"],
            row["anchor_row"],
        )
    )
    rows = torch.tensor([item["anchor_row"] for item in candidates], dtype=torch.long)
    descriptors = (
        torch.stack([item["descriptor"] for item in candidates])
        if candidates
        else torch.empty(0, anchors.shape[1])
    )
    return {
        "schema": "lafgs_v10_actionable_anchor_descriptor_proposal",
        "version": 1,
        "loo_used": False,
        "geometry_mutation_count": 0,
        "anchor_addition_count": 0,
        "descriptor_count_per_anchor": 1,
        "feedback_descriptor_exact_copy": False,
        "feedback_descriptor_robust_aggregate": True,
        "candidate_anchor_rows": rows,
        "candidate_descriptors": descriptors,
        "candidate_count": int(rows.numel()),
        "candidate_audit": [
            {key: value for key, value in item.items() if key != "descriptor"}
            for item in candidates
        ],
        "rejection_counts": dict(rejected),
        "contract": {
            "maximum_positive_rank": int(maximum_positive_rank),
            "minimum_pose_families": int(minimum_pose_families),
            "maximum_median_dispersion_deg": float(maximum_median_dispersion_deg),
            "maximum_p90_dispersion_deg": float(maximum_p90_dispersion_deg),
            "minimum_predicted_flip_families": int(minimum_predicted_flip_families),
            "minimum_predicted_flip_rows": int(minimum_predicted_flip_rows),
            "maximum_update_angle_deg": float(maximum_update_angle_deg),
        },
    }
