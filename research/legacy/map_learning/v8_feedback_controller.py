"""Strict map-side feedback actions for the V2 projective mainline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math

import torch


def task_error(
    translation_error_cm: float,
    rotation_error_deg: float,
    *,
    translation_scale_cm: float = 5.0,
    rotation_scale_deg: float = 5.0,
) -> float:
    """Return the frozen continuous localization objective."""

    return math.hypot(
        float(translation_error_cm) / float(translation_scale_cm),
        float(rotation_error_deg) / float(rotation_scale_deg),
    )


def propose_feedback_anchor_quarantine(
    *,
    anchor_ids: torch.Tensor,
    feedback_records: Sequence[Mapping],
    minimum_pose_families: int = 2,
    minimum_queries: int = 2,
    minimum_query_task_gain: float = 0.01,
    maximum_quarantine_fraction: float = 0.01,
) -> dict:
    """Aggregate strict false-attractor evidence into a reversible proposal.

    Evidence is admitted only from an update-authorized precision replay.  An
    Anchor is excluded if it is ever the positive alternative in the same
    feedback batch.  The proposal is additionally bounded by a small active-set
    change, and does not mutate the input map.
    """

    ids = torch.as_tensor(anchor_ids).long().reshape(-1).cpu()
    if ids.numel() == 0 or torch.unique(ids).numel() != ids.numel():
        raise ValueError("feedback controller requires unique non-empty Anchor IDs")
    if not 0.0 <= float(maximum_quarantine_fraction) <= 0.03:
        raise ValueError("feedback quarantine must remain within the 3% safety cap")
    known = set(ids.tolist())
    positive_ids: set[int] = set()
    evidence: dict[int, dict[str, object]] = defaultdict(
        lambda: {"families": set(), "queries": set(), "task_gain": 0.0}
    )
    authorized_queries = 0
    for record in feedback_records:
        diagnosis = record["diagnosis"]
        if diagnosis.get("can_drive_map_update") is not True:
            continue
        if diagnosis.get("category") != "precision_deficit":
            continue
        precision = diagnosis["precision_diagnostic"]
        current = task_error(
            diagnosis["translation_error_cm"], diagnosis["rotation_error_deg"]
        )
        alternative = task_error(
            precision["alternative_translation_error_cm"],
            precision["alternative_rotation_error_deg"],
        )
        gain = current - alternative
        if not math.isfinite(gain) or gain < float(minimum_query_task_gain):
            continue
        control = diagnosis["descriptor_control_evidence"]
        positives = torch.as_tensor(control["positive_anchor_ids"]).long().tolist()
        harmful = torch.as_tensor(
            control["false_attractor_anchor_ids"]
        ).long().tolist()
        if len(positives) != len(harmful):
            raise ValueError("positive and false-attractor evidence must align")
        family = int(record["pose_family_id"])
        query = int(record["query_index"])
        authorized_queries += 1
        positive_ids.update(map(int, positives))
        for anchor_id in set(map(int, harmful)):
            if anchor_id not in known:
                raise ValueError("feedback false-attractor is outside the active map")
            item = evidence[anchor_id]
            item["families"].add(family)
            item["queries"].add(query)
            item["task_gain"] = float(item["task_gain"]) + gain

    candidates = []
    for anchor_id, item in evidence.items():
        families = item["families"]
        queries = item["queries"]
        if (
            anchor_id not in positive_ids
            and len(families) >= int(minimum_pose_families)
            and len(queries) >= int(minimum_queries)
        ):
            candidates.append(
                {
                    "anchor_id": anchor_id,
                    "pose_family_count": len(families),
                    "query_count": len(queries),
                    "cumulative_task_gain": float(item["task_gain"]),
                }
            )
    candidates.sort(
        key=lambda item: (
            -item["cumulative_task_gain"],
            -item["pose_family_count"],
            item["anchor_id"],
        )
    )
    maximum = int(math.floor(ids.numel() * float(maximum_quarantine_fraction)))
    accepted = candidates[:maximum]
    accepted_ids = torch.tensor(
        [item["anchor_id"] for item in accepted], dtype=torch.long
    )
    id_to_row = {int(anchor_id): row for row, anchor_id in enumerate(ids.tolist())}
    accepted_rows = torch.tensor(
        sorted(id_to_row[int(anchor_id)] for anchor_id in accepted_ids.tolist()),
        dtype=torch.long,
    )
    return {
        "schema": "lafgs_v8_feedback_anchor_quarantine_proposal",
        "version": 1,
        "proposed_anchor_ids": accepted_ids,
        "proposed_anchor_rows": accepted_rows,
        "proposed_anchor_count": int(accepted_ids.numel()),
        "candidate_count_before_cap": len(candidates),
        "authorized_precision_query_count": authorized_queries,
        "positive_protection_anchor_count": len(positive_ids),
        "maximum_quarantine_fraction": float(maximum_quarantine_fraction),
        "actual_quarantine_fraction": float(accepted_ids.numel() / ids.numel()),
        "minimum_pose_families": int(minimum_pose_families),
        "minimum_queries": int(minimum_queries),
        "minimum_query_task_gain": float(minimum_query_task_gain),
        "candidate_audit": candidates,
        "reversible": True,
        "map_mutation_count": 0,
        "feedback_descriptors_copied": False,
    }


def materialize_quarantined_map(
    state: Mapping,
    proposed_anchor_rows: torch.Tensor,
) -> tuple[dict, torch.Tensor]:
    """Materialize a reversible active subset while preserving Anchor IDs."""

    count = int(torch.as_tensor(state["anchor_ids"]).numel())
    quarantined = torch.as_tensor(proposed_anchor_rows).long().reshape(-1).cpu()
    if quarantined.numel() and (
        int(quarantined.min()) < 0 or int(quarantined.max()) >= count
    ):
        raise ValueError("quarantined Anchor row is out of range")
    if not torch.equal(quarantined, torch.unique(quarantined, sorted=True)):
        raise ValueError("quarantined Anchor rows must be unique and sorted")
    active = torch.ones(count, dtype=torch.bool)
    active[quarantined] = False
    selected = torch.nonzero(active, as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        raise ValueError("feedback action cannot empty the map")
    output = {}
    for key, value in state.items():
        if key == "projective_anchor_observations":
            continue
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == count:
            output[key] = value[selected].clone()
        elif isinstance(value, list) and len(value) == count:
            output[key] = [value[row] for row in selected.tolist()]
        else:
            output[key] = value
    source = state["projective_anchor_observations"]
    offsets = torch.as_tensor(source["observation_offsets"]).long()
    queries = torch.as_tensor(source["query_indices"]).long()
    keypoints = torch.as_tensor(source["keypoint_indices"]).long()
    query_parts, keypoint_parts, output_offsets = [], [], [0]
    for row in selected.tolist():
        start, stop = int(offsets[row]), int(offsets[row + 1])
        query_parts.append(queries[start:stop])
        keypoint_parts.append(keypoints[start:stop])
        output_offsets.append(output_offsets[-1] + stop - start)
    output["projective_anchor_observations"] = {
        **{
            key: value
            for key, value in source.items()
            if key not in {"observation_offsets", "query_indices", "keypoint_indices"}
        },
        "observation_offsets": torch.tensor(output_offsets, dtype=torch.long),
        "query_indices": torch.cat(query_parts),
        "keypoint_indices": torch.cat(keypoint_parts),
    }
    output["canonical_anchor_count"] = int(selected.numel())
    output["micro_anchor_count"] = int(selected.numel())
    output["provenance"] = {
        **dict(state.get("provenance", {})),
        "v8_feedback_anchor_quarantine": True,
        "v8_feedback_quarantine_reversible": True,
        "v8_feedback_quarantined_source_anchor_ids": torch.as_tensor(
            state["anchor_ids"]
        )[quarantined].clone(),
        "feedback_descriptors_copied_into_map": False,
        "uses_test_queries": False,
    }
    return output, selected
