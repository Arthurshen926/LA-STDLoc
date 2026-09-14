"""Forward-safe identity calibration for V21 adaptation diagnostics.

The V21 correspondence teacher intentionally exposes Tier-C ``UNIQUE`` rows
as diagnostics, not as deployable map/metric supervision.  This module adds a
second, independent evidence gate over the *complete adaptation prefix*:

* the same frozen fine identity must recur in independent trajectory blocks;
* its frozen mapping Track must span independent mapping view families; and
* block-balanced native query descriptors must agree with a descriptor medoid.

Rows passing that gate are only ``provisional_action_positive`` observations.
They may be used to build a quarantined candidate after adaptation, but never
authorize deployment.  Control and confirmation remain held out, and no
unlabelled/ambiguous/wrong-Top1 Anchor is manufactured into a negative.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import itertools
import math
import os
from pathlib import Path
import re
import uuid

import torch
import torch.nn.functional as F

from common.hashing import canonical_json


SCHEMA = "lafgs_v21_adaptation_identity_calibration"
VERSION = 1
ROLE = "adaptation"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

SEMANTICS = {
    "source_truth": "tier_c_diagnostic_unique_only",
    "identity_key": "frozen_stable_map_fine_identity_id",
    "adaptation_independence_unit": "predeclared_trajectory_block",
    "mapping_independence_unit": "frozen_mapping_track_bank_view_family",
    "descriptor_evidence": "native_query_descriptor_block_centroid_medoid",
    "descriptor_balance": "one_l2_normalized_centroid_per_adaptation_block",
    "wrong_top1_definition": "winner_fine_identity_differs_from_positive_fine_identity",
    "existing_correct_top1_role": "preservation_positive_not_negative",
    "negative_labels_created": False,
    "candidate_scope": "quarantined_candidate_generation_only",
    "deployment_authorized": False,
}


def sha256_json(value: Mapping) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def _source_record(value: object, *, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"V21 identity calibration {label} source is missing")
    path = str(value.get("path", ""))
    digest = _require_sha256(value.get("sha256"), label=f"{label} SHA256")
    size = int(value.get("size_bytes", 0))
    if not path or size <= 0:
        raise ValueError(f"V21 identity calibration {label} source is empty")
    return {"path": path, "sha256": digest, "size_bytes": size}


def calibration_thresholds(
    *,
    minimum_adaptation_blocks: int,
    minimum_adaptation_sequences: int,
    minimum_mapping_observations: int,
    minimum_mapping_families: int,
    minimum_descriptor_medoid_cosine: float,
) -> dict:
    """Build and validate a JSON-stable provisional-positive gate."""

    values = {
        "minimum_adaptation_blocks": int(minimum_adaptation_blocks),
        "minimum_adaptation_sequences": int(minimum_adaptation_sequences),
        "minimum_mapping_observations": int(minimum_mapping_observations),
        "minimum_mapping_families": int(minimum_mapping_families),
        "minimum_descriptor_medoid_cosine": float(
            minimum_descriptor_medoid_cosine
        ),
    }
    if any(
        values[name] < 1
        for name in (
            "minimum_adaptation_blocks",
            "minimum_adaptation_sequences",
            "minimum_mapping_observations",
            "minimum_mapping_families",
        )
    ):
        raise ValueError("V21 identity calibration count thresholds must be positive")
    cosine = values["minimum_descriptor_medoid_cosine"]
    if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
        raise ValueError("V21 identity calibration cosine threshold is invalid")
    return values


def block_descriptor_medoid(
    descriptors: torch.Tensor,
    block_ids: Sequence[str],
) -> dict:
    """Return a block-balanced native descriptor medoid and agreement scores.

    Dense adjacent frames must not outvote an independent trajectory block.
    We therefore average normalized observations *within* each block first,
    normalize each block centroid, and choose the centroid with maximum mean
    cosine to the other block centroids.  Lexicographic block ordering makes
    exact ties deterministic.
    """

    values = torch.as_tensor(descriptors).float().cpu()
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("descriptor medoid input must be a non-empty [N,D] tensor")
    if len(block_ids) != values.shape[0] or not bool(torch.isfinite(values).all()):
        raise ValueError("descriptor medoid rows and block IDs do not align")
    if values.shape[1] <= 0:
        raise ValueError("descriptor medoid dimension must be positive")
    normalized = F.normalize(values, dim=1)
    by_block: dict[str, list[int]] = defaultdict(list)
    for row, block_id in enumerate(block_ids):
        block = str(block_id)
        if not block:
            raise ValueError("descriptor medoid block ID is empty")
        by_block[block].append(row)
    ordered_blocks = sorted(by_block)
    centroids = []
    for block in ordered_blocks:
        centroid = normalized[by_block[block]].mean(0)
        norm = float(torch.linalg.vector_norm(centroid))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError("descriptor block centroid has zero/invalid norm")
        centroids.append(centroid / norm)
    centroid_tensor = torch.stack(centroids).contiguous()
    similarities = centroid_tensor @ centroid_tensor.T
    medoid_index = int(similarities.mean(1).argmax())
    medoid = centroid_tensor[medoid_index].contiguous()
    edge_cosines = normalized @ medoid
    if len(ordered_blocks) == 1:
        minimum = -1.0
        median = -1.0
        pairwise_median = -1.0
        defined = False
    else:
        medoid_cosines = similarities[medoid_index]
        minimum = float(medoid_cosines.min())
        median = float(medoid_cosines.median())
        upper = similarities[
            torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
        ]
        pairwise_median = float(upper.median())
        defined = True
    return {
        "descriptor_medoid": medoid,
        "descriptor_medoid_block_id": ordered_blocks[medoid_index],
        "adaptation_block_ids": tuple(ordered_blocks),
        "adaptation_block_count": len(ordered_blocks),
        "descriptor_cross_block_defined": defined,
        "descriptor_medoid_min_cosine": minimum,
        "descriptor_medoid_median_cosine": median,
        "descriptor_pairwise_median_cosine": pairwise_median,
        "descriptor_edge_to_medoid_median_cosine": float(edge_cosines.median()),
    }


def mapping_identity_support(
    *,
    target_fine_identity_ids: torch.Tensor,
    anchor_fine_identity_ids: torch.Tensor,
    observation_offsets: torch.Tensor,
    observation_query_indices: torch.Tensor,
    provenance_observation_rows: torch.Tensor,
    provenance_observation_valid: torch.Tensor,
    mapping_view_family_ids: torch.Tensor,
    family_roles: Mapping[int, str],
) -> dict:
    """Aggregate frozen mapping support for selected fine identities.

    Action evidence uses only provenance-valid observations whose mapping
    family was assigned to the V19 ``track_bank``.  Counts over all valid
    mapping observations are retained for diagnosis but never drive the gate.
    """

    targets = torch.as_tensor(target_fine_identity_ids).long().cpu().reshape(-1)
    anchor_fine = torch.as_tensor(anchor_fine_identity_ids).long().cpu().reshape(-1)
    offsets = torch.as_tensor(observation_offsets).long().cpu().reshape(-1)
    observation_queries = (
        torch.as_tensor(observation_query_indices).long().cpu().reshape(-1)
    )
    provenance_rows = (
        torch.as_tensor(provenance_observation_rows).long().cpu().reshape(-1)
    )
    provenance_valid = (
        torch.as_tensor(provenance_observation_valid).bool().cpu().reshape(-1)
    )
    mapping_families = (
        torch.as_tensor(mapping_view_family_ids).long().cpu().reshape(-1)
    )
    if (
        targets.numel() == 0
        or not torch.equal(targets, torch.unique(targets, sorted=True))
        or offsets.shape != (anchor_fine.numel() + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != observation_queries.numel()
        or bool((offsets[1:] < offsets[:-1]).any())
        or provenance_rows.shape != provenance_valid.shape
        or provenance_rows.numel() != observation_queries.numel()
    ):
        raise ValueError("V21 mapping identity registries do not align")
    edge_count = int(observation_queries.numel())
    if edge_count and (
        int(observation_queries.min()) < 0
        or int(observation_queries.max()) >= mapping_families.numel()
        or int(provenance_rows.min()) != 0
        or int(provenance_rows.max()) != edge_count - 1
        or torch.unique(provenance_rows).numel() != edge_count
    ):
        raise ValueError("V21 mapping observation registry is not a permutation")
    normalized_roles = {int(key): str(value) for key, value in family_roles.items()}
    used_families = set(int(value) for value in mapping_families.tolist())
    if any(family not in normalized_roles for family in used_families):
        raise ValueError("V21 mapping family role registry is incomplete")
    if not any(normalized_roles[value] == "track_bank" for value in used_families):
        raise ValueError("V21 mapping Track bank is empty")

    global_valid = torch.zeros(edge_count, dtype=torch.bool)
    global_valid[provenance_rows] = provenance_valid
    track_bank_query = torch.tensor(
        [normalized_roles[int(value)] == "track_bank" for value in mapping_families],
        dtype=torch.bool,
    )
    target_to_row = {int(value): row for row, value in enumerate(targets.tolist())}
    track_observations = torch.zeros(targets.numel(), dtype=torch.long)
    all_observations = torch.zeros_like(track_observations)
    anchor_rows = torch.zeros_like(track_observations)
    track_families: list[set[int]] = [set() for _ in range(targets.numel())]
    all_families: list[set[int]] = [set() for _ in range(targets.numel())]
    for anchor_row, fine_identity in enumerate(anchor_fine.tolist()):
        target_row = target_to_row.get(int(fine_identity))
        if target_row is None:
            continue
        anchor_rows[target_row] += 1
        lower = int(offsets[anchor_row])
        upper = int(offsets[anchor_row + 1])
        queries = observation_queries[lower:upper]
        valid = global_valid[lower:upper]
        valid_families = mapping_families[queries[valid]]
        all_observations[target_row] += int(valid.sum())
        all_families[target_row].update(map(int, valid_families.tolist()))
        track_valid = valid & track_bank_query[queries]
        selected_families = mapping_families[queries[track_valid]]
        track_observations[target_row] += int(track_valid.sum())
        track_families[target_row].update(map(int, selected_families.tolist()))
    return {
        "fine_identity_ids": targets.contiguous(),
        "identity_anchor_row_count": anchor_rows.contiguous(),
        "mapping_track_bank_observation_count": track_observations.contiguous(),
        "mapping_track_bank_family_count": torch.tensor(
            [len(values) for values in track_families], dtype=torch.long
        ),
        "mapping_all_valid_observation_count": all_observations.contiguous(),
        "mapping_all_valid_family_count": torch.tensor(
            [len(values) for values in all_families], dtype=torch.long
        ),
    }


def build_identity_evidence(
    *,
    observation_fine_identity_ids: torch.Tensor,
    observation_anchor_rows: torch.Tensor,
    observation_descriptors: torch.Tensor,
    observation_query_indices: torch.Tensor,
    observation_block_ids: Sequence[str],
    observation_sequence_ids: Sequence[str],
    wrong_top1_anchor_row: torch.Tensor,
    wrong_top1_fine_identity: torch.Tensor,
    baseline_failure: torch.Tensor,
    baseline_inlier: torch.Tensor,
    truth_anchor_scores: torch.Tensor,
    winner_scores: torch.Tensor,
    mapping_support: Mapping,
) -> dict:
    """Build one deterministic evidence row per diagnostic fine identity."""

    fine_ids = torch.as_tensor(observation_fine_identity_ids).long().cpu().reshape(-1)
    anchor_rows = torch.as_tensor(observation_anchor_rows).long().cpu().reshape(-1)
    descriptors = torch.as_tensor(observation_descriptors).float().cpu()
    query_indices = torch.as_tensor(observation_query_indices).long().cpu().reshape(-1)
    wrong_anchor = torch.as_tensor(wrong_top1_anchor_row).bool().cpu().reshape(-1)
    wrong_fine = torch.as_tensor(wrong_top1_fine_identity).bool().cpu().reshape(-1)
    failures = torch.as_tensor(baseline_failure).bool().cpu().reshape(-1)
    inliers = torch.as_tensor(baseline_inlier).bool().cpu().reshape(-1)
    truth_scores = torch.as_tensor(truth_anchor_scores).float().cpu().reshape(-1)
    top1_scores = torch.as_tensor(winner_scores).float().cpu().reshape(-1)
    count = int(fine_ids.numel())
    if (
        count == 0
        or descriptors.ndim != 2
        or descriptors.shape[0] != count
        or any(
            value.shape != (count,)
            for value in (
                anchor_rows,
                query_indices,
                wrong_anchor,
                wrong_fine,
                failures,
                inliers,
                truth_scores,
                top1_scores,
            )
        )
        or len(observation_block_ids) != count
        or len(observation_sequence_ids) != count
        or not bool(torch.isfinite(descriptors).all())
        or not bool(torch.isfinite(truth_scores).all())
        or not bool(torch.isfinite(top1_scores).all())
    ):
        raise ValueError("V21 diagnostic identity observations do not align")
    identities = torch.unique(fine_ids, sorted=True)
    support_ids = torch.as_tensor(mapping_support.get("fine_identity_ids")).long()
    if not torch.equal(identities, support_ids):
        raise ValueError("V21 mapping support identity registry differs")
    support_columns = {}
    for name in (
        "identity_anchor_row_count",
        "mapping_track_bank_observation_count",
        "mapping_track_bank_family_count",
        "mapping_all_valid_observation_count",
        "mapping_all_valid_family_count",
    ):
        value = torch.as_tensor(mapping_support.get(name)).long().cpu().reshape(-1)
        if value.shape != identities.shape or bool((value < 0).any()):
            raise ValueError("V21 mapping support columns do not align")
        support_columns[name] = value

    scalar_columns: dict[str, list] = defaultdict(list)
    descriptor_medoids = []
    medoid_blocks = []
    block_registries = []
    for fine_identity in identities.tolist():
        rows = torch.where(fine_ids == int(fine_identity))[0]
        row_list = rows.tolist()
        medoid = block_descriptor_medoid(
            descriptors[rows], [observation_block_ids[row] for row in row_list]
        )
        descriptor_medoids.append(medoid["descriptor_medoid"])
        medoid_blocks.append(medoid["descriptor_medoid_block_id"])
        block_registries.append(medoid["adaptation_block_ids"])
        sequences = {str(observation_sequence_ids[row]) for row in row_list}
        wrong_fine_rows = wrong_fine[rows]
        failure_wrong = failures[rows] & wrong_fine_rows
        scalar_columns["representative_anchor_rows"].append(
            int(anchor_rows[rows].min())
        )
        scalar_columns["adaptation_edge_count"].append(int(rows.numel()))
        scalar_columns["adaptation_query_count"].append(
            int(torch.unique(query_indices[rows]).numel())
        )
        scalar_columns["adaptation_block_count"].append(
            int(medoid["adaptation_block_count"])
        )
        scalar_columns["adaptation_sequence_count"].append(len(sequences))
        scalar_columns["wrong_top1_anchor_row_count"].append(
            int(wrong_anchor[rows].sum())
        )
        scalar_columns["wrong_top1_fine_identity_count"].append(
            int(wrong_fine_rows.sum())
        )
        scalar_columns["baseline_failure_wrong_top1_count"].append(
            int(failure_wrong.sum())
        )
        scalar_columns["baseline_failure_query_count"].append(
            int(torch.unique(query_indices[rows][failure_wrong]).numel())
        )
        scalar_columns["baseline_inlier_wrong_top1_count"].append(
            int((inliers[rows] & wrong_fine_rows).sum())
        )
        for name in (
            "descriptor_cross_block_defined",
            "descriptor_medoid_min_cosine",
            "descriptor_medoid_median_cosine",
            "descriptor_pairwise_median_cosine",
            "descriptor_edge_to_medoid_median_cosine",
        ):
            scalar_columns[name].append(medoid[name])
        selected_truth_scores = truth_scores[rows]
        margins = top1_scores[rows] - selected_truth_scores
        scalar_columns["truth_anchor_score_median"].append(
            float(selected_truth_scores.median())
        )
        scalar_columns["winner_margin_over_truth_median"].append(
            float(margins.median())
        )

    evidence = {
        "fine_identity_ids": identities.contiguous(),
        "representative_anchor_rows": torch.tensor(
            scalar_columns["representative_anchor_rows"], dtype=torch.long
        ),
        "descriptor_medoids": torch.stack(descriptor_medoids).contiguous(),
        "descriptor_medoid_block_ids": tuple(medoid_blocks),
        "adaptation_block_id_registries": tuple(block_registries),
    }
    integer_names = (
        "adaptation_edge_count",
        "adaptation_query_count",
        "adaptation_block_count",
        "adaptation_sequence_count",
        "wrong_top1_anchor_row_count",
        "wrong_top1_fine_identity_count",
        "baseline_failure_wrong_top1_count",
        "baseline_failure_query_count",
        "baseline_inlier_wrong_top1_count",
    )
    float_names = (
        "descriptor_medoid_min_cosine",
        "descriptor_medoid_median_cosine",
        "descriptor_pairwise_median_cosine",
        "descriptor_edge_to_medoid_median_cosine",
        "truth_anchor_score_median",
        "winner_margin_over_truth_median",
    )
    for name in integer_names:
        evidence[name] = torch.tensor(scalar_columns[name], dtype=torch.long)
    evidence["descriptor_cross_block_defined"] = torch.tensor(
        scalar_columns["descriptor_cross_block_defined"], dtype=torch.bool
    )
    for name in float_names:
        evidence[name] = torch.tensor(scalar_columns[name], dtype=torch.float32)
    evidence.update(support_columns)
    return evidence


def accepted_identity_mask(evidence: Mapping, thresholds: Mapping) -> torch.Tensor:
    normalized = calibration_thresholds(**dict(thresholds))
    return (
        torch.as_tensor(evidence["descriptor_cross_block_defined"]).bool()
        & (
            torch.as_tensor(evidence["adaptation_block_count"]).long()
            >= normalized["minimum_adaptation_blocks"]
        )
        & (
            torch.as_tensor(evidence["adaptation_sequence_count"]).long()
            >= normalized["minimum_adaptation_sequences"]
        )
        & (
            torch.as_tensor(evidence["mapping_track_bank_observation_count"]).long()
            >= normalized["minimum_mapping_observations"]
        )
        & (
            torch.as_tensor(evidence["mapping_track_bank_family_count"]).long()
            >= normalized["minimum_mapping_families"]
        )
        & (
            torch.as_tensor(evidence["descriptor_medoid_min_cosine"]).float()
            >= normalized["minimum_descriptor_medoid_cosine"]
        )
    ).contiguous()


def _identity_acceptance_for_observations(
    evidence: Mapping,
    accepted: torch.Tensor,
    observation_fine_identity_ids: torch.Tensor,
) -> torch.Tensor:
    identities = torch.as_tensor(evidence["fine_identity_ids"]).long().cpu()
    fine_ids = torch.as_tensor(observation_fine_identity_ids).long().cpu()
    selected = torch.as_tensor(accepted).bool().cpu()
    positions = torch.searchsorted(identities, fine_ids)
    if (
        positions.numel()
        and (
            int(positions.max()) >= identities.numel()
            or not torch.equal(identities[positions], fine_ids)
        )
    ):
        raise ValueError("V21 observation fine identity is absent from evidence")
    return selected[positions]


def threshold_grid_summary(
    *,
    evidence: Mapping,
    observation_fine_identity_ids: torch.Tensor,
    observation_query_indices: torch.Tensor,
    wrong_top1_fine_identity: torch.Tensor,
    baseline_failure: torch.Tensor,
    block_minimums: Sequence[int],
    sequence_minimums: Sequence[int],
    mapping_observation_minimums: Sequence[int],
    mapping_family_minimums: Sequence[int],
    descriptor_cosine_minimums: Sequence[float],
) -> list[dict]:
    fine_ids = torch.as_tensor(observation_fine_identity_ids).long().cpu()
    queries = torch.as_tensor(observation_query_indices).long().cpu()
    wrong = torch.as_tensor(wrong_top1_fine_identity).bool().cpu()
    failures = torch.as_tensor(baseline_failure).bool().cpu()
    if any(value.shape != fine_ids.shape for value in (queries, wrong, failures)):
        raise ValueError("V21 threshold-grid observation columns do not align")
    output = []
    combinations = itertools.product(
        sorted(set(map(int, block_minimums))),
        sorted(set(map(int, sequence_minimums))),
        sorted(set(map(int, mapping_observation_minimums))),
        sorted(set(map(int, mapping_family_minimums))),
        sorted(set(map(float, descriptor_cosine_minimums))),
    )
    for blocks, sequences, observations, families, cosine in combinations:
        thresholds = calibration_thresholds(
            minimum_adaptation_blocks=blocks,
            minimum_adaptation_sequences=sequences,
            minimum_mapping_observations=observations,
            minimum_mapping_families=families,
            minimum_descriptor_medoid_cosine=cosine,
        )
        accepted = accepted_identity_mask(evidence, thresholds)
        selected = _identity_acceptance_for_observations(
            evidence, accepted, fine_ids
        )
        promotion = selected & wrong
        failure_promotion = promotion & failures
        output.append(
            {
                "thresholds": thresholds,
                "accepted_identity_count": int(accepted.sum()),
                "provisional_positive_edge_count": int(selected.sum()),
                "promotion_wrong_top1_edge_count": int(promotion.sum()),
                "preservation_correct_top1_edge_count": int(
                    (selected & ~wrong).sum()
                ),
                "baseline_failure_wrong_top1_edge_count": int(
                    failure_promotion.sum()
                ),
                "baseline_failure_query_count": int(
                    torch.unique(queries[failure_promotion]).numel()
                ),
                "query_count": int(torch.unique(queries[selected]).numel()),
            }
        )
    if not output:
        raise ValueError("V21 identity calibration threshold grid is empty")
    return output


def build_query_provisional_record(
    *,
    query: Mapping,
    evidence: Mapping,
    accepted_identities: torch.Tensor,
) -> dict:
    """Build a positive-only CSR for one adaptation query."""

    count = int(query["keypoint_count"])
    rows = torch.as_tensor(query["diagnostic_unique_query_rows"]).long().cpu()
    anchors = torch.as_tensor(query["diagnostic_unique_anchor_rows"]).long().cpu()
    fine_ids = torch.as_tensor(
        query["diagnostic_unique_fine_identity_ids"]
    ).long().cpu()
    winners = torch.as_tensor(query["winner_anchor_rows"]).long().cpu()
    winner_fine = torch.as_tensor(query["winner_fine_identity_ids"]).long().cpu()
    inliers = torch.as_tensor(query["baseline_inlier"]).bool().cpu()
    truth_scores = torch.as_tensor(query["truth_anchor_scores"]).float().cpu()
    winner_scores = torch.as_tensor(query["winner_scores"]).float().cpu()
    diagnostic_count = int(rows.numel())
    if (
        count < 0
        or any(
            value.shape != (diagnostic_count,)
            for value in (
                anchors,
                fine_ids,
                winners,
                winner_fine,
                inliers,
                truth_scores,
                winner_scores,
            )
        )
        or (
            diagnostic_count
            and (
                int(rows.min()) < 0
                or int(rows.max()) >= count
                or not torch.equal(rows, torch.unique(rows, sorted=True))
            )
        )
    ):
        raise ValueError("V21 query diagnostic UNIQUE rows do not align")
    accepted = _identity_acceptance_for_observations(
        evidence, accepted_identities, fine_ids
    )
    wrong_anchor = anchors != winners
    wrong_fine = fine_ids != winner_fine
    counts = torch.zeros(count, dtype=torch.long)
    if diagnostic_count:
        counts[rows] = accepted.long()
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    provisional_anchors = anchors[accepted].contiguous()
    provisional_fine = fine_ids[accepted].contiguous()
    promotion = accepted & wrong_fine
    preservation = accepted & ~wrong_fine
    record = {
        "query_index": int(query["query_index"]),
        "image_name": str(query["image_name"]),
        "image_sha256": str(query["image_sha256"]),
        "sequence_id": str(query["sequence_id"]),
        "frame_index": int(query["frame_index"]),
        "block_id": str(query["block_id"]),
        "role": ROLE,
        "source_record_sha256": str(query["source_record_sha256"]),
        "pose_w2c_sha256": str(query["pose_w2c_sha256"]),
        "keypoint_count": count,
        "keypoints_sha256": str(query["keypoints_sha256"]),
        "descriptors_sha256": str(query["descriptors_sha256"]),
        "baseline_r5": bool(query["baseline_r5"]),
        "diagnostic_unique_query_rows": rows.contiguous(),
        "diagnostic_unique_anchor_rows": anchors.contiguous(),
        "diagnostic_unique_fine_identity_ids": fine_ids.contiguous(),
        "winner_anchor_rows": winners.contiguous(),
        "winner_fine_identity_ids": winner_fine.contiguous(),
        "wrong_top1_anchor_row": wrong_anchor.contiguous(),
        "wrong_top1_fine_identity": wrong_fine.contiguous(),
        "baseline_inlier": inliers.contiguous(),
        "truth_anchor_scores": truth_scores.contiguous(),
        "winner_scores": winner_scores.contiguous(),
        "winner_margin_over_truth": (winner_scores - truth_scores).contiguous(),
        "accepted_unique_mask": accepted.contiguous(),
        "provisional_positive_offsets": offsets.contiguous(),
        "provisional_positive_anchor_rows": provisional_anchors,
        "provisional_positive_fine_identity_ids": provisional_fine,
        "promotion_wrong_top1_count": int(promotion.sum()),
        "preservation_correct_top1_count": int(preservation.sum()),
        "negative_anchor_rows": None,
        "ambiguous_or_unlabelled_are_negative": False,
        "deployment_authorized": False,
    }
    validate_query_provisional_record(record)
    return record


def validate_query_provisional_record(
    record: Mapping, *, anchor_count: int | None = None
) -> None:
    count = int(record.get("keypoint_count", -1))
    rows = torch.as_tensor(record.get("diagnostic_unique_query_rows")).long()
    diagnostic_count = int(rows.numel())
    anchors = torch.as_tensor(record.get("diagnostic_unique_anchor_rows")).long()
    fine_ids = torch.as_tensor(
        record.get("diagnostic_unique_fine_identity_ids")
    ).long()
    winners = torch.as_tensor(record.get("winner_anchor_rows")).long()
    winner_fine = torch.as_tensor(record.get("winner_fine_identity_ids")).long()
    wrong_anchor = torch.as_tensor(record.get("wrong_top1_anchor_row"))
    wrong_fine = torch.as_tensor(record.get("wrong_top1_fine_identity"))
    inliers = torch.as_tensor(record.get("baseline_inlier"))
    truth_scores = torch.as_tensor(record.get("truth_anchor_scores")).float()
    winner_scores = torch.as_tensor(record.get("winner_scores")).float()
    margins = torch.as_tensor(record.get("winner_margin_over_truth")).float()
    accepted = torch.as_tensor(record.get("accepted_unique_mask"))
    if (
        count < 0
        or int(record.get("query_index", -1)) < 0
        or record.get("role") != ROLE
        or record.get("negative_anchor_rows") is not None
        or record.get("ambiguous_or_unlabelled_are_negative") is not False
        or record.get("deployment_authorized") is not False
        or any(
            value.shape != (diagnostic_count,)
            for value in (
                anchors,
                fine_ids,
                winners,
                winner_fine,
                wrong_anchor,
                wrong_fine,
                inliers,
                truth_scores,
                winner_scores,
                margins,
                accepted,
            )
        )
        or any(value.dtype != torch.bool for value in (wrong_anchor, wrong_fine, inliers, accepted))
    ):
        raise ValueError("V21 provisional query contract is invalid")
    for name in (
        "image_sha256",
        "source_record_sha256",
        "pose_w2c_sha256",
        "keypoints_sha256",
        "descriptors_sha256",
    ):
        _require_sha256(record.get(name), label=name)
    if diagnostic_count and (
        int(rows.min()) < 0
        or int(rows.max()) >= count
        or not torch.equal(rows, torch.unique(rows, sorted=True))
        or int(anchors.min()) < 0
        or (anchor_count is not None and int(anchors.max()) >= anchor_count)
        or not bool(torch.isfinite(truth_scores).all())
        or not bool(torch.isfinite(winner_scores).all())
    ):
        raise ValueError("V21 provisional diagnostic rows are invalid")
    if not torch.equal(wrong_anchor, anchors != winners) or not torch.equal(
        wrong_fine, fine_ids != winner_fine
    ):
        raise ValueError("V21 wrong-Top1 masks differ from frozen identities")
    if not torch.allclose(margins, winner_scores - truth_scores, atol=1e-7, rtol=0.0):
        raise ValueError("V21 winner margins differ from frozen scores")
    offsets = torch.as_tensor(record.get("provisional_positive_offsets")).long()
    provisional_anchors = torch.as_tensor(
        record.get("provisional_positive_anchor_rows")
    ).long()
    provisional_fine = torch.as_tensor(
        record.get("provisional_positive_fine_identity_ids")
    ).long()
    counts = torch.zeros(count, dtype=torch.long)
    if diagnostic_count:
        counts[rows] = accepted.long()
    expected_offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    if (
        not torch.equal(offsets, expected_offsets)
        or not torch.equal(provisional_anchors, anchors[accepted])
        or not torch.equal(provisional_fine, fine_ids[accepted])
        or int(record.get("promotion_wrong_top1_count", -1))
        != int((accepted & wrong_fine).sum())
        or int(record.get("preservation_correct_top1_count", -1))
        != int((accepted & ~wrong_fine).sum())
    ):
        raise ValueError("V21 provisional positive CSR differs from its evidence")


def validate_identity_evidence(
    evidence: Mapping, *, thresholds: Mapping, descriptor_dim: int
) -> torch.Tensor:
    ids = torch.as_tensor(evidence.get("fine_identity_ids")).long()
    count = int(ids.numel())
    medoids = torch.as_tensor(evidence.get("descriptor_medoids")).float()
    if (
        count <= 0
        or ids.ndim != 1
        or not torch.equal(ids, torch.unique(ids, sorted=True))
        or medoids.shape != (count, descriptor_dim)
        or not bool(torch.isfinite(medoids).all())
        or not torch.allclose(
            torch.linalg.vector_norm(medoids, dim=1),
            torch.ones(count),
            atol=1e-5,
            rtol=1e-5,
        )
        or len(evidence.get("descriptor_medoid_block_ids", ())) != count
        or len(evidence.get("adaptation_block_id_registries", ())) != count
    ):
        raise ValueError("V21 identity evidence registry is invalid")
    integer_names = (
        "representative_anchor_rows",
        "adaptation_edge_count",
        "adaptation_query_count",
        "adaptation_block_count",
        "adaptation_sequence_count",
        "wrong_top1_anchor_row_count",
        "wrong_top1_fine_identity_count",
        "baseline_failure_wrong_top1_count",
        "baseline_failure_query_count",
        "baseline_inlier_wrong_top1_count",
        "identity_anchor_row_count",
        "mapping_track_bank_observation_count",
        "mapping_track_bank_family_count",
        "mapping_all_valid_observation_count",
        "mapping_all_valid_family_count",
    )
    for name in integer_names:
        value = torch.as_tensor(evidence.get(name)).long()
        if value.shape != (count,) or bool((value < 0).any()):
            raise ValueError(f"V21 identity evidence {name} is invalid")
    defined = torch.as_tensor(evidence.get("descriptor_cross_block_defined"))
    if defined.shape != (count,) or defined.dtype != torch.bool:
        raise ValueError("V21 cross-block descriptor mask is invalid")
    float_names = (
        "descriptor_medoid_min_cosine",
        "descriptor_medoid_median_cosine",
        "descriptor_pairwise_median_cosine",
        "descriptor_edge_to_medoid_median_cosine",
        "truth_anchor_score_median",
        "winner_margin_over_truth_median",
    )
    for name in float_names:
        value = torch.as_tensor(evidence.get(name)).float()
        if value.shape != (count,) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"V21 identity evidence {name} is invalid")
    blocks = torch.as_tensor(evidence["adaptation_block_count"]).long()
    if not torch.equal(defined, blocks >= 2):
        raise ValueError("V21 cross-block definition differs from block counts")
    if bool(
        (
            torch.as_tensor(evidence["mapping_track_bank_observation_count"])
            > torch.as_tensor(evidence["mapping_all_valid_observation_count"])
        ).any()
    ) or bool(
        (
            torch.as_tensor(evidence["mapping_track_bank_family_count"])
            > torch.as_tensor(evidence["mapping_all_valid_family_count"])
        ).any()
    ):
        raise ValueError("V21 Track-bank support exceeds all-valid support")
    return accepted_identity_mask(evidence, thresholds)


def aggregate_record_counts(records: Sequence[Mapping]) -> dict:
    return {
        "query_count": len(records),
        "baseline_failure_query_count": sum(
            not bool(record["baseline_r5"]) for record in records
        ),
        "diagnostic_unique_edge_count": sum(
            int(torch.as_tensor(record["diagnostic_unique_query_rows"]).numel())
            for record in records
        ),
        "wrong_top1_anchor_row_edge_count": sum(
            int(torch.as_tensor(record["wrong_top1_anchor_row"]).sum())
            for record in records
        ),
        "wrong_top1_fine_identity_edge_count": sum(
            int(torch.as_tensor(record["wrong_top1_fine_identity"]).sum())
            for record in records
        ),
        "provisional_positive_edge_count": sum(
            int(torch.as_tensor(record["provisional_positive_anchor_rows"]).numel())
            for record in records
        ),
        "promotion_wrong_top1_edge_count": sum(
            int(record["promotion_wrong_top1_count"]) for record in records
        ),
        "preservation_correct_top1_edge_count": sum(
            int(record["preservation_correct_top1_count"]) for record in records
        ),
        "baseline_failure_promotion_edge_count": sum(
            int(record["promotion_wrong_top1_count"])
            for record in records
            if not bool(record["baseline_r5"])
        ),
        "baseline_failure_query_with_promotion_count": sum(
            not bool(record["baseline_r5"])
            and int(record["promotion_wrong_top1_count"]) > 0
            for record in records
        ),
        "query_with_provisional_positive_count": sum(
            int(torch.as_tensor(record["provisional_positive_anchor_rows"]).numel())
            > 0
            for record in records
        ),
    }


def validate_identity_calibration_payload(payload: Mapping) -> None:
    if not (
        payload.get("schema") == SCHEMA
        and payload.get("version") == VERSION
        and payload.get("protocol") == "test_adapted"
        and payload.get("uses_test_queries") is True
        and payload.get("test_adapted") is True
        and payload.get("role") == ROLE
        and payload.get("complete_adaptation_registry_consumed") is True
        and payload.get("formation_stage")
        == "after_complete_adaptation_before_control_scoring"
        and payload.get("control_or_confirmation_features_consumed") is False
        and payload.get("control_or_confirmation_outcomes_consumed") is False
        and payload.get("source_teacher_mutating_action_authorized") is False
        and payload.get("provisional_action_positive_only") is True
        and payload.get("negative_labels_created") is False
        and payload.get("ambiguous_or_unlabelled_are_negative") is False
        and payload.get("artifact_writes_map") is False
        and payload.get("candidate_deployment_authorized") is False
        and payload.get("heldout_control_required_before_confirmation") is True
        and payload.get("heldout_confirmation_required_before_deployment") is True
        and payload.get("semantics") == SEMANTICS
    ):
        raise ValueError("unsupported V21 identity calibration payload")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 identity calibration input lineage is missing")
    primary = {
        name: _source_record(inputs.get(name), label=name)
        for name in (
            "correspondence_truth",
            "stable_map",
            "split_manifest",
            "mapping_provenance",
            "teacher_validation",
        )
    }
    cache_sources = inputs.get("frontend_caches")
    producer_sources = inputs.get("producer_sources")
    if not isinstance(cache_sources, list) or not cache_sources:
        raise ValueError("V21 identity calibration frontend lineage is empty")
    if not isinstance(producer_sources, list) or not producer_sources:
        raise ValueError("V21 identity calibration producer lineage is empty")
    [_source_record(value, label="frontend cache") for value in cache_sources]
    [_source_record(value, label="producer") for value in producer_sources]
    for name, source in primary.items():
        if payload.get(f"{name}_sha256") != source["sha256"]:
            raise ValueError("V21 identity calibration SHA lineage differs")
    registry = payload.get("frontend_shard_registry")
    if (
        not isinstance(registry, Mapping)
        or payload.get("frontend_shard_registry_sha256")
        != registry.get("registry_sha256")
        or registry.get("role") != ROLE
    ):
        raise ValueError("V21 identity calibration frontend registry is invalid")
    thresholds = payload.get("candidate_thresholds")
    if not isinstance(thresholds, Mapping) or payload.get(
        "candidate_thresholds_sha256"
    ) != sha256_json(thresholds):
        raise ValueError("V21 identity calibration thresholds are not bound")
    descriptor_dim = int(payload.get("descriptor_dim", 0))
    evidence = payload.get("identity_evidence")
    if not isinstance(evidence, Mapping) or descriptor_dim <= 0:
        raise ValueError("V21 identity calibration evidence is missing")
    accepted = validate_identity_evidence(
        evidence, thresholds=thresholds, descriptor_dim=descriptor_dim
    )
    recorded_accepted = torch.as_tensor(
        evidence.get("accepted_identity_mask")
    )
    if recorded_accepted.dtype != torch.bool or not torch.equal(
        accepted, recorded_accepted
    ):
        raise ValueError("V21 accepted identities differ from frozen thresholds")
    records = payload.get("records")
    registry_rows = sorted(
        registry.get("rows", ()), key=lambda value: int(value["ordinal"])
    )
    anchor_count = int(payload.get("anchor_count", 0))
    if (
        not isinstance(records, list)
        or len(records) != len(registry_rows)
        or anchor_count <= 0
        or int(payload.get("query_count", -1)) != len(records)
    ):
        raise ValueError("V21 identity calibration query coverage is incomplete")
    accepted_ids = set(
        torch.as_tensor(evidence["fine_identity_ids"])[accepted].long().tolist()
    )
    for record, registry_row in zip(records, registry_rows):
        validate_query_provisional_record(record, anchor_count=anchor_count)
        if (
            int(record["query_index"]) != int(registry_row["query_index"])
            or record["image_name"] != registry_row["image_name"]
            or record["image_sha256"] != registry_row["image_sha256"]
            or record["source_record_sha256"]
            != registry_row["source_record_sha256"]
        ):
            raise ValueError("V21 identity calibration record registry differs")
        observed_acceptance = torch.as_tensor(record["accepted_unique_mask"]).bool()
        observed_ids = torch.as_tensor(
            record["diagnostic_unique_fine_identity_ids"]
        ).long()
        expected_acceptance = torch.tensor(
            [int(value) in accepted_ids for value in observed_ids.tolist()],
            dtype=torch.bool,
        )
        if not torch.equal(observed_acceptance, expected_acceptance):
            raise ValueError("V21 query acceptance differs from identity evidence")
    aggregate = aggregate_record_counts(records)
    if payload.get("counts") != aggregate:
        raise ValueError("V21 identity calibration aggregate counts differ")
    if int(accepted.sum()) != int(payload.get("accepted_identity_count", -1)):
        raise ValueError("V21 accepted identity count differs")
    available = aggregate["provisional_positive_edge_count"] > 0
    if (
        payload.get("provisional_action_positive_available") is not available
        or payload.get("quarantined_candidate_generation_allowed") is not available
    ):
        raise ValueError("V21 provisional candidate availability differs")
    grid = payload.get("threshold_grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("V21 identity calibration threshold grid is missing")
    for row in grid:
        if not isinstance(row, Mapping) or not isinstance(
            row.get("thresholds"), Mapping
        ):
            raise ValueError("V21 identity calibration threshold grid is malformed")
        calibration_thresholds(**dict(row["thresholds"]))


def atomic_torch_save_fresh(payload: Mapping, output: str | Path) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V21 identity calibration output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        torch.save(dict(payload), temporary)
        validate_identity_calibration_payload(
            torch.load(temporary, map_location="cpu", weights_only=False)
        )
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"V21 identity calibration output appeared while running: {output}"
            ) from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "ROLE",
    "SCHEMA",
    "SEMANTICS",
    "VERSION",
    "accepted_identity_mask",
    "aggregate_record_counts",
    "atomic_torch_save_fresh",
    "block_descriptor_medoid",
    "build_identity_evidence",
    "build_query_provisional_record",
    "calibration_thresholds",
    "mapping_identity_support",
    "sha256_json",
    "threshold_grid_summary",
    "validate_identity_calibration_payload",
    "validate_identity_evidence",
    "validate_query_provisional_record",
]
