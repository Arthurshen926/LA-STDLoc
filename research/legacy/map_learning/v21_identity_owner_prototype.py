"""Strict V21 calibration-medoid owner prototypes and exact cached replay.

This is an independent candidate arm from the geometry/PoseLib recovery arm.
It consumes only the complete adaptation frontend cache and a validated V21
identity-calibration artifact.  Every accepted fine identity contributes at
most one block-balanced prototype to an owner Anchor; frozen Anchor fields are
bit-exact and no negative labels are created.

The identity evidence remains provisional.  A candidate produced here is
quarantined and non-deployable until held-out control passes a predeclared
gate and confirmation subsequently succeeds.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import re
from typing import Callable

import torch
import torch.nn.functional as F

from localization.pose_solver import solve_absolute_pose
from map_learning.v21_identity_calibration import (
    accepted_identity_mask,
    block_descriptor_medoid,
    validate_identity_calibration_payload,
)
from map_learning.v21_pose_feedback_transductive import (
    EVALUATION_SCHEMA as BASE_EVALUATION_SCHEMA,
    PROTOTYPE_FEATURE_FIELD,
    PROTOTYPE_OWNER_FIELD,
    _bit_exact,
    _clone_frozen_value,
    _same_source,
    _source_identity,
    _validate_stable_map,
    evaluate_cached_record,
    summarize_cached_evaluation,
    validate_cached_evaluation_payload as validate_base_cached_evaluation,
    validate_complete_cache_payloads,
)
from map_learning.v21_test_cache import tensor_sha256


CANDIDATE_SCHEMA = "lafgs_v21_identity_owner_prototype_candidate"
EVALUATION_SCHEMA = "lafgs_v21_identity_owner_prototype_cached_evaluation"
VERSION = 1
METADATA_FIELD = "v21_identity_owner_prototype"
MATCHING_SEMANTICS = "global_owner_prototype_top1"
PROTOTYPE_FEATURE_SOURCE = (
    "normalized_equal_block_mean_after_exact_calibration_medoid_recompute"
)
CONTROL_GATE_POLICY = {
    "minimum_paired_r5_gain_count": 1,
    "maximum_paired_r5_loss_count": 0,
    "maximum_median_paired_delta_task_error": 0.0,
}


def _canonical_source(value: object, *, label: str) -> dict:
    path, digest = _source_identity(value, label=label)
    size = int(value.get("size_bytes", 0)) if isinstance(value, Mapping) else 0
    if size <= 0:
        raise ValueError(f"V21 {label} source size is invalid")
    return {"path": path, "sha256": digest, "size_bytes": size}


def assert_base_fields_bit_exact(
    stable_map: Mapping, candidate_map: Mapping
) -> None:
    additions = {PROTOTYPE_FEATURE_FIELD, PROTOTYPE_OWNER_FIELD, METADATA_FIELD}
    if set(candidate_map) != set(stable_map) | additions:
        raise ValueError("V21 identity candidate changed the frozen field registry")
    for key, value in stable_map.items():
        if not _bit_exact(value, candidate_map[key]):
            raise ValueError(f"V21 identity candidate changed frozen field: {key}")


def _stable_identity_registry(stable_map: Mapping) -> torch.Tensor:
    features, _ = _validate_stable_map(stable_map)
    fine_ids = torch.as_tensor(stable_map.get("fine_identity_ids")).long().cpu()
    if fine_ids.shape != (features.shape[0],) or bool((fine_ids < 0).any()):
        raise ValueError("V21 stable-map fine-identity registry is invalid")
    return fine_ids


def _cache_record_join(
    calibration_record: Mapping, cache_record: Mapping
) -> None:
    if not (
        int(calibration_record.get("query_index", -1))
        == int(cache_record.get("query_index", -2))
        and calibration_record.get("image_name") == cache_record.get("image_name")
        and calibration_record.get("image_sha256")
        == cache_record.get("image_sha256")
        and calibration_record.get("source_record_sha256")
        == cache_record.get("source_record_sha256")
        and calibration_record.get("pose_w2c_sha256")
        == cache_record.get("pose_w2c_sha256")
        and int(calibration_record.get("keypoint_count", -1))
        == int(torch.as_tensor(cache_record.get("keypoints")).shape[0])
        and calibration_record.get("keypoints_sha256")
        == tensor_sha256(torch.as_tensor(cache_record.get("keypoints")).float())
        and calibration_record.get("descriptors_sha256")
        == tensor_sha256(torch.as_tensor(cache_record.get("descriptors")).float())
        and calibration_record.get("role") == "adaptation"
        and cache_record.get("role") == "adaptation"
    ):
        raise ValueError("V21 calibration/adaptation cache query binding differs")


def _block_balanced_prototype(
    *,
    descriptors: torch.Tensor,
    block_ids: Sequence[str],
    sequence_ids: Sequence[str],
) -> dict:
    """Recompute the calibration medoid, then average all block centroids.

    The deployed prototype is deliberately not the selected medoid block
    vector.  It is the normalized, equal-block spherical mean of every verified
    block centroid, ensuring the candidate cannot copy one same-frame row.
    """

    values = torch.as_tensor(descriptors).float().cpu()
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or len(block_ids) != values.shape[0]
        or len(sequence_ids) != values.shape[0]
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("V21 identity prototype descriptor evidence is invalid")
    # The calibration audit first normalized each cached query descriptor while
    # compacting the frontend record, then ``block_descriptor_medoid``
    # normalized that compact bank again.  Reproduce both operations exactly;
    # omitting the first step is numerically close but breaks bit-exact medoid
    # lineage under float32 rounding.
    calibration_descriptors = F.normalize(values, dim=1).contiguous()
    normalized = F.normalize(calibration_descriptors, dim=1)
    by_block: dict[str, list[int]] = defaultdict(list)
    block_sequence: dict[str, str] = {}
    for row, (raw_block, raw_sequence) in enumerate(zip(block_ids, sequence_ids)):
        block = str(raw_block)
        sequence = str(raw_sequence)
        if not block or not sequence:
            raise ValueError("V21 identity prototype block/sequence ID is empty")
        existing = block_sequence.setdefault(block, sequence)
        if existing != sequence:
            raise ValueError("V21 adaptation block spans multiple test sequences")
        by_block[block].append(row)
    ordered_blocks = sorted(by_block)
    centroids = []
    for block in ordered_blocks:
        centroid = normalized[by_block[block]].mean(0)
        norm = float(torch.linalg.vector_norm(centroid))
        if not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError("V21 identity prototype block centroid is invalid")
        centroids.append(centroid / norm)
    centroid_tensor = torch.stack(centroids).contiguous()
    medoid = block_descriptor_medoid(calibration_descriptors, block_ids)
    prototype = F.normalize(centroid_tensor.mean(0), dim=0).contiguous()
    source_hashes = tuple(tensor_sha256(row) for row in calibration_descriptors)
    prototype_hash = tensor_sha256(prototype)
    if len(ordered_blocks) < 2 or prototype_hash in source_hashes:
        raise ValueError("V21 identity prototype is not genuinely multi-block")
    medoid_hash = tensor_sha256(medoid["descriptor_medoid"])
    centroid_hashes = tuple(tensor_sha256(value) for value in centroid_tensor)
    try:
        medoid_index = centroid_hashes.index(medoid_hash)
    except ValueError as error:
        raise ValueError("V21 identity medoid is absent from block centroids") from error
    medoid_cosines = centroid_tensor @ medoid["descriptor_medoid"]
    return {
        "prototype": prototype,
        "prototype_sha256": prototype_hash,
        "normalized_source_descriptors": calibration_descriptors,
        "source_descriptor_sha256s": source_hashes,
        "block_centroids": centroid_tensor,
        "block_centroid_sha256s": centroid_hashes,
        "block_ids": tuple(ordered_blocks),
        "block_sequence_ids": tuple(block_sequence[value] for value in ordered_blocks),
        "medoid": medoid["descriptor_medoid"].contiguous(),
        "medoid_sha256": medoid_hash,
        "medoid_index": medoid_index,
        "medoid_min_cosine": float(medoid_cosines.min()),
        "medoid_median_cosine": float(medoid_cosines.median()),
    }


def derive_verified_identity_actions(
    *,
    stable_map: Mapping,
    calibration: Mapping,
    adaptation_records: Sequence[Mapping],
) -> list[dict]:
    """Join accepted calibration edges to cached descriptors and recompute all evidence."""

    validate_identity_calibration_payload(calibration)
    if not (
        calibration.get("role") == "adaptation"
        and calibration.get("candidate_deployment_authorized") is False
        and calibration.get("provisional_action_positive_only") is True
        and calibration.get("complete_adaptation_registry_consumed") is True
        and calibration.get("counts", {}).get(
            "preservation_correct_top1_edge_count", 0
        )
        > 0
    ):
        raise ValueError("V21 strict provisional calibration contract differs")
    thresholds = dict(calibration["candidate_thresholds"])
    if not (
        int(thresholds["minimum_adaptation_blocks"]) >= 3
        and int(thresholds["minimum_adaptation_sequences"]) >= 2
        and int(thresholds["minimum_mapping_observations"]) >= 2
        and int(thresholds["minimum_mapping_families"]) >= 2
        and float(thresholds["minimum_descriptor_medoid_cosine"]) >= 0.8
    ):
        raise ValueError("V21 identity prototype requires the strict calibration gate")
    calibration_records = list(calibration.get("records", ()))
    if len(calibration_records) != len(adaptation_records):
        raise ValueError("V21 calibration/adaptation query coverage differs")
    evidence = calibration["identity_evidence"]
    identity_ids = torch.as_tensor(evidence["fine_identity_ids"]).long().cpu()
    accepted = accepted_identity_mask(evidence, thresholds)
    if not torch.equal(
        accepted, torch.as_tensor(evidence["accepted_identity_mask"]).bool().cpu()
    ):
        raise ValueError("V21 strict identity acceptance differs from its thresholds")
    accepted_ids = set(identity_ids[accepted].tolist())
    if len(accepted_ids) != int(calibration.get("accepted_identity_count", -1)):
        raise ValueError("V21 accepted identity registry differs")
    fine_ids = _stable_identity_registry(stable_map)
    observations: dict[int, dict[str, list]] = {
        int(value): defaultdict(list) for value in accepted_ids
    }
    for calibration_record, cache_record in zip(
        calibration_records, adaptation_records
    ):
        _cache_record_join(calibration_record, cache_record)
        rows = torch.as_tensor(
            calibration_record["diagnostic_unique_query_rows"]
        ).long().cpu()
        anchors = torch.as_tensor(
            calibration_record["diagnostic_unique_anchor_rows"]
        ).long().cpu()
        record_ids = torch.as_tensor(
            calibration_record["diagnostic_unique_fine_identity_ids"]
        ).long().cpu()
        accepted_rows = torch.as_tensor(
            calibration_record["accepted_unique_mask"]
        ).bool().cpu()
        wrong_top1 = torch.as_tensor(
            calibration_record["wrong_top1_fine_identity"]
        ).bool().cpu()
        count = int(rows.numel())
        if any(
            value.shape != (count,)
            for value in (anchors, record_ids, accepted_rows, wrong_top1)
        ):
            raise ValueError("V21 calibration diagnostic columns do not align")
        descriptor_bank = torch.as_tensor(cache_record["descriptors"]).float().cpu()
        if rows.numel() and (
            int(rows.min()) < 0 or int(rows.max()) >= descriptor_bank.shape[0]
        ):
            raise ValueError("V21 calibration query row is outside its cache")
        if anchors.numel() and not torch.equal(fine_ids[anchors], record_ids):
            raise ValueError("V21 calibration fine identity differs from stable map")
        for local in torch.where(accepted_rows)[0].tolist():
            identity = int(record_ids[local])
            if identity not in observations:
                raise ValueError("V21 accepted query row has an unaccepted identity")
            target = observations[identity]
            target["descriptors"].append(descriptor_bank[int(rows[local])])
            target["block_ids"].append(str(cache_record["block_id"]))
            target["sequence_ids"].append(str(cache_record["sequence_id"]))
            target["query_indices"].append(int(cache_record["query_index"]))
            target["query_rows"].append(int(rows[local]))
            target["anchor_rows"].append(int(anchors[local]))
            target["wrong_top1"].append(bool(wrong_top1[local]))
    total_edges = sum(len(value["descriptors"]) for value in observations.values())
    if total_edges != int(calibration["counts"]["provisional_positive_edge_count"]):
        raise ValueError("V21 accepted calibration edge coverage differs")

    evidence_row = {int(value): row for row, value in enumerate(identity_ids.tolist())}
    actions = []
    for identity in sorted(accepted_ids):
        row = evidence_row[identity]
        source = observations[identity]
        if not source["descriptors"]:
            raise ValueError("V21 accepted identity has no adaptation observations")
        recomputed = _block_balanced_prototype(
            descriptors=torch.stack(source["descriptors"]),
            block_ids=source["block_ids"],
            sequence_ids=source["sequence_ids"],
        )
        stored_medoid = torch.as_tensor(evidence["descriptor_medoids"])[row].float()
        stored_blocks = tuple(evidence["adaptation_block_id_registries"][row])
        sequence_count = len(set(source["sequence_ids"]))
        owner = int(torch.as_tensor(evidence["representative_anchor_rows"])[row])
        checks = {
            "medoid_sha256": tensor_sha256(stored_medoid)
            == recomputed["medoid_sha256"],
            "block_registry": stored_blocks == recomputed["block_ids"],
            "edge_count": int(
                torch.as_tensor(evidence["adaptation_edge_count"])[row]
            )
            == len(source["descriptors"]),
            "block_count": int(
                torch.as_tensor(evidence["adaptation_block_count"])[row]
            )
            == len(recomputed["block_ids"]),
            "sequence_count": int(
                torch.as_tensor(evidence["adaptation_sequence_count"])[row]
            )
            == sequence_count,
            "strict_block_count": len(recomputed["block_ids"])
            >= int(thresholds["minimum_adaptation_blocks"]),
            "strict_sequence_count": sequence_count
            >= int(thresholds["minimum_adaptation_sequences"]),
            "mapping_observation_count": int(
                torch.as_tensor(evidence["mapping_track_bank_observation_count"])[
                    row
                ]
            )
            >= int(thresholds["minimum_mapping_observations"]),
            "mapping_family_count": int(
                torch.as_tensor(evidence["mapping_track_bank_family_count"])[row]
            )
            >= int(thresholds["minimum_mapping_families"]),
            "medoid_min_cosine": math.isclose(
                float(torch.as_tensor(evidence["descriptor_medoid_min_cosine"])[row]),
                recomputed["medoid_min_cosine"],
                rel_tol=1e-6,
                abs_tol=1e-6,
            ),
            "strict_medoid_min_cosine": recomputed["medoid_min_cosine"]
            >= float(thresholds["minimum_descriptor_medoid_cosine"]),
            "representative_owner_observed": owner in set(source["anchor_rows"]),
            "owner_identity": int(fine_ids[owner]) == identity,
        }
        failed_checks = sorted(name for name, passed in checks.items() if not passed)
        if failed_checks:
            raise ValueError(
                "V21 stored/recomputed identity evidence differs for fine identity "
                f"{identity}: {', '.join(failed_checks)}"
            )
        actions.append(
            {
                "fine_identity_id": identity,
                "owner_anchor_row": owner,
                "prototype": recomputed["prototype"],
                "prototype_sha256": recomputed["prototype_sha256"],
                "stored_medoid": stored_medoid.contiguous(),
                "stored_medoid_sha256": tensor_sha256(stored_medoid),
                "block_centroids": recomputed["block_centroids"],
                "block_centroid_sha256s": recomputed["block_centroid_sha256s"],
                "block_ids": recomputed["block_ids"],
                "block_sequence_ids": recomputed["block_sequence_ids"],
                "source_descriptor_sha256s": recomputed[
                    "source_descriptor_sha256s"
                ],
                "source_query_indices": tuple(sorted(set(source["query_indices"]))),
                "source_edge_count": len(source["descriptors"]),
                "adaptation_block_count": len(recomputed["block_ids"]),
                "adaptation_sequence_count": sequence_count,
                "mapping_track_bank_observation_count": int(
                    torch.as_tensor(evidence["mapping_track_bank_observation_count"])[
                        row
                    ]
                ),
                "mapping_track_bank_family_count": int(
                    torch.as_tensor(evidence["mapping_track_bank_family_count"])[row]
                ),
                "descriptor_medoid_min_cosine": recomputed[
                    "medoid_min_cosine"
                ],
                "prototype_to_stored_medoid_cosine": float(
                    recomputed["prototype"] @ stored_medoid
                ),
                "promotion_wrong_top1_edge_count": sum(source["wrong_top1"]),
                "preservation_correct_top1_edge_count": len(source["wrong_top1"])
                - sum(source["wrong_top1"]),
            }
        )
    if len({value["owner_anchor_row"] for value in actions}) != len(actions):
        raise ValueError("V21 strict identities do not map to unique owner Anchors")
    return actions


def build_identity_owner_prototype_candidate(
    *,
    stable_map: Mapping,
    calibration: Mapping,
    adaptation_cache_payloads: Sequence[Mapping],
    stable_map_source: Mapping,
    calibration_source: Mapping,
    adaptation_cache_sources: Sequence[Mapping],
    producer_sources: Sequence[Mapping],
    maximum_total_prototypes: int = 128,
    prototype_activation_threshold: float | None = None,
) -> dict:
    """Materialize the complete strict-identity arm without held-out inputs."""

    features, _ = _validate_stable_map(stable_map)
    payloads, records, baseline_contract = validate_complete_cache_payloads(
        adaptation_cache_payloads, required_role="adaptation"
    )
    if not all(
        payload.get("training_consumers_allowed") is True
        and payload.get("training_consumer_allowed") is True
        for payload in payloads
    ):
        raise ValueError("V21 identity candidate received held-out cache data")
    if len(payloads) != len(adaptation_cache_sources):
        raise ValueError("V21 identity candidate cache source registry differs")
    _same_source(
        calibration.get("inputs", {}).get("stable_map"),
        stable_map_source,
        label="identity calibration stable map",
    )
    declared_calibration_caches = {
        _source_identity(value, label="identity calibration cache")
        for value in calibration.get("inputs", {}).get("frontend_caches", ())
    }
    actual_cache_sources = {
        _source_identity(value, label="identity candidate cache")
        for value in adaptation_cache_sources
    }
    if (
        declared_calibration_caches != actual_cache_sources
        or len(actual_cache_sources) != len(adaptation_cache_sources)
    ):
        raise ValueError("V21 identity calibration/cache lineage differs")
    for payload in payloads:
        _same_source(
            payload.get("inputs", {}).get("stable_map"),
            stable_map_source,
            label="identity candidate stable map",
        )
        if payload.get("shard_registry") != calibration.get(
            "frontend_shard_registry"
        ):
            raise ValueError("V21 identity calibration/cache registries differ")
    if (
        int(payloads[0]["anchor_count"]) != features.shape[0]
        or int(payloads[0]["descriptor_dim"]) != features.shape[1]
        or int(maximum_total_prototypes) < 1
        or (
            prototype_activation_threshold is not None
            and (
                not math.isfinite(float(prototype_activation_threshold))
                or not 0.0 <= float(prototype_activation_threshold) <= 1.0
            )
        )
    ):
        raise ValueError("V21 identity candidate dimensions/budget are invalid")
    actions = derive_verified_identity_actions(
        stable_map=stable_map,
        calibration=calibration,
        adaptation_records=records,
    )
    if len(actions) > int(maximum_total_prototypes):
        raise ValueError("V21 strict identity set exceeds the prototype budget")
    prototypes = torch.stack([value["prototype"] for value in actions]).contiguous()
    owners = torch.tensor(
        [value["owner_anchor_row"] for value in actions], dtype=torch.long
    )
    selected_actions = []
    for index, action in enumerate(actions):
        selected = dict(action)
        selected.pop("prototype")
        selected["prototype_index"] = index
        selected["prototype_uses_multiple_blocks"] = True
        selected["prototype_is_single_observation_copy"] = False
        selected_actions.append(selected)
    all_source_queries = sorted(
        {
            query
            for action in actions
            for query in action["source_query_indices"]
        }
    )
    metadata = {
        "schema": CANDIDATE_SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "formation_role": "adaptation",
        "complete_adaptation_registry_consumed": True,
        "adaptation_features_consumed": True,
        "control_features_consumed": False,
        "confirmation_features_consumed": False,
        "control_or_confirmation_outcomes_consumed": False,
        "identity_evidence_authority": "provisional_candidate_only",
        "identity_truth_claimed": False,
        "negative_anchor_labels_created": False,
        "base_anchor_fields_bit_exact": True,
        "base_anchor_descriptors_retained": True,
        "base_anchor_features_moved_or_lowered": False,
        "geometry_changed": False,
        "matching_semantics": MATCHING_SEMANTICS,
        "prototype_feature_source": PROTOTYPE_FEATURE_SOURCE,
        "prototype_activation_threshold": (
            None
            if prototype_activation_threshold is None
            else float(prototype_activation_threshold)
        ),
        "deployment_authorized": False,
        "controller_authorized": False,
        "heldout_control_required": True,
        "heldout_confirmation_required": True,
        "confirmation_evaluation_authorized": False,
        "inputs": {
            "stable_map": dict(stable_map_source),
            "identity_calibration": dict(calibration_source),
            "adaptation_caches": [dict(value) for value in adaptation_cache_sources],
            "split_manifest": dict(payloads[0]["inputs"]["split_manifest"]),
            "producer_sources": [dict(value) for value in producer_sources],
        },
        "frontend_shard_registry_sha256": payloads[0]["shard_registry"][
            "registry_sha256"
        ],
        "preprocessing_config_sha256": payloads[0][
            "preprocessing_config_sha256"
        ],
        "baseline_contract": baseline_contract,
        "calibration_thresholds": dict(calibration["candidate_thresholds"]),
        "maximum_total_prototypes": int(maximum_total_prototypes),
        "maximum_prototypes_per_fine_identity": 1,
        "accepted_identity_count": len(actions),
        "added_prototype_count": len(actions),
        "source_query_count": len(all_source_queries),
        "source_query_indices": torch.tensor(all_source_queries, dtype=torch.long),
        "prototype_features_sha256": tensor_sha256(prototypes),
        "prototype_owner_rows_sha256": tensor_sha256(owners),
        "selected_actions": selected_actions,
    }
    candidate = _clone_frozen_value(stable_map)
    candidate[PROTOTYPE_FEATURE_FIELD] = prototypes
    candidate[PROTOTYPE_OWNER_FIELD] = owners
    candidate[METADATA_FIELD] = metadata
    validate_identity_owner_candidate(candidate, stable_map=stable_map)
    return candidate


def validate_identity_owner_candidate(
    candidate: Mapping, *, stable_map: Mapping
) -> dict:
    features, _ = _validate_stable_map(stable_map)
    fine_ids = _stable_identity_registry(stable_map)
    assert_base_fields_bit_exact(stable_map, candidate)
    prototypes = torch.as_tensor(candidate.get(PROTOTYPE_FEATURE_FIELD)).float().cpu()
    owners = torch.as_tensor(candidate.get(PROTOTYPE_OWNER_FIELD)).long().cpu()
    metadata = candidate.get(METADATA_FIELD)
    if not (
        prototypes.ndim == 2
        and prototypes.shape == (owners.numel(), features.shape[1])
        and owners.numel() > 0
        and torch.unique(owners).numel() == owners.numel()
        and int(owners.min()) >= 0
        and int(owners.max()) < features.shape[0]
        and bool(torch.isfinite(prototypes).all())
        and torch.allclose(
            torch.linalg.vector_norm(prototypes, dim=1),
            torch.ones(owners.numel()),
            atol=1e-6,
            rtol=1e-5,
        )
        and isinstance(metadata, Mapping)
        and metadata.get("schema") == CANDIDATE_SCHEMA
        and metadata.get("version") == VERSION
        and metadata.get("protocol") == "test_adapted"
        and metadata.get("uses_test_queries") is True
        and metadata.get("test_adapted") is True
        and metadata.get("formation_role") == "adaptation"
        and metadata.get("complete_adaptation_registry_consumed") is True
        and metadata.get("adaptation_features_consumed") is True
        and metadata.get("control_features_consumed") is False
        and metadata.get("confirmation_features_consumed") is False
        and metadata.get("control_or_confirmation_outcomes_consumed") is False
        and metadata.get("identity_evidence_authority")
        == "provisional_candidate_only"
        and metadata.get("identity_truth_claimed") is False
        and metadata.get("negative_anchor_labels_created") is False
        and metadata.get("base_anchor_fields_bit_exact") is True
        and metadata.get("base_anchor_descriptors_retained") is True
        and metadata.get("base_anchor_features_moved_or_lowered") is False
        and metadata.get("geometry_changed") is False
        and metadata.get("matching_semantics") == MATCHING_SEMANTICS
        and metadata.get("prototype_feature_source") == PROTOTYPE_FEATURE_SOURCE
        and metadata.get("deployment_authorized") is False
        and metadata.get("controller_authorized") is False
        and metadata.get("heldout_control_required") is True
        and metadata.get("heldout_confirmation_required") is True
        and metadata.get("confirmation_evaluation_authorized") is False
        and int(metadata.get("accepted_identity_count", -1)) == owners.numel()
        and int(metadata.get("added_prototype_count", -1)) == owners.numel()
        and int(metadata.get("maximum_prototypes_per_fine_identity", -1)) == 1
        and owners.numel() <= int(metadata.get("maximum_total_prototypes", 0))
        and metadata.get("prototype_features_sha256")
        == tensor_sha256(prototypes)
        and metadata.get("prototype_owner_rows_sha256") == tensor_sha256(owners)
    ):
        raise ValueError("V21 identity owner-prototype candidate contract is invalid")
    threshold = metadata.get("prototype_activation_threshold")
    if threshold is not None and (
        not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("V21 identity prototype activation threshold is invalid")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 identity candidate input lineage is missing")
    for field in ("stable_map", "identity_calibration", "split_manifest"):
        _canonical_source(inputs.get(field), label=f"identity candidate {field}")
    for field in ("adaptation_caches", "producer_sources"):
        sources = inputs.get(field)
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"V21 identity candidate {field} lineage is empty")
        identities = [
            _source_identity(value, label=f"identity candidate {field}")
            for value in sources
        ]
        if len(set(identities)) != len(identities):
            raise ValueError(f"V21 identity candidate {field} lineage is duplicated")
    actions = metadata.get("selected_actions")
    thresholds = metadata.get("calibration_thresholds")
    if not isinstance(actions, list) or not isinstance(thresholds, Mapping):
        raise ValueError("V21 identity candidate action/threshold registry is missing")
    required_thresholds = {
        "minimum_adaptation_blocks": 3,
        "minimum_adaptation_sequences": 2,
        "minimum_mapping_observations": 2,
        "minimum_mapping_families": 2,
        "minimum_descriptor_medoid_cosine": 0.8,
    }
    try:
        threshold_contract_valid = all(
            float(thresholds[field]) >= minimum
            for field, minimum in required_thresholds.items()
        )
    except (KeyError, TypeError, ValueError):
        threshold_contract_valid = False
    if not threshold_contract_valid:
        raise ValueError("V21 identity candidate strict thresholds are invalid")
    if len(actions) != owners.numel():
        raise ValueError("V21 identity candidate action count differs")
    seen_identities = set()
    action_queries = set()
    for index, action in enumerate(actions):
        identity = int(action.get("fine_identity_id", -1))
        owner = int(action.get("owner_anchor_row", -1))
        centroids = torch.as_tensor(action.get("block_centroids")).float().cpu()
        medoid = torch.as_tensor(action.get("stored_medoid")).float().cpu()
        block_ids = tuple(action.get("block_ids", ()))
        block_sequences = tuple(action.get("block_sequence_ids", ()))
        source_hashes = tuple(action.get("source_descriptor_sha256s", ()))
        centroid_hashes = tuple(action.get("block_centroid_sha256s", ()))
        medoid_index = next(
            (
                offset
                for offset, value in enumerate(centroid_hashes)
                if value == action.get("stored_medoid_sha256")
            ),
            -1,
        )
        recomputed = F.normalize(centroids.mean(0), dim=0) if centroids.ndim == 2 else medoid
        similarities = (
            centroids @ centroids.T
            if centroids.ndim == 2
            else torch.empty(0, 0)
        )
        recomputed_medoid_index = (
            int(similarities.mean(1).argmax()) if centroids.ndim == 2 else -1
        )
        recomputed_min_cosine = (
            float(similarities[recomputed_medoid_index].min())
            if recomputed_medoid_index >= 0 and centroids.shape[0] > 1
            else -1.0
        )
        source_queries = tuple(map(int, action.get("source_query_indices", ())))
        if not (
            int(action.get("prototype_index", -1)) == index
            and identity >= 0
            and identity not in seen_identities
            and owner == int(owners[index])
            and int(fine_ids[owner]) == identity
            and centroids.shape[0] == len(block_ids) == len(block_sequences)
            and centroids.shape[1:] == (features.shape[1],)
            and len(set(block_ids)) == len(block_ids)
            and tuple(sorted(block_ids)) == block_ids
            and len(set(block_sequences))
            >= int(thresholds["minimum_adaptation_sequences"])
            and len(block_ids) >= int(thresholds["minimum_adaptation_blocks"])
            and bool(torch.isfinite(centroids).all())
            and torch.allclose(
                torch.linalg.vector_norm(centroids, dim=1),
                torch.ones(centroids.shape[0]),
                atol=1e-6,
                rtol=1e-5,
            )
            and centroid_hashes == tuple(tensor_sha256(value) for value in centroids)
            and medoid_index >= 0
            and medoid_index == recomputed_medoid_index
            and torch.equal(medoid, centroids[medoid_index])
            and action.get("stored_medoid_sha256") == tensor_sha256(medoid)
            and torch.allclose(prototypes[index], recomputed, atol=1e-7, rtol=1e-6)
            and action.get("prototype_sha256") == tensor_sha256(prototypes[index])
            and action.get("prototype_sha256") not in source_hashes
            and all(
                isinstance(value, str)
                and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                for value in source_hashes
            )
            and int(action.get("source_edge_count", -1)) == len(source_hashes)
            and int(action.get("adaptation_block_count", -1)) == len(block_ids)
            and int(action.get("adaptation_sequence_count", -1))
            == len(set(block_sequences))
            and int(action.get("mapping_track_bank_observation_count", -1))
            >= int(thresholds["minimum_mapping_observations"])
            and int(action.get("mapping_track_bank_family_count", -1))
            >= int(thresholds["minimum_mapping_families"])
            and float(action.get("descriptor_medoid_min_cosine", -2.0))
            >= float(thresholds["minimum_descriptor_medoid_cosine"])
            and math.isclose(
                float(action.get("descriptor_medoid_min_cosine", math.nan)),
                recomputed_min_cosine,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            and math.isclose(
                float(action.get("prototype_to_stored_medoid_cosine", math.nan)),
                float(prototypes[index] @ medoid),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
            and int(action.get("promotion_wrong_top1_edge_count", -1)) >= 0
            and int(action.get("preservation_correct_top1_edge_count", -1)) >= 0
            and int(action["promotion_wrong_top1_edge_count"])
            + int(action["preservation_correct_top1_edge_count"])
            == int(action["source_edge_count"])
            and action.get("prototype_uses_multiple_blocks") is True
            and action.get("prototype_is_single_observation_copy") is False
            and source_queries == tuple(sorted(set(source_queries)))
            and source_queries
        ):
            raise ValueError("V21 identity owner-prototype action row is invalid")
        seen_identities.add(identity)
        action_queries.update(source_queries)
    declared_queries = torch.as_tensor(metadata.get("source_query_indices")).long()
    if not (
        torch.equal(
            declared_queries,
            torch.tensor(sorted(action_queries), dtype=torch.long),
        )
        and int(metadata.get("source_query_count", -1)) == len(action_queries)
    ):
        raise ValueError("V21 identity candidate source query registry differs")
    return dict(metadata)


def control_gate(summary: Mapping, *, role: str) -> dict:
    if role != "control":
        return {
            "evaluated": False,
            "passed": False,
            "reason": "evaluation_role_is_not_heldout_control",
            "policy": CONTROL_GATE_POLICY,
        }
    median = float(summary["paired_delta_task_error"]["median"])
    requirements = {
        "paired_r5_gain": int(summary["paired_r5_gain_count"])
        >= CONTROL_GATE_POLICY["minimum_paired_r5_gain_count"],
        "no_paired_r5_loss": int(summary["paired_r5_loss_count"])
        <= CONTROL_GATE_POLICY["maximum_paired_r5_loss_count"],
        "nonpositive_median_task_delta": median
        <= CONTROL_GATE_POLICY["maximum_median_paired_delta_task_error"],
    }
    passed = all(requirements.values())
    return {
        "evaluated": True,
        "passed": passed,
        "reason": "all_predeclared_requirements_pass" if passed else "control_gate_failed",
        "policy": CONTROL_GATE_POLICY,
        "requirements": requirements,
    }


def build_identity_owner_cached_evaluation(
    *,
    stable_map: Mapping,
    candidate: Mapping,
    cache_payloads: Sequence[Mapping],
    stable_map_source: Mapping,
    candidate_source: Mapping,
    cache_sources: Sequence[Mapping],
    producer_sources: Sequence[Mapping],
    matcher_chunk_size: int = 8192,
    device: str | torch.device = "cpu",
    solver: Callable = solve_absolute_pose,
) -> dict:
    features, xyz = _validate_stable_map(stable_map)
    metadata = validate_identity_owner_candidate(candidate, stable_map=stable_map)
    _same_source(
        metadata.get("inputs", {}).get("stable_map"),
        stable_map_source,
        label="identity candidate stable map",
    )
    payloads, records, baseline_contract = validate_complete_cache_payloads(
        cache_payloads
    )
    role = str(payloads[0]["role"])
    if role not in {"adaptation", "control"}:
        raise ValueError("V21 identity arm evaluates adaptation or control only")
    if len(payloads) != len(cache_sources):
        raise ValueError("V21 identity evaluation cache lineage differs")
    for payload in payloads:
        _same_source(
            payload.get("inputs", {}).get("stable_map"),
            stable_map_source,
            label="identity evaluation stable map",
        )
        _same_source(
            payload.get("inputs", {}).get("split_manifest"),
            metadata.get("inputs", {}).get("split_manifest"),
            label="identity evaluation split",
        )
        if payload.get("preprocessing_config_sha256") != metadata.get(
            "preprocessing_config_sha256"
        ):
            raise ValueError("V21 identity candidate/evaluation frontend differs")
    if baseline_contract != metadata.get("baseline_contract"):
        raise ValueError("V21 identity candidate/evaluation PoseLib contract differs")
    if role == "control":
        formation = {
            _source_identity(value, label="identity formation cache")
            for value in metadata["inputs"]["adaptation_caches"]
        }
        evaluation = {
            _source_identity(value, label="identity control cache")
            for value in cache_sources
        }
        if formation & evaluation:
            raise ValueError("V21 heldout control cache entered candidate formation")
    normalized_features = F.normalize(
        F.normalize(features.to(device=device, dtype=torch.float32), dim=1), dim=1
    )
    prototypes = torch.as_tensor(
        candidate[PROTOTYPE_FEATURE_FIELD], device=device
    ).float()
    owners = torch.as_tensor(candidate[PROTOTYPE_OWNER_FIELD], device=device).long()
    threshold = metadata.get("prototype_activation_threshold")
    evaluated = [
        evaluate_cached_record(
            record=record,
            anchor_features=normalized_features,
            anchor_xyz=xyz,
            extra_prototypes=prototypes,
            prototype_owner_rows=owners,
            baseline_contract=baseline_contract,
            matcher_chunk_size=matcher_chunk_size,
            device=device,
            anchor_features_normalized=True,
            prototype_activation_threshold=threshold,
            solver=solver,
        )
        for record in records
    ]
    summary = summarize_cached_evaluation(evaluated)
    gate = control_gate(summary, role=role)
    return {
        "schema": EVALUATION_SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "evaluation_role": role,
        "matching_semantics": MATCHING_SEMANTICS,
        "pose_solver_semantics": "single_standard_poselib_absolute_pose",
        "standard_r5_definition_inherited": True,
        "candidate_formation_feedback_consumed": False,
        "heldout_outcomes_feed_candidate": False,
        "candidate_map_mutated": False,
        "deployment_authorized": False,
        "catastrophe_definition": "baseline_r5_success_to_candidate_r5_failure",
        "prototype_activation_threshold": threshold,
        "inputs": {
            "stable_map": dict(stable_map_source),
            "candidate_map": dict(candidate_source),
            "frontend_caches": [dict(value) for value in cache_sources],
            "producer_sources": [dict(value) for value in producer_sources],
        },
        "candidate_formation_role": "adaptation",
        "candidate_source_query_indices": metadata["source_query_indices"].clone(),
        "evaluation_query_count": len(evaluated),
        "baseline_contract": baseline_contract,
        "matcher_chunk_size": int(matcher_chunk_size),
        "records": evaluated,
        "summary": summary,
        "control_gate": gate,
        "confirmation_evaluation_authorized": bool(gate["passed"]),
    }


def validate_identity_owner_cached_evaluation(payload: Mapping) -> None:
    if not (
        payload.get("schema") == EVALUATION_SCHEMA
        and payload.get("version") == VERSION
        and payload.get("evaluation_role") in {"adaptation", "control"}
        and payload.get("deployment_authorized") is False
        and payload.get("heldout_outcomes_feed_candidate") is False
        and payload.get("candidate_map_mutated") is False
    ):
        raise ValueError("V21 identity cached evaluation contract is invalid")
    proxy = dict(payload)
    proxy["schema"] = BASE_EVALUATION_SCHEMA
    proxy.pop("prototype_activation_threshold", None)
    proxy.pop("control_gate", None)
    proxy.pop("confirmation_evaluation_authorized", None)
    proxy_inputs = dict(proxy["inputs"])
    proxy_inputs.pop("producer_sources", None)
    proxy["inputs"] = proxy_inputs
    validate_base_cached_evaluation(proxy)
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 identity evaluation lineage is missing")
    producers = inputs.get("producer_sources")
    if not isinstance(producers, list) or not producers:
        raise ValueError("V21 identity evaluation producer lineage is empty")
    producer_ids = [
        _source_identity(value, label="identity evaluation producer")
        for value in producers
    ]
    if len(set(producer_ids)) != len(producer_ids):
        raise ValueError("V21 identity evaluation producer lineage is duplicated")
    expected_gate = control_gate(
        payload["summary"], role=str(payload["evaluation_role"])
    )
    if (
        payload.get("control_gate") != expected_gate
        or payload.get("confirmation_evaluation_authorized")
        is not bool(expected_gate["passed"])
    ):
        raise ValueError("V21 identity evaluation control gate differs")


__all__ = [
    "CANDIDATE_SCHEMA",
    "CONTROL_GATE_POLICY",
    "EVALUATION_SCHEMA",
    "MATCHING_SEMANTICS",
    "METADATA_FIELD",
    "VERSION",
    "assert_base_fields_bit_exact",
    "build_identity_owner_cached_evaluation",
    "build_identity_owner_prototype_candidate",
    "control_gate",
    "derive_verified_identity_actions",
    "validate_identity_owner_cached_evaluation",
    "validate_identity_owner_candidate",
]
