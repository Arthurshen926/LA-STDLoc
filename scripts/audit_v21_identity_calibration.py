#!/usr/bin/env python3
"""Audit recurrent V21 adaptation identities and emit positive-only CSR."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from map_learning.v21_correspondence_truth import (
    STATUS_UNIQUE,
    resolve_teacher_action,
    validate_correspondence_payload,
)
from map_learning.v21_identity_calibration import (
    ROLE,
    SCHEMA,
    SEMANTICS,
    VERSION,
    accepted_identity_mask,
    aggregate_record_counts,
    atomic_torch_save_fresh,
    build_identity_evidence,
    build_query_provisional_record,
    calibration_thresholds,
    mapping_identity_support,
    sha256_json,
    threshold_grid_summary,
)
from map_learning.v21_test_cache import (
    tensor_sha256,
    validate_cache_payload,
    validate_shard_registry,
    validate_split_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_SOURCES = (
    "map_learning/v21_correspondence_truth.py",
    "map_learning/v21_identity_calibration.py",
    "map_learning/v21_test_cache.py",
    "map_learning/v21_test_protocol.py",
    "scripts/audit_v21_identity_calibration.py",
)


def _source(path: str | Path, *, expected_sha256: str | None = None) -> dict:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"V21 identity calibration source SHA256 differs: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": int(resolved.stat().st_size),
    }


def _verify_sources(sources: list[dict[str, Any]]) -> None:
    for source in sources:
        path = Path(str(source["path"]))
        if (
            not path.is_file()
            or int(path.stat().st_size) != int(source["size_bytes"])
            or sha256_file(path) != source["sha256"]
        ):
            raise RuntimeError(f"V21 identity calibration source changed: {path}")


def _same_source(left: Any, right: dict, *, label: str) -> None:
    if not isinstance(left, dict):
        raise ValueError(f"V21 identity calibration {label} lineage is absent")
    if (
        Path(str(left.get("path", ""))).expanduser().resolve()
        != Path(right["path"])
        or left.get("sha256") != right["sha256"]
        or int(left.get("size_bytes", -1)) != right["size_bytes"]
    ):
        raise ValueError(f"V21 identity calibration {label} lineage differs")


def _compact_truth_records(payload: dict) -> dict[int, dict]:
    output = {}
    for record in payload["records"]:
        query = int(record["query_index"])
        status = torch.as_tensor(record["diagnostic_truth_status"]).long().cpu()
        rows = torch.where(status == STATUS_UNIQUE)[0]
        offsets = torch.as_tensor(
            record["diagnostic_positive_offsets"]
        ).long().cpu()
        anchors = torch.as_tensor(
            record["diagnostic_positive_anchor_rows"]
        ).long().cpu()
        if rows.numel() and bool((offsets[rows + 1] - offsets[rows] != 1).any()):
            raise ValueError("V21 diagnostic UNIQUE row does not have one Anchor")
        if query in output:
            raise ValueError("V21 correspondence query is duplicated")
        output[query] = {
            "query_index": query,
            "image_name": str(record["image_name"]),
            "image_sha256": str(record["image_sha256"]),
            "sequence_id": str(record["sequence_id"]),
            "frame_index": int(record["frame_index"]),
            "block_id": str(record["block_id"]),
            "source_record_sha256": str(record["source_record_sha256"]),
            "pose_w2c_sha256": str(record["pose_w2c_sha256"]),
            "keypoint_count": int(record["keypoint_count"]),
            "keypoints_sha256": str(record["keypoints_sha256"]),
            "descriptors_sha256": str(record["descriptors_sha256"]),
            "diagnostic_unique_query_rows": rows.contiguous(),
            "diagnostic_unique_anchor_rows": anchors[offsets[rows]].contiguous(),
        }
    return output


def _compact_frontend_record(frontend: dict, truth: dict) -> dict:
    identity_fields = (
        "query_index",
        "image_name",
        "image_sha256",
        "sequence_id",
        "frame_index",
        "block_id",
        "source_record_sha256",
        "pose_w2c_sha256",
    )
    if any(frontend.get(name) != truth.get(name) for name in identity_fields):
        raise ValueError("V21 correspondence/frontend query identities differ")
    keypoints = torch.as_tensor(frontend["keypoints"]).float().cpu()
    descriptors = torch.as_tensor(frontend["descriptors"]).float().cpu()
    count = int(truth["keypoint_count"])
    if (
        keypoints.shape != (count, 2)
        or descriptors.ndim != 2
        or descriptors.shape[0] != count
        or tensor_sha256(keypoints) != truth["keypoints_sha256"]
        or tensor_sha256(descriptors) != truth["descriptors_sha256"]
    ):
        raise ValueError("V21 correspondence/frontend feature registries differ")
    rows = truth["diagnostic_unique_query_rows"]
    winners = torch.as_tensor(frontend["winner_anchor_rows"]).long().cpu()[rows]
    winner_scores = torch.as_tensor(frontend["winner_scores"]).float().cpu()[rows]
    inlier_mask = torch.zeros(count, dtype=torch.bool)
    inlier_rows = torch.as_tensor(frontend["baseline_inliers"]).long().cpu()
    inlier_mask[inlier_rows] = True
    compact = dict(truth)
    compact.update(
        {
            "descriptors": F.normalize(descriptors[rows], dim=1).contiguous(),
            "winner_anchor_rows": winners.contiguous(),
            "winner_scores": winner_scores.contiguous(),
            "baseline_inlier": inlier_mask[rows].contiguous(),
            "baseline_r5": bool(frontend["baseline_r5"]),
        }
    )
    return compact


def _descriptor_quantiles(values: torch.Tensor) -> dict:
    tensor = torch.as_tensor(values).float().cpu()
    if tensor.numel() == 0:
        return {}
    probabilities = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    quantiles = torch.quantile(tensor, probabilities)
    return {
        label: float(value)
        for label, value in zip(
            ("min", "p10", "p25", "median", "p75", "p90", "max"),
            quantiles.tolist(),
        )
    }


@torch.inference_mode()
def materialize(args: argparse.Namespace) -> dict:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    truth_source = _source(
        args.correspondence_truth,
        expected_sha256=args.expected_correspondence_truth_sha256,
    )
    cache_sources = [
        _source(path) for path in sorted(set(map(str, args.frontend_cache)))
    ]
    map_source = _source(
        args.stable_map, expected_sha256=args.expected_stable_map_sha256
    )
    split_source = _source(
        args.split_manifest, expected_sha256=args.expected_split_manifest_sha256
    )
    provenance_source = _source(
        args.mapping_provenance,
        expected_sha256=args.expected_mapping_provenance_sha256,
    )
    teacher_source = _source(
        args.teacher_validation,
        expected_sha256=args.expected_teacher_validation_sha256,
    )
    producer_sources = [_source(REPOSITORY_ROOT / path) for path in PRODUCER_SOURCES]
    all_sources = [
        truth_source,
        *cache_sources,
        map_source,
        split_source,
        provenance_source,
        teacher_source,
        *producer_sources,
    ]

    with Path(split_source["path"]).open("r", encoding="utf-8") as stream:
        split_manifest = json.load(stream)
    adaptation_split = validate_split_manifest(split_manifest, role=ROLE)
    split_role_counts = {
        role: sum(record["role"] == role for record in split_manifest["records"])
        for role in ("adaptation", "control", "confirmation", "embargo")
    }

    correspondence = torch.load(
        truth_source["path"], map_location="cpu", weights_only=False
    )
    validate_correspondence_payload(correspondence)
    if not (
        correspondence.get("role") == ROLE
        and correspondence.get("action_authorized") is False
        and correspondence.get("planner_diagnostic_consumers_allowed") is True
        and correspondence.get("teacher_action_decision", {}).get("tier_name")
        == "tier_c"
        and correspondence.get("teacher_action_decision", {}).get(
            "requested_action"
        )
        == "planner_priority"
    ):
        raise ValueError("V21 calibration requires blocked Tier-C planner diagnostics")
    _same_source(
        correspondence["inputs"]["stable_map"], map_source, label="stable map"
    )
    _same_source(
        correspondence["inputs"]["mapping_provenance"],
        provenance_source,
        label="mapping provenance",
    )
    _same_source(
        correspondence["inputs"]["teacher_validation"],
        teacher_source,
        label="teacher validation",
    )
    expected_cache_lineage = sorted(
        correspondence["inputs"]["frontend_caches"],
        key=lambda value: str(value["path"]),
    )
    if len(expected_cache_lineage) != len(cache_sources):
        raise ValueError("V21 correspondence/frontend cache source count differs")
    for expected, actual in zip(expected_cache_lineage, cache_sources):
        _same_source(expected, actual, label="frontend cache")
    registry = dict(correspondence["frontend_shard_registry"])
    validate_shard_registry(registry)
    if registry.get("split_manifest_sha256") != split_source["sha256"]:
        raise ValueError("V21 correspondence/split SHA lineage differs")
    expected_anchor_count = int(correspondence["anchor_count"])
    truth_by_query = _compact_truth_records(correspondence)
    del correspondence
    gc.collect()

    compact_by_query: dict[int, dict] = {}
    seen_shards = set()
    descriptor_dim = None
    for cache_source in cache_sources:
        cache = torch.load(cache_source["path"], map_location="cpu", weights_only=False)
        validate_cache_payload(cache)
        if cache.get("role") != ROLE or cache.get("shard_registry") != registry:
            raise ValueError("V21 calibration accepts complete adaptation caches only")
        if int(cache.get("anchor_count", -1)) != expected_anchor_count:
            raise ValueError("V21 correspondence/frontend Anchor counts differ")
        _same_source(
            cache["inputs"]["stable_map"], map_source, label="cache stable map"
        )
        _same_source(
            cache["inputs"]["split_manifest"], split_source, label="cache split"
        )
        shard = int(cache["shard_index"])
        if shard in seen_shards:
            raise ValueError("V21 calibration frontend shard is duplicated")
        seen_shards.add(shard)
        descriptor_dim = descriptor_dim or int(cache["descriptor_dim"])
        if int(cache["descriptor_dim"]) != descriptor_dim:
            raise ValueError("V21 calibration descriptor dimensions differ")
        for frontend in cache["records"]:
            query_index = int(frontend["query_index"])
            truth = truth_by_query.get(query_index)
            if truth is None or query_index in compact_by_query:
                raise ValueError("V21 calibration frontend query registry differs")
            compact_by_query[query_index] = _compact_frontend_record(frontend, truth)
        del cache
        gc.collect()
    if seen_shards != set(range(int(registry["shard_count"]))):
        raise ValueError("V21 calibration requires all adaptation cache shards")
    registry_rows = sorted(registry["rows"], key=lambda value: int(value["ordinal"]))
    ordered_queries = [int(row["query_index"]) for row in registry_rows]
    if (
        set(compact_by_query) != set(truth_by_query)
        or set(compact_by_query) != set(ordered_queries)
        or len(adaptation_split) != len(ordered_queries)
        or [int(record["query_index"]) for record in adaptation_split]
        != sorted(ordered_queries)
    ):
        raise ValueError("V21 calibration did not consume the complete adaptation prefix")
    compact_records = [compact_by_query[query] for query in ordered_queries]
    del truth_by_query, compact_by_query
    gc.collect()

    stable_map = torch.load(map_source["path"], map_location="cpu", weights_only=False)
    observations = stable_map.get("projective_anchor_observations")
    anchor_fine_ids = torch.as_tensor(stable_map.get("fine_identity_ids")).long()
    anchor_features = torch.as_tensor(stable_map.get("anchor_features")).float()
    anchor_count = int(anchor_fine_ids.numel())
    if not (
        stable_map.get("schema") == "lafgs_materialized_anchor_map"
        and isinstance(observations, dict)
        and observations.get("schema") == "lafgs_projective_anchor_observations"
        and anchor_count == int(torch.as_tensor(stable_map["anchor_ids"]).numel())
        and anchor_count == expected_anchor_count
        and anchor_count > 0
        and anchor_features.shape == (anchor_count, int(descriptor_dim))
        and not bool((anchor_fine_ids < 0).any())
    ):
        raise ValueError("V21 calibration frozen stable-map contract differs")
    normalized_anchor_features = F.normalize(anchor_features, dim=1)
    for record in compact_records:
        anchors = record["diagnostic_unique_anchor_rows"]
        winners = record["winner_anchor_rows"]
        if anchors.numel() and (
            int(anchors.min()) < 0
            or int(anchors.max()) >= anchor_count
            or int(winners.min()) < 0
            or int(winners.max()) >= anchor_count
        ):
            raise ValueError("V21 diagnostic/winner Anchor row is outside the map")
        descriptors = record["descriptors"]
        truth_scores = (descriptors * normalized_anchor_features[anchors]).sum(1)
        recomputed_winner = (descriptors * normalized_anchor_features[winners]).sum(1)
        if not torch.allclose(
            recomputed_winner, record["winner_scores"], atol=2e-5, rtol=1e-5
        ):
            raise ValueError("V21 cached Top1 scores differ from the frozen map")
        record["diagnostic_unique_fine_identity_ids"] = anchor_fine_ids[
            anchors
        ].clone()
        record["winner_fine_identity_ids"] = anchor_fine_ids[winners].clone()
        record["truth_anchor_scores"] = truth_scores.contiguous()
    observation_offsets = torch.as_tensor(
        observations["observation_offsets"]
    ).long().clone()
    observation_queries = torch.as_tensor(
        observations["query_indices"]
    ).long().clone()
    mapping_query_names = list(stable_map.get("v6_mapping_query_names", ()))
    anchor_fine_ids = anchor_fine_ids.clone()
    del stable_map, observations, anchor_features, normalized_anchor_features
    gc.collect()

    fine_ids = torch.cat(
        [record["diagnostic_unique_fine_identity_ids"] for record in compact_records]
    )
    anchor_rows = torch.cat(
        [record["diagnostic_unique_anchor_rows"] for record in compact_records]
    )
    descriptors = torch.cat([record["descriptors"] for record in compact_records])
    query_indices = torch.cat(
        [
            torch.full(
                (record["diagnostic_unique_query_rows"].numel(),),
                int(record["query_index"]),
                dtype=torch.long,
            )
            for record in compact_records
        ]
    )
    wrong_anchor = torch.cat(
        [
            record["diagnostic_unique_anchor_rows"] != record["winner_anchor_rows"]
            for record in compact_records
        ]
    )
    wrong_fine = torch.cat(
        [
            record["diagnostic_unique_fine_identity_ids"]
            != record["winner_fine_identity_ids"]
            for record in compact_records
        ]
    )
    baseline_failure = torch.cat(
        [
            torch.full(
                (record["diagnostic_unique_query_rows"].numel(),),
                not bool(record["baseline_r5"]),
                dtype=torch.bool,
            )
            for record in compact_records
        ]
    )
    baseline_inlier = torch.cat(
        [record["baseline_inlier"] for record in compact_records]
    )
    truth_scores = torch.cat(
        [record["truth_anchor_scores"] for record in compact_records]
    )
    winner_scores = torch.cat([record["winner_scores"] for record in compact_records])
    block_ids = [
        str(record["block_id"])
        for record in compact_records
        for _ in range(record["diagnostic_unique_query_rows"].numel())
    ]
    sequence_ids = [
        str(record["sequence_id"])
        for record in compact_records
        for _ in range(record["diagnostic_unique_query_rows"].numel())
    ]
    if fine_ids.numel() == 0:
        raise ValueError("V21 Tier-C diagnostic UNIQUE registry is empty")
    target_identities = torch.unique(fine_ids, sorted=True)

    provenance = torch.load(
        provenance_source["path"], map_location="cpu", weights_only=False
    )
    if not (
        provenance.get("schema")
        == "lafgs_v18_mapping_observation_gaussian_provenance"
        and provenance.get("uses_test_queries") is False
        and provenance.get("descriptor_independent") is True
        and int(provenance.get("anchor_count", -1)) == anchor_count
        and int(provenance.get("global_observation_count", -1))
        == observation_queries.numel()
        and list(provenance.get("mapping_query_names", ())) == mapping_query_names
        and provenance.get("inputs", {}).get("anchor_map_sha256")
        == map_source["sha256"]
        and Path(str(provenance.get("inputs", {}).get("anchor_map", "")))
        .expanduser()
        .resolve()
        == Path(map_source["path"])
    ):
        raise ValueError("V21 calibration mapping provenance contract differs")
    provenance_rows = torch.as_tensor(provenance["observation_rows"]).long().clone()
    provenance_valid = torch.as_tensor(provenance["observation_valid"]).bool().clone()
    mapping_families = torch.as_tensor(
        provenance["mapping_view_family_ids"]
    ).long().clone()
    del provenance
    gc.collect()

    teacher = torch.load(
        teacher_source["path"], map_location="cpu", weights_only=False
    )
    action = resolve_teacher_action(
        teacher, tier_name="tier_c", requested_action="planner_priority"
    )
    teacher_inputs = teacher.get("inputs", {})
    if not (
        action.get("planner_diagnostic_authorized") is True
        and action.get("action_authorized") is False
        and teacher_inputs.get("anchor_map_sha256") == map_source["sha256"]
        and teacher_inputs.get("mapping_provenance_sha256")
        == provenance_source["sha256"]
    ):
        raise ValueError("V21 calibration V19 teacher lineage differs")
    family_roles = dict(teacher["family_roles"])
    del teacher
    gc.collect()

    mapping_support = mapping_identity_support(
        target_fine_identity_ids=target_identities,
        anchor_fine_identity_ids=anchor_fine_ids,
        observation_offsets=observation_offsets,
        observation_query_indices=observation_queries,
        provenance_observation_rows=provenance_rows,
        provenance_observation_valid=provenance_valid,
        mapping_view_family_ids=mapping_families,
        family_roles=family_roles,
    )
    evidence = build_identity_evidence(
        observation_fine_identity_ids=fine_ids,
        observation_anchor_rows=anchor_rows,
        observation_descriptors=descriptors,
        observation_query_indices=query_indices,
        observation_block_ids=block_ids,
        observation_sequence_ids=sequence_ids,
        wrong_top1_anchor_row=wrong_anchor,
        wrong_top1_fine_identity=wrong_fine,
        baseline_failure=baseline_failure,
        baseline_inlier=baseline_inlier,
        truth_anchor_scores=truth_scores,
        winner_scores=winner_scores,
        mapping_support=mapping_support,
    )
    thresholds = calibration_thresholds(
        minimum_adaptation_blocks=args.minimum_adaptation_blocks,
        minimum_adaptation_sequences=args.minimum_adaptation_sequences,
        minimum_mapping_observations=args.minimum_mapping_observations,
        minimum_mapping_families=args.minimum_mapping_families,
        minimum_descriptor_medoid_cosine=args.minimum_descriptor_medoid_cosine,
    )
    accepted = accepted_identity_mask(evidence, thresholds)
    evidence["accepted_identity_mask"] = accepted
    records = [
        build_query_provisional_record(
            query=record,
            evidence=evidence,
            accepted_identities=accepted,
        )
        for record in compact_records
    ]
    counts = aggregate_record_counts(records)
    grid = threshold_grid_summary(
        evidence=evidence,
        observation_fine_identity_ids=fine_ids,
        observation_query_indices=query_indices,
        wrong_top1_fine_identity=wrong_fine,
        baseline_failure=baseline_failure,
        block_minimums=args.grid_adaptation_blocks,
        sequence_minimums=args.grid_adaptation_sequences,
        mapping_observation_minimums=args.grid_mapping_observations,
        mapping_family_minimums=args.grid_mapping_families,
        descriptor_cosine_minimums=args.grid_descriptor_cosines,
    )
    available = counts["provisional_positive_edge_count"] > 0
    cross_block = torch.as_tensor(evidence["descriptor_cross_block_defined"]).bool()
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": ROLE,
        "complete_adaptation_registry_consumed": True,
        "formation_stage": "after_complete_adaptation_before_control_scoring",
        "control_or_confirmation_features_consumed": False,
        "control_or_confirmation_outcomes_consumed": False,
        "split_metadata_only_for_heldout_roles": True,
        "source_teacher_mutating_action_authorized": False,
        "provisional_action_positive_only": True,
        "negative_labels_created": False,
        "ambiguous_or_unlabelled_are_negative": False,
        "artifact_writes_map": False,
        "candidate_deployment_authorized": False,
        "heldout_control_required_before_confirmation": True,
        "heldout_confirmation_required_before_deployment": True,
        "quarantined_candidate_generation_allowed": available,
        "provisional_action_positive_available": available,
        "calibration_decision": (
            "GO_PROVISIONAL_HELDOUT_REQUIRED" if available else "STOP_NO_COVERAGE"
        ),
        "semantics": SEMANTICS,
        "correspondence_truth_sha256": truth_source["sha256"],
        "stable_map_sha256": map_source["sha256"],
        "split_manifest_sha256": split_source["sha256"],
        "mapping_provenance_sha256": provenance_source["sha256"],
        "teacher_validation_sha256": teacher_source["sha256"],
        "frontend_shard_registry": registry,
        "frontend_shard_registry_sha256": registry["registry_sha256"],
        "anchor_count": anchor_count,
        "descriptor_dim": int(descriptor_dim),
        "query_count": len(records),
        "diagnostic_identity_count": int(target_identities.numel()),
        "accepted_identity_count": int(accepted.sum()),
        "candidate_thresholds": thresholds,
        "candidate_thresholds_sha256": sha256_json(thresholds),
        "threshold_selection": (
            "explicit_cli_gate_using_complete_adaptation_only; heldout roles unused"
        ),
        "threshold_grid": grid,
        "split_role_counts": split_role_counts,
        "descriptor_cross_block_min_cosine_quantiles": _descriptor_quantiles(
            torch.as_tensor(evidence["descriptor_medoid_min_cosine"])[cross_block]
        ),
        "counts": counts,
        "identity_evidence": evidence,
        "inputs": {
            "correspondence_truth": truth_source,
            "frontend_caches": cache_sources,
            "stable_map": map_source,
            "split_manifest": split_source,
            "mapping_provenance": provenance_source,
            "teacher_validation": teacher_source,
            "producer_sources": producer_sources,
        },
        "records": records,
    }
    _verify_sources(all_sources)
    atomic_torch_save_fresh(payload, output)
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "calibration_decision": payload["calibration_decision"],
        "candidate_thresholds": thresholds,
        "diagnostic_identity_count": int(target_identities.numel()),
        "accepted_identity_count": int(accepted.sum()),
        "counts": counts,
        "threshold_grid": grid,
        "descriptor_cross_block_min_cosine_quantiles": payload[
            "descriptor_cross_block_min_cosine_quantiles"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correspondence-truth", required=True)
    parser.add_argument("--frontend-cache", action="append", required=True)
    parser.add_argument("--stable-map", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--mapping-provenance", required=True)
    parser.add_argument("--teacher-validation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-correspondence-truth-sha256")
    parser.add_argument("--expected-stable-map-sha256")
    parser.add_argument("--expected-split-manifest-sha256")
    parser.add_argument("--expected-mapping-provenance-sha256")
    parser.add_argument("--expected-teacher-validation-sha256")
    parser.add_argument("--minimum-adaptation-blocks", type=int, default=3)
    parser.add_argument("--minimum-adaptation-sequences", type=int, default=2)
    parser.add_argument("--minimum-mapping-observations", type=int, default=2)
    parser.add_argument("--minimum-mapping-families", type=int, default=2)
    parser.add_argument(
        "--minimum-descriptor-medoid-cosine", type=float, default=0.8
    )
    parser.add_argument(
        "--grid-adaptation-blocks", type=int, nargs="+", default=[2, 3]
    )
    parser.add_argument(
        "--grid-adaptation-sequences", type=int, nargs="+", default=[1, 2]
    )
    parser.add_argument(
        "--grid-mapping-observations", type=int, nargs="+", default=[2]
    )
    parser.add_argument(
        "--grid-mapping-families", type=int, nargs="+", default=[2, 3]
    )
    parser.add_argument(
        "--grid-descriptor-cosines",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(materialize(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
