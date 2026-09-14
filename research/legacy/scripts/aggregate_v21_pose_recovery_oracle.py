#!/usr/bin/env python3
"""Strictly aggregate a complete V21 pose-recovery oracle shard registry.

This is a diagnostic pose-recovery upper bound.  Aggregation never turns
geometry or Track-consensus evidence into a deployable controller action.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import uuid

import torch

from common.hashing import sha256_file
from map_learning.v21_correspondence_truth import (
    validate_correspondence_payload,
)
from map_learning.v21_gaussian_support import validate_support_payload
from map_learning.v21_pose_leverage import (
    GAUSSIAN_GEOMETRY_SOURCE,
    TRACK_CONSENSUS_DIAGNOSTIC,
    summarize_pose_recovery,
)
from map_learning.v21_test_cache import tensor_sha256, validate_cache_payload


SHARD_SCHEMA = "lafgs_v21_pose_recovery_oracle_shard"
OUTPUT_SCHEMA = "lafgs_v21_pose_recovery_oracle_aggregate"
VERSION = 1
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
POSITIVE_SOURCES = frozenset(
    {GAUSSIAN_GEOMETRY_SOURCE, TRACK_CONSENSUS_DIAGNOSTIC}
)


def _source_identity(value: object, *, label: str) -> tuple[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"V21 {label} source identity is missing")
    raw_path = str(value.get("path", ""))
    digest = str(value.get("sha256", ""))
    if not raw_path or SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"V21 {label} source identity is invalid")
    resolved = str(Path(raw_path).expanduser().resolve())
    if resolved != raw_path:
        raise ValueError(f"V21 {label} path is not canonical")
    return resolved, digest


def _oracle_source_contract(payload: dict) -> tuple[str, str]:
    declared = payload.get("positive_source")
    if declared is None:
        return GAUSSIAN_GEOMETRY_SOURCE, "legacy_geometry_v1"
    source = str(declared)
    if source not in POSITIVE_SOURCES:
        raise ValueError("V21 pose-recovery positive source is unsupported")
    return source, "explicit_positive_source_v1"


def _validate_oracle_envelope(payload: dict) -> tuple[str, str]:
    records = payload.get("records")
    summary = payload.get("summary")
    inputs = payload.get("input")
    positive_source, source_contract = _oracle_source_contract(payload)
    legacy = source_contract == "legacy_geometry_v1"
    common_contract = (
        payload.get("schema") == SHARD_SCHEMA
        and payload.get("version") == VERSION
        and payload.get("protocol") == "test_adapted"
        and payload.get("uses_test_queries") is True
        and payload.get("role") == "adaptation"
        and payload.get("loo_used") is False
        and payload.get("ground_truth_pose_is_feedback_authority") is True
        and payload.get("topk_is_candidate_mining_only") is True
        and payload.get("unlabeled_rows_are_negative") is False
        and int(payload.get("negative_anchor_label_count", -1)) == 0
        and payload.get("gaussian_support_is_geometry_only") is True
        and payload.get("correspondence_identity_authority_present") is False
        and payload.get("controller_authorized_query_count_must_be_zero") is True
        and isinstance(inputs, dict)
        and isinstance(payload.get("parameters"), dict)
        and isinstance(records, list)
        and isinstance(summary, dict)
        and int(summary.get("query_count", -1)) == len(records)
        and int(summary.get("controller_authorized_query_count", -1)) == 0
    )
    if legacy:
        source_contract_valid = (
            payload.get("all_action_authority_is_exact_poselib") is True
            and inputs.get("correspondence_truth") is None
            and all("positive_source" not in record for record in records)
            and summary == summarize_pose_recovery(records)
        )
    else:
        source_contract_valid = (
            payload.get("all_pose_recovery_claims_use_exact_poselib") is True
            and payload.get("exact_poselib_is_controller_action_authority") is False
            and all(
                record.get("positive_source") == positive_source
                for record in records
            )
            and summary
            == summarize_pose_recovery(records, positive_source=positive_source)
            and (
                (positive_source == TRACK_CONSENSUS_DIAGNOSTIC)
                == isinstance(inputs.get("correspondence_truth"), dict)
            )
        )
        if positive_source == TRACK_CONSENSUS_DIAGNOSTIC:
            source_contract_valid = source_contract_valid and (
                payload.get("track_consensus_identity_evidence_present") is True
                and payload.get(
                    "track_consensus_identity_evidence_is_deployment_authority"
                )
                is False
                and payload.get("track_consensus_diagnostic_is_action_authority")
                is False
            )
    if not (
        common_contract and source_contract_valid
    ):
        raise ValueError("V21 pose-recovery oracle shard contract is invalid")
    shard_count = int(payload.get("shard_count", 0))
    shard_index = int(payload.get("shard_index", -1))
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("V21 pose-recovery oracle shard coordinate is invalid")
    if int(payload.get("source_query_count", 0)) <= 0:
        raise ValueError("V21 pose-recovery oracle source count is invalid")
    queries = set()
    for record in records:
        baseline = record.get("baseline") if isinstance(record, dict) else None
        query_index = int(record.get("query_index", -1)) if isinstance(record, dict) else -1
        if (
            query_index < 0
            or query_index in queries
            or not isinstance(baseline, dict)
            or not isinstance(baseline.get("r5_success"), bool)
            or not str(record.get("route", ""))
            or bool(record.get("controller_authorized", False))
            or not str(record.get("image_name", ""))
            or not str(record.get("sequence_id", ""))
            or not str(record.get("block_id", ""))
        ):
            raise ValueError("V21 pose-recovery oracle record is invalid")
        queries.add(query_index)
    return positive_source, source_contract


def _diagnostic_truth_row_count(record: dict) -> int:
    legal = record.get("legal_positive_csr", {})
    return int(
        legal.get(
            "source_decisive_row_count",
            legal.get("legal_positive_row_count", 0),
        )
    )


def _distribution(records: list[dict], key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(record)
    output = {}
    for name, rows in sorted(grouped.items()):
        failures = [row for row in rows if not row["baseline"]["r5_success"]]
        recovered = [
            row
            for row in failures
            if isinstance(row.get("one_assignment_lower_bound"), dict)
            and row["one_assignment_lower_bound"].get("r5_success") is True
        ]
        output[name] = {
            "query_count": len(rows),
            "baseline_r5_success_count": len(rows) - len(failures),
            "baseline_failure_count": len(failures),
            "one_assignment_recovered_failure_count": len(recovered),
            "diagnostic_truth_available_query_count": int(
                sum(_diagnostic_truth_row_count(row) > 0 for row in rows)
            ),
            "route_counts": dict(sorted(Counter(str(row["route"]) for row in rows).items())),
        }
    return output


def summarize_aggregate(
    records: list[dict], *, positive_source: str | None = None
) -> dict:
    """Return an all-query summary without granting deployment authority."""

    declared_sources = {
        str(record["positive_source"])
        for record in records
        if "positive_source" in record
    }
    if positive_source is None:
        if len(declared_sources) > 1:
            raise ValueError("V21 aggregate mixes positive sources")
        positive_source = (
            next(iter(declared_sources))
            if declared_sources
            else GAUSSIAN_GEOMETRY_SOURCE
        )
    if positive_source not in POSITIVE_SOURCES:
        raise ValueError("V21 aggregate positive source is unsupported")
    base = summarize_pose_recovery(records, positive_source=positive_source)
    failures = [record for record in records if not record["baseline"]["r5_success"]]
    one_assignment = [
        record
        for record in failures
        if isinstance(record.get("one_assignment_lower_bound"), dict)
        and record["one_assignment_lower_bound"].get("r5_success") is True
    ]
    recovered_bundles = [
        record
        for record in failures
        if isinstance(record.get("recovery_bundle"), dict)
        and int(record["recovery_bundle"].get("exact_delta_r5", 0)) == 1
    ]
    bundle_sizes = [
        int(torch.as_tensor(record["recovery_bundle"]["query_rows"]).numel())
        for record in recovered_bundles
    ]
    replay_counts = [int(record.get("exact_replay_count", 0)) for record in records]
    total = len(records)
    baseline_success = int(base["baseline_r5_success_count"])
    upper_bound_success = baseline_success + len(one_assignment)
    output = {
        **base,
        "positive_source": positive_source,
        "baseline_r5_rate": float(baseline_success / total) if total else 0.0,
        "one_assignment_recovered_failure_count": len(one_assignment),
        "one_assignment_recovery_upper_bound_r5_success_count": upper_bound_success,
        "one_assignment_recovery_upper_bound_r5_rate": (
            float(upper_bound_success / total) if total else 0.0
        ),
        "recovery_bundle_count": len(recovered_bundles),
        "recovery_bundle_sizes": bundle_sizes,
        "recovery_bundle_size_histogram": dict(
            sorted((str(size), count) for size, count in Counter(bundle_sizes).items())
        ),
        "exact_replay_count_total": int(sum(replay_counts)),
        "exact_replay_count_min": int(min(replay_counts, default=0)),
        "exact_replay_count_max": int(max(replay_counts, default=0)),
        "sequence_distribution": _distribution(records, "sequence_id"),
        "block_distribution": _distribution(records, "block_id"),
        "pose_recovery_is_diagnostic_upper_bound_only": True,
        "deployment_authorized": False,
        "controller_authorized_query_count": 0,
    }
    if positive_source == TRACK_CONSENSUS_DIAGNOSTIC:
        available = [
            record for record in records if _diagnostic_truth_row_count(record) > 0
        ]
        failure_available = [
            record
            for record in available
            if not record["baseline"]["r5_success"]
        ]
        diagnostic_rows = [
            _diagnostic_truth_row_count(record) for record in records
        ]
        diagnostic_edges = [
            int(record["legal_positive_csr"].get("legal_positive_edge_count", 0))
            for record in records
        ]
        output.update(
            {
                "track_consensus_diagnostic_available": True,
                "track_consensus_diagnostic_truth_available_query_count": len(
                    available
                ),
                "track_consensus_diagnostic_truth_query_coverage": (
                    float(len(available) / total) if total else 0.0
                ),
                "track_consensus_diagnostic_truth_available_failure_count": len(
                    failure_available
                ),
                "track_consensus_diagnostic_positive_row_count_total": int(
                    sum(diagnostic_rows)
                ),
                "track_consensus_diagnostic_positive_edge_count_total": int(
                    sum(diagnostic_edges)
                ),
                "track_consensus_one_assignment_recovered_failure_count": len(
                    one_assignment
                ),
                "track_consensus_exact_bundle_recovered_failure_count": len(
                    recovered_bundles
                ),
                "track_consensus_identity_evidence_is_deployment_authority": False,
                "interpretation": (
                    "Track-consensus planner diagnostic plus exact PoseLib recovery "
                    "upper bound; no map/metric action authority"
                ),
            }
        )
    else:
        output.update(
            {
                "geometry_recovery_is_upper_bound_only": True,
                "interpretation": (
                    "all-query geometry/PoseLib recovery upper bound; no "
                    "correspondence identity authority and no deployable map action"
                ),
            }
        )
    return output


def aggregate_payloads(
    entries: list[tuple[Path, str, dict]],
    source_records: list[tuple[int, int, dict]],
) -> tuple[list[dict], dict]:
    """Validate complete oracle shards against the full frontend registry."""

    if not entries or not source_records:
        raise ValueError("V21 aggregation requires shards and source records")
    source_contracts = [
        _validate_oracle_envelope(payload) for _, _, payload in entries
    ]
    if len(set(source_contracts)) != 1:
        raise ValueError("V21 pose-recovery shards mix positive sources/contracts")
    positive_source, _ = source_contracts[0]
    first = entries[0][2]
    shard_count = int(first["shard_count"])
    source_count = int(first["source_query_count"])
    common_input = first["input"]
    common_parameters = first["parameters"]
    coordinates = set()
    artifact_paths = set()
    artifact_digests = set()
    for path, digest, payload in entries:
        coordinate = int(payload["shard_index"])
        if (
            int(payload["shard_count"]) != shard_count
            or int(payload["source_query_count"]) != source_count
            or payload["input"] != common_input
            or payload["parameters"] != common_parameters
        ):
            raise ValueError("V21 pose-recovery shard contracts differ")
        if coordinate in coordinates or str(path) in artifact_paths or digest in artifact_digests:
            raise ValueError("V21 pose-recovery shard set is duplicated")
        coordinates.add(coordinate)
        artifact_paths.add(str(path))
        artifact_digests.add(digest)
    if coordinates != set(range(shard_count)):
        raise ValueError("V21 pose-recovery shards do not cover the full registry")
    if source_count != len(source_records):
        raise ValueError("V21 pose-recovery source_query_count differs from frontend")

    expected = {}
    for ordinal, (cache_index, local_index, source) in enumerate(source_records):
        query_index = int(source["query_index"])
        if query_index in expected:
            raise ValueError("V21 frontend source query registry is duplicated")
        expected[query_index] = {
            "ordinal": ordinal,
            "oracle_shard_index": ordinal % shard_count,
            "source_cache_index": int(cache_index),
            "source_record_index": int(local_index),
            "image_name": str(source["image_name"]),
            "sequence_id": str(source["sequence_id"]),
            "block_id": str(source["block_id"]),
            "source_record_sha256": str(source["source_record_sha256"]),
        }

    merged = []
    seen = set()
    for _, _, payload in sorted(entries, key=lambda value: int(value[2]["shard_index"])):
        coordinate = int(payload["shard_index"])
        expected_count = sum(
            row["oracle_shard_index"] == coordinate for row in expected.values()
        )
        if len(payload["records"]) != expected_count:
            raise ValueError("V21 oracle shard omitted or added a source record")
        for record in payload["records"]:
            query_index = int(record["query_index"])
            target = expected.get(query_index)
            if query_index in seen or target is None:
                raise ValueError("V21 oracle record query registry differs from frontend")
            if (
                target["oracle_shard_index"] != coordinate
                or int(record.get("source_cache_index", -1))
                != target["source_cache_index"]
                or int(record.get("source_record_index", -1))
                != target["source_record_index"]
                or str(record.get("image_name")) != target["image_name"]
                or str(record.get("sequence_id")) != target["sequence_id"]
                or str(record.get("block_id")) != target["block_id"]
            ):
                raise ValueError("V21 oracle record lineage differs from frontend")
            seen.add(query_index)
            merged.append(record)
    if seen != set(expected):
        raise ValueError("V21 oracle records do not exactly cover frontend queries")
    merged.sort(key=lambda record: expected[int(record["query_index"])]["ordinal"])
    return merged, summarize_aggregate(merged, positive_source=positive_source)


def _load_frontend_sources(input_contract: dict) -> tuple[list[tuple[int, int, dict]], dict]:
    identities = input_contract.get("adaptation_caches")
    if not isinstance(identities, list) or not identities:
        raise ValueError("V21 oracle input has no adaptation caches")
    caches = []
    frozen_map = (
        str(Path(str(input_contract.get("frozen_map", ""))).expanduser().resolve()),
        str(input_contract.get("frozen_map_sha256", "")),
    )
    for offset, identity in enumerate(identities):
        path_text, digest = _source_identity(identity, label=f"adaptation cache {offset}")
        path = Path(path_text)
        if sha256_file(path) != digest:
            raise ValueError("V21 adaptation cache SHA256 differs")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_cache_payload(payload)
        if payload.get("role") != "adaptation":
            raise ValueError("V21 oracle references a non-adaptation frontend cache")
        stable = payload.get("inputs", {}).get("stable_map")
        if _source_identity(stable, label="frontend stable map") != frozen_map:
            raise ValueError("V21 frontend cache frozen-map lineage differs")
        caches.append((path, digest, payload))
    first = caches[0][2]
    registry = first["shard_registry"]
    registry_sha = str(registry["registry_sha256"])
    cache_shard_count = int(first["shard_count"])
    coordinates = set()
    for input_offset, (_, _, cache) in enumerate(caches):
        if (
            int(cache["shard_count"]) != cache_shard_count
            or str(cache["shard_registry"]["registry_sha256"]) != registry_sha
        ):
            raise ValueError("V21 frontend cache shard registries differ")
        coordinate = int(cache["shard_index"])
        if coordinate in coordinates or coordinate != input_offset:
            raise ValueError("V21 frontend cache shard coordinate is duplicated")
        coordinates.add(coordinate)
    if coordinates != set(range(cache_shard_count)):
        raise ValueError("V21 frontend caches do not cover their full registry")
    caches.sort(key=lambda value: int(value[2]["shard_index"]))
    ordinal_by_query = {
        int(row["query_index"]): int(row["ordinal"]) for row in registry["rows"]
    }
    records = [
        (cache_index, local_index, record)
        for cache_index, (_, _, cache) in enumerate(caches)
        for local_index, record in enumerate(cache["records"])
    ]
    if (
        len(records) != int(registry["role_query_count"])
        or {int(record["query_index"]) for _, _, record in records}
        != set(ordinal_by_query)
    ):
        raise ValueError("V21 frontend caches do not exactly cover their registry")
    records.sort(key=lambda row: ordinal_by_query[int(row[2]["query_index"])])
    return records, registry


def _verify_common_inputs(
    input_contract: dict,
    *,
    frontend_registry: dict,
    source_records: list[tuple[int, int, dict]],
    positive_source: str,
) -> None:
    map_path = Path(str(input_contract.get("frozen_map", ""))).expanduser().resolve()
    map_sha = str(input_contract.get("frozen_map_sha256", ""))
    if str(map_path) != str(input_contract.get("frozen_map")) or sha256_file(map_path) != map_sha:
        raise ValueError("V21 frozen map path/SHA256 differs")
    map_payload = torch.load(map_path, map_location="cpu", weights_only=False)
    anchor_xyz = torch.as_tensor(map_payload.get("anchor_xyz"))
    if (
        map_payload.get("schema") != "lafgs_materialized_anchor_map"
        or anchor_xyz.ndim != 2
        or anchor_xyz.shape[1] != 3
        or anchor_xyz.shape[0] <= 0
    ):
        raise ValueError("V21 frozen map registry is invalid")
    supports = input_contract.get("gaussian_support")
    if not isinstance(supports, list):
        raise ValueError("V21 Gaussian support input registry is invalid")
    adaptation_sources = {
        _source_identity(value, label="adaptation cache")
        for value in input_contract.get("adaptation_caches", [])
    }
    referenced_caches = set()
    support_entries = []
    for offset, identity in enumerate(supports):
        path_text, digest = _source_identity(identity, label=f"Gaussian support {offset}")
        path = Path(path_text)
        if sha256_file(path) != digest:
            raise ValueError("V21 Gaussian support SHA256 differs")
        support = torch.load(path, map_location="cpu", weights_only=False)
        validate_support_payload(support)
        if (
            str(support.get("stable_map_sha256")) != map_sha
            or str(support.get("frontend_shard_registry_sha256"))
            != str(frontend_registry["registry_sha256"])
            or support.get("frontend_shard_registry") != frontend_registry
            or _source_identity(
                support.get("inputs", {}).get("stable_map"),
                label="Gaussian stable map",
            )
            != (str(map_path), map_sha)
        ):
            raise ValueError("V21 Gaussian support map/frontend lineage differs")
        referenced_caches.update(
            _source_identity(value, label="Gaussian frontend cache")
            for value in support["inputs"]["frontend_caches"]
        )
        support_entries.append((path, digest, support))
    if supports and referenced_caches != adaptation_sources:
        raise ValueError("V21 Gaussian support does not cover all frontend caches")

    truth_identity = input_contract.get("correspondence_truth")
    if positive_source != TRACK_CONSENSUS_DIAGNOSTIC:
        if truth_identity is not None:
            raise ValueError("V21 geometry aggregation cannot consume Track truth")
        return
    if len(support_entries) != 1:
        raise ValueError("V21 Track aggregation requires one complete Gaussian support")
    truth_path_text, truth_sha = _source_identity(
        truth_identity, label="correspondence truth"
    )
    truth_path = Path(truth_path_text)
    if sha256_file(truth_path) != truth_sha:
        raise ValueError("V21 correspondence truth SHA256 differs")
    truth = torch.load(truth_path, map_location="cpu", weights_only=False)
    validate_correspondence_payload(truth)
    decision = truth.get("teacher_action_decision", {})
    support_path, support_sha, _ = support_entries[0]
    truth_inputs = truth.get("inputs", {})
    if not (
        truth.get("action_authorized") is False
        and truth.get("training_consumers_allowed") is False
        and truth.get("planner_diagnostic_consumers_allowed") is True
        and truth.get("artifact_writes_map") is False
        and truth.get("exact_poselib_recovery_is_identity_truth") is False
        and decision.get("planner_diagnostic_authorized") is True
        and decision.get("action_authorized") is False
        and int(truth.get("anchor_count", -1)) == int(anchor_xyz.shape[0])
        and _source_identity(
            truth_inputs.get("stable_map"), label="correspondence stable map"
        )
        == (str(map_path), map_sha)
        and _source_identity(
            truth_inputs.get("gaussian_support"),
            label="correspondence Gaussian support",
        )
        == (str(support_path), support_sha)
        and truth.get("frontend_shard_registry") == frontend_registry
        and truth.get("frontend_shard_registry_sha256")
        == frontend_registry["registry_sha256"]
    ):
        raise ValueError("V21 correspondence truth diagnostic/map registry differs")
    truth_cache_sources = {
        _source_identity(value, label="correspondence frontend cache")
        for value in truth_inputs.get("frontend_caches", [])
    }
    if truth_cache_sources != adaptation_sources:
        raise ValueError("V21 correspondence truth frontend lineage differs")
    frontend_by_query = {
        int(record["query_index"]): record for _, _, record in source_records
    }
    if len(frontend_by_query) != len(source_records):
        raise ValueError("V21 frontend query registry is duplicated")
    joined_queries = set()
    for truth_record in truth["records"]:
        query_index = int(truth_record["query_index"])
        frontend = frontend_by_query.get(query_index)
        if frontend is None or query_index in joined_queries:
            raise ValueError("V21 correspondence truth query registry differs")
        if not (
            truth_record["image_name"] == frontend["image_name"]
            and truth_record["image_sha256"] == frontend["image_sha256"]
            and truth_record["sequence_id"] == frontend["sequence_id"]
            and int(truth_record["frame_index"]) == int(frontend["frame_index"])
            and truth_record["block_id"] == frontend["block_id"]
            and truth_record["role"] == frontend["role"] == "adaptation"
            and truth_record["source_record_sha256"]
            == frontend["source_record_sha256"]
            and truth_record["pose_w2c_sha256"] == frontend["pose_w2c_sha256"]
            and int(truth_record["keypoint_count"])
            == int(torch.as_tensor(frontend["keypoints"]).shape[0])
            and truth_record["keypoints_sha256"]
            == tensor_sha256(torch.as_tensor(frontend["keypoints"]).float())
            and truth_record["descriptors_sha256"]
            == tensor_sha256(torch.as_tensor(frontend["descriptors"]).float())
            and truth_record["action_authorized"] is False
        ):
            raise ValueError("V21 correspondence truth row binding differs")
        joined_queries.add(query_index)
    if joined_queries != set(frontend_by_query):
        raise ValueError("V21 correspondence truth does not cover all frontend queries")


def _validate_output(payload: dict) -> None:
    records = payload.get("records")
    registry = payload.get("frontend_query_registry")
    oracle_shards = payload.get("oracle_shards")
    positive_source = str(payload.get("positive_source", ""))
    input_contract = payload.get("input")
    source_fields = {
        str(record["positive_source"])
        for record in records or []
        if isinstance(record, dict) and "positive_source" in record
    }
    source_records_valid = source_fields in ({positive_source}, set()) and not (
        positive_source == TRACK_CONSENSUS_DIAGNOSTIC and not source_fields
    )
    source_contract_valid = (
        positive_source in POSITIVE_SOURCES
        and isinstance(input_contract, dict)
        and (
            (positive_source == TRACK_CONSENSUS_DIAGNOSTIC)
            == isinstance(input_contract.get("correspondence_truth"), dict)
        )
        and source_records_valid
        and (
            payload.get("track_consensus_identity_evidence_present") is True
        )
        == (positive_source == TRACK_CONSENSUS_DIAGNOSTIC)
        and payload.get(
            "track_consensus_identity_evidence_is_deployment_authority"
        )
        is False
    )
    if not (
        payload.get("schema") == OUTPUT_SCHEMA
        and payload.get("version") == VERSION
        and payload.get("protocol") == "test_adapted"
        and payload.get("uses_test_queries") is True
        and payload.get("role") == "adaptation"
        and payload.get("pose_recovery_is_diagnostic_upper_bound_only") is True
        and payload.get("deployment_authorized") is False
        and payload.get("correspondence_identity_authority_present") is False
        and source_contract_valid
        and isinstance(records, list)
        and isinstance(registry, list)
        and isinstance(oracle_shards, list)
        and bool(oracle_shards)
        and isinstance(payload.get("parameters"), dict)
        and len(records) == len(registry) == int(payload.get("source_query_count", -1))
        and payload.get("summary")
        == summarize_aggregate(records, positive_source=positive_source)
        and int(payload["summary"].get("controller_authorized_query_count", -1)) == 0
    ):
        raise ValueError("V21 pose-recovery aggregate output contract is invalid")
    shard_identities = [
        _source_identity(value, label="aggregate oracle shard")
        for value in oracle_shards
    ]
    if len(set(shard_identities)) != len(shard_identities):
        raise ValueError("V21 aggregate oracle shard lineage is duplicated")
    query_indices = set()
    image_names = set()
    source_records = set()
    for ordinal, (record, source) in enumerate(zip(records, registry)):
        source_record_sha = str(source.get("source_record_sha256", ""))
        if (
            int(source.get("ordinal", -1)) != ordinal
            or int(record["query_index"]) != int(source["query_index"])
            or str(record["image_name"]) != str(source["image_name"])
            or bool(record.get("controller_authorized", False))
            or SHA256_PATTERN.fullmatch(source_record_sha) is None
            or int(source["query_index"]) in query_indices
            or str(source["image_name"]) in image_names
            or source_record_sha in source_records
        ):
            raise ValueError("V21 aggregate record order differs from frontend registry")
        query_indices.add(int(source["query_index"]))
        image_names.add(str(source["image_name"]))
        source_records.add(source_record_sha)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    entries = []
    for raw_path in args.shard:
        path = raw_path.expanduser().resolve()
        digest = sha256_file(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        entries.append((path, digest, payload))
    source_contracts = [
        _validate_oracle_envelope(payload) for _, _, payload in entries
    ]
    if len(set(source_contracts)) != 1:
        raise ValueError("V21 pose-recovery shards mix positive sources/contracts")
    positive_source, source_contract = source_contracts[0]
    common_input = entries[0][2]["input"]
    source_records, frontend_registry = _load_frontend_sources(common_input)
    _verify_common_inputs(
        common_input,
        frontend_registry=frontend_registry,
        source_records=source_records,
        positive_source=positive_source,
    )
    records, summary = aggregate_payloads(entries, source_records)
    frontend_query_registry = [
        {
            "ordinal": ordinal,
            "query_index": int(record["query_index"]),
            "image_name": str(record["image_name"]),
            "source_record_sha256": str(record["source_record_sha256"]),
        }
        for ordinal, (_, _, record) in enumerate(source_records)
    ]
    output = {
        "schema": OUTPUT_SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": "adaptation",
        "positive_source": positive_source,
        "source_contract": source_contract,
        "pose_recovery_is_diagnostic_upper_bound_only": True,
        "geometry_recovery_is_upper_bound_only": (
            positive_source == GAUSSIAN_GEOMETRY_SOURCE
        ),
        "deployment_authorized": False,
        "correspondence_identity_authority_present": False,
        "track_consensus_identity_evidence_present": (
            positive_source == TRACK_CONSENSUS_DIAGNOSTIC
        ),
        "track_consensus_identity_evidence_is_deployment_authority": False,
        "track_consensus_diagnostic_is_action_authority": False,
        "controller_authorized_query_count_must_be_zero": True,
        "source_query_count": len(source_records),
        "oracle_shards": [
            {"path": str(path), "sha256": digest} for path, digest, _ in entries
        ],
        "input": common_input,
        "parameters": entries[0][2]["parameters"],
        "frontend_shard_registry_sha256": frontend_registry["registry_sha256"],
        "frontend_query_registry": frontend_query_registry,
        "records": records,
        "summary": summary,
    }
    for path, digest, _ in entries:
        if sha256_file(path) != digest:
            raise RuntimeError("V21 oracle shard changed while aggregating")
    _verify_common_inputs(
        common_input,
        frontend_registry=frontend_registry,
        source_records=source_records,
        positive_source=positive_source,
    )
    _validate_output(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(output, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        _validate_output(reloaded)
        try:
            os.link(temporary, output_path)
        except FileExistsError as error:
            raise FileExistsError(
                f"V21 aggregate output appeared while running: {output_path}"
            ) from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
