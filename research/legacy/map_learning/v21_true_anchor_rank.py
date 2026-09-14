"""Exact global-rank audit for V21 Tier-C UNIQUE correspondences.

This module measures whether a diagnostic Track-consensus Anchor is already
present near the front of the frozen descriptor ranking.  Tier-C evidence is
planner-only: it is never exposed as map/metric supervision, a negative label,
or controller authority.  The audit consumes the complete adaptation prefix
and may stratify results with the already-materialized Gaussian/PoseLib oracle,
but neither source authorizes an action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import re
import uuid

import torch
import torch.nn.functional as F

from map_learning.v21_correspondence_truth import (
    STATUS_UNIQUE,
    validate_correspondence_payload,
)
from map_learning.v21_test_cache import (
    tensor_sha256,
    validate_cache_payload,
    validate_shard_registry,
)


SCHEMA = "lafgs_v21_true_anchor_global_rank_audit"
VERSION = 1
ROLE = "adaptation"
RANK_KS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

RANK_DEFINITION = {
    "similarity": "float32_l2_normalized_global_cosine",
    "strict_rank": "one_plus_count_of_frozen_anchor_scores_strictly_greater_than_true_anchor_score",
    "tie_break_rank": "strict_rank_plus_equal_score_anchor_rows_lower_than_true_anchor_row",
    "reported_rank_at_k_uses": "strict_rank",
    "anchor_scope": "all_frozen_stable_map_anchor_rows",
    "query_scope": "tier_c_diagnostic_unique_rows_in_complete_adaptation_prefix",
}

SEMANTICS = {
    "correspondence_source": "tier_c_track_consensus_diagnostic_unique_only",
    "correspondence_source_is_action_authority": False,
    "gaussian_oracle_role": "pose_recovery_bundle_overlap_stratification_only",
    "gaussian_oracle_is_identity_truth": False,
    "rank_is_action_authority": False,
    "negative_labels_created": False,
    "unlabelled_or_ambiguous_are_negative": False,
    "artifact_writes_map_or_metric": False,
    "deployment_authorized": False,
}

COLUMN_DTYPES = {
    "edge_query_ordinals": torch.int64,
    "edge_query_indices": torch.int64,
    "edge_query_rows": torch.int64,
    "true_anchor_rows": torch.int64,
    "true_anchor_scores": torch.float32,
    "strict_greater_counts": torch.int64,
    "strict_ranks": torch.int64,
    "lower_row_equal_score_counts": torch.int64,
    "exact_tie_break_ranks": torch.int64,
    "baseline_r5_success": torch.bool,
    "baseline_pose_inlier": torch.bool,
    "baseline_winner_anchor_rows": torch.int64,
    "baseline_winner_scores": torch.float32,
    "baseline_winner_is_true_anchor": torch.bool,
    "geometry_recovery_bundle_query_row_overlap": torch.bool,
    "geometry_recovery_bundle_exact_anchor_overlap": torch.bool,
}


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def source_record(value: object, *, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"V21 rank audit {label} source is missing")
    path = str(value.get("path", ""))
    digest = _require_sha256(value.get("sha256"), label=f"{label} SHA256")
    size = int(value.get("size_bytes", 0))
    if not path or size <= 0:
        raise ValueError(f"V21 rank audit {label} source is empty")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _source_identity(value: object, *, label: str) -> tuple[str, str]:
    source = source_record(value, label=label)
    return str(Path(source["path"]).expanduser().resolve()), source["sha256"]


def _same_source(left: object, right: object, *, label: str) -> None:
    if _source_identity(left, label=label) != _source_identity(right, label=label):
        raise ValueError(f"V21 rank audit {label} lineage differs")


def exact_global_true_anchor_ranks(
    *,
    query_descriptors: torch.Tensor,
    anchor_features: torch.Tensor,
    true_anchor_rows: torch.Tensor,
    device: str | torch.device = "cpu",
    query_batch_size: int = 256,
    anchor_chunk_size: int = 32768,
) -> dict:
    """Compute strict cosine ranks against every frozen Anchor.

    Strict rank intentionally follows ``1 + count(score > true_score)``.  An
    additional lower-row tie count records the production matcher's stable
    row-order tie break without changing the requested Rank@K definition.
    """

    descriptors = torch.as_tensor(query_descriptors).float().cpu()
    features = torch.as_tensor(anchor_features).float().cpu()
    anchors = torch.as_tensor(true_anchor_rows).long().cpu().reshape(-1)
    if (
        descriptors.ndim != 2
        or features.ndim != 2
        or descriptors.shape[0] != anchors.numel()
        or descriptors.shape[1] != features.shape[1]
        or descriptors.shape[0] == 0
        or features.shape[0] == 0
        or int(query_batch_size) < 1
        or int(anchor_chunk_size) < 1
        or not bool(torch.isfinite(descriptors).all())
        or not bool(torch.isfinite(features).all())
        or bool((descriptors.norm(dim=1) <= 1e-12).any())
        or bool((features.norm(dim=1) <= 1e-12).any())
        or int(anchors.min()) < 0
        or int(anchors.max()) >= features.shape[0]
    ):
        raise ValueError("V21 rank audit descriptor/Anchor registries differ")

    target = torch.device(device)
    # Match SparseLocalizer/global matcher semantics: the frozen bank is
    # normalized on load and again by the historical global matcher.
    bank = F.normalize(F.normalize(features.to(target), dim=1), dim=1)
    strict_counts = torch.empty(anchors.numel(), dtype=torch.long)
    lower_equal_counts = torch.empty_like(strict_counts)
    true_scores_out = torch.empty(anchors.numel(), dtype=torch.float32)
    anchor_count = int(bank.shape[0])
    for query_start in range(0, anchors.numel(), int(query_batch_size)):
        query_stop = min(query_start + int(query_batch_size), anchors.numel())
        local_descriptors = F.normalize(
            descriptors[query_start:query_stop].to(target), dim=1
        )
        local_anchors = anchors[query_start:query_stop].to(target)
        true_scores = (
            local_descriptors * bank.index_select(0, local_anchors)
        ).sum(1)
        greater = torch.zeros(local_anchors.numel(), dtype=torch.long, device=target)
        lower_equal = torch.zeros_like(greater)
        local_rows = torch.arange(local_anchors.numel(), device=target)
        for anchor_start in range(0, anchor_count, int(anchor_chunk_size)):
            anchor_stop = min(anchor_start + int(anchor_chunk_size), anchor_count)
            scores = local_descriptors @ bank[anchor_start:anchor_stop].T
            greater_mask = scores > true_scores[:, None]
            true_in_chunk = (local_anchors >= anchor_start) & (
                local_anchors < anchor_stop
            )
            if bool(true_in_chunk.any()):
                local_true_columns = local_anchors[true_in_chunk] - anchor_start
                greater_mask[
                    local_rows[true_in_chunk], local_true_columns
                ] = False
            greater += greater_mask.sum(1)
            candidate_rows = torch.arange(
                anchor_start, anchor_stop, device=target
            )
            lower_equal += (
                (scores == true_scores[:, None])
                & (candidate_rows[None, :] < local_anchors[:, None])
            ).sum(1)
        strict_counts[query_start:query_stop] = greater.cpu()
        lower_equal_counts[query_start:query_stop] = lower_equal.cpu()
        true_scores_out[query_start:query_stop] = true_scores.cpu()
    strict_ranks = strict_counts + 1
    tie_break_ranks = strict_ranks + lower_equal_counts
    return {
        "true_anchor_scores": true_scores_out.contiguous(),
        "strict_greater_counts": strict_counts.contiguous(),
        "strict_ranks": strict_ranks.contiguous(),
        "lower_row_equal_score_counts": lower_equal_counts.contiguous(),
        "exact_tie_break_ranks": tie_break_ranks.contiguous(),
    }


def _nearest_rank(values: torch.Tensor, fraction: float) -> int:
    ordered = torch.sort(torch.as_tensor(values).long().reshape(-1)).values
    if ordered.numel() == 0:
        return 0
    index = max(0, math.ceil(float(fraction) * ordered.numel()) - 1)
    return int(ordered[index])


def _stratum_summary(payload: Mapping, mask: torch.Tensor) -> dict:
    selected = torch.as_tensor(mask).bool().reshape(-1)
    ranks = torch.as_tensor(payload["strict_ranks"]).long()[selected]
    queries = torch.as_tensor(payload["edge_query_indices"]).long()[selected]
    top1 = torch.as_tensor(payload["baseline_winner_is_true_anchor"]).bool()[selected]
    recalls = {}
    for k in RANK_KS:
        count = int((ranks <= k).sum())
        recalls[str(k)] = {
            "count": count,
            "fraction": float(count / ranks.numel()) if ranks.numel() else 0.0,
        }
    return {
        "edge_count": int(ranks.numel()),
        "query_count": int(torch.unique(queries).numel()),
        "baseline_exact_true_anchor_top1_count": int(top1.sum()),
        "baseline_exact_true_anchor_top1_fraction": (
            float(top1.float().mean()) if top1.numel() else 0.0
        ),
        "rank_min": int(ranks.min()) if ranks.numel() else 0,
        "rank_p50_nearest": _nearest_rank(ranks, 0.50),
        "rank_p75_nearest": _nearest_rank(ranks, 0.75),
        "rank_p90_nearest": _nearest_rank(ranks, 0.90),
        "rank_max": int(ranks.max()) if ranks.numel() else 0,
        "rank_at_k": recalls,
    }


def summarize_rank_audit(payload: Mapping) -> dict:
    count = int(torch.as_tensor(payload["strict_ranks"]).numel())
    all_rows = torch.ones(count, dtype=torch.bool)
    success = torch.as_tensor(payload["baseline_r5_success"]).bool()
    inlier = torch.as_tensor(payload["baseline_pose_inlier"]).bool()
    bundle = torch.as_tensor(
        payload["geometry_recovery_bundle_query_row_overlap"]
    ).bool()
    exact_bundle = torch.as_tensor(
        payload["geometry_recovery_bundle_exact_anchor_overlap"]
    ).bool()
    masks = {
        "all": all_rows,
        "baseline_r5_success": success,
        "baseline_r5_failure": ~success,
        "baseline_pose_inlier": inlier,
        "baseline_pose_non_inlier": ~inlier,
        "baseline_r5_success_and_pose_inlier": success & inlier,
        "baseline_r5_success_and_pose_non_inlier": success & ~inlier,
        "baseline_r5_failure_and_pose_inlier": ~success & inlier,
        "baseline_r5_failure_and_pose_non_inlier": ~success & ~inlier,
        "geometry_recovery_bundle_query_row_overlap": bundle,
        "no_geometry_recovery_bundle_query_row_overlap": ~bundle,
        "geometry_recovery_bundle_exact_anchor_overlap": exact_bundle,
        "geometry_bundle_row_overlap_and_baseline_failure": bundle & ~success,
    }
    return {
        "unique_edge_count": count,
        "strata": {
            name: _stratum_summary(payload, mask) for name, mask in masks.items()
        },
        "interpretation": (
            "Exact descriptor rank of Tier-C diagnostic UNIQUE truth; all labels "
            "and Gaussian bundle overlaps remain diagnostic-only"
        ),
        "deployment_authorized": False,
    }


def _complete_frontend_records(
    frontend_caches: Sequence[Mapping], frontend_sources: Sequence[Mapping]
) -> tuple[list[dict], list[dict], dict]:
    if not frontend_caches or len(frontend_caches) != len(frontend_sources):
        raise ValueError("V21 rank audit frontend cache/source list differs")
    entries = []
    for payload, source in zip(frontend_caches, frontend_sources):
        validate_cache_payload(payload)
        if payload.get("role") != ROLE:
            raise ValueError("V21 rank audit accepts adaptation caches only")
        entries.append((int(payload["shard_index"]), payload, source_record(source, label="frontend cache")))
    entries.sort(key=lambda value: value[0])
    first = entries[0][1]
    registry = first["shard_registry"]
    validate_shard_registry(registry)
    shard_count = int(first["shard_count"])
    if [value[0] for value in entries] != list(range(shard_count)):
        raise ValueError("V21 rank audit requires complete frontend shards")
    by_query = {}
    ordered_sources = []
    for _, payload, source in entries:
        if payload["shard_registry"] != registry or int(payload["shard_count"]) != shard_count:
            raise ValueError("V21 rank audit frontend registries differ")
        ordered_sources.append(source)
        for local_index, record in enumerate(payload["records"]):
            query = int(record["query_index"])
            if query in by_query:
                raise ValueError("V21 rank audit frontend query is duplicated")
            by_query[query] = (payload, local_index, record)
    rows = sorted(registry["rows"], key=lambda value: int(value["ordinal"]))
    if len(rows) != len(by_query) or len(rows) != int(registry["role_query_count"]):
        raise ValueError("V21 rank audit frontend coverage is incomplete")
    ordered = []
    query_registry = []
    for row in rows:
        query = int(row["query_index"])
        if query not in by_query:
            raise ValueError("V21 rank audit frontend registry row is absent")
        payload, local_index, record = by_query[query]
        if any(
            record.get(name) != row.get(name)
            for name in ("query_index", "image_name", "image_sha256", "source_record_sha256")
        ):
            raise ValueError("V21 rank audit frontend record identity differs")
        ordered.append(record)
        query_registry.append(
            {
                "ordinal": int(row["ordinal"]),
                "query_index": query,
                "image_name": str(record["image_name"]),
                "image_sha256": str(record["image_sha256"]),
                "source_record_sha256": str(record["source_record_sha256"]),
                "cache_shard_index": int(payload["shard_index"]),
                "cache_local_record_index": int(local_index),
            }
        )
    return ordered, query_registry, {"registry": registry, "sources": ordered_sources}


def build_true_anchor_rank_audit(
    *,
    stable_map: Mapping,
    frontend_caches: Sequence[Mapping],
    correspondence_truth: Mapping,
    geometry_oracle: Mapping,
    stable_map_source: Mapping,
    frontend_cache_sources: Sequence[Mapping],
    correspondence_truth_source: Mapping,
    geometry_oracle_source: Mapping,
    producer_sources: Sequence[Mapping],
    device: str | torch.device = "cpu",
    query_batch_size: int = 256,
    anchor_chunk_size: int = 32768,
) -> dict:
    """Join immutable adaptation diagnostics and materialize exact ranks."""

    features = torch.as_tensor(stable_map.get("anchor_features")).float().cpu()
    xyz = torch.as_tensor(stable_map.get("anchor_xyz")).float().cpu()
    if (
        stable_map.get("schema") != "lafgs_materialized_anchor_map"
        or features.ndim != 2
        or xyz.shape != (features.shape[0], 3)
        or features.shape[0] <= 0
        or not bool(torch.isfinite(features).all())
        or bool((features.norm(dim=1) <= 1e-12).any())
    ):
        raise ValueError("V21 rank audit stable map is invalid")
    stable_source = source_record(stable_map_source, label="stable map")
    cache_records, query_registry, cache_info = _complete_frontend_records(
        frontend_caches, frontend_cache_sources
    )
    registry = cache_info["registry"]
    cache_sources = cache_info["sources"]
    for cache in frontend_caches:
        _same_source(
            cache.get("inputs", {}).get("stable_map"), stable_source, label="stable map"
        )
        if int(cache.get("anchor_count", -1)) != features.shape[0] or int(
            cache.get("descriptor_dim", -1)
        ) != features.shape[1]:
            raise ValueError("V21 rank audit cache/map dimensions differ")

    validate_correspondence_payload(correspondence_truth)
    if not (
        correspondence_truth.get("role") == ROLE
        and correspondence_truth.get("action_authorized") is False
        and correspondence_truth.get("training_consumers_allowed") is False
        and correspondence_truth.get("planner_diagnostic_consumers_allowed") is True
        and correspondence_truth.get("artifact_writes_map") is False
        and correspondence_truth.get("negative_labels_created") is False
        and correspondence_truth.get("frontend_shard_registry") == registry
        and int(correspondence_truth.get("anchor_count", -1)) == features.shape[0]
    ):
        raise ValueError("V21 rank audit requires planner-only Tier-C truth")
    truth_source = source_record(correspondence_truth_source, label="correspondence truth")
    truth_inputs = correspondence_truth["inputs"]
    _same_source(truth_inputs.get("stable_map"), stable_source, label="truth stable map")
    declared_truth_caches = {
        _source_identity(value, label="truth frontend cache")
        for value in truth_inputs.get("frontend_caches", ())
    }
    if declared_truth_caches != {
        _source_identity(value, label="frontend cache") for value in cache_sources
    }:
        raise ValueError("V21 rank audit truth/frontend cache lineage differs")

    oracle_source = source_record(geometry_oracle_source, label="geometry oracle")
    oracle_input = geometry_oracle.get("input")
    oracle_summary = geometry_oracle.get("summary")
    if not (
        geometry_oracle.get("schema") == "lafgs_v21_pose_recovery_oracle_aggregate"
        and int(geometry_oracle.get("version", 0)) == 1
        and geometry_oracle.get("role") == ROLE
        and geometry_oracle.get("uses_test_queries") is True
        and geometry_oracle.get("geometry_recovery_is_upper_bound_only") is True
        and geometry_oracle.get("deployment_authorized") is False
        and geometry_oracle.get("correspondence_identity_authority_present") is False
        and isinstance(oracle_input, Mapping)
        and isinstance(oracle_summary, Mapping)
        and int(oracle_summary.get("controller_authorized_query_count", -1)) == 0
        and isinstance(geometry_oracle.get("records"), list)
    ):
        raise ValueError("V21 rank audit geometry oracle is not diagnostic-only")
    if (
        str(Path(str(oracle_input.get("frozen_map", ""))).expanduser().resolve())
        != str(Path(stable_source["path"]).expanduser().resolve())
        or oracle_input.get("frozen_map_sha256") != stable_source["sha256"]
    ):
        raise ValueError("V21 rank audit oracle/stable-map lineage differs")
    declared_oracle_caches = {
        (str(Path(str(value.get("path", ""))).expanduser().resolve()), str(value.get("sha256", "")))
        for value in oracle_input.get("adaptation_caches", ())
    }
    if declared_oracle_caches != {
        _source_identity(value, label="frontend cache") for value in cache_sources
    }:
        raise ValueError("V21 rank audit oracle/frontend cache lineage differs")

    truth_by_query = {int(record["query_index"]): record for record in correspondence_truth["records"]}
    oracle_by_query = {int(record["query_index"]): record for record in geometry_oracle["records"]}
    expected_queries = {int(record["query_index"]) for record in cache_records}
    if set(truth_by_query) != expected_queries or set(oracle_by_query) != expected_queries:
        raise ValueError("V21 rank audit truth/oracle query coverage differs")

    descriptors = []
    edge_columns: dict[str, list] = {
        "edge_query_ordinals": [],
        "edge_query_indices": [],
        "edge_query_rows": [],
        "true_anchor_rows": [],
        "baseline_r5_success": [],
        "baseline_pose_inlier": [],
        "baseline_winner_anchor_rows": [],
        "baseline_winner_scores": [],
        "baseline_winner_is_true_anchor": [],
        "geometry_recovery_bundle_query_row_overlap": [],
        "geometry_recovery_bundle_exact_anchor_overlap": [],
    }
    for ordinal, frontend in enumerate(cache_records):
        query = int(frontend["query_index"])
        truth = truth_by_query[query]
        oracle = oracle_by_query[query]
        if not (
            truth["image_name"] == frontend["image_name"] == oracle["image_name"]
            and truth["source_record_sha256"] == frontend["source_record_sha256"]
            and truth["descriptors_sha256"]
            == tensor_sha256(torch.as_tensor(frontend["descriptors"]).float())
            and int(truth["keypoint_count"])
            == int(torch.as_tensor(frontend["descriptors"]).shape[0])
            and bool(oracle.get("controller_authorized", False)) is False
        ):
            raise ValueError("V21 rank audit per-query lineage differs")
        status = torch.as_tensor(truth["diagnostic_truth_status"]).to(torch.int8)
        offsets = torch.as_tensor(truth["diagnostic_positive_offsets"]).long()
        anchors = torch.as_tensor(truth["diagnostic_positive_anchor_rows"]).long()
        unique_rows = torch.nonzero(status == STATUS_UNIQUE, as_tuple=False).reshape(-1)
        inlier_rows = set(torch.as_tensor(frontend["baseline_inliers"]).long().tolist())
        bundle = oracle.get("recovery_bundle")
        bundle_by_row = {}
        if isinstance(bundle, Mapping):
            bundle_rows = torch.as_tensor(bundle.get("query_rows")).long().reshape(-1)
            bundle_anchors = torch.as_tensor(bundle.get("anchor_rows")).long().reshape(-1)
            if bundle_rows.shape != bundle_anchors.shape or torch.unique(bundle_rows).numel() != bundle_rows.numel():
                raise ValueError("V21 rank audit oracle bundle is malformed")
            bundle_by_row = dict(zip(bundle_rows.tolist(), bundle_anchors.tolist()))
        for row in unique_rows.tolist():
            lower, upper = int(offsets[row]), int(offsets[row + 1])
            if upper - lower != 1:
                raise ValueError("V21 rank audit UNIQUE row has non-unique Anchor")
            true_anchor = int(anchors[lower])
            winner = int(torch.as_tensor(frontend["winner_anchor_rows"])[row])
            descriptors.append(torch.as_tensor(frontend["descriptors"])[row].float())
            edge_columns["edge_query_ordinals"].append(ordinal)
            edge_columns["edge_query_indices"].append(query)
            edge_columns["edge_query_rows"].append(row)
            edge_columns["true_anchor_rows"].append(true_anchor)
            edge_columns["baseline_r5_success"].append(bool(frontend["baseline_r5"]))
            edge_columns["baseline_pose_inlier"].append(row in inlier_rows)
            edge_columns["baseline_winner_anchor_rows"].append(winner)
            edge_columns["baseline_winner_scores"].append(float(torch.as_tensor(frontend["winner_scores"])[row]))
            edge_columns["baseline_winner_is_true_anchor"].append(winner == true_anchor)
            edge_columns["geometry_recovery_bundle_query_row_overlap"].append(row in bundle_by_row)
            edge_columns["geometry_recovery_bundle_exact_anchor_overlap"].append(bundle_by_row.get(row) == true_anchor)
    if not descriptors:
        raise ValueError("V21 rank audit has no diagnostic UNIQUE edge")

    tensor_columns = {
        name: torch.tensor(values, dtype=COLUMN_DTYPES[name]).contiguous()
        for name, values in edge_columns.items()
    }
    rank_columns = exact_global_true_anchor_ranks(
        query_descriptors=torch.stack(descriptors),
        anchor_features=features,
        true_anchor_rows=tensor_columns["true_anchor_rows"],
        device=device,
        query_batch_size=query_batch_size,
        anchor_chunk_size=anchor_chunk_size,
    )
    tensor_columns.update(rank_columns)
    output = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "role": ROLE,
        "complete_adaptation_registry_consumed": True,
        "control_features_consumed": False,
        "confirmation_features_consumed": False,
        "control_or_confirmation_outcomes_consumed": False,
        "diagnostic_only": True,
        "action_authorized": False,
        "controller_authorized": False,
        "deployment_authorized": False,
        "negative_anchor_labels_created": False,
        "rank_definition": RANK_DEFINITION,
        "rank_ks": RANK_KS,
        "semantics": SEMANTICS,
        "inputs": {
            "stable_map": stable_source,
            "frontend_caches": cache_sources,
            "correspondence_truth": truth_source,
            "geometry_oracle": oracle_source,
            "producer_sources": [source_record(value, label="producer") for value in producer_sources],
        },
        "stable_map_sha256": stable_source["sha256"],
        "correspondence_truth_sha256": truth_source["sha256"],
        "geometry_oracle_sha256": oracle_source["sha256"],
        "frontend_shard_registry": registry,
        "frontend_shard_registry_sha256": registry["registry_sha256"],
        "anchor_count": int(features.shape[0]),
        "descriptor_dim": int(features.shape[1]),
        "query_count": len(cache_records),
        "unique_edge_count": len(descriptors),
        "query_registry": query_registry,
        **tensor_columns,
    }
    output["column_sha256"] = {
        name: tensor_sha256(output[name]) for name in COLUMN_DTYPES
    }
    output["summary"] = summarize_rank_audit(output)
    validate_true_anchor_rank_audit(output)
    return output


def validate_true_anchor_rank_audit(payload: Mapping) -> None:
    if not (
        payload.get("schema") == SCHEMA
        and payload.get("version") == VERSION
        and payload.get("protocol") == "test_adapted"
        and payload.get("uses_test_queries") is True
        and payload.get("test_adapted") is True
        and payload.get("role") == ROLE
        and payload.get("complete_adaptation_registry_consumed") is True
        and payload.get("control_features_consumed") is False
        and payload.get("confirmation_features_consumed") is False
        and payload.get("control_or_confirmation_outcomes_consumed") is False
        and payload.get("diagnostic_only") is True
        and payload.get("action_authorized") is False
        and payload.get("controller_authorized") is False
        and payload.get("deployment_authorized") is False
        and payload.get("negative_anchor_labels_created") is False
        and payload.get("rank_definition") == RANK_DEFINITION
        and tuple(payload.get("rank_ks", ())) == RANK_KS
        and payload.get("semantics") == SEMANTICS
    ):
        raise ValueError("unsupported V21 true-Anchor rank audit")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 rank audit input lineage is missing")
    stable = source_record(inputs.get("stable_map"), label="stable map")
    truth = source_record(inputs.get("correspondence_truth"), label="correspondence truth")
    oracle = source_record(inputs.get("geometry_oracle"), label="geometry oracle")
    caches = inputs.get("frontend_caches")
    producers = inputs.get("producer_sources")
    if not isinstance(caches, list) or not caches or not isinstance(producers, list) or not producers:
        raise ValueError("V21 rank audit source registries are empty")
    [source_record(value, label="frontend cache") for value in caches]
    [source_record(value, label="producer") for value in producers]
    if not (
        payload.get("stable_map_sha256") == stable["sha256"]
        and payload.get("correspondence_truth_sha256") == truth["sha256"]
        and payload.get("geometry_oracle_sha256") == oracle["sha256"]
    ):
        raise ValueError("V21 rank audit primary SHA lineage differs")
    registry = payload.get("frontend_shard_registry")
    validate_shard_registry(registry)
    if payload.get("frontend_shard_registry_sha256") != registry["registry_sha256"]:
        raise ValueError("V21 rank audit frontend registry SHA differs")
    edge_count = int(payload.get("unique_edge_count", -1))
    anchor_count = int(payload.get("anchor_count", 0))
    query_count = int(payload.get("query_count", -1))
    query_registry = payload.get("query_registry")
    if (
        edge_count <= 0
        or anchor_count <= 0
        or int(payload.get("descriptor_dim", 0)) <= 0
        or not isinstance(query_registry, list)
        or len(query_registry) != query_count
        or query_count != int(registry["role_query_count"])
    ):
        raise ValueError("V21 rank audit dimensions/coverage are invalid")
    query_indices = set()
    ordinal_by_query = {}
    for ordinal, record in enumerate(query_registry):
        if (
            int(record.get("ordinal", -1)) != ordinal
            or int(record.get("query_index", -1)) < 0
            or int(record["query_index"]) in query_indices
            or not str(record.get("image_name", ""))
        ):
            raise ValueError("V21 rank audit query registry is invalid")
        _require_sha256(record.get("image_sha256"), label="query image")
        _require_sha256(record.get("source_record_sha256"), label="query source")
        query_indices.add(int(record["query_index"]))
        ordinal_by_query[int(record["query_index"])] = ordinal
    hashes = payload.get("column_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("V21 rank audit column hashes are missing")
    columns = {}
    for name, dtype in COLUMN_DTYPES.items():
        value = torch.as_tensor(payload.get(name))
        if value.shape != (edge_count,) or value.dtype != dtype:
            raise ValueError("V21 rank audit columns do not align")
        if hashes.get(name) != tensor_sha256(value):
            raise ValueError("V21 rank audit column SHA differs")
        columns[name] = value
    if not (
        bool(torch.isfinite(columns["true_anchor_scores"]).all())
        and bool(torch.isfinite(columns["baseline_winner_scores"]).all())
        and int(columns["true_anchor_rows"].min()) >= 0
        and int(columns["true_anchor_rows"].max()) < anchor_count
        and int(columns["baseline_winner_anchor_rows"].min()) >= 0
        and int(columns["baseline_winner_anchor_rows"].max()) < anchor_count
        and torch.equal(columns["strict_ranks"], columns["strict_greater_counts"] + 1)
        and torch.equal(
            columns["exact_tie_break_ranks"],
            columns["strict_ranks"] + columns["lower_row_equal_score_counts"],
        )
        and int(columns["strict_ranks"].min()) >= 1
        and int(columns["strict_ranks"].max()) <= anchor_count
        and int(columns["exact_tie_break_ranks"].max()) <= anchor_count
        and torch.equal(
            columns["baseline_winner_is_true_anchor"],
            columns["baseline_winner_anchor_rows"] == columns["true_anchor_rows"],
        )
        and not bool(
            (
                columns["geometry_recovery_bundle_exact_anchor_overlap"]
                & ~columns["geometry_recovery_bundle_query_row_overlap"]
            ).any()
        )
    ):
        raise ValueError("V21 rank audit rank/Anchor invariants differ")
    previous = None
    seen_edges = set()
    for ordinal, query, row in zip(
        columns["edge_query_ordinals"].tolist(),
        columns["edge_query_indices"].tolist(),
        columns["edge_query_rows"].tolist(),
    ):
        key = (int(ordinal), int(row))
        if (
            int(query) not in query_indices
            or ordinal_by_query[int(query)] != int(ordinal)
            or int(row) < 0
            or key in seen_edges
            or (previous is not None and key <= previous)
        ):
            raise ValueError("V21 rank audit edge ordering/registry differs")
        seen_edges.add(key)
        previous = key
    if payload.get("summary") != summarize_rank_audit(payload):
        raise ValueError("V21 rank audit summary differs")


def atomic_torch_save_fresh(payload: Mapping, output: str | Path) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V21 rank audit output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(dict(payload), temporary)
        validate_true_anchor_rank_audit(
            torch.load(temporary, map_location="cpu", weights_only=False)
        )
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"V21 rank audit output appeared while running: {output}"
            ) from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "RANK_DEFINITION",
    "RANK_KS",
    "SCHEMA",
    "SEMANTICS",
    "VERSION",
    "atomic_torch_save_fresh",
    "build_true_anchor_rank_audit",
    "exact_global_true_anchor_ranks",
    "source_record",
    "summarize_rank_audit",
    "validate_true_anchor_rank_audit",
]
