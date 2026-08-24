#!/usr/bin/env python3
"""Write paired diagnostics for two identity-safe V6 feedback evaluations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from common.hashing import sha256_file
from common.v6_contracts import FEEDBACK_SCHEMA, require_schema
from common.v6_pipeline_contract import FORMAL_FEEDBACK_CANDIDATE_ARMS


EVALUATION_SCHEMA = "lafgs_v6_query_local_feedback_evaluation"
OUTPUT_SCHEMA = "lafgs_v6_paired_feedback_diagnostics"
WINNER_CLASSES = (
    "exact_identity_correct",
    "geometry_compatible_ambiguous",
    "identity_projective_incompatible",
    "negative",
)
WINNER_MASK_FIELDS = (
    "top1_exact_identity_correct_mask",
    "top1_geometry_compatible_ambiguous_mask",
    "top1_identity_projective_incompatible_mask",
    "top1_negative_mask",
)
FAILURE_LAYERS = ("L1", "L2", "L3", "L4")
REQUIRED_PRODUCER_SOURCES = {
    "scripts/evaluate_v6_self_localization.py",
    "common/v6_contracts.py",
    "common/v6_pipeline_contract.py",
    "evidence/observation_provider.py",
    "map_learning/v6_feedback_evaluator.py",
    "map_learning/self_localization_feedback.py",
    "evidence/projective_loo.py",
    "evidence/projective_reconstruction.py",
    "localization/matcher.py",
    "localization/pose_solver.py",
    "topology/layered_sufficiency.py",
    "topology/pose_information.py",
}
PAIRED_PROTOCOL_FIELDS = (
    "positive_identity",
    "positive_radius_px",
    "alpha_minimum",
    "required_matching_rank",
    "required_visibility_rank",
    "required_detectable_rank",
    "pose_logdet_target",
    "pose_min_eigenvalue_target",
    "ransac_reprojection_px",
    "ransac_seed",
    "loo_pose_neighbors",
    "affected_anchor_policy",
    "global_top1",
    "pose_solves_per_query",
    "retrieval",
    "refinement",
)


def _is_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _require_sha(path: Path, expected: str, *, label: str) -> str:
    path = path.resolve()
    expected = str(expected).lower()
    if not _is_sha256(expected):
        raise ValueError(f"{label} expected SHA256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _finite_nonnegative(value: object, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _load_evaluation(path: Path, expected_sha256: str, *, label: str) -> dict:
    _require_sha(path, expected_sha256, label=label)
    value = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a dictionary")
    if value.get("schema") != EVALUATION_SCHEMA or int(value.get("version", -1)) != 4:
        raise ValueError(f"{label} is not a V6 identity-safe evaluation v4")
    if value.get("uses_source_mapping_rgb") is not False or value.get("uses_test_queries") is not False:
        raise ValueError(f"{label} is not mapping-only/test-free")
    feedback = value.get("feedback")
    if not isinstance(feedback, dict):
        raise ValueError(f"{label} embedded feedback is missing")
    require_schema(feedback, FEEDBACK_SCHEMA, label=f"{label} embedded feedback")
    contract = value.get("contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{label} evaluation contract is missing")
    if contract.get("global_top1") is not True:
        raise ValueError(f"{label} did not use global Top-1")
    if int(contract.get("pose_solves_per_query", -1)) != 1:
        raise ValueError(f"{label} did not use one PoseLib solve per query")
    missing_protocol = [key for key in PAIRED_PROTOCOL_FIELDS if key not in contract]
    if missing_protocol:
        raise ValueError(f"{label} evaluation protocol is incomplete: {missing_protocol}")
    protocol = {key: contract[key] for key in PAIRED_PROTOCOL_FIELDS}
    calibration_binding_map_role = contract.get("calibration_binding_map_role")
    calibration_binding_source_map_sha = contract.get(
        "calibration_binding_source_map_sha256"
    )
    calibration_binding_candidate_arm = contract.get(
        "calibration_binding_candidate_arm"
    )
    if calibration_binding_map_role not in {"current_map", "candidate_parent_map"}:
        raise ValueError(f"{label} calibration-binding map role is invalid")
    if not _is_sha256(calibration_binding_source_map_sha):
        raise ValueError(f"{label} calibration-binding source map SHA is invalid")
    if calibration_binding_map_role == "current_map":
        if (
            calibration_binding_source_map_sha != value.get("input_sha256", {}).get("map")
            or calibration_binding_candidate_arm is not None
        ):
            raise ValueError(f"{label} baseline calibration-binding lineage differs")
    elif calibration_binding_candidate_arm not in FORMAL_FEEDBACK_CANDIDATE_ARMS:
        raise ValueError(f"{label} calibration-binding candidate arm is invalid")
    pose_logdet_target = float(protocol["pose_logdet_target"])
    if not math.isfinite(pose_logdet_target):
        raise ValueError(f"{label} pose logdet target is not finite")
    _finite_nonnegative(
        protocol["pose_min_eigenvalue_target"],
        label=f"{label} pose minimum-eigenvalue target",
    )

    producer = value.get("producer")
    if not isinstance(producer, dict) or producer.get("worktree_clean") is not True:
        raise ValueError(f"{label} producer registry is missing or dirty")
    source_sha256 = producer.get("source_sha256")
    if (
        not isinstance(source_sha256, dict)
        or not source_sha256
        or not REQUIRED_PRODUCER_SOURCES <= set(source_sha256)
        or any(not _is_sha256(digest) for digest in source_sha256.values())
    ):
        raise ValueError(f"{label} producer source SHA registry is invalid")
    if not isinstance(producer.get("torch_version"), str):
        raise ValueError(f"{label} producer Torch version is missing")

    outer_input = value.get("input_sha256")
    feedback_input = feedback.get("input_sha256")
    if not isinstance(outer_input, dict) or not isinstance(feedback_input, dict):
        raise ValueError(f"{label} input SHA registry is missing")
    cache_sha = outer_input.get("observation_cache")
    map_sha = outer_input.get("map")
    metric_sha = outer_input.get("metric")
    scene_calibration_sha = outer_input.get("scene_calibration")
    feedback_calibration_binding_sha = outer_input.get(
        "feedback_calibration_binding"
    )
    if not all(
        _is_sha256(digest)
        for digest in (
            cache_sha,
            map_sha,
            metric_sha,
            scene_calibration_sha,
            feedback_calibration_binding_sha,
        )
    ):
        raise ValueError(
            f"{label} map/metric/cache/calibration SHA registry is invalid"
        )
    if feedback_input.get("query_cache") != cache_sha:
        raise ValueError(f"{label} outer/embedded observation cache differs")
    if feedback_input.get("map") != outer_input.get("map"):
        raise ValueError(f"{label} outer/embedded map SHA differs")
    for field in ("scene_calibration", "feedback_calibration_binding"):
        embedded_sha = feedback_input.get(field)
        if not _is_sha256(embedded_sha):
            raise ValueError(f"{label} embedded {field} SHA is invalid")
        if embedded_sha != outer_input[field]:
            raise ValueError(f"{label} outer/embedded {field} SHA differs")
    feedback_rank_fields = {
        "required_matching_rank": "required_matching_rank",
        "required_visibility_rank": "required_visibility_rank",
        "required_detectable_rank": "required_detectable_rank",
    }
    if any(
        int(feedback.get(feedback_key, -1)) != int(contract[contract_key])
        for feedback_key, contract_key in feedback_rank_fields.items()
    ):
        raise ValueError(f"{label} outer/embedded required-rank contract differs")
    for key in ("pose_logdet_target", "pose_min_eigenvalue_target"):
        if float(feedback.get(key, float("nan"))) != float(contract[key]):
            raise ValueError(
                f"{label} outer/embedded pose-information target differs at {key}"
            )

    names = list(feedback.get("query_names", ()))
    records = list(feedback.get("records", ()))
    queries = list(value.get("queries", ()))
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
        or len(names) != len(records)
        or len(names) != len(queries)
    ):
        raise ValueError(f"{label} query registries are empty or misaligned")
    normalized_records = []
    normalized_queries = []
    for query_index, (name, record, query) in enumerate(zip(names, records, queries)):
        if not isinstance(record, dict) or not isinstance(query, dict):
            raise ValueError(f"{label} query payload is not a dictionary")
        if (
            str(record.get("image_name")) != name
            or int(record.get("query_index", -1)) != query_index
            or str(query.get("image_name")) != name
            or int(query.get("query_index", -1)) != query_index
        ):
            raise ValueError(f"{label} ordered query registry differs at {query_index}")
        query_rows = torch.as_tensor(record.get("query_rows"), dtype=torch.long).reshape(-1)
        winners = torch.as_tensor(record.get("winner_anchor_ids"), dtype=torch.long).reshape(-1)
        if not torch.equal(query_rows, torch.arange(query_rows.numel())):
            raise ValueError(f"{label} query rows are not a complete ordered registry")
        if winners.shape != query_rows.shape or bool((winners < 0).any()):
            raise ValueError(f"{label} winner rows are invalid")
        masks = [
            torch.as_tensor(record.get(field), dtype=torch.bool).reshape(-1)
            for field in WINNER_MASK_FIELDS
        ]
        if any(mask.shape != query_rows.shape for mask in masks):
            raise ValueError(f"{label} winner-class masks are misaligned")
        partition = torch.stack(masks).long().sum(0)
        if query_rows.numel() and not bool((partition == 1).all()):
            raise ValueError(f"{label} winner classes are not a partition")
        inliers = torch.as_tensor(record.get("inlier_query_rows"), dtype=torch.long).reshape(-1)
        if inliers.numel() and (
            int(inliers.min()) < 0
            or int(inliers.max()) >= query_rows.numel()
            or inliers.numel() != torch.unique(inliers).numel()
        ):
            raise ValueError(f"{label} PoseLib inlier row registry is invalid")
        inlier_clean = torch.as_tensor(
            record.get("inlier_clean_mask"), dtype=torch.bool
        ).reshape(-1)
        if inlier_clean.shape != inliers.shape or not torch.equal(
            inlier_clean, masks[0][inliers]
        ):
            raise ValueError(f"{label} PoseLib inlier identity labels differ")
        failure_layers = tuple(str(layer) for layer in record.get("failure_layers", ()))
        if len(failure_layers) != len(set(failure_layers)) or not set(failure_layers) <= set(FAILURE_LAYERS):
            raise ValueError(f"{label} failure layers are invalid")
        te_cm = _finite_nonnegative(record.get("te_cm"), label=f"{label} te_cm")
        ae_deg = _finite_nonnegative(record.get("ae_deg"), label=f"{label} ae_deg")
        query_te = _finite_nonnegative(query.get("te_cm"), label=f"{label} query te_cm")
        query_ae = _finite_nonnegative(query.get("ae_deg"), label=f"{label} query ae_deg")
        if te_cm != query_te or ae_deg != query_ae:
            raise ValueError(f"{label} feedback/query pose errors differ")
        pose_success = record.get("pose_success")
        if not isinstance(pose_success, bool) or pose_success != (te_cm < 5.0 and ae_deg < 5.0):
            raise ValueError(f"{label} pose-success label differs from 5cm/5deg")
        independent = record.get("independent_mapping_validation_query")
        if not isinstance(independent, bool) or query.get("independent_mapping_validation_query") is not independent:
            raise ValueError(f"{label} independent-validation labels differ")
        positive_rows = int(query.get("positive_rows", -1))
        rank_at_1 = int(query.get("correct_anchor_rank_le_1", -1))
        rank_at_16 = int(query.get("correct_anchor_rank_le_16", -1))
        correct_winners = int(query.get("correct_winners", -1))
        if not (0 <= rank_at_1 <= rank_at_16 <= positive_rows <= query_rows.numel()):
            raise ValueError(f"{label} correct-rank counts are invalid")
        if correct_winners != int(masks[0].sum()):
            raise ValueError(f"{label} correct-winner count differs from identity masks")
        query_class_counts = (
            "top1_exact_identity_correct_rows",
            "top1_geometry_ambiguous_rows",
            "top1_identity_incompatible_rows",
            "top1_negative_rows",
        )
        if any(int(query.get(key, -1)) != int(mask.sum()) for key, mask in zip(query_class_counts, masks)):
            raise ValueError(f"{label} query winner-class counts differ from feedback")
        if (
            int(query.get("correspondences", -1)) != query_rows.numel()
            or int(query.get("inliers", -1)) != inliers.numel()
            or int(query.get("clean_inliers", -1)) != int(inlier_clean.sum())
            or int(query.get("pose_solves", -1)) != 1
        ):
            raise ValueError(f"{label} query correspondence/PoseLib counts differ")
        positive_pairs = torch.as_tensor(
            record.get("exact_identity_positive_pairs"), dtype=torch.long
        ).reshape(-1, 2)
        identity_positive_count = int(record.get("identity_positive_count", -1))
        if (
            identity_positive_count != positive_pairs.shape[0]
            or positive_rows != identity_positive_count
            or (
                positive_pairs.numel()
                and (
                    int(positive_pairs[:, 0].min()) < 0
                    or int(positive_pairs[:, 0].max()) >= query_rows.numel()
                    or int(positive_pairs[:, 1].min()) < 0
                    or positive_pairs.shape[0] != torch.unique(positive_pairs, dim=0).shape[0]
                )
            )
        ):
            raise ValueError(f"{label} exact-positive registry/count differs")
        correct_rank = int(record.get("correct_anchor_rank", -1))
        if correct_rank < 0 or (correct_rank == 0) != (positive_rows == 0):
            raise ValueError(f"{label} correct Anchor rank is invalid")
        estimated_pose = record.get("estimated_pose_w2c")
        if estimated_pose is None:
            raise ValueError(f"{label} estimated pose is missing")
        estimated_pose = torch.as_tensor(estimated_pose).float()
        if estimated_pose.shape != (4, 4) or not bool(torch.isfinite(estimated_pose).all()):
            raise ValueError(f"{label} estimated pose is invalid")
        normalized_records.append(
            {
                "query_index": query_index,
                "image_name": name,
                "query_rows": query_rows,
                "winners": winners,
                "winner_masks": masks,
                "inlier_rows": inliers,
                "failure_layers": failure_layers,
                "pose_success": pose_success,
                "te_cm": te_cm,
                "ae_deg": ae_deg,
                "correct_anchor_rank": correct_rank,
                "independent": independent,
                "estimated_pose_w2c": estimated_pose,
                "exact_identity_positive_pairs": positive_pairs,
            }
        )
        normalized_queries.append(
            {
                "positive_rows": positive_rows,
                "rank_at_1": rank_at_1,
                "rank_at_16": rank_at_16,
                "correct_winners": correct_winners,
            }
        )
    expected_class_totals = {
        key: sum(int(record["winner_masks"][index].sum()) for record in normalized_records)
        for index, key in enumerate(
            (
                "top1_exact_identity_correct_count",
                "top1_geometry_compatible_ambiguous_count",
                "top1_identity_projective_incompatible_count",
                "top1_negative_count",
            )
        )
    }
    if any(int(feedback.get(key, -1)) != count for key, count in expected_class_totals.items()):
        raise ValueError(f"{label} feedback winner-class totals differ")
    expected_failure_counts = {
        layer: sum(layer in record["failure_layers"] for record in normalized_records)
        for layer in FAILURE_LAYERS
    }
    if feedback.get("failure_layer_counts") != expected_failure_counts:
        raise ValueError(f"{label} feedback failure-layer totals differ")
    independent_count = sum(record["independent"] for record in normalized_records)
    if (
        int(feedback.get("independent_mapping_validation_query_count", -1))
        != independent_count
        or int(contract.get("independent_mapping_validation_query_count", -1))
        != independent_count
        or contract.get("independent_mapping_validation_available")
        is not bool(independent_count)
    ):
        raise ValueError(f"{label} independent-validation registry/count differs")
    summary = value.get("summary")
    if not isinstance(summary, dict) or int(summary.get("anchor_count", 0)) < 1:
        raise ValueError(f"{label} map Anchor count is missing")
    return {
        "payload": value,
        "feedback": feedback,
        "names": names,
        "records": normalized_records,
        "queries": normalized_queries,
        "map_sha256": str(map_sha),
        "metric_sha256": str(metric_sha),
        "cache_sha256": str(cache_sha),
        "scene_calibration_sha256": str(scene_calibration_sha),
        "feedback_calibration_binding_sha256": str(
            feedback_calibration_binding_sha
        ),
        "calibration_binding_map_role": calibration_binding_map_role,
        "calibration_binding_source_map_sha256": str(
            calibration_binding_source_map_sha
        ),
        "calibration_binding_candidate_arm": calibration_binding_candidate_arm,
        "anchor_count": int(summary["anchor_count"]),
        "protocol": protocol,
        "producer_source_sha256": dict(source_sha256),
        "torch_version": producer["torch_version"],
    }


def _winner_classes(record: dict) -> list[str]:
    stacked = torch.stack(record["winner_masks"])
    return [WINNER_CLASSES[int(index)] for index in torch.argmax(stacked.long(), dim=0)]


def _metric_summary(records: Sequence[dict]) -> dict:
    if not records:
        raise ValueError("metric summary requires at least one query")
    te = torch.tensor([record["te_cm"] for record in records], dtype=torch.float64)
    ae = torch.tensor([record["ae_deg"] for record in records], dtype=torch.float64)
    tail = max(int(math.ceil(0.05 * len(records))), 1)

    def distribution(values: torch.Tensor) -> dict:
        ordered = torch.sort(values).values
        return {
            "median": float(torch.quantile(values, 0.5)),
            "mean": float(values.mean()),
            "p90": float(torch.quantile(values, 0.9)),
            "cvar95": float(ordered[-tail:].mean()),
        }

    success = (te < 5.0) & (ae < 5.0)
    return {
        "query_count": len(records),
        "translation_cm": distribution(te),
        "rotation_deg": distribution(ae),
        "recall_5cm_5deg_percent": float(success.double().mean() * 100.0),
        "catastrophic_100cm_count": int((te >= 100.0).sum()),
    }


def _numeric_delta(candidate: object, baseline: object) -> object:
    if isinstance(candidate, dict) and isinstance(baseline, dict):
        return {key: _numeric_delta(candidate[key], baseline[key]) for key in candidate}
    if isinstance(candidate, (int, float)) and isinstance(baseline, (int, float)):
        return candidate - baseline
    raise TypeError("metric summaries have incompatible structures")


def _rank_summary(queries: Sequence[dict], records: Sequence[dict]) -> dict:
    positive_rows = sum(query["positive_rows"] for query in queries)
    rank_at_1 = sum(query["rank_at_1"] for query in queries)
    rank_at_16 = sum(query["rank_at_16"] for query in queries)
    correct_winners = sum(query["correct_winners"] for query in queries)
    ranks = [record["correct_anchor_rank"] for record in records if record["correct_anchor_rank"] > 0]
    return {
        "positive_row_count": positive_rows,
        "correct_winner_count": correct_winners,
        "correct_anchor_recall_at_1_percent": 100.0 * rank_at_1 / max(positive_rows, 1),
        "correct_anchor_recall_at_16_percent": 100.0 * rank_at_16 / max(positive_rows, 1),
        "query_minimum_correct_rank_available_count": len(ranks),
        "query_minimum_correct_rank_median": None if not ranks else float(torch.quantile(torch.tensor(ranks, dtype=torch.float64), 0.5)),
        "query_minimum_correct_rank_mean": None if not ranks else float(torch.tensor(ranks, dtype=torch.float64).mean()),
    }


def _rank_delta(candidate: dict, baseline: dict) -> dict:
    result = {}
    for key in candidate:
        left, right = candidate[key], baseline[key]
        result[key] = None if left is None or right is None else left - right
    return result


def _pose_output_diagnostics(baseline: Sequence[dict], candidate: Sequence[dict]) -> dict:
    if any(record["estimated_pose_w2c"] is None for record in baseline) or any(
        record["estimated_pose_w2c"] is None for record in candidate
    ):
        return {
            "available": False,
            "changed_query_count": None,
            "unchanged_query_count": None,
            "changed_query_indices": None,
            "maximum_absolute_matrix_delta": None,
            "unavailable_reason": "estimated_pose_w2c_missing_from_one_or_both_inputs",
        }
    changed = []
    maximum = 0.0
    for base, proposed in zip(baseline, candidate):
        delta = torch.abs(proposed["estimated_pose_w2c"] - base["estimated_pose_w2c"])
        maximum = max(maximum, float(delta.max()))
        if not torch.equal(proposed["estimated_pose_w2c"], base["estimated_pose_w2c"]):
            changed.append(base["query_index"])
    return {
        "available": True,
        "changed_query_count": len(changed),
        "unchanged_query_count": len(baseline) - len(changed),
        "changed_query_indices": changed,
        "maximum_absolute_matrix_delta": maximum,
        "unavailable_reason": None,
    }


def _coverage(
    records: Sequence[dict],
    updated: torch.Tensor | None,
    *,
    unavailable_reason: str | None,
) -> dict:
    if updated is None:
        return {
            "available": False,
            "unavailable_reason": unavailable_reason,
            "updated_anchor_count": None,
            "winner": None,
            "poselib_inlier": None,
            "catastrophic_queries": None,
        }
    updated = torch.unique(updated.long(), sorted=True)
    winner_total = 0
    winner_hit = 0
    winner_query_hit = 0
    inlier_total = 0
    inlier_hit = 0
    inlier_query_hit = 0
    catastrophic = 0
    catastrophic_winner_hit = 0
    catastrophic_inlier_hit = 0
    for record in records:
        winner_mask = torch.isin(record["winners"], updated)
        inlier_mask = winner_mask[record["inlier_rows"]]
        winner_total += int(winner_mask.numel())
        winner_hit += int(winner_mask.sum())
        winner_query_hit += int(bool(winner_mask.any()))
        inlier_total += int(inlier_mask.numel())
        inlier_hit += int(inlier_mask.sum())
        inlier_query_hit += int(bool(inlier_mask.any()))
        if record["te_cm"] >= 100.0:
            catastrophic += 1
            catastrophic_winner_hit += int(bool(winner_mask.any()))
            catastrophic_inlier_hit += int(bool(inlier_mask.any()))
    return {
        "available": True,
        "unavailable_reason": None,
        "updated_anchor_count": int(updated.numel()),
        "winner": {
            "occurrence_count": winner_total,
            "updated_occurrence_count": winner_hit,
            "updated_occurrence_percent": 100.0 * winner_hit / max(winner_total, 1),
            "query_count_with_updated_winner": winner_query_hit,
            "query_coverage_percent": 100.0 * winner_query_hit / max(len(records), 1),
        },
        "poselib_inlier": {
            "occurrence_count": inlier_total,
            "updated_occurrence_count": inlier_hit,
            "updated_occurrence_percent": 100.0 * inlier_hit / max(inlier_total, 1),
            "query_count_with_updated_inlier": inlier_query_hit,
            "query_coverage_percent": 100.0 * inlier_query_hit / max(len(records), 1),
        },
        "catastrophic_queries": {
            "count": catastrophic,
            "with_updated_winner_count": catastrophic_winner_hit,
            "with_updated_winner_percent": 100.0 * catastrophic_winner_hit / max(catastrophic, 1),
            "with_updated_inlier_count": catastrophic_inlier_hit,
            "with_updated_inlier_percent": 100.0 * catastrophic_inlier_hit / max(catastrophic, 1),
        },
    }


def _paired_scope(
    baseline: dict,
    candidate: dict,
    indices: Sequence[int],
    *,
    updated_anchor_rows: torch.Tensor | None,
    updated_anchor_unavailable_reason: str | None,
    baseline_anchor_identity_digests: Sequence[str],
    candidate_anchor_identity_digests: Sequence[str],
) -> dict:
    base_records = [baseline["records"][index] for index in indices]
    proposed_records = [candidate["records"][index] for index in indices]
    base_queries = [baseline["queries"][index] for index in indices]
    proposed_queries = [candidate["queries"][index] for index in indices]
    if not base_records:
        return {
            "available": False,
            "query_count": 0,
            "query_indices": [],
            "unavailable_reason": "empty_scope",
        }

    transitions = {
        source: {target: 0 for target in WINNER_CLASSES} for source in WINNER_CLASSES
    }
    changed_rows = []
    wrong_to_correct = []
    correct_to_wrong = []
    inlier_changed_queries = []
    inlier_pair_changes = []
    inlier_gained = 0
    inlier_lost = 0
    pose_gained = []
    pose_lost = []
    failure = {
        layer: {"entered_query_indices": [], "exited_query_indices": []}
        for layer in FAILURE_LAYERS
    }
    rank_improved = []
    rank_regressed = []
    rank_equal = []
    rank_availability_gained = []
    rank_availability_lost = []
    rank_unavailable_in_both = []
    for base, proposed in zip(base_records, proposed_records):
        if not torch.equal(base["query_rows"], proposed["query_rows"]):
            raise ValueError(f"paired query rows differ for {base['image_name']}")
        base_classes = _winner_classes(base)
        proposed_classes = _winner_classes(proposed)
        base_winner_identities = [
            baseline_anchor_identity_digests[int(row)] for row in base["winners"]
        ]
        proposed_winner_identities = [
            candidate_anchor_identity_digests[int(row)] for row in proposed["winners"]
        ]
        for row, (source, target) in enumerate(zip(base_classes, proposed_classes)):
            transitions[source][target] += 1
            base_exact = source == "exact_identity_correct"
            proposed_exact = target == "exact_identity_correct"
            if not base_exact and proposed_exact:
                wrong_to_correct.append([base["query_index"], row])
            if base_exact and not proposed_exact:
                correct_to_wrong.append([base["query_index"], row])
            if base_winner_identities[row] != proposed_winner_identities[row]:
                changed_rows.append(
                    {
                        "query_index": base["query_index"],
                        "image_name": base["image_name"],
                        "query_row": row,
                        "baseline_winner_local_row": int(base["winners"][row]),
                        "candidate_winner_local_row": int(proposed["winners"][row]),
                        "baseline_winner_identity_sha256": base_winner_identities[row],
                        "candidate_winner_identity_sha256": proposed_winner_identities[row],
                    }
                )
        base_inliers = {
            (int(row), base_winner_identities[int(row)])
            for row in base["inlier_rows"].tolist()
        }
        proposed_inliers = {
            (int(row), proposed_winner_identities[int(row)])
            for row in proposed["inlier_rows"].tolist()
        }
        if base_inliers != proposed_inliers:
            inlier_changed_queries.append(base["query_index"])
            inlier_pair_changes.append(
                {
                    "query_index": base["query_index"],
                    "image_name": base["image_name"],
                    "baseline_pairs": [list(pair) for pair in sorted(base_inliers)],
                    "candidate_pairs": [list(pair) for pair in sorted(proposed_inliers)],
                    "gained_pairs": [
                        list(pair) for pair in sorted(proposed_inliers - base_inliers)
                    ],
                    "lost_pairs": [
                        list(pair) for pair in sorted(base_inliers - proposed_inliers)
                    ],
                }
            )
        inlier_gained += len(proposed_inliers - base_inliers)
        inlier_lost += len(base_inliers - proposed_inliers)
        if not base["pose_success"] and proposed["pose_success"]:
            pose_gained.append(base["query_index"])
        if base["pose_success"] and not proposed["pose_success"]:
            pose_lost.append(base["query_index"])
        for layer in FAILURE_LAYERS:
            if layer not in base["failure_layers"] and layer in proposed["failure_layers"]:
                failure[layer]["entered_query_indices"].append(base["query_index"])
            if layer in base["failure_layers"] and layer not in proposed["failure_layers"]:
                failure[layer]["exited_query_indices"].append(base["query_index"])
        base_rank = base["correct_anchor_rank"]
        proposed_rank = proposed["correct_anchor_rank"]
        if base_rank == 0 and proposed_rank > 0:
            rank_availability_gained.append(base["query_index"])
            rank_improved.append(base["query_index"])
        elif base_rank > 0 and proposed_rank == 0:
            rank_availability_lost.append(base["query_index"])
            rank_regressed.append(base["query_index"])
        elif base_rank == 0 and proposed_rank == 0:
            rank_unavailable_in_both.append(base["query_index"])
        else:
            if proposed_rank < base_rank:
                rank_improved.append(base["query_index"])
            elif proposed_rank > base_rank:
                rank_regressed.append(base["query_index"])
            else:
                rank_equal.append(base["query_index"])

    baseline_metrics = _metric_summary(base_records)
    candidate_metrics = _metric_summary(proposed_records)
    baseline_rank = _rank_summary(base_queries, base_records)
    candidate_rank = _rank_summary(proposed_queries, proposed_records)
    te_better = sum(proposed["te_cm"] < base["te_cm"] for base, proposed in zip(base_records, proposed_records))
    te_worse = sum(proposed["te_cm"] > base["te_cm"] for base, proposed in zip(base_records, proposed_records))
    ae_better = sum(proposed["ae_deg"] < base["ae_deg"] for base, proposed in zip(base_records, proposed_records))
    ae_worse = sum(proposed["ae_deg"] > base["ae_deg"] for base, proposed in zip(base_records, proposed_records))
    for layer in FAILURE_LAYERS:
        baseline_count = sum(layer in record["failure_layers"] for record in base_records)
        candidate_count = sum(
            layer in record["failure_layers"] for record in proposed_records
        )
        failure[layer]["baseline_count"] = baseline_count
        failure[layer]["candidate_count"] = candidate_count
        failure[layer]["delta_candidate_minus_baseline"] = (
            candidate_count - baseline_count
        )
        failure[layer]["entered_count"] = len(failure[layer]["entered_query_indices"])
        failure[layer]["exited_count"] = len(failure[layer]["exited_query_indices"])
    return {
        "available": True,
        "query_count": len(indices),
        "query_indices": list(indices),
        "unavailable_reason": None,
        "metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "delta_candidate_minus_baseline": _numeric_delta(candidate_metrics, baseline_metrics),
            "paired_translation_better_count": te_better,
            "paired_translation_worse_count": te_worse,
            "paired_translation_equal_count": len(indices) - te_better - te_worse,
            "paired_rotation_better_count": ae_better,
            "paired_rotation_worse_count": ae_worse,
            "paired_rotation_equal_count": len(indices) - ae_better - ae_worse,
        },
        "top1": {
            "row_count": sum(record["query_rows"].numel() for record in base_records),
            "changed_row_count": len(changed_rows),
            "changed_query_count": len({row["query_index"] for row in changed_rows}),
            "changed_rows": changed_rows,
            "exact_wrong_to_correct_count": len(wrong_to_correct),
            "exact_wrong_to_correct_rows": wrong_to_correct,
            "exact_correct_to_wrong_count": len(correct_to_wrong),
            "exact_correct_to_wrong_rows": correct_to_wrong,
            "winner_class_transition": transitions,
            "winner_identifier_contract": (
                "sha256_of_sorted_unique_projective_query_keypoint_pairs"
            ),
            "anchor_identity_comparable": True,
            "changed_row_semantics": "stable_projective_observation_identity_change",
        },
        "poselib_inlier_pair_set": {
            "changed_query_count": len(inlier_changed_queries),
            "changed_query_indices": inlier_changed_queries,
            "changed_queries": inlier_pair_changes,
            "gained_pair_count": inlier_gained,
            "lost_pair_count": inlier_lost,
        },
        "pose_success_flips": {
            "failure_to_success_count": len(pose_gained),
            "failure_to_success_query_indices": pose_gained,
            "success_to_failure_count": len(pose_lost),
            "success_to_failure_query_indices": pose_lost,
        },
        "failure_layer_transitions": failure,
        "correct_rank": {
            "baseline": baseline_rank,
            "candidate": candidate_rank,
            "delta_candidate_minus_baseline": _rank_delta(candidate_rank, baseline_rank),
            "improved_query_count": len(rank_improved),
            "improved_query_indices": rank_improved,
            "regressed_query_count": len(rank_regressed),
            "regressed_query_indices": rank_regressed,
            "equal_query_count": len(rank_equal),
            "availability_gained_query_count": len(rank_availability_gained),
            "availability_gained_query_indices": rank_availability_gained,
            "availability_lost_query_count": len(rank_availability_lost),
            "availability_lost_query_indices": rank_availability_lost,
            "unavailable_in_both_query_count": len(rank_unavailable_in_both),
            "unavailable_in_both_query_indices": rank_unavailable_in_both,
        },
        "pose_output_changed": _pose_output_diagnostics(base_records, proposed_records),
        "updated_anchor_coverage": _coverage(
            proposed_records,
            updated_anchor_rows,
            unavailable_reason=updated_anchor_unavailable_reason,
        ),
    }


def _anchor_identity_digest(identity: tuple[tuple[int, int], ...]) -> str:
    serialized = json.dumps(identity, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def _load_anchor_map(
    path: Path,
    expected_sha256: str,
    *,
    label: str,
    evaluation: dict,
) -> dict:
    actual = _require_sha(path, expected_sha256, label=label)
    if actual != evaluation["map_sha256"]:
        raise ValueError(f"{label} is not bound to its feedback")
    state = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or state.get("schema") != "lafgs_materialized_anchor_map"
        or int(state.get("version", -1)) != 1
    ):
        raise ValueError(f"{label} schema differs")
    provenance = state.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("uses_source_mapping_rgb") is not False
        or provenance.get("uses_test_queries") is not False
    ):
        raise ValueError(f"{label} is not mapping-only/test-free")
    if list(state.get("v6_mapping_query_names", ())) != evaluation["names"]:
        raise ValueError(f"{label} query registry differs")
    anchor_ids = torch.as_tensor(state.get("anchor_ids"), dtype=torch.long)
    anchor_count = int(anchor_ids.numel())
    if (
        anchor_ids.ndim != 1
        or anchor_count < 1
        or not torch.equal(anchor_ids, torch.arange(anchor_count))
        or anchor_count != evaluation["anchor_count"]
    ):
        raise ValueError(f"{label} Anchor registry differs")
    for record in evaluation["records"]:
        if record["winners"].numel() and int(record["winners"].max()) >= anchor_count:
            raise ValueError(f"{label} feedback winner is outside the map registry")
        positive_pairs = record["exact_identity_positive_pairs"]
        if positive_pairs.numel() and int(positive_pairs[:, 1].max()) >= anchor_count:
            raise ValueError(f"{label} feedback positive is outside the map registry")

    csr = state.get("projective_anchor_observations")
    if (
        not isinstance(csr, dict)
        or csr.get("schema") != "lafgs_projective_anchor_observations"
        or int(csr.get("version", -1)) != 1
    ):
        raise ValueError(f"{label} projective identity CSR is missing")
    offsets = torch.as_tensor(csr.get("observation_offsets"), dtype=torch.long)
    query_indices = torch.as_tensor(csr.get("query_indices"), dtype=torch.long).reshape(-1)
    keypoint_indices = torch.as_tensor(
        csr.get("keypoint_indices"), dtype=torch.long
    ).reshape(-1)
    if (
        offsets.ndim != 1
        or offsets.numel() != anchor_count + 1
        or int(offsets[0]) != 0
        or bool((offsets[1:] < offsets[:-1]).any())
        or int(offsets[-1]) != query_indices.numel()
        or query_indices.shape != keypoint_indices.shape
        or bool((query_indices < 0).any())
        or bool((query_indices >= len(evaluation["names"])).any())
        or bool((keypoint_indices < 0).any())
    ):
        raise ValueError(f"{label} projective identity CSR is invalid")
    digests = []
    for anchor_row in range(anchor_count):
        start, stop = int(offsets[anchor_row]), int(offsets[anchor_row + 1])
        observations = [
            (int(query), int(keypoint))
            for query, keypoint in zip(
                query_indices[start:stop].tolist(),
                keypoint_indices[start:stop].tolist(),
            )
        ]
        identity = tuple(sorted(set(observations)))
        if not identity or len(identity) != len(observations):
            raise ValueError(f"{label} Anchor projective identity is empty or duplicated")
        digests.append(_anchor_identity_digest(identity))
    if len(set(digests)) != anchor_count:
        raise ValueError(f"{label} projective Anchor fingerprints are not unique")
    return {
        "path": str(path.resolve()),
        "sha256": actual,
        "state": state,
        "provenance": provenance,
        "anchor_count": anchor_count,
        "identity_digests": digests,
    }


def _load_map_pair(
    *,
    baseline_map: Path,
    expected_baseline_map_sha256: str,
    candidate_map: Path,
    expected_candidate_map_sha256: str,
    baseline: dict,
    candidate: dict,
) -> tuple[dict, dict, torch.Tensor | None]:
    baseline_input = _load_anchor_map(
        baseline_map,
        expected_baseline_map_sha256,
        label="baseline map",
        evaluation=baseline,
    )
    candidate_input = _load_anchor_map(
        candidate_map,
        expected_candidate_map_sha256,
        label="candidate map",
        evaluation=candidate,
    )
    parent_sha = candidate_input["provenance"].get("v6_parent_map_sha256")
    if parent_sha != baseline_input["sha256"]:
        raise ValueError("candidate map parent is not the baseline map")
    candidate_arm = candidate["calibration_binding_candidate_arm"]
    if candidate_input["provenance"].get("v6_latest_proposal_arm") != candidate_arm:
        raise ValueError("candidate map arm differs from calibration-binding arm")
    history = candidate_input["provenance"].get("v6_proposal_history")
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
        raise ValueError("candidate map proposal history is missing")
    if (
        history[-1].get("parent_map_sha256") != baseline_input["sha256"]
        or history[-1].get("arm") != candidate_arm
    ):
        raise ValueError("candidate map proposal history differs from calibration")

    baseline_identities = set(baseline_input["identity_digests"])
    candidate_identities = set(candidate_input["identity_digests"])
    candidate_input["matched_parent_anchor_count"] = len(
        baseline_identities & candidate_identities
    )
    candidate_input["new_anchor_count"] = len(candidate_identities - baseline_identities)
    candidate_input["dropped_parent_anchor_count"] = len(
        baseline_identities - candidate_identities
    )

    report = candidate_input["state"].get("v6_descriptor_distillation")
    updated = None
    if isinstance(report, dict) and "updated_anchor_rows" in report:
        valid_reports = {
            "lafgs_v6_counterfactual_descriptor_distillation": 2,
            "lafgs_v6_counterfactual_descriptor_loss_distillation": 4,
        }
        if valid_reports.get(report.get("schema")) != int(report.get("version", -1)):
            raise ValueError("candidate descriptor distillation contract differs")
        raw_updated = torch.as_tensor(
            report["updated_anchor_rows"], dtype=torch.long
        ).reshape(-1)
        updated = torch.unique(raw_updated, sorted=True)
        if (
            updated.numel() != raw_updated.numel()
            or (
                updated.numel()
                and (
                    int(updated.min()) < 0
                    or int(updated.max()) >= candidate_input["anchor_count"]
                )
            )
        ):
            raise ValueError("candidate updated Anchor registry is invalid")
        declared_count = report.get("updated_anchor_count")
        if declared_count is not None and int(declared_count) != updated.numel():
            raise ValueError("candidate updated Anchor count differs")
    return baseline_input, candidate_input, updated


def compare_feedback_files(
    *,
    baseline_feedback: Path,
    expected_baseline_feedback_sha256: str,
    candidate_feedback: Path,
    expected_candidate_feedback_sha256: str,
    baseline_map: Path,
    expected_baseline_map_sha256: str,
    candidate_map: Path,
    expected_candidate_map_sha256: str,
) -> dict:
    baseline = _load_evaluation(
        baseline_feedback, expected_baseline_feedback_sha256, label="baseline feedback"
    )
    candidate = _load_evaluation(
        candidate_feedback, expected_candidate_feedback_sha256, label="candidate feedback"
    )
    if baseline["names"] != candidate["names"]:
        raise ValueError("baseline and candidate query registries differ")
    if baseline["calibration_binding_map_role"] != "current_map":
        raise ValueError("baseline calibration binding does not bind its current map")
    if candidate["calibration_binding_map_role"] != "candidate_parent_map":
        raise ValueError("candidate calibration binding does not bind its parent map")
    if (
        baseline["calibration_binding_source_map_sha256"]
        != baseline["map_sha256"]
    ):
        raise ValueError("baseline calibration-binding source map differs")
    if (
        candidate["calibration_binding_source_map_sha256"]
        != baseline["map_sha256"]
    ):
        raise ValueError("candidate calibration-binding parent is not the baseline")
    if baseline["cache_sha256"] != candidate["cache_sha256"]:
        raise ValueError("baseline and candidate observation caches differ")
    if (
        baseline["scene_calibration_sha256"]
        != candidate["scene_calibration_sha256"]
    ):
        raise ValueError("baseline and candidate scene calibrations differ")
    if (
        baseline["feedback_calibration_binding_sha256"]
        != candidate["feedback_calibration_binding_sha256"]
    ):
        raise ValueError(
            "baseline and candidate feedback calibration bindings differ"
        )
    if baseline["protocol"] != candidate["protocol"]:
        raise ValueError("baseline and candidate evaluation protocols differ")
    if baseline["producer_source_sha256"] != candidate["producer_source_sha256"]:
        raise ValueError("baseline and candidate evaluator source registries differ")
    if baseline["torch_version"] != candidate["torch_version"]:
        raise ValueError("baseline and candidate Torch versions differ")
    baseline_map_input, candidate_map_input, updated = _load_map_pair(
        baseline_map=baseline_map,
        expected_baseline_map_sha256=expected_baseline_map_sha256,
        candidate_map=candidate_map,
        expected_candidate_map_sha256=expected_candidate_map_sha256,
        baseline=baseline,
        candidate=candidate,
    )
    full_indices = list(range(len(candidate["records"])))
    independent_indices = [
        index
        for index, (base, proposed) in enumerate(
            zip(baseline["records"], candidate["records"])
        )
        if base["independent"] and proposed["independent"]
    ]
    updated_unavailable_reason = "candidate_map_has_no_updated_anchor_rows"
    independent = _paired_scope(
        baseline,
        candidate,
        independent_indices,
        updated_anchor_rows=updated,
        updated_anchor_unavailable_reason=updated_unavailable_reason,
        baseline_anchor_identity_digests=baseline_map_input["identity_digests"],
        candidate_anchor_identity_digests=candidate_map_input["identity_digests"],
    )
    baseline_map_report = {
        key: baseline_map_input[key]
        for key in ("path", "sha256", "anchor_count")
    }
    baseline_map_report["unique_projective_identity_count"] = baseline_map_input[
        "anchor_count"
    ]
    candidate_map_report = {
        key: candidate_map_input[key]
        for key in (
            "path",
            "sha256",
            "anchor_count",
            "matched_parent_anchor_count",
            "new_anchor_count",
            "dropped_parent_anchor_count",
        )
    }
    candidate_map_report["parent_map_sha256"] = candidate_map_input["provenance"][
        "v6_parent_map_sha256"
    ]
    candidate_map_report["unique_projective_identity_count"] = candidate_map_input[
        "anchor_count"
    ]
    return {
        "schema": OUTPUT_SCHEMA,
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "valid": True,
        "comparison_contract": {
            "paired_ordered_query_registry": True,
            "shared_observation_cache_sha256": baseline["cache_sha256"],
            "shared_scene_calibration_sha256": baseline[
                "scene_calibration_sha256"
            ],
            "shared_feedback_calibration_binding_sha256": baseline[
                "feedback_calibration_binding_sha256"
            ],
            "immutable_calibration_source_map_sha256": baseline[
                "map_sha256"
            ],
            "candidate_calibration_binding_arm": candidate[
                "calibration_binding_candidate_arm"
            ],
            "identity_safe_feedback_required": True,
            "independent_subset_source": (
                "baseline_and_candidate.feedback.records."
                "independent_mapping_validation_query"
            ),
            "winner_identifier_semantics": (
                "sha256_of_sorted_unique_projective_query_keypoint_pairs"
            ),
            "cross_map_anchor_identity_verified": True,
            "evaluator_protocol_equal": True,
            "evaluator_source_registry_equal": True,
        },
        "inputs": {
            "baseline_feedback": {
                "path": str(baseline_feedback.resolve()),
                "sha256": expected_baseline_feedback_sha256.lower(),
                "map_sha256": baseline["map_sha256"],
            },
            "candidate_feedback": {
                "path": str(candidate_feedback.resolve()),
                "sha256": expected_candidate_feedback_sha256.lower(),
                "map_sha256": candidate["map_sha256"],
            },
            "baseline_map": baseline_map_report,
            "candidate_map": candidate_map_report,
        },
        "scopes": {
            "full": _paired_scope(
                baseline,
                candidate,
                full_indices,
                updated_anchor_rows=updated,
                updated_anchor_unavailable_reason=updated_unavailable_reason,
                baseline_anchor_identity_digests=baseline_map_input[
                    "identity_digests"
                ],
                candidate_anchor_identity_digests=candidate_map_input[
                    "identity_digests"
                ],
            ),
            "independent_validation_intersection": independent,
        },
    }


def _atomic_json(payload: dict, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()) != payload:
            raise RuntimeError("temporary paired diagnostics did not reload exactly")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace | SimpleNamespace) -> dict:
    result = compare_feedback_files(
        baseline_feedback=args.baseline_feedback,
        expected_baseline_feedback_sha256=args.expected_baseline_feedback_sha256,
        candidate_feedback=args.candidate_feedback,
        expected_candidate_feedback_sha256=args.expected_candidate_feedback_sha256,
        baseline_map=args.baseline_map,
        expected_baseline_map_sha256=args.expected_baseline_map_sha256,
        candidate_map=args.candidate_map,
        expected_candidate_map_sha256=args.expected_candidate_map_sha256,
    )
    _atomic_json(result, args.output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-feedback", type=Path, required=True)
    parser.add_argument("--expected-baseline-feedback-sha256", required=True)
    parser.add_argument("--candidate-feedback", type=Path, required=True)
    parser.add_argument("--expected-candidate-feedback-sha256", required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--expected-baseline-map-sha256", required=True)
    parser.add_argument("--candidate-map", type=Path, required=True)
    parser.add_argument("--expected-candidate-map-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
