#!/usr/bin/env python3
"""Fail-closed mechanism gate for the mapping-only frontend descriptor Arm B."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

from common.evaluation_code import frontend_descriptor_evaluation_code_identity
from common.hashing import sha256_file


REPORT_SCHEMA = "lafgs_frontend_ceiling_probe_audit_bundle"
DESCRIPTOR_SCHEMA = "lafgs_mapping_descriptor_identity_ceiling_probe"
DESCRIPTOR_EQUAL_ENERGY_SCHEMA = "lafgs_mapping_descriptor_equal_energy_ceiling_probe"
GATE_SCHEMA = "lafgs_frontend_descriptor_arm_b_mechanism_gate"
GATE_EQUAL_ENERGY_SCHEMA = "lafgs_frontend_descriptor_equal_energy_mechanism_gate"
EXPECTED_CROSSFIT_BLOCKS = 8
EXPECTED_MINIMUM_SUPPORT_VIEWS = 2
EXPECTED_TOPKS = (1, 2, 4, 8, 16, 32)
EXPECTED_DIRECTION_NAMES = ("selection_to_gate", "gate_to_selection")
NON_REGRESSION_ABS_TOLERANCE = 1e-12
REPORTED_DELTA_ABS_TOLERANCE = 1e-15


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Descriptor report must contain a JSON object")
    return payload


def _sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be 64 hexadecimal digits")
    return normalized


def _resolved_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def _mapping(value: object, *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _nonnegative_int(value: object, *, label: str) -> int:
    number = _exact_int(value, label=label)
    if number < 0:
        raise ValueError(f"{label} must be nonnegative")
    return number


def _unit_interval(value: object, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return number


def _validate_source_entry(
    entry: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    label: str,
) -> dict:
    payload = _mapping(entry, label=f"{label} report entry")
    reported_path_text = str(payload.get("path", ""))
    if not reported_path_text or "://" in reported_path_text:
        raise ValueError(f"{label} report path is not a local artifact")
    reported_path = Path(reported_path_text).expanduser().resolve()
    if reported_path != expected_path:
        raise ValueError(f"{label} report path differs from expected")
    if payload.get("sha256") != expected_sha256:
        raise ValueError(f"{label} report SHA-256 differs from expected")
    if payload.get("status") != "present_unverified":
        raise ValueError(f"{label} report inspection status is not exact")
    if payload.get("expected_sha256") is not None:
        raise ValueError(f"{label} report unexpectedly rewrites expected SHA-256")
    size_bytes = _nonnegative_int(
        payload.get("size_bytes"), label=f"{label} report size"
    )
    if size_bytes != expected_path.stat().st_size:
        raise ValueError(f"{label} report size differs from current artifact")
    if sha256_file(expected_path) != expected_sha256:
        raise ValueError(f"{label} current artifact SHA-256 differs from expected")
    return {
        "path": str(expected_path),
        "sha256": expected_sha256,
        "size_bytes": size_bytes,
    }


def _validate_attested_artifact(
    entry: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    label: str,
) -> dict:
    payload = _mapping(entry, label=f"{label} attestation")
    reported_path_text = str(payload.get("path", ""))
    if not reported_path_text or "://" in reported_path_text:
        raise ValueError(f"{label} attestation path is not local")
    if Path(reported_path_text).expanduser().resolve() != expected_path:
        raise ValueError(f"{label} attestation path differs from expected")
    if payload.get("sha256") != expected_sha256:
        raise ValueError(f"{label} attestation SHA-256 differs from expected")
    if payload.get("verified") is not True:
        raise ValueError(f"{label} attestation is not verified")
    if sha256_file(expected_path) != expected_sha256:
        raise ValueError(f"{label} current artifact SHA-256 differs from expected")
    return {"path": str(expected_path), "sha256": expected_sha256}


def _validate_recall_summary(summary: object, *, label: str) -> Mapping:
    payload = _mapping(summary, label=label)
    row_counts = _mapping(payload.get("row_counts"), label=f"{label} row_counts")
    expected_count_keys = {
        "replayed_rows",
        "positive_eligible_rows",
        "track_positive_eligible_rows",
        "reserve_positive_eligible_rows",
        "top1_correct",
        "top1_false",
        "top1_ambiguous",
    }
    if set(row_counts) != expected_count_keys:
        raise ValueError(f"{label} row-count registry differs")
    counts = {
        key: _nonnegative_int(row_counts[key], label=f"{label} {key}")
        for key in expected_count_keys
    }
    if counts["top1_correct"] > counts["positive_eligible_rows"]:
        raise ValueError(f"{label} top-1 correct count exceeds eligible rows")

    topk_keys = {str(value) for value in EXPECTED_TOPKS}
    recall = _mapping(payload.get("positive_recall_at_k"), label=f"{label} recall")
    if set(recall) != topk_keys:
        raise ValueError(f"{label} top-K registry differs from preregistration")
    previous = -math.inf
    for topk in EXPECTED_TOPKS:
        value = _unit_interval(recall[str(topk)], label=f"{label} R@{topk}")
        if value + NON_REGRESSION_ABS_TOLERANCE < previous:
            raise ValueError(f"{label} recall is not monotonic in K")
        previous = value
    expected_r1 = counts["top1_correct"] / max(counts["positive_eligible_rows"], 1)
    if not math.isclose(
        float(recall["1"]),
        expected_r1,
        rel_tol=0.0,
        abs_tol=REPORTED_DELTA_ABS_TOLERANCE,
    ):
        raise ValueError(f"{label} R@1 differs from additive counts")

    by_kind = _mapping(
        payload.get("positive_recall_at_k_by_anchor_kind"),
        label=f"{label} recall by anchor kind",
    )
    if set(by_kind) != {"track_core", "gaussian_reserve"}:
        raise ValueError(f"{label} anchor-kind registry differs")
    for kind in ("track_core", "gaussian_reserve"):
        values = _mapping(by_kind[kind], label=f"{label} {kind} recall")
        if set(values) != topk_keys:
            raise ValueError(f"{label} {kind} top-K registry differs")
        previous = -math.inf
        for topk in EXPECTED_TOPKS:
            value = _unit_interval(values[str(topk)], label=f"{label} {kind} R@{topk}")
            if value + NON_REGRESSION_ABS_TOLERANCE < previous:
                raise ValueError(f"{label} {kind} recall is not monotonic")
            previous = value
    return payload


def _validate_reported_delta(
    delta: object,
    *,
    baseline: Mapping,
    candidate: Mapping,
    label: str,
) -> None:
    payload = _mapping(delta, label=f"{label} delta")
    topk_keys = {str(value) for value in EXPECTED_TOPKS}
    if set(payload) != topk_keys | {"by_anchor_kind"}:
        raise ValueError(f"{label} delta registry differs")
    for topk in EXPECTED_TOPKS:
        key = str(topk)
        expected = float(candidate["positive_recall_at_k"][key]) - float(
            baseline["positive_recall_at_k"][key]
        )
        actual = _finite_number(payload[key], label=f"{label} delta R@{key}")
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=REPORTED_DELTA_ABS_TOLERANCE,
        ):
            raise ValueError(f"{label} reported R@{key} delta is not exact")
    by_kind = _mapping(
        payload.get("by_anchor_kind"), label=f"{label} delta by anchor kind"
    )
    if set(by_kind) != {"track_core", "gaussian_reserve"}:
        raise ValueError(f"{label} delta anchor-kind registry differs")
    for kind in ("track_core", "gaussian_reserve"):
        values = _mapping(by_kind[kind], label=f"{label} {kind} delta")
        if set(values) != topk_keys:
            raise ValueError(f"{label} {kind} delta top-K registry differs")
        for topk in EXPECTED_TOPKS:
            key = str(topk)
            expected = float(
                candidate["positive_recall_at_k_by_anchor_kind"][kind][key]
            ) - float(baseline["positive_recall_at_k_by_anchor_kind"][kind][key])
            actual = _finite_number(values[key], label=f"{label} {kind} delta R@{key}")
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=REPORTED_DELTA_ABS_TOLERANCE,
            ):
                raise ValueError(f"{label} reported {kind} R@{key} delta is not exact")


def _validate_paired_rows(baseline: Mapping, candidate: Mapping, *, label: str) -> None:
    paired_keys = (
        "replayed_rows",
        "positive_eligible_rows",
        "track_positive_eligible_rows",
        "reserve_positive_eligible_rows",
    )
    for key in paired_keys:
        if baseline["row_counts"][key] != candidate["row_counts"][key]:
            raise ValueError(f"{label} candidate and baseline {key} differ")


def _validate_direction_resources(
    direction: Mapping,
    *,
    expected_reference_descriptor_dim: int,
    expected_candidate_descriptor_dim: int,
    label: str,
) -> None:
    support = _mapping(direction.get("support"), label=f"{label} support")
    if (
        _exact_int(
            support.get("minimum_support_views"),
            label=f"{label} minimum support views",
        )
        != EXPECTED_MINIMUM_SUPPORT_VIEWS
    ):
        raise ValueError(f"{label} minimum support views differs")
    supported_anchor_count = _exact_int(
        support.get("supported_anchor_count"),
        label=f"{label} supported anchor count",
    )
    if supported_anchor_count <= 0:
        raise ValueError(f"{label} has no supported anchors")
    if (
        _exact_int(
            support.get("positive_edge_count"),
            label=f"{label} positive edge count",
        )
        <= 0
    ):
        raise ValueError(f"{label} has no positive support edges")
    memory = _mapping(
        support.get("map_descriptor_memory_float32"),
        label=f"{label} descriptor memory",
    )
    if (
        memory.get("formula") != "supported_anchor_count * descriptor_dim * 4"
        or memory.get("bytes_per_scalar") != 4
        or memory.get("frozen_superpoint_dim") != expected_reference_descriptor_dim
        or memory.get("candidate_dim") != expected_candidate_descriptor_dim
        or memory.get("frozen_superpoint_bytes")
        != supported_anchor_count * expected_reference_descriptor_dim * 4
        or memory.get("candidate_bytes")
        != supported_anchor_count * expected_candidate_descriptor_dim * 4
    ):
        raise ValueError(f"{label} descriptor memory attestation differs")
    expected_memory_ratio = (
        expected_candidate_descriptor_dim / expected_reference_descriptor_dim
    )
    memory_ratio = _finite_number(
        memory.get("candidate_to_superpoint_ratio"),
        label=f"{label} descriptor memory ratio",
    )
    if not math.isclose(
        memory_ratio,
        expected_memory_ratio,
        rel_tol=0.0,
        abs_tol=REPORTED_DELTA_ABS_TOLERANCE,
    ):
        raise ValueError(f"{label} descriptor memory ratio differs")

    resources = _mapping(
        direction.get("ranking_resources"), label=f"{label} ranking resources"
    )
    baseline = _mapping(
        resources.get("frozen_superpoint"),
        label=f"{label} baseline ranking resources",
    )
    candidate = _mapping(
        resources.get("candidate"),
        label=f"{label} candidate ranking resources",
    )
    if baseline.get("descriptor_dim") != expected_reference_descriptor_dim:
        raise ValueError(f"{label} baseline ranking dimension differs")
    if candidate.get("descriptor_dim") != expected_candidate_descriptor_dim:
        raise ValueError(f"{label} candidate ranking dimension differs")
    paired_resources = {}
    for key in ("query_rows", "score_elements"):
        baseline_value = _nonnegative_int(
            baseline.get(key), label=f"{label} baseline {key}"
        )
        candidate_value = _nonnegative_int(
            candidate.get(key), label=f"{label} candidate {key}"
        )
        if baseline_value != candidate_value:
            raise ValueError(f"{label} paired ranking {key} differs")
        paired_resources[key] = baseline_value
    if paired_resources["score_elements"] != (
        paired_resources["query_rows"] * supported_anchor_count
    ):
        raise ValueError(f"{label} ranking score-element formula differs")
    reported_macs = {}
    for name, values in (("baseline", baseline), ("candidate", candidate)):
        reported_macs[name] = _nonnegative_int(
            values.get("dot_product_multiply_accumulates"),
            label=f"{label} {name} ranking MACs",
        )
        seconds = _finite_number(
            values.get("ranking_wall_seconds"),
            label=f"{label} {name} ranking wall time",
        )
        if seconds < 0.0:
            raise ValueError(f"{label} {name} ranking wall time is negative")
    if reported_macs["baseline"] != (
        paired_resources["score_elements"] * expected_reference_descriptor_dim
    ) or reported_macs["candidate"] != (
        paired_resources["score_elements"] * expected_candidate_descriptor_dim
    ):
        raise ValueError(f"{label} ranking MAC formula differs")
    ratios = _mapping(
        resources.get("candidate_to_superpoint_ratio"),
        label=f"{label} ranking ratios",
    )
    mac_ratio = _finite_number(
        ratios.get("dot_product_multiply_accumulates"),
        label=f"{label} ranking MAC ratio",
    )
    if not math.isclose(
        mac_ratio,
        expected_memory_ratio,
        rel_tol=0.0,
        abs_tol=REPORTED_DELTA_ABS_TOLERANCE,
    ):
        raise ValueError(f"{label} ranking MAC ratio differs")
    wall_ratio = _finite_number(
        ratios.get("ranking_wall_seconds"),
        label=f"{label} ranking wall-time ratio",
    )
    baseline_seconds = float(baseline["ranking_wall_seconds"])
    candidate_seconds = float(candidate["ranking_wall_seconds"])
    expected_wall_ratio = candidate_seconds / max(baseline_seconds, 1e-12)
    if not math.isclose(
        wall_ratio,
        expected_wall_ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{label} ranking wall-time ratio differs")


def _validate_bundle(
    report: dict,
    *,
    report_path: Path,
    expected_report_sha256: str,
    source_paths: Mapping[str, Path],
    source_sha256: Mapping[str, str],
    candidate_weights_path: Path,
    candidate_weights_sha256: str,
    expected_query_count: int,
    expected_requested_keypoint_count: int,
    expected_reference_descriptor_dim: int,
    expected_candidate_descriptor_dim: int,
    expected_effective_candidate_descriptor_dim: int,
    expected_validated_descriptor_rows: int,
    expected_teacher_schema: str,
    candidate_representation: str,
    expected_evaluation_code: Mapping | None,
) -> tuple[Mapping, dict]:
    if candidate_representation not in {
        "native_candidate",
        "equal_energy_superpoint_candidate",
    }:
        raise ValueError("Unsupported candidate representation")
    if candidate_representation == "native_candidate":
        if (
            expected_effective_candidate_descriptor_dim
            != expected_candidate_descriptor_dim
        ):
            raise ValueError("Native candidate dimensions differ")
    elif expected_effective_candidate_descriptor_dim != (
        expected_reference_descriptor_dim + expected_candidate_descriptor_dim
    ):
        raise ValueError("Equal-energy descriptor dimension is not additive")
    if sha256_file(report_path) != expected_report_sha256:
        raise ValueError("Descriptor report SHA-256 differs from expected")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("version") != 1
        or report.get("mapping_only") is not True
        or report.get("uses_test_queries") is not False
        or report.get("deployment_modified") is not False
    ):
        raise ValueError("Descriptor audit bundle is not mapping-only/test-free")
    if (
        expected_evaluation_code is not None
        and report.get("evaluation_code") != expected_evaluation_code
    ):
        raise ValueError("Descriptor evaluator code identity differs")
    if "detector_repeatability" in report:
        raise ValueError("Descriptor Arm B report also contains detector Arm A")

    source_entries = _mapping(report.get("source_artifacts"), label="source_artifacts")
    if set(source_entries) != set(source_paths):
        raise ValueError("Descriptor report source-artifact registry differs")
    locked_sources = {
        name: _validate_source_entry(
            source_entries[name],
            expected_path=source_paths[name],
            expected_sha256=source_sha256[name],
            label=name,
        )
        for name in source_paths
    }
    if (
        Path(str(report.get("probe_cache", ""))).expanduser().resolve()
        != source_paths["probe_cache"]
    ):
        raise ValueError("Descriptor report probe-cache path differs")

    descriptor = _mapping(
        report.get("descriptor_identity"), label="descriptor_identity"
    )
    expected_descriptor_schema = (
        DESCRIPTOR_SCHEMA
        if candidate_representation == "native_candidate"
        else DESCRIPTOR_EQUAL_ENERGY_SCHEMA
    )
    if (
        descriptor.get("schema") != expected_descriptor_schema
        or descriptor.get("version") != 1
        or descriptor.get("mapping_only") is not True
        or descriptor.get("uses_test_queries") is not False
    ):
        raise ValueError("Descriptor result is not mapping-only/test-free")

    attestation = _mapping(
        descriptor.get("attestation"), label="descriptor attestation"
    )
    _validate_attested_artifact(
        attestation.get("artifact"),
        expected_path=candidate_weights_path,
        expected_sha256=candidate_weights_sha256,
        label="candidate weights",
    )
    reference_artifacts = _mapping(
        attestation.get("reference_artifacts"),
        label="descriptor reference artifacts",
    )
    if set(reference_artifacts) != {"query_cache", "teacher"}:
        raise ValueError("Descriptor reference-artifact registry differs")
    for name in ("query_cache", "teacher"):
        _validate_attested_artifact(
            reference_artifacts[name],
            expected_path=source_paths[name],
            expected_sha256=source_sha256[name],
            label=f"attested {name}",
        )
    exact_attestation = {
        "query_count": expected_query_count,
        "requested_keypoint_count": expected_requested_keypoint_count,
        "reference_descriptor_dim": expected_reference_descriptor_dim,
        "candidate_descriptor_dim": expected_candidate_descriptor_dim,
        "validated_descriptor_rows": expected_validated_descriptor_rows,
        "validated_detector_keypoints": 0,
    }
    for key, expected in exact_attestation.items():
        if _exact_int(attestation.get(key), label=f"attestation {key}") != expected:
            raise ValueError(f"Descriptor attestation {key} differs from expected")
    if expected_query_count <= 0 or expected_validated_descriptor_rows <= 0:
        raise ValueError("Expected query and descriptor-row counts must be positive")

    protocol = _mapping(descriptor.get("protocol"), label="descriptor protocol")
    expected_protocol = {
        "query_coordinates": "exact_frozen_superpoint_keypoint_rows",
        "positive_labels": expected_teacher_schema,
        "map_bank": "same_positive_edges_view_balanced_support_only",
        "ranking": "global_cosine",
        "crossfit": "bidirectional_temporal_block",
        "crossfit_blocks": EXPECTED_CROSSFIT_BLOCKS,
        "minimum_support_views": EXPECTED_MINIMUM_SUPPORT_VIEWS,
        "topks": list(EXPECTED_TOPKS),
        "candidate_detector_used": False,
        "descriptor_dimension_policy": (
            "native_dimensions_may_differ; rows_edges_folds_K_are_paired"
        ),
    }
    if candidate_representation == "equal_energy_superpoint_candidate":
        expected_protocol = {
            "query_coordinates": "exact_frozen_superpoint_keypoint_rows",
            "positive_labels": expected_teacher_schema,
            "map_bank": "same_positive_edges_view_balanced_support_only",
            "ranking": "single_global_cosine",
            "crossfit": "bidirectional_temporal_block",
            "crossfit_blocks": EXPECTED_CROSSFIT_BLOCKS,
            "minimum_support_views": EXPECTED_MINIMUM_SUPPORT_VIEWS,
            "topks": list(EXPECTED_TOPKS),
            "candidate_detector_used": False,
            "candidate_representation": (
                "l2_concat(l2(superpoint),l2(candidate))/sqrt(2)"
            ),
            "score_identity": ("0.5*cosine_superpoint+0.5*cosine_candidate"),
            "source_candidate_descriptor_dim": (expected_candidate_descriptor_dim),
            "effective_candidate_descriptor_dim": (
                expected_effective_candidate_descriptor_dim
            ),
            "learned_fusion_parameters": False,
            "source_specific_descriptor_routing": False,
        }
    if dict(protocol) != expected_protocol:
        raise ValueError("Descriptor protocol differs from preregistration")

    split = _mapping(descriptor.get("split"), label="descriptor split")
    if (
        split.get("policy") != "per_sequence_alternating_contiguous_temporal_blocks"
        or split.get("block_count") != EXPECTED_CROSSFIT_BLOCKS
        or split.get("uses_test_queries") is not False
    ):
        raise ValueError("Descriptor split differs from preregistration")
    selection_count = _exact_int(
        split.get("selection_query_count"), label="selection query count"
    )
    gate_count = _exact_int(split.get("gate_query_count"), label="gate query count")
    if selection_count <= 0 or gate_count <= 0:
        raise ValueError("Descriptor split contains an empty direction")
    if selection_count + gate_count != expected_query_count:
        raise ValueError("Descriptor split query counts do not cover the audit")
    assignments = _mapping(split.get("assignments"), label="split assignments")
    if len(assignments) != expected_query_count:
        raise ValueError("Descriptor split assignment count differs")
    assignment_counts = [0, 0]
    for name, value in assignments.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Descriptor split has an invalid query name")
        block = _exact_int(value, label=f"split block for {name}")
        if not 0 <= block < EXPECTED_CROSSFIT_BLOCKS:
            raise ValueError("Descriptor split block lies outside preregistration")
        assignment_counts[block % 2] += 1
    if assignment_counts != [selection_count, gate_count]:
        raise ValueError("Descriptor split assignments disagree with counts")

    directions = descriptor.get("directions")
    if not isinstance(directions, list) or len(directions) != 2:
        raise ValueError("Descriptor report is not bidirectional")
    if tuple(row.get("direction") for row in directions) != EXPECTED_DIRECTION_NAMES:
        raise ValueError("Descriptor direction registry differs")
    expected_direction_counts = (
        (selection_count, gate_count),
        (gate_count, selection_count),
    )
    direction_summaries = []
    for raw_direction, (support_count, heldout_count) in zip(
        directions, expected_direction_counts
    ):
        direction = _mapping(raw_direction, label="descriptor direction")
        name = str(direction["direction"])
        support = _mapping(direction.get("support"), label=f"{name} support")
        if support.get("support_query_count") != support_count:
            raise ValueError(f"{name} support query count differs")
        if direction.get("heldout_query_count") != heldout_count:
            raise ValueError(f"{name} held-out query count differs")
        baseline = _validate_recall_summary(
            direction.get("frozen_superpoint"), label=f"{name} frozen SuperPoint"
        )
        candidate = _validate_recall_summary(
            direction.get("candidate"), label=f"{name} candidate"
        )
        _validate_paired_rows(baseline, candidate, label=name)
        _validate_reported_delta(
            direction.get("delta_candidate_minus_superpoint"),
            baseline=baseline,
            candidate=candidate,
            label=name,
        )
        _validate_direction_resources(
            direction,
            expected_reference_descriptor_dim=expected_reference_descriptor_dim,
            expected_candidate_descriptor_dim=(
                expected_effective_candidate_descriptor_dim
            ),
            label=name,
        )
        if (
            direction["ranking_resources"]["frozen_superpoint"]["query_rows"]
            != baseline["row_counts"]["replayed_rows"]
            or direction["ranking_resources"]["candidate"]["query_rows"]
            != candidate["row_counts"]["replayed_rows"]
        ):
            raise ValueError(f"{name} ranking/query row attestation differs")
        direction_summaries.append((baseline, candidate))

    pooled = _mapping(descriptor.get("pooled"), label="pooled descriptor metrics")
    if set(pooled) != {"frozen_superpoint", "candidate"}:
        raise ValueError("Pooled descriptor-arm registry differs")
    pooled_baseline = _validate_recall_summary(
        pooled["frozen_superpoint"], label="pooled frozen SuperPoint"
    )
    pooled_candidate = _validate_recall_summary(
        pooled["candidate"], label="pooled candidate"
    )
    _validate_paired_rows(pooled_baseline, pooled_candidate, label="pooled")
    for arm, pooled_summary, index in (
        ("frozen_superpoint", pooled_baseline, 0),
        ("candidate", pooled_candidate, 1),
    ):
        for key, pooled_value in pooled_summary["row_counts"].items():
            directional_total = sum(
                int(summary[index]["row_counts"][key])
                for summary in direction_summaries
            )
            if pooled_value != directional_total:
                raise ValueError(f"Pooled {arm} {key} is not direction-additive")
    if (
        pooled_baseline["row_counts"]["positive_eligible_rows"] <= 0
        or pooled_baseline["row_counts"]["track_positive_eligible_rows"] <= 0
        or pooled_baseline["row_counts"]["reserve_positive_eligible_rows"] <= 0
    ):
        raise ValueError("Pooled Arm-B gate lacks one preregistered metric domain")
    _validate_reported_delta(
        descriptor.get("delta_candidate_minus_superpoint"),
        baseline=pooled_baseline,
        candidate=pooled_candidate,
        label="pooled",
    )
    return descriptor, locked_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--expected-teacher-sha256", required=True)
    parser.add_argument("--probe-cache", type=Path, required=True)
    parser.add_argument("--expected-probe-cache-sha256", required=True)
    parser.add_argument("--candidate-weights", type=Path, required=True)
    parser.add_argument("--expected-candidate-weights-sha256", required=True)
    parser.add_argument("--expected-query-count", type=int, required=True)
    parser.add_argument("--expected-requested-keypoint-count", type=int, required=True)
    parser.add_argument("--expected-reference-descriptor-dim", type=int, required=True)
    parser.add_argument("--expected-candidate-descriptor-dim", type=int, required=True)
    parser.add_argument(
        "--candidate-representation",
        choices=("native_candidate", "equal_energy_superpoint_candidate"),
        default="native_candidate",
    )
    parser.add_argument("--expected-effective-candidate-descriptor-dim", type=int)
    parser.add_argument("--expected-validated-descriptor-rows", type=int, required=True)
    parser.add_argument("--expected-teacher-schema", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report_path = _resolved_file(args.report, label="descriptor report")
    report_sha256 = _sha256(
        args.expected_report_sha256, label="Expected descriptor report SHA-256"
    )
    source_paths = {
        "state": _resolved_file(args.state, label="state"),
        "query_cache": _resolved_file(args.query_cache, label="query cache"),
        "teacher": _resolved_file(args.teacher, label="teacher"),
        "probe_cache": _resolved_file(args.probe_cache, label="probe cache"),
    }
    source_sha256 = {
        "state": _sha256(args.expected_state_sha256, label="Expected state SHA-256"),
        "query_cache": _sha256(
            args.expected_query_cache_sha256,
            label="Expected query-cache SHA-256",
        ),
        "teacher": _sha256(
            args.expected_teacher_sha256, label="Expected teacher SHA-256"
        ),
        "probe_cache": _sha256(
            args.expected_probe_cache_sha256,
            label="Expected probe-cache SHA-256",
        ),
    }
    candidate_weights_path = _resolved_file(
        args.candidate_weights, label="candidate weights"
    )
    candidate_weights_sha256 = _sha256(
        args.expected_candidate_weights_sha256,
        label="Expected candidate-weights SHA-256",
    )
    output_path = args.output.expanduser().resolve()
    protected_inputs = {report_path, candidate_weights_path, *source_paths.values()}
    if output_path in protected_inputs:
        raise ValueError("Gate output must not overwrite a source artifact")

    report = _read_json(report_path)
    expected_candidate_descriptor_dim = int(args.expected_candidate_descriptor_dim)
    expected_effective_candidate_descriptor_dim = int(
        args.expected_effective_candidate_descriptor_dim
        if args.expected_effective_candidate_descriptor_dim is not None
        else expected_candidate_descriptor_dim
    )
    equal_energy = args.candidate_representation == "equal_energy_superpoint_candidate"
    evaluation_code = (
        frontend_descriptor_evaluation_code_identity(require_clean=True)
        if equal_energy
        else None
    )
    descriptor, locked_sources = _validate_bundle(
        report,
        report_path=report_path,
        expected_report_sha256=report_sha256,
        source_paths=source_paths,
        source_sha256=source_sha256,
        candidate_weights_path=candidate_weights_path,
        candidate_weights_sha256=candidate_weights_sha256,
        expected_query_count=int(args.expected_query_count),
        expected_requested_keypoint_count=int(args.expected_requested_keypoint_count),
        expected_reference_descriptor_dim=int(args.expected_reference_descriptor_dim),
        expected_candidate_descriptor_dim=expected_candidate_descriptor_dim,
        expected_effective_candidate_descriptor_dim=(
            expected_effective_candidate_descriptor_dim
        ),
        expected_validated_descriptor_rows=int(args.expected_validated_descriptor_rows),
        expected_teacher_schema=str(args.expected_teacher_schema),
        candidate_representation=str(args.candidate_representation),
        expected_evaluation_code=evaluation_code,
    )

    direction_comparisons = {}
    gates = {}
    for direction in descriptor["directions"]:
        name = str(direction["direction"])
        baseline = float(direction["frozen_superpoint"]["positive_recall_at_k"]["1"])
        candidate = float(direction["candidate"]["positive_recall_at_k"]["1"])
        delta = candidate - baseline
        direction_comparisons[name] = {
            "frozen_superpoint": baseline,
            "candidate": candidate,
            "delta_candidate_minus_superpoint": delta,
        }
        gates[f"{name}_candidate_r1_strictly_positive"] = delta > 0.0

    pooled_baseline = descriptor["pooled"]["frozen_superpoint"]
    pooled_candidate = descriptor["pooled"]["candidate"]
    pooled_r8_delta = float(pooled_candidate["positive_recall_at_k"]["8"]) - float(
        pooled_baseline["positive_recall_at_k"]["8"]
    )
    pooled_kind_r1 = {}
    for kind in ("track_core", "gaussian_reserve"):
        baseline = float(
            pooled_baseline["positive_recall_at_k_by_anchor_kind"][kind]["1"]
        )
        candidate = float(
            pooled_candidate["positive_recall_at_k_by_anchor_kind"][kind]["1"]
        )
        delta = candidate - baseline
        pooled_kind_r1[kind] = {
            "frozen_superpoint": baseline,
            "candidate": candidate,
            "delta_candidate_minus_superpoint": delta,
        }
        gates[f"pooled_{kind}_r1_non_regression"] = (
            delta >= -NON_REGRESSION_ABS_TOLERANCE
        )
    gates["pooled_r8_non_regression"] = pooled_r8_delta >= -NON_REGRESSION_ABS_TOLERANCE
    passed = all(gates.values())
    gate_report = {
        "schema": GATE_EQUAL_ENERGY_SCHEMA if equal_energy else GATE_SCHEMA,
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "single_factor": (
            "equal_energy_single_descriptor_at_exact_superpoint_rows"
            if equal_energy
            else "descriptor_identity_at_exact_superpoint_rows"
        ),
        "valid": True,
        "protocol": {
            "crossfit": "bidirectional_temporal_block",
            "crossfit_blocks": EXPECTED_CROSSFIT_BLOCKS,
            "minimum_support_views": EXPECTED_MINIMUM_SUPPORT_VIEWS,
            "topks": list(EXPECTED_TOPKS),
            "strict_positive_r1_delta_threshold": 0.0,
            "non_regression_absolute_tolerance": NON_REGRESSION_ABS_TOLERANCE,
            "reported_delta_validation_absolute_tolerance": (
                REPORTED_DELTA_ABS_TOLERANCE
            ),
            "candidate_representation": str(args.candidate_representation),
            "source_candidate_descriptor_dim": (expected_candidate_descriptor_dim),
            "effective_candidate_descriptor_dim": (
                expected_effective_candidate_descriptor_dim
            ),
        },
        "comparisons": {
            "direction_r1": direction_comparisons,
            "pooled_r8": {
                "frozen_superpoint": float(
                    pooled_baseline["positive_recall_at_k"]["8"]
                ),
                "candidate": float(pooled_candidate["positive_recall_at_k"]["8"]),
                "delta_candidate_minus_superpoint": pooled_r8_delta,
            },
            "pooled_r1_by_anchor_kind": pooled_kind_r1,
        },
        "gates": gates,
        "mechanism_gate_passed": passed,
        "advance_to_mapping_only_descriptor_rebuild": passed,
        "decision": "GO" if passed else "STOP",
        "limitations": [
            "GO authorizes only a mapping-only descriptor rebuild and later pose gate; it is not a pose or test-set claim.",
            "STOP rejects this locked candidate under the fixed identity problem; it does not prove every frontend is at the same ceiling.",
        ],
        "inputs": {
            "descriptor_report": {
                "path": str(report_path),
                "sha256": report_sha256,
            },
            "source_artifacts": locked_sources,
            "candidate_weights": {
                "path": str(candidate_weights_path),
                "sha256": candidate_weights_sha256,
            },
            **(
                {"evaluation_code": evaluation_code}
                if evaluation_code is not None
                else {}
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(gate_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gate_report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
