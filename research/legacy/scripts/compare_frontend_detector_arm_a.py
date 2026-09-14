#!/usr/bin/env python3
"""Fail-closed mechanism gate for mapping-only XFeat detector Arm A."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path

import torch

from common.evaluation_code import frontend_detector_evaluation_code_identity
from common.hashing import sha256_file
from map_learning.frontend_upper_bound import tensor_sha256


REPORT_SCHEMA = "lafgs_frontend_ceiling_probe_audit_bundle"
DETECTOR_SCHEMA = "lafgs_mapping_detector_repeatability_ceiling_probe"
PROBE_SCHEMA = "lafgs_frontend_ceiling_probe_cache"
PRODUCER_SCHEMA = "lafgs_xfeat_arm_a_producer"
GATE_SCHEMA = "lafgs_frontend_detector_arm_a_mechanism_gate"
EXPECTED_RADII = (2.0, 4.0, 8.0)
EXPECTED_RADIUS_KEYS = tuple(str(value) for value in EXPECTED_RADII)
EXPECTED_QUERY_COUNT = 2000
EXPECTED_REQUESTED_KEYPOINT_COUNT = 1024
EXPECTED_REFERENCE_NMS_RADIUS = 4
EXPECTED_CANDIDATE_NMS_KERNEL = 5
EXPECTED_CANDIDATE_NMS_RADIUS = 2
EXPECTED_DETECTION_THRESHOLD = 0.05
EXPECTED_REFERENCE_DESCRIPTOR_DIM = 256
EXPECTED_CANDIDATE_DESCRIPTOR_DIM = 64
EXPECTED_TEACHER_SCHEMA = "lafgs_v9_active_map_complete_positive_teacher"
EXPECTED_DEPTH_ABS_TOLERANCE_M = 0.05
EXPECTED_DEPTH_REL_TOLERANCE = 0.02
EXPECTED_ALPHA_MINIMUM = 0.01
NON_REGRESSION_ABS_TOLERANCE = 1e-12
REPORTED_VALUE_ABS_TOLERANCE = 1e-15

EXPECTED_XFEAT_WEIGHTS_SHA256 = (
    "0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b"
)
EXPECTED_XFEAT_TREE = "4f804566cb1cf72469b7d7174fba9308885c5c5a"
EXPECTED_XFEAT_MODEL_SHA256 = (
    "d9a665f18fcea5eaf3e278925e1a92103afcba9051e05b2334f3daa29f411964"
)
EXPECTED_XFEAT_INTERPOLATOR_SHA256 = (
    "d63a6163eb6fff81e8720231f62537a42a69fccb44dc8851b04de5115daab4da"
)
EXPECTED_XFEAT_WRAPPER_SHA256 = (
    "f1b0f73c77e34381a46578866bb1531b98180e8d870c0fc61fbfdbd29ac64f31"
)
EXPECTED_IMPLEMENTATION_ID = (
    "xfeat_tree_4f804566cb1c__model_d9a665f18fce__arm_a_v1"
)
EXPECTED_PRODUCER_IMPLEMENTATION_FILES = (
    "map_learning/xfeat_arm_a.py",
    "map_learning/xfeat_arm_b.py",
    "map_learning/frontend_upper_bound.py",
    "data/datasets.py",
    "data/images.py",
    "features/multiview_fusion.py",
)


def _mapping(value: object, *, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def _nonnegative_int(value: object, *, label: str) -> int:
    result = _exact_int(value, label=label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _sha256(value: object, *, label: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be 64 hexadecimal digits")
    return result


def _resolved_file(path: str | Path, *, label: str) -> Path:
    text = str(path)
    if "://" in text:
        raise ValueError(f"{label} must be a local file")
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise ValueError(f"{label} is not a file: {result}")
    return result


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _validate_local_entry(
    entry: object,
    *,
    expected_path: Path,
    expected_sha256: str,
    label: str,
    status: str | None = None,
) -> dict:
    payload = _mapping(entry, label=f"{label} entry")
    path_text = str(payload.get("path", ""))
    if not path_text or "://" in path_text:
        raise ValueError(f"{label} path is not local")
    if Path(path_text).expanduser().resolve() != expected_path:
        raise ValueError(f"{label} path differs from expected")
    if payload.get("sha256") != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs from expected")
    if status not in {None, "metadata"} and payload.get("status") != status:
        raise ValueError(f"{label} inspection status differs")
    if status in {"present_unverified", "metadata"}:
        if payload.get("expected_sha256") is not None:
            if status == "present_unverified":
                raise ValueError(f"{label} unexpectedly rewrites expected SHA-256")
        if _nonnegative_int(payload.get("size_bytes"), label=f"{label} size") != (
            expected_path.stat().st_size
        ):
            raise ValueError(f"{label} size differs from current artifact")
    if status is None and payload.get("verified") is not True:
        raise ValueError(f"{label} is not verified")
    if sha256_file(expected_path) != expected_sha256:
        raise ValueError(f"{label} current artifact SHA-256 differs from expected")
    return {"path": str(expected_path), "sha256": expected_sha256}


def _validate_current_code_entry(entry: object, *, relative: str, root: Path) -> None:
    payload = _mapping(entry, label=f"producer code {relative}")
    expected_path = (root / relative).resolve()
    if Path(str(payload.get("path", ""))).expanduser().resolve() != expected_path:
        raise ValueError(f"producer code path differs for {relative}")
    current_sha256 = sha256_file(expected_path)
    if payload.get("sha256") != current_sha256:
        raise ValueError(f"producer code SHA-256 differs for {relative}")


def _validate_xfeat_code_entry(
    entry: object, *, expected_sha256: str, label: str
) -> None:
    payload = _mapping(entry, label=f"XFeat {label}")
    path = _resolved_file(str(payload.get("path", "")), label=f"XFeat {label}")
    if payload.get("sha256") != expected_sha256:
        raise ValueError(f"XFeat {label} preregistered SHA-256 differs")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"XFeat {label} current SHA-256 differs")


def _expected_candidate_protocol() -> dict:
    return {
        "name": "xfeat_single_image_sparse",
        "requested_keypoint_count": EXPECTED_REQUESTED_KEYPOINT_COUNT,
        "keypoint_heatmap": "softmax_65_discard_dustbin_then_8x8_unpack",
        "nms_kernel_size": EXPECTED_CANDIDATE_NMS_KERNEL,
        "nms_radius": EXPECTED_CANDIDATE_NMS_RADIUS,
        "nms_passes": 1,
        "strict_probability_threshold": EXPECTED_DETECTION_THRESHOLD,
        "score_semantics": "nearest_probability_times_bilinear_reliability",
        "origin_padding_sentinel_excluded": True,
        "sort": "descending_score_stable_row_major_ties",
        "top_k_before_native_mask": True,
        "native_mask_filter": "sample_mask_at_grid_uv_nearest_round",
    }


def _query_names_sha256(names: Sequence[str]) -> str:
    encoded = "".join(f"{len(name)}:{name}\n" for name in names).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_probe(
    probe: object,
    *,
    source_paths: Mapping[str, Path],
    source_sha256: Mapping[str, str],
    candidate_weights_path: Path,
    candidate_weights_sha256: str,
    expected_query_count: int,
    expected_validated_detector_keypoints: int,
) -> tuple[tuple[str, ...], dict]:
    payload = _mapping(probe, label="probe cache")
    if set(payload) != {
        "schema",
        "version",
        "mapping_only",
        "uses_test_queries",
        "reference",
        "frontend",
        "capabilities",
        "producer",
        "queries",
    }:
        raise ValueError("Arm-A probe top-level registry differs")
    if (
        payload.get("schema") != PROBE_SCHEMA
        or payload.get("version") != 1
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
    ):
        raise ValueError("Arm-A probe is not mapping-only/test-free")
    if dict(_mapping(payload.get("capabilities"), label="probe capabilities")) != {
        "detector_repeatability": True,
        "descriptor_identity": False,
    }:
        raise ValueError("Arm-A probe capabilities are not detector-only")

    reference = _mapping(payload.get("reference"), label="probe reference")
    for name in ("query_cache", "teacher"):
        _validate_local_entry(
            {
                "path": reference.get(f"{name}_path"),
                "sha256": reference.get(f"{name}_sha256"),
                "verified": True,
            },
            expected_path=source_paths[name],
            expected_sha256=source_sha256[name],
            label=f"probe {name}",
        )
    raw_reference_names = reference.get("query_names")
    if not isinstance(raw_reference_names, list) or not all(
        isinstance(name, str) and name for name in raw_reference_names
    ):
        raise ValueError("probe reference query-name registry is invalid")
    reference_names = tuple(raw_reference_names)
    if (
        len(reference_names) != expected_query_count
        or len(set(reference_names)) != expected_query_count
        or reference.get("query_names_sha256") != _query_names_sha256(reference_names)
        or reference.get("teacher_schema") != EXPECTED_TEACHER_SCHEMA
    ):
        raise ValueError("probe reference query names/teacher schema differ")
    reference_protocol = _mapping(
        reference.get("reference_detector_protocol"),
        label="reference detector protocol",
    )
    if dict(reference_protocol) != {
        "name": "frozen_superpoint",
        "requested_keypoint_count": EXPECTED_REQUESTED_KEYPOINT_COUNT,
        "nms_radius": EXPECTED_REFERENCE_NMS_RADIUS,
        "mask_filter": "sample_mask_at_grid_uv_nearest_round",
        "top_k_before_native_mask": True,
    }:
        raise ValueError("reference K/NMS protocol differs from preregistration")

    frontend = _mapping(payload.get("frontend"), label="probe frontend")
    if (
        frontend.get("name") != "xfeat_sparse_64d_detector_only"
        or frontend.get("family") != "independent_local_frontend"
        or frontend.get("implementation_id") != EXPECTED_IMPLEMENTATION_ID
        or frontend.get("coordinate_convention")
        != "reference_grid_index_then_cached_pixel_center_offset"
        or frontend.get("descriptor_dim") != EXPECTED_CANDIDATE_DESCRIPTOR_DIM
        or frontend.get("requested_keypoint_count")
        != EXPECTED_REQUESTED_KEYPOINT_COUNT
    ):
        raise ValueError("candidate frontend identity/K differs from preregistration")
    _validate_local_entry(
        frontend.get("weights"),
        expected_path=candidate_weights_path,
        expected_sha256=candidate_weights_sha256,
        label="probe candidate weights",
        status="metadata",
    )
    code = _mapping(frontend.get("code"), label="probe frontend code")
    if code.get("xfeat_tree") != EXPECTED_XFEAT_TREE or code.get("git_clean") is not True:
        raise ValueError("XFeat tree/clean attestation differs")
    for name, expected in (
        ("model", EXPECTED_XFEAT_MODEL_SHA256),
        ("interpolator", EXPECTED_XFEAT_INTERPOLATOR_SHA256),
        ("wrapper", EXPECTED_XFEAT_WRAPPER_SHA256),
    ):
        _validate_xfeat_code_entry(code.get(name), expected_sha256=expected, label=name)

    producer = _mapping(payload.get("producer"), label="probe producer")
    exact_producer = {
        "schema": PRODUCER_SCHEMA,
        "version": 1,
        "arm": "A_detector_repeatability",
        "device": "cpu",
        "dtype": "float32",
        "gpu_used": False,
        "network_access_used": False,
        "candidate_detector_used": True,
        "shared_forward_descriptor_output_used": False,
        "candidate_descriptor_rows_materialized": False,
        "pair_matcher_used": False,
        "query_count": expected_query_count,
        "all_queries_identity_xfeat_resize": True,
        "all_queries_round_floor_mask_equivalent": True,
    }
    for key, expected in exact_producer.items():
        if producer.get(key) != expected:
            raise ValueError(f"Arm-A producer field {key} differs")
    if dict(_mapping(producer.get("detector_protocol"), label="producer protocol")) != (
        _expected_candidate_protocol()
    ):
        raise ValueError("candidate detector protocol differs from preregistration")
    root = Path(__file__).resolve().parents[1]
    implementation_files = _mapping(
        producer.get("implementation_files"), label="producer implementation files"
    )
    if set(implementation_files) != set(EXPECTED_PRODUCER_IMPLEMENTATION_FILES):
        raise ValueError("producer implementation-file registry differs")
    for relative in EXPECTED_PRODUCER_IMPLEMENTATION_FILES:
        _validate_current_code_entry(implementation_files[relative], relative=relative, root=root)
    cli = _mapping(producer.get("cli"), label="producer CLI")
    expected_cli = (root / "scripts/materialize_xfeat_arm_a.py").resolve()
    if (
        Path(str(cli.get("path", ""))).expanduser().resolve() != expected_cli
        or cli.get("sha256") != sha256_file(expected_cli)
    ):
        raise ValueError("producer CLI code identity differs")

    consumer = _mapping(
        producer.get("consumer_validation"), label="producer consumer validation"
    )
    exact_consumer = {
        "query_count": expected_query_count,
        "requested_keypoint_count": EXPECTED_REQUESTED_KEYPOINT_COUNT,
        "reference_descriptor_dim": EXPECTED_REFERENCE_DESCRIPTOR_DIM,
        "candidate_descriptor_dim": None,
        "validated_descriptor_rows": 0,
        "validated_detector_keypoints": expected_validated_detector_keypoints,
    }
    for key, expected in exact_consumer.items():
        if consumer.get(key) != expected:
            raise ValueError(f"producer consumer attestation {key} differs")

    queries = _mapping(payload.get("queries"), label="probe queries")
    if len(queries) != expected_query_count:
        raise ValueError("probe query count differs from preregistration")
    if tuple(queries) != reference_names:
        raise ValueError("probe query order differs from reference registry")
    total_before = 0
    total_after = 0
    for name, raw_record in queries.items():
        if not isinstance(name, str) or not name:
            raise ValueError("probe contains an invalid query name")
        record = _mapping(raw_record, label=f"probe query {name}")
        if "descriptor_at_reference_keypoints" in record:
            raise ValueError("detector probe contains descriptor rows")
        _sha256(
            record.get("reference_keypoints_sha256"),
            label=f"{name} reference-keypoint SHA-256",
        )
        keypoints = torch.as_tensor(record.get("detector_keypoints")).detach().cpu().float()
        scores = torch.as_tensor(record.get("detector_scores")).detach().cpu().float().reshape(-1)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2 or scores.numel() != keypoints.shape[0]:
            raise ValueError(f"probe detector row shape differs for {name}")
        if not bool(torch.isfinite(keypoints).all()) or not bool(torch.isfinite(scores).all()):
            raise ValueError(f"probe detector rows are non-finite for {name}")
        if scores.numel() > 1 and not bool((scores[:-1] >= scores[1:]).all()):
            raise ValueError(f"probe detector scores are not ordered for {name}")
        before = _nonnegative_int(
            record.get("detected_count_before_mask"), label=f"{name} pre-mask count"
        )
        after = _nonnegative_int(
            record.get("detected_count_after_mask"), label=f"{name} post-mask count"
        )
        if after != keypoints.shape[0] or not after <= before <= EXPECTED_REQUESTED_KEYPOINT_COUNT:
            raise ValueError(f"probe detector K/count contract differs for {name}")
        if record.get("detector_keypoints_sha256") != tensor_sha256(keypoints):
            raise ValueError(f"probe keypoint hash differs for {name}")
        if record.get("detector_scores_sha256") != tensor_sha256(scores):
            raise ValueError(f"probe score hash differs for {name}")
        lineage = _mapping(record.get("detector_lineage"), label=f"{name} lineage")
        expected_lineage = {
            key: expected
            for key, expected in _expected_candidate_protocol().items()
            if key not in {"name", "requested_keypoint_count", "top_k_before_native_mask"}
        }
        expected_lineage["top_k_before_native_mask"] = (
            EXPECTED_REQUESTED_KEYPOINT_COUNT
        )
        for key, expected in expected_lineage.items():
            if lineage.get(key) != expected:
                raise ValueError(f"probe detector lineage {key} differs for {name}")
        if (
            lineage.get("positive_top_k_count_before_mask") != before
            or lineage.get("post_mask_count") != after
            or _nonnegative_int(
                lineage.get("candidate_count_after_threshold_nms"),
                label=f"{name} post-NMS count",
            )
            < before
        ):
            raise ValueError(f"probe detector lineage counts differ for {name}")
        if (
            lineage.get("shared_forward_descriptor_output_used") is not False
            or lineage.get("candidate_descriptor_rows_materialized") is not False
            or lineage.get("pair_matcher_used") is not False
        ):
            raise ValueError(f"probe detector lineage is not detector-only for {name}")
        proof = _mapping(
            lineage.get("mask_equivalence_proof"), label=f"{name} mask proof"
        )
        for key in (
            "required_native_hw_divisible_by_32",
            "native_hw_divisible_by_32",
            "identity_xfeat_resize",
            "integer_xfeat_coordinates",
            "round_floor_indices_equal",
            "round_floor_mask_decisions_equal",
        ):
            if proof.get(key) is not True:
                raise ValueError(f"probe mask proof {key} differs for {name}")
        if proof.get("checked_pre_mask_rows") != before:
            raise ValueError(f"probe mask proof row count differs for {name}")
        round_indices_sha256 = _sha256(
            proof.get("round_indices_sha256"), label=f"{name} round-index SHA-256"
        )
        floor_indices_sha256 = _sha256(
            proof.get("floor_indices_sha256"), label=f"{name} floor-index SHA-256"
        )
        if round_indices_sha256 != floor_indices_sha256:
            raise ValueError(f"probe round/floor index hashes differ for {name}")
        round_mask_sha256 = _sha256(
            proof.get("round_mask_keep_sha256"), label=f"{name} round-mask SHA-256"
        )
        floor_mask_sha256 = _sha256(
            proof.get("floor_mask_keep_sha256"), label=f"{name} floor-mask SHA-256"
        )
        if round_mask_sha256 != floor_mask_sha256:
            raise ValueError(f"probe round/floor mask hashes differ for {name}")
        total_before += before
        total_after += after
    if producer.get("detected_count_before_mask") != total_before:
        raise ValueError("producer aggregate pre-mask count differs")
    if producer.get("detected_count_after_mask") != total_after:
        raise ValueError("producer aggregate post-mask count differs")
    if total_after != expected_validated_detector_keypoints:
        raise ValueError("probe validated detector count differs")
    return tuple(queries), {
        "query_count": len(queries),
        "detected_count_before_mask": total_before,
        "detected_count_after_mask": total_after,
    }


def _validate_reachability(
    value: object, *, label: str, include_query_count: bool
) -> dict:
    payload = _mapping(value, label=label)
    expected_keys = {"by_anchor_kind"}
    if include_query_count:
        expected_keys.add("query_count")
    else:
        expected_keys.update({"target_count", "hit_count"})
    if set(payload) != expected_keys:
        raise ValueError(f"{label} registry differs")
    kinds = _mapping(payload.get("by_anchor_kind"), label=f"{label} anchor kinds")
    if set(kinds) != {"all", "track_core", "gaussian_reserve"}:
        raise ValueError(f"{label} anchor-kind registry differs")
    normalized = {"by_anchor_kind": {}}
    for kind in ("all", "track_core", "gaussian_reserve"):
        row = _mapping(kinds[kind], label=f"{label} {kind}")
        expected_row_keys = {"target_count", "hit_count"}
        if include_query_count:
            expected_row_keys.add("reachable_fraction")
        if set(row) != expected_row_keys:
            raise ValueError(f"{label} {kind} registry differs")
        target = _nonnegative_int(row.get("target_count"), label=f"{label} {kind} target")
        hits = _mapping(row.get("hit_count"), label=f"{label} {kind} hits")
        if tuple(hits) != EXPECTED_RADIUS_KEYS:
            raise ValueError(f"{label} {kind} radius registry differs")
        hit_values = {
            key: _nonnegative_int(hits[key], label=f"{label} {kind}@{key} hits")
            for key in EXPECTED_RADIUS_KEYS
        }
        previous = -1
        for key in EXPECTED_RADIUS_KEYS:
            if hit_values[key] < previous or hit_values[key] > target:
                raise ValueError(f"{label} {kind} hit counts are invalid")
            previous = hit_values[key]
        normalized["by_anchor_kind"][kind] = {
            "target_count": target,
            "hit_count": hit_values,
        }
        if include_query_count:
            fractions = _mapping(
                row.get("reachable_fraction"), label=f"{label} {kind} fractions"
            )
            if tuple(fractions) != EXPECTED_RADIUS_KEYS:
                raise ValueError(f"{label} {kind} fraction registry differs")
            for key in EXPECTED_RADIUS_KEYS:
                actual = _finite_number(
                    fractions[key], label=f"{label} {kind}@{key} fraction"
                )
                expected = hit_values[key] / max(target, 1)
                if not math.isclose(
                    actual, expected, rel_tol=0.0, abs_tol=REPORTED_VALUE_ABS_TOLERANCE
                ):
                    raise ValueError(f"{label} {kind}@{key} fraction is not count-derived")
    all_row = normalized["by_anchor_kind"]["all"]
    if not include_query_count:
        if payload.get("target_count") != all_row["target_count"]:
            raise ValueError(f"{label} duplicated target count differs")
        if dict(_mapping(payload.get("hit_count"), label=f"{label} duplicated hits")) != (
            all_row["hit_count"]
        ):
            raise ValueError(f"{label} duplicated hit counts differ")
    else:
        normalized["query_count"] = _nonnegative_int(
            payload.get("query_count"), label=f"{label} query count"
        )
    if (
        normalized["by_anchor_kind"]["track_core"]["target_count"]
        + normalized["by_anchor_kind"]["gaussian_reserve"]["target_count"]
        != all_row["target_count"]
    ):
        raise ValueError(f"{label} anchor-kind targets are not exhaustive")
    for key in EXPECTED_RADIUS_KEYS:
        if (
            normalized["by_anchor_kind"]["track_core"]["hit_count"][key]
            + normalized["by_anchor_kind"]["gaussian_reserve"]["hit_count"][key]
            != all_row["hit_count"][key]
        ):
            raise ValueError(f"{label} anchor-kind hits are not exhaustive")
    return normalized


def _validate_report(
    report: object,
    *,
    report_path: Path,
    expected_report_sha256: str,
    source_paths: Mapping[str, Path],
    source_sha256: Mapping[str, str],
    candidate_weights_path: Path,
    candidate_weights_sha256: str,
    expected_query_count: int,
    expected_validated_detector_keypoints: int,
    expected_target_counts: Mapping[str, int],
    expected_evaluation_code: Mapping,
) -> tuple[Mapping, dict, tuple[str, ...]]:
    if sha256_file(report_path) != expected_report_sha256:
        raise ValueError("detector report SHA-256 differs from expected")
    payload = _mapping(report, label="detector report")
    if set(payload) != {
        "schema",
        "version",
        "mapping_only",
        "uses_test_queries",
        "deployment_modified",
        "evaluation_code",
        "probe_cache",
        "source_artifacts",
        "detector_repeatability",
    }:
        raise ValueError("detector report top-level registry differs or mixes another arm")
    if (
        payload.get("schema") != REPORT_SCHEMA
        or payload.get("version") != 1
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("deployment_modified") is not False
    ):
        raise ValueError("detector report is not mapping-only/test-free")
    if payload.get("evaluation_code") != expected_evaluation_code:
        raise ValueError("detector evaluator clean-code identity differs")
    if Path(str(payload.get("probe_cache", ""))).expanduser().resolve() != source_paths[
        "probe_cache"
    ]:
        raise ValueError("detector report probe-cache path differs")
    entries = _mapping(payload.get("source_artifacts"), label="report sources")
    if set(entries) != set(source_paths):
        raise ValueError("detector report source registry differs")
    locked_sources = {
        name: _validate_local_entry(
            entries[name],
            expected_path=source_paths[name],
            expected_sha256=source_sha256[name],
            label=f"report {name}",
            status="present_unverified",
        )
        for name in source_paths
    }

    detector = _mapping(
        payload.get("detector_repeatability"), label="detector result"
    )
    if (
        detector.get("schema") != DETECTOR_SCHEMA
        or detector.get("version") != 1
        or detector.get("mapping_only") is not True
        or detector.get("uses_test_queries") is not False
    ):
        raise ValueError("detector result is not mapping-only/test-free")
    attestation = _mapping(detector.get("attestation"), label="detector attestation")
    _validate_local_entry(
        attestation.get("artifact"),
        expected_path=candidate_weights_path,
        expected_sha256=candidate_weights_sha256,
        label="attested candidate weights",
    )
    reference_artifacts = _mapping(
        attestation.get("reference_artifacts"), label="attested references"
    )
    if set(reference_artifacts) != {"query_cache", "teacher"}:
        raise ValueError("detector reference-artifact registry differs")
    for name in ("query_cache", "teacher"):
        _validate_local_entry(
            reference_artifacts[name],
            expected_path=source_paths[name],
            expected_sha256=source_sha256[name],
            label=f"attested {name}",
        )
    exact_attestation = {
        "query_count": expected_query_count,
        "requested_keypoint_count": EXPECTED_REQUESTED_KEYPOINT_COUNT,
        "reference_descriptor_dim": EXPECTED_REFERENCE_DESCRIPTOR_DIM,
        "candidate_descriptor_dim": None,
        "validated_descriptor_rows": 0,
        "validated_detector_keypoints": expected_validated_detector_keypoints,
    }
    for key, expected in exact_attestation.items():
        if attestation.get(key) != expected:
            raise ValueError(f"detector attestation {key} differs")
    config = _mapping(detector.get("config"), label="detector config")
    if dict(config) != {
        "radii_px": list(EXPECTED_RADII),
        "depth_abs_tolerance_m": EXPECTED_DEPTH_ABS_TOLERANCE_M,
        "depth_rel_tolerance": EXPECTED_DEPTH_REL_TOLERANCE,
        "alpha_minimum": EXPECTED_ALPHA_MINIMUM,
        "target_universe": "frozen_map_gt_projection_depth_alpha_mask_legal",
        "same_requested_k": True,
    }:
        raise ValueError("detector evaluator protocol differs from preregistration")

    baseline = _validate_reachability(
        detector.get("frozen_superpoint"), label="pooled baseline", include_query_count=True
    )
    candidate = _validate_reachability(
        detector.get("candidate"), label="pooled candidate", include_query_count=True
    )
    if baseline["query_count"] != expected_query_count or candidate["query_count"] != expected_query_count:
        raise ValueError("pooled detector query count differs")
    for kind in ("all", "track_core", "gaussian_reserve"):
        if baseline["by_anchor_kind"][kind]["target_count"] != candidate["by_anchor_kind"][kind]["target_count"]:
            raise ValueError(f"candidate/baseline {kind} target universes differ")
        if baseline["by_anchor_kind"][kind]["target_count"] != expected_target_counts[kind]:
            raise ValueError(f"{kind} target count differs from exact lineage")
        if expected_target_counts[kind] <= 0:
            raise ValueError(f"{kind} target domain is empty")

    deltas = _mapping(
        detector.get("delta_candidate_minus_superpoint"), label="detector deltas"
    )
    if set(deltas) != {"all", "track_core", "gaussian_reserve"}:
        raise ValueError("detector delta anchor-kind registry differs")
    for kind in ("all", "track_core", "gaussian_reserve"):
        values = _mapping(deltas[kind], label=f"{kind} deltas")
        if tuple(values) != EXPECTED_RADIUS_KEYS:
            raise ValueError(f"{kind} delta radius registry differs")
        total = baseline["by_anchor_kind"][kind]["target_count"]
        for key in EXPECTED_RADIUS_KEYS:
            expected = (
                candidate["by_anchor_kind"][kind]["hit_count"][key]
                - baseline["by_anchor_kind"][kind]["hit_count"][key]
            ) / total
            actual = _finite_number(values[key], label=f"{kind}@{key} delta")
            if not math.isclose(
                actual, expected, rel_tol=0.0, abs_tol=REPORTED_VALUE_ABS_TOLERANCE
            ):
                raise ValueError(f"{kind}@{key} delta is not count-derived")

    rows = detector.get("per_query")
    if not isinstance(rows, list) or len(rows) != expected_query_count:
        raise ValueError("detector per-query registry differs")
    query_names = []
    seen_query_names = set()
    aggregates = {
        arm: {
            kind: {"target_count": 0, "hit_count": {key: 0 for key in EXPECTED_RADIUS_KEYS}}
            for kind in ("all", "track_core", "gaussian_reserve")
        }
        for arm in ("baseline", "candidate")
    }
    candidate_keypoint_total = 0
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, label=f"per-query row {index}")
        if set(row) != {
            "query_name",
            "legal_anchor_count",
            "reference_keypoint_count",
            "candidate_keypoint_count",
            "frozen_superpoint",
            "candidate",
        }:
            raise ValueError("detector per-query row registry differs")
        name = str(row.get("query_name", ""))
        if not name or name in seen_query_names:
            raise ValueError("detector per-query names are invalid or duplicated")
        query_names.append(name)
        seen_query_names.add(name)
        reference_count = _nonnegative_int(
            row.get("reference_keypoint_count"), label=f"{name} reference keypoints"
        )
        candidate_count = _nonnegative_int(
            row.get("candidate_keypoint_count"), label=f"{name} candidate keypoints"
        )
        if reference_count > EXPECTED_REQUESTED_KEYPOINT_COUNT or candidate_count > EXPECTED_REQUESTED_KEYPOINT_COUNT:
            raise ValueError(f"{name} post-mask keypoint count exceeds K")
        candidate_keypoint_total += candidate_count
        per_baseline = _validate_reachability(
            row.get("frozen_superpoint"), label=f"{name} baseline", include_query_count=False
        )
        per_candidate = _validate_reachability(
            row.get("candidate"), label=f"{name} candidate", include_query_count=False
        )
        legal = _nonnegative_int(
            row.get("legal_anchor_count"), label=f"{name} legal anchors"
        )
        if (
            legal != per_baseline["by_anchor_kind"]["all"]["target_count"]
            or legal != per_candidate["by_anchor_kind"]["all"]["target_count"]
        ):
            raise ValueError(f"{name} legal-anchor target count differs")
        for kind in ("all", "track_core", "gaussian_reserve"):
            if per_baseline["by_anchor_kind"][kind]["target_count"] != per_candidate["by_anchor_kind"][kind]["target_count"]:
                raise ValueError(f"{name} candidate/baseline {kind} targets differ")
            for arm, values in (("baseline", per_baseline), ("candidate", per_candidate)):
                aggregate = aggregates[arm][kind]
                source = values["by_anchor_kind"][kind]
                aggregate["target_count"] += source["target_count"]
                for key in EXPECTED_RADIUS_KEYS:
                    aggregate["hit_count"][key] += source["hit_count"][key]
    if candidate_keypoint_total != expected_validated_detector_keypoints:
        raise ValueError("per-query candidate keypoints are not attestation-additive")
    for arm, pooled in (("baseline", baseline), ("candidate", candidate)):
        for kind in ("all", "track_core", "gaussian_reserve"):
            if aggregates[arm][kind] != pooled["by_anchor_kind"][kind]:
                raise ValueError(f"pooled {arm} {kind} counts are not query-additive")
    return detector, locked_sources, tuple(query_names)


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
    parser.add_argument("--expected-validated-detector-keypoints", type=int, required=True)
    parser.add_argument("--expected-all-target-count", type=int, required=True)
    parser.add_argument("--expected-track-target-count", type=int, required=True)
    parser.add_argument("--expected-reserve-target-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report_path = _resolved_file(args.report, label="detector report")
    expected_report_sha256 = _sha256(
        args.expected_report_sha256, label="expected detector report SHA-256"
    )
    source_paths = {
        "state": _resolved_file(args.state, label="state"),
        "query_cache": _resolved_file(args.query_cache, label="query cache"),
        "teacher": _resolved_file(args.teacher, label="teacher"),
        "probe_cache": _resolved_file(args.probe_cache, label="probe cache"),
    }
    source_sha256 = {
        "state": _sha256(args.expected_state_sha256, label="expected state SHA-256"),
        "query_cache": _sha256(
            args.expected_query_cache_sha256, label="expected query-cache SHA-256"
        ),
        "teacher": _sha256(
            args.expected_teacher_sha256, label="expected teacher SHA-256"
        ),
        "probe_cache": _sha256(
            args.expected_probe_cache_sha256, label="expected probe-cache SHA-256"
        ),
    }
    candidate_weights_path = _resolved_file(
        args.candidate_weights, label="candidate weights"
    )
    candidate_weights_sha256 = _sha256(
        args.expected_candidate_weights_sha256,
        label="expected candidate-weights SHA-256",
    )
    if candidate_weights_sha256 != EXPECTED_XFEAT_WEIGHTS_SHA256:
        raise ValueError("candidate weights differ from preregistered XFeat checkpoint")
    expected_query_count = int(args.expected_query_count)
    expected_validated_detector_keypoints = int(
        args.expected_validated_detector_keypoints
    )
    if expected_query_count != EXPECTED_QUERY_COUNT:
        raise ValueError("query count differs from preregistered Stairs mapping split")
    if expected_validated_detector_keypoints <= 0:
        raise ValueError("expected query/detector counts must be positive")
    if expected_validated_detector_keypoints > (
        expected_query_count * EXPECTED_REQUESTED_KEYPOINT_COUNT
    ):
        raise ValueError("validated post-mask detector rows exceed the global K cap")
    expected_target_counts = {
        "all": int(args.expected_all_target_count),
        "track_core": int(args.expected_track_target_count),
        "gaussian_reserve": int(args.expected_reserve_target_count),
    }
    if any(value <= 0 for value in expected_target_counts.values()):
        raise ValueError("all preregistered target domains must be non-empty")
    if expected_target_counts["track_core"] + expected_target_counts[
        "gaussian_reserve"
    ] != expected_target_counts["all"]:
        raise ValueError("expected target counts are not anchor-kind exhaustive")
    output_path = Path(args.output).expanduser().resolve()
    if output_path in {report_path, candidate_weights_path, *source_paths.values()}:
        raise ValueError("gate output must not overwrite a source artifact")

    expected_evaluation_code = frontend_detector_evaluation_code_identity(
        require_clean=True
    )
    detector, locked_sources, report_names = _validate_report(
        json.loads(report_path.read_text(encoding="utf-8")),
        report_path=report_path,
        expected_report_sha256=expected_report_sha256,
        source_paths=source_paths,
        source_sha256=source_sha256,
        candidate_weights_path=candidate_weights_path,
        candidate_weights_sha256=candidate_weights_sha256,
        expected_query_count=expected_query_count,
        expected_validated_detector_keypoints=expected_validated_detector_keypoints,
        expected_target_counts=expected_target_counts,
        expected_evaluation_code=expected_evaluation_code,
    )
    probe_names, probe_counts = _validate_probe(
        _torch_load(source_paths["probe_cache"]),
        source_paths=source_paths,
        source_sha256=source_sha256,
        candidate_weights_path=candidate_weights_path,
        candidate_weights_sha256=candidate_weights_sha256,
        expected_query_count=expected_query_count,
        expected_validated_detector_keypoints=expected_validated_detector_keypoints,
    )
    if probe_names != report_names:
        raise ValueError("probe/report query-name registries differ")

    baseline = detector["frozen_superpoint"]["by_anchor_kind"]
    candidate = detector["candidate"]["by_anchor_kind"]
    metric_specs = (
        ("all_at_2px", "all", "2.0", False),
        ("all_at_4px", "all", "4.0", True),
        ("all_at_8px", "all", "8.0", False),
        ("track_core_at_4px", "track_core", "4.0", False),
        ("gaussian_reserve_at_4px", "gaussian_reserve", "4.0", False),
    )
    comparisons = {}
    gates = {}
    for name, kind, radius, strict in metric_specs:
        baseline_hits = int(baseline[kind]["hit_count"][radius])
        candidate_hits = int(candidate[kind]["hit_count"][radius])
        target_count = int(baseline[kind]["target_count"])
        delta_hits = candidate_hits - baseline_hits
        delta_fraction = delta_hits / target_count
        comparisons[name] = {
            "target_count": target_count,
            "frozen_superpoint_hit_count": baseline_hits,
            "candidate_hit_count": candidate_hits,
            "delta_hit_count": delta_hits,
            "frozen_superpoint_reachable_fraction": baseline_hits / target_count,
            "candidate_reachable_fraction": candidate_hits / target_count,
            "delta_reachable_fraction": delta_fraction,
        }
        gates[
            f"{name}_{'strict_positive' if strict else 'non_regression'}"
        ] = (
            candidate_hits > baseline_hits
            if strict
            else (
                candidate_hits >= baseline_hits
                and delta_fraction >= -NON_REGRESSION_ABS_TOLERANCE
            )
        )
    passed = all(gates.values())
    gate_report = {
        "schema": GATE_SCHEMA,
        "version": 1,
        "mapping_only": True,
        "uses_test_queries": False,
        "single_factor": "same_K_single_image_detector_repeatability_only",
        "valid": True,
        "protocol": {
            "requested_keypoint_count": EXPECTED_REQUESTED_KEYPOINT_COUNT,
            "reference_superpoint_nms_radius": EXPECTED_REFERENCE_NMS_RADIUS,
            "candidate_xfeat_nms_kernel_size": EXPECTED_CANDIDATE_NMS_KERNEL,
            "candidate_xfeat_strict_probability_threshold": (
                EXPECTED_DETECTION_THRESHOLD
            ),
            "radii_px": list(EXPECTED_RADII),
            "target_universe": "frozen_map_gt_projection_depth_alpha_mask_legal",
            "strict_positive_metric": "pooled_all_at_4px_integer_hit_count",
            "non_regression_metrics": [
                "pooled_all_at_2px",
                "pooled_all_at_8px",
                "pooled_track_core_at_4px",
                "pooled_gaussian_reserve_at_4px",
            ],
            "non_regression_absolute_tolerance": NON_REGRESSION_ABS_TOLERANCE,
            "reported_value_validation_absolute_tolerance": (
                REPORTED_VALUE_ABS_TOLERANCE
            ),
        },
        "comparisons": comparisons,
        "gates": gates,
        "mechanism_gate_passed": passed,
        "advance_to_mapping_only_detector_rebuild": passed,
        "decision": "GO" if passed else "STOP",
        "limitations": [
            "GO authorizes only a mapping-only XFeat-detector rebuild and a later pose gate; it is not a test-set or pose claim.",
            "STOP rejects this locked same-K detector candidate; it does not reject all frontend redesigns.",
        ],
        "inputs": {
            "detector_report": {
                "path": str(report_path),
                "sha256": expected_report_sha256,
            },
            "source_artifacts": locked_sources,
            "candidate_weights": {
                "path": str(candidate_weights_path),
                "sha256": candidate_weights_sha256,
            },
            "evaluation_code": expected_evaluation_code,
            "probe_counts": probe_counts,
            "exact_target_counts": expected_target_counts,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(gate_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(gate_report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
