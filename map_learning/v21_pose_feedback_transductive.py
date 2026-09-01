"""Quarantined V21 test-adapted owner-prototype feedback and cached replay.

This module deliberately separates two operations:

* candidate formation consumes adaptation records and exact PoseLib recovery
  bundles from the Gaussian geometry oracle; and
* cached evaluation consumes an immutable frontend cache from any one role.

Gaussian recovery establishes a pose-valid counterfactual, not projective
identity.  Consequently every candidate produced here remains experimental,
test-adapted, non-deployable, and unable to mutate the frozen map.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import math
import os
from pathlib import Path
import re
import uuid
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from localization.matcher import global_owner_prototype_top1
from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.v8_feedback_controller import task_error
from map_learning.v21_identity_calibration import (
    validate_identity_calibration_payload,
)
from map_learning.v21_test_cache import (
    tensor_sha256,
    validate_cache_payload,
)


CANDIDATE_SCHEMA = "lafgs_v21_pose_feedback_transductive_candidate"
EVALUATION_SCHEMA = "lafgs_v21_pose_feedback_transductive_cached_evaluation"
VERSION = 1
METADATA_FIELD = "v21_pose_feedback_transductive"
PROTOTYPE_FEATURE_FIELD = "anchor_extra_prototype_features"
PROTOTYPE_OWNER_FIELD = "anchor_extra_prototype_owner_rows"
GAUSSIAN_POSITIVE_SOURCE = "gaussian_geometry"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

R5_TRANSLATION_CM = 5.0
R5_ROTATION_DEG = 5.0


def _as_cpu(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device="cpu").contiguous()


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def source_record(
    path: str | Path,
    *,
    sha256_file_fn: Callable[[str | Path], str],
    expected_sha256: str | None = None,
) -> dict:
    """Resolve and hash one immutable input file."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = sha256_file_fn(resolved)
    if expected_sha256 is not None and digest != _require_sha256(
        expected_sha256, label="expected source SHA256"
    ):
        raise ValueError(f"V21 source SHA256 differs: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": int(resolved.stat().st_size),
    }


def verify_source_record(
    source: Mapping, *, sha256_file_fn: Callable[[str | Path], str]
) -> None:
    """Fail if a bound source changed after it was initially read."""

    path = Path(str(source.get("path", ""))).expanduser().resolve()
    digest = _require_sha256(source.get("sha256"), label="source SHA256")
    if (
        str(path) != str(source.get("path"))
        or not path.is_file()
        or int(path.stat().st_size) != int(source.get("size_bytes", -1))
        or sha256_file_fn(path) != digest
    ):
        raise RuntimeError(f"V21 bound source changed: {path}")


def _source_identity(source: object, *, label: str) -> tuple[str, str]:
    if not isinstance(source, Mapping):
        raise ValueError(f"V21 {label} source is missing")
    raw_path = str(source.get("path", ""))
    digest = _require_sha256(source.get("sha256"), label=f"V21 {label} SHA256")
    path = str(Path(raw_path).expanduser().resolve())
    if not raw_path or raw_path != path:
        raise ValueError(f"V21 {label} path is not canonical")
    return path, digest


def _same_source(left: object, right: Mapping, *, label: str) -> None:
    if _source_identity(left, label=label) != (
        str(right["path"]),
        str(right["sha256"]),
    ):
        raise ValueError(f"V21 {label} lineage differs")


def _bit_exact(left: Any, right: Any) -> bool:
    """Recursively compare frozen map values without numeric coercion."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and tensor_sha256(left) == tensor_sha256(right)
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_bit_exact(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_bit_exact(a, b) for a, b in zip(left, right))
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return bool(left == right)


def _clone_frozen_value(value: Any) -> Any:
    """Clone mutable map storage so a candidate cannot alias the stable map."""

    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _clone_frozen_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_frozen_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_frozen_value(item) for item in value)
    return value


def assert_base_fields_bit_exact(
    stable_map: Mapping, candidate_map: Mapping
) -> None:
    """Prove the candidate only appends the two extension fields and metadata."""

    additions = {PROTOTYPE_FEATURE_FIELD, PROTOTYPE_OWNER_FIELD, METADATA_FIELD}
    if set(candidate_map) != set(stable_map) | additions:
        raise ValueError("V21 candidate changed the frozen map field registry")
    for key, value in stable_map.items():
        if not _bit_exact(value, candidate_map[key]):
            raise ValueError(f"V21 candidate changed frozen base field: {key}")


def _validate_stable_map(stable_map: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    if stable_map.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("V21 stable map schema is unsupported")
    if PROTOTYPE_FEATURE_FIELD in stable_map or PROTOTYPE_OWNER_FIELD in stable_map:
        raise ValueError("V21 transductive action requires an unextended stable map")
    features = torch.as_tensor(stable_map.get("anchor_features"))
    xyz = torch.as_tensor(stable_map.get("anchor_xyz"))
    anchor_ids = torch.as_tensor(stable_map.get("anchor_ids"))
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or xyz.shape != (features.shape[0], 3)
        or anchor_ids.shape != (features.shape[0],)
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(xyz).all())
    ):
        raise ValueError("V21 stable map Anchor registry is invalid")
    return features.detach().cpu(), xyz.detach().cpu()


def validate_baseline_contract(contract: object) -> dict:
    """Validate the exact matcher/PoseLib contract inherited by cached replay."""

    if not isinstance(contract, Mapping):
        raise ValueError("V21 baseline contract is missing")
    result = dict(contract)
    if not (
        result.get("matching")
        == "exact_global_cosine_top1_lower_anchor_row_tie_break"
        and result.get("pose_solver")
        == "single_standard_poselib_absolute_pose"
        and result.get("cached_keypoints")
        == "native_integer_grid_without_pixel_center_offset"
        and result.get("pose_solver_points_2d") == "cached_keypoints_plus_0.5"
        and result.get("r5")
        == "translation_cm_strictly_below_5_and_rotation_deg_strictly_below_5"
        and math.isfinite(float(result.get("pixel_center_offset", math.nan)))
        and float(result["pixel_center_offset"]) == 0.5
        and math.isfinite(float(result.get("reprojection_error_px", math.nan)))
        and math.isfinite(float(result.get("confidence", math.nan)))
        and float(result["reprojection_error_px"]) > 0.0
        and 0.0 < float(result["confidence"]) < 1.0
        and int(result.get("maximum_iterations", 0)) > 0
        and int(result.get("minimum_iterations", 0)) > 0
        and int(result["maximum_iterations"]) >= int(result["minimum_iterations"])
        and isinstance(result.get("seed"), int)
        and not isinstance(result.get("seed"), bool)
    ):
        raise ValueError("V21 baseline matcher/PoseLib contract is invalid")
    return result


def validate_complete_cache_payloads(
    payloads: Sequence[Mapping], *, required_role: str | None = None
) -> tuple[list[Mapping], list[dict], dict]:
    """Validate and order a complete one-role frontend cache registry."""

    if not payloads:
        raise ValueError("V21 frontend cache set is empty")
    for payload in payloads:
        validate_cache_payload(payload)
    first = payloads[0]
    role = str(first["role"])
    if required_role is not None and role != required_role:
        raise ValueError(f"V21 frontend cache role must be {required_role}")
    shard_count = int(first["shard_count"])
    registry = first["shard_registry"]
    baseline = validate_baseline_contract(first.get("baseline_contract"))
    stable_source = _source_identity(
        first.get("inputs", {}).get("stable_map"), label="cache stable map"
    )
    split_source = _source_identity(
        first.get("inputs", {}).get("split_manifest"), label="cache split"
    )
    coordinates: set[int] = set()
    for payload in payloads:
        coordinate = int(payload["shard_index"])
        if (
            payload["role"] != role
            or int(payload["shard_count"]) != shard_count
            or payload["shard_registry"] != registry
            or validate_baseline_contract(payload.get("baseline_contract")) != baseline
            or payload.get("frontend_contract") != first.get("frontend_contract")
            or _source_identity(
                payload.get("inputs", {}).get("stable_map"),
                label="cache stable map",
            )
            != stable_source
            or _source_identity(
                payload.get("inputs", {}).get("split_manifest"),
                label="cache split",
            )
            != split_source
            or coordinate in coordinates
        ):
            raise ValueError("V21 frontend cache shard contracts differ")
        coordinates.add(coordinate)
    if coordinates != set(range(shard_count)):
        raise ValueError("V21 frontend caches do not cover the full shard registry")
    ordered_payloads = sorted(payloads, key=lambda value: int(value["shard_index"]))
    ordinal_by_query = {
        int(row["query_index"]): int(row["ordinal"])
        for row in registry["rows"]
    }
    records = [dict(record) for payload in ordered_payloads for record in payload["records"]]
    if (
        len(records) != int(registry["role_query_count"])
        or len(ordinal_by_query) != len(records)
        or {int(record["query_index"]) for record in records} != set(ordinal_by_query)
    ):
        raise ValueError("V21 frontend caches do not exactly cover their registry")
    records.sort(key=lambda record: ordinal_by_query[int(record["query_index"])])
    return ordered_payloads, records, baseline


def replay_pose_with_contract(
    *,
    keypoints: torch.Tensor,
    anchor_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
    baseline_contract: Mapping,
    solver: Callable = solve_absolute_pose,
) -> dict:
    """Run exactly the cached standard PoseLib plant."""

    contract = validate_baseline_contract(baseline_contract)
    xy = _as_cpu(keypoints, dtype=torch.float32)
    rows = _as_cpu(anchor_rows, dtype=torch.long).reshape(-1)
    xyz = _as_cpu(anchor_xyz, dtype=torch.float32)
    calibration = _as_cpu(intrinsic, dtype=torch.float32)
    truth = _as_cpu(ground_truth_w2c, dtype=torch.float32)
    if xy.shape != (rows.numel(), 2):
        raise ValueError("V21 pose keypoints and Anchor rows do not align")
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= xyz.shape[0]):
        raise ValueError("V21 pose assignment is outside the stable map")
    estimate = solver(
        xy.numpy(),
        xyz[rows].numpy(),
        calibration.numpy(),
        reprojection_error_px=float(contract["reprojection_error_px"]),
        confidence=float(contract["confidence"]),
        max_iterations=int(contract["maximum_iterations"]),
        min_iterations=int(contract["minimum_iterations"]),
        seed=int(contract["seed"]),
    )
    pose = np.asarray(estimate.pose_w2c)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("V21 exact cached replay returned an invalid pose")
    rotation_deg, translation_cm = pose_error(pose, truth.numpy())
    inliers = np.asarray(estimate.inliers, dtype=np.int64).reshape(-1)
    if inliers.size and (
        int(inliers.min()) < 0 or int(inliers.max()) >= rows.numel()
    ):
        raise ValueError("V21 exact cached replay returned invalid inliers")
    objective = task_error(translation_cm, rotation_deg)
    return {
        "pose_w2c": torch.from_numpy(pose.copy()).float(),
        "translation_error_cm": float(translation_cm),
        "rotation_error_deg": float(rotation_deg),
        "task_error": float(objective),
        "r5_success": bool(
            translation_cm < R5_TRANSLATION_CM
            and rotation_deg < R5_ROTATION_DEG
        ),
        "inlier_count": int(inliers.size),
        "inlier_query_rows": torch.from_numpy(inliers.copy()).long(),
    }


def _assert_pose_outcome_matches(
    observed: Mapping, expected: Mapping, *, label: str, require_pose_exact: bool
) -> None:
    for key in ("translation_error_cm", "rotation_error_deg", "task_error"):
        if not math.isclose(
            float(observed[key]), float(expected.get(key, math.nan)), rel_tol=1e-6, abs_tol=1e-4
        ):
            raise ValueError(f"V21 {label} {key} differs from exact replay")
    if bool(observed["r5_success"]) != bool(expected.get("r5_success")):
        raise ValueError(f"V21 {label} R5 differs from exact replay")
    if expected.get("inlier_count") is not None and int(observed["inlier_count"]) != int(
        expected["inlier_count"]
    ):
        raise ValueError(f"V21 {label} inlier count differs from exact replay")
    if expected.get("inlier_query_rows") is not None and not torch.equal(
        observed["inlier_query_rows"],
        _as_cpu(expected["inlier_query_rows"], dtype=torch.long).reshape(-1),
    ):
        raise ValueError(f"V21 {label} inlier rows differ from exact replay")
    if require_pose_exact and expected.get("pose_w2c") is not None and not torch.equal(
        observed["pose_w2c"], _as_cpu(expected["pose_w2c"], dtype=torch.float32)
    ):
        raise ValueError(f"V21 {label} pose differs from exact replay")


def _assert_cached_baseline(record: Mapping, replay: Mapping) -> None:
    expected = {
        "pose_w2c": record.get("baseline_pose_w2c"),
        "translation_error_cm": record.get("baseline_translation_error_cm"),
        "rotation_error_deg": record.get("baseline_rotation_error_deg"),
        "task_error": record.get("baseline_task_error"),
        "r5_success": record.get("baseline_r5"),
        "inlier_count": record.get("baseline_inlier_count"),
        "inlier_query_rows": record.get("baseline_inliers"),
    }
    _assert_pose_outcome_matches(
        replay, expected, label="cached baseline", require_pose_exact=True
    )


def _patch_assignments(
    winners: torch.Tensor, rows: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    output = _as_cpu(winners, dtype=torch.long).reshape(-1).clone()
    rows = _as_cpu(rows, dtype=torch.long).reshape(-1)
    anchors = _as_cpu(anchors, dtype=torch.long).reshape(-1)
    if rows.shape != anchors.shape or torch.unique(rows).numel() != rows.numel():
        raise ValueError("V21 correction bundle rows are invalid")
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= output.numel()):
        raise ValueError("V21 correction bundle query row is outside the cache")
    output[rows] = anchors
    return output


def _calibration_edge_set(record: Mapping, *, anchor_count: int) -> set[tuple[int, int]]:
    offsets = _as_cpu(record.get("provisional_positive_offsets"), dtype=torch.long).reshape(-1)
    anchors = _as_cpu(record.get("provisional_positive_anchor_rows"), dtype=torch.long).reshape(-1)
    count = int(record.get("keypoint_count", -1))
    if (
        offsets.shape != (count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != anchors.numel()
        or bool((offsets[1:] < offsets[:-1]).any())
        or (anchors.numel() and (int(anchors.min()) < 0 or int(anchors.max()) >= anchor_count))
    ):
        raise ValueError("V21 provisional calibration CSR is invalid")
    return {
        (row, int(anchors[index]))
        for row in range(count)
        for index in range(int(offsets[row]), int(offsets[row + 1]))
    }


def validate_gaussian_oracle_aggregate(
    payload: Mapping,
    *,
    stable_map_source: Mapping,
    adaptation_cache_sources: Sequence[Mapping],
    adaptation_records: Sequence[Mapping],
) -> list[dict]:
    """Validate the legacy/current Gaussian aggregate against exact inputs."""

    records = payload.get("records")
    registry = payload.get("frontend_query_registry")
    inputs = payload.get("input")
    explicit_source = payload.get("positive_source")
    legacy_geometry = explicit_source is None
    if not (
        payload.get("schema") == "lafgs_v21_pose_recovery_oracle_aggregate"
        and payload.get("version") == 1
        and payload.get("protocol") == "test_adapted"
        and payload.get("uses_test_queries") is True
        and payload.get("role") == "adaptation"
        and payload.get("deployment_authorized") is False
        and (
            payload.get("pose_recovery_is_diagnostic_upper_bound_only") is True
            or (
                legacy_geometry
                and payload.get("pose_recovery_is_diagnostic_upper_bound_only")
                is None
            )
        )
        and payload.get("correspondence_identity_authority_present") is False
        and payload.get("geometry_recovery_is_upper_bound_only") is True
        and isinstance(records, list)
        and isinstance(registry, list)
        and isinstance(inputs, Mapping)
        and len(records) == len(registry) == len(adaptation_records)
        and int(payload.get("source_query_count", -1)) == len(records)
        and (legacy_geometry or explicit_source == GAUSSIAN_POSITIVE_SOURCE)
        and inputs.get("correspondence_truth") is None
    ):
        raise ValueError("V21 Gaussian pose-recovery aggregate contract is invalid")
    if legacy_geometry and any("positive_source" in record for record in records):
        raise ValueError("V21 legacy Gaussian aggregate mixes positive sources")
    if not legacy_geometry and any(
        record.get("positive_source") != GAUSSIAN_POSITIVE_SOURCE for record in records
    ):
        raise ValueError("V21 Gaussian aggregate record source differs")
    gaussian_support = inputs.get("gaussian_support")
    oracle_shards = payload.get("oracle_shards")
    if (
        not isinstance(gaussian_support, list)
        or not gaussian_support
        or not isinstance(oracle_shards, list)
        or not oracle_shards
    ):
        raise ValueError("V21 Gaussian aggregate support/shard lineage is missing")
    support_identities = [
        _source_identity(value, label="oracle Gaussian support")
        for value in gaussian_support
    ]
    shard_identities = [
        _source_identity(value, label="oracle source shard")
        for value in oracle_shards
    ]
    if len(set(support_identities)) != len(support_identities) or len(
        set(shard_identities)
    ) != len(shard_identities):
        raise ValueError("V21 Gaussian aggregate support/shard lineage is duplicated")
    frozen_map = (
        str(Path(str(inputs.get("frozen_map", ""))).expanduser().resolve()),
        str(inputs.get("frozen_map_sha256", "")),
    )
    if frozen_map != (str(stable_map_source["path"]), str(stable_map_source["sha256"])):
        raise ValueError("V21 Gaussian oracle stable-map lineage differs")
    oracle_cache_sources = {
        _source_identity(value, label="oracle adaptation cache")
        for value in inputs.get("adaptation_caches", ())
    }
    expected_cache_sources = {
        (str(value["path"]), str(value["sha256"]))
        for value in adaptation_cache_sources
    }
    if oracle_cache_sources != expected_cache_sources:
        raise ValueError("V21 Gaussian oracle adaptation-cache lineage differs")
    ordered = []
    for ordinal, (oracle_record, registry_row, cache_record) in enumerate(
        zip(records, registry, adaptation_records)
    ):
        if not (
            int(registry_row.get("ordinal", -1)) == ordinal
            and int(oracle_record.get("query_index", -1))
            == int(registry_row.get("query_index", -2))
            == int(cache_record["query_index"])
            and oracle_record.get("image_name")
            == registry_row.get("image_name")
            == cache_record["image_name"]
            and registry_row.get("source_record_sha256")
            == cache_record["source_record_sha256"]
            and oracle_record.get("sequence_id") == cache_record["sequence_id"]
            and oracle_record.get("block_id") == cache_record["block_id"]
            and oracle_record.get("controller_authorized") is False
            and oracle_record.get("legal_positive_csr", {}).get(
                "positive_evidence_mode"
            )
            == "gaussian_geometry_supported_upper_bound"
            and oracle_record.get("legal_positive_csr", {}).get(
                "geometry_supported_candidate"
            )
            is True
            and oracle_record.get("legal_positive_csr", {}).get(
                "deployable_positive_authorized"
            )
            is False
            and oracle_record.get("protection", {}).get(
                "positive_evidence_mode"
            )
            == "gaussian_geometry_supported_upper_bound"
        ):
            raise ValueError("V21 Gaussian oracle query registry differs from cache")
        ordered.append(dict(oracle_record))
    return ordered


def _validate_calibration_join(
    calibration: Mapping,
    *,
    stable_map_source: Mapping,
    cache_sources: Sequence[Mapping],
    cache_records: Sequence[Mapping],
    anchor_count: int,
) -> dict[int, set[tuple[int, int]]]:
    validate_identity_calibration_payload(calibration)
    _same_source(
        calibration.get("inputs", {}).get("stable_map"),
        stable_map_source,
        label="calibration stable map",
    )
    declared_caches = {
        _source_identity(value, label="calibration frontend cache")
        for value in calibration.get("inputs", {}).get("frontend_caches", ())
    }
    expected_caches = {
        (str(value["path"]), str(value["sha256"])) for value in cache_sources
    }
    if declared_caches != expected_caches:
        raise ValueError("V21 provisional calibration cache lineage differs")
    if int(calibration.get("anchor_count", -1)) != anchor_count:
        raise ValueError("V21 provisional calibration Anchor registry differs")
    calibration_records = calibration.get("records", ())
    if len(calibration_records) != len(cache_records):
        raise ValueError("V21 provisional calibration query coverage differs")
    output = {}
    for source, target in zip(cache_records, calibration_records):
        query_index = int(source["query_index"])
        if not (
            query_index == int(target.get("query_index", -1))
            and target.get("image_name") == source["image_name"]
            and target.get("image_sha256") == source["image_sha256"]
            and target.get("source_record_sha256") == source["source_record_sha256"]
            and target.get("pose_w2c_sha256") == source["pose_w2c_sha256"]
            and int(target.get("keypoint_count", -1))
            == int(torch.as_tensor(source["keypoints"]).shape[0])
            and target.get("keypoints_sha256")
            == tensor_sha256(torch.as_tensor(source["keypoints"]).float())
            and target.get("descriptors_sha256")
            == tensor_sha256(torch.as_tensor(source["descriptors"]).float())
        ):
            raise ValueError("V21 provisional calibration query binding differs")
        output[query_index] = _calibration_edge_set(
            target, anchor_count=anchor_count
        )
    return output


def build_pose_feedback_transductive_candidate(
    *,
    stable_map: Mapping,
    adaptation_cache_payloads: Sequence[Mapping],
    gaussian_oracle_aggregate: Mapping,
    stable_map_source: Mapping,
    adaptation_cache_sources: Sequence[Mapping],
    oracle_source: Mapping,
    provisional_calibration: Mapping | None = None,
    calibration_source: Mapping | None = None,
    maximum_bundle_size: int = 8,
    maximum_source_queries: int = 16,
    maximum_total_prototypes: int = 64,
    maximum_prototypes_per_anchor: int = 4,
    require_one_assignment_translation_below_cm: float | None = 4.0,
    require_provisional_edge: bool = False,
    solver: Callable = solve_absolute_pose,
) -> dict:
    """Form one deterministic positive-only, non-deployable candidate map."""

    features, xyz = _validate_stable_map(stable_map)
    ordered_payloads, cache_records, baseline_contract = validate_complete_cache_payloads(
        adaptation_cache_payloads, required_role="adaptation"
    )
    if not all(
        payload.get("training_consumers_allowed") is True
        and payload.get("training_consumer_allowed") is True
        for payload in ordered_payloads
    ):
        raise ValueError("V21 candidate formation received held-out cache data")
    if len(adaptation_cache_sources) != len(ordered_payloads):
        raise ValueError("V21 adaptation cache source registry differs")
    if len(
        {
            (str(value.get("path")), str(value.get("sha256")))
            for value in adaptation_cache_sources
        }
    ) != len(adaptation_cache_sources):
        raise ValueError("V21 adaptation cache sources are duplicated")
    expected_cache_sources = {
        _source_identity(
            payload.get("inputs", {}).get("stable_map"), label="cache stable map"
        )
        for payload in ordered_payloads
    }
    if expected_cache_sources != {
        (str(stable_map_source["path"]), str(stable_map_source["sha256"]))
    }:
        raise ValueError("V21 candidate cache/stable-map lineage differs")
    if int(ordered_payloads[0]["anchor_count"]) != features.shape[0] or int(
        ordered_payloads[0]["descriptor_dim"]
    ) != features.shape[1]:
        raise ValueError("V21 candidate cache/map descriptor registry differs")
    oracle_records = validate_gaussian_oracle_aggregate(
        gaussian_oracle_aggregate,
        stable_map_source=stable_map_source,
        adaptation_cache_sources=adaptation_cache_sources,
        adaptation_records=cache_records,
    )
    if (
        int(maximum_bundle_size) < 1
        or int(maximum_source_queries) < 1
        or int(maximum_total_prototypes) < 1
        or int(maximum_prototypes_per_anchor) < 1
        or (
            require_one_assignment_translation_below_cm is not None
            and (
                not math.isfinite(float(require_one_assignment_translation_below_cm))
                or not 0.0 < float(require_one_assignment_translation_below_cm) <= 5.0
            )
        )
    ):
        raise ValueError("V21 transductive candidate budgets are invalid")
    if require_provisional_edge and provisional_calibration is None:
        raise ValueError("V21 provisional-edge requirement needs calibration")
    if (provisional_calibration is None) != (calibration_source is None):
        raise ValueError("V21 provisional calibration payload/source must be paired")
    provisional_edges = (
        _validate_calibration_join(
            provisional_calibration,
            stable_map_source=stable_map_source,
            cache_sources=adaptation_cache_sources,
            cache_records=cache_records,
            anchor_count=features.shape[0],
        )
        if provisional_calibration is not None
        else {}
    )
    cache_by_query = {int(record["query_index"]): record for record in cache_records}
    candidates = []
    audits = []
    for oracle_record in oracle_records:
        query_index = int(oracle_record["query_index"])
        cache_record = cache_by_query[query_index]
        reason = None
        bundle = oracle_record.get("recovery_bundle")
        one_assignment = oracle_record.get("one_assignment_lower_bound")
        if bool(oracle_record.get("baseline", {}).get("r5_success", True)):
            reason = "baseline_success_has_no_recovery_action"
        elif not isinstance(one_assignment, Mapping) or one_assignment.get("r5_success") is not True:
            reason = "one_assignment_does_not_recover_r5"
        elif not isinstance(bundle, Mapping):
            reason = "exact_recovery_bundle_absent"
        elif int(bundle.get("exact_delta_r5", 0)) != 1 or bundle.get("inclusion_minimal") is not True:
            reason = "bundle_is_not_exact_minimal_r5_recovery"
        if reason is not None:
            audits.append({"query_index": query_index, "eligible": False, "reason": reason})
            continue
        rows = _as_cpu(bundle.get("query_rows"), dtype=torch.long).reshape(-1)
        owners = _as_cpu(bundle.get("anchor_rows"), dtype=torch.long).reshape(-1)
        if (
            rows.shape != owners.shape
            or rows.numel() == 0
            or rows.numel() > int(maximum_bundle_size)
            or torch.unique(rows).numel() != rows.numel()
            or rows.numel() > int(maximum_total_prototypes)
            or (owners.numel() and (int(owners.min()) < 0 or int(owners.max()) >= features.shape[0]))
        ):
            audits.append(
                {
                    "query_index": query_index,
                    "eligible": False,
                    "reason": "bundle_shape_or_size_is_ineligible",
                }
            )
            continue
        descriptor_bank = _as_cpu(cache_record["descriptors"], dtype=torch.float32)
        if rows.numel() and int(rows.max()) >= descriptor_bank.shape[0]:
            raise ValueError("V21 recovery bundle row is outside its source query")
        correction = oracle_record.get("correction_candidates")
        if not isinstance(correction, Mapping):
            raise ValueError("V21 recovery oracle correction assignment is missing")
        assignment_rows = _as_cpu(correction.get("candidate_rows"), dtype=torch.long).reshape(-1)
        assignment_owners = _as_cpu(
            correction.get("candidate_positive_anchor_rows"), dtype=torch.long
        ).reshape(-1)
        winners = _as_cpu(cache_record["winner_anchor_rows"], dtype=torch.long).reshape(-1)
        physical_keypoints = _as_cpu(cache_record["keypoints"], dtype=torch.float32) + float(
            baseline_contract["pixel_center_offset"]
        )
        exact_baseline = replay_pose_with_contract(
            keypoints=physical_keypoints,
            anchor_rows=winners,
            anchor_xyz=xyz,
            intrinsic=cache_record["intrinsics"],
            ground_truth_w2c=cache_record["pose_w2c"],
            baseline_contract=baseline_contract,
            solver=solver,
        )
        _assert_cached_baseline(cache_record, exact_baseline)
        _assert_pose_outcome_matches(
            exact_baseline,
            oracle_record["baseline"],
            label="oracle baseline",
            require_pose_exact=True,
        )
        exact_one_assignment = replay_pose_with_contract(
            keypoints=physical_keypoints,
            anchor_rows=_patch_assignments(winners, assignment_rows, assignment_owners),
            anchor_xyz=xyz,
            intrinsic=cache_record["intrinsics"],
            ground_truth_w2c=cache_record["pose_w2c"],
            baseline_contract=baseline_contract,
            solver=solver,
        )
        _assert_pose_outcome_matches(
            exact_one_assignment,
            one_assignment,
            label="one-assignment oracle",
            require_pose_exact=True,
        )
        if not exact_one_assignment["r5_success"] or (
            require_one_assignment_translation_below_cm is not None
            and not float(exact_one_assignment["translation_error_cm"])
            < float(require_one_assignment_translation_below_cm)
        ):
            audits.append(
                {
                    "query_index": query_index,
                    "eligible": False,
                    "reason": "one_assignment_misses_required_translation_margin",
                }
            )
            continue
        exact_bundle = replay_pose_with_contract(
            keypoints=physical_keypoints,
            anchor_rows=_patch_assignments(winners, rows, owners),
            anchor_xyz=xyz,
            intrinsic=cache_record["intrinsics"],
            ground_truth_w2c=cache_record["pose_w2c"],
            baseline_contract=baseline_contract,
            solver=solver,
        )
        if not isinstance(bundle.get("pose"), Mapping):
            raise ValueError("V21 recovery bundle exact pose is missing")
        _assert_pose_outcome_matches(
            exact_bundle,
            bundle["pose"],
            label="recovery bundle",
            require_pose_exact=True,
        )
        if not exact_bundle["r5_success"]:
            raise ValueError("V21 recovery bundle fails official R5 replay")
        supported = provisional_edges.get(query_index, set())
        provisional_mask = torch.tensor(
            [(int(row), int(owner)) in supported for row, owner in zip(rows, owners)],
            dtype=torch.bool,
        )
        if require_provisional_edge and not bool(provisional_mask.all()):
            audits.append(
                {
                    "query_index": query_index,
                    "eligible": False,
                    "reason": "bundle_not_fully_covered_by_provisional_edges",
                    "provisional_edge_count": int(provisional_mask.sum()),
                    "bundle_size": int(rows.numel()),
                }
            )
            continue
        descriptors = F.normalize(descriptor_bank[rows], dim=1)
        if not bool(torch.isfinite(descriptors).all()) or bool(
            (descriptor_bank[rows].norm(dim=1) <= 0).any()
        ):
            raise ValueError("V21 source bundle descriptor is invalid")
        candidates.append(
            {
                "query_index": query_index,
                "image_name": str(cache_record["image_name"]),
                "source_record_sha256": str(cache_record["source_record_sha256"]),
                "query_rows": rows,
                "owner_rows": owners,
                "features": descriptors.contiguous(),
                "provisional_edge_mask": provisional_mask,
                "one_assignment": exact_one_assignment,
                "bundle_pose": exact_bundle,
            }
        )
    candidates.sort(
        key=lambda value: (
            float(value["one_assignment"]["translation_error_cm"]),
            float(value["one_assignment"]["rotation_error_deg"]),
            int(value["query_rows"].numel()),
            int(value["query_index"]),
        )
    )
    accepted = []
    owner_counts = Counter()
    total = 0
    for candidate in candidates:
        owners = [int(value) for value in candidate["owner_rows"].tolist()]
        local_counts = Counter(owners)
        size = len(owners)
        if len(accepted) >= int(maximum_source_queries):
            reason = "maximum_source_queries_reached"
        elif total + size > int(maximum_total_prototypes):
            reason = "maximum_total_prototypes_reached"
        elif any(
            owner_counts[owner] + count > int(maximum_prototypes_per_anchor)
            for owner, count in local_counts.items()
        ):
            reason = "maximum_prototypes_per_anchor_reached"
        else:
            reason = None
        if reason is not None:
            audits.append(
                {
                    "query_index": int(candidate["query_index"]),
                    "eligible": True,
                    "selected": False,
                    "reason": reason,
                }
            )
            continue
        start = total
        total += size
        owner_counts.update(local_counts)
        accepted.append(candidate)
        audits.append(
            {
                "query_index": int(candidate["query_index"]),
                "eligible": True,
                "selected": True,
                "reason": "accepted_complete_exact_recovery_bundle",
                "prototype_start": start,
                "prototype_stop": total,
            }
        )
    if not accepted:
        raise ValueError("V21 Gaussian oracle yields no eligible transductive bundle")
    prototype_features = torch.cat([value["features"] for value in accepted], dim=0)
    prototype_owners = torch.cat([value["owner_rows"] for value in accepted], dim=0)
    selected_actions = []
    cursor = 0
    for value in accepted:
        size = int(value["query_rows"].numel())
        selected_actions.append(
            {
                "query_index": int(value["query_index"]),
                "image_name": value["image_name"],
                "source_record_sha256": value["source_record_sha256"],
                "query_rows": value["query_rows"],
                "owner_anchor_rows": value["owner_rows"],
                "prototype_indices": torch.arange(cursor, cursor + size, dtype=torch.long),
                "one_assignment_translation_error_cm": float(
                    value["one_assignment"]["translation_error_cm"]
                ),
                "one_assignment_rotation_error_deg": float(
                    value["one_assignment"]["rotation_error_deg"]
                ),
                "bundle_translation_error_cm": float(
                    value["bundle_pose"]["translation_error_cm"]
                ),
                "bundle_rotation_error_deg": float(
                    value["bundle_pose"]["rotation_error_deg"]
                ),
                "bundle_standard_r5_success": True,
                "exact_poselib_replayed_during_materialization": True,
                "pose_valid_edge_claimed": True,
                "identity_truth_claimed": False,
                "provisional_edge_mask": value["provisional_edge_mask"],
                "all_edges_provisional": bool(value["provisional_edge_mask"].all()),
                "prototype_features_sha256": tensor_sha256(value["features"]),
                "prototype_owner_rows_sha256": tensor_sha256(value["owner_rows"]),
            }
        )
        cursor += size
    metadata = {
        "schema": CANDIDATE_SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "formation_role": "adaptation",
        "adaptation_features_consumed": True,
        "control_features_consumed": False,
        "confirmation_features_consumed": False,
        "control_or_confirmation_outcomes_consumed": False,
        "prototype_feature_source": "adaptation_query_descriptor_exact_recovery_bundle_row",
        "matching_semantics": "global_owner_prototype_top1",
        "base_anchor_fields_bit_exact": True,
        "base_anchor_features_moved_or_lowered": False,
        "geometry_changed": False,
        "pose_valid_edge_claimed": True,
        "identity_truth_claimed": False,
        "gaussian_geometry_is_identity_truth": False,
        "negative_anchor_labels_created": False,
        "deployment_authorized": False,
        "controller_authorized": False,
        "heldout_control_required": True,
        "heldout_confirmation_required": True,
        "inputs": {
            "stable_map": dict(stable_map_source),
            "split_manifest": dict(
                ordered_payloads[0].get("inputs", {})["split_manifest"]
            ),
            "adaptation_caches": [dict(value) for value in adaptation_cache_sources],
            "gaussian_geometry_oracle_aggregate": dict(oracle_source),
            "gaussian_geometry_support": [
                dict(value)
                for value in gaussian_oracle_aggregate["input"]["gaussian_support"]
            ],
            "pose_recovery_oracle_shards": [
                dict(value) for value in gaussian_oracle_aggregate["oracle_shards"]
            ],
            "provisional_identity_calibration": (
                dict(calibration_source) if calibration_source is not None else None
            ),
        },
        "frontend_shard_registry_sha256": ordered_payloads[0]["shard_registry"][
            "registry_sha256"
        ],
        "preprocessing_config_sha256": ordered_payloads[0][
            "preprocessing_config_sha256"
        ],
        "baseline_contract": baseline_contract,
        "budgets": {
            "maximum_bundle_size": int(maximum_bundle_size),
            "maximum_source_queries": int(maximum_source_queries),
            "maximum_total_prototypes": int(maximum_total_prototypes),
            "maximum_prototypes_per_anchor": int(maximum_prototypes_per_anchor),
            "require_one_assignment_translation_below_cm": (
                float(require_one_assignment_translation_below_cm)
                if require_one_assignment_translation_below_cm is not None
                else None
            ),
            "require_provisional_edge": bool(require_provisional_edge),
        },
        "source_query_count": len(cache_records),
        "eligible_bundle_count": len(candidates),
        "selected_source_query_count": len(accepted),
        "added_prototype_count": int(prototype_owners.numel()),
        "prototype_features_sha256": tensor_sha256(prototype_features),
        "prototype_owner_rows_sha256": tensor_sha256(prototype_owners),
        "selected_actions": selected_actions,
        "selection_audits": audits,
    }
    candidate_map = _clone_frozen_value(stable_map)
    candidate_map[PROTOTYPE_FEATURE_FIELD] = prototype_features.contiguous()
    candidate_map[PROTOTYPE_OWNER_FIELD] = prototype_owners.contiguous()
    candidate_map[METADATA_FIELD] = metadata
    validate_candidate_map(candidate_map, stable_map=stable_map)
    return candidate_map


def validate_candidate_map(candidate_map: Mapping, *, stable_map: Mapping) -> dict:
    """Validate extension shape, semantics, action registry, and frozen fields."""

    features, _ = _validate_stable_map(stable_map)
    assert_base_fields_bit_exact(stable_map, candidate_map)
    prototypes = torch.as_tensor(candidate_map.get(PROTOTYPE_FEATURE_FIELD))
    owners = torch.as_tensor(candidate_map.get(PROTOTYPE_OWNER_FIELD)).long().reshape(-1)
    metadata = candidate_map.get(METADATA_FIELD)
    if not (
        prototypes.shape == (owners.numel(), features.shape[1])
        and owners.numel() > 0
        and bool(torch.isfinite(prototypes).all())
        and torch.allclose(
            prototypes.float().norm(dim=1),
            torch.ones(owners.numel()),
            rtol=1e-5,
            atol=1e-6,
        )
        and int(owners.min()) >= 0
        and int(owners.max()) < features.shape[0]
        and isinstance(metadata, Mapping)
        and metadata.get("schema") == CANDIDATE_SCHEMA
        and metadata.get("version") == VERSION
        and metadata.get("protocol") == "test_adapted"
        and metadata.get("uses_test_queries") is True
        and metadata.get("test_adapted") is True
        and metadata.get("formation_role") == "adaptation"
        and metadata.get("adaptation_features_consumed") is True
        and metadata.get("control_features_consumed") is False
        and metadata.get("confirmation_features_consumed") is False
        and metadata.get("control_or_confirmation_outcomes_consumed") is False
        and metadata.get("base_anchor_fields_bit_exact") is True
        and metadata.get("base_anchor_features_moved_or_lowered") is False
        and metadata.get("geometry_changed") is False
        and metadata.get("pose_valid_edge_claimed") is True
        and metadata.get("identity_truth_claimed") is False
        and metadata.get("gaussian_geometry_is_identity_truth") is False
        and metadata.get("negative_anchor_labels_created") is False
        and metadata.get("deployment_authorized") is False
        and metadata.get("controller_authorized") is False
        and int(metadata.get("added_prototype_count", -1)) == owners.numel()
        and metadata.get("prototype_features_sha256")
        == tensor_sha256(prototypes)
        and metadata.get("prototype_owner_rows_sha256") == tensor_sha256(owners)
    ):
        raise ValueError("V21 transductive candidate contract is invalid")
    inputs = metadata.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 transductive candidate input lineage is missing")
    for name in (
        "stable_map",
        "split_manifest",
        "gaussian_geometry_oracle_aggregate",
    ):
        _source_identity(inputs.get(name), label=f"candidate {name}")
    cache_sources = inputs.get("adaptation_caches")
    if not isinstance(cache_sources, list) or not cache_sources:
        raise ValueError("V21 candidate adaptation cache lineage is empty")
    cache_identities = [
        _source_identity(value, label="candidate adaptation cache")
        for value in cache_sources
    ]
    if len(set(cache_identities)) != len(cache_identities):
        raise ValueError("V21 candidate adaptation cache lineage is duplicated")
    for field, label in (
        ("gaussian_geometry_support", "candidate Gaussian support"),
        ("pose_recovery_oracle_shards", "candidate oracle shard"),
    ):
        values = inputs.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"V21 {label} lineage is empty")
        identities = [_source_identity(value, label=label) for value in values]
        if len(set(identities)) != len(identities):
            raise ValueError(f"V21 {label} lineage is duplicated")
    calibration_source = inputs.get("provisional_identity_calibration")
    if calibration_source is not None:
        _source_identity(calibration_source, label="candidate provisional calibration")
    validate_baseline_contract(metadata.get("baseline_contract"))
    budgets = metadata.get("budgets")
    if not isinstance(budgets, Mapping) or not (
        int(budgets.get("maximum_bundle_size", 0)) >= 1
        and int(budgets.get("maximum_source_queries", 0)) >= 1
        and int(budgets.get("maximum_total_prototypes", 0)) >= 1
        and int(budgets.get("maximum_prototypes_per_anchor", 0)) >= 1
        and isinstance(budgets.get("require_provisional_edge"), bool)
    ):
        raise ValueError("V21 transductive candidate budgets are invalid")
    margin = budgets.get("require_one_assignment_translation_below_cm")
    if margin is not None and not 0.0 < float(margin) <= 5.0:
        raise ValueError("V21 transductive one-assignment margin is invalid")
    if budgets["require_provisional_edge"] and calibration_source is None:
        raise ValueError("V21 candidate provisional-edge gate lacks calibration")
    actions = metadata.get("selected_actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("V21 transductive candidate action registry is empty")
    seen_indices = []
    seen_queries = set()
    for action in actions:
        indices = _as_cpu(action.get("prototype_indices"), dtype=torch.long).reshape(-1)
        query_rows = _as_cpu(action.get("query_rows"), dtype=torch.long).reshape(-1)
        action_owners = _as_cpu(action.get("owner_anchor_rows"), dtype=torch.long).reshape(-1)
        one_translation = float(
            action.get("one_assignment_translation_error_cm", math.nan)
        )
        one_rotation = float(
            action.get("one_assignment_rotation_error_deg", math.nan)
        )
        bundle_translation = float(
            action.get("bundle_translation_error_cm", math.nan)
        )
        bundle_rotation = float(
            action.get("bundle_rotation_error_deg", math.nan)
        )
        if not (
            indices.shape == query_rows.shape == action_owners.shape
            and indices.numel() > 0
            and indices.numel() <= int(budgets["maximum_bundle_size"])
            and int(indices.min()) >= 0
            and int(indices.max()) < owners.numel()
            and torch.unique(indices).numel() == indices.numel()
            and int(query_rows.min()) >= 0
            and torch.unique(query_rows).numel() == query_rows.numel()
            and torch.equal(owners[indices], action_owners)
            and action.get("prototype_features_sha256")
            == tensor_sha256(prototypes[indices])
            and action.get("prototype_owner_rows_sha256")
            == tensor_sha256(action_owners)
            and action.get("bundle_standard_r5_success") is True
            and action.get("exact_poselib_replayed_during_materialization") is True
            and action.get("pose_valid_edge_claimed") is True
            and action.get("identity_truth_claimed") is False
            and all(
                math.isfinite(value) and value >= 0.0
                for value in (
                    one_translation,
                    one_rotation,
                    bundle_translation,
                    bundle_rotation,
                )
            )
            and one_translation < R5_TRANSLATION_CM
            and one_rotation < R5_ROTATION_DEG
            and bundle_translation < R5_TRANSLATION_CM
            and bundle_rotation < R5_ROTATION_DEG
            and (
                margin is None or one_translation < float(margin)
            )
        ):
            raise ValueError("V21 transductive candidate action row is invalid")
        query_index = int(action.get("query_index", -1))
        provisional_mask = torch.as_tensor(
            action.get("provisional_edge_mask")
        ).bool().reshape(-1)
        if (
            query_index < 0
            or query_index in seen_queries
            or provisional_mask.shape != indices.shape
            or bool(action.get("all_edges_provisional"))
            != bool(provisional_mask.all())
            or (
                budgets["require_provisional_edge"]
                and not bool(provisional_mask.all())
            )
        ):
            raise ValueError("V21 transductive candidate provisional action differs")
        seen_queries.add(query_index)
        seen_indices.extend(indices.tolist())
    if seen_indices != list(range(owners.numel())):
        raise ValueError("V21 transductive prototype registry is incomplete")
    if not (
        len(actions) == int(metadata.get("selected_source_query_count", -1))
        and len(actions) <= int(budgets["maximum_source_queries"])
        and owners.numel() <= int(budgets["maximum_total_prototypes"])
        and int(metadata.get("source_query_count", 0)) >= len(actions)
        and int(metadata.get("eligible_bundle_count", -1)) >= len(actions)
    ):
        raise ValueError("V21 transductive candidate counts differ")
    owner_counts = torch.bincount(owners, minlength=features.shape[0])
    if int(owner_counts.max()) > int(budgets["maximum_prototypes_per_anchor"]):
        raise ValueError("V21 transductive per-Anchor prototype budget differs")
    return dict(metadata)


def evaluate_cached_record(
    *,
    record: Mapping,
    anchor_features: torch.Tensor,
    anchor_xyz: torch.Tensor,
    extra_prototypes: torch.Tensor,
    prototype_owner_rows: torch.Tensor,
    baseline_contract: Mapping,
    matcher_chunk_size: int = 8192,
    device: str | torch.device = "cpu",
    anchor_features_normalized: bool = False,
    prototype_activation_threshold: float | None = None,
    solver: Callable = solve_absolute_pose,
) -> dict:
    """Evaluate one immutable cached query with exact owner-prototype Top1."""

    contract = validate_baseline_contract(baseline_contract)
    base = torch.as_tensor(anchor_features, device=device).float()
    xyz = _as_cpu(anchor_xyz, dtype=torch.float32)
    prototypes = torch.as_tensor(extra_prototypes, device=device).float()
    owners = torch.as_tensor(prototype_owner_rows, device=device).long().reshape(-1)
    descriptors = torch.as_tensor(record["descriptors"], device=device).float()
    if (
        base.ndim != 2
        or descriptors.ndim != 2
        or descriptors.shape[1] != base.shape[1]
        or prototypes.shape != (owners.numel(), base.shape[1])
        or xyz.shape != (base.shape[0], 3)
        or int(matcher_chunk_size) < 1
    ):
        raise ValueError("V21 cached evaluation descriptor registries differ")
    if not anchor_features_normalized:
        # SparseLocalizer performs one normalization while loading the identity
        # map and one historical matcher normalization immediately afterwards.
        base = F.normalize(F.normalize(base, dim=1), dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    matches = global_owner_prototype_top1(
        descriptors,
        base,
        prototypes,
        owners,
        chunk_size=int(matcher_chunk_size),
        anchor_descriptors_normalized=True,
        prototype_activation_threshold=prototype_activation_threshold,
    )
    candidate_rows = matches.anchor_indices.detach().cpu().long()
    candidate_scores = matches.scores.detach().cpu().float()
    baseline_rows = _as_cpu(record["winner_anchor_rows"], dtype=torch.long).reshape(-1)
    baseline_scores = _as_cpu(record["winner_scores"], dtype=torch.float32).reshape(-1)
    prototype_scores = F.normalize(descriptors, dim=1) @ prototypes.T
    best_prototype_scores, best_prototype_indices = prototype_scores.max(dim=1)
    expected_prototype_winner = best_prototype_scores.detach().cpu() > baseline_scores
    if prototype_activation_threshold is not None:
        expected_prototype_winner &= best_prototype_scores.detach().cpu() >= float(
            prototype_activation_threshold
        )
    expected_rows = baseline_rows.clone()
    expected_rows[expected_prototype_winner] = owners[
        best_prototype_indices[expected_prototype_winner.to(device)]
    ].detach().cpu()
    expected_scores = baseline_scores.clone()
    expected_scores[expected_prototype_winner] = best_prototype_scores.detach().cpu()[
        expected_prototype_winner
    ]
    if not torch.equal(candidate_rows, expected_rows) or not torch.allclose(
        candidate_scores, expected_scores, rtol=1e-6, atol=5e-7
    ):
        raise ValueError(
            "V21 owner-prototype result differs from cached exact base Top1 semantics"
        )
    physical_keypoints = _as_cpu(record["keypoints"], dtype=torch.float32) + float(
        contract["pixel_center_offset"]
    )
    baseline = replay_pose_with_contract(
        keypoints=physical_keypoints,
        anchor_rows=baseline_rows,
        anchor_xyz=xyz,
        intrinsic=record["intrinsics"],
        ground_truth_w2c=record["pose_w2c"],
        baseline_contract=contract,
        solver=solver,
    )
    _assert_cached_baseline(record, baseline)
    candidate = replay_pose_with_contract(
        keypoints=physical_keypoints,
        anchor_rows=candidate_rows,
        anchor_xyz=xyz,
        intrinsic=record["intrinsics"],
        ground_truth_w2c=record["pose_w2c"],
        baseline_contract=contract,
        solver=solver,
    )
    winner_flip = candidate_rows != baseline_rows
    prototype_winner = expected_prototype_winner
    r5_gain = bool(not baseline["r5_success"] and candidate["r5_success"])
    r5_loss = bool(baseline["r5_success"] and not candidate["r5_success"])
    return {
        "query_index": int(record["query_index"]),
        "image_name": str(record["image_name"]),
        "sequence_id": str(record["sequence_id"]),
        "block_id": str(record["block_id"]),
        "role": str(record["role"]),
        "source_record_sha256": str(record["source_record_sha256"]),
        "baseline": baseline,
        "candidate": candidate,
        "paired_delta_translation_error_cm": float(
            candidate["translation_error_cm"] - baseline["translation_error_cm"]
        ),
        "paired_delta_rotation_error_deg": float(
            candidate["rotation_error_deg"] - baseline["rotation_error_deg"]
        ),
        "paired_delta_task_error": float(candidate["task_error"] - baseline["task_error"]),
        "paired_delta_r5": int(candidate["r5_success"]) - int(baseline["r5_success"]),
        "r5_gain": r5_gain,
        "r5_loss": r5_loss,
        "catastrophe": r5_loss,
        "catastrophe_definition": "baseline_r5_success_to_candidate_r5_failure",
        "r5_flip": bool(r5_gain or r5_loss),
        "winner_flip_count": int(winner_flip.sum()),
        "winner_flip_fraction": float(winner_flip.float().mean()),
        "winner_flip_query_rows": torch.nonzero(winner_flip, as_tuple=False).reshape(-1),
        "baseline_winner_anchor_rows_at_flips": baseline_rows[winner_flip],
        "candidate_winner_anchor_rows_at_flips": candidate_rows[winner_flip],
        "prototype_winner_count": int(prototype_winner.sum()),
        "candidate_winner_anchor_rows": candidate_rows,
        "candidate_winner_scores": candidate_scores,
        "candidate_formation_feedback_consumed": False,
    }


def _quantiles(values: Sequence[float]) -> dict:
    if not values:
        return {}
    tensor = torch.tensor(list(values), dtype=torch.float64)
    levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    result = torch.quantile(tensor, levels)
    return {
        name: float(value)
        for name, value in zip(("min", "p25", "median", "p75", "max"), result)
    }


def summarize_cached_evaluation(records: Sequence[Mapping]) -> dict:
    """Summarize paired R5 and continuous-error effects without feedback."""

    total = len(records)
    baseline_success = sum(bool(record["baseline"]["r5_success"]) for record in records)
    candidate_success = sum(bool(record["candidate"]["r5_success"]) for record in records)
    gains = sum(bool(record["r5_gain"]) for record in records)
    losses = sum(bool(record["r5_loss"]) for record in records)
    catastrophes = sum(bool(record["catastrophe"]) for record in records)
    winner_flips = sum(int(record["winner_flip_count"]) for record in records)
    return {
        "query_count": total,
        "baseline_r5_success_count": baseline_success,
        "candidate_r5_success_count": candidate_success,
        "baseline_r5_rate": float(baseline_success / total) if total else 0.0,
        "candidate_r5_rate": float(candidate_success / total) if total else 0.0,
        "paired_r5_gain_count": gains,
        "paired_r5_loss_count": losses,
        "paired_r5_net_count": gains - losses,
        "paired_r5_rate_delta": float((candidate_success - baseline_success) / total)
        if total
        else 0.0,
        "catastrophe_count": catastrophes,
        "catastrophe_definition": "baseline_r5_success_to_candidate_r5_failure",
        "query_with_winner_flip_count": sum(
            int(record["winner_flip_count"]) > 0 for record in records
        ),
        "winner_flip_count_total": winner_flips,
        "paired_delta_translation_error_cm": _quantiles(
            [float(record["paired_delta_translation_error_cm"]) for record in records]
        ),
        "paired_delta_rotation_error_deg": _quantiles(
            [float(record["paired_delta_rotation_error_deg"]) for record in records]
        ),
        "paired_delta_task_error": _quantiles(
            [float(record["paired_delta_task_error"]) for record in records]
        ),
    }


def build_cached_evaluation_payload(
    *,
    stable_map: Mapping,
    candidate_map: Mapping,
    cache_payloads: Sequence[Mapping],
    stable_map_source: Mapping,
    candidate_map_source: Mapping,
    cache_sources: Sequence[Mapping],
    matcher_chunk_size: int = 8192,
    device: str | torch.device = "cpu",
    solver: Callable = solve_absolute_pose,
) -> dict:
    """Evaluate one complete role while keeping it outside candidate formation."""

    features, xyz = _validate_stable_map(stable_map)
    candidate_metadata = validate_candidate_map(candidate_map, stable_map=stable_map)
    _same_source(
        candidate_metadata.get("inputs", {}).get("stable_map"),
        stable_map_source,
        label="candidate stable map",
    )
    ordered_payloads, cache_records, baseline_contract = validate_complete_cache_payloads(
        cache_payloads
    )
    role = str(ordered_payloads[0]["role"])
    if len(cache_sources) != len(ordered_payloads):
        raise ValueError("V21 evaluation cache source registry differs")
    for payload in ordered_payloads:
        _same_source(
            payload.get("inputs", {}).get("stable_map"),
            stable_map_source,
            label="evaluation cache stable map",
        )
        _same_source(
            payload.get("inputs", {}).get("split_manifest"),
            candidate_metadata.get("inputs", {}).get("split_manifest"),
            label="evaluation split manifest",
        )
        if (
            payload.get("preprocessing_config_sha256")
            != candidate_metadata.get("preprocessing_config_sha256")
        ):
            raise ValueError("V21 candidate/evaluation preprocessing contracts differ")
    if candidate_metadata.get("baseline_contract") != baseline_contract:
        raise ValueError("V21 candidate/evaluation baseline contracts differ")
    if role in {"control", "confirmation"}:
        formation_sources = {
            (str(value["path"]), str(value["sha256"]))
            for value in candidate_metadata.get("inputs", {}).get("adaptation_caches", ())
        }
        evaluation_sources = {
            (str(value["path"]), str(value["sha256"])) for value in cache_sources
        }
        if formation_sources & evaluation_sources:
            raise ValueError("V21 held-out evaluation cache was consumed by candidate formation")
    device_features = features.to(device=device, dtype=torch.float32)
    normalized_features = F.normalize(
        F.normalize(device_features, dim=1), dim=1
    )
    prototypes = torch.as_tensor(
        candidate_map[PROTOTYPE_FEATURE_FIELD], device=device
    ).float()
    owners = torch.as_tensor(
        candidate_map[PROTOTYPE_OWNER_FIELD], device=device
    ).long()
    records = [
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
            solver=solver,
        )
        for record in cache_records
    ]
    summary = summarize_cached_evaluation(records)
    return {
        "schema": EVALUATION_SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "evaluation_role": role,
        "matching_semantics": "global_owner_prototype_top1",
        "pose_solver_semantics": "single_standard_poselib_absolute_pose",
        "standard_r5_definition_inherited": True,
        "candidate_formation_feedback_consumed": False,
        "heldout_outcomes_feed_candidate": False,
        "candidate_map_mutated": False,
        "deployment_authorized": False,
        "catastrophe_definition": "baseline_r5_success_to_candidate_r5_failure",
        "inputs": {
            "stable_map": dict(stable_map_source),
            "candidate_map": dict(candidate_map_source),
            "frontend_caches": [dict(value) for value in cache_sources],
        },
        "candidate_formation_role": "adaptation",
        "candidate_source_query_indices": torch.tensor(
            [
                int(action["query_index"])
                for action in candidate_metadata["selected_actions"]
            ],
            dtype=torch.long,
        ),
        "evaluation_query_count": len(records),
        "baseline_contract": baseline_contract,
        "matcher_chunk_size": int(matcher_chunk_size),
        "records": records,
        "summary": summary,
    }


def validate_cached_evaluation_payload(payload: Mapping) -> None:
    records = payload.get("records")
    if not (
        payload.get("schema") == EVALUATION_SCHEMA
        and payload.get("version") == VERSION
        and payload.get("protocol") == "test_adapted"
        and payload.get("uses_test_queries") is True
        and payload.get("test_adapted") is True
        and payload.get("evaluation_role") in {"adaptation", "control", "confirmation"}
        and payload.get("matching_semantics") == "global_owner_prototype_top1"
        and payload.get("standard_r5_definition_inherited") is True
        and payload.get("candidate_formation_feedback_consumed") is False
        and payload.get("heldout_outcomes_feed_candidate") is False
        and payload.get("candidate_map_mutated") is False
        and payload.get("deployment_authorized") is False
        and isinstance(records, list)
        and int(payload.get("evaluation_query_count", -1)) == len(records)
        and payload.get("summary") == summarize_cached_evaluation(records)
    ):
        raise ValueError("V21 cached transductive evaluation contract is invalid")
    validate_baseline_contract(payload.get("baseline_contract"))
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 cached evaluation input lineage is missing")
    _source_identity(inputs.get("stable_map"), label="evaluation stable map")
    _source_identity(inputs.get("candidate_map"), label="evaluation candidate map")
    cache_sources = inputs.get("frontend_caches")
    if not isinstance(cache_sources, list) or not cache_sources:
        raise ValueError("V21 cached evaluation frontend lineage is empty")
    cache_identities = [
        _source_identity(value, label="evaluation frontend cache")
        for value in cache_sources
    ]
    if len(set(cache_identities)) != len(cache_identities):
        raise ValueError("V21 cached evaluation frontend lineage is duplicated")
    source_queries = torch.as_tensor(
        payload.get("candidate_source_query_indices")
    ).long().reshape(-1)
    if source_queries.numel() == 0 or torch.unique(source_queries).numel() != source_queries.numel():
        raise ValueError("V21 cached evaluation candidate source registry is invalid")
    query_indices = set()
    for record in records:
        baseline = record.get("baseline")
        candidate = record.get("candidate")
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise ValueError("V21 cached evaluation pose outcome is missing")
        for outcome in (baseline, candidate):
            numeric = torch.tensor(
                [
                    float(outcome.get("translation_error_cm", math.nan)),
                    float(outcome.get("rotation_error_deg", math.nan)),
                    float(outcome.get("task_error", math.nan)),
                ],
                dtype=torch.float64,
            )
            pose = torch.as_tensor(outcome.get("pose_w2c"))
            inliers = torch.as_tensor(outcome.get("inlier_query_rows")).long().reshape(-1)
            expected_task = task_error(float(numeric[0]), float(numeric[1]))
            expected_r5 = bool(
                float(numeric[0]) < R5_TRANSLATION_CM
                and float(numeric[1]) < R5_ROTATION_DEG
            )
            if not (
                bool(torch.isfinite(numeric).all())
                and bool((numeric[:2] >= 0).all())
                and pose.shape == (4, 4)
                and bool(torch.isfinite(pose).all())
                and math.isclose(
                    float(outcome["task_error"]),
                    expected_task,
                    rel_tol=1e-7,
                    abs_tol=1e-8,
                )
                and outcome.get("r5_success") is expected_r5
                and int(outcome.get("inlier_count", -1)) == inliers.numel()
                and torch.unique(inliers).numel() == inliers.numel()
            ):
                raise ValueError("V21 cached evaluation pose outcome is invalid")
        baseline_success = bool(baseline["r5_success"])
        candidate_success = bool(candidate["r5_success"])
        gain = not baseline_success and candidate_success
        loss = baseline_success and not candidate_success
        candidate_rows = torch.as_tensor(
            record.get("candidate_winner_anchor_rows")
        ).long().reshape(-1)
        candidate_scores = torch.as_tensor(
            record.get("candidate_winner_scores")
        ).float().reshape(-1)
        flip_rows = torch.as_tensor(
            record.get("winner_flip_query_rows")
        ).long().reshape(-1)
        baseline_flip_anchors = torch.as_tensor(
            record.get("baseline_winner_anchor_rows_at_flips")
        ).long().reshape(-1)
        candidate_flip_anchors = torch.as_tensor(
            record.get("candidate_winner_anchor_rows_at_flips")
        ).long().reshape(-1)
        count = candidate_rows.numel()
        expected_deltas = (
            float(candidate["translation_error_cm"])
            - float(baseline["translation_error_cm"]),
            float(candidate["rotation_error_deg"])
            - float(baseline["rotation_error_deg"]),
            float(candidate["task_error"]) - float(baseline["task_error"]),
        )
        observed_deltas = (
            float(record.get("paired_delta_translation_error_cm", math.nan)),
            float(record.get("paired_delta_rotation_error_deg", math.nan)),
            float(record.get("paired_delta_task_error", math.nan)),
        )
        query_index = int(record.get("query_index", -1))
        if not (
            record.get("role") == payload["evaluation_role"]
            and record.get("candidate_formation_feedback_consumed") is False
            and query_index >= 0
            and query_index not in query_indices
            and SHA256_PATTERN.fullmatch(str(record.get("source_record_sha256", "")))
            is not None
            and record.get("r5_gain") is gain
            and record.get("r5_loss") is loss
            and record.get("catastrophe") is loss
            and record.get("r5_flip") is (gain or loss)
            and int(record.get("paired_delta_r5", 99))
            == int(candidate_success) - int(baseline_success)
            and all(
                math.isclose(observed, expected, rel_tol=1e-7, abs_tol=1e-8)
                for observed, expected in zip(observed_deltas, expected_deltas)
            )
            and candidate_scores.shape == candidate_rows.shape
            and bool(torch.isfinite(candidate_scores).all())
            and flip_rows.shape
            == baseline_flip_anchors.shape
            == candidate_flip_anchors.shape
            and int(record.get("winner_flip_count", -1)) == flip_rows.numel()
            and torch.unique(flip_rows).numel() == flip_rows.numel()
            and (
                flip_rows.numel() == 0
                or (
                    int(flip_rows.min()) >= 0
                    and int(flip_rows.max()) < count
                    and bool((baseline_flip_anchors != candidate_flip_anchors).all())
                    and torch.equal(candidate_rows[flip_rows], candidate_flip_anchors)
                )
            )
            and math.isclose(
                float(record.get("winner_flip_fraction", math.nan)),
                float(flip_rows.numel() / count) if count else 0.0,
                rel_tol=1e-7,
                abs_tol=1e-8,
            )
            and 0 <= int(record.get("prototype_winner_count", -1)) <= count
        ):
            raise ValueError("V21 cached transductive evaluation record is invalid")
        query_indices.add(query_index)


def atomic_torch_save_fresh(
    payload: Mapping,
    output: str | Path,
    *,
    validator: Callable[[Mapping], None],
) -> Path:
    """Save through a reloaded temporary and expose via a fresh hardlink."""

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(dict(payload), temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        validator(reloaded)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"V21 output appeared while running: {output}") from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return output
