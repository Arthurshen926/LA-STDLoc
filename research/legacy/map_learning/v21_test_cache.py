"""Contracts for leakage-explicit V21 real-test frontend/baseline caches.

The cache is intentionally a *test-adapted* artifact.  Only the adaptation
role may be consumed by a training job; control and confirmation shards are
held out even though all three roles contain the same immutable observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import os
from pathlib import Path
import re
import uuid

import torch

from common.hashing import canonical_json
from map_learning.v21_test_protocol import (
    PRIMARY_ROLES,
    SCHEMA as SPLIT_SCHEMA,
    VERSION as SPLIT_VERSION,
    validate_test_protocol_manifest,
)


CACHE_SCHEMA = "lafgs_v21_test_frontend_baseline_cache"
CACHE_VERSION = 1
ALLOWED_ROLES = frozenset(PRIMARY_ROLES)
MANIFEST_ROLES = frozenset({*ALLOWED_ROLES, "embargo"})
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SHARD_ASSIGNMENT = "ordered_role_query_registry_modulo_shard_count"
VALID_MASK_SEMANTICS = {
    "dataset_mask": "object_and_sky_and_distortion_channel_conjunction",
    "resize": "nearest_to_colmap_camera_hw",
    "before_detection": "invalid_rgb_pixels_are_zeroed",
    "after_detection": "rounded_keypoint_lookup_filters_invalid_rows",
    "cached_rows": "post_mask_native_superpoint_rows_only",
    "native_valid_keypoint_mask": "all_true_for_every_cached_row",
    "cached_raster": "omitted_to_avoid_per_query_h_by_w_duplication",
    "derived_mask_binding": "dtype_shape_and_content_sha256",
}


def sha256_json(value: Mapping) -> str:
    """Hash a JSON-safe mapping with the repository canonicalization."""

    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes deterministically."""

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    header = canonical_json(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    ).encode("ascii")
    digest = hashlib.sha256(header)
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def pose_w2c_sha256(value: torch.Tensor) -> str:
    """Match the V21 split manifest's exact little-endian FP32 pose hash."""

    import numpy as np

    pose = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    if pose.shape != (4, 4):
        raise ValueError("pose_w2c hash input must have shape [4,4]")
    digest = hashlib.sha256(b"pose_w2c:<f4:4x4:")
    digest.update(pose.tobytes())
    return digest.hexdigest()


def ordered_test_camera_sha256(records: Sequence[Mapping]) -> str:
    """Reproduce the split manifest's ordered test-camera registry digest."""

    identities = [
        {
            "query_index": int(record["query_index"]),
            "image_name": str(record["image_name"]),
            "image_path": str(record["image_path"]),
            "image_sha256": record.get("image_sha256"),
            "pose_w2c_sha256": str(record["pose_w2c_sha256"]),
        }
        for record in sorted(
            records,
            key=lambda row: (int(row["query_index"]), str(row["image_name"])),
        )
    ]
    return hashlib.sha256(
        canonical_json({"value": identities}).encode("ascii")
    ).hexdigest()


def _require_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def _nonnegative_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a non-negative integer") from error
    if result < 0 or result != value:
        raise ValueError(f"{label} must be a non-negative integer")
    return result


def _is_embargoed(record: Mapping) -> bool:
    values = [
        record.get("embargo", False),
        record.get("embargoed", False),
        record.get("is_embargoed", False),
    ]
    if any(value not in {True, False, None} for value in values):
        raise ValueError("V21 record embargo flags must be boolean")
    return any(value is True for value in values)


def validate_split_manifest(payload: Mapping, *, role: str) -> list[dict]:
    """Validate and return one role in its canonical query-index order.

    The whole manifest is validated before role selection.  This prevents an
    invalid held-out record or duplicate query identity from being hidden by a
    shard/role filter.
    """

    if role not in ALLOWED_ROLES:
        raise ValueError(f"V21 cache role must be one of {sorted(ALLOWED_ROLES)}")
    if (
        payload.get("schema") != SPLIT_SCHEMA
        or payload.get("version") != SPLIT_VERSION
        or payload.get("protocol") != "test_adapted"
        or payload.get("ordering_policy")
        != "forward_adaptation_control_confirmation"
        or payload.get("uses_test_queries") is not True
        or payload.get("test_adapted") is not True
    ):
        raise ValueError("unsupported or non-test-adapted V21 split manifest")
    validate_test_protocol_manifest(dict(payload))
    _require_sha256(
        payload.get("stable_map_sha256"), label="V21 stable-map SHA256"
    )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("V21 split manifest records must be a non-empty list")

    normalized = []
    query_indices: set[int] = set()
    image_names: set[str] = set()
    paths: set[str] = set()
    required = {
        "query_index",
        "image_name",
        "image_path",
        "image_sha256",
        "sequence_id",
        "frame_index",
        "pose_w2c_sha256",
        "block_id",
        "role",
    }
    for offset, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"V21 split record {offset} must be a mapping")
        missing = required - set(raw)
        if missing:
            raise ValueError(
                f"V21 split record {offset} misses fields: {sorted(missing)}"
            )
        record = dict(raw)
        query_index = _nonnegative_integer(
            record["query_index"], label="V21 query_index"
        )
        frame_index = _nonnegative_integer(
            record["frame_index"], label="V21 frame_index"
        )
        block_id = str(record["block_id"])
        record_role = str(record["role"])
        image_name = str(record["image_name"])
        image_path = str(record["image_path"])
        sequence_id = str(record["sequence_id"])
        if record_role not in MANIFEST_ROLES:
            raise ValueError("V21 split record has an unsupported role")
        if not image_name or not image_path or not sequence_id or not block_id:
            raise ValueError("V21 image/sequence identities must be non-empty")
        if record["image_sha256"] is not None:
            _require_sha256(record["image_sha256"], label="V21 image SHA256")
        _require_sha256(record["pose_w2c_sha256"], label="V21 pose SHA256")
        if query_index in query_indices:
            raise ValueError("V21 split query indices must be globally unique")
        if image_name in image_names or image_path in paths:
            raise ValueError("V21 split image identities must be globally unique")
        query_indices.add(query_index)
        image_names.add(image_name)
        paths.add(image_path)
        record.update(
            {
                "query_index": query_index,
                "frame_index": frame_index,
                "block_id": block_id,
                "role": record_role,
                "image_name": image_name,
                "image_path": image_path,
                "sequence_id": sequence_id,
            }
        )
        normalized.append(record)

    dataset_registry = payload.get("dataset_registry")
    if not isinstance(dataset_registry, Mapping):
        raise ValueError("V21 split dataset registry is missing")
    expected_camera_sha = ordered_test_camera_sha256(normalized)
    if dataset_registry.get("ordered_test_camera_sha256") != expected_camera_sha:
        raise ValueError("V21 ordered test-camera registry SHA256 differs")

    declared_roles = payload.get("roles")
    if isinstance(declared_roles, Mapping):
        role_names = set(declared_roles)
    elif isinstance(declared_roles, Sequence) and not isinstance(
        declared_roles, (str, bytes)
    ):
        role_names = {str(value) for value in declared_roles}
    else:
        raise ValueError("V21 split roles registry is missing")
    if role_names != set(MANIFEST_ROLES):
        raise ValueError("V21 split roles registry differs from its records")

    selected = [record for record in normalized if record["role"] == role]
    if not selected:
        raise ValueError(f"V21 split contains no {role} records")
    if any(record["image_sha256"] is None for record in selected):
        raise ValueError(f"V21 {role} images must be content-addressable")
    if any(_is_embargoed(record) for record in selected):
        raise ValueError(f"V21 {role} role is embargoed and cannot be materialized")
    return sorted(selected, key=lambda row: (row["query_index"], row["image_name"]))


def build_shard_registry(
    records: Sequence[Mapping],
    *,
    role: str,
    shard_count: int,
    split_manifest_sha256: str,
) -> dict:
    """Build the complete deterministic registry carried by every shard."""

    if role not in ALLOWED_ROLES:
        raise ValueError("V21 shard registry role is invalid")
    shard_count = int(shard_count)
    if shard_count <= 0:
        raise ValueError("V21 shard count must be positive")
    manifest_sha = _require_sha256(
        split_manifest_sha256, label="V21 split manifest SHA256"
    )
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda row: (int(row["query_index"]), str(row["image_name"])),
    )
    if any(str(record.get("role")) != role for record in ordered):
        raise ValueError("V21 shard registry mixes roles")
    rows = []
    for ordinal, record in enumerate(ordered):
        rows.append(
            {
                "ordinal": ordinal,
                "shard_index": ordinal % shard_count,
                "query_index": int(record["query_index"]),
                "image_name": str(record["image_name"]),
                "image_sha256": _require_sha256(
                    record["image_sha256"], label="V21 registry image SHA256"
                ),
                "source_record_sha256": sha256_json(record),
            }
        )
    core = {
        "schema": "lafgs_v21_test_cache_shard_registry",
        "version": 1,
        "role": role,
        "split_manifest_sha256": manifest_sha,
        "assignment": SHARD_ASSIGNMENT,
        "shard_count": shard_count,
        "role_query_count": len(rows),
        "rows": rows,
    }
    return {**core, "registry_sha256": sha256_json(core)}


def validate_shard_registry(registry: Mapping) -> None:
    """Validate a complete role registry and deterministic shard assignment."""

    core = dict(registry)
    registry_sha = core.pop("registry_sha256", None)
    role = str(registry.get("role"))
    shard_count = int(registry.get("shard_count", 0))
    rows = registry.get("rows")
    if (
        registry.get("schema") != "lafgs_v21_test_cache_shard_registry"
        or registry.get("version") != 1
        or registry.get("assignment") != SHARD_ASSIGNMENT
        or role not in ALLOWED_ROLES
        or shard_count <= 0
        or not isinstance(rows, list)
        or not rows
        or int(registry.get("role_query_count", -1)) != len(rows)
        or registry_sha != sha256_json(core)
    ):
        raise ValueError("V21 cache shard registry is invalid")
    _require_sha256(
        registry.get("split_manifest_sha256"),
        label="V21 registry split-manifest SHA256",
    )
    ordinals = []
    queries = set()
    images = set()
    source_records = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("V21 shard registry row must be a mapping")
        ordinal = _nonnegative_integer(row.get("ordinal"), label="V21 ordinal")
        query = _nonnegative_integer(
            row.get("query_index"), label="V21 registry query_index"
        )
        image_name = str(row.get("image_name", ""))
        source_record = _require_sha256(
            row.get("source_record_sha256"),
            label="V21 source-record SHA256",
        )
        _require_sha256(row.get("image_sha256"), label="V21 image SHA256")
        if (
            not image_name
            or int(row.get("shard_index", -1)) != ordinal % shard_count
            or query in queries
            or image_name in images
            or source_record in source_records
        ):
            raise ValueError("V21 shard registry row is duplicated or misassigned")
        ordinals.append(ordinal)
        queries.add(query)
        images.add(image_name)
        source_records.add(source_record)
    if sorted(ordinals) != list(range(len(rows))):
        raise ValueError("V21 shard registry ordinals are incomplete")


def records_for_shard(registry: Mapping, *, shard_index: int) -> list[dict]:
    validate_shard_registry(registry)
    shard_count = int(registry.get("shard_count", 0))
    shard_index = int(shard_index)
    if not 0 <= shard_index < shard_count:
        raise ValueError("V21 shard index is outside [0, shard_count)")
    return [
        dict(row)
        for row in registry.get("rows", [])
        if int(row["shard_index"]) == shard_index
    ]


def training_consumer_policy(role: str) -> dict:
    if role not in ALLOWED_ROLES:
        raise ValueError("V21 consumer policy role is invalid")
    allowed = role == "adaptation"
    return {
        "training_consumer_allowed": allowed,
        "training_consumers_allowed": allowed,
        "held_out_from_training": not allowed,
        "allowed_training_consumer": (
            "v21_test_adapter_training_only" if allowed else None
        ),
        "control_or_confirmation_forbidden_for_training": not allowed,
        "may_update_stable_map": False,
        "may_update_frontend_weights": False,
        "may_select_hyperparameters": role == "control",
        "may_confirm_once": role == "confirmation",
    }


def _validate_source_record(source: object, *, label: str) -> dict:
    if not isinstance(source, Mapping):
        raise ValueError(f"V21 {label} source record is missing")
    path = str(source.get("path", ""))
    digest = _require_sha256(source.get("sha256"), label=f"V21 {label} SHA256")
    size = _nonnegative_integer(
        source.get("size_bytes"), label=f"V21 {label} source size"
    )
    if not path or size <= 0:
        raise ValueError(f"V21 {label} source record is empty")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _as_cpu_tensor(value: object, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device="cpu").contiguous()


def build_query_record(
    *,
    split_record: Mapping,
    keypoints: torch.Tensor,
    descriptors: torch.Tensor,
    scores: torch.Tensor,
    image_hw: Sequence[int],
    valid_mask: torch.Tensor | None,
    intrinsics: torch.Tensor,
    pose_w2c: torch.Tensor,
    winner_anchor_rows: torch.Tensor,
    winner_anchor_ids: torch.Tensor,
    winner_scores: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    baseline_inliers: torch.Tensor,
    rotation_error_deg: float,
    translation_error_cm: float,
    task_error: float,
) -> dict:
    """Build one fully aligned, CPU-resident cache row."""

    xy = _as_cpu_tensor(keypoints, dtype=torch.float32).reshape(-1, 2)
    descriptor = _as_cpu_tensor(descriptors, dtype=torch.float32)
    confidence = _as_cpu_tensor(scores, dtype=torch.float32).reshape(-1)
    winners = _as_cpu_tensor(winner_anchor_rows, dtype=torch.long).reshape(-1)
    winner_ids = _as_cpu_tensor(winner_anchor_ids, dtype=torch.long).reshape(-1)
    winner_values = _as_cpu_tensor(winner_scores, dtype=torch.float32).reshape(-1)
    count = xy.shape[0]
    if (
        descriptor.ndim != 2
        or descriptor.shape[0] != count
        or confidence.numel() != count
        or winners.numel() != count
        or winner_ids.numel() != count
        or winner_values.numel() != count
    ):
        raise ValueError("V21 native frontend and winner rows do not align")
    if not bool(
        torch.isfinite(xy).all()
        and torch.isfinite(descriptor).all()
        and torch.isfinite(confidence).all()
        and torch.isfinite(winner_values).all()
    ):
        raise ValueError("V21 frontend cache contains non-finite values")
    height, width = (int(image_hw[0]), int(image_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError("V21 image dimensions must be positive")
    cached_mask = None
    if valid_mask is not None:
        cached_mask = _as_cpu_tensor(valid_mask, dtype=torch.bool).squeeze()
        if cached_mask.shape != (height, width):
            raise ValueError("V21 valid mask does not align with image_hw")
    mask_digest = tensor_sha256(cached_mask) if cached_mask is not None else None
    mask_valid_pixel_count = (
        int(cached_mask.sum()) if cached_mask is not None else None
    )
    mask_valid_fraction = (
        float(cached_mask.float().mean()) if cached_mask is not None else None
    )
    gt = _as_cpu_tensor(pose_w2c, dtype=torch.float32)
    estimate = _as_cpu_tensor(baseline_pose_w2c, dtype=torch.float32)
    calibration = _as_cpu_tensor(intrinsics, dtype=torch.float32)
    inliers = _as_cpu_tensor(baseline_inliers, dtype=torch.long).reshape(-1)
    if gt.shape != (4, 4) or estimate.shape != (4, 4):
        raise ValueError("V21 GT and baseline poses must have shape [4,4]")
    if pose_w2c_sha256(gt) != split_record["pose_w2c_sha256"]:
        raise ValueError("V21 GT pose differs from the split-manifest pose hash")
    if calibration.shape != (3, 3):
        raise ValueError("V21 intrinsics must have shape [3,3]")
    if inliers.numel() and (
        int(inliers.min()) < 0
        or int(inliers.max()) >= count
        or torch.unique(inliers).numel() != inliers.numel()
    ):
        raise ValueError("V21 baseline inlier rows are invalid")
    rotation = float(rotation_error_deg)
    translation = float(translation_error_cm)
    task = float(task_error)
    if not all(torch.isfinite(torch.tensor([rotation, translation, task])).tolist()):
        raise ValueError("V21 baseline errors must be finite")
    r5 = translation < 5.0 and rotation < 5.0
    role = str(split_record["role"])
    return {
        "query_index": int(split_record["query_index"]),
        "image_name": str(split_record["image_name"]),
        "image_path": str(split_record["image_path"]),
        "image_sha256": str(split_record["image_sha256"]),
        "sequence_id": str(split_record["sequence_id"]),
        "sequence_query_index": int(split_record["sequence_query_index"]),
        "frame_index": int(split_record["frame_index"]),
        "block_id": str(split_record["block_id"]),
        "within_block_index": int(split_record["within_block_index"]),
        "role": role,
        "source_record_sha256": sha256_json(dict(split_record)),
        "keypoints": xy,
        "descriptors": descriptor,
        "scores": confidence,
        "image_hw": torch.tensor([height, width], dtype=torch.long),
        "valid_mask_present": cached_mask is not None,
        "valid_mask_raster_cached": False,
        "valid_mask_sha256": mask_digest,
        "valid_mask_valid_pixel_count": mask_valid_pixel_count,
        "valid_mask_valid_fraction": mask_valid_fraction,
        "native_valid_keypoint_mask": torch.ones(count, dtype=torch.bool),
        "valid_mask_semantics": dict(VALID_MASK_SEMANTICS),
        "intrinsics": calibration,
        "pose_w2c": gt,
        "pose_w2c_sha256": str(split_record["pose_w2c_sha256"]),
        "winner_query_rows": torch.arange(count, dtype=torch.long),
        "winner_anchor_rows": winners,
        "winner_anchor_ids": winner_ids,
        "winner_scores": winner_values,
        "baseline_pose_w2c": estimate,
        "baseline_inliers": inliers,
        "baseline_inlier_count": int(inliers.numel()),
        "baseline_rotation_error_deg": rotation,
        "baseline_translation_error_cm": translation,
        "baseline_task_error": task,
        "baseline_r5": r5,
        "training_consumer_allowed": role == "adaptation",
        "training_consumers_allowed": role == "adaptation",
    }


def validate_cache_payload(payload: Mapping) -> None:
    """Validate the cache envelope and all within-record tensor alignment."""

    role = str(payload.get("role"))
    if (
        payload.get("schema") != CACHE_SCHEMA
        or payload.get("version") != CACHE_VERSION
        or payload.get("protocol") != "test_adapted"
        or payload.get("uses_test_queries") is not True
        or payload.get("test_adapted") is not True
        or role not in ALLOWED_ROLES
    ):
        raise ValueError("unsupported V21 test frontend cache contract")
    policy = training_consumer_policy(role)
    if payload.get("consumer_policy") != policy:
        raise ValueError("V21 cache consumer policy differs from its role")
    if bool(payload.get("training_consumers_allowed")) != bool(
        policy["training_consumers_allowed"]
    ) or bool(payload.get("training_consumer_allowed")) != bool(
        policy["training_consumer_allowed"]
    ):
        raise ValueError("V21 cache training permission differs from its role")
    frontend_contract = payload.get("frontend_contract")
    if (
        not isinstance(frontend_contract, Mapping)
        or payload.get("preprocessing_config_sha256")
        != sha256_json(frontend_contract)
    ):
        raise ValueError("V21 preprocessing config SHA256 is invalid")
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V21 cache input lineage is missing")
    required_sources = {
        name: _validate_source_record(inputs.get(name), label=name)
        for name in (
            "split_manifest",
            "stable_map",
            "identity_metric",
            "frontend_weights",
            "mainline_config",
            "dataset_registry",
        )
    }
    if (
        payload.get("split_manifest_sha256")
        != required_sources["split_manifest"]["sha256"]
        or frontend_contract.get("superpoint_weights_sha256")
        != required_sources["frontend_weights"]["sha256"]
        or frontend_contract.get("mainline_config_sha256")
        != required_sources["mainline_config"]["sha256"]
        or inputs["mainline_config"].get("resolved_sha256")
        != frontend_contract.get("resolved_mainline_config_sha256")
    ):
        raise ValueError("V21 cache preprocessing/input SHA lineage differs")
    _require_sha256(
        inputs["mainline_config"].get("resolved_sha256"),
        label="V21 resolved mainline config SHA256",
    )
    optional_sources = {}
    for optional_name in ("scene_calibration", "valid_mask_source"):
        if inputs.get(optional_name) is not None:
            optional_sources[optional_name] = _validate_source_record(
                inputs[optional_name], label=optional_name
            )
    all_sources = inputs.get("all_source_files")
    if not isinstance(all_sources, list) or not all_sources:
        raise ValueError("V21 complete source-file registry is missing")
    source_by_path = {}
    for offset, source in enumerate(all_sources):
        normalized = _validate_source_record(
            source, label=f"all_source_files[{offset}]"
        )
        if normalized["path"] in source_by_path:
            raise ValueError("V21 complete source-file registry contains duplicates")
        source_by_path[normalized["path"]] = normalized
    if any(
        source_by_path.get(source["path"]) != source
        for source in required_sources.values()
    ):
        raise ValueError("V21 required input is absent from complete source registry")
    shard_count = int(payload.get("shard_count", 0))
    shard_index = int(payload.get("shard_index", -1))
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("V21 cache shard coordinates are invalid")
    registry = payload.get("shard_registry")
    if not isinstance(registry, Mapping):
        raise ValueError("V21 cache shard registry is missing")
    validate_shard_registry(registry)
    if (
        registry.get("role") != role
        or int(registry.get("shard_count", 0)) != shard_count
        or payload.get("split_manifest_sha256")
        != registry.get("split_manifest_sha256")
    ):
        raise ValueError("V21 cache shard registry is invalid")
    expected = records_for_shard(registry, shard_index=shard_index)
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("V21 cache record count differs from its shard registry")
    if int(payload.get("query_count", -1)) != len(records):
        raise ValueError("V21 cache query_count differs from records")
    if int(payload.get("role_query_count", -1)) != int(
        registry.get("role_query_count", -1)
    ):
        raise ValueError("V21 cache full role query count differs")
    anchor_count = int(payload.get("anchor_count", 0))
    descriptor_dim = int(payload.get("descriptor_dim", 0))
    if anchor_count <= 0 or descriptor_dim <= 0:
        raise ValueError("V21 cache map dimensions are invalid")
    for record, expected_row in zip(records, expected):
        if (
            int(record.get("query_index", -1)) != expected_row["query_index"]
            or str(record.get("image_name")) != expected_row["image_name"]
            or record.get("source_record_sha256")
            != expected_row["source_record_sha256"]
            or str(record.get("role")) != role
            or bool(record.get("training_consumers_allowed"))
            != (role == "adaptation")
            or bool(record.get("training_consumer_allowed"))
            != (role == "adaptation")
            or record.get("image_sha256") != expected_row["image_sha256"]
        ):
            raise ValueError("V21 cache record identity differs from shard registry")
        image_source = source_by_path.get(str(record.get("image_path", "")))
        if (
            image_source is None
            or image_source["sha256"] != record.get("image_sha256")
        ):
            raise ValueError("V21 query image is absent from complete source registry")
        keypoints = torch.as_tensor(record.get("keypoints"))
        descriptors = torch.as_tensor(record.get("descriptors"))
        scores = torch.as_tensor(record.get("scores"))
        winners = torch.as_tensor(record.get("winner_anchor_rows"))
        winner_ids = torch.as_tensor(record.get("winner_anchor_ids"))
        winner_scores = torch.as_tensor(record.get("winner_scores"))
        valid_rows = torch.as_tensor(record.get("native_valid_keypoint_mask"))
        query_rows = torch.as_tensor(record.get("winner_query_rows"))
        count = keypoints.shape[0] if keypoints.ndim == 2 else -1
        if (
            keypoints.shape != (count, 2)
            or descriptors.shape != (count, descriptor_dim)
            or scores.shape != (count,)
            or winners.shape != (count,)
            or winner_ids.shape != (count,)
            or winner_scores.shape != (count,)
            or valid_rows.shape != (count,)
            or query_rows.shape != (count,)
            or valid_rows.dtype != torch.bool
            or not bool(valid_rows.all())
            or not torch.equal(query_rows.long(), torch.arange(count))
        ):
            raise ValueError("V21 cache frontend/winner columns do not align")
        if winners.numel() and (
            int(winners.min()) < 0 or int(winners.max()) >= anchor_count
        ):
            raise ValueError("V21 cache winner row is outside the frozen map")
        inliers = torch.as_tensor(record.get("baseline_inliers"))
        if (
            inliers.ndim != 1
            or int(record.get("baseline_inlier_count", -1)) != inliers.numel()
            or (
                inliers.numel()
                and (
                    int(inliers.min()) < 0
                    or int(inliers.max()) >= count
                    or torch.unique(inliers).numel() != inliers.numel()
                )
            )
        ):
            raise ValueError("V21 cache baseline inliers are invalid")
        if torch.as_tensor(record.get("image_hw")).shape != (2,):
            raise ValueError("V21 cache image_hw is invalid")
        height, width = map(int, torch.as_tensor(record["image_hw"]).tolist())
        mask_present = bool(record.get("valid_mask_present"))
        mask_digest = record.get("valid_mask_sha256")
        mask_count = record.get("valid_mask_valid_pixel_count")
        mask_fraction = record.get("valid_mask_valid_fraction")
        if (
            record.get("valid_mask_raster_cached") is not False
            or record.get("valid_mask") is not None
        ):
            raise ValueError("V21 cache must not duplicate full valid-mask rasters")
        if mask_present:
            _require_sha256(mask_digest, label="V21 derived valid-mask SHA256")
            mask_count = _nonnegative_integer(
                mask_count, label="V21 valid-mask pixel count"
            )
            if (
                "valid_mask_source" not in optional_sources
                or mask_count > height * width
                or not 0.0 <= float(mask_fraction) <= 1.0
                or not math.isclose(
                    float(mask_fraction),
                    mask_count / (height * width),
                    rel_tol=1e-7,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError("V21 compact valid-mask binding is invalid")
        elif mask_digest is not None or mask_count is not None or mask_fraction is not None:
            raise ValueError("V21 absent valid mask must not expose derived statistics")
        if record.get("valid_mask_semantics") != VALID_MASK_SEMANTICS:
            raise ValueError("V21 cache valid-mask semantics differ")
        if torch.as_tensor(record.get("intrinsics")).shape != (3, 3):
            raise ValueError("V21 cache intrinsics are invalid")
        pose_w2c = torch.as_tensor(record.get("pose_w2c"))
        if pose_w2c.shape != (4, 4):
            raise ValueError("V21 cache GT pose is invalid")
        if (
            pose_w2c_sha256(pose_w2c) != record.get("pose_w2c_sha256")
            or SHA256_PATTERN.fullmatch(str(record.get("image_sha256"))) is None
        ):
            raise ValueError("V21 cache GT/image source hash differs")
        if torch.as_tensor(record.get("baseline_pose_w2c")).shape != (4, 4):
            raise ValueError("V21 cache baseline pose is invalid")
        expected_r5 = (
            float(record.get("baseline_translation_error_cm")) < 5.0
            and float(record.get("baseline_rotation_error_deg")) < 5.0
        )
        if record.get("baseline_r5") is not expected_r5:
            raise ValueError("V21 cache R5 label differs from baseline errors")
        expected_task = math.hypot(
            float(record.get("baseline_translation_error_cm")) / 5.0,
            float(record.get("baseline_rotation_error_deg")) / 5.0,
        )
        if not math.isclose(
            float(record.get("baseline_task_error")),
            expected_task,
            rel_tol=1e-7,
            abs_tol=1e-8,
        ):
            raise ValueError("V21 cache task error differs from baseline pose errors")


def atomic_torch_save_fresh(payload: Mapping, output: str | Path) -> Path:
    """Validate, save, and expose a cache without replacing an existing file."""

    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"V21 cache output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        torch.save(dict(payload), temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        validate_cache_payload(reloaded)
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"V21 cache output appeared during materialization: {output}"
            ) from error
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return output
