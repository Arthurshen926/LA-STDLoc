"""Mapping-only detector-density and sparse-frontend contract audit.

This module is deliberately read-only.  It inspects already materialized query
caches and emits a paired deployment-factor manifest; it never runs the
frontend, rebuilds evidence, or evaluates localization poses.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


AUDIT_SCHEMA = "lafgs_mapping_density_protocol_audit"
FACTOR_SCHEMA = "lafgs_mapping_density_paired_factor_manifest"


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    position = float(fraction) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize(values: Sequence[int | float]) -> dict[str, float | int | None]:
    """Return compact, JSON-safe distribution diagnostics."""
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "median": None,
            "p90": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(finite),
        "minimum": finite[0],
        "p10": _quantile(finite, 0.10),
        "median": _quantile(finite, 0.50),
        "p90": _quantile(finite, 0.90),
        "maximum": finite[-1],
        "mean": float(sum(finite) / len(finite)),
    }


def explain_area_density_resolution(
    deployment: Mapping[str, Any], image_areas_px: Sequence[int]
) -> dict[str, Any]:
    """Explain the exact median-area and min/max clamp used by the mainline."""
    base = int(deployment["keypoints"])
    reference = deployment.get("keypoint_reference_area_px")
    minimum = int(deployment.get("keypoint_minimum", 1))
    maximum = int(deployment.get("keypoint_maximum", base))
    if reference is None or not image_areas_px:
        return {
            "base_keypoints": base,
            "reference_area_px": reference,
            "median_image_area_px": None,
            "area_scaled_unclamped_keypoints": base,
            "resolved_keypoints": base,
            "clamp_reason": "area_scaling_disabled",
            "minimum": minimum,
            "maximum": maximum,
        }
    areas = sorted(int(area) for area in image_areas_px)
    median_area = int(areas[len(areas) // 2])
    scaled = round(base * median_area / float(reference))
    resolved = max(minimum, min(maximum, scaled))
    if scaled < minimum:
        reason = "minimum_clamp"
    elif scaled > maximum:
        reason = "maximum_clamp"
    else:
        reason = "area_scaled_unclamped"
    return {
        "base_keypoints": base,
        "reference_area_px": float(reference),
        "median_image_area_px": median_area,
        "area_scaled_unclamped_keypoints": int(scaled),
        "resolved_keypoints": int(resolved),
        "clamp_reason": reason,
        "minimum": minimum,
        "maximum": maximum,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_identity(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256(resolved),
    }


def _unique_ints(values: Sequence[Any]) -> list[int]:
    return sorted({int(value) for value in values if value is not None})


def _factor_manifest(
    *,
    scene: str,
    query_cache_path: Path,
    cache_signature: str | None,
    mapping_graph: dict[str, Any] | None,
    mapping_keypoints_target: int,
    observed_requests: list[int],
    expected_nms_radius: int,
    nms_status: str,
    deployment_keypoint_factors: Sequence[int],
) -> dict[str, Any]:
    mapping_target_satisfied = observed_requests == [int(mapping_keypoints_target)]
    blockers = []
    if not mapping_target_satisfied:
        blockers.append("mapping_query_cache_does_not_attest_target_k_mapping")
    if nms_status != "pass":
        blockers.append(f"mapping_nms_contract_{nms_status}")
    if mapping_graph is None:
        blockers.append("mapping_graph_identity_missing")
    graph_identity = {
        "query_cache_path": str(query_cache_path),
        "query_cache_signature": cache_signature,
        "mapping_graph_artifact": mapping_graph,
        "k_mapping_target": int(mapping_keypoints_target),
        "observed_mapping_request_values": observed_requests,
        "nms_radius_target": int(expected_nms_radius),
    }
    return {
        "schema": FACTOR_SCHEMA,
        "version": 1,
        "scene": scene,
        "uses_test_queries": False,
        "audit_only": True,
        "factor_axis": "k_deployment",
        "ready_for_paired_deployment_factor": not blockers,
        "blocked_reasons": blockers,
        "immutable_mapping_graph": graph_identity,
        "variants": [
            {
                "name": f"deployment_k{int(value)}",
                "k_mapping": int(mapping_keypoints_target),
                "k_deployment": int(value),
                "mapping_graph_identity": graph_identity,
            }
            for value in deployment_keypoint_factors
        ],
        "pairing_contract": {
            "require_identical_query_cache_signature": True,
            "require_identical_mapping_graph_sha256": True,
            "require_identical_map_and_metric": True,
            "allowed_varying_field": "k_deployment",
            "forbid_mapping_cache_rebuild_between_variants": True,
            "forbid_mapping_and_deployment_density_joint_change": True,
        },
    }


def audit_mapping_density(
    *,
    scene: str,
    query_cache_path: str | Path,
    deployment: Mapping[str, Any],
    mapping_keypoints_target: int = 2048,
    expected_nms_radius: int = 4,
    deployment_keypoint_factors: Sequence[int] = (1024, 2048),
    mapping_graph_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit one existing mapping cache without invoking any model code."""
    query_cache_path = Path(query_cache_path).expanduser().resolve()
    payload = torch.load(query_cache_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("query cache must be a mapping")
    cache = payload.get("queries", payload)
    if not isinstance(cache, dict):
        raise ValueError("query-cache records must be a mapping")
    records = [
        (name, record)
        for name, record in cache.items()
        if isinstance(record, dict) and "native_keypoints" in record
    ]
    if not records:
        raise ValueError("query cache has no native sparse mapping records")

    requested = []
    before_mask = []
    after_mask = []
    native_rows = []
    valid_pixels = []
    valid_fractions = []
    image_areas = []
    entry_nms = []
    metadata_row_mismatches = []
    for name, record in records:
        metadata = record.get("native_sparse_metadata", {})
        rows = int(torch.as_tensor(record["native_keypoints"]).shape[0])
        height, width = map(int, record["native_input_hw"])
        area = height * width
        valid_mask = torch.as_tensor(record["native_valid_mask"], dtype=torch.bool)
        # Per-record torch reductions repeatedly enter the global thread pool
        # and dominate this otherwise read-only audit on large caches.  The
        # masks are CPU tensors, so NumPy's contiguous scalar reduction is both
        # exact and substantially cheaper here.
        valid_count = int(valid_mask.numpy().sum())
        request = metadata.get("detect_num")
        before = metadata.get("keypoint_count_before_mask")
        after = metadata.get("keypoint_count_after_mask")
        requested.append(request)
        before_mask.append(before if before is not None else rows)
        after_mask.append(after if after is not None else rows)
        native_rows.append(rows)
        image_areas.append(area)
        valid_pixels.append(valid_count)
        valid_fractions.append(valid_count / float(area))
        entry_nms.append(metadata.get("nms_radius"))
        if after is not None and int(after) != rows:
            metadata_row_mismatches.append(name)

    signature_payload = payload.get("signature_payload", {})
    signature_request = signature_payload.get("native_sparse_keypoint_count")
    signature_nms = signature_payload.get("native_sparse_nms_radius")
    observed_requests = _unique_ints([signature_request, *requested])
    observed_nms = _unique_ints([signature_nms, *entry_nms])
    if not observed_nms:
        nms_status = "unattested"
    elif observed_nms == [int(expected_nms_radius)]:
        nms_status = "pass"
    else:
        nms_status = "mismatch"

    area_resolution = explain_area_density_resolution(deployment, image_areas)
    mapping_target_satisfied = observed_requests == [int(mapping_keypoints_target)]
    full_request_rate = sum(
        int(request is not None and int(before) == int(request))
        for request, before in zip(requested, before_mask)
    ) / len(records)
    masked_drop_rate = sum(
        int(int(after) < int(before))
        for before, after in zip(before_mask, after_mask)
    ) / len(records)
    graph_identity = _artifact_identity(mapping_graph_path)
    factor_manifest = _factor_manifest(
        scene=scene,
        query_cache_path=query_cache_path,
        cache_signature=payload.get("signature"),
        mapping_graph=graph_identity,
        mapping_keypoints_target=mapping_keypoints_target,
        observed_requests=observed_requests,
        expected_nms_radius=expected_nms_radius,
        nms_status=nms_status,
        deployment_keypoint_factors=deployment_keypoint_factors,
    )
    gaps = []
    if not mapping_target_satisfied:
        gaps.append(
            {
                "kind": "mapping_density_below_target",
                "target": int(mapping_keypoints_target),
                "observed_request_values": observed_requests,
                "area_resolution": area_resolution,
            }
        )
    if nms_status != "pass":
        gaps.append(
            {
                "kind": "nms_contract_not_attested"
                if nms_status == "unattested"
                else "nms_contract_mismatch",
                "target": int(expected_nms_radius),
                "observed_values": observed_nms,
            }
        )
    if metadata_row_mismatches:
        gaps.append(
            {
                "kind": "native_row_metadata_mismatch",
                "query_count": len(metadata_row_mismatches),
                "examples": metadata_row_mismatches[:8],
            }
        )

    return {
        "schema": AUDIT_SCHEMA,
        "version": 1,
        "scene": scene,
        "uses_test_queries": False,
        "audit_only": True,
        "rebuilds_gpu_cache": False,
        "mutates_frontend": False,
        "mutates_map": False,
        "sources": {
            "query_cache": str(query_cache_path),
            "query_cache_signature": payload.get("signature"),
            "query_cache_signature_version": signature_payload.get("version"),
            "mapping_graph": graph_identity,
        },
        "protocol": {
            "k_mapping_target": int(mapping_keypoints_target),
            "k_deployment_factors": [
                int(value) for value in deployment_keypoint_factors
            ],
            "expected_nms_radius": int(expected_nms_radius),
            "mapping_target_satisfied": mapping_target_satisfied,
            "nms_contract_status": nms_status,
            "observed_mapping_request_values": observed_requests,
            "observed_nms_radius_values": observed_nms,
            "area_density_resolution": area_resolution,
        },
        "mapping_cache": {
            "query_count": len(records),
            "native_input_area_px": summarize(image_areas),
            "requested_detect_num": summarize(
                [int(value) for value in requested if value is not None]
            ),
            "keypoint_count_before_mask": summarize(before_mask),
            "keypoint_count_after_mask": summarize(after_mask),
            "native_tensor_rows": summarize(native_rows),
            "valid_mask_pixel_count": summarize(valid_pixels),
            "valid_mask_fraction": summarize(valid_fractions),
            "detector_reached_requested_count_rate": float(full_request_rate),
            "queries_with_masked_keypoint_drop_rate": float(masked_drop_rate),
            "metadata_row_mismatch_count": len(metadata_row_mismatches),
        },
        "mechanism_gaps": gaps,
        "factor_manifest": factor_manifest,
        "decision": (
            "protocol_ready"
            if not factor_manifest["blocked_reasons"]
            else "blocked_before_paired_deployment_factor"
        ),
        "pose_gain_claimed": False,
    }
