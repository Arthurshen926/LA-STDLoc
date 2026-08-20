#!/usr/bin/env python3
"""Audit R1 artifact-cache equivalence for numeric calibration reuse.

The R1 cache intentionally changes observation reliability used while fusing
the already-materialized map.  It must not change any query-localization row,
rendered geometry sample, query registry entry, or frozen Track input.  This
audit proves that narrower contract and emits the V2 equivalence schema
consumed by the existing calibration-rebind tool.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import torch

from common.hashing import sha256_file
from common.tensor_identity import recursive_bitwise_equal


_ADDED_QUERY_FIELDS = {
    "native_artifact_exposure",
    "native_artifact_reliability",
    "native_raw_clean_descriptor_cosine",
    "native_raw_clean_detector_score_stability",
    "native_raw_clean_position_displacement_px",
    "native_raw_clean_position_stability",
}
_CHANGED_QUERY_FIELDS = {"native_appearance_reliability", "source"}
_LOCALIZATION_QUERY_FIELDS = {
    "native_descriptors",
    "native_input_hw",
    "native_K",
    "native_keypoints",
    "native_scores",
    "native_valid_keypoint_mask",
    "pose_w2c",
}
_UNIT_INTERVAL_FIELDS = _ADDED_QUERY_FIELDS - {
    "native_raw_clean_position_displacement_px"
}
_SOURCE_PATHS = (
    "common/tensor_identity.py",
    "docs/evidence/rendered_rgb_track_artifact_stability_preregistration.json",
    "scripts/audit_rendered_track_artifact_cache_equivalence.py",
)


def _load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _producer_identity() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("artifact-cache equivalence producer must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "torch_version": torch.__version__,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in _SOURCE_PATHS
        },
    }


def _finite_aligned_vector(value: Any, *, count: int, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1 or tensor.shape[0] != count:
        raise ValueError(f"{label} must be an exact [{count}] vector")
    if tensor.dtype not in (torch.float16, torch.float32, torch.float64):
        raise ValueError(f"{label} must be floating point")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must be finite")
    return tensor


def audit_artifact_cache_equivalence(
    source: Mapping[str, Any],
    refreshed: Mapping[str, Any],
    track_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a strict, calibration-focused cache-equivalence audit."""
    if (
        source.get("schema") != "lafgs_rendered_rgb_appearance_ensemble_cache"
        or source.get("version") != 1
    ):
        raise ValueError("source cache is not the frozen appearance ensemble")
    if (
        refreshed.get("schema") != "lafgs_rendered_rgb_artifact_stability_cache"
        or refreshed.get("version") != 1
    ):
        raise ValueError("refreshed cache is not rendered artifact stability R1")
    for label, payload in (("source", source), ("refreshed", refreshed)):
        if (
            payload.get("uses_source_mapping_rgb") is not False
            or payload.get("uses_test_queries") is not False
        ):
            raise ValueError(f"{label} cache is not source-image-free mapping-only")

    source_queries = source.get("queries")
    refreshed_queries = refreshed.get("queries")
    if not isinstance(source_queries, Mapping) or not isinstance(
        refreshed_queries, Mapping
    ):
        raise ValueError("query cache records must be mappings")
    source_names = list(source_queries)
    refreshed_names = list(refreshed_queries)
    if source_names != refreshed_names or not source_names:
        raise ValueError("artifact cache changed the ordered query registry")
    if list(track_payload.get("query_names", [])) != source_names:
        raise ValueError("frozen Track payload query registry differs")
    if track_payload.get("rendered_rgb_only") is not True:
        raise ValueError("frozen Track payload is not rendered-RGB-only")

    allowed_top_level_additions = {"artifact_stability"}
    if set(refreshed) != set(source) | allowed_top_level_additions:
        raise ValueError("artifact cache has an unexpected top-level field change")
    for key in source:
        if key in {"schema", "queries"}:
            continue
        if not recursive_bitwise_equal(source[key], refreshed[key]):
            raise ValueError(f"artifact cache changed frozen top-level field {key}")
    stability_contract = refreshed.get("artifact_stability")
    if not isinstance(stability_contract, Mapping) or (
        stability_contract.get("topology_frozen") is not True
        or stability_contract.get("descriptors_remain_v14_appearance_descriptors")
        is not True
    ):
        raise ValueError("artifact cache stability contract is incomplete")

    exact_shared_fields: set[str] | None = None
    for name in source_names:
        source_record = source_queries[name]
        refreshed_record = refreshed_queries[name]
        if not isinstance(source_record, Mapping) or not isinstance(
            refreshed_record, Mapping
        ):
            raise ValueError(f"query record {name} is not a mapping")
        if set(refreshed_record) != set(source_record) | _ADDED_QUERY_FIELDS:
            raise ValueError(f"query record {name} has an unauthorized field change")
        shared = set(source_record) - _CHANGED_QUERY_FIELDS
        if exact_shared_fields is None:
            exact_shared_fields = shared
        elif shared != exact_shared_fields:
            raise ValueError("source query records have inconsistent field schemas")
        if not _LOCALIZATION_QUERY_FIELDS <= shared:
            raise ValueError("source query record lacks a localization input field")
        for key in shared:
            if not recursive_bitwise_equal(source_record[key], refreshed_record[key]):
                raise ValueError(f"query {name} changed frozen field {key}")
        if refreshed_record.get("source") != (
            "gaussian_rendered_rgb_artifact_stability_r1"
        ):
            raise ValueError(f"query {name} has the wrong R1 source tag")
        row_count = int(torch.as_tensor(source_record["native_keypoints"]).shape[0])
        if tuple(torch.as_tensor(source_record["native_keypoints"]).shape) != (
            row_count,
            2,
        ):
            raise ValueError(f"query {name} keypoints are not [N,2]")
        source_reliability = _finite_aligned_vector(
            source_record["native_appearance_reliability"],
            count=row_count,
            label=f"{name} source appearance reliability",
        )
        refreshed_reliability = _finite_aligned_vector(
            refreshed_record["native_appearance_reliability"],
            count=row_count,
            label=f"{name} refreshed appearance reliability",
        )
        if source_reliability.dtype != refreshed_reliability.dtype:
            raise ValueError(f"query {name} changed appearance reliability dtype")
        for key in _ADDED_QUERY_FIELDS:
            values = _finite_aligned_vector(
                refreshed_record[key], count=row_count, label=f"{name} {key}"
            )
            if key in _UNIT_INTERVAL_FIELDS and not bool(
                ((values >= 0) & (values <= 1)).all()
            ):
                raise ValueError(f"query {name} {key} is outside [0,1]")
            if key == "native_raw_clean_position_displacement_px" and not bool(
                (values >= 0).all()
            ):
                raise ValueError(f"query {name} displacement is negative")

    return {
        "query_count": len(source_names),
        "query_order_exact": True,
        "source_track_query_registry_exact": True,
        "localization_query_rows_bitwise_exact": True,
        "rendered_geometry_samples_bitwise_exact": True,
        "artifact_only_allowed_differences": True,
        "source_image_free_mapping_only": True,
        "changed_query_fields": sorted(_CHANGED_QUERY_FIELDS),
        "added_query_fields": sorted(_ADDED_QUERY_FIELDS),
        "exact_shared_query_fields": sorted(exact_shared_fields or ()),
        "content_equivalent_track_payload_reuse_authorized": True,
        "calibration_numeric_reuse_authorized": True,
    }


def _atomic_json(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        reloaded = json.loads(temporary.read_text())
        if reloaded != payload:
            raise RuntimeError("temporary equivalence report did not reload exactly")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--refreshed-cache", type=Path, required=True)
    parser.add_argument("--expected-refreshed-cache-sha256", required=True)
    parser.add_argument("--source-track-payload", type=Path, required=True)
    parser.add_argument("--expected-source-track-payload-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source_cache.resolve()
    refreshed_path = args.refreshed_cache.resolve()
    track_path = args.source_track_payload.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError("refusing to overwrite an equivalence report")
    entries = {
        "source_cache": (source_path, args.expected_source_cache_sha256.lower()),
        "refreshed_cache": (
            refreshed_path,
            args.expected_refreshed_cache_sha256.lower(),
        ),
        "source_track_payload": (
            track_path,
            args.expected_source_track_payload_sha256.lower(),
        ),
    }
    sources: dict[str, dict[str, str]] = {}
    for label, (path, expected) in entries.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
        sources[label] = {"path": str(path), "sha256": actual}
    audit = audit_artifact_cache_equivalence(
        _load(source_path), _load(refreshed_path), _load(track_path)
    )
    checks = {
        "source_cache_sha256_expected": True,
        "refreshed_cache_sha256_expected": True,
        "source_track_payload_sha256_expected": True,
        "query_order_exact": audit["query_order_exact"],
        "localization_query_rows_bitwise_exact": audit[
            "localization_query_rows_bitwise_exact"
        ],
        "rendered_geometry_samples_bitwise_exact": audit[
            "rendered_geometry_samples_bitwise_exact"
        ],
        "artifact_only_allowed_differences": audit["artifact_only_allowed_differences"],
        "source_track_query_registry_exact": audit["source_track_query_registry_exact"],
        "source_image_free_mapping_only": audit["source_image_free_mapping_only"],
        "content_equivalent_track_payload_reuse_authorized": audit[
            "content_equivalent_track_payload_reuse_authorized"
        ],
        "calibration_numeric_reuse_authorized": audit[
            "calibration_numeric_reuse_authorized"
        ],
    }
    report = {
        "schema": "lafgs_mapping_sparse_refresh_equivalence",
        "version": 2,
        "uses_test_queries": False,
        "valid": all(checks.values()),
        "sources": sources,
        "checks": checks,
        "audit": audit,
        "producer_identity": _producer_identity(),
    }
    _atomic_json(report, output)
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output), "valid": True},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
