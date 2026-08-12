#!/usr/bin/env python3
"""Bind frozen numeric calibration to one valid pair-factor Track payload."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from common.hashing import canonical_json, sha256_file
from common.calibration import (
    validate_equivalent_query_cache_calibration_parent,
)


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _normalized_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be 64 lowercase hex digits")
    return digest


def materialize_pair_factor_calibration(
    *,
    parent_path: Path,
    query_cache_path: Path,
    track_payload_path: Path,
    payload_lineage_audit_path: Path,
    expected_parent_calibration_sha256: str,
    expected_query_cache_sha256: str,
    expected_payload_lineage_audit_sha256: str,
    expected_mapping_keypoints: int,
    expected_nms_radius: int,
    expected_pair_budget: int,
) -> dict:
    parent_path = Path(parent_path).resolve()
    query_cache_path = Path(query_cache_path).resolve()
    track_payload_path = Path(track_payload_path).resolve()
    payload_lineage_audit_path = Path(payload_lineage_audit_path).resolve()
    expected_parent_calibration_sha256 = _normalized_sha256(
        expected_parent_calibration_sha256,
        label="Expected parent-calibration SHA-256",
    )
    expected_query_cache_sha256 = _normalized_sha256(
        expected_query_cache_sha256,
        label="Expected query-cache SHA-256",
    )
    expected_payload_lineage_audit_sha256 = _normalized_sha256(
        expected_payload_lineage_audit_sha256,
        label="Expected payload-lineage-audit SHA-256",
    )
    expected_pair_budget = int(expected_pair_budget)
    expected_mapping_keypoints = int(expected_mapping_keypoints)
    expected_nms_radius = int(expected_nms_radius)
    if expected_mapping_keypoints <= 0:
        raise ValueError("Expected mapping keypoints must be positive")
    if expected_nms_radius <= 0:
        raise ValueError("Expected NMS radius must be positive")
    if expected_pair_budget <= 0:
        raise ValueError("Expected pair budget must be positive")
    parent_calibration_sha256 = sha256_file(parent_path)
    if parent_calibration_sha256 != expected_parent_calibration_sha256:
        raise ValueError("Parent calibration SHA-256 differs from the frozen contract")
    payload_lineage_audit_sha256 = sha256_file(payload_lineage_audit_path)
    if payload_lineage_audit_sha256 != expected_payload_lineage_audit_sha256:
        raise ValueError(
            "Payload-lineage-audit SHA-256 differs from the frozen contract"
        )
    parent = json.loads(parent_path.read_text())
    audit = json.loads(payload_lineage_audit_path.read_text())
    sources = dict(parent.get("sources", {}))
    if (
        parent.get("schema") != "lafgs_mapping_only_scene_calibration"
        or int(parent.get("version", 0)) < 2
        or sources.get("uses_test_queries") is not False
    ):
        raise ValueError("Parent calibration is not a mapping-only V2 contract")
    if Path(str(sources.get("query_cache", ""))).resolve() != query_cache_path:
        raise ValueError("Parent calibration names a different query cache")
    actual_query_cache_sha256 = sha256_file(query_cache_path)
    if actual_query_cache_sha256 != str(expected_query_cache_sha256):
        raise ValueError("Query-cache SHA-256 differs from frozen factor contract")
    validate_equivalent_query_cache_calibration_parent(
        parent,
        parent_path=parent_path,
        query_cache_path=query_cache_path,
    )
    payload = torch.load(track_payload_path, map_location="cpu", weights_only=False)
    track_payload_sha256 = sha256_file(track_payload_path)
    provenance = dict(payload.get("provenance", {}))
    if (
        payload.get("schema") != "lafgs_track_first_payload"
        or payload.get("version") != 1
        or provenance.get("uses_test_queries") is not False
    ):
        raise ValueError("Track payload is not a valid mapping-only contract")
    if provenance.get("query_cache_sha256") != actual_query_cache_sha256:
        raise ValueError("Track payload is bound to a different query cache")
    checks = dict(audit.get("checks", {}))
    required_audit_checks = {
        "assignment_algorithm_exact",
        "assignment_parameters_frozen",
        "base_state_path_bound",
        "base_state_sha256_bound",
        "factor_has_no_assignment",
        "factor_input_lineage_present",
        "factor_input_lineage_copied_to_payload",
        "factor_manifest_path_bound",
        "factor_manifest_sha256_bound",
        "factor_query_cache_path_bound",
        "factor_query_cache_sha256_bound",
        "manifest_query_cache_path_bound",
        "manifest_query_cache_sha256_bound",
        "factor_sha256_bound",
        "mapping_keypoints_expected",
        "mapping_nms_radius_expected",
        "pair_policy_parameters_present",
        "pair_policy_parallax_diverse",
        "pair_sidecar_exact_budget",
        "pair_sidecar_per_pair_columns_aligned",
        "frozen_bootstrap_manifest_path_bound",
        "frozen_bootstrap_manifest_sha256_bound",
        "query_cache_path_bound",
        "query_cache_sha256_bound",
        "source_factor_path_bound",
        "tracks_fields_equal",
        "tracks_values_equal",
        "track_geometry_fields_equal",
        "track_geometry_values_equal",
        "assignment_shapes_valid",
        "assignment_indices_valid",
        "assignment_costs_valid",
    }
    if (
        audit.get("schema") != "lafgs_pair_policy_payload_lineage_audit"
        or audit.get("version") != 1
        or audit.get("uses_test_queries") is not False
        or audit.get("valid") is not True
        or audit.get("pair_policy") != "parallax_diverse"
        or audit.get("expected_mapping_keypoints") != expected_mapping_keypoints
        or audit.get("mapping_keypoints") != expected_mapping_keypoints
        or audit.get("expected_nms_radius") != expected_nms_radius
        or audit.get("mapping_nms_radius") != expected_nms_radius
        or audit.get("expected_pair_budget") != expected_pair_budget
        or audit.get("exact_pair_budget") != expected_pair_budget
        or not required_audit_checks.issubset(checks)
        or not all(checks.values())
    ):
        raise ValueError(
            "Payload lineage audit is not a valid parallax-diverse contract"
        )
    if (
        Path(str(audit.get("payload", ""))).resolve() != track_payload_path
        or audit.get("payload_sha256") != track_payload_sha256
        or audit.get("query_cache_sha256") != actual_query_cache_sha256
    ):
        raise ValueError("Payload lineage audit names different inputs")
    factor_path = Path(str(audit.get("factor", ""))).resolve()
    base_state_path = Path(str(audit.get("base_state", ""))).resolve()
    if (
        not factor_path.is_file()
        or sha256_file(factor_path) != audit.get("factor_sha256")
        or provenance.get("source_factor_sha256") != audit.get("factor_sha256")
        or Path(str(provenance.get("source_factor", ""))).resolve() != factor_path
        or not base_state_path.is_file()
        or sha256_file(base_state_path) != audit.get("base_state_sha256")
        or provenance.get("base_state_sha256") != audit.get("base_state_sha256")
        or Path(str(provenance.get("base_state", ""))).resolve() != base_state_path
    ):
        raise ValueError("Payload lineage audit factor/base binding failed")

    result = copy.deepcopy(parent)
    result["uses_test_queries"] = False
    result["sources"] = {
        **sources,
        "query_cache": str(query_cache_path),
        "query_cache_sha256": actual_query_cache_sha256,
        "track_payload": str(track_payload_path),
        "track_payload_sha256": track_payload_sha256,
        "payload_lineage_audit": str(payload_lineage_audit_path),
        "payload_lineage_audit_sha256": (payload_lineage_audit_sha256),
        "uses_test_queries": False,
    }
    result["lineage"] = {
        "mode": "frozen_numeric_pair_factor",
        "parent_calibration": str(parent_path),
        "parent_calibration_sha256": parent_calibration_sha256,
        "expected_parent_calibration_sha256": (expected_parent_calibration_sha256),
        "payload_lineage_audit": str(payload_lineage_audit_path),
        "payload_lineage_audit_sha256": (payload_lineage_audit_sha256),
        "expected_payload_lineage_audit_sha256": (
            expected_payload_lineage_audit_sha256
        ),
        "expected_pair_budget": expected_pair_budget,
        "expected_mapping_keypoints": expected_mapping_keypoints,
        "expected_nms_radius": expected_nms_radius,
        "parameters_sha256": _canonical_sha256(parent["parameters"]),
        "policy_sha256": _canonical_sha256(parent["policy"]),
        "statistics_reused_from_parent": True,
        "parameters_reused_from_parent": True,
        "uses_test_queries": False,
    }
    if result["parameters"] != parent["parameters"]:
        raise AssertionError("Frozen calibration parameters changed")
    if result["policy"] != parent["policy"]:
        raise AssertionError("Frozen calibration policy changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--payload-lineage-audit", type=Path, required=True)
    parser.add_argument("--expected-parent-calibration-sha256", required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--expected-payload-lineage-audit-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_pair_factor_calibration(
        parent_path=args.parent,
        query_cache_path=args.query_cache,
        track_payload_path=args.track_payload,
        payload_lineage_audit_path=args.payload_lineage_audit,
        expected_parent_calibration_sha256=(args.expected_parent_calibration_sha256),
        expected_query_cache_sha256=args.expected_query_cache_sha256,
        expected_payload_lineage_audit_sha256=(
            args.expected_payload_lineage_audit_sha256
        ),
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_pair_budget=args.expected_pair_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "lineage": result["lineage"],
                "sources": result["sources"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
