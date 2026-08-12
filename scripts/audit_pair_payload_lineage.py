#!/usr/bin/env python3
"""Audit a replayed pair-policy payload before canonical evidence building."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file


def _normalized_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be 64 lowercase hex digits")
    return digest


def _tensor_exact(left, right) -> bool:
    left, right = torch.as_tensor(left), torch.as_tensor(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if not left.dtype.is_floating_point:
        return torch.equal(left, right)
    return (
        torch.equal(torch.isnan(left), torch.isnan(right))
        and torch.equal(torch.isposinf(left), torch.isposinf(right))
        and torch.equal(torch.isneginf(left), torch.isneginf(right))
        and torch.equal(left.nan_to_num(), right.nan_to_num())
    )


def _value_exact(left, right) -> bool:
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        return set(left) == set(right) and all(
            _value_exact(left[name], right[name]) for name in left
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(
            right, (list, tuple)
        ):
            return False
        return len(left) == len(right) and all(
            _value_exact(lvalue, rvalue)
            for lvalue, rvalue in zip(left, right)
        )
    if torch.is_tensor(left) or torch.is_tensor(right):
        return _tensor_exact(left, right)
    return left == right


def _assignment_parameters_from_manifest(manifest: dict) -> dict:
    frozen = dict(manifest["arguments"])
    return {
        "topk": int(frozen["geometry_teacher_provenance_topk"]),
        "minimum_consensus_rate": float(
            frozen["geometry_teacher_provenance_min_consensus_rate"]
        ),
        "minimum_views": int(
            frozen["geometry_teacher_provenance_min_views"]
        ),
        "group_maximum_landmarks": int(
            frozen["geometry_teacher_provenance_group_max_landmarks"]
        ),
        "group_minimum_relative_mass": float(
            frozen[
                "geometry_teacher_provenance_group_min_relative_mass"
            ]
        ),
        "group_minimum_consensus_rate": float(
            frozen[
                "geometry_teacher_provenance_group_min_consensus_rate"
            ]
        ),
        "depth_absolute_tolerance_m": float(
            frozen[
                "geometry_teacher_provenance_depth_abs_tolerance_m"
            ]
        ),
        "depth_relative_tolerance": float(
            frozen["geometry_teacher_provenance_depth_rel_tolerance"]
        ),
    }


def _shape(value) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(torch.as_tensor(value).shape)
    except (TypeError, ValueError):
        return None


def _pair_sidecar_contract(
    factor: dict, *, expected_pair_budget: int
) -> dict[str, bool]:
    sidecar = factor.get("pair_sidecar")
    if not isinstance(sidecar, dict):
        return {"pair_sidecar_present": False}
    policy = sidecar.get("policy")
    pair = sidecar.get("pair")
    if not isinstance(policy, dict) or not isinstance(pair, dict):
        return {
            "pair_sidecar_present": True,
            "pair_sidecar_policy_present": isinstance(policy, dict),
            "pair_sidecar_pair_table_present": isinstance(pair, dict),
        }
    required_per_pair = {
        "left_query_index",
        "right_query_index",
        "baseline_m",
        "axis_angle_deg",
        "mapping_point_joint_visibility_count",
        "mapping_point_overlap_jaccard",
        "mapping_point_parallax_median_deg",
        "raw_match_count",
        "descriptor_accepted_before_epipolar_count",
        "epipolar_accepted_top1_count",
        "cycle_supported_edge_count",
        "conflict_rejected_edge_count",
        "final_component_edge_count",
        "triangulated_track_count",
        "actual_triangulation_parallax_median_deg",
    }
    left = torch.as_tensor(pair.get("left_query_index", [])).long().reshape(-1)
    right = torch.as_tensor(pair.get("right_query_index", [])).long().reshape(-1)
    pair_count = int(left.numel())
    query_count = len(factor.get("query_names", []))
    aligned = required_per_pair.issubset(pair)
    if aligned:
        aligned = all(
            _shape(value) is not None
            and len(_shape(value)) >= 1
            and _shape(value)[0] == pair_count
            for name, value in pair.items()
            if name not in {"final_track_offsets", "final_track_indices"}
        )
    offsets = torch.as_tensor(pair.get("final_track_offsets", [])).long()
    indices = torch.as_tensor(pair.get("final_track_indices", [])).long()
    offsets_valid = (
        offsets.shape == (pair_count + 1,)
        and offsets.numel() > 0
        and int(offsets[0]) == 0
        and bool((offsets[1:] >= offsets[:-1]).all())
        and int(offsets[-1]) == int(indices.numel())
    )
    pair_indices_valid = (
        right.shape == left.shape
        and pair_count == int(expected_pair_budget)
        and bool((left >= 0).all())
        and bool((right >= 0).all())
        and bool((left < right).all())
        and bool((right < query_count).all())
        and len(set(zip(left.tolist(), right.tolist()))) == pair_count
    )
    return {
        "pair_sidecar_present": True,
        "pair_sidecar_schema": sidecar.get("schema")
        == "lafgs_mapping_track_pair_sidecar",
        "pair_sidecar_version": sidecar.get("version") == 1,
        "pair_sidecar_mapping_only": policy.get("uses_test_queries") is False,
        "pair_sidecar_policy_descriptor_free": policy.get(
            "uses_descriptors_for_selection"
        )
        is False,
        "pair_sidecar_policy_parallax_diverse": policy.get("name")
        == "parallax_diverse",
        "pair_sidecar_overlap_constraint": policy.get(
            "overlap_constraint_applied"
        )
        is True,
        "pair_sidecar_triangulation_attached": sidecar.get(
            "triangulation_attached"
        )
        is True,
        "pair_sidecar_exact_budget": policy.get("exact_pair_budget")
        == int(expected_pair_budget)
        == pair_count,
        "pair_sidecar_pair_indices_valid": pair_indices_valid,
        "pair_sidecar_per_pair_columns_aligned": bool(aligned),
        "pair_sidecar_final_track_csr_valid": bool(offsets_valid),
    }


def _mapping_keypoint_contract(
    factor: dict, *, expected_mapping_keypoints: int
) -> dict[str, bool]:
    expected_mapping_keypoints = int(expected_mapping_keypoints)
    return {
        "mapping_keypoints_expected": expected_mapping_keypoints > 0
        and factor.get("mapping_keypoint_factor")
        == expected_mapping_keypoints,
    }


def audit_pair_payload(
    payload: dict,
    factor: dict,
    base_state: dict,
    *,
    payload_path: Path,
    factor_path: Path,
    base_state_path: Path,
    query_cache_path: Path,
    expected_query_cache_sha256: str,
    frozen_bootstrap_manifest_path: Path,
    expected_frozen_bootstrap_manifest_sha256: str,
    expected_mapping_keypoints: int,
    expected_pair_budget: int,
) -> dict:
    assignment_fields = {
        "track_landmark_index",
        "track_assignment_cost",
        "landmark_best_track_index",
        "track_landmark_offsets",
        "track_landmark_indices",
        "track_landmark_costs",
    }
    payload_path = Path(payload_path).resolve()
    factor_path = Path(factor_path).resolve()
    base_state_path = Path(base_state_path).resolve()
    query_cache_path = Path(query_cache_path).resolve()
    frozen_bootstrap_manifest_path = Path(
        frozen_bootstrap_manifest_path
    ).resolve()
    expected_query_cache_sha256 = _normalized_sha256(
        expected_query_cache_sha256, label="Expected query-cache SHA-256"
    )
    expected_frozen_bootstrap_manifest_sha256 = _normalized_sha256(
        expected_frozen_bootstrap_manifest_sha256,
        label="Expected frozen-bootstrap-manifest SHA-256",
    )
    expected_mapping_keypoints = int(expected_mapping_keypoints)
    expected_pair_budget = int(expected_pair_budget)
    if expected_mapping_keypoints <= 0:
        raise ValueError("Expected mapping keypoints must be positive")
    if expected_pair_budget <= 0:
        raise ValueError("Expected pair budget must be positive")
    payload_sha256 = sha256_file(payload_path)
    factor_sha256 = sha256_file(factor_path)
    base_state_sha256 = sha256_file(base_state_path)
    query_cache_sha256 = sha256_file(query_cache_path)
    frozen_bootstrap_manifest_sha256 = sha256_file(
        frozen_bootstrap_manifest_path
    )
    frozen_bootstrap_manifest = json.loads(
        frozen_bootstrap_manifest_path.read_text()
    )
    expected_assignment_parameters = _assignment_parameters_from_manifest(
        frozen_bootstrap_manifest
    )
    provenance = dict(payload.get("provenance", {}))
    provenance_factor = Path(str(provenance.get("source_factor", "")))
    provenance_base = Path(str(provenance.get("base_state", "")))
    provenance_query_cache = Path(str(provenance.get("query_cache", "")))
    provenance_manifest = Path(
        str(provenance.get("frozen_bootstrap_manifest", ""))
    )
    checks = {
        "payload_schema": payload.get("schema") == "lafgs_track_first_payload",
        "payload_version": payload.get("version") == 1,
        "provenance_schema": provenance.get("schema")
        == "lafgs_replayed_track_provenance_assignment",
        "provenance_version": provenance.get("version") == 1,
        "payload_mapping_only": provenance.get("uses_test_queries") is False,
        "factor_schema": factor.get("schema") == "lafgs_pair_policy_track_factor",
        "factor_version": factor.get("version") == 1,
        "mapping_only": factor.get("uses_test_queries") is False,
        **_mapping_keypoint_contract(
            factor,
            expected_mapping_keypoints=expected_mapping_keypoints,
        ),
        "frozen_bootstrap_mapping_keypoints_expected": int(
            frozen_bootstrap_manifest.get("arguments", {}).get(
                "native_keypoint_count", -1
            )
        )
        == expected_mapping_keypoints,
        "density_frozen": factor.get("density_factor_mutated") is False,
        "descriptor_frozen": factor.get("descriptor_factor_mutated") is False,
        "selector_frozen": factor.get("selector_factor_mutated") is False,
        "pair_policy_parallax_diverse": factor.get("pair_policy")
        == "parallax_diverse",
        "factor_has_no_assignment": "assignment" not in factor,
        "query_names_equal": payload.get("query_names") == factor.get("query_names"),
        "query_bins_equal": torch.equal(
            torch.as_tensor(payload.get("query_bins")),
            torch.as_tensor(factor.get("query_bins")),
        ),
        "landmark_indices_equal": torch.equal(
            torch.as_tensor(payload.get("landmark_indices")),
            torch.as_tensor(base_state.get("landmark_indices")),
        ),
        "assignment_fields_complete": assignment_fields
        == set(payload.get("assignment", {})),
        "assignment_algorithm_exact": provenance.get("assignment_algorithm")
        == "frozen_2dgs_splat_provenance_exact_replay",
        "assignment_parameters_frozen": provenance.get(
            "assignment_parameters"
        ) == expected_assignment_parameters,
        "source_factor_path_bound": provenance_factor.resolve()
        == factor_path,
        "factor_sha256_bound": provenance.get("source_factor_sha256")
        == factor_sha256,
        "base_state_path_bound": provenance_base.resolve()
        == base_state_path,
        "base_state_sha256_bound": provenance.get("base_state_sha256")
        == base_state_sha256,
        "query_cache_path_bound": provenance_query_cache.resolve()
        == query_cache_path,
        "query_cache_sha256_bound": provenance.get("query_cache_sha256")
        == expected_query_cache_sha256
        == query_cache_sha256,
        "expected_query_cache_sha256_bound": provenance.get(
            "expected_query_cache_sha256"
        ) == expected_query_cache_sha256,
        "frozen_bootstrap_manifest_path_bound": provenance_manifest.resolve()
        == frozen_bootstrap_manifest_path,
        "frozen_bootstrap_manifest_sha256_bound": provenance.get(
            "frozen_bootstrap_manifest_sha256"
        )
        == expected_frozen_bootstrap_manifest_sha256
        == frozen_bootstrap_manifest_sha256,
        "expected_frozen_bootstrap_manifest_sha256_bound": (
            provenance.get(
                "expected_frozen_bootstrap_manifest_sha256",
                frozen_bootstrap_manifest_sha256,
            )
            == expected_frozen_bootstrap_manifest_sha256
        ),
        "pair_sidecar_equal": _value_exact(
            payload.get("pair_sidecar"), factor.get("pair_sidecar")
        ),
        **_pair_sidecar_contract(
            factor, expected_pair_budget=int(expected_pair_budget)
        ),
    }
    for table in ("tracks", "track_geometry"):
        checks[f"{table}_fields_equal"] = set(payload.get(table, {})) == set(
            factor.get(table, {})
        )
        checks[f"{table}_values_equal"] = all(
            _tensor_exact(payload[table][name], factor[table][name])
            for name in factor.get(table, {})
        )
    offsets = torch.as_tensor(
        payload.get("assignment", {}).get("track_landmark_offsets", [])
    ).long()
    indices = torch.as_tensor(
        payload.get("assignment", {}).get("track_landmark_indices", [])
    ).long()
    costs = torch.as_tensor(
        payload.get("assignment", {}).get("track_landmark_costs", [])
    )
    track_landmark_index = torch.as_tensor(
        payload.get("assignment", {}).get("track_landmark_index", [])
    ).long()
    track_assignment_cost = torch.as_tensor(
        payload.get("assignment", {}).get("track_assignment_cost", [])
    )
    landmark_best_track_index = torch.as_tensor(
        payload.get("assignment", {}).get("landmark_best_track_index", [])
    ).long()
    track_count = int(torch.as_tensor(factor["tracks"]["track_level"]).numel())
    landmark_count = int(
        torch.as_tensor(base_state.get("landmark_indices", [])).numel()
    )
    checks["assignment_csr_valid"] = (
        offsets.shape == (track_count + 1,)
        and offsets.numel() > 0
        and int(offsets[0]) == 0
        and bool((offsets[1:] >= offsets[:-1]).all())
        and int(offsets[-1]) == int(indices.numel())
    )
    checks["assignment_shapes_valid"] = (
        track_landmark_index.shape == (track_count,)
        and track_assignment_cost.shape == (track_count,)
        and landmark_best_track_index.shape == (landmark_count,)
        and costs.shape == indices.shape
    )
    checks["assignment_indices_valid"] = (
        bool((track_landmark_index >= -1).all())
        and bool((track_landmark_index < landmark_count).all())
        and bool((landmark_best_track_index >= -1).all())
        and bool((landmark_best_track_index < track_count).all())
        and bool((indices >= 0).all())
        and bool((indices < landmark_count).all())
    )
    assigned = track_landmark_index >= 0
    checks["assignment_costs_valid"] = (
        not bool(torch.isnan(track_assignment_cost).any())
        and bool(torch.isfinite(track_assignment_cost[assigned]).all())
        and bool(torch.isinf(track_assignment_cost[~assigned]).all())
        and bool((track_assignment_cost[assigned] >= 0).all())
        and bool(torch.isfinite(costs).all())
        and bool((costs >= 0).all())
    )
    valid = all(checks.values())
    return {
        "schema": "lafgs_pair_policy_payload_lineage_audit",
        "version": 1,
        "uses_test_queries": False,
        "valid": bool(valid),
        "pair_policy": str(factor.get("pair_policy")),
        "expected_mapping_keypoints": expected_mapping_keypoints,
        "mapping_keypoints": int(factor.get("mapping_keypoint_factor", -1)),
        "expected_pair_budget": int(expected_pair_budget),
        "exact_pair_budget": int(
            factor.get("pair_sidecar", {})
            .get("policy", {})
            .get("exact_pair_budget", -1)
        ),
        "checks": checks,
        "payload": str(payload_path.resolve()),
        "payload_sha256": payload_sha256,
        "factor": str(factor_path.resolve()),
        "factor_sha256": factor_sha256,
        "base_state": str(base_state_path.resolve()),
        "base_state_sha256": base_state_sha256,
        "query_cache": str(query_cache_path.resolve()),
        "query_cache_sha256": query_cache_sha256,
        "frozen_bootstrap_manifest": str(
            frozen_bootstrap_manifest_path.resolve()
        ),
        "frozen_bootstrap_manifest_sha256": (
            frozen_bootstrap_manifest_sha256
        ),
        "assignment_algorithm": provenance.get("assignment_algorithm"),
        "assignment_parameters": provenance.get("assignment_parameters"),
        "assignment": {
            "field_names": sorted(payload.get("assignment", {})),
            "track_count": track_count,
            "group_edge_count": int(indices.numel()),
            "assigned_track_count": int(
                (torch.as_tensor(
                    payload["assignment"]["track_landmark_index"]
                ) >= 0).sum()
            ),
            "assigned_landmark_count": int(
                (torch.as_tensor(
                    payload["assignment"]["landmark_best_track_index"]
                ) >= 0).sum()
            ),
        },
        "diagnostics": {
            name: value
            for name, value in payload.get("diagnostics", {}).items()
            if name.startswith("geometry_teacher_")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--base-state", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument(
        "--frozen-bootstrap-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--expected-frozen-bootstrap-manifest-sha256", required=True
    )
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def load(path: Path) -> dict:
        return torch.load(path, map_location="cpu", weights_only=False)

    report = audit_pair_payload(
        load(args.payload),
        load(args.factor),
        load(args.base_state),
        payload_path=args.payload,
        factor_path=args.factor,
        base_state_path=args.base_state,
        query_cache_path=args.query_cache,
        expected_query_cache_sha256=args.expected_query_cache_sha256,
        frozen_bootstrap_manifest_path=args.frozen_bootstrap_manifest,
        expected_frozen_bootstrap_manifest_sha256=(
            args.expected_frozen_bootstrap_manifest_sha256
        ),
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_pair_budget=args.expected_pair_budget,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
