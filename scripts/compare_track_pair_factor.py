#!/usr/bin/env python3
"""Compare paired Track builds under a fail-closed single-factor contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.hashing import sha256_file


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return normalized


def _number(payload: dict, *path: str) -> float:
    value = payload
    for key in path:
        value = value[key]
    if value is None:
        raise ValueError(f"Missing comparison value: {'/'.join(path)}")
    return float(value)


def _input_entry(report: dict, name: str) -> dict:
    value = report.get("inputs", {}).get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Factor report lacks {name} input lineage")
    return value


def _validate_report(
    report: dict,
    *,
    path: Path,
    expected_report_sha256: str,
    expected_policy: str,
    expected_mapping_keypoints: int,
    expected_nms_radius: int,
    expected_pair_budget: int,
    expected_query_count: int,
    expected_query_names_sha256: str,
    expected_manifest_sha256: str,
    expected_query_cache_sha256: str,
    expected_frozen_track_payload_sha256: str,
    expected_policy_parameters: dict,
) -> None:
    if sha256_file(path) != expected_report_sha256:
        raise ValueError("Factor report SHA-256 differs from expected")
    if (
        report.get("schema") != "lafgs_pair_policy_track_factor"
        or report.get("version") != 1
        or report.get("uses_test_queries") is not False
        or report.get("pair_policy") != expected_policy
        or report.get("mapping_keypoint_factor") != expected_mapping_keypoints
        or report.get("mapping_nms_radius") != expected_nms_radius
        or report.get("exact_pair_budget") != expected_pair_budget
        or report.get("mapping_query_count") != expected_query_count
        or report.get("query_names_sha256") != expected_query_names_sha256
        or report.get("pair_policy_parameters") != expected_policy_parameters
    ):
        raise ValueError("Factor report differs from the preregistered contract")
    expected_inputs = {
        "manifest": expected_manifest_sha256,
        "query_cache": expected_query_cache_sha256,
        "frozen_track_payload": expected_frozen_track_payload_sha256,
    }
    for name, expected_sha256 in expected_inputs.items():
        entry = _input_entry(report, name)
        artifact = Path(str(entry.get("path", ""))).resolve()
        if (
            entry.get("sha256") != expected_sha256
            or not artifact.is_file()
            or sha256_file(artifact) != expected_sha256
        ):
            raise ValueError(f"Factor report {name} input differs")
    artifact = Path(str(report.get("artifact", ""))).resolve()
    artifact_sha256 = report.get("artifact_sha256")
    if (
        not artifact.is_file()
        or not isinstance(artifact_sha256, str)
        or sha256_file(artifact) != artifact_sha256
    ):
        raise ValueError("Factor artifact is missing or differs from its report")
    if int(_number(report, "pair", "pair_count")) != expected_pair_budget:
        raise ValueError("Factor report pair table does not have the exact budget")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--expected-control-sha256", required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--expected-variant-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--expected-query-count", type=int, required=True)
    parser.add_argument("--expected-query-names-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--expected-frozen-track-payload-sha256", required=True)
    parser.add_argument("--minimum-overlap-jaccard", type=float, required=True)
    parser.add_argument("--minimum-joint-visibility-points", type=int, required=True)
    parser.add_argument("--parallax-saturation-deg", type=float, required=True)
    parser.add_argument("--diversity-weight", type=float, required=True)
    parser.add_argument("--candidate-pool-per-camera", type=int, required=True)
    parser.add_argument("--scene-points-per-camera", type=int, required=True)
    parser.add_argument("--maximum-scene-points", type=int, required=True)
    parser.add_argument("--scene-point-voxel-size-m", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control_path = args.control.resolve()
    variant_path = args.variant.resolve()
    control_sha256 = _sha256(
        args.expected_control_sha256, label="Expected control report SHA-256"
    )
    variant_sha256 = _sha256(
        args.expected_variant_sha256, label="Expected variant report SHA-256"
    )
    query_names_sha256 = _sha256(
        args.expected_query_names_sha256, label="Expected query-name SHA-256"
    )
    manifest_sha256 = _sha256(
        args.expected_manifest_sha256, label="Expected manifest SHA-256"
    )
    query_cache_sha256 = _sha256(
        args.expected_query_cache_sha256, label="Expected query-cache SHA-256"
    )
    frozen_track_payload_sha256 = _sha256(
        args.expected_frozen_track_payload_sha256,
        label="Expected frozen Track payload SHA-256",
    )
    expected_policy_parameters = {
        "minimum_overlap_jaccard": float(args.minimum_overlap_jaccard),
        "minimum_joint_visibility_points": int(args.minimum_joint_visibility_points),
        "parallax_saturation_deg": float(args.parallax_saturation_deg),
        "diversity_weight": float(args.diversity_weight),
        "candidate_pool_per_camera": int(args.candidate_pool_per_camera),
        "scene_points_per_camera": int(args.scene_points_per_camera),
        "maximum_scene_points": int(args.maximum_scene_points),
        "scene_point_voxel_size_m": float(args.scene_point_voxel_size_m),
    }
    control = _read(control_path)
    variant = _read(variant_path)
    for report, path, digest, policy in (
        (control, control_path, control_sha256, "nearest"),
        (variant, variant_path, variant_sha256, "parallax_diverse"),
    ):
        _validate_report(
            report,
            path=path,
            expected_report_sha256=digest,
            expected_policy=policy,
            expected_mapping_keypoints=int(args.expected_mapping_keypoints),
            expected_nms_radius=int(args.expected_nms_radius),
            expected_pair_budget=int(args.expected_pair_budget),
            expected_query_count=int(args.expected_query_count),
            expected_query_names_sha256=query_names_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_query_cache_sha256=query_cache_sha256,
            expected_frozen_track_payload_sha256=(frozen_track_payload_sha256),
            expected_policy_parameters=expected_policy_parameters,
        )
    if control["inputs"] != variant["inputs"]:
        raise ValueError("Factor arms do not have identical frozen inputs")

    metrics = {
        "pair_count": (
            _number(control, "pair", "pair_count"),
            _number(variant, "pair", "pair_count"),
        ),
        "pair_parallax_below_1deg_fraction": (
            _number(control, "pair", "mapping_point_parallax_below_1deg_fraction"),
            _number(variant, "pair", "mapping_point_parallax_below_1deg_fraction"),
        ),
        "pair_parallax_median_deg": (
            _number(control, "pair", "mapping_point_parallax_median_deg", "median"),
            _number(variant, "pair", "mapping_point_parallax_median_deg", "median"),
        ),
        "triangulated_tracks": (
            _number(control, "track", "triangulated_track_count"),
            _number(variant, "track", "triangulated_track_count"),
        ),
        "broad_eligible_tracks": (
            _number(control, "track", "broad_eligible_track_count"),
            _number(variant, "track", "broad_eligible_track_count"),
        ),
        "triangulated_covariance_p90_m2": (
            _number(control, "track", "triangulated_covariance_trace_m2", "p90"),
            _number(variant, "track", "triangulated_covariance_trace_m2", "p90"),
        ),
        "broad_support_query_p10": (
            _number(control, "track", "broad_track_support_per_mapping_query", "p10"),
            _number(variant, "track", "broad_track_support_per_mapping_query", "p10"),
        ),
    }
    comparisons = {
        name: {
            "control": baseline,
            "variant": revised,
            "delta": revised - baseline,
            "relative": None if baseline == 0 else revised / baseline - 1.0,
        }
        for name, (baseline, revised) in metrics.items()
    }
    gates = {
        "exact_global_pair_budget": metrics["pair_count"][0]
        == metrics["pair_count"][1]
        == int(args.expected_pair_budget),
        "low_parallax_fraction_reduced_by_10pp": (
            metrics["pair_parallax_below_1deg_fraction"][1]
            <= metrics["pair_parallax_below_1deg_fraction"][0] - 0.10
        ),
        "triangulated_tracks_retained_95pct": (
            metrics["triangulated_tracks"][1]
            >= 0.95 * metrics["triangulated_tracks"][0]
        ),
        "broad_tracks_retained_98pct": (
            metrics["broad_eligible_tracks"][1]
            >= 0.98 * metrics["broad_eligible_tracks"][0]
        ),
        "triangulated_covariance_p90_not_worse_5pct": (
            metrics["triangulated_covariance_p90_m2"][1]
            <= 1.05 * metrics["triangulated_covariance_p90_m2"][0]
        ),
        "broad_query_support_p10_retained_95pct": (
            metrics["broad_support_query_p10"][1]
            >= 0.95 * metrics["broad_support_query_p10"][0]
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": "lafgs_pair_policy_mechanism_gate",
        "version": 2,
        "uses_test_queries": False,
        "single_factor": "camera_pair_policy",
        "valid": True,
        "mapping_keypoints": int(args.expected_mapping_keypoints),
        "mapping_nms_radius": int(args.expected_nms_radius),
        "exact_pair_budget": int(args.expected_pair_budget),
        "query_count": int(args.expected_query_count),
        "query_names_sha256": query_names_sha256,
        "pair_policy_parameters": expected_policy_parameters,
        "comparisons": comparisons,
        "gates": gates,
        "mechanism_gate_passed": passed,
        "advance_to_full_pipeline_pose": passed,
        "decision": "GO_TO_PIPELINE" if passed else "STOP_BEFORE_PIPELINE",
        "limitations": [
            "A mechanism pass authorizes compact pipeline/mapping-pose validation; it is not a pose claim."
        ],
        "inputs": {
            "control": {"path": str(control_path), "sha256": control_sha256},
            "variant": {"path": str(variant_path), "sha256": variant_sha256},
            "factor_inputs": control["inputs"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
