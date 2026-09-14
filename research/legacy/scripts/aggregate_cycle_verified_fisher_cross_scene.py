#!/usr/bin/env python3
"""Aggregate the two mapping-only P8 Stage-B mechanism gates."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Sequence
import uuid

from common.hashing import canonical_json, sha256_file
from evidence.cycle_verified_fisher import CONTROL_POLICY_NAME, POLICY_NAME


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
SCENE_CONTRACTS = {
    "stairs": {
        "mapping_keypoints": 1024,
        "nms_radius": 4,
        "pair_budget": 7450,
        "candidate_pair_count": 14835,
        "candidate_component_count": 2,
    },
    "greatcourt": {
        "mapping_keypoints": 2048,
        "nms_radius": 4,
        "pair_budget": 5254,
        "candidate_pair_count": 9875,
        "candidate_component_count": 1,
    },
}
MATCHER_CONTRACT = {
    "minimum_similarity": 0.65,
    "minimum_margin": 0.01,
    "maximum_epipolar_error_px": 2.0,
    "epipolar_candidate_topk": 1,
    "epipolar_recovered_minimum_similarity": -1.0,
    "epipolar_recovered_minimum_margin": -1.0,
}
STAGE_A_GATE_NAMES = {
    "candidate_union_exact_and_bounded",
    "selected_exact_pair_budget",
    "candidate_components_exact",
    "selected_components_preserved",
    "selected_zero_isolates",
    "selected_minimum_degree_at_least_one",
    "verified_fisher_utility_improves_5pct",
    "verified_triangles_retain_98pct",
    "verified_triangle_camera_fraction_not_lower",
}
STAGE_B_GATE_NAMES = {
    "triangulated_tracks_retain_98pct",
    "broad_eligible_tracks_retain_98pct",
    "high_confidence_tracks_retain_98pct",
    "triangulated_covariance_p90_not_worse_5pct",
    "broad_mapping_query_coverage_not_lower",
    "control_probe_rows_reused",
    "variant_probe_rows_reused",
    "same_probe_matcher_contract",
}
STAGE_B_COMPARISON_NAMES = {
    "triangulated_tracks",
    "broad_eligible_tracks",
    "high_confidence_tracks",
    "triangulated_covariance_p90_m2",
    "mapping_query_with_broad_track_fraction",
}
STAGE_B_INPUT_NAMES = {
    "query_cache",
    "pair_proposals",
    "pair_match_probe",
    "pair_selection",
    "stage_a_gate",
    "control_factor",
    "control_report",
    "variant_factor",
    "variant_report",
}
STAGE_A_INPUT_NAMES = {
    "query_cache",
    "pair_proposals",
    "pair_match_probe",
    "pair_selection",
}
TRACK_LINEAGE_NAMES = {
    "manifest",
    "frozen_track_payload",
    "query_cache",
    "pair_proposals",
    "pair_match_probe",
    "pair_selection",
    "stage_a_gate",
}
TRACK_SCIENCE_KEYS = {
    "allow_chain_tracks",
    "depth_sampling",
    "exact_pair_budget",
    "huber_delta_px",
    "mapping_keypoints",
    "mapping_nms_radius",
    "matcher",
    "maximum_axis_angle_deg",
    "maximum_baseline_m",
    "maximum_condition_number",
    "maximum_covariance_trace_m2",
    "maximum_observations_per_landmark",
    "maximum_rendered_depth_residual_m",
    "maximum_reprojection_px",
    "minimum_baseline_m",
    "minimum_parallax_deg",
    "minimum_rendered_depth_observations",
    "minimum_track_views",
    "minimum_view_bins",
    "pair_neighbors",
    "parallax_quantile",
    "require_cycle",
    "surface_support_enabled",
    "triangulation_iterations",
    "view_bins",
    "view_direction_weight",
}
# These values are frozen per scene by the preregistered cache/Track contract.
# They are parameters of the same compiled algorithm, not policy-identity fields.
SCENE_SPECIFIC_TRACK_FIELDS = {
    "exact_pair_budget",
    "mapping_keypoints",
    "mapping_nms_radius",
    "maximum_covariance_trace_m2",
    "maximum_rendered_depth_residual_m",
    "maximum_reprojection_px",
}
COMPILED_TRACK_SCIENCE_CONTRACT = {
    "allow_chain_tracks": True,
    "depth_sampling": "native_depth_at_sparse_keypoints_or_nearest_pixel_v1",
    "huber_delta_px": 2.0,
    "matcher": MATCHER_CONTRACT,
    "maximum_axis_angle_deg": 75.0,
    "maximum_baseline_m": 5.0,
    "maximum_condition_number": 1_000_000.0,
    "maximum_observations_per_landmark": 32,
    "minimum_baseline_m": 0.03,
    "minimum_parallax_deg": 1.0,
    "minimum_rendered_depth_observations": 2,
    "minimum_track_views": 3,
    "minimum_view_bins": 2,
    "pair_neighbors": 6,
    "parallax_quantile": 0.75,
    "require_cycle": True,
    "surface_support_enabled": False,
    "triangulation_iterations": 3,
    "view_bins": 8,
    "view_direction_weight": 0.5,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stairs-stage-b-gate", type=Path, required=True)
    parser.add_argument("--expected-stairs-stage-b-gate-sha256", required=True)
    parser.add_argument("--greatcourt-stage-b-gate", type=Path, required=True)
    parser.add_argument("--expected-greatcourt-stage-b-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _expected_sha256(value: object, *, label: str) -> str:
    digest = str(value).strip().lower()
    if SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return digest


def _local_file(path: object, *, label: str) -> Path:
    if not isinstance(path, (str, Path)) or "://" in str(path):
        raise ValueError(f"{label} must be a local file")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _attest_file(
    path: object,
    expected_sha256: object,
    *,
    label: str,
    registry: dict[Path, str],
) -> Path:
    resolved = _local_file(path, label=label)
    expected = _expected_sha256(expected_sha256, label=f"expected {label} SHA-256")
    prior = registry.get(resolved)
    if prior is not None and prior != expected:
        raise ValueError(f"{label} conflicts with an earlier hash contract")
    if prior is not None:
        return resolved
    if sha256_file(resolved) != expected:
        raise ValueError(f"{label} SHA-256 differs from the explicit contract")
    registry[resolved] = expected
    return resolved


def _json_object(path: Path, *, label: str) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _reference(
    value: object,
    *,
    label: str,
    registry: dict[Path, str],
    attest: bool = True,
) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a path/SHA-256 object")
    if set(value) - {"path", "sha256", "content_sha256", "mapping_scope"}:
        raise ValueError(f"{label} contains unsupported lineage fields")
    if "path" not in value or "sha256" not in value:
        raise ValueError(f"{label} lacks path/SHA-256 lineage")
    digest = _expected_sha256(value["sha256"], label=f"{label} SHA-256")
    if attest:
        path = _attest_file(value["path"], digest, label=label, registry=registry)
    else:
        path = _local_file(value["path"], label=label)
    result = {"path": path, "sha256": digest}
    if "content_sha256" in value:
        result["content_sha256"] = _expected_sha256(
            value["content_sha256"], label=f"{label} content SHA-256"
        )
    if "mapping_scope" in value:
        mapping_scope = value["mapping_scope"]
        if (
            not isinstance(mapping_scope, dict)
            or mapping_scope.get("uses_test_queries") is not False
            or mapping_scope.get("mode")
            not in {
                "query_cache_explicit_mapping_only",
                "mapping_sparse_refresh_equivalence_v2",
            }
        ):
            raise ValueError(f"{label} does not attest an accepted mapping scope")
        equivalence = mapping_scope.get("equivalence_report")
        if mapping_scope["mode"] == "mapping_sparse_refresh_equivalence_v2":
            _reference(
                equivalence,
                label=f"{label} mapping-scope equivalence",
                registry=registry,
            )
        elif equivalence is not None:
            raise ValueError(f"{label} has an unexpected equivalence report")
        result["mapping_scope"] = deepcopy(mapping_scope)
    return result


def _same_reference(left: dict, right: dict) -> bool:
    return (
        left["path"] == right["path"]
        and left["sha256"] == right["sha256"]
        and left.get("content_sha256") == right.get("content_sha256")
        and left.get("mapping_scope") == right.get("mapping_scope")
    )


def _same_file_reference(left: dict, right: dict) -> bool:
    return (
        left["path"] == right["path"]
        and left["sha256"] == right["sha256"]
        and left.get("content_sha256") == right.get("content_sha256")
    )


def _scene_contract(payload: object, *, scene: str, label: str) -> dict:
    expected = {"scene": scene, **SCENE_CONTRACTS[scene]}
    if payload != expected:
        raise ValueError(f"{label} is not the exact preregistered {scene} contract")
    return deepcopy(expected)


def _validate_stage_a(
    *,
    scene: str,
    gate: dict,
    stage_b_inputs: dict[str, dict],
    registry: dict[Path, str],
) -> dict:
    stage_a_summary = gate.get("stage_a")
    if not isinstance(stage_a_summary, dict):
        raise ValueError(f"{scene} Stage-B gate lacks its Stage-A summary")
    stage_a_ref = _reference(
        {
            "path": stage_a_summary.get("gate_path"),
            "sha256": stage_a_summary.get("gate_sha256"),
        },
        label=f"{scene} Stage-A gate",
        registry=registry,
    )
    if stage_a_summary.get("passed") is not True or not _same_reference(
        stage_a_ref, stage_b_inputs["stage_a_gate"]
    ):
        raise ValueError(f"{scene} Stage-A summary/input lineage differs")
    payload = _json_object(stage_a_ref["path"], label=f"{scene} Stage-A gate")
    gates = payload.get("gates")
    if (
        payload.get("schema") != "lafgs_cycle_verified_fisher_stage_a_gate"
        or payload.get("version") != 1
        or payload.get("valid") is not True
        or payload.get("mapping_only") is not True
        or payload.get("uses_test_queries") is not False
        or payload.get("policy") != POLICY_NAME
        or payload.get("control_subset") != "attested_nearest_pairs_from_same_probe"
        or payload.get("stage_a_passed") is not True
        or payload.get("advance_to_reuse_only_track_build") is not True
        or payload.get("decision") != "GO_TO_TRACK_REUSE"
        or not isinstance(gates, dict)
        or set(gates) != STAGE_A_GATE_NAMES
        or not all(value is True for value in gates.values())
    ):
        raise ValueError(f"{scene} embedded Stage-A gate is not a valid 9/9 pass")
    _scene_contract(
        payload.get("scene_contract"), scene=scene, label=f"{scene} Stage-A"
    )
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != STAGE_A_INPUT_NAMES:
        raise ValueError(f"{scene} Stage-A input registry is incomplete")
    for name in sorted(STAGE_A_INPUT_NAMES):
        observed = _reference(
            inputs[name],
            label=f"{scene} Stage-A {name}",
            registry=registry,
        )
        if not _same_reference(observed, stage_b_inputs[name]):
            raise ValueError(f"{scene} Stage-A/{name} lineage differs from Stage-B")
    return {
        "path": stage_a_ref["path"],
        "sha256": stage_a_ref["sha256"],
        "payload": payload,
    }


def _validate_rebind(
    *,
    scene: str,
    payload: object,
    query_cache: dict,
    frozen_track: dict,
    registry: dict[Path, str],
) -> dict:
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "lafgs_equivalent_query_cache_rebind"
        or payload.get("version") != 1
        or payload.get("uses_test_queries") is not False
    ):
        raise ValueError(f"{scene} Track cache rebind is invalid")
    references = {}
    for name in (
        "equivalence_report",
        "parent_manifest",
        "refreshed_cache",
        "source_cache",
        "source_track_payload",
    ):
        references[name] = _reference(
            payload.get(name),
            label=f"{scene} Track cache rebind {name}",
            registry=registry,
        )
    if not _same_file_reference(
        references["refreshed_cache"], query_cache
    ) or not _same_file_reference(references["source_track_payload"], frozen_track):
        raise ValueError(f"{scene} Track cache rebind differs from frozen inputs")
    return deepcopy(payload)


def _validate_track_report(
    *,
    scene: str,
    role: str,
    report_ref: dict,
    factor_ref: dict,
    stage_a: dict,
    stage_b_inputs: dict[str, dict],
    registry: dict[Path, str],
) -> dict:
    report = _json_object(report_ref["path"], label=f"{scene} {role} report")
    contract = {"scene": scene, **SCENE_CONTRACTS[scene]}
    expected_policy = CONTROL_POLICY_NAME if role == "control" else POLICY_NAME
    expected_subset = (
        "attested_nearest_same_probe_control"
        if role == "control"
        else "cycle_verified_fisher_selection"
    )
    parameters = report.get("pair_policy_parameters")
    science = (
        parameters.get("track_science_contract")
        if isinstance(parameters, dict)
        else None
    )
    if (
        report.get("schema") != "lafgs_pair_policy_track_factor"
        or report.get("version") != 1
        or report.get("uses_test_queries") is not False
        or report.get("reuse_only") is not True
        or report.get("pair_policy") != expected_policy
        or report.get("scene_contract") != contract
        or report.get("mapping_keypoint_factor") != contract["mapping_keypoints"]
        or report.get("mapping_nms_radius") != contract["nms_radius"]
        or report.get("exact_pair_budget") != contract["pair_budget"]
        or Path(str(report.get("artifact", ""))).expanduser().resolve()
        != factor_ref["path"]
        or report.get("artifact_sha256") != factor_ref["sha256"]
        or report.get("probe_matcher") != MATCHER_CONTRACT
        or not isinstance(parameters, dict)
        or parameters.get("reuse_only") is not True
        or parameters.get("pair_subset_role") != expected_subset
        or parameters.get("probe_matcher") != MATCHER_CONTRACT
        or not isinstance(science, dict)
        or set(science) != TRACK_SCIENCE_KEYS
        or science.get("matcher") != MATCHER_CONTRACT
        or science.get("mapping_keypoints") != contract["mapping_keypoints"]
        or science.get("mapping_nms_radius") != contract["nms_radius"]
        or science.get("exact_pair_budget") != contract["pair_budget"]
    ):
        raise ValueError(f"{scene} {role} Track report contract is invalid")
    inputs = report.get("inputs")
    if (
        not isinstance(inputs, dict)
        or not TRACK_LINEAGE_NAMES.issubset(inputs)
        or inputs.get("pair_subset_role") != expected_subset
        or inputs.get("probe_matcher") != MATCHER_CONTRACT
    ):
        raise ValueError(f"{scene} {role} Track report lineage is incomplete")
    lineage = {}
    stage_b_names = {
        "query_cache",
        "pair_proposals",
        "pair_match_probe",
        "pair_selection",
        "stage_a_gate",
    }
    for name in sorted(TRACK_LINEAGE_NAMES):
        observed = _reference(
            inputs[name],
            label=f"{scene} {role} Track {name}",
            registry=registry,
        )
        lineage[name] = observed
        if name in stage_b_names and not _same_reference(
            observed, stage_b_inputs[name]
        ):
            raise ValueError(f"{scene} {role} Track/{name} lineage differs")
    if not _same_reference(lineage["stage_a_gate"], stage_a):
        raise ValueError(f"{scene} {role} Track does not bind the Stage-A gate")
    rebind = _validate_rebind(
        scene=scene,
        payload=inputs.get("equivalent_query_cache_rebind"),
        query_cache=lineage["query_cache"],
        frozen_track=lineage["frozen_track_payload"],
        registry=registry,
    )
    normalized_science = deepcopy(science)
    for name in SCENE_SPECIFIC_TRACK_FIELDS:
        normalized_science.pop(name)
    if normalized_science != COMPILED_TRACK_SCIENCE_CONTRACT:
        raise ValueError(f"{scene} {role} Track uses a different compiled contract")
    track = report.get("track")
    covariance = (
        track.get("triangulated_covariance_trace_m2", {}).get("p90")
        if isinstance(track, dict)
        else None
    )
    metrics = {
        "triangulated_tracks": track.get("triangulated_track_count")
        if isinstance(track, dict)
        else None,
        "broad_eligible_tracks": track.get("broad_eligible_track_count")
        if isinstance(track, dict)
        else None,
        "high_confidence_tracks": track.get("high_confidence_track_count")
        if isinstance(track, dict)
        else None,
        "mapping_query_with_broad_track_fraction": track.get(
            "mapping_query_with_broad_track_fraction"
        )
        if isinstance(track, dict)
        else None,
        "triangulated_covariance_p90_m2": covariance,
    }
    for name, value in metrics.items():
        if (
            value is None
            and role == "variant"
            and name == ("triangulated_covariance_p90_m2")
        ):
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{scene} {role} Track metric {name} is invalid")
        metrics[name] = float(value)
    coverage = metrics["mapping_query_with_broad_track_fraction"]
    if coverage is None or coverage > 1.0:
        raise ValueError(f"{scene} {role} Track mapping coverage is invalid")
    return {
        "payload": report,
        "lineage": lineage,
        "rebind": rebind,
        "parameters": parameters,
        "normalized_science": normalized_science,
        "metrics": metrics,
        "subset_role": expected_subset,
    }


def _comparison(control: float, variant: float | None) -> dict:
    if variant is None:
        return {"control": control, "variant": None, "delta": None, "ratio": None}
    return {
        "control": control,
        "variant": variant,
        "delta": variant - control,
        "ratio": None if control == 0 else variant / control,
    }


def _expected_stage_b_gates(*, control: dict, variant: dict) -> dict[str, bool]:
    covariance = variant["triangulated_covariance_p90_m2"]
    return {
        "triangulated_tracks_retain_98pct": variant["triangulated_tracks"]
        >= 0.98 * control["triangulated_tracks"],
        "broad_eligible_tracks_retain_98pct": variant["broad_eligible_tracks"]
        >= 0.98 * control["broad_eligible_tracks"],
        "high_confidence_tracks_retain_98pct": variant["high_confidence_tracks"]
        >= 0.98 * control["high_confidence_tracks"],
        "triangulated_covariance_p90_not_worse_5pct": covariance is not None
        and covariance <= 1.05 * control["triangulated_covariance_p90_m2"],
        "broad_mapping_query_coverage_not_lower": variant[
            "mapping_query_with_broad_track_fraction"
        ]
        >= control["mapping_query_with_broad_track_fraction"],
    }


def _compiled_identity(
    *,
    gate: dict,
    stage_a: dict,
    control: dict,
    variant: dict,
) -> dict:
    return {
        "algorithm": "p8_cycle_verified_fisher_v1",
        "policy": gate["policy"],
        "stage_a": {
            "schema": stage_a["payload"]["schema"],
            "version": stage_a["payload"]["version"],
            "control_subset": stage_a["payload"]["control_subset"],
            "gate_names": sorted(stage_a["payload"]["gates"]),
        },
        "selector": {
            "candidate_union_maximum_budget_multiple": 2.0,
            "maximum_cycle_reprojection_error_px": 2.0,
            "minimum_camera_degree": 1,
            "tie_break": "lexicographic_pair_index",
        },
        "stage_b": {
            "schema": gate["schema"],
            "version": gate["version"],
            "gate_names": sorted(gate["stage_b"]["gates"]),
        },
        "track": {
            "report_schema": control["payload"]["schema"],
            "report_version": control["payload"]["version"],
            "control_policy": control["payload"]["pair_policy"],
            "variant_policy": variant["payload"]["pair_policy"],
            "control_subset_role": control["subset_role"],
            "variant_subset_role": variant["subset_role"],
            "matcher": deepcopy(MATCHER_CONTRACT),
            "reuse_only": True,
            "normalized_science_contract": control["normalized_science"],
        },
    }


def _validate_scene_gate(
    *,
    scene: str,
    gate_path: Path,
    expected_gate_sha256: str,
    registry: dict[Path, str],
) -> dict:
    path = _attest_file(
        gate_path,
        expected_gate_sha256,
        label=f"{scene} Stage-B gate",
        registry=registry,
    )
    gate = _json_object(path, label=f"{scene} Stage-B gate")
    if (
        gate.get("schema") != "lafgs_cycle_verified_fisher_mechanism_gate"
        or gate.get("version") != 2
        or gate.get("valid") is not True
        or gate.get("mapping_only") is not True
        or gate.get("uses_test_queries") is not False
        or gate.get("policy") != POLICY_NAME
        or gate.get("requires_other_scene") is not True
    ):
        raise ValueError(f"{scene} is not a valid mapping-only Stage-B gate")
    contract = _scene_contract(
        gate.get("scene_contract"), scene=scene, label=f"{scene} Stage-B"
    )
    inputs = gate.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != STAGE_B_INPUT_NAMES:
        raise ValueError(f"{scene} Stage-B input registry is incomplete")
    input_refs = {
        name: _reference(
            inputs[name],
            label=f"{scene} Stage-B {name}",
            registry=registry,
        )
        for name in sorted(STAGE_B_INPUT_NAMES)
    }
    stage_a = _validate_stage_a(
        scene=scene,
        gate=gate,
        stage_b_inputs=input_refs,
        registry=registry,
    )
    control = _validate_track_report(
        scene=scene,
        role="control",
        report_ref=input_refs["control_report"],
        factor_ref=input_refs["control_factor"],
        stage_a=stage_a,
        stage_b_inputs=input_refs,
        registry=registry,
    )
    variant = _validate_track_report(
        scene=scene,
        role="variant",
        report_ref=input_refs["variant_report"],
        factor_ref=input_refs["variant_factor"],
        stage_a=stage_a,
        stage_b_inputs=input_refs,
        registry=registry,
    )
    control_parameters = deepcopy(control["parameters"])
    variant_parameters = deepcopy(variant["parameters"])
    control_parameters.pop("pair_subset_role")
    variant_parameters.pop("pair_subset_role")
    control_inputs = deepcopy(control["payload"]["inputs"])
    variant_inputs = deepcopy(variant["payload"]["inputs"])
    control_inputs.pop("pair_subset_role")
    variant_inputs.pop("pair_subset_role")
    if (
        control_parameters != variant_parameters
        or control_inputs != variant_inputs
        or control["rebind"] != variant["rebind"]
    ):
        raise ValueError(f"{scene} Track arms do not share exact frozen inputs")
    stage_b = gate.get("stage_b")
    gates = stage_b.get("gates") if isinstance(stage_b, dict) else None
    comparisons = stage_b.get("comparisons") if isinstance(stage_b, dict) else None
    if (
        not isinstance(stage_b, dict)
        or not isinstance(gates, dict)
        or set(gates) != STAGE_B_GATE_NAMES
        or any(type(value) is not bool for value in gates.values())
        or not isinstance(comparisons, dict)
        or set(comparisons) != STAGE_B_COMPARISON_NAMES
    ):
        raise ValueError(f"{scene} Stage-B scientific gates are malformed")
    expected_comparisons = {
        name: _comparison(control["metrics"][name], variant["metrics"][name])
        for name in STAGE_B_COMPARISON_NAMES
    }
    if comparisons != expected_comparisons:
        raise ValueError(f"{scene} Stage-B comparisons differ from Track reports")
    expected_scientific_gates = _expected_stage_b_gates(
        control=control["metrics"], variant=variant["metrics"]
    )
    if any(
        gates[name] is not expected
        for name, expected in expected_scientific_gates.items()
    ):
        raise ValueError(f"{scene} Stage-B gates differ from Track metrics")
    for name in (
        "control_probe_rows_reused",
        "variant_probe_rows_reused",
        "same_probe_matcher_contract",
    ):
        if gates[name] is not True:
            raise ValueError(f"{scene} Stage-B {name} is a lineage failure")
    scientific_pass = all(gates.values())
    expected_decision = (
        "SCENE_PASS_REQUIRES_OTHER_SCENE" if scientific_pass else "STOP_SCENE_MECHANISM"
    )
    if (
        stage_b.get("passed") is not scientific_pass
        or gate.get("scene_specific_mechanism_pass") is not scientific_pass
        or gate.get("decision") != expected_decision
    ):
        raise ValueError(f"{scene} Stage-B decision is inconsistent with its gates")
    identity = _compiled_identity(
        gate=gate, stage_a=stage_a, control=control, variant=variant
    )
    return {
        "path": path,
        "sha256": registry[path],
        "contract": contract,
        "passed": scientific_pass,
        "decision": expected_decision,
        "gates": deepcopy(gates),
        "identity": identity,
    }


def _atomic_json_save(payload: dict, output: Path, *, overwrite: bool) -> Path:
    output = Path(output).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def run(args: argparse.Namespace) -> dict:
    registry: dict[Path, str] = {}
    scenes = {
        "stairs": _validate_scene_gate(
            scene="stairs",
            gate_path=args.stairs_stage_b_gate,
            expected_gate_sha256=args.expected_stairs_stage_b_gate_sha256,
            registry=registry,
        ),
        "greatcourt": _validate_scene_gate(
            scene="greatcourt",
            gate_path=args.greatcourt_stage_b_gate,
            expected_gate_sha256=args.expected_greatcourt_stage_b_gate_sha256,
            registry=registry,
        ),
    }
    if scenes["stairs"]["path"] == scenes["greatcourt"]["path"]:
        raise ValueError("One Stage-B gate cannot represent both scene domains")
    identities = [scenes[name]["identity"] for name in ("stairs", "greatcourt")]
    if identities[0] != identities[1]:
        raise ValueError(
            "Stairs and GreatCourt do not share one compiled P8 policy identity"
        )
    identity = identities[0]
    identity_sha256 = hashlib.sha256(
        canonical_json(identity).encode("ascii")
    ).hexdigest()
    output = Path(args.output).expanduser().resolve()
    if output in registry:
        raise ValueError("Cross-scene output must not overwrite an attested input")
    # Rehash every recursively consumed file immediately before the decision is
    # persisted, closing the input-mutation window across the two scene audits.
    for path, expected in registry.items():
        if sha256_file(path) != expected:
            raise RuntimeError("A cross-scene input changed during aggregation")
    cross_scene_pass = all(scene["passed"] for scene in scenes.values())
    report = {
        "schema": "lafgs_cycle_verified_fisher_cross_scene_gate",
        "version": 1,
        "valid": True,
        "mapping_only": True,
        "uses_test_queries": False,
        "policy": POLICY_NAME,
        "compiled_policy_identity": identity,
        "compiled_policy_identity_sha256": identity_sha256,
        "scene_gates": {
            name: {
                "path": str(scene["path"]),
                "sha256": scene["sha256"],
                "scene_contract": scene["contract"],
                "embedded_stage_a_passed": True,
                "track_inputs_and_reuse_lineage_valid": True,
                "stage_b_gates": scene["gates"],
                "scene_specific_mechanism_pass": scene["passed"],
                "decision": scene["decision"],
            }
            for name, scene in scenes.items()
        },
        "cross_scene_gates": {
            "stairs_scene_mechanism_pass": scenes["stairs"]["passed"],
            "greatcourt_scene_mechanism_pass": scenes["greatcourt"]["passed"],
            "distinct_indoor_outdoor_scene_contracts": True,
            "same_compiled_policy_identity": True,
            "all_inputs_mapping_only_and_test_free": True,
        },
        "cross_scene_mechanism_pass": cross_scene_pass,
        "advance_to_fullchain_mapping_pose": cross_scene_pass,
        "authorizes_test": False,
        "decision": (
            "GO_TO_FULLCHAIN_MAPPING_POSE"
            if cross_scene_pass
            else "STOP_BEFORE_FULLCHAIN"
        ),
        "limitations": [
            "A Go authorizes only the preregistered mapping-only fullchain and "
            "mapping-pose Stage C; formal test remains forbidden.",
            "This gate does not claim pose improvement or authorize a method-default "
            "switch.",
        ],
    }
    saved = _atomic_json_save(report, output, overwrite=bool(args.overwrite))
    report["output"] = str(saved)
    report["output_sha256"] = sha256_file(saved)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["cross_scene_mechanism_pass"]:
        raise SystemExit(2)


def entrypoint(argv: Sequence[str] | None = None) -> None:
    try:
        main(argv)
    except SystemExit:
        raise
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    entrypoint()
