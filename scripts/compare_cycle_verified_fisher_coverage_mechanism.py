#!/usr/bin/env python3
"""Evaluate one completed paired P8 coverage-V2 Track build at Stage B."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from scripts.cycle_verified_fisher_cli_common import (
    atomic_json_save,
    attest_file,
    torch_load,
    validate_output_target,
)
from scripts.cycle_verified_fisher_coverage_track_common import (
    CONTROL_POLICY_NAME,
    CONTROL_SUBSET_ROLE,
    STAGE_B_PRODUCER_SCHEMA,
    STAGE_B_PRODUCER_SOURCE_PATHS,
    VARIANT_POLICY_NAME,
    VARIANT_SUBSET_ROLE,
    artifact_schema_contract,
    load_compiled_scene_inputs,
    load_completed_arms,
    preregistration,
    recursive_equal,
    reference_registry_unchanged,
    require_clean_identity,
    scene_preregistration,
    stage_b_producer_identity,
    track_metrics,
    required_artifact_keys,
    validate_code_identity,
    validate_completion_manifest,
)
from scripts.run_track_pair_factor import _track_report


SCHEMA = "lafgs_cycle_verified_fisher_coverage_mechanism_gate"
BASE_GATE_NAMES = {
    "triangulated_tracks_retain_98pct",
    "broad_eligible_tracks_retain_98pct",
    "high_confidence_tracks_retain_98pct",
    "triangulated_covariance_p90_not_worse_5pct",
    "broad_mapping_query_coverage_not_lower",
    "control_probe_rows_reused",
    "variant_probe_rows_reused",
    "same_probe_matcher_contract",
}
STAIRS_RETENTION_GATE_NAMES = {
    "v1_triangulated_tracks_retain_98pct",
    "v1_broad_eligible_tracks_retain_98pct",
    "v1_high_confidence_tracks_retain_98pct",
    "v1_triangulated_covariance_p90_not_worse_5pct",
    "v1_broad_mapping_query_coverage_not_lower",
}
SCENE_SPECIFIC_SCIENCE_FIELDS = {
    "mapping_keypoints",
    "exact_pair_budget",
    "maximum_reprojection_px",
    "maximum_covariance_trace_m2",
    "maximum_rendered_depth_residual_m",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--completion-manifest", type=Path, required=True)
    parser.add_argument("--expected-completion-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _comparison(control: float | int, variant: float | int | None) -> dict:
    if variant is None:
        return {
            "control": control,
            "variant": None,
            "delta": None,
            "ratio": None,
        }
    return {
        "control": control,
        "variant": variant,
        "delta": variant - control,
        "ratio": None if control == 0 else variant / control,
    }


def _validate_metrics(metrics: dict, *, role: str) -> None:
    expected = {
        "triangulated_tracks",
        "broad_eligible_tracks",
        "high_confidence_tracks",
        "triangulated_covariance_p90_m2",
        "mapping_query_with_broad_track_fraction",
    }
    if not isinstance(metrics, dict) or set(metrics) != expected:
        raise ValueError(f"{role} Track metrics are incomplete")
    for name in (
        "triangulated_tracks",
        "broad_eligible_tracks",
        "high_confidence_tracks",
    ):
        if isinstance(metrics[name], bool) or not isinstance(metrics[name], int):
            raise ValueError(f"{role} {name} is not an exact integer count")
        if metrics[name] < 0:
            raise ValueError(f"{role} {name} is negative")
    covariance = metrics["triangulated_covariance_p90_m2"]
    if covariance is not None and (
        isinstance(covariance, bool)
        or not isinstance(covariance, (int, float))
        or not math.isfinite(float(covariance))
        or float(covariance) < 0.0
    ):
        raise ValueError(f"{role} covariance p90 is invalid")
    coverage = metrics["mapping_query_with_broad_track_fraction"]
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, (int, float))
        or not math.isfinite(float(coverage))
        or float(coverage) < 0.0
        or float(coverage) > 1.0
    ):
        raise ValueError(f"{role} broad-query coverage is invalid")


def _base_gates(*, control: dict, variant: dict) -> dict[str, bool]:
    thresholds = preregistration()["stage_b_thresholds"]
    covariance = variant["triangulated_covariance_p90_m2"]
    return {
        "triangulated_tracks_retain_98pct": variant["triangulated_tracks"]
        >= thresholds["minimum_triangulated_track_retention_ratio"]
        * control["triangulated_tracks"],
        "broad_eligible_tracks_retain_98pct": variant["broad_eligible_tracks"]
        >= thresholds["minimum_broad_eligible_track_retention_ratio"]
        * control["broad_eligible_tracks"],
        "high_confidence_tracks_retain_98pct": variant["high_confidence_tracks"]
        >= thresholds["minimum_high_confidence_track_retention_ratio"]
        * control["high_confidence_tracks"],
        "triangulated_covariance_p90_not_worse_5pct": covariance is not None
        and covariance
        <= thresholds["maximum_triangulated_covariance_p90_ratio"]
        * control["triangulated_covariance_p90_m2"],
        "broad_mapping_query_coverage_not_lower": variant[
            "mapping_query_with_broad_track_fraction"
        ]
        - control["mapping_query_with_broad_track_fraction"]
        >= thresholds["minimum_broad_query_coverage_delta"],
        "control_probe_rows_reused": True,
        "variant_probe_rows_reused": True,
        "same_probe_matcher_contract": True,
    }


def _baseline_reference(name: str) -> dict:
    reference = preregistration()["stairs_v1_reference"].get(name)
    if not isinstance(reference, dict):
        raise RuntimeError(f"Stairs V1 preregistration lacks {name}")
    return deepcopy(reference)


def _attest_baseline(name: str) -> tuple[Path, dict]:
    reference = _baseline_reference(name)
    path = attest_file(
        Path(reference["path"]), reference["sha256"], label=f"Stairs V1 {name}"
    )
    return path, reference


def _control_scientific_projection_status(
    *, v2_control: dict, v1_control_factor: dict, v1_control_metrics: dict
) -> dict[str, bool]:
    """Compare only the preregistered scientific projection of the control."""
    v2_factor = v2_control["factor"]["payload"]
    tensor_fields = ("query_bins", "tracks", "track_geometry")
    tensor_parity = all(
        recursive_equal(v2_factor.get(name), v1_control_factor.get(name))
        for name in tensor_fields
    ) and recursive_equal(
        v2_factor.get("pair_sidecar", {}).get("pair"),
        v1_control_factor.get("pair_sidecar", {}).get("pair"),
    )
    return {
        "tensor_parity": tensor_parity,
        "metric_parity": recursive_equal(v2_control["metrics"], v1_control_metrics),
    }


def _stairs_retention_gates(*, metrics: dict, thresholds: dict) -> dict[str, bool]:
    covariance = metrics["triangulated_covariance_p90_m2"]
    return {
        "v1_triangulated_tracks_retain_98pct": metrics["triangulated_tracks"]
        >= thresholds["triangulated_tracks_at_least"],
        "v1_broad_eligible_tracks_retain_98pct": metrics[
            "broad_eligible_tracks"
        ]
        >= thresholds["broad_eligible_tracks_at_least"],
        "v1_high_confidence_tracks_retain_98pct": metrics[
            "high_confidence_tracks"
        ]
        >= thresholds["high_confidence_tracks_at_least"],
        "v1_triangulated_covariance_p90_not_worse_5pct": covariance is not None
        and covariance <= thresholds["triangulated_covariance_p90_m2_at_most"],
        "v1_broad_mapping_query_coverage_not_lower": metrics[
            "mapping_query_with_broad_track_fraction"
        ]
        == thresholds["mapping_query_with_broad_track_fraction_exact"],
    }


def _load_stairs_v1_reference(*, v2_control: dict, v2_variant: dict) -> dict:
    control_path, control_ref = _attest_baseline("control_factor")
    control_report_path, control_report_ref = _attest_baseline("control_report")
    variant_path, variant_ref = _attest_baseline("variant_factor")
    variant_report_path, variant_report_ref = _attest_baseline("variant_report")
    gate_path, gate_ref = _attest_baseline("stage_b_gate")
    control_factor = torch_load(control_path)
    variant_factor = torch_load(variant_path)
    control_report = json.loads(control_report_path.read_text())
    variant_report = json.loads(variant_report_path.read_text())
    gate = json.loads(gate_path.read_text())
    if (
        control_factor.get("schema") != "lafgs_pair_policy_track_factor"
        or control_factor.get("version") != 1
        or control_factor.get("uses_test_queries") is not False
        or control_factor.get("pair_policy")
        != "cycle_verified_fisher_nearest_control"
        or variant_factor.get("schema") != "lafgs_pair_policy_track_factor"
        or variant_factor.get("version") != 1
        or variant_factor.get("uses_test_queries") is not False
        or variant_factor.get("pair_policy") != "cycle_verified_fisher"
    ):
        raise ValueError("Frozen Stairs V1 Track factors are invalid")
    expected_metrics = preregistration()["stairs_v1_reference"]
    control_metrics = track_metrics(control_factor, query_count=2000)
    variant_metrics = track_metrics(variant_factor, query_count=2000)
    if (
        control_metrics != expected_metrics["control_metrics"]
        or variant_metrics != expected_metrics["variant_metrics"]
    ):
        raise ValueError("Stairs V1 metrics differ from the factor recomputation")
    for role, report, factor, factor_ref in (
        ("control", control_report, control_factor, control_ref),
        ("variant", variant_report, variant_factor, variant_ref),
    ):
        expected_track = _track_report(
            factor["tracks"], factor["track_geometry"], query_count=2000
        )
        if (
            report.get("schema") != "lafgs_pair_policy_track_factor"
            or report.get("uses_test_queries") is not False
            or Path(str(report.get("artifact", ""))).resolve()
            != Path(factor_ref["path"]).resolve()
            or report.get("artifact_sha256") != factor_ref["sha256"]
            or not recursive_equal(report.get("track"), expected_track)
        ):
            raise ValueError(f"Stairs V1 {role} report differs from its factor")
    gate_inputs = gate.get("inputs")
    expected_gate_refs = {
        "control_factor": control_ref,
        "control_report": control_report_ref,
        "variant_factor": variant_ref,
        "variant_report": variant_report_ref,
    }
    if (
        gate.get("schema") != "lafgs_cycle_verified_fisher_mechanism_gate"
        or gate.get("version") != 2
        or gate.get("uses_test_queries") is not False
        or gate.get("mapping_only") is not True
        or gate.get("valid") is not True
        or gate.get("scene_specific_mechanism_pass") is not True
        or gate.get("decision") != "SCENE_PASS_REQUIRES_OTHER_SCENE"
        or not isinstance(gate.get("stage_b", {}).get("gates"), dict)
        or not all(gate["stage_b"]["gates"].values())
        or not isinstance(gate_inputs, dict)
    ):
        raise ValueError("Frozen Stairs V1 Stage-B gate is not a Pass")
    for name, reference in expected_gate_refs.items():
        observed = gate_inputs.get(name)
        if (
            not isinstance(observed, dict)
            or Path(str(observed.get("path", ""))).resolve()
            != Path(reference["path"]).resolve()
            or observed.get("sha256") != reference["sha256"]
        ):
            raise ValueError(f"Stairs V1 Stage-B gate changed {name}")
    parity = _control_scientific_projection_status(
        v2_control=v2_control,
        v1_control_factor=control_factor,
        v1_control_metrics=control_metrics,
    )
    if not all(parity.values()):
        raise ValueError(
            "Stairs V2 nearest control differs from the frozen V1 scientific projection"
        )
    parity_gate = {
        "v1_nearest_control_scientific_projection_exact": True
    }
    thresholds = expected_metrics["retention_gates"]
    v2_metrics = v2_variant["metrics"]
    gates = _stairs_retention_gates(metrics=v2_metrics, thresholds=thresholds)
    return {
        "references": {
            "stage_b_gate": gate_ref,
            "control_factor": control_ref,
            "control_report": control_report_ref,
            "variant_factor": variant_ref,
            "variant_report": variant_report_ref,
        },
        "control_metrics": control_metrics,
        "variant_metrics": variant_metrics,
        "thresholds": deepcopy(thresholds),
        "gates": gates,
        "control_parity_gate": parity_gate,
        "v2_control_exact_scientific_tensor_parity": parity["tensor_parity"],
        "v2_control_exact_metric_parity": parity["metric_parity"],
    }


def _compiled_identity(*, arms: dict) -> dict:
    control_science = deepcopy(arms["control"]["science"])
    variant_science = deepcopy(arms["variant"]["science"])
    if control_science != variant_science:
        raise ValueError("Paired Track arms use different science contracts")
    for name in SCENE_SPECIFIC_SCIENCE_FIELDS:
        control_science.pop(name)
    return {
        "algorithm": "p8_cycle_verified_fisher_coverage_v2",
        "control_policy": CONTROL_POLICY_NAME,
        "variant_policy": VARIANT_POLICY_NAME,
        "control_subset_role": CONTROL_SUBSET_ROLE,
        "variant_subset_role": VARIANT_SUBSET_ROLE,
        "base_gate_names": sorted(BASE_GATE_NAMES),
        "stairs_retention_gate_names": sorted(STAIRS_RETENTION_GATE_NAMES),
        "normalized_track_science_contract": control_science,
        "track_producer_identity": deepcopy(
            arms["control"]["factor"]["payload"]["track_producer_identity"]
        ),
    }


def evaluate_scene(
    *, scene: str, completion_manifest: Path, expected_completion_sha256: str
) -> dict:
    registry = load_compiled_scene_inputs(scene)
    completion = validate_completion_manifest(
        path=completion_manifest,
        expected_sha256=expected_completion_sha256,
        expected_scene=scene,
    )
    arms = load_completed_arms(completion=completion, registry=registry)
    control = arms["control"]["metrics"]
    variant = arms["variant"]["metrics"]
    _validate_metrics(control, role="control")
    _validate_metrics(variant, role="variant")
    if control["triangulated_covariance_p90_m2"] is None:
        raise ValueError("Control covariance p90 is undefined")
    comparisons = {
        name: _comparison(control[name], variant[name]) for name in control
    }
    base_gates = _base_gates(control=control, variant=variant)
    if set(base_gates) != BASE_GATE_NAMES:
        raise RuntimeError("Compiled base Stage-B gate set changed")
    stairs = (
        _load_stairs_v1_reference(
            v2_control=arms["control"], v2_variant=arms["variant"]
        )
        if scene == "stairs"
        else None
    )
    retention_gates = {} if stairs is None else stairs["gates"]
    parity_gate = {} if stairs is None else stairs["control_parity_gate"]
    if scene == "stairs" and set(retention_gates) != STAIRS_RETENTION_GATE_NAMES:
        raise RuntimeError("Compiled Stairs V1 retention gate set changed")
    passed = (
        all(base_gates.values())
        and all(parity_gate.values())
        and all(retention_gates.values())
    )
    reference_registry_unchanged(registry)
    return {
        "registry": registry,
        "completion": completion,
        "arms": arms,
        "control": control,
        "variant": variant,
        "comparisons": comparisons,
        "base_gates": base_gates,
        "stairs": stairs,
        "stairs_v1_retention_gates": retention_gates,
        "stairs_control_parity_gate": parity_gate,
        "passed": passed,
        "compiled_identity": _compiled_identity(arms=arms),
    }


def gate_payload(*, scene: str, evaluation: dict, producer: dict) -> dict:
    registry = evaluation["registry"]
    completion = evaluation["completion"]
    stairs = evaluation["stairs"]
    passed = evaluation["passed"]
    report = {
        "schema": SCHEMA,
        "version": 1,
        "uses_test_queries": False,
        "mapping_only": True,
        "valid": True,
        "scene": scene,
        "policy": VARIANT_POLICY_NAME,
        "control_policy": CONTROL_POLICY_NAME,
        "scene_contract": {"scene": scene, **scene_preregistration(scene)},
        "stage_a": {
            "cross_scene_gate": {
                "path": str(registry["cross_scene_stage_a_gate"]["path"]),
                "sha256": registry["cross_scene_stage_a_gate"]["sha256"],
            },
            "scene_gate": {
                "path": str(registry["scene_stage_a_gate"]["path"]),
                "sha256": registry["scene_stage_a_gate"]["sha256"],
            },
            "passed": True,
        },
        "inputs": {
            "completion_manifest": {
                "path": str(completion["path"]),
                "sha256": completion["sha256"],
            }
        },
        "paired_track": {
            "run_uuid": completion["payload"]["run_uuid"],
            "track_producer_identity": deepcopy(completion["producer"]),
            "completion_manifest_validated": True,
            "partial_or_resume_rejected": True,
            "greatcourt_stage_b_parent": deepcopy(
                completion["payload"]["inputs"].get("greatcourt_stage_b_parent")
            ),
        },
        "stage_b": {
            "comparisons": deepcopy(evaluation["comparisons"]),
            "base_gates": deepcopy(evaluation["base_gates"]),
            "base_passed": all(evaluation["base_gates"].values()),
            "stairs_v1_retention_gates": deepcopy(
                evaluation["stairs_v1_retention_gates"]
            ),
            "stairs_control_parity_gate": deepcopy(
                evaluation["stairs_control_parity_gate"]
            ),
            "stairs_v1_retention_passed": (
                None
                if scene == "greatcourt"
                else all(evaluation["stairs_v1_retention_gates"].values())
            ),
            "passed": passed,
        },
        "stairs_v1_reference": (
            None
            if stairs is None
            else {
                "references": deepcopy(stairs["references"]),
                "control_metrics": deepcopy(stairs["control_metrics"]),
                "variant_metrics": deepcopy(stairs["variant_metrics"]),
                "thresholds": deepcopy(stairs["thresholds"]),
                "v2_control_exact_scientific_tensor_parity": stairs[
                    "v2_control_exact_scientific_tensor_parity"
                ],
                "v2_control_exact_metric_parity": stairs[
                    "v2_control_exact_metric_parity"
                ],
            }
        ),
        "compiled_identity": deepcopy(evaluation["compiled_identity"]),
        "stage_b_producer_identity": deepcopy(producer),
        "scene_specific_mechanism_pass": passed,
        "requires_other_scene": True,
        "advance_to_existing_fullchain": False,
        "advance_to_mapping_pose": False,
        "authorizes_test": False,
        "changes_method_default": False,
        "decision": (
            "SCENE_PASS_REQUIRES_OTHER_SCENE"
            if passed
            else "STOP_SCENE_MECHANISM"
        ),
    }
    # Do not duplicate the many absolute artifact paths inside scene_contract.
    report["scene_contract"] = {
        "scene": scene,
        "mapping_keypoints": registry["compiled"]["mapping_keypoints"],
        "nms_radius": registry["compiled"]["mapping_nms_radius"],
        "pair_budget": registry["compiled"]["exact_pair_budget"],
        "candidate_pair_count": registry["compiled"]["candidate_pair_count"],
        "candidate_component_count": registry["compiled"][
            "candidate_component_count"
        ],
    }
    if set(report) != required_artifact_keys("scene_stage_b_gate"):
        raise RuntimeError("Compiled scene Stage-B output schema changed")
    return report


def validate_stage_b_gate(
    *, scene: str, path: Path, expected_sha256: str
) -> dict:
    path = attest_file(path, expected_sha256, label=f"{scene} V2 Stage-B gate")
    observed = json.loads(path.read_text())
    contract = artifact_schema_contract("scene_stage_b_gate")
    if (
        set(observed) != required_artifact_keys("scene_stage_b_gate")
        or observed.get("schema") != contract["schema"]
        or observed.get("version") != contract["version"]
        or observed.get("uses_test_queries") is not False
        or observed.get("mapping_only") is not True
        or observed.get("valid") is not True
        or observed.get("scene") != scene
        or set(observed.get("inputs", {})) != {"completion_manifest"}
    ):
        raise ValueError(f"{scene} V2 Stage-B gate is invalid")
    producer = validate_code_identity(
        observed.get("stage_b_producer_identity"),
        schema=STAGE_B_PRODUCER_SCHEMA,
        algorithm="p8_cycle_verified_fisher_coverage_v2_stage_b",
        entrypoint=(
            "python -m scripts.compare_cycle_verified_fisher_coverage_mechanism"
        ),
        source_paths=STAGE_B_PRODUCER_SOURCE_PATHS,
        device=None,
        label=f"{scene} Stage-B",
    )
    completion_ref = observed["inputs"]["completion_manifest"]
    if not isinstance(completion_ref, dict):
        raise ValueError(f"{scene} Stage-B completion reference is invalid")
    evaluation = evaluate_scene(
        scene=scene,
        completion_manifest=Path(str(completion_ref.get("path", ""))),
        expected_completion_sha256=str(completion_ref.get("sha256", "")),
    )
    expected = gate_payload(scene=scene, evaluation=evaluation, producer=producer)
    if observed != expected:
        raise ValueError(f"{scene} Stage-B gate differs from recursive replay")
    return {
        "path": path,
        "sha256": sha256_file(path),
        "payload": observed,
        "evaluation": evaluation,
    }


def run(args: argparse.Namespace) -> dict:
    evaluation = evaluate_scene(
        scene=args.scene,
        completion_manifest=args.completion_manifest,
        expected_completion_sha256=args.expected_completion_manifest_sha256,
    )
    producer = stage_b_producer_identity()
    require_clean_identity(producer, label="P8 coverage-V2 Stage-B producer")
    report = gate_payload(scene=args.scene, evaluation=evaluation, producer=producer)
    output = validate_output_target(
        args.output,
        protected_paths=[
            evaluation["completion"]["path"],
            *(
                value["path"]
                for value in evaluation["completion"]["artifacts"].values()
            ),
        ],
    )
    if output.exists():
        raise FileExistsError("P8 V2 Stage-B output already exists")
    atomic_json_save(report, output, overwrite=False)
    return {
        **report,
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def main(argv: Sequence[str] | None = None) -> None:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["scene_specific_mechanism_pass"]:
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
