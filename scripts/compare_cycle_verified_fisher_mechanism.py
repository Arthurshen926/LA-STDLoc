#!/usr/bin/env python3
"""Compare P8 Stage-A/B evidence with strict lineage and STOP/GO semantics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import POLICY_NAME
from scripts.cycle_verified_fisher_cli_common import (
    assert_selection_metrics,
    atomic_json_save,
    attest_file,
    evaluate_pair_subsets,
    load_mapping_cache,
    load_probe,
    load_selection,
    load_track_factor,
    selection_pairs,
    validate_output_target,
    validate_scene_contract,
    validate_variant_reuse_lineage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-probe-content-sha256", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-selection-content-sha256", required=True)
    parser.add_argument("--control-factor", type=Path, required=True)
    parser.add_argument("--expected-control-factor-sha256", required=True)
    parser.add_argument("--control-report", type=Path, required=True)
    parser.add_argument("--expected-control-report-sha256", required=True)
    parser.add_argument("--variant-factor", type=Path, required=True)
    parser.add_argument("--expected-variant-factor-sha256", required=True)
    parser.add_argument("--variant-report", type=Path, required=True)
    parser.add_argument("--expected-variant-report-sha256", required=True)
    parser.add_argument("--expected-query-names-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--expected-candidate-pair-count", type=int, required=True)
    parser.add_argument("--expected-candidate-components", type=int, required=True)
    parser.add_argument(
        "--maximum-cycle-reprojection-error-px", type=float, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_report(
    *,
    path: Path,
    expected_sha256: str,
    factor: dict,
    expected_policy: str,
    cache: dict,
    mapping_keypoints: int,
    nms_radius: int,
    pair_budget: int,
) -> dict:
    path = attest_file(path, expected_sha256, label=f"{expected_policy} report")
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "lafgs_pair_policy_track_factor"
        or int(payload.get("version", -1)) != 1
        or payload.get("uses_test_queries") is not False
        or payload.get("pair_policy") != expected_policy
        or int(payload.get("mapping_keypoint_factor", -1)) != int(mapping_keypoints)
        or int(payload.get("mapping_nms_radius", -1)) != int(nms_radius)
        or int(payload.get("exact_pair_budget", -1)) != int(pair_budget)
        or int(payload.get("mapping_query_count", -1)) != len(cache["names"])
        or payload.get("query_names_sha256") != cache["query_names_sha256"]
        or Path(str(payload.get("artifact", ""))).resolve() != factor["path"]
        or payload.get("artifact_sha256") != factor["sha256"]
    ):
        raise ValueError(f"{expected_policy} report differs from its factor contract")
    inputs = payload.get("inputs")
    query_cache = inputs.get("query_cache") if isinstance(inputs, dict) else None
    if (
        not isinstance(query_cache, dict)
        or Path(str(query_cache.get("path", ""))).resolve() != cache["path"]
        or query_cache.get("sha256") != cache["sha256"]
    ):
        raise ValueError(f"{expected_policy} report names a different query cache")
    track = payload.get("track")
    if not isinstance(track, dict):
        raise ValueError(f"{expected_policy} report lacks Track mechanism metrics")
    return {"path": path, "sha256": sha256_file(path), "payload": payload}


def _finite_number(payload: dict, *path: str) -> float:
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing mechanism metric: {'/'.join(path)}")
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Mechanism metric is not finite: {'/'.join(path)}")
    return result


def _comparison(control: float, variant: float | None) -> dict:
    if variant is None:
        return {"control": control, "variant": None, "delta": None, "ratio": None}
    return {
        "control": control,
        "variant": variant,
        "delta": variant - control,
        "ratio": None if control == 0 else variant / control,
    }


def _track_metrics(report: dict, *, allow_undefined_covariance: bool) -> dict:
    result = {
        "triangulated_tracks": _finite_number(
            report, "track", "triangulated_track_count"
        ),
        "broad_eligible_tracks": _finite_number(
            report, "track", "broad_eligible_track_count"
        ),
        "high_confidence_tracks": _finite_number(
            report, "track", "high_confidence_track_count"
        ),
        "mapping_query_with_broad_track_fraction": _finite_number(
            report, "track", "mapping_query_with_broad_track_fraction"
        ),
    }
    covariance = report.get("track", {}).get(
        "triangulated_covariance_trace_m2", {}
    ).get("p90")
    if covariance is None and allow_undefined_covariance:
        result["triangulated_covariance_p90_m2"] = None
    elif covariance is None:
        raise ValueError("Control triangulated covariance p90 is undefined")
    else:
        result["triangulated_covariance_p90_m2"] = float(covariance)
        if not math.isfinite(result["triangulated_covariance_p90_m2"]):
            raise ValueError("Triangulated covariance p90 is not finite")
    if any(
        result[name] < 0
        for name in (
            "triangulated_tracks",
            "broad_eligible_tracks",
            "high_confidence_tracks",
        )
    ):
        raise ValueError("Track count metrics must be non-negative")
    coverage = result["mapping_query_with_broad_track_fraction"]
    if coverage < 0.0 or coverage > 1.0:
        raise ValueError("Mapping-query Track coverage must lie in [0, 1]")
    if (
        result["triangulated_covariance_p90_m2"] is not None
        and result["triangulated_covariance_p90_m2"] < 0.0
    ):
        raise ValueError("Triangulated covariance p90 must be non-negative")
    return result


def run(args: argparse.Namespace) -> dict:
    contract = validate_scene_contract(
        scene=args.scene,
        mapping_keypoints=args.expected_mapping_keypoints,
        nms_radius=args.expected_nms_radius,
        pair_budget=args.expected_pair_budget,
        candidate_pair_count=args.expected_candidate_pair_count,
        candidate_component_count=args.expected_candidate_components,
    )
    if float(args.maximum_cycle_reprojection_error_px) != 2.0:
        raise ValueError("P8 V1 cycle reprojection threshold is frozen at 2.0 px")
    cache = load_mapping_cache(
        path=args.query_cache,
        expected_file_sha256=args.expected_query_cache_sha256,
        expected_query_names_sha256=args.expected_query_names_sha256,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
    )
    probe = load_probe(
        path=args.probe,
        expected_file_sha256=args.expected_probe_sha256,
        expected_content_sha256=args.expected_probe_content_sha256,
        cache=cache,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_candidate_pair_count=args.expected_candidate_pair_count,
    )
    selection = load_selection(
        path=args.selection,
        expected_file_sha256=args.expected_selection_sha256,
        expected_content_sha256=args.expected_selection_content_sha256,
        probe=probe,
        expected_pair_budget=args.expected_pair_budget,
    )
    factor_common = {
        "expected_query_names": cache["names"],
        "expected_query_names_sha256": cache["query_names_sha256"],
        "expected_query_cache_path": cache["path"],
        "expected_query_cache_sha256": cache["sha256"],
        "expected_mapping_keypoints": args.expected_mapping_keypoints,
        "expected_nms_radius": args.expected_nms_radius,
        "expected_pair_budget": args.expected_pair_budget,
    }
    control_factor = load_track_factor(
        path=args.control_factor,
        expected_file_sha256=args.expected_control_factor_sha256,
        expected_policy="nearest",
        **factor_common,
    )
    variant_factor = load_track_factor(
        path=args.variant_factor,
        expected_file_sha256=args.expected_variant_factor_sha256,
        expected_policy=POLICY_NAME,
        **factor_common,
    )
    validate_variant_reuse_lineage(
        factor=variant_factor["payload"], probe=probe, selection=selection
    )
    control_report = _load_report(
        path=args.control_report,
        expected_sha256=args.expected_control_report_sha256,
        factor=control_factor,
        expected_policy="nearest",
        cache=cache,
        mapping_keypoints=args.expected_mapping_keypoints,
        nms_radius=args.expected_nms_radius,
        pair_budget=args.expected_pair_budget,
    )
    variant_report = _load_report(
        path=args.variant_report,
        expected_sha256=args.expected_variant_report_sha256,
        factor=variant_factor,
        expected_policy=POLICY_NAME,
        cache=cache,
        mapping_keypoints=args.expected_mapping_keypoints,
        nms_radius=args.expected_nms_radius,
        pair_budget=args.expected_pair_budget,
    )

    stage_a_evaluations = evaluate_pair_subsets(
        probe=probe["payload"],
        cache=cache,
        subsets={
            "control": control_factor["pairs"],
            "variant": selection_pairs(selection["payload"]),
        },
        maximum_reprojection_error_px=args.maximum_cycle_reprojection_error_px,
    )
    control_stage_a = stage_a_evaluations["control"]
    variant_stage_a = stage_a_evaluations["variant"]
    assert_selection_metrics(selection["payload"], variant_stage_a)
    selected_graph = selection["payload"]["graph"]
    candidate_graph = selection["payload"]["candidate_graph"]
    stage_a_comparisons = {
        name: _comparison(control_stage_a[name], variant_stage_a[name])
        for name in (
            "confidence_weighted_fisher_utility_sum",
            "completed_verified_keypoint_triangle_count",
            "completed_verified_triangle_camera_fraction",
        )
    }
    stage_a_gates = {
        "candidate_union_exact_and_bounded": (
            int(selection["payload"]["candidate_pair_count"])
            == int(args.expected_candidate_pair_count)
            and int(args.expected_candidate_pair_count)
            <= 2 * int(args.expected_pair_budget)
        ),
        "selected_exact_pair_budget": int(selection["payload"]["exact_pair_budget"])
        == int(args.expected_pair_budget),
        "candidate_components_exact": int(candidate_graph["component_count"])
        == int(args.expected_candidate_components),
        "selected_components_preserved": int(selected_graph["component_count"])
        == int(candidate_graph["component_count"]),
        "selected_zero_isolates": int(selected_graph["isolated_camera_count"]) == 0,
        "selected_minimum_degree_at_least_one": int(selected_graph["minimum_degree"])
        >= 1,
        "verified_fisher_utility_improves_5pct": (
            variant_stage_a["confidence_weighted_fisher_utility_sum"]
            >= 1.05 * control_stage_a["confidence_weighted_fisher_utility_sum"]
        ),
        "verified_triangles_retain_98pct": (
            variant_stage_a["completed_verified_keypoint_triangle_count"]
            >= 0.98 * control_stage_a["completed_verified_keypoint_triangle_count"]
        ),
        "verified_triangle_camera_fraction_not_lower": (
            variant_stage_a["completed_verified_triangle_camera_fraction"]
            >= control_stage_a["completed_verified_triangle_camera_fraction"]
        ),
    }

    control_track = _track_metrics(
        control_report["payload"], allow_undefined_covariance=False
    )
    variant_track = _track_metrics(
        variant_report["payload"], allow_undefined_covariance=True
    )
    stage_b_comparisons = {
        name: _comparison(control_track[name], variant_track[name])
        for name in control_track
    }
    stage_b_gates = {
        "triangulated_tracks_retain_98pct": variant_track["triangulated_tracks"]
        >= 0.98 * control_track["triangulated_tracks"],
        "broad_eligible_tracks_retain_98pct": variant_track[
            "broad_eligible_tracks"
        ]
        >= 0.98 * control_track["broad_eligible_tracks"],
        "high_confidence_tracks_retain_98pct": variant_track[
            "high_confidence_tracks"
        ]
        >= 0.98 * control_track["high_confidence_tracks"],
        "triangulated_covariance_p90_not_worse_5pct": (
            variant_track["triangulated_covariance_p90_m2"] is not None
            and variant_track["triangulated_covariance_p90_m2"]
            <= 1.05 * control_track["triangulated_covariance_p90_m2"]
        ),
        "broad_mapping_query_coverage_not_lower": variant_track[
            "mapping_query_with_broad_track_fraction"
        ]
        >= control_track["mapping_query_with_broad_track_fraction"],
        "selected_probe_rows_reused": True,
    }
    stage_a_passed = all(stage_a_gates.values())
    stage_b_passed = all(stage_b_gates.values())
    passed = stage_a_passed and stage_b_passed
    artifacts = {
        "query_cache": cache,
        "pair_match_probe": probe,
        "pair_selection": selection,
        "control_factor": control_factor,
        "control_report": control_report,
        "variant_factor": variant_factor,
        "variant_report": variant_report,
    }
    output_target = validate_output_target(
        args.output, protected_paths=[artifact["path"] for artifact in artifacts.values()]
    )
    for artifact in artifacts.values():
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("A frozen input changed during mechanism comparison")
    report = {
        "schema": "lafgs_cycle_verified_fisher_mechanism_gate",
        "version": 1,
        "uses_test_queries": False,
        "valid": True,
        "policy": POLICY_NAME,
        "scene_contract": contract,
        "stage_a": {
            "control": control_stage_a,
            "variant": variant_stage_a,
            "comparisons": stage_a_comparisons,
            "gates": stage_a_gates,
            "passed": stage_a_passed,
        },
        "stage_b": {
            "comparisons": stage_b_comparisons,
            "gates": stage_b_gates,
            "passed": stage_b_passed,
        },
        "mechanism_gate_passed": passed,
        "advance_to_fullchain_mapping_pose": passed,
        "decision": "GO_TO_FULLCHAIN" if passed else "STOP_BEFORE_FULLCHAIN",
        "inputs": {
            name: {
                "path": str(artifact["path"]),
                "sha256": artifact["sha256"],
                **(
                    {"content_sha256": artifact["content_sha256"]}
                    if "content_sha256" in artifact
                    else {}
                ),
            }
            for name, artifact in artifacts.items()
        },
        "limitations": [
            "A two-scene mechanism pass authorizes fullchain/mapping-pose validation; it is not a pose claim."
        ],
    }
    output = atomic_json_save(
        report, output_target, overwrite=bool(args.overwrite)
    )
    report["output"] = str(output)
    report["output_sha256"] = sha256_file(output)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["mechanism_gate_passed"]:
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
