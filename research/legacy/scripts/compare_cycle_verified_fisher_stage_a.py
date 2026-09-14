#!/usr/bin/env python3
"""Gate P8 selection against the nearest subset from the same match probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from scripts.cycle_verified_fisher_cli_common import (
    add_mapping_scope_arguments,
    assert_selection_metrics,
    atomic_json_save,
    evaluate_pair_subsets,
    load_mapping_cache,
    load_probe,
    load_proposals,
    load_selection,
    mapping_scope_kwargs,
    selection_pairs,
    validate_output_target,
    validate_probe_proposal_lineage,
    validate_scene_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    add_mapping_scope_arguments(parser)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--expected-proposals-sha256", required=True)
    parser.add_argument("--expected-proposals-content-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-probe-content-sha256", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-selection-content-sha256", required=True)
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


def _comparison(control: float, variant: float) -> dict:
    return {
        "control": control,
        "variant": variant,
        "delta": variant - control,
        "ratio": None if control == 0 else variant / control,
    }


def _stage_a_gates(
    *,
    selection: dict,
    control: dict,
    variant: dict,
    expected_pair_budget: int,
    expected_candidate_pair_count: int,
    expected_candidate_components: int,
) -> dict[str, bool]:
    selected_graph = selection["graph"]
    candidate_graph = selection["candidate_graph"]
    return {
        "candidate_union_exact_and_bounded": (
            int(selection["candidate_pair_count"])
            == int(expected_candidate_pair_count)
            and int(expected_candidate_pair_count) <= 2 * int(expected_pair_budget)
        ),
        "selected_exact_pair_budget": int(selection["exact_pair_budget"])
        == int(expected_pair_budget),
        "candidate_components_exact": int(candidate_graph["component_count"])
        == int(expected_candidate_components),
        "selected_components_preserved": int(selected_graph["component_count"])
        == int(candidate_graph["component_count"]),
        "selected_zero_isolates": int(selected_graph["isolated_camera_count"]) == 0,
        "selected_minimum_degree_at_least_one": int(selected_graph["minimum_degree"])
        >= 1,
        "verified_fisher_utility_improves_5pct": variant[
            "confidence_weighted_fisher_utility_sum"
        ]
        >= 1.05 * control["confidence_weighted_fisher_utility_sum"],
        "verified_triangles_retain_98pct": variant[
            "completed_verified_keypoint_triangle_count"
        ]
        >= 0.98 * control["completed_verified_keypoint_triangle_count"],
        "verified_triangle_camera_fraction_not_lower": variant[
            "completed_verified_triangle_camera_fraction"
        ]
        >= control["completed_verified_triangle_camera_fraction"],
    }


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
        **mapping_scope_kwargs(args),
    )
    proposals = load_proposals(
        path=args.proposals,
        expected_file_sha256=args.expected_proposals_sha256,
        expected_content_sha256=args.expected_proposals_content_sha256,
        cache=cache,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        expected_pair_budget=args.expected_pair_budget,
        expected_candidate_pair_count=args.expected_candidate_pair_count,
        expected_candidate_components=args.expected_candidate_components,
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
    validate_probe_proposal_lineage(probe=probe, proposals=proposals)
    selection = load_selection(
        path=args.selection,
        expected_file_sha256=args.expected_selection_sha256,
        expected_content_sha256=args.expected_selection_content_sha256,
        probe=probe,
        expected_pair_budget=args.expected_pair_budget,
    )
    evaluations = evaluate_pair_subsets(
        probe=probe["payload"],
        cache=cache,
        subsets={
            "control": proposals["nearest_pairs"],
            "variant": selection_pairs(selection["payload"]),
        },
        maximum_reprojection_error_px=args.maximum_cycle_reprojection_error_px,
    )
    control = evaluations["control"]
    variant = evaluations["variant"]
    assert_selection_metrics(selection["payload"], variant)
    comparisons = {
        name: _comparison(control[name], variant[name])
        for name in (
            "confidence_weighted_fisher_utility_sum",
            "completed_verified_keypoint_triangle_count",
            "completed_verified_triangle_camera_fraction",
        )
    }
    gates = _stage_a_gates(
        selection=selection["payload"],
        control=control,
        variant=variant,
        expected_pair_budget=args.expected_pair_budget,
        expected_candidate_pair_count=args.expected_candidate_pair_count,
        expected_candidate_components=args.expected_candidate_components,
    )
    passed = all(gates.values())
    artifacts = {
        "query_cache": cache,
        "pair_proposals": proposals,
        "pair_match_probe": probe,
        "pair_selection": selection,
    }
    output_target = validate_output_target(
        args.output, protected_paths=[value["path"] for value in artifacts.values()]
    )
    for artifact in artifacts.values():
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("A Stage-A input changed during comparison")
    report = {
        "schema": "lafgs_cycle_verified_fisher_stage_a_gate",
        "version": 1,
        "uses_test_queries": False,
        "mapping_only": True,
        "valid": True,
        "policy": "cycle_verified_fisher",
        "scene_contract": contract,
        "control_subset": "attested_nearest_pairs_from_same_probe",
        "control": control,
        "variant": variant,
        "comparisons": comparisons,
        "gates": gates,
        "stage_a_passed": passed,
        "advance_to_reuse_only_track_build": passed,
        "decision": "GO_TO_TRACK_REUSE" if passed else "STOP_BEFORE_TRACK_REUSE",
        "inputs": {
            name: {
                "path": str(value["path"]),
                "sha256": value["sha256"],
                **(
                    {"content_sha256": value["content_sha256"]}
                    if "content_sha256" in value
                    else {}
                ),
                **(
                    {"mapping_scope": value["mapping_scope"]}
                    if "mapping_scope" in value
                    else {}
                ),
            }
            for name, value in artifacts.items()
        },
    }
    output = atomic_json_save(report, output_target, overwrite=bool(args.overwrite))
    report["output"] = str(output)
    report["output_sha256"] = sha256_file(output)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["stage_a_passed"]:
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
