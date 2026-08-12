#!/usr/bin/env python3
"""Gate P8 V2 against nearest control with coverage as a hard invariant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import (
    assert_verified_cycle_table_exact,
    materialize_verified_cycle_table,
)
from scripts.cycle_verified_fisher_cli_common import (
    add_mapping_scope_arguments,
    assert_selection_metrics,
    atomic_json_save,
    evaluate_pair_subsets_from_verified_table,
    load_coverage_selection,
    load_mapping_cache,
    load_probe,
    load_proposals,
    load_selection,
    load_verified_cycle_table,
    mapping_scope_kwargs,
    selection_pairs,
    validate_output_target,
    validate_probe_proposal_lineage,
    validate_scene_contract,
    validate_v2_frozen_source_contract,
)


STAIRS_V1_SELECTION_CONTRACT = {
    "sha256": "7d08bed0ead859ae917724beebe22af844b87bbd7e0f9579b834f5915a13c16f",
    "content_sha256": (
        "721617c2e084e1c8fe75e29cec6d818a0374d7522977c241880033489f9cf93f"
    ),
}


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
    parser.add_argument("--verified-cycle-table", type=Path, required=True)
    parser.add_argument("--expected-verified-cycle-table-sha256", required=True)
    parser.add_argument(
        "--expected-verified-cycle-table-content-sha256", required=True
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--expected-selection-content-sha256", required=True)
    parser.add_argument("--stairs-v1-selection", type=Path)
    parser.add_argument("--expected-stairs-v1-selection-sha256")
    parser.add_argument("--expected-stairs-v1-selection-content-sha256")
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
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate lineage and rematerialize geometry without writing a gate.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _comparison(control: float, variant: float) -> dict:
    return {
        "control": control,
        "variant": variant,
        "delta": variant - control,
        "ratio": None if control == 0 else variant / control,
    }


def _gates(
    *,
    selection: dict,
    control: dict,
    variant: dict,
    expected_pair_budget: int,
    expected_candidate_pair_count: int,
    expected_candidate_components: int,
    stairs_v1: dict | None,
) -> dict[str, bool]:
    candidate_graph = selection["candidate_graph"]
    selected_graph = selection["graph"]
    certificate = selection["coverage_certificate"]
    control_cameras = control["completed_verified_triangle_camera_index"]
    variant_cameras = set(variant["completed_verified_triangle_camera_index"])
    # Camera coverage is exact membership, not a post-hoc fraction threshold.
    target_exact = certificate["target_camera_index"] == control_cameras
    target_covered = set(control_cameras).issubset(variant_cameras)
    gates = {
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
        "coverage_scaffold_reserves_half_budget_for_utility": certificate[
            "stage1_scaffold_at_most_half_budget"
        ]
        is True,
        "control_target_membership_exact": target_exact,
        "all_control_target_cameras_hard_covered": target_covered,
        "verified_fisher_utility_improves_5pct": variant[
            "confidence_weighted_fisher_utility_sum"
        ]
        >= 1.05 * control["confidence_weighted_fisher_utility_sum"],
        "verified_triangles_retain_98pct": variant[
            "completed_verified_keypoint_triangle_count"
        ]
        >= 0.98 * control["completed_verified_keypoint_triangle_count"],
    }
    if stairs_v1 is not None:
        gates.update(
            {
                "stairs_v1_fisher_utility_retain_98pct": variant[
                    "confidence_weighted_fisher_utility_sum"
                ]
                >= 0.98 * stairs_v1["confidence_weighted_fisher_utility_sum"],
                "stairs_v1_verified_triangles_retain_98pct": variant[
                    "completed_verified_keypoint_triangle_count"
                ]
                >= 0.98
                * stairs_v1["completed_verified_keypoint_triangle_count"],
                "stairs_v1_covered_camera_count_not_lower": variant[
                    "completed_verified_triangle_camera_count"
                ]
                >= stairs_v1["completed_verified_triangle_camera_count"],
            }
        )
    return gates


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
        raise ValueError("P8 V2 cycle reprojection threshold is frozen at 2.0 px")
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
    frozen_sources = validate_v2_frozen_source_contract(
        scene=args.scene, cache=cache, probe=probe, proposals=proposals
    )
    verified = load_verified_cycle_table(
        path=args.verified_cycle_table,
        expected_file_sha256=args.expected_verified_cycle_table_sha256,
        expected_content_sha256=(
            args.expected_verified_cycle_table_content_sha256
        ),
        probe=probe,
        expected_maximum_reprojection_error_px=(
            args.maximum_cycle_reprojection_error_px
        ),
    )
    rematerialized = materialize_verified_cycle_table(
        pair_match_probe=probe["payload"],
        keypoints=cache["keypoints"],
        camera_K=cache["camera_K"],
        pose_w2c=cache["pose_w2c"],
        maximum_reprojection_error_px=args.maximum_cycle_reprojection_error_px,
    )
    assert_verified_cycle_table_exact(verified["payload"], rematerialized)
    if sha256_file(cache["path"]) != cache["sha256"]:
        raise RuntimeError("Query cache changed during verified geometry replay")
    if sha256_file(probe["path"]) != probe["sha256"]:
        raise RuntimeError("Pair probe changed during verified geometry replay")
    if bool(args.verify_only):
        return {
            "schema": "lafgs_cycle_verified_fisher_coverage_stage_a_verification",
            "version": 1,
            "valid": True,
            "uses_test_queries": False,
            "scene_contract": contract,
            "frozen_source_contract": frozen_sources,
            "verified_geometry_independently_rematerialized_exact": True,
            "stage_a_gate_written": False,
        }
    selection = load_coverage_selection(
        path=args.selection,
        expected_file_sha256=args.expected_selection_sha256,
        expected_content_sha256=args.expected_selection_content_sha256,
        probe=probe,
        coverage_reference_pairs=proposals["nearest_pairs"],
        verified_cycle_table=verified,
        expected_pair_budget=args.expected_pair_budget,
    )
    stairs_v1 = None
    stairs_v1_record = None
    stairs_v1_arguments = (
        args.stairs_v1_selection,
        args.expected_stairs_v1_selection_sha256,
        args.expected_stairs_v1_selection_content_sha256,
    )
    if args.scene == "stairs":
        if any(value is None for value in stairs_v1_arguments):
            raise ValueError("Stairs V2 requires the frozen successful V1 selection")
        if (
            args.expected_stairs_v1_selection_sha256
            != STAIRS_V1_SELECTION_CONTRACT["sha256"]
            or args.expected_stairs_v1_selection_content_sha256
            != STAIRS_V1_SELECTION_CONTRACT["content_sha256"]
        ):
            raise ValueError("Stairs V1 guard differs from its frozen artifact hashes")
        stairs_v1_record = load_selection(
            path=args.stairs_v1_selection,
            expected_file_sha256=args.expected_stairs_v1_selection_sha256,
            expected_content_sha256=(
                args.expected_stairs_v1_selection_content_sha256
            ),
            probe=probe,
            expected_pair_budget=args.expected_pair_budget,
        )
    elif any(value is not None for value in stairs_v1_arguments):
        raise ValueError("Stairs V1 guard arguments are forbidden for GreatCourt")
    subsets = {
        "control": proposals["nearest_pairs"],
        "variant": selection_pairs(selection["payload"]),
    }
    if stairs_v1_record is not None:
        subsets["stairs_v1"] = selection_pairs(stairs_v1_record["payload"])
    evaluations = evaluate_pair_subsets_from_verified_table(
        probe=probe["payload"],
        verified_cycle_table=verified["payload"],
        subsets=subsets,
    )
    control, variant = evaluations["control"], evaluations["variant"]
    stairs_v1 = evaluations.get("stairs_v1")
    assert_selection_metrics(selection["payload"], variant)
    gates = _gates(
        selection=selection["payload"],
        control=control,
        variant=variant,
        expected_pair_budget=args.expected_pair_budget,
        expected_candidate_pair_count=args.expected_candidate_pair_count,
        expected_candidate_components=args.expected_candidate_components,
        stairs_v1=stairs_v1,
    )
    passed = all(gates.values())
    artifacts = {
        "query_cache": cache,
        "pair_proposals": proposals,
        "pair_match_probe": probe,
        "verified_cycle_table": verified,
        "pair_selection": selection,
    }
    if stairs_v1_record is not None:
        artifacts["stairs_v1_pair_selection"] = stairs_v1_record
    output_target = validate_output_target(
        args.output, protected_paths=[value["path"] for value in artifacts.values()]
    )
    if any(
        sha256_file(value["path"]) != value["sha256"]
        for value in artifacts.values()
    ):
        raise RuntimeError("A P8 V2 Stage-A input changed during comparison")
    comparisons = {
        name: _comparison(control[name], variant[name])
        for name in (
            "confidence_weighted_fisher_utility_sum",
            "completed_verified_keypoint_triangle_count",
            "completed_verified_triangle_camera_count",
        )
    }
    report = {
        "schema": "lafgs_cycle_verified_fisher_coverage_stage_a_gate",
        "version": 1,
        "uses_test_queries": False,
        "mapping_only": True,
        "valid": True,
        "policy": "cycle_verified_fisher_coverage",
        "scene_contract": contract,
        "frozen_source_contract": frozen_sources,
        "control_subset": "attested_nearest_pairs_from_same_probe",
        "coverage_semantics": (
            "lexicographic_hard_membership_before_fisher_objective"
        ),
        "verified_geometry_independently_rematerialized_exact": True,
        "candidate_universe_verified_camera": verified["payload"][
            "candidate_verified_camera"
        ],
        "control": control,
        "variant": variant,
        "stairs_v1": stairs_v1,
        "comparisons": comparisons,
        "gates": gates,
        "stage_a_passed": passed,
        "requires_other_scene": True,
        "requires_v2_aware_track_lineage_implementation": True,
        "advance_to_reuse_only_track_build": False,
        "decision": (
            "SCENE_STAGE_A_PASS_REQUIRES_OTHER_SCENE"
            if passed
            else "STOP_BEFORE_TRACK_REUSE"
        ),
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
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("stage_a_passed") is False:
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
