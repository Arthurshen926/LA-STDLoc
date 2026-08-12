#!/usr/bin/env python3
"""Select P8 V2 with hard nearest-control verified-triangle camera coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import (
    select_cycle_verified_fisher_coverage_pairs,
)
from scripts.cycle_verified_fisher_cli_common import (
    add_mapping_scope_arguments,
    atomic_torch_save,
    load_mapping_cache,
    load_probe,
    load_proposals,
    load_verified_cycle_table,
    mapping_scope_kwargs,
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
    parser.add_argument("--verified-cycle-table", type=Path, required=True)
    parser.add_argument("--expected-verified-cycle-table-sha256", required=True)
    parser.add_argument(
        "--expected-verified-cycle-table-content-sha256", required=True
    )
    parser.add_argument("--expected-query-names-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--expected-candidate-pair-count", type=int, required=True)
    parser.add_argument("--expected-candidate-components", type=int, required=True)
    parser.add_argument("--minimum-camera-degree", type=int, required=True)
    parser.add_argument(
        "--maximum-cycle-reprojection-error-px", type=float, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict:
    contract = validate_scene_contract(
        scene=args.scene,
        mapping_keypoints=args.expected_mapping_keypoints,
        nms_radius=args.expected_nms_radius,
        pair_budget=args.expected_pair_budget,
        candidate_pair_count=args.expected_candidate_pair_count,
        candidate_component_count=args.expected_candidate_components,
    )
    if int(args.minimum_camera_degree) != 1:
        raise ValueError("P8 V2 minimum camera degree is frozen at one")
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
    output_target = validate_output_target(
        args.output,
        protected_paths=[
            cache["path"],
            proposals["path"],
            probe["path"],
            verified["path"],
        ],
    )
    _, selection = select_cycle_verified_fisher_coverage_pairs(
        pair_match_probe=probe["payload"],
        coverage_reference_pairs=proposals["nearest_pairs"],
        keypoints=cache["keypoints"],
        camera_K=cache["camera_K"],
        pose_w2c=cache["pose_w2c"],
        pair_budget=args.expected_pair_budget,
        minimum_camera_degree=args.minimum_camera_degree,
        maximum_cycle_reprojection_error_px=(
            args.maximum_cycle_reprojection_error_px
        ),
        verified_cycle_table=verified["payload"],
    )
    if (
        int(selection["candidate_graph"]["component_count"])
        != int(args.expected_candidate_components)
        or int(selection["candidate_pair_count"])
        != int(args.expected_candidate_pair_count)
    ):
        raise RuntimeError("Coverage selector differs from the scene contract")
    inputs = (cache, proposals, probe, verified)
    if any(sha256_file(value["path"]) != value["sha256"] for value in inputs):
        raise RuntimeError("A P8 V2 selector input changed during selection")
    output = atomic_torch_save(
        selection, output_target, overwrite=bool(args.overwrite)
    )
    return {
        "schema": selection["schema"],
        "version": selection["version"],
        "policy": selection["policy"],
        "scene_contract": contract,
        "uses_test_queries": selection["uses_test_queries"],
        "candidate_graph": selection["candidate_graph"],
        "selected_graph": selection["graph"],
        "coverage_certificate": selection["coverage_certificate"],
        "verified_triangle": selection["verified_triangle"],
        "selection_content_sha256": selection["content_sha256"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "inputs": {
            name: {
                "path": str(value["path"]),
                "sha256": value["sha256"],
                **(
                    {"content_sha256": value["content_sha256"]}
                    if "content_sha256" in value
                    else {}
                ),
            }
            for name, value in zip(
                (
                    "query_cache",
                    "pair_proposals",
                    "pair_match_probe",
                    "verified_cycle_table",
                ),
                inputs,
            )
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2, sort_keys=True))


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
