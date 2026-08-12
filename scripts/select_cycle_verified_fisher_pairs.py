#!/usr/bin/env python3
"""Select the exact-budget P8 graph from one SHA-bound candidate match probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import select_cycle_verified_fisher_pairs
from scripts.cycle_verified_fisher_cli_common import (
    atomic_torch_save,
    load_mapping_cache,
    load_probe,
    validate_output_target,
    validate_scene_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-probe-content-sha256", required=True)
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
        raise ValueError("P8 V1 minimum camera degree is frozen at one")
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
    output_target = validate_output_target(
        args.output, protected_paths=[cache["path"], probe["path"]]
    )
    _, selection = select_cycle_verified_fisher_pairs(
        pair_match_probe=probe["payload"],
        keypoints=cache["keypoints"],
        camera_K=cache["camera_K"],
        pose_w2c=cache["pose_w2c"],
        pair_budget=int(args.expected_pair_budget),
        minimum_camera_degree=int(args.minimum_camera_degree),
        maximum_cycle_reprojection_error_px=float(
            args.maximum_cycle_reprojection_error_px
        ),
    )
    if (
        int(selection["candidate_graph"]["component_count"])
        != int(args.expected_candidate_components)
        or int(selection["candidate_pair_count"])
        != int(args.expected_candidate_pair_count)
    ):
        raise RuntimeError("Selector output differs from the scene candidate contract")
    if sha256_file(cache["path"]) != cache["sha256"]:
        raise RuntimeError("Query cache changed while selecting the P8 graph")
    if sha256_file(probe["path"]) != probe["sha256"]:
        raise RuntimeError("Pair-match probe changed while selecting the P8 graph")
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
        "verified_triangle": selection["verified_triangle"],
        "selection_content_sha256": selection["content_sha256"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "inputs": {
            "query_cache": {"path": str(cache["path"]), "sha256": cache["sha256"]},
            "probe": {
                "path": str(probe["path"]),
                "sha256": probe["sha256"],
                "content_sha256": probe["content_sha256"],
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))


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
