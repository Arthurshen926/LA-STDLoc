#!/usr/bin/env python3
"""Materialize one reusable, hash-bound P8 verified-triangle table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import materialize_verified_cycle_table
from scripts.cycle_verified_fisher_cli_common import (
    add_mapping_scope_arguments,
    atomic_torch_save,
    load_mapping_cache,
    load_probe,
    mapping_scope_kwargs,
    validate_output_target,
    validate_scene_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    add_mapping_scope_arguments(parser)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--expected-probe-sha256", required=True)
    parser.add_argument("--expected-probe-content-sha256", required=True)
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
    table = materialize_verified_cycle_table(
        pair_match_probe=probe["payload"],
        keypoints=cache["keypoints"],
        camera_K=cache["camera_K"],
        pose_w2c=cache["pose_w2c"],
        maximum_reprojection_error_px=args.maximum_cycle_reprojection_error_px,
    )
    if sha256_file(cache["path"]) != cache["sha256"]:
        raise RuntimeError("Query cache changed while materializing verified cycles")
    if sha256_file(probe["path"]) != probe["sha256"]:
        raise RuntimeError("Pair probe changed while materializing verified cycles")
    output = atomic_torch_save(table, output_target, overwrite=bool(args.overwrite))
    return {
        "schema": table["schema"],
        "version": table["version"],
        "uses_test_queries": table["uses_test_queries"],
        "scene_contract": contract,
        "verified_triangle_count": int(
            table["verified_triangle"]["pair_index"].shape[0]
        ),
        "content_sha256": table["content_sha256"],
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
    report = run(build_parser().parse_args(argv))
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
