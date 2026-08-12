#!/usr/bin/env python3
"""Materialize the one bounded mapping-only P8 candidate-edge match probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import (
    bounded_union_candidate_pool,
    build_pair_match_probe,
)
from scripts.cycle_verified_fisher_cli_common import (
    atomic_torch_save,
    load_mapping_cache,
    load_track_factor,
    validate_matcher_contract,
    validate_output_target,
    validate_scene_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--nearest-factor", type=Path, required=True)
    parser.add_argument("--expected-nearest-factor-sha256", required=True)
    parser.add_argument("--geometry-factor", type=Path, required=True)
    parser.add_argument("--expected-geometry-factor-sha256", required=True)
    parser.add_argument("--expected-query-names-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--expected-candidate-pair-count", type=int, required=True)
    parser.add_argument("--expected-candidate-components", type=int, required=True)
    parser.add_argument("--minimum-similarity", type=float, required=True)
    parser.add_argument("--minimum-margin", type=float, required=True)
    parser.add_argument("--maximum-epipolar-error-px", type=float, required=True)
    parser.add_argument("--epipolar-candidate-topk", type=int, required=True)
    parser.add_argument(
        "--epipolar-recovered-minimum-similarity", type=float, required=True
    )
    parser.add_argument(
        "--epipolar-recovered-minimum-margin", type=float, required=True
    )
    parser.add_argument("--device", required=True)
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
    cache = load_mapping_cache(
        path=args.query_cache,
        expected_file_sha256=args.expected_query_cache_sha256,
        expected_query_names_sha256=args.expected_query_names_sha256,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
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
    nearest = load_track_factor(
        path=args.nearest_factor,
        expected_file_sha256=args.expected_nearest_factor_sha256,
        expected_policy="nearest",
        **factor_common,
    )
    geometry = load_track_factor(
        path=args.geometry_factor,
        expected_file_sha256=args.expected_geometry_factor_sha256,
        expected_policy="parallax_diverse",
        **factor_common,
    )
    output_target = validate_output_target(
        args.output,
        protected_paths=[cache["path"], nearest["path"], geometry["path"]],
    )
    candidate_pairs, candidate_graph = bounded_union_candidate_pool(
        pair_sets=(nearest["pairs"], geometry["pairs"]),
        query_count=len(cache["names"]),
        maximum_pair_count=2 * int(args.expected_pair_budget),
    )
    if (
        len(candidate_pairs) != int(args.expected_candidate_pair_count)
        or int(candidate_graph["component_count"])
        != int(args.expected_candidate_components)
        or int(candidate_graph["isolated_camera_count"]) != 0
    ):
        raise ValueError("Candidate union differs from the P8 V1 scene contract")
    matcher = validate_matcher_contract({
        "minimum_similarity": float(args.minimum_similarity),
        "minimum_margin": float(args.minimum_margin),
        "maximum_epipolar_error_px": float(args.maximum_epipolar_error_px),
        "epipolar_candidate_topk": int(args.epipolar_candidate_topk),
        "epipolar_recovered_minimum_similarity": float(
            args.epipolar_recovered_minimum_similarity
        ),
        "epipolar_recovered_minimum_margin": float(
            args.epipolar_recovered_minimum_margin
        ),
    })
    probe = build_pair_match_probe(
        candidate_pairs=candidate_pairs,
        descriptors=cache["descriptors"],
        keypoints=cache["keypoints"],
        detector_scores=cache["scores"],
        camera_K=cache["camera_K"],
        pose_w2c=cache["pose_w2c"],
        query_names_sha256=cache["query_names_sha256"],
        query_cache_sha256=cache["sha256"],
        mapping_keypoint_count=int(args.expected_mapping_keypoints),
        mapping_nms_radius=int(args.expected_nms_radius),
        candidate_pool_construction=(
            "frozen_nearest_union_equally_budgeted_mapping_geometry_v1"
        ),
        candidate_pool_parameters={
            "nearest_factor_sha256": nearest["sha256"],
            "mapping_geometry_factor_sha256": geometry["sha256"],
            "per_arm_exact_pair_budget": int(args.expected_pair_budget),
            "maximum_union_pair_count": 2 * int(args.expected_pair_budget),
            "observed_union_graph": candidate_graph,
        },
        matcher_parameters=matcher,
        device=args.device,
    )
    for artifact in (cache, nearest, geometry):
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("A frozen input changed while materializing the probe")
    output = atomic_torch_save(
        probe, output_target, overwrite=bool(args.overwrite)
    )
    return {
        "schema": probe["schema"],
        "version": probe["version"],
        "scene_contract": contract,
        "mapping_only": True,
        "uses_test_queries": probe["uses_test_queries"],
        "candidate_graph": candidate_graph,
        "match_count": int(probe["matches"]["source_keypoint_index"].numel()),
        "probe_content_sha256": probe["content_sha256"],
        "output": str(output),
        "output_sha256": sha256_file(output),
        "inputs": {
            "query_cache": {"path": str(cache["path"]), "sha256": cache["sha256"]},
            "nearest_factor": {
                "path": str(nearest["path"]),
                "sha256": nearest["sha256"],
            },
            "geometry_factor": {
                "path": str(geometry["path"]),
                "sha256": geometry["sha256"],
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
