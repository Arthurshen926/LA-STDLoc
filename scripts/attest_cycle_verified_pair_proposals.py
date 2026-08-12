#!/usr/bin/env python3
"""Bind two archived pair tables to the fresh P8 mapping-cache contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import torch

from common.hashing import sha256_file
from evidence.cycle_verified_fisher import materialize_pair_proposal_table
from scripts.cycle_verified_fisher_cli_common import (
    add_mapping_scope_arguments,
    atomic_torch_save,
    attest_file,
    load_mapping_cache,
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
    parser.add_argument("--nearest-source", type=Path, required=True)
    parser.add_argument("--expected-nearest-source-sha256", required=True)
    parser.add_argument("--geometry-source", type=Path, required=True)
    parser.add_argument("--expected-geometry-source-sha256", required=True)
    parser.add_argument("--expected-query-names-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--expected-pair-budget", type=int, required=True)
    parser.add_argument("--expected-candidate-pair-count", type=int, required=True)
    parser.add_argument("--expected-candidate-components", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _source_pair_table(
    *,
    path: Path,
    expected_sha256: str,
    expected_policy: str,
    cache: dict,
    mapping_keypoints: int,
    nms_radius: int,
    pair_budget: int,
) -> dict:
    """Read only the pair table; never promote an archived factor's lineage."""
    path = attest_file(path, expected_sha256, label=f"{expected_policy} proposal source")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "lafgs_pair_policy_track_factor"
        or int(payload.get("version", -1)) != 1
        or payload.get("uses_test_queries") is not False
        or payload.get("pair_policy") != expected_policy
        or int(payload.get("mapping_keypoint_factor", -1)) != int(mapping_keypoints)
        or any(
            payload.get(name) is not False
            for name in (
                "descriptor_factor_mutated",
                "density_factor_mutated",
                "selector_factor_mutated",
            )
        )
    ):
        raise ValueError(f"{expected_policy} proposal source is not an admissible archive")
    names = [str(value) for value in payload.get("query_names", [])]
    if names != cache["names"]:
        raise ValueError(f"{expected_policy} proposal source has a different query order")
    if payload.get("query_names_sha256") not in (None, cache["query_names_sha256"]):
        raise ValueError(f"{expected_policy} proposal source has stale query-name lineage")
    if payload.get("mapping_nms_radius") not in (None, int(nms_radius)):
        raise ValueError(f"{expected_policy} proposal source has a different NMS radius")
    sidecar = payload.get("pair_sidecar")
    pair = sidecar.get("pair") if isinstance(sidecar, dict) else None
    policy = sidecar.get("policy") if isinstance(sidecar, dict) else None
    if (
        not isinstance(pair, dict)
        or not isinstance(policy, dict)
        or policy.get("uses_test_queries") is not False
    ):
        raise ValueError(f"{expected_policy} proposal source lacks a mapping-only pair table")
    left = torch.as_tensor(pair.get("left_query_index"), dtype=torch.long).reshape(-1)
    right = torch.as_tensor(pair.get("right_query_index"), dtype=torch.long).reshape(-1)
    pairs = list(zip(left.tolist(), right.tolist()))
    if len(pairs) != int(pair_budget) or pairs != sorted(set(pairs)):
        raise ValueError(f"{expected_policy} proposal source violates the exact budget")
    if any(
        left_index < 0
        or left_index >= right_index
        or right_index >= len(names)
        for left_index, right_index in pairs
    ):
        raise ValueError(f"{expected_policy} proposal source pair index is invalid")
    unavailable = [
        name
        for name in ("mapping_nms_radius", "query_names_sha256", "input_lineage")
        if payload.get(name) is None
    ]
    return {
        "path": path,
        "sha256": sha256_file(path),
        "pairs": pairs,
        "unavailable_source_lineage": unavailable,
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
    cache = load_mapping_cache(
        path=args.query_cache,
        expected_file_sha256=args.expected_query_cache_sha256,
        expected_query_names_sha256=args.expected_query_names_sha256,
        expected_mapping_keypoints=args.expected_mapping_keypoints,
        expected_nms_radius=args.expected_nms_radius,
        **mapping_scope_kwargs(args),
    )
    common = {
        "cache": cache,
        "mapping_keypoints": args.expected_mapping_keypoints,
        "nms_radius": args.expected_nms_radius,
        "pair_budget": args.expected_pair_budget,
    }
    nearest = _source_pair_table(
        path=args.nearest_source,
        expected_sha256=args.expected_nearest_source_sha256,
        expected_policy="nearest",
        **common,
    )
    geometry = _source_pair_table(
        path=args.geometry_source,
        expected_sha256=args.expected_geometry_source_sha256,
        expected_policy="parallax_diverse",
        **common,
    )
    output_target = validate_output_target(
        args.output,
        protected_paths=[cache["path"], nearest["path"], geometry["path"]],
    )
    proposals = materialize_pair_proposal_table(
        nearest_pairs=nearest["pairs"],
        geometry_pairs=geometry["pairs"],
        query_count=len(cache["names"]),
        query_names_sha256=cache["query_names_sha256"],
        query_cache_path=str(cache["path"]),
        query_cache_sha256=cache["sha256"],
        mapping_keypoint_count=args.expected_mapping_keypoints,
        mapping_nms_radius=args.expected_nms_radius,
        mapping_scope=cache["mapping_scope"],
        exact_pair_budget=args.expected_pair_budget,
        nearest_source_path=str(nearest["path"]),
        nearest_source_sha256=nearest["sha256"],
        nearest_unavailable_source_lineage=nearest["unavailable_source_lineage"],
        geometry_source_path=str(geometry["path"]),
        geometry_source_sha256=geometry["sha256"],
        geometry_unavailable_source_lineage=geometry[
            "unavailable_source_lineage"
        ],
    )
    union = proposals["candidate_union"]
    if (
        int(union["pair_count"]) != int(args.expected_candidate_pair_count)
        or int(union["component_count"]) != int(args.expected_candidate_components)
        or int(union["isolated_camera_count"]) != 0
    ):
        raise ValueError("Proposal union differs from the P8 scene contract")
    for artifact in (cache, nearest, geometry):
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("A proposal input changed during attestation")
    output = atomic_torch_save(proposals, output_target, overwrite=bool(args.overwrite))
    return {
        "schema": proposals["schema"],
        "version": proposals["version"],
        "scene_contract": contract,
        "mapping_only": True,
        "uses_test_queries": False,
        "mapping_scope": cache["mapping_scope"],
        "source_contract": proposals["source_contract"],
        "candidate_union": union,
        "proposal_content_sha256": proposals["content_sha256"],
        "output": str(output),
        "output_sha256": sha256_file(output),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))


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
