#!/usr/bin/env python3
"""Materialize the fresh mapping-only P9 E1 feature/depth cache on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence
import uuid

from common.hashing import sha256_file
from map_learning.fixed_pair_matcher_ceiling import (
    BundledXFeatSpec,
    load_bundled_models,
    load_source_image_manifest,
    materialize_feature_cache,
    validate_bundled_xfeat_artifact,
    validate_feature_cache,
    validate_mapping_context,
)
from scripts.fixed_pair_matcher_ceiling_common import (
    atomic_torch_save_fresh,
    attest_file,
    configure_formal_cpu_runtime,
    load_mapping_scope,
    load_scene_gate,
    producer_identity,
    require_fixed_path,
    scene_contract,
    torch_load,
    validate_fresh_file_output,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--expected-query-cache-sha256", required=True)
    parser.add_argument("--mapping-scope-equivalence", type=Path, required=True)
    parser.add_argument("--expected-mapping-scope-equivalence-sha256", required=True)
    parser.add_argument("--xfeat-worktree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-commit", required=True)
    parser.add_argument("--expected-xfeat-tree", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--stairs-pair-gate", type=Path)
    parser.add_argument("--expected-stairs-pair-gate-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict:
    configure_formal_cpu_runtime()
    if args.device != "cpu":
        raise ValueError("P9 feature-cache producer is CPU-only")
    contract = scene_contract(args.scene)
    parent_arguments = (
        args.stairs_pair_gate,
        args.expected_stairs_pair_gate_sha256,
    )
    parent_stairs_gate = None
    if args.scene == "greatcourt":
        if not all(value is not None for value in parent_arguments):
            raise ValueError(
                "GreatCourt P9 feature build requires the exact passing Stairs gate"
            )
        parent = load_scene_gate(
            path=args.stairs_pair_gate,
            expected_file_sha256=args.expected_stairs_pair_gate_sha256,
            expected_scene="stairs",
            require_pass=True,
        )
        parent_stairs_gate = {
            "path": str(parent["path"]),
            "sha256": parent["sha256"],
            "scientific_projection": parent["scientific_projection"],
        }
    elif any(value is not None for value in parent_arguments):
        raise ValueError("Stairs P9 feature build must not accept a parent scene gate")
    dataset = require_fixed_path(
        args.dataset, contract["dataset_root"], label="dataset"
    )
    if not dataset.is_dir():
        raise FileNotFoundError(f"P9 dataset not found: {dataset}")
    query_cache_path = require_fixed_path(
        args.query_cache, contract["query_cache"]["path"], label="query cache"
    )
    query_cache_path = attest_file(
        query_cache_path,
        args.expected_query_cache_sha256,
        label="mapping query cache",
    )
    if args.expected_query_cache_sha256 != contract["query_cache"]["sha256"]:
        raise ValueError("P9 query-cache SHA differs from fixed scene registry")
    proof_path = require_fixed_path(
        args.mapping_scope_equivalence,
        contract["mapping_scope_equivalence"]["path"],
        label="mapping-scope proof",
    )
    if (
        args.expected_mapping_scope_equivalence_sha256
        != contract["mapping_scope_equivalence"]["sha256"]
    ):
        raise ValueError("P9 mapping-scope proof SHA differs from scene registry")
    mapping_scope = load_mapping_scope(
        path=proof_path,
        expected_file_sha256=args.expected_mapping_scope_equivalence_sha256,
        query_cache_path=query_cache_path,
        query_cache_sha256=contract["query_cache"]["sha256"],
        query_count=contract["query_count"],
        mapping_keypoints=contract["requested_keypoint_count"],
    )
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = validate_fresh_file_output(
        args.output,
        protected=[query_cache_path, proof_path, checkpoint],
    )
    if output.name != "p9_fixed_pair_feature_cache.pt":
        raise ValueError("P9 feature cache output must use its fixed filename")
    producer = producer_identity(
        entrypoint="python -m scripts.materialize_fixed_pair_feature_cache"
    )
    artifact_spec = BundledXFeatSpec(
        worktree=args.xfeat_worktree,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_parent_commit=args.expected_parent_commit,
        expected_xfeat_tree=args.expected_xfeat_tree,
    )
    artifact = validate_bundled_xfeat_artifact(artifact_spec)
    query_cache = torch_load(query_cache_path)
    source_image_manifest = load_source_image_manifest(scene=args.scene)
    context = validate_mapping_context(
        query_cache,
        query_cache_path=query_cache_path,
        expected_query_cache_sha256=contract["query_cache"]["sha256"],
        dataset_root=dataset,
        images=contract["images"],
        expected_query_names_sha256=contract["query_names_sha256"],
        expected_query_count=contract["query_count"],
        requested_keypoint_count=contract["requested_keypoint_count"],
        mapping_scope=mapping_scope,
        source_image_manifest=source_image_manifest,
    )
    extractor, _, interpolators, state_summary = load_bundled_models(artifact)
    payload = materialize_feature_cache(
        scene=args.scene,
        context=context,
        artifact=artifact,
        extractor=extractor,
        interpolators=interpolators,
        state_summary=state_summary,
        producer_identity=producer,
        run_uuid=uuid.uuid4().hex,
        parent_stairs_gate=parent_stairs_gate,
    )
    if sha256_file(query_cache_path) != contract["query_cache"]["sha256"]:
        raise RuntimeError("P9 query cache changed during feature materialization")
    validate_bundled_xfeat_artifact(artifact_spec)
    for name, record in payload["queries"].items():
        if (
            sha256_file(record["image_lineage"]["source_image_path"])
            != record["image_lineage"]["source_image_sha256"]
        ):
            raise RuntimeError(f"P9 source mapping image changed: {name}")
    atomic_torch_save_fresh(
        payload,
        output,
        validator=lambda value: validate_feature_cache(
            value, expected_scene=args.scene
        ),
    )
    reloaded = torch_load(output)
    validation = validate_feature_cache(reloaded, expected_scene=args.scene)
    return {
        "schema": payload["schema"],
        "scene": args.scene,
        "mapping_only": True,
        "uses_test_queries": False,
        "output": str(output),
        "output_sha256": sha256_file(output),
        **validation,
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
