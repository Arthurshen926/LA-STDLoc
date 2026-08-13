#!/usr/bin/env python3
"""Build both P9 fixed-pair matcher arms and an atomic completion marker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Sequence
import uuid

from common.hashing import sha256_file
from evidence.fixed_pair_matcher_ceiling import (
    materialize_paired_probe,
    validate_paired_probe,
)
from map_learning.fixed_pair_matcher_ceiling import (
    BundledXFeatSpec,
    load_bundled_models,
    validate_bundled_xfeat_artifact,
)
from scripts.fixed_pair_matcher_ceiling_common import (
    completion_payload,
    configure_formal_cpu_runtime,
    load_completion,
    load_feature_cache,
    load_fixed_proposals,
    producer_identity,
    scene_contract,
    torch_load,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("stairs", "greatcourt"), required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--expected-feature-cache-sha256", required=True)
    parser.add_argument("--expected-feature-cache-content-sha256", required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--expected-proposals-sha256", required=True)
    parser.add_argument("--expected-proposals-content-sha256", required=True)
    parser.add_argument("--xfeat-worktree", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-parent-commit", required=True)
    parser.add_argument("--expected-xfeat-tree", required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def _summary(probe: dict) -> dict:
    return {
        "schema": "lafgs_p9_fixed_pair_match_probe_summary",
        "version": 1,
        "scene": probe["scene"],
        "mapping_only": True,
        "uses_test_queries": False,
        "run_uuid": probe["run_uuid"],
        "query_count": probe["query_count"],
        "pair_count": probe["pair_table"]["pair_count"],
        "pair_table_sha256": probe["pair_table"]["pair_table_sha256"],
        "content_sha256": probe["content_sha256"],
        "arms": {
            name: probe["arms"][name]["metrics"]
            for name in ("mnn_control", "lighterglue_variant")
        },
    }


def run(args: argparse.Namespace) -> dict:
    configure_formal_cpu_runtime()
    if args.device != "cpu":
        raise ValueError("P9 paired-probe producer is CPU-only")
    scene_contract(args.scene)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"P9 paired output root must be fresh: {output_root}")
    feature = load_feature_cache(
        scene=args.scene,
        path=args.feature_cache,
        expected_file_sha256=args.expected_feature_cache_sha256,
        expected_content_sha256=args.expected_feature_cache_content_sha256,
    )
    proposals = load_fixed_proposals(
        scene=args.scene,
        path=args.proposals,
        expected_file_sha256=args.expected_proposals_sha256,
        expected_content_sha256=args.expected_proposals_content_sha256,
        feature_cache=feature["payload"],
    )
    producer = producer_identity(
        entrypoint="python -m scripts.materialize_fixed_pair_match_probe"
    )
    if (
        feature["payload"].get("producer_identity", {}).get("compiled_identity")
        != producer["compiled_identity"]
    ):
        raise ValueError("P9 feature cache and probe producer identities differ")
    artifact_spec = BundledXFeatSpec(
        worktree=args.xfeat_worktree,
        checkpoint=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_parent_commit=args.expected_parent_commit,
        expected_xfeat_tree=args.expected_xfeat_tree,
    )
    artifact = validate_bundled_xfeat_artifact(artifact_spec)
    _, matcher, _, state_summary = load_bundled_models(artifact)
    run_uuid = feature["payload"]["run_uuid"]
    probe = materialize_paired_probe(
        scene=args.scene,
        feature_cache=feature["payload"],
        feature_cache_path=str(feature["path"]),
        feature_cache_sha256=feature["sha256"],
        pairs=proposals["pairs"],
        proposal_lineage={
            "path": str(proposals["path"]),
            "sha256": proposals["sha256"],
            "content_sha256": proposals["content_sha256"],
            "arm": "nearest",
            "pair_table_sha256": proposals["pair_table_sha256"],
            "match_rows_reused": False,
        },
        matcher=matcher,
        matcher_identity={
            "checkpoint": artifact["checkpoint"],
            "state": state_summary,
        },
        run_uuid=run_uuid,
        producer_identity=producer,
    )
    for artifact_input in (feature, proposals):
        if sha256_file(artifact_input["path"]) != artifact_input["sha256"]:
            raise RuntimeError("P9 frozen input changed during paired matching")
    validate_bundled_xfeat_artifact(artifact_spec)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = output_root.with_name(
        f".{output_root.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    probe_path = temporary_root / "fixed_pair_match_probe.pt"
    summary_path = temporary_root / "fixed_pair_match_probe.json"
    completion_path = temporary_root / "paired_match_completion.json"
    try:
        temporary_root.mkdir()
        import torch

        torch.save(probe, probe_path)
        summary_path.write_text(
            json.dumps(_summary(probe), indent=2, sort_keys=True) + "\n"
        )
        reloaded = torch_load(probe_path)
        validate_paired_probe(
            reloaded,
            feature_cache=feature["payload"],
            expected_scene=args.scene,
            expected_pairs=proposals["pairs"],
        )
        completion = completion_payload(
            scene=args.scene,
            run_uuid=run_uuid,
            producer=producer,
            compiled_identity=producer["compiled_identity"],
            feature_cache=feature,
            proposals=proposals,
            probe_path=probe_path,
            probe_logical_path=output_root / probe_path.name,
            probe=probe,
            summary_path=summary_path,
            summary_logical_path=output_root / summary_path.name,
        )
        completion_path.write_text(
            json.dumps(completion, indent=2, sort_keys=True) + "\n"
        )
        if json.loads(completion_path.read_text()) != completion:
            raise RuntimeError("P9 completion failed its temporary-root reload")
        if output_root.exists():
            raise FileExistsError("P9 output root appeared before atomic commit")
        os.replace(temporary_root, output_root)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise
    final_completion = output_root / completion_path.name
    load_completion(
        path=final_completion,
        expected_file_sha256=sha256_file(final_completion),
    )
    return {
        "schema": probe["schema"],
        "scene": args.scene,
        "mapping_only": True,
        "uses_test_queries": False,
        "output_root": str(output_root),
        "probe": {
            "path": str(output_root / probe_path.name),
            "sha256": sha256_file(output_root / probe_path.name),
            "content_sha256": probe["content_sha256"],
        },
        "completion": {
            "path": str(final_completion),
            "sha256": sha256_file(final_completion),
        },
        "arms": _summary(probe)["arms"],
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
