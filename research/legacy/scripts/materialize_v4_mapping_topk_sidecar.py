#!/usr/bin/env python3
"""Materialize one exact mapping-LOO Top-8 candidate sidecar per scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file
from scripts.v4_assignment_sidecar_common import (
    add_mapping_input_arguments,
    load_mapping_context,
    require_sha,
)
from topology.assignment_replay import (
    materialize_mapping_topk,
    validate_mapping_topk,
)


SOURCE_PATHS = (
    "scripts/materialize_v4_mapping_topk_sidecar.py",
    "scripts/v4_assignment_sidecar_common.py",
    "scripts/evaluate_rendered_track_fullmap.py",
    "topology/assignment_replay.py",
    "topology/deployment_revision.py",
    "evidence/tracks.py",
    "localization/matcher.py",
)


def producer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("mapping Top-K producer worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "torch_version": torch.__version__,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in SOURCE_PATHS
        },
    }


def atomic_save(payload: dict, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        validate_mapping_topk(reloaded)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_mapping_input_arguments(parser)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    identity = producer_identity()
    context = load_mapping_context(args)

    def progress(completed: int, total: int) -> None:
        if completed % 25 == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "event": "mapping_loo_topk_materialization",
                        "queries_complete": completed,
                        "query_count": total,
                    }
                ),
                flush=True,
            )

    sidecar = materialize_mapping_topk(
        state=context.state,
        metric_state_path=context.paths["metric"],
        teacher=context.teacher,
        query_cache=context.cache,
        device=torch.device(args.device),
        anchor_bank_updater=context.updater,
        topk=int(args.topk),
        deployment_row_limit=int(args.deployment_row_limit),
        progress=progress,
    )
    sidecar.update(
        {
            "producer_identity": identity,
            "inputs": {label: str(path) for label, path in context.paths.items()},
            "input_sha256": context.input_sha256,
            "loo": {
                "query_descriptor_excluded_from_affected_anchor_fusion": True,
                "affected_anchor_updates": context.updater.affected_anchor_updates,
                "minimum_affected_anchors_per_query": int(
                    context.affected_anchors_per_query.min()
                ),
                "maximum_affected_anchors_per_query": int(
                    context.affected_anchors_per_query.max()
                ),
                "mean_affected_anchors_per_query": float(
                    context.affected_anchors_per_query.float().mean()
                ),
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if producer_identity() != identity:
        raise RuntimeError("mapping Top-K producer identity changed")
    for label, path in context.paths.items():
        require_sha(path, context.input_sha256[label], label)
    atomic_save(sidecar, output)
    print(
        json.dumps(
            {"output": str(output), "output_sha256": sha256_file(output)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
