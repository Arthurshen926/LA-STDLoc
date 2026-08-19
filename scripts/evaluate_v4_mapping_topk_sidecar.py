#!/usr/bin/env python3
"""Replay Top-1 or partial assignment from a shared exact mapping sidecar."""

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
    load_replay_context,
    require_sha,
)
from topology.assignment_replay import replay_mapping_topk, validate_mapping_topk


SOURCE_PATHS = (
    "scripts/evaluate_v4_mapping_topk_sidecar.py",
    "scripts/v4_assignment_sidecar_common.py",
    "topology/assignment_replay.py",
    "topology/deployment_revision.py",
    "localization/matcher.py",
    "localization/pose_solver.py",
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
        raise RuntimeError("mapping sidecar evaluator worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "torch_version": torch.__version__,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in SOURCE_PATHS
        },
    }


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if reloaded.get("schema") != payload.get("schema"):
            raise RuntimeError("temporary mapping statistics did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()).get("schema") != payload.get("schema"):
            raise RuntimeError("temporary mapping report did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_mapping_input_arguments(parser)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--expected-sidecar-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment-topk", type=int, default=0)
    parser.add_argument("--assignment-dustbin-score", type=float, default=-1.0)
    parser.add_argument("--assignment-maximum-regret", type=float)
    parser.add_argument("--pose-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    identity = producer_identity()
    context = load_replay_context(args)
    sidecar_path = args.sidecar.resolve()
    sidecar_sha256 = require_sha(
        sidecar_path, args.expected_sidecar_sha256, "mapping Top-K sidecar"
    )
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    validate_mapping_topk(sidecar)
    if sidecar.get("input_sha256") != context.input_sha256 or sidecar.get("inputs") != {
        label: str(path) for label, path in context.paths.items()
    }:
        raise ValueError("mapping Top-K sidecar input lineage differs")
    sidecar_identity = sidecar.get("producer_identity", {})
    if sidecar_identity.get("git_commit") != identity["git_commit"]:
        raise ValueError("mapping Top-K sidecar producer commit differs")
    repository = Path(__file__).resolve().parents[1]
    for relative, expected in sidecar_identity.get("source_sha256", {}).items():
        if sha256_file(repository / relative) != expected:
            raise ValueError(f"mapping Top-K producer source differs: {relative}")

    parameters = context.calibration["parameters"]
    statistics = replay_mapping_topk(
        sidecar=sidecar,
        state=context.state,
        teacher=context.teacher,
        assignment_topk=int(args.assignment_topk),
        assignment_dustbin_score=float(args.assignment_dustbin_score),
        assignment_maximum_regret=args.assignment_maximum_regret,
        ransac_reprojection_px=float(parameters["ransac_reprojection_px"]),
        clean_reprojection_px=float(parameters["clean_radius_px"]),
        task_translation_m=float(parameters["task_translation_m"]),
        task_rotation_deg=float(parameters["task_rotation_deg"]),
        seed=int(args.seed),
        pose_workers=int(args.pose_workers),
    )
    statistics = {
        "schema": "lafgs_v4_mapping_topk_replay_statistics",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        **statistics,
    }
    if producer_identity() != identity:
        raise RuntimeError("mapping sidecar evaluator identity changed")
    require_sha(sidecar_path, sidecar_sha256, "mapping Top-K sidecar")
    for label, path in context.paths.items():
        require_sha(path, context.input_sha256[label], label)
    output_dir.mkdir(parents=True, exist_ok=False)
    statistics_path = output_dir / "mapping_topk_replay_statistics.pt"
    atomic_save(statistics, statistics_path)
    report = {
        "schema": "lafgs_v4_mapping_topk_replay_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "producer_identity": identity,
        "inputs": {label: str(path) for label, path in context.paths.items()},
        "input_sha256": context.input_sha256,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sidecar_sha256,
        "configuration": {
            "assignment_topk": int(args.assignment_topk),
            "assignment_dustbin_score": float(args.assignment_dustbin_score),
            "assignment_maximum_regret": args.assignment_maximum_regret,
            "pose_workers": int(args.pose_workers),
            "seed": int(args.seed),
            "one_standard_poselib_call_per_query": True,
        },
        "statistics": str(statistics_path),
        "statistics_sha256": sha256_file(statistics_path),
        "summary": statistics["summary"],
    }
    atomic_json(report, output_dir / "mapping_topk_replay_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
