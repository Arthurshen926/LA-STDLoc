#!/usr/bin/env python3
"""Evaluate every preregistered assignment candidate from one shared load."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file
from scripts.evaluate_v4_mapping_topk_sidecar import (
    apply_confidence_dustbin,
    atomic_json,
    atomic_save,
)
from scripts.v4_assignment_sidecar_common import (
    add_mapping_input_arguments,
    load_replay_context,
    require_sha,
)
from topology.assignment_replay import replay_mapping_topk, validate_mapping_topk


SOURCE_PATHS = (
    "scripts/evaluate_v4_mapping_topk_candidates.py",
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
        raise RuntimeError("multi-candidate evaluator worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "torch_version": torch.__version__,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in SOURCE_PATHS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_mapping_input_arguments(parser)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--expected-sidecar-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-workers", type=int, default=4)
    parser.add_argument("--pose-workers-per-candidate", type=int, default=4)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    candidate_workers = int(args.candidate_workers)
    pose_workers = int(args.pose_workers_per_candidate)
    if candidate_workers < 1 or pose_workers < 1:
        raise ValueError("candidate and pose worker counts must be positive")
    prereg_path = args.preregistration.resolve()
    prereg = json.loads(prereg_path.read_text())
    if prereg.get("uses_test_queries") is not False:
        raise ValueError("candidate preregistration must be mapping-only")
    candidates = dict(prereg["candidates"])
    if not candidates:
        raise ValueError("candidate preregistration is empty")
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
    repository = Path(__file__).resolve().parents[1]
    sidecar_identity = sidecar.get("producer_identity", {})
    if sidecar_identity.get("worktree_clean") is not True:
        raise ValueError("mapping Top-K sidecar producer was not clean")
    for relative, expected in sidecar_identity.get("source_sha256", {}).items():
        if sha256_file(repository / relative) != expected:
            raise ValueError(f"mapping Top-K producer source differs: {relative}")
    parameters = context.calibration["parameters"]

    def evaluate(item: tuple[str, dict]) -> tuple[str, dict, dict]:
        name, configuration = item
        filtered = apply_confidence_dustbin(
            sidecar,
            minimum_margin=float(configuration["assignment_minimum_top1_margin"]),
            dustbin_score=float(configuration["assignment_dustbin_score"]),
        )
        statistics = replay_mapping_topk(
            sidecar=filtered,
            state=context.state,
            teacher=context.teacher,
            assignment_topk=int(configuration["assignment_topk"]),
            assignment_dustbin_score=float(configuration["assignment_dustbin_score"]),
            assignment_maximum_regret=float(configuration["assignment_maximum_regret"]),
            ransac_reprojection_px=float(parameters["ransac_reprojection_px"]),
            clean_reprojection_px=float(parameters["clean_radius_px"]),
            task_translation_m=float(parameters["task_translation_m"]),
            task_rotation_deg=float(parameters["task_rotation_deg"]),
            seed=int(prereg["pose"]["seed"]),
            pose_workers=pose_workers,
        )
        return (
            name,
            configuration,
            {
                "schema": "lafgs_v4_mapping_topk_replay_statistics",
                "version": 1,
                "uses_source_mapping_rgb": False,
                "uses_test_queries": False,
                **statistics,
            },
        )

    output_root.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(
        max_workers=min(candidate_workers, len(candidates))
    ) as pool:
        results = list(pool.map(evaluate, candidates.items()))
    if producer_identity() != identity:
        raise RuntimeError("multi-candidate evaluator identity changed")
    require_sha(sidecar_path, sidecar_sha256, "mapping Top-K sidecar")
    for label, path in context.paths.items():
        require_sha(path, context.input_sha256[label], label)
    summaries = {}
    for name, configuration, statistics in results:
        output = output_root / name
        output.mkdir()
        statistics_path = output / "mapping_topk_replay_statistics.pt"
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
            "preregistration": str(prereg_path),
            "preregistration_sha256": sha256_file(prereg_path),
            "configuration": {
                **configuration,
                "confidence_rejected_query_rows": int(
                    sum(
                        int(
                            (
                                torch.as_tensor(record["scores"])[:, 0]
                                - torch.as_tensor(record["scores"])[:, 1]
                                < float(configuration["assignment_minimum_top1_margin"])
                            ).sum()
                        )
                        for record in sidecar["records"]
                    )
                ),
                "candidate_workers": candidate_workers,
                "pose_workers_per_candidate": pose_workers,
                "seed": int(prereg["pose"]["seed"]),
                "one_standard_poselib_call_per_query": True,
            },
            "statistics": str(statistics_path),
            "statistics_sha256": sha256_file(statistics_path),
            "summary": statistics["summary"],
        }
        atomic_json(report, output / "mapping_topk_replay_report.json")
        summaries[name] = statistics["summary"]
    completion = {
        "schema": "lafgs_v4_mapping_topk_candidate_batch_completion",
        "version": 1,
        "uses_test_queries": False,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sidecar_sha256,
        "candidates": list(candidates),
        "summaries": summaries,
    }
    atomic_json(completion, output_root / "completion.json")
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
