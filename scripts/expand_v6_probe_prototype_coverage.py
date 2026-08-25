#!/usr/bin/env python3
"""Expand a probe candidate over training-side controllable Anchor support."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file
from evidence.observation_provider import GaussianRenderObservationProvider
from map_learning.v6_control_actions import expand_probe_prototype_support
from topology.v6_anchor_map import compact_projective_deployment_map, identity_metric_state


def _load_pt(path: Path, expected: str, label: str) -> tuple[dict, str]:
    actual = sha256_file(path.resolve())
    if actual != str(expected).lower():
        raise ValueError(f"{label} SHA differs")
    return torch.load(path.resolve(), map_location="cpu", weights_only=False), actual


def _save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--probe-cache", type=Path, required=True)
    parser.add_argument("--expected-probe-cache-sha256", required=True)
    parser.add_argument("--probe-feedback", type=Path, required=True)
    parser.add_argument("--expected-probe-feedback-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-total-prototypes", type=int, default=8192)
    parser.add_argument("--maximum-prototypes-per-anchor", type=int, default=2)
    parser.add_argument("--duplicate-cosine", type=float, default=0.995)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("probe coverage producer requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    state, checkpoint_sha = _load_pt(args.checkpoint, args.expected_checkpoint_sha256, "checkpoint")
    cache, cache_sha = _load_pt(args.probe_cache, args.expected_probe_cache_sha256, "probe cache")
    feedback_sha = sha256_file(args.probe_feedback.resolve())
    if feedback_sha != str(args.expected_probe_feedback_sha256).lower():
        raise ValueError("probe feedback SHA differs")
    feedback = json.loads(args.probe_feedback.resolve().read_text())
    provider = GaussianRenderObservationProvider(cache, query_names=list(cache["query_names"]))
    proposal = expand_probe_prototype_support(
        state,
        provider,
        feedback,
        maximum_total_prototypes=args.maximum_total_prototypes,
        maximum_prototypes_per_anchor=args.maximum_prototypes_per_anchor,
        duplicate_cosine=args.duplicate_cosine,
    )
    proposal["provenance"] = {
        **dict(proposal.get("provenance", {})),
        "v6_producer_git_commit": commit,
        "v6_probe_coverage_parent_checkpoint_sha256": checkpoint_sha,
        "v6_probe_coverage_feedback_sha256": feedback_sha,
    }
    compact = compact_projective_deployment_map(proposal)
    output.mkdir(parents=True)
    checkpoint = output / "training_checkpoint.pt"
    deployment = output / "deployment_map.pt"
    _save(proposal, checkpoint)
    _save(compact, deployment)
    deployment_sha = sha256_file(deployment)
    metric = identity_metric_state(compact, map_path=str(deployment), map_sha256=deployment_sha)
    metric_path = output / "deployment_identity_metric.pt"
    _save(metric, metric_path)
    report = {
        "schema": "lafgs_v6_probe_prototype_coverage_proposal",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "producer_git_commit": commit,
        "input_sha256": {
            "checkpoint": checkpoint_sha,
            "probe_cache": cache_sha,
            "probe_feedback": feedback_sha,
        },
        "output": {
            "training_checkpoint": str(checkpoint),
            "training_checkpoint_sha256": sha256_file(checkpoint),
            "deployment_map": str(deployment),
            "deployment_map_sha256": deployment_sha,
            "deployment_metric": str(metric_path),
            "deployment_metric_sha256": sha256_file(metric_path),
        },
        "control": {
            key: value.tolist() if isinstance(value, torch.Tensor) else value
            for key, value in proposal["v6_probe_prototype_coverage"].items()
        },
    }
    report_path = output / "proposal.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["control"], indent=2, sort_keys=True))
    print(report_path)
    print(sha256_file(report_path))


if __name__ == "__main__":
    main()
