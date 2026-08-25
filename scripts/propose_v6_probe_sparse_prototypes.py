#!/usr/bin/env python3
"""Create one probe-trained sparse-prototype V6 candidate without test data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file
from evidence.observation_provider import GaussianRenderObservationProvider
from map_learning.v6_control_actions import probe_conditioned_sparse_prototype_proposal
from topology.v6_anchor_map import compact_projective_deployment_map, identity_metric_state


def _clean_commit() -> str:
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("probe prototype producer requires a clean worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha_load_pt(path: Path, expected: str, label: str) -> tuple[dict, str]:
    path = path.resolve()
    actual = sha256_file(path)
    if actual != str(expected).lower():
        raise ValueError(f"{label} SHA differs")
    return torch.load(path, map_location="cpu", weights_only=False), actual


def _sha_load_json(path: Path, expected: str, label: str) -> tuple[dict, str]:
    path = path.resolve()
    actual = sha256_file(path)
    if actual != str(expected).lower():
        raise ValueError(f"{label} SHA differs")
    return json.loads(path.read_text()), actual


def _atomic_save(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--probe-cache", type=Path, required=True)
    parser.add_argument("--expected-probe-cache-sha256", required=True)
    parser.add_argument("--probe-feedback", type=Path, required=True)
    parser.add_argument("--expected-probe-feedback-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ransac-reprojection-px", type=float, required=True)
    parser.add_argument("--maximum-candidates-per-query", type=int, default=256)
    parser.add_argument("--maximum-correction-set-size", type=int, default=8)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--maximum-extra-prototypes", type=int, default=128)
    parser.add_argument("--maximum-prototypes-per-anchor", type=int, default=1)
    parser.add_argument("--duplicate-cosine", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    producer_commit = _clean_commit()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    state, map_sha = _sha_load_pt(args.map, args.expected_map_sha256, "map")
    cache, cache_sha = _sha_load_pt(
        args.probe_cache, args.expected_probe_cache_sha256, "probe cache"
    )
    feedback, feedback_sha = _sha_load_json(
        args.probe_feedback,
        args.expected_probe_feedback_sha256,
        "probe feedback",
    )
    provider = GaussianRenderObservationProvider(
        cache, query_names=list(cache["query_names"])
    )
    proposal = probe_conditioned_sparse_prototype_proposal(
        state,
        provider,
        feedback,
        source_map_sha256=map_sha,
        probe_cache_sha256=cache_sha,
        probe_feedback_sha256=feedback_sha,
        reprojection_error_px=args.ransac_reprojection_px,
        maximum_candidates_per_query=args.maximum_candidates_per_query,
        maximum_correction_set_size=args.maximum_correction_set_size,
        beam_width=args.beam_width,
        maximum_extra_prototypes=args.maximum_extra_prototypes,
        maximum_prototypes_per_anchor=args.maximum_prototypes_per_anchor,
        duplicate_cosine=args.duplicate_cosine,
        seed=args.seed,
    )
    proposal["provenance"]["v6_producer_git_commit"] = producer_commit
    compact = compact_projective_deployment_map(proposal)
    output.mkdir(parents=True)
    checkpoint_path = output / "training_checkpoint.pt"
    deployment_path = output / "deployment_map.pt"
    _atomic_save(proposal, checkpoint_path)
    _atomic_save(compact, deployment_path)
    deployment_sha = sha256_file(deployment_path)
    metric = identity_metric_state(
        compact, map_path=str(deployment_path), map_sha256=deployment_sha
    )
    metric_path = output / "deployment_identity_metric.pt"
    _atomic_save(metric, metric_path)
    report = {
        "schema": "lafgs_v6_probe_sparse_prototype_proposal",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "producer_git_commit": producer_commit,
        "input_sha256": {
            "map": map_sha,
            "probe_cache": cache_sha,
            "probe_feedback": feedback_sha,
        },
        "output": {
            "training_checkpoint": str(checkpoint_path),
            "training_checkpoint_sha256": sha256_file(checkpoint_path),
            "deployment_map": str(deployment_path),
            "deployment_map_sha256": deployment_sha,
            "deployment_metric": str(metric_path),
            "deployment_metric_sha256": sha256_file(metric_path),
        },
        "control": _jsonable(proposal["v6_probe_prototype_control"]),
    }
    report_path = output / "proposal.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["control"]["training_replay"], indent=2, sort_keys=True))
    print(report_path)
    print(sha256_file(report_path))


if __name__ == "__main__":
    main()
