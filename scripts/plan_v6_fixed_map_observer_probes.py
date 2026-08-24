#!/usr/bin/env python3
"""Plan mapping-only virtual probes for the fixed V6 localization plant."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.v6_observer_probes import build_fixed_map_observer_probe_plan


def _load(path: Path, expected: str, label: str) -> tuple[dict, str]:
    path = path.resolve()
    actual = sha256_file(path)
    if actual != str(expected).lower():
        raise ValueError(f"{label} SHA differs")
    return torch.load(path, map_location="cpu", weights_only=False), actual


def _save(payload: dict, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--expected-observation-cache-sha256", required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--expected-feedback-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-pose-budget", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=512)
    parser.add_argument("--anchor-projection-stride", type=int, default=16)
    args = parser.parse_args()
    state, map_sha = _load(args.map, args.expected_map_sha256, "map")
    cache, cache_sha = _load(
        args.observation_cache,
        args.expected_observation_cache_sha256,
        "observation cache",
    )
    evaluation, feedback_sha = _load(
        args.feedback, args.expected_feedback_sha256, "feedback"
    )
    feedback = evaluation.get("feedback", evaluation)
    observations = GaussianRenderObservationProvider(
        cache,
        query_names=list(state["v6_mapping_query_names"]),
        query_bins=state.get("v6_mapping_query_bins"),
    )
    payload = build_fixed_map_observer_probe_plan(
        state,
        observations,
        feedback,
        map_sha256=map_sha,
        observation_cache_sha256=cache_sha,
        feedback_sha256=feedback_sha,
        selected_pose_budget=args.selected_pose_budget,
        maximum_candidates=args.maximum_candidates,
        anchor_projection_stride=args.anchor_projection_stride,
    )
    _save(payload, args.output)
    print(args.output.resolve())
    print(sha256_file(args.output.resolve()))


if __name__ == "__main__":
    main()
