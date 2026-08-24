#!/usr/bin/env python3
"""Evaluate a rendered virtual-probe cache against one fixed V6 map."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v6_virtual_probe_evaluator import evaluate_fixed_map_virtual_probes


def _load(path: Path, expected: str, label: str) -> tuple[dict, str]:
    actual = sha256_file(path.resolve())
    if actual != str(expected).lower():
        raise ValueError(f"{label} SHA differs")
    return torch.load(path.resolve(), map_location="cpu", weights_only=False), actual


def _atomic_json(payload: dict, output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--probe-cache", type=Path, required=True)
    parser.add_argument("--expected-probe-cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--positive-radius-px", type=float, default=2.0)
    parser.add_argument("--alpha-minimum", type=float, default=0.05)
    parser.add_argument("--ransac-reprojection-px", type=float, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    state, map_sha = _load(args.map, args.expected_map_sha256, "map")
    cache, cache_sha = _load(
        args.probe_cache, args.expected_probe_cache_sha256, "probe cache"
    )
    result = evaluate_fixed_map_virtual_probes(
        state,
        cache,
        map_sha256=map_sha,
        probe_cache_sha256=cache_sha,
        positive_radius_px=args.positive_radius_px,
        alpha_minimum=args.alpha_minimum,
        ransac_reprojection_px=args.ransac_reprojection_px,
        seed=args.seed,
        device=args.device,
    )
    _atomic_json(result, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(args.output.resolve())
    print(sha256_file(args.output.resolve()))


if __name__ == "__main__":
    main()
