#!/usr/bin/env python3
"""Run paired exact cached evaluation of a V21 owner-prototype candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v21_pose_feedback_transductive import (
    atomic_torch_save_fresh,
    build_cached_evaluation_payload,
    source_record,
    validate_cached_evaluation_payload,
    verify_source_record,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--expected-stable-map-sha256", required=True)
    parser.add_argument("--candidate-map", type=Path, required=True)
    parser.add_argument("--expected-candidate-map-sha256", required=True)
    parser.add_argument("--frontend-cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--expected-frontend-cache-sha256", action="append", required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcher-chunk-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(source: dict) -> dict:
    return torch.load(source["path"], map_location="cpu", weights_only=False)


def main() -> None:
    args = _parse_args()
    if len(args.frontend_cache) != len(args.expected_frontend_cache_sha256):
        raise ValueError("each frontend cache needs one expected SHA256")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    stable_source = source_record(
        args.stable_map,
        sha256_file_fn=sha256_file,
        expected_sha256=args.expected_stable_map_sha256,
    )
    candidate_source = source_record(
        args.candidate_map,
        sha256_file_fn=sha256_file,
        expected_sha256=args.expected_candidate_map_sha256,
    )
    cache_sources = [
        source_record(path, sha256_file_fn=sha256_file, expected_sha256=digest)
        for path, digest in zip(
            args.frontend_cache, args.expected_frontend_cache_sha256
        )
    ]
    if len({value["path"] for value in cache_sources}) != len(cache_sources) or len(
        {value["sha256"] for value in cache_sources}
    ) != len(cache_sources):
        raise ValueError("frontend cache source registry is duplicated")
    stable_map = _load(stable_source)
    candidate_map = _load(candidate_source)
    caches = [_load(value) for value in cache_sources]
    result = build_cached_evaluation_payload(
        stable_map=stable_map,
        candidate_map=candidate_map,
        cache_payloads=caches,
        stable_map_source=stable_source,
        candidate_map_source=candidate_source,
        cache_sources=cache_sources,
        matcher_chunk_size=args.matcher_chunk_size,
        device=args.device,
    )
    for value in [stable_source, candidate_source, *cache_sources]:
        verify_source_record(value, sha256_file_fn=sha256_file)
    output = atomic_torch_save_fresh(
        result,
        args.output,
        validator=validate_cached_evaluation_payload,
    )
    print(output)


if __name__ == "__main__":
    main()
