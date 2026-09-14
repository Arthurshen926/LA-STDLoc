#!/usr/bin/env python3
"""Evaluate sparse LGCV fused with query-specific Top-K geometric feedback."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.hashing import sha256_file
from localization import matcher as matcher_module
from localization import pose_solver as pose_solver_module
from map_learning import v22_sparse_lgcv_feedback as feedback
from map_learning import v21_pose_feedback_transductive as replay_module
from map_learning import v21_topk_geometric_feedback as topk_module


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--expected-stable-map-sha256", required=True)
    parser.add_argument("--frontend-cache", type=Path, action="append", required=True)
    parser.add_argument("--expected-frontend-cache-sha256", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matcher-chunk-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path, expected: str) -> tuple[dict, dict]:
    resolved = path.expanduser().resolve()
    digest = sha256_file(resolved)
    if digest != expected:
        raise ValueError(f"V22 sparse LGCV source SHA differs: {resolved}")
    return torch.load(resolved, map_location="cpu", weights_only=False), {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": int(resolved.stat().st_size),
    }


def main() -> None:
    args = _args()
    if len(args.frontend_cache) != len(args.expected_frontend_cache_sha256):
        raise ValueError("each frontend cache requires one expected SHA")
    stable, stable_source = _load(args.stable_map, args.expected_stable_map_sha256)
    caches = [
        _load(path, digest)
        for path, digest in zip(
            args.frontend_cache, args.expected_frontend_cache_sha256
        )
    ]
    producers = []
    for path in [
        Path(feedback.__file__).resolve(),
        Path(topk_module.__file__).resolve(),
        Path(replay_module.__file__).resolve(),
        Path(matcher_module.__file__).resolve(),
        Path(pose_solver_module.__file__).resolve(),
        Path(__file__).resolve(),
    ]:
        producers.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
        )
    payload = feedback.build_evaluation(
        stable_map=stable,
        cache_payloads=[value[0] for value in caches],
        stable_map_source=stable_source,
        cache_sources=[value[1] for value in caches],
        producer_sources=producers,
        device=args.device,
        matcher_chunk_size=args.matcher_chunk_size,
    )
    for source in [stable_source, *(value[1] for value in caches), *producers]:
        if sha256_file(source["path"]) != source["sha256"]:
            raise RuntimeError(f"V22 sparse LGCV source changed: {source['path']}")
    output = feedback.atomic_torch_save_fresh(payload, args.output)
    print(output)
    print(payload["summary"])


if __name__ == "__main__":
    main()
