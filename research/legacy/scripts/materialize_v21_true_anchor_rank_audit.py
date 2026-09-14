#!/usr/bin/env python3
"""Materialize the diagnostic-only V21 exact true-Anchor global-rank audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning import v21_true_anchor_rank as rank_audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--expected-stable-map-sha256", required=True)
    parser.add_argument("--frontend-cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--expected-frontend-cache-sha256", action="append", required=True
    )
    parser.add_argument("--correspondence-truth", type=Path, required=True)
    parser.add_argument("--expected-correspondence-truth-sha256", required=True)
    parser.add_argument("--geometry-oracle", type=Path, required=True)
    parser.add_argument("--expected-geometry-oracle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--anchor-chunk-size", type=int, default=32768)
    return parser.parse_args()


def _load_source(path: Path, expected_sha256: str, *, label: str) -> tuple[dict, dict]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    digest = sha256_file(resolved)
    if digest != str(expected_sha256):
        raise ValueError(f"{label} SHA256 differs")
    source = {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": int(resolved.stat().st_size),
    }
    return torch.load(resolved, map_location="cpu", weights_only=False), source


def _producer_source(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _verify_unchanged(source: dict) -> None:
    path = Path(source["path"])
    if int(path.stat().st_size) != int(source["size_bytes"]) or sha256_file(path) != source["sha256"]:
        raise RuntimeError(f"input changed while auditing: {path}")


def main() -> None:
    args = _parse_args()
    if len(args.frontend_cache) != len(args.expected_frontend_cache_sha256):
        raise ValueError("frontend cache paths and expected SHA256 values differ")
    stable_map, stable_source = _load_source(
        args.stable_map,
        args.expected_stable_map_sha256,
        label="stable map",
    )
    cache_entries = [
        _load_source(path, digest, label=f"frontend cache {offset}")
        for offset, (path, digest) in enumerate(
            zip(args.frontend_cache, args.expected_frontend_cache_sha256)
        )
    ]
    truth, truth_source = _load_source(
        args.correspondence_truth,
        args.expected_correspondence_truth_sha256,
        label="correspondence truth",
    )
    oracle, oracle_source = _load_source(
        args.geometry_oracle,
        args.expected_geometry_oracle_sha256,
        label="geometry oracle",
    )
    module_path = Path(rank_audit.__file__).resolve()
    script_path = Path(__file__).resolve()
    producer_sources = [_producer_source(module_path), _producer_source(script_path)]
    payload = rank_audit.build_true_anchor_rank_audit(
        stable_map=stable_map,
        frontend_caches=[value[0] for value in cache_entries],
        correspondence_truth=truth,
        geometry_oracle=oracle,
        stable_map_source=stable_source,
        frontend_cache_sources=[value[1] for value in cache_entries],
        correspondence_truth_source=truth_source,
        geometry_oracle_source=oracle_source,
        producer_sources=producer_sources,
        device=args.device,
        query_batch_size=args.query_batch_size,
        anchor_chunk_size=args.anchor_chunk_size,
    )
    for source in [
        stable_source,
        *(value[1] for value in cache_entries),
        truth_source,
        oracle_source,
        *producer_sources,
    ]:
        _verify_unchanged(source)
    output = rank_audit.atomic_torch_save_fresh(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256_file(output),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
