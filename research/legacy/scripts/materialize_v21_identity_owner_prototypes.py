#!/usr/bin/env python3
"""Materialize the strict calibration-medoid V21 owner-prototype arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v21_identity_owner_prototype import (
    METADATA_FIELD,
    build_identity_owner_prototype_candidate,
    validate_identity_owner_candidate,
)
from map_learning.v21_pose_feedback_transductive import (
    atomic_torch_save_fresh,
    source_record,
    verify_source_record,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_SOURCES = (
    "localization/matcher.py",
    "map_learning/v21_identity_calibration.py",
    "map_learning/v21_identity_owner_prototype.py",
    "map_learning/v21_pose_feedback_transductive.py",
    "map_learning/v21_test_cache.py",
    "scripts/materialize_v21_identity_owner_prototypes.py",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--expected-stable-map-sha256", required=True)
    parser.add_argument("--identity-calibration", type=Path, required=True)
    parser.add_argument("--expected-identity-calibration-sha256", required=True)
    parser.add_argument("--adaptation-cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--expected-adaptation-cache-sha256", action="append", required=True
    )
    parser.add_argument("--maximum-total-prototypes", type=int, default=128)
    parser.add_argument("--prototype-activation-threshold", type=float)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(source: dict) -> dict:
    return torch.load(source["path"], map_location="cpu", weights_only=False)


def main() -> None:
    args = _parse_args()
    if len(args.adaptation_cache) != len(args.expected_adaptation_cache_sha256):
        raise ValueError("each adaptation cache needs one expected SHA256")
    stable_source = source_record(
        args.stable_map,
        sha256_file_fn=sha256_file,
        expected_sha256=args.expected_stable_map_sha256,
    )
    calibration_source = source_record(
        args.identity_calibration,
        sha256_file_fn=sha256_file,
        expected_sha256=args.expected_identity_calibration_sha256,
    )
    cache_sources = [
        source_record(path, sha256_file_fn=sha256_file, expected_sha256=digest)
        for path, digest in zip(
            args.adaptation_cache, args.expected_adaptation_cache_sha256
        )
    ]
    if len({source["path"] for source in cache_sources}) != len(cache_sources):
        raise ValueError("adaptation cache source registry is duplicated")
    producer_sources = [
        source_record(path, sha256_file_fn=sha256_file)
        for path in (REPOSITORY_ROOT / value for value in PRODUCER_SOURCES)
    ]
    stable_map = _load(stable_source)
    calibration = _load(calibration_source)
    caches = [_load(source) for source in cache_sources]
    candidate = build_identity_owner_prototype_candidate(
        stable_map=stable_map,
        calibration=calibration,
        adaptation_cache_payloads=caches,
        stable_map_source=stable_source,
        calibration_source=calibration_source,
        adaptation_cache_sources=cache_sources,
        producer_sources=producer_sources,
        maximum_total_prototypes=args.maximum_total_prototypes,
        prototype_activation_threshold=args.prototype_activation_threshold,
    )
    for source in [
        stable_source,
        calibration_source,
        *cache_sources,
        *producer_sources,
    ]:
        verify_source_record(source, sha256_file_fn=sha256_file)
    output = atomic_torch_save_fresh(
        candidate,
        args.output,
        validator=lambda value: validate_identity_owner_candidate(
            value, stable_map=stable_map
        ),
    )
    metadata = candidate[METADATA_FIELD]
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": sha256_file(output),
                "accepted_identity_count": metadata["accepted_identity_count"],
                "added_prototype_count": metadata["added_prototype_count"],
                "source_query_count": metadata["source_query_count"],
                "prototype_activation_threshold": metadata[
                    "prototype_activation_threshold"
                ],
                "deployment_authorized": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
