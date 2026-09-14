#!/usr/bin/env python3
"""Materialize a quarantined V21 owner-prototype candidate from adaptation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.v21_pose_feedback_transductive import (
    atomic_torch_save_fresh,
    build_pose_feedback_transductive_candidate,
    source_record,
    validate_candidate_map,
    verify_source_record,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stable-map", type=Path, required=True)
    parser.add_argument("--expected-stable-map-sha256", required=True)
    parser.add_argument("--adaptation-cache", type=Path, action="append", required=True)
    parser.add_argument(
        "--expected-adaptation-cache-sha256", action="append", required=True
    )
    parser.add_argument("--gaussian-oracle-aggregate", type=Path, required=True)
    parser.add_argument("--expected-gaussian-oracle-aggregate-sha256", required=True)
    parser.add_argument("--provisional-calibration", type=Path)
    parser.add_argument("--expected-provisional-calibration-sha256")
    parser.add_argument("--maximum-bundle-size", type=int, default=8)
    parser.add_argument("--maximum-source-queries", type=int, default=16)
    parser.add_argument("--maximum-total-prototypes", type=int, default=64)
    parser.add_argument("--maximum-prototypes-per-anchor", type=int, default=4)
    parser.add_argument(
        "--one-assignment-max-translation-cm", type=float, default=4.0
    )
    parser.add_argument(
        "--allow-one-assignment-r5-only",
        action="store_true",
        help="Disable the default strict one-assignment TE<4cm margin gate.",
    )
    parser.add_argument("--require-provisional-edge", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(source: dict) -> dict:
    return torch.load(source["path"], map_location="cpu", weights_only=False)


def main() -> None:
    args = _parse_args()
    if len(args.adaptation_cache) != len(args.expected_adaptation_cache_sha256):
        raise ValueError("each adaptation cache needs one expected SHA256")
    if (args.provisional_calibration is None) != (
        args.expected_provisional_calibration_sha256 is None
    ):
        raise ValueError("provisional calibration path and expected SHA must be paired")
    stable_source = source_record(
        args.stable_map,
        sha256_file_fn=sha256_file,
        expected_sha256=args.expected_stable_map_sha256,
    )
    cache_sources = [
        source_record(path, sha256_file_fn=sha256_file, expected_sha256=digest)
        for path, digest in zip(
            args.adaptation_cache, args.expected_adaptation_cache_sha256
        )
    ]
    if len({value["path"] for value in cache_sources}) != len(cache_sources) or len(
        {value["sha256"] for value in cache_sources}
    ) != len(cache_sources):
        raise ValueError("adaptation cache source registry is duplicated")
    oracle_source = source_record(
        args.gaussian_oracle_aggregate,
        sha256_file_fn=sha256_file,
        expected_sha256=args.expected_gaussian_oracle_aggregate_sha256,
    )
    calibration_source = (
        source_record(
            args.provisional_calibration,
            sha256_file_fn=sha256_file,
            expected_sha256=args.expected_provisional_calibration_sha256,
        )
        if args.provisional_calibration is not None
        else None
    )
    stable_map = _load(stable_source)
    caches = [_load(value) for value in cache_sources]
    oracle = _load(oracle_source)
    embedded_oracle_sources = [
        source_record(
            value["path"],
            sha256_file_fn=sha256_file,
            expected_sha256=value["sha256"],
        )
        for value in [
            *oracle.get("input", {}).get("gaussian_support", ()),
            *oracle.get("oracle_shards", ()),
        ]
    ]
    calibration = _load(calibration_source) if calibration_source is not None else None
    candidate = build_pose_feedback_transductive_candidate(
        stable_map=stable_map,
        adaptation_cache_payloads=caches,
        gaussian_oracle_aggregate=oracle,
        stable_map_source=stable_source,
        adaptation_cache_sources=cache_sources,
        oracle_source=oracle_source,
        provisional_calibration=calibration,
        calibration_source=calibration_source,
        maximum_bundle_size=args.maximum_bundle_size,
        maximum_source_queries=args.maximum_source_queries,
        maximum_total_prototypes=args.maximum_total_prototypes,
        maximum_prototypes_per_anchor=args.maximum_prototypes_per_anchor,
        require_one_assignment_translation_below_cm=(
            None
            if args.allow_one_assignment_r5_only
            else args.one_assignment_max_translation_cm
        ),
        require_provisional_edge=args.require_provisional_edge,
    )
    for value in [
        stable_source,
        *cache_sources,
        oracle_source,
        *embedded_oracle_sources,
    ]:
        verify_source_record(value, sha256_file_fn=sha256_file)
    if calibration_source is not None:
        verify_source_record(calibration_source, sha256_file_fn=sha256_file)
    output = atomic_torch_save_fresh(
        candidate,
        args.output,
        validator=lambda value: validate_candidate_map(value, stable_map=stable_map),
    )
    print(output)


if __name__ == "__main__":
    main()
