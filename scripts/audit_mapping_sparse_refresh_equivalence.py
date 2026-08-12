#!/usr/bin/env python3
"""Prove that an attested sparse refresh preserves frozen Track inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.mapping_density_factor import audit_sparse_refresh_equivalence


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--refreshed-cache", type=Path, required=True)
    parser.add_argument("--source-track-payload", type=Path, required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--expected-source-track-payload-sha256", required=True)
    parser.add_argument("--expected-mapping-keypoints", type=int, required=True)
    parser.add_argument("--expected-nms-radius", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_cache = args.source_cache.resolve()
    refreshed_cache = args.refreshed_cache.resolve()
    source_track_payload = args.source_track_payload.resolve()
    source_cache_sha256 = sha256_file(source_cache)
    refreshed_cache_sha256 = sha256_file(refreshed_cache)
    source_track_payload_sha256 = sha256_file(source_track_payload)
    source = _load(args.source_cache)
    refreshed = _load(args.refreshed_cache)
    audit = audit_sparse_refresh_equivalence(source, refreshed)
    expected_source_cache_sha256 = args.expected_source_cache_sha256.lower()
    expected_source_track_payload_sha256 = (
        args.expected_source_track_payload_sha256.lower()
    )
    checks = {
        "source_cache_sha256_expected": source_cache_sha256
        == expected_source_cache_sha256,
        "source_track_payload_sha256_expected": source_track_payload_sha256
        == expected_source_track_payload_sha256,
        "mapping_keypoints_expected": audit["target_k_mapping"]
        == int(args.expected_mapping_keypoints),
        "nms_radius_expected": audit["target_nms_radius"]
        == int(args.expected_nms_radius),
        "query_order_exact": audit["query_order_exact"],
        "content_equivalent_track_payload_reuse_authorized": audit[
            "content_equivalent_track_payload_reuse_authorized"
        ],
    }
    valid = all(checks.values())
    report = {
        "schema": "lafgs_mapping_sparse_refresh_equivalence",
        "version": 2,
        "uses_test_queries": False,
        "valid": valid,
        "sources": {
            "source_cache": {
                "path": str(source_cache),
                "sha256": source_cache_sha256,
            },
            "refreshed_cache": {
                "path": str(refreshed_cache),
                "sha256": refreshed_cache_sha256,
            },
            "source_track_payload": {
                "path": str(source_track_payload),
                "sha256": source_track_payload_sha256,
            },
        },
        "expected": {
            "source_cache_sha256": expected_source_cache_sha256,
            "source_track_payload_sha256": (expected_source_track_payload_sha256),
            "mapping_keypoints": int(args.expected_mapping_keypoints),
            "nms_radius": int(args.expected_nms_radius),
        },
        "checks": checks,
        "audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
