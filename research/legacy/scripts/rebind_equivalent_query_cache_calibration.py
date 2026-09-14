#!/usr/bin/env python3
"""Rebind frozen calibration numbers to an exact-equivalent query cache."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from common.calibration import (
    validate_equivalent_query_cache_calibration_parent,
)
from common.hashing import sha256_file


def _sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return normalized


def rebind_equivalent_query_cache_calibration(
    *,
    parent: dict,
    parent_path: Path,
    parent_sha256: str,
    equivalence_path: Path,
    equivalence_sha256: str,
    source_cache_path: Path,
    source_cache_sha256: str,
    refreshed_cache_path: Path,
    refreshed_cache_sha256: str,
    source_track_payload_path: Path,
    source_track_payload_sha256: str,
) -> dict:
    parent_path = parent_path.resolve()
    equivalence_path = equivalence_path.resolve()
    source_cache_path = source_cache_path.resolve()
    refreshed_cache_path = refreshed_cache_path.resolve()
    source_track_payload_path = source_track_payload_path.resolve()
    parent_sha256 = _sha256(parent_sha256, label="Parent calibration SHA-256")
    equivalence_sha256 = _sha256(equivalence_sha256, label="Equivalence report SHA-256")
    source_cache_sha256 = _sha256(source_cache_sha256, label="Source cache SHA-256")
    refreshed_cache_sha256 = _sha256(
        refreshed_cache_sha256, label="Refreshed cache SHA-256"
    )
    source_track_payload_sha256 = _sha256(
        source_track_payload_sha256, label="Source Track payload SHA-256"
    )
    sources = parent.get("sources")
    if (
        parent.get("schema") != "lafgs_mapping_only_scene_calibration"
        or int(parent.get("version", 0)) < 2
        or not isinstance(sources, dict)
        or sources.get("uses_test_queries") is not False
        or Path(str(sources.get("query_cache", ""))).resolve() != source_cache_path
        or Path(str(sources.get("track_payload", ""))).resolve()
        != source_track_payload_path
    ):
        raise ValueError("Parent calibration is not the frozen mapping control")
    rebound = deepcopy(parent)
    rebound["sources"] = {
        **sources,
        "query_cache": str(refreshed_cache_path),
        "query_cache_sha256": refreshed_cache_sha256,
        "uses_test_queries": False,
    }
    rebound["equivalent_query_cache_rebind"] = {
        "schema": "lafgs_equivalent_query_cache_calibration_rebind",
        "version": 1,
        "uses_test_queries": False,
        "parent_calibration": {
            "path": str(parent_path),
            "sha256": parent_sha256,
        },
        "equivalence_report": {
            "path": str(equivalence_path),
            "sha256": equivalence_sha256,
        },
        "source_cache": {
            "path": str(source_cache_path),
            "sha256": source_cache_sha256,
        },
        "refreshed_cache": {
            "path": str(refreshed_cache_path),
            "sha256": refreshed_cache_sha256,
        },
        "source_track_payload": {
            "path": str(source_track_payload_path),
            "sha256": source_track_payload_sha256,
        },
    }
    return rebound


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--equivalence-report", type=Path, required=True)
    parser.add_argument("--expected-equivalence-report-sha256", required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--refreshed-cache", type=Path, required=True)
    parser.add_argument("--expected-refreshed-cache-sha256", required=True)
    parser.add_argument("--source-track-payload", type=Path, required=True)
    parser.add_argument("--expected-source-track-payload-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_path = args.parent.resolve()
    equivalence_path = args.equivalence_report.resolve()
    source_track_payload_path = args.source_track_payload.resolve()
    if sha256_file(parent_path) != args.expected_parent_sha256.lower():
        raise ValueError("Parent calibration SHA-256 differs from expected")
    if sha256_file(equivalence_path) != (
        args.expected_equivalence_report_sha256.lower()
    ):
        raise ValueError("Equivalence report SHA-256 differs from expected")
    if sha256_file(source_track_payload_path) != (
        args.expected_source_track_payload_sha256.lower()
    ):
        raise ValueError("Source Track payload SHA-256 differs from expected")
    for path, label in (
        (args.source_cache.resolve(), "Source cache"),
        (args.refreshed_cache.resolve(), "Refreshed cache"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("Refusing to overwrite an existing rebound calibration")
    parent = json.loads(parent_path.read_text())
    rebound = rebind_equivalent_query_cache_calibration(
        parent=parent,
        parent_path=parent_path,
        parent_sha256=args.expected_parent_sha256,
        equivalence_path=equivalence_path,
        equivalence_sha256=args.expected_equivalence_report_sha256,
        source_cache_path=args.source_cache,
        source_cache_sha256=args.expected_source_cache_sha256,
        refreshed_cache_path=args.refreshed_cache,
        refreshed_cache_sha256=args.expected_refreshed_cache_sha256,
        source_track_payload_path=source_track_payload_path,
        source_track_payload_sha256=args.expected_source_track_payload_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(rebound, indent=2, sort_keys=True) + "\n")
    try:
        validate_equivalent_query_cache_calibration_parent(
            rebound,
            parent_path=temporary,
            query_cache_path=args.refreshed_cache,
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
