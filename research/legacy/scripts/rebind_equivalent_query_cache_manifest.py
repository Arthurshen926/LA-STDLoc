#!/usr/bin/env python3
"""Bind an exact sparse-cache refresh to a frozen bootstrap contract.

The parent manifest remains immutable.  The output changes only the declared
query-cache path/SHA and records the exact equivalence audit that authorizes
reuse of the frozen Track payload with the refreshed cache.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from common.hashing import sha256_file


def _sha256(value: str, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return normalized


def _resolved_entry(value: object, *, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain path and SHA-256")
    path = Path(str(value.get("path", ""))).resolve()
    sha256 = _sha256(str(value.get("sha256", "")), label=f"{label} SHA-256")
    return path, sha256


def rebind_equivalent_query_cache_manifest(
    *,
    parent: dict,
    equivalence: dict,
    parent_manifest_path: Path,
    parent_manifest_sha256: str,
    equivalence_path: Path,
    equivalence_sha256: str,
    source_cache_path: Path,
    source_cache_sha256: str,
    refreshed_cache_path: Path,
    refreshed_cache_sha256: str,
    source_track_payload_path: Path,
    source_track_payload_sha256: str,
) -> dict:
    """Return a fresh-cache-bound copy of a frozen bootstrap manifest."""
    parent_manifest_path = parent_manifest_path.resolve()
    equivalence_path = equivalence_path.resolve()
    source_cache_path = source_cache_path.resolve()
    refreshed_cache_path = refreshed_cache_path.resolve()
    source_track_payload_path = source_track_payload_path.resolve()
    parent_manifest_sha256 = _sha256(
        parent_manifest_sha256, label="Parent manifest SHA-256"
    )
    equivalence_sha256 = _sha256(equivalence_sha256, label="Equivalence report SHA-256")
    source_cache_sha256 = _sha256(source_cache_sha256, label="Source cache SHA-256")
    refreshed_cache_sha256 = _sha256(
        refreshed_cache_sha256, label="Refreshed cache SHA-256"
    )
    source_track_payload_sha256 = _sha256(
        source_track_payload_sha256, label="Source Track payload SHA-256"
    )
    if equivalence.get("schema") != "lafgs_mapping_sparse_refresh_equivalence":
        raise ValueError("Unexpected sparse-refresh equivalence schema")
    if equivalence.get("version") != 2:
        raise ValueError("Sparse-refresh equivalence must use version 2")
    if equivalence.get("uses_test_queries") is not False:
        raise ValueError("Sparse-refresh equivalence must be mapping-only")
    if equivalence.get("valid") is not True:
        raise ValueError("Sparse-refresh equivalence is not valid")
    checks = equivalence.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("Sparse-refresh equivalence checks did not all pass")
    audit = equivalence.get("audit")
    if (
        not isinstance(audit, dict)
        or audit.get("content_equivalent_track_payload_reuse_authorized") is not True
    ):
        raise ValueError("Sparse refresh does not authorize Track-payload reuse")
    report_source_path, report_source_sha = _resolved_entry(
        equivalence.get("sources", {}).get("source_cache"),
        label="Equivalence source cache",
    )
    report_refreshed_path, report_refreshed_sha = _resolved_entry(
        equivalence.get("sources", {}).get("refreshed_cache"),
        label="Equivalence refreshed cache",
    )
    report_track_path, report_track_sha = _resolved_entry(
        equivalence.get("sources", {}).get("source_track_payload"),
        label="Equivalence source Track payload",
    )
    if (report_source_path, report_source_sha) != (
        source_cache_path,
        source_cache_sha256,
    ):
        raise ValueError("Equivalence report names a different source cache")
    if (report_refreshed_path, report_refreshed_sha) != (
        refreshed_cache_path,
        refreshed_cache_sha256,
    ):
        raise ValueError("Equivalence report names a different refreshed cache")
    if (report_track_path, report_track_sha) != (
        source_track_payload_path,
        source_track_payload_sha256,
    ):
        raise ValueError("Equivalence report names a different Track payload")
    arguments = parent.get("arguments")
    inputs = parent.get("inputs")
    if not isinstance(arguments, dict) or not isinstance(inputs, dict):
        raise ValueError("Parent bootstrap manifest lacks arguments/inputs")
    parent_query_path = Path(str(arguments.get("query_cache_path", ""))).resolve()
    input_query = inputs.get("query_cache_path")
    if not isinstance(input_query, dict):
        raise ValueError("Parent bootstrap manifest lacks query-cache input lineage")
    input_query_path = Path(str(input_query.get("path", ""))).resolve()
    if parent_query_path != source_cache_path or input_query_path != source_cache_path:
        raise ValueError("Parent bootstrap manifest names a different source cache")

    rebound = deepcopy(parent)
    rebound["arguments"]["query_cache_path"] = str(refreshed_cache_path)
    rebound["inputs"]["query_cache_path"] = {
        "path": str(refreshed_cache_path),
        "sha256": refreshed_cache_sha256,
    }
    rebound["equivalent_query_cache_rebind"] = {
        "schema": "lafgs_equivalent_query_cache_rebind",
        "version": 1,
        "uses_test_queries": False,
        "parent_manifest": {
            "path": str(parent_manifest_path),
            "sha256": parent_manifest_sha256,
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
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
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
    parent_manifest = args.parent_manifest.resolve()
    equivalence_report = args.equivalence_report.resolve()
    if sha256_file(parent_manifest) != args.expected_parent_manifest_sha256.lower():
        raise ValueError("Parent manifest SHA-256 differs from expected")
    if (
        sha256_file(equivalence_report)
        != args.expected_equivalence_report_sha256.lower()
    ):
        raise ValueError("Equivalence report SHA-256 differs from expected")
    source_track_payload = args.source_track_payload.resolve()
    if sha256_file(source_track_payload) != (
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
        raise ValueError("Refusing to overwrite an existing rebound manifest")
    parent = json.loads(parent_manifest.read_text())
    equivalence = json.loads(equivalence_report.read_text())
    rebound = rebind_equivalent_query_cache_manifest(
        parent=parent,
        equivalence=equivalence,
        parent_manifest_path=parent_manifest,
        parent_manifest_sha256=args.expected_parent_manifest_sha256,
        equivalence_path=equivalence_report,
        equivalence_sha256=args.expected_equivalence_report_sha256,
        source_cache_path=args.source_cache,
        source_cache_sha256=args.expected_source_cache_sha256,
        refreshed_cache_path=args.refreshed_cache,
        refreshed_cache_sha256=args.expected_refreshed_cache_sha256,
        source_track_payload_path=source_track_payload,
        source_track_payload_sha256=args.expected_source_track_payload_sha256,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rebound, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(output), "sha256": sha256_file(output)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
