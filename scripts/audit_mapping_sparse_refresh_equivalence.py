#!/usr/bin/env python3
"""Prove that an attested sparse refresh preserves frozen Track inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = _load(args.source_cache)
    refreshed = _load(args.refreshed_cache)
    audit = audit_sparse_refresh_equivalence(source, refreshed)
    report = {
        "schema": "lafgs_mapping_sparse_refresh_equivalence",
        "version": 1,
        "uses_test_queries": False,
        "sources": {
            "source_cache": str(args.source_cache.resolve()),
            "refreshed_cache": str(args.refreshed_cache.resolve()),
            "source_track_payload": str(args.source_track_payload.resolve()),
        },
        "audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
