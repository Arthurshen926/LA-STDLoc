#!/usr/bin/env python3
"""Audit that two mapping caches differ only by native sparse K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evidence.mapping_density_factor import audit_density_cache_pair


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-cache", type=Path, required=True)
    parser.add_argument("--high-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = _load(args.control_cache)
    high = _load(args.high_cache)
    report = {
        "schema": "lafgs_mapping_density_cache_pair_contract",
        "version": 1,
        "uses_test_queries": False,
        "sources": {
            "control_cache": str(args.control_cache.resolve()),
            "high_cache": str(args.high_cache.resolve()),
        },
        "audit": audit_density_cache_pair(control, high),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
