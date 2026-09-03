#!/usr/bin/env python3
"""Audit whether a rendered mapping prior is safe enough to deploy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.mapping_prior_admission import audit_mapping_prior_admission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--expected-candidates-sha256", required=True)
    parser.add_argument("--high-identity-reliability", type=float, required=True)
    parser.add_argument("--high-geometry-reliability", type=float, required=True)
    parser.add_argument("--minimum-high-reliability-count", type=int, required=True)
    parser.add_argument("--minimum-high-reliability-fraction", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.candidates.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if sha256_file(source) != args.expected_candidates_sha256:
        raise ValueError("candidate SHA256 mismatch")
    candidates = torch.load(source, map_location="cpu", weights_only=False)
    report = audit_mapping_prior_admission(
        candidates,
        high_identity_reliability=args.high_identity_reliability,
        high_geometry_reliability=args.high_geometry_reliability,
        minimum_high_reliability_count=args.minimum_high_reliability_count,
        minimum_high_reliability_fraction=args.minimum_high_reliability_fraction,
    )
    if sha256_file(source) != args.expected_candidates_sha256:
        raise ValueError("candidate changed during admission audit")
    report["input"] = {
        "candidates": str(source),
        "candidates_sha256": args.expected_candidates_sha256,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
