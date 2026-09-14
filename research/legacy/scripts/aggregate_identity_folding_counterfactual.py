#!/usr/bin/env python3
"""Aggregate mapping-only identity-folding replays across RANSAC seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topology.equivalence_counterfactual import (
    aggregate_identity_folding_summaries,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.input]
    for report in reports:
        if report.get("schema") != "lafgs_identity_folding_counterfactual":
            raise ValueError("unsupported identity-folding report schema")
        if report.get("uses_test_queries") is not False:
            raise ValueError("seed aggregation accepts mapping-only reports")
    aggregate = aggregate_identity_folding_summaries(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
