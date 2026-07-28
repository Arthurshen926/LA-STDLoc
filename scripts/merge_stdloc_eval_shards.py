#!/usr/bin/env python3
"""Merge disjoint STDLoc evaluation shards and recompute global metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--expected-lists", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for directory in map(Path, args.results):
        rows.extend(json.loads((directory / "results.json").read_text()))
    names = [row["image_name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("evaluation shards contain duplicate image names")
    expected = []
    for path in map(Path, args.expected_lists):
        expected.extend(json.loads(path.read_text()))
    if len(expected) != len(set(expected)):
        raise ValueError("expected camera lists contain duplicates")
    if set(names) != set(expected):
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(f"camera coverage mismatch: missing={missing}, extra={extra}")
    by_name = {row["image_name"]: row for row in rows}
    rows = [by_name[name] for name in expected]

    te = np.asarray([float(row["sparse_TE"]) for row in rows])
    ae = np.asarray([float(row["sparse_AE"]) for row in rows])
    diagnostic_keys = sorted(
        set.intersection(
            *[
                {
                    key
                    for key, value in row["sparse"].items()
                    if key.startswith("sparse_diag_")
                    and isinstance(value, (int, float))
                }
                for row in rows
            ]
        )
    )
    diagnostics = {
        f"{key}_mean": float(np.mean([row["sparse"][key] for row in rows]))
        for key in diagnostic_keys
    }
    summary = {
        "schema": "stdloc_merged_evaluation_shards",
        "evaluation_camera_count": len(rows),
        "sparse": {
            "median_te": float(np.median(te)),
            "mean_te": float(np.mean(te)),
            "p90_te": float(np.percentile(te, 90)),
            "median_ae": float(np.median(ae)),
            "recall_5cm_5d": float(np.mean((te <= 5) & (ae <= 5))),
        },
        "sparse_diagnostics": diagnostics,
        "source_results": [str(Path(path).resolve()) for path in args.results],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output.parent / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
