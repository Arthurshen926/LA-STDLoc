#!/usr/bin/env python3
"""Compose a first-pass replay with same-solver continuation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--continuation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = json.loads(Path(args.base).read_text())
    continuation = json.loads(Path(args.continuation).read_text())
    rows = list(base["results"])
    replacement = {row["query"]: row for row in continuation["results"]}
    unknown = sorted(set(replacement) - {row["query"] for row in rows})
    if unknown:
        raise ValueError(f"continuation contains unknown queries: {unknown[:3]}")
    rows = [replacement.get(row["query"], row) for row in rows]
    if len({row["query"] for row in rows}) != len(rows):
        raise ValueError("composed replay contains duplicate queries")
    te = np.asarray([row["te_cm"] for row in rows])
    re = np.asarray([row["re_deg"] for row in rows])
    hypotheses = np.asarray([row["hypotheses"] for row in rows])
    payload = {
        "schema": "lafgs_composed_single_solver_replay",
        "split": base["split"],
        "base": str(Path(args.base).resolve()),
        "continuation": str(Path(args.continuation).resolve()),
        "query_count": len(rows),
        "continuation_query_count": len(replacement),
        "anchor_count": base["anchor_count"],
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(re)),
        "mean_ae_deg": float(np.mean(re)),
        "recall_5cm_percent": float(100 * np.mean(te <= 5)),
        "recall_5cm_5deg_percent": float(
            100 * np.mean((te <= 5) & (re <= 5))
        ),
        "mean_hypotheses": float(np.mean(hypotheses)),
        "median_hypotheses": float(np.median(hypotheses)),
        "p90_hypotheses": float(np.percentile(hypotheses, 90)),
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "results"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
