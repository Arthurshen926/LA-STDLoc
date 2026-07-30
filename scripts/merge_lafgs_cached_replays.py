#!/usr/bin/env python3
"""Merge disjoint cached deployment replay shards with identity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge(paths: list[str]) -> dict:
    payloads = [json.loads(Path(path).read_text()) for path in paths]
    if not payloads:
        raise ValueError("at least one replay is required")
    identity_fields = ("schema", "split", "map", "metric_state", "anchor_count")
    reference = {field: payloads[0].get(field) for field in identity_fields}
    rows = []
    for payload in payloads:
        current = {field: payload.get(field) for field in identity_fields}
        if current != reference:
            raise ValueError("replay shard identity mismatch")
        rows.extend(payload["results"])
    names = [str(row["query"]) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("replay shards contain duplicate queries")
    rows.sort(key=lambda row: str(row["query"]))
    translation = np.asarray([float(row["te_cm"]) for row in rows])
    rotation = np.asarray([float(row["re_deg"]) for row in rows])
    hypotheses = np.asarray(
        [float(row["hypotheses"]) for row in rows if row.get("hypotheses") is not None]
    )
    dependency = [
        bool(row.get("dependency_sampler_used", False)) for row in rows
    ]
    rescue = [bool(row.get("dependency_rescue_used", False)) for row in rows]
    inlier_ratio = np.asarray(
        [
            float(row["inlier_count"]) / max(int(row["match_count"]), 1)
            for row in rows
        ]
    )
    raw_precision = np.asarray(
        [float(row["raw_gt_precision_2px"]) for row in rows]
    )
    inlier_precision = np.asarray(
        [float(row["inlier_gt_precision_2px"]) for row in rows]
    )
    matching_ms = np.asarray(
        [float(row["matching_ms"]) for row in rows if "matching_ms" in row]
    )
    context_ms = np.asarray(
        [float(row.get("context_ms", 0.0)) for row in rows]
    )
    ransac_ms = np.asarray(
        [float(row["ransac_ms"]) for row in rows if "ransac_ms" in row]
    )
    output = {
        **reference,
        "runtime_scope": "cached_descriptor_matching_and_pnp",
        "feature_extraction_included": False,
        "source_shards": [str(Path(path).resolve()) for path in paths],
        "query_count": len(rows),
        "median_te_cm": float(np.median(translation)),
        "mean_te_cm": float(np.mean(translation)),
        "p90_te_cm": float(np.percentile(translation, 90)),
        "median_ae_deg": float(np.median(rotation)),
        "mean_ae_deg": float(np.mean(rotation)),
        "recall_5cm_5deg_percent": float(
            100.0 * np.mean((translation <= 5.0) & (rotation <= 5.0))
        ),
        "mean_solver_inlier_ratio_percent": float(100.0 * inlier_ratio.mean()),
        "raw_gt_precision_2px_percent": float(100.0 * raw_precision.mean()),
        "inlier_gt_precision_2px_percent": float(
            100.0 * inlier_precision.mean()
        ),
        "dependency_sampler_query_count": int(sum(dependency)),
        "dependency_sampler_query_fraction_percent": float(
            100.0 * np.mean(dependency)
        ),
        "dependency_rescue_query_count": int(sum(rescue)),
        "mean_hypotheses": float(np.mean(hypotheses))
        if hypotheses.size
        else None,
        "median_hypotheses": float(np.median(hypotheses))
        if hypotheses.size
        else None,
        "p90_hypotheses": float(np.percentile(hypotheses, 90))
        if hypotheses.size
        else None,
        "matching_ms_per_query": float(matching_ms.mean())
        if matching_ms.size
        else None,
        "context_ms_per_query": float(context_ms.mean()),
        "ransac_ms_per_query": float(ransac_ms.mean())
        if ransac_ms.size
        else None,
        "total_ms_per_query": (
            float(matching_ms.mean() + context_ms.mean() + ransac_ms.mean())
            if matching_ms.size and ransac_ms.size
            else None
        ),
        "results": rows,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = merge(args.input)
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "results"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
