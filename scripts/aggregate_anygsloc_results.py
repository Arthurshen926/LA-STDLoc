#!/usr/bin/env python3
"""Hash-bind and aggregate completed AnyGSLoc Base experiment cells."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

from common.hashing import sha256_file
from scripts.run_anygsloc_matrix import complete_cell


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def resolve_root(cell: dict[str, Any], output_root: Path) -> Path:
    existing = cell.get("existing_base")
    if existing and (Path(existing) / "projective_map/report.json").is_file():
        return Path(existing)
    return output_root / cell["family"] / cell["scene"]


def load_cell(cell: dict[str, Any], output_root: Path, seed: int) -> tuple[dict, list, list]:
    root = resolve_root(cell, output_root)
    report_path = root / "projective_map/report.json"
    summary_path = root / "evaluation" / f"base_seed{seed}" / "summary.json"
    results_path = root / "evaluation" / f"base_seed{seed}" / "results.json"
    report = json.loads(report_path.read_text())
    summary = json.loads(summary_path.read_text())
    results = json.loads(results_path.read_text())
    if (
        report.get("uses_source_mapping_rgb") is not False
        or report.get("uses_test_queries") is not False
        or report.get("contracts", {}).get("feedback_used") is not False
        or summary.get("evaluated_split") != "test"
        or summary.get("input_map_sha256") != report["output"]["map_sha256"]
        or len(results) != int(summary["query_count"])
    ):
        raise ValueError(f"cell violates AnyGSLoc Base scope: {cell['key']}")
    row = {
        "key": cell["key"],
        "artifact_root": str(root.resolve()),
        "anchor_count": int(report["counts"]["total_anchors"]),
        "query_count": int(summary["query_count"]),
        "median_te_cm": float(summary["median_te_cm"]),
        "median_re_deg": float(summary["median_ae_deg"]),
        "p90_te_cm": float(summary["p90_te_cm"]),
        "mean_te_cm": float(summary["mean_te_cm"]),
        "r5_percent": float(summary["recall_5cm_5deg_percent"]),
        "catastrophic_100cm_count": int(summary["catastrophic_100cm_count"]),
        "mean_latency_ms": float(summary["total_ms_mean"]),
        "p50_latency_ms": float(summary["total_ms_p50"]),
        "p90_latency_ms": float(summary["total_ms_p90"]),
        "map_build_seconds": float(report["timing_seconds"]["total"]),
        "map_bytes": (root / "projective_map/projective_anchor_map.pt").stat().st_size,
        "sha256": {
            "map_report": sha256_file(report_path),
            "map": report["output"]["map_sha256"],
            "metric": report["output"]["metric_sha256"],
            "summary": sha256_file(summary_path),
            "results": sha256_file(results_path),
        },
    }
    te = [float(item["translation_error_cm"]) for item in results]
    re = [float(item["rotation_error_deg"]) for item in results]
    return row, te, re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=Path("configs/anygsloc_experiments.json"))
    parser.add_argument("--group", choices=("primary_24_scene", "prior_robustness"), default="primary_24_scene")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    cells = [complete_cell(cell) for cell in matrix[args.group]]
    output_root = Path(matrix["output_root"])
    rows, all_te, all_re = [], [], []
    for cell in cells:
        row, te, re = load_cell(cell, output_root, args.seed)
        rows.append(row)
        all_te.extend(te)
        all_re.extend(re)
    aggregate = {
        "schema": "anygsloc_base_experiment_aggregate",
        "version": 1,
        "group": args.group,
        "seed": args.seed,
        "scientific_scope": {
            "offline_self_localization_feedback": False,
            "descriptor_or_metric_training": False,
            "test_query_map_adaptation": False,
            "online_refinement": False,
        },
        "scene_count": len(rows),
        "query_count": len(all_te),
        "pooled": {
            "median_te_cm": statistics.median(all_te),
            "median_re_deg": statistics.median(all_re),
            "p90_te_cm": percentile(all_te, 0.9),
            "mean_te_cm": statistics.fmean(all_te),
            "r5_percent": 100.0 * sum(t <= 5.0 and r <= 5.0 for t, r in zip(all_te, all_re)) / len(all_te),
            "catastrophic_100cm_count": sum(t > 100.0 for t in all_te),
        },
        "scenes": rows,
    }
    atomic_json(args.output.resolve(), aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
