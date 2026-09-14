#!/usr/bin/env python3
"""Aggregate paired real/render map-cleaning ablations without selecting a map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file


def _paired(reference: dict, candidate: dict) -> dict:
    left = {int(row["query_index"]): row for row in reference["rows"]}
    right = {int(row["query_index"]): row for row in candidate["rows"]}
    if set(left) != set(right):
        raise ValueError("paired ablation query registries differ")
    indices = sorted(left)
    baseline_t = torch.tensor([left[index]["translation_error_cm"] for index in indices])
    proposal_t = torch.tensor([right[index]["translation_error_cm"] for index in indices])
    baseline_r = torch.tensor([left[index]["rotation_error_deg"] for index in indices])
    proposal_r = torch.tensor([right[index]["rotation_error_deg"] for index in indices])
    baseline_success = (baseline_t < 5.0) & (baseline_r < 5.0)
    proposal_success = (proposal_t < 5.0) & (proposal_r < 5.0)
    delta = proposal_t - baseline_t
    worst = torch.argsort(delta, descending=True)[:5]
    best = torch.argsort(delta)[:5]

    def examples(rows: torch.Tensor) -> list[dict]:
        return [
            {
                "query_index": indices[int(row)],
                "baseline_translation_cm": float(baseline_t[row]),
                "proposal_translation_cm": float(proposal_t[row]),
                "delta_translation_cm": float(delta[row]),
            }
            for row in rows
        ]

    return {
        "query_count": len(indices),
        "translation_improved_count": int((delta < 0).sum()),
        "translation_worsened_count": int((delta > 0).sum()),
        "baseline_success_proposal_failure_count": int(
            (baseline_success & (~proposal_success)).sum()
        ),
        "baseline_failure_proposal_success_count": int(
            ((~baseline_success) & proposal_success).sum()
        ),
        "baseline_catastrophic_proposal_noncatastrophic_count": int(
            ((baseline_t >= 50.0) & (proposal_t < 50.0)).sum()
        ),
        "baseline_noncatastrophic_proposal_catastrophic_count": int(
            ((baseline_t < 50.0) & (proposal_t >= 50.0)).sum()
        ),
        "median_paired_translation_delta_cm": float(delta.median()),
        "mean_paired_translation_delta_cm": float(delta.mean()),
        "largest_regressions": examples(worst),
        "largest_improvements": examples(best),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, nargs=4, required=True)
    parser.add_argument("--confirmation", type=Path, nargs=4, required=True)
    parser.add_argument(
        "--names",
        nargs=4,
        default=("baseline", "strict_retire", "bounded_reaggregate", "combined"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    real = {name: json.loads(path.read_text()) for name, path in zip(args.names, args.real)}
    confirmation = {
        name: json.loads(path.read_text())
        for name, path in zip(args.names, args.confirmation)
    }
    report = {
        "schema": "lafgs_v7_anchor_contamination_paired_ablation",
        "version": 1,
        "status": "PASS",
        "uses_test_queries": False,
        "threshold_tuning_from_results": False,
        "formal_method_selected": False,
        "real_mapping_rgb_evaluation_only": {
            name: {
                "metrics": real[name]["metrics"],
                **(
                    {}
                    if name == "baseline"
                    else {"paired_vs_baseline": _paired(real["baseline"], real[name])}
                ),
            }
            for name in args.names
        },
        "independent_render_confirmation": {
            name: {
                "metrics": confirmation[name]["metrics"],
                **(
                    {}
                    if name == "baseline"
                    else {
                        "paired_vs_baseline": _paired(
                            confirmation["baseline"], confirmation[name]
                        )
                    }
                ),
            }
            for name in args.names
        },
        "inputs": {
            "real": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.real
            ],
            "confirmation": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in args.confirmation
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
