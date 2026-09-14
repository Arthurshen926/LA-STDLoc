#!/usr/bin/env python3
"""Aggregate post-hoc false-winner/Anchor-contamination enrichment evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.v7_anchor_contamination import enrichment_table


def _bucket_report(rows: torch.Tensor, evidence: dict) -> dict:
    anchor_rows = torch.as_tensor(rows).long()
    valid_fraction = evidence["valid_observation_fraction"][anchor_rows]
    unique = torch.unique(anchor_rows)
    return {
        "occurrences": enrichment_table(
            anchor_rows=anchor_rows,
            anchor_positive=evidence["pure_contamination"],
        ),
        "unique_anchors": enrichment_table(
            anchor_rows=unique,
            anchor_positive=evidence["pure_contamination"],
        ),
        "unique_anchor_count": int(unique.numel()),
        "valid_observation_fraction_mean": (
            float(valid_fraction.mean()) if valid_fraction.numel() else math.nan
        ),
        "valid_observation_fraction_median": (
            float(valid_fraction.median()) if valid_fraction.numel() else math.nan
        ),
        "zero_valid_observation_fraction": (
            float((valid_fraction == 0).float().mean())
            if valid_fraction.numel()
            else math.nan
        ),
        "below_25pct_valid_observation_fraction": (
            float((valid_fraction < 0.25).float().mean())
            if valid_fraction.numel()
            else math.nan
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--contamination-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence = torch.load(
        args.contamination_evidence, map_location="cpu", weights_only=False
    )
    buckets: dict[str, list[torch.Tensor]] = {}
    rows = []
    inputs = []
    for path in args.shards:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != "lafgs_v7_false_winner_contamination_audit_shard":
            raise ValueError("unexpected false-winner audit shard")
        if payload["input"]["contamination_evidence_sha256"] != sha256_file(
            args.contamination_evidence
        ):
            raise ValueError("false-winner shard evidence lineage differs")
        inputs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
        rows.extend(payload["rows"])
        for key, value in payload["anchor_row_buckets"].items():
            buckets.setdefault(key, []).append(value)
    rows.sort(key=lambda item: int(item["query_index"]))
    packed = {key: torch.cat(parts) for key, parts in buckets.items()}
    report_buckets = {key: _bucket_report(value, evidence) for key, value in packed.items()}
    wrong_pure = report_buckets["wrong"]["occurrences"]["event_positive_fraction"]
    correct_pure = report_buckets["correct"]["occurrences"]["event_positive_fraction"]
    report = {
        "schema": "lafgs_v7_false_winner_contamination_audit",
        "version": 1,
        "status": "PASS",
        "posthoc_test_rgb_diagnostic": True,
        "may_update_or_select_map": False,
        "threshold_tuning_from_results": False,
        "map_mutation_count": 0,
        "query_count": len(rows),
        "failed_query_count": sum(int(row["pose_failed"]) for row in rows),
        "buckets": report_buckets,
        "wrong_vs_correct_pure_contamination_ratio": (
            wrong_pure / correct_pure if correct_pure > 0 else math.nan
        ),
        "target_queries": [
            {
                key: value
                for key, value in row.items()
                if key != "target_detail"
            }
            | (
                {
                    "wrong_pure_contamination_count": int(
                        row["target_detail"]["wrong_anchor_pure_contamination"].sum()
                    ),
                    "wrong_zero_valid_fraction": float(
                        (
                            row["target_detail"]["wrong_anchor_valid_fraction"] == 0
                        ).float().mean()
                    ),
                    "wrong_below_25pct_valid_fraction": float(
                        (
                            row["target_detail"]["wrong_anchor_valid_fraction"] < 0.25
                        ).float().mean()
                    ),
                    "wrong_valid_fraction_median": float(
                        row["target_detail"]["wrong_anchor_valid_fraction"].median()
                    ),
                }
                if "target_detail" in row
                else {}
            )
            for row in rows
            if "target_detail" in row
        ],
        "input": {
            "contamination_evidence": str(args.contamination_evidence.resolve()),
            "contamination_evidence_sha256": sha256_file(
                args.contamination_evidence
            ),
            "shards": inputs,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
