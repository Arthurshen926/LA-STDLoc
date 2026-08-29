#!/usr/bin/env python3
"""Seal disjoint design/control Observer batches by pose family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from common.hashing import sha256_file


def split_batches(payloads: list[dict], *, design_fraction: float, seed: int) -> tuple[dict, dict]:
    if not 0.0 < design_fraction < 1.0:
        raise ValueError("design fraction must be in (0, 1)")
    identity = payloads[0]["input"]
    if any(
        item.get("schema") != "lafgs_v9_no_loo_causal_feedback_batch"
        or item.get("uses_test_queries") is not False
        or item.get("loo_used") is not False
        or item.get("accepted_query_row_policy") != "v2_row_valid_only"
        or item.get("input") != identity
        for item in payloads
    ):
        raise ValueError("Observer inputs do not share the corrected no-LOO contract")
    records = [row for item in payloads for row in item["records"]]
    if len({int(row["query_index"]) for row in records}) != len(records):
        raise ValueError("Observer batches overlap")

    # Read only the tiny Observer records required to bind each query to a family.
    enriched = []
    import torch

    for row in records:
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError("Observer record SHA256 differs")
        record = torch.load(path, map_location="cpu", weights_only=False)
        enriched.append((int(record["pose_family_id"]), row))
    families = sorted(
        {family for family, _ in enriched},
        key=lambda family: hashlib.sha256(f"{seed}:{family}".encode()).digest(),
    )
    design_count = max(1, min(len(families) - 1, math.floor(len(families) * design_fraction)))
    design_families = set(families[:design_count])

    def materialize(role: str, selected_families: set[int]) -> dict:
        selected = [row for family, row in enriched if family in selected_families]
        selected.sort(key=lambda row: int(row["query_index"]))
        categories: dict[str, int] = {}
        authorized_queries = authorized_rows = 0
        for row in selected:
            category = str(row["category"])
            categories[category] = categories.get(category, 0) + 1
            if row["can_train_metric"]:
                authorized_queries += 1
                record = torch.load(row["path"], map_location="cpu", weights_only=False)
                authorized_rows += int(record["training_evidence"]["query_rows"].numel())
        return {
            "schema": "lafgs_v9_no_loo_causal_feedback_batch",
            "version": 1,
            "status": "PASS",
            "role": role,
            "loo_used": False,
            "uses_test_queries": False,
            "map_mutation_count": 0,
            "accepted_query_row_policy": "v2_row_valid_only",
            "split_policy": "sha256_ranked_pose_family",
            "split_seed": int(seed),
            "design_fraction": float(design_fraction),
            "pose_family_count": len(selected_families),
            "query_count": len(selected),
            "category_counts": categories,
            "authorized_metric_query_count": authorized_queries,
            "authorized_metric_training_row_count": authorized_rows,
            "input": identity,
            "records": selected,
        }

    design = materialize("controller_design", design_families)
    control = materialize("heldout_control", set(families) - design_families)
    return design, control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--design-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=1420260828)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inputs = [path.resolve() for path in args.observer_batches]
    payloads = [json.loads(path.read_text()) for path in inputs]
    design, control = split_batches(
        payloads, design_fraction=args.design_fraction, seed=args.seed
    )
    lineage = [
        {"path": str(path), "sha256": sha256_file(path)} for path in inputs
    ]
    for payload in (design, control):
        payload["source_observer_batches"] = lineage
    args.output_dir.mkdir(parents=True)
    for name, payload in (("design", design), ("control", control)):
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps({"design": design["query_count"], "control": control["query_count"]}, indent=2))


if __name__ == "__main__":
    main()
