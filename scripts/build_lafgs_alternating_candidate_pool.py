#!/usr/bin/env python3
"""Rebase inactive structure candidates behind the current active map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


ROW_FIELDS = (
    "source_primitive_ids",
    "track_cluster_ids",
    "anchor_xyz",
    "anchor_features",
    "anchor_type",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-map", required=True)
    parser.add_argument("--source-candidate-pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    current_path = Path(args.current_map).resolve()
    source_path = Path(args.source_candidate_pool).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current = torch.load(
        current_path, map_location="cpu", weights_only=False
    )
    source = torch.load(
        source_path, map_location="cpu", weights_only=False
    )
    if current.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("current map is not materialized")
    if source.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("source candidate pool is not materialized")
    update = current.get("alternating_structure_update")
    if not isinstance(update, dict):
        raise ValueError("current map lacks an alternating structure report")
    source_base_count = int(update["base_anchor_count"])
    accepted_rows = {
        int(value)
        for value in update.get("accepted_candidate_pool_rows", ())
    }
    available = range(
        source_base_count, int(source["anchor_ids"].numel())
    )
    inactive_rows = torch.as_tensor(
        [row for row in available if row not in accepted_rows],
        dtype=torch.long,
    )
    output = dict(current)
    for key in ROW_FIELDS:
        output[key] = torch.cat(
            (
                torch.as_tensor(current[key]),
                torch.as_tensor(source[key])[inactive_rows],
            )
        )
    output["anchor_ids"] = torch.arange(
        output["anchor_xyz"].shape[0], dtype=torch.long
    )
    output["alternating_candidate_pool"] = {
        "round": int(args.round),
        "current_map_path": str(current_path),
        "source_candidate_pool_path": str(source_path),
        "source_base_count": source_base_count,
        "accepted_source_rows": sorted(accepted_rows),
        "inactive_source_rows": inactive_rows.tolist(),
        "inactive_candidate_count": int(inactive_rows.numel()),
    }
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "current_anchor_count": int(current["anchor_ids"].numel()),
                "inactive_candidate_count": int(inactive_rows.numel()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
