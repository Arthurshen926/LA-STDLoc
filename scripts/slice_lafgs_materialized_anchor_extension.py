#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.micro_anchors import (
    truncate_materialized_anchor_extension,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--prefix", default="full_prior_micro_anchor")
    args = parser.parse_args()

    source_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted(
        {int(value) for value in args.budgets.split(",") if value.strip()}
    )
    state = torch.load(source_path, map_location="cpu", weights_only=False)
    summary = {}
    for budget in budgets:
        output = truncate_materialized_anchor_extension(state, budget)
        config = dict(output.get("config", {}))
        config.update(
            {
                "extension_slice_source": str(source_path),
                "requested_new_anchor_budget": int(budget),
            }
        )
        output["config"] = config
        output_path = output_dir / f"{args.prefix}_{budget:04d}.pt"
        torch.save(output, output_path)
        summary[str(budget)] = {
            "path": str(output_path),
            "canonical_anchor_count": int(output["canonical_anchor_count"]),
            "selected_extension_count": int(
                output["selected_extension_count"]
            ),
            "total_anchor_count": int(output["anchor_ids"].numel()),
        }
    summary_path = output_dir / f"{args.prefix}_slice_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
