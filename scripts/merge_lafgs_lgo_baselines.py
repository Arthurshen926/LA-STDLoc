#!/usr/bin/env python3
"""Merge budget-disjoint LGO baseline files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in args.inputs
    ]
    first = payloads[0]
    states = {}
    for payload in payloads:
        for key in ("anchor_map", "function_graph", "query_cache"):
            if payload[key] != first[key]:
                raise ValueError(f"{key} mismatch")
        overlap = set(states) & set(payload["states"])
        if overlap:
            raise ValueError(f"duplicate baseline states: {overlap}")
        states.update(payload["states"])
    if set(states) != {"30000", "35000", "40000", "45000"}:
        raise ValueError("merged baselines must contain all nested states")
    output = {
        "schema": "lafgs_lgo_baselines",
        "version": 1,
        "anchor_map": first["anchor_map"],
        "function_graph": first["function_graph"],
        "query_cache": first["query_cache"],
        "states": states,
        "shard_configs": [payload["config"] for payload in payloads],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                budget: state["metrics"]
                for budget, state in sorted(states.items())
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
