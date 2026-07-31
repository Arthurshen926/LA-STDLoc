#!/usr/bin/env python3
"""Materialize a fixed-budget SLPS ablation without changing its ordering."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args()
    budget = max(int(args.budget), 4)
    source = Path(args.selector).resolve()
    output = Path(args.output).resolve()
    state = torch.load(source, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_slps_selector":
        raise ValueError("fixed-budget state requires an SLPS selector")
    state = dict(state)
    state["selector_config"] = {
        **dict(state["selector_config"]),
        "budgets": [budget],
        "safe_probability_threshold": -0.01,
        "catastrophic_probability_threshold": 1.01,
        "minimum_probability_margin": 0.0,
        "relative_utility_lcb_threshold": float("-inf"),
    }
    state["deployment_policy"] = {
        "type": "fixed_learned_order_budget",
        "budget": budget,
        "source_selector": str(source),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, output)
    summary = {
        "output": str(output),
        "source_selector": str(source),
        "budget": budget,
        "selector_config": state["selector_config"],
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
