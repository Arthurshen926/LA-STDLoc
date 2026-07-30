#!/usr/bin/env python3
"""Restrict context codes to repeatedly observed confusion families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-state", required=True)
    parser.add_argument("--confusion-graph", required=True)
    parser.add_argument("--minimum-occurrences", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    state = torch.load(
        args.context_state, map_location="cpu", weights_only=False
    )
    graph = torch.load(
        args.confusion_graph, map_location="cpu", weights_only=False
    )
    if int(graph["anchor_count"]) != len(state["anchor_context"]):
        raise ValueError("confusion graph and context state do not align")
    active = {
        int(edge[key])
        for edge in graph["edges"]
        if int(edge["occurrences"]) >= int(args.minimum_occurrences)
        for key in ("correct_anchor", "confusing_anchor")
    }
    mask = torch.zeros(len(state["anchor_context"]), dtype=torch.bool)
    if active:
        mask[torch.as_tensor(sorted(active)).long()] = True
    context = torch.as_tensor(state["anchor_context"]).clone()
    context[~mask] = 0
    output = dict(state)
    output["version"] = max(int(state.get("version", 1)), 6)
    output["anchor_context"] = context
    output["context_active_mask"] = mask
    output["config"] = {
        **state["config"],
        "context_scope": "frequent_confusion_families_only",
        "context_minimum_confusion_occurrences": int(
            args.minimum_occurrences
        ),
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    summary = {
        "output": str(path),
        "active_anchor_count": int(mask.sum()),
        "active_anchor_fraction": float(mask.float().mean()),
        "minimum_occurrences": int(args.minimum_occurrences),
    }
    path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(summary)


if __name__ == "__main__":
    main()
