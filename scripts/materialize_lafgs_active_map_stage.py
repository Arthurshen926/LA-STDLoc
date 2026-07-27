#!/usr/bin/env python3
"""Materialize an accepted intermediate stage from an Active Map V2 report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.run_lafgs_active_map_v2 import (
    _initial_active_rows,
    _load,
    _materialize,
    _overlay_initial_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-map", required=True)
    parser.add_argument("--universe-map", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--accepted-operation-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    initial = _load(Path(args.initial_map))
    universe = _load(Path(args.universe_map))
    active = _initial_active_rows(initial, universe)
    features = _overlay_initial_features(initial, universe, active)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    accepted = 0
    replayed = []
    for operation in report["operations"]:
        if not operation["accepted"]:
            continue
        if accepted >= args.accepted_operation_count:
            break
        active[torch.as_tensor(operation["add_rows"], dtype=torch.long)] = True
        active[torch.as_tensor(operation["retire_rows"], dtype=torch.long)] = False
        replayed.append(operation)
        accepted += 1
    if accepted != args.accepted_operation_count:
        raise ValueError(
            f"Requested {args.accepted_operation_count} accepted operations, "
            f"but only replayed {accepted}"
        )
    stage_report = {
        "schema": "lafgs_dynamic_active_map_stage",
        "version": 1,
        "source_report": str(Path(args.report).resolve()),
        "accepted_operation_count": accepted,
        "anchor_count": int(active.sum()),
        "operations": replayed,
    }
    state = _materialize(universe, active, stage_report)
    state["anchor_features"] = features[active]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    print(json.dumps(stage_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
