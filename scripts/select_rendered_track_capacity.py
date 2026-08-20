#!/usr/bin/env python3
"""Select a small train-only harmful-attractor prune set for rendered Tracks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from topology.deployment_revision import (
    select_revision,
    subset_map_and_metric,
    subset_teacher,
)


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args) -> dict:
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
    statistics = torch.load(args.statistics, map_location="cpu", weights_only=False)
    metric = torch.load(args.metric_state, map_location="cpu", weights_only=False)
    if statistics.get("uses_source_mapping_rgb") is not False:
        raise ValueError("selection statistics do not attest rendered-RGB-only")
    if statistics.get("uses_test_queries") is not False:
        raise ValueError("selection statistics contain test queries")
    if int(teacher["anchor_count"]) != int(state["anchor_ids"].numel()):
        raise ValueError("teacher and rendered Track map differ")
    if not bool((torch.as_tensor(state["anchor_type"]).long() == 1).all()):
        raise ValueError("rendered Track capacity selector only accepts Track maps")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    pruned, selection = select_revision(
        teacher,
        statistics,
        matching_rows_target=args.matching_rows_target,
        revisable_mask=torch.ones(int(teacher["anchor_count"]), dtype=torch.bool),
        maximum_prune_fraction=args.maximum_prune_fraction,
        minimum_counterfactual_gain=args.minimum_counterfactual_gain,
        maximum_tail_nonimproving_wins=args.maximum_tail_nonimproving_wins,
    )
    keep = torch.ones(int(teacher["anchor_count"]), dtype=torch.bool)
    keep[pruned] = False
    output_map = args.output_dir / "selected_anchor_map.pt"
    output_metric = args.output_dir / "selected_metric_state.pt"
    output_teacher = args.output_dir / "selected_positive_teacher.pt"
    revised_map, revised_metric = subset_map_and_metric(
        state, metric, keep, output_map=output_map
    )
    revised_teacher = subset_teacher(teacher, keep, output_map)
    revised_map["provenance"] = {
        **revised_map.get("provenance", {}),
        "rendered_track_train_only_capacity_selection": {
            "source_anchor_count": int(keep.numel()),
            "retained_anchor_count": int(keep.sum()),
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        },
    }
    _atomic_save(revised_map, output_map)
    _atomic_save(revised_metric, output_metric)
    _atomic_save(revised_teacher, output_teacher)
    counters = statistics["counters"]
    report = {
        "schema": "lafgs_rendered_track_train_only_capacity_selection",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "source_anchor_count": int(keep.numel()),
        "retained_anchor_count": int(keep.sum()),
        "pruned_anchor_count": int(pruned.numel()),
        "pruned_anchor_rows": pruned.tolist(),
        "pruned_statistics": {
            name: float(torch.as_tensor(values)[pruned].sum())
            for name, values in counters.items()
        },
        "selection": selection,
        "config": {
            "matching_rows_target": args.matching_rows_target,
            "maximum_prune_fraction": args.maximum_prune_fraction,
            "minimum_counterfactual_gain": args.minimum_counterfactual_gain,
            "maximum_tail_nonimproving_wins": args.maximum_tail_nonimproving_wins,
        },
        "inputs": {
            "anchor_map": str(args.anchor_map.resolve()),
            "teacher": str(args.teacher.resolve()),
            "statistics": str(args.statistics.resolve()),
            "metric_state": str(args.metric_state.resolve()),
        },
        "input_sha256": {
            "anchor_map": sha256_file(args.anchor_map),
            "teacher": sha256_file(args.teacher),
            "statistics": sha256_file(args.statistics),
            "metric_state": sha256_file(args.metric_state),
        },
        "outputs": {
            "anchor_map": str(output_map.resolve()),
            "teacher": str(output_teacher.resolve()),
            "metric_state": str(output_metric.resolve()),
        },
        "output_sha256": {
            "anchor_map": sha256_file(output_map),
            "teacher": sha256_file(output_teacher),
            "metric_state": sha256_file(output_metric),
        },
    }
    (args.output_dir / "capacity_selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matching-rows-target", type=int, default=32)
    parser.add_argument("--maximum-prune-fraction", type=float, default=0.02)
    parser.add_argument("--minimum-counterfactual-gain", type=float, default=4.0)
    parser.add_argument("--maximum-tail-nonimproving-wins", type=int, default=2)
    args = parser.parse_args()
    for field in ("anchor_map", "teacher", "statistics", "metric_state"):
        setattr(args, field, getattr(args, field).resolve())
    args.output_dir = args.output_dir.resolve()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
