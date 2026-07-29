#!/usr/bin/env python3
"""Calibrate high-recall appearance modes against deployed hard assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.appearance_calibration import (
    calibrate_appearance_modes,
    parse_nonpositive_biases,
    validate_dynamic_baseline_binding,
)
from localization_training.shared_metric import SharedLowRankMetric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode-pool", required=True)
    parser.add_argument("--base-family", default="")
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--biases", default="0,-0.01,-0.02,-0.03,-0.05,-0.08"
    )
    parser.add_argument("--minimum-legal-activations", type=int, default=3)
    parser.add_argument(
        "--minimum-activation-precision", type=float, default=0.8
    )
    parser.add_argument("--false-activation-cost", type=float, default=2.0)
    parser.add_argument("--maximum-selected", type=int, default=2048)
    parser.add_argument(
        "--allow-unbound-dynamic-base",
        action="store_true",
        help="Allow a legacy dynamic artifact without family provenance.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    pool = torch.load(
        args.mode_pool, map_location="cpu", weights_only=False
    )
    positives = torch.load(
        args.complete_positive_teacher,
        map_location="cpu",
        weights_only=False,
    )
    dynamic = torch.load(
        args.dynamic_outcomes, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    base = (
        torch.load(
            args.base_family, map_location="cpu", weights_only=False
        )
        if args.base_family
        else None
    )
    if base is not None:
        validate_dynamic_baseline_binding(
            dynamic,
            base_family_path=args.base_family,
            allow_unbound=args.allow_unbound_dynamic_base,
        )

    def progress(completed: int, query_count: int) -> None:
        if completed % 25 == 0:
            print(
                json.dumps(
                    {"completed": completed, "query_count": query_count}
                ),
                flush=True,
            )

    output = calibrate_appearance_modes(
        pool=pool,
        positives=positives,
        dynamic=dynamic,
        cache=cache_payload.get("queries", cache_payload),
        metric=metric,
        biases=parse_nonpositive_biases(args.biases),
        minimum_legal_activations=args.minimum_legal_activations,
        minimum_activation_precision=args.minimum_activation_precision,
        false_activation_cost=args.false_activation_cost,
        maximum_selected=args.maximum_selected,
        device=device,
        base_family=base,
        config=vars(args),
        progress=progress,
    )
    summary = {"schema": output["schema"], **output.pop("summary")}
    summary["config"] = vars(args)
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(path)


if __name__ == "__main__":
    main()
