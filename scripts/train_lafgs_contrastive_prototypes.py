#!/usr/bin/env python3
"""Optimize only involved family prototypes with V19 contrastive evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.prototype_optimization import (
    ContrastivePrototypeOptimizationConfig,
    optimize_contrastive_prototypes,
)
from localization_training.shared_metric import SharedLowRankMetric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--family-state", required=True)
    parser.add_argument("--contrastive-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--maximum-residual", type=float, default=0.025)
    parser.add_argument("--maximum-negative-bias", type=float, default=0.12)
    parser.add_argument("--minimum-temperature", type=float, default=0.9)
    parser.add_argument("--maximum-temperature", type=float, default=1.1)
    parser.add_argument("--train-temperature", action="store_true")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--trust-weight", type=float, default=1.0)
    parser.add_argument("--bias-trust-weight", type=float, default=0.5)
    parser.add_argument("--maximum-pairs-per-step", type=int, default=256)
    parser.add_argument("--smoothmax-temperature", type=float, default=0.02)
    parser.add_argument(
        "--all-appearance-modes",
        action="store_true",
        help="Diagnostic only: include real-only appearance-pool modes.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    family = torch.load(
        args.family_state, map_location="cpu", weights_only=False
    )
    evidence = torch.load(
        args.contrastive_evidence, map_location="cpu", weights_only=False
    )
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"])
    metric.load_state_dict(metric_payload["metric_state_dict"])
    config = ContrastivePrototypeOptimizationConfig(
        steps=args.steps,
        learning_rate=args.learning_rate,
        maximum_residual=args.maximum_residual,
        maximum_negative_bias=args.maximum_negative_bias,
        minimum_temperature=args.minimum_temperature,
        maximum_temperature=args.maximum_temperature,
        train_temperature=args.train_temperature,
        margin=args.margin,
        trust_weight=args.trust_weight,
        bias_trust_weight=args.bias_trust_weight,
        maximum_pairs_per_step=args.maximum_pairs_per_step,
        smoothmax_temperature=args.smoothmax_temperature,
        synthetic_modes_only=not args.all_appearance_modes,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress(event: dict) -> None:
        print(json.dumps(event), flush=True)

    def checkpoint(step: int, output: dict) -> None:
        torch.save(
            output,
            output_dir / f"contrastive_family_step_{step:04d}.pt",
        )

    output, history = optimize_contrastive_prototypes(
        state=state,
        metric=metric,
        family=family,
        evidence=evidence,
        config=config,
        device=torch.device(args.device),
        checkpoint_steps={args.steps},
        checkpoint_callback=checkpoint,
        progress=progress,
    )
    summary = {
        "schema": "lafgs_confusion_contrastive_training_summary",
        "version": 1,
        "map": str(Path(args.map).resolve()),
        "metric_state": str(Path(args.metric_state).resolve()),
        "family_state": str(Path(args.family_state).resolve()),
        "contrastive_evidence": str(
            Path(args.contrastive_evidence).resolve()
        ),
        "output_family": str(
            output_dir / f"contrastive_family_step_{args.steps:04d}.pt"
        ),
        "history": history,
        "config": vars(args),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(summary["output_family"])


if __name__ == "__main__":
    main()
