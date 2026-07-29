#!/usr/bin/env python3
"""Train validated secondary prototypes while freezing geometry and metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from localization_training.prototype_optimization import (
    PrototypeOptimizationConfig,
    optimize_basin_prototypes,
)
from localization_training.shared_metric import SharedLowRankMetric


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--family-state", required=True)
    parser.add_argument("--basin-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--checkpoint-steps", default="50,100,200,300")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--maximum-residual", type=float, default=0.05)
    parser.add_argument("--maximum-negative-bias", type=float, default=0.12)
    parser.add_argument("--minimum-temperature", type=float, default=0.85)
    parser.add_argument("--maximum-temperature", type=float, default=1.15)
    parser.add_argument("--train-temperature", action="store_true")
    parser.add_argument("--hyperedge-weight", type=float, default=1.0)
    parser.add_argument("--sibling-weight", type=float, default=2.0)
    parser.add_argument("--trust-weight", type=float, default=0.2)
    parser.add_argument("--bias-trust-weight", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument(
        "--translation-reward-scale-cm", type=float, default=15.0
    )
    parser.add_argument(
        "--rotation-reward-scale-deg", type=float, default=2.0
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    metric_payload = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    family = torch.load(
        args.family_state, map_location="cpu", weights_only=False
    )
    teacher = torch.load(
        args.basin_teacher, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    metric = SharedLowRankMetric(**metric_payload["metric_config"])
    metric.load_state_dict(metric_payload["metric_state_dict"])
    config = PrototypeOptimizationConfig(
        **{
            key: getattr(args, key)
            for key in PrototypeOptimizationConfig.__dataclass_fields__
        }
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {
        int(value)
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }

    def save_checkpoint(step: int, output: dict) -> None:
        torch.save(
            output,
            output_dir / f"family_prototypes_step_{step:04d}.pt",
        )

    def progress(event: dict) -> None:
        print(json.dumps(event), flush=True)

    output, history = optimize_basin_prototypes(
        state=state,
        metric=metric,
        family=family,
        teacher=teacher,
        cache=cache_payload.get("queries", cache_payload),
        config=config,
        device=torch.device(args.device),
        checkpoint_steps=checkpoints,
        checkpoint_callback=save_checkpoint,
        progress=progress,
    )
    trainable = output["prototype_only_training"][
        "trainable_prototype_indices"
    ]
    teacher_query_count = output["prototype_only_training"][
        "teacher_query_count"
    ]
    (output_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "schema": "lafgs_basin_prototype_only_training",
                "trainable_prototype_count": int(trainable.numel()),
                "teacher_query_count": int(teacher_query_count),
                "history": history,
                "config": vars(args),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
