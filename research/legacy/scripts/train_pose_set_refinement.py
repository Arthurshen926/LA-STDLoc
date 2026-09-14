#!/usr/bin/env python3
"""Run one exact-oracle descriptor M-step without changing deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from map_learning.pose_set_refinement import (
    build_expanded_pose_set_constraints,
    build_pose_set_constraints,
    materialize_pose_set_map,
    train_pose_set_residual,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--pose-set-oracle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-residual-norm", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--trust-weight", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=1.0)
    parser.add_argument("--clean-minimum-margin", type=float, default=0.05)
    parser.add_argument("--clean-margin-slack", type=float, default=0.01)
    parser.add_argument("--maximum-clean-constraints", type=int, default=8192)
    parser.add_argument(
        "--expand-oracle-targets",
        action="store_true",
        help=(
            "Use exact pose actions only to seed identities, then train from "
            "their complete mapping-only positive support."
        ),
    )
    parser.add_argument("--minimum-target-views", type=int, default=3)
    parser.add_argument("--maximum-constraints-per-target", type=int, default=32)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    parser.add_argument("--holdout-remainder", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.complete_positive_teacher, map_location="cpu", weights_only=False
    )
    cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    oracle = json.loads(args.pose_set_oracle.read_text())
    builder = (
        build_expanded_pose_set_constraints
        if args.expand_oracle_targets
        else build_pose_set_constraints
    )
    builder_options = {}
    if args.expand_oracle_targets:
        builder_options = {
            "minimum_target_views": int(args.minimum_target_views),
            "maximum_constraints_per_target": int(
                args.maximum_constraints_per_target
            ),
        }
    constraints, clean_constraints, trainable, constraint_report = builder(
        state=state,
        metric_state_path=args.metric_state,
        teacher=teacher,
        query_cache=cache,
        oracle=oracle,
        device=device,
        clean_minimum_margin=float(args.clean_minimum_margin),
        clean_margin_slack=float(args.clean_margin_slack),
        maximum_clean_constraints=int(args.maximum_clean_constraints),
        **builder_options,
    )
    residual, training_report = train_pose_set_residual(
        state=state,
        constraints=constraints,
        clean_constraints=clean_constraints,
        trainable_anchors=trainable,
        maximum_norm=float(args.maximum_residual_norm),
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        margin=float(args.margin),
        temperature=float(args.temperature),
        trust_weight=float(args.trust_weight),
        clean_weight=float(args.clean_weight),
        holdout_modulus=int(args.holdout_modulus),
        holdout_remainder=int(args.holdout_remainder),
        device=device,
    )
    report = {
        "schema": "lafgs_pose_set_descriptor_refinement",
        "version": 1,
        "changes_default_mainline": False,
        "uses_test_queries": False,
        "source_map": str(args.map.resolve()),
        "metric_state": str(args.metric_state.resolve()),
        "pose_set_oracle": str(args.pose_set_oracle.resolve()),
        "maximum_residual_norm": float(args.maximum_residual_norm),
        "requested_steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "margin": float(args.margin),
        "temperature": float(args.temperature),
        "trust_weight": float(args.trust_weight),
        "clean_weight": float(args.clean_weight),
        "clean_minimum_margin": float(args.clean_minimum_margin),
        "clean_margin_slack": float(args.clean_margin_slack),
        "expand_oracle_targets": bool(args.expand_oracle_targets),
        **constraint_report,
        **training_report,
    }
    output = materialize_pose_set_map(
        state=state,
        trainable_anchors=trainable,
        residual=residual,
        report=report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
