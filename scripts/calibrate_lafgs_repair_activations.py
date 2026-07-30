#!/usr/bin/env python3
"""Calibrate repair modes by leave-one-trajectory-out activation precision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from localization_training.repair_activation_calibration import (
    RepairActivationCalibrationConfig,
    calibrate_repair_route_activations,
)
from localization_training.shared_metric import SharedLowRankMetric


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routed-audit", required=True)
    parser.add_argument("--routed-teacher", required=True)
    parser.add_argument("--routed-family", required=True)
    parser.add_argument("--base-family", required=True)
    parser.add_argument("--positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--teacher-output", required=True)
    parser.add_argument("--family-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-global-precision", type=float, default=0.8)
    parser.add_argument(
        "--minimum-trajectory-precision", type=float, default=0.5
    )
    parser.add_argument(
        "--minimum-supported-trajectories", type=int, default=2
    )
    parser.add_argument(
        "--minimum-true-activations-per-trajectory",
        type=int,
        default=1,
    )
    parser.add_argument("--activation-margin", type=float, default=0.0)
    parser.add_argument("--descriptor-batch-size", type=int, default=8192)
    args = parser.parse_args()

    paths = {
        "audit": Path(args.routed_audit).resolve(),
        "teacher": Path(args.routed_teacher).resolve(),
        "family": Path(args.routed_family).resolve(),
        "base_family": Path(args.base_family).resolve(),
        "positive_teacher": Path(args.positive_teacher).resolve(),
        "query_cache": Path(args.query_cache).resolve(),
        "dynamic": Path(args.dynamic_outcomes).resolve(),
        "metric": Path(args.metric_state).resolve(),
        "map": Path(args.map).resolve(),
    }
    payload = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    device = torch.device(args.device)
    metric_payload = payload["metric"]
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    config = RepairActivationCalibrationConfig(
        minimum_global_precision=args.minimum_global_precision,
        minimum_trajectory_precision=args.minimum_trajectory_precision,
        minimum_supported_trajectories=args.minimum_supported_trajectories,
        minimum_true_activations_per_trajectory=(
            args.minimum_true_activations_per_trajectory
        ),
        activation_margin=args.activation_margin,
        descriptor_batch_size=args.descriptor_batch_size,
    )
    audit, teacher, family, report = calibrate_repair_route_activations(
        routed_audit=payload["audit"],
        routed_teacher=payload["teacher"],
        routed_family=payload["family"],
        base_family_count=len(
            torch.as_tensor(payload["base_family"]["prototype_features"])
        ),
        positive_teacher=payload["positive_teacher"],
        query_cache=payload["query_cache"],
        dynamic_outcomes=payload["dynamic"],
        metric=metric,
        anchor_count=len(torch.as_tensor(payload["map"]["anchor_ids"])),
        config=config,
        device=device,
    )
    _save(Path(args.audit_output).resolve(), audit)
    _save(Path(args.teacher_output).resolve(), teacher)
    _save(Path(args.family_output).resolve(), family)
    report_path = Path(args.report_output).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "modes"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
