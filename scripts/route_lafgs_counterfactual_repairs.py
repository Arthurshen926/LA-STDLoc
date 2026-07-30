#!/usr/bin/env python3
"""Route counterfactual repairs by repeated event support and add family modes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from localization_training.counterfactual_repair_routing import (
    RepairRoutingConfig,
    route_counterfactual_repairs,
)
from localization_training.shared_metric import SharedLowRankMetric


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--positive-teacher", required=True)
    parser.add_argument("--family-prototype-state", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--teacher-output", required=True)
    parser.add_argument("--family-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--minimum-primary-repair-trajectories", type=int, default=2
    )
    parser.add_argument(
        "--minimum-family-repair-observations", type=int, default=2
    )
    parser.add_argument(
        "--minimum-family-repair-trajectories", type=int, default=2
    )
    parser.add_argument(
        "--minimum-descriptor-cosine", type=float, default=0.6
    )
    parser.add_argument("--family-prototype-bias", type=float, default=-0.05)
    args = parser.parse_args()
    payload = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in {
            "audit": args.audit,
            "teacher": args.positive_teacher,
            "family": args.family_prototype_state,
            "map": args.map,
            "metric": args.metric_state,
            "cache": args.query_cache,
        }.items()
    }
    device = torch.device(args.device)
    metric_payload = payload["metric"]
    metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
    metric.load_state_dict(metric_payload["metric_state_dict"])
    metric.eval()
    config = RepairRoutingConfig(
        minimum_primary_repair_trajectories=(
            args.minimum_primary_repair_trajectories
        ),
        minimum_family_repair_observations=(
            args.minimum_family_repair_observations
        ),
        minimum_family_repair_trajectories=(
            args.minimum_family_repair_trajectories
        ),
        minimum_descriptor_cosine=args.minimum_descriptor_cosine,
        family_prototype_bias=args.family_prototype_bias,
    )
    audit, teacher, family = route_counterfactual_repairs(
        audit=payload["audit"],
        positive_teacher=payload["teacher"],
        family=payload["family"],
        query_cache=payload["cache"],
        metric=metric,
        anchor_count=len(payload["map"]["anchor_ids"]),
        config=config,
        device=device,
    )
    _save(Path(args.audit_output).resolve(), audit)
    _save(Path(args.teacher_output).resolve(), teacher)
    _save(Path(args.family_output).resolve(), family)
    print(audit["summary"]["event_support_routing"])


if __name__ == "__main__":
    main()
