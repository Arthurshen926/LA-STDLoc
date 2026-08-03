#!/usr/bin/env python3
"""Evaluate a compact LaFGS map with one sparse PoseLib solve per query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import load_mainline_config
from data.datasets import ColmapDataset
from evaluation.bootstrap import materialize_a0
from evaluation.evaluator import evaluate_dataset
from localization.localizer import SparseLocalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path)
    parser.add_argument("--metric-state", type=Path)
    parser.add_argument(
        "--stage-state",
        type=Path,
        help="Evaluate A0 by materializing a Stage-A state with an identity metric.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default="configs/paper_mainline.yaml"
    )
    parser.add_argument("--split", choices=("mapping", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.stage_state:
        if args.map or args.metric_state:
            parser.error("--stage-state cannot be combined with --map/--metric-state")
        map_path, metric_path = materialize_a0(
            args.stage_state, args.output / "materialized_a0", args.config
        )
    else:
        if args.map is None or args.metric_state is None:
            parser.error("A1 evaluation requires both --map and --metric-state")
        map_path, metric_path = args.map, args.metric_state
    deployment = load_mainline_config(args.config).values["deployment"]
    dataset = ColmapDataset(args.dataset, images=args.images)
    localizer = SparseLocalizer(
        map_path,
        metric_path,
        device=args.device,
        keypoint_count=deployment["keypoints"],
        reprojection_error_px=deployment["reprojection_error_px"],
        confidence=deployment["confidence"],
        max_iterations=deployment["maximum_iterations"],
        min_iterations=deployment["minimum_iterations"],
        seed=args.seed,
    )
    result = evaluate_dataset(
        dataset=dataset,
        localizer=localizer,
        cameras=dataset.split(args.split),
        output=args.output,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
