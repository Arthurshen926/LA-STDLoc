#!/usr/bin/env python3
"""Evaluate a compact LaFGS map with one sparse PoseLib solve per query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.config import load_mainline_config
from data.datasets import ColmapDataset
from evaluation.evaluator import evaluate_dataset
from localization.localizer import SparseLocalizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default="configs/paper_mainline.yaml"
    )
    parser.add_argument("--split", choices=("mapping", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    deployment = load_mainline_config(args.config).values["deployment"]
    dataset = ColmapDataset(args.dataset, images=args.images)
    localizer = SparseLocalizer(
        args.map,
        args.metric_state,
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
