#!/usr/bin/env python3
"""Evaluate immutable map ablations on named post-hoc test cases only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evaluation.evaluator import pose_error
from localization.localizer import SparseLocalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--query-indices", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = ColmapDataset(args.dataset, images="processed")
    cameras = dataset.split("test")
    localizer = SparseLocalizer(
        args.map,
        args.metric,
        device=args.device,
        keypoint_count=2048,
        nms_radius=4,
        reprojection_error_px=11.954343111400277,
        confidence=0.99999,
        max_iterations=100000,
        min_iterations=1000,
        seed=2026,
        profile_mode=True,
    )
    rows = []
    for query_index in args.query_indices:
        camera = cameras[query_index]
        result = localizer.localize(
            dataset.load_image(camera),
            fov_x=camera.fov_x,
            fov_y=camera.fov_y,
            valid_mask=dataset.valid_mask(camera),
        )
        rotation, translation = pose_error(result.pose.pose_w2c, camera.pose_w2c)
        rows.append(
            {
                "query_index": query_index,
                "image_name": camera.image_name,
                "translation_error_cm": float(translation),
                "rotation_error_deg": float(rotation),
                "inlier_count": int(result.pose.inliers.size),
            }
        )
    payload = {
        "schema": "lafgs_v7_anchor_contamination_posthoc_test_cases",
        "version": 1,
        "posthoc_test_rgb_diagnostic": True,
        "may_update_or_select_map": False,
        "threshold_tuning_from_results": False,
        "map_mutation_count": 0,
        "input": {
            "map_sha256": sha256_file(args.map),
            "metric_sha256": sha256_file(args.metric),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(rows, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
