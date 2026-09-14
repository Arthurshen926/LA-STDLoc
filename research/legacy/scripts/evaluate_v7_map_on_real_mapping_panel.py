#!/usr/bin/env python3
"""Evaluate one immutable map on a fixed non-test real mapping-RGB panel."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evaluation.evaluator import pose_error
from localization.localizer import SparseLocalizer


def _uniform_indices(count: int, limit: int) -> list[int]:
    if limit <= 0 or limit >= count:
        return list(range(count))
    return torch.div(
        torch.arange(limit, dtype=torch.long) * count,
        limit,
        rounding_mode="floor",
    ).tolist()


def _metrics(rows: list[dict]) -> dict:
    translation = torch.tensor([row["translation_error_cm"] for row in rows])
    rotation = torch.tensor([row["rotation_error_deg"] for row in rows])
    success = (translation < 5.0) & (rotation < 5.0)
    return {
        "query_count": len(rows),
        "median_translation_cm": float(translation.median()),
        "mean_translation_cm": float(translation.mean()),
        "p90_translation_cm": float(torch.quantile(translation, 0.9)),
        "recall_5cm_5deg_percent": 100.0 * float(success.float().mean()),
        "catastrophic_50cm_count": int((translation >= 50.0).sum()),
        "median_rotation_deg": float(rotation.median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    dataset = ColmapDataset(args.dataset, images=args.images)
    cameras = dataset.split("mapping")
    indices = _uniform_indices(len(cameras), args.panel_size)
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
    for panel_row, query_index in enumerate(indices):
        camera = cameras[query_index]
        image = dataset.load_image(camera)
        result = localizer.localize(
            image, fov_x=camera.fov_x, fov_y=camera.fov_y, valid_mask=None
        )
        rotation, translation = pose_error(result.pose.pose_w2c, camera.pose_w2c)
        rows.append(
            {
                "panel_row": panel_row,
                "query_index": query_index,
                "image_name": camera.image_name,
                "translation_error_cm": float(translation),
                "rotation_error_deg": float(rotation),
                "keypoint_count": int(result.sparse_features.keypoints.shape[0]),
                "inlier_count": int(result.pose.inliers.size),
            }
        )
        if (panel_row + 1) % 16 == 0 or panel_row + 1 == len(indices):
            print(f"real mapping panel {panel_row + 1}/{len(indices)}", flush=True)
    payload = {
        "schema": "lafgs_v7_real_mapping_rgb_map_ablation",
        "version": 1,
        "status": "PASS",
        "uses_source_mapping_rgb_for_evaluation_only": True,
        "source_mapping_descriptors_written_to_map": False,
        "uses_test_queries": False,
        "threshold_tuning_from_results": False,
        "panel_policy": "uniform_128_frozen_before_variant_results",
        "panel_indices": indices,
        "input": {
            "dataset": str(args.dataset.resolve()),
            "map": str(args.map.resolve()),
            "map_sha256": sha256_file(args.map),
            "metric": str(args.metric.resolve()),
            "metric_sha256": sha256_file(args.metric),
        },
        "metrics": _metrics(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
