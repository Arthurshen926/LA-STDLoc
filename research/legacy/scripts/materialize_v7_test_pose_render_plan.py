#!/usr/bin/env python3
"""Create a metadata-only, explicitly non-formal test-pose render plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from common.v7_contracts import sha256_file
from data.datasets import ColmapDataset
from evidence.v7_query_planner import plan_v7_test_pose_render_diagnostic


def _intrinsic(camera) -> torch.Tensor:
    focal_x = camera.width / (2.0 * math.tan(camera.fov_x / 2.0))
    focal_y = camera.height / (2.0 * math.tan(camera.fov_y / 2.0))
    return torch.tensor(
        [
            [focal_x, 0.0, camera.width / 2.0],
            [0.0, focal_y, camera.height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    test = dataset.split("test")
    global_indices = torch.arange(len(test), dtype=torch.long)
    shard_rows = global_indices[global_indices.remainder(args.shard_count) == args.shard_index]
    selected = [test[int(row)] for row in shard_rows]
    plan = plan_v7_test_pose_render_diagnostic(
        mapping_pose_w2c=torch.stack(
            [torch.as_tensor(camera.pose_w2c) for camera in mapping]
        ),
        mapping_names=[camera.image_name for camera in mapping],
        test_pose_w2c=torch.stack(
            [torch.as_tensor(camera.pose_w2c) for camera in selected]
        ),
        test_intrinsics=torch.stack([_intrinsic(camera) for camera in selected]),
        test_image_hw=torch.tensor(
            [[camera.height, camera.width] for camera in selected], dtype=torch.long
        ),
        test_names=[camera.image_name for camera in selected],
        query_indices=shard_rows,
    )
    args.output_dir.mkdir(parents=True)
    plan_path = args.output_dir / "query_plan.pt"
    temporary = plan_path.with_name(f".{plan_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(plan, temporary)
        os.replace(temporary, plan_path)
    finally:
        temporary.unlink(missing_ok=True)
    test_list = args.dataset.resolve() / "dataset_test.txt"
    manifest = {
        "schema": "lafgs_v7_test_pose_render_diagnostic_plan_manifest",
        "version": 1,
        "formal_protocol_eligible": False,
        "transductive_pose_distribution_oracle": True,
        "uses_test_pose_metadata": True,
        "uses_test_rgb": False,
        "test_image_files_opened": False,
        "dataset": str(args.dataset.resolve()),
        "test_camera_list": str(test_list),
        "test_camera_list_sha256": sha256_file(test_list),
        "mapping_camera_count": len(mapping),
        "total_test_camera_count": len(test),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "query_count": len(selected),
        "query_indices": shard_rows.tolist(),
        "output": {
            "query_plan": str(plan_path.resolve()),
            "query_plan_sha256": sha256_file(plan_path),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
