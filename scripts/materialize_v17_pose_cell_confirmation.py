#!/usr/bin/env python3
"""Materialize a pose-cell-fresh confirmation plan with sealed provenance."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.v17_pose_cell_planner import plan_v17_pose_cell_confirmation


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


def _save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--prior-plans", type=Path, nargs="+", required=True)
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1720260830)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    dataset = ColmapDataset(args.dataset, images=args.images)
    mapping = dataset.split("mapping")
    poses = torch.stack(
        [torch.as_tensor(camera.pose_w2c, dtype=torch.float64) for camera in mapping]
    )
    intrinsics = torch.stack([_intrinsic(camera) for camera in mapping])
    image_hw = torch.tensor(
        [[camera.height, camera.width] for camera in mapping], dtype=torch.long
    )
    names = [camera.image_name for camera in mapping]
    map_path = args.map.resolve()
    map_state = torch.load(map_path, map_location="cpu", weights_only=False)
    prior_poses = []
    prior_families = []
    prior_inputs = []
    for input_path in args.prior_plans:
        path = input_path.resolve()
        plan = torch.load(path, map_location="cpu", weights_only=False)
        pose = torch.as_tensor(plan["pose_w2c"], dtype=torch.float64)
        family = torch.as_tensor(plan["pose_family_ids"]).long()
        if pose.shape[0] != family.numel():
            raise ValueError("prior pose/family registry differs")
        prior_poses.append(pose)
        prior_families.append(family)
        prior_inputs.append({"path": str(path), "sha256": sha256_file(path)})
    plan = plan_v17_pose_cell_confirmation(
        pose_w2c=poses,
        intrinsics=intrinsics,
        image_hw=image_hw,
        names=names,
        anchor_xyz=map_state["anchor_xyz"],
        prior_pose_w2c=torch.cat(prior_poses),
        prior_source_family_ids=torch.cat(prior_families),
        seed=args.seed,
        maximum_queries=args.count,
    )
    args.output_dir.mkdir(parents=True)
    output_path = args.output_dir / "confirmation_query_plan.pt"
    _save(plan, output_path)
    manifest = {
        "schema": "lafgs_v17_pose_cell_confirmation_registry",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "trajectory_interpolation_candidate_count": 0,
        "map": str(map_path),
        "map_sha256": sha256_file(map_path),
        "prior_plans": prior_inputs,
        "confirmation": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "query_count": int(plan["query_count"]),
            "plan_sha256": plan["plan_sha256"],
        },
        "planner_contract": plan["planner_contract"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
