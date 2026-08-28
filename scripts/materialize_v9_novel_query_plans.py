#!/usr/bin/env python3
"""Materialize disjoint interpolation-free V9 feedback/confirmation plans."""

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
from evidence.v9_novel_query_planner import plan_v9_novel_queries


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


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pose_digests(plan: dict) -> set[str]:
    return {
        hashlib.sha256(pose.contiguous().numpy().tobytes()).hexdigest()
        for pose in plan["pose_w2c"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--ambiguity-proposal", type=Path)
    parser.add_argument("--feedback-count", type=int, default=256)
    parser.add_argument("--confirmation-count", type=int, default=128)
    parser.add_argument("--feedback-seed", type=int, default=920260828)
    parser.add_argument("--confirmation-seed", type=int, default=920260829)
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
    map_state = torch.load(args.map, map_location="cpu", weights_only=False)
    anchor_xyz = torch.as_tensor(map_state["anchor_xyz"]).float()
    ambiguity_xyz = torch.empty(0, 3)
    ambiguity_sha = None
    if args.ambiguity_proposal is not None:
        proposal = torch.load(
            args.ambiguity_proposal, map_location="cpu", weights_only=False
        )
        rows = torch.as_tensor(proposal["proposed_anchor_rows"]).long()
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= len(anchor_xyz)):
            raise ValueError("ambiguity proposal rows differ from the fixed V2 map")
        ambiguity_xyz = anchor_xyz[rows]
        ambiguity_sha = sha256_file(args.ambiguity_proposal)

    common = {
        "pose_w2c": poses,
        "intrinsics": intrinsics,
        "image_hw": image_hw,
        "names": names,
        "anchor_xyz": anchor_xyz,
        "ambiguity_xyz": ambiguity_xyz,
    }
    feedback = plan_v9_novel_queries(
        role="feedback_query",
        seed=args.feedback_seed,
        maximum_queries=args.feedback_count,
        **common,
    )
    confirmation = plan_v9_novel_queries(
        role="confirmation_query",
        seed=args.confirmation_seed,
        maximum_queries=args.confirmation_count,
        forbidden_pose_family_ids=feedback["pose_family_ids"].tolist(),
        **common,
    )
    overlap = _pose_digests(feedback) & _pose_digests(confirmation)
    if overlap:
        raise RuntimeError("feedback and confirmation pose registries overlap")
    for plan in (feedback, confirmation):
        if plan["loo_used"] or plan["trajectory_interpolation_candidate_count"]:
            raise RuntimeError("V9 no-LOO/no-interpolation contract was violated")

    args.output_dir.mkdir(parents=True)
    feedback_path = args.output_dir / "feedback_query_plan.pt"
    confirmation_path = args.output_dir / "confirmation_query_plan.pt"
    _atomic_save(feedback, feedback_path)
    _atomic_save(confirmation, confirmation_path)
    manifest = {
        "schema": "lafgs_v9_novel_query_plan_registry",
        "version": 1,
        "loo_used": False,
        "trajectory_interpolation_candidate_count": 0,
        "uses_test_queries": False,
        "mapping_camera_count": len(mapping),
        "map": str(args.map.resolve()),
        "map_sha256": sha256_file(args.map),
        "ambiguity_proposal": (
            None
            if args.ambiguity_proposal is None
            else str(args.ambiguity_proposal.resolve())
        ),
        "ambiguity_proposal_sha256": ambiguity_sha,
        "feedback": {
            "path": str(feedback_path.resolve()),
            "sha256": sha256_file(feedback_path),
            "query_count": feedback["query_count"],
            "plan_sha256": feedback["plan_sha256"],
        },
        "confirmation": {
            "path": str(confirmation_path.resolve()),
            "sha256": sha256_file(confirmation_path),
            "query_count": confirmation["query_count"],
            "plan_sha256": confirmation["plan_sha256"],
        },
        "pose_registry_overlap_count": 0,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
