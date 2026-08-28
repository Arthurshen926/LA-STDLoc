#!/usr/bin/env python3
"""Materialize disjoint V10 safety and fresh-confirmation confusion plans."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evidence.v10_confusion_query_planner import plan_v10_confusion_queries


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
    parser.add_argument("--descriptor-proposal", type=Path, required=True)
    parser.add_argument("--authorized-action-audit", type=Path, required=True)
    parser.add_argument("--feedback-batches", type=Path, nargs="+", required=True)
    parser.add_argument("--prior-plans", type=Path, nargs="*", default=[])
    parser.add_argument("--safety-count", type=int, default=128)
    parser.add_argument("--confirmation-count", type=int, default=128)
    parser.add_argument("--safety-seed", type=int, default=1020260828)
    parser.add_argument("--confirmation-seed", type=int, default=1020260829)
    parser.add_argument("--views-per-pair", type=int, default=4)
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
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    proposal_path = args.descriptor_proposal.resolve()
    proposal = torch.load(proposal_path, map_location="cpu", weights_only=False)
    action_audit_path = args.authorized_action_audit.resolve()
    action_audit = torch.load(action_audit_path, map_location="cpu", weights_only=False)
    if action_audit.get("schema") != "lafgs_v10_single_descriptor_action_gain_audit":
        raise ValueError("V10 authorized action audit schema differs")
    priority_anchor_rows = torch.as_tensor(
        action_audit["authorized_anchor_rows"]
    ).long().tolist()
    candidate_rows = set(
        torch.as_tensor(proposal["candidate_anchor_rows"]).long().tolist()
    )
    pair_support: Counter[tuple[int, int]] = Counter()
    feedback_inputs = []
    for batch_input in args.feedback_batches:
        batch_path = batch_input.resolve()
        batch = json.loads(batch_path.read_text())
        feedback_inputs.append({"path": str(batch_path), "sha256": sha256_file(batch_path)})
        for item in batch["records"]:
            record = torch.load(item["path"], map_location="cpu", weights_only=False)
            if not record["can_train_metric"]:
                continue
            evidence = record["training_evidence"]
            positive = torch.as_tensor(evidence["positive_anchor_rows"]).long()
            negative = torch.as_tensor(evidence["negative_anchor_rows"]).long()
            for left, right in zip(positive.tolist(), negative.tolist()):
                if int(left) in candidate_rows:
                    pair_support[(int(left), int(right))] += 1
    if not pair_support:
        raise RuntimeError("V10 proposal has no confusion-pair evidence")
    ordered_pairs = sorted(
        pair_support, key=lambda pair: (-pair_support[pair], pair[0], pair[1])
    )
    pairs = torch.tensor(ordered_pairs, dtype=torch.long)
    forbidden = set()
    prior_inputs = []
    for prior_input in args.prior_plans:
        path = prior_input.resolve()
        plan = torch.load(path, map_location="cpu", weights_only=False)
        forbidden.update(torch.as_tensor(plan["pose_family_ids"]).long().tolist())
        prior_inputs.append({"path": str(path), "sha256": sha256_file(path)})
    common = {
        "pose_w2c": poses,
        "intrinsics": intrinsics,
        "image_hw": image_hw,
        "names": names,
        "anchor_xyz": state["anchor_xyz"],
        "confusion_pairs": pairs,
        "priority_anchor_rows": priority_anchor_rows,
        "maximum_views_per_confusion_pair": args.views_per_pair,
    }
    safety = plan_v10_confusion_queries(
        role="feedback_query",
        feedback_stage="safety",
        seed=args.safety_seed,
        maximum_queries=args.safety_count,
        forbidden_pose_family_ids=sorted(forbidden),
        **common,
    )
    forbidden.update(safety["pose_family_ids"].tolist())
    confirmation = plan_v10_confusion_queries(
        role="confirmation_query",
        feedback_stage=None,
        seed=args.confirmation_seed,
        maximum_queries=args.confirmation_count,
        forbidden_pose_family_ids=sorted(forbidden),
        **common,
    )
    if _pose_digests(safety) & _pose_digests(confirmation):
        raise RuntimeError("V10 safety and confirmation poses overlap")
    if set(safety["pose_family_ids"].tolist()) & set(
        confirmation["pose_family_ids"].tolist()
    ):
        raise RuntimeError("V10 safety and confirmation families overlap")
    args.output_dir.mkdir(parents=True)
    safety_path = args.output_dir / "safety_feedback_plan.pt"
    confirmation_path = args.output_dir / "fresh_confirmation_plan.pt"
    _save(safety, safety_path)
    _save(confirmation, confirmation_path)
    manifest = {
        "schema": "lafgs_v10_confusion_query_plan_registry",
        "version": 1,
        "loo_used": False,
        "trajectory_interpolation_candidate_count": 0,
        "ambiguity_full_look_at": False,
        "uses_test_queries": False,
        "map": str(map_path),
        "map_sha256": sha256_file(map_path),
        "descriptor_proposal": str(proposal_path),
        "descriptor_proposal_sha256": sha256_file(proposal_path),
        "authorized_action_audit": str(action_audit_path),
        "authorized_action_audit_sha256": sha256_file(action_audit_path),
        "authorized_action_anchor_count": len(priority_anchor_rows),
        "feedback_inputs": feedback_inputs,
        "prior_plan_inputs": prior_inputs,
        "confusion_pair_count": len(ordered_pairs),
        "safety": {
            "path": str(safety_path.resolve()),
            "sha256": sha256_file(safety_path),
            "query_count": safety["query_count"],
        },
        "confirmation": {
            "path": str(confirmation_path.resolve()),
            "sha256": sha256_file(confirmation_path),
            "query_count": confirmation["query_count"],
        },
        "safety_confirmation_family_overlap_count": 0,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
