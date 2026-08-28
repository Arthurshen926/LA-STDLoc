#!/usr/bin/env python3
"""Paired fresh-confirmation evaluation for baseline and V9 metric proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from localization.matcher import global_cosine_top1
from map_learning.metric import SharedLowRankMetric
from map_learning.v9_causal_feedback import standard_pose_replay


def _load_metric(path: Path, anchor_ids: torch.Tensor, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    protocol = state.get("protocol")
    if not (
        protocol
        in {
            "v9_no_loo_causal_shared_metric",
            "rendered_track_map_bound_identity",
            "v6_identity_shared_metric",
        }
        and state.get("loo_used", False) is False
        and torch.equal(
            torch.as_tensor(state["landmark_indices"]).long(), anchor_ids.long()
        )
    ):
        raise ValueError("proposal metric violates the V9 no-LOO map binding")
    metric = SharedLowRankMetric(**state["metric_config"]).to(device)
    metric.load_state_dict(state["metric_state_dict"], strict=True)
    return metric.eval()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-batch", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--proposal-map", type=Path, required=True)
    parser.add_argument("--proposal-metric", type=Path, required=True)
    parser.add_argument(
        "--expected-role",
        choices=("feedback_query", "confirmation_query"),
        default="confirmation_query",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    batch_path = args.certified_batch.resolve()
    batch = json.loads(batch_path.read_text())
    if not (
        batch.get("view_role") == args.expected_role
        and batch.get("uses_test_queries") is False
        and batch.get("map_mutation_count") == 0
    ):
        raise ValueError("evaluation requires a sealed fresh confirmation batch")
    baseline_path = args.baseline_map.resolve()
    proposal_path = args.proposal_map.resolve()
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    proposal = torch.load(proposal_path, map_location="cpu", weights_only=False)
    proposal_ids = torch.as_tensor(proposal["anchor_ids"]).long()
    if torch.unique(proposal_ids).numel() != proposal_ids.numel():
        raise ValueError("proposal Anchor IDs must remain unique")
    device = torch.device(args.device)
    baseline_features = F.normalize(
        torch.as_tensor(baseline["anchor_features"], device=device).float(), dim=1
    )
    proposal_features = F.normalize(
        torch.as_tensor(proposal["anchor_features"], device=device).float(), dim=1
    )
    baseline_xyz = torch.as_tensor(baseline["anchor_xyz"]).float()
    proposal_xyz = torch.as_tensor(proposal["anchor_xyz"]).float()
    metric_path = args.proposal_metric.resolve()
    metric = _load_metric(metric_path, proposal_ids, device)
    records = []
    selected = [
        item
        for index, item in enumerate(batch["records"])
        if index % args.shard_count == args.shard_index
    ]
    for index, item in enumerate(selected):
        source_path = Path(item["path"]).resolve()
        if sha256_file(source_path) != item["sha256"]:
            raise ValueError("confirmation record SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if source["certificate"]["decision"] != "ACCEPT":
            continue
        query = F.normalize(
            torch.as_tensor(source["descriptors"], device=device).float(), dim=1
        )
        baseline_matches = global_cosine_top1(
            query,
            baseline_features,
            anchor_descriptors_normalized=True,
        )
        transformed_query, residual = metric(query)
        if bool(
            (torch.linalg.norm(residual, dim=1) > metric.max_residual_norm + 1e-6).any()
        ):
            raise RuntimeError("proposal query metric exceeded its trust region")
        proposal_matches = global_cosine_top1(
            transformed_query,
            proposal_features,
            anchor_descriptors_normalized=True,
        )
        keypoints = torch.as_tensor(source["keypoints"]).float() + 0.5
        baseline_pose = standard_pose_replay(
            keypoints=keypoints,
            anchor_rows=baseline_matches.anchor_indices.cpu(),
            anchor_xyz=baseline_xyz,
            intrinsic=source["intrinsics"],
            ground_truth_w2c=source["pose_w2c"],
        )
        proposal_pose = standard_pose_replay(
            keypoints=keypoints,
            anchor_rows=proposal_matches.anchor_indices.cpu(),
            anchor_xyz=proposal_xyz,
            intrinsic=source["intrinsics"],
            ground_truth_w2c=source["pose_w2c"],
        )
        baseline_pose["pose_w2c"] = baseline_pose["pose_w2c"].tolist()
        proposal_pose["pose_w2c"] = proposal_pose["pose_w2c"].tolist()
        records.append(
            {
                "query_index": int(source["query_index"]),
                "pose_family_id": int(source["pose_family_id"]),
                "baseline": baseline_pose,
                "proposal": proposal_pose,
                "task_gain": baseline_pose["task_error"]
                - proposal_pose["task_error"],
            }
        )
        if (index + 1) % 8 == 0 or index + 1 == len(selected):
            print(
                f"V9 confirmation shard {args.shard_index}: "
                f"{index + 1}/{len(selected)}",
                flush=True,
            )
    output = {
        "schema": "lafgs_v9_paired_confirmation_shard",
        "version": 1,
        "loo_used": False,
        "uses_test_queries": False,
        "confirmation_can_train_or_select": False,
        "evaluation_role": args.expected_role,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_query_count": len(batch["records"]),
        "accepted_query_count": len(records),
        "input": {
            "certified_batch": str(batch_path),
            "certified_batch_sha256": sha256_file(batch_path),
            "baseline_map": str(baseline_path),
            "baseline_map_sha256": sha256_file(baseline_path),
            "proposal_map": str(proposal_path),
            "proposal_map_sha256": sha256_file(proposal_path),
            "proposal_metric": str(metric_path),
            "proposal_metric_sha256": sha256_file(metric_path),
        },
        "records": records,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
