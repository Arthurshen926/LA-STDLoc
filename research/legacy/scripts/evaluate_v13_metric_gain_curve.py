#!/usr/bin/env python3
"""Exact Top-1 + PoseLib replay for a frozen low-rank action gain curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from localization.matcher import global_cosine_top1
from map_learning.metric import SharedLowRankMetric
from map_learning.v8_safety_actions import certified_feedback_row_mask
from map_learning.v9_causal_feedback import standard_pose_replay


def _arm(alpha: float) -> str:
    return "alpha_" + (f"{alpha:.4f}".rstrip("0").rstrip(".").replace(".", "p"))


def _load_metric(path: Path, anchor_ids: torch.Tensor, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not (
        state.get("protocol")
        in {
            "v9_no_loo_causal_shared_metric",
            "v19_mapping_track_identity_pretraining",
        }
        and state.get("loo_used", False) is False
        and torch.equal(torch.as_tensor(state["landmark_indices"]).long(), anchor_ids)
    ):
        raise ValueError("gain curve requires a frozen no-LOO shared metric")
    metric = SharedLowRankMetric(**state["metric_config"]).to(device)
    metric.load_state_dict(state["metric_state_dict"], strict=True)
    return metric.eval()


@torch.inference_mode()
def _scaled(metric, descriptor: torch.Tensor, alpha: float):
    normalized = F.normalize(descriptor.float(), dim=1)
    _, residual = metric(normalized)
    return F.normalize(normalized + float(alpha) * residual, dim=1)


def _json_pose(pose: dict) -> dict:
    pose = dict(pose)
    pose["pose_w2c"] = pose["pose_w2c"].tolist()
    return pose


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-batch", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--active-set-map", type=Path)
    parser.add_argument("--joint-only", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--expected-role", choices=("feedback_query", "confirmation_query"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    alphas = sorted(set(float(value) for value in args.alphas))
    if not alphas or alphas[0] < 0.0 or alphas[-1] > 1.25 or 0.0 in alphas:
        parser.error("candidate alphas must be unique values in (0, 1.25]")
    if args.joint_only and args.active_set_map is None:
        parser.error("--joint-only requires --active-set-map")

    batch_path = args.certified_batch.resolve()
    batch = json.loads(batch_path.read_text())
    if not (
        batch.get("view_role") == args.expected_role
        and batch.get("uses_test_queries") is False
        and batch.get("map_mutation_count") == 0
    ):
        raise ValueError("gain curve requires a sealed non-test certified batch")
    map_path = args.baseline_map.resolve()
    map_payload = torch.load(map_path, map_location="cpu", weights_only=False)
    anchor_ids = torch.as_tensor(map_payload["anchor_ids"]).long()
    if torch.unique(anchor_ids).numel() != anchor_ids.numel():
        raise ValueError("baseline Anchor IDs must be unique")
    target = torch.device(args.device)
    native_anchor = F.normalize(
        torch.as_tensor(map_payload["anchor_features"], device=target).float(), dim=1
    )
    anchor_xyz = torch.as_tensor(map_payload["anchor_xyz"]).float()
    metric_path = args.metric.resolve()
    metric = _load_metric(metric_path, anchor_ids, target)
    transformed_anchors = (
        {}
        if args.joint_only
        else {_arm(alpha): _scaled(metric, native_anchor, alpha) for alpha in alphas}
    )
    active_path = args.active_set_map.resolve() if args.active_set_map else None
    active_anchor = active_xyz = None
    transformed_active = {}
    if active_path is not None:
        active_payload = torch.load(active_path, map_location="cpu", weights_only=False)
        active_ids = torch.as_tensor(active_payload["anchor_ids"]).long()
        if not set(active_ids.tolist()).issubset(set(anchor_ids.tolist())):
            raise ValueError("active-set action may only select from the frozen M0")
        active_anchor = F.normalize(
            torch.as_tensor(active_payload["anchor_features"], device=target).float(), dim=1
        )
        active_xyz = torch.as_tensor(active_payload["anchor_xyz"]).float()
        transformed_active = {
            _arm(alpha) + "_active": _scaled(metric, active_anchor, alpha)
            for alpha in alphas
        }

    selected = [
        item
        for index, item in enumerate(batch["records"])
        if index % args.shard_count == args.shard_index
    ]
    records = []
    for local_index, item in enumerate(selected):
        source_path = Path(item["path"]).resolve()
        if sha256_file(source_path) != item["sha256"]:
            raise ValueError("certified query record SHA256 differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if source["certificate"]["decision"] != "ACCEPT":
            continue
        valid = certified_feedback_row_mask(source["certificate"])
        if valid.numel() != torch.as_tensor(source["descriptors"]).shape[0]:
            raise ValueError("V2 row-valid mask does not align with descriptors")
        source_query_rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        query = F.normalize(
            torch.as_tensor(source["descriptors"])[source_query_rows]
            .to(device=target)
            .float(),
            dim=1,
        )
        keypoints = torch.as_tensor(source["keypoints"])[source_query_rows].float() + 0.5
        baseline_matches = global_cosine_top1(
            query, native_anchor, anchor_descriptors_normalized=True
        )
        row = {
            "query_index": int(source["query_index"]),
            "pose_family_id": int(source["pose_family_id"]),
            "valid_query_row_count": int(source_query_rows.numel()),
            "invalid_query_row_count": int(valid.numel() - source_query_rows.numel()),
            "baseline": _json_pose(
                standard_pose_replay(
                    keypoints=keypoints,
                    anchor_rows=baseline_matches.anchor_indices.cpu(),
                    anchor_xyz=anchor_xyz,
                    intrinsic=source["intrinsics"],
                    ground_truth_w2c=source["pose_w2c"],
                )
            ),
        }
        if active_anchor is not None and not args.joint_only:
            active_matches = global_cosine_top1(
                query, active_anchor, anchor_descriptors_normalized=True
            )
            row["active"] = _json_pose(
                standard_pose_replay(
                    keypoints=keypoints,
                    anchor_rows=active_matches.anchor_indices.cpu(),
                    anchor_xyz=active_xyz,
                    intrinsic=source["intrinsics"],
                    ground_truth_w2c=source["pose_w2c"],
                )
            )
            row["active"]["top1_changed_count"] = int(
                (active_ids[active_matches.anchor_indices.cpu()] != anchor_ids[baseline_matches.anchor_indices.cpu()]).sum()
            )
        transformed_query_base = {
            _arm(alpha): _scaled(metric, query, alpha) for alpha in alphas
        }
        for alpha in alphas:
            arm = _arm(alpha)
            if not args.joint_only:
                matches = global_cosine_top1(
                    transformed_query_base[arm],
                    transformed_anchors[arm],
                    anchor_descriptors_normalized=True,
                )
                row[arm] = _json_pose(
                    standard_pose_replay(
                        keypoints=keypoints,
                        anchor_rows=matches.anchor_indices.cpu(),
                        anchor_xyz=anchor_xyz,
                        intrinsic=source["intrinsics"],
                        ground_truth_w2c=source["pose_w2c"],
                    )
                )
                row[arm]["top1_changed_count"] = int(
                    (matches.anchor_indices != baseline_matches.anchor_indices).sum()
                )
            if active_anchor is not None:
                joint_arm = arm + "_active"
                joint_matches = global_cosine_top1(
                    transformed_query_base[arm],
                    transformed_active[joint_arm],
                    anchor_descriptors_normalized=True,
                )
                row[joint_arm] = _json_pose(
                    standard_pose_replay(
                        keypoints=keypoints,
                        anchor_rows=joint_matches.anchor_indices.cpu(),
                        anchor_xyz=active_xyz,
                        intrinsic=source["intrinsics"],
                        ground_truth_w2c=source["pose_w2c"],
                    )
                )
                row[joint_arm]["top1_changed_count"] = int(
                    (
                        active_ids[joint_matches.anchor_indices.cpu()]
                        != anchor_ids[baseline_matches.anchor_indices.cpu()]
                    ).sum()
                )
        records.append(row)
        if (local_index + 1) % 4 == 0 or local_index + 1 == len(selected):
            print(
                f"V13 {args.expected_role} shard {args.shard_index}: "
                f"{local_index + 1}/{len(selected)}",
                flush=True,
            )

    output = {
        "schema": "lafgs_v13_metric_gain_curve_shard",
        "version": 1,
        "uses_test_queries": False,
        "loo_used": False,
        "accepted_query_row_policy": "v2_row_valid_only",
        "evaluation_role": args.expected_role,
        "candidate_alphas": alphas,
        "candidate_arms": (
            [_arm(alpha) + "_active" for alpha in alphas]
            if args.joint_only
            else (
                [_arm(alpha) for alpha in alphas]
                if active_path is None
                else ["active"]
                + [_arm(alpha) for alpha in alphas]
                + [_arm(alpha) + "_active" for alpha in alphas]
            )
        ),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "source_query_count": len(batch["records"]),
        "accepted_query_count": len(records),
        "input": {
            "certified_batch": str(batch_path),
            "certified_batch_sha256": sha256_file(batch_path),
            "baseline_map": str(map_path),
            "baseline_map_sha256": sha256_file(map_path),
            "metric": str(metric_path),
            "metric_sha256": sha256_file(metric_path),
            "active_set_map": None if active_path is None else str(active_path),
            "active_set_map_sha256": None if active_path is None else sha256_file(active_path),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
