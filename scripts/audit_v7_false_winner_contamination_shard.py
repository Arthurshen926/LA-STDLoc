#!/usr/bin/env python3
"""Trace real-test false Top-1 winners to frozen mapping contamination evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from data.datasets import ColmapDataset
from evaluation.evaluator import pose_error
from evidence.v7_render_real_gap import projected_match_correctness
from localization.localizer import SparseLocalizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--contamination-evidence", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")

    evidence = torch.load(
        args.contamination_evidence, map_location="cpu", weights_only=False
    )
    source_map = torch.load(args.map, map_location="cpu", weights_only=False)
    if (
        evidence.get("schema") != "lafgs_v7_anchor_contamination_evidence"
        or evidence["input"]["anchor_map_sha256"] != sha256_file(args.map)
        or not torch.equal(evidence["anchor_ids"], source_map["anchor_ids"])
    ):
        raise ValueError("contamination evidence does not align with the Full map")

    dataset = ColmapDataset(args.dataset, images=args.images)
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
    buckets = {
        "all": [],
        "correct": [],
        "wrong": [],
        "failure_wrong": [],
        "wrong_pose_inlier": [],
    }
    rows = []
    targets = {135, 136, 139, 140}
    for local_row, query_index in enumerate(
        range(args.shard_index, len(cameras), args.shard_count)
    ):
        camera = cameras[query_index]
        image = dataset.load_image(camera)
        valid_mask = dataset.valid_mask(camera)
        result = localizer.localize(
            image,
            fov_x=camera.fov_x,
            fov_y=camera.fov_y,
            valid_mask=valid_mask,
        )
        anchors = result.matches.anchor_indices.detach().cpu().long()
        correct = projected_match_correctness(
            result.sparse_features.keypoints[result.matches.keypoint_indices],
            localizer.anchor_xyz[result.matches.anchor_indices],
            camera.pose_w2c,
            result.intrinsic,
            maximum_reprojection_px=4.0,
        ).detach().cpu()
        rotation, translation = pose_error(result.pose.pose_w2c, camera.pose_w2c)
        failed = bool(translation >= 5.0 or rotation >= 5.0)
        inlier_rows = torch.zeros(anchors.numel(), dtype=torch.bool)
        if result.pose.inliers.size:
            inlier_rows[torch.from_numpy(result.pose.inliers).long()] = True
        buckets["all"].append(anchors)
        buckets["correct"].append(anchors[correct])
        buckets["wrong"].append(anchors[~correct])
        if failed:
            buckets["failure_wrong"].append(anchors[~correct])
        buckets["wrong_pose_inlier"].append(anchors[(~correct) & inlier_rows])
        row = {
            "query_index": query_index,
            "image_name": camera.image_name,
            "translation_error_cm": float(translation),
            "rotation_error_deg": float(rotation),
            "pose_failed": failed,
            "match_count": int(anchors.numel()),
            "correct_count": int(correct.sum()),
            "wrong_count": int((~correct).sum()),
            "wrong_pose_inlier_count": int(((~correct) & inlier_rows).sum()),
        }
        if query_index in targets:
            wrong = anchors[~correct]
            row["target_detail"] = {
                "wrong_anchor_rows": wrong,
                "wrong_anchor_valid_fraction": evidence[
                    "valid_observation_fraction"
                ][wrong],
                "wrong_anchor_pure_contamination": evidence["pure_contamination"][
                    wrong
                ],
            }
        rows.append(row)
        if (local_row + 1) % 25 == 0:
            print(
                f"false-winner shard {args.shard_index}: {local_row + 1}", flush=True
            )

    packed = {
        key: torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
        for key, parts in buckets.items()
    }
    payload = {
        "schema": "lafgs_v7_false_winner_contamination_audit_shard",
        "version": 1,
        "posthoc_test_rgb_diagnostic": True,
        "may_update_or_select_map": False,
        "threshold_tuning_from_results": False,
        "map_mutation_count": 0,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "input": {
            "map_sha256": sha256_file(args.map),
            "metric_sha256": sha256_file(args.metric),
            "contamination_evidence_sha256": sha256_file(
                args.contamination_evidence
            ),
        },
        "anchor_row_buckets": packed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "query_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
