#!/usr/bin/env python3
"""Isolate detector allocation with fixed descriptors, map, Top-1 and PoseLib."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from features.scene_specific_detector import fuse_scene_reliability, load_scene_detector_checkpoint
from features.superpoint import SuperPoint
from localization.matcher import global_cosine_top1
from localization.pose_solver import pose_error, solve_absolute_pose


def _summary(rows: list[dict]) -> dict:
    te = np.asarray([row["translation_error_cm"] for row in rows])
    ae = np.asarray([row["rotation_error_deg"] for row in rows])
    return {
        "query_count": len(rows), "median_te_cm": float(np.median(te)),
        "p90_te_cm": float(np.percentile(te, 90)), "median_ae_deg": float(np.median(ae)),
        "r5_percent": 100 * float(np.mean((te <= 5) & (ae <= 5))),
        "catastrophic_100cm_count": int(np.sum(te >= 100)),
        "mean_gt4px_percent": 100 * float(np.mean([row["gt4px"] for row in rows])),
        "mean_positive_hit_percent": 100 * float(np.mean([row["positive_hit"] for row in rows])),
        "mean_spatial_cells": float(np.mean([row["spatial_cells"] for row in rows])),
        "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in rows])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--detector-lineage-map", type=Path,
        help="Map that defined detector targets when matching uses a larger map",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keypoints", type=int, default=2048)
    args = parser.parse_args()
    from common.hashing import sha256_file
    device = torch.device(args.device)
    state = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    head = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        target_map = args.detector_lineage_map or args.anchor_map
        head = load_scene_detector_checkpoint(
            checkpoint, map_sha256=sha256_file(target_map)
        ).to(device).eval()
    encoder = SuperPoint().to(device).eval()
    encoder.nms_radius = 4
    anchors = F.normalize(torch.as_tensor(state["anchor_features"], device=device).float(), dim=1)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    paths = sorted(args.dataset_root.glob("confirmation_*.pt"))
    if not paths:
        raise FileNotFoundError("confirmation dataset is empty")
    arms = {"superpoint": []}
    if head is not None:
        arms["scene_detector"] = []
    for query_index, path in enumerate(paths):
        record = torch.load(path, map_location="cpu", weights_only=False)
        image = record["rgb_u8"].to(device=device, dtype=torch.float32)[None] / 255
        labels = record["labels"].to(device)
        with torch.inference_mode():
            dense, sp_scores = encoder._dense_outputs(image)
            score_maps = {"superpoint": sp_scores}
            if head is not None:
                detector_logits = head(dense, output_hw=tuple(sp_scores.shape[-2:]))
                score_maps["scene_detector"] = fuse_scene_reliability(
                    sp_scores, detector_logits
                )
            for arm, scores in score_maps.items():
                started = time.perf_counter()
                sparse = encoder._sparse_from_dense(dense, scores, top_k=args.keypoints, detection_threshold=0.0)[0]
                matches = global_cosine_top1(sparse["descriptors"], anchors, anchor_descriptors_normalized=True)
                keypoints = sparse["keypoints"].cpu()
                winner_xyz = xyz[matches.anchor_indices.cpu()]
                estimate = solve_absolute_pose(
                    (keypoints + 0.5).numpy(), winner_xyz.numpy(), record["intrinsic"].numpy(), seed=2026 + query_index
                )
                ae, te = pose_error(estimate.pose_w2c, record["pose_w2c"].numpy())
                camera = winner_xyz @ record["pose_w2c"][:3, :3].T + record["pose_w2c"][:3, 3]
                projected = camera @ record["intrinsic"].T
                uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
                gt4 = ((uv - (keypoints + 0.5)).norm(dim=1) <= 4).float().mean()
                label_xy = (keypoints / 8).long()
                label_xy[:, 0].clamp_(0, labels.shape[1] - 1)
                label_xy[:, 1].clamp_(0, labels.shape[0] - 1)
                positive_hit = (labels[label_xy[:, 1], label_xy[:, 0]] == 1).float().mean()
                cell_x = (keypoints[:, 0] * 4 / image.shape[-1]).long().clamp(0, 3)
                cell_y = (keypoints[:, 1] * 4 / image.shape[-2]).long().clamp(0, 3)
                arms[arm].append({
                    "query_name": record["query_name"], "translation_error_cm": te,
                    "rotation_error_deg": ae, "gt4px": float(gt4),
                    "positive_hit": float(positive_hit),
                    "spatial_cells": int(torch.unique(cell_y * 4 + cell_x).numel()),
                    "runtime_ms": 1000 * (time.perf_counter() - started),
                })
    report = {
        "schema": "lafgs_v8_scene_detector_isolation", "version": 1,
        "uses_test_rgb": False, "same_anchor_map": True,
        "same_descriptors_matcher_poselib": True,
        "matching_map_sha256": sha256_file(args.anchor_map),
        "detector_target_map_sha256": (
            sha256_file(args.detector_lineage_map or args.anchor_map)
            if head is not None else None
        ),
        "arms": {name: {"summary": _summary(rows), "rows": rows} for name, rows in arms.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: value["summary"] for name, value in report["arms"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
