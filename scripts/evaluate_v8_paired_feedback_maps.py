#!/usr/bin/env python3
"""Paired map-action evaluation with one shared SuperPoint extraction per RGB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common.hashing import sha256_file
from evaluation.evaluator import pose_error
from localization.localizer import SparseLocalizer
from localization.matcher import global_owner_prototype_top1
from localization.pose_solver import poselib_camera, solve_absolute_pose


def _metrics(rows: list[dict]) -> dict:
    translation = np.asarray([row["translation_error_cm"] for row in rows])
    rotation = np.asarray([row["rotation_error_deg"] for row in rows])
    success = (translation < 5.0) & (rotation < 5.0)
    return {
        "query_count": len(rows),
        "median_translation_cm": float(np.median(translation)),
        "p90_translation_cm": float(np.quantile(translation, 0.9)),
        "mean_translation_cm": float(np.mean(translation)),
        "median_rotation_deg": float(np.median(rotation)),
        "recall_5cm_5deg_percent": float(success.mean() * 100.0),
        "catastrophic_50cm_count": int((translation >= 50.0).sum()),
    }


def _paired(baseline: list[dict], proposal: list[dict]) -> dict:
    old = np.asarray([row["translation_error_cm"] for row in baseline])
    new = np.asarray([row["translation_error_cm"] for row in proposal])
    old_r = np.asarray([row["rotation_error_deg"] for row in baseline])
    new_r = np.asarray([row["rotation_error_deg"] for row in proposal])
    delta = new - old
    old_success = (old < 5.0) & (old_r < 5.0)
    new_success = (new < 5.0) & (new_r < 5.0)
    return {
        "translation_improved_count": int((delta < 0).sum()),
        "translation_worsened_count": int((delta > 0).sum()),
        "translation_unchanged_count": int((delta == 0).sum()),
        "median_paired_translation_delta_cm": float(np.median(delta)),
        "mean_paired_translation_delta_cm": float(np.mean(delta)),
        "r5_recovered_count": int(((~old_success) & new_success).sum()),
        "r5_lost_count": int((old_success & (~new_success)).sum()),
        "catastrophic_resolved_count": int(((old >= 50) & (new < 50)).sum()),
        "catastrophic_created_count": int(((old < 50) & (new >= 50)).sum()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certified-manifest", type=Path, required=True)
    parser.add_argument("--baseline-map", type=Path, required=True)
    parser.add_argument("--baseline-metric", type=Path, required=True)
    parser.add_argument("--proposal-maps", type=Path, nargs="+", required=True)
    parser.add_argument("--proposal-metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (
        len(args.proposal_maps) == len(args.proposal_metrics) == len(args.names)
    ):
        raise ValueError("paired proposal registries differ")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.certified_manifest.read_text())
    if not (
        manifest.get("view_role") in {"feedback_query", "confirmation_query"}
        and manifest.get("uses_test_queries") is False
    ):
        raise ValueError("paired feedback evaluation requires a non-test batch")
    batch = [item for item in manifest["records"] if item["decision"] == "ACCEPT"]
    paths = [args.baseline_map, *args.proposal_maps]
    metrics = [args.baseline_metric, *args.proposal_metrics]
    names = ["baseline", *args.names]
    localizers = [
        SparseLocalizer(
            map_path,
            metric_path,
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
        for map_path, metric_path in zip(paths, metrics)
    ]
    rows = {name: [] for name in names}
    for index, item in enumerate(batch):
        source_path = Path(item["path"])
        if sha256_file(source_path) != item["sha256"]:
            raise ValueError("certified record SHA differs")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        sparse = localizers[0].frontend(source["rgb_float16"].float())
        intrinsic = torch.as_tensor(source["intrinsics"]).float().numpy()
        camera = poselib_camera(intrinsic)
        points_2d = sparse.keypoints.cpu().numpy() + 0.5
        for name, localizer in zip(names, localizers):
            matches = global_owner_prototype_top1(
                sparse.descriptors,
                localizer.anchor_features,
                localizer.anchor_extra_prototype_features,
                localizer.anchor_extra_prototype_owner_rows,
                anchor_descriptors_normalized=True,
            )
            points_3d = localizer.anchor_xyz[matches.anchor_indices].cpu().numpy()
            pose = solve_absolute_pose(
                points_2d,
                points_3d,
                intrinsic,
                reprojection_error_px=11.954343111400277,
                confidence=0.99999,
                max_iterations=100000,
                min_iterations=1000,
                seed=2026,
                progressive_sampling=False,
                camera=camera,
            )
            rotation, translation = pose_error(
                pose.pose_w2c, torch.as_tensor(source["pose_w2c"]).numpy()
            )
            rows[name].append(
                {
                    "query_index": int(source["query_index"]),
                    "translation_error_cm": float(translation),
                    "rotation_error_deg": float(rotation),
                    "inlier_count": int(pose.inliers.size),
                }
            )
        if (index + 1) % 8 == 0 or index + 1 == len(batch):
            print(f"paired {index + 1}/{len(batch)}", flush=True)
    payload = {
        "schema": "lafgs_v8_paired_feedback_map_evaluation",
        "version": 1,
        "status": "PASS",
        "view_role": manifest["view_role"],
        "uses_test_queries": False,
        "shared_frontend_extraction": True,
        "input": {
            "certified_manifest": str(args.certified_manifest.resolve()),
            "certified_manifest_sha256": sha256_file(args.certified_manifest),
            "maps": [
                {"name": name, "path": str(path.resolve()), "sha256": sha256_file(path)}
                for name, path in zip(names, paths)
            ],
        },
        "results": {
            name: {
                "metrics": _metrics(rows[name]),
                "rows": rows[name],
                **(
                    {}
                    if name == "baseline"
                    else {"paired_vs_baseline": _paired(rows["baseline"], rows[name])}
                ),
            }
            for name in names
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v["metrics"] for k, v in payload["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
