#!/usr/bin/env python3
"""Evaluate one immutable map on a fixed non-test certified render batch."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from evaluation.evaluator import pose_error
from localization.localizer import SparseLocalizer


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
    parser.add_argument("--certified-manifest", type=Path, required=True)
    parser.add_argument(
        "--view-role",
        choices=("confirmation_query", "feedback_query"),
        default="confirmation_query",
    )
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.certified_manifest.read_text())
    if not (
        manifest.get("view_role") == args.view_role
        and manifest.get("uses_test_queries") is False
    ):
        raise ValueError("evaluation requires a frozen non-test certified batch")
    batch_records = [
        item for item in manifest["records"] if item.get("decision") == "ACCEPT"
    ]
    if not batch_records:
        raise ValueError("evaluation has no ACCEPT certified rows")
    if args.view_role == "confirmation_query" and len(batch_records) != len(
        manifest["records"]
    ):
        raise ValueError("formal confirmation batch must contain only ACCEPT rows")
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
    for index, item in enumerate(batch_records):
        record_path = Path(item["path"])
        if sha256_file(record_path) != item["sha256"]:
            raise ValueError("certified render record SHA mismatch")
        record = torch.load(record_path, map_location="cpu", weights_only=False)
        rgb = record["rgb_float16"].float()
        intrinsic = torch.as_tensor(record["intrinsics"]).float()
        height, width = rgb.shape[-2:]
        fov_x = 2.0 * math.atan(width / (2.0 * float(intrinsic[0, 0])))
        fov_y = 2.0 * math.atan(height / (2.0 * float(intrinsic[1, 1])))
        result = localizer.localize(
            rgb, fov_x=fov_x, fov_y=fov_y, valid_mask=None
        )
        rotation, translation = pose_error(
            result.pose.pose_w2c, torch.as_tensor(record["pose_w2c"]).numpy()
        )
        rows.append(
            {
                "query_index": int(record["query_index"]),
                "translation_error_cm": float(translation),
                "rotation_error_deg": float(rotation),
                "keypoint_count": int(result.sparse_features.keypoints.shape[0]),
                "inlier_count": int(result.pose.inliers.size),
            }
        )
        if (index + 1) % 16 == 0 or index + 1 == len(batch_records):
            print(
                f"{args.view_role} {index + 1}/{len(batch_records)}",
                flush=True,
            )
    payload = {
        "schema": "lafgs_v7_anchor_contamination_confirmation_ablation",
        "version": 1,
        "status": "PASS",
        "uses_test_queries": False,
        "map_mutation_count": 0,
        "threshold_tuning_from_results": False,
        "view_role": args.view_role,
        "control_selection_only": args.view_role == "feedback_query",
        "input": {
            "certified_manifest": str(args.certified_manifest.resolve()),
            "certified_manifest_sha256": sha256_file(args.certified_manifest),
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
