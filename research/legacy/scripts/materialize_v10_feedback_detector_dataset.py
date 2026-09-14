#!/usr/bin/env python3
"""Build leakage-safe query-detector targets from frozen M0 feedback outcomes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.scene_detector_supervision import (
    build_feedback_match_heatmap,
    build_pose_contribution_weights,
)


SCHEMA = "lafgs_v11_pose_contribution_detector_dataset"


def _atomic_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _project_top1_error(
    source: dict, top1_xyz: torch.Tensor, keypoints: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = torch.as_tensor(top1_xyz).float()
    pose = torch.as_tensor(source["pose_w2c"]).float()
    intrinsic = torch.as_tensor(source["intrinsics"]).float()
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    projected = camera @ intrinsic.T
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    error = torch.linalg.norm(uv - (torch.as_tensor(keypoints).float() + 0.5), dim=1)
    error[camera[:, 2] <= 1e-6] = torch.inf
    return error, camera[:, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-modulus", type=int, default=5)
    args = parser.parse_args()
    if args.validation_modulus < 2:
        parser.error("validation modulus must be at least two")
    anchor_map = torch.load(args.anchor_map, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(anchor_map["anchor_xyz"]).float()
    map_sha = sha256_file(args.anchor_map)
    args.output.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "validation": 0}
    class_counts = {"train": {"positive": 0, "negative": 0}, "validation": {"positive": 0, "negative": 0}}
    seen_queries = set()
    records = []
    for manifest_path in args.observer_manifests:
        manifest = json.loads(manifest_path.read_text())
        if not (
            manifest.get("schema") == "lafgs_v9_no_loo_causal_feedback_batch"
            and manifest.get("loo_used") is False
            and manifest.get("uses_test_queries") is False
            and manifest["input"]["map_sha256"] == map_sha
        ):
            raise ValueError("observer manifest violates V10 detector lineage")
        for item in manifest["records"]:
            record_path = Path(item["path"])
            if sha256_file(record_path) != item["sha256"]:
                raise ValueError("observer record SHA256 differs")
            observer = torch.load(record_path, map_location="cpu", weights_only=False)
            if observer.get("loo_used") is not False:
                raise ValueError("feedback detector cannot consume LOO evidence")
            source_path = Path(observer["source_record"])
            if sha256_file(source_path) != observer["source_record_sha256"]:
                raise ValueError("certified source record SHA256 differs")
            source = torch.load(source_path, map_location="cpu", weights_only=False)
            if source["certificate"]["decision"] != "ACCEPT":
                continue
            query_index = int(observer["query_index"])
            if query_index in seen_queries:
                raise ValueError("feedback queries overlap")
            seen_queries.add(query_index)
            top1 = torch.as_tensor(observer["topk_anchor_rows"]).long()[:, 0]
            source_query_rows = torch.as_tensor(observer["source_query_rows"]).long()
            keypoints = torch.as_tensor(source["keypoints"])[source_query_rows]
            if bool(((top1 < 0) | (top1 >= xyz.shape[0])).any()):
                raise ValueError("feedback Top-1 row is outside M0")
            error, camera_depth = _project_top1_error(source, xyz[top1], keypoints)
            labels = build_feedback_match_heatmap(
                image_hw=tuple(torch.as_tensor(source["image_hw"]).long().tolist()),
                keypoints=keypoints,
                reprojection_error_px=error,
                row_valid=torch.ones(keypoints.shape[0], dtype=torch.bool),
                row_uncertain=torch.zeros(keypoints.shape[0], dtype=torch.bool),
            ).cpu()
            topk_scores = torch.as_tensor(observer["topk_scores"]).float()
            margin = topk_scores[:, 0] - topk_scores[:, 1]
            contribution_weights = build_pose_contribution_weights(
                labels=labels,
                keypoints=keypoints,
                image_hw=tuple(torch.as_tensor(source["image_hw"]).long().tolist()),
                reprojection_error_px=error,
                camera_depth=camera_depth,
                match_margin=margin,
            ).cpu()
            split = "validation" if query_index % args.validation_modulus == 0 else "train"
            output_path = args.output / f"{split}_{query_index:04d}.pt"
            payload = {
                "schema": SCHEMA,
                "version": 1,
                "split": split,
                "query_index": query_index,
                "pose_family_id": int(observer["pose_family_id"]),
                "source_record": str(source_path.resolve()),
                "source_record_sha256": sha256_file(source_path),
                "labels": labels,
                "contribution_weights": contribution_weights,
                "positive_count": int((labels == 1).sum()),
                "negative_count": int((labels == 0).sum()),
                "loo_used": False,
                "uses_test_rgb": False,
                "map_sha256": map_sha,
            }
            _atomic_save(payload, output_path)
            counts[split] += 1
            class_counts[split]["positive"] += payload["positive_count"]
            class_counts[split]["negative"] += payload["negative_count"]
            records.append(str(output_path.resolve()))
    report = {
        "schema": f"{SCHEMA}_manifest",
        "version": 1,
        "loo_used": False,
        "uses_test_rgb": False,
        "uses_real_training_rgb": False,
        "anchor_map": str(args.anchor_map.resolve()),
        "anchor_map_sha256": map_sha,
        "observer_manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.observer_manifests
        ],
        "split_policy": f"query_index_mod_{args.validation_modulus}_equals_zero",
        "counts": counts,
        "class_counts": class_counts,
        "records": records,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
