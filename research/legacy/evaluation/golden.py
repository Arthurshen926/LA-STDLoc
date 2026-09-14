"""Compact golden sparse-localization fixtures and parity checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common.hashing import sha256_file


QUERY_FIELDS = (
    "keypoint_xy",
    "keypoint_detector_score",
    "topk_landmark_idx",
    "topk_scores",
    "matcher_raw_keypoint_idx",
    "matcher_raw_landmark_idx",
    "matcher_raw_scores",
    "hard_post_keypoint_idx",
    "hard_post_landmark_idx",
    "hard_post_scores",
    "hard_post_inliers",
    "gt_pose_w2c",
    "pred_pose_w2c",
    "K",
    "width",
    "height",
    "reprojection_error",
    "confidence",
    "max_iterations",
    "min_iterations",
    "ransac_seed",
)


def pack_oracle_dump(source: str | Path, output: str | Path) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank_source = source / "landmark_bank.npz"
    if not bank_source.is_file():
        raise FileNotFoundError(bank_source)
    with np.load(bank_source, allow_pickle=False) as bank:
        np.savez_compressed(
            output / "landmark_bank.npz",
            **{name: bank[name] for name in bank.files},
        )

    records = []
    for index, path in enumerate(sorted(source.glob("query_*.npz"))):
        with np.load(path, allow_pickle=False) as payload:
            missing = [name for name in QUERY_FIELDS if name not in payload]
            if missing:
                raise ValueError(f"{path.name} misses fields: {missing}")
            name = f"query_{index:04d}.npz"
            values = {field: payload[field] for field in QUERY_FIELDS}
            values["topk_landmark_idx"] = values["topk_landmark_idx"][:, :1]
            values["topk_scores"] = values["topk_scores"][:, :1]
            np.savez_compressed(
                output / name,
                image_name=payload["image_name"],
                **values,
            )
            records.append(
                {
                    "image_name": str(payload["image_name"].item()),
                    "file": name,
                    "sha256": sha256_file(output / name),
                    "keypoint_count": int(payload["keypoint_xy"].shape[0]),
                    "match_count": int(payload["hard_post_keypoint_idx"].shape[0]),
                    "inlier_count": int(payload["hard_post_inliers"].shape[0]),
                }
            )
    manifest = {
        "schema": "lafgs_sparse_golden_fixture",
        "version": 1,
        "query_count": len(records),
        "landmark_bank_sha256": sha256_file(output / "landmark_bank.npz"),
        "queries": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_fixture(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != "lafgs_sparse_golden_fixture":
        raise ValueError("unsupported golden fixture")
    if sha256_file(path / "landmark_bank.npz") != manifest["landmark_bank_sha256"]:
        raise ValueError("golden landmark bank hash mismatch")
    for record in manifest["queries"]:
        if sha256_file(path / record["file"]) != record["sha256"]:
            raise ValueError(f"golden query hash mismatch: {record['file']}")
    return manifest
