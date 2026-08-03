"""Dataset-level evaluation for the minimal one-shot sparse runtime."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from data.datasets import CameraRecord, ColmapDataset
from evaluation.metrics import pose_error, summarize_pose_errors
from localization.localizer import SparseLocalizer


def evaluate_dataset(
    *,
    dataset: ColmapDataset,
    localizer: SparseLocalizer,
    cameras: Iterable[CameraRecord],
    output: str | Path | None = None,
) -> dict[str, Any]:
    rows = []
    rotation_errors = []
    translation_errors = []
    for camera in cameras:
        started = time.perf_counter()
        result = localizer.localize(
            dataset.load_image(camera),
            fov_x=camera.fov_x,
            fov_y=camera.fov_y,
            valid_mask=dataset.valid_mask(camera),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        rotation, translation = pose_error(result.pose.pose_w2c, camera.pose_w2c)
        rotation_errors.append(rotation)
        translation_errors.append(translation)
        rows.append(
            {
                "image_name": camera.image_name,
                "pose_w2c": result.pose.pose_w2c.tolist(),
                "gt_pose_w2c": camera.pose_w2c.tolist(),
                "rotation_error_deg": rotation,
                "translation_error_cm": translation,
                "keypoints": int(result.sparse_features.keypoints.shape[0]),
                "matches": int(result.matches.scores.numel()),
                "inliers": int(result.pose.inliers.size),
                "ransac_iterations": int(result.pose.diagnostics.get("iterations", 0)),
                "runtime_ms": elapsed_ms,
            }
        )
    summary = summarize_pose_errors(rotation_errors, translation_errors)
    summary["runtime_ms_mean"] = float(np.mean([row["runtime_ms"] for row in rows]))
    payload = {
        "schema": "lafgs_sparse_evaluation",
        "version": 1,
        "summary": summary,
        "queries": rows,
    }
    if output is not None:
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return payload
