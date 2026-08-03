"""Dataset-level evaluation for the minimal one-shot sparse runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from data.datasets import CameraRecord, ColmapDataset
from evaluation.metrics import pose_error, summarize_pose_errors
from localization.localizer import SparseLocalizer


def _gt_reprojection_diagnostics(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
    inliers: np.ndarray,
) -> dict[str, int]:
    points_2d = np.asarray(points_2d, dtype=np.float64)
    points_3d = np.asarray(points_3d, dtype=np.float64)
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if points_2d.shape != (points_3d.shape[0], 2):
        raise ValueError("2D and 3D correspondence rows do not align")
    camera_points = (
        pose_w2c[:3, :3] @ points_3d.T + pose_w2c[:3, 3:4]
    ).T
    valid = camera_points[:, 2] > 1e-8
    projected = np.full_like(points_2d, np.nan)
    projected_h = (intrinsic @ camera_points[valid].T).T
    projected[valid] = projected_h[:, :2] / projected_h[:, 2:3]
    errors = np.full(points_2d.shape[0], np.inf, dtype=np.float64)
    errors[valid] = np.linalg.norm(projected[valid] - points_2d[valid], axis=1)
    inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    inliers = inliers[(inliers >= 0) & (inliers < errors.size)]
    inlier_errors = errors[inliers]
    return {
        "raw_count": int(errors.size),
        "raw_correct_2px": int(np.count_nonzero(errors <= 2.0)),
        "raw_correct_4px": int(np.count_nonzero(errors <= 4.0)),
        "inlier_count": int(inlier_errors.size),
        "inlier_correct_2px": int(np.count_nonzero(inlier_errors <= 2.0)),
        "inlier_correct_4px": int(np.count_nonzero(inlier_errors <= 4.0)),
    }


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * float(numerator) / max(int(denominator), 1)


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
        result = localizer.localize(
            dataset.load_image(camera),
            fov_x=camera.fov_x,
            fov_y=camera.fov_y,
            valid_mask=dataset.valid_mask(camera),
        )
        rotation, translation = pose_error(result.pose.pose_w2c, camera.pose_w2c)
        points_2d = (
            result.sparse_features.keypoints[result.matches.keypoint_indices]
            .cpu()
            .numpy()
            + 0.5
        )
        points_3d = (
            localizer.anchor_xyz[result.matches.anchor_indices].cpu().numpy()
        )
        cleanliness = _gt_reprojection_diagnostics(
            points_2d,
            points_3d,
            camera.pose_w2c,
            result.intrinsic,
            result.pose.inliers,
        )
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
                **cleanliness,
                "raw_gt_precision_2px_percent": _percent(
                    cleanliness["raw_correct_2px"], cleanliness["raw_count"]
                ),
                "raw_gt_precision_4px_percent": _percent(
                    cleanliness["raw_correct_4px"], cleanliness["raw_count"]
                ),
                "inlier_gt_precision_2px_percent": _percent(
                    cleanliness["inlier_correct_2px"], cleanliness["inlier_count"]
                ),
                "inlier_gt_precision_4px_percent": _percent(
                    cleanliness["inlier_correct_4px"], cleanliness["inlier_count"]
                ),
                **result.runtime_ms,
                "runtime_ms": result.runtime_ms["total_ms"],
            }
        )
    summary = summarize_pose_errors(rotation_errors, translation_errors)
    raw_count = sum(row["raw_count"] for row in rows)
    inlier_count = sum(row["inlier_count"] for row in rows)
    for threshold in (2, 4):
        raw = _percent(
            sum(row[f"raw_correct_{threshold}px"] for row in rows), raw_count
        )
        inlier = _percent(
            sum(row[f"inlier_correct_{threshold}px"] for row in rows), inlier_count
        )
        summary[f"raw_gt_precision_{threshold}px_percent"] = raw
        summary[f"matcher_raw_gt_precision_{threshold}px_percent"] = raw
        summary[f"inlier_gt_precision_{threshold}px_percent"] = inlier
    summary["solver_inlier_ratio_percent"] = _percent(inlier_count, raw_count)
    summary["mean_hypotheses"] = float(
        np.mean([row["ransac_iterations"] for row in rows])
    )
    summary["catastrophic_100cm_count"] = int(
        np.count_nonzero(np.asarray(translation_errors) >= 100.0)
    )
    for name in ("frontend_ms", "matching_ms", "ransac_ms", "total_ms"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        summary[f"{name}_mean"] = float(np.mean(values))
        summary[f"{name}_p50"] = float(np.percentile(values, 50))
        summary[f"{name}_p90"] = float(np.percentile(values, 90))
    summary["runtime_ms_mean"] = summary["total_ms_mean"]
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
