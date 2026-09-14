"""Pose metrics used by LaFGS evaluation."""

from __future__ import annotations

import math

import cv2
import numpy as np


def pose_error(
    predicted_w2c: np.ndarray, ground_truth_w2c: np.ndarray
) -> tuple[float, float]:
    predicted_w2c = np.asarray(predicted_w2c)
    ground_truth_w2c = np.asarray(ground_truth_w2c)
    relative_rotation = ground_truth_w2c[:3, :3] @ predicted_w2c[:3, :3].T
    rotation_degrees = float(
        np.linalg.norm(cv2.Rodrigues(relative_rotation)[0]) * 180.0 / math.pi
    )
    predicted_center = np.linalg.inv(predicted_w2c)[:3, 3]
    ground_truth_center = np.linalg.inv(ground_truth_w2c)[:3, 3]
    translation_cm = float(
        np.linalg.norm(predicted_center - ground_truth_center) * 100.0
    )
    return rotation_degrees, translation_cm


def summarize_pose_errors(
    rotation_degrees: list[float], translation_cm: list[float]
) -> dict[str, float | int]:
    rotation = np.asarray(rotation_degrees, dtype=np.float64)
    translation = np.asarray(translation_cm, dtype=np.float64)
    if rotation.shape != translation.shape or rotation.size == 0:
        raise ValueError("rotation and translation errors must be nonempty and aligned")
    return {
        "query_count": int(rotation.size),
        "median_te_cm": float(np.median(translation)),
        "mean_te_cm": float(np.mean(translation)),
        "p90_te_cm": float(np.percentile(translation, 90)),
        "median_ae_deg": float(np.median(rotation)),
        "mean_ae_deg": float(np.mean(rotation)),
        "p90_ae_deg": float(np.percentile(rotation, 90)),
        "recall_2cm_2deg_percent": float(
            100.0 * np.mean((translation < 2.0) & (rotation < 2.0))
        ),
        "recall_5cm_5deg_percent": float(
            100.0 * np.mean((translation < 5.0) & (rotation < 5.0))
        ),
    }
