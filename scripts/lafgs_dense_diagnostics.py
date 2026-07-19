"""Pure diagnostic helpers for standalone LaFGS dense-refinement experiments."""

import numpy as np


_TIGHT_PRECISION_THRESHOLDS = (
    (0.25, "0p25px"),
    (0.50, "0p5px"),
    (0.75, "0p75px"),
    (1.00, "1px"),
)


def _project_world_points(points3d, intrinsic, pose_w2c):
    """Project world points with the evaluator's pinhole convention."""
    camera_points = points3d @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-6)
    projected = np.full((points3d.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        normalized = camera_points[valid, :2] / camera_points[valid, 2:3]
        projected[valid] = normalized @ intrinsic[:2, :2].T + intrinsic[:2, 2]
    return projected, valid


def gt_reprojection_diagnostics(
    query_xy, points3d, intrinsic, gt_pose_w2c, inliers, scores=None
):
    """Measure correspondence cleanliness without feeding it to PnP."""
    query_xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    pose = np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4)
    if query_xy.shape[0] != points3d.shape[0]:
        raise ValueError("query_xy and points3d must have the same count")

    projected, _ = _project_world_points(points3d, intrinsic, pose)
    residual = (query_xy + 0.5) - projected
    errors = np.linalg.norm(residual, axis=1)

    def summarize(prefix, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        indices = indices[(indices >= 0) & (indices < query_xy.shape[0])]
        selected = errors[indices]
        finite = selected[np.isfinite(selected)]
        result = {
            f"{prefix}_count": int(indices.size),
            f"{prefix}_gt_projected_ratio": float(np.isfinite(selected).mean())
            if selected.size
            else 0.0,
        }
        if finite.size:
            finite_residual = residual[indices][np.isfinite(selected)]
            result.update(
                {
                    f"{prefix}_gt_reproj_px_mean": float(finite.mean()),
                    f"{prefix}_gt_reproj_px_median": float(np.median(finite)),
                    f"{prefix}_gt_reproj_px_p95": float(np.percentile(finite, 95)),
                    f"{prefix}_gt_precision_2px": float(np.mean(finite <= 2.0)),
                    f"{prefix}_gt_precision_4px": float(np.mean(finite <= 4.0)),
                    f"{prefix}_gt_precision_6px": float(np.mean(finite <= 6.0)),
                    f"{prefix}_gt_precision_12px": float(np.mean(finite <= 12.0)),
                    f"{prefix}_gt_residual_x_mean_px": float(finite_residual[:, 0].mean()),
                    f"{prefix}_gt_residual_y_mean_px": float(finite_residual[:, 1].mean()),
                    f"{prefix}_gt_residual_x_median_px": float(
                        np.median(finite_residual[:, 0])
                    ),
                    f"{prefix}_gt_residual_y_median_px": float(
                        np.median(finite_residual[:, 1])
                    ),
                }
            )
            for threshold, label in _TIGHT_PRECISION_THRESHOLDS:
                result[f"{prefix}_gt_precision_{label}"] = float(
                    np.mean(finite <= threshold)
                )
        else:
            for suffix in (
                "gt_reproj_px_mean",
                "gt_reproj_px_median",
                "gt_reproj_px_p95",
                "gt_precision_2px",
                "gt_precision_4px",
                "gt_precision_6px",
                "gt_precision_12px",
                *(
                    f"gt_precision_{label}"
                    for _, label in _TIGHT_PRECISION_THRESHOLDS
                ),
                "gt_residual_x_mean_px",
                "gt_residual_y_mean_px",
                "gt_residual_x_median_px",
                "gt_residual_y_median_px",
            ):
                result[f"{prefix}_{suffix}"] = None
        return result

    result = summarize("dense_all", np.arange(query_xy.shape[0], dtype=np.int64))
    result.update(summarize("dense_ransac_inlier", inliers))
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != query_xy.shape[0]:
            raise ValueError("scores must have one value per correspondence")
        finite_scores = np.isfinite(scores)
        result["dense_score_finite_ratio"] = float(finite_scores.mean())
        if finite_scores.any():
            ranking = np.argsort(-np.where(finite_scores, scores, -np.inf), kind="stable")
            for fraction in (0.05, 0.10, 0.25):
                count = min(
                    query_xy.shape[0], max(1, int(np.ceil(query_xy.shape[0] * fraction)))
                )
                selected = errors[ranking[:count]]
                selected = selected[np.isfinite(selected)]
                prefix = f"dense_score_top_{int(fraction * 100):02d}pct"
                result[f"{prefix}_count"] = int(selected.size)
                result[f"{prefix}_gt_precision_2px"] = (
                    float(np.mean(selected <= 2.0)) if selected.size else 0.0
                )
                result[f"{prefix}_gt_precision_4px"] = (
                    float(np.mean(selected <= 4.0)) if selected.size else 0.0
                )
                for threshold, label in _TIGHT_PRECISION_THRESHOLDS:
                    result[f"{prefix}_gt_precision_{label}"] = (
                        float(np.mean(selected <= threshold)) if selected.size else 0.0
                    )
                result[f"{prefix}_gt_reproj_px_median"] = (
                    float(np.median(selected)) if selected.size else None
                )
    return result


def candidate_displacement_diagnostics(
    query_xy,
    rendered_xy,
    points3d,
    intrinsic,
    gt_pose_w2c,
):
    """Measure whether local matching responds correctly to a seed-pose shift.

    ``gt_shift`` is the feature-grid displacement that would align a rendered
    source point to the GT image.  ``predicted_shift`` is the displacement
    selected by the final local candidate matcher.  A value near one for the
    response gain means the candidate graph compensates the seed error; zero
    means it remains pinned to the rendered cell.  This is diagnostic-only and
    never enters the pose solve or training loss.
    """
    query_xy = np.asarray(query_xy, dtype=np.float64).reshape(-1, 2)
    rendered_xy = np.asarray(rendered_xy, dtype=np.float64).reshape(-1, 2)
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    pose = np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4)
    if query_xy.shape != rendered_xy.shape or query_xy.shape[0] != points3d.shape[0]:
        raise ValueError("query_xy, rendered_xy, and points3d must share a count")

    projected, _ = _project_world_points(points3d, intrinsic, pose)
    gt_target_xy = projected - 0.5
    gt_shift = gt_target_xy - rendered_xy
    predicted_shift = query_xy - rendered_xy
    residual = predicted_shift - gt_shift
    finite = (
        np.isfinite(gt_shift).all(axis=1)
        & np.isfinite(predicted_shift).all(axis=1)
        & np.isfinite(residual).all(axis=1)
    )
    result = {
        "candidate_shift_count": int(query_xy.shape[0]),
        "candidate_shift_gt_projected_ratio": float(finite.mean())
        if finite.size
        else 0.0,
    }
    if not np.any(finite):
        result.update(
            {
                "candidate_shift_response_gain": None,
                "candidate_shift_response_gain_x": None,
                "candidate_shift_response_gain_y": None,
                "candidate_shift_direction_cosine_mean": None,
                "candidate_shift_direction_positive_fraction": None,
                "candidate_shift_error_px_median": None,
                "candidate_shift_error_px_mean": None,
            }
        )
        return result

    gt_shift = gt_shift[finite]
    predicted_shift = predicted_shift[finite]
    residual = residual[finite]
    gt_norm_sq = np.sum(np.square(gt_shift))
    response_gain = (
        float(np.sum(predicted_shift * gt_shift) / gt_norm_sq)
        if gt_norm_sq > 1e-12
        else None
    )
    result.update(
        {
            "candidate_shift_gt_norm_px_mean": float(np.linalg.norm(gt_shift, axis=1).mean()),
            "candidate_shift_gt_norm_px_median": float(
                np.median(np.linalg.norm(gt_shift, axis=1))
            ),
            "candidate_shift_pred_norm_px_mean": float(
                np.linalg.norm(predicted_shift, axis=1).mean()
            ),
            "candidate_shift_pred_norm_px_median": float(
                np.median(np.linalg.norm(predicted_shift, axis=1))
            ),
            "candidate_shift_gt_x_mean_px": float(gt_shift[:, 0].mean()),
            "candidate_shift_gt_y_mean_px": float(gt_shift[:, 1].mean()),
            "candidate_shift_pred_x_mean_px": float(predicted_shift[:, 0].mean()),
            "candidate_shift_pred_y_mean_px": float(predicted_shift[:, 1].mean()),
            "candidate_shift_response_gain": response_gain,
            "candidate_shift_response_gain_x": _axis_response_gain(
                predicted_shift[:, 0], gt_shift[:, 0]
            ),
            "candidate_shift_response_gain_y": _axis_response_gain(
                predicted_shift[:, 1], gt_shift[:, 1]
            ),
            "candidate_shift_error_px_mean": float(
                np.linalg.norm(residual, axis=1).mean()
            ),
            "candidate_shift_error_px_median": float(
                np.median(np.linalg.norm(residual, axis=1))
            ),
        }
    )
    gt_norm = np.linalg.norm(gt_shift, axis=1)
    pred_norm = np.linalg.norm(predicted_shift, axis=1)
    directional = (gt_norm > 1e-6) & (pred_norm > 1e-6)
    if np.any(directional):
        cosine = (
            np.sum(predicted_shift[directional] * gt_shift[directional], axis=1)
            / (pred_norm[directional] * gt_norm[directional])
        )
        result["candidate_shift_direction_cosine_mean"] = float(cosine.mean())
        result["candidate_shift_direction_positive_fraction"] = float(
            np.mean(cosine > 0.0)
        )
    else:
        result["candidate_shift_direction_cosine_mean"] = None
        result["candidate_shift_direction_positive_fraction"] = None
    for threshold, label in _TIGHT_PRECISION_THRESHOLDS:
        result[f"candidate_shift_precision_{label}"] = float(
            np.mean(np.linalg.norm(residual, axis=1) <= threshold)
        )
    return result


def _axis_response_gain(predicted, target):
    denominator = float(np.sum(np.square(target)))
    if denominator <= 1e-12:
        return None
    return float(np.sum(predicted * target) / denominator)


def gt_local_basin_diagnostics(
    rendered_xy,
    points3d,
    intrinsic,
    gt_pose_w2c,
    radius_px,
):
    """Measure whether GT projection lies inside a seed-pose local window.

    This is diagnostics only.  It separates an insufficient local search basin
    from descriptor ambiguity: a refinement cannot recover a correct match if
    the source point's GT projection is outside the final candidate window.
    Coordinates follow the evaluator convention where an integer feature index
    is lifted and passed to PnP at its physical ``index + 0.5`` cell center.
    """
    rendered_xy = np.asarray(rendered_xy, dtype=np.float64).reshape(-1, 2)
    points3d = np.asarray(points3d, dtype=np.float64).reshape(-1, 3)
    intrinsic = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3)
    pose = np.asarray(gt_pose_w2c, dtype=np.float64).reshape(4, 4)
    if rendered_xy.shape[0] != points3d.shape[0]:
        raise ValueError("rendered_xy and points3d must have the same count")
    radius_px = float(radius_px)
    if radius_px < 0.0:
        raise ValueError("radius_px must be non-negative")

    camera_points = points3d @ pose[:3, :3].T + pose[:3, 3]
    valid = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-6)
    projected = np.full_like(rendered_xy, np.nan)
    if np.any(valid):
        normalized = camera_points[valid, :2] / camera_points[valid, 2:3]
        projected[valid] = normalized @ intrinsic[:2, :2].T + intrinsic[:2, 2]

    # Convert the projected physical pixel center into the feature index used
    # to construct the evaluator's integer local candidate window.
    offset = (projected - 0.5) - rendered_xy
    finite = np.isfinite(offset).all(axis=1)
    selected = offset[finite]
    result = {
        "seed_local_basin_count": int(rendered_xy.shape[0]),
        "seed_local_basin_gt_projected_ratio": float(finite.mean())
        if finite.size
        else 0.0,
        "seed_local_basin_radius_px": radius_px,
    }
    if selected.size == 0:
        result.update(
            {
                "seed_local_basin_square_coverage": 0.0,
                "seed_local_basin_offset_px_mean": None,
                "seed_local_basin_offset_px_median": None,
                "seed_local_basin_offset_px_p95": None,
                "seed_local_basin_offset_x_mean_px": None,
                "seed_local_basin_offset_y_mean_px": None,
            }
        )
        return result

    distance = np.linalg.norm(selected, axis=1)
    # A continuous target has a representable integer candidate if it is no
    # more than half a cell beyond a window edge in each axis.
    coverage = np.all(np.abs(selected) <= radius_px + 0.5, axis=1)
    result.update(
        {
            "seed_local_basin_square_coverage": float(np.mean(coverage)),
            "seed_local_basin_offset_px_mean": float(distance.mean()),
            "seed_local_basin_offset_px_median": float(np.median(distance)),
            "seed_local_basin_offset_px_p95": float(np.percentile(distance, 95)),
            "seed_local_basin_offset_x_mean_px": float(selected[:, 0].mean()),
            "seed_local_basin_offset_y_mean_px": float(selected[:, 1].mean()),
        }
    )
    return result
