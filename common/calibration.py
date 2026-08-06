"""Mapping-only calibration for scale- and resolution-adaptive LaFGS runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import torch


REFERENCE_IMAGE_DIAGONAL_PX = math.hypot(1920.0, 1080.0)
REFERENCE_FOCAL_PX = 1672.028076171875
REFERENCE_EFFECTIVE_BASELINE_M = 2.06756077


def _quantile(values: torch.Tensor, value: float, fallback: float) -> float:
    values = torch.as_tensor(values).double().reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return float(fallback)
    return float(torch.quantile(values, float(value)))


def _queries(payload: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    return payload.get("queries", payload)


def _camera_centers(
    queries: Mapping[str, Mapping[str, Any]], names: list[str]
) -> torch.Tensor:
    centers = []
    for name in names:
        pose = torch.as_tensor(queries[name]["pose_w2c"]).double()
        centers.append(-(pose[:3, :3].T @ pose[:3, 3]))
    return torch.stack(centers)


def _effective_track_baselines(
    centers: torch.Tensor, track_payload: Mapping[str, Any]
) -> torch.Tensor:
    observations = track_payload["tracks"]
    track = torch.as_tensor(observations["track_index"]).long()
    query = torch.as_tensor(observations["query_index"]).long()
    if track.numel() < 2:
        return centers.new_empty(0)
    # The first-to-last span of each real co-visible track is insensitive to
    # video frame rate. Adjacent observation pairs collapse toward zero on
    # dense indoor sequences and are therefore not an effective geometry scale.
    key = track * (centers.shape[0] + 1) + query
    order = torch.argsort(key, stable=True)
    sorted_track, sorted_query = track[order], query[order]
    starts = torch.cat(
        (torch.ones(1, dtype=torch.bool), sorted_track[1:] != sorted_track[:-1])
    )
    ends = torch.cat(
        (sorted_track[1:] != sorted_track[:-1], torch.ones(1, dtype=torch.bool))
    )
    left, right = sorted_query[starts], sorted_query[ends]
    keep = left != right
    if not bool(keep.any()):
        return centers.new_empty(0)
    baseline = torch.linalg.norm(centers[right[keep]] - centers[left[keep]], dim=1)
    return baseline[baseline > 1e-9]


def _visible_track_depths(
    queries: Mapping[str, Mapping[str, Any]],
    names: list[str],
    track_payload: Mapping[str, Any],
) -> torch.Tensor:
    geometry = track_payload["track_geometry"]
    observations = track_payload["tracks"]
    track = torch.as_tensor(observations["track_index"]).long()
    query = torch.as_tensor(observations["query_index"]).long()
    triangulated = torch.as_tensor(geometry["triangulated"]).bool()
    keep = triangulated[track]
    if not bool(keep.any()):
        return torch.empty(0, dtype=torch.float64)
    track, query = track[keep], query[keep]
    xyz = torch.as_tensor(geometry["triangulated_xyz"]).double()[track]
    poses = torch.stack(
        [torch.as_tensor(queries[name]["pose_w2c"]).double() for name in names]
    )
    camera = torch.bmm(poses[query, :3, :3], xyz[:, :, None])[:, :, 0]
    camera += poses[query, :3, 3]
    depth = camera[:, 2]
    return depth[torch.isfinite(depth) & (depth > 0)]


@dataclass(frozen=True)
class MappingSceneStatistics:
    query_count: int
    image_diagonal_px: float
    focal_px: float
    valid_keypoints_p10: float
    valid_keypoints_median: float
    effective_baseline_m: float
    visible_depth_median_m: float
    camera_extent_diagonal_m: float
    track_count: int
    triangulated_track_count: int
    stable_track_reprojection_quantile: float
    stable_track_reprojection_p90_px: float


@dataclass(frozen=True)
class AdaptiveParameters:
    image_pixel_scale: float
    angular_pixel_scale: float
    # Kept as a compatibility alias for artifacts produced by adaptive V2.
    pixel_scale: float
    metric_scale: float
    covariance_scale: float
    kcs_radius_px: float
    positive_radius_px: float
    clean_radius_px: float
    negative_radius_px: float
    ransac_reprojection_px: float
    harm_radius_px: float
    semidense_local_radius_px: float
    semidense_sigma_px: float
    projected_neighbor_radius_px: float
    track_reprojection_median_px: float
    track_reprojection_p90_px: float
    track_covariance_trace_m2: float
    assignment_distance_m: float
    dependency_voxel_m: float
    base_voxel_m: float
    surface_point_plane_m: float
    surface_max_distance_m: float
    surface_group_voxel_m: float
    depth_residual_m: float
    evidence_depth_abs_tolerance_m: float
    matching_rows_target: int
    stage_a_steps: int
    metric_steps: int
    view_bin_count: int
    trajectory_bin_count: int
    task_translation_m: float
    task_rotation_deg: float


def derive_mapping_statistics(
    query_cache: Mapping[str, Any],
    track_payload: Mapping[str, Any] | None = None,
    *,
    track_residual_quantile: float = 0.975,
) -> MappingSceneStatistics:
    queries = _queries(query_cache)
    names = (
        list(track_payload["query_names"])
        if track_payload is not None
        else sorted(queries)
    )
    if not names:
        raise ValueError("mapping calibration requires at least one query")
    missing = set(names) - set(queries)
    if missing:
        raise ValueError(f"query cache misses mapping views: {sorted(missing)[:3]}")
    centers = _camera_centers(queries, names)
    diagonals, focals, keypoint_counts = [], [], []
    for name in names:
        record = queries[name]
        height, width = (int(value) for value in record["native_input_hw"])
        diagonals.append(math.hypot(width, height))
        K = torch.as_tensor(record["native_K"]).double()
        focals.append(math.sqrt(float(K[0, 0] * K[1, 1])))
        keypoint_counts.append(
            int(torch.as_tensor(record["native_keypoints"]).shape[0])
        )
    diagonal = torch.as_tensor(diagonals, dtype=torch.float64)
    focal = torch.as_tensor(focals, dtype=torch.float64)
    counts = torch.as_tensor(keypoint_counts, dtype=torch.float64)
    baseline = centers.new_empty(0)
    depths = centers.new_empty(0)
    track_count = triangulated_count = 0
    stable_track_reprojection = centers.new_empty(0)
    if track_payload is not None:
        baseline = _effective_track_baselines(centers, track_payload)
        depths = _visible_track_depths(queries, names, track_payload)
        triangulated = torch.as_tensor(
            track_payload["track_geometry"]["triangulated"]
        ).bool()
        track_count = int(triangulated.numel())
        triangulated_count = int(triangulated.sum())
        geometry = track_payload["track_geometry"]
        reprojection = geometry.get("triangulation_reprojection_p90_px")
        if reprojection is not None:
            stable = triangulated.clone()
            confidence = geometry.get("track_confidence_level")
            if confidence is not None:
                stable &= torch.as_tensor(confidence).long() >= 2
            stable_track_reprojection = torch.as_tensor(reprojection).double()[stable]
            stable_track_reprojection = stable_track_reprojection[
                torch.isfinite(stable_track_reprojection)
                & (stable_track_reprojection > 0)
            ]
    extent = torch.linalg.norm(centers.amax(0) - centers.amin(0))
    if baseline.numel() == 0 and centers.shape[0] > 1:
        fallback = torch.linalg.norm(centers[1:] - centers[:-1], dim=1)
        baseline = fallback[fallback > 1e-9]
    effective_baseline = _quantile(
        baseline, 0.5, REFERENCE_EFFECTIVE_BASELINE_M
    )
    if track_payload is None:
        # Before Track-First evidence exists, a dense video's adjacent-frame
        # displacement reflects frame rate. A small trajectory-extent floor is
        # a more stable proxy and is replaced by track spans after Stage A.
        effective_baseline = max(effective_baseline, float(extent) * 0.05)
    effective_baseline = max(effective_baseline, 1e-6)
    return MappingSceneStatistics(
        query_count=len(names),
        image_diagonal_px=_quantile(
            diagonal, 0.5, REFERENCE_IMAGE_DIAGONAL_PX
        ),
        focal_px=_quantile(focal, 0.5, 1.0),
        valid_keypoints_p10=_quantile(counts, 0.1, 1.0),
        valid_keypoints_median=_quantile(counts, 0.5, 1.0),
        effective_baseline_m=effective_baseline,
        visible_depth_median_m=_quantile(depths, 0.5, effective_baseline),
        camera_extent_diagonal_m=float(extent),
        track_count=track_count,
        triangulated_track_count=triangulated_count,
        stable_track_reprojection_quantile=float(track_residual_quantile),
        stable_track_reprojection_p90_px=_quantile(
            stable_track_reprojection, track_residual_quantile, 0.0
        ),
    )


def derive_adaptive_parameters(
    statistics: MappingSceneStatistics,
    policy: Mapping[str, Any] | None = None,
) -> AdaptiveParameters:
    """Resolve all dimensional thresholds from one scene-independent policy."""
    policy = {} if policy is None else dict(policy)
    reference_diagonal = float(
        policy.get("reference_image_diagonal_px", REFERENCE_IMAGE_DIAGONAL_PX)
    )
    reference_focal = float(
        policy.get("reference_focal_px", REFERENCE_FOCAL_PX)
    )
    reference_baseline = float(
        policy.get("reference_effective_baseline_m", REFERENCE_EFFECTIVE_BASELINE_M)
    )
    image_pixel_scale = statistics.image_diagonal_px / reference_diagonal
    # Reprojection, epipolar, and PnP thresholds express angular error.  Their
    # pixel realization therefore scales with focal length, not image extent.
    angular_pixel_scale = statistics.focal_px / reference_focal
    metric_scale = statistics.effective_baseline_m / reference_baseline
    covariance_scale = metric_scale * metric_scale
    target_fraction = float(policy.get("matching_rows_fraction", 0.04735))
    row_target = round(statistics.valid_keypoints_p10 * target_fraction)
    row_target = max(int(policy.get("matching_rows_minimum", 32)), row_target)
    row_target = min(int(policy.get("matching_rows_maximum", 192)), row_target)
    stage_epochs = float(policy.get("stage_a_query_epochs", 4.33))
    metric_epochs = float(policy.get("metric_query_epochs", 0.76))
    bins = round(
        math.sqrt(
            statistics.query_count
            / float(policy.get("queries_per_pose_bin_squared", 16.0))
        )
    )
    bins = max(int(policy.get("pose_bins_minimum", 2)), bins)
    bins = min(int(policy.get("pose_bins_maximum", 8)), bins)
    angular_ransac_px = max(2.0, 12.0 * angular_pixel_scale)
    track_residual_cap_px = float(
        policy.get("ransac_reprojection_maximum_px", 12.0)
    )
    track_residual_floor_px = min(
        track_residual_cap_px,
        statistics.stable_track_reprojection_p90_px,
    )
    ransac_reprojection_px = max(
        angular_ransac_px,
        track_residual_floor_px,
    )
    return AdaptiveParameters(
        image_pixel_scale=image_pixel_scale,
        angular_pixel_scale=angular_pixel_scale,
        pixel_scale=image_pixel_scale,
        metric_scale=metric_scale,
        covariance_scale=covariance_scale,
        kcs_radius_px=max(0.5, 1.0 * angular_pixel_scale),
        positive_radius_px=max(0.5, 2.0 * angular_pixel_scale),
        clean_radius_px=max(1.0, 4.0 * angular_pixel_scale),
        negative_radius_px=max(2.0, 8.0 * angular_pixel_scale),
        ransac_reprojection_px=ransac_reprojection_px,
        harm_radius_px=ransac_reprojection_px,
        semidense_local_radius_px=max(2.0, 8.0 * angular_pixel_scale),
        semidense_sigma_px=max(0.5, 2.0 * angular_pixel_scale),
        projected_neighbor_radius_px=max(8.0, 64.0 * angular_pixel_scale),
        track_reprojection_median_px=max(0.75, 3.0 * angular_pixel_scale),
        track_reprojection_p90_px=max(2.0, 15.0 * angular_pixel_scale),
        track_covariance_trace_m2=max(1e-8, 0.2 * covariance_scale),
        assignment_distance_m=max(1e-4, 0.2 * metric_scale),
        dependency_voxel_m=max(1e-4, 0.5 * metric_scale),
        base_voxel_m=max(1e-4, 1.0 * metric_scale),
        surface_point_plane_m=max(1e-4, 0.03 * metric_scale),
        surface_max_distance_m=max(1e-4, 0.15 * metric_scale),
        surface_group_voxel_m=max(1e-4, 0.25 * metric_scale),
        depth_residual_m=max(
            1e-4,
            0.15 * metric_scale,
            float(policy.get("depth_residual_fraction", 0.01))
            * statistics.visible_depth_median_m,
        ),
        evidence_depth_abs_tolerance_m=max(1e-4, 0.05 * metric_scale),
        matching_rows_target=row_target,
        stage_a_steps=max(1, math.ceil(stage_epochs * statistics.query_count)),
        metric_steps=max(1, math.ceil(metric_epochs * statistics.query_count)),
        view_bin_count=bins,
        trajectory_bin_count=bins,
        # These are task tolerances, not scene-size normalization constants.
        task_translation_m=float(policy.get("task_translation_m", 0.05)),
        task_rotation_deg=float(policy.get("task_rotation_deg", 5.0)),
    )


def calibrate_scene(
    query_cache_path: str | Path,
    track_payload_path: str | Path | None = None,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    query_cache = torch.load(
        Path(query_cache_path), map_location="cpu", weights_only=False
    )
    track_payload = (
        torch.load(Path(track_payload_path), map_location="cpu", weights_only=False)
        if track_payload_path is not None
        else None
    )
    policy = {} if policy is None else dict(policy)
    statistics = derive_mapping_statistics(
        query_cache,
        track_payload,
        track_residual_quantile=float(
            policy.get("ransac_track_residual_quantile", 0.975)
        ),
    )
    parameters = derive_adaptive_parameters(statistics, policy)
    return {
        "schema": "lafgs_mapping_only_scene_calibration",
        "version": 2,
        "statistics": asdict(statistics),
        "parameters": asdict(parameters),
        "policy": policy,
        "sources": {
            "query_cache": str(Path(query_cache_path).resolve()),
            "track_payload": (
                str(Path(track_payload_path).resolve())
                if track_payload_path is not None
                else None
            ),
            "uses_test_queries": False,
        },
    }
