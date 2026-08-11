"""Rebuild Track-First geometry with frozen-prior weak-axis support."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping

import torch

from evidence.triangulation import robust_triangulate_associations


def _observation_geometry(payload: Mapping, query_payload: Mapping):
    cache = query_payload.get("queries", query_payload)
    names = list(payload["query_names"])
    tracks = payload["tracks"]
    query_index = torch.as_tensor(tracks["query_index"]).long()
    keypoint_index = torch.as_tensor(tracks["keypoint_index"]).long()
    count = int(query_index.numel())
    uv = torch.empty((count, 2), dtype=torch.float32)
    depth = torch.empty(count, dtype=torch.float32)
    camera_K = torch.stack(
        [torch.as_tensor(cache[name]["native_K"]).float() for name in names]
    )
    pose_w2c = torch.stack(
        [torch.as_tensor(cache[name]["pose_w2c"]).float() for name in names]
    )
    order = torch.argsort(query_index, stable=True)
    ordered_query = query_index[order]
    unique, counts = torch.unique_consecutive(ordered_query, return_counts=True)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    for group, query in enumerate(unique.tolist()):
        rows = order[int(offsets[group]) : int(offsets[group + 1])]
        record = cache[names[query]]
        local = keypoint_index[rows]
        keypoints = torch.as_tensor(record["native_keypoints"]).float()[local]
        uv[rows] = keypoints + float(record.get("pixel_center_offset", 0.5))
        native_depth = torch.as_tensor(record["native_depth"]).float()
        x = keypoints[:, 0].round().long().clamp(0, native_depth.shape[1] - 1)
        y = keypoints[:, 1].round().long().clamp(0, native_depth.shape[0] - 1)
        depth[rows] = native_depth[y, x]
    return uv, depth, camera_K, pose_w2c


def rebuild_surface_supported_track_payload(
    *,
    payload: Mapping,
    query_payload: Mapping,
    parameters: Mapping,
    maximum_weak_information_ratio: float = 0.25,
    maximum_correction_depth_sigmas: float = 4.0,
    minimum_depth_improvement_fraction: float = 0.10,
    maximum_reprojection_increase_px: float = 0.05,
) -> tuple[dict, dict]:
    """Re-triangulate an immutable track graph without re-running matching."""
    uv, depth, camera_K, pose_w2c = _observation_geometry(payload, query_payload)
    tracks = payload["tracks"]
    old_geometry = payload["track_geometry"]
    depth_sigma = float(parameters["depth_residual_m"])
    geometry = robust_triangulate_associations(
        landmark_count=int(torch.as_tensor(old_geometry["triangulated"]).numel()),
        landmark_index=tracks["track_index"],
        query_index=tracks["query_index"],
        uv=uv,
        confidence=tracks["confidence"],
        camera_K=camera_K,
        pose_w2c=pose_w2c,
        query_bin=payload["query_bins"],
        rendered_depth=depth,
        maximum_observations_per_landmark=32,
        minimum_views=3,
        minimum_view_bins=2,
        huber_delta_px=2.0,
        iterations=3,
        minimum_parallax_deg=1.0,
        parallax_quantile=0.75,
        maximum_reprojection_px=float(parameters["positive_radius_px"]),
        maximum_condition_number=1e6,
        maximum_covariance_trace_m2=(
            0.01 * float(parameters["covariance_scale"])
        ),
        maximum_rendered_depth_residual_m=depth_sigma,
        minimum_rendered_depth_observations=2,
        surface_support_enabled=True,
        surface_support_huber_m=depth_sigma,
        surface_support_maximum_correction_m=(
            depth_sigma * float(maximum_correction_depth_sigmas)
        ),
        surface_support_maximum_weak_information_ratio=(
            maximum_weak_information_ratio
        ),
        surface_support_minimum_depth_improvement_fraction=(
            minimum_depth_improvement_fraction
        ),
        surface_support_maximum_reprojection_increase_px=(
            maximum_reprojection_increase_px
        ),
        surface_support_covariance_sigma_m=depth_sigma,
    )
    geometry["track_confidence_level"] = torch.as_tensor(
        old_geometry["track_confidence_level"]
    ).clone()
    # Preserve the immutable image-only estimate as a separate topology signal.
    # Surface support may promote a track into the candidate reserve, but it
    # cannot retroactively make that track part of the image-stable core.
    geometry["triangulation_image_only_xyz"] = torch.as_tensor(
        old_geometry["triangulated_xyz"]
    ).clone()
    geometry["triangulation_image_only_covariance_trace"] = torch.as_tensor(
        old_geometry["triangulation_covariance_trace"]
    ).clone()
    geometry["triangulation_image_only_covariance_matrix"] = torch.as_tensor(
        old_geometry["triangulation_covariance_matrix"]
    ).clone()
    geometry["triangulation_image_only_reprojection_median_px"] = torch.as_tensor(
        old_geometry["triangulation_reprojection_median_px"]
    ).clone()
    geometry["triangulation_image_only_reprojection_p90_px"] = torch.as_tensor(
        old_geometry["triangulation_reprojection_p90_px"]
    ).clone()
    revised = deepcopy(dict(payload))
    revised["version"] = max(int(payload.get("version", 1)), 2)
    revised["track_geometry"] = geometry
    supported = torch.as_tensor(geometry["triangulation_surface_supported"]).bool()
    old_covariance = torch.as_tensor(
        old_geometry["triangulation_covariance_trace"]
    ).float()
    new_covariance = torch.as_tensor(
        geometry["triangulation_covariance_trace"]
    ).float()
    old_threshold = float(parameters["track_covariance_trace_m2"])
    old_eligible = (
        torch.as_tensor(old_geometry["triangulated"]).bool()
        & (old_covariance <= old_threshold)
    )
    new_eligible = (
        torch.as_tensor(geometry["triangulated"]).bool()
        & (new_covariance <= old_threshold)
    )
    report = {
        "schema": "lafgs_surface_supported_track_geometry",
        "version": 1,
        "track_count": int(supported.numel()),
        "surface_supported_count": int(supported.sum()),
        "surface_supported_rate": float(supported.float().mean()),
        "covariance_eligible_before": int(old_eligible.sum()),
        "covariance_eligible_after": int(new_eligible.sum()),
        "covariance_eligible_gain": int(new_eligible.sum() - old_eligible.sum()),
        "high_confidence_before": int(
            torch.as_tensor(old_geometry["triangulation_high_confidence"]).sum()
        ),
        "high_confidence_after": int(
            torch.as_tensor(geometry["triangulation_high_confidence"]).sum()
        ),
        "correction_median_m": float(
            torch.as_tensor(geometry["triangulation_surface_correction_m"])[supported].median()
            if bool(supported.any()) else 0.0
        ),
        "correction_p95_m": float(
            torch.quantile(
                torch.as_tensor(geometry["triangulation_surface_correction_m"])[supported],
                0.95,
            ) if bool(supported.any()) else 0.0
        ),
        "correction_saturation_rate": float(
            (
                torch.as_tensor(
                    geometry["triangulation_surface_correction_m"]
                )[supported]
                >= 0.99
                * depth_sigma
                * float(maximum_correction_depth_sigmas)
            ).float().mean()
            if bool(supported.any()) else 0.0
        ),
        "final_depth_residual_median_m": float(
            torch.as_tensor(
                geometry["triangulation_rendered_depth_absolute_median_m"]
            )[supported].median()
            if bool(supported.any()) else 0.0
        ),
        "reprojection_delta_median_px": float(
            torch.as_tensor(
                geometry["triangulation_surface_reprojection_delta_px"]
            )[supported].median() if bool(supported.any()) else 0.0
        ),
        "uses_test_queries": False,
        "identity_source": "immutable_real_image_track_graph",
        "surface_source": "frozen_rgb_gaussian_rendered_depth",
    }
    revised["surface_supported_geometry"] = report
    return revised, report
