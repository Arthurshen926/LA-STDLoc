"""Tri-state render admissibility certificate for V7 feedback queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch


SCHEMA = "lafgs_v7_render_admissibility_certificate"
VERSION = 1
DECISIONS = ("ACCEPT", "UNCERTAIN", "REJECT")


@dataclass(frozen=True)
class CertificateThresholds:
    minimum_alpha_supported_fraction: float = 0.70
    reject_alpha_supported_fraction: float = 0.20
    minimum_positive_depth_fraction: float = 0.70
    reject_positive_depth_fraction: float = 0.20
    minimum_valid_keypoint_fraction: float = 0.65
    reject_valid_keypoint_fraction: float = 0.15
    maximum_black_or_hole_fraction: float = 0.30
    reject_black_or_hole_fraction: float = 0.80
    maximum_depth_discontinuity_keypoint_fraction: float = 0.30
    maximum_support_distance_baselines: float = 2.5
    reject_support_distance_baselines: float = 5.0
    minimum_source_family_support: int = 1
    minimum_expected_depth_ratio: float = 0.25
    maximum_expected_depth_ratio: float = 4.0
    alpha_minimum: float = 0.05
    black_luminance_maximum: float = 0.01
    border_fraction: float = 0.02
    depth_discontinuity_relative: float = 0.08


def _validate_thresholds(value: CertificateThresholds) -> None:
    fields = asdict(value)
    for name, number in fields.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ValueError(f"invalid render-certificate threshold: {name}")
        if not math.isfinite(float(number)) or float(number) < 0:
            raise ValueError(f"invalid render-certificate threshold: {name}")
    if not (
        value.reject_alpha_supported_fraction
        <= value.minimum_alpha_supported_fraction
        and value.reject_positive_depth_fraction
        <= value.minimum_positive_depth_fraction
        and value.reject_valid_keypoint_fraction
        <= value.minimum_valid_keypoint_fraction
        and value.maximum_black_or_hole_fraction
        <= value.reject_black_or_hole_fraction
        and value.maximum_support_distance_baselines
        <= value.reject_support_distance_baselines
    ):
        raise ValueError("render-certificate accept/reject thresholds are inverted")


def _image_tensors(
    rgb: torch.Tensor, alpha: torch.Tensor, depth: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = torch.as_tensor(rgb, dtype=torch.float32)
    alpha = torch.as_tensor(alpha, dtype=torch.float32).squeeze()
    depth = torch.as_tensor(depth, dtype=torch.float32).squeeze()
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("render certificate RGB must have shape [3,H,W]")
    if alpha.shape != rgb.shape[1:] or depth.shape != rgb.shape[1:]:
        raise ValueError("render RGB, alpha, and depth rasters must align")
    if not bool(torch.isfinite(rgb).all()):
        raise ValueError("render RGB contains non-finite values")
    return rgb, alpha, depth


def _sample_rows(mask: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("post-detector keypoints must have shape [N,2]")
    height, width = mask.shape
    xy = torch.round(keypoints).long()
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    result = torch.zeros(keypoints.shape[0], dtype=torch.bool)
    if bool(inside.any()):
        result[inside] = mask[xy[inside, 1], xy[inside, 0]]
    return result


def _depth_discontinuity(depth: torch.Tensor, *, relative: float) -> torch.Tensor:
    finite = torch.isfinite(depth) & (depth > 0)
    valid_values = depth[finite]
    if valid_values.numel() == 0:
        return torch.ones_like(finite)
    scale = valid_values.median().clamp_min(1e-6)
    horizontal = torch.zeros_like(depth)
    vertical = torch.zeros_like(depth)
    horizontal[:, 1:] = (depth[:, 1:] - depth[:, :-1]).abs()
    vertical[1:, :] = (depth[1:, :] - depth[:-1, :]).abs()
    return (~finite) | (torch.maximum(horizontal, vertical) > float(relative) * scale)


def extreme_distortion_row_mask(
    distortion: torch.Tensor, keypoints: torch.Tensor
) -> torch.Tensor:
    """Flag only robust extreme distortion outliers after keypoint detection."""

    value = torch.as_tensor(distortion, dtype=torch.float32).squeeze().abs()
    if value.ndim != 2:
        raise ValueError("2DGS distortion raster must reduce to shape [H,W]")
    finite = torch.isfinite(value)
    samples = value[finite]
    if samples.numel() == 0:
        return torch.ones(torch.as_tensor(keypoints).shape[0], dtype=torch.bool)
    median = samples.median()
    mad = (samples - median).abs().median().clamp_min(1e-8)
    extreme = (~finite) | (value > median + 20.0 * mad)
    return _sample_rows(extreme, keypoints)


def certify_v7_render(
    *,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    keypoints: torch.Tensor,
    nearest_mapping_distance_m: float,
    median_adjacent_baseline_m: float,
    source_family_support: int,
    expected_median_depth_m: float | None = None,
    artifact_row_mask: torch.Tensor | None = None,
    thresholds: CertificateThresholds = CertificateThresholds(),
) -> dict[str, Any]:
    """Certify a render after detection; never masks RGB before SuperPoint."""

    _validate_thresholds(thresholds)
    rgb, alpha, depth = _image_tensors(rgb, alpha, depth)
    keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
    finite_positive_depth = torch.isfinite(depth) & (depth > 0)
    alpha_supported = torch.isfinite(alpha) & (alpha >= thresholds.alpha_minimum)
    pixel_valid = alpha_supported & finite_positive_depth
    discontinuity = _depth_discontinuity(
        depth, relative=thresholds.depth_discontinuity_relative
    )
    height, width = depth.shape
    border = torch.zeros_like(pixel_valid)
    margin = max(1, round(min(height, width) * thresholds.border_fraction))
    border[:margin] = True
    border[-margin:] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    luminance = rgb.mean(dim=0)
    black_or_hole = (luminance <= thresholds.black_luminance_maximum) | (~alpha_supported)

    row_pixel_valid = _sample_rows(pixel_valid, keypoints)
    row_discontinuity = _sample_rows(discontinuity, keypoints)
    row_border = _sample_rows(border, keypoints)
    if artifact_row_mask is None:
        artifact_rows = torch.zeros(keypoints.shape[0], dtype=torch.bool)
    else:
        artifact_rows = torch.as_tensor(artifact_row_mask, dtype=torch.bool).reshape(-1)
        if artifact_rows.numel() != keypoints.shape[0]:
            raise ValueError("artifact row mask must align with detected keypoints")
    row_valid = row_pixel_valid & (~row_discontinuity) & (~row_border) & (~artifact_rows)
    row_uncertain = row_pixel_valid & (~row_valid)

    total = max(int(depth.numel()), 1)
    row_count = max(int(keypoints.shape[0]), 1)
    alpha_fraction = float(alpha_supported.sum()) / total
    depth_fraction = float(finite_positive_depth.sum()) / total
    black_fraction = float(black_or_hole.sum()) / total
    valid_keypoint_fraction = float(row_valid.sum()) / row_count
    discontinuity_fraction = float(row_discontinuity.sum()) / row_count
    artifact_fraction = float(artifact_rows.sum()) / row_count
    support_distance = float(nearest_mapping_distance_m) / max(
        float(median_adjacent_baseline_m), 1e-8
    )
    finite_depth_values = depth[finite_positive_depth]
    median_depth = (
        float(finite_depth_values.median()) if finite_depth_values.numel() else math.nan
    )
    expected_ratio = None
    if expected_median_depth_m is not None:
        expected = float(expected_median_depth_m)
        expected_ratio = median_depth / expected if expected > 0 and math.isfinite(median_depth) else math.nan

    reject_reasons: list[str] = []
    uncertain_reasons: list[str] = []
    if alpha_fraction < thresholds.reject_alpha_supported_fraction:
        reject_reasons.append("insufficient_alpha_support")
    elif alpha_fraction < thresholds.minimum_alpha_supported_fraction:
        uncertain_reasons.append("marginal_alpha_support")
    if depth_fraction < thresholds.reject_positive_depth_fraction:
        reject_reasons.append("insufficient_positive_depth")
    elif depth_fraction < thresholds.minimum_positive_depth_fraction:
        uncertain_reasons.append("marginal_positive_depth")
    if black_fraction > thresholds.reject_black_or_hole_fraction:
        reject_reasons.append("black_or_hole_dominated")
    elif black_fraction > thresholds.maximum_black_or_hole_fraction:
        uncertain_reasons.append("elevated_black_or_hole_fraction")
    if valid_keypoint_fraction < thresholds.reject_valid_keypoint_fraction:
        reject_reasons.append("insufficient_valid_keypoints")
    elif valid_keypoint_fraction < thresholds.minimum_valid_keypoint_fraction:
        uncertain_reasons.append("marginal_valid_keypoints")
    if discontinuity_fraction > thresholds.maximum_depth_discontinuity_keypoint_fraction:
        uncertain_reasons.append("depth_discontinuity_rows")
    if support_distance > thresholds.reject_support_distance_baselines:
        reject_reasons.append("outside_camera_support_envelope")
    elif support_distance > thresholds.maximum_support_distance_baselines:
        uncertain_reasons.append("weak_mapping_camera_support")
    if int(source_family_support) < thresholds.minimum_source_family_support:
        uncertain_reasons.append("insufficient_source_family_support")
    if expected_ratio is not None and (
        not math.isfinite(expected_ratio)
        or expected_ratio < thresholds.minimum_expected_depth_ratio
        or expected_ratio > thresholds.maximum_expected_depth_ratio
    ):
        reject_reasons.append("expected_depth_curtain_mismatch")
    if reject_reasons:
        decision = "REJECT"
    elif uncertain_reasons:
        decision = "UNCERTAIN"
    else:
        decision = "ACCEPT"
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "decision": decision,
        "can_drive_map_update": decision == "ACCEPT",
        "detector_input": "complete_unmasked_rgb",
        "quality_mask_stage": "post_detector_row_sampling",
        "reject_reasons": reject_reasons,
        "uncertain_reasons": uncertain_reasons,
        "signals": {
            "alpha_supported_fraction": alpha_fraction,
            "finite_positive_depth_fraction": depth_fraction,
            "black_or_hole_fraction": black_fraction,
            "valid_keypoint_fraction": valid_keypoint_fraction,
            "depth_discontinuity_keypoint_fraction": discontinuity_fraction,
            "extreme_artifact_keypoint_fraction": artifact_fraction,
            "nearest_mapping_distance_baselines": support_distance,
            "source_family_support": int(source_family_support),
            "median_rendered_depth_m": median_depth,
            "expected_depth_ratio": expected_ratio,
        },
        "row_valid": row_valid,
        "row_uncertain": row_uncertain,
        "thresholds": asdict(thresholds),
    }
