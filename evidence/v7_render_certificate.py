"""Tri-state render admissibility certificate for V7 feedback queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


SCHEMA = "lafgs_v7_render_admissibility_certificate"
VERSION = 2
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
    rgb_structure_window_reference_px: int = 9
    rgb_structure_support_threshold: float = 0.20
    rgb_structure_dilate_reference_px: int = 5
    rgb_structure_reference_short_side: int = 1080
    distortion_mad_multiplier: float = 20.0
    distortion_tail_quantile: float = 0.995


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
    if not 0.0 < value.distortion_tail_quantile < 1.0:
        raise ValueError("distortion_tail_quantile must lie strictly inside (0, 1)")
    if (
        value.rgb_structure_window_reference_px < 1
        or value.rgb_structure_reference_short_side < 1
    ):
        raise ValueError("RGB structure window and reference size must be positive")
    if not 0.0 < value.rgb_structure_support_threshold <= 1.0:
        raise ValueError("rgb_structure_support_threshold must lie inside (0, 1]")
    if value.distortion_mad_multiplier <= 0.0:
        raise ValueError("distortion_mad_multiplier must be positive")


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
    keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=mask.device)
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
    result = torch.zeros(
        keypoints.shape[0], dtype=torch.bool, device=mask.device
    )
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


def _border_mask(height: int, width: int, fraction: float, device) -> torch.Tensor:
    border = torch.zeros((int(height), int(width)), dtype=torch.bool, device=device)
    margin = max(1, round(min(int(height), int(width)) * float(fraction)))
    border[:margin] = True
    border[-margin:] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    return border


def _scaled_radius(value: int, *, height: int, width: int, reference: int) -> int:
    scale = min(height, width) / float(reference)
    return max(1, int(round(int(value) * scale)))


def _local_average(value: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel = max(3, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    pad = kernel // 2
    padded = F.pad(value[None, None], (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1)[0, 0]


def _normalized_map(value: torch.Tensor, quantile: float = 0.97) -> torch.Tensor:
    samples = value.detach().reshape(-1)
    if samples.numel() > 65536:
        stride = math.ceil(samples.numel() / 65536)
        samples = samples[::stride]
    scale = torch.quantile(samples, float(quantile)).clamp_min(1e-6)
    return (value / scale).clamp(0.0, 1.0)


def rgb_structure_support_mask(
    rgb: torch.Tensor,
    *,
    support_threshold: float = 0.20,
    window_reference_px: int = 9,
    dilate_reference_px: int = 5,
    reference_short_side: int = 1080,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a post-detector RGB structure prior and its continuous score.

    The mask is never applied to detector input. It only decides whether an
    already detected row lies near reproducible local intensity structure.
    Spatial parameters scale with the render short side so the certificate is
    stable across formal render resolutions.
    """

    image = torch.as_tensor(rgb, dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("RGB structure input must have shape [3,H,W]")
    gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    height, width = gray.shape
    window = _scaled_radius(
        window_reference_px,
        height=height,
        width=width,
        reference=reference_short_side,
    )
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    sobel_x = gray.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    padded = F.pad(gray[None, None], (1, 1, 1, 1), mode="replicate")
    gradient_x = F.conv2d(padded, sobel_x[None, None])[0, 0]
    gradient_y = F.conv2d(padded, sobel_x.t()[None, None])[0, 0]
    gradient = torch.sqrt(gradient_x.square() + gradient_y.square())
    local_mean = _local_average(gray, window)
    local_variance = _local_average((gray - local_mean).square(), window)
    local_gradient = _local_average(gradient, window)
    score = torch.maximum(
        _normalized_map(local_gradient),
        _normalized_map(torch.sqrt(local_variance.clamp_min(0.0))),
    )
    support = score >= float(support_threshold)
    dilation = (
        0
        if int(dilate_reference_px) == 0
        else _scaled_radius(
            dilate_reference_px,
            height=height,
            width=width,
            reference=reference_short_side,
        )
    )
    kernel = 2 * dilation + 1
    support = (
        F.max_pool2d(
            support.float()[None, None],
            kernel_size=kernel,
            stride=1,
            padding=dilation,
        )[0, 0]
        > 0
    )
    return support, score


def render_quality_pixel_masks(
    *,
    rgb: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    distortion: torch.Tensor | None = None,
    thresholds: CertificateThresholds = CertificateThresholds(),
) -> dict[str, torch.Tensor]:
    """Expose V2 as tri-state pixel evidence for offline detector supervision.

    Depth boundaries and image borders are uncertain because they can contain
    useful structure.  Definite holes, unsupported RGB regions, and extreme
    distortion are negative.  This function is never a pre-SuperPoint mask in
    the deployed frontend.
    """

    _validate_thresholds(thresholds)
    image, opacity, rendered_depth = _image_tensors(rgb, alpha, depth)
    height, width = rendered_depth.shape
    pixel_valid = (
        torch.isfinite(opacity)
        & (opacity >= thresholds.alpha_minimum)
        & torch.isfinite(rendered_depth)
        & (rendered_depth > 0)
    )
    discontinuity = _depth_discontinuity(
        rendered_depth, relative=thresholds.depth_discontinuity_relative
    )
    border = _border_mask(
        height, width, thresholds.border_fraction, rendered_depth.device
    )
    structure, structure_score = rgb_structure_support_mask(
        image,
        support_threshold=thresholds.rgb_structure_support_threshold,
        window_reference_px=thresholds.rgb_structure_window_reference_px,
        dilate_reference_px=thresholds.rgb_structure_dilate_reference_px,
        reference_short_side=thresholds.rgb_structure_reference_short_side,
    )
    if distortion is None:
        extreme = torch.zeros_like(pixel_valid)
    else:
        value = torch.as_tensor(
            distortion, device=rendered_depth.device, dtype=torch.float32
        ).squeeze().abs()
        if value.shape != rendered_depth.shape:
            raise ValueError("distortion raster must align with render pixels")
        finite = torch.isfinite(value)
        samples = value[finite]
        if samples.numel() == 0:
            extreme = torch.ones_like(pixel_valid)
        else:
            median = samples.median()
            mad = (samples - median).abs().median().clamp_min(1e-8)
            robust = median + thresholds.distortion_mad_multiplier * mad
            tail = torch.quantile(samples, thresholds.distortion_tail_quantile)
            extreme = (~finite) | (value > torch.maximum(robust, tail))
    uncertain = discontinuity | border
    invalid = (~pixel_valid) | (~structure) | extreme
    valid = pixel_valid & structure & (~extreme) & (~uncertain)
    return {
        "valid": valid,
        "invalid": invalid & (~uncertain),
        "uncertain": uncertain,
        "pixel_support": pixel_valid,
        "rgb_structure_support": structure,
        "rgb_structure_score": structure_score,
        "depth_discontinuity": discontinuity,
        "border": border,
        "extreme_distortion": extreme,
    }


def extreme_distortion_row_mask(
    distortion: torch.Tensor,
    keypoints: torch.Tensor,
    *,
    mad_multiplier: float = 20.0,
    tail_quantile: float = 0.995,
) -> torch.Tensor:
    """Flag distortion only when it is both robustly and rank-extreme.

    A broad secondary distortion mode is not an extreme artifact by itself.
    Requiring the configured upper tail prevents a bimodal render from marking
    most architectural keypoints invalid, while isolated spikes remain flagged.
    """

    value = torch.as_tensor(distortion, dtype=torch.float32).squeeze().abs()
    if value.ndim != 2:
        raise ValueError("2DGS distortion raster must reduce to shape [H,W]")
    finite = torch.isfinite(value)
    samples = value[finite]
    if samples.numel() == 0:
        return torch.ones(
            torch.as_tensor(keypoints).shape[0],
            dtype=torch.bool,
            device=value.device,
        )
    if not 0.0 < float(tail_quantile) < 1.0:
        raise ValueError("distortion tail quantile must lie strictly inside (0, 1)")
    median = samples.median()
    mad = (samples - median).abs().median().clamp_min(1e-8)
    robust_threshold = median + float(mad_multiplier) * mad
    tail_threshold = torch.quantile(samples, float(tail_quantile))
    threshold = torch.maximum(robust_threshold, tail_threshold)
    extreme = (~finite) | (value > threshold)
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
    border = _border_mask(height, width, thresholds.border_fraction, depth.device)
    luminance = rgb.mean(dim=0)
    black_or_hole = (luminance <= thresholds.black_luminance_maximum) | (~alpha_supported)

    row_pixel_valid = _sample_rows(pixel_valid, keypoints)
    row_discontinuity = _sample_rows(discontinuity, keypoints)
    row_border = _sample_rows(border, keypoints)
    structure_support, structure_score = rgb_structure_support_mask(
        rgb,
        support_threshold=thresholds.rgb_structure_support_threshold,
        window_reference_px=thresholds.rgb_structure_window_reference_px,
        dilate_reference_px=thresholds.rgb_structure_dilate_reference_px,
        reference_short_side=thresholds.rgb_structure_reference_short_side,
    )
    row_structure_supported = _sample_rows(structure_support, keypoints)
    if artifact_row_mask is None:
        artifact_rows = torch.zeros(
            keypoints.shape[0], dtype=torch.bool, device=row_pixel_valid.device
        )
    else:
        artifact_rows = torch.as_tensor(
            artifact_row_mask, dtype=torch.bool, device=row_pixel_valid.device
        ).reshape(-1)
        if artifact_rows.numel() != keypoints.shape[0]:
            raise ValueError("artifact row mask must align with detected keypoints")
    row_valid = (
        row_pixel_valid
        & (~row_discontinuity)
        & (~row_border)
        & (~artifact_rows)
        & row_structure_supported
    )
    row_uncertain = row_pixel_valid & (~row_valid)

    total = max(int(depth.numel()), 1)
    row_count = max(int(keypoints.shape[0]), 1)
    alpha_fraction = float(alpha_supported.sum()) / total
    depth_fraction = float(finite_positive_depth.sum()) / total
    black_fraction = float(black_or_hole.sum()) / total
    valid_keypoint_fraction = float(row_valid.sum()) / row_count
    discontinuity_fraction = float(row_discontinuity.sum()) / row_count
    artifact_fraction = float(artifact_rows.sum()) / row_count
    structure_keypoint_fraction = float(row_structure_supported.sum()) / row_count
    structure_pixel_fraction = float(structure_support.sum()) / total
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
        "quality_fusion_policy": "basic_geometry_and_rank_extreme_distortion_and_rgb_structure_v2",
        "reject_reasons": reject_reasons,
        "uncertain_reasons": uncertain_reasons,
        "signals": {
            "alpha_supported_fraction": alpha_fraction,
            "finite_positive_depth_fraction": depth_fraction,
            "black_or_hole_fraction": black_fraction,
            "valid_keypoint_fraction": valid_keypoint_fraction,
            "depth_discontinuity_keypoint_fraction": discontinuity_fraction,
            "extreme_artifact_keypoint_fraction": artifact_fraction,
            "rgb_structure_supported_keypoint_fraction": structure_keypoint_fraction,
            "rgb_structure_supported_pixel_fraction": structure_pixel_fraction,
            "rgb_structure_score_mean": float(structure_score.mean()),
            "nearest_mapping_distance_baselines": support_distance,
            "source_family_support": int(source_family_support),
            "median_rendered_depth_m": median_depth,
            "expected_depth_ratio": expected_ratio,
        },
        "row_valid": row_valid,
        "row_uncertain": row_uncertain,
        "row_reasons": {
            "invalid_pixel_support": ~row_pixel_valid,
            "depth_discontinuity": row_discontinuity,
            "border": row_border,
            "extreme_distortion": artifact_rows,
            "low_rgb_structure_support": ~row_structure_supported,
        },
        "thresholds": asdict(thresholds),
    }
