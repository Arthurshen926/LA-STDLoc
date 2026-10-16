"""Frozen mainline kernel extracted from priors.mapping_supersampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from anygsloc.priors.rendering import render_from_pose_gsplat


def mapping_render_sampling_contract(
    *, gaussian_type: str, supersampling_factor: int
) -> dict:
    factor = int(supersampling_factor)
    if factor < 1 or factor > 4:
        raise ValueError("mapping supersampling factor must lie inside [1,4]")
    if str(gaussian_type) not in {"2dgs", "3dgs"}:
        raise ValueError("unsupported Gaussian renderer type")
    return {
        "supersampling_factor": factor,
        "render_resolution_scale": factor,
        "output_resolution_scale": 1,
        "rgb_downsampling": "area_box" if factor > 1 else "identity",
        "alpha_downsampling": "area_box" if factor > 1 else "identity",
        "depth_downsampling": ("alpha_weighted_area_box" if factor > 1 else "identity"),
        "depth_invalid_sample_policy": (
            "exclude_nonfinite_nonpositive_depth_and_invalid_alpha"
            if factor > 1
            else "identity"
        ),
        "requested_rasterize_mode": "antialiased",
        "effective_rasterize_mode": (
            "omitted_unsupported_by_2dgs_wrapper"
            if str(gaussian_type) == "2dgs"
            else "antialiased"
        ),
        "rasterize_mode_argument_forwarded": str(gaussian_type) != "2dgs",
        "native_pixel_coordinate_system_preserved": True,
    }


def _channel_first(
    value: torch.Tensor, *, name: str, expected_hw: tuple[int, int]
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{name} render batch must contain one view")
        tensor = tensor[0]
    if tensor.ndim == 2:
        tensor = tensor[None]
    elif tensor.ndim == 3:
        if tuple(tensor.shape[-2:]) == expected_hw:
            pass
        elif tuple(tensor.shape[:2]) == expected_hw and tensor.shape[-1] <= 4:
            tensor = tensor.permute(2, 0, 1)
    if (
        tensor.ndim != 3
        or tensor.shape[0] > 4
        or tuple(tensor.shape[-2:]) != expected_hw
    ):
        raise ValueError(f"{name} render has unsupported shape {tuple(tensor.shape)}")
    return tensor


def downsample_mapping_render_package(
    package: dict,
    *,
    target_height: int,
    target_width: int,
    supersampling_factor: int,
) -> dict:
    """Downsample RGB/alpha/depth while preserving the native camera grid."""

    factor = int(supersampling_factor)
    height, width = int(target_height), int(target_width)
    if factor < 1 or height < 1 or width < 1:
        raise ValueError("mapping render dimensions/factor must be positive")
    expected = (height * factor, width * factor)
    rgb = _channel_first(package["render"], name="RGB", expected_hw=expected).float()[
        :3
    ]
    alpha_value = package.get("alphas", package.get("rend_alpha"))
    depth_value = package.get("depth")
    if alpha_value is None or depth_value is None:
        raise ValueError("mapping supersampling requires alpha and depth")
    alpha = _channel_first(alpha_value, name="alpha", expected_hw=expected).float()[:1]
    depth = _channel_first(depth_value, name="depth", expected_hw=expected).float()[:1]
    if (
        rgb.shape[-2:] != expected
        or alpha.shape[-2:] != expected
        or depth.shape[-2:] != expected
    ):
        raise ValueError("supersampled render dimensions do not match the camera")
    if factor > 1:
        size = (height, width)
        rgb = F.interpolate(rgb[None], size=size, mode="area")[0]
        alpha_down = F.interpolate(alpha[None], size=size, mode="area")[0]
        # An uncovered subpixel may contain NaN/Inf depth. Multiplying by
        # zero alpha does not remove NaNs; exclude invalid samples before
        # multiplication and normalize using only their valid support.
        depth_valid = torch.isfinite(depth) & (depth > 0)
        depth_valid &= torch.isfinite(alpha) & (alpha > 0)
        depth_alpha = torch.where(depth_valid, alpha, torch.zeros_like(alpha))
        safe_depth = torch.where(depth_valid, depth, torch.zeros_like(depth))
        depth_numerator = F.interpolate(
            (safe_depth * depth_alpha)[None], size=size, mode="area"
        )[0]
        depth_support = F.interpolate(depth_alpha[None], size=size, mode="area")[0]
        depth = torch.where(
            depth_support > 1e-8,
            depth_numerator / depth_support.clamp_min(1e-8),
            torch.zeros_like(depth_numerator),
        )
        alpha = alpha_down
    output = {
        "render": rgb,
        "alphas": alpha,
        "rend_alpha": alpha,
        "depth": depth,
    }
    distortion = package.get("rend_dist")
    if distortion is not None:
        distortion = _channel_first(
            distortion, name="distortion", expected_hw=expected
        ).float()[:1]
        if distortion.shape[-2:] != expected:
            raise ValueError("distortion render dimensions do not match the camera")
        if factor > 1:
            distortion = F.interpolate(
                distortion[None], size=(height, width), mode="area"
            )[0]
        output["rend_dist"] = distortion
    return output


@torch.inference_mode()
def render_mapping_rgbd(
    model,
    pose_w2c: torch.Tensor,
    *,
    fov_x: float,
    fov_y: float,
    width: int,
    height: int,
    background: torch.Tensor,
    gaussian_type: str,
    supersampling_factor: int = 1,
) -> tuple[dict, dict]:
    """Render a mapping view at higher resolution then return native pixels."""

    factor = int(supersampling_factor)
    if factor < 1 or factor > 4:
        raise ValueError("mapping supersampling factor must lie inside [1,4]")
    package = render_from_pose_gsplat(
        model,
        pose_w2c,
        fov_x,
        fov_y,
        int(width) * factor,
        int(height) * factor,
        bg_color=background,
        render_mode="RGB+ED",
        rgb_only=True,
        rasterize_mode="antialiased",
    )
    native = downsample_mapping_render_package(
        package,
        target_height=int(height),
        target_width=int(width),
        supersampling_factor=factor,
    )
    contract = mapping_render_sampling_contract(
        gaussian_type=gaussian_type, supersampling_factor=factor
    )
    return native, contract
