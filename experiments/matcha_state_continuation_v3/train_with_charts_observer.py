#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import json
sys.path.insert(0, "/root/MAtCha")
sys.path.insert(0, "/root/MAtCha/2d-gaussian-splatting")
from exact_resume import Controller, add_arguments
from typing import Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import copy
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import SH2RGB
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.point_utils import depth_to_normal
from matcha.dm_scene.charts import (
    load_charts_data, 
    build_priors_from_charts_data,
    schedule_regularization_factor_1,
    schedule_regularization_factor_2,
    get_gaussian_parameters_from_charts_data,
)
from matcha.dm_scene.chart_sampling import (
    chart_regularization_indices_for_iteration,
    canonical_chart_key,
    chart_quality_attenuation,
    chart_scale_multiplier,
    load_chart_selection_keep,
    select_chart_initialization_indices,
)
from matcha.dm_regularization.depth import compute_depth_order_loss
from matcha.dm_utils.rendering import normal2curv
from matcha.cambridge_masks import CambridgeMaskLookup, combine_optional_masks, mask_to_weight
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import matplotlib.pyplot as plt


# Old confidence to increasing weight function
# def confidence_to_weight(confidence:torch.Tensor):
#     conf_weights = confidence - 1.
#     return 1. - torch.exp(-conf_weights**2 / 2)

# New confidence to increasing weight function
def confidence_to_weight(confidence:torch.Tensor):
    conf_weights = confidence - 1.
    return torch.sigmoid((conf_weights - 2.) * 2.)


def prepare_rgb_loss_mask(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3:
        raise RuntimeError(f"Expected mask with 2 or 3 dimensions, got {tuple(mask.shape)}")
    mask = mask.to(device=image.device, dtype=image.dtype)
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask[None], size=image.shape[-2:], mode="nearest")[0]
    if mask.shape[0] not in (1, image.shape[0]):
        raise RuntimeError(f"Mask channels {mask.shape[0]} do not match image channels {image.shape[0]}")
    if mask.sum() <= 0:
        raise RuntimeError("RGB mask removed all pixels for one training view")
    return mask


def photometric_difference(diff: torch.Tensor, loss_type: str = "l1", charbonnier_eps: float = 1e-3) -> torch.Tensor:
    if loss_type == "l1":
        return diff.abs()
    if loss_type == "charbonnier":
        return torch.sqrt(diff.square() + charbonnier_eps ** 2)
    raise RuntimeError(f"Unsupported RGB loss type {loss_type!r}")


def build_robust_rgb_weight(
    image: torch.Tensor,
    gt_image: torch.Tensor,
    base_weight: torch.Tensor,
    loss_type: str = "l1",
    charbonnier_eps: float = 1e-3,
    rgb_robust_quantile: float = None,
    rgb_robust_start_iter: int = 0,
    rgb_robust_min_weight: float = 0.0,
    iteration: int = None,
) -> torch.Tensor:
    if rgb_robust_quantile is None or rgb_robust_quantile <= 0.0 or rgb_robust_quantile >= 1.0:
        return base_weight
    if iteration is None or iteration < rgb_robust_start_iter:
        return base_weight
    if rgb_robust_min_weight < 0.0 or rgb_robust_min_weight > 1.0:
        raise RuntimeError(f"rgb_robust_min_weight must be in [0, 1], got {rgb_robust_min_weight}")

    residual = photometric_difference(image - gt_image, loss_type, charbonnier_eps).detach().mean(dim=0, keepdim=True)
    valid = base_weight.mean(dim=0, keepdim=True) > 0
    if valid.sum() <= 1:
        return base_weight

    threshold = torch.quantile(residual[valid], rgb_robust_quantile)
    robust_weight = torch.where(
        residual <= threshold,
        torch.ones_like(residual),
        torch.full_like(residual, rgb_robust_min_weight),
    )
    return base_weight * robust_weight


def compute_rgb_loss(
    image: torch.Tensor,
    gt_image: torch.Tensor,
    mask: torch.Tensor = None,
    lambda_dssim: float = 0.2,
    loss_type: str = "l1",
    charbonnier_eps: float = 1e-3,
    rgb_robust_quantile: float = None,
    rgb_robust_start_iter: int = 0,
    rgb_robust_min_weight: float = 0.0,
    iteration: int = None,
    transient_weight: torch.Tensor = None,
):
    if mask is None:
        loss_weight = torch.ones((1, *image.shape[-2:]), device=image.device, dtype=image.dtype)
    else:
        loss_weight = prepare_rgb_loss_mask(mask, image)
    if transient_weight is not None:
        if transient_weight.ndim == 2:
            transient_weight = transient_weight[None]
        if transient_weight.ndim != 3:
            raise RuntimeError(
                f"Expected transient weight with 2 or 3 dimensions, got {tuple(transient_weight.shape)}"
            )
        transient_weight = transient_weight.to(device=image.device, dtype=image.dtype).clamp_min(0.0)
        if transient_weight.shape[-2:] != image.shape[-2:]:
            transient_weight = F.interpolate(
                transient_weight[None],
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
        if transient_weight.shape[0] not in (1, image.shape[0]):
            raise RuntimeError(
                f"Transient weight channels {transient_weight.shape[0]} do not match image channels {image.shape[0]}"
            )
        loss_weight = loss_weight * transient_weight
    loss_weight = build_robust_rgb_weight(
        image,
        gt_image,
        loss_weight,
        loss_type=loss_type,
        charbonnier_eps=charbonnier_eps,
        rgb_robust_quantile=rgb_robust_quantile,
        rgb_robust_start_iter=rgb_robust_start_iter,
        rgb_robust_min_weight=rgb_robust_min_weight,
        iteration=iteration,
    )
    expanded_loss_weight = loss_weight.expand_as(image)
    denominator = expanded_loss_weight.sum()
    if denominator <= 0:
        Ll1 = (photometric_difference(image - gt_image, loss_type, charbonnier_eps) * expanded_loss_weight).sum()
    else:
        Ll1 = (
            photometric_difference(image - gt_image, loss_type, charbonnier_eps)
            * expanded_loss_weight
        ).sum() / denominator
    if lambda_dssim > 0.0:
        dssim = 1.0 - ssim(image * loss_weight, gt_image * loss_weight)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * dssim
    else:
        loss = Ll1
    return {"Ll1": Ll1, "loss": loss, "dssim": dssim if lambda_dssim > 0.0 else None}


class PerImageAffineColorCorrection(torch.nn.Module):
    def __init__(self, image_names):
        super().__init__()
        unique_image_names = list(dict.fromkeys(str(image_name) for image_name in image_names))
        if not unique_image_names:
            raise RuntimeError("Per-image color correction requires at least one image name")
        self.image_name_to_index = {
            image_name: index for index, image_name in enumerate(unique_image_names)
        }
        self.log_scales = torch.nn.Parameter(torch.zeros((len(unique_image_names), 3, 1, 1)))
        self.biases = torch.nn.Parameter(torch.zeros((len(unique_image_names), 3, 1, 1)))

    def forward(self, image: torch.Tensor, image_name: str) -> torch.Tensor:
        image_name = str(image_name)
        if image_name not in self.image_name_to_index:
            raise RuntimeError(f"Missing color correction parameters for image {image_name!r}")
        index = self.image_name_to_index[image_name]
        scale = torch.exp(self.log_scales[index]).to(device=image.device, dtype=image.dtype)
        bias = self.biases[index].to(device=image.device, dtype=image.dtype)
        return image * scale + bias

    def identity_regularization(self) -> torch.Tensor:
        return self.log_scales.square().mean() + self.biases.square().mean()


class PerImageAppearanceEmbeddingColorCorrection(torch.nn.Module):
    def __init__(self, image_names, embedding_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        unique_image_names = list(dict.fromkeys(str(image_name) for image_name in image_names))
        if not unique_image_names:
            raise RuntimeError("Per-image appearance embedding requires at least one image name")
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise RuntimeError("Appearance embedding and hidden dimensions must be positive")
        self.image_name_to_index = {
            image_name: index for index, image_name in enumerate(unique_image_names)
        }
        self.embeddings = torch.nn.Embedding(len(unique_image_names), embedding_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 6),
        )
        torch.nn.init.zeros_(self.embeddings.weight)
        torch.nn.init.constant_(self.mlp[0].bias, 0.01)
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

    def _index_tensor(self, image_name: str, device) -> torch.Tensor:
        image_name = str(image_name)
        if image_name not in self.image_name_to_index:
            raise RuntimeError(f"Missing appearance embedding for image {image_name!r}")
        return torch.tensor(self.image_name_to_index[image_name], device=device, dtype=torch.long)

    def affine_parameters(self, image_name: str) -> torch.Tensor:
        index = self._index_tensor(image_name, self.embeddings.weight.device)
        return self.mlp(self.embeddings(index))

    def forward(self, image: torch.Tensor, image_name: str) -> torch.Tensor:
        params = self.affine_parameters(image_name).to(device=image.device, dtype=image.dtype)
        log_scale = params[:3].reshape(3, 1, 1)
        bias = params[3:].reshape(3, 1, 1)
        return image * torch.exp(log_scale) + bias

    def identity_regularization(self) -> torch.Tensor:
        all_params = self.mlp(self.embeddings.weight)
        return self.embeddings.weight.square().mean() + all_params.square().mean()


class PerImageTransientModel(torch.nn.Module):
    def __init__(self, image_names, grid_size: int = 16):
        super().__init__()
        unique_image_names = list(dict.fromkeys(str(image_name) for image_name in image_names))
        if not unique_image_names:
            raise RuntimeError("Per-image transient model requires at least one image name")
        if grid_size <= 0:
            raise RuntimeError(f"Transient grid size must be positive, got {grid_size}")
        self.image_name_to_index = {
            image_name: index for index, image_name in enumerate(unique_image_names)
        }
        self.residuals = torch.nn.Parameter(torch.zeros((len(unique_image_names), 3, grid_size, grid_size)))
        self.alpha_logits = torch.nn.Parameter(torch.full((len(unique_image_names), 1, grid_size, grid_size), -10.0))
        self.confidence_logits = torch.nn.Parameter(torch.full((len(unique_image_names), 1, grid_size, grid_size), 10.0))

    def _index(self, image_name: str) -> int:
        image_name = str(image_name)
        if image_name not in self.image_name_to_index:
            raise RuntimeError(f"Missing transient parameters for image {image_name!r}")
        return self.image_name_to_index[image_name]

    def _upsample(self, tensor: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        return F.interpolate(tensor[None], size=shape, mode="bilinear", align_corners=False)[0]

    def forward(self, image: torch.Tensor, image_name: str):
        index = self._index(image_name)
        shape = image.shape[-2:]
        residual = self._upsample(self.residuals[index], shape).to(device=image.device, dtype=image.dtype)
        alpha = torch.sigmoid(self._upsample(self.alpha_logits[index], shape)).to(device=image.device, dtype=image.dtype)
        confidence = torch.sigmoid(self._upsample(self.confidence_logits[index], shape)).to(device=image.device, dtype=image.dtype)
        return image + alpha * residual, confidence

    def regularization(
        self,
        image_name: str = None,
        alpha_weight: float = 0.1,
        residual_weight: float = 0.01,
        confidence_weight: float = 0.001,
    ) -> torch.Tensor:
        if image_name is None:
            residuals = self.residuals
            alpha = torch.sigmoid(self.alpha_logits)
            confidence = torch.sigmoid(self.confidence_logits)
        else:
            index = self._index(image_name)
            residuals = self.residuals[index:index + 1]
            alpha = torch.sigmoid(self.alpha_logits[index:index + 1])
            confidence = torch.sigmoid(self.confidence_logits[index:index + 1])
        alpha_loss = alpha.mean()
        residual_loss = (alpha * residuals.abs()).mean()
        confidence_loss = (1.0 - confidence).mean()
        return (
            alpha_weight * alpha_loss
            + residual_weight * residual_loss
            + confidence_weight * confidence_loss
        )


def masked_mean(values: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    loss_mask = mask.to(device=values.device, dtype=values.dtype)
    if values.ndim >= 4 and values.ndim == loss_mask.ndim + 1 and loss_mask.shape[0] == values.shape[0]:
        loss_mask = loss_mask[:, None]
    else:
        while loss_mask.ndim < values.ndim:
            loss_mask = loss_mask.unsqueeze(0)
    if loss_mask.shape[-2:] != values.shape[-2:]:
        leading_shape = loss_mask.shape[:-2]
        loss_mask = F.interpolate(
            loss_mask.reshape(-1, 1, *loss_mask.shape[-2:]),
            size=values.shape[-2:],
            mode="nearest",
        ).reshape(*leading_shape, *values.shape[-2:])
    expanded_mask = loss_mask.expand_as(values)
    denominator = expanded_mask.sum()
    if denominator <= 0:
        return (values * expanded_mask).sum()
    return (values * expanded_mask).sum() / denominator


def get_cambridge_masks_for_cameras(mask_lookup, cameras, shape, device) -> torch.Tensor:
    return torch.stack([
        mask_lookup.get_mask(camera.image_name, shape, device)
        for camera in cameras
    ])


def apply_geometry_mask_to_charts_data(charts_data: dict, geometry_mask: torch.Tensor = None, invalid_conf: float = -2.0) -> dict:
    if geometry_mask is None:
        return charts_data

    confs = charts_data["confs"]
    keep_mask = geometry_mask.to(device=confs.device, dtype=torch.bool)
    if keep_mask.ndim == 4 and keep_mask.shape[1] == 1:
        keep_mask = keep_mask[:, 0]
    if keep_mask.ndim == 2 and confs.ndim == 3:
        keep_mask = keep_mask[None].expand(confs.shape[0], -1, -1)
    if keep_mask.shape[-2:] != confs.shape[-2:]:
        keep_mask = F.interpolate(
            keep_mask.to(dtype=torch.float32)[:, None],
            size=confs.shape[-2:],
            mode="nearest",
        )[:, 0] > 0.5
    if keep_mask.shape != confs.shape:
        raise RuntimeError(f"Geometry mask shape {tuple(keep_mask.shape)} does not match chart confs {tuple(confs.shape)}")

    masked_charts_data = dict(charts_data)
    invalid_values = torch.full_like(confs, invalid_conf)
    masked_charts_data["confs"] = torch.where(keep_mask, confs, invalid_values)
    return masked_charts_data


def build_chart_outlier_mask(
    charts_data: dict,
    max_abs_point: float = None,
    max_abs_depth: float = None,
) -> torch.Tensor:
    if (max_abs_point is None or max_abs_point <= 0) and (max_abs_depth is None or max_abs_depth <= 0):
        return None

    if "confs" in charts_data:
        reference = charts_data["confs"]
    elif "depths" in charts_data:
        reference = charts_data["depths"]
    elif "pts" in charts_data:
        reference = charts_data["pts"][..., 0]
    else:
        raise RuntimeError("Chart outlier mask requires at least one of confs, depths, or pts")
    keep_mask = torch.ones_like(reference, dtype=torch.bool)

    pts = charts_data.get("pts")
    if pts is not None:
        point_mask = torch.isfinite(pts).all(dim=-1)
        if max_abs_point is not None and max_abs_point > 0:
            point_mask = point_mask & (pts.abs().amax(dim=-1) <= max_abs_point)
        keep_mask = keep_mask & point_mask.to(device=keep_mask.device)

    depths = charts_data.get("depths")
    if depths is not None:
        depth_mask = torch.isfinite(depths)
        if max_abs_depth is not None and max_abs_depth > 0:
            depth_mask = depth_mask & (depths.abs() <= max_abs_depth)
        keep_mask = keep_mask & depth_mask.to(device=keep_mask.device)

    return keep_mask


def build_chart_confidence_mask(
    charts_data: dict,
    min_chart_conf: float = None,
    min_chart_conf_quantile: float = None,
) -> torch.Tensor:
    if (min_chart_conf is None) and (min_chart_conf_quantile is None):
        return None
    confs = charts_data["confs"]
    keep_mask = torch.isfinite(confs)
    if min_chart_conf is not None:
        keep_mask = keep_mask & (confs >= min_chart_conf)
    if min_chart_conf_quantile is not None:
        if min_chart_conf_quantile <= 0.0 or min_chart_conf_quantile >= 1.0:
            raise RuntimeError(f"min_chart_conf_quantile must be in (0, 1), got {min_chart_conf_quantile}")
        quantile_mask = torch.zeros_like(keep_mask)
        flat_confs = confs.reshape(confs.shape[0], -1)
        flat_quantile_mask = quantile_mask.reshape(confs.shape[0], -1)
        for chart_index in range(flat_confs.shape[0]):
            valid_values = flat_confs[chart_index][torch.isfinite(flat_confs[chart_index])]
            if valid_values.numel() == 0:
                continue
            threshold = torch.quantile(valid_values, min_chart_conf_quantile)
            flat_quantile_mask[chart_index] = flat_confs[chart_index] >= threshold
        keep_mask = keep_mask & quantile_mask
    return keep_mask


def load_chart_reliabilities(
    reliability_json: Optional[str],
    cameras,
    device,
) -> Optional[torch.Tensor]:
    """Load optional per-chart reliability scores produced by the selector."""
    if reliability_json is None:
        return None
    with open(reliability_json, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scores = payload.get("chart_scores", payload.get("scores", payload))
    if not isinstance(scores, dict):
        raise RuntimeError(
            "chart_reliability_json must be a name-to-score mapping or contain chart_scores"
        )
    canonical_scores = {}
    for raw_name, record in scores.items():
        canonical_name = canonical_chart_key(raw_name)
        if canonical_name in canonical_scores:
            raise RuntimeError(
                "chart_reliability_json maps multiple records to the same chart "
                f"key {canonical_name!r}"
            )
        canonical_scores[canonical_name] = record

    values = []
    missing = []
    for camera in cameras:
        image_name = str(camera.image_name)
        record = canonical_scores.get(canonical_chart_key(image_name))
        if record is None:
            missing.append(image_name)
            values.append(1.0)
            continue
        if isinstance(record, dict):
            record = record.get("reliability", record.get("combined_reliability"))
        if record is None:
            raise RuntimeError(f"No reliability scalar for chart {image_name!r}")
        values.append(float(record))
    if missing:
        preview = ", ".join(missing[:5])
        print(
            f"[WARNING] Reliability file omitted {len(missing)} chart(s); "
            f"using 1.0 for them (first: {preview})."
        )
    reliabilities = torch.tensor(values, device=device, dtype=torch.float32)
    if not torch.isfinite(reliabilities).all():
        raise RuntimeError("chart_reliability_json contains non-finite scores")
    reliabilities = reliabilities.clamp(0.0, 1.0)
    print(
        "[INFO] Loaded chart reliabilities: "
        f"min={reliabilities.min().item():.3f}, mean={reliabilities.mean().item():.3f}, "
        f"max={reliabilities.max().item():.3f}."
    )
    return reliabilities


def build_chart_initialization_selection_mask(
    selection_json: Optional[str],
    cameras,
    shape: tuple[int, int],
    device,
) -> Optional[torch.Tensor]:
    """Mask chart geometry used to seed Gaussians without discarding priors."""
    keep = load_chart_selection_keep(
        selection_json,
        cameras,
        device,
        argument_name="chart_init_selection_json",
    )
    if keep is None:
        return None
    print(
        f"[INFO] Chart initialization selection keeps {int(keep.sum().item())}/"
        f"{len(keep)} charts from {selection_json}."
    )
    return keep[:, None, None].expand(len(keep), *shape)


def build_chart_prior_selection_mask(
    selection_json: Optional[str],
    cameras,
    shape: tuple[int, int],
    device,
) -> Optional[torch.Tensor]:
    """Disable early chart depth/normal priors outside a joint selection.

    Unlike ``chart_init_selection_json``, this is deliberately a downstream
    gate: excluded Charts remain in the camera scene for bookkeeping, but
    contribute exactly zero prior weight.  Keeping the two flags separate
    preserves legacy initialization-only ablations while making a strict
    jointly selected set enforceable end-to-end.
    """
    keep = load_chart_selection_keep(
        selection_json,
        cameras,
        device,
        argument_name="chart_prior_selection_json",
    )
    if keep is None:
        return None
    print(
        f"[INFO] Chart prior selection keeps {int(keep.sum().item())}/"
        f"{len(keep)} charts from {selection_json}; excluded charts have zero prior weight."
    )
    return keep[:, None, None].expand(len(keep), *shape)


def build_chart_late_prior_selection_mask(
    selection_json: Optional[str],
    cameras,
    shape: tuple[int, int],
    device,
) -> Optional[torch.Tensor]:
    """Load a stricter Chart subset for late, incremental regularization.

    This does not affect Gaussian initialization or early Chart anchoring. It
    is a separate stage because coverage matters most before RGB supervision
    stabilizes, while late refinement can prefer the jointly selected quality
    subset.
    """
    keep = load_chart_selection_keep(
        selection_json,
        cameras,
        device,
        argument_name="chart_late_prior_selection_json",
    )
    if keep is None:
        return None
    print(
        f"[INFO] Late Chart prior selection keeps {int(keep.sum().item())}/"
        f"{len(keep)} charts from {selection_json}; it activates only after the configured iteration."
    )
    return keep[:, None, None].expand(len(keep), *shape)


def build_chart_confidence_weights(
    charts_data: dict,
    chart_prior_conf_weight_floor: float = None,
    chart_prior_conf_weight_power: float = 1.0,
) -> torch.Tensor:
    if chart_prior_conf_weight_floor is None:
        return None
    if chart_prior_conf_weight_floor < 0.0 or chart_prior_conf_weight_floor > 1.0:
        raise RuntimeError(
            f"chart_prior_conf_weight_floor must be in [0, 1], got {chart_prior_conf_weight_floor}"
        )
    if chart_prior_conf_weight_power <= 0.0:
        raise RuntimeError(f"chart_prior_conf_weight_power must be positive, got {chart_prior_conf_weight_power}")

    confs = charts_data["confs"].to(dtype=torch.float32)
    weights = torch.zeros_like(confs)
    flat_confs = confs.reshape(confs.shape[0], -1)
    flat_weights = weights.reshape(confs.shape[0], -1)
    for chart_index in range(flat_confs.shape[0]):
        chart_confs = flat_confs[chart_index]
        valid = torch.isfinite(chart_confs)
        if not valid.any():
            continue
        valid_values = chart_confs[valid]
        conf_min = valid_values.min()
        conf_max = valid_values.max()
        if torch.isclose(conf_max, conf_min):
            flat_weights[chart_index][valid] = 1.0
            continue
        normalized = ((chart_confs[valid] - conf_min) / (conf_max - conf_min)).clamp(0.0, 1.0)
        flat_weights[chart_index][valid] = (
            chart_prior_conf_weight_floor
            + (1.0 - chart_prior_conf_weight_floor) * normalized.pow(chart_prior_conf_weight_power)
        )
    return weights


def combine_optional_weight_masks(*masks: torch.Tensor) -> torch.Tensor:
    valid_masks = [mask.to(dtype=torch.float32) for mask in masks if mask is not None]
    if not valid_masks:
        return None

    combined = valid_masks[0]
    for mask in valid_masks[1:]:
        if mask.shape != combined.shape:
            raise RuntimeError(f"Weight mask shape {tuple(mask.shape)} does not match {tuple(combined.shape)}")
        combined = combined * mask
    return combined


def resize_spatial_mask(mask: torch.Tensor, shape: tuple[int, int], mode: str = "nearest") -> torch.Tensor:
    if mask is None:
        return None
    if mask.shape[-2:] == shape:
        return mask
    original_dtype = mask.dtype
    leading_shape = mask.shape[:-2]
    flat_mask = mask.to(dtype=torch.float32).reshape(-1, 1, *mask.shape[-2:])
    if mode == "nearest":
        resized = F.interpolate(flat_mask, size=shape, mode=mode)
    else:
        resized = F.interpolate(flat_mask, size=shape, mode=mode, align_corners=False)
    resized = resized.reshape(*leading_shape, *shape)
    if original_dtype == torch.bool:
        return resized > 0.5
    return resized.to(dtype=original_dtype)


def _robust_standardize_depths(depths: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    standardized = torch.zeros_like(depths)
    for index in range(depths.shape[0]):
        valid_values = depths[index][valid[index]]
        if valid_values.numel() == 0:
            continue
        median = valid_values.median()
        mad = (valid_values - median).abs().median().clamp_min(eps)
        standardized[index] = (depths[index] - median) / mad
        standardized[index] = torch.where(valid[index], standardized[index], torch.zeros_like(standardized[index]))
    return standardized


def build_relative_depth_consistency_weights(
    reference_depths: torch.Tensor,
    candidate_depths: torch.Tensor,
    min_weight: float = 0.25,
    sigma: float = 0.5,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    if min_weight < 0.0 or min_weight > 1.0:
        raise RuntimeError(f"Depth consistency min_weight must be in [0, 1], got {min_weight}")
    if sigma <= 0.0:
        raise RuntimeError(f"Depth consistency sigma must be positive, got {sigma}")
    reference_depths = reference_depths.to(dtype=torch.float32)
    candidate_depths = candidate_depths.to(device=reference_depths.device, dtype=torch.float32)
    if reference_depths.ndim == 4 and reference_depths.shape[1] == 1:
        reference_depths = reference_depths[:, 0]
    if candidate_depths.ndim == 4 and candidate_depths.shape[1] == 1:
        candidate_depths = candidate_depths[:, 0]
    if reference_depths.ndim != 3 or candidate_depths.ndim != 3:
        raise RuntimeError(
            f"Depth consistency expects tensors shaped (N,H,W), got {tuple(reference_depths.shape)} and {tuple(candidate_depths.shape)}"
        )
    if candidate_depths.shape[0] != reference_depths.shape[0]:
        raise RuntimeError(
            f"Depth consistency view count mismatch: {candidate_depths.shape[0]} vs {reference_depths.shape[0]}"
        )
    if candidate_depths.shape[-2:] != reference_depths.shape[-2:]:
        candidate_depths = resize_spatial_mask(candidate_depths, reference_depths.shape[-2:], mode="bilinear")

    valid = torch.isfinite(reference_depths) & torch.isfinite(candidate_depths)
    if valid_mask is not None:
        valid = valid & resize_spatial_mask(valid_mask.to(dtype=torch.bool), reference_depths.shape[-2:], mode="nearest")

    reference_standard = _robust_standardize_depths(reference_depths, valid)
    candidate_standard = _robust_standardize_depths(candidate_depths, valid)
    difference = (reference_standard - candidate_standard).abs()
    consistency = torch.exp(-difference.square() / (2.0 * sigma ** 2))
    weights = min_weight + (1.0 - min_weight) * consistency
    return torch.where(valid, weights, torch.zeros_like(weights))


def load_depth_consistency_npz(path: str, key: str = "depths") -> torch.Tensor:
    data = np.load(path)
    if key not in data:
        raise RuntimeError(f"Depth consistency file {path} does not contain key {key!r}")
    return torch.from_numpy(data[key]).to(dtype=torch.float32)


def load_chart_consistency_weights_npz(
    path: str,
    key: str = "consistency_weights",
) -> torch.Tensor:
    """Load explicit chart-prior weights without re-normalizing depth maps."""
    data = np.load(path)
    if key not in data:
        raise RuntimeError(f"Chart consistency file {path} does not contain key {key!r}")
    weights = torch.from_numpy(data[key]).to(dtype=torch.float32)
    if weights.ndim == 4 and weights.shape[1] == 1:
        weights = weights[:, 0]
    if weights.ndim != 3:
        raise RuntimeError(
            "Chart consistency weights must have shape (N,H,W) or (N,1,H,W), "
            f"got {tuple(weights.shape)}"
        )
    if not torch.isfinite(weights).all():
        raise RuntimeError("Chart consistency weights contain non-finite values")
    return weights.clamp(0.0, 1.0)


def load_chart_consensus_depths_npz(
    path: str,
    key: str = "depths",
) -> torch.Tensor:
    """Load native-chart-coordinate consensus depths from a sidecar."""
    data = np.load(path)
    if key not in data:
        raise RuntimeError(f"Chart consensus depth file {path} does not contain key {key!r}")
    depths = torch.from_numpy(data[key]).to(dtype=torch.float32)
    if depths.ndim != 3:
        raise RuntimeError(
            f"Chart consensus depths must have shape (N,H,W), got {tuple(depths.shape)}"
        )
    return depths


def blend_chart_consensus_depths(
    charts_data: dict,
    consensus_depths: torch.Tensor,
    blend: float,
) -> tuple[dict, dict[str, float]]:
    """Apply a fallback-safe consensus correction to chart priors only.

    The sidecar is expressed in the original chart depth coordinates and is
    identical to the native depth wherever peer support was insufficient.
    Applying its *delta* preserves the existing smoothed ``prior_depths`` at
    those fallback pixels, so this cannot accidentally replace every prior
    just because a sidecar was supplied.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError("chart consensus depth blend must lie in [0, 1]")
    source_depths = charts_data["depths"]
    source_prior_depths = charts_data["prior_depths"]
    if source_depths.shape != source_prior_depths.shape:
        raise RuntimeError("Chart depths and prior_depths must have identical shapes")
    if consensus_depths.shape != source_depths.shape:
        raise RuntimeError(
            "Chart consensus depth shape does not match charts_data depths: "
            f"{tuple(consensus_depths.shape)} vs {tuple(source_depths.shape)}"
        )
    consensus_depths = consensus_depths.to(
        device=source_depths.device,
        dtype=source_depths.dtype,
    )
    valid = (
        torch.isfinite(source_depths)
        & torch.isfinite(consensus_depths)
        & (source_depths > 0)
        & (consensus_depths > 0)
    )
    raw_delta = torch.where(
        valid,
        consensus_depths - source_depths,
        torch.zeros_like(source_depths),
    )
    delta = raw_delta * float(blend)
    result = dict(charts_data)
    result["depths"] = source_depths + delta
    result["prior_depths"] = source_prior_depths + delta
    changed = valid & raw_delta.abs().gt(1e-6)
    diagnostics = {
        "blend": float(blend),
        "changed_fraction": float(changed.float().mean().item()),
        "mean_abs_raw_delta": float(raw_delta[changed].abs().mean().item()) if changed.any() else 0.0,
        "max_abs_raw_delta": float(raw_delta[changed].abs().max().item()) if changed.any() else 0.0,
    }
    return result, diagnostics


def compute_large_splat_radius_loss(
    radii: torch.Tensor,
    visibility_filter: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if threshold is None or threshold <= 0.0:
        return torch.zeros((), device=radii.device, dtype=radii.dtype)
    visible_radii = radii[visibility_filter].to(dtype=torch.float32)
    if visible_radii.numel() == 0:
        return torch.zeros((), device=radii.device, dtype=torch.float32)
    excess = torch.clamp_min(visible_radii - threshold, 0.0)
    return excess.mean() / threshold


def should_apply_large_splat_radius_loss(iteration: int, reg: float, start_iter: int = 0) -> bool:
    if reg is None or reg <= 0.0:
        return False
    if start_iter is None:
        return True
    return iteration >= max(int(start_iter), 0)


def compute_masked_depth_order_loss(mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
    if mask is None:
        return compute_depth_order_loss(reduction="mean", **kwargs)
    loss_map = compute_depth_order_loss(reduction="none", **kwargs)
    return masked_mean(loss_map, mask)


def training(
    dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, 
    use_refined_charts, use_mip_filter, dense_data_path, use_chart_view_every_n_iter,
    normal_consistency_from, distortion_from,
    depthanythingv2_checkpoint_dir, depthanything_encoder, 
    dense_regul, cambridge_mask_pickle=None, cambridge_mask_dataset_path=None, cambridge_mask_indices=None,
    cambridge_geometry_mask_pickle=None, cambridge_geometry_mask_dataset_path=None, cambridge_geometry_mask_indices=None,
    rgb_loss_type="l1", rgb_charbonnier_eps=1e-3,
    cambridge_rgb_mask_floor=0.0, cambridge_rgb_mask_feather=0,
    rgb_robust_quantile=None, rgb_robust_start_iter=0, rgb_robust_min_weight=0.0,
    use_color_correction=False, color_correction_lr=1e-3, color_correction_reg=1e-2,
    use_appearance_embedding=False, appearance_embedding_dim=16, appearance_hidden_dim=32,
    use_transient_model=False, transient_grid_size=16, transient_lr=1e-3,
    transient_alpha_reg=0.1, transient_residual_reg=0.01, transient_confidence_reg=0.001,
    chart_depth_consistency_npz=None, chart_depth_consistency_key="depths",
    chart_depth_consistency_min_weight=0.25, chart_depth_consistency_sigma=0.5,
    chart_consistency_weight_npz=None, chart_consistency_weight_key="consistency_weights",
    chart_consensus_depth_npz=None, chart_consensus_depth_key="depths",
    chart_consensus_depth_blend=0.25,
    large_splat_radius_reg=0.0, large_splat_radius_threshold=20.0, large_splat_radius_from=0,
    max_chart_gaussians=None, max_chart_abs_point=None, max_chart_abs_depth=None,
    min_chart_conf=None, min_chart_conf_quantile=None,
    chart_prior_conf_weight_floor=None, chart_prior_conf_weight_power=1.0,
    chart_prior_reliability_json=None,
    chart_prior_quality_weight_floor=1.0, chart_prior_quality_weight_power=0.0,
    chart_init_sampler="random", chart_init_scale_policy="legacy_linear",
    chart_init_scale_multiplier_override=None,
    chart_init_voxel_bins=96, chart_init_seed=0, chart_reliability_json=None,
    chart_init_reliability_weight=4.0,
    chart_init_selection_json=None,
    chart_prior_selection_json=None,
    chart_late_prior_selection_json=None,
    chart_late_prior_selection_from=3001,
    early_stop_iteration=None,
):
    
    save_log_images = False
    save_log_images_every_n_iter = 200
    
    if args.exact_resume is not None:
        raise RuntimeError("native observer never restores checkpoints")
    exact = Controller(args, locals())
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    
    # Sparse data
    scene = Scene(dataset, gaussians, shuffle=False)
    
    viewpoint_cams = scene.getTrainCameras()

    geometry_mask_lookup = None
    if cambridge_geometry_mask_pickle is None and cambridge_geometry_mask_indices is not None:
        cambridge_geometry_mask_pickle = cambridge_mask_pickle
    if cambridge_geometry_mask_pickle is not None:
        if cambridge_geometry_mask_dataset_path is None:
            cambridge_geometry_mask_dataset_path = cambridge_mask_dataset_path or dense_data_path or dataset.source_path
        geometry_mask_lookup = CambridgeMaskLookup(
            cambridge_geometry_mask_dataset_path,
            cambridge_geometry_mask_pickle,
            mask_indices=cambridge_geometry_mask_indices or [0, 1, 2],
        )
        print(
            f"[INFO] Using Cambridge geometry/depth masks from {cambridge_geometry_mask_pickle} "
            f"with indices {geometry_mask_lookup.mask_indices}"
        )

    # Dense data
    if dense_data_path is not None:
        use_dense_supervision = True
        
        print(f"[INFO] Loading dense supervision data from: {dense_data_path}")
        dense_dataset = copy.deepcopy(dataset)
        dense_dataset.source_path = dense_data_path
        dense_dataset.model_path = os.path.join(dataset.model_path, 'dense_data')
        os.makedirs(dense_dataset.model_path, exist_ok=True)
        dense_gaussians = GaussianModel(dataset.sh_degree)
        dense_scene = Scene(dense_dataset, dense_gaussians, shuffle=False)
        
        dense_viewpoint_cams = dense_scene.getTrainCameras()
        dense_viewpoint_idx_stack = None
        print(f"          > Number of dense cameras: {len(dense_viewpoint_cams)}")
        
        print(f"          > Changing spatial lr scale from {gaussians.spatial_lr_scale} to {dense_gaussians.spatial_lr_scale}")
        gaussians.spatial_lr_scale = dense_gaussians.spatial_lr_scale
        
        print(f"          > Resolution of dense data: {dense_viewpoint_cams[0].original_image.shape}")
    else:
        use_dense_supervision = False
        use_chart_view_every_n_iter = 1
    print(f"[INFO] Charts will be used for regularization every {use_chart_view_every_n_iter} iteration(s).")
    
    # ===================================================================================
    # Create gaussians from charts data
    print("[INFO] Loading charts data...")
    charts_data_path = f'{dataset.source_path}/charts_data.npz'
    if use_refined_charts:
        charts_data_path = f'{dataset.source_path}/refined_charts_data.npz'
    print("Using charts data from: ", charts_data_path)
    charts_data = load_charts_data(charts_data_path)
    charts_data['confs'] = charts_data['confs'] # - 1.  # Was not there before
    print("[WARNING] Confidence values are not being subtracted by 1.0 as in the original implementation.")
    print("Minimum confidence: ", charts_data['confs'].min())
    print("Maximum confidence: ", charts_data['confs'].max())
    # The strict G4Splat gate encodes rejected chart pixels with -2.0 and the
    # mesh initializer already uses ``conf_th=-1``.  Keep the same invariant
    # for every downstream consumer: a rejected chart must not re-enter via
    # its depth/normal prior during the early chart-regularized phase.
    chart_hard_gate_mask = torch.isfinite(charts_data["confs"]) & (charts_data["confs"] > -1.0)
    chart_hard_gate_active = chart_hard_gate_mask.reshape(
        chart_hard_gate_mask.shape[0], -1
    ).any(dim=1)
    print(
        "[INFO] Hard chart gate keeps "
        f"{int(chart_hard_gate_active.sum().item())}/{len(chart_hard_gate_active)} chart(s) "
        "for initialization and regularization."
    )
    create_gaussians_from_charts_data = True
    chart_outlier_mask = build_chart_outlier_mask(
        charts_data,
        max_abs_point=max_chart_abs_point,
        max_abs_depth=max_chart_abs_depth,
    )
    if chart_outlier_mask is not None:
        print(
            f"[INFO] Chart outlier mask keeps {chart_outlier_mask.float().mean().item() * 100:.2f}% "
            f"of chart pixels (max_abs_point={max_chart_abs_point}, max_abs_depth={max_chart_abs_depth})."
        )
    chart_confidence_mask = build_chart_confidence_mask(
        charts_data,
        min_chart_conf=min_chart_conf,
        min_chart_conf_quantile=min_chart_conf_quantile,
    )
    if chart_confidence_mask is not None:
        print(
            f"[INFO] Chart confidence mask keeps {chart_confidence_mask.float().mean().item() * 100:.2f}% "
            f"of chart pixels (min_chart_conf={min_chart_conf}, "
            f"min_chart_conf_quantile={min_chart_conf_quantile})."
        )
    
    if create_gaussians_from_charts_data:    
        h_charts, w_charts = charts_data['pts'].shape[-3:-1]
        charts_data_for_gaussian_init = charts_data
        geometry_masks_for_init = None
        if geometry_mask_lookup is not None:
            geometry_masks_for_init = get_cambridge_masks_for_cameras(
                geometry_mask_lookup,
                viewpoint_cams,
                (h_charts, w_charts),
                charts_data['confs'].device,
            )
            print(
                f"[INFO] Geometry mask keeps {geometry_masks_for_init.float().mean().item() * 100:.2f}% "
                "of chart initialization pixels."
            )
        selected_chart_mask = build_chart_initialization_selection_mask(
            chart_init_selection_json,
            viewpoint_cams,
            (h_charts, w_charts),
            charts_data["confs"].device,
        )
        init_chart_mask = combine_optional_masks(
            chart_hard_gate_mask,
            geometry_masks_for_init,
            chart_outlier_mask,
            chart_confidence_mask,
            selected_chart_mask,
        )
        charts_data_for_gaussian_init = apply_geometry_mask_to_charts_data(
            charts_data,
            init_chart_mask,
            invalid_conf=-2.0,
        )
        _images = [
            torch.nn.functional.interpolate(cam.original_image[None].cuda(), (h_charts, w_charts), mode="bilinear", antialias=True)[0].permute(1, 2, 0)
            for cam in viewpoint_cams
        ]
        chart_reliabilities = load_chart_reliabilities(
            chart_reliability_json,
            viewpoint_cams,
            charts_data["confs"].device,
        )
        gaussian_params = get_gaussian_parameters_from_charts_data(
            charts_data=charts_data_for_gaussian_init,
            images=_images, 
            conf_th=-1.,  # TODO: Try higher values
            ratio_th=5.,
            normal_scale=1e-10,
            normalized_scales=0.5,
            return_chart_ids=True,
        )
        
        n_max_gaussians = max_chart_gaussians if max_chart_gaussians is not None else -1
        if n_max_gaussians > -1 and n_max_gaussians < len(gaussian_params['means']):
            print(
                f"Downsampling {len(gaussian_params['means'])} gaussians to {n_max_gaussians} "
                f"with {chart_init_sampler}..."
            )
            sample_idx, sampling_report = select_chart_initialization_indices(
                gaussian_params["means"],
                gaussian_params["scales"],
                n_max_gaussians,
                source_chart_ids=gaussian_params.get("chart_ids"),
                chart_reliabilities=chart_reliabilities,
                sampler=chart_init_sampler,
                voxel_bins=chart_init_voxel_bins,
                seed=chart_init_seed,
                reliability_weight=chart_init_reliability_weight,
            )
        else:
            print(f"Using all {len(gaussian_params['means'])} gaussians...")
            sampling_report = {
                "sampler": "identity",
                "input_count": len(gaussian_params["means"]),
                "output_count": len(gaussian_params["means"]),
            }
            sample_idx = torch.arange(
                len(gaussian_params['means']),
                device=gaussian_params["means"].device,
                dtype=torch.long,
            )
        computed_scale_factor = chart_scale_multiplier(
            len(gaussian_params["means"]),
            len(sample_idx),
            chart_init_scale_policy,
        )
        if chart_init_scale_multiplier_override is not None:
            if chart_init_scale_multiplier_override <= 0:
                raise ValueError("chart_init_scale_multiplier_override must be positive")
            scale_factor = float(chart_init_scale_multiplier_override)
            scale_multiplier_source = "override"
        else:
            scale_factor = computed_scale_factor
            scale_multiplier_source = "policy"
        sampling_report["scale_policy"] = chart_init_scale_policy
        sampling_report["scale_multiplier"] = scale_factor
        sampling_report["computed_scale_multiplier"] = computed_scale_factor
        sampling_report["scale_multiplier_source"] = scale_multiplier_source
        # ``source_chart_count`` is retained for compatibility with existing
        # experiment reports, but it is the number of cameras in the chart
        # scene—not necessarily the number that contributed surfels.  A
        # hard-gated scene still has 40 cameras while only 37 may have valid
        # geometry, and a quality-aware initializer can reduce that further.
        sampling_report["source_chart_count"] = len(viewpoint_cams)
        source_chart_ids = gaussian_params.get("chart_ids")
        if source_chart_ids is not None:
            sampling_report["geometry_source_chart_count"] = int(
                torch.unique(source_chart_ids).numel()
            )
            sampling_report["sampled_source_chart_count"] = int(
                torch.unique(source_chart_ids[sample_idx]).numel()
            )
            # A scalar reliability report is not enough to verify that a
            # quality-aware sampler actually changed the initialization.  Keep
            # the per-camera contribution so a later render regression can be
            # traced to a concrete chart rather than inferred from a global
            # mean.  This is deliberately recorded after every gate and
            # optional selection mask have been applied.
            source_counts = torch.bincount(
                source_chart_ids.detach().to(dtype=torch.long, device="cpu"),
                minlength=len(viewpoint_cams),
            )
            sampled_counts = torch.bincount(
                source_chart_ids[sample_idx].detach().to(dtype=torch.long, device="cpu"),
                minlength=len(viewpoint_cams),
            )
            reliability_values = (
                chart_reliabilities.detach().to(dtype=torch.float32, device="cpu")
                if chart_reliabilities is not None
                else torch.ones(len(viewpoint_cams), dtype=torch.float32)
            )
            contributions = []
            for chart_index, camera in enumerate(viewpoint_cams):
                input_count = int(source_counts[chart_index].item())
                if input_count == 0:
                    continue
                output_count = int(sampled_counts[chart_index].item())
                contributions.append(
                    {
                        "chart_index": chart_index,
                        "image_name": str(camera.image_name),
                        "reliability": float(reliability_values[chart_index].item()),
                        "input_gaussians": input_count,
                        "sampled_gaussians": output_count,
                        "sampled_fraction": float(output_count / input_count),
                    }
                )
            sampling_report["chart_source_contributions"] = contributions
        sampling_report_path = os.path.join(dataset.model_path, "chart_initialization_sampling.json")
        with open(sampling_report_path, "w", encoding="utf-8") as handle:
            json.dump(sampling_report, handle, indent=2)
        print(f"[INFO] Chart initialization sampling: {sampling_report}")
        
        _means = gaussian_params['means'][sample_idx]
        _scales = gaussian_params['scales'][..., :2][sample_idx] * scale_factor
        _quaternions = gaussian_params['quaternions'][sample_idx]
        _colors = gaussian_params['colors'][sample_idx]
        if use_dense_supervision:
            with torch.no_grad():
                _means = torch.cat([_means, dense_gaussians.get_xyz.detach()], dim=0)
                _scales = torch.cat([_scales, dense_gaussians.get_scaling.detach()], dim=0)
                _quaternions = torch.cat([_quaternions, dense_gaussians.get_rotation.detach()], dim=0)
                _colors = torch.cat([_colors, SH2RGB(dense_gaussians._features_dc.detach()[:, 0])], dim=0)
        gaussians.create_from_parameters(_means, _scales, _quaternions, _colors, gaussians.spatial_lr_scale)
        print("[INFO] Gaussians created from charts data.")
    
    # Delete unused variables
    if use_dense_supervision:
        del dense_gaussians, dense_scene
    del _means, _scales, _quaternions, _colors, gaussian_params, sample_idx
    gc.collect()
    torch.cuda.empty_cache()
    
    # ===================================================================================
    
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_idx_stack = None
    chart_sampling_phase = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_prior_depth_for_log = 0.0
    ema_prior_normal_for_log = 0.0
    ema_prior_curvature_for_log = 0.0
    ema_prior_anisotropy_for_log = 0.0
    
    # Set mip filter
    if use_mip_filter:
        print("[INFO] Using mip filter during training.")
        gaussians.set_mip_filter(use_mip_filter)
        gaussians.compute_mip_filter(cameras=dense_viewpoint_cams if use_dense_supervision else viewpoint_cams)
    
    # ===================================================================================
    # Build priors from charts data
    print("[INFO] Building priors from charts data...")
    charts_data_for_priors = charts_data
    if chart_consensus_depth_npz is not None:
        consensus_depths = load_chart_consensus_depths_npz(
            chart_consensus_depth_npz,
            key=chart_consensus_depth_key,
        )
        charts_data_for_priors, consensus_depth_report = blend_chart_consensus_depths(
            charts_data,
            consensus_depths,
            chart_consensus_depth_blend,
        )
        print(
            "[INFO] Applied direct cross-view consensus depth correction to chart priors: "
            f"file={chart_consensus_depth_npz}, key={chart_consensus_depth_key}, "
            f"blend={consensus_depth_report['blend']:.3f}, "
            f"changed={consensus_depth_report['changed_fraction'] * 100:.2f}%, "
            f"mean_abs_delta={consensus_depth_report['mean_abs_raw_delta']:.5f}."
        )
    charts_priors = build_priors_from_charts_data(
        charts_data_for_priors,
        viewpoint_cams,
        normal_chunk_size=4,
    )
    charts_scale_factor = charts_priors['scale_factor']
    charts_prior_depths = charts_priors['prior_depths']
    charts_depths = charts_priors['depths']
    charts_confs = charts_priors['confs']
    charts_normals = charts_priors['normals']
    charts_curvs = charts_priors['curvs']
    chart_geometry_masks = None
    if geometry_mask_lookup is not None:
        chart_geometry_masks = get_cambridge_masks_for_cameras(
            geometry_mask_lookup,
            viewpoint_cams,
            charts_depths.shape[-2:],
            charts_depths.device,
        )
        print(
            f"[INFO] Geometry mask keeps {chart_geometry_masks.float().mean().item() * 100:.2f}% "
            "of chart prior pixels."
        )
    chart_confidence_weights = build_chart_confidence_weights(
        charts_data,
        chart_prior_conf_weight_floor=chart_prior_conf_weight_floor,
        chart_prior_conf_weight_power=chart_prior_conf_weight_power,
    )
    if chart_confidence_weights is not None:
        print(
            f"[INFO] Using chart confidence prior weights with floor={chart_prior_conf_weight_floor}, "
            f"power={chart_prior_conf_weight_power}."
        )
    prior_shape = charts_depths.shape[-2:]
    chart_hard_gate_prior_mask = resize_spatial_mask(
        chart_hard_gate_mask,
        prior_shape,
        mode="nearest",
    )
    chart_outlier_prior_mask = resize_spatial_mask(chart_outlier_mask, prior_shape, mode="nearest")
    chart_confidence_prior_mask = resize_spatial_mask(chart_confidence_mask, prior_shape, mode="nearest")
    chart_confidence_prior_weights = resize_spatial_mask(chart_confidence_weights, prior_shape, mode="bilinear")
    chart_prior_selection_mask = build_chart_prior_selection_mask(
        chart_prior_selection_json,
        viewpoint_cams,
        prior_shape,
        charts_depths.device,
    )
    chart_late_prior_selection_mask = build_chart_late_prior_selection_mask(
        chart_late_prior_selection_json,
        viewpoint_cams,
        prior_shape,
        charts_depths.device,
    )
    chart_prior_weights = combine_optional_weight_masks(
        chart_hard_gate_prior_mask,
        chart_geometry_masks,
        chart_outlier_prior_mask,
        chart_confidence_prior_mask,
        chart_confidence_prior_weights,
        chart_prior_selection_mask,
    )
    if chart_depth_consistency_npz is not None:
        external_depths = load_depth_consistency_npz(
            chart_depth_consistency_npz,
            key=chart_depth_consistency_key,
        ).to(device=charts_depths.device)
        chart_depth_consistency_weights = build_relative_depth_consistency_weights(
            charts_depths[:, 0] if charts_depths.ndim == 4 and charts_depths.shape[1] == 1 else charts_depths,
            external_depths,
            min_weight=chart_depth_consistency_min_weight,
            sigma=chart_depth_consistency_sigma,
            valid_mask=chart_prior_weights,
        )
        chart_prior_weights = combine_optional_weight_masks(
            chart_prior_weights,
            chart_depth_consistency_weights,
        )
        print(
            f"[INFO] Using external chart depth consistency weights from {chart_depth_consistency_npz} "
            f"key={chart_depth_consistency_key}, mean={chart_depth_consistency_weights.mean().item() * 100:.2f}%."
        )
    if chart_consistency_weight_npz is not None:
        explicit_chart_consistency_weights = load_chart_consistency_weights_npz(
            chart_consistency_weight_npz,
            key=chart_consistency_weight_key,
        ).to(device=charts_depths.device)
        if explicit_chart_consistency_weights.shape[0] != len(viewpoint_cams):
            raise RuntimeError(
                "Chart consistency view count does not match chart cameras: "
                f"{explicit_chart_consistency_weights.shape[0]} vs {len(viewpoint_cams)}"
            )
        explicit_chart_consistency_weights = resize_spatial_mask(
            explicit_chart_consistency_weights,
            prior_shape,
            mode="bilinear",
        )
        chart_prior_weights = combine_optional_weight_masks(
            chart_prior_weights,
            explicit_chart_consistency_weights,
        )
        print(
            "[INFO] Applied explicit cross-view chart consistency weights from "
            f"{chart_consistency_weight_npz} key={chart_consistency_weight_key}, "
            f"mean={explicit_chart_consistency_weights.mean().item() * 100:.2f}%."
        )
    if chart_prior_reliability_json is not None:
        chart_prior_reliabilities = load_chart_reliabilities(
            chart_prior_reliability_json,
            viewpoint_cams,
            charts_depths.device,
        )
        chart_quality_weights = chart_quality_attenuation(
            chart_prior_reliabilities,
            floor=chart_prior_quality_weight_floor,
            power=chart_prior_quality_weight_power,
        )
        quality_shape = (len(chart_quality_weights),) + (1,) * (chart_prior_weights.ndim - 1)
        chart_prior_weights = chart_prior_weights * chart_quality_weights.view(quality_shape)
        print(
            "[INFO] Applied quality-aware chart-prior attenuation: "
            f"floor={chart_prior_quality_weight_floor:.3f}, "
            f"power={chart_prior_quality_weight_power:.3f}, "
            f"min={chart_quality_weights.min().item():.3f}, "
            f"mean={chart_quality_weights.mean().item():.3f}, "
            f"max={chart_quality_weights.max().item():.3f}."
        )
    elif (
        chart_prior_quality_weight_floor != 1.0
        or chart_prior_quality_weight_power != 0.0
    ):
        raise ValueError(
            "chart_prior_quality_weight_* requires --chart_prior_reliability_json"
        )
    chart_regularization_indices = [
        chart_index
        for chart_index in range(len(viewpoint_cams))
        if bool(chart_prior_weights[chart_index].gt(0).any().item())
    ]
    if not chart_regularization_indices:
        raise RuntimeError("Hard chart gate removed every chart regularization view")
    print(
        "[INFO] Chart regularization will sample only "
        f"{len(chart_regularization_indices)}/{len(viewpoint_cams)} enabled chart(s)."
    )
    chart_late_prior_weights = None
    chart_late_regularization_indices = None
    if chart_late_prior_selection_mask is not None:
        if chart_late_prior_selection_from is None or int(chart_late_prior_selection_from) < 1:
            raise ValueError(
                "chart_late_prior_selection_from must be positive when "
                "chart_late_prior_selection_json is set"
            )
        chart_late_prior_weights = combine_optional_weight_masks(
            chart_prior_weights,
            chart_late_prior_selection_mask,
        )
        chart_late_regularization_indices = [
            chart_index
            for chart_index in range(len(viewpoint_cams))
            if bool(chart_late_prior_weights[chart_index].gt(0).any().item())
        ]
        if not chart_late_regularization_indices:
            raise RuntimeError("Late Chart selection removed every enabled chart regularization view")
        print(
            "[INFO] Late Chart regularization will sample only "
            f"{len(chart_late_regularization_indices)}/{len(viewpoint_cams)} enabled chart(s) "
            f"from iteration {int(chart_late_prior_selection_from)}."
        )
    if chart_prior_weights is not None and (chart_outlier_mask is not None or chart_confidence_mask is not None):
        print(
            f"[INFO] Combined chart prior weights have mean {chart_prior_weights.float().mean().item() * 100:.2f}%."
        )
    print("[INFO] Charts priors built.")
    
    dense_depth_priors = None
    if use_dense_supervision and dense_regul != "none":
        print("[INFO] Building depth priors from dense data...")
        # TODO: Build priors from dense data
        from matcha.pointmap.depthanythingv2 import load_model as load_depthanythingv2
        from matcha.pointmap.depthanythingv2 import apply_depthanything

        dav2 = load_depthanythingv2(checkpoint_dir=depthanythingv2_checkpoint_dir, encoder=depthanything_encoder, device='cuda')
        dav2.eval()
        dense_depth_priors = []
        with torch.no_grad():
            for i_image in range(len(dense_viewpoint_cams)):
                gt_image = dense_viewpoint_cams[i_image].original_image.permute(1, 2, 0)
                supervision_disparity = apply_depthanything(dav2, image=gt_image)
                supervision_disparity = (supervision_disparity - supervision_disparity.min()) / (supervision_disparity.max() - supervision_disparity.min())
                supervision_depth = 1. / (0.1 + 0.9 * supervision_disparity)
                dense_depth_priors.append(supervision_depth.squeeze().unsqueeze(0).to('cpu'))
                if (i_image + 1) % 100 == 0 or i_image + 1 == len(dense_viewpoint_cams):
                    print(
                        "[INFO] Dense depth prior progress "
                        f"{i_image + 1}/{len(dense_viewpoint_cams)}."
                    )
        del dav2
        gc.collect()
        torch.cuda.empty_cache()
        print("[INFO] Depth priors for dense data built.")
    # ===================================================================================
    
    print(f"\n[INFO] Normal consistency from iteration {normal_consistency_from} with lambda_normal {opt.lambda_normal}")
    print(f"[INFO] Distortion from iteration {distortion_from} with lambda_dist {opt.lambda_dist}")
    if use_dense_supervision:
        print(f"[INFO] Dense regularization strength: {dense_regul}")
    # TODO: Should the sparse depth order regularization be used during dense supervision? It's not clear.
    # use_depth_order_regularization = not use_dense_supervision
    use_depth_order_regularization = True
    if use_depth_order_regularization:
        print(f"[INFO] Using depth order regularization for charts.")

    color_correction = None
    color_correction_optimizer = None
    if use_color_correction or use_appearance_embedding:
        color_correction_image_names = [camera.image_name for camera in viewpoint_cams]
        if use_dense_supervision:
            color_correction_image_names.extend(camera.image_name for camera in dense_viewpoint_cams)
        if use_appearance_embedding:
            color_correction = PerImageAppearanceEmbeddingColorCorrection(
                color_correction_image_names,
                embedding_dim=appearance_embedding_dim,
                hidden_dim=appearance_hidden_dim,
            ).cuda()
            color_correction_name = "appearance embedding color correction"
        else:
            color_correction = PerImageAffineColorCorrection(color_correction_image_names).cuda()
            color_correction_name = "affine color correction"
        color_correction_optimizer = torch.optim.Adam(
            color_correction.parameters(),
            lr=color_correction_lr,
        )
        print(
            f"[INFO] Using per-image {color_correction_name} for "
            f"{len(color_correction.image_name_to_index)} image(s), "
            f"lr={color_correction_lr}, reg={color_correction_reg}."
        )

    transient_model = None
    transient_optimizer = None
    if use_transient_model:
        transient_image_names = [camera.image_name for camera in viewpoint_cams]
        if use_dense_supervision:
            transient_image_names.extend(camera.image_name for camera in dense_viewpoint_cams)
        transient_model = PerImageTransientModel(
            transient_image_names,
            grid_size=transient_grid_size,
        ).cuda()
        transient_optimizer = torch.optim.Adam(transient_model.parameters(), lr=transient_lr)
        print(
            f"[INFO] Using per-image transient model for "
            f"{len(transient_model.image_name_to_index)} image(s), "
            f"grid={transient_grid_size}, lr={transient_lr}, "
            f"reg=(alpha={transient_alpha_reg}, residual={transient_residual_reg}, "
            f"confidence={transient_confidence_reg})."
        )

    rgb_mask_lookup = None
    if cambridge_mask_pickle is not None:
        if cambridge_mask_dataset_path is None:
            cambridge_mask_dataset_path = dense_data_path or dataset.source_path
        rgb_mask_lookup = CambridgeMaskLookup(
            cambridge_mask_dataset_path,
            cambridge_mask_pickle,
            mask_indices=cambridge_mask_indices or [0, 2],
        )
        print(
            f"[INFO] Using Cambridge RGB masks from {cambridge_mask_pickle} "
            f"with indices {rgb_mask_lookup.mask_indices}"
        )

    assert exact.prepare(locals()) is None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    exact.restore_rng_last()
    exact.verify_restored_boundary(locals())
    for iteration in range(first_iter, opt.iterations + 1):        

        exact.before_step(iteration)
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera index
        if iteration % use_chart_view_every_n_iter == 0:
            phase, active_chart_indices = chart_regularization_indices_for_iteration(
                iteration,
                chart_regularization_indices,
                late_indices=chart_late_regularization_indices,
                late_from=chart_late_prior_selection_from,
            )
            if phase != chart_sampling_phase:
                viewpoint_idx_stack = None
                chart_sampling_phase = phase
                print(
                    f"[INFO] Chart regularization sampling switched to {phase} pool "
                    f"({len(active_chart_indices)} chart(s)) at iteration {iteration}."
                )
            if not viewpoint_idx_stack:
                viewpoint_idx_stack = list(active_chart_indices)
            viewpoint_idx = viewpoint_idx_stack.pop(randint(0, len(viewpoint_idx_stack)-1))
            viewpoint_cam = viewpoint_cams[viewpoint_idx]
        else:
            if not dense_viewpoint_idx_stack:
                dense_viewpoint_idx_stack = list(range(len(dense_viewpoint_cams)))
            viewpoint_idx = dense_viewpoint_idx_stack.pop(randint(0, len(dense_viewpoint_idx_stack)-1))
            viewpoint_cam = dense_viewpoint_cams[viewpoint_idx]
        
        exact.observe_view(locals())
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        rgb_mask = None
        if rgb_mask_lookup is not None:
            rgb_mask = rgb_mask_lookup.get_mask(viewpoint_cam.image_name, gt_image.shape[-2:], gt_image.device)
            rgb_mask = mask_to_weight(
                rgb_mask,
                invalid_weight=cambridge_rgb_mask_floor,
                feather_pixels=cambridge_rgb_mask_feather,
            ).to(device=gt_image.device, dtype=gt_image.dtype)
        image_for_rgb_loss = image
        if color_correction is not None:
            image_for_rgb_loss = color_correction(image, viewpoint_cam.image_name)
        transient_weight = None
        if transient_model is not None:
            image_for_rgb_loss, transient_weight = transient_model(image_for_rgb_loss, viewpoint_cam.image_name)
        rgb_losses = compute_rgb_loss(
            image_for_rgb_loss,
            gt_image,
            rgb_mask,
            opt.lambda_dssim,
            loss_type=rgb_loss_type,
            charbonnier_eps=rgb_charbonnier_eps,
            rgb_robust_quantile=rgb_robust_quantile,
            rgb_robust_start_iter=rgb_robust_start_iter,
            rgb_robust_min_weight=rgb_robust_min_weight,
            iteration=iteration,
            transient_weight=transient_weight,
        )
        Ll1 = rgb_losses["Ll1"]
        loss = rgb_losses["loss"]
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > normal_consistency_from else 0.0
        lambda_dist = opt.lambda_dist if iteration > distortion_from else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        
        # loss
        total_loss = loss + dist_loss + normal_loss
        if color_correction is not None:
            total_loss = total_loss + color_correction_reg * color_correction.identity_regularization()
        if transient_model is not None:
            total_loss = total_loss + transient_model.regularization(
                viewpoint_cam.image_name,
                alpha_weight=transient_alpha_reg,
                residual_weight=transient_residual_reg,
                confidence_weight=transient_confidence_reg,
            )
        
        # ===================================================================================
        surf_depth = render_pkg['surf_depth']
        total_regularization_loss = 0.
        lambda_anisotropy = 0.1  # 0.01 works well for depth_ratio = 0.
        anisotropy_max_ratio = 5.
        
        if iteration % use_chart_view_every_n_iter == 0:
            rend_curvature = normal2curv(render_pkg['rend_normal'], torch.ones_like(render_pkg['rend_normal'][0:1]))
            active_chart_prior_weights = (
                chart_late_prior_weights
                if chart_sampling_phase == "late"
                else chart_prior_weights
            )
            chart_geometry_mask = (
                active_chart_prior_weights[viewpoint_idx]
                if active_chart_prior_weights is not None
                else None
            )
            
            # ---Charts regularization---
            initial_regularization_factor = 0.5
            regularization_factor = schedule_regularization_factor_2(iteration, initial_regularization_factor)
            lambda_prior_depth = regularization_factor * 0.75
            lambda_prior_depth_derivative = regularization_factor * 0.5  # TODO: Changed. Was 0. before.
            lambda_prior_normal = regularization_factor * 0.5
            lambda_prior_curvature = regularization_factor * 0.25  # 0.5?
            
            confidence_weighting = 0.5
            
            # TODO: Weight charts_normals with rendered alpha, as rendered normals (and surface normals) are not normalized
            
            if False:
                depths_to_regularize = [surf_depth]
            else:
                # depth_to_regularize = render_pkg['rend_depth']
                depths_to_regularize = [surf_depth, render_pkg['rend_depth']]
            
            # Depth regularization
            if False:  # New (should be used with depth_ratio = 1.)
                depth_prior_loss = torch.zeros_like(total_loss)
                for depth_to_regularize in depths_to_regularize:
                    depth_term = (
                        # confidence_weighting * charts_confs[viewpoint_idx] *  # (1, h, w)
                        1.5 * (1. - torch.exp(-(charts_confs[viewpoint_idx] - 1.) ** 2 / 2.)) *  # (1, h, w)
                        # 1.5 * confidence_to_weight(charts_confs[viewpoint_idx]) *  # (1, h, w)
                        torch.log(1. + charts_scale_factor * (charts_depths[viewpoint_idx] - depth_to_regularize).abs())
                        # (charts_scale_factor * (charts_depths[viewpoint_idx] - surf_depth).abs())
                    )
                    depth_prior_loss += lambda_prior_depth * masked_mean(depth_term, chart_geometry_mask)
                    if lambda_prior_depth_derivative > 0:
                        surf_normal_to_regularize = depth_to_normal(viewpoint_cam, depth_to_regularize)
                        surf_normal_to_regularize = surf_normal_to_regularize.permute(2,0,1)
                        depth_derivative_term = (
                            torch.exp(-(charts_confs[viewpoint_idx] - 1.) ** 2 / 2.)
                            # * (1. - confidence_to_weight(charts_confs[viewpoint_idx]))
                            * (1. - (surf_normal_to_regularize * charts_normals[viewpoint_idx]).sum(dim=0))
                        )
                        depth_prior_loss += lambda_prior_depth_derivative * masked_mean(
                            depth_derivative_term,
                            chart_geometry_mask,
                        )
            else:  # Old (should be used with depth_ratio = 0. or 0.5)
                # Depth regularization
                depth_term = (
                    confidence_weighting * charts_confs[viewpoint_idx] *  # (1, h, w)
                    torch.log(1. + charts_scale_factor * (charts_depths[viewpoint_idx] - surf_depth).abs())
                    # (charts_scale_factor * (charts_depths[viewpoint_idx] - surf_depth).abs())
                )
                depth_prior_loss = lambda_prior_depth * masked_mean(depth_term, chart_geometry_mask)
                if lambda_prior_depth_derivative > 0:
                    depth_derivative_term = (
                        torch.exp(-(charts_confs[viewpoint_idx] - 1.) ** 2 / 2.)
                        * (1. - (surf_normal * charts_normals[viewpoint_idx]).sum(dim=0))
                    )
                    depth_prior_loss += lambda_prior_depth_derivative * masked_mean(
                        depth_derivative_term,
                        chart_geometry_mask,
                    )
            
            # Normal regularization
            normal_prior_loss = lambda_prior_normal * masked_mean(
                1. - (rend_normal * charts_normals[viewpoint_idx]).sum(dim=0),
                chart_geometry_mask,
            )
            
            # Curvature regularization
            curv_prior_loss = lambda_prior_curvature * masked_mean(
                (charts_curvs[viewpoint_idx] - rend_curvature).abs(),
                chart_geometry_mask,
            )
            # TODO: Should the curvature be applied to the surf normal?
            
            # Depth order regularization
            if use_depth_order_regularization:
                # TODO: Hyperparameters and scheduling should not be hardcoded
                depth_order_loss_max_pixel_shift_ratio = 0.05
                depth_order_loss_log_space = True
                depth_order_loss_log_scale = 20.
                
                # Scheduling
                lambda_depth_order = 0.
                if iteration > 1_500:
                    lambda_depth_order = 1.
                if iteration > 3_000:
                    lambda_depth_order = 0.1
                if iteration > 4_500:
                    lambda_depth_order = 0.01
                if iteration > 6_000:
                    lambda_depth_order = 0.001
                    
                # Compute depth prior loss
                order_supervision_depth = charts_prior_depths[viewpoint_idx].to(surf_depth.device)
                if lambda_depth_order > 0:
                    depth_order_prior_loss = lambda_depth_order * compute_masked_depth_order_loss(
                        depth=surf_depth, 
                        prior_depth=order_supervision_depth, 
                        mask=chart_geometry_mask,
                        scene_extent=gaussians.spatial_lr_scale,  # TODO: Check if this is correct
                        max_pixel_shift_ratio=depth_order_loss_max_pixel_shift_ratio,
                        normalize_loss=True,
                        log_space=depth_order_loss_log_space,
                        log_scale=depth_order_loss_log_scale,
                        debug=False,
                    )
                else:
                    depth_order_prior_loss = torch.zeros_like(loss.detach())
                depth_prior_loss = depth_prior_loss + depth_order_prior_loss
            
            # Total loss
            total_regularization_loss = depth_prior_loss + normal_prior_loss + curv_prior_loss
        
        else:
            # TODO: Hyperparameters and scheduling should not be hardcoded
            dense_depth_loss_max_pixel_shift_ratio = 0.05
            dense_depth_loss_log_space = True
            dense_depth_loss_log_scale = 20.
            lambda_anisotropy = 0.                
            dense_depth_regularization_schedule = dense_regul
            
            # ---Scheduling: default---
            # Gives more weight to the dense depth regularization than charts regularization,
            # and progressively decreases the weight of the dense depth regularization.
            if dense_depth_regularization_schedule == "default":
                lambda_dense_depth = 0.  # TODO: Should it be 0. or 1.?
                use_chart_view_every_n_iter = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter = 999_999
                    lambda_dense_depth = 1.
                if iteration > 7_000:
                    lambda_dense_depth = 0.1
                if iteration > 15_000:
                    lambda_dense_depth = 0.01
                if iteration > 20_000:
                    lambda_dense_depth = 0.001
                if iteration > 25_000:
                    lambda_dense_depth = 0.0001
                
            # ---Scheduling: strong---
            # Removes the charts regularization and only uses a strong dense depth regularization.
            elif dense_depth_regularization_schedule == "strong":
                # lambda_dense_depth = 0.
                lambda_dense_depth = 1.
                use_chart_view_every_n_iter = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter = 999_999
                    lambda_dense_depth = 1.
                    
            # ---Scheduling: weak---
            # Applies a weak dense depth regularization, while still using the charts regularization.
            elif dense_depth_regularization_schedule == "weak":
                lambda_dense_depth = 0.
                use_chart_view_every_n_iter = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter = 15
                    lambda_dense_depth = 0.1

            # ---Scheduling: weak_dense_only---
            # Matches ``weak``'s dense-depth amplitude after the anchor phase,
            # but stops Chart regularization. This is a causal ablation for
            # separating schedule gains from late-Chart gains.
            elif dense_depth_regularization_schedule == "weak_dense_only":
                lambda_dense_depth = 0.
                use_chart_view_every_n_iter = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter = 999_999
                    lambda_dense_depth = 0.1

            # ---Scheduling: weak_dense_only_decay---
            # Keep the screen-validated weak dense-only anchor phase exactly
            # unchanged through 7k iterations, then progressively relax the
            # pseudo-depth term.  The dense prediction is valuable for
            # resolving early geometry, but retaining its 7k amplitude for a
            # long photometric refinement would continue to bias appearance
            # and opacity toward an imperfect monocular prior.
            elif dense_depth_regularization_schedule == "weak_dense_only_decay":
                lambda_dense_depth = 0.
                use_chart_view_every_n_iter = 5
                if iteration > 3_000:
                    use_chart_view_every_n_iter = 999_999
                    lambda_dense_depth = 0.1
                if iteration > 7_000:
                    lambda_dense_depth = 0.01
                if iteration > 15_000:
                    lambda_dense_depth = 0.001
                if iteration > 20_000:
                    lambda_dense_depth = 0.0001
                if iteration > 25_000:
                    lambda_dense_depth = 0.00001
            
            # ---Scheduling: none---
            # Disables the dense depth regularization, while still using the charts regularization.
            elif dense_depth_regularization_schedule == "none":
                lambda_dense_depth = 0.
                use_chart_view_every_n_iter = 5
                    
            else:
                raise ValueError(f"Invalid dense depth regularization schedule: {dense_depth_regularization_schedule}")
            
            # Compute depth prior loss
            dense_supervision_depth = (
                dense_depth_priors[viewpoint_idx].to(surf_depth.device)
                if lambda_dense_depth > 0 else None
            )
            dense_geometry_mask = None
            if geometry_mask_lookup is not None:
                dense_geometry_mask = geometry_mask_lookup.get_mask(
                    viewpoint_cam.image_name,
                    surf_depth.shape[-2:],
                    surf_depth.device,
                )
            if lambda_dense_depth > 0:
                if dense_supervision_depth is None:
                    raise RuntimeError(
                        "Dense depth regularization is enabled but no dense depth prior was built"
                    )
                depth_prior_loss = lambda_dense_depth * compute_masked_depth_order_loss(
                    depth=surf_depth, 
                    prior_depth=dense_supervision_depth, 
                    mask=dense_geometry_mask,
                    scene_extent=gaussians.spatial_lr_scale,  # TODO: Check if this is correct
                    max_pixel_shift_ratio=dense_depth_loss_max_pixel_shift_ratio,
                    normalize_loss=True,
                    log_space=dense_depth_loss_log_space,
                    log_scale=dense_depth_loss_log_scale,
                    debug=False,
                )
            else:
                depth_prior_loss = torch.zeros_like(loss.detach())
            
            # TODO: Add dense regularization for normal and curvature
            normal_prior_loss = torch.zeros_like(loss.detach())
            curv_prior_loss = torch.zeros_like(loss.detach())
            
            # Total loss
            total_regularization_loss = depth_prior_loss + normal_prior_loss + curv_prior_loss
            
        # Anisotropy regularization
        if lambda_anisotropy > 0.:
            gaussians_scaling = gaussians.get_scaling
            anisotropy_loss = lambda_anisotropy * (
                torch.clamp_min(gaussians_scaling.max(dim=1).values / gaussians_scaling.min(dim=1).values, anisotropy_max_ratio) 
                - anisotropy_max_ratio
            ).mean()
            total_regularization_loss = total_regularization_loss + anisotropy_loss

        if should_apply_large_splat_radius_loss(iteration, large_splat_radius_reg, large_splat_radius_from):
            large_splat_loss = large_splat_radius_reg * compute_large_splat_radius_loss(
                radii,
                visibility_filter,
                threshold=large_splat_radius_threshold,
            )
            total_regularization_loss = total_regularization_loss + large_splat_loss
        
        total_loss = total_loss + total_regularization_loss
        
        # ===================================================================================

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_prior_depth_for_log = 0.4 * depth_prior_loss.item() + 0.6 * ema_prior_depth_for_log
            ema_prior_normal_for_log = 0.4 * normal_prior_loss.item() + 0.6 * ema_prior_normal_for_log
            ema_prior_curvature_for_log = 0.4 * curv_prior_loss.item() + 0.6 * ema_prior_curvature_for_log
            if lambda_anisotropy > 0.:
                ema_prior_anisotropy_for_log = 0.4 * anisotropy_loss.item() + 0.6 * ema_prior_anisotropy_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz.detach())}",
                    "p_depth": f"{ema_prior_depth_for_log:.{5}f}",
                    "p_normal": f"{ema_prior_normal_for_log:.{5}f}",
                    "p_curvature": f"{ema_prior_curvature_for_log:.{5}f}"
                }
                if lambda_anisotropy > 0:
                    loss_dict["aniso"] = f"{ema_prior_anisotropy_for_log:.{5}f}"
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
                
            if (iteration % save_log_images_every_n_iter == 0) or (iteration == 1):
                # Save log image with rgb, depth, normal, curvature
                if iteration % use_chart_view_every_n_iter == 0:
                    supervision_depth = charts_depths[viewpoint_idx]
                    supervision_normal = charts_normals[viewpoint_idx]
                else:
                    supervision_depth = (
                        dense_supervision_depth.squeeze().unsqueeze(0)
                        if dense_supervision_depth is not None
                        else torch.zeros_like(surf_depth)
                    )
                    supervision_normal = torch.zeros_like(charts_normals[0])
                
                if save_log_images:
                    figsize = 30
                    height, width = gt_image.shape[-2:]
                    nrows = 2
                    ncols = 3
                    plt.figure(figsize=(figsize, figsize * height / width * nrows / ncols))
                    plt.subplot(nrows, ncols, 1)
                    plt.title("GT Image")
                    plt.imshow(gt_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    plt.subplot(nrows, ncols, 2)
                    plt.title("Charts Depth")
                    plt.imshow(supervision_depth[0].cpu().numpy(), cmap="Spectral")
                    plt.colorbar()
                    plt.subplot(nrows, ncols, 3)
                    plt.title("Charts Normal")
                    plt.imshow((-supervision_normal + 1).permute(1, 2, 0).clamp(0, 2).cpu().numpy() / 2)
                    plt.subplot(nrows, ncols, 4)
                    plt.title("Rendered Image")
                    plt.imshow(image.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy())
                    plt.subplot(nrows, ncols, 5)
                    plt.title("Rendered Depth")
                    plt.imshow(surf_depth.detach()[0].cpu().numpy(), cmap="Spectral")
                    plt.colorbar()
                    plt.subplot(nrows, ncols, 6)
                    plt.title("Rendered Normal")
                    plt.imshow((-rend_normal.detach() + 1).permute(1, 2, 0).clamp(0, 2).cpu().numpy() / 2)
                    # save image
                    plt.savefig(f"{dataset.model_path}/{iteration}.png")
                    plt.close()
                
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Keep the original optimization horizon (and therefore all
            # learning-rate schedules) while materializing an early-stop
            # prior.  Setting --iterations to the stop point is not
            # equivalent when schedules depend on opt.iterations.
            if early_stop_iteration is not None and iteration >= early_stop_iteration:
                progress_bar.close()
                print(
                    "\n[ITER {}] Early-stop checkpoint materialized; "
                    "leaving the 30k schedule unchanged.".format(iteration)
                )
                break


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(cameras=dense_viewpoint_cams if use_dense_supervision else viewpoint_cams)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:  # don't update in the end of training
                    torch.cuda.empty_cache()
                    if gaussians.use_mip_filter:
                        gaussians.compute_mip_filter(cameras=dense_viewpoint_cams if use_dense_supervision else viewpoint_cams)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                if color_correction_optimizer is not None:
                    color_correction_optimizer.step()
                if transient_optimizer is not None:
                    transient_optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if color_correction_optimizer is not None:
                    color_correction_optimizer.zero_grad(set_to_none=True)
                if transient_optimizer is not None:
                    transient_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if network_gui.conn is not None:
                raise RuntimeError("exact-resume: live GUI is unsupported")
            if exact.boundary(locals()):
                progress_bar.close()
                break

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
                if network_gui.conn is not None:
                    raise RuntimeError("exact-resume: GUI connection refused before render")
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=None)  # 6009
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument(
        "--early_stop_iteration",
        type=int,
        default=None,
        help=(
            "Stop immediately after this iteration is saved without changing "
            "the configured optimization horizon or learning-rate schedules."
        ),
    )
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--use_refined_charts", action="store_true", default=False)
    parser.add_argument("--use_mip_filter", action="store_true", default=False)
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument("--use_chart_view_every_n_iter", type=int, default=999_999)
    parser.add_argument("--normal_consistency_from", type=int, default=3500)
    parser.add_argument("--distortion_from", type=int, default=1500)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str, default='../Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default', help='Strength of dense regularization. Can be "default", "strong", "weak", "weak_dense_only", "weak_dense_only_decay", or "none".')
    parser.add_argument("--cambridge_mask_pickle", type=str, default=None)
    parser.add_argument("--cambridge_mask_dataset_path", type=str, default=None)
    parser.add_argument("--cambridge_mask_indices", nargs="*", type=int, default=None)
    parser.add_argument("--cambridge_geometry_mask_pickle", type=str, default=None)
    parser.add_argument("--cambridge_geometry_mask_dataset_path", type=str, default=None)
    parser.add_argument("--cambridge_geometry_mask_indices", nargs="*", type=int, default=None)
    parser.add_argument("--rgb_loss_type", choices=["l1", "charbonnier"], default="l1")
    parser.add_argument("--rgb_charbonnier_eps", type=float, default=1e-3)
    parser.add_argument("--cambridge_rgb_mask_floor", type=float, default=0.0)
    parser.add_argument("--cambridge_rgb_mask_feather", type=int, default=0)
    parser.add_argument("--rgb_robust_quantile", type=float, default=None)
    parser.add_argument("--rgb_robust_start_iter", type=int, default=0)
    parser.add_argument("--rgb_robust_min_weight", type=float, default=0.0)
    parser.add_argument("--use_color_correction", action="store_true", default=False)
    parser.add_argument("--color_correction_lr", type=float, default=1e-3)
    parser.add_argument("--color_correction_reg", type=float, default=1e-2)
    parser.add_argument("--use_appearance_embedding", action="store_true", default=False)
    parser.add_argument("--appearance_embedding_dim", type=int, default=16)
    parser.add_argument("--appearance_hidden_dim", type=int, default=32)
    parser.add_argument("--use_transient_model", action="store_true", default=False)
    parser.add_argument("--transient_grid_size", type=int, default=16)
    parser.add_argument("--transient_lr", type=float, default=1e-3)
    parser.add_argument("--transient_alpha_reg", type=float, default=0.1)
    parser.add_argument("--transient_residual_reg", type=float, default=0.01)
    parser.add_argument("--transient_confidence_reg", type=float, default=0.001)
    parser.add_argument("--chart_depth_consistency_npz", type=str, default=None)
    parser.add_argument("--chart_depth_consistency_key", type=str, default="depths")
    parser.add_argument("--chart_depth_consistency_min_weight", type=float, default=0.25)
    parser.add_argument("--chart_depth_consistency_sigma", type=float, default=0.5)
    parser.add_argument(
        "--chart_consistency_weight_npz",
        type=str,
        default=None,
        help=(
            "Optional explicit (N,H,W) cross-view chart-prior weight sidecar. "
            "Unlike chart_depth_consistency_npz, it is multiplied directly."
        ),
    )
    parser.add_argument(
        "--chart_consistency_weight_key",
        type=str,
        default="consistency_weights",
    )
    parser.add_argument(
        "--chart_consensus_depth_npz",
        type=str,
        default=None,
        help=(
            "Optional fallback-safe (N,H,W) multi-view consensus depth sidecar. "
            "It modifies chart regularization targets only, never initialization."
        ),
    )
    parser.add_argument("--chart_consensus_depth_key", type=str, default="depths")
    parser.add_argument(
        "--chart_consensus_depth_blend",
        type=float,
        default=0.25,
        help="Interpolation from native chart depth to consensus depth in [0,1].",
    )
    parser.add_argument("--large_splat_radius_reg", type=float, default=0.0)
    parser.add_argument("--large_splat_radius_threshold", type=float, default=20.0)
    parser.add_argument("--large_splat_radius_from", type=int, default=0)
    parser.add_argument("--max_chart_gaussians", type=int, default=None)
    parser.add_argument("--max_chart_abs_point", type=float, default=None)
    parser.add_argument("--max_chart_abs_depth", type=float, default=None)
    parser.add_argument("--min_chart_conf", type=float, default=None)
    parser.add_argument("--min_chart_conf_quantile", type=float, default=None)
    parser.add_argument("--chart_prior_conf_weight_floor", type=float, default=None)
    parser.add_argument("--chart_prior_conf_weight_power", type=float, default=1.0)
    parser.add_argument(
        "--chart_prior_reliability_json",
        type=str,
        default=None,
        help=(
            "Optional quality sidecar used only to attenuate early chart depth/normal "
            "priors; it does not alter chart-initialization sampling."
        ),
    )
    parser.add_argument(
        "--chart_prior_quality_weight_floor",
        type=float,
        default=1.0,
        help="Minimum multiplier for a hard-gated chart prior (default: neutral 1.0).",
    )
    parser.add_argument(
        "--chart_prior_quality_weight_power",
        type=float,
        default=0.0,
        help="Reliability exponent for chart-prior attenuation (default: neutral 0.0).",
    )
    parser.add_argument(
        "--chart_init_sampler",
        choices=["random", "voxel_stratified"],
        default="random",
        help="How to cap chart-initialized surfels; voxel_stratified preserves 3D support.",
    )
    parser.add_argument(
        "--chart_init_scale_policy",
        choices=["legacy_linear", "sqrt", "none"],
        default="legacy_linear",
        help="Scale multiplier after chart surfel capping.",
    )
    parser.add_argument(
        "--chart_init_scale_multiplier_override",
        type=float,
        default=None,
        help=(
            "Optional positive fixed multiplier for a controlled chart-selection "
            "ablation; overrides the multiplier implied by chart_init_scale_policy."
        ),
    )
    parser.add_argument(
        "--chart_init_voxel_bins",
        type=int,
        default=96,
        help="Coarse bins per world axis for voxel_stratified chart sampling.",
    )
    parser.add_argument("--chart_init_seed", type=int, default=0)
    parser.add_argument(
        "--chart_reliability_json",
        type=str,
        default=None,
        help="Optional quality-aware chart score JSON available to the in-voxel sampler.",
    )
    parser.add_argument(
        "--chart_init_reliability_weight",
        type=float,
        default=4.0,
        help=(
            "Strength of chart reliability in the in-voxel tie breaker; set 0 for "
            "geometry-only sampling while retaining the score sidecar for audit."
        ),
    )
    parser.add_argument(
        "--chart_init_selection_json",
        type=str,
        default=None,
        help="Optional selected Chart list; only these charts seed Gaussian geometry.",
    )
    parser.add_argument(
        "--chart_prior_selection_json",
        type=str,
        default=None,
        help=(
            "Optional selected Chart list; unselected charts have zero early depth/normal "
            "prior weight while remaining in the scene for bookkeeping."
        ),
    )
    parser.add_argument(
        "--chart_late_prior_selection_json",
        type=str,
        default=None,
        help=(
            "Optional audited Chart subset used only for late depth/normal regularization; "
            "initialization and early Chart anchoring stay unchanged."
        ),
    )
    parser.add_argument(
        "--chart_late_prior_selection_from",
        type=int,
        default=3001,
        help="First iteration at which --chart_late_prior_selection_json becomes active.",
    )
    add_arguments(parser)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.early_stop_iteration is not None:
        if not 0 < args.early_stop_iteration <= args.iterations:
            parser.error("--early_stop_iteration must lie inside [1, --iterations]")
        args.save_iterations.append(args.early_stop_iteration)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if args.port is None:
        import time
        current_time = time.strftime("%H%M%S", time.localtime())[2:]
        args.port = int(current_time)
        print(f"Randomly selected port: {args.port}")
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args), op.extract(args), pp.extract(args), 
        args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
        args.start_checkpoint, args.use_refined_charts, args.use_mip_filter, 
        args.dense_data_path, args.use_chart_view_every_n_iter,
        args.normal_consistency_from, args.distortion_from,
        args.depthanythingv2_checkpoint_dir, args.depthanything_encoder,
        args.dense_regul, args.cambridge_mask_pickle, args.cambridge_mask_dataset_path,
        args.cambridge_mask_indices, args.cambridge_geometry_mask_pickle,
        args.cambridge_geometry_mask_dataset_path, args.cambridge_geometry_mask_indices,
        args.rgb_loss_type, args.rgb_charbonnier_eps,
        args.cambridge_rgb_mask_floor, args.cambridge_rgb_mask_feather,
        args.rgb_robust_quantile, args.rgb_robust_start_iter, args.rgb_robust_min_weight,
        args.use_color_correction, args.color_correction_lr, args.color_correction_reg,
        args.use_appearance_embedding, args.appearance_embedding_dim, args.appearance_hidden_dim,
        args.use_transient_model, args.transient_grid_size, args.transient_lr,
        args.transient_alpha_reg, args.transient_residual_reg, args.transient_confidence_reg,
        args.chart_depth_consistency_npz, args.chart_depth_consistency_key,
        args.chart_depth_consistency_min_weight, args.chart_depth_consistency_sigma,
        args.chart_consistency_weight_npz, args.chart_consistency_weight_key,
        args.chart_consensus_depth_npz, args.chart_consensus_depth_key,
        args.chart_consensus_depth_blend,
        args.large_splat_radius_reg, args.large_splat_radius_threshold, args.large_splat_radius_from,
        args.max_chart_gaussians, args.max_chart_abs_point, args.max_chart_abs_depth,
        args.min_chart_conf, args.min_chart_conf_quantile,
        args.chart_prior_conf_weight_floor, args.chart_prior_conf_weight_power,
        args.chart_prior_reliability_json,
        args.chart_prior_quality_weight_floor, args.chart_prior_quality_weight_power,
        args.chart_init_sampler, args.chart_init_scale_policy,
        args.chart_init_scale_multiplier_override,
        args.chart_init_voxel_bins, args.chart_init_seed, args.chart_reliability_json,
        args.chart_init_reliability_weight,
        args.chart_init_selection_json,
        args.chart_prior_selection_json,
        args.chart_late_prior_selection_json,
        args.chart_late_prior_selection_from,
        args.early_stop_iteration,
    )

    # All done
    print("\nTraining complete.")
