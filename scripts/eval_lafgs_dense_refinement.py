#!/usr/bin/env python
"""Evaluate an experimental dense refinement seeded by saved LaFGS poses.

This is deliberately outside the LaFGS V2 inference path.  It evaluates two
different map-side representations from the exact same saved sparse poses:

* ``lafgs_field`` renders the distilled LaFGS descriptors, with non-bank
  primitives suppressed for both the feature render and the 3D lifting
  geometry.  A complete frozen 2DGS render is still used as an occlusion
  consistency check.
* ``prior_rgb`` renders the complete frozen 2DGS RGB image at the real-query
  encoder resolution, then encodes it with the same frozen image encoder as
  the real query.  A separate feature-grid render supplies PnP geometry.

Ground-truth poses are read only after refinement to report metrics.  The
script writes raw candidates; a separate validation-only gate selector decides
whether an update is accepted for a final test report.
"""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_from_pose_gsplat
from localization_training.pose_refiner import project_points, weighted_gauss_newton_refine
from scripts.lafgs_dense_diagnostics import (
    candidate_displacement_diagnostics,
    gt_local_basin_diagnostics,
    gt_reprojection_diagnostics,
)
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from stdloc import get_intrinsic, lift_2d_to_3d
from utils.camera_utils import loadCam
from utils.image_utils import get_resolution_from_longest_edge
from utils.pose_utils import cal_pose_error, solve_pose


def file_sha256(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_query_valid_masks(path):
    """Load cached image-valid masks used by dense-field training.

    The feature cache is not used as supervision during evaluation.  It only
    carries the precomputed artifact/distortion validity mask so that the
    local candidate graph sees the same admissible query cells as training.
    """
    if not path:
        return {}, {"path": None, "sha256": None, "query_count": 0}
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    queries = payload.get("queries") if isinstance(payload, dict) else None
    if not isinstance(queries, dict):
        raise ValueError("query_cache must contain a queries dictionary")
    masks = {}
    for raw_name, entry in queries.items():
        if not isinstance(entry, dict) or "valid_mask" not in entry:
            raise ValueError(f"query cache entry {raw_name!r} is missing valid_mask")
        mask = torch.as_tensor(entry["valid_mask"], dtype=torch.bool).squeeze()
        if mask.ndim != 2:
            raise ValueError(f"query cache valid mask for {raw_name!r} is not 2D")
        masks[str(raw_name).replace("\\", "/")] = mask.contiguous()
    return masks, {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "query_count": int(len(masks)),
    }


def _resize_query_valid_mask(mask, height, width, device):
    mask = torch.as_tensor(mask, device=device, dtype=torch.float32).squeeze()
    if mask.ndim != 2:
        raise ValueError("query valid mask must be a 2D image")
    if tuple(mask.shape) != (int(height), int(width)):
        mask = F.interpolate(
            mask[None, None],
            size=(int(height), int(width)),
            mode="nearest",
        )[0, 0]
    return mask >= 0.5


def _as_scalar_map(value):
    """Convert renderer scalar outputs such as [1, H, W, 1] to [H, W]."""
    if value is None:
        return None
    value = torch.as_tensor(value)
    value = value.squeeze()
    if value.ndim == 2:
        return value
    if value.ndim == 3:
        if value.shape[0] == 1:
            return value[0]
        if value.shape[-1] == 1:
            return value[..., 0]
    raise ValueError(f"Expected a scalar image map, got shape {tuple(value.shape)}")


def _normalized_feature_map(feature_extractor, image, size):
    with torch.no_grad():
        feature_map = feature_extractor(image[None])["feature_map"]
        feature_map = F.interpolate(
            feature_map, size=size, mode="bilinear", align_corners=False
        )[0]
    return F.normalize(feature_map, p=2, dim=0)


def prior_rgb_encoder_resolution(source_image_size, feature_height, feature_width):
    """Return the image size at which a rendered RGB query is encoded.

    The real branch feeds the encoder the original query image and only then
    interpolates its descriptors to the native feature grid.  Rendering RGB
    directly at that feature-grid resolution changes the encoder receptive
    field and makes the two descriptor domains incomparable.
    """
    if source_image_size is None:
        return int(feature_height), int(feature_width)
    if len(source_image_size) != 2:
        raise ValueError("source_image_size must be (height, width)")
    height, width = (int(source_image_size[0]), int(source_image_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("source_image_size must be positive")
    return height, width


def query_feature_maps(feature_extractor, image, longest_edge, feature_grid="fine"):
    """Extract query descriptors on the same image grid used by the field.

    LaFGS V2's map training caches SuperPoint descriptors at the native
    ``longest_edge / 8`` grid.  Rendering a field trained on that grid at full
    image resolution and comparing it to an upsampled descriptor map shifts
    the feature-cell centers under ``align_corners=False``.  Native-grid
    evaluation avoids that coordinate-system change.
    """
    height, width = get_resolution_from_longest_edge(
        image.shape[-2], image.shape[-1], longest_edge
    )
    native_size = (max(height // 8, 1), max(width // 8, 1))
    native = _normalized_feature_map(feature_extractor, image, native_size)
    if feature_grid == "native":
        return native, None
    if feature_grid != "fine":
        raise ValueError(f"Unsupported feature grid: {feature_grid}")
    fine = _normalized_feature_map(feature_extractor, image, (height, width))
    coarse = F.interpolate(
        fine[None], size=(height // 8, width // 8), mode="bilinear", align_corners=False
    )[0]
    return F.normalize(fine, p=2, dim=0), F.normalize(coarse, p=2, dim=0)


def _masked_dual_softmax(similarity, query_valid, rendered_valid, temperature):
    """Dual softmax without allowing invalid rendered cells into either norm."""
    if similarity.ndim != 3:
        raise ValueError("similarity must have shape [B, N, M]")
    query_valid = torch.as_tensor(query_valid, device=similarity.device, dtype=torch.bool)
    rendered_valid = torch.as_tensor(
        rendered_valid, device=similarity.device, dtype=torch.bool
    )
    if query_valid.ndim == 1:
        query_valid = query_valid[None].expand(similarity.shape[0], -1)
    if rendered_valid.ndim == 1:
        rendered_valid = rendered_valid[None].expand(similarity.shape[0], -1)
    if query_valid.shape != similarity.shape[:2] or rendered_valid.shape != (
        similarity.shape[0],
        similarity.shape[2],
    ):
        raise ValueError("valid masks do not match similarity")
    valid = query_valid[:, :, None] & rendered_valid[:, None, :]
    logits = similarity / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~valid, -1e4)
    corr = torch.softmax(logits, dim=-1) * torch.softmax(logits, dim=-2)
    return torch.where(valid, corr, torch.zeros_like(corr))


def _mnn_pairs(corr, threshold=0.0):
    """Return stable batched mutual-nearest pairs and their dual-softmax score."""
    if corr.ndim != 3:
        raise ValueError("corr must have shape [B, N, M]")
    if corr.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=corr.device)
        return empty, empty, empty, corr.new_empty(0)
    mask = corr > float(threshold)
    mask &= corr == corr.max(dim=-1, keepdim=True).values
    mask &= corr == corr.max(dim=-2, keepdim=True).values
    batch, left, right = torch.where(mask)
    return batch, left, right, corr[batch, left, right]


def _coarse_valid_mask(fine_valid, cell_size, min_fraction):
    height, width = fine_valid.shape[-2:]
    if height % cell_size or width % cell_size:
        raise ValueError("fine feature size must be divisible by the coarse cell size")
    pooled = F.avg_pool2d(
        fine_valid.float()[None, None], kernel_size=cell_size, stride=cell_size
    )[0, 0]
    return pooled >= float(min_fraction)


def build_dense_matches(
    query_fine,
    query_coarse,
    rendered_fine,
    rendered_valid,
    *,
    coarse_temperature,
    fine_temperature,
    coarse_threshold,
    fine_threshold,
    valid_cell_fraction,
    max_coarse_matches,
    max_dense_matches,
    query_valid=None,
):
    """Coarse-to-fine matching with renderer-validity-aware dual softmax."""
    channels, fine_h, fine_w = query_fine.shape
    _, coarse_h, coarse_w = query_coarse.shape
    cell = fine_h // coarse_h
    if fine_h != coarse_h * cell or fine_w != coarse_w * cell or cell <= 0:
        raise ValueError("incompatible query fine/coarse feature shapes")
    if rendered_fine.shape != query_fine.shape:
        raise ValueError("rendered and query fine feature shapes must agree")
    if rendered_valid.shape != (fine_h, fine_w):
        raise ValueError("rendered validity shape does not match fine feature map")
    if query_valid is None:
        query_valid = torch.ones(
            (fine_h, fine_w), device=query_fine.device, dtype=torch.bool
        )
    else:
        query_valid = torch.as_tensor(
            query_valid, device=query_fine.device, dtype=torch.bool
        ).squeeze()
        if query_valid.shape != (fine_h, fine_w):
            raise ValueError("query validity shape does not match fine feature map")

    rendered_coarse = F.interpolate(
        rendered_fine[None], size=(coarse_h, coarse_w), mode="bilinear", align_corners=False
    )[0]
    rendered_coarse = F.normalize(rendered_coarse, p=2, dim=0)
    coarse_valid = _coarse_valid_mask(rendered_valid, cell, valid_cell_fraction)
    query_coarse_valid = _coarse_valid_mask(query_valid, cell, valid_cell_fraction)
    if int(coarse_valid.sum().item()) < 4 or int(query_coarse_valid.sum().item()) < 4:
        return None, {
            "coarse_valid_cells": int(coarse_valid.sum().item()),
            "query_coarse_valid_cells": int(query_coarse_valid.sum().item()),
            "coarse_matches": 0,
            "fine_matches": 0,
        }

    query_vectors = query_coarse.permute(1, 2, 0).reshape(1, -1, channels)
    rendered_vectors = rendered_coarse.reshape(1, channels, -1)
    coarse_similarity = torch.matmul(query_vectors, rendered_vectors)
    coarse_corr = _masked_dual_softmax(
        coarse_similarity,
        query_coarse_valid.reshape(-1),
        coarse_valid.reshape(-1),
        coarse_temperature,
    )
    _, coarse_query, coarse_render, coarse_score = _mnn_pairs(
        coarse_corr, threshold=coarse_threshold
    )
    if coarse_query.numel() < 4:
        return None, {
            "coarse_valid_cells": int(coarse_valid.sum().item()),
            "coarse_matches": int(coarse_query.numel()),
            "fine_matches": 0,
        }
    if max_coarse_matches > 0 and coarse_query.numel() > max_coarse_matches:
        keep = torch.topk(coarse_score, k=int(max_coarse_matches), sorted=False).indices
        coarse_query = coarse_query[keep]
        coarse_render = coarse_render[keep]
        coarse_score = coarse_score[keep]

    pixels_per_cell = cell * cell
    query_windows = F.unfold(
        query_fine[None], (cell, cell), stride=cell
    ).reshape(1, channels, pixels_per_cell, -1)[0, :, :, coarse_query].permute(2, 1, 0)
    rendered_windows = F.unfold(
        rendered_fine[None], (cell, cell), stride=cell
    ).reshape(1, channels, pixels_per_cell, -1)[0, :, :, coarse_render].permute(2, 1, 0)
    rendered_valid_windows = F.unfold(
        rendered_valid.float()[None, None], (cell, cell), stride=cell
    ).reshape(pixels_per_cell, -1)[:, coarse_render].transpose(0, 1) >= 0.5
    query_valid_windows = F.unfold(
        query_valid.float()[None, None], (cell, cell), stride=cell
    ).reshape(pixels_per_cell, -1)[:, coarse_query].transpose(0, 1) >= 0.5

    fine_similarity = torch.bmm(query_windows, rendered_windows.transpose(1, 2))
    fine_corr = _masked_dual_softmax(
        fine_similarity,
        query_valid_windows,
        rendered_valid_windows,
        fine_temperature,
    )
    batch, fine_query, fine_render, fine_score = _mnn_pairs(
        fine_corr, threshold=fine_threshold
    )
    if fine_query.numel() < 4:
        return None, {
            "coarse_valid_cells": int(coarse_valid.sum().item()),
            "coarse_matches": int(coarse_query.numel()),
            "fine_matches": int(fine_query.numel()),
        }

    query_xy = torch.stack(
        [
            (coarse_query[batch] % coarse_w) * cell + (fine_query % cell),
            (coarse_query[batch] // coarse_w) * cell + (fine_query // cell),
        ],
        dim=1,
    ).float()
    rendered_xy = torch.stack(
        [
            (coarse_render[batch] % coarse_w) * cell + (fine_render % cell),
            (coarse_render[batch] // coarse_w) * cell + (fine_render // cell),
        ],
        dim=1,
    ).float()
    valid_pairs = (
        rendered_valid[rendered_xy[:, 1].long(), rendered_xy[:, 0].long()]
        & query_valid[query_xy[:, 1].long(), query_xy[:, 0].long()]
    )
    query_xy = query_xy[valid_pairs]
    rendered_xy = rendered_xy[valid_pairs]
    fine_score = fine_score[valid_pairs]
    if max_dense_matches > 0 and fine_score.numel() > max_dense_matches:
        keep = torch.topk(fine_score, k=int(max_dense_matches), sorted=False).indices
        query_xy = query_xy[keep]
        rendered_xy = rendered_xy[keep]
        fine_score = fine_score[keep]
    diagnostics = {
        "coarse_valid_cells": int(coarse_valid.sum().item()),
        "query_coarse_valid_cells": int(query_coarse_valid.sum().item()),
        "coarse_matches": int(coarse_query.numel()),
        "fine_matches": int(query_xy.shape[0]),
        "fine_score_mean": float(fine_score.mean().item()) if fine_score.numel() else 0.0,
    }
    if query_xy.shape[0] < 4:
        return None, diagnostics
    return (query_xy, rendered_xy, fine_score), diagnostics


def _ulfloc_geometric_support(
    query_xy,
    rendered_xy,
    *,
    neighbors,
    angle_thresh_cos,
    scale_thresh,
    scale_limit,
):
    """Score ULF-Loc-style local geometric consistency of coarse matches.

    This is the tensor-only counterpart of ULF-Loc's ``compute_geometric_support``.
    It deliberately operates on the matched image coordinates before fine matching:
    repeated facade cells can have high descriptor scores but usually cannot preserve
    local triangle orientation, angle, and scale simultaneously.  No pose or GT is
    consumed here.
    """
    query_xy = torch.as_tensor(query_xy)
    rendered_xy = torch.as_tensor(rendered_xy, device=query_xy.device, dtype=query_xy.dtype)
    if query_xy.ndim != 2 or rendered_xy.shape != query_xy.shape or query_xy.shape[1] != 2:
        raise ValueError("geometric support expects matching [N, 2] coordinate arrays")
    count = int(query_xy.shape[0])
    if count < 3:
        return query_xy.new_zeros((count,))
    neighbors = min(max(int(neighbors), 1), count - 1)
    if neighbors < 2:
        return query_xy.new_zeros((count,))
    if float(scale_limit) <= 1.0:
        raise ValueError("geometric support scale_limit must exceed one")

    # ULF-Loc uses KNN in the query image, then checks whether the matched
    # rendered neighborhoods preserve local 2D similarity geometry.
    distance = torch.cdist(query_xy, query_xy)
    knn = distance.topk(neighbors + 1, dim=-1, largest=False).indices[:, 1:]
    query_knn = query_xy[knn]
    rendered_knn = rendered_xy[knn]
    query_rel = query_knn - query_xy[:, None, :]
    rendered_rel = rendered_knn - rendered_xy[:, None, :]

    def normalize(vectors):
        return vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-8)

    query_unit = normalize(query_rel)
    rendered_unit = normalize(rendered_rel)
    query_dot = torch.matmul(query_unit, query_unit.transpose(1, 2))
    rendered_dot = torch.matmul(rendered_unit, rendered_unit.transpose(1, 2))
    angle_consistent = (query_dot - rendered_dot).abs() < (1.0 - float(angle_thresh_cos))

    query_j = query_rel[:, :, None, :]
    query_k = query_rel[:, None, :, :]
    rendered_j = rendered_rel[:, :, None, :]
    rendered_k = rendered_rel[:, None, :, :]
    query_cross = query_j[..., 0] * query_k[..., 1] - query_j[..., 1] * query_k[..., 0]
    rendered_cross = (
        rendered_j[..., 0] * rendered_k[..., 1]
        - rendered_j[..., 1] * rendered_k[..., 0]
    )
    orientation_consistent = (query_cross * rendered_cross) > 0.0

    # Degenerate or extremely distant triples should neither support nor veto
    # nearby correspondences.  This follows the 50-pixel guard in ULF-Loc.
    invalid_edge = (
        (query_rel.norm(dim=-1) < 1e-6)
        | (rendered_rel.norm(dim=-1) < 1e-6)
        | (query_rel.norm(dim=-1) > 50.0)
        | (rendered_rel.norm(dim=-1) > 50.0)
    )
    invalid_triangle = invalid_edge[:, :, None] & invalid_edge[:, None, :]
    query_jk = (query_j - query_k).norm(dim=-1)
    rendered_jk = (rendered_j - rendered_k).norm(dim=-1)
    invalid_triangle |= (query_jk < 1e-6) | (rendered_jk < 1e-6)

    query_anchor = query_xy[:, None, None, :]
    rendered_anchor = rendered_xy[:, None, None, :]
    query_a = (query_j - query_anchor).norm(dim=-1) + 1e-8
    query_b = (query_k - query_anchor).norm(dim=-1) + 1e-8
    query_c = query_jk + 1e-8
    rendered_a = (rendered_j - rendered_anchor).norm(dim=-1) + 1e-8
    rendered_b = (rendered_k - rendered_anchor).norm(dim=-1) + 1e-8
    rendered_c = rendered_jk + 1e-8
    scale_a = rendered_a / query_a
    scale_b = rendered_b / query_b
    scale_c = rendered_c / query_c
    scale_consistent = (
        ((scale_a - scale_b).abs() < float(scale_thresh))
        & ((scale_a - scale_c).abs() < float(scale_thresh))
        & ((scale_b - scale_c).abs() < float(scale_thresh))
    )
    lower, upper = 1.0 / float(scale_limit), float(scale_limit)
    scale_consistent &= (
        (scale_a > lower)
        & (scale_a < upper)
        & (scale_b > lower)
        & (scale_b < upper)
        & (scale_c > lower)
        & (scale_c < upper)
    )

    support = angle_consistent & orientation_consistent & scale_consistent & ~invalid_triangle
    diagonal = torch.arange(neighbors, device=query_xy.device)
    support[:, diagonal, diagonal] = False
    return support.float().sum(dim=(1, 2))


def build_ulfloc_dense_matches(
    query_fine,
    query_coarse,
    rendered_fine,
    rendered_valid,
    *,
    coarse_temperature,
    fine_temperature,
    coarse_threshold,
    fine_threshold,
    valid_cell_fraction,
    max_coarse_matches,
    max_dense_matches,
    geometric_filter,
    geometric_neighbors,
    geometric_support_threshold,
    geometric_angle_cos,
    geometric_scale_threshold,
    geometric_scale_limit,
    query_valid=None,
):
    """ULF-Loc's RGB-render coarse-to-fine matcher with explicit validity masks.

    Unlike ``build_local_dense_matches``, this intentionally does a global
    coarse proposal stage, geometric support rejection, then cell-wise fine
    matching.  It is kept separate from the LaFGS-field matcher because it is
    an RGB-prior dense stage and has no dependency on a localization feature
    field.
    """
    channels, fine_h, fine_w = query_fine.shape
    _, coarse_h, coarse_w = query_coarse.shape
    cell = fine_h // coarse_h
    if fine_h != coarse_h * cell or fine_w != coarse_w * cell or cell <= 0:
        raise ValueError("incompatible query fine/coarse feature shapes")
    if rendered_fine.shape != query_fine.shape:
        raise ValueError("rendered and query fine feature shapes must agree")
    if rendered_valid.shape != (fine_h, fine_w):
        raise ValueError("rendered validity shape does not match fine feature map")
    if query_valid is None:
        query_valid = torch.ones((fine_h, fine_w), device=query_fine.device, dtype=torch.bool)
    else:
        query_valid = torch.as_tensor(query_valid, device=query_fine.device, dtype=torch.bool).squeeze()
        if query_valid.shape != (fine_h, fine_w):
            raise ValueError("query validity shape does not match fine feature map")

    rendered_coarse = F.interpolate(
        rendered_fine[None], size=(coarse_h, coarse_w), mode="bilinear", align_corners=False
    )[0]
    rendered_coarse = F.normalize(rendered_coarse, p=2, dim=0)
    rendered_coarse_valid = _coarse_valid_mask(rendered_valid, cell, valid_cell_fraction)
    query_coarse_valid = _coarse_valid_mask(query_valid, cell, valid_cell_fraction)
    base_diagnostics = {
        "coarse_valid_cells": int(rendered_coarse_valid.sum().item()),
        "query_coarse_valid_cells": int(query_coarse_valid.sum().item()),
        "coarse_matches_before_geometric_filter": 0,
        "coarse_matches": 0,
        "fine_matches": 0,
        "ulfloc_geometric_filter": bool(geometric_filter),
    }
    if int(rendered_coarse_valid.sum().item()) < 4 or int(query_coarse_valid.sum().item()) < 4:
        return None, base_diagnostics

    query_vectors = query_coarse.permute(1, 2, 0).reshape(1, -1, channels)
    rendered_vectors = rendered_coarse.reshape(1, channels, -1)
    coarse_similarity = torch.matmul(query_vectors, rendered_vectors)
    coarse_corr = _masked_dual_softmax(
        coarse_similarity,
        query_coarse_valid.reshape(-1),
        rendered_coarse_valid.reshape(-1),
        coarse_temperature,
    )
    _, coarse_query, coarse_render, coarse_score = _mnn_pairs(
        coarse_corr, threshold=coarse_threshold
    )
    base_diagnostics["coarse_matches_before_geometric_filter"] = int(coarse_query.numel())
    if coarse_query.numel() < 4:
        return None, base_diagnostics
    if max_coarse_matches > 0 and coarse_query.numel() > int(max_coarse_matches):
        keep = torch.topk(coarse_score, k=int(max_coarse_matches), sorted=False).indices
        coarse_query = coarse_query[keep]
        coarse_render = coarse_render[keep]
        coarse_score = coarse_score[keep]

    if geometric_filter:
        coarse_query_xy = torch.stack(
            [(coarse_query % coarse_w) * cell, (coarse_query // coarse_w) * cell], dim=1
        ).to(dtype=query_fine.dtype)
        coarse_render_xy = torch.stack(
            [(coarse_render % coarse_w) * cell, (coarse_render // coarse_w) * cell], dim=1
        ).to(dtype=query_fine.dtype)
        support = _ulfloc_geometric_support(
            coarse_query_xy,
            coarse_render_xy,
            neighbors=geometric_neighbors,
            angle_thresh_cos=geometric_angle_cos,
            scale_thresh=geometric_scale_threshold,
            scale_limit=geometric_scale_limit,
        )
        base_diagnostics["ulfloc_geometric_support_mean"] = (
            float(support.mean().item()) if support.numel() else 0.0
        )
        keep = support > float(geometric_support_threshold)
        coarse_query = coarse_query[keep]
        coarse_render = coarse_render[keep]
        coarse_score = coarse_score[keep]
    else:
        base_diagnostics["ulfloc_geometric_support_mean"] = None
    base_diagnostics["coarse_matches"] = int(coarse_query.numel())
    if coarse_query.numel() < 4:
        return None, base_diagnostics

    pixels_per_cell = cell * cell
    query_windows = F.unfold(query_fine[None], (cell, cell), stride=cell).reshape(
        1, channels, pixels_per_cell, -1
    )[0, :, :, coarse_query].permute(2, 1, 0)
    rendered_windows = F.unfold(rendered_fine[None], (cell, cell), stride=cell).reshape(
        1, channels, pixels_per_cell, -1
    )[0, :, :, coarse_render].permute(2, 1, 0)
    query_valid_windows = F.unfold(query_valid.float()[None, None], (cell, cell), stride=cell).reshape(
        pixels_per_cell, -1
    )[:, coarse_query].transpose(0, 1) >= 0.5
    rendered_valid_windows = F.unfold(
        rendered_valid.float()[None, None], (cell, cell), stride=cell
    ).reshape(pixels_per_cell, -1)[:, coarse_render].transpose(0, 1) >= 0.5
    fine_similarity = torch.bmm(query_windows, rendered_windows.transpose(1, 2))
    fine_corr = _masked_dual_softmax(
        fine_similarity, query_valid_windows, rendered_valid_windows, fine_temperature
    )
    batch, fine_query, fine_render, fine_score = _mnn_pairs(fine_corr, threshold=fine_threshold)
    if fine_query.numel() < 4:
        base_diagnostics["fine_matches"] = int(fine_query.numel())
        return None, base_diagnostics

    query_xy = torch.stack(
        [
            (coarse_query[batch] % coarse_w) * cell + (fine_query % cell),
            (coarse_query[batch] // coarse_w) * cell + (fine_query // cell),
        ],
        dim=1,
    ).float()
    rendered_xy = torch.stack(
        [
            (coarse_render[batch] % coarse_w) * cell + (fine_render % cell),
            (coarse_render[batch] // coarse_w) * cell + (fine_render // cell),
        ],
        dim=1,
    ).float()
    valid_pairs = (
        query_valid[query_xy[:, 1].long(), query_xy[:, 0].long()]
        & rendered_valid[rendered_xy[:, 1].long(), rendered_xy[:, 0].long()]
    )
    query_xy = query_xy[valid_pairs]
    rendered_xy = rendered_xy[valid_pairs]
    fine_score = fine_score[valid_pairs]
    if max_dense_matches > 0 and fine_score.numel() > int(max_dense_matches):
        keep = torch.topk(fine_score, k=int(max_dense_matches), sorted=False).indices
        query_xy = query_xy[keep]
        rendered_xy = rendered_xy[keep]
        fine_score = fine_score[keep]
    base_diagnostics.update(
        {
            "fine_matches": int(query_xy.shape[0]),
            "fine_score_mean": float(fine_score.mean().item()) if fine_score.numel() else 0.0,
        }
    )
    if query_xy.shape[0] < 4:
        return None, base_diagnostics
    return (query_xy, rendered_xy, fine_score), base_diagnostics


def build_local_dense_matches(
    query_fine,
    rendered_fine,
    rendered_valid,
    *,
    radius_px,
    anchor_stride,
    temperature,
    batch_size,
    min_similarity,
    max_dense_matches,
    correspondence_mode="hard",
    query_valid=None,
    geometric_filter=False,
    geometric_neighbors=8,
    geometric_support_threshold=4.0,
    geometric_angle_cos=0.9659,
    geometric_scale_threshold=0.1,
    geometric_scale_limit=3.0,
    rendered_anchor_xy=None,
    query_anchor_xy=None,
):
    """Match each rendered anchor only inside a seed-pose local query window.

    A dense refinement is conditioned on the sparse pose used to render the
    map.  The globally matched variant above deliberately remains available as
    a diagnostic, but it can turn repeated facade texture into a coherent,
    wrong global PnP consensus.  This routine instead asks whether the current
    render can be aligned to the image locally around its predicted pixel.
    """
    if query_fine.shape != rendered_fine.shape:
        raise ValueError("rendered and query fine feature shapes must agree")
    channels, height, width = query_fine.shape
    if rendered_valid.shape != (height, width):
        raise ValueError("rendered validity shape does not match fine feature map")
    if query_valid is None:
        query_valid = torch.ones(
            (height, width), device=query_fine.device, dtype=torch.bool
        )
    else:
        query_valid = torch.as_tensor(
            query_valid, device=query_fine.device, dtype=torch.bool
        ).squeeze()
        if query_valid.shape != (height, width):
            raise ValueError("query validity shape does not match fine feature map")
    radius_px = int(radius_px)
    anchor_stride = int(anchor_stride)
    batch_size = int(batch_size)
    if radius_px < 0 or anchor_stride <= 0 or batch_size <= 0:
        raise ValueError("local matching radius/stride/batch_size are invalid")
    if correspondence_mode not in {"hard", "soft"}:
        raise ValueError(f"Unsupported local correspondence mode: {correspondence_mode}")

    if (rendered_anchor_xy is None) != (query_anchor_xy is None):
        raise ValueError(
            "rendered_anchor_xy and query_anchor_xy must be supplied together"
        )
    if rendered_anchor_xy is None:
        grid_y = torch.arange(0, height, anchor_stride, device=query_fine.device)
        grid_x = torch.arange(0, width, anchor_stride, device=query_fine.device)
        yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
        valid_grid = rendered_valid[yy, xx]
        rendered_xy = torch.stack([xx[valid_grid], yy[valid_grid]], dim=1).long()
        query_anchor_xy = rendered_xy.clone()
        anchor_source = "render_grid"
    else:
        rendered_xy = torch.as_tensor(
            rendered_anchor_xy, device=query_fine.device, dtype=torch.float32
        ).reshape(-1, 2).round().long()
        query_anchor_xy = torch.as_tensor(
            query_anchor_xy, device=query_fine.device, dtype=torch.float32
        ).reshape(-1, 2).round().long()
        if rendered_xy.shape != query_anchor_xy.shape:
            raise ValueError("rendered and query anchor arrays must have the same shape")
        inside_render = (
            (rendered_xy[:, 0] >= 0)
            & (rendered_xy[:, 0] < width)
            & (rendered_xy[:, 1] >= 0)
            & (rendered_xy[:, 1] < height)
        )
        rendered_xy = rendered_xy[inside_render]
        query_anchor_xy = query_anchor_xy[inside_render]
        if rendered_xy.numel():
            render_support = rendered_valid[rendered_xy[:, 1], rendered_xy[:, 0]]
            rendered_xy = rendered_xy[render_support]
            query_anchor_xy = query_anchor_xy[render_support]
        anchor_source = "provided"
    initial_anchor_count = int(rendered_xy.shape[0])
    if rendered_xy.shape[0] < 4:
        return None, {
            "coarse_valid_cells": 0,
            "coarse_matches": 0,
            "local_anchor_count": initial_anchor_count,
            "local_anchor_source": anchor_source,
            "fine_matches": 0,
        }
    # Keep the spatial sampling deterministic and bounded before constructing
    # local descriptor windows.  A later score-based cap is still applied.
    if max_dense_matches > 0 and rendered_xy.shape[0] > int(max_dense_matches):
        keep = torch.linspace(
            0,
            rendered_xy.shape[0] - 1,
            int(max_dense_matches),
            device=query_fine.device,
        ).round().long()
        rendered_xy = rendered_xy[keep]
        query_anchor_xy = query_anchor_xy[keep]

    offsets = torch.arange(
        -radius_px, radius_px + 1, device=query_fine.device, dtype=torch.long
    )
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    dx = dx.reshape(1, -1)
    dy = dy.reshape(1, -1)
    query_parts = []
    rendered_parts = []
    score_parts = []
    best_similarity_parts = []
    margin_parts = []
    candidate_count_parts = []
    for start in range(0, rendered_xy.shape[0], batch_size):
        anchors = rendered_xy[start : start + batch_size]
        query_anchors = query_anchor_xy[start : start + batch_size]
        query_x = query_anchors[:, 0:1] + dx
        query_y = query_anchors[:, 1:2] + dy
        valid_window = (
            (query_x >= 0)
            & (query_x < width)
            & (query_y >= 0)
            & (query_y < height)
        )
        query_x = query_x.clamp(0, width - 1)
        query_y = query_y.clamp(0, height - 1)
        valid_window = valid_window & query_valid[query_y, query_x]
        candidate_count = valid_window.sum(dim=1)
        row_has_valid = valid_window.any(dim=1)
        if not bool(row_has_valid.all().item()):
            # Keep the temporary softmax finite.  These rows are excluded
            # below, so the forced candidate cannot enter PnP.
            valid_window = valid_window.clone()
            valid_window[~row_has_valid, 0] = True
        query_windows = query_fine[:, query_y, query_x].permute(1, 2, 0)
        rendered_vectors = rendered_fine[:, anchors[:, 1], anchors[:, 0]].T
        similarity = (query_windows * rendered_vectors[:, None, :]).sum(dim=-1)
        similarity = similarity.masked_fill(~valid_window, -torch.inf)
        topk = torch.topk(similarity, k=min(2, similarity.shape[1]), dim=1)
        best_similarity = topk.values[:, 0]
        best_index = topk.indices[:, 0]
        second_similarity = (
            topk.values[:, 1]
            if topk.values.shape[1] > 1
            else torch.full_like(best_similarity, -1.0)
        )
        best_x = query_x.gather(1, best_index[:, None]).squeeze(1)
        best_y = query_y.gather(1, best_index[:, None]).squeeze(1)
        # The softmax confidence and the top-2 margin are used only to rank
        # correspondences for optional PROSAC; PnP never sees this as a weight.
        local_prob = torch.softmax(
            similarity / max(float(temperature), 1e-6), dim=1
        ).gather(1, best_index[:, None]).squeeze(1)
        # A masked or boundary window can have exactly one valid candidate.
        # Its second top-k value is -inf, which is not evidence of a confident
        # match.  Letting that turn into an infinite margin incorrectly puts
        # these non-informative rows first in PROSAC/RANSAC ordering.
        margin = torch.where(
            torch.isfinite(second_similarity),
            best_similarity - second_similarity,
            torch.zeros_like(best_similarity),
        )
        score = local_prob * margin.clamp_min(0.0)
        if correspondence_mode == "soft":
            probability = torch.softmax(
                similarity / max(float(temperature), 1e-6), dim=1
            )
            selected_x = (probability * query_x.to(dtype=probability.dtype)).sum(dim=1)
            selected_y = (probability * query_y.to(dtype=probability.dtype)).sum(dim=1)
            entropy = -(probability.clamp_min(1e-12).log() * probability).sum(dim=1)
            support = valid_window.sum(dim=1).clamp_min(2).to(dtype=entropy.dtype)
            confidence = (1.0 - entropy / support.log()).clamp(0.0, 1.0)
            score = confidence * margin.clamp_min(0.0)
        else:
            selected_x = best_x.to(dtype=best_similarity.dtype)
            selected_y = best_y.to(dtype=best_similarity.dtype)
        keep = row_has_valid & torch.isfinite(best_similarity) & (
            best_similarity >= float(min_similarity)
        )
        if keep.any():
            query_parts.append(torch.stack([selected_x[keep], selected_y[keep]], dim=1))
            rendered_parts.append(anchors[keep])
            score_parts.append(score[keep])
            best_similarity_parts.append(best_similarity[keep])
            margin_parts.append(margin[keep])
            candidate_count_parts.append(candidate_count[keep])

    if not score_parts:
        return None, {
            "coarse_valid_cells": 0,
            "coarse_matches": 0,
            "local_anchor_count": initial_anchor_count,
            "local_anchor_source": anchor_source,
            "fine_matches": 0,
        }
    query_xy = torch.cat(query_parts, dim=0)
    rendered_xy = torch.cat(rendered_parts, dim=0)
    score = torch.cat(score_parts, dim=0)
    best_similarity = torch.cat(best_similarity_parts, dim=0)
    margin = torch.cat(margin_parts, dim=0)
    candidate_count = torch.cat(candidate_count_parts, dim=0)

    if correspondence_mode == "hard":
        # Several rendered surfels can otherwise collapse onto the same query
        # location.  Retain only its highest-confidence proposal before PnP.
        ranked = torch.argsort(score, descending=True, stable=True)
        query_linear = query_xy[:, 1] * width + query_xy[:, 0]
        ranked_query = query_linear[ranked]
        unique = torch.ones(ranked_query.shape[0], dtype=torch.bool, device=ranked.device)
        unique[1:] = ranked_query[1:] != ranked_query[:-1]
        keep = ranked[unique]
        query_xy = query_xy[keep]
        rendered_xy = rendered_xy[keep]
        score = score[keep]
        best_similarity = best_similarity[keep]
        margin = margin[keep]
        candidate_count = candidate_count[keep]
    if max_dense_matches > 0 and score.numel() > int(max_dense_matches):
        keep = torch.topk(score, k=int(max_dense_matches), sorted=False).indices
        query_xy = query_xy[keep]
        rendered_xy = rendered_xy[keep]
        score = score[keep]
        best_similarity = best_similarity[keep]
        margin = margin[keep]
        candidate_count = candidate_count[keep]

    local_before_lgcv = int(query_xy.shape[0])
    local_lgcv_retained = local_before_lgcv
    if bool(geometric_filter) and query_xy.shape[0] >= 3:
        support = _ulfloc_geometric_support(
            query_xy,
            rendered_xy,
            neighbors=int(geometric_neighbors),
            angle_thresh_cos=float(geometric_angle_cos),
            scale_thresh=float(geometric_scale_threshold),
            scale_limit=float(geometric_scale_limit),
        )
        keep = support >= float(geometric_support_threshold)
        query_xy = query_xy[keep]
        rendered_xy = rendered_xy[keep]
        score = score[keep]
        best_similarity = best_similarity[keep]
        margin = margin[keep]
        candidate_count = candidate_count[keep]
        local_lgcv_retained = int(query_xy.shape[0])

    rounded_query = query_xy.round().long()
    if rounded_query.numel() > 0:
        rounded_query[:, 0].clamp_(0, width - 1)
        rounded_query[:, 1].clamp_(0, height - 1)
        unique_query_cells = torch.unique(rounded_query, dim=0).shape[0]
    else:
        unique_query_cells = 0

    diagnostics = {
        "coarse_valid_cells": 0,
        "coarse_matches": 0,
        "local_anchor_count": initial_anchor_count,
        "local_anchor_source": anchor_source,
        "fine_matches": int(query_xy.shape[0]),
        "fine_score_mean": float(score.mean().item()) if score.numel() else 0.0,
        "local_best_similarity_mean": (
            float(best_similarity.mean().item()) if best_similarity.numel() else 0.0
        ),
        "local_best_margin_mean": float(margin.mean().item()) if margin.numel() else 0.0,
        "local_score_finite_fraction": (
            float(torch.isfinite(score).float().mean().item()) if score.numel() else 1.0
        ),
        "local_single_candidate_count": int((candidate_count == 1).sum().item()),
        "local_query_cell_unique_count": int(unique_query_cells),
        "local_query_cell_unique_fraction": float(
            unique_query_cells / max(int(query_xy.shape[0]), 1)
        ),
        "local_correspondence_mode": correspondence_mode,
        "local_lgcv_filter": bool(geometric_filter),
        "local_matches_before_lgcv": local_before_lgcv,
        "local_lgcv_retained": local_lgcv_retained,
        "local_lgcv_retained_fraction": float(
            local_lgcv_retained / max(local_before_lgcv, 1)
        ),
    }
    if query_xy.shape[0] < 4:
        return None, diagnostics
    return (query_xy.float(), rendered_xy.float(), score), diagnostics


def _resize_pixel_center_coordinates(points_xy, source_width, source_height, width, height):
    """Map pixel-index coordinates through resize with explicit center semantics."""
    points_xy = torch.as_tensor(points_xy, dtype=torch.float32)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape [N, 2]")
    source_width = int(source_width)
    source_height = int(source_height)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source correspondence resolution must be positive")
    scale = points_xy.new_tensor(
        [float(width) / float(source_width), float(height) / float(source_height)]
    )
    return (points_xy + 0.5) * scale - 0.5


def _deduplicate_pair_anchors(rendered_xy, query_xy, seed_scores, width):
    """Keep the highest-scored sparse seed when expanded regions overlap.

    Sparse RANSAC inliers cluster heavily around repeated local texture.  Keeping
    all overlapping expansions would silently turn them into a density weight in
    the dense PnP solve.  This deterministic first-winner rule preserves the
    source score ordering while allowing each rendered grid cell only once.
    """
    if rendered_xy.numel() == 0:
        return rendered_xy, query_xy, seed_scores
    order = torch.argsort(seed_scores, descending=True, stable=True)
    rendered_xy = rendered_xy[order]
    query_xy = query_xy[order]
    seed_scores = seed_scores[order]
    linear = (rendered_xy[:, 1] * int(width) + rendered_xy[:, 0]).detach().cpu().tolist()
    seen = set()
    keep_list = []
    for index, value in enumerate(linear):
        if value not in seen:
            seen.add(value)
            keep_list.append(index)
    keep = torch.as_tensor(keep_list, device=rendered_xy.device, dtype=torch.long)
    return rendered_xy[keep], query_xy[keep], seed_scores[keep]


def build_pair_inlier_anchors(
    sparse_correspondence,
    pose_w2c,
    intrinsic,
    rendered_valid,
    *,
    width,
    height,
    expansion_radius_px,
    expansion_stride_px,
    max_anchors,
):
    """Expand sparse RANSAC inliers into paired local dense anchor regions.

    The query-side anchor is the measured sparse keypoint.  The render-side
    anchor is the same 3D point projected under the current sparse pose.  Each
    pair is locally expanded by the same feature-grid offset, so dense matching
    can refine the local surface while remaining tied to an actual sparse PnP
    inlier.  No GT pose or GT correspondence is involved.
    """
    if not isinstance(sparse_correspondence, dict):
        raise ValueError("sparse_correspondence must be a dictionary")
    query_input = torch.as_tensor(
        sparse_correspondence.get("p2d", []), dtype=torch.float32, device=rendered_valid.device
    ).reshape(-1, 2)
    points3d = torch.as_tensor(
        sparse_correspondence.get("p3d", []), dtype=torch.float32, device=rendered_valid.device
    ).reshape(-1, 3)
    if query_input.shape[0] != points3d.shape[0]:
        raise ValueError("sparse correspondence p2d/p3d counts differ")
    source_width = int(sparse_correspondence.get("width", 0))
    source_height = int(sparse_correspondence.get("height", 0))
    if source_width <= 0 or source_height <= 0:
        raise ValueError("sparse correspondence is missing source width/height")
    source_scores = torch.as_tensor(
        sparse_correspondence.get("scores", []), dtype=torch.float32, device=rendered_valid.device
    ).reshape(-1)
    if source_scores.numel() == 0:
        source_scores = torch.ones(query_input.shape[0], dtype=torch.float32, device=rendered_valid.device)
    if source_scores.shape[0] != query_input.shape[0]:
        raise ValueError("sparse correspondence score count differs from p2d/p3d")
    input_count = int(query_input.shape[0])
    if input_count == 0:
        return None, {
            "pair_inlier_input_count": 0,
            "pair_inlier_projected_count": 0,
            "pair_inlier_anchor_count": 0,
        }

    query_seed = _resize_pixel_center_coordinates(
        query_input, source_width, source_height, width, height
    ).to(device=rendered_valid.device)
    projected_uv, projected_valid = project_points(
        points3d,
        torch.as_tensor(intrinsic, dtype=torch.float32, device=rendered_valid.device),
        torch.as_tensor(pose_w2c, dtype=torch.float32, device=rendered_valid.device),
    )
    rendered_seed = projected_uv - 0.5
    finite = (
        projected_valid
        & torch.isfinite(query_seed).all(dim=1)
        & torch.isfinite(rendered_seed).all(dim=1)
        & torch.isfinite(source_scores)
    )
    query_seed = query_seed[finite]
    rendered_seed = rendered_seed[finite]
    source_scores = source_scores[finite]
    projected_count = int(query_seed.shape[0])
    if projected_count == 0:
        return None, {
            "pair_inlier_input_count": input_count,
            "pair_inlier_projected_count": 0,
            "pair_inlier_anchor_count": 0,
        }

    expansion_radius_px = int(expansion_radius_px)
    expansion_stride_px = int(expansion_stride_px)
    if expansion_radius_px < 0 or expansion_stride_px <= 0:
        raise ValueError("pair seed expansion radius/stride are invalid")
    offsets = torch.arange(
        -expansion_radius_px,
        expansion_radius_px + 1,
        expansion_stride_px,
        dtype=torch.float32,
        device=rendered_valid.device,
    )
    offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
    offset_xy = torch.stack([offset_x.reshape(-1), offset_y.reshape(-1)], dim=1)
    rendered_anchor = (rendered_seed[:, None, :] + offset_xy[None]).reshape(-1, 2)
    query_anchor = (query_seed[:, None, :] + offset_xy[None]).reshape(-1, 2)
    anchor_scores = source_scores[:, None].expand(-1, offset_xy.shape[0]).reshape(-1)
    rendered_anchor = rendered_anchor.round().long()
    query_anchor = query_anchor.round().long()
    inside = (
        (rendered_anchor[:, 0] >= 0)
        & (rendered_anchor[:, 0] < int(width))
        & (rendered_anchor[:, 1] >= 0)
        & (rendered_anchor[:, 1] < int(height))
    )
    rendered_anchor = rendered_anchor[inside]
    query_anchor = query_anchor[inside]
    anchor_scores = anchor_scores[inside]
    if rendered_anchor.numel():
        support = rendered_valid[rendered_anchor[:, 1], rendered_anchor[:, 0]]
        rendered_anchor = rendered_anchor[support]
        query_anchor = query_anchor[support]
        anchor_scores = anchor_scores[support]
    rendered_anchor, query_anchor, anchor_scores = _deduplicate_pair_anchors(
        rendered_anchor, query_anchor, anchor_scores, width
    )
    if int(max_anchors) > 0 and rendered_anchor.shape[0] > int(max_anchors):
        keep = torch.argsort(anchor_scores, descending=True, stable=True)[: int(max_anchors)]
        rendered_anchor = rendered_anchor[keep]
        query_anchor = query_anchor[keep]
        anchor_scores = anchor_scores[keep]
    diagnostics = {
        "pair_inlier_input_count": input_count,
        "pair_inlier_projected_count": projected_count,
        "pair_inlier_anchor_count": int(rendered_anchor.shape[0]),
        "pair_inlier_expansion_radius_px": expansion_radius_px,
        "pair_inlier_expansion_stride_px": expansion_stride_px,
        "pair_inlier_source_width": source_width,
        "pair_inlier_source_height": source_height,
    }
    if rendered_anchor.shape[0] < 4:
        return None, diagnostics
    return (rendered_anchor, query_anchor), diagnostics


def _pose_delta(reference_w2c, candidate_w2c):
    reference = np.asarray(reference_w2c, dtype=np.float64).reshape(4, 4)
    candidate = np.asarray(candidate_w2c, dtype=np.float64).reshape(4, 4)
    ref_c2w = np.linalg.inv(reference)
    cand_c2w = np.linalg.inv(candidate)
    translation = float(np.linalg.norm(ref_c2w[:3, 3] - cand_c2w[:3, 3]))
    relative_rotation = candidate[:3, :3] @ reference[:3, :3].T
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    rotation = float(np.degrees(np.arccos(cosine)))
    return translation, rotation


def _prior_gn_weights(points3d, target_uv, intrinsic, pose_w2c, match_score, robust_delta_px):
    """Build normalized local-GN weights from seed-pose residuals.

    Dense local matches are measurements conditioned on ``pose_w2c``.  A
    correct refinement therefore starts at that pose and suppresses a local
    match only when its residual is incompatible with the expected local
    window, rather than asking global RANSAC to replace the pose outright.
    """
    target_uv = torch.as_tensor(target_uv, device=points3d.device, dtype=points3d.dtype)
    intrinsic = torch.as_tensor(intrinsic, device=points3d.device, dtype=points3d.dtype)
    pose = torch.as_tensor(pose_w2c, device=points3d.device, dtype=points3d.dtype)
    score = torch.as_tensor(match_score, device=points3d.device, dtype=points3d.dtype).reshape(-1)
    projected, valid = project_points(points3d, intrinsic, pose)
    residual = torch.linalg.norm(projected - target_uv, dim=1)
    valid = valid & torch.isfinite(residual) & torch.isfinite(score)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if bool(valid.any().item()):
        mean_score = score[valid].mean().clamp_min(1e-6)
        score = (score / mean_score).clamp_max(10.0)
    delta = max(float(robust_delta_px), 1e-6)
    robust = torch.where(residual <= delta, torch.ones_like(residual), delta / residual.clamp_min(1e-6))
    weights = torch.nan_to_num(
        score * robust * valid.to(dtype=points3d.dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return weights, residual, valid


def _prior_gn_refine(points3d, target_uv, intrinsic, pose_w2c, match_score, args):
    """One bounded, prior-centered local dense pose update.

    The result is accepted only if it stays in a physically specified trust
    region around the sparse seed.  This makes the dense branch a refiner, not
    a second global relocalizer.
    """
    target_uv = torch.as_tensor(target_uv, device=points3d.device, dtype=points3d.dtype)
    pose = torch.as_tensor(pose_w2c, device=points3d.device, dtype=points3d.dtype)
    intrinsic_t = torch.as_tensor(intrinsic, device=points3d.device, dtype=points3d.dtype)
    weights, seed_residual, valid = _prior_gn_weights(
        points3d,
        target_uv,
        intrinsic_t,
        pose,
        match_score,
        args.prior_gn_robust_delta_px,
    )
    effective = weights > 0
    diagnostics = {
        "prior_gn_weighted_match_count": int(effective.sum().item()),
        "prior_gn_seed_residual_px_median": float(seed_residual[valid].median().item())
        if bool(valid.any().item())
        else None,
        "prior_gn_seed_residual_px_mean": float(seed_residual[valid].mean().item())
        if bool(valid.any().item())
        else None,
        "prior_gn_weight_mean": float(weights[effective].mean().item())
        if bool(effective.any().item())
        else 0.0,
    }
    if int(effective.sum().item()) < int(args.prior_gn_min_matches):
        diagnostics["prior_gn_failure"] = "insufficient_weighted_matches"
        return np.asarray(pose_w2c, dtype=np.float32), np.empty(0, dtype=np.int64), diagnostics

    parameter_scale = torch.tensor(
        [
            float(args.prior_gn_translation_scale_m),
            float(args.prior_gn_translation_scale_m),
            float(args.prior_gn_translation_scale_m),
            math.radians(float(args.prior_gn_rotation_scale_deg)),
            math.radians(float(args.prior_gn_rotation_scale_deg)),
            math.radians(float(args.prior_gn_rotation_scale_deg)),
        ],
        device=points3d.device,
        dtype=points3d.dtype,
    )
    with torch.no_grad():
        refined, info = weighted_gauss_newton_refine(
            points3d,
            target_uv,
            intrinsic_t,
            pose,
            weights=weights,
            num_iterations=int(args.prior_gn_iterations),
            damping=float(args.prior_gn_damping),
            parameter_scale=parameter_scale,
        )
    candidate = refined.detach().cpu().numpy().astype(np.float32)
    delta_translation, delta_rotation = _pose_delta(pose_w2c, candidate)
    diagnostics.update(
        {
            "prior_gn_initial_rmse_px": float(info["initial_rmse"].item()),
            "prior_gn_final_rmse_px": float(info["final_rmse"].item()),
            "prior_gn_condition_number": float(info["condition_number"].item()),
            "prior_gn_raw_condition_number": float(info["raw_condition_number"].item()),
            "prior_gn_delta_translation_m": delta_translation,
            "prior_gn_delta_rotation_deg": delta_rotation,
        }
    )
    within_trust_region = (
        np.isfinite(candidate).all()
        and delta_translation <= float(args.prior_gn_max_translation_m)
        and delta_rotation <= float(args.prior_gn_max_rotation_deg)
    )
    if not within_trust_region:
        diagnostics["prior_gn_failure"] = "outside_trust_region"
        return np.asarray(pose_w2c, dtype=np.float32), np.empty(0, dtype=np.int64), diagnostics

    projected, projected_valid = project_points(
        points3d,
        intrinsic_t,
        torch.as_tensor(candidate, device=points3d.device, dtype=points3d.dtype),
    )
    final_residual = torch.linalg.norm(projected - target_uv, dim=1)
    inliers = torch.nonzero(
        projected_valid & torch.isfinite(final_residual) & (final_residual <= float(args.prior_gn_robust_delta_px)),
        as_tuple=False,
    ).squeeze(1).detach().cpu().numpy().astype(np.int64)
    diagnostics["prior_gn_failure"] = None
    return candidate, inliers, diagnostics


def _prepare_lafgs_field(gaussians, state_path):
    state = torch.load(state_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("LaFGS feature state must be a dictionary")
    indices = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long)
    features = torch.as_tensor(state.get("landmark_features"), dtype=torch.float32)
    if indices.ndim != 1 or features.ndim != 2 or indices.numel() != features.shape[0]:
        raise ValueError("LaFGS state has incompatible landmark indices/features")
    point_count = int(gaussians.get_xyz.shape[0])
    if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= point_count:
        raise ValueError("LaFGS landmark indices are outside the prior 2DGS map")
    target = gaussians._loc_feature
    target_dim = int(target.reshape(point_count, -1).shape[1])
    if features.shape[1] != target_dim:
        raise ValueError(
            f"LaFGS descriptor dimension mismatch: state={features.shape[1]}, map={target_dim}"
        )
    device_indices = indices.to(device=target.device)
    normalized = F.normalize(features, p=2, dim=1).to(device=target.device, dtype=target.dtype)
    with torch.no_grad():
        target.data[device_indices] = normalized.reshape(
            device_indices.numel(), *target.shape[1:]
        )
        landmark_xyz = state.get("landmark_xyz")
        if landmark_xyz is not None:
            landmark_xyz = torch.as_tensor(
                landmark_xyz, device=gaussians._xyz.device, dtype=gaussians._xyz.dtype
            )
            if landmark_xyz.shape == (device_indices.numel(), 3):
                gaussians._xyz.data[device_indices] = landmark_xyz
        loc_opacity = torch.full_like(gaussians._opacity.detach(), -20.0)
        loc_opacity[device_indices] = gaussians._opacity.detach()[device_indices]
    gaussians._loc_opacity = torch.nn.Parameter(loc_opacity, requires_grad=False)
    return {
        "state_path": str(Path(state_path).resolve()),
        "state_sha256": file_sha256(state_path),
        "landmark_count": int(indices.numel()),
        "landmark_indices_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
        "feature_dim": target_dim,
        "feature_render_nonbank_opacity_logit": -20.0,
    }


def _render_bank_geometry(gaussians, pose, fovx, fovy, height, width):
    """Render depth/alpha with exactly the field's opacity support."""
    if not hasattr(gaussians, "_loc_opacity"):
        raise ValueError("LaFGS field geometry requires loc_opacity")
    original_opacity = gaussians._opacity
    try:
        # The renderer has no geometry-opacity override.  This inference-only
        # swap is restored before returning and never touches optimization.
        gaussians._opacity = gaussians._loc_opacity
        return render_from_pose_gsplat(
            gaussians,
            pose,
            fovx,
            fovy,
            width,
            height,
            render_mode="RGB+ED",
            rgb_only=True,
            rasterize_mode="antialiased",
        )
    finally:
        gaussians._opacity = original_opacity


def _render_representation(
    gaussians,
    feature_extractor,
    mode,
    pose_w2c,
    fovx,
    fovy,
    height,
    width,
    prior_rgb_source_image_size=None,
    prior_rgb_render_resolution="source",
):
    pose = torch.as_tensor(pose_w2c, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        if mode == "lafgs_field":
            package = render_from_pose_gsplat(
                gaussians,
                pose,
                fovx,
                fovy,
                width,
                height,
                render_mode="RGB+ED",
                rgb_only=False,
                norm_feat_bf_render=True,
                use_loc_opacity=True,
                return_loc_meta=True,
                rasterize_mode="antialiased",
            )
            rendered_features = package["feature_map"]
            feature_alpha = _as_scalar_map(package.get("loc_alphas"))
            bank_geometry = _render_bank_geometry(
                gaussians, pose, fovx, fovy, height, width
            )
            bank_depth = _as_scalar_map(bank_geometry.get("depth"))
            bank_median_depth = _as_scalar_map(bank_geometry.get("rend_median"))
            bank_alpha = _as_scalar_map(
                bank_geometry.get("rend_alpha", bank_geometry.get("alphas"))
            )
        elif mode == "prior_rgb":
            encoder_height, encoder_width = prior_rgb_encoder_resolution(
                prior_rgb_source_image_size, height, width
            )
            if prior_rgb_render_resolution == "source":
                render_height, render_width = encoder_height, encoder_width
            elif prior_rgb_render_resolution == "feature":
                # This is the original ULF-Loc schedule: render at the dense
                # feature resolution, resize that render back to the query
                # image size, then apply the frozen image encoder.
                render_height, render_width = int(height), int(width)
            else:
                raise ValueError(
                    "prior_rgb_render_resolution must be 'source' or 'feature'"
                )
            rgb_package = render_from_pose_gsplat(
                gaussians,
                pose,
                fovx,
                fovy,
                render_width,
                render_height,
                render_mode="RGB+ED",
                rgb_only=True,
                rasterize_mode="antialiased",
            )
            rendered_rgb = rgb_package["render"][:3].clamp(0.0, 1.0)
            if tuple(rendered_rgb.shape[-2:]) != (encoder_height, encoder_width):
                rendered_rgb = F.interpolate(
                    rendered_rgb[None],
                    size=(encoder_height, encoder_width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            rendered_features = _normalized_feature_map(
                feature_extractor, rendered_rgb, (height, width)
            )
            # PnP coordinates live on the feature grid, so keep its lifting
            # depth on exactly that grid rather than resizing high-res depth.
            package = render_from_pose_gsplat(
                gaussians,
                pose,
                fovx,
                fovy,
                width,
                height,
                render_mode="RGB+ED",
                rgb_only=True,
                rasterize_mode="antialiased",
            )
            feature_alpha = None
            bank_depth = None
            bank_median_depth = None
            bank_alpha = None
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    depth = _as_scalar_map(package.get("depth"))
    rgb_alpha = _as_scalar_map(package.get("rend_alpha", package.get("alphas")))
    return (
        rendered_features,
        depth,
        rgb_alpha,
        feature_alpha,
        bank_depth,
        bank_median_depth,
        bank_alpha,
    )


def _refine_once(
    gaussians,
    feature_extractor,
    mode,
    query_fine,
    query_coarse,
    pose_w2c,
    fovx,
    fovy,
    dense_cfg,
    args,
    gt_pose_w2c=None,
    query_valid_mask=None,
    prior_rgb_source_image_size=None,
    sparse_pair_correspondence=None,
):
    height, width = query_fine.shape[-2:]
    if args.matching_mode == "global" and query_coarse is None:
        raise ValueError("global matching requires a coarse query feature map")
    (
        rendered_features,
        depth,
        rgb_alpha,
        feature_alpha,
        bank_depth,
        bank_median_depth,
        bank_alpha,
    ) = _render_representation(
        gaussians,
        feature_extractor,
        mode,
        pose_w2c,
        fovx,
        fovy,
        height,
        width,
        prior_rgb_source_image_size=prior_rgb_source_image_size,
        prior_rgb_render_resolution=args.prior_rgb_render_resolution,
    )
    if rendered_features is None or depth is None or rgb_alpha is None:
        return np.asarray(pose_w2c, dtype=np.float32), {
            "solver_success": False,
            "failure": "missing_renderer_output",
        }
    rendered_valid = torch.isfinite(depth) & (depth > float(args.min_depth))
    rendered_valid &= torch.isfinite(rgb_alpha) & (rgb_alpha >= float(args.alpha_min))
    if feature_alpha is not None:
        rendered_valid &= torch.isfinite(feature_alpha) & (
            feature_alpha >= float(args.alpha_min)
        )
    depth_consistent = None
    if bank_depth is not None:
        tolerance = float(args.field_depth_consistency_abs_m) + float(
            args.field_depth_consistency_rel
        ) * depth.abs()
        depth_consistent = (
            torch.isfinite(bank_depth)
            & torch.isfinite(depth)
            & (bank_depth > float(args.min_depth))
            & (torch.abs(bank_depth - depth) <= tolerance)
        )
        if bank_alpha is not None:
            depth_consistent &= torch.isfinite(bank_alpha) & (
                bank_alpha >= float(args.alpha_min)
            )
        rendered_valid &= depth_consistent
    if args.matching_mode == "global":
        matches, diagnostics = build_dense_matches(
            query_fine,
            query_coarse,
            rendered_features,
            rendered_valid,
            coarse_temperature=float(dense_cfg["coarse_dual_softmax_temp"]),
            fine_temperature=float(dense_cfg["fine_dual_softmax_temp"]),
            coarse_threshold=float(dense_cfg["coarse_threshold"]),
            fine_threshold=float(dense_cfg["fine_threshold"]),
            valid_cell_fraction=float(args.valid_cell_fraction),
            max_coarse_matches=int(args.max_coarse_matches),
            max_dense_matches=int(args.max_dense_matches),
            query_valid=query_valid_mask,
        )
    elif args.matching_mode == "ulfloc":
        matches, diagnostics = build_ulfloc_dense_matches(
            query_fine,
            query_coarse,
            rendered_features,
            rendered_valid,
            coarse_temperature=float(dense_cfg["coarse_dual_softmax_temp"]),
            fine_temperature=float(dense_cfg["fine_dual_softmax_temp"]),
            coarse_threshold=float(dense_cfg["coarse_threshold"]),
            fine_threshold=float(dense_cfg["fine_threshold"]),
            valid_cell_fraction=float(args.valid_cell_fraction),
            max_coarse_matches=int(args.max_coarse_matches),
            max_dense_matches=int(args.max_dense_matches),
            geometric_filter=bool(args.ulfloc_geometric_filter),
            geometric_neighbors=int(args.ulfloc_geometric_neighbors),
            geometric_support_threshold=float(args.ulfloc_geometric_support_threshold),
            geometric_angle_cos=float(args.ulfloc_geometric_angle_cos),
            geometric_scale_threshold=float(args.ulfloc_geometric_scale_threshold),
            geometric_scale_limit=float(args.ulfloc_geometric_scale_limit),
            query_valid=query_valid_mask,
        )
    elif args.matching_mode == "pair_inlier_local":
        intrinsic = get_intrinsic(fovx, fovy, width, height)
        anchors, pair_diagnostics = build_pair_inlier_anchors(
            sparse_pair_correspondence,
            pose_w2c,
            intrinsic,
            rendered_valid,
            width=width,
            height=height,
            expansion_radius_px=int(args.pair_seed_expansion_radius_px),
            expansion_stride_px=int(args.pair_seed_expansion_stride_px),
            max_anchors=int(args.pair_seed_max_anchors),
        )
        if anchors is None:
            diagnostics = pair_diagnostics
            matches = None
        else:
            rendered_anchor_xy, query_anchor_xy = anchors
            matches, diagnostics = build_local_dense_matches(
                query_fine,
                rendered_features,
                rendered_valid,
                radius_px=int(args.local_radius_px),
                anchor_stride=int(args.local_anchor_stride),
                temperature=float(args.local_temperature),
                batch_size=int(args.local_batch_size),
                min_similarity=float(args.local_min_similarity),
                max_dense_matches=int(args.max_dense_matches),
                correspondence_mode=args.local_correspondence_mode,
                query_valid=query_valid_mask,
                geometric_filter=True,
                geometric_neighbors=int(args.ulfloc_geometric_neighbors),
                geometric_support_threshold=float(args.ulfloc_geometric_support_threshold),
                geometric_angle_cos=float(args.ulfloc_geometric_angle_cos),
                geometric_scale_threshold=float(args.ulfloc_geometric_scale_threshold),
                geometric_scale_limit=float(args.ulfloc_geometric_scale_limit),
                rendered_anchor_xy=rendered_anchor_xy,
                query_anchor_xy=query_anchor_xy,
            )
            diagnostics.update(pair_diagnostics)
    else:
        matches, diagnostics = build_local_dense_matches(
            query_fine,
            rendered_features,
            rendered_valid,
            radius_px=int(args.local_radius_px),
            anchor_stride=int(args.local_anchor_stride),
            temperature=float(args.local_temperature),
            batch_size=int(args.local_batch_size),
            min_similarity=float(args.local_min_similarity),
            max_dense_matches=int(args.max_dense_matches),
            correspondence_mode=args.local_correspondence_mode,
            query_valid=query_valid_mask,
            geometric_filter=(
                bool(args.local_lgcv_filter)
                or str(args.matching_mode) == "pair_local"
            ),
            geometric_neighbors=int(args.ulfloc_geometric_neighbors),
            geometric_support_threshold=float(args.ulfloc_geometric_support_threshold),
            geometric_angle_cos=float(args.ulfloc_geometric_angle_cos),
            geometric_scale_threshold=float(args.ulfloc_geometric_scale_threshold),
            geometric_scale_limit=float(args.ulfloc_geometric_scale_limit),
        )
    diagnostics.update(
        {
            "render_valid_fraction": float(rendered_valid.float().mean().item()),
            "render_alpha_fraction": float((rgb_alpha >= float(args.alpha_min)).float().mean().item()),
            "field_alpha_fraction": (
                float((feature_alpha >= float(args.alpha_min)).float().mean().item())
                if feature_alpha is not None
                else None
            ),
            "field_depth_consistent_fraction": (
                float(depth_consistent.float().mean().item())
                if depth_consistent is not None
                else None
            ),
            "matching_mode": args.matching_mode,
            "query_valid_fraction": (
                float(query_valid_mask.float().mean().item())
                if query_valid_mask is not None
                else 1.0
            ),
        }
    )
    if mode == "prior_rgb":
        encoder_height, encoder_width = prior_rgb_encoder_resolution(
            prior_rgb_source_image_size, height, width
        )
        diagnostics.update(
            {
                "prior_rgb_encoder_height": int(encoder_height),
                "prior_rgb_encoder_width": int(encoder_width),
                "prior_rgb_feature_height": int(height),
                "prior_rgb_feature_width": int(width),
                "prior_rgb_render_resolution": args.prior_rgb_render_resolution,
            }
        )
    if matches is None:
        diagnostics.update({"solver_success": False, "failure": "insufficient_matches"})
        return np.asarray(pose_w2c, dtype=np.float32), diagnostics

    query_xy, rendered_xy, match_score = matches
    intrinsic = get_intrinsic(fovx, fovy, width, height)
    pose_c2w = np.linalg.inv(np.asarray(pose_w2c, dtype=np.float32))
    lift_depth = depth
    lift_depth_source = "full_expected"
    if mode == "lafgs_field" and args.field_lift_depth_source == "bank_expected":
        if bank_depth is None:
            raise ValueError("bank_expected lifting requires bank depth")
        lift_depth = bank_depth
        lift_depth_source = "bank_expected"
    elif mode == "lafgs_field" and args.field_lift_depth_source == "bank_median":
        if bank_median_depth is None:
            raise ValueError("bank_median lifting requires bank median depth")
        lift_depth = bank_median_depth
        lift_depth_source = "bank_median"
    diagnostics["lift_depth_source"] = lift_depth_source
    points3d = lift_2d_to_3d(
        rendered_xy,
        torch.as_tensor(intrinsic, device="cuda"),
        torch.as_tensor(pose_c2w, device="cuda"),
        lift_depth,
    )
    finite = torch.isfinite(points3d).all(dim=1)
    query_xy = query_xy[finite]
    points3d = points3d[finite]
    match_score = match_score[finite]
    diagnostics["finite_points3d"] = int(points3d.shape[0])
    if points3d.shape[0] < 4:
        diagnostics.update({"solver_success": False, "failure": "insufficient_finite_points"})
        return np.asarray(pose_w2c, dtype=np.float32), diagnostics

    if gt_pose_w2c is not None and args.matching_mode == "local":
        diagnostics.update(
            gt_local_basin_diagnostics(
                rendered_xy.detach().cpu().numpy(),
                points3d.detach().cpu().numpy(),
                intrinsic,
                gt_pose_w2c,
                radius_px=int(args.local_radius_px),
            )
        )
        diagnostics.update(
            candidate_displacement_diagnostics(
                query_xy.detach().cpu().numpy(),
                rendered_xy.detach().cpu().numpy(),
                points3d.detach().cpu().numpy(),
                intrinsic,
                gt_pose_w2c,
            )
        )

    measurement_uv = query_xy + 0.5
    if args.dense_pose_solver == "prior_gn":
        candidate, inliers, solver_diagnostics = _prior_gn_refine(
            points3d,
            measurement_uv,
            intrinsic,
            pose_w2c,
            match_score,
            args,
        )
        diagnostics.update(solver_diagnostics)
    else:
        candidate, inliers = solve_pose(
            measurement_uv.detach().cpu().numpy(),
            points3d.detach().cpu().numpy(),
            intrinsic,
            dense_cfg["solver"],
            float(dense_cfg["reprojection_error"]),
            float(dense_cfg["confidence"]),
            int(dense_cfg["max_iterations"]),
            int(dense_cfg["min_iterations"]),
            scores=match_score.detach().cpu().numpy(),
            progressive_sampling=bool(args.dense_progressive_sampling),
            ransac_seed=int(dense_cfg.get("ransac_seed", 0)),
        )
    inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    success = bool(inliers.size >= 4 and np.isfinite(candidate).all())
    if not success:
        candidate = np.asarray(pose_w2c, dtype=np.float32)
    delta_translation, delta_rotation = _pose_delta(pose_w2c, candidate)
    diagnostics.update(
        {
            "solver_success": success,
            "inliers": int(inliers.size) if success else 0,
            "inlier_ratio": float(inliers.size / max(int(points3d.shape[0]), 1)) if success else 0.0,
            "pose_delta_translation_m": delta_translation,
            "pose_delta_rotation_deg": delta_rotation,
            "failure": None if success else (
                diagnostics.get("prior_gn_failure")
                if args.dense_pose_solver == "prior_gn"
                else "pnp_failed"
            ),
            "dense_pose_solver": args.dense_pose_solver,
        }
    )
    if gt_pose_w2c is not None:
        diagnostics.update(
            gt_reprojection_diagnostics(
                query_xy.detach().cpu().numpy(),
                points3d.detach().cpu().numpy(),
                intrinsic,
                gt_pose_w2c,
                inliers,
                scores=match_score.detach().cpu().numpy(),
            )
        )
    return np.asarray(candidate, dtype=np.float32), diagnostics


def _load_input_records(path):
    with open(path) as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("input results must be the stdloc results.json list")
    by_name = {}
    for record in records:
        name = record.get("image_name")
        sparse = record.get("sparse", {})
        if not name or "pose_w2c" not in sparse:
            raise ValueError("every input record needs image_name and sparse.pose_w2c")
        by_name[str(name)] = record
    return by_name


def _load_sparse_pair_correspondences(path):
    """Load sparse RANSAC-inlier pairs emitted by ``stdloc.py`` diagnostics.

    The loader intentionally reads only image identity, measured 2D points,
    selected 3D points, scores, and sparse image resolution.  The JSONL dump
    also carries GT poses for diagnostics, but those fields never enter the
    refinement path.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"sparse correspondence dump is missing: {path}")
    by_name = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"sparse correspondence line {line_number} is not an object")
            image_name = str(payload.get("image_name", ""))
            if not image_name:
                raise ValueError(f"sparse correspondence line {line_number} lacks image_name")
            if image_name in by_name:
                raise ValueError(f"duplicate sparse correspondence image: {image_name}")
            p2d = np.asarray(payload.get("p2d", []), dtype=np.float32).reshape(-1, 2)
            p3d = np.asarray(payload.get("p3d", []), dtype=np.float32).reshape(-1, 3)
            if p2d.shape[0] != p3d.shape[0]:
                raise ValueError(
                    f"sparse correspondence line {line_number} has mismatched p2d/p3d counts"
                )
            scores = np.asarray(payload.get("scores", []), dtype=np.float32).reshape(-1)
            if scores.size not in {0, p2d.shape[0]}:
                raise ValueError(
                    f"sparse correspondence line {line_number} has mismatched scores"
                )
            width = int(payload.get("width", 0))
            height = int(payload.get("height", 0))
            if width <= 0 or height <= 0:
                raise ValueError(
                    f"sparse correspondence line {line_number} has invalid resolution"
                )
            by_name[image_name] = {
                "p2d": p2d,
                "p3d": p3d,
                "scores": scores,
                "width": width,
                "height": height,
                "candidate_stage": str(payload.get("candidate_stage", "unknown")),
            }
    if not by_name:
        raise ValueError("sparse correspondence dump contains no records")
    return by_name


def _metric_summary(records, pose_key):
    # ``cal_pose_error`` follows the STDLoc result schema: TE is already cm.
    # Keep this standalone experiment byte-for-byte comparable to stdloc.py.
    te = np.asarray([record[f"{pose_key}_TE"] for record in records], dtype=np.float64)
    ae = np.asarray([record[f"{pose_key}_AE"] for record in records], dtype=np.float64)
    return {
        "count": int(te.size),
        "median_te_cm": float(np.median(te)) if te.size else None,
        "mean_te_cm": float(np.mean(te)) if te.size else None,
        "median_ae_deg": float(np.median(ae)) if ae.size else None,
        "mean_ae_deg": float(np.mean(ae)) if ae.size else None,
        "recall_5cm_5deg": float(np.mean((te <= 5.0) & (ae <= 5.0))) if te.size else None,
        "recall_2m_5deg": float(np.mean((te <= 200.0) & (ae <= 5.0))) if te.size else None,
    }


def _make_gaussians(dataset):
    if dataset.gaussian_type == "3dgs":
        return GaussianModel(dataset.sh_degree)
    if dataset.gaussian_type == "2dgs":
        return GaussianModel_2dgs(dataset.sh_degree)
    raise ValueError(f"Unsupported gaussian type: {dataset.gaussian_type}")


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def evaluate(args, dataset):
    output_dir = Path(args.output_dir).resolve()
    results_path = output_dir / "results.json"
    summary_path = output_dir / "summary.json"
    existing = {}
    if args.resume and results_path.exists():
        existing = {
            record["image_name"]: record
            for record in json.loads(results_path.read_text())
            if isinstance(record, dict) and "image_name" in record
        }

    input_records = _load_input_records(args.input_results)
    pair_correspondences = {}
    if args.matching_mode == "pair_inlier_local":
        pair_correspondences = _load_sparse_pair_correspondences(
            args.sparse_correspondences
        )
        missing_pair_records = sorted(set(input_records).difference(pair_correspondences))
        if missing_pair_records:
            raise ValueError(
                "sparse correspondence dump does not cover every input result image: "
                f"{missing_pair_records[:3]}"
            )
    query_valid_masks, query_cache_manifest = _load_query_valid_masks(args.query_cache)
    with open(args.cfg) as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    dense_cfg = dict(config["dense"])
    for key in (
        "coarse_dual_softmax_temp",
        "fine_dual_softmax_temp",
        "coarse_threshold",
        "fine_threshold",
        "solver",
        "confidence",
        "reprojection_error",
        "max_iterations",
        "min_iterations",
    ):
        if key not in dense_cfg:
            raise ValueError(f"dense config is missing {key}")
    dense_cfg.setdefault("ransac_seed", 0)
    dense_reprojection_error_override = getattr(
        args, "dense_reprojection_error_override", None
    )
    if dense_reprojection_error_override is not None:
        dense_cfg["reprojection_error"] = float(dense_reprojection_error_override)

    gaussians = _make_gaussians(dataset)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.iteration,
        shuffle=False,
        # Dense evaluation only consumes the held-out query images.  Keeping
        # this lazy avoids materializing the 895 training images per worker.
        preload_cameras=False,
    )
    field_manifest = None
    if args.mode == "lafgs_field":
        if not args.field_state:
            raise ValueError("lafgs_field mode requires --field_state")
        field_manifest = _prepare_lafgs_field(gaussians, args.field_state)

    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    # Candidate-validation records are a subset of train cameras, whereas the
    # final protocol uses test cameras.  Keep only CameraInfo here and lazily
    # materialize the exact requested image inside the loop.
    camera_infos = {
        camera.image_name: camera
        for camera in list(scene.scene_info.train_cameras)
        + list(scene.scene_info.test_cameras)
    }
    records = []
    missing = sorted(set(input_records) - set(camera_infos))
    if missing:
        raise ValueError(f"input result images are absent from scene test split: {missing[:3]}")
    ordered_names = [record["image_name"] for record in json.loads(Path(args.input_results).read_text())]
    if args.max_queries > 0:
        ordered_names = ordered_names[: int(args.max_queries)]
    if args.query_cache:
        missing_masks = sorted(
            set(ordered_names).difference(query_valid_masks)
        )
        if missing_masks:
            raise ValueError(
                "query_cache does not contain valid masks for evaluation images: "
                f"{missing_masks[:3]}"
            )
    start = time.time()
    for number, image_name in enumerate(tqdm(ordered_names, desc=f"Dense {args.mode}"), start=1):
        if image_name in existing:
            records.append(existing[image_name])
            continue
        source = input_records[image_name]
        camera = loadCam(dataset, number - 1, camera_infos[image_name], 1.0)
        sparse_pose = np.asarray(source["sparse"]["pose_w2c"], dtype=np.float32)
        gt_pose = np.asarray(source["gt_pose_w2c"], dtype=np.float32)
        seed_pose = (
            gt_pose.copy()
            if args.initial_pose_source == "gt"
            else sparse_pose.copy()
        )
        query_image = camera.original_image.cuda(non_blocking=True)
        query_fine, query_coarse = query_feature_maps(
            feature_extractor,
            query_image,
            dataset.longest_edge,
            feature_grid=args.feature_grid,
        )
        query_valid_mask = None
        if args.query_cache:
            query_valid_mask = _resize_query_valid_mask(
                query_valid_masks[image_name],
                query_fine.shape[1],
                query_fine.shape[2],
                query_fine.device,
            )
            # Match training exactly: invalid pixels have neither descriptor
            # energy nor candidate-window support.
            query_fine = F.normalize(
                query_fine * query_valid_mask[None].to(dtype=query_fine.dtype),
                p=2,
                dim=0,
            )
        raw_pose = seed_pose
        iterations = []
        for _ in range(int(args.dense_iterations)):
            candidate, diagnostics = _refine_once(
                gaussians,
                feature_extractor,
                args.mode,
                query_fine,
                query_coarse,
                raw_pose,
                camera.FoVx,
                camera.FoVy,
                dense_cfg,
                args,
                gt_pose_w2c=(gt_pose if args.dense_gt_diagnostics else None),
                query_valid_mask=query_valid_mask,
                prior_rgb_source_image_size=tuple(int(value) for value in query_image.shape[-2:]),
                sparse_pair_correspondence=pair_correspondences.get(image_name),
            )
            iterations.append(diagnostics)
            if not diagnostics.get("solver_success", False):
                break
            raw_pose = candidate

        sparse_ae, sparse_te = cal_pose_error(sparse_pose, gt_pose)
        seed_ae, seed_te = cal_pose_error(seed_pose, gt_pose)
        raw_ae, raw_te = cal_pose_error(raw_pose, gt_pose)
        # ``sparse_pose`` and ``seed_pose`` are normally identical, but GT-seed
        # diagnostics deliberately replace the seed.  Keep both deltas so a
        # diagnostic run cannot be misread as an enormous dense update.
        sparse_delta_translation, sparse_delta_rotation = _pose_delta(sparse_pose, raw_pose)
        seed_delta_translation, seed_delta_rotation = _pose_delta(seed_pose, raw_pose)
        final_diag = iterations[-1] if iterations else {}
        record = {
            "image_name": image_name,
            "sparse_pose_w2c": sparse_pose.tolist(),
            "seed_pose_w2c": seed_pose.tolist(),
            "raw_dense_pose_w2c": raw_pose.tolist(),
            "gt_pose_w2c": gt_pose.tolist(),
            "sparse_AE": float(sparse_ae),
            "sparse_TE": float(sparse_te),
            "seed_AE": float(seed_ae),
            "seed_TE": float(seed_te),
            "raw_dense_AE": float(raw_ae),
            "raw_dense_TE": float(raw_te),
            "raw_solver_success": bool(final_diag.get("solver_success", False)),
            "raw_dense_inliers": int(final_diag.get("inliers", 0)),
            "raw_dense_match_count": int(final_diag.get("finite_points3d", 0)),
            "raw_dense_inlier_ratio": float(final_diag.get("inlier_ratio", 0.0)),
            # Legacy fields are relative to the sparse estimate.
            "raw_pose_delta_translation_m": float(sparse_delta_translation),
            "raw_pose_delta_rotation_deg": float(sparse_delta_rotation),
            "raw_pose_delta_from_sparse_translation_m": float(sparse_delta_translation),
            "raw_pose_delta_from_sparse_rotation_deg": float(sparse_delta_rotation),
            "raw_pose_delta_from_seed_translation_m": float(seed_delta_translation),
            "raw_pose_delta_from_seed_rotation_deg": float(seed_delta_rotation),
            "iterations": iterations,
        }
        records.append(record)
        if args.checkpoint_every > 0 and number % int(args.checkpoint_every) == 0:
            _write_json(results_path, records)
        del camera, query_image, query_fine, query_coarse, query_valid_mask
        torch.cuda.empty_cache()

    _write_json(results_path, records)
    metadata = {
        "schema_version": 1,
        "experimental_only": True,
        "mode": args.mode,
        "gt_used_for_refinement": False,
        "gt_used_for_render_seed": bool(args.initial_pose_source == "gt"),
        "dense_gt_diagnostics_enabled": bool(args.dense_gt_diagnostics),
        "input_results": str(Path(args.input_results).resolve()),
        "input_results_sha256": file_sha256(args.input_results),
        "sparse_correspondences": (
            {
                "path": str(Path(args.sparse_correspondences).resolve()),
                "sha256": file_sha256(args.sparse_correspondences),
                "record_count": int(len(pair_correspondences)),
                "uses_gt_pose_for_refinement": False,
            }
            if args.matching_mode == "pair_inlier_local"
            else None
        ),
        "query_cache": query_cache_manifest,
        "cfg": str(Path(args.cfg).resolve()),
        "cfg_sha256": file_sha256(args.cfg),
        "model_path": str(Path(dataset.model_path).resolve()),
        "source_path": str(Path(dataset.source_path).resolve()),
        "iteration": int(args.iteration),
        "dense_config": dense_cfg,
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"source_path", "model_path"}
        },
        "field_manifest": field_manifest,
    }
    _write_json(output_dir / "manifest.json", metadata)
    summary = {
        "schema_version": 1,
        "experimental_only": True,
        "mode": args.mode,
        "count": len(records),
        "runtime_seconds": float(time.time() - start),
        "sparse": _metric_summary(records, "sparse"),
        "raw_dense": _metric_summary(records, "raw_dense"),
        "raw_solver_success_rate": float(
            np.mean([record["raw_solver_success"] for record in records])
        ) if records else 0.0,
        "raw_avg_inliers": float(np.mean([record["raw_dense_inliers"] for record in records])) if records else 0.0,
        "raw_avg_matches": float(np.mean([record["raw_dense_match_count"] for record in records])) if records else 0.0,
        "raw_avg_pose_delta_translation_m": float(
            np.mean([record["raw_pose_delta_translation_m"] for record in records])
        ) if records else 0.0,
        "raw_avg_pose_delta_rotation_deg": float(
            np.mean([record["raw_pose_delta_rotation_deg"] for record in records])
        ) if records else 0.0,
        "raw_avg_pose_delta_from_seed_translation_m": float(
            np.mean([record["raw_pose_delta_from_seed_translation_m"] for record in records])
        ) if records else 0.0,
        "raw_avg_pose_delta_from_seed_rotation_deg": float(
            np.mean([record["raw_pose_delta_from_seed_rotation_deg"] for record in records])
        ) if records else 0.0,
    }
    _write_json(summary_path, summary)
    print(f"Output path: {output_dir}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(
        description="Experimental LaFGS-seeded dense pose refinement"
    )
    model = ModelParams(parser, sentinel=True)
    PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--input_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("lafgs_field", "prior_rgb"), required=True)
    parser.add_argument("--field_state", default=None)
    parser.add_argument(
        "--query_cache",
        default="",
        help="Optional cached valid masks; enforces the dense training query mask at eval.",
    )
    parser.add_argument("--dense_iterations", type=int, default=1)
    parser.add_argument("--alpha_min", type=float, default=0.25)
    parser.add_argument("--min_depth", type=float, default=0.05)
    parser.add_argument("--valid_cell_fraction", type=float, default=0.5)
    parser.add_argument("--field_depth_consistency_abs_m", type=float, default=0.05)
    parser.add_argument("--field_depth_consistency_rel", type=float, default=0.01)
    parser.add_argument("--max_coarse_matches", type=int, default=1024)
    parser.add_argument("--max_dense_matches", type=int, default=4096)
    parser.add_argument(
        "--matching_mode",
        choices=("global", "local", "pair_local", "pair_inlier_local", "ulfloc"),
        default="global",
        help=(
            "Global/local matching, legacy render-anchored local+LGCV, strict "
            "sparse-RANSAC-inlier Pair+LGCV, or ULF-Loc RGB matching."
        ),
    )
    parser.add_argument(
        "--feature_grid",
        choices=("fine", "native"),
        default="fine",
        help="Native uses the exact H/8 x W/8 descriptor grid used in LaFGS training.",
    )
    parser.add_argument("--local_radius_px", type=int, default=24)
    parser.add_argument("--local_anchor_stride", type=int, default=4)
    parser.add_argument("--local_temperature", type=float, default=0.07)
    parser.add_argument("--local_batch_size", type=int, default=256)
    parser.add_argument("--local_min_similarity", type=float, default=-1.0)
    parser.add_argument(
        "--local_correspondence_mode",
        choices=("hard", "soft"),
        default="hard",
        help="Soft uses the local descriptor distribution's subpixel expectation.",
    )
    parser.add_argument(
        "--local_lgcv_filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply ULF-style local geometric consistency to local dense pairs. "
            "pair_local and pair_inlier_local enable the same filter unconditionally."
        ),
    )
    parser.add_argument(
        "--sparse_correspondences",
        default="",
        help=(
            "JSONL sparse correspondence dump from stdloc.py. Required only for "
            "pair_inlier_local; only RANSAC-inlier p2d/p3d/scores are consumed."
        ),
    )
    parser.add_argument(
        "--pair_seed_expansion_radius_px",
        type=int,
        default=4,
        help="Feature-grid radius used to expand each sparse RANSAC inlier pair.",
    )
    parser.add_argument(
        "--pair_seed_expansion_stride_px",
        type=int,
        default=2,
        help="Feature-grid stride used to expand each sparse RANSAC inlier pair.",
    )
    parser.add_argument(
        "--pair_seed_max_anchors",
        type=int,
        default=2048,
        help="Maximum deduplicated render anchors retained by pair_inlier_local.",
    )
    parser.add_argument(
        "--prior_rgb_render_resolution",
        choices=("source", "feature"),
        default="source",
        help=(
            "Render RGB at the original query resolution, or use ULF-Loc's "
            "feature-grid render followed by source-resolution encoding."
        ),
    )
    parser.add_argument(
        "--ulfloc_geometric_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply ULF-Loc-style coarse geometric support filtering before fine matching.",
    )
    parser.add_argument("--ulfloc_geometric_neighbors", type=int, default=8)
    parser.add_argument("--ulfloc_geometric_support_threshold", type=float, default=4.0)
    parser.add_argument("--ulfloc_geometric_angle_cos", type=float, default=0.9659)
    parser.add_argument("--ulfloc_geometric_scale_threshold", type=float, default=0.1)
    parser.add_argument("--ulfloc_geometric_scale_limit", type=float, default=3.0)
    parser.add_argument(
        "--field_lift_depth_source",
        choices=("full_expected", "bank_expected", "bank_median"),
        default="bank_expected",
        help="Lift field descriptors using the same surface support used to render them.",
    )
    parser.add_argument(
        "--dense_reprojection_error_override",
        type=float,
        default=None,
        help="Validation-only PnP threshold override in image pixels.",
    )
    parser.add_argument(
        "--dense_progressive_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use match-score ordered PROSAC for the dense PnP solve.",
    )
    parser.add_argument(
        "--dense_pose_solver",
        choices=("ransac_pnp", "prior_gn"),
        default="ransac_pnp",
        help="RANSAC absolute pose or bounded local Gauss-Newton around the seed pose.",
    )
    parser.add_argument("--prior_gn_iterations", type=int, default=1)
    parser.add_argument("--prior_gn_damping", type=float, default=100.0)
    parser.add_argument("--prior_gn_translation_scale_m", type=float, default=0.05)
    parser.add_argument("--prior_gn_rotation_scale_deg", type=float, default=0.5)
    parser.add_argument("--prior_gn_robust_delta_px", type=float, default=0.75)
    parser.add_argument("--prior_gn_max_translation_m", type=float, default=0.10)
    parser.add_argument("--prior_gn_max_rotation_deg", type=float, default=0.75)
    parser.add_argument("--prior_gn_min_matches", type=int, default=32)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--initial_pose_source",
        choices=("sparse", "gt"),
        default="sparse",
        help="Use gt only for an explicit diagnostic render-seed experiment.",
    )
    parser.add_argument(
        "--dense_gt_diagnostics",
        action="store_true",
        help="Report GT reprojection cleanliness after matching; never used by PnP.",
    )
    args = get_combined_args(parser)
    if int(args.dense_iterations) <= 0:
        raise ValueError("dense_iterations must be positive")
    if not (0.0 <= float(args.alpha_min) <= 1.0):
        raise ValueError("alpha_min must be in [0, 1]")
    if not (0.0 < float(args.valid_cell_fraction) <= 1.0):
        raise ValueError("valid_cell_fraction must be in (0, 1]")
    if float(args.field_depth_consistency_abs_m) < 0.0:
        raise ValueError("field_depth_consistency_abs_m must be non-negative")
    if float(args.field_depth_consistency_rel) < 0.0:
        raise ValueError("field_depth_consistency_rel must be non-negative")
    if int(args.local_radius_px) < 0:
        raise ValueError("local_radius_px must be non-negative")
    if int(args.local_anchor_stride) <= 0 or int(args.local_batch_size) <= 0:
        raise ValueError("local anchor stride and batch size must be positive")
    if float(args.local_temperature) <= 0.0:
        raise ValueError("local_temperature must be positive")
    if int(args.pair_seed_expansion_radius_px) < 0:
        raise ValueError("pair_seed_expansion_radius_px must be non-negative")
    if int(args.pair_seed_expansion_stride_px) <= 0:
        raise ValueError("pair_seed_expansion_stride_px must be positive")
    if int(args.pair_seed_max_anchors) < 0:
        raise ValueError("pair_seed_max_anchors must be non-negative")
    if int(args.prior_gn_iterations) <= 0 or int(args.prior_gn_min_matches) < 4:
        raise ValueError("prior-GN iterations/min-matches are invalid")
    if min(
        float(args.prior_gn_damping),
        float(args.prior_gn_translation_scale_m),
        float(args.prior_gn_rotation_scale_deg),
        float(args.prior_gn_robust_delta_px),
        float(args.prior_gn_max_translation_m),
        float(args.prior_gn_max_rotation_deg),
    ) <= 0.0:
        raise ValueError("prior-GN scales, damping, threshold, and trust region must be positive")
    if args.matching_mode in {"global", "ulfloc"} and args.feature_grid != "fine":
        raise ValueError("global and ulfloc matching require the fine feature grid")
    if args.matching_mode == "ulfloc" and args.mode != "prior_rgb":
        raise ValueError("ulfloc matching is only valid for the RGB 2DGS-prior mode")
    if args.matching_mode == "pair_inlier_local" and not str(
        args.sparse_correspondences
    ).strip():
        raise ValueError("pair_inlier_local requires --sparse_correspondences")
    if int(args.ulfloc_geometric_neighbors) <= 0:
        raise ValueError("ulfloc_geometric_neighbors must be positive")
    if not (0.0 < float(args.ulfloc_geometric_angle_cos) <= 1.0):
        raise ValueError("ulfloc_geometric_angle_cos must be in (0, 1]")
    if float(args.ulfloc_geometric_scale_threshold) <= 0.0:
        raise ValueError("ulfloc_geometric_scale_threshold must be positive")
    if float(args.ulfloc_geometric_scale_limit) <= 1.0:
        raise ValueError("ulfloc_geometric_scale_limit must exceed one")
    dense_reprojection_error_override = getattr(
        args, "dense_reprojection_error_override", None
    )
    if (
        dense_reprojection_error_override is not None
        and float(dense_reprojection_error_override) <= 0.0
    ):
        raise ValueError("dense_reprojection_error_override must be positive")
    dataset = model.extract(args)
    evaluate(args, dataset)


if __name__ == "__main__":
    main()
