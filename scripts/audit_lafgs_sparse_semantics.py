#!/usr/bin/env python3
"""P0 audit for LaFGS sparse descriptor, coordinate and visibility semantics.

The script intentionally compares three descriptor sources on the same native
SuperPoint keypoints: direct sparse extraction, native dense-map sampling and
the cache/deployment pyramid.  It also proves the explicit ``+0.5`` PnP
convention with a synthetic regression and records raster-contribution versus
primitive-center visibility at the actual processed resolution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# Allow direct ``python scripts/...`` execution without a caller-provided
# PYTHONPATH.  Training launchers still use the repository root as usual.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from encoders.feature_extractor import FeatureExtractor
from localization_training.correspondence import bilinear_sample_features
from localization_training.direct_landmark_teacher import (
    filter_depth_consistent_landmarks,
    make_intrinsics_from_fov,
    project_landmarks_to_query,
)
from localization_training.ulf_initializer import (
    PIXEL_CENTER_OFFSET,
    grid_index_to_physical,
    sample_dense_descriptors_at_image_uv,
    sample_mask_at_grid_uv,
)
from scene import Scene
from train_lafgs_map import (
    _cache_signature,
    _gaussian_model_for_type,
    _load_masks,
    _load_or_build_query_cache,
    _masked_camera_image,
    _native_feature_input,
    _render_depth_alpha,
    _render_full_visibility,
    _uniformly_subsample_cameras,
    build_parser,
)
from utils.pose_utils import cal_pose_error, solve_pose


def _summary(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    quantiles = torch.quantile(
        values,
        torch.tensor([0.5, 0.9, 0.99], dtype=values.dtype, device=values.device),
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "p50": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _synthetic_pixel_center_regression():
    """Show that grid index -> ``+0.5`` recovers the physical PnP points."""
    rng = np.random.default_rng(20260720)
    points = np.concatenate(
        [
            rng.uniform(-1.0, 1.0, size=(48, 2)),
            rng.uniform(3.0, 8.0, size=(48, 1)),
        ],
        axis=1,
    ).astype(np.float32)
    width, height = 640, 480
    K = np.array(
        [[580.0, 0.0, width / 2.0], [0.0, 575.0, height / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )
    rvec = np.array([0.08, -0.04, 0.03], dtype=np.float32)
    tvec = np.array([0.12, -0.08, 0.25], dtype=np.float32)
    rotation, _ = cv2.Rodrigues(rvec)
    gt = np.eye(4, dtype=np.float32)
    gt[:3, :3] = rotation
    gt[:3, 3] = tvec
    physical, _ = cv2.projectPoints(points, rvec, tvec, K, np.zeros((4, 1)))
    physical = physical.reshape(-1, 2).astype(np.float32)
    grid = physical - float(PIXEL_CENTER_OFFSET)

    corrected, corrected_inliers = solve_pose(
        grid + float(PIXEL_CENTER_OFFSET),
        points,
        K,
        solver="opencv",
        reprojection_error=1e-4,
        confidence=0.999,
        max_iterations=1000,
        min_iterations=0,
        ransac_seed=2026,
    )
    legacy, legacy_inliers = solve_pose(
        physical + float(PIXEL_CENTER_OFFSET),
        points,
        K,
        solver="opencv",
        reprojection_error=10.0,
        confidence=0.999,
        max_iterations=1000,
        min_iterations=0,
        ransac_seed=2026,
    )
    corrected_ae, corrected_te = cal_pose_error(corrected, gt)
    legacy_ae, legacy_te = cal_pose_error(legacy, gt)
    return {
        "image_hw": [height, width],
        "point_count": int(points.shape[0]),
        "corrected_grid_to_pnp_offset": float(PIXEL_CENTER_OFFSET),
        "corrected_inliers": int(np.asarray(corrected_inliers).size),
        "legacy_inliers": int(np.asarray(legacy_inliers).size),
        "corrected_rotation_error_deg": float(corrected_ae),
        "corrected_translation_error_cm": float(corrected_te),
        "legacy_rotation_error_deg": float(legacy_ae),
        "legacy_translation_error_cm": float(legacy_te),
        "legacy_minus_corrected_translation_cm": float(legacy_te - corrected_te),
        "passed": bool(corrected_ae < 1e-4 and corrected_te < 1e-4),
    }


def _bilinear_support_mask(valid_mask, grid_uv):
    """Require every descriptor-map cell touched by bilinear sampling to be valid."""
    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=grid_uv.device)
    if valid_mask.ndim != 2:
        raise ValueError(f"Expected a two-dimensional cache mask, got {tuple(valid_mask.shape)}")
    grid_uv = torch.as_tensor(grid_uv, device=valid_mask.device)
    height, width = valid_mask.shape
    lower = torch.floor(grid_uv).long()
    upper = lower + 1
    in_bounds = (
        (lower[:, 0] >= 0)
        & (lower[:, 1] >= 0)
        & (upper[:, 0] < width)
        & (upper[:, 1] < height)
    )
    result = torch.zeros(grid_uv.shape[0], dtype=torch.bool, device=grid_uv.device)
    if bool(in_bounds.any().item()):
        x0, y0 = lower[in_bounds, 0], lower[in_bounds, 1]
        x1, y1 = upper[in_bounds, 0], upper[in_bounds, 1]
        result[in_bounds] = (
            valid_mask[y0, x0]
            & valid_mask[y0, x1]
            & valid_mask[y1, x0]
            & valid_mask[y1, x1]
        )
    return result


def _descriptor_audit_for_camera(
    camera, cached, feature_extractor, masks, longest_edge, top_k
):
    contract = cached.get("frontend_metadata", {}).get(
        "query_feature_contract", "legacy_full_then_resized_map"
    )
    if contract == "native_resized_input":
        image, full_valid_mask = _native_feature_input(camera, masks, longest_edge)
    else:
        image, full_valid_mask = _masked_camera_image(camera, masks)
    height, width = image.shape[-2:]
    sparse = feature_extractor.detectAndCompute(image[None], top_k=top_k)[0]
    keypoints = sparse["keypoints"]
    direct = F.normalize(sparse["descriptors"], dim=-1)
    sparse_valid = sample_mask_at_grid_uv(full_valid_mask, keypoints)
    keypoints = keypoints[sparse_valid]
    direct = direct[sparse_valid]
    dense, _ = feature_extractor.detectAndComputeDense(image[None])
    native_dense = sample_dense_descriptors_at_image_uv(
        dense,
        grid_index_to_physical(keypoints),
        (height, width),
    )
    direct_dense_cosine = (direct * native_dense).sum(dim=1)

    cached_feature_map = cached["feature_map"].cuda().float()
    cache_height, cache_width = cached_feature_map.shape[-2:]
    physical = grid_index_to_physical(keypoints)
    cache_physical = physical.clone()
    cache_physical[:, 0] *= float(cache_width) / float(width)
    cache_physical[:, 1] *= float(cache_height) / float(height)
    cache_grid = cache_physical - float(cached.get("pixel_center_offset", 0.0))
    cache_mask = cached["valid_mask"].cuda()
    cache_valid = sample_mask_at_grid_uv(cache_mask, cache_grid)
    cache_bilinear_supported = _bilinear_support_mask(cache_mask, cache_grid)
    cached_descriptors = bilinear_sample_features(cached_feature_map, cache_grid)
    cached_descriptors = F.normalize(cached_descriptors, dim=-1)
    direct_cache_cosine = (direct * cached_descriptors).sum(dim=1)
    direct_cache_nearest_valid = direct_cache_cosine[cache_valid]
    direct_cache_cosine = direct_cache_cosine[cache_bilinear_supported]

    K_full = make_intrinsics_from_fov(
        camera.FoVx,
        camera.FoVy,
        width,
        height,
        device=image.device,
        dtype=torch.float32,
    )
    expected_cache_K = K_full.clone()
    expected_cache_K[0] *= float(cache_width) / float(width)
    expected_cache_K[1] *= float(cache_height) / float(height)
    cache_K = cached["K"].cuda().float()
    metadata = dict(cached.get("frontend_metadata", {}))
    return {
        "image_name": str(camera.image_name),
        "processed_image_hw": [int(height), int(width)],
        "native_dense_hw": [int(dense.shape[-2]), int(dense.shape[-1])],
        "native_effective_hw": [int(dense.shape[-2]) * 8, int(dense.shape[-1]) * 8],
        "cache_feature_hw": [int(cache_height), int(cache_width)],
        "cache_metadata": metadata,
        "sparse_keypoints_before_mask": int(sparse["keypoints"].shape[0]),
        "sparse_keypoints_after_mask": int(keypoints.shape[0]),
        "sparse_keypoints_invalid_after_mask": int(
            (~sample_mask_at_grid_uv(full_valid_mask, keypoints)).sum().item()
        ),
        "direct_vs_native_dense_cosine": _summary(direct_dense_cosine),
        "direct_vs_cache_cosine": _summary(direct_cache_cosine),
        "direct_vs_cache_nearest_valid_cosine": _summary(
            direct_cache_nearest_valid
        ),
        "cache_valid_keypoints": int(cache_valid.sum().item()),
        "cache_bilinear_supported_keypoints": int(
            cache_bilinear_supported.sum().item()
        ),
        "intrinsics_resize_max_abs_error": float(
            (cache_K - expected_cache_K).abs().max().item()
        ),
        "cache_pixel_center_offset": float(cached.get("pixel_center_offset", 0.0)),
        "_direct_dense_values": direct_dense_cosine.detach().cpu(),
        "_direct_cache_values": direct_cache_cosine.detach().cpu(),
        "_direct_cache_nearest_valid_values": direct_cache_nearest_valid.detach().cpu(),
    }


def _visibility_audit_for_camera(
    camera,
    cached,
    gaussians,
    masks,
    background,
    norm_before_render,
    longest_edge,
    maximum_points,
):
    contract = cached.get("frontend_metadata", {}).get(
        "query_feature_contract", "legacy_full_then_resized_map"
    )
    if contract == "native_resized_input":
        image, full_valid_mask = _native_feature_input(camera, masks, longest_edge)
    else:
        image, full_valid_mask = _masked_camera_image(camera, masks)
    height, width = image.shape[-2:]
    contribution = _render_full_visibility(gaussians, camera, width, height)
    contribution_ids = torch.nonzero(contribution, as_tuple=False).reshape(-1)
    if maximum_points > 0:
        contribution_ids = contribution_ids[:maximum_points]
    depth, alpha = _render_depth_alpha(
        gaussians,
        camera,
        height,
        width,
        background,
        norm_before_render,
    )
    K = make_intrinsics_from_fov(
        camera.FoVx,
        camera.FoVy,
        width,
        height,
        device=gaussians.get_xyz.device,
        dtype=gaussians.get_xyz.dtype,
    )
    uv, point_depth, projected = project_landmarks_to_query(
        gaussians.get_xyz[contribution_ids].detach(),
        K,
        camera.world_view_transform.transpose(0, 1).cuda(),
        height,
        width,
        pixel_center_offset=PIXEL_CENTER_OFFSET,
    )
    center_depth = filter_depth_consistent_landmarks(
        uv,
        point_depth,
        projected,
        target_depth=depth,
        target_alpha=alpha,
        alpha_threshold=0.2,
        abs_tolerance=1e-3,
        rel_tolerance=0.01,
    )
    center_depth &= sample_mask_at_grid_uv(full_valid_mask, uv)
    cache_height, cache_width = cached["feature_map"].shape[-2:]
    coarse_contribution = _render_full_visibility(
        gaussians, camera, cache_width, cache_height
    )
    return {
        "image_name": str(camera.image_name),
        "processed_image_hw": [int(height), int(width)],
        "cache_feature_hw": [int(cache_height), int(cache_width)],
        "raster_contribution_visible_full": int(contribution.sum().item()),
        "raster_contribution_visible_coarse": int(coarse_contribution.sum().item()),
        "center_depth_checked_contribution_points": int(contribution_ids.numel()),
        "center_depth_accepted": int(center_depth.sum().item()),
        "contribution_visible_but_center_depth_rejected": int((~center_depth).sum().item()),
        "uses_center_depth_for_kcs_or_gwff": False,
        "valid_mask_policy": "object_and_sky_and_distortion_v1",
    }


def main():
    parser, model_params = build_parser()
    parser.description = __doc__
    # The shared training parser requires this for map checkpoints, but an
    # audit only writes the explicit --audit_json artifact below.
    for action in parser._actions:
        if action.dest == "output_dir":
            action.required = False
            action.default = None
            break
    parser.add_argument("--audit_json", required=True)
    parser.add_argument("--audit_split", choices=["train", "test"], default="test")
    parser.add_argument("--audit_max_views", type=int, default=3)
    parser.add_argument("--audit_sparse_keypoints", type=int, default=2048)
    parser.add_argument("--audit_visibility_views", type=int, default=1)
    parser.add_argument("--audit_visibility_max_points", type=int, default=100000)
    args = parser.parse_args()
    dataset = model_params.extract(args)
    output = Path(args.audit_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.query_cache_path:
        args.query_cache_path = str(output.parent / "query_cache_v7_audit.pt")

    gaussians = _gaussian_model_for_type(dataset.gaussian_type, dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.load_iteration,
        shuffle=False,
        preload_cameras=False,
        load_test_cameras=True,
    )
    if not scene.loaded_iter:
        raise RuntimeError("A frozen Gaussian map checkpoint is required for the audit")
    for parameter in gaussians.parameters():
        parameter.requires_grad_(False)
    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    for parameter in feature_extractor.parameters():
        parameter.requires_grad_(False)
    all_cameras = (
        scene.getTrainCameras()
        if args.audit_split == "train"
        else scene.getTestCameras()
    )
    cameras = _uniformly_subsample_cameras(
        sorted(all_cameras, key=lambda camera: str(camera.image_name)),
        args.audit_max_views,
    )
    if not cameras:
        raise RuntimeError(f"No {args.audit_split} cameras are available for audit")
    masks = _load_masks(dataset)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        device="cuda",
    )
    signature, signature_payload = _cache_signature(dataset, args)
    cache = _load_or_build_query_cache(
        args.query_cache_path,
        signature,
        signature_payload,
        cameras,
        gaussians,
        feature_extractor,
        masks,
        background,
        dataset.norm_before_render,
        dataset.longest_edge,
        args.query_feature_contract,
        require_proposal_scores=False,
        cache_policy=args.query_cache_policy,
    )

    descriptor_views = []
    all_direct_dense = []
    all_direct_cache = []
    all_direct_cache_nearest_valid = []
    for camera in cameras:
        record = _descriptor_audit_for_camera(
            camera,
            cache[str(camera.image_name).replace("\\", "/")],
            feature_extractor,
            masks,
            dataset.longest_edge,
            args.audit_sparse_keypoints,
        )
        descriptor_views.append(record)
        all_direct_dense.append(record.pop("_direct_dense_values"))
        all_direct_cache.append(record.pop("_direct_cache_values"))
        all_direct_cache_nearest_valid.append(
            record.pop("_direct_cache_nearest_valid_values")
        )

    visibility_views = []
    for camera in cameras[: max(int(args.audit_visibility_views), 0)]:
        visibility_views.append(
            _visibility_audit_for_camera(
                camera,
                cache[str(camera.image_name).replace("\\", "/")],
                gaussians,
                masks,
                background,
                dataset.norm_before_render,
                dataset.longest_edge,
                args.audit_visibility_max_points,
            )
        )

    aggregate_direct_dense = _summary(torch.cat(all_direct_dense))
    aggregate_direct_cache = _summary(torch.cat(all_direct_cache))
    aggregate_direct_cache_nearest_valid = _summary(
        torch.cat(all_direct_cache_nearest_valid)
    )
    output_payload = {
        "schema_version": 1,
        "dataset": {
            "model_path": str(Path(dataset.model_path).resolve()),
            "source_path": str(Path(dataset.source_path).resolve()),
            "images": str(dataset.images),
            "load_iteration": int(args.load_iteration),
            "longest_edge": int(dataset.longest_edge),
            "feature_type": str(dataset.feature_type),
            "gaussian_type": str(dataset.gaussian_type),
        },
        "cache": {
            "path": str(Path(args.query_cache_path).resolve()),
            "signature": signature,
            "signature_payload": signature_payload,
        },
        "descriptor_source_parity": {
            "native_sparse_vs_dense": aggregate_direct_dense,
            "native_sparse_vs_cache": aggregate_direct_cache,
            "native_sparse_vs_cache_nearest_valid": (
                aggregate_direct_cache_nearest_valid
            ),
            "per_view": descriptor_views,
            "pass_native_sparse_dense_p50": bool(
                aggregate_direct_dense["p50"] is not None
                and aggregate_direct_dense["p50"] >= 0.999
            ),
            "pass_native_sparse_cache_bilinear_supported_p50": bool(
                aggregate_direct_cache["p50"] is not None
                and aggregate_direct_cache["p50"] >= 0.999
            ),
            "note": (
                "V8 uses one native resized-input SuperPoint contract for "
                "cache, KCS/GWFF and detector deployment. The primary cache "
                "statistic excludes keypoints whose bilinear cache sample "
                "touches an invalid mask cell; nearest-cell-only values are "
                "retained separately to expose mask-boundary behavior."
            ),
        },
        "coordinate_resize_audit": {
            "pixel_center_offset": float(PIXEL_CENTER_OFFSET),
            "per_view_intrinsics_and_shape": [
                {
                    "image_name": item["image_name"],
                    "processed_image_hw": item["processed_image_hw"],
                    "native_dense_hw": item["native_dense_hw"],
                    "native_effective_hw": item["native_effective_hw"],
                    "cache_feature_hw": item["cache_feature_hw"],
                    "cache_pixel_center_offset": item["cache_pixel_center_offset"],
                    "intrinsics_resize_max_abs_error": item["intrinsics_resize_max_abs_error"],
                    "cache_metadata": item["cache_metadata"],
                }
                for item in descriptor_views
            ],
            "synthetic_pnp": _synthetic_pixel_center_regression(),
        },
        "visibility_and_mask_audit": {
            "mask_policy": "object_and_sky_and_distortion_v1",
            "rasterizer_visibility": "primitive contributes nonzero rendered gradient",
            "per_view": visibility_views,
            "sparse_keypoints_invalid_after_mask": int(
                sum(item["sparse_keypoints_invalid_after_mask"] for item in descriptor_views)
            ),
        },
    }
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
