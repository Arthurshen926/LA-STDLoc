import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import get_render_visible_mask, render_from_pose_gsplat
from localization_training.detector_free_map import (
    background_dustbin_loss,
    bounded_geometry_losses,
    build_detector_free_observations,
    build_native_sparse_observations,
    build_score_proposal_observations,
    descriptor_losses_active,
    descriptor_trust_loss,
    hard_hypothesis_retrieval_loss,
    jitter_detector_free_observations,
    local_correlation_peak_loss,
    local_soft_correspondences,
    materialize_descriptor_residual,
    multiview_descriptor_loss,
    native_association_geometry_losses,
    observation_adaptive_trust_weights,
    pose_layer_loss,
    random_negative_retrieval_loss,
)
from localization_training.direct_landmark_teacher import (
    filter_depth_consistent_landmarks,
    make_intrinsics_from_fov,
    project_landmarks_to_query,
)
from localization_training.episode_sampler import split_support_query_cameras
from localization_training.landmark_distill import (
    coverage_balanced_score,
    coverage_preserving_sample,
    robust_normalize,
)
from localization_training.pose_information import compute_pose_information
from localization_training.pose_refiner import se3_exp
from localization_training.surface_anchor import (
    build_pure_geometric_scaffold,
    materialize_bounded_surface_anchors,
)
from localization_training.ulf_initializer import (
    PIXEL_CENTER_OFFSET,
    geometry_view_weights,
    grid_index_to_physical,
    nearest_keypoint_distance,
    sample_dense_descriptors_at_image_uv,
    sample_mask_at_grid_uv,
    surface_normals_from_rotation,
)
from scene import Scene
from utils.general_utils import safe_state, seed_everything
from utils.image_utils import get_resolution_from_longest_edge


def _gaussian_model_for_type(gaussian_type, sh_degree):
    from scene.gaussian_model import GaussianModel, GaussianModel_2dgs

    gaussian_type = str(gaussian_type).lower()
    if gaussian_type == "2dgs":
        return GaussianModel_2dgs(sh_degree)
    if gaussian_type == "3dgs":
        return GaussianModel(sh_degree)
    raise ValueError(f"Unsupported Gaussian type: {gaussian_type}")


def _file_sha256(path, chunk_size=1024 * 1024):
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args):
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _write_reproducibility_manifest(
    output_dir,
    dataset,
    args,
    *,
    landmark_path=None,
    scaffold_diagnostics=None,
):
    paths = {
        "landmark_path": None if landmark_path is None else str(landmark_path),
        "protected_core_path": str(args.protected_core_path or ""),
        "initial_state_path": str(args.initial_state_path or ""),
        "query_cache_path": str(args.query_cache_path or ""),
        "visibility_cache_path": str(args.visibility_cache_path or ""),
    }
    manifest = {
        "version": 1,
        "command": [sys.executable, *sys.argv],
        "arguments": vars(args),
        "dataset": {
            "model_path": os.path.abspath(dataset.model_path),
            "source_path": os.path.abspath(dataset.source_path),
            "images": str(dataset.images),
            "resolution": int(dataset.resolution),
            "longest_edge": int(dataset.longest_edge),
            "gaussian_type": str(dataset.gaussian_type),
            "feature_type": str(dataset.feature_type),
        },
        "inputs": {
            key: {
                "path": value,
                "sha256": (
                    _file_sha256(value)
                    if value
                    and key not in {"query_cache_path", "visibility_cache_path"}
                    else None
                ),
            }
            for key, value in paths.items()
        },
        "scaffold": dict(scaffold_diagnostics or {}),
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "status_porcelain": _git_output("status", "--porcelain"),
            "diff_sha256": hashlib.sha256(
                _git_output("diff", "--binary").encode("utf-8")
            ).hexdigest(),
        },
    }
    path = Path(output_dir) / "reproducibility_manifest.json"
    with path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return path


def _load_landmark_indices(model_path, landmark_path, point_count):
    path = Path(landmark_path)
    if not path.is_absolute():
        path = Path(model_path) / path
    with path.open("rb") as handle:
        indices = torch.as_tensor(pickle.load(handle), dtype=torch.long).reshape(-1)
    if indices.numel() == 0:
        raise ValueError("Localization scaffold is empty")
    if int(indices.min()) < 0 or int(indices.max()) >= int(point_count):
        raise ValueError(
            "Localization scaffold contains out-of-range primitive indices: "
            f"count={indices.numel()} points={point_count}"
        )
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise ValueError("Localization scaffold must contain unique primitive IDs")
    return indices, path


def _load_masks(dataset):
    candidates = (
        Path(dataset.source_path) / dataset.images / "masks.pkl",
        Path(dataset.source_path) / "masks.pkl",
    )
    for path in candidates:
        if path.exists():
            with path.open("rb") as handle:
                return pickle.load(handle)
    return None


def _resize_mask(mask, target_hw):
    mask = torch.as_tensor(mask, device="cuda").bool()
    while mask.ndim > 2:
        mask = mask.squeeze(0)
    return (
        F.interpolate(mask[None, None].float(), size=target_hw, mode="nearest")
        .squeeze(0)
        .squeeze(0)
        .bool()
    )


_VALID_MASK_POLICY = "object_and_sky_and_distortion_v1"
_QUERY_FEATURE_CONTRACT_LEGACY = "legacy_full_then_resized_map"
_QUERY_FEATURE_CONTRACT_NATIVE = "native_resized_input"


def _valid_mask_from_scene_masks(masks, image_name, target_hw, *, device="cuda"):
    """Return the common object/sky/distortion validity domain.

    All sparse selection paths must use this mask before sampling keypoints or
    descriptors.  Keeping it in one helper prevents the cache, KCS and GWFF
    paths from silently seeing different image support.
    """
    target_hw = (int(target_hw[0]), int(target_hw[1]))
    if masks is None or image_name not in masks:
        return torch.ones(target_hw, dtype=torch.bool, device=device)
    channels = masks[image_name]
    if len(channels) < 3:
        raise ValueError(
            f"Mask entry for {image_name!r} must contain object, sky and distortion masks"
        )
    object_mask = _resize_mask(channels[0], target_hw).to(device=device)
    sky_mask = _resize_mask(channels[1], target_hw).to(device=device)
    distortion_mask = _resize_mask(channels[2], target_hw).to(device=device)
    return object_mask & sky_mask & distortion_mask


def _masked_camera_image(camera, masks):
    image = camera.original_image.cuda()
    image_name = _camera_cache_key(camera)
    valid_mask = _valid_mask_from_scene_masks(
        masks,
        image_name,
        image.shape[-2:],
        device=image.device,
    )
    return image * valid_mask[None].to(dtype=image.dtype), valid_mask


def _native_feature_input(camera, masks, longest_edge):
    """Resize RGB before encoding so native SuperPoint descriptors stay valid."""
    image = camera.original_image.cuda()
    full_valid_mask = _valid_mask_from_scene_masks(
        masks,
        _camera_cache_key(camera),
        image.shape[-2:],
        device=image.device,
    )
    target_height, target_width = get_resolution_from_longest_edge(
        image.shape[-2], image.shape[-1], longest_edge
    )
    target_hw = (int(target_height), int(target_width))
    if tuple(image.shape[-2:]) != target_hw:
        image = F.interpolate(
            image[None], size=target_hw, mode="bilinear", align_corners=False
        )[0]
        valid_mask = F.interpolate(
            full_valid_mask[None, None].float(), size=target_hw, mode="nearest"
        )[0, 0].bool()
    else:
        valid_mask = full_valid_mask
    return image * valid_mask[None].to(dtype=image.dtype), valid_mask


def _query_feature_outputs(
    camera,
    feature_extractor,
    *,
    longest_edge,
    masks=None,
    query_feature_contract=_QUERY_FEATURE_CONTRACT_LEGACY,
):
    query_feature_contract = str(query_feature_contract)
    if query_feature_contract == _QUERY_FEATURE_CONTRACT_NATIVE:
        image, full_valid_mask = _native_feature_input(camera, masks, longest_edge)
    elif query_feature_contract == _QUERY_FEATURE_CONTRACT_LEGACY:
        image, full_valid_mask = _masked_camera_image(camera, masks)
    else:
        raise ValueError(
            f"Unknown query feature contract: {query_feature_contract!r}"
        )
    fine_height, fine_width = get_resolution_from_longest_edge(
        image.shape[-2],
        image.shape[-1],
        longest_edge,
    )
    with torch.no_grad():
        outputs = feature_extractor(image[None])
        encoder_feature_map = outputs["feature_map"]
        if query_feature_contract == _QUERY_FEATURE_CONTRACT_NATIVE:
            expected_hw = (
                max(int(fine_height) // 8, 1),
                max(int(fine_width) // 8, 1),
            )
            if tuple(encoder_feature_map.shape[-2:]) != expected_hw:
                raise RuntimeError(
                    "Native SuperPoint descriptor map does not match the "
                    f"resized-image stride-8 grid: got={tuple(encoder_feature_map.shape[-2:])} "
                    f"expected={expected_hw}"
                )
            feature_map = encoder_feature_map[0]
        else:
            target_height = max(int(fine_height) // 8, 1)
            target_width = max(int(fine_width) // 8, 1)
            feature_map = F.interpolate(
                encoder_feature_map,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )[0]
        feature_map = F.normalize(feature_map.float(), dim=0)
        score_map = outputs.get("scores")
        if score_map is None:
            score_map = outputs.get("repeatability")
        if score_map is not None:
            score_map = torch.as_tensor(score_map[0]).squeeze().float()
    valid_mask = _valid_mask_from_scene_masks(
        masks,
        _camera_cache_key(camera),
        feature_map.shape[-2:],
        device=feature_map.device,
    )
    feature_map = feature_map * valid_mask[None]
    proposal_score_map = None
    if score_map is not None:
        score_map = F.adaptive_max_pool2d(
            score_map[None, None],
            output_size=feature_map.shape[-2:],
        )[0, 0]
        proposal_score_map = score_map * valid_mask
    metadata = {
        "input_image_hw": [int(image.shape[-2]), int(image.shape[-1])],
        "encoder_dense_hw": [
            int(encoder_feature_map.shape[-2]),
            int(encoder_feature_map.shape[-1]),
        ],
        "feature_grid_hw": [int(feature_map.shape[-2]), int(feature_map.shape[-1])],
        "query_feature_contract": query_feature_contract,
        "feature_resize_mode": (
            "resize_image_then_native_stride8"
            if query_feature_contract == _QUERY_FEATURE_CONTRACT_NATIVE
            else "encoder_full_then_coarse_bilinear"
        ),
        "descriptor_source": (
            "superpoint_native_dense_resized_input"
            if query_feature_contract == _QUERY_FEATURE_CONTRACT_NATIVE
            else "superpoint_dense_then_coarse_bilinear"
        ),
        "pixel_center_offset": float(PIXEL_CENTER_OFFSET),
        "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "full_valid_fraction": float(full_valid_mask.float().mean().item()),
        "grid_valid_fraction": float(valid_mask.float().mean().item()),
    }
    return feature_map, proposal_score_map, valid_mask, metadata


def _squeeze_render_map(value):
    if value is None:
        return None
    value = torch.as_tensor(value).squeeze()
    if value.ndim == 3 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D render map, got {tuple(value.shape)}")
    return value


@torch.no_grad()
def _render_depth_alpha(gaussians, camera, height, width, background, norm_before_render):
    pose_w2c = camera.world_view_transform.transpose(0, 1).cuda()
    render_pkg = render_from_pose_gsplat(
        gaussians,
        pose_w2c,
        camera.FoVx,
        camera.FoVy,
        width,
        height,
        bg_color=background,
        render_mode="RGB+ED",
        rgb_only=True,
        norm_feat_bf_render=norm_before_render,
        rasterize_mode="antialiased",
    )
    depth = _squeeze_render_map(render_pkg.get("depth"))
    alpha = render_pkg.get("alphas")
    if alpha is None:
        alpha = render_pkg.get("rend_alpha")
    alpha = _squeeze_render_map(alpha)
    if depth is None or alpha is None:
        raise RuntimeError("Frozen RGB map did not return depth and alpha")
    return depth, alpha


def _camera_cache_key(camera):
    return str(getattr(camera, "image_name", "")).replace("\\", "/")


def _camera_names_sha256(names):
    """Hash a canonical camera-name set for direct-holdout auditability."""
    normalized = sorted(str(name).replace("\\", "/") for name in names)
    return hashlib.sha256(("\n".join(normalized) + "\n").encode("utf-8")).hexdigest()


def _uniformly_subsample_cameras(cameras, maximum):
    cameras = list(cameras)
    maximum = int(maximum)
    if maximum <= 0 or len(cameras) <= maximum:
        return cameras
    positions = torch.linspace(0, len(cameras) - 1, maximum).round().long().tolist()
    return [cameras[position] for position in positions]


def _cache_signature(dataset, args):
    observation_source = str(getattr(args, "observation_source", "anchor"))
    native_sparse_enabled = observation_source in {
        "native",
        "native_plus_anchor",
    }
    payload = {
        "version": 9,
        "query_feature_contract": str(args.query_feature_contract),
        "feature_resize_mode": (
            "resize_image_then_native_stride8"
            if str(args.query_feature_contract) == _QUERY_FEATURE_CONTRACT_NATIVE
            else "encoder_full_then_coarse_bilinear"
        ),
        "descriptor_source": (
            "superpoint_native_dense_resized_input"
            if str(args.query_feature_contract) == _QUERY_FEATURE_CONTRACT_NATIVE
            else "superpoint_dense_then_coarse_bilinear"
        ),
        "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
        "pixel_center_offset": float(PIXEL_CENTER_OFFSET),
        "valid_mask_policy": _VALID_MASK_POLICY,
        "model_path": os.path.abspath(dataset.model_path),
        "source_path": os.path.abspath(dataset.source_path),
        "load_iteration": int(args.load_iteration),
        "feature_type": str(dataset.feature_type),
        "images": str(dataset.images),
        "resolution": int(dataset.resolution),
        "longest_edge": int(dataset.longest_edge),
        "white_background": bool(dataset.white_background),
        "norm_before_render": bool(dataset.norm_before_render),
        "native_sparse_enabled": native_sparse_enabled,
        "native_sparse_keypoint_count": (
            int(getattr(args, "native_keypoint_count", 2048))
            if native_sparse_enabled
            else 0
        ),
        "native_sparse_coordinate_convention": (
            "superpoint_grid_index_then_pnp_plus_half_v1"
            if native_sparse_enabled
            else ""
        ),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _cache_payload_compatible(cached_payload, expected_payload):
    if not isinstance(cached_payload, dict):
        return False
    required_keys = (
        "version",
        "query_feature_contract",
        "feature_resize_mode",
        "descriptor_source",
        "coordinate_convention",
        "pixel_center_offset",
        "valid_mask_policy",
        "model_path",
        "source_path",
        "load_iteration",
        "feature_type",
        "images",
        "resolution",
        "longest_edge",
        "white_background",
        "norm_before_render",
        "native_sparse_enabled",
        "native_sparse_keypoint_count",
        "native_sparse_coordinate_convention",
    )
    return all(
        cached_payload.get(key) == expected_payload.get(key)
        for key in required_keys
    )


@torch.no_grad()
def _build_query_cache(
    cameras,
    gaussians,
    feature_extractor,
    masks,
    background,
    norm_before_render,
    longest_edge,
    query_feature_contract,
    include_native_sparse=False,
    native_keypoint_count=2048,
    existing=None,
):
    cache = {} if existing is None else dict(existing)
    for camera in tqdm(cameras, desc="Detector-free real-query cache"):
        name = _camera_cache_key(camera)
        feature_map, proposal_score_map, valid_mask, metadata = _query_feature_outputs(
            camera,
            feature_extractor,
            longest_edge=longest_edge,
            masks=masks,
            query_feature_contract=query_feature_contract,
        )
        height, width = feature_map.shape[-2:]
        previous = cache.get(name, {})
        if "depth" in previous and "alpha" in previous:
            depth = previous["depth"]
            alpha = previous["alpha"]
        else:
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
            device=feature_map.device,
            dtype=feature_map.dtype,
        )
        cache[name] = {
            "feature_map": feature_map.detach().to(device="cpu", dtype=torch.float16),
            "depth": torch.as_tensor(depth).detach().to(device="cpu", dtype=torch.float32),
            "alpha": torch.as_tensor(alpha).detach().to(device="cpu", dtype=torch.float16),
            "K": K.detach().cpu(),
            "pose_w2c": camera.world_view_transform.transpose(0, 1).detach().cpu(),
            "valid_mask": valid_mask.detach().cpu(),
            "pixel_center_offset": float(PIXEL_CENTER_OFFSET),
            "frontend_metadata": metadata,
        }
        if proposal_score_map is not None:
            cache[name]["proposal_score_map"] = proposal_score_map.detach().to(
                device="cpu", dtype=torch.float16
            )
        if include_native_sparse:
            # This is the exact sparse frontend contract used by
            # ``loc_sparse_ulfloc_native``: resize RGB first, mask it before
            # detection, retain native detector coordinates, and defer the
            # +0.5 conversion until PnP.
            native_image, native_valid_mask = _native_feature_input(
                camera, masks, longest_edge
            )
            native_sparse = feature_extractor.detectAndCompute(
                native_image[None], top_k=int(native_keypoint_count)
            )[0]
            native_keypoints = native_sparse["keypoints"]
            native_descriptors = F.normalize(native_sparse["descriptors"], dim=1)
            native_scores = native_sparse["keypoint_scores"]
            native_keep = sample_mask_at_grid_uv(native_valid_mask, native_keypoints)
            native_keypoints = native_keypoints[native_keep]
            native_descriptors = native_descriptors[native_keep]
            native_scores = native_scores[native_keep]
            native_height, native_width = native_image.shape[-2:]
            if "native_depth" in previous and "native_alpha" in previous:
                native_depth = previous["native_depth"]
                native_alpha = previous["native_alpha"]
            else:
                native_depth, native_alpha = _render_depth_alpha(
                    gaussians,
                    camera,
                    native_height,
                    native_width,
                    background,
                    norm_before_render,
                )
            native_K = make_intrinsics_from_fov(
                camera.FoVx,
                camera.FoVy,
                native_width,
                native_height,
                device=native_image.device,
                dtype=native_image.dtype,
            )
            cache[name].update(
                {
                    "native_keypoints": native_keypoints.detach().cpu(),
                    "native_descriptors": native_descriptors.detach().to(
                        device="cpu", dtype=torch.float16
                    ),
                    "native_scores": native_scores.detach().to(
                        device="cpu", dtype=torch.float16
                    ),
                    "native_valid_mask": native_valid_mask.detach().cpu(),
                    "native_depth": torch.as_tensor(native_depth)
                    .detach()
                    .to(device="cpu", dtype=torch.float32),
                    "native_alpha": torch.as_tensor(native_alpha)
                    .detach()
                    .to(device="cpu", dtype=torch.float16),
                    "native_K": native_K.detach().cpu(),
                    "native_input_hw": [int(native_height), int(native_width)],
                    "native_sparse_metadata": {
                        "detect_and_compute": True,
                        "detect_num": int(native_keypoint_count),
                        "keypoint_count_before_mask": int(
                            native_sparse["keypoints"].shape[0]
                        ),
                        "keypoint_count_after_mask": int(native_keypoints.shape[0]),
                        "coordinate_convention": (
                            "superpoint_grid_index_then_pnp_plus_half_v1"
                        ),
                    },
                }
            )
            del native_image, native_valid_mask, native_sparse
        del feature_map, depth, alpha
    return cache


def _load_or_build_query_cache(
    path,
    signature,
    signature_payload,
    cameras,
    gaussians,
    feature_extractor,
    masks,
    background,
    norm_before_render,
    longest_edge,
    query_feature_contract=_QUERY_FEATURE_CONTRACT_LEGACY,
    require_proposal_scores=False,
    require_native_sparse=False,
    native_keypoint_count=2048,
    cache_policy="reuse_or_build",
):
    cache_policy = str(cache_policy)
    if cache_policy not in {"reuse_or_build", "readonly", "refresh"}:
        raise ValueError(f"Unknown query cache policy: {cache_policy}")
    path = Path(path) if path else None
    if cache_policy == "refresh":
        cached = {}
    elif path is not None and path.exists():
        payload = torch.load(path, map_location="cpu")
        signature_matches = payload.get("signature") == signature
        legacy_matches = _cache_payload_compatible(
            payload.get("signature_payload"), signature_payload
        )
        if not signature_matches and not legacy_matches:
            if cache_policy == "readonly":
                raise ValueError(
                    "Detector-free query cache is incompatible with the current "
                    f"dataset/frontend protocol: {path}"
                )
            print(
                f"Ignoring incompatible detector-free query cache: {path}"
            )
            cached = {}
        else:
            cached = payload.get("queries", {})
        missing = [
            _camera_cache_key(camera)
            for camera in cameras
            if _camera_cache_key(camera) not in cached
            or (
                require_proposal_scores
                and "proposal_score_map"
                not in cached[_camera_cache_key(camera)]
            )
            or (
                require_native_sparse
                and not {
                    "native_keypoints",
                    "native_descriptors",
                    "native_scores",
                    "native_valid_mask",
                    "native_depth",
                    "native_alpha",
                    "native_K",
                    "native_input_hw",
                }.issubset(cached[_camera_cache_key(camera)])
            )
        ]
        if not missing:
            print(f"Loaded detector-free query cache: {path} queries={len(cached)}")
            return cached
        if cache_policy == "readonly":
            raise ValueError(
                "Read-only detector-free query cache is incomplete for the "
                f"requested cameras/frontend outputs: {path}; missing={len(missing)}"
            )
    else:
        if cache_policy == "readonly":
            raise ValueError(
                "Read-only detector-free query cache does not exist: "
                f"{path}"
            )
        cached = {}
    missing_cameras = [
        camera
        for camera in cameras
        if _camera_cache_key(camera) not in cached
        or (
            require_proposal_scores
            and "proposal_score_map" not in cached[_camera_cache_key(camera)]
        )
        or (
            require_native_sparse
            and not {
                "native_keypoints",
                "native_descriptors",
                "native_scores",
                "native_valid_mask",
                "native_depth",
                "native_alpha",
                "native_K",
                "native_input_hw",
            }.issubset(cached[_camera_cache_key(camera)])
        )
    ]
    cache = _build_query_cache(
        missing_cameras,
        gaussians,
        feature_extractor,
        masks,
        background,
        norm_before_render,
        longest_edge,
        query_feature_contract,
        include_native_sparse=require_native_sparse,
        native_keypoint_count=native_keypoint_count,
        existing=cached,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        torch.save(
            {
                "version": 3,
                "signature": signature,
                "signature_payload": signature_payload,
                "queries": cache,
            },
            temporary_path,
        )
        os.replace(temporary_path, path)
        print(f"Saved detector-free query cache: {path}")
    return cache


def _cached_observations(
    cached,
    base_bank_xyz,
    args,
    *,
    max_observations,
    bank_visibility_mask=None,
    prediction_bank_xyz=None,
):
    feature_map = cached["feature_map"].cuda().float()
    return build_detector_free_observations(
        base_bank_xyz,
        feature_map,
        cached["K"].cuda().float(),
        cached["pose_w2c"].cuda().float(),
        prediction_bank_xyz=prediction_bank_xyz,
        target_depth=cached["depth"].cuda().float(),
        target_alpha=cached["alpha"].cuda().float(),
        bank_visibility_mask=bank_visibility_mask,
        query_valid_mask=cached.get("valid_mask"),
        alpha_threshold=args.alpha_threshold,
        depth_abs_tolerance=args.depth_abs_tolerance,
        depth_rel_tolerance=args.depth_rel_tolerance,
        max_observations=max_observations,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        depth_bins=args.depth_bins,
        pixel_center_offset=float(cached.get("pixel_center_offset", 0.0)),
    )


def _cached_native_observations(
    cached,
    base_bank_xyz,
    args,
    *,
    max_observations,
    bank_visibility_mask=None,
    prediction_bank_xyz=None,
):
    required = {
        "native_keypoints",
        "native_descriptors",
        "native_scores",
        "native_valid_mask",
        "native_depth",
        "native_alpha",
        "native_K",
        "native_input_hw",
    }
    missing = sorted(required - set(cached))
    if missing:
        raise ValueError(
            "Native sparse observations were requested but the query cache is "
            f"missing: {', '.join(missing)}"
        )
    native_height, native_width = map(int, cached["native_input_hw"])
    return build_native_sparse_observations(
        base_bank_xyz,
        cached["native_keypoints"].cuda().float(),
        cached["native_descriptors"].cuda().float(),
        cached["native_scores"].cuda().float(),
        cached["native_K"].cuda().float(),
        cached["pose_w2c"].cuda().float(),
        image_size=(native_height, native_width),
        prediction_bank_xyz=prediction_bank_xyz,
        target_depth=cached["native_depth"].cuda().float(),
        target_alpha=cached["native_alpha"].cuda().float(),
        bank_visibility_mask=bank_visibility_mask,
        query_valid_mask=cached["native_valid_mask"].cuda().bool(),
        max_observations=max_observations,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        positive_radius_px=args.native_association_radius_px,
        unmatched_fraction=args.native_unmatched_fraction,
        sampling_mode=args.native_sampling_mode,
        pixel_center_offset=float(PIXEL_CENTER_OFFSET),
    )


def _primary_observations(
    cached,
    base_bank_xyz,
    args,
    *,
    max_observations,
    bank_visibility_mask=None,
    prediction_bank_xyz=None,
):
    """Return deployment-aligned observations and optional anchor auxiliary."""
    source = str(args.observation_source)
    if source == "anchor":
        observations = _cached_observations(
            cached,
            base_bank_xyz,
            args,
            max_observations=max_observations,
            bank_visibility_mask=bank_visibility_mask,
            prediction_bank_xyz=prediction_bank_xyz,
        )
        return observations, None
    observations = _cached_native_observations(
        cached,
        base_bank_xyz,
        args,
        max_observations=max_observations,
        bank_visibility_mask=bank_visibility_mask,
        prediction_bank_xyz=prediction_bank_xyz,
    )
    anchor_auxiliary = None
    if source == "native_plus_anchor" and float(args.native_anchor_aux_weight) > 0.0:
        anchor_auxiliary = _cached_observations(
            cached,
            base_bank_xyz,
            args,
            max_observations=max_observations,
            bank_visibility_mask=bank_visibility_mask,
            prediction_bank_xyz=prediction_bank_xyz,
        )
    return observations, anchor_auxiliary


@torch.no_grad()
def _basic_visibility_counts(
    xyz,
    query_names,
    cache,
    args,
    *,
    chunk_size=262144,
):
    """Count depth-consistent support views without inspecting image descriptors."""
    xyz = torch.as_tensor(xyz).detach().float()
    counts = torch.zeros(xyz.shape[0], dtype=torch.int16, device=xyz.device)
    for name in tqdm(query_names, desc="Pure scaffold visibility"):
        cached = cache[name]
        feature_map = cached["feature_map"]
        height, width = feature_map.shape[-2:]
        K = cached["K"].to(device=xyz.device, dtype=xyz.dtype)
        pose_w2c = cached["pose_w2c"].to(device=xyz.device, dtype=xyz.dtype)
        target_depth = cached["depth"].to(device=xyz.device, dtype=xyz.dtype)
        target_alpha = cached["alpha"].to(device=xyz.device, dtype=xyz.dtype)
        valid_mask = cached.get("valid_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=xyz.device, dtype=torch.bool).squeeze()
        for start in range(0, xyz.shape[0], int(chunk_size)):
            end = min(start + int(chunk_size), xyz.shape[0])
            uv, depth, projected = project_landmarks_to_query(
                xyz[start:end],
                K,
                pose_w2c,
                height,
                width,
                pixel_center_offset=float(cached.get("pixel_center_offset", 0.0)),
            )
            visible = filter_depth_consistent_landmarks(
                uv,
                depth,
                projected,
                target_depth=target_depth,
                target_alpha=target_alpha,
                alpha_threshold=args.alpha_threshold,
                abs_tolerance=args.depth_abs_tolerance,
                rel_tolerance=args.depth_rel_tolerance,
            )
            if valid_mask is not None and bool(visible.any().item()):
                rounded = uv.round().long()
                visible_indices = torch.nonzero(visible, as_tuple=False).reshape(-1)
                visible[visible_indices] &= valid_mask[
                    rounded[visible_indices, 1],
                    rounded[visible_indices, 0],
                ]
            counts[start:end] += visible.to(dtype=counts.dtype)
    return counts


def _render_full_visibility(gaussians, camera, width, height):
    """Use raster contribution, rather than primitive-center depth, for visibility."""
    xyz = gaussians._xyz
    previous_requires_grad = bool(xyz.requires_grad)
    xyz.requires_grad_(True)
    try:
        with torch.enable_grad():
            visible = get_render_visible_mask(gaussians, camera, width, height)
    finally:
        xyz.grad = None
        xyz.requires_grad_(previous_requires_grad)
    return visible.detach().bool()


def _automatic_ulf_voxel_size(xyz, budget):
    xyz = torch.as_tensor(xyz).float()
    if xyz.numel() == 0:
        return 1.0
    extent = (xyz.amax(dim=0) - xyz.amin(dim=0)).clamp_min(1e-4)
    volume = float(extent.prod().item())
    return max((volume / max(int(budget), 1)) ** (1.0 / 3.0), 1e-4)


def _build_ulf_consensus_landmark_indices(
    gaussians,
    cameras,
    masks,
    feature_extractor,
    args,
):
    """ULF-Loc KCS over full 2DGS primitives with contribution visibility.

    The vote is emitted only when a visible surface primitive projects within
    ``ulf_consensus_radius_px`` of a native sparse SuperPoint keypoint.  The
    final coverage-balanced extraction is deterministic and records any
    necessary fallback when the consensus pool is too small for the requested
    bank size.
    """
    if str(feature_extractor.feature_type) != "sp":
        raise ValueError("ULF consensus sampling currently requires SuperPoint")
    xyz = gaussians.get_xyz.detach().float()
    opacity = gaussians.get_opacity.detach().reshape(-1)
    eligible = torch.isfinite(opacity) & (opacity >= float(args.scaffold_min_opacity))
    votes = torch.zeros(xyz.shape[0], dtype=torch.int32, device=xyz.device)
    visibility_counts = torch.zeros_like(votes)
    cameras = _uniformly_subsample_cameras(
        cameras,
        args.ulf_consensus_max_views,
    )
    view_records = []
    radius_px = float(args.ulf_consensus_radius_px)
    for camera in tqdm(cameras, desc="ULF keypoint-consensus sampling"):
        image, valid_mask = _native_feature_input(
            camera, masks, args.longest_edge
        )
        height, width = image.shape[-2:]
        sparse = feature_extractor.detectAndCompute(
            image[None],
            top_k=args.ulf_consensus_keypoints,
        )[0]
        keypoints = sparse["keypoints"]
        keypoint_valid = sample_mask_at_grid_uv(valid_mask, keypoints)
        keypoints = keypoints[keypoint_valid]
        render_visible = _render_full_visibility(gaussians, camera, width, height)
        K = make_intrinsics_from_fov(
            camera.FoVx,
            camera.FoVy,
            width,
            height,
            device=xyz.device,
            dtype=xyz.dtype,
        )
        uv, _, projected = project_landmarks_to_query(
            xyz,
            K,
            camera.world_view_transform.transpose(0, 1).cuda(),
            height,
            width,
            pixel_center_offset=PIXEL_CENTER_OFFSET,
        )
        in_valid_mask = sample_mask_at_grid_uv(valid_mask, uv)
        visible = eligible & projected & render_visible & in_valid_mask
        visible_indices = torch.nonzero(visible, as_tuple=False).reshape(-1)
        if visible_indices.numel() > int(args.ulf_consensus_max_candidates_per_view) > 0:
            candidate_opacity = opacity[visible_indices]
            _, keep = torch.topk(
                candidate_opacity,
                int(args.ulf_consensus_max_candidates_per_view),
                largest=True,
                sorted=False,
            )
            visible_indices = visible_indices[keep]
        if visible_indices.numel() > 0:
            visibility_counts[visible_indices] += 1
        matched_count = 0
        if visible_indices.numel() > 0 and keypoints.numel() > 0:
            distance = nearest_keypoint_distance(
                uv[visible_indices],
                keypoints,
                chunk_size=args.ulf_consensus_distance_chunk,
            )
            matched = distance <= radius_px
            if bool(matched.any().item()):
                votes[visible_indices[matched]] += 1
                matched_count = int(matched.sum().item())
        view_records.append(
            {
                "image_name": _camera_cache_key(camera),
                "processed_image_hw": [int(height), int(width)],
                "sparse_keypoints_before_mask": int(sparse["keypoints"].shape[0]),
                "sparse_keypoints_after_mask": int(keypoints.shape[0]),
                "contribution_visible_primitives": int(render_visible.sum().item()),
                "eligible_visible_primitives": int(visible_indices.shape[0]),
                "consensus_votes": matched_count,
            }
        )

    min_votes = max(int(args.ulf_consensus_min_votes), 1)
    consensus_eligible = eligible & (votes >= min_votes)
    candidate_eligible = consensus_eligible
    fallback_to_non_consensus = False
    requested_budget = min(int(args.scaffold_budget), int(eligible.sum().item()))
    if int(candidate_eligible.sum().item()) < requested_budget:
        candidate_eligible = eligible
        fallback_to_non_consensus = True
    vote_score = votes.float() + 0.01 * visibility_counts.float()
    voxel_size = float(args.ulf_consensus_voxel_size)
    if voxel_size <= 0.0:
        voxel_size = _automatic_ulf_voxel_size(xyz[candidate_eligible], requested_budget)
    selected = coverage_balanced_score(
        xyz,
        requested_budget,
        vote_score,
        voxel_size=voxel_size,
        max_per_voxel=args.ulf_consensus_max_per_voxel,
        eligible=candidate_eligible,
        allow_overflow=True,
    )
    if selected.numel() != requested_budget:
        raise RuntimeError(
            "ULF consensus scaffold could not satisfy the requested budget: "
            f"requested={requested_budget} selected={selected.numel()}"
        )
    selected_votes = votes[selected]
    diagnostics = {
        "mode": "ulf_keypoint_consensus",
        "budget": int(selected.numel()),
        "requested_budget": int(args.scaffold_budget),
        "eligible_primitives": int(eligible.sum().item()),
        "consensus_eligible_primitives": int(consensus_eligible.sum().item()),
        "fallback_to_non_consensus": bool(fallback_to_non_consensus),
        "selected_with_consensus": int((selected_votes >= min_votes).sum().item()),
        "selected_vote_mean": float(selected_votes.float().mean().item()),
        "selected_vote_max": int(selected_votes.max().item()) if selected_votes.numel() else 0,
        "minimum_votes": min_votes,
        "consensus_radius_px": radius_px,
        "consensus_sparse_keypoints": int(args.ulf_consensus_keypoints),
        "consensus_view_count": int(len(cameras)),
        "distance_chunk_size": int(args.ulf_consensus_distance_chunk),
        "candidate_cap_per_view": int(args.ulf_consensus_max_candidates_per_view),
        "voxel_size": float(voxel_size),
        "max_per_voxel": int(args.ulf_consensus_max_per_voxel),
        "visibility": "2dgs_raster_contribution_gradient",
        "visibility_resolution": "resized_feature_input_resolution",
        "query_feature_contract": _QUERY_FEATURE_CONTRACT_NATIVE,
        "coordinate_convention": "grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "views": view_records,
    }
    return selected.detach().cpu(), diagnostics


def _build_ulf_geometry_features(
    cameras,
    gaussians,
    landmark_indices,
    masks,
    feature_extractor,
    fallback_features,
    args,
):
    """Fuse native dense SuperPoint descriptors with surface-normal weights."""
    landmark_indices_gpu = landmark_indices.to(device=gaussians.get_xyz.device)
    bank_xyz = gaussians.get_xyz[landmark_indices_gpu].detach().float()
    rotations = gaussians.get_rotation[landmark_indices_gpu].detach().float()
    scales = gaussians.get_scaling[landmark_indices_gpu].detach().float()
    normals = surface_normals_from_rotation(rotations, scales)
    feature_sum = torch.zeros(
        (bank_xyz.shape[0], fallback_features.shape[1]),
        device=bank_xyz.device,
        dtype=torch.float32,
    )
    weight_sum = torch.zeros(bank_xyz.shape[0], device=bank_xyz.device)
    observation_count = torch.zeros(
        bank_xyz.shape[0], dtype=torch.long, device=bank_xyz.device
    )
    cameras = _uniformly_subsample_cameras(cameras, args.ulf_fusion_max_views)
    sampled_weight_sum = 0.0
    sampled_weight_count = 0
    native_size_mismatch_views = 0
    for camera in tqdm(cameras, desc="ULF geometry-weighted feature fusion"):
        image, valid_mask = _native_feature_input(
            camera, masks, args.longest_edge
        )
        height, width = image.shape[-2:]
        dense_features, _ = feature_extractor.detectAndComputeDense(image[None])
        expected_hw = (int(dense_features.shape[-2]) * 8, int(dense_features.shape[-1]) * 8)
        native_size_mismatch_views += int(expected_hw != (int(height), int(width)))
        visible = _render_full_visibility(gaussians, camera, width, height)[
            landmark_indices_gpu
        ]
        K = make_intrinsics_from_fov(
            camera.FoVx,
            camera.FoVy,
            width,
            height,
            device=bank_xyz.device,
            dtype=bank_xyz.dtype,
        )
        grid_uv, _, projected = project_landmarks_to_query(
            bank_xyz,
            K,
            camera.world_view_transform.transpose(0, 1).cuda(),
            height,
            width,
            pixel_center_offset=PIXEL_CENTER_OFFSET,
        )
        valid = visible & projected & sample_mask_at_grid_uv(valid_mask, grid_uv)
        if not bool(valid.any().item()):
            continue
        compact_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        sampled = sample_dense_descriptors_at_image_uv(
            dense_features,
            grid_index_to_physical(grid_uv[compact_indices]),
            (height, width),
        )
        weights = geometry_view_weights(
            bank_xyz[compact_indices],
            normals[compact_indices],
            camera.camera_center.cuda(),
        )
        if float(args.ulf_fusion_min_cosine) > 0.0:
            weights = weights * (weights >= float(args.ulf_fusion_min_cosine))
        useful = weights > 0.0
        if not bool(useful.any().item()):
            continue
        compact_indices = compact_indices[useful]
        sampled = sampled[useful]
        weights = weights[useful].float()
        feature_sum.index_add_(0, compact_indices, sampled.float() * weights[:, None])
        weight_sum.index_add_(0, compact_indices, weights)
        observation_count.index_add_(
            0,
            compact_indices,
            torch.ones_like(compact_indices, dtype=observation_count.dtype),
        )
        sampled_weight_sum += float(weights.sum().item())
        sampled_weight_count += int(weights.numel())

    observed = weight_sum > 1e-8
    result = F.normalize(fallback_features.float(), dim=-1).clone()
    if bool(observed.any().item()):
        result[observed] = F.normalize(
            feature_sum[observed] / weight_sum[observed, None],
            dim=-1,
        )
    diagnostics = {
        "initialization_mode": "ulf_geometry_weighted_fusion",
        "observed_landmarks": int(observed.sum().item()),
        "unobserved_landmarks": int((~observed).sum().item()),
        "observation_count_mean": float(observation_count.float().mean().item()),
        "observation_count_median": float(observation_count.float().median().item()),
        "observation_count_max": int(observation_count.max().item()),
        "geometry_weight_mean": sampled_weight_sum / max(sampled_weight_count, 1),
        "geometry_weighted_samples": int(sampled_weight_count),
        "fusion_view_count": int(len(cameras)),
        "native_stride8_size_mismatch_views": int(native_size_mismatch_views),
        "visibility": "2dgs_raster_contribution_gradient",
        "visibility_resolution": "resized_feature_input_resolution",
        "query_feature_contract": _QUERY_FEATURE_CONTRACT_NATIVE,
        "coordinate_convention": "grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
    }
    return result, observation_count, diagnostics


def _load_or_build_landmark_indices(
    dataset,
    gaussians,
    args,
    *,
    visibility_counts=None,
    cameras=None,
    masks=None,
    feature_extractor=None,
):
    if args.scaffold_mode == "file":
        indices, path = _load_landmark_indices(
            dataset.model_path,
            args.landmark_path,
            point_count=gaussians.get_xyz.shape[0],
        )
        return indices, path, {
            "mode": "file",
            "path": str(path),
            "budget": int(indices.numel()),
        }

    output_path = Path(args.generated_landmark_path)
    if not output_path.is_absolute():
        output_path = Path(args.output_dir) / output_path
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() and not args.regenerate_scaffold:
        indices, path = _load_landmark_indices(
            dataset.model_path,
            str(output_path),
            point_count=gaussians.get_xyz.shape[0],
        )
        metadata = {}
        if metadata_path.exists():
            with metadata_path.open() as handle:
                metadata = json.load(handle)
        metadata.setdefault("mode", f"{args.scaffold_mode}_cached")
        metadata.setdefault("budget", int(indices.numel()))
        return indices, path, metadata

    if args.scaffold_mode == "ulf_consensus":
        if cameras is None or feature_extractor is None:
            raise ValueError("ULF consensus scaffold requires cameras and a feature extractor")
        indices, diagnostics = _build_ulf_consensus_landmark_indices(
            gaussians,
            cameras,
            masks,
            feature_extractor,
            args,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(indices, handle)
        with metadata_path.open("w") as handle:
            json.dump(diagnostics, handle, indent=2, sort_keys=True)
        print(
            "Saved ULF keypoint-consensus localization scaffold: "
            f"{output_path} count={indices.numel()}"
        )
        return indices, output_path, diagnostics

    opacity = gaussians.get_opacity.detach().reshape(-1)
    eligible = torch.isfinite(opacity) & (opacity >= float(args.scaffold_min_opacity))
    if visibility_counts is not None:
        visibility_counts = torch.as_tensor(
            visibility_counts,
            device=eligible.device,
        ).reshape(-1)
        if visibility_counts.numel() != eligible.numel():
            raise ValueError("scaffold visibility counts must match the primitive count")
        eligible &= visibility_counts >= int(args.scaffold_min_visible_views)
    protected_core = None
    if args.scaffold_mode == "protected_union":
        protected_core, protected_path = _load_landmark_indices(
            dataset.model_path,
            args.protected_core_path,
            point_count=gaussians.get_xyz.shape[0],
        )
        if protected_core.numel() > int(args.scaffold_budget):
            raise ValueError("protected core exceeds the scaffold budget")
        protected_mask = torch.zeros_like(eligible)
        protected_mask[protected_core.to(device=eligible.device)] = True
        eligible &= ~protected_mask
        requested_budget = int(args.scaffold_budget) - int(protected_core.numel())
    else:
        protected_path = None
        requested_budget = int(args.scaffold_budget)
    if requested_budget > 0:
        scaffold = build_pure_geometric_scaffold(
            gaussians.get_xyz.detach(),
            gaussians.get_rotation.detach(),
            requested_budget,
            eligible=eligible,
            normal_bins=args.scaffold_normal_bins,
            voxel_size=args.scaffold_voxel_size,
            search_steps=args.scaffold_search_steps,
            seed=args.scaffold_seed,
        )
    else:
        from localization_training.surface_anchor import GeometricScaffold

        scaffold = GeometricScaffold(
            indices=torch.empty(0, dtype=torch.long, device=eligible.device),
            voxel_size=0.0,
            group_count=0,
            eligible_count=int(eligible.sum().item()),
            diagnostics={"mode": "protected_core_only"},
        )
    if protected_core is not None:
        geometric_extra_count = int(scaffold.indices.numel())
        scaffold.indices = torch.cat(
            [
                protected_core.to(device=scaffold.indices.device),
                scaffold.indices,
            ]
        )
        scaffold.diagnostics.update(
            {
                "mode": "protected_core_plus_pure_geometry",
                "protected_core_count": int(protected_core.numel()),
                "protected_core_path": str(protected_path),
                "geometric_extra_count": geometric_extra_count,
            }
        )
    if visibility_counts is not None:
        selected_visibility = visibility_counts[scaffold.indices]
        scaffold.diagnostics.update(
            {
                "visibility_prefilter": True,
                "minimum_visible_views": int(args.scaffold_min_visible_views),
                "selected_visible_views_min": int(selected_visibility.min().item()),
                "selected_visible_views_median": float(
                    selected_visibility.float().median().item()
                ),
                "selected_visible_views_mean": float(
                    selected_visibility.float().mean().item()
                ),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(scaffold.indices.detach().cpu(), handle)
    with metadata_path.open("w") as handle:
        json.dump(scaffold.diagnostics, handle, indent=2, sort_keys=True)
    print(
        "Saved pure-geometric localization scaffold: "
        f"{output_path} count={scaffold.indices.numel()}"
    )
    return scaffold.indices.cpu(), output_path, scaffold.diagnostics


def _cached_score_proposal_observations(cached, anchor_observations, args):
    score_map = cached.get("proposal_score_map")
    if score_map is None or int(args.generic_proposal_count) <= 0:
        return None
    return build_score_proposal_observations(
        anchor_observations,
        score_map.cuda().float(),
        max_proposals=args.generic_proposal_count,
        nms_radius=args.generic_proposal_nms_radius,
        score_threshold=args.generic_proposal_score_threshold,
        positive_search_radius_px=args.generic_proposal_positive_radius,
        include_unmatched=args.generic_proposal_include_unmatched,
    )


def _visibility_signature(dataset, args, landmark_indices):
    digest = hashlib.sha256()
    digest.update(landmark_indices.detach().cpu().numpy().tobytes())
    payload = {
        "version": 5,
        "model_path": os.path.abspath(dataset.model_path),
        "source_path": os.path.abspath(dataset.source_path),
        "load_iteration": int(args.load_iteration),
        "images": str(dataset.images),
        "resolution": int(dataset.resolution),
        "longest_edge": int(dataset.longest_edge),
        "query_feature_contract": str(args.query_feature_contract),
        "feature_resize_mode": (
            "resize_image_then_native_stride8"
            if str(args.query_feature_contract) == _QUERY_FEATURE_CONTRACT_NATIVE
            else "encoder_full_then_coarse_bilinear"
        ),
        "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "visibility_resolution": (
            "native_sparse_input"
            if str(args.observation_source) in {"native", "native_plus_anchor"}
            else "feature_grid"
        ),
        "landmark_sha256": digest.hexdigest(),
        "landmark_count": int(landmark_indices.numel()),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _render_bank_visibility(gaussians, camera, width, height, landmark_indices):
    xyz = gaussians._xyz
    previous_requires_grad = bool(xyz.requires_grad)
    xyz.requires_grad_(True)
    try:
        with torch.enable_grad():
            visible = get_render_visible_mask(
                gaussians,
                camera,
                width,
                height,
            )
    finally:
        xyz.grad = None
        xyz.requires_grad_(previous_requires_grad)
    return visible[landmark_indices.cuda()].detach().cpu().bool()


def _load_or_build_visibility_cache(
    path,
    signature,
    signature_payload,
    cameras,
    query_cache,
    gaussians,
    landmark_indices,
    native_sparse=False,
):
    path = Path(path) if path else None
    if path is not None and path.exists():
        payload = torch.load(path, map_location="cpu")
        if payload.get("signature") != signature:
            raise ValueError(
                f"Visibility cache signature mismatch for {path}; use a matching scaffold"
            )
        cached = payload.get("visibility", {})
        missing = [
            _camera_cache_key(camera)
            for camera in cameras
            if _camera_cache_key(camera) not in cached
        ]
        if not missing:
            print(f"Loaded rasterizer visibility cache: {path} views={len(cached)}")
            return cached
    visibility = {}
    for camera in tqdm(cameras, desc="2DGS contribution visibility"):
        name = _camera_cache_key(camera)
        if native_sparse:
            height, width = map(int, query_cache[name]["native_input_hw"])
        else:
            height, width = query_cache[name]["feature_map"].shape[-2:]
        visibility[name] = _render_bank_visibility(
            gaussians,
            camera,
            width,
            height,
            landmark_indices,
        )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": 2,
                "signature": signature,
                "signature_payload": signature_payload,
                "visibility": visibility,
            },
            path,
        )
        print(f"Saved rasterizer visibility cache: {path}")
    return visibility


@torch.no_grad()
def _build_mvinit_features(
    query_names,
    cache,
    bank_xyz,
    fallback_features,
    args,
    visibility_cache=None,
):
    feature_dim = int(fallback_features.shape[1])
    feature_sum = torch.zeros(
        (bank_xyz.shape[0], feature_dim), device=bank_xyz.device, dtype=torch.float32
    )
    observation_count = torch.zeros(
        bank_xyz.shape[0], device=bank_xyz.device, dtype=torch.long
    )
    for name in tqdm(query_names, desc="Detector-free MVInit"):
        observations = _cached_observations(
            cache[name],
            bank_xyz,
            args,
            max_observations=args.mvinit_max_observations,
            bank_visibility_mask=(
                None if visibility_cache is None else visibility_cache[name]
            ),
        )
        if observations.source_indices.numel() == 0:
            continue
        feature_sum.index_add_(
            0,
            observations.source_indices,
            observations.query_features.float(),
        )
        observation_count.index_add_(
            0,
            observations.source_indices,
            torch.ones_like(observations.source_indices),
        )
    observed = observation_count > 0
    result = F.normalize(fallback_features.float(), dim=-1)
    if bool(observed.any().item()):
        result = result.clone()
        result[observed] = F.normalize(feature_sum[observed], dim=-1)
    mvinit_mode = str(args.mvinit_mode)
    if mvinit_mode == "medoid" and bool(observed.any().item()):
        mean_features = result.clone()
        best_score = result.new_full((result.shape[0],), -torch.inf)
        medoid_features = result.clone()
        for name in tqdm(query_names, desc="Detector-free MVInit medoid"):
            observations = _cached_observations(
                cache[name],
                bank_xyz,
                args,
                max_observations=args.mvinit_max_observations,
                bank_visibility_mask=(
                    None if visibility_cache is None else visibility_cache[name]
                ),
            )
            source = observations.source_indices
            if source.numel() == 0:
                continue
            candidates = F.normalize(observations.query_features.float(), dim=-1)
            score = (candidates * mean_features[source]).sum(dim=-1)
            better = score > best_score[source]
            if bool(better.any().item()):
                selected_source = source[better]
                best_score[selected_source] = score[better]
                medoid_features[selected_source] = candidates[better]
        result[observed] = F.normalize(medoid_features[observed], dim=-1)
    elif mvinit_mode != "mean":
        raise ValueError(f"Unknown MVInit mode: {mvinit_mode}")
    diagnostics = {
        "mvinit_mode": mvinit_mode,
        "observed_landmarks": int(observed.sum().item()),
        "unobserved_landmarks": int((~observed).sum().item()),
        "observation_count_mean": float(observation_count.float().mean().item()),
        "observation_count_median": float(observation_count.float().median().item()),
        "observation_count_max": int(observation_count.max().item()),
    }
    return result, observation_count, diagnostics


def _load_initial_features(
    path,
    landmark_indices,
    feature_dim,
    device,
    *,
    fallback_features=None,
    alignment="exact",
):
    state = torch.load(path, map_location="cpu")
    state_indices = torch.as_tensor(state.get("landmark_indices"), dtype=torch.long)
    features = torch.as_tensor(state.get("landmark_features"), dtype=torch.float32)
    features = features.reshape(state_indices.numel(), -1)
    if int(features.shape[1]) != int(feature_dim):
        raise ValueError(
            f"Initial state descriptor dimension is {features.shape[1]}, expected {feature_dim}"
        )
    landmark_indices_cpu = landmark_indices.cpu().reshape(-1)
    state_indices = state_indices.reshape(-1)
    state = dict(state)
    raw_anchor_offset = state.get("raw_anchor_offset")
    raw_offset_valid = False
    if raw_anchor_offset is not None:
        raw_anchor_offset = torch.as_tensor(
            raw_anchor_offset, dtype=torch.float32
        )
        raw_offset_valid = (
            raw_anchor_offset.numel() == state_indices.numel() * 3
        )
        if raw_offset_valid:
            raw_anchor_offset = raw_anchor_offset.reshape(
                state_indices.numel(), 3
            )
    if torch.equal(state_indices, landmark_indices_cpu):
        if raw_offset_valid:
            state["raw_anchor_offset"] = raw_anchor_offset
        state["_raw_anchor_offset_alignment_valid"] = raw_offset_valid
        return F.normalize(features.to(device), dim=-1), state, int(
            landmark_indices.numel()
        )
    if str(alignment) != "overlap":
        raise ValueError("Initial state landmark IDs do not match the fixed scaffold")
    if fallback_features is None:
        raise ValueError("overlap-aligned initialization requires fallback_features")
    order = torch.argsort(state_indices)
    sorted_indices = state_indices[order]
    positions = torch.searchsorted(sorted_indices, landmark_indices_cpu)
    in_range = positions < sorted_indices.numel()
    matched = torch.zeros_like(in_range)
    matched[in_range] = (
        sorted_indices[positions[in_range]] == landmark_indices_cpu[in_range]
    )
    result = F.normalize(fallback_features.float(), dim=-1).clone()
    if bool(matched.any().item()):
        source_position = order[positions[matched]]
        result[matched.to(device=result.device)] = F.normalize(
            features[source_position].to(device), dim=-1
        )
        if raw_offset_valid:
            aligned_offset = torch.zeros(
                landmark_indices.numel(),
                3,
                dtype=raw_anchor_offset.dtype,
            )
            aligned_offset[matched] = raw_anchor_offset[source_position]
            state["raw_anchor_offset"] = aligned_offset
    elif raw_offset_valid:
        state["raw_anchor_offset"] = torch.zeros(
            landmark_indices.numel(),
            3,
            dtype=raw_anchor_offset.dtype,
        )
    state["_raw_anchor_offset_alignment_valid"] = raw_offset_valid
    return result, state, int(matched.sum().item())


def _fallback_features(gaussians, landmark_indices, feature_dim):
    features = gaussians.get_loc_feature[landmark_indices.cuda()].reshape(
        landmark_indices.numel(), -1
    )
    if int(features.shape[1]) != int(feature_dim):
        return F.normalize(
            torch.randn(
                landmark_indices.numel(),
                feature_dim,
                device=gaussians.get_xyz.device,
            ),
            dim=-1,
        )
    finite = torch.isfinite(features).all(dim=1)
    normalized = F.normalize(
        torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0), dim=-1
    )
    invalid = (~finite) | (torch.linalg.norm(normalized, dim=-1) < 1e-6)
    if bool(invalid.any().item()):
        normalized[invalid] = F.normalize(
            torch.randn(
                int(invalid.sum().item()),
                feature_dim,
                device=normalized.device,
            ),
            dim=-1,
        )
    return normalized


def _mean_diagnostics(records):
    keys = sorted({key for record in records for key in record})
    return {
        key: float(
            sum(float(record[key]) for record in records if key in record)
            / max(sum(1 for record in records if key in record), 1)
        )
        for key in keys
    }


def _all_parameter_gradients_finite(parameters):
    for parameter in parameters:
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all().item()
        ):
            return False
    return True


def _clear_inactive_phase_gradients(
    residual,
    raw_anchor_offset,
    dustbin_score,
    *,
    descriptor_update_active,
    geometry_update_active,
    dustbin_update_active,
):
    """Prevent AdamW, including decoupled decay, from changing frozen state."""
    if not descriptor_update_active:
        residual.grad = None
    if not geometry_update_active:
        raw_anchor_offset.grad = None
    if dustbin_score is not None and not dustbin_update_active:
        dustbin_score.grad = None


@torch.no_grad()
def _collect_landmark_statistics(
    features,
    query_names,
    cache,
    bank_xyz,
    args,
    visibility_cache=None,
    base_bank_xyz=None,
):
    count = int(bank_xyz.shape[0])
    device = bank_xyz.device
    observation_count = torch.zeros(count, device=device)
    correct_count = torch.zeros(count, device=device)
    source_top1_count = torch.zeros(count, device=device)
    proposal_observation_count = torch.zeros(count, device=device)
    proposal_correct_count = torch.zeros(count, device=device)
    margin_sum = torch.zeros(count, device=device)
    entropy_sum = torch.zeros(count, device=device)
    reprojection_sum = torch.zeros(count, device=device)
    translation_fim_sum = torch.zeros(count, device=device)
    uv_sum = torch.zeros((count, 2), device=device)
    depth_sum = torch.zeros(count, device=device)
    normalized_features = F.normalize(features, dim=1)
    records = []
    for name in tqdm(query_names, desc="One-time landmark statistics"):
        observations = _cached_observations(
            cache[name],
            bank_xyz if base_bank_xyz is None else base_bank_xyz,
            args,
            max_observations=args.statistics_observations,
            bank_visibility_mask=(
                None if visibility_cache is None else visibility_cache[name]
            ),
            prediction_bank_xyz=bank_xyz,
        )
        source = observations.source_indices
        if source.numel() == 0:
            continue
        query = F.normalize(observations.query_features, dim=1)
        scores = query @ normalized_features.T
        score_topk = min(max(int(args.statistics_hypothesis_topk), 2), count)
        top_values, top_indices = torch.topk(scores, score_topk, dim=1)
        top1 = top_indices[:, 0]
        top1_distance = torch.linalg.norm(
            observations.bank_uv[top1] - observations.query_uv, dim=1
        )
        clean = observations.bank_visible[top1] & (
            top1_distance <= float(args.positive_radius_px)
        )
        source_score = scores.gather(1, source[:, None]).squeeze(1)
        competitor = torch.where(
            top1 == source,
            top_values[:, 1],
            top_values[:, 0],
        )
        margin = source_score - competitor
        probability = torch.softmax(
            top_values / max(float(args.temperature), 1e-6), dim=1
        )
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1)
        entropy /= max(float(torch.log(torch.tensor(score_topk)).item()), 1.0)
        local = local_soft_correspondences(
            features,
            observations,
            radius=args.local_radius,
            temperature=args.local_temperature,
        )
        local_error = torch.linalg.norm(
            local.expected_uv - observations.query_uv, dim=1
        )
        top1_projected = observations.bank_projected[top1]
        finite_top1_distance = top1_distance[
            top1_projected & torch.isfinite(top1_distance)
        ]
        ones = torch.ones_like(source, dtype=torch.float32)
        observation_count.index_add_(0, source, ones)
        correct_count.index_add_(0, source, clean.float())
        source_top1_count.index_add_(0, source, (top1 == source).float())
        margin_sum.index_add_(0, source, margin)
        entropy_sum.index_add_(0, source, entropy)
        reprojection_sum.index_add_(0, source, local_error)
        height, width = observations.query_feature_map.shape[-2:]
        normalized_uv = observations.query_uv.clone()
        normalized_uv[:, 0] /= max(float(width - 1), 1.0)
        normalized_uv[:, 1] /= max(float(height - 1), 1.0)
        uv_sum.index_add_(0, source, normalized_uv)
        depth_sum.index_add_(0, source, observations.source_depth)
        if source.numel() >= 6:
            information = compute_pose_information(
                bank_xyz[source],
                observations.K,
                observations.pose_w2c,
                weights=clean.float().clamp_min(0.05),
                translation_scale=args.pose_translation_scale_m,
                rotation_scale=float(
                    torch.deg2rad(torch.tensor(args.pose_rotation_scale_deg)).item()
                ),
            )
            translation_fim_sum.index_add_(
                0, source, information.translation_scores
            )
        proposal_observations = _cached_score_proposal_observations(
            cache[name],
            observations,
            args,
        )
        proposal_precision = 0.0
        proposal_count = 0
        if proposal_observations is not None:
            proposal_source_valid = proposal_observations.source_indices >= 0
            if bool(proposal_source_valid.any().item()):
                proposal_source = proposal_observations.source_indices[
                    proposal_source_valid
                ]
                proposal_query = F.normalize(
                    proposal_observations.query_features[proposal_source_valid],
                    dim=1,
                )
                proposal_scores = proposal_query @ normalized_features.T
                proposal_top1 = proposal_scores.argmax(dim=1)
                proposal_distance = torch.linalg.norm(
                    proposal_observations.bank_uv[proposal_top1]
                    - proposal_observations.query_uv[proposal_source_valid],
                    dim=1,
                )
                proposal_clean = (
                    proposal_observations.bank_visible[proposal_top1]
                    & (proposal_distance <= float(args.positive_radius_px))
                )
                proposal_ones = torch.ones_like(
                    proposal_source, dtype=torch.float32
                )
                proposal_observation_count.index_add_(
                    0, proposal_source, proposal_ones
                )
                proposal_correct_count.index_add_(
                    0, proposal_source, proposal_clean.float()
                )
                proposal_precision = float(proposal_clean.float().mean().item())
                proposal_count = int(proposal_source.numel())
        records.append(
            {
                "observations": int(source.numel()),
                "top1_clean_precision": float(clean.float().mean().item()),
                "top1_reprojection_median_px": float(
                    finite_top1_distance.median().item()
                    if finite_top1_distance.numel()
                    else 0.0
                ),
                "top1_projected_ratio": float(top1_projected.float().mean().item()),
                "source_margin_mean": float(margin.mean().item()),
                "hypothesis_entropy_mean": float(entropy.mean().item()),
                "proposal_matched_observations": proposal_count,
                "proposal_top1_clean_precision": proposal_precision,
            }
        )
    proposal_weight = float(args.distill_proposal_weight)
    effective_count = observation_count + proposal_weight * proposal_observation_count
    effective_correct = correct_count + proposal_weight * proposal_correct_count
    denominator = observation_count.clamp_min(1.0)
    effective_denominator = effective_count.clamp_min(1.0)
    matchability = (effective_correct + 1.0) / (effective_count + 2.0)
    statistics = {
        "observation_count": observation_count,
        "correct_count": correct_count,
        "source_top1_count": source_top1_count,
        "source_identity_rate": (source_top1_count + 1.0)
        / (observation_count + 2.0),
        "proposal_observation_count": proposal_observation_count,
        "proposal_correct_count": proposal_correct_count,
        "effective_observation_count": effective_count,
        "false_top1_rate": (
            1.0 - effective_correct / effective_denominator
        ).clamp(0.0, 1.0),
        "matchability": matchability,
        "margin": margin_sum / denominator,
        "entropy": entropy_sum / denominator,
        "reprojection_error": reprojection_sum / denominator,
        "translation_fim": translation_fim_sum / denominator,
        "mean_uv": uv_sum / denominator[:, None],
        "mean_depth": depth_sum / denominator,
    }
    diagnostics = _mean_diagnostics(records)
    diagnostics.update(
        {
            "query_count": len(records),
            "observed_landmark_count": int((effective_count > 0).sum().item()),
            "proposal_observed_landmark_count": int(
                (proposal_observation_count > 0).sum().item()
            ),
            "matchability_mean": float(
                matchability[effective_count > 0].mean().item()
                if bool((effective_count > 0).any())
                else 0.0
            ),
        }
    )
    return statistics, diagnostics


@torch.no_grad()
def _distill_final_landmark_bank(
    output_dir,
    landmark_indices,
    features,
    bank_xyz,
    raw_anchor_offset,
    statistics,
    args,
    config,
    mvinit_observation_count,
    dustbin_score,
):
    observation_count = statistics.get(
        "effective_observation_count", statistics["observation_count"]
    )
    matchability = statistics["matchability"]
    false_top1_rate = statistics.get(
        "false_top1_rate", 1.0 - matchability
    )
    repeatability = (
        observation_count
        / observation_count[observation_count > 0].median().clamp_min(1.0)
        if bool((observation_count > 0).any())
        else torch.zeros_like(observation_count)
    ).clamp(0.0, 1.0)
    margin_quality = torch.sigmoid(robust_normalize(statistics["margin"]))
    entropy_quality = (1.0 - statistics["entropy"]).clamp(0.0, 1.0)
    reprojection_quality = torch.exp(
        -statistics["reprojection_error"].clamp_min(0.0)
        / max(float(args.distill_reprojection_scale_px), 1e-6)
    )
    fim_quality = torch.sigmoid(
        robust_normalize(statistics["translation_fim"])
    )
    utility = (
        matchability
        * (0.5 + 0.5 * repeatability)
        * (0.5 + 0.5 * margin_quality)
        * (0.5 + 0.5 * entropy_quality)
        * (0.5 + 0.5 * reprojection_quality)
        * (0.75 + 0.25 * fim_quality)
    )
    extent = bank_xyz.quantile(0.99, dim=0) - bank_xyz.quantile(0.01, dim=0)
    voxel_size = float(args.distill_voxel_size)
    if voxel_size <= 0.0:
        voxel_size = float(torch.linalg.norm(extent).item()) / 40.0
    observed_eligible = observation_count >= float(args.distill_min_observations)
    eligible = (
        observed_eligible
        & (matchability >= float(args.distill_matchability_threshold))
        & (false_top1_rate <= float(args.distill_false_top1_max))
    )
    eligibility_relaxed = "none"
    rank_pool_size = int(eligible.sum().item())
    if int(eligible.sum().item()) < int(args.distill_budget):
        observed_indices = torch.nonzero(
            observed_eligible, as_tuple=False
        ).reshape(-1)
        if observed_indices.numel() > 0:
            reliable_order = torch.argsort(
                matchability[observed_indices], descending=True, stable=True
            )
            # Keep a larger matchability-first reservoir.  Coverage and FIM
            # can then choose the final fixed-size bank instead of becoming
            # inert because the fallback exposed exactly K candidates.
            requested_pool = max(
                int(args.distill_budget),
                int(
                    round(
                        float(args.distill_budget)
                        * max(float(args.distill_rank_pool_multiplier), 1.0)
                    )
                ),
            )
            keep = min(requested_pool, int(observed_indices.numel()))
            eligible = torch.zeros_like(eligible)
            eligible[observed_indices[reliable_order[:keep]]] = True
            rank_pool_size = keep
        eligibility_relaxed = "matchability_rank_pool"
    effective_budget = min(int(args.distill_budget), int(eligible.sum().item()))
    if effective_budget <= 0:
        raise ValueError("No observed landmarks are available for final distillation")
    if effective_budget < int(args.distill_budget):
        eligibility_relaxed = "observed_shortfall"
        if bool(args.distill_require_exact_budget):
            raise ValueError(
                "Final landmark distillation could not satisfy the requested "
                f"budget ({effective_budget}/{int(args.distill_budget)} observed "
                "eligible landmarks). Increase --statistics_observations or use "
                "a smaller fixed comparison budget."
            )
    selected, selection_meta = coverage_preserving_sample(
        bank_xyz,
        base_score=matchability,
        utility=utility,
        num=effective_budget,
        min_observations=eligible,
        base_preserve_ratio=args.distill_matchability_preserve_ratio,
        utility_preserve_ratio=args.distill_utility_preserve_ratio,
        high_confidence=(matchability >= float(args.distill_high_confidence)),
        high_confidence_ratio=args.distill_high_confidence_ratio,
        voxel_size=voxel_size,
        max_per_voxel=args.distill_max_per_voxel,
        uv=statistics["mean_uv"] * 1000.0,
        image_size=(1000, 1000),
        grid_size=args.distill_grid_size,
        max_per_grid=args.distill_max_per_grid,
        depth=statistics["mean_depth"],
        depth_bins=args.distill_depth_bins,
        max_per_depth_bin=args.distill_max_per_depth_bin,
        allow_unbalanced_fallback=True,
    )
    output_dir = Path(output_dir)
    selected_global = landmark_indices[selected.cpu()]
    sampled_path = output_dir / "distilled_sampled_idx.pkl"
    with sampled_path.open("wb") as handle:
        pickle.dump(selected_global.detach().cpu(), handle)
    distilled_config = dict(config)
    distilled_config["one_time_landmark_distillation"] = True
    distilled_config["distillation"] = {
        "source_pool_size": int(landmark_indices.numel()),
        "final_budget": int(selected.numel()),
        "voxel_size": voxel_size,
        "uses_matchability": True,
        "matchability_first": True,
        "matchability_threshold": float(args.distill_matchability_threshold),
        "false_top1_max": float(args.distill_false_top1_max),
        "proposal_weight": float(args.distill_proposal_weight),
        "uses_conditional_translation_fim": True,
        "uses_3d_image_depth_coverage": True,
        "eligibility_relaxed": eligibility_relaxed,
        "rank_pool_size": int(rank_pool_size),
        "rank_pool_multiplier": float(args.distill_rank_pool_multiplier),
        "requested_budget": int(args.distill_budget),
        "observed_budget_shortfall": int(args.distill_budget) - effective_budget,
    }
    distilled_statistics = {
        key: value[selected].detach().cpu() for key, value in statistics.items()
    }
    distilled_meta = {
        "landmark_indices": selected_global.detach().cpu(),
        "candidate_quality": utility[selected].detach().cpu(),
        "utility": utility[selected].detach().cpu(),
        "matchability": matchability[selected].detach().cpu(),
        "false_top1_rate": false_top1_rate[selected].detach().cpu(),
        "source_identity_rate": statistics["source_identity_rate"][
            selected
        ].detach().cpu(),
        "repeatability": repeatability[selected].detach().cpu(),
        "margin": statistics["margin"][selected].detach().cpu(),
        "reproj_error": statistics["reprojection_error"][selected].detach().cpu(),
        "information": statistics["translation_fim"][selected].detach().cpu(),
        "translation_fim": statistics["translation_fim"][selected].detach().cpu(),
        "coverage_uv": statistics["mean_uv"][selected].detach().cpu(),
        "coverage_image_size": torch.tensor([1000.0, 1000.0]),
        "coverage_grid_size": int(args.distill_grid_size),
        "coverage_depth": statistics["mean_depth"][selected].detach().cpu(),
        "coverage_depth_bins": int(args.distill_depth_bins),
        "selection_meta": selection_meta,
    }
    meta_path = output_dir / "landmark_meta.pt"
    torch.save(distilled_meta, meta_path)
    state_path = output_dir / "distilled_lafgs_map_state.pt"
    state = {
        "version": 7,
        "iteration": int(args.steps),
        "landmark_indices": selected_global.detach().cpu(),
        "landmark_features": F.normalize(features[selected], dim=1).detach().cpu(),
        "landmark_xyz": bank_xyz[selected].detach().cpu(),
        "raw_anchor_offset": raw_anchor_offset[selected].detach().cpu(),
        "mvinit_observation_count": mvinit_observation_count[selected].detach().cpu(),
        "landmark_statistics": distilled_statistics,
        "selection_meta": selection_meta,
        "config": distilled_config,
    }
    if dustbin_score is not None:
        state["dustbin_score"] = float(dustbin_score.detach().item())
    torch.save(state, state_path)
    return {
        "sampled_idx_path": str(sampled_path),
        "state_path": str(state_path),
        "landmark_meta_path": str(meta_path),
        "source_pool_size": int(landmark_indices.numel()),
        "selected_count": int(selected.numel()),
        "utility_selected_mean": float(utility[selected].mean().item()),
        "matchability_selected_mean": float(matchability[selected].mean().item()),
        "translation_fim_selected_mean": float(
            statistics["translation_fim"][selected].mean().item()
        ),
        "eligibility_relaxed": eligibility_relaxed,
        "rank_pool_size": int(rank_pool_size),
    }


@torch.no_grad()
def _validate_descriptor_field(
    features,
    validation_names,
    cache,
    bank_xyz,
    args,
    visibility_cache=None,
    dustbin_score=None,
    base_bank_xyz=None,
):
    records = []
    for name in tqdm(validation_names, desc="Detector-free validation"):
        observations, anchor_auxiliary = _primary_observations(
            cache[name],
            bank_xyz if base_bank_xyz is None else base_bank_xyz,
            args,
            max_observations=args.validation_observations,
            bank_visibility_mask=(
                None if visibility_cache is None else visibility_cache[name]
            ),
            prediction_bank_xyz=bank_xyz,
        )
        if observations.query_features.numel() == 0:
            continue
        retrieval = hard_hypothesis_retrieval_loss(
            features,
            observations,
            hypothesis_topk=args.hypothesis_topk,
            temperature=args.temperature,
            positive_radius_px=args.positive_radius_px,
            negative_radius_px=args.negative_radius_px,
            margin=args.retrieval_margin,
            missed_positive_weight=args.missed_positive_weight,
            missed_positive_margin=args.missed_positive_margin,
            unmatched_rejection_weight=args.unmatched_rejection_weight,
            unmatched_max_similarity=args.unmatched_max_similarity,
            dustbin_score=dustbin_score,
        )
        record = dict(retrieval.diagnostics)
        proposal_observations = (
            _cached_score_proposal_observations(cache[name], observations, args)
            if str(args.observation_source) == "anchor"
            else None
        )
        if proposal_observations is not None:
            proposal_retrieval = hard_hypothesis_retrieval_loss(
                features,
                proposal_observations,
                hypothesis_topk=args.hypothesis_topk,
                temperature=args.temperature,
                positive_radius_px=args.positive_radius_px,
                negative_radius_px=args.negative_radius_px,
                margin=args.retrieval_margin,
                missed_positive_weight=args.missed_positive_weight,
                missed_positive_margin=args.missed_positive_margin,
                unmatched_rejection_weight=args.unmatched_rejection_weight,
                unmatched_max_similarity=args.unmatched_max_similarity,
                dustbin_score=dustbin_score,
            )
            record.update(
                {
                    f"generic_proposal_{key}": value
                    for key, value in proposal_retrieval.diagnostics.items()
                }
            )
            record["generic_proposal_loss"] = float(
                proposal_retrieval.loss.item()
            )
            record["generic_proposal_observations"] = int(
                proposal_observations.source_indices.numel()
            )
        if str(args.observation_source) == "anchor":
            auxiliary_observations = observations
            auxiliary_scale = 1.0
        else:
            auxiliary_observations = anchor_auxiliary
            auxiliary_scale = float(args.native_anchor_aux_weight)
        if auxiliary_observations is not None:
            record["mv_loss"] = float(
                auxiliary_scale
                * multiview_descriptor_loss(features, auxiliary_observations).item()
            )
        else:
            record["mv_loss"] = 0.0
        if args.local_weight > 0.0 and auxiliary_observations is not None:
            local_loss, local_diagnostics = local_correlation_peak_loss(
                features, auxiliary_observations,
                radius=args.local_radius,
                target_sigma=args.local_target_sigma,
                temperature=args.local_temperature,
            )
            record["local_loss"] = float(local_loss.item())
            record.update(local_diagnostics)
        record["visible_observations"] = int(observations.query_features.shape[0])
        record["matched_observations"] = int(
            (observations.source_indices >= 0).sum().item()
        )
        record["unmatched_observations"] = int(
            (observations.source_indices < 0).sum().item()
        )
        record["native_candidate_observations"] = float(
            str(args.observation_source) != "anchor"
        )
        records.append(record)
    summary = _mean_diagnostics(records)
    summary["validation_query_count"] = len(records)
    return summary


def _state_config(
    args,
    dataset,
    train_names,
    validation_names,
    landmark_path,
    scaffold_diagnostics,
):
    return {
        "method": "lafgs_kcs_gwff_alternating_native_candidate_v1",
        "detector_free": True,
        "query_cache_policy": str(args.query_cache_policy),
        "query_feature_contract": str(args.query_feature_contract),
        "scene_detector_used_for_map_training": False,
        "observation_source": str(args.observation_source),
        "native_sparse_keypoint_count": int(args.native_keypoint_count),
        "native_sparse_coordinate_convention": (
            "superpoint_grid_index_then_pnp_plus_half_v1"
        ),
        "native_association_radius_px": float(args.native_association_radius_px),
        "native_unmatched_fraction": float(args.native_unmatched_fraction),
        "native_sampling_mode": str(args.native_sampling_mode),
        "native_anchor_aux_weight": float(args.native_anchor_aux_weight),
        "frozen_generic_proposal_head": bool(args.generic_proposal_count > 0),
        "geometry_frozen": not bool(args.geometry_weight > 0.0),
        "raw_xyz_trainable": False,
        "bounded_anchor_trainable": bool(args.geometry_weight > 0.0),
        "dynamic_landmark_selection": False,
        "one_time_landmark_distillation": bool(args.distill_budget > 0),
        "online_rendering": False,
        "fim_loss_enabled": False,
        "pair_enabled": False,
        "topology_enabled": False,
        "objective": str(args.objective),
        "steps": int(args.steps),
        # Keep the requested checkpoint grid in every state.  It makes a
        # missing intermediate checkpoint a visible protocol failure instead
        # of silently changing the validation selection set.
        "checkpoint_save_steps": sorted(
            {int(step) for step in args.save_steps} | {int(args.steps)}
        ),
        "feature_lr": float(args.feature_lr),
        "geometry_lr": float(args.geometry_lr),
        "weight_decay": float(args.weight_decay),
        "mv_weight": float(args.mv_weight),
        "retrieval_weight": float(args.retrieval_weight),
        "trust_weight": float(args.trust_weight),
        "local_weight": float(args.local_weight),
        "local_correlation_enabled": bool(args.local_weight > 0.0),
        "local_radius": int(args.local_radius),
        "local_target_sigma": float(args.local_target_sigma),
        "local_temperature": float(args.local_temperature),
        "temperature": float(args.temperature),
        "hypothesis_topk": int(args.hypothesis_topk),
        "random_negative_count": int(args.random_negative_count),
        "positive_radius_px": float(args.positive_radius_px),
        "negative_radius_px": float(args.negative_radius_px),
        "retrieval_margin": float(args.retrieval_margin),
        "missed_positive_weight": float(args.missed_positive_weight),
        "missed_positive_margin": float(args.missed_positive_margin),
        "unmatched_rejection_weight": float(args.unmatched_rejection_weight),
        "unmatched_max_similarity": float(args.unmatched_max_similarity),
        "visibility_mode": str(args.visibility_mode),
        "proposal_jitter_std": float(args.proposal_jitter_std),
        "proposal_jitter_max": float(args.proposal_jitter_max),
        "generic_proposal_weight": float(args.generic_proposal_weight),
        "generic_proposal_count": int(args.generic_proposal_count),
        "generic_proposal_nms_radius": int(args.generic_proposal_nms_radius),
        "generic_proposal_score_threshold": float(
            args.generic_proposal_score_threshold
        ),
        "generic_proposal_positive_radius": float(
            args.generic_proposal_positive_radius
        ),
        "generic_proposal_include_unmatched": bool(
            args.generic_proposal_include_unmatched
        ),
        "dustbin_weight": float(args.dustbin_weight),
        "dustbin_background_count": int(args.dustbin_background_count),
        "dustbin_background_alpha_max": float(args.dustbin_background_alpha_max),
        "dustbin_no_anchor": bool(args.dustbin_no_anchor),
        "geometry_start_step": int(args.geometry_start_step),
        "geometry_weight": float(args.geometry_weight),
        "geometry_mode": str(args.geometry_mode),
        "geometry_association_max_reprojection_px": float(
            args.geometry_association_max_reprojection_px
        ),
        "geometry_association_min_margin": float(
            args.geometry_association_min_margin
        ),
        "surface_weight": float(args.surface_weight),
        "depth_weight": float(args.depth_weight),
        "reprojection_weight": float(args.reprojection_weight),
        "tangent_bound_m": float(args.tangent_bound_m),
        "normal_bound_m": float(args.normal_bound_m),
        "pose_start_step": int(args.pose_start_step),
        "pose_interval": int(args.pose_interval),
        "pose_weight": float(args.pose_weight),
        "pose_gradient_mode": str(args.pose_gradient_mode),
        "pose_iterations": int(args.pose_iterations),
        "trust_observation_power": float(args.trust_observation_power),
        "initial_state_blend": float(args.initial_state_blend),
        "initial_state_alignment": str(args.initial_state_alignment),
        "initialization_mode": str(args.initialization_mode),
        "mvinit_mode": str(args.mvinit_mode),
        "descriptor_end_step": int(args.descriptor_end_step),
        "mvinit_max_observations": int(args.mvinit_max_observations),
        "ulf_consensus_keypoints": int(args.ulf_consensus_keypoints),
        "ulf_consensus_radius_px": float(args.ulf_consensus_radius_px),
        "ulf_consensus_min_votes": int(args.ulf_consensus_min_votes),
        "ulf_fusion_min_cosine": float(args.ulf_fusion_min_cosine),
        "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "distill_budget": int(args.distill_budget),
        "distill_require_exact_budget": bool(args.distill_require_exact_budget),
        "distill_rank_pool_multiplier": float(
            args.distill_rank_pool_multiplier
        ),
        "statistics_observations": int(args.statistics_observations),
        "distill_min_observations": int(args.distill_min_observations),
        "distill_matchability_threshold": float(
            args.distill_matchability_threshold
        ),
        "distill_false_top1_max": float(args.distill_false_top1_max),
        "distill_proposal_weight": float(args.distill_proposal_weight),
        "max_observations": int(args.max_observations),
        "validation_ratio": float(args.validation_ratio),
        "split_mode": str(args.split_mode),
        "split_seed": int(args.split_seed),
        "camera_order": "image_name_lexicographic",
        "train_camera_count": int(len(train_names)),
        "validation_camera_count": int(len(validation_names)),
        "train_camera_names_sha256": _camera_names_sha256(train_names),
        "validation_camera_names_sha256": _camera_names_sha256(validation_names),
        "input_camera_names_sha256": _camera_names_sha256(
            list(train_names) + list(validation_names)
        ),
        "model_path": os.path.abspath(dataset.model_path),
        "source_path": os.path.abspath(dataset.source_path),
        "map_iteration": int(args.load_iteration),
        "landmark_path": str(landmark_path),
        "scaffold": dict(scaffold_diagnostics),
    }


def _save_state(
    path,
    iteration,
    landmark_indices,
    features,
    config,
    diagnostics,
    mvinit_observation_count,
    dustbin_score=None,
    landmark_xyz=None,
    raw_anchor_offset=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 6,
        "iteration": int(iteration),
        "landmark_indices": landmark_indices.detach().cpu(),
        "landmark_features": F.normalize(features.detach().float(), dim=-1).cpu(),
        "config": dict(config),
        "diagnostics": dict(diagnostics),
        "mvinit_observation_count": mvinit_observation_count.detach().cpu(),
    }
    if landmark_xyz is not None:
        state["landmark_xyz"] = torch.as_tensor(landmark_xyz).detach().float().cpu()
    if raw_anchor_offset is not None:
        state["raw_anchor_offset"] = (
            torch.as_tensor(raw_anchor_offset).detach().float().cpu()
        )
    if dustbin_score is not None:
        state["dustbin_score"] = float(
            torch.as_tensor(dustbin_score).detach().cpu().item()
        )
    torch.save(state, path)
    sampled_path = Path(path).parent / "sampled_idx.pkl"
    with sampled_path.open("wb") as handle:
        pickle.dump(landmark_indices.detach().cpu(), handle)


def _checkpoint_integrity(output_dir, requested_steps):
    """Return a deterministic audit of the requested map checkpoints."""
    output_dir = Path(output_dir)
    requested_steps = sorted({int(step) for step in requested_steps})
    saved_steps = [
        step
        for step in requested_steps
        if (output_dir / f"{step}_lafgs_map_state.pt").is_file()
    ]
    missing_steps = sorted(set(requested_steps) - set(saved_steps))
    return {
        "requested_steps": requested_steps,
        "saved_steps": saved_steps,
        "missing_steps": missing_steps,
        "complete": not missing_steps,
    }


def train(dataset, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    native_observation_mode = str(args.observation_source) in {
        "native",
        "native_plus_anchor",
    }
    if native_observation_mode and str(args.query_feature_contract) != _QUERY_FEATURE_CONTRACT_NATIVE:
        raise ValueError(
            "Native candidate supervision requires --query_feature_contract "
            f"{_QUERY_FEATURE_CONTRACT_NATIVE} so training and deployment share "
            "the same resized SuperPoint input."
        )
    if native_observation_mode and int(args.generic_proposal_count) > 0:
        raise ValueError(
            "generic proposal maps are a dense auxiliary and cannot be mixed "
            "into native candidate supervision; use native_plus_anchor instead"
        )
    if str(args.geometry_mode) == "native_association" and not native_observation_mode:
        raise ValueError(
            "native_association geometry requires --observation_source native "
            "or native_plus_anchor"
        )
    if (
        str(args.geometry_mode) == "native_association"
        and str(args.pose_gradient_mode) == "geometry"
        and float(args.pose_weight) > 0.0
    ):
        raise ValueError(
            "native_association geometry has no local-anchor pose measurement; "
            "run fixed-pose BA with --pose_gradient_mode off"
        )
    if (
        str(args.geometry_mode) == "native_association"
        and float(args.geometry_weight) > 0.0
        and int(args.descriptor_end_step) >= 0
    ):
        raise ValueError(
            "native_association is an alternating fixed-descriptor BA stage; "
            "set --descriptor_end_step -1 and run descriptor refresh separately"
        )
    if (
        args.scaffold_mode == "ulf_consensus"
        or args.initialization_mode == "ulf_geometry"
    ) and str(args.query_feature_contract) != _QUERY_FEATURE_CONTRACT_NATIVE:
        raise ValueError(
            "ULF KCS/GWFF requires --query_feature_contract "
            f"{_QUERY_FEATURE_CONTRACT_NATIVE}; otherwise its native sparse "
            "descriptors would be initialized against a different cache contract."
        )
    gaussians = _gaussian_model_for_type(dataset.gaussian_type, dataset.sh_degree)
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.load_iteration,
        shuffle=False,
        preload_cameras=False,
        load_test_cameras=False,
    )
    if not scene.loaded_iter:
        raise ValueError("A pretrained Gaussian map checkpoint is required")
    for parameter in gaussians.parameters():
        parameter.requires_grad_(False)

    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    for parameter in feature_extractor.parameters():
        parameter.requires_grad_(False)

    all_train_cameras = sorted(
        scene.getTrainCameras(),
        key=lambda camera: _camera_cache_key(camera),
    )
    if float(args.validation_ratio) <= 0.0:
        train_cameras = all_train_cameras
        validation_cameras = []
    else:
        train_cameras, validation_cameras = split_support_query_cameras(
            all_train_cameras,
            query_ratio=args.validation_ratio,
            seed=args.split_seed + 1,
            mode=args.split_mode,
        )
    train_cameras = _uniformly_subsample_cameras(
        train_cameras, args.max_train_views
    )
    validation_cameras = _uniformly_subsample_cameras(
        validation_cameras, args.max_validation_views
    )
    train_names = [_camera_cache_key(camera) for camera in train_cameras]
    validation_names = [_camera_cache_key(camera) for camera in validation_cameras]
    print(
        "LaFGS detector-free split: "
        f"train={len(train_names)} validation={len(validation_names)} "
        f"mode={args.split_mode} seed={args.split_seed}"
    )

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
        train_cameras + validation_cameras,
        gaussians,
        feature_extractor,
        masks,
        background,
        dataset.norm_before_render,
        dataset.longest_edge,
        args.query_feature_contract,
        require_proposal_scores=bool(args.generic_proposal_count > 0),
        require_native_sparse=(
            str(args.observation_source) in {"native", "native_plus_anchor"}
        ),
        native_keypoint_count=args.native_keypoint_count,
        cache_policy=args.query_cache_policy,
    )
    scaffold_visibility_counts = None
    if (
        args.scaffold_mode in {"pure_geometry", "protected_union"}
        and int(args.scaffold_min_visible_views) > 0
    ):
        scaffold_visibility_counts = _basic_visibility_counts(
            gaussians.get_xyz.detach(),
            train_names,
            cache,
            args,
            chunk_size=args.scaffold_visibility_chunk_size,
        )
        visible_eligible = scaffold_visibility_counts >= int(
            args.scaffold_min_visible_views
        )
        print(
            "Pure scaffold basic-visibility prefilter: "
            f"eligible={int(visible_eligible.sum().item())}/"
            f"{visible_eligible.numel()} "
            f"minimum_views={args.scaffold_min_visible_views}"
        )

    landmark_indices, landmark_path, scaffold_diagnostics = (
        _load_or_build_landmark_indices(
            dataset,
            gaussians,
            args,
            visibility_counts=scaffold_visibility_counts,
            cameras=train_cameras,
            masks=masks,
            feature_extractor=feature_extractor,
        )
    )
    _write_reproducibility_manifest(
        output_dir,
        dataset,
        args,
        landmark_path=landmark_path,
        scaffold_diagnostics=scaffold_diagnostics,
    )
    base_bank_xyz = gaussians.get_xyz[landmark_indices.cuda()].detach().float()
    base_bank_rotation = (
        gaussians.get_rotation[landmark_indices.cuda()].detach().float()
    )
    visibility_cache = None
    if args.visibility_mode == "rasterizer":
        visibility_signature, visibility_payload = _visibility_signature(
            dataset,
            args,
            landmark_indices,
        )
        visibility_cache = _load_or_build_visibility_cache(
            args.visibility_cache_path,
            visibility_signature,
            visibility_payload,
            train_cameras + validation_cameras,
            cache,
            gaussians,
            landmark_indices,
            native_sparse=(
                str(args.observation_source) in {"native", "native_plus_anchor"}
            ),
        )

    fallback = _fallback_features(
        gaussians,
        landmark_indices,
        feature_dim=feature_extractor.feature_dim,
    )
    state_only_initialization = bool(
        args.initial_state_path and float(args.initial_state_blend) >= 1.0
    )
    if args.initialization_mode == "ulf_geometry" and not state_only_initialization:
        mvinit_features, mvinit_observation_count, mvinit_diagnostics = (
            _build_ulf_geometry_features(
                train_cameras,
                gaussians,
                landmark_indices,
                masks,
                feature_extractor,
                fallback,
                args,
            )
        )
    elif args.initialization_mode == "ulf_geometry":
        # A descriptor continuation with blend=1 reuses the bootstrap state
        # exactly. Avoid recomputing the frozen ULF fusion merely to construct
        # an unused fallback.
        mvinit_features = fallback
        mvinit_observation_count = torch.zeros(
            fallback.shape[0], dtype=torch.long, device=fallback.device
        )
        mvinit_diagnostics = {
            "initialization_mode": "ulf_geometry_weighted_fusion",
            "initializer_reused_from_exact_state": True,
            "observed_landmarks": 0,
            "unobserved_landmarks": int(fallback.shape[0]),
        }
    else:
        mvinit_features, mvinit_observation_count, mvinit_diagnostics = (
            _build_mvinit_features(
                train_names,
                cache,
                base_bank_xyz,
                fallback,
                args,
                visibility_cache=visibility_cache,
            )
        )
    initial_state = None
    if args.initial_state_path:
        prior_features, initial_state, initial_state_match_count = _load_initial_features(
            args.initial_state_path,
            landmark_indices,
            feature_extractor.feature_dim,
            base_bank_xyz.device,
            fallback_features=mvinit_features,
            alignment=args.initial_state_alignment,
        )
        blend = float(max(0.0, min(1.0, args.initial_state_blend)))
        initial_features = F.normalize(
            blend * prior_features + (1.0 - blend) * mvinit_features,
            dim=-1,
        )
        mvinit_diagnostics["loaded_initial_state"] = 1.0
        mvinit_diagnostics["initial_state_path"] = os.path.abspath(
            args.initial_state_path
        )
        mvinit_diagnostics["initial_state_blend"] = blend
        mvinit_diagnostics["initial_state_match_count"] = int(
            initial_state_match_count
        )
        mvinit_diagnostics["initial_state_alignment"] = str(
            args.initial_state_alignment
        )
    else:
        initial_features = mvinit_features

    config = _state_config(
        args,
        dataset,
        train_names,
        validation_names,
        landmark_path,
        scaffold_diagnostics,
    )
    residual = torch.nn.Parameter(torch.zeros_like(initial_features))
    raw_anchor_offset = torch.nn.Parameter(torch.zeros_like(base_bank_xyz))
    if (
        initial_state is not None
        and "raw_anchor_offset" in initial_state
        and torch.as_tensor(initial_state["raw_anchor_offset"]).numel()
        == raw_anchor_offset.numel()
        and bool(
            initial_state.get("_raw_anchor_offset_alignment_valid", False)
        )
    ):
        initial_raw_offset = torch.as_tensor(
            initial_state["raw_anchor_offset"],
            device=base_bank_xyz.device,
            dtype=base_bank_xyz.dtype,
        ).reshape_as(raw_anchor_offset)
        with torch.no_grad():
            raw_anchor_offset.copy_(initial_raw_offset)
        mvinit_diagnostics["loaded_initial_anchor_offset"] = 1.0
    dustbin_score = None
    optimizer_parameters = [residual, raw_anchor_offset]
    if float(args.dustbin_weight) > 0.0:
        dustbin_initial_value = (
            float(initial_state["dustbin_score"])
            if initial_state is not None and "dustbin_score" in initial_state
            else float(args.dustbin_init)
        )
        dustbin_score = torch.nn.Parameter(
            initial_features.new_tensor(dustbin_initial_value)
        )
        optimizer_parameters.append(dustbin_score)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [residual],
                "lr": args.feature_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [raw_anchor_offset],
                "lr": args.geometry_lr,
                "weight_decay": 0.0,
            },
            *(
                [
                    {
                        "params": [dustbin_score],
                        "lr": args.feature_lr,
                        "weight_decay": 0.0,
                    }
                ]
                if dustbin_score is not None
                else []
            ),
        ],
    )
    cuda_generator = torch.Generator(device=base_bank_xyz.device).manual_seed(
        args.train_seed
    )
    camera_rng = random.Random(args.train_seed)
    trust_weights = observation_adaptive_trust_weights(
        mvinit_observation_count,
        power=args.trust_observation_power,
        minimum=args.trust_weight_min,
        maximum=args.trust_weight_max,
    ).to(device=base_bank_xyz.device)
    train_order = list(train_names)
    camera_rng.shuffle(train_order)
    order_position = 0
    history = []
    requested_checkpoint_steps = sorted(
        {int(step) for step in args.save_steps} | {int(args.steps)}
    )
    save_steps = set(requested_checkpoint_steps)

    initial_xyz = materialize_bounded_surface_anchors(
        base_bank_xyz,
        base_bank_rotation,
        raw_anchor_offset,
        tangent_bound_m=args.tangent_bound_m,
        normal_bound_m=args.normal_bound_m,
    )
    initial_validation = _validate_descriptor_field(
        initial_features,
        validation_names,
        cache,
        initial_xyz,
        args,
        visibility_cache=visibility_cache,
        dustbin_score=dustbin_score,
        base_bank_xyz=base_bank_xyz,
    )
    _save_state(
        output_dir / "0_lafgs_map_state.pt",
        0,
        landmark_indices,
        initial_features,
        config,
        {**mvinit_diagnostics, **initial_validation},
        mvinit_observation_count,
        dustbin_score=dustbin_score,
        landmark_xyz=initial_xyz,
        raw_anchor_offset=raw_anchor_offset,
    )

    empty_observation_steps = 0
    empty_observation_checkpoint_steps = []

    def save_checkpoint(step, checkpoint_xyz, *, after_empty_observation=False):
        checkpoint_features = materialize_descriptor_residual(
            initial_features,
            residual,
            residual_scale=args.residual_scale,
            max_residual_norm=args.max_residual_norm,
        )
        validation = _validate_descriptor_field(
            checkpoint_features,
            validation_names,
            cache,
            checkpoint_xyz.detach(),
            args,
            visibility_cache=visibility_cache,
            dustbin_score=dustbin_score,
            base_bank_xyz=base_bank_xyz,
        )
        recent = _mean_diagnostics(history[-min(len(history), 200) :])
        if after_empty_observation:
            recent["checkpoint_saved_after_empty_observation"] = 1.0
        _save_state(
            output_dir / f"{step}_lafgs_map_state.pt",
            step,
            landmark_indices,
            checkpoint_features,
            config,
            {**mvinit_diagnostics, **recent, **validation},
            mvinit_observation_count,
            dustbin_score=dustbin_score,
            landmark_xyz=checkpoint_xyz,
            raw_anchor_offset=raw_anchor_offset,
        )

    progress = tqdm(range(1, args.steps + 1), desc=f"LaFGS map {args.objective}")
    for step in progress:
        if order_position >= len(train_order):
            camera_rng.shuffle(train_order)
            order_position = 0
        query_name = train_order[order_position]
        order_position += 1
        current_xyz = materialize_bounded_surface_anchors(
            base_bank_xyz,
            base_bank_rotation,
            raw_anchor_offset,
            tangent_bound_m=args.tangent_bound_m,
            normal_bound_m=args.normal_bound_m,
        )
        observations, anchor_auxiliary = _primary_observations(
            cache[query_name],
            base_bank_xyz,
            args,
            max_observations=args.max_observations,
            bank_visibility_mask=(
                None
                if visibility_cache is None
                else visibility_cache[query_name]
            ),
            prediction_bank_xyz=current_xyz,
        )
        if observations.query_features.numel() == 0:
            empty_observation_steps += 1
            if step in save_steps:
                save_checkpoint(
                    step,
                    current_xyz,
                    after_empty_observation=True,
                )
                empty_observation_checkpoint_steps.append(step)
            continue
        retrieval_observations = (
            jitter_detector_free_observations(
                observations,
                standard_deviation=args.proposal_jitter_std,
                maximum=args.proposal_jitter_max,
                generator=cuda_generator,
            )
            if str(args.observation_source) == "anchor"
            else observations
        )
        proposal_observations = (
            _cached_score_proposal_observations(cache[query_name], observations, args)
            if str(args.observation_source) == "anchor"
            else None
        )

        features = materialize_descriptor_residual(
            initial_features,
            residual,
            residual_scale=args.residual_scale,
            max_residual_norm=args.max_residual_norm,
        )
        descriptor_active = descriptor_losses_active(
            step,
            args.descriptor_end_step,
        )
        descriptor_scale = float(descriptor_active)
        if str(args.observation_source) == "anchor":
            auxiliary_observations = observations
            auxiliary_scale = 1.0
        else:
            auxiliary_observations = anchor_auxiliary
            auxiliary_scale = float(args.native_anchor_aux_weight)
        if auxiliary_observations is not None:
            mv_loss = auxiliary_scale * multiview_descriptor_loss(
                features, auxiliary_observations
            )
        else:
            mv_loss = features.sum() * 0.0
        if not descriptor_active:
            retrieval_loss = features.sum() * 0.0
            retrieval_diagnostics = {"descriptor_retrieval_skipped": 1.0}
        elif args.objective == "mv":
            retrieval_loss = features.sum() * 0.0
            retrieval_diagnostics = {}
        elif args.objective == "random":
            retrieval = random_negative_retrieval_loss(
                features,
                retrieval_observations,
                negative_count=args.random_negative_count,
                temperature=args.temperature,
                positive_radius_px=args.positive_radius_px,
                negative_radius_px=args.negative_radius_px,
                generator=cuda_generator,
                dustbin_score=dustbin_score,
            )
            retrieval_loss = retrieval.loss
            retrieval_diagnostics = retrieval.diagnostics
        elif args.objective == "hard":
            retrieval = hard_hypothesis_retrieval_loss(
                features,
                retrieval_observations,
                hypothesis_topk=args.hypothesis_topk,
                temperature=args.temperature,
                positive_radius_px=args.positive_radius_px,
                negative_radius_px=args.negative_radius_px,
                margin=args.retrieval_margin,
                missed_positive_weight=args.missed_positive_weight,
                missed_positive_margin=args.missed_positive_margin,
                unmatched_rejection_weight=args.unmatched_rejection_weight,
                unmatched_max_similarity=args.unmatched_max_similarity,
                dustbin_score=dustbin_score,
            )
            retrieval_loss = retrieval.loss
            retrieval_diagnostics = retrieval.diagnostics
        else:
            raise ValueError(f"Unknown objective: {args.objective}")
        if (
            descriptor_active
            and proposal_observations is not None
            and args.objective != "mv"
        ):
            proposal_retrieval = hard_hypothesis_retrieval_loss(
                features,
                proposal_observations,
                hypothesis_topk=args.hypothesis_topk,
                temperature=args.temperature,
                positive_radius_px=args.positive_radius_px,
                negative_radius_px=args.negative_radius_px,
                margin=args.retrieval_margin,
                missed_positive_weight=args.missed_positive_weight,
                missed_positive_margin=args.missed_positive_margin,
                unmatched_rejection_weight=args.unmatched_rejection_weight,
                unmatched_max_similarity=args.unmatched_max_similarity,
                dustbin_score=dustbin_score,
            )
            proposal_retrieval_loss = proposal_retrieval.loss
            proposal_retrieval_diagnostics = {
                f"generic_proposal_{key}": value
                for key, value in proposal_retrieval.diagnostics.items()
            }
        else:
            proposal_retrieval_loss = features.sum() * 0.0
            proposal_retrieval_diagnostics = {}
        trust_loss = descriptor_trust_loss(
            features,
            initial_features,
            weights=trust_weights,
        )
        if args.local_weight > 0.0 and auxiliary_observations is not None:
            local_loss, local_diagnostics = local_correlation_peak_loss(
                features,
                auxiliary_observations,
                radius=args.local_radius,
                target_sigma=args.local_target_sigma,
                temperature=args.local_temperature,
            )
        else:
            local_loss = features.sum() * 0.0
            local_diagnostics = {}
        if dustbin_score is not None and auxiliary_observations is not None:
            dustbin_loss, dustbin_diagnostics = background_dustbin_loss(
                features,
                auxiliary_observations,
                dustbin_score,
                sample_count=args.dustbin_background_count,
                exclusion_radius_px=args.dustbin_exclusion_radius,
                hypothesis_topk=args.hypothesis_topk,
                temperature=args.temperature,
                background_alpha_max=args.dustbin_background_alpha_max,
                allow_no_anchor=args.dustbin_no_anchor,
                generator=cuda_generator,
            )
        else:
            dustbin_loss = features.sum() * 0.0
            dustbin_diagnostics = {}
        geometry_active = (
            float(args.geometry_weight) > 0.0
            and step >= int(args.geometry_start_step)
        )
        if geometry_active:
            if str(args.geometry_mode) == "native_association":
                (
                    surface_loss,
                    depth_loss,
                    geometry_reprojection_loss,
                    geometry_diagnostics,
                ) = native_association_geometry_losses(
                    current_xyz,
                    raw_anchor_offset,
                    features,
                    observations,
                    max_reprojection_error_px=args.geometry_association_max_reprojection_px,
                    min_score_margin=args.geometry_association_min_margin,
                    alpha_threshold=args.alpha_threshold,
                    depth_scale_floor=args.depth_scale_floor,
                )
                local_correspondences = None
            else:
                (
                    surface_loss,
                    depth_loss,
                    geometry_reprojection_loss,
                    local_correspondences,
                    geometry_diagnostics,
                ) = bounded_geometry_losses(
                    current_xyz,
                    raw_anchor_offset,
                    features,
                    observations,
                    local_radius=args.local_radius,
                    local_temperature=args.local_temperature,
                    depth_scale_floor=args.depth_scale_floor,
                )
        else:
            surface_loss = raw_anchor_offset.sum() * 0.0
            depth_loss = raw_anchor_offset.sum() * 0.0
            geometry_reprojection_loss = raw_anchor_offset.sum() * 0.0
            local_correspondences = None
            geometry_diagnostics = {"bounded_geometry_active": 0.0}
        pose_active = (
            str(args.pose_gradient_mode) != "off"
            and float(args.pose_weight) > 0.0
            and step >= int(args.pose_start_step)
            and step % max(int(args.pose_interval), 1) == 0
            and (
                str(args.pose_gradient_mode) == "feature"
                or geometry_active
            )
        )
        if pose_active:
            if str(args.pose_gradient_mode) == "feature":
                pose_correspondences = local_soft_correspondences(
                    features,
                    observations,
                    radius=args.local_radius,
                    temperature=args.local_temperature,
                )
            else:
                pose_correspondences = local_correspondences
            if pose_correspondences is None:
                raise RuntimeError(
                    "The configured pose layer requires local anchor correspondences; "
                    "it is intentionally unavailable in native_association geometry mode"
                )
            pose_noise = torch.randn(
                6,
                device=current_xyz.device,
                dtype=current_xyz.dtype,
                generator=cuda_generator,
            )
            pose_noise[:3] *= float(args.pose_init_translation_std_m)
            pose_noise[3:] *= float(
                torch.deg2rad(torch.tensor(args.pose_init_rotation_std_deg)).item()
            )
            pose_init = se3_exp(pose_noise) @ observations.pose_w2c
            pose_loss, pose_diagnostics = pose_layer_loss(
                current_xyz,
                observations,
                pose_correspondences,
                pose_init,
                num_iterations=args.pose_iterations,
                damping=args.pose_damping,
                min_points=args.pose_min_points,
                max_points=args.pose_max_points,
                translation_scale_m=args.pose_translation_scale_m,
                rotation_scale_degrees=args.pose_rotation_scale_deg,
                max_condition_number=args.pose_max_condition_number,
                gradient_mode=args.pose_gradient_mode,
            )
        else:
            pose_loss = raw_anchor_offset.sum() * 0.0
            pose_diagnostics = {"pose_layer_active": 0.0}
        loss = (
            descriptor_scale
            * (
                args.mv_weight * mv_loss
                + args.retrieval_weight * retrieval_loss
                + args.generic_proposal_weight * proposal_retrieval_loss
                + args.local_weight * local_loss
                + args.trust_weight * trust_loss
                + args.dustbin_weight * dustbin_loss
            )
            + (
                args.geometry_weight
                * (
                    args.surface_weight * surface_loss
                    + args.depth_weight * depth_loss
                    + args.reprojection_weight * geometry_reprojection_loss
                )
            )
            + args.pose_weight * pose_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        descriptor_update_active = bool(descriptor_active) or (
            pose_active and str(args.pose_gradient_mode) == "feature"
        )
        geometry_update_active = geometry_active or (
            pose_active and str(args.pose_gradient_mode) == "geometry"
        )
        _clear_inactive_phase_gradients(
            residual,
            raw_anchor_offset,
            dustbin_score,
            descriptor_update_active=descriptor_update_active,
            geometry_update_active=geometry_update_active,
            dustbin_update_active=bool(descriptor_active),
        )
        gradients_finite = _all_parameter_gradients_finite(optimizer_parameters)
        if gradients_finite:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                optimizer_parameters,
                args.gradient_clip_norm,
            )
            gradients_finite = bool(torch.isfinite(grad_norm).item())
        else:
            grad_norm = loss.new_tensor(float("nan"))
        optimizer_step_skipped = not gradients_finite
        if gradients_finite:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            raw_anchor_offset.clamp_(-float(args.raw_offset_clip), float(args.raw_offset_clip))
            current_xyz = materialize_bounded_surface_anchors(
                base_bank_xyz,
                base_bank_rotation,
                raw_anchor_offset,
                tangent_bound_m=args.tangent_bound_m,
                normal_bound_m=args.normal_bound_m,
            )
            displacement = torch.linalg.norm(current_xyz - base_bank_xyz, dim=1)
        record = {
            "step": step,
            "descriptor_active": float(descriptor_active),
            "descriptor_update_active": float(descriptor_update_active),
            "geometry_update_active": float(geometry_update_active),
            "loss": float(loss.detach().item()),
            "mv_loss": float(mv_loss.detach().item()),
            "retrieval_loss": float(retrieval_loss.detach().item()),
            "generic_proposal_loss": float(
                proposal_retrieval_loss.detach().item()
            ),
            "local_loss": float(local_loss.detach().item()),
            "trust_loss": float(trust_loss.detach().item()),
            "dustbin_loss": float(dustbin_loss.detach().item()),
            "surface_loss": float(surface_loss.detach().item()),
            "depth_loss": float(depth_loss.detach().item()),
            "geometry_reprojection_loss": float(
                geometry_reprojection_loss.detach().item()
            ),
            "pose_loss": float(pose_loss.detach().item()),
            "anchor_displacement_mean_m": float(displacement.mean().item()),
            "anchor_displacement_p95_m": float(
                torch.quantile(displacement, 0.95).item()
            ),
            "anchor_displacement_max_m": float(displacement.max().item()),
            "grad_norm": float(grad_norm.detach().item()),
            "optimizer_step_skipped_nonfinite": float(optimizer_step_skipped),
            "visible_observations": int(observations.query_features.shape[0]),
            "matched_observations": int(
                (observations.source_indices >= 0).sum().item()
            ),
            "unmatched_observations": int(
                (observations.source_indices < 0).sum().item()
            ),
            "native_candidate_observations": float(
                str(args.observation_source) != "anchor"
            ),
            "anchor_auxiliary_observations": (
                0
                if anchor_auxiliary is None
                else int(anchor_auxiliary.query_features.shape[0])
            ),
            "generic_proposal_observations": (
                0
                if proposal_observations is None
                else int(proposal_observations.source_indices.numel())
            ),
            **retrieval_diagnostics,
            **proposal_retrieval_diagnostics,
            **local_diagnostics,
            **dustbin_diagnostics,
            **geometry_diagnostics,
            **pose_diagnostics,
        }
        history.append(record)
        if step % args.log_interval == 0:
            recent = _mean_diagnostics(history[-args.log_interval :])
            progress.set_postfix(
                loss=f"{recent.get('loss', 0.0):.4f}",
                mv=f"{recent.get('mv_loss', 0.0):.4f}",
                retr=f"{recent.get('retrieval_loss', 0.0):.4f}",
                prop=f"{recent.get('generic_proposal_loss', 0.0):.4f}",
                local=f"{recent.get('local_loss', 0.0):.4f}",
                geo=f"{recent.get('geometry_reprojection_loss', 0.0):.3f}",
                pose=f"{recent.get('pose_loss', 0.0):.3f}",
            )
        if step in save_steps:
            save_checkpoint(step, current_xyz)

    final_features = materialize_descriptor_residual(
        initial_features,
        residual,
        residual_scale=args.residual_scale,
        max_residual_norm=args.max_residual_norm,
    )
    final_xyz = materialize_bounded_surface_anchors(
        base_bank_xyz,
        base_bank_rotation,
        raw_anchor_offset,
        tangent_bound_m=args.tangent_bound_m,
        normal_bound_m=args.normal_bound_m,
    )
    final_validation = _validate_descriptor_field(
        final_features,
        validation_names,
        cache,
        final_xyz,
        args,
        visibility_cache=visibility_cache,
        dustbin_score=dustbin_score,
        base_bank_xyz=base_bank_xyz,
    )
    distillation_summary = {"enabled": False}
    landmark_statistics_summary = {}
    if int(args.distill_budget) > 0:
        statistics, landmark_statistics_summary = _collect_landmark_statistics(
            final_features,
            train_names,
            cache,
            final_xyz,
            args,
            visibility_cache=visibility_cache,
            base_bank_xyz=base_bank_xyz,
        )
        distillation_summary = {
            "enabled": True,
            **_distill_final_landmark_bank(
                output_dir,
                landmark_indices,
                final_features,
                final_xyz,
                raw_anchor_offset,
                statistics,
                args,
                config,
                mvinit_observation_count,
                dustbin_score,
            ),
        }
    with torch.no_grad():
        feature_cosine = (
            F.normalize(final_features, dim=-1)
            * F.normalize(initial_features, dim=-1)
        ).sum(dim=-1)
        bounded_local_offset = torch.tanh(raw_anchor_offset)
        tangent_offset = (
            bounded_local_offset[:, :2] * float(args.tangent_bound_m)
        )
        normal_offset = (
            bounded_local_offset[:, 2].abs() * float(args.normal_bound_m)
        )
        tangent_norm = torch.linalg.norm(tangent_offset, dim=1)
        summary = {
            "config": config,
            "mvinit": mvinit_diagnostics,
            "initial_validation": initial_validation,
            "final_validation": final_validation,
            "landmark_statistics": landmark_statistics_summary,
            "distillation": distillation_summary,
            "feature_drift": {
                "cosine_mean": float(feature_cosine.mean().item()),
                "cosine_p01": float(torch.quantile(feature_cosine, 0.01).item()),
                "l2_mean": float(
                    torch.linalg.norm(final_features - initial_features, dim=-1)
                    .mean()
                    .item()
                ),
                "residual_raw_l2_mean": float(
                    torch.linalg.norm(residual, dim=-1).mean().item()
                ),
                "residual_raw_l2_max": float(
                    torch.linalg.norm(residual, dim=-1).max().item()
                ),
            },
            "dustbin": {
                "enabled": dustbin_score is not None,
                "score": (
                    None
                    if dustbin_score is None
                    else float(dustbin_score.detach().item())
                ),
            },
            "history_tail": history[-min(len(history), 200) :],
            "training_control": {
                "empty_observation_steps": int(empty_observation_steps),
                "empty_observation_checkpoint_steps": list(
                    empty_observation_checkpoint_steps
                ),
            },
            "geometry_invariant": {
                "base_bank_xyz_trainable": bool(base_bank_xyz.requires_grad),
                # raw_anchor_offset remains a Parameter so staged runs can
                # reuse one optimizer layout.  It is not trainable in a
                # descriptor-only phase unless a configured loss can update it.
                "bounded_anchor_parameter_requires_grad": bool(
                    raw_anchor_offset.requires_grad
                ),
                "bounded_anchor_trainable": bool(args.geometry_weight > 0.0),
                "anchor_displacement_mean_m": float(
                    torch.linalg.norm(final_xyz - base_bank_xyz, dim=1).mean().item()
                ),
                "anchor_displacement_p95_m": float(
                    torch.quantile(
                        torch.linalg.norm(final_xyz - base_bank_xyz, dim=1), 0.95
                    ).item()
                ),
                "anchor_displacement_max_m": float(
                    torch.linalg.norm(final_xyz - base_bank_xyz, dim=1).max().item()
                ),
                "tangent_displacement_mean_m": float(tangent_norm.mean().item()),
                "tangent_displacement_p95_m": float(
                    torch.quantile(tangent_norm, 0.95).item()
                ),
                "tangent_displacement_max_m": float(tangent_norm.max().item()),
                "normal_displacement_abs_mean_m": float(
                    normal_offset.mean().item()
                ),
                "normal_displacement_abs_p95_m": float(
                    torch.quantile(normal_offset, 0.95).item()
                ),
                "normal_displacement_abs_max_m": float(normal_offset.max().item()),
                "gaussian_parameter_grad_count": int(
                    sum(
                        parameter.grad is not None
                        for parameter in gaussians.parameters()
                    )
                ),
            },
        }
    _save_state(
        output_dir / f"{args.steps}_lafgs_map_state.pt",
        args.steps,
        landmark_indices,
        final_features,
        config,
        {**mvinit_diagnostics, **_mean_diagnostics(history[-min(len(history), 200):]), **final_validation},
        mvinit_observation_count,
        dustbin_score=dustbin_score,
        landmark_xyz=final_xyz,
        raw_anchor_offset=raw_anchor_offset,
    )
    summary["checkpoint_integrity"] = _checkpoint_integrity(
        output_dir, requested_checkpoint_steps
    )
    with (output_dir / "training_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    missing_checkpoint_steps = summary["checkpoint_integrity"]["missing_steps"]
    if missing_checkpoint_steps:
        raise RuntimeError(
            "Requested LaFGS checkpoint(s) were not written: "
            + ", ".join(str(step) for step in missing_checkpoint_steps)
        )
    print(f"Saved detector-free LaFGS map training output: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Detector-free localization descriptor reconstruction on a fixed Gaussian surface"
    )
    model_params = ModelParams(parser)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--query_feature_contract",
        choices=[
            _QUERY_FEATURE_CONTRACT_LEGACY,
            _QUERY_FEATURE_CONTRACT_NATIVE,
        ],
        default=_QUERY_FEATURE_CONTRACT_NATIVE,
        help=(
            "Image/descriptor contract for map reconstruction. Native mode "
            "resizes RGB before SuperPoint and preserves its stride-8 map."
        ),
    )
    parser.add_argument(
        "--scaffold_mode",
        choices=["file", "pure_geometry", "protected_union", "ulf_consensus"],
        default="ulf_consensus",
    )
    parser.add_argument("--landmark_path", default="detector/sampled_idx.pkl")
    parser.add_argument(
        "--protected_core_path",
        default="detector/sampled_idx.pkl",
        help="Strong landmark IDs kept at the front of a protected_union scaffold.",
    )
    parser.add_argument("--generated_landmark_path", default="pure_geometry_scaffold.pkl")
    parser.add_argument("--regenerate_scaffold", action="store_true")
    parser.add_argument("--scaffold_budget", type=int, default=16384)
    parser.add_argument("--scaffold_min_opacity", type=float, default=0.05)
    parser.add_argument("--scaffold_min_visible_views", type=int, default=1)
    parser.add_argument("--scaffold_visibility_chunk_size", type=int, default=262144)
    parser.add_argument("--scaffold_normal_bins", type=int, default=6)
    parser.add_argument("--scaffold_voxel_size", type=float, default=0.0)
    parser.add_argument("--scaffold_search_steps", type=int, default=14)
    parser.add_argument("--scaffold_seed", type=int, default=2026)
    parser.add_argument(
        "--ulf_consensus_keypoints",
        type=int,
        default=2048,
        help="Native sparse SuperPoint keypoints retained per support view for KCS.",
    )
    parser.add_argument("--ulf_consensus_radius_px", type=float, default=1.0)
    parser.add_argument("--ulf_consensus_min_votes", type=int, default=1)
    parser.add_argument(
        "--ulf_consensus_max_views",
        type=int,
        default=0,
        help="Zero uses every support camera; otherwise uniformly subsample views.",
    )
    parser.add_argument("--ulf_consensus_distance_chunk", type=int, default=8192)
    parser.add_argument(
        "--ulf_consensus_max_candidates_per_view",
        type=int,
        default=0,
        help="Optional opacity-ranked cap for KCS; zero preserves all visible primitives.",
    )
    parser.add_argument("--ulf_consensus_voxel_size", type=float, default=0.0)
    parser.add_argument("--ulf_consensus_max_per_voxel", type=int, default=8)
    parser.add_argument(
        "--initialization_mode",
        choices=["mvinit", "ulf_geometry"],
        default="ulf_geometry",
        help="Descriptor initializer for a newly built landmark bank.",
    )
    parser.add_argument(
        "--ulf_fusion_max_views",
        type=int,
        default=0,
        help="Zero fuses every support camera; otherwise uniformly subsample views.",
    )
    parser.add_argument("--ulf_fusion_min_cosine", type=float, default=0.0)
    parser.add_argument("--initial_state_path", default="")
    parser.add_argument("--initial_state_blend", type=float, default=0.0)
    parser.add_argument(
        "--initial_state_alignment",
        choices=["exact", "overlap"],
        default="exact",
    )
    parser.add_argument("--query_cache_path", default="")
    parser.add_argument(
        "--query_cache_policy",
        choices=["reuse_or_build", "readonly", "refresh"],
        default="reuse_or_build",
        help=(
            "Use readonly for formal experiments so an incompatible frontend "
            "protocol fails instead of silently rebuilding shared query features."
        ),
    )
    parser.add_argument(
        "--visibility_mode",
        choices=["depth", "rasterizer"],
        default="rasterizer",
    )
    parser.add_argument("--visibility_cache_path", default="")
    parser.add_argument(
        "--observation_source",
        choices=["anchor", "native", "native_plus_anchor"],
        default="anchor",
        help=(
            "anchor preserves the legacy projection-sampled objective; native "
            "uses deployed SuperPoint detectAndCompute proposals; "
            "native_plus_anchor adds a low-weight projection auxiliary."
        ),
    )
    parser.add_argument(
        "--native_keypoint_count",
        type=int,
        default=2048,
        help="Native SuperPoint proposal count cached per query image.",
    )
    parser.add_argument(
        "--native_association_radius_px",
        type=float,
        default=2.0,
        help="GT reprojection radius used only to label native proposals.",
    )
    parser.add_argument(
        "--native_unmatched_fraction",
        type=float,
        default=0.25,
        help=(
            "Unmatched ratio only for the label_balanced ablation; ignored by "
            "the deployment-aligned detector_grid default."
        ),
    )
    parser.add_argument(
        "--native_sampling_mode",
        choices=["detector_grid", "label_balanced"],
        default="detector_grid",
        help=(
            "detector_grid samples actual SuperPoint proposals before GT labels; "
            "label_balanced is an explicitly non-deployment ablation."
        ),
    )
    parser.add_argument(
        "--native_anchor_aux_weight",
        type=float,
        default=0.1,
        help="Relative anchor MV/local auxiliary weight in native_plus_anchor mode.",
    )
    parser.add_argument("--objective", choices=["mv", "random", "hard"], default="hard")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument(
        "--save_steps", nargs="*", type=int, default=[1000, 2000, 3000, 4000, 5000]
    )
    parser.add_argument("--feature_lr", type=float, default=1.5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=10.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--max_residual_norm", type=float, default=0.0)
    parser.add_argument("--mv_weight", type=float, default=1.0)
    parser.add_argument("--retrieval_weight", type=float, default=0.5)
    parser.add_argument("--trust_weight", type=float, default=0.02)
    parser.add_argument("--trust_observation_power", type=float, default=0.5)
    parser.add_argument("--trust_weight_min", type=float, default=0.25)
    parser.add_argument("--trust_weight_max", type=float, default=4.0)
    parser.add_argument("--local_weight", type=float, default=0.05)
    parser.add_argument("--local_radius", type=int, default=3)
    parser.add_argument("--local_target_sigma", type=float, default=1.0)
    parser.add_argument("--local_temperature", type=float, default=0.07)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hypothesis_topk", type=int, default=32)
    parser.add_argument("--random_negative_count", type=int, default=32)
    parser.add_argument("--positive_radius_px", type=float, default=2.0)
    parser.add_argument("--negative_radius_px", type=float, default=6.0)
    parser.add_argument("--retrieval_margin", type=float, default=0.05)
    parser.add_argument("--missed_positive_weight", type=float, default=1.0)
    parser.add_argument("--missed_positive_margin", type=float, default=0.05)
    parser.add_argument("--unmatched_rejection_weight", type=float, default=0.0)
    parser.add_argument("--unmatched_max_similarity", type=float, default=0.5)
    parser.add_argument("--proposal_jitter_std", type=float, default=0.0)
    parser.add_argument("--proposal_jitter_max", type=float, default=0.0)
    parser.add_argument("--generic_proposal_weight", type=float, default=0.0)
    parser.add_argument("--generic_proposal_count", type=int, default=0)
    parser.add_argument("--generic_proposal_nms_radius", type=int, default=2)
    parser.add_argument("--generic_proposal_score_threshold", type=float, default=0.0)
    parser.add_argument("--generic_proposal_positive_radius", type=float, default=2.0)
    parser.add_argument(
        "--generic_proposal_include_unmatched",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--descriptor_end_step",
        type=int,
        default=0,
        help=(
            "Last step for descriptor losses; zero keeps them active throughout "
            "and a negative value freezes descriptors for the whole run."
        ),
    )
    parser.add_argument("--dustbin_weight", type=float, default=0.02)
    parser.add_argument("--dustbin_init", type=float, default=0.2)
    parser.add_argument("--dustbin_background_count", type=int, default=128)
    parser.add_argument("--dustbin_exclusion_radius", type=float, default=6.0)
    parser.add_argument("--dustbin_background_alpha_max", type=float, default=0.05)
    parser.add_argument(
        "--dustbin_no_anchor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use valid pixels outside all visible-anchor neighborhoods when low-alpha background is absent.",
    )
    parser.add_argument("--geometry_start_step", type=int, default=2000)
    parser.add_argument("--geometry_lr", type=float, default=2e-3)
    parser.add_argument("--geometry_weight", type=float, default=0.1)
    parser.add_argument("--surface_weight", type=float, default=0.05)
    parser.add_argument("--depth_weight", type=float, default=0.25)
    parser.add_argument("--reprojection_weight", type=float, default=1.0)
    parser.add_argument("--tangent_bound_m", type=float, default=0.005)
    parser.add_argument("--normal_bound_m", type=float, default=0.002)
    parser.add_argument("--raw_offset_clip", type=float, default=3.0)
    parser.add_argument("--depth_scale_floor", type=float, default=0.25)
    parser.add_argument(
        "--geometry_mode",
        choices=["anchor_local", "native_association"],
        default="anchor_local",
        help=(
            "anchor_local is the legacy dense-anchor geometry branch; "
            "native_association performs fixed-descriptor BA on GT-clean "
            "native top-1 matches."
        ),
    )
    parser.add_argument("--geometry_association_max_reprojection_px", type=float, default=2.0)
    parser.add_argument("--geometry_association_min_margin", type=float, default=0.0)
    parser.add_argument("--pose_start_step", type=int, default=3500)
    parser.add_argument("--pose_interval", type=int, default=4)
    parser.add_argument("--pose_weight", type=float, default=0.01)
    parser.add_argument(
        "--pose_gradient_mode",
        choices=["off", "feature", "geometry"],
        default="geometry",
    )
    parser.add_argument("--pose_iterations", type=int, default=2)
    parser.add_argument("--pose_damping", type=float, default=1e-3)
    parser.add_argument("--pose_min_points", type=int, default=24)
    parser.add_argument("--pose_max_points", type=int, default=96)
    parser.add_argument("--pose_init_translation_std_m", type=float, default=0.01)
    parser.add_argument("--pose_init_rotation_std_deg", type=float, default=0.5)
    parser.add_argument("--pose_translation_scale_m", type=float, default=0.02)
    parser.add_argument("--pose_rotation_scale_deg", type=float, default=2.0)
    parser.add_argument("--pose_max_condition_number", type=float, default=5e4)
    parser.add_argument("--distill_budget", type=int, default=0)
    parser.add_argument(
        "--distill_require_exact_budget",
        action="store_true",
        help=(
            "Fail instead of silently shrinking the final distilled bank when "
            "the requested fixed landmark budget is not observed."
        ),
    )
    parser.add_argument(
        "--distill_rank_pool_multiplier",
        type=float,
        default=2.0,
        help=(
            "When threshold-qualified landmarks are fewer than the final "
            "budget, retain this multiple of the budget by matchability before "
            "coverage/FIM selection."
        ),
    )
    parser.add_argument("--statistics_observations", type=int, default=1024)
    parser.add_argument("--statistics_hypothesis_topk", type=int, default=32)
    parser.add_argument("--distill_min_observations", type=int, default=2)
    parser.add_argument(
        "--distill_matchability_threshold", type=float, default=0.5
    )
    parser.add_argument("--distill_false_top1_max", type=float, default=0.5)
    parser.add_argument(
        "--distill_proposal_weight",
        type=float,
        default=1.0,
        help="Weight of frozen generic-proposal retrieval in reliability filtering.",
    )
    parser.add_argument("--distill_reprojection_scale_px", type=float, default=2.0)
    parser.add_argument(
        "--distill_matchability_preserve_ratio", type=float, default=0.30
    )
    parser.add_argument("--distill_utility_preserve_ratio", type=float, default=0.35)
    parser.add_argument("--distill_high_confidence", type=float, default=0.75)
    parser.add_argument("--distill_high_confidence_ratio", type=float, default=0.10)
    parser.add_argument("--distill_voxel_size", type=float, default=0.0)
    parser.add_argument("--distill_max_per_voxel", type=int, default=8)
    parser.add_argument("--distill_grid_size", type=int, default=8)
    parser.add_argument("--distill_max_per_grid", type=int, default=512)
    parser.add_argument("--distill_depth_bins", type=int, default=8)
    parser.add_argument("--distill_max_per_depth_bin", type=int, default=4096)
    parser.add_argument(
        "--mvinit_max_observations",
        type=int,
        default=0,
        help="Per-view MVInit cap; zero uses every visible anchor.",
    )
    parser.add_argument(
        "--mvinit_mode",
        choices=["mean", "medoid"],
        default="mean",
    )
    parser.add_argument("--max_observations", type=int, default=512)
    parser.add_argument("--validation_observations", type=int, default=512)
    parser.add_argument("--grid_rows", type=int, default=8)
    parser.add_argument("--grid_cols", type=int, default=8)
    parser.add_argument("--depth_bins", type=int, default=4)
    parser.add_argument("--alpha_threshold", type=float, default=0.2)
    parser.add_argument("--depth_abs_tolerance", type=float, default=1e-3)
    parser.add_argument("--depth_rel_tolerance", type=float, default=0.01)
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument(
        "--split_mode",
        choices=["random", "sequence_block", "temporal_block"],
        default="temporal_block",
    )
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--train_seed", type=int, default=2026)
    parser.add_argument("--max_train_views", type=int, default=0)
    parser.add_argument("--max_validation_views", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--quiet", action="store_true")
    return parser, model_params


if __name__ == "__main__":
    parser, model_params = build_parser()
    args = parser.parse_args()
    safe_state(args.quiet)
    seed_everything(args.train_seed)
    dataset = model_params.extract(args)
    train(dataset, args)
