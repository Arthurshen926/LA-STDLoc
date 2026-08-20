from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.cli import ModelParams
from common.calibration import write_query_calibration_sidecar
from features.extractor import FeatureExtractor
from priors.rendering import get_render_visible_mask, render_from_pose_gsplat
from map_learning.stage_a_loss import (
    build_native_sparse_observations,
    descriptor_trust_loss,
    hard_hypothesis_retrieval_loss,
    materialize_descriptor_residual,
    native_semidense_neighborhood_loss,
    observation_adaptive_trust_weights,
)
from evidence.visibility import (
    filter_depth_consistent_landmarks,
    make_intrinsics_from_fov,
    project_landmarks_to_query,
)
from data.splits import split_support_query_cameras
from topology.sampling import (
    coverage_balanced_score,
)
from map_learning.trust import (
    HardCandidateTeacherCache,
    hard_candidate_preservation_loss,
)
from priors.geometry import (
    GaussianPriorGeometry,
)
from evidence.camera_pair_policy import (
    mapping_scene_points_from_depth_samples,
)
from evidence.triangulation import (
    attach_pair_triangulation_statistics,
    assign_triangulated_tracks_to_landmarks,
    build_cycle_consistent_tracks,
    camera_pose_bins,
    robust_triangulate_associations,
)
from evidence.parallel_triangulation import (
    robust_triangulate_associations_fresh_cpu,
)
from evidence.track_provenance_assignment import (
    assign_tracks_by_splat_provenance,
)
from features.multiview_fusion import (
    PIXEL_CENTER_OFFSET,
    accumulate_cosine_histogram,
    consensus_eligibility,
    cosine_histogram_trim_thresholds,
    geometry_view_weights,
    grid_index_to_physical,
    nearest_keypoint_distance,
    sample_dense_descriptors_at_image_uv,
    sample_mask_at_grid_uv,
    surface_normals_from_rotation,
)
from data.scene import FrozenScene
from common.runtime import configure_output, seed_everything
from data.images import resolution_from_longest_edge

get_resolution_from_longest_edge = resolution_from_longest_edge


def _gaussian_model_for_type(gaussian_type, sh_degree):
    from priors.models import GaussianModel2D, GaussianModel3D

    gaussian_type = str(gaussian_type).lower()
    if gaussian_type == "2dgs":
        return GaussianModel2D(sh_degree)
    if gaussian_type == "3dgs":
        return GaussianModel3D(sh_degree)
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


def _load_rgb_prior_contract(dataset, args, primitive_count):
    requested = str(args.rgb_prior_manifest_path or "").strip()
    manifest_path = (
        Path(requested).expanduser()
        if requested
        else Path(dataset.model_path) / "rgb_prior_manifest.json"
    )
    if not manifest_path.is_absolute():
        manifest_path = Path(dataset.model_path) / manifest_path
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        if bool(args.require_rgb_prior_manifest):
            raise FileNotFoundError(
                "RGB Gaussian prior manifest is required but missing: "
                f"{manifest_path}"
            )
        return {
            "validated": False,
            "manifest_path": str(manifest_path),
            "prior_kind": "unverified_legacy",
        }

    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if bool(manifest.get("localization_state_present", True)):
        raise ValueError("RGB prior manifest still contains localization state")
    if bool(manifest.get("detector_state_present", True)):
        raise ValueError("RGB prior manifest still contains detector state")
    if str(manifest.get("gaussian_type", "")).lower() != str(
        dataset.gaussian_type
    ).lower():
        raise ValueError(
            "RGB prior Gaussian type does not match the requested loader"
        )
    if int(manifest.get("primitive_count", -1)) != int(primitive_count):
        raise ValueError(
            "RGB prior primitive count does not match the loaded checkpoint"
        )
    used_feature_loss = bool(
        manifest.get("prior_training_used_feature_loss", True)
    )
    if used_feature_loss and not bool(args.allow_feature_stripped_prior):
        raise ValueError(
            "The prior geometry/topology was trained with feature loss. "
            "Use a true rgb_only prior or explicitly pass "
            "--allow_feature_stripped_prior for the compatibility ablation."
        )
    exported_ply = Path(str(manifest.get("exported_ply", ""))).resolve()
    if not exported_ply.is_file():
        raise FileNotFoundError(
            f"Manifest-exported Gaussian PLY is missing: {exported_ply}"
        )
    expected_sha = str(manifest.get("exported_ply_sha256", ""))
    actual_sha = _file_sha256(exported_ply)
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError("RGB prior PLY hash does not match its manifest")
    expected_model_root = exported_ply.parents[2]
    if expected_model_root != Path(dataset.model_path).resolve():
        raise ValueError(
            "RGB prior manifest belongs to a different model root: "
            f"{expected_model_root} != {Path(dataset.model_path).resolve()}"
        )
    return {
        **manifest,
        "validated": True,
        "manifest_path": str(manifest_path),
    }


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
    rgb_prior_contract=None,
):
    paths = {
        "landmark_path": None if landmark_path is None else str(landmark_path),
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
        "rgb_prior": dict(rgb_prior_contract or {}),
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


def _ulf_parity_feature_input(camera, masks):
    """Return ULF-Loc's native-resolution, RGB-masked encoder input.

    Unlike the normal native contract, strict parity neither changes image
    scale nor filters detected keypoints after masking.  ULF-Loc only masks
    the RGB image before its SuperPoint call.
    """
    return _masked_camera_image(camera, masks)


def _ulf_parity_fusion_input(camera, masks):
    """Return ULF-Loc GWFF's raw RGB and its separate validity mask.

    The reference implementation masks RGB before KCS detection, but GWFF
    encodes the original support image and applies object/sky/distortion
    validity only when accumulating projected observations.  Retaining this
    asymmetry is necessary for a true parity initialization.
    """
    image = camera.original_image.cuda()
    valid_mask = _valid_mask_from_scene_masks(
        masks,
        _camera_cache_key(camera),
        image.shape[-2:],
        device=image.device,
    )
    return image, valid_mask


def _query_feature_outputs(
    camera,
    feature_extractor,
    *,
    longest_edge,
    masks=None,
):
    image, full_valid_mask = _native_feature_input(camera, masks, longest_edge)
    fine_height, fine_width = get_resolution_from_longest_edge(
        image.shape[-2],
        image.shape[-1],
        longest_edge,
    )
    with torch.no_grad():
        outputs = feature_extractor(image[None])
        encoder_feature_map = outputs["feature_map"]
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
        "query_feature_contract": _QUERY_FEATURE_CONTRACT_NATIVE,
        "feature_resize_mode": "resize_image_then_native_stride8",
        "descriptor_source": "superpoint_native_dense_resized_input",
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


def _pose_diverse_subsample_cameras(cameras, maximum):
    """Deterministically cover camera-center space with farthest-point sampling."""
    cameras = list(cameras)
    maximum = int(maximum)
    if maximum <= 0 or len(cameras) <= maximum:
        return cameras
    centers = []
    for camera in cameras:
        center = getattr(camera, "camera_center", None)
        if center is None:
            return _uniformly_subsample_cameras(cameras, maximum)
        center = torch.as_tensor(center).detach().float().cpu().reshape(-1)
        if center.numel() != 3 or not bool(torch.isfinite(center).all().item()):
            return _uniformly_subsample_cameras(cameras, maximum)
        centers.append(center)
    centers = torch.stack(centers, dim=0)
    squared_distance_to_centroid = (
        centers - centers.mean(dim=0, keepdim=True)
    ).square().sum(dim=1)
    selected_mask = torch.zeros(len(cameras), dtype=torch.bool)
    selected_indices = []
    first_index = int(torch.argmax(squared_distance_to_centroid).item())
    selected_indices.append(first_index)
    selected_mask[first_index] = True
    nearest_selected_distance = (centers - centers[first_index]).square().sum(dim=1)
    for _ in range(1, maximum):
        candidate_scores = nearest_selected_distance.masked_fill(
            selected_mask, -torch.inf
        )
        next_index = int(torch.argmax(candidate_scores).item())
        selected_indices.append(next_index)
        selected_mask[next_index] = True
        nearest_selected_distance = torch.minimum(
            nearest_selected_distance,
            (centers - centers[next_index]).square().sum(dim=1),
        )
    return [cameras[index] for index in sorted(selected_indices)]


def _subsample_ulf_support_cameras(cameras, maximum, sampling):
    sampling = str(sampling)
    if sampling == "uniform":
        return _uniformly_subsample_cameras(cameras, maximum)
    if sampling == "pose_diverse":
        return _pose_diverse_subsample_cameras(cameras, maximum)
    raise ValueError(f"Unsupported ULF support-view sampling: {sampling}")


def _camera_view_bin_ids(cameras, bin_count):
    """Assign support cameras to deterministic camera-center coverage bins."""
    cameras = list(cameras)
    bin_count = int(bin_count)
    if bin_count <= 0 or not cameras:
        return [0] * len(cameras), 0
    bin_count = min(bin_count, len(cameras))
    centers = []
    for camera in cameras:
        center = getattr(camera, "camera_center", None)
        if center is None:
            # A temporal fallback is deterministic, but callers record it so it
            # can never be mistaken for genuine pose-space coverage.
            labels = [min(bin_count - 1, index * bin_count // len(cameras)) for index in range(len(cameras))]
            return labels, bin_count
        center = torch.as_tensor(center).detach().float().cpu().reshape(-1)
        if center.numel() != 3 or not bool(torch.isfinite(center).all().item()):
            labels = [min(bin_count - 1, index * bin_count // len(cameras)) for index in range(len(cameras))]
            return labels, bin_count
        centers.append(center)
    centers = torch.stack(centers, dim=0)
    selected = []
    selected_mask = torch.zeros(len(cameras), dtype=torch.bool)
    first = int(torch.argmax((centers - centers.mean(dim=0)).square().sum(dim=1)).item())
    selected.append(first)
    selected_mask[first] = True
    nearest = (centers - centers[first]).square().sum(dim=1)
    for _ in range(1, bin_count):
        next_index = int(torch.argmax(nearest.masked_fill(selected_mask, -torch.inf)).item())
        selected.append(next_index)
        selected_mask[next_index] = True
        nearest = torch.minimum(nearest, (centers - centers[next_index]).square().sum(dim=1))
    prototypes = centers[torch.as_tensor(selected, dtype=torch.long)]
    labels = torch.cdist(centers, prototypes).argmin(dim=1).tolist()
    return [int(label) for label in labels], int(bin_count)


def _camera_trajectory_bin_ids(cameras, bin_count):
    """Split each image trajectory into deterministic chronological support bins."""
    cameras = list(cameras)
    bin_count = int(bin_count)
    if bin_count <= 0 or not cameras:
        return [0] * len(cameras), 0
    groups = {}
    for index, camera in enumerate(cameras):
        name = _camera_cache_key(camera)
        parent = str(Path(name).parent).replace("\\", "/")
        # A flat image directory is still one trajectory; chronological image
        # names are then split into bins below.
        groups.setdefault(parent or ".", []).append((name, index))
    labels = [0] * len(cameras)
    offset = 0
    for _, records in sorted(groups.items()):
        records.sort(key=lambda item: item[0])
        local_bins = min(bin_count, len(records))
        for rank, (_, index) in enumerate(records):
            labels[index] = offset + min(local_bins - 1, rank * local_bins // len(records))
        offset += local_bins
    return labels, offset


def _ulf_support_feature_input(camera, masks, longest_edge, policy, *, fusion=False):
    """Build the frozen KCS/GWFF support-image contract."""
    policy = str(policy)
    if policy == "support_rgb_only":
        if int(longest_edge) > 0:
            raise ValueError(
                "support_rgb_only requires --longest_edge 0 to preserve native ULF support semantics"
            )
        if fusion:
            image, valid_mask = _ulf_parity_fusion_input(camera, masks)
        else:
            image, valid_mask = _ulf_parity_feature_input(camera, masks)
        return image, valid_mask, False
    raise ValueError(f"Unsupported ULF support mask policy: {policy!r}")


def _cache_signature(dataset, args):
    model_path = Path(dataset.model_path).resolve()
    prior_manifest_path = model_path / "rgb_prior_manifest.json"
    if prior_manifest_path.is_file():
        prior_manifest = json.loads(prior_manifest_path.read_text())
        prior_fingerprint = {
            "manifest_present": True,
            "gaussian_type": str(prior_manifest.get("gaussian_type", "")),
            "primitive_count": int(prior_manifest.get("primitive_count", -1)),
            "geometry_sha256": str(prior_manifest.get("geometry_sha256", "")),
            "appearance_sha256": str(prior_manifest.get("appearance_sha256", "")),
            "exported_ply_sha256": str(prior_manifest.get("exported_ply_sha256", "")),
        }
    else:
        ply_path = (
            model_path
            / "point_cloud"
            / f"iteration_{int(args.load_iteration)}"
            / "point_cloud.ply"
        )
        stat = ply_path.stat() if ply_path.is_file() else None
        prior_fingerprint = {
            "manifest_present": False,
            "ply_size": int(stat.st_size) if stat is not None else -1,
            "ply_mtime_ns": int(stat.st_mtime_ns) if stat is not None else -1,
        }
    payload = {
        "version": 11,
        "query_feature_contract": _QUERY_FEATURE_CONTRACT_NATIVE,
        "feature_resize_mode": "resize_image_then_native_stride8",
        "descriptor_source": "superpoint_native_dense_resized_input",
        "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
        "pixel_center_offset": float(PIXEL_CENTER_OFFSET),
        "valid_mask_policy": _VALID_MASK_POLICY,
        "model_path": str(model_path),
        # Query caches contain rendered depth/alpha. Their identity must include
        # the frozen RGB prior, not just the image/frontend protocol.
        "rgb_prior_fingerprint": prior_fingerprint,
        "source_path": os.path.abspath(dataset.source_path),
        "load_iteration": int(args.load_iteration),
        "feature_type": str(dataset.feature_type),
        "images": str(dataset.images),
        "resolution": int(dataset.resolution),
        "longest_edge": int(dataset.longest_edge),
        "white_background": bool(dataset.white_background),
        "norm_before_render": bool(dataset.norm_before_render),
        "native_sparse_enabled": True,
        "native_sparse_keypoint_count": int(args.native_keypoint_count),
        "native_sparse_nms_radius": int(args.native_nms_radius),
        "native_sparse_coordinate_convention": (
            "superpoint_grid_index_then_pnp_plus_half_v1"
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
        "rgb_prior_fingerprint",
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
        "native_sparse_nms_radius",
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
                        "requested_keypoint_count": int(native_keypoint_count),
                        "nms_radius": int(feature_extractor.nms_radius),
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
            write_query_calibration_sidecar(path, {"queries": cached})
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
        write_query_calibration_sidecar(path, {"queries": cache})
        print(f"Saved detector-free query cache: {path}")
    return cache


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
        "feature_map",
        "frontend_metadata",
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
    frontend_metadata = cached["frontend_metadata"]
    if str(frontend_metadata.get("query_feature_contract")) != _QUERY_FEATURE_CONTRACT_NATIVE:
        raise ValueError(
            "Native semidense supervision requires a dense SuperPoint map "
            "computed from the same resized RGB input as detectAndCompute"
        )
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
        query_feature_map=cached["feature_map"].cuda().float(),
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
    """Return deployment-aligned native SuperPoint observations."""
    return (
        _cached_native_observations(
            cached,
            base_bank_xyz,
            args,
            max_observations=max_observations,
            bank_visibility_mask=bank_visibility_mask,
            prediction_bank_xyz=prediction_bank_xyz,
        ),
        None,
    )


def _native_auxiliary_contract(args):
    """Persist the deployment-aligned descriptor objective contract."""
    return {
        "schema_version": 1,
        "observation_source": "native",
        "objective": "hard",
        "native_outcome_mode": bool(args.native_outcome_mode),
        "native_sampling_mode": str(args.native_sampling_mode),
        "effective_native_retrieval_weight": float(args.retrieval_weight),
        "effective_trust_weight": float(args.trust_weight),
        "pure_native": True,
    }


def _validate_native_objective_semantics(args):
    """Validate the single deployment-aligned Stage-A objective."""
    if str(args.observation_source) != "native":
        raise ValueError("the paper release supports only native observations")
    if str(args.objective) != "hard":
        raise ValueError("the paper release supports only the hard native objective")
    if str(args.native_sampling_mode) != "detector_grid":
        raise ValueError("native Stage-A requires detector_grid sampling")
    numeric_names = ("retrieval_weight", "trust_weight")
    for name in numeric_names:
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite for native supervision")
    if not bool(args.native_outcome_mode) and float(args.retrieval_weight) != 0.0:
        raise ValueError(
            "retrieval_weight requires native_outcome_mode; set it to zero for "
            "an initialization-only stage"
        )
    outcome_values = {
        "native_global_attractor_weight": 0.0,
        "native_global_attractor_support_power": 0.5,
        "native_global_attractor_max_score": 4.0,
    }
    for name, default in outcome_values.items():
        if float(getattr(args, name, default)) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if int(args.native_global_attractor_min_incoming) < 1:
        raise ValueError("native_global_attractor_min_incoming must be positive")
    if float(args.native_global_attractor_max_score) <= 0.0:
        raise ValueError("native_global_attractor_max_score must be positive")


def _validate_ulf_initializer_semantics(args):
    """Validate the explicit robust KCS/GWFF extension independently of parity."""
    robust_scaffold = str(args.scaffold_mode) == "ulf_robust_consensus"
    robust_fusion = str(args.initialization_mode) == "ulf_robust_geometry"
    if not (robust_scaffold or robust_fusion):
        return
    if float(args.ulf_consensus_radius_px) < 0.0:
        raise ValueError("ulf_consensus_radius_px must be non-negative")
    for name in (
        "ulf_consensus_min_votes",
        "ulf_consensus_min_visible_views",
        "ulf_consensus_view_bins",
        "ulf_consensus_min_distinct_view_bins",
        "ulf_consensus_trajectory_bins",
        "ulf_consensus_min_distinct_trajectory_bins",
    ):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be non-negative")
    if not 0.0 <= float(args.ulf_consensus_min_rate) <= 1.0:
        raise ValueError("ulf_consensus_min_rate must be in [0, 1]")
    if not 0.0 <= float(args.ulf_consensus_extent_quantile) < 0.5:
        raise ValueError("ulf_consensus_extent_quantile must be in [0, 0.5)")
    if not 0.0 <= float(args.scaffold_opacity_keep_quantile) < 1.0:
        raise ValueError("scaffold_opacity_keep_quantile must be in [0, 1)")
    if not -1.0 <= float(args.ulf_fusion_descriptor_min_cosine) <= 1.0:
        raise ValueError("ulf_fusion_descriptor_min_cosine must be in [-1, 1]")
    if not 0.0 <= float(args.ulf_fusion_descriptor_trim_fraction) < 1.0:
        raise ValueError("ulf_fusion_descriptor_trim_fraction must be in [0, 1)")
    if int(args.ulf_fusion_trim_histogram_bins) < 2:
        raise ValueError("ulf_fusion_trim_histogram_bins must be at least two")
    if (
        int(args.ulf_consensus_min_distinct_view_bins) > 0
        and int(args.ulf_consensus_view_bins) <= 0
    ):
        raise ValueError(
            "ulf_consensus_min_distinct_view_bins requires ulf_consensus_view_bins"
        )
    if (
        int(args.ulf_consensus_min_distinct_trajectory_bins) > 0
        and int(args.ulf_consensus_trajectory_bins) <= 0
    ):
        raise ValueError(
            "ulf_consensus_min_distinct_trajectory_bins requires "
            "ulf_consensus_trajectory_bins"
        )
    if (
        bool(args.ulf_consensus_independent_bin_scoring)
        and int(args.ulf_consensus_view_bins) <= 0
        and int(args.ulf_consensus_trajectory_bins) <= 0
    ):
        raise ValueError(
            "ulf_consensus_independent_bin_scoring requires "
            "ulf_consensus_view_bins or ulf_consensus_trajectory_bins"
        )
    if (
        str(args.ulf_support_mask_policy) == "support_rgb_only"
        and int(args.longest_edge) > 0
    ):
        raise ValueError("support_rgb_only requires --longest_edge 0")


def _native_candidate_loss_kwargs(
    args,
    *,
    global_attractor_scores=None,
):
    """Return explicit native-candidate weights only for native proposals."""
    # The global false-attractor prior is a train-split artifact.  Validation
    # must never build or consume it, so callers receive a zero-weight no-op
    # unless they explicitly supply the frozen training prior.
    global_attractor_enabled = global_attractor_scores is not None
    return {
        "native_outcome_mode": bool(args.native_outcome_mode),
        "native_nce_weight": 0.0,
        "native_keep_weight": float(args.native_keep_weight),
        "native_keep_margin": float(args.native_keep_margin),
        "native_keep_loose_weight": 0.0,
        "native_keep_loose_radius_px": 4.0,
        "native_keep_loose_margin": 0.025,
        "native_swap_weight": float(args.native_swap_weight),
        "native_swap_margin": float(args.native_swap_margin),
        "native_miss_weight": float(args.native_miss_weight),
        "native_miss_margin": float(args.native_miss_margin),
        "native_reject_weight": 0.0,
        "native_reject_threshold": 0.5,
        "native_attractor_weight": 0.0,
        "native_attractor_margin": 0.05,
        "native_global_attractor_weight": (
            float(args.native_global_attractor_weight)
            if global_attractor_enabled
            else 0.0
        ),
        "native_global_attractor_scores": global_attractor_scores,
    }


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


def _automatic_ulf_voxel_size(xyz, budget, extent_quantile=0.0):
    xyz = torch.as_tensor(xyz).float()
    if xyz.numel() == 0:
        return 1.0
    extent_quantile = float(extent_quantile)
    if not 0.0 <= extent_quantile < 0.5:
        raise ValueError("extent_quantile must be in [0, 0.5)")
    if extent_quantile > 0.0:
        bounds = torch.quantile(
            xyz,
            torch.tensor(
                [extent_quantile, 1.0 - extent_quantile],
                device=xyz.device,
                dtype=xyz.dtype,
            ),
            dim=0,
        )
        extent = bounds[1] - bounds[0]
    else:
        extent = xyz.amax(dim=0) - xyz.amin(dim=0)
    extent = extent.clamp_min(1e-4)
    volume = float(extent.prod().item())
    return max((volume / max(int(budget), 1)) ** (1.0 / 3.0), 1e-4)


def _resolve_consensus_capacity(
    consensus_count: int,
    requested_budget: int,
    *,
    allow_nonconsensus_fallback: bool,
    allow_underfill: bool,
) -> tuple[int, bool, str]:
    """Resolve a safety cap without silently changing KCS eligibility."""
    consensus_count = int(consensus_count)
    requested_budget = int(requested_budget)
    if consensus_count <= 0:
        raise RuntimeError("Robust KCS gates produced no consensus landmarks")
    if consensus_count >= requested_budget:
        return requested_budget, False, "consensus_saturation_cap"
    if allow_nonconsensus_fallback and allow_underfill:
        raise ValueError(
            "non-consensus fallback and consensus underfill are mutually exclusive"
        )
    if allow_nonconsensus_fallback:
        return requested_budget, True, "fixed_with_nonconsensus_fallback"
    if allow_underfill:
        return consensus_count, False, "consensus_saturation_cap"
    raise RuntimeError(
        "Robust KCS gates produced too few consensus landmarks: "
        f"eligible={consensus_count} budget={requested_budget}. Relax a "
        "named gate, enable consensus underfill, or explicitly enable "
        "non-consensus fallback for a fixed-budget compatibility run."
    )


def _build_ulf_robust_consensus_landmark_indices(
    gaussians,
    cameras,
    masks,
    feature_extractor,
    args,
):
    """Build a non-parity KCS bank with explicit multi-view consensus gates.

    This deliberately does not alter the strict ULF random-kNN sampler.  It is
    an experimental LaFGS selection rule whose candidates must demonstrate
    support across visible views, camera-center bins, and temporal trajectory
    bins before coverage selection can use them.
    """
    if str(feature_extractor.feature_type) != "sp":
        raise ValueError("Robust ULF consensus sampling currently requires SuperPoint")
    xyz = gaussians.get_xyz.detach().float()
    opacity = gaussians.get_opacity.detach().reshape(-1)
    finite_opacity = torch.isfinite(opacity)
    opacity_threshold = float(args.scaffold_min_opacity)
    opacity_quantile = float(args.scaffold_opacity_keep_quantile)
    if opacity_quantile > 0.0:
        finite_values = opacity[finite_opacity]
        if finite_values.numel() == 0:
            raise ValueError("Robust ULF consensus found no finite opacity values")
        quantile_threshold = float(
            torch.quantile(finite_values.float(), opacity_quantile).item()
        )
        opacity_threshold = max(opacity_threshold, quantile_threshold)
    base_eligible = finite_opacity & (opacity >= opacity_threshold)
    if int(args.scaffold_budget) <= 0:
        raise ValueError("Robust ULF consensus requires a positive scaffold budget")
    if (
        int(base_eligible.sum().item()) < int(args.scaffold_budget)
        and not bool(args.ulf_consensus_allow_underfill)
    ):
        raise ValueError(
            "Robust ULF consensus budget exceeds the opacity-eligible primitive pool"
        )
    cameras = _subsample_ulf_support_cameras(
        cameras,
        args.ulf_consensus_max_views,
        args.ulf_support_view_sampling,
    )
    view_labels, view_bin_count = _camera_view_bin_ids(
        cameras, args.ulf_consensus_view_bins
    )
    trajectory_labels, trajectory_bin_count = _camera_trajectory_bin_ids(
        cameras, args.ulf_consensus_trajectory_bins
    )
    if int(args.ulf_consensus_min_distinct_view_bins) > 0 and view_bin_count <= 0:
        raise ValueError("robust KCS distinct-view gate requires --ulf_consensus_view_bins")
    if (
        int(args.ulf_consensus_min_distinct_trajectory_bins) > 0
        and trajectory_bin_count <= 0
    ):
        raise ValueError(
            "robust KCS distinct-trajectory gate requires --ulf_consensus_trajectory_bins"
        )
    votes = torch.zeros(xyz.shape[0], dtype=torch.int32, device=xyz.device)
    visibility_counts = torch.zeros_like(votes)
    view_vote_mask = (
        torch.zeros(
            (xyz.shape[0], view_bin_count), dtype=torch.bool, device=xyz.device
        )
        if view_bin_count > 0
        else None
    )
    view_visibility_mask = (
        torch.zeros(
            (xyz.shape[0], view_bin_count), dtype=torch.bool, device=xyz.device
        )
        if view_bin_count > 0
        else None
    )
    trajectory_vote_mask = (
        torch.zeros(
            (xyz.shape[0], trajectory_bin_count),
            dtype=torch.bool,
            device=xyz.device,
        )
        if trajectory_bin_count > 0
        else None
    )
    trajectory_visibility_mask = (
        torch.zeros(
            (xyz.shape[0], trajectory_bin_count),
            dtype=torch.bool,
            device=xyz.device,
        )
        if trajectory_bin_count > 0
        else None
    )
    policy = str(args.ulf_support_mask_policy)
    radius_px = float(args.ulf_consensus_radius_px)
    view_records = []
    for view_index, camera in enumerate(
        tqdm(cameras, desc="Robust ULF keypoint-consensus sampling")
    ):
        image, valid_mask, post_detection_filter = _ulf_support_feature_input(
            camera, masks, args.longest_edge, policy
        )
        height, width = image.shape[-2:]
        sparse = feature_extractor.detectAndCompute(
            image[None], top_k=args.ulf_consensus_keypoints
        )[0]
        keypoints = sparse["keypoints"]
        if post_detection_filter:
            keypoints = keypoints[sample_mask_at_grid_uv(valid_mask, keypoints)]
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
        valid_projection = (
            sample_mask_at_grid_uv(valid_mask, uv)
            if post_detection_filter
            else torch.ones_like(projected)
        )
        visible = base_eligible & projected & render_visible & valid_projection
        visible_indices = torch.nonzero(visible, as_tuple=False).reshape(-1)
        if visible_indices.numel() > int(args.ulf_consensus_max_candidates_per_view) > 0:
            _, keep = torch.topk(
                opacity[visible_indices],
                int(args.ulf_consensus_max_candidates_per_view),
                largest=True,
                sorted=False,
            )
            visible_indices = visible_indices[keep]
        if visible_indices.numel() > 0:
            visibility_counts[visible_indices] += 1
            if view_visibility_mask is not None:
                view_visibility_mask[
                    visible_indices, view_labels[view_index]
                ] = True
            if trajectory_visibility_mask is not None:
                trajectory_visibility_mask[
                    visible_indices, trajectory_labels[view_index]
                ] = True
        matched_indices = visible_indices.new_empty((0,), dtype=torch.long)
        if visible_indices.numel() > 0 and keypoints.numel() > 0:
            distance = nearest_keypoint_distance(
                uv[visible_indices],
                keypoints,
                chunk_size=args.ulf_consensus_distance_chunk,
            )
            matched_indices = visible_indices[distance <= radius_px]
            if matched_indices.numel() > 0:
                votes[matched_indices] += 1
                if view_vote_mask is not None:
                    view_vote_mask[matched_indices, view_labels[view_index]] = True
                if trajectory_vote_mask is not None:
                    trajectory_vote_mask[
                        matched_indices, trajectory_labels[view_index]
                    ] = True
        view_records.append(
            {
                "image_name": _camera_cache_key(camera),
                "processed_image_hw": [int(height), int(width)],
                "sparse_keypoints_before_mask": int(sparse["keypoints"].shape[0]),
                "sparse_keypoints_after_mask": int(keypoints.shape[0]),
                "contribution_visible_primitives": int(render_visible.sum().item()),
                "eligible_visible_primitives": int(visible_indices.numel()),
                "consensus_votes": int(matched_indices.numel()),
                "view_bin": int(view_labels[view_index]) if view_bin_count else None,
                "trajectory_bin": (
                    int(trajectory_labels[view_index]) if trajectory_bin_count else None
                ),
            }
        )

    distinct_views = (
        view_vote_mask.sum(dim=1, dtype=torch.int32)
        if view_vote_mask is not None
        else None
    )
    visible_view_bins = (
        view_visibility_mask.sum(dim=1, dtype=torch.int32)
        if view_visibility_mask is not None
        else None
    )
    distinct_trajectories = (
        trajectory_vote_mask.sum(dim=1, dtype=torch.int32)
        if trajectory_vote_mask is not None
        else None
    )
    visible_trajectory_bins = (
        trajectory_visibility_mask.sum(dim=1, dtype=torch.int32)
        if trajectory_visibility_mask is not None
        else None
    )
    independent_bin_scoring = bool(args.ulf_consensus_independent_bin_scoring)
    if independent_bin_scoring:
        if distinct_views is not None:
            consensus_votes = distinct_views
            consensus_visibility = visible_view_bins
            consensus_evidence = "camera_center_bins"
        elif distinct_trajectories is not None:
            consensus_votes = distinct_trajectories
            consensus_visibility = visible_trajectory_bins
            consensus_evidence = "trajectory_bins"
        else:
            raise ValueError(
                "independent-bin KCS scoring requires camera-center or trajectory bins"
            )
    else:
        consensus_votes = votes
        consensus_visibility = visibility_counts
        consensus_evidence = "frames"
    min_votes = max(int(args.ulf_consensus_min_votes), 1)
    consensus_eligible, consensus_rate = consensus_eligibility(
        consensus_votes,
        consensus_visibility,
        minimum_votes=min_votes,
        minimum_visible_views=int(args.ulf_consensus_min_visible_views),
        minimum_rate=float(args.ulf_consensus_min_rate),
        distinct_view_bins=distinct_views,
        minimum_distinct_view_bins=int(args.ulf_consensus_min_distinct_view_bins),
        distinct_trajectory_bins=distinct_trajectories,
        minimum_distinct_trajectory_bins=int(
            args.ulf_consensus_min_distinct_trajectory_bins
        ),
    )
    consensus_eligible &= base_eligible
    requested_budget = int(args.scaffold_budget)

    # Persist gate statistics before enforcing the capacity requirement.  A
    # strict KCS run can legitimately fail to fill its requested bank; without
    # this audit it is impossible to distinguish an over-constrained gate from
    # an implementation or visibility regression.
    progressive = base_eligible.clone()
    gate_counts = {"base_eligible": int(progressive.sum().item())}
    gate_masks = [
        ("minimum_votes", consensus_votes >= min_votes),
        (
            "minimum_visible_views",
            consensus_visibility >= int(args.ulf_consensus_min_visible_views),
        ),
        ("minimum_consensus_rate", consensus_rate >= float(args.ulf_consensus_min_rate)),
    ]
    if int(args.ulf_consensus_min_distinct_view_bins) > 0:
        gate_masks.append(
            (
                "minimum_distinct_view_bins",
                distinct_views >= int(args.ulf_consensus_min_distinct_view_bins),
            )
        )
    if int(args.ulf_consensus_min_distinct_trajectory_bins) > 0:
        gate_masks.append(
            (
                "minimum_distinct_trajectory_bins",
                distinct_trajectories
                >= int(args.ulf_consensus_min_distinct_trajectory_bins),
            )
        )
    for name, gate in gate_masks:
        progressive &= gate
        gate_counts[f"after_{name}"] = int(progressive.sum().item())

    def _eligible_count(mask):
        return int((base_eligible & mask).sum().item())

    def _gate_stat(values):
        values = values[base_eligible].float()
        if values.numel() == 0:
            return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
        quantiles = torch.quantile(
            values, torch.tensor([0.1, 0.5, 0.9], device=values.device)
        )
        return {
            "count": int(values.numel()),
            "mean": float(values.mean().item()),
            "p10": float(quantiles[0].item()),
            "p50": float(quantiles[1].item()),
            "p90": float(quantiles[2].item()),
            "max": float(values.max().item()),
        }

    gate_audit = {
        "mode": "ulf_robust_keypoint_consensus_gate_audit_v1",
        "requested_budget": requested_budget,
        "support_view_count": int(len(cameras)),
        "gate_configuration": {
            "minimum_opacity": opacity_threshold,
            "opacity_keep_quantile": opacity_quantile,
            "minimum_votes": min_votes,
            "minimum_visible_views": int(args.ulf_consensus_min_visible_views),
            "minimum_consensus_rate": float(args.ulf_consensus_min_rate),
            "minimum_distinct_view_bins": int(
                args.ulf_consensus_min_distinct_view_bins
            ),
            "minimum_distinct_trajectory_bins": int(
                args.ulf_consensus_min_distinct_trajectory_bins
            ),
        },
        "progressive_eligible_counts": gate_counts,
        "individual_gate_eligible_counts": {
            "minimum_votes": _eligible_count(consensus_votes >= min_votes),
            "minimum_visible_views": _eligible_count(
                consensus_visibility >= int(args.ulf_consensus_min_visible_views)
            ),
            "minimum_consensus_rate": _eligible_count(
                consensus_rate >= float(args.ulf_consensus_min_rate)
            ),
            "minimum_distinct_view_bins": (
                _eligible_count(
                    distinct_views
                    >= int(args.ulf_consensus_min_distinct_view_bins)
                )
                if int(args.ulf_consensus_min_distinct_view_bins) > 0
                else None
            ),
            "minimum_distinct_trajectory_bins": (
                _eligible_count(
                    distinct_trajectories
                    >= int(args.ulf_consensus_min_distinct_trajectory_bins)
                )
                if int(args.ulf_consensus_min_distinct_trajectory_bins) > 0
                else None
            ),
        },
        "rate_sweep_after_minimum_votes_and_visibility": {
            f"{threshold:g}": int(
                (
                    base_eligible
                    & (consensus_votes >= min_votes)
                    & (
                        consensus_visibility
                        >= int(args.ulf_consensus_min_visible_views)
                    )
                    & (consensus_rate >= threshold)
                )
                .sum()
                .item()
            )
            for threshold in (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1)
        },
        "consensus_eligible_primitives": int(consensus_eligible.sum().item()),
        "vote_statistics_over_base_eligible": _gate_stat(votes),
        "visibility_statistics_over_base_eligible": _gate_stat(visibility_counts),
        "consensus_evidence": consensus_evidence,
        "independent_bin_scoring": independent_bin_scoring,
        "independent_vote_statistics_over_base_eligible": _gate_stat(
            consensus_votes
        ),
        "independent_visibility_statistics_over_base_eligible": _gate_stat(
            consensus_visibility
        ),
        "consensus_rate_statistics_over_base_eligible": _gate_stat(consensus_rate),
        "distinct_view_bin_statistics_over_base_eligible": (
            _gate_stat(distinct_views) if distinct_views is not None else None
        ),
        "distinct_trajectory_bin_statistics_over_base_eligible": (
            _gate_stat(distinct_trajectories)
            if distinct_trajectories is not None
            else None
        ),
        "visible_view_bin_statistics_over_base_eligible": (
            _gate_stat(visible_view_bins) if visible_view_bins is not None else None
        ),
        "visible_trajectory_bin_statistics_over_base_eligible": (
            _gate_stat(visible_trajectory_bins)
            if visible_trajectory_bins is not None
            else None
        ),
    }
    gate_audit_path = Path(args.output_dir) / "robust_kcs_gate_audit.json"
    gate_audit_path.parent.mkdir(parents=True, exist_ok=True)
    with gate_audit_path.open("w") as handle:
        json.dump(gate_audit, handle, indent=2, sort_keys=True)
    print(f"Saved robust KCS gate audit: {gate_audit_path}")

    default_allow_fallback = False
    allow_fallback = (
        default_allow_fallback
        if args.ulf_consensus_allow_nonconsensus_fallback is None
        else bool(args.ulf_consensus_allow_nonconsensus_fallback)
    )
    consensus_count = int(consensus_eligible.sum().item())
    effective_budget, fallback_to_non_consensus, capacity_policy = (
        _resolve_consensus_capacity(
            consensus_count,
            requested_budget,
            allow_nonconsensus_fallback=allow_fallback,
            allow_underfill=bool(args.ulf_consensus_allow_underfill),
        )
    )
    # Rate only becomes a score after it has first been enforced as a gate.
    vote_score = (
        consensus_votes.float()
        + 0.25
        * consensus_votes.float().max().clamp_min(1.0)
        * consensus_rate
    )
    vote_score += 0.01 * consensus_visibility.float()
    voxel_size = float(args.ulf_consensus_voxel_size)
    if voxel_size <= 0.0:
        voxel_size = _automatic_ulf_voxel_size(
            xyz[base_eligible],
            effective_budget,
            extent_quantile=args.ulf_consensus_extent_quantile,
        )
    if fallback_to_non_consensus:
        # Preserve every primitive that passed the named consensus gates.
        # Capacity-only fallback points fill the remaining fixed protocol
        # budget without displacing that reliable core.
        consensus_core = torch.nonzero(
            consensus_eligible, as_tuple=False
        ).reshape(-1)
        fill_budget = requested_budget - int(consensus_core.numel())
        coverage_fill = coverage_balanced_score(
            xyz,
            fill_budget,
            vote_score,
            voxel_size=voxel_size,
            max_per_voxel=args.ulf_consensus_max_per_voxel,
            eligible=base_eligible & ~consensus_eligible,
            allow_overflow=True,
        )
        selected = torch.cat((consensus_core, coverage_fill))
    else:
        selected = coverage_balanced_score(
            xyz,
            effective_budget,
            vote_score,
            voxel_size=voxel_size,
            max_per_voxel=args.ulf_consensus_max_per_voxel,
            eligible=consensus_eligible,
            allow_overflow=True,
        )
    if selected.numel() != effective_budget:
        raise RuntimeError(
            "Robust ULF consensus scaffold could not satisfy its resolved capacity: "
            f"requested_cap={requested_budget} resolved={effective_budget} "
            f"selected={selected.numel()}"
        )
    selected_votes = consensus_votes[selected]
    selected_rates = consensus_rate[selected]
    diagnostics = {
        "mode": "ulf_robust_keypoint_consensus_v1",
        "strict_ulf_parity": False,
        "budget": int(selected.numel()),
        "requested_budget": requested_budget,
        "resolved_budget": effective_budget,
        "capacity_policy": capacity_policy,
        "underfilled_to_consensus_saturation": bool(
            effective_budget < requested_budget
            and not fallback_to_non_consensus
        ),
        "eligible_primitives": int(base_eligible.sum().item()),
        "consensus_eligible_primitives": int(consensus_eligible.sum().item()),
        "fallback_to_non_consensus": bool(fallback_to_non_consensus),
        "fallback_nonconsensus_count": int(
            requested_budget - consensus_count
            if fallback_to_non_consensus
            else 0
        ),
        "effective_minimum_opacity": opacity_threshold,
        "opacity_keep_quantile": opacity_quantile,
        "selected_with_consensus": int((selected_votes >= min_votes).sum().item()),
        "selected_vote_mean": float(selected_votes.float().mean().item()),
        "selected_vote_max": int(selected_votes.max().item()),
        "selected_consensus_rate_mean": float(selected_rates.mean().item()),
        "selected_consensus_rate_p10": float(torch.quantile(selected_rates, 0.1).item()),
        "minimum_votes": min_votes,
        "minimum_visible_views": int(args.ulf_consensus_min_visible_views),
        "minimum_consensus_rate": float(args.ulf_consensus_min_rate),
        "consensus_evidence": consensus_evidence,
        "independent_bin_scoring": independent_bin_scoring,
        "distinct_view_bins": int(view_bin_count),
        "minimum_distinct_view_bins": int(args.ulf_consensus_min_distinct_view_bins),
        # ``distinct_trajectory_bins`` is the realized scene-wide total because
        # trajectory bins are allocated independently per image sequence.  Keep
        # the configured per-sequence policy separate so cached scaffolds can
        # be validated after a resume.
        "trajectory_bins_per_group": int(args.ulf_consensus_trajectory_bins),
        "distinct_trajectory_bins": int(trajectory_bin_count),
        "minimum_distinct_trajectory_bins": int(
            args.ulf_consensus_min_distinct_trajectory_bins
        ),
        "gate_audit_path": str(gate_audit_path),
        "selected_distinct_view_bins_mean": (
            float(distinct_views[selected].float().mean().item())
            if distinct_views is not None
            else 0.0
        ),
        "selected_distinct_trajectory_bins_mean": (
            float(distinct_trajectories[selected].float().mean().item())
            if distinct_trajectories is not None
            else 0.0
        ),
        "consensus_radius_px": radius_px,
        "consensus_sparse_keypoints": int(args.ulf_consensus_keypoints),
        "consensus_view_count": int(len(cameras)),
        "support_view_sampling": str(args.ulf_support_view_sampling),
        "support_mask_policy": policy,
        "post_detection_mask_filter": bool(policy == "deployment_post_filter"),
        "distance_chunk_size": int(args.ulf_consensus_distance_chunk),
        "candidate_cap_per_view": int(args.ulf_consensus_max_candidates_per_view),
        "voxel_size": float(voxel_size),
        "voxel_extent_quantile": float(args.ulf_consensus_extent_quantile),
        "max_per_voxel": int(args.ulf_consensus_max_per_voxel),
        "visibility": "2dgs_raster_contribution_gradient",
        "visibility_resolution": "native_support_input_resolution",
        "query_feature_contract": _QUERY_FEATURE_CONTRACT_NATIVE,
        "coordinate_convention": "grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "views": view_records,
    }
    return selected.detach().cpu(), diagnostics


def _build_ulf_robust_geometry_features(
    cameras,
    gaussians,
    landmark_indices,
    masks,
    feature_extractor,
    fallback_features,
    args,
):
    """Fuse GWFF observations after descriptor-prototype outlier trimming.

    The first pass is a conventional geometry-weighted prototype.  A second
    streaming pass estimates each landmark's descriptor-cosine quantile, and
    the final pass recomputes the prototype from only retained observations.
    No observation tensor is retained across cameras, so this remains usable
    for full-support 2DGS banks.
    """
    if float(args.ulf_fusion_min_cosine) != 0.0:
        raise ValueError(
            "ulf_fusion_min_cosine is a legacy geometry-weight threshold, not a "
            "descriptor cosine gate; use --ulf_fusion_descriptor_min_cosine"
        )
    landmark_indices_gpu = landmark_indices.to(device=gaussians.get_xyz.device)
    bank_xyz = gaussians.get_xyz[landmark_indices_gpu].detach().float()
    rotations = gaussians.get_rotation[landmark_indices_gpu].detach().float()
    scales = gaussians.get_scaling[landmark_indices_gpu].detach().float()
    normals = surface_normals_from_rotation(rotations, scales)
    cameras = _subsample_ulf_support_cameras(
        cameras, args.ulf_fusion_max_views, args.ulf_support_view_sampling
    )
    fusion_view_labels, fusion_view_bin_count = _camera_view_bin_ids(
        cameras, args.ulf_fusion_view_bins
    )
    exact_bin_balance = (
        bool(args.ulf_fusion_exact_bin_balance) and fusion_view_bin_count > 0
    )
    fusion_view_balance = None
    if fusion_view_bin_count > 0 and not exact_bin_balance:
        label_tensor = torch.as_tensor(
            fusion_view_labels, dtype=torch.long, device=bank_xyz.device
        )
        bin_counts = torch.bincount(
            label_tensor, minlength=fusion_view_bin_count
        ).clamp_min(1)
        fusion_view_balance = (
            float(len(cameras))
            / float(fusion_view_bin_count)
            / bin_counts[label_tensor].float()
        )
    policy = str(args.ulf_support_mask_policy)
    feature_dim = int(fallback_features.shape[1])
    bank_count = int(bank_xyz.shape[0])
    pretrim_count = torch.zeros(bank_count, dtype=torch.long, device=bank_xyz.device)
    sampled_weight_sum = 0.0
    sampled_weight_count = 0
    native_size_mismatch_views = 0
    first_pass_records = []

    def for_each_observation(description, callback, *, record_views=False):
        nonlocal sampled_weight_sum, sampled_weight_count, native_size_mismatch_views
        records = []
        for view_index, camera in enumerate(tqdm(cameras, desc=description)):
            image, valid_mask, _ = _ulf_support_feature_input(
                camera,
                masks,
                args.longest_edge,
                policy,
                fusion=True,
            )
            height, width = image.shape[-2:]
            dense_features, _ = feature_extractor.detectAndComputeDense(image[None])
            expected_hw = (
                int(dense_features.shape[-2]) * 8,
                int(dense_features.shape[-1]) * 8,
            )
            if record_views:
                native_size_mismatch_views += int(
                    expected_hw != (int(height), int(width))
                )
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
            # Fusion always applies support validity at the projected point.
            # ``support_rgb_only`` differs from deployment only in the sparse
            # detector/KCS post-filter, mirroring ULF's RGB-versus-GWFF split.
            valid = visible & projected & sample_mask_at_grid_uv(valid_mask, grid_uv)
            compact_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
            useful_count = 0
            if compact_indices.numel() > 0:
                sampled = sample_dense_descriptors_at_image_uv(
                    dense_features,
                    grid_index_to_physical(grid_uv[compact_indices]),
                    (height, width),
                )
                weights = geometry_view_weights(
                    bank_xyz[compact_indices],
                    normals[compact_indices],
                    camera.camera_center.cuda(),
                ).float()
                if fusion_view_balance is not None:
                    weights = weights * fusion_view_balance[view_index]
                useful = weights > 0.0
                compact_indices = compact_indices[useful]
                sampled = sampled[useful]
                weights = weights[useful]
                useful_count = int(compact_indices.numel())
                if useful_count:
                    callback(
                        compact_indices,
                        sampled.float(),
                        weights,
                        (
                            int(fusion_view_labels[view_index])
                            if fusion_view_bin_count > 0
                            else -1
                        ),
                    )
                    if record_views:
                        sampled_weight_sum += float(weights.sum().item())
                        sampled_weight_count += useful_count
            if record_views:
                records.append(
                    {
                        "image_name": _camera_cache_key(camera),
                        "processed_image_hw": [int(height), int(width)],
                        "contribution_visible_landmarks": int(visible.sum().item()),
                        "valid_projected_landmarks": int(valid.sum().item()),
                        "geometry_weighted_samples": useful_count,
                    }
                )
            del dense_features
        return records

    accumulator_leading_shape = (
        (fusion_view_bin_count, bank_count)
        if exact_bin_balance
        else (bank_count,)
    )
    prototype_sum = torch.zeros(
        (*accumulator_leading_shape, feature_dim),
        device=bank_xyz.device,
        dtype=torch.float32,
    )
    prototype_weight = torch.zeros(
        accumulator_leading_shape,
        device=bank_xyz.device,
        dtype=torch.float32,
    )

    def accumulate_weighted(
        descriptor_sum,
        weight_sum,
        indices,
        sampled,
        weights,
        view_bin,
    ):
        if exact_bin_balance:
            descriptor_sum[view_bin].index_add_(
                0, indices, sampled * weights[:, None]
            )
            weight_sum[view_bin].index_add_(0, indices, weights)
        else:
            descriptor_sum.index_add_(0, indices, sampled * weights[:, None])
            weight_sum.index_add_(0, indices, weights)

    def finalize_weighted(descriptor_sum, weight_sum, fallback):
        if not exact_bin_balance:
            observed = weight_sum > 1e-8
            result = F.normalize(fallback.float(), dim=-1).clone()
            if bool(observed.any().item()):
                result[observed] = F.normalize(
                    descriptor_sum[observed] / weight_sum[observed, None],
                    dim=-1,
                )
            return result, observed
        observed_by_bin = weight_sum > 1e-8
        observed = observed_by_bin.any(dim=0)
        result = F.normalize(fallback.float(), dim=-1).clone()
        if bool(observed.any().item()):
            per_bin = F.normalize(
                descriptor_sum
                / weight_sum.clamp_min(1e-8)[..., None],
                dim=-1,
            )
            per_bin = per_bin * observed_by_bin[..., None]
            equal_bin_mean = per_bin.sum(dim=0) / observed_by_bin.sum(
                dim=0
            ).clamp_min(1)[..., None]
            result[observed] = F.normalize(equal_bin_mean[observed], dim=-1)
        return result, observed

    def accumulate_prototype(indices, sampled, weights, view_bin):
        accumulate_weighted(
            prototype_sum,
            prototype_weight,
            indices,
            sampled,
            weights,
            view_bin,
        )
        pretrim_count.index_add_(
            0, indices, torch.ones_like(indices, dtype=pretrim_count.dtype)
        )

    first_pass_records = for_each_observation(
        "Robust ULF GWFF prototype", accumulate_prototype, record_views=True
    )
    prototype, prototype_observed = finalize_weighted(
        prototype_sum, prototype_weight, fallback_features
    )

    reference = prototype
    reference_observed = prototype_observed

    trim_fraction = float(args.ulf_fusion_descriptor_trim_fraction)
    descriptor_min_cosine = float(args.ulf_fusion_descriptor_min_cosine)
    needs_trim = trim_fraction > 0.0 or descriptor_min_cosine > -1.0
    thresholds = torch.full((bank_count,), -1.0, device=bank_xyz.device)
    posttrim_count = pretrim_count.clone()
    result = reference
    trim_fractions = torch.full(
        (bank_count,), trim_fraction, dtype=torch.float32, device=bank_xyz.device
    )
    if needs_trim:
        histogram = torch.zeros(
            (bank_count, int(args.ulf_fusion_trim_histogram_bins)),
            dtype=torch.int32,
            device=bank_xyz.device,
        )

        def accumulate_histogram(indices, sampled, _weights, _view_bin):
            cosine = (sampled * reference[indices]).sum(dim=1)
            histogram.copy_(accumulate_cosine_histogram(histogram, indices, cosine))

        for_each_observation("Robust ULF GWFF cosine histogram", accumulate_histogram)
        thresholds = cosine_histogram_trim_thresholds(
            histogram, trim_fractions
        ).to(
            device=bank_xyz.device
        )
        thresholds = torch.maximum(
            thresholds,
            thresholds.new_full(thresholds.shape, descriptor_min_cosine),
        )
        trimmed_sum = torch.zeros_like(prototype_sum)
        trimmed_weight = torch.zeros_like(prototype_weight)
        posttrim_count = torch.zeros_like(pretrim_count)

        def accumulate_trimmed(indices, sampled, weights, view_bin):
            cosine = (sampled * reference[indices]).sum(dim=1)
            keep = cosine >= thresholds[indices]
            if not bool(keep.any().item()):
                return
            indices = indices[keep]
            sampled = sampled[keep]
            weights = weights[keep]
            accumulate_weighted(
                trimmed_sum,
                trimmed_weight,
                indices,
                sampled,
                weights,
                view_bin,
            )
            posttrim_count.index_add_(
                0, indices, torch.ones_like(indices, dtype=posttrim_count.dtype)
            )

        for_each_observation("Robust ULF GWFF trimmed fusion", accumulate_trimmed)
        result, retained = finalize_weighted(
            trimmed_sum, trimmed_weight, fallback_features
        )

    observed = posttrim_count > 0
    retained_fraction = float(
        posttrim_count.sum().item() / max(int(pretrim_count.sum().item()), 1)
    )
    diagnostics = {
        "initialization_mode": "ulf_robust_geometry_weighted_fusion_v1",
        "strict_ulf_parity": False,
        "observed_landmarks": int(observed.sum().item()),
        "unobserved_landmarks": int((~observed).sum().item()),
        "observation_count_mean": float(posttrim_count.float().mean().item()),
        "observation_count_median": float(posttrim_count.float().median().item()),
        "observation_count_max": int(posttrim_count.max().item()),
        "pretrim_observation_count": int(pretrim_count.sum().item()),
        "retained_observation_count": int(posttrim_count.sum().item()),
        "retained_observation_fraction": retained_fraction,
        "descriptor_trim_enabled": bool(needs_trim),
        "descriptor_trim_fraction": trim_fraction,
        "descriptor_min_cosine": descriptor_min_cosine,
        "descriptor_trim_histogram_bins": int(args.ulf_fusion_trim_histogram_bins),
        "fusion_reference_mode": "mean",
        "reference_to_prototype_cosine_mean": (
            float((reference[reference_observed] * prototype[reference_observed]).sum(dim=1).mean().item())
            if bool(reference_observed.any().item())
            else None
        ),
        "descriptor_threshold_mean_observed": (
            float(thresholds[reference_observed].mean().item())
            if bool(reference_observed.any().item())
            else -1.0
        ),
        "geometry_weight_mean": sampled_weight_sum / max(sampled_weight_count, 1),
        "geometry_weighted_samples": int(sampled_weight_count),
        "fusion_view_count": int(len(cameras)),
        "fusion_view_bin_count": int(fusion_view_bin_count),
        "fusion_view_bin_balanced": bool(fusion_view_bin_count > 0),
        "fusion_view_bin_balance_mode": (
            "per_landmark_bin_then_equal_bin"
            if exact_bin_balance
            else (
                "inverse_global_frame_count"
                if fusion_view_bin_count > 0
                else "frame_weighted"
            )
        ),
        "support_view_sampling": str(args.ulf_support_view_sampling),
        "support_mask_policy": policy,
        "native_stride8_size_mismatch_views": int(native_size_mismatch_views),
        "visibility": "2dgs_raster_contribution_gradient",
        "visibility_resolution": "native_support_input_resolution",
        "query_feature_contract": _QUERY_FEATURE_CONTRACT_NATIVE,
        "coordinate_convention": "grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "views": first_pass_records,
    }
    return result, posttrim_count, diagnostics


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
        if args.scaffold_mode == "ulf_robust_consensus":
            _assert_cached_consensus_scaffold(metadata, args)
        metadata.setdefault("mode", f"{args.scaffold_mode}_cached")
        metadata.setdefault("budget", int(indices.numel()))
        return indices, path, metadata

    if args.scaffold_mode != "ulf_robust_consensus":
        raise ValueError(f"unsupported paper scaffold mode: {args.scaffold_mode}")
    if cameras is None or feature_extractor is None:
        raise ValueError("KCS requires mapping cameras and SuperPoint")
    indices, diagnostics = _build_ulf_robust_consensus_landmark_indices(
        gaussians,
        cameras,
        masks,
        feature_extractor,
        args,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(indices.detach().cpu(), handle)
    with metadata_path.open("w") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
    print(
        "Saved robust KCS localization scaffold: "
        f"{output_path} count={indices.numel()}"
    )
    return indices.cpu(), output_path, diagnostics


def _assert_cached_consensus_scaffold(metadata, args):
    """Reject a cached KCS scaffold produced under a different policy."""
    if not metadata:
        raise ValueError(
            "cached robust scaffold lacks metadata; regenerate the scaffold"
        )
    expected = {
        "requested_budget": int(args.scaffold_budget),
        "consensus_sparse_keypoints": int(args.ulf_consensus_keypoints),
        "consensus_radius_px": float(args.ulf_consensus_radius_px),
        "minimum_votes": max(int(args.ulf_consensus_min_votes), 1),
        "minimum_visible_views": int(args.ulf_consensus_min_visible_views),
        "minimum_consensus_rate": float(args.ulf_consensus_min_rate),
        "distinct_view_bins": int(args.ulf_consensus_view_bins),
        "minimum_distinct_view_bins": int(
            args.ulf_consensus_min_distinct_view_bins
        ),
        "minimum_distinct_trajectory_bins": int(
            args.ulf_consensus_min_distinct_trajectory_bins
        ),
        "independent_bin_scoring": bool(
            args.ulf_consensus_independent_bin_scoring
        ),
        "candidate_cap_per_view": int(
            args.ulf_consensus_max_candidates_per_view
        ),
        "max_per_voxel": int(args.ulf_consensus_max_per_voxel),
        "voxel_extent_quantile": float(args.ulf_consensus_extent_quantile),
        "support_view_sampling": str(args.ulf_support_view_sampling),
        "support_mask_policy": str(args.ulf_support_mask_policy),
    }
    for key, target in expected.items():
        if key not in metadata:
            raise ValueError(
                f"cached robust scaffold lacks {key}; regenerate the scaffold"
            )
        value = metadata[key]
        if isinstance(target, float):
            matches = math.isclose(
                float(value), target, rel_tol=1e-7, abs_tol=1e-7
            )
        else:
            matches = value == target
        if not matches:
            raise ValueError(
                "cached robust scaffold policy mismatch for "
                f"{key}: {value!r} != {target!r}; regenerate the scaffold"
            )
    target_trajectory_bins = int(args.ulf_consensus_trajectory_bins)
    cached_trajectory_bins = metadata.get("trajectory_bins_per_group")
    if cached_trajectory_bins is not None:
        trajectory_policy_matches = int(cached_trajectory_bins) == target_trajectory_bins
    else:
        # Legacy diagnostics stored the realized total across all sequences in
        # ``distinct_trajectory_bins``.  Reconstruct the per-sequence policy
        # from the persisted view records instead of comparing that total with
        # the configured bins-per-sequence value.
        grouped_records = {}
        for record in metadata.get("views", ()):
            image_name = str(record.get("image_name", ""))
            label = record.get("trajectory_bin")
            if not image_name or label is None:
                continue
            parent = str(Path(image_name).parent).replace("\\", "/") or "."
            grouped_records.setdefault(parent, []).append(int(label))
        if grouped_records:
            trajectory_policy_matches = all(
                len(set(labels)) == min(target_trajectory_bins, len(labels))
                for labels in grouped_records.values()
            )
        else:
            trajectory_policy_matches = (
                metadata.get("distinct_trajectory_bins") == target_trajectory_bins
            )
    if not trajectory_policy_matches:
        raise ValueError(
            "cached robust scaffold policy mismatch for trajectory_bins_per_group: "
            f"{cached_trajectory_bins!r} != {target_trajectory_bins!r}; "
            "regenerate the scaffold"
        )
    if bool(args.ulf_consensus_allow_underfill):
        if metadata.get("capacity_policy") != "consensus_saturation_cap":
            raise ValueError(
                "cached robust scaffold used fixed non-consensus fallback; "
                "regenerate it for adaptive consensus saturation"
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
        "query_feature_contract": _QUERY_FEATURE_CONTRACT_NATIVE,
        "feature_resize_mode": "resize_image_then_native_stride8",
        "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "visibility_resolution": "native_sparse_input",
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
                "version": 3,
                "signature": signature,
                "signature_payload": signature_payload,
                "visibility": visibility,
            },
            path,
        )
        print(f"Saved rasterizer visibility cache: {path}")
    return visibility


@torch.no_grad()
def _load_initial_features(
    path,
    landmark_indices,
    feature_dim,
    device,
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
    mvinit_observation_count = state.get("mvinit_observation_count")
    mvinit_count_valid = False
    if mvinit_observation_count is not None:
        mvinit_observation_count = torch.as_tensor(
            mvinit_observation_count, dtype=torch.long
        ).reshape(-1)
        mvinit_count_valid = (
            mvinit_observation_count.numel() == state_indices.numel()
        )
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
    if not torch.equal(state_indices, landmark_indices_cpu):
        raise ValueError("Initial state landmark IDs do not match the fixed scaffold")
    if mvinit_count_valid:
        state["mvinit_observation_count"] = mvinit_observation_count
    state["_mvinit_observation_count_alignment_valid"] = mvinit_count_valid
    if raw_offset_valid:
        state["raw_anchor_offset"] = raw_anchor_offset
    state["_raw_anchor_offset_alignment_valid"] = raw_offset_valid
    return F.normalize(features.to(device), dim=-1), state, int(
        landmark_indices.numel()
    )


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


def _history_windows(records, window_count=10):
    """Summarize the full optimization course without storing every record.

    Long native-residual runs retain only ``history_tail`` in their output to
    keep the artifact compact.  That makes an early instability or a
    keep/swap/miss/reject collapse invisible after a 5K-step run.  Contiguous
    record windows preserve the trajectory at fixed size while keeping the
    complete tail available for detailed late-stage inspection.
    """
    window_count = int(window_count)
    if window_count <= 0:
        raise ValueError("window_count must be positive")
    if not records:
        return []
    effective_count = min(window_count, len(records))
    windows = []
    for window_index in range(effective_count):
        start = window_index * len(records) // effective_count
        end = (window_index + 1) * len(records) // effective_count
        window_records = records[start:end]
        if not window_records:
            continue
        first = window_records[0]
        last = window_records[-1]
        windows.append(
            {
                "start_step": int(first.get("step", start + 1)),
                "end_step": int(last.get("step", end)),
                "record_count": len(window_records),
                "diagnostics": _mean_diagnostics(window_records),
            }
        )
    return windows


def _all_parameter_gradients_finite(parameters):
    for parameter in parameters:
        if parameter.grad is not None and not bool(
            torch.isfinite(parameter.grad).all().item()
        ):
            return False
    return True


@torch.no_grad()
def _stable_clip_grad_norm(parameters, max_norm):
    """Clip finite gradients without float32 norm overflow.

    Native sparse residual updates occasionally produce a few very large but
    finite descriptor entries. ``torch.nn.utils.clip_grad_norm_`` accumulates
    the global L2 norm in the gradient dtype, which can overflow in float32
    before clipping is applied.  Compute the reduction in float64 so those
    samples receive the intended bounded update instead of being discarded.
    """
    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.zeros((), dtype=torch.float64), False

    norms = []
    for gradient in gradients:
        values = (
            gradient.coalesce().values()
            if gradient.is_sparse
            else gradient
        )
        norms.append(torch.linalg.vector_norm(values.detach(), dtype=torch.float64))
    total_norm = torch.linalg.vector_norm(torch.stack(norms), dtype=torch.float64)
    if not bool(torch.isfinite(total_norm).item()):
        return total_norm, False

    clip_limit = float(max_norm)
    clipped = bool(total_norm.item() > clip_limit)
    if clipped:
        coefficient = (clip_limit / (total_norm + 1e-6)).to(
            device=gradients[0].device,
            dtype=gradients[0].dtype,
        )
        for gradient in gradients:
            gradient.mul_(coefficient)
    return total_norm, clipped


@torch.no_grad()
def _assign_tracks_by_splat_provenance(
    *,
    tracks,
    track_geometry,
    keypoints,
    query_names,
    cache,
    bank_xyz,
    gaussians,
    cameras_by_name,
    landmark_global_indices,
    background,
    args,
):
    """Compatibility bridge to the replayable provenance API."""
    return assign_tracks_by_splat_provenance(
        tracks=tracks,
        track_geometry=track_geometry,
        keypoints=keypoints,
        query_names=query_names,
        cache=cache,
        bank_xyz=bank_xyz,
        gaussians=gaussians,
        cameras_by_name=cameras_by_name,
        landmark_global_indices=landmark_global_indices,
        background=background,
        topk=args.geometry_teacher_provenance_topk,
        minimum_consensus_rate=(
            args.geometry_teacher_provenance_min_consensus_rate
        ),
        minimum_views=args.geometry_teacher_provenance_min_views,
        group_maximum_landmarks=(
            args.geometry_teacher_provenance_group_max_landmarks
        ),
        group_minimum_relative_mass=(
            args.geometry_teacher_provenance_group_min_relative_mass
        ),
        group_minimum_consensus_rate=(
            args.geometry_teacher_provenance_group_min_consensus_rate
        ),
        depth_absolute_tolerance_m=(
            args.geometry_teacher_provenance_depth_abs_tolerance_m
        ),
        depth_relative_tolerance=(
            args.geometry_teacher_provenance_depth_rel_tolerance
        ),
    )


def _track_triangulation_backend(args, track_count: int):
    if (
        int(args.geometry_teacher_triangulation_cpu_workers) > 1
        and int(track_count)
        >= int(args.geometry_teacher_parallel_triangulation_min_tracks)
    ):
        return robust_triangulate_associations_fresh_cpu, {
            "worker_count": int(
                args.geometry_teacher_triangulation_cpu_workers
            )
        }
    return robust_triangulate_associations, {}


@torch.no_grad()
def _collect_track_first_geometry_teacher(
    query_names,
    cache,
    bank_xyz,
    args,
    *,
    provenance_context=None,
    return_track_payload=False,
):
    """Build G2 image-side tracks, triangulate them, then associate geometry."""
    descriptors = []
    keypoints = []
    detector_scores = []
    camera_intrinsics = []
    camera_poses = []
    depth_sources = []
    image_hw = []
    for name in query_names:
        cached = cache[name]
        descriptors.append(cached["native_descriptors"].float())
        keypoints.append(
            cached["native_keypoints"].float() + float(PIXEL_CENTER_OFFSET)
        )
        detector_scores.append(cached["native_scores"].float())
        camera_intrinsics.append(cached["native_K"].float())
        camera_poses.append(cached["pose_w2c"].float())
        depth_sources.append(
            cached.get(
                "native_depth_at_keypoints",
                cached.get("native_depth"),
            )
        )
        if depth_sources[-1] is None:
            raise ValueError(
                f"Geometry teacher cache lacks native depth for {name}"
            )
        native_hw = cached.get("native_input_hw")
        if native_hw is None:
            depth = cached.get("native_depth")
            if depth is None or torch.as_tensor(depth).ndim < 2:
                raise ValueError(
                    f"Geometry teacher cache lacks native_input_hw for {name}"
                )
            native_hw = torch.as_tensor(depth).shape[-2:]
        image_hw.append(torch.as_tensor(native_hw, dtype=torch.long))
    camera_intrinsics = torch.stack(camera_intrinsics)
    camera_poses = torch.stack(camera_poses)

    def sample_depth(depth_source, query_keypoints, rows=None):
        source = torch.as_tensor(depth_source)
        selected = (
            torch.arange(query_keypoints.shape[0], dtype=torch.long)
            if rows is None
            else torch.as_tensor(rows, dtype=torch.long)
        )
        if source.ndim == 1:
            return source[selected]
        physical = query_keypoints[selected] - float(PIXEL_CENTER_OFFSET)
        x = physical[:, 0].round().long().clamp(0, int(source.shape[1]) - 1)
        y = physical[:, 1].round().long().clamp(0, int(source.shape[0]) - 1)
        return source[y, x]

    depth_at_keypoints = [
        sample_depth(depth, query_keypoints).float()
        for depth, query_keypoints in zip(depth_sources, keypoints)
    ]
    pair_sidecar_requested = bool(args.save_track_pair_sidecar) or (
        str(args.geometry_teacher_track_pair_policy) != "nearest"
    )
    pair_scene_points = None
    if pair_sidecar_requested:
        pair_scene_points = mapping_scene_points_from_depth_samples(
            keypoints,
            depth_at_keypoints,
            camera_intrinsics,
            camera_poses,
            points_per_camera=(
                args.geometry_teacher_track_pair_scene_points_per_camera
            ),
            maximum_points=(
                args.geometry_teacher_track_pair_maximum_scene_points
            ),
            voxel_size_m=(
                args.geometry_teacher_track_pair_scene_point_voxel_size_m
            ),
        )
    track_result = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        detector_scores=detector_scores,
        camera_K=camera_intrinsics,
        pose_w2c=camera_poses,
        pair_neighbors=args.geometry_teacher_track_pair_neighbors,
        pair_policy=args.geometry_teacher_track_pair_policy,
        pair_image_hw=(torch.stack(image_hw) if pair_sidecar_requested else None),
        pair_scene_points_xyz=pair_scene_points,
        pair_minimum_overlap_jaccard=(
            args.geometry_teacher_track_pair_min_overlap_jaccard
        ),
        pair_minimum_joint_visibility_points=(
            args.geometry_teacher_track_pair_min_joint_visibility_points
        ),
        pair_parallax_saturation_deg=(
            args.geometry_teacher_track_pair_parallax_saturation_deg
        ),
        pair_diversity_weight=(
            args.geometry_teacher_track_pair_diversity_weight
        ),
        pair_candidate_pool_per_camera=(
            args.geometry_teacher_track_pair_candidate_pool_per_camera
        ),
        minimum_baseline_m=args.geometry_teacher_track_min_baseline_m,
        maximum_baseline_m=args.geometry_teacher_track_max_baseline_m,
        maximum_axis_angle_deg=args.geometry_teacher_track_max_axis_angle_deg,
        minimum_similarity=args.geometry_teacher_track_min_similarity,
        minimum_margin=args.geometry_teacher_track_min_margin,
        maximum_epipolar_error_px=(
            args.geometry_teacher_track_max_epipolar_error_px
        ),
        epipolar_candidate_topk=(
            args.geometry_teacher_track_epipolar_candidate_topk
        ),
        epipolar_recovered_minimum_similarity=(
            args.geometry_teacher_track_epipolar_recovered_min_similarity
        ),
        epipolar_recovered_minimum_margin=(
            args.geometry_teacher_track_epipolar_recovered_min_margin
        ),
        minimum_track_views=args.geometry_teacher_min_views,
        require_cycle=args.geometry_teacher_track_require_cycle,
        allow_chain_tracks=(
            args.geometry_teacher_track_allow_chain_tracks
        ),
        return_pair_sidecar=pair_sidecar_requested,
        device="cuda",
    )
    if pair_sidecar_requested:
        tracks, track_diagnostics, track_pair_sidecar = track_result
    else:
        tracks, track_diagnostics = track_result
        track_pair_sidecar = None
    if int(track_diagnostics["track_count"]) == 0:
        raise RuntimeError("Track-first geometry teacher produced no tracks")
    observation_query = tracks["query_index"]
    observation_keypoint = tracks["keypoint_index"]
    observation_uv = torch.stack(
        [
            keypoints[int(query)][int(keypoint)]
            for query, keypoint in zip(
                observation_query.tolist(), observation_keypoint.tolist()
            )
        ]
    )
    rendered_depth = torch.empty(observation_query.numel(), dtype=torch.float32)
    observation_order = torch.argsort(observation_query, stable=True)
    observation_counts = torch.bincount(
        observation_query, minlength=len(keypoints)
    )
    observation_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), observation_counts.cumsum(0))
    )
    for query in range(len(keypoints)):
        begin = int(observation_offsets[query])
        end = int(observation_offsets[query + 1])
        if begin == end:
            continue
        rows = observation_order[begin:end]
        rendered_depth[rows] = depth_at_keypoints[query][
            observation_keypoint[rows]
        ]
    query_bins = camera_pose_bins(
        camera_poses,
        int(args.geometry_teacher_view_bins),
        direction_weight=float(args.geometry_teacher_view_direction_weight),
    )
    triangulate, triangulation_extra = _track_triangulation_backend(
        args, int(track_diagnostics["track_count"])
    )
    track_geometry = triangulate(
        landmark_count=int(track_diagnostics["track_count"]),
        landmark_index=tracks["track_index"],
        query_index=observation_query,
        uv=observation_uv,
        confidence=tracks["confidence"],
        camera_K=camera_intrinsics,
        pose_w2c=camera_poses,
        query_bin=query_bins,
        rendered_depth=rendered_depth,
        maximum_observations_per_landmark=(
            args.geometry_teacher_max_observations_per_landmark
        ),
        minimum_views=args.geometry_teacher_min_views,
        minimum_view_bins=args.geometry_teacher_min_view_bins,
        huber_delta_px=args.geometry_teacher_huber_delta_px,
        iterations=args.geometry_teacher_iterations,
        minimum_parallax_deg=args.geometry_teacher_min_parallax_deg,
        parallax_quantile=args.geometry_teacher_parallax_quantile,
        maximum_reprojection_px=args.geometry_teacher_max_reprojection_px,
        maximum_condition_number=args.geometry_teacher_max_condition_number,
        maximum_covariance_trace_m2=(
            args.geometry_teacher_max_covariance_trace_m2
        ),
        maximum_rendered_depth_residual_m=(
            args.geometry_teacher_max_rendered_depth_residual_m
        ),
        minimum_rendered_depth_observations=(
            args.geometry_teacher_min_rendered_depth_observations
        ),
        surface_support_enabled=args.geometry_teacher_surface_support,
        surface_support_huber_m=args.geometry_teacher_surface_huber_m,
        surface_support_maximum_correction_m=(
            args.geometry_teacher_surface_max_correction_m
        ),
        surface_support_maximum_weak_information_ratio=(
            args.geometry_teacher_surface_max_weak_information_ratio
        ),
        surface_support_minimum_depth_improvement_fraction=(
            args.geometry_teacher_surface_min_depth_improvement_fraction
        ),
        surface_support_maximum_reprojection_increase_px=(
            args.geometry_teacher_surface_max_reprojection_increase_px
        ),
        surface_support_covariance_sigma_m=(
            args.geometry_teacher_surface_covariance_sigma_m
        ),
        **triangulation_extra,
    )
    track_geometry["track_confidence_level"] = tracks[
        "track_level"
    ].clone()
    if track_pair_sidecar is not None:
        track_pair_sidecar = attach_pair_triangulation_statistics(
            track_pair_sidecar, tracks, track_geometry, camera_poses
        )
    provenance_diagnostics = {}
    if str(args.geometry_teacher_identity_mode) == "track_first_provenance":
        if provenance_context is None:
            raise ValueError(
                "track_first_provenance requires frozen renderer context"
            )
        geometry, assignment, provenance_diagnostics = (
            _assign_tracks_by_splat_provenance(
                tracks=tracks,
                track_geometry=track_geometry,
                keypoints=keypoints,
                query_names=query_names,
                cache=cache,
                bank_xyz=bank_xyz,
                args=args,
                **provenance_context,
            )
        )
    else:
        geometry, assignment = assign_triangulated_tracks_to_landmarks(
            track_geometry,
            bank_xyz,
            maximum_distance_m=(
                args.geometry_teacher_track_assignment_max_distance_m
            ),
            minimum_margin_m=(
                args.geometry_teacher_track_assignment_min_margin_m
            ),
            require_high_confidence=True,
            device="cuda",
        )
    track_landmark = assignment["track_landmark_index"]
    track_high_confidence = track_geometry[
        "triangulation_high_confidence"
    ]
    support_by_query = [[] for _ in query_names]
    track_group_offsets = assignment.get("track_landmark_offsets")
    track_group_indices = assignment.get("track_landmark_indices")
    for track, query in zip(
        tracks["track_index"].tolist(), observation_query.tolist()
    ):
        if not bool(track_high_confidence[track]):
            continue
        if track_group_offsets is not None and track_group_indices is not None:
            begin = int(track_group_offsets[track])
            end = int(track_group_offsets[track + 1])
            support_by_query[query].extend(
                track_group_indices[begin:end].tolist()
            )
        else:
            landmark = int(track_landmark[track])
            if landmark >= 0:
                support_by_query[query].append(landmark)
    support_offsets = [0]
    support_indices = []
    for support in support_by_query:
        unique = torch.unique(torch.as_tensor(support, dtype=torch.long))
        support_indices.append(unique)
        support_offsets.append(support_offsets[-1] + int(unique.numel()))
    support_indices = (
        torch.cat(support_indices)
        if support_indices
        else torch.zeros(0, dtype=torch.long)
    )
    statistics = {
        "query_support_offsets": torch.as_tensor(
            support_offsets, dtype=torch.long
        ),
        "query_support_indices": support_indices,
        "query_support_query_count": torch.as_tensor(
            len(query_names), dtype=torch.long
        ),
    }
    diagnostics = {
        **track_diagnostics,
        "geometry_teacher_identity_mode": str(
            args.geometry_teacher_identity_mode
        ),
        "geometry_teacher_triangulated_track_count": int(
            track_geometry["triangulated"].sum().item()
        ),
        "geometry_teacher_high_confidence_track_count": int(
            track_high_confidence.sum().item()
        ),
        "geometry_teacher_assigned_landmark_count": int(
            geometry["track_assigned"].sum().item()
        ),
        "geometry_teacher_high_confidence_landmark_count": int(
            geometry["triangulation_high_confidence"].sum().item()
        ),
        "geometry_teacher_query_support_edge_count": int(
            support_indices.numel()
        ),
        **provenance_diagnostics,
    }
    if "landmark_track_count" in geometry:
        track_count_per_landmark = geometry["landmark_track_count"]
        assigned_landmarks = track_count_per_landmark > 0
        multi_track = track_count_per_landmark > 1
        effective_support = geometry["landmark_effective_track_support"]
        xyz_max_residual = geometry[
            "landmark_track_xyz_max_residual_m"
        ]
        diagnostics.update(
            {
                "geometry_teacher_multi_track_landmark_count": int(
                    multi_track.sum().item()
                ),
                "geometry_teacher_multi_track_fraction": float(
                    multi_track.float().sum().item()
                    / max(int(assigned_landmarks.sum().item()), 1)
                ),
                "geometry_teacher_effective_track_support_mean": float(
                    effective_support[assigned_landmarks].mean().item()
                    if bool(assigned_landmarks.any())
                    else 0.0
                ),
                "geometry_teacher_track_conflict_gt_1cm_count": int(
                    (
                        multi_track & (xyz_max_residual > 0.01)
                    ).sum().item()
                ),
                "geometry_teacher_track_conflict_gt_3cm_count": int(
                    (
                        multi_track & (xyz_max_residual > 0.03)
                    ).sum().item()
                ),
                "geometry_teacher_track_conflict_gt_5cm_count": int(
                    (
                        multi_track & (xyz_max_residual > 0.05)
                    ).sum().item()
                ),
            }
        )
    if not bool(return_track_payload):
        return statistics, geometry, diagnostics
    track_payload = {
        "version": 1,
        "schema": "lafgs_track_first_payload",
        "query_names": list(query_names),
        "tracks": {
            name: torch.as_tensor(value).detach().cpu()
            for name, value in tracks.items()
        },
        "track_geometry": {
            name: torch.as_tensor(value).detach().cpu()
            for name, value in track_geometry.items()
        },
        "assignment": {
            name: torch.as_tensor(value).detach().cpu()
            for name, value in assignment.items()
            if torch.is_tensor(value)
        },
        "query_bins": query_bins.detach().cpu(),
        "diagnostics": dict(diagnostics),
    }
    if track_pair_sidecar is not None:
        track_payload["pair_sidecar"] = track_pair_sidecar
    return statistics, geometry, diagnostics, track_payload


def _csr_positive_responsibilities(
    positive_offsets,
    positive_reprojection_errors,
    *,
    sigma_px,
):
    """Normalize Gaussian reprojection responsibilities within each CSR row."""
    offsets = torch.as_tensor(positive_offsets, dtype=torch.long)
    errors = torch.as_tensor(
        positive_reprojection_errors,
        device=offsets.device,
        dtype=torch.float32,
    ).reshape(-1)
    counts = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(
        torch.arange(counts.numel(), device=offsets.device), counts
    )
    if rows.numel() != errors.numel():
        raise ValueError("Positive CSR offsets/errors are inconsistent")
    if rows.numel() == 0:
        return rows, errors
    sigma = max(float(sigma_px), 1e-6)
    unnormalized = torch.exp(-0.5 * (errors / sigma).square())
    denominator = torch.zeros(
        counts.numel(), device=errors.device, dtype=errors.dtype
    )
    denominator.index_add_(0, rows, unnormalized)
    weights = unnormalized / denominator[rows].clamp_min(1e-12)
    return rows, weights


@torch.no_grad()
def _collect_native_global_attractor_statistics(
    features,
    query_names,
    cache,
    bank_xyz,
    args,
    *,
    visibility_cache=None,
    base_bank_xyz=None,
    max_observations=None,
):
    """Estimate train-only false-attractor priors for a fixed native bank.

    A KCS landmark can be repeatable while still attracting unrelated native
    keypoints in repeated facades.  This pass records the *target* landmark of
    every geometrically wrong top-1 prediction, unlike source-side
    matchability statistics.  It is evaluated on the training split only,
    detached from descriptor gradients, and then kept fixed for the residual
    stage so the deployment candidate graph itself is never edited.
    """
    landmark_count = int(bank_xyz.shape[0])
    device = bank_xyz.device
    incoming_count = torch.zeros(landmark_count, device=device)
    decisive_count = torch.zeros(landmark_count, device=device)
    ambiguous_count = torch.zeros(landmark_count, device=device)
    false_count = torch.zeros(landmark_count, device=device)
    correct_count = torch.zeros(landmark_count, device=device)
    visible_proposal_opportunity_count = torch.zeros(
        landmark_count, device=device
    )
    normalized_features = F.normalize(features.detach(), dim=1)
    records = []
    observation_limit = (
        int(args.max_observations)
        if max_observations is None
        else int(max_observations)
    )
    for name in tqdm(query_names, desc="Native false-attractor prior"):
        observations, _ = _primary_observations(
            cache[name],
            bank_xyz if base_bank_xyz is None else base_bank_xyz,
            args,
            max_observations=observation_limit,
            bank_visibility_mask=(
                None if visibility_cache is None else visibility_cache[name]
            ),
            prediction_bank_xyz=bank_xyz,
        )
        if observations.query_features.numel() == 0:
            continue
        query = F.normalize(
            observations.query_features.detach(), dim=1
        )
        top1 = (query @ normalized_features.T).argmax(dim=1)
        top1_distance = torch.linalg.norm(
            observations.bank_uv[top1]
            - observations.query_uv,
            dim=1,
        )
        clean = observations.bank_visible[top1] & (
            top1_distance <= float(args.positive_radius_px)
        )
        ambiguous = (
            ~clean
            & observations.bank_projected[top1]
            & (top1_distance < float(args.negative_radius_px))
        )
        false = ~clean & ~ambiguous
        ones = torch.ones_like(top1, dtype=incoming_count.dtype)
        incoming_count.index_add_(0, top1, ones)
        decisive_count.index_add_(0, top1[~ambiguous], ones[~ambiguous])
        ambiguous_count.index_add_(0, top1[ambiguous], ones[ambiguous])
        correct_count.index_add_(0, top1[clean], ones[clean])
        false_count.index_add_(0, top1[false], ones[false])
        visible_proposal_opportunity_count.add_(
            observations.bank_visible.float() * float(top1.numel())
        )
        records.append(
            {
                "observations": int(top1.numel()),
                "top1_clean_precision": float(clean.float().mean().item()),
                "unique_top1_landmarks": int(torch.unique(top1).numel()),
            }
        )

    observed = decisive_count > 0
    false_incoming_rate = torch.zeros_like(incoming_count)
    false_incoming_rate[observed] = (
        false_count[observed] / decisive_count[observed]
    )
    opportunity_observed = visible_proposal_opportunity_count > 0
    false_rate = torch.zeros_like(incoming_count)
    false_rate[opportunity_observed] = (
        false_count[opportunity_observed]
        / visible_proposal_opportunity_count[opportunity_observed]
    )
    min_incoming = max(int(args.native_global_attractor_min_incoming), 1)
    eligible = incoming_count >= float(min_incoming)
    support_reference = (
        torch.log1p(incoming_count[eligible]).median()
        if bool(eligible.any().item())
        else incoming_count.new_tensor(1.0)
    )
    support = torch.zeros_like(incoming_count)
    if bool(eligible.any().item()):
        support[eligible] = (
            torch.log1p(incoming_count[eligible])
            / support_reference.clamp_min(1e-8)
        ).pow(float(args.native_global_attractor_support_power))
    raw_score = false_rate * support
    score = torch.zeros_like(raw_score)
    positive = raw_score > 0.0
    if bool(positive.any().item()):
        score[positive] = raw_score[positive] / raw_score[positive].mean().clamp_min(
            1e-8
        )
        score.clamp_(max=float(args.native_global_attractor_max_score))
    diagnostics = _mean_diagnostics(records)
    diagnostics.update(
        {
            "native_global_attractor_prior_enabled": 1.0,
            "native_global_attractor_prior_query_count": int(len(records)),
            "native_global_attractor_prior_incoming_count": int(
                incoming_count.sum().item()
            ),
            "native_global_attractor_prior_false_count": int(
                false_count.sum().item()
            ),
            "native_global_attractor_prior_ambiguous_count": int(
                ambiguous_count.sum().item()
            ),
            "native_global_attractor_prior_raw_false_rate": float(
                false_count.sum().item()
                / decisive_count.sum().clamp_min(1.0).item()
            ),
            "native_global_attractor_prior_visible_opportunities": float(
                visible_proposal_opportunity_count.sum().item()
            ),
            "native_global_attractor_prior_eligible_landmarks": int(
                eligible.sum().item()
            ),
            "native_global_attractor_prior_nonzero_landmarks": int(
                positive.sum().item()
            ),
            "native_global_attractor_prior_score_mean": float(
                score[positive].mean().item() if bool(positive.any()) else 0.0
            ),
            "native_global_attractor_prior_score_max": float(score.max().item()),
            "native_global_attractor_prior_support_reference": float(
                support_reference.item()
            ),
        }
    )
    return {
        "incoming_count": incoming_count,
        "decisive_count": decisive_count,
        "ambiguous_count": ambiguous_count,
        "false_count": false_count,
        "correct_count": correct_count,
        "visible_proposal_opportunity_count": visible_proposal_opportunity_count,
        "false_incoming_rate": false_incoming_rate,
        "false_rate": false_rate,
        "score": score,
    }, diagnostics


@torch.no_grad()
def _validate_descriptor_field(
    features,
    validation_names,
    cache,
    bank_xyz,
    args,
    visibility_cache=None,
    base_bank_xyz=None,
):
    records = []
    for name in tqdm(validation_names, desc="Detector-free validation"):
        observations, _ = _primary_observations(
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
            dustbin_score=None,
            **_native_candidate_loss_kwargs(args),
        )
        record = dict(retrieval.diagnostics)
        record["visible_observations"] = int(observations.query_features.shape[0])
        record["matched_observations"] = int(
            (observations.source_indices >= 0).sum().item()
        )
        record["unmatched_observations"] = int(
            (observations.source_indices < 0).sum().item()
        )
        record["native_candidate_observations"] = 1.0
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
    *,
    initial_state=None,
    rgb_prior_contract=None,
):
    """Return the frozen paper-mainline training contract."""
    del initial_state
    return {
        "schema_version": 1,
        "method": "lafgs_track_centric_descriptor_reconstruction",
        "rgb_prior_frozen": True,
        "rgb_prior_contract": dict(rgb_prior_contract or {}),
        "geometry_trainable": False,
        "observation_source": "native_superpoint",
        "query_feature_contract": str(args.query_feature_contract),
        "coordinate_convention": "superpoint_grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "native_sparse_keypoint_count": int(args.native_keypoint_count),
        "native_sparse_nms_radius": int(args.native_nms_radius),
        "native_association_radius_px": float(args.native_association_radius_px),
        "native_unmatched_fraction": float(args.native_unmatched_fraction),
        "native_sampling_mode": str(args.native_sampling_mode),
        "visibility_mode": str(args.visibility_mode),
        "initialization": {
            "mode": "robust_kcs_gwff",
            "scaffold": dict(scaffold_diagnostics),
            "support_mask_policy": str(args.ulf_support_mask_policy),
            "support_view_sampling": str(args.ulf_support_view_sampling),
            "kcs_keypoints": int(args.ulf_consensus_keypoints),
            "kcs_radius_px": float(args.ulf_consensus_radius_px),
            "kcs_min_visible_views": int(args.ulf_consensus_min_visible_views),
            "kcs_min_votes": int(args.ulf_consensus_min_votes),
            "kcs_min_rate": float(args.ulf_consensus_min_rate),
            "kcs_view_bins": int(args.ulf_consensus_view_bins),
            "kcs_min_distinct_view_bins": int(
                args.ulf_consensus_min_distinct_view_bins
            ),
            "kcs_trajectory_bins": int(args.ulf_consensus_trajectory_bins),
            "kcs_min_distinct_trajectory_bins": int(
                args.ulf_consensus_min_distinct_trajectory_bins
            ),
            "gwff_trim_fraction": float(
                args.ulf_fusion_descriptor_trim_fraction
            ),
            "gwff_descriptor_min_cosine": float(
                args.ulf_fusion_descriptor_min_cosine
            ),
            "gwff_reference_mode": "mean",
            "gwff_view_bins": int(args.ulf_fusion_view_bins),
        },
        "stage_a": {
            "enabled": bool(args.native_outcome_mode),
            "steps": int(args.steps),
            "checkpoint_save_steps": sorted(
                {int(step) for step in args.save_steps} | {int(args.steps)}
            ),
            "feature_lr": float(args.feature_lr),
            "weight_decay": float(args.weight_decay),
            "retrieval_weight": float(args.retrieval_weight),
            "trust_weight": float(args.trust_weight),
            "hypothesis_topk": int(args.hypothesis_topk),
            "positive_radius_px": float(args.positive_radius_px),
            "negative_radius_px": float(args.negative_radius_px),
            "keep_weight": float(args.native_keep_weight),
            "keep_margin": float(args.native_keep_margin),
            "swap_weight": float(args.native_swap_weight),
            "swap_margin": float(args.native_swap_margin),
            "miss_weight": float(args.native_miss_weight),
            "miss_margin": float(args.native_miss_margin),
            "global_attractor_weight": float(
                args.native_global_attractor_weight
            ),
            "local_peak_weight": float(args.native_semidense_weight),
            "local_peak_start_step": int(args.native_semidense_start_step),
            "local_peak_interval": int(args.native_semidense_interval),
            "local_identity_weight": float(
                args.native_semidense_local_identity_weight
            ),
            "margin_preservation_weight": float(
                args.native_semidense_margin_preservation_weight
            ),
            "protected_set_weight": float(args.native_protected_set_weight),
            "protected_set_start_step": int(
                args.native_protected_set_start_step
            ),
            "protected_set_interval": int(args.native_protected_set_interval),
        },
        "track_first_evidence": {
            "enabled": bool(args.save_independent_geometry_teacher),
            "identity_mode": str(args.geometry_teacher_identity_mode),
            "track_payload": bool(args.save_track_micro_anchor_payload),
        },
        "split": {
            "validation_ratio": float(args.validation_ratio),
            "mode": str(args.split_mode),
            "seed": int(args.split_seed),
            "camera_order": "image_name_lexicographic",
            "train_camera_count": int(len(train_names)),
            "validation_camera_count": int(len(validation_names)),
            "train_camera_names_sha256": _camera_names_sha256(train_names),
            "validation_camera_names_sha256": _camera_names_sha256(
                validation_names
            ),
        },
        "input": {
            "model_path": os.path.abspath(dataset.model_path),
            "source_path": os.path.abspath(dataset.source_path),
            "map_iteration": int(args.load_iteration),
            "landmark_path": str(landmark_path),
        },
    }


def _save_state(
    path,
    iteration,
    landmark_indices,
    features,
    config,
    diagnostics,
    mvinit_observation_count,
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
    if str(args.query_feature_contract) != _QUERY_FEATURE_CONTRACT_NATIVE:
        raise ValueError(
            "Native candidate supervision requires --query_feature_contract "
            f"{_QUERY_FEATURE_CONTRACT_NATIVE} so training and deployment share "
            "the same resized SuperPoint input."
        )
    _validate_native_objective_semantics(args)
    _validate_ulf_initializer_semantics(args)
    gaussians = _gaussian_model_for_type(dataset.gaussian_type, dataset.sh_degree)
    scene = FrozenScene(
        dataset,
        gaussians,
        load_iteration=args.load_iteration,
        load_test_cameras=False,
    )
    if not scene.loaded_iter:
        raise ValueError("A pretrained Gaussian map checkpoint is required")
    for parameter in gaussians.parameters():
        parameter.requires_grad_(False)
    rgb_prior_contract = _load_rgb_prior_contract(
        dataset,
        args,
        primitive_count=gaussians.get_xyz.shape[0],
    )
    print(
        "RGB Gaussian prior contract: "
        f"validated={rgb_prior_contract.get('validated', False)} "
        f"kind={rgb_prior_contract.get('prior_kind', 'unknown')} "
        f"type={dataset.gaussian_type} "
        f"primitives={gaussians.get_xyz.shape[0]}"
    )

    feature_extractor = FeatureExtractor(
        dataset.feature_type, nms_radius=args.native_nms_radius
    ).cuda().eval()
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
        require_proposal_scores=False,
        require_native_sparse=True,
        native_keypoint_count=args.native_keypoint_count,
        cache_policy=args.query_cache_policy,
    )
    scaffold_visibility_counts = None

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
        rgb_prior_contract=rgb_prior_contract,
    )
    landmark_indices_cuda = landmark_indices.cuda()
    rgb_bank_xyz = gaussians.get_xyz[landmark_indices_cuda].detach().float()
    base_bank_xyz = rgb_bank_xyz
    base_bank_rotation = (
        gaussians.get_rotation[landmark_indices_cuda].detach().float()
    )
    base_bank_scaling = (
        gaussians.get_scaling[landmark_indices_cuda].detach().float()
    )
    prior_geometry = GaussianPriorGeometry(
        str(dataset.gaussian_type),
        xyz=base_bank_xyz,
        rotation=base_bank_rotation,
        scaling=base_bank_scaling,
    )
    base_bank_normals = prior_geometry.proxy_normals.detach()
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
            native_sparse=True,
        )

    fallback = _fallback_features(
        gaussians,
        landmark_indices,
        feature_dim=feature_extractor.feature_dim,
    )
    state_only_initialization = bool(
        args.initial_state_path and float(args.initial_state_blend) >= 1.0
    )
    if not state_only_initialization:
        mvinit_features, mvinit_observation_count, mvinit_diagnostics = (
            _build_ulf_robust_geometry_features(
                train_cameras,
                gaussians,
                landmark_indices,
                masks,
                feature_extractor,
                fallback,
                args,
            )
        )
    else:
        # A descriptor continuation with blend=1 reuses the bootstrap state
        # exactly. Avoid recomputing the frozen ULF fusion merely to construct
        # an unused fallback.
        mvinit_features = fallback
        mvinit_observation_count = torch.zeros(
            fallback.shape[0], dtype=torch.long, device=fallback.device
        )
        mvinit_diagnostics = {
            "initialization_mode": "ulf_robust_geometry_weighted_fusion_v1",
            "initializer_reused_from_exact_state": True,
            "observed_landmarks": 0,
            "unobserved_landmarks": int(fallback.shape[0]),
        }
    initial_state = None
    if args.initial_state_path:
        prior_features, initial_state, initial_state_match_count = _load_initial_features(
            args.initial_state_path,
            landmark_indices,
            feature_extractor.feature_dim,
            base_bank_xyz.device,
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
        mvinit_diagnostics["initial_state_alignment"] = "exact"
        if bool(
            initial_state.get(
                "_mvinit_observation_count_alignment_valid", False
            )
        ):
            inherited_mvinit_count = torch.as_tensor(
                initial_state["mvinit_observation_count"],
                dtype=torch.long,
                device=base_bank_xyz.device,
            ).reshape(-1)
            if inherited_mvinit_count.numel() != landmark_indices.numel():
                raise ValueError(
                    "Aligned MVInit observation counts do not match the "
                    "fixed landmark bank"
                )
            mvinit_observation_count = inherited_mvinit_count
            mvinit_diagnostics.update(
                {
                    "inherited_mvinit_observation_count": True,
                    "observation_count_mean": float(
                        mvinit_observation_count.float().mean().item()
                    ),
                    "observation_count_median": float(
                        mvinit_observation_count.float().median().item()
                    ),
                    "observation_count_max": int(
                        mvinit_observation_count.max().item()
                    ),
                    "observed_landmarks": int(
                        (mvinit_observation_count > 0).sum().item()
                    ),
                    "unobserved_landmarks": int(
                        (mvinit_observation_count == 0).sum().item()
                    ),
                }
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
        initial_state=initial_state,
        rgb_prior_contract=rgb_prior_contract,
    )
    residual = torch.nn.Parameter(torch.zeros_like(initial_features))
    raw_anchor_offset = torch.zeros_like(base_bank_xyz)
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
        if float(initial_raw_offset.abs().max().item()) > 1e-8:
            raise ValueError(
                "the paper release does not resume train-time anchor offsets"
            )
        mvinit_diagnostics["verified_zero_initial_anchor_offset"] = 1.0
    optimizer_parameters = [residual]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [residual],
                "lr": args.feature_lr,
                "weight_decay": args.weight_decay,
            },
        ],
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

    initial_xyz = base_bank_xyz
    native_global_attractor_scores = None
    native_global_attractor_diagnostics = {
        "native_global_attractor_prior_enabled": 0.0,
    }
    if (
        bool(args.native_outcome_mode)
        and float(args.native_global_attractor_weight) > 0.0
    ):
        (
            native_global_attractor_statistics,
            native_global_attractor_diagnostics,
        ) = _collect_native_global_attractor_statistics(
            initial_features,
            train_names,
            cache,
            initial_xyz,
            args,
            visibility_cache=visibility_cache,
            base_bank_xyz=base_bank_xyz,
        )
        native_global_attractor_scores = native_global_attractor_statistics[
            "score"
        ].detach()
        global_attractor_path = output_dir / "native_global_attractor_prior.pt"
        torch.save(
            {
                "version": 1,
                "split": "train_only",
                "train_camera_names_sha256": _camera_names_sha256(train_names),
                "landmark_indices": landmark_indices.detach().cpu(),
                "statistics": {
                    name: value.detach().cpu()
                    for name, value in native_global_attractor_statistics.items()
                },
                "diagnostics": dict(native_global_attractor_diagnostics),
            },
            global_attractor_path,
        )
        config["native_global_attractor_prior"] = {
            "enabled": True,
            "split": "train_only",
            "path": str(global_attractor_path.resolve()),
            "train_camera_names_sha256": _camera_names_sha256(train_names),
        }
    else:
        config["native_global_attractor_prior"] = {"enabled": False}
    native_loss_kwargs = _native_candidate_loss_kwargs(
        args,
        global_attractor_scores=native_global_attractor_scores,
    )
    initial_validation = _validate_descriptor_field(
        initial_features,
        validation_names,
        cache,
        initial_xyz,
        args,
        visibility_cache=visibility_cache,
        base_bank_xyz=base_bank_xyz,
    )
    _save_state(
        output_dir / "0_lafgs_map_state.pt",
        0,
        landmark_indices,
        initial_features,
        config,
        {
            **mvinit_diagnostics,
            **native_global_attractor_diagnostics,
            **initial_validation,
        },
        mvinit_observation_count,
        landmark_xyz=initial_xyz,
        raw_anchor_offset=raw_anchor_offset,
    )

    empty_observation_steps = 0
    empty_observation_checkpoint_steps = []
    semidense_reference_features = initial_features.detach().clone()
    protected_set_teacher = None
    if float(args.native_protected_set_weight) > 0.0:
        protected_set_teacher = HardCandidateTeacherCache(
            refresh_visits=args.native_protected_set_refresh_visits,
            solver="poselib",
            reprojection_error=args.native_protected_set_ransac_reprojection_px,
            confidence=0.99999,
            max_iterations=args.native_protected_set_ransac_max_iterations,
            min_iterations=args.native_protected_set_ransac_min_iterations,
            ransac_seed=args.native_protected_set_ransac_seed,
            min_inliers=4,
            max_pose_error_cm=args.native_protected_set_max_pose_error_cm,
            max_useful=args.native_protected_set_max_useful,
            max_harmful=args.native_protected_set_max_harmful,
            useful_grid_rows=args.native_protected_set_grid_rows,
            useful_grid_cols=args.native_protected_set_grid_cols,
            useful_depth_bins=args.native_protected_set_depth_bins,
            useful_surface_voxel_m=(
                args.native_protected_set_surface_voxel_m
            ),
            useful_max_per_surface_group=(
                args.native_protected_set_max_per_surface_group
            ),
            harmful_mode="all_false",
        )

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
            {
                **mvinit_diagnostics,
                **native_global_attractor_diagnostics,
                **recent,
                **validation,
            },
            mvinit_observation_count,
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
        current_xyz = base_bank_xyz
        observations, _ = _primary_observations(
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
        retrieval_observations = observations

        features = materialize_descriptor_residual(
            initial_features,
            residual,
            residual_scale=args.residual_scale,
            max_residual_norm=args.max_residual_norm,
        )
        descriptor_active = True
        descriptor_scale = float(descriptor_active)
        if not descriptor_active:
            retrieval_loss = features.sum() * 0.0
            retrieval_diagnostics = {"descriptor_retrieval_skipped": 1.0}
        else:
            step_native_loss_kwargs = dict(native_loss_kwargs)
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
                dustbin_score=None,
                **step_native_loss_kwargs,
            )
            retrieval_loss = retrieval.loss
            retrieval_diagnostics = {
                **retrieval.diagnostics,
            }
        trust_loss = descriptor_trust_loss(
            features,
            initial_features,
            weights=trust_weights,
        )
        protected_set_active = (
            descriptor_active
            and protected_set_teacher is not None
            and step >= int(args.native_protected_set_start_step)
            and step
            % max(int(args.native_protected_set_interval), 1)
            == 0
        )
        if protected_set_active:
            normalized_query = F.normalize(
                observations.query_features.detach(), dim=1
            )
            protected_scores = normalized_query @ F.normalize(
                features, dim=1
            ).T
            protected_top1_scores, protected_top1_indices = (
                protected_scores.max(dim=1)
            )
            protected_distance = torch.linalg.norm(
                observations.bank_uv[protected_top1_indices]
                - observations.query_uv,
                dim=1,
            )
            protected_projected = observations.bank_projected[
                protected_top1_indices
            ]
            protected_visible = observations.bank_visible[
                protected_top1_indices
            ]
            protected_correct = protected_visible & (
                protected_distance <= float(args.positive_radius_px)
            )
            protected_neutral = (
                ~protected_correct
                & protected_projected
                & (
                    protected_distance
                    < float(args.negative_radius_px)
                )
            )
            protected_rows = torch.arange(
                observations.query_uv.shape[0],
                device=features.device,
                dtype=torch.long,
            )
            protected_targets = protected_set_teacher.build(
                query_name,
                keypoint_xy=observations.query_uv,
                keypoint_ids=protected_rows,
                candidate_keypoint_idx=protected_rows,
                candidate_landmark_idx=protected_top1_indices.detach(),
                candidate_scores=protected_top1_scores.detach(),
                deployment_mask=torch.ones_like(
                    protected_rows, dtype=torch.bool
                ),
                gt_correct_mask=protected_correct.detach(),
                gt_neutral_mask=protected_neutral.detach(),
                landmark_xyz=current_xyz.detach(),
                K=observations.K,
                pose_gt_w2c=observations.pose_w2c,
            )
            (
                protected_set_loss,
                protected_set_loss_diagnostics,
            ) = hard_candidate_preservation_loss(
                protected_top1_scores,
                protected_targets,
                temperature=args.native_protected_set_temperature,
                margin=args.native_protected_set_margin,
                score_target=args.native_protected_set_score_target,
            )
            protected_set_diagnostics = {
                "native_protected_set_active": 1.0,
                **{
                    key: value
                    for key, value in protected_targets.diagnostics.items()
                    if isinstance(value, (bool, int, float))
                },
                **protected_set_loss_diagnostics,
            }
        else:
            protected_set_loss = features.sum() * 0.0
            protected_set_diagnostics = {
                "native_protected_set_active": 0.0
            }
        native_semidense_active = (
            descriptor_active
            and float(args.native_semidense_weight) > 0.0
            and step >= int(args.native_semidense_start_step)
            and step % max(int(args.native_semidense_interval), 1) == 0
        )
        if native_semidense_active:
            (
                native_semidense_loss,
                native_semidense_diagnostics,
            ) = native_semidense_neighborhood_loss(
                features,
                current_xyz.detach(),
                base_bank_normals,
                observations,
                positive_radius_px=args.positive_radius_px,
                max_anchors=args.native_semidense_max_anchors,
                neighbors_per_anchor=args.native_semidense_neighbors,
                neighborhood_radius_m=args.native_semidense_neighborhood_radius_m,
                normal_cosine=args.native_semidense_normal_cosine,
                local_radius_px=args.native_semidense_local_radius_px,
                target_sigma_px=args.native_semidense_target_sigma_px,
                temperature=args.native_semidense_temperature,
                protected_v2=args.native_semidense_protected_v2,
                measurement_min_reprojection_px=(
                    args.native_semidense_measurement_min_reprojection_px
                ),
                measurement_max_reprojection_px=(
                    args.native_semidense_measurement_max_reprojection_px
                ),
                surface_point_plane_m=(
                    args.native_semidense_surface_point_plane_m
                ),
                surface_max_distance_m=(
                    args.native_semidense_surface_max_distance_m
                ),
                surface_normal_cosine=(
                    args.native_semidense_surface_normal_cosine
                ),
                projected_neighbor_radius_px=(
                    args.native_semidense_projected_neighbor_radius_px
                ),
                local_identity_weight=(
                    args.native_semidense_local_identity_weight
                ),
                margin_preservation_weight=(
                    args.native_semidense_margin_preservation_weight
                ),
                reference_bank_features=semidense_reference_features,
            )
        else:
            native_semidense_loss = features.sum() * 0.0
            native_semidense_diagnostics = {
                "native_semidense_active": 0.0
            }
        semidense_gradient_diagnostics = {
            "native_semidense_gradient_audit_active": 0.0,
            "native_semidense_global_grad_norm": 0.0,
            "native_semidense_local_grad_norm": 0.0,
            "native_semidense_global_local_grad_cosine": 0.0,
            "native_semidense_gradient_conflict": 0.0,
            "native_semidense_effective_weight_scale": 1.0,
            "native_semidense_gradient_ratio_after_cap": 0.0,
            "native_semidense_alternating_local_step": float(
                native_semidense_active
                and args.native_semidense_alternate_global
            ),
        }
        semidense_weight_scale = 1.0
        if native_semidense_active and (
            bool(args.native_semidense_gradient_audit)
            or float(args.native_semidense_max_gradient_ratio) > 0.0
        ):
            global_gradient = torch.autograd.grad(
                args.retrieval_weight * retrieval_loss,
                residual,
                retain_graph=True,
                allow_unused=True,
            )[0]
            local_gradient = torch.autograd.grad(
                args.native_semidense_weight * native_semidense_loss,
                residual,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if global_gradient is not None and local_gradient is not None:
                global_flat = global_gradient.detach().reshape(-1).float()
                local_flat = local_gradient.detach().reshape(-1).float()
                global_norm = torch.linalg.norm(global_flat)
                local_norm = torch.linalg.norm(local_flat)
                denominator = (global_norm * local_norm).clamp_min(1e-12)
                cosine = torch.dot(global_flat, local_flat) / denominator
                maximum_ratio = float(
                    args.native_semidense_max_gradient_ratio
                )
                if maximum_ratio > 0.0:
                    if float(local_norm.item()) <= 0.0:
                        semidense_weight_scale = 1.0
                    elif float(global_norm.item()) <= 0.0:
                        semidense_weight_scale = 0.0
                    else:
                        semidense_weight_scale = min(
                            1.0,
                            maximum_ratio
                            * float(global_norm.item())
                            / float(local_norm.item()),
                        )
                semidense_gradient_diagnostics = {
                    "native_semidense_gradient_audit_active": 1.0,
                    "native_semidense_global_grad_norm": float(
                        global_norm.item()
                    ),
                    "native_semidense_local_grad_norm": float(
                        local_norm.item()
                    ),
                    "native_semidense_global_local_grad_cosine": float(
                        cosine.item()
                    ),
                    "native_semidense_gradient_conflict": float(
                        cosine.item() < 0.0
                    ),
                    "native_semidense_effective_weight_scale": float(
                        semidense_weight_scale
                    ),
                    "native_semidense_gradient_ratio_after_cap": float(
                        semidense_weight_scale
                        * float(local_norm.item())
                        / max(float(global_norm.item()), 1e-12)
                    ),
                    "native_semidense_alternating_local_step": float(
                        args.native_semidense_alternate_global
                    ),
                }
        global_retrieval_scale = float(
            not (
                native_semidense_active
                and bool(args.native_semidense_alternate_global)
            )
        )
        loss = (
            descriptor_scale
            * (
                global_retrieval_scale
                * args.retrieval_weight
                * retrieval_loss
                + semidense_weight_scale
                * args.native_semidense_weight
                * native_semidense_loss
                + args.trust_weight * trust_loss
                + args.native_protected_set_weight
                * protected_set_loss
            )
        )
        loss_finite = bool(torch.isfinite(loss.detach()).item())
        optimizer.zero_grad(set_to_none=True)
        if loss_finite:
            loss.backward()
        descriptor_update_active = bool(descriptor_active)
        gradients_finite = loss_finite and _all_parameter_gradients_finite(
            optimizer_parameters
        )
        gradient_clipped = False
        if gradients_finite:
            grad_norm, gradient_clipped = _stable_clip_grad_norm(
                optimizer_parameters,
                args.gradient_clip_norm,
            )
            gradients_finite = bool(torch.isfinite(grad_norm).item())
        else:
            grad_norm = loss.new_tensor(float("nan"), dtype=torch.float64)
        optimizer_step_skipped = not gradients_finite
        if gradients_finite:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            if (
                gradients_finite
                and native_semidense_active
                and step
                % max(
                    int(args.native_semidense_reference_refresh_steps), 1
                )
                == 0
            ):
                semidense_reference_features.copy_(
                    materialize_descriptor_residual(
                        initial_features,
                        residual,
                        residual_scale=args.residual_scale,
                        max_residual_norm=args.max_residual_norm,
                    )
                )
        grad_norm_value = float(grad_norm.detach().item())
        grad_norm_finite = math.isfinite(grad_norm_value)
        loss_value = float(loss.detach().item())
        record = {
            "step": step,
            "descriptor_active": float(descriptor_active),
            "descriptor_update_active": float(descriptor_update_active),
            "loss": loss_value if math.isfinite(loss_value) else 0.0,
            "loss_nonfinite": float(not math.isfinite(loss_value)),
            "retrieval_loss": float(retrieval_loss.detach().item()),
            "native_semidense_loss": float(
                native_semidense_loss.detach().item()
            ),
            "trust_loss": float(trust_loss.detach().item()),
            "native_protected_set_loss": float(
                protected_set_loss.detach().item()
            ),
            "grad_norm": grad_norm_value if grad_norm_finite else 0.0,
            "grad_norm_nonfinite": float(not grad_norm_finite),
            "gradient_clip_applied": float(gradient_clipped),
            "optimizer_step_skipped_nonfinite": float(optimizer_step_skipped),
            "visible_observations": int(observations.query_features.shape[0]),
            "matched_observations": int(
                (observations.source_indices >= 0).sum().item()
            ),
            "unmatched_observations": int(
                (observations.source_indices < 0).sum().item()
            ),
            "native_candidate_observations": 1.0,
            "native_input_count": int(
                observations.native_input_count
                if observations.native_input_count is not None
                else observations.query_features.shape[0]
            ),
            "native_valid_count": int(
                observations.native_valid_count
                if observations.native_valid_count is not None
                else observations.query_features.shape[0]
            ),
            "native_selected_count": int(
                observations.native_selected_count
                if observations.native_selected_count is not None
                else observations.query_features.shape[0]
            ),
            "native_selected_input_ratio": float(
                (
                    observations.native_selected_count
                    if observations.native_selected_count is not None
                    else observations.query_features.shape[0]
                )
                / max(
                    int(
                        observations.native_input_count
                        if observations.native_input_count is not None
                        else observations.query_features.shape[0]
                    ),
                    1,
                )
            ),
            "configured_max_observations": int(
                observations.configured_max_observations
                if observations.configured_max_observations is not None
                else args.max_observations
            ),
            "native_selection_coverage_ratio": float(
                observations.query_features.shape[0]
                / max(
                    int(
                        observations.native_valid_count
                        if observations.native_valid_count is not None
                        else observations.query_features.shape[0]
                    ),
                    1,
                )
            ),
            **retrieval_diagnostics,
            **native_semidense_diagnostics,
            **protected_set_diagnostics,
            **semidense_gradient_diagnostics,
        }
        history.append(record)
        if step % args.log_interval == 0:
            recent = _mean_diagnostics(history[-args.log_interval :])
            progress.set_postfix(
                loss=f"{recent.get('loss', 0.0):.4f}",
                retr=f"{recent.get('retrieval_loss', 0.0):.4f}",
                semi=f"{recent.get('native_semidense_loss', 0.0):.4f}",
            )
        if step in save_steps:
            save_checkpoint(step, current_xyz)

    final_features = materialize_descriptor_residual(
        initial_features,
        residual,
        residual_scale=args.residual_scale,
        max_residual_norm=args.max_residual_norm,
    )
    final_xyz = base_bank_xyz
    final_validation = _validate_descriptor_field(
        final_features,
        validation_names,
        cache,
        final_xyz,
        args,
        visibility_cache=visibility_cache,
        base_bank_xyz=base_bank_xyz,
    )
    track_evidence_summary = {}
    if bool(args.save_independent_geometry_teacher):
        identity_mode = str(args.geometry_teacher_identity_mode)
        if identity_mode not in {"track_first", "track_first_provenance"}:
            raise ValueError("the paper release supports only Track-First geometry")
        provenance_context = None
        if identity_mode == "track_first_provenance":
            if str(dataset.gaussian_type) != "2dgs":
                raise ValueError(
                    "Exact splat-provenance geometry assignment is defined only for 2DGS"
                )
            provenance_context = {
                "gaussians": gaussians,
                "cameras_by_name": {
                    _camera_cache_key(camera): camera
                    for camera in train_cameras
                },
                "landmark_global_indices": landmark_indices_cuda,
                "background": background,
            }
        teacher_output = _collect_track_first_geometry_teacher(
            train_names,
            cache,
            final_xyz,
            args,
            provenance_context=provenance_context,
            return_track_payload=bool(args.save_track_micro_anchor_payload),
        )
        track_statistics, track_geometry, track_evidence_summary = teacher_output[:3]
        if len(teacher_output) == 4:
            payload = teacher_output[3]
            torch.save(
                {
                    **payload,
                    "train_camera_names_sha256": _camera_names_sha256(train_names),
                    "landmark_indices": landmark_indices.detach().cpu(),
                },
                output_dir / "track_micro_anchor_payload.pt",
            )
        torch.save(
            {
                "version": 1,
                "split": "train_only",
                "train_camera_names_sha256": _camera_names_sha256(train_names),
                "landmark_indices": landmark_indices.detach().cpu(),
                "statistics": {
                    name: value.detach().cpu()
                    for name, value in track_statistics.items()
                },
                "geometry_evidence": {
                    name: (
                        value.detach().cpu()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for name, value in track_geometry.items()
                },
                "diagnostics": dict(track_evidence_summary),
            },
            output_dir / "track_geometry_evidence.pt",
        )
    with torch.no_grad():
        feature_cosine = (
            F.normalize(final_features, dim=-1)
            * F.normalize(initial_features, dim=-1)
        ).sum(dim=-1)
        anchor_displacement = final_xyz - base_bank_xyz
        local_anchor_displacement = torch.einsum(
            "nji,nj->ni",
            prior_geometry.frame,
            anchor_displacement,
        )
        if str(dataset.gaussian_type).lower() == "2dgs":
            tangent_norm = torch.linalg.norm(
                local_anchor_displacement[:, :2], dim=1
            )
            normal_abs = local_anchor_displacement[:, 2].abs()
        else:
            tangent_norm = torch.linalg.norm(
                local_anchor_displacement, dim=1
            )
            normal_abs = local_anchor_displacement.abs().max(dim=1).values
        summary = {
            "config": config,
            "mvinit": mvinit_diagnostics,
            "initial_validation": initial_validation,
            "final_validation": final_validation,
            "track_first_evidence": track_evidence_summary,
            "native_global_attractor_prior": native_global_attractor_diagnostics,
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
            "history_windows": _history_windows(history),
            "history_tail": history[-min(len(history), 200) :],
            "training_control": {
                "empty_observation_steps": int(empty_observation_steps),
                "empty_observation_checkpoint_steps": list(
                    empty_observation_checkpoint_steps
                ),
            },
            "geometry_invariant": {
                "base_bank_xyz_trainable": bool(base_bank_xyz.requires_grad),
                "anchor_geometry_trainable": False,
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
                    normal_abs.mean().item()
                ),
                "normal_displacement_abs_p95_m": float(
                    torch.quantile(normal_abs, 0.95).item()
                ),
                "normal_displacement_abs_max_m": float(normal_abs.max().item()),
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
        {
            **mvinit_diagnostics,
            **native_global_attractor_diagnostics,
            **_mean_diagnostics(history[-min(len(history), 200):]),
            **final_validation,
        },
        mvinit_observation_count,
        landmark_xyz=final_xyz,
        raw_anchor_offset=raw_anchor_offset,
    )
    torch.save(
        {
            "version": 1,
            "landmark_indices": landmark_indices.detach().cpu(),
            "fixed_bank": True,
            "feature_dim": int(final_features.shape[1]),
            "state_path": str(
                (output_dir / f"{args.steps}_lafgs_map_state.pt").resolve()
            ),
        },
        output_dir / "landmark_meta.pt",
    )
    summary["checkpoint_integrity"] = _checkpoint_integrity(
        output_dir, requested_checkpoint_steps
    )
    with (output_dir / "training_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    missing_checkpoint_steps = summary["checkpoint_integrity"]["missing_steps"]
    if missing_checkpoint_steps:
        raise RuntimeError(
            "Requested LaFGS checkpoint(s) were not written: "
            + ", ".join(str(step) for step in missing_checkpoint_steps)
        )
    print(f"Saved detector-free LaFGS map training output: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build the frozen LaFGS paper localization map"
    )
    model_params = ModelParams(parser)
    parser.add_argument('--load_iteration', type=int, default=30000)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--rgb_prior_manifest_path', default='')
    parser.add_argument('--require_rgb_prior_manifest', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--allow_feature_stripped_prior', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--query_feature_contract', choices=['native_resized_input'], default='native_resized_input')
    parser.add_argument('--scaffold_mode', choices=['file', 'ulf_robust_consensus'], default='ulf_robust_consensus')
    parser.add_argument('--landmark_path', default='sampled_idx.pkl')
    parser.add_argument('--generated_landmark_path', default='robust_kcs_ids.pkl')
    parser.add_argument(
        '--regenerate_scaffold',
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument('--scaffold_budget', type=int, default=16384)
    parser.add_argument('--scaffold_min_opacity', type=float, default=0.05)
    parser.add_argument('--scaffold_opacity_keep_quantile', type=float, default=0.0)
    parser.add_argument('--ulf_consensus_keypoints', type=int, default=2048)
    parser.add_argument('--ulf_consensus_radius_px', type=float, default=1.0)
    parser.add_argument('--ulf_consensus_min_votes', type=int, default=1)
    parser.add_argument('--ulf_consensus_min_visible_views', type=int, default=0)
    parser.add_argument('--ulf_consensus_min_rate', type=float, default=0.0)
    parser.add_argument('--ulf_consensus_view_bins', type=int, default=0)
    parser.add_argument('--ulf_consensus_min_distinct_view_bins', type=int, default=0)
    parser.add_argument('--ulf_consensus_trajectory_bins', type=int, default=0)
    parser.add_argument('--ulf_consensus_min_distinct_trajectory_bins', type=int, default=0)
    parser.add_argument('--ulf_consensus_independent_bin_scoring', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ulf_consensus_allow_nonconsensus_fallback', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--ulf_consensus_allow_underfill', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ulf_consensus_max_views', type=int, default=0)
    parser.add_argument('--ulf_support_view_sampling', choices=['uniform', 'pose_diverse'], default='uniform')
    parser.add_argument('--ulf_consensus_distance_chunk', type=int, default=8192)
    parser.add_argument('--ulf_consensus_max_candidates_per_view', type=int, default=0)
    parser.add_argument('--ulf_consensus_voxel_size', type=float, default=0.0)
    parser.add_argument('--ulf_consensus_extent_quantile', type=float, default=0.0)
    parser.add_argument('--ulf_consensus_max_per_voxel', type=int, default=8)
    parser.add_argument('--ulf_support_mask_policy', choices=['support_rgb_only'], default='support_rgb_only')
    parser.add_argument('--initialization_mode', choices=['ulf_robust_geometry'], default='ulf_robust_geometry')
    parser.add_argument('--ulf_fusion_max_views', type=int, default=0)
    parser.add_argument('--ulf_fusion_view_bins', type=int, default=0)
    parser.add_argument('--ulf_fusion_exact_bin_balance', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ulf_fusion_min_cosine', type=float, default=0.0)
    parser.add_argument('--ulf_fusion_descriptor_min_cosine', type=float, default=-1.0)
    parser.add_argument('--ulf_fusion_descriptor_trim_fraction', type=float, default=0.0)
    parser.add_argument('--ulf_fusion_trim_histogram_bins', type=int, default=64)
    parser.add_argument('--initial_state_path', default='')
    parser.add_argument('--initial_state_blend', type=float, default=0.0)
    parser.add_argument('--query_cache_path', default='')
    parser.add_argument('--query_cache_policy', choices=['reuse_or_build', 'readonly', 'refresh'], default='reuse_or_build')
    parser.add_argument('--visibility_mode', choices=['rasterizer'], default='rasterizer')
    parser.add_argument('--visibility_cache_path', default='')
    parser.add_argument('--observation_source', choices=['native'], default='native')
    parser.add_argument('--native_keypoint_count', type=int, default=2048)
    parser.add_argument('--native_nms_radius', type=int, default=4)
    parser.add_argument('--native_association_radius_px', type=float, default=2.0)
    parser.add_argument('--native_unmatched_fraction', type=float, default=0.25)
    parser.add_argument('--native_sampling_mode', choices=['detector_grid'], default='detector_grid')
    parser.add_argument('--objective', choices=['hard'], default='hard')
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--save_steps', type=int, nargs='*', default=[1000, 2000, 3000, 4000, 5000])
    parser.add_argument('--feature_lr', type=float, default=0.00015)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--gradient_clip_norm', type=float, default=10.0)
    parser.add_argument('--residual_scale', type=float, default=1.0)
    parser.add_argument('--max_residual_norm', type=float, default=0.0)
    parser.add_argument('--retrieval_weight', type=float, default=0.5)
    parser.add_argument('--trust_weight', type=float, default=0.02)
    parser.add_argument('--trust_observation_power', type=float, default=0.5)
    parser.add_argument('--trust_weight_min', type=float, default=0.25)
    parser.add_argument('--trust_weight_max', type=float, default=4.0)
    parser.add_argument('--native_semidense_weight', type=float, default=0.0)
    parser.add_argument('--native_semidense_start_step', type=int, default=2500)
    parser.add_argument('--native_semidense_interval', type=int, default=1)
    parser.add_argument('--native_semidense_max_anchors', type=int, default=64)
    parser.add_argument('--native_semidense_neighbors', type=int, default=1)
    parser.add_argument('--native_semidense_neighborhood_radius_m', type=float, default=0.25)
    parser.add_argument('--native_semidense_normal_cosine', type=float, default=0.8)
    parser.add_argument('--native_semidense_local_radius_px', type=float, default=8.0)
    parser.add_argument('--native_semidense_target_sigma_px', type=float, default=2.0)
    parser.add_argument('--native_semidense_temperature', type=float, default=0.07)
    parser.add_argument('--native_semidense_protected_v2', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--native_semidense_measurement_min_reprojection_px', type=float, default=2.0)
    parser.add_argument('--native_semidense_measurement_max_reprojection_px', type=float, default=8.0)
    parser.add_argument('--native_semidense_surface_point_plane_m', type=float, default=0.03)
    parser.add_argument('--native_semidense_surface_max_distance_m', type=float, default=0.15)
    parser.add_argument('--native_semidense_surface_normal_cosine', type=float, default=0.95)
    parser.add_argument('--native_semidense_projected_neighbor_radius_px', type=float, default=64.0)
    parser.add_argument('--native_semidense_local_identity_weight', type=float, default=0.0)
    parser.add_argument('--native_semidense_margin_preservation_weight', type=float, default=0.0)
    parser.add_argument('--native_semidense_gradient_audit', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--native_semidense_reference_refresh_steps', type=int, default=500)
    parser.add_argument('--native_semidense_alternate_global', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--native_semidense_max_gradient_ratio', type=float, default=0.0)
    parser.add_argument('--native_protected_set_weight', type=float, default=0.0)
    parser.add_argument('--native_protected_set_start_step', type=int, default=1000)
    parser.add_argument('--native_protected_set_interval', type=int, default=5)
    parser.add_argument('--native_protected_set_refresh_visits', type=int, default=1)
    parser.add_argument('--native_protected_set_ransac_seed', type=int, default=0)
    parser.add_argument('--native_protected_set_ransac_reprojection_px', type=float, default=8.0)
    parser.add_argument('--native_protected_set_ransac_max_iterations', type=int, default=5000)
    parser.add_argument('--native_protected_set_ransac_min_iterations', type=int, default=100)
    parser.add_argument('--native_protected_set_max_pose_error_cm', type=float, default=100.0)
    parser.add_argument('--native_protected_set_max_useful', type=int, default=96)
    parser.add_argument('--native_protected_set_max_harmful', type=int, default=96)
    parser.add_argument('--native_protected_set_grid_rows', type=int, default=4)
    parser.add_argument('--native_protected_set_grid_cols', type=int, default=4)
    parser.add_argument('--native_protected_set_depth_bins', type=int, default=4)
    parser.add_argument('--native_protected_set_surface_voxel_m', type=float, default=0.25)
    parser.add_argument('--native_protected_set_max_per_surface_group', type=int, default=2)
    parser.add_argument('--native_protected_set_temperature', type=float, default=0.05)
    parser.add_argument('--native_protected_set_margin', type=float, default=0.05)
    parser.add_argument('--native_protected_set_score_target', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--hypothesis_topk', type=int, default=32)
    parser.add_argument('--positive_radius_px', type=float, default=2.0)
    parser.add_argument('--negative_radius_px', type=float, default=6.0)
    parser.add_argument('--retrieval_margin', type=float, default=0.05)
    parser.add_argument('--missed_positive_weight', type=float, default=1.0)
    parser.add_argument('--missed_positive_margin', type=float, default=0.05)
    parser.add_argument('--unmatched_rejection_weight', type=float, default=0.0)
    parser.add_argument('--unmatched_max_similarity', type=float, default=0.5)
    parser.add_argument('--native_outcome_mode', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--native_keep_weight', type=float, default=1.0)
    parser.add_argument('--native_keep_margin', type=float, default=0.05)
    parser.add_argument('--native_swap_weight', type=float, default=1.0)
    parser.add_argument('--native_swap_margin', type=float, default=0.05)
    parser.add_argument('--native_miss_weight', type=float, default=1.0)
    parser.add_argument('--native_miss_margin', type=float, default=0.05)
    parser.add_argument('--native_global_attractor_weight', type=float, default=0.0)
    parser.add_argument('--native_global_attractor_min_incoming', type=int, default=4)
    parser.add_argument('--native_global_attractor_support_power', type=float, default=0.5)
    parser.add_argument('--native_global_attractor_max_score', type=float, default=4.0)
    parser.add_argument('--save_independent_geometry_teacher', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save_track_micro_anchor_payload', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save_track_pair_sidecar', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--geometry_teacher_max_observations_per_landmark', type=int, default=32)
    parser.add_argument('--geometry_teacher_triangulation_cpu_workers', type=int, default=2)
    parser.add_argument('--geometry_teacher_parallel_triangulation_min_tracks', type=int, default=5000)
    parser.add_argument('--geometry_teacher_min_views', type=int, default=3)
    parser.add_argument('--geometry_teacher_view_bins', type=int, default=8)
    parser.add_argument('--geometry_teacher_min_view_bins', type=int, default=2)
    parser.add_argument('--geometry_teacher_huber_delta_px', type=float, default=2.0)
    parser.add_argument('--geometry_teacher_iterations', type=int, default=3)
    parser.add_argument('--geometry_teacher_min_parallax_deg', type=float, default=1.0)
    parser.add_argument('--geometry_teacher_max_reprojection_px', type=float, default=2.0)
    parser.add_argument('--geometry_teacher_max_condition_number', type=float, default=1000000.0)
    parser.add_argument('--geometry_teacher_identity_mode', choices=['track_first', 'track_first_provenance'], default='track_first')
    parser.add_argument('--geometry_teacher_view_direction_weight', type=float, default=0.5)
    parser.add_argument('--geometry_teacher_parallax_quantile', type=float, default=0.75)
    parser.add_argument('--geometry_teacher_max_covariance_trace_m2', type=float, default=0.01)
    parser.add_argument('--geometry_teacher_max_rendered_depth_residual_m', type=float, default=0.15)
    parser.add_argument('--geometry_teacher_min_rendered_depth_observations', type=int, default=2)
    parser.add_argument('--geometry_teacher_surface_support', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--geometry_teacher_surface_huber_m', type=float, default=0.02)
    parser.add_argument('--geometry_teacher_surface_max_correction_m', type=float, default=0.08)
    parser.add_argument('--geometry_teacher_surface_max_weak_information_ratio', type=float, default=0.25)
    parser.add_argument('--geometry_teacher_surface_min_depth_improvement_fraction', type=float, default=0.10)
    parser.add_argument('--geometry_teacher_surface_max_reprojection_increase_px', type=float, default=0.05)
    parser.add_argument('--geometry_teacher_surface_covariance_sigma_m', type=float, default=0.02)
    parser.add_argument('--geometry_teacher_track_pair_neighbors', type=int, default=6)
    parser.add_argument('--geometry_teacher_track_pair_policy', choices=['nearest', 'parallax_diverse'], default='nearest')
    parser.add_argument('--geometry_teacher_track_pair_min_overlap_jaccard', type=float, default=0.15)
    parser.add_argument('--geometry_teacher_track_pair_min_joint_visibility_points', type=int, default=8)
    parser.add_argument('--geometry_teacher_track_pair_parallax_saturation_deg', type=float, default=2.0)
    parser.add_argument('--geometry_teacher_track_pair_diversity_weight', type=float, default=0.20)
    parser.add_argument('--geometry_teacher_track_pair_candidate_pool_per_camera', type=int, default=48)
    parser.add_argument('--geometry_teacher_track_pair_scene_points_per_camera', type=int, default=8)
    parser.add_argument('--geometry_teacher_track_pair_maximum_scene_points', type=int, default=4096)
    parser.add_argument('--geometry_teacher_track_pair_scene_point_voxel_size_m', type=float, default=0.02)
    parser.add_argument('--geometry_teacher_track_min_baseline_m', type=float, default=0.03)
    parser.add_argument('--geometry_teacher_track_max_baseline_m', type=float, default=5.0)
    parser.add_argument('--geometry_teacher_track_max_axis_angle_deg', type=float, default=75.0)
    parser.add_argument('--geometry_teacher_track_min_similarity', type=float, default=0.65)
    parser.add_argument('--geometry_teacher_track_min_margin', type=float, default=0.01)
    parser.add_argument('--geometry_teacher_track_max_epipolar_error_px', type=float, default=2.0)
    parser.add_argument('--geometry_teacher_track_epipolar_candidate_topk', type=int, default=1)
    parser.add_argument('--geometry_teacher_track_epipolar_recovered_min_similarity', type=float, default=-1.0)
    parser.add_argument('--geometry_teacher_track_epipolar_recovered_min_margin', type=float, default=-1.0)
    parser.add_argument('--geometry_teacher_track_require_cycle', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--geometry_teacher_track_allow_chain_tracks', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--geometry_teacher_track_assignment_max_distance_m', type=float, default=0.2)
    parser.add_argument('--geometry_teacher_track_assignment_min_margin_m', type=float, default=0.0)
    parser.add_argument('--geometry_teacher_provenance_topk', type=int, default=4)
    parser.add_argument('--geometry_teacher_provenance_min_consensus_rate', type=float, default=0.35)
    parser.add_argument('--geometry_teacher_provenance_min_views', type=int, default=2)
    parser.add_argument('--geometry_teacher_provenance_group_max_landmarks', type=int, default=1)
    parser.add_argument('--geometry_teacher_provenance_group_min_relative_mass', type=float, default=0.25)
    parser.add_argument('--geometry_teacher_provenance_group_min_consensus_rate', type=float, default=0.1)
    parser.add_argument('--geometry_teacher_provenance_depth_abs_tolerance_m', type=float, default=0.05)
    parser.add_argument('--geometry_teacher_provenance_depth_rel_tolerance', type=float, default=0.02)
    parser.add_argument('--max_observations', type=int, default=2048)
    parser.add_argument('--validation_observations', type=int, default=2048)
    parser.add_argument('--grid_rows', type=int, default=8)
    parser.add_argument('--grid_cols', type=int, default=8)
    parser.add_argument('--alpha_threshold', type=float, default=0.2)
    parser.add_argument('--depth_abs_tolerance', type=float, default=0.001)
    parser.add_argument('--depth_rel_tolerance', type=float, default=0.01)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--split_mode', choices=['random', 'sequence_block', 'temporal_block', 'stratified_temporal_block'], default='temporal_block')
    parser.add_argument('--split_seed', type=int, default=2026)
    parser.add_argument('--train_seed', type=int, default=2026)
    parser.add_argument('--max_train_views', type=int, default=0)
    parser.add_argument('--max_validation_views', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=25)
    parser.add_argument('--quiet', action="store_true")
    return parser, model_params

def main() -> None:
    parser, model_params = build_parser()
    args = parser.parse_args()
    configure_output(args.quiet)
    seed_everything(args.train_seed)
    dataset = model_params.extract(args)
    train(dataset, args)


if __name__ == "__main__":
    main()
