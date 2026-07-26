import argparse
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import get_render_visible_mask, render_from_pose_gsplat
from localization_training.detector_free_map import (
    _select_observation_rows,
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
    native_association_matches,
    native_semidense_neighborhood_loss,
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
    coverage_ranked_fill,
    coverage_preserving_sample,
    hard_score_core,
    robust_normalize,
    top_score_reservoir,
    ulf_random_knn_vote_sample,
    wilson_lower_confidence,
)
from localization_training.hard_candidate_teacher import (
    HardCandidateTeacherCache,
    derive_hard_candidate_targets,
    hard_candidate_preservation_loss,
)
from localization_training.functional_replay import (
    per_landmark_gradient_conflict,
    protected_functional_replay_loss,
)
from localization_training.gaussian_prior import (
    GaussianPriorGeometry,
    validate_gaussian_anchor_resume,
)
from localization_training.geometry_teacher import (
    assign_triangulated_tracks_to_landmarks,
    build_cycle_consistent_tracks,
    camera_pose_bins,
    robust_triangulate_associations,
    transfer_triangulated_track_groups_to_landmarks,
)
from localization_training.splat_provenance import (
    bank_splat_provenance_2dgs,
)
from localization_training.pose_information import compute_pose_information
from localization_training.pose_refiner import se3_exp
from localization_training.surface_anchor import (
    bounded_surface_local_offsets,
    build_pure_geometric_scaffold,
)
from localization_training.ulf_initializer import (
    PIXEL_CENTER_OFFSET,
    adaptive_cosine_histogram_trim_schedule,
    accumulate_cosine_histogram,
    consensus_eligibility,
    cosine_histogram_trim_thresholds,
    geometry_view_weights,
    grid_index_to_physical,
    nearest_keypoint_distance,
    sample_dense_descriptors_at_image_uv,
    sample_mask_at_grid_uv,
    surface_normals_from_rotation,
    update_weighted_cosine_medoid_state,
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


def _ulf_parity_feature_input(camera, masks):
    """Return ULF-Loc's native-resolution, RGB-masked encoder input.

    Unlike the normal native contract, strict parity neither changes image
    scale nor filters detected keypoints after masking.  ULF-Loc only masks
    the RGB image before its SuperPoint call.
    """
    return _masked_camera_image(camera, masks)


def _resolve_ulf_parity_kcs_mask_policy(policy):
    """Return the explicit support-mask semantics for strict KCS controls.

    ``rgb_only`` reproduces the ULF KCS convention: validity masks alter the
    encoder input but do not remove detected keypoints or projected primitive
    centers afterwards.  ``deployment_post_filter`` uses the production sparse
    domain at both locations.  Keeping this decision explicit is important:
    the two choices have identical RGB inputs but different candidate sets.
    """
    policy = str(policy)
    if policy not in {"rgb_only", "deployment_post_filter"}:
        raise ValueError(f"Unsupported ULF-parity KCS mask policy: {policy!r}")
    return policy


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
    """Make the support-mask choice explicit for robust KCS/GWFF experiments."""
    policy = str(policy)
    if policy == "deployment_post_filter":
        image, valid_mask = _native_feature_input(camera, masks, longest_edge)
        return image, valid_mask, True
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


def _build_functional_replay_bank(
    reference_features,
    train_names,
    cache,
    bank_xyz,
    args,
    *,
    visibility_cache=None,
    base_bank_xyz=None,
):
    """Freeze deployment behavior from every training query for later replay."""
    topm = min(int(args.functional_replay_topm), int(reference_features.shape[0]))
    rows_per_query = int(args.functional_replay_rows_per_query)
    query_features = []
    protected_ids = []
    candidate_ids = []
    candidate_logits = []
    reference_margins = []
    importance = []
    core_masks = []
    query_indices = []
    diagnostics = {
        "functional_replay_source_query_count": 0.0,
        "functional_replay_source_clean_count": 0.0,
        "functional_replay_source_core_count": 0.0,
        "functional_replay_source_selected_count": 0.0,
    }
    reference_bank = F.normalize(reference_features.detach(), dim=1)
    for query_index, query_name in enumerate(
        tqdm(train_names, desc="Functional replay bank")
    ):
        observations, _ = _primary_observations(
            cache[query_name],
            bank_xyz,
            args,
            max_observations=args.max_observations,
            bank_visibility_mask=(
                None
                if visibility_cache is None
                else visibility_cache[query_name]
            ),
            prediction_bank_xyz=(
                bank_xyz if base_bank_xyz is None else bank_xyz
            ),
        )
        if observations.query_features.numel() == 0:
            continue
        normalized_query = F.normalize(
            observations.query_features.detach(), dim=1
        )
        scores = normalized_query @ reference_bank.T
        top_logits, top_indices = torch.topk(scores, k=topm, dim=1)
        top1 = top_indices[:, 0]
        top1_distance = torch.linalg.norm(
            observations.bank_uv[top1] - observations.query_uv, dim=1
        )
        top1_correct = observations.bank_visible[top1] & (
            top1_distance <= float(args.positive_radius_px)
        )
        clean_rows = torch.nonzero(top1_correct, as_tuple=False).reshape(-1)
        diagnostics["functional_replay_source_query_count"] += 1.0
        diagnostics["functional_replay_source_clean_count"] += float(
            clean_rows.numel()
        )
        if clean_rows.numel() == 0:
            continue

        core_weights = scores.new_zeros(scores.shape[0])
        if bool(args.functional_replay_build_pnp_core):
            projected = observations.bank_projected[top1]
            neutral = (
                ~top1_correct
                & projected
                & (top1_distance < float(args.negative_radius_px))
            )
            row_ids = torch.arange(
                scores.shape[0], device=scores.device, dtype=torch.long
            )
            targets = derive_hard_candidate_targets(
                keypoint_xy=observations.query_uv,
                keypoint_ids=row_ids,
                candidate_keypoint_idx=row_ids,
                candidate_landmark_idx=top1.detach(),
                candidate_scores=top_logits[:, 0].detach(),
                deployment_mask=torch.ones_like(row_ids, dtype=torch.bool),
                gt_correct_mask=top1_correct.detach(),
                gt_neutral_mask=neutral.detach(),
                landmark_xyz=bank_xyz.detach(),
                K=observations.K,
                pose_gt_w2c=observations.pose_w2c,
                solver="poselib",
                reprojection_error=float(
                    args.functional_replay_ransac_reprojection_px
                ),
                confidence=0.99999,
                max_iterations=int(
                    args.functional_replay_ransac_max_iterations
                ),
                min_iterations=int(
                    args.functional_replay_ransac_min_iterations
                ),
                ransac_seed=int(args.train_seed),
                min_inliers=4,
                max_pose_error_cm=float(
                    args.functional_replay_max_pose_error_cm
                ),
                max_useful=int(
                    args.functional_replay_core_rows_per_query
                ),
                max_harmful=0,
                useful_grid_rows=int(args.functional_replay_grid_rows),
                useful_grid_cols=int(args.functional_replay_grid_cols),
                useful_depth_bins=int(args.functional_replay_depth_bins),
                useful_surface_voxel_m=float(
                    args.functional_replay_surface_voxel_m
                ),
                useful_max_per_surface_group=int(
                    args.functional_replay_max_per_surface_group
                ),
                harmful_mode="all_false",
            )
            core_weights = targets.useful_weights.detach()

        core_rows = torch.nonzero(
            (core_weights > 0.0) & top1_correct, as_tuple=False
        ).reshape(-1)
        margins = top_logits[:, 0] - top_logits[:, 1]
        # Core rows are always retained. Remaining capacity protects the
        # narrowest clean margins, which are the first to flip under rank
        # promotion.
        selected = core_rows
        if selected.numel() > rows_per_query:
            order = torch.argsort(
                core_weights[selected], descending=True, stable=True
            )
            selected = selected[order[:rows_per_query]]
        remaining = rows_per_query - int(selected.numel())
        if remaining > 0:
            noncore_clean = clean_rows[core_weights[clean_rows] <= 0.0]
            if noncore_clean.numel() > 0:
                order = torch.argsort(
                    margins[noncore_clean], descending=False, stable=True
                )
                selected = torch.cat(
                    [selected, noncore_clean[order[:remaining]]]
                )
        if selected.numel() == 0:
            continue
        selected_core = core_weights[selected] > 0.0
        selected_importance = torch.ones_like(margins[selected])
        selected_importance[selected_core] += float(
            args.functional_replay_pnp_core_weight
        ) * core_weights[selected][selected_core]
        query_features.append(normalized_query[selected].cpu())
        protected_ids.append(top1[selected].cpu())
        candidate_ids.append(top_indices[selected].cpu())
        candidate_logits.append(top_logits[selected].cpu())
        reference_margins.append(margins[selected].cpu())
        importance.append(selected_importance.cpu())
        core_masks.append(selected_core.cpu())
        query_indices.append(
            torch.full(
                (selected.numel(),), query_index, dtype=torch.long
            )
        )
        diagnostics["functional_replay_source_core_count"] += float(
            selected_core.sum().item()
        )
        diagnostics["functional_replay_source_selected_count"] += float(
            selected.numel()
        )

    if not query_features:
        raise RuntimeError(
            "Functional replay requested but no GT-clean deployment rows were found"
        )
    return {
        "version": 1,
        "split": "all_train",
        "train_camera_names": list(train_names),
        "query_features": torch.cat(query_features),
        "protected_landmark_indices": torch.cat(protected_ids),
        "reference_candidate_indices": torch.cat(candidate_ids),
        "reference_candidate_logits": torch.cat(candidate_logits),
        "reference_margins": torch.cat(reference_margins),
        "importance": torch.cat(importance),
        "pnp_core_mask": torch.cat(core_masks),
        "query_indices": torch.cat(query_indices),
        "diagnostics": diagnostics,
    }


def _sample_functional_replay(replay_bank, batch_size, generator, device):
    count = int(replay_bank["query_features"].shape[0])
    sample_count = min(int(batch_size), count)
    indices = torch.randint(
        count, (sample_count,), generator=generator, device="cpu"
    )
    keys = (
        "query_features",
        "protected_landmark_indices",
        "reference_candidate_indices",
        "reference_candidate_logits",
        "reference_margins",
        "importance",
        "pnp_core_mask",
    )
    return {
        key: replay_bank[key][indices].to(device=device, non_blocking=True)
        for key in keys
    }


def _native_anchor_auxiliary_scale(args):
    """Return the configured scale for projection-anchor-only losses."""
    source = str(args.observation_source)
    if source == "anchor":
        return 1.0
    if source == "native_plus_anchor":
        return float(args.native_anchor_aux_weight)
    return 0.0


def _native_auxiliary_contract(args):
    """Persist effective rather than merely parser-default auxiliary weights."""
    source = str(args.observation_source)
    anchor_scale = _native_anchor_auxiliary_scale(args)
    effective_anchor_weights = {
        "mv": anchor_scale * float(args.mv_weight),
        "local": anchor_scale * float(args.local_weight),
        "dustbin": anchor_scale * float(args.dustbin_weight),
    }
    return {
        "schema_version": 1,
        "observation_source": source,
        "objective": str(args.objective),
        "native_outcome_mode": bool(args.native_outcome_mode),
        "native_rank_budget_mode": bool(
            getattr(args, "native_rank_budget_mode", False)
        ),
        "native_rank_stage_a_steps": int(
            getattr(args, "native_rank_stage_a_steps", 0)
        ),
        "native_rank_steps": int(getattr(args, "native_rank_steps", 1)),
        "native_sampling_mode": str(args.native_sampling_mode),
        "anchor_auxiliary_scale": anchor_scale,
        "anchor_auxiliary_observations_enabled": source == "native_plus_anchor",
        "effective_anchor_weights": effective_anchor_weights,
        "effective_native_retrieval_weight": (
            float(args.retrieval_weight) if source != "anchor" else 0.0
        ),
        "effective_trust_weight": float(args.trust_weight),
        # Trust regularizes descriptor drift but does not introduce projected
        # anchor observations, so it remains compatible with a pure native run.
        "pure_native": source == "native"
        and all(weight == 0.0 for weight in effective_anchor_weights.values()),
    }


def _validate_native_objective_semantics(args):
    """Fail closed when a native stage records inert auxiliary settings.

    A plain ``native`` source never materializes projected anchor observations.
    Nonzero anchor-only options would therefore be silently ignored while still
    appearing in a checkpoint config.  Reject them instead of allowing an
    experiment to be mislabeled as a mixed or pure-native objective.
    """
    source = str(args.observation_source)
    if source not in {"native", "native_plus_anchor"}:
        return

    numeric_names = (
        "native_anchor_aux_weight",
        "mv_weight",
        "local_weight",
        "dustbin_weight",
        "retrieval_weight",
        "trust_weight",
    )
    for name in numeric_names:
        value = float(getattr(args, name))
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite for native supervision")

    native_rank_budget_mode = bool(
        getattr(args, "native_rank_budget_mode", False)
    )
    native_rank_stage_a_steps = int(
        getattr(args, "native_rank_stage_a_steps", 0)
    )
    native_rank_steps = int(getattr(args, "native_rank_steps", 1))
    if native_rank_budget_mode:
        if bool(args.native_outcome_mode) and native_rank_stage_a_steps <= 0:
            raise ValueError(
                "native_rank_budget_mode replaces rather than augments "
                "native_outcome_mode unless rank/Stage-A alternation is enabled"
            )
        if str(args.objective) != "hard":
            raise ValueError("native_rank_budget_mode requires --objective hard")
        if str(args.native_sampling_mode) != "detector_grid":
            raise ValueError(
                "native_rank_budget_mode requires --native_sampling_mode detector_grid"
            )
        if float(args.native_rank_temperature) <= 0.0:
            raise ValueError("native_rank_temperature must be positive")
        rank_values = (
            float(args.native_rank_margin_at1),
            float(args.native_rank_margin_at4),
            float(args.native_rank_margin_at8),
            float(args.native_rank_margin_at32),
            float(args.native_rank_top1_weight),
            float(args.native_rank_keep_weight),
            float(args.native_rank_reference_clean_weight),
            float(args.native_rank_reference_clean_margin),
            float(args.native_rank_band_rank1),
            float(args.native_rank_band_rank2_4),
            float(args.native_rank_band_rank5_32),
            float(args.native_rank_band_rank33_plus),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in rank_values):
            raise ValueError("native rank margins, weights, and quotas must be finite and non-negative")
        if sum(rank_values[-4:]) <= 0.0:
            raise ValueError("native rank band quotas must have a positive sum")
        if native_rank_stage_a_steps < 0:
            raise ValueError("native_rank_stage_a_steps must be non-negative")
        if native_rank_steps <= 0:
            raise ValueError("native_rank_steps must be positive")
        if (
            float(args.native_rank_reference_clean_weight) > 0.0
            and float(args.trust_weight) <= 0.0
            and float(args.max_residual_norm) <= 0.0
        ):
            raise ValueError(
                "reference-clean rank protection requires descriptor trust "
                "or a positive residual norm cap"
            )
    functional_replay_weight = float(
        getattr(args, "functional_replay_weight", 0.0)
    )
    if functional_replay_weight < 0.0 or not math.isfinite(
        functional_replay_weight
    ):
        raise ValueError("functional_replay_weight must be finite and non-negative")
    if functional_replay_weight > 0.0:
        if str(args.objective) != "hard":
            raise ValueError("functional replay requires --objective hard")
        if int(getattr(args, "functional_replay_topm", 64)) < 2:
            raise ValueError("functional_replay_topm must be at least two")
        if int(getattr(args, "functional_replay_rows_per_query", 64)) <= 0:
            raise ValueError(
                "functional_replay_rows_per_query must be positive"
            )
        core_rows = int(
            getattr(args, "functional_replay_core_rows_per_query", 16)
        )
        if core_rows <= 0 or core_rows > int(
            getattr(args, "functional_replay_rows_per_query", 64)
        ):
            raise ValueError(
                "functional_replay_core_rows_per_query must be positive and "
                "not exceed functional_replay_rows_per_query"
            )
        if int(getattr(args, "functional_replay_batch_size", 256)) <= 0:
            raise ValueError("functional_replay_batch_size must be positive")
        if float(getattr(args, "functional_replay_temperature", 0.05)) <= 0.0:
            raise ValueError("functional_replay_temperature must be positive")
        if float(getattr(args, "functional_replay_margin_slack", 0.005)) < 0.0:
            raise ValueError(
                "functional_replay_margin_slack must be non-negative"
            )

    if bool(args.native_outcome_mode):
        if str(args.objective) != "hard":
            raise ValueError("native_outcome_mode requires --objective hard")
        if str(args.native_sampling_mode) != "detector_grid":
            raise ValueError(
                "native_outcome_mode requires --native_sampling_mode detector_grid"
            )
        new_outcome_defaults = {
            "native_keep_loose_weight": 0.0,
            "native_keep_loose_margin": 0.025,
            "native_attractor_weight": 0.0,
            "native_attractor_margin": 0.05,
            "native_global_attractor_weight": 0.0,
            "native_global_attractor_support_power": 0.5,
            "native_global_attractor_max_score": 4.0,
        }
        for name, default in new_outcome_defaults.items():
            if float(getattr(args, name, default)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if int(getattr(args, "native_global_attractor_min_incoming", 4)) < 1:
            raise ValueError("native_global_attractor_min_incoming must be positive")
        if float(getattr(args, "native_global_attractor_max_score", 4.0)) <= 0.0:
            raise ValueError("native_global_attractor_max_score must be positive")
        if float(getattr(args, "native_keep_loose_radius_px", 4.0)) < float(
            getattr(args, "native_association_radius_px", 2.0)
        ):
            raise ValueError(
                "native_keep_loose_radius_px must be at least "
                "native_association_radius_px"
            )

    if source == "native":
        inert_anchor_options = {
            "native_anchor_aux_weight": float(args.native_anchor_aux_weight),
            "mv_weight": float(args.mv_weight),
            "local_weight": float(args.local_weight),
            "dustbin_weight": float(args.dustbin_weight),
        }
        nonzero = [
            name for name, value in inert_anchor_options.items() if value != 0.0
        ]
        if nonzero:
            raise ValueError(
                "--observation_source native does not create anchor observations; "
                "set these inert options to zero: " + ", ".join(nonzero)
            )
        if (
            not bool(args.native_outcome_mode)
            and not native_rank_budget_mode
            and float(args.retrieval_weight) != 0.0
        ):
            raise ValueError(
                "--observation_source native without native_outcome_mode has no "
                "deployment-aligned descriptor objective; set retrieval_weight to "
                "zero for an initialization-only or fixed-descriptor stage"
            )
        return

    if float(args.native_anchor_aux_weight) <= 0.0:
        raise ValueError(
            "--observation_source native_plus_anchor requires a positive "
            "--native_anchor_aux_weight"
        )
    if all(
        float(getattr(args, name)) == 0.0
        for name in ("mv_weight", "local_weight", "dustbin_weight")
    ):
        raise ValueError(
            "native_plus_anchor requires at least one nonzero anchor loss "
            "weight: mv_weight, local_weight, or dustbin_weight"
        )


def _validate_distillation_semantics(args):
    """Reject incomplete quality-reservoir settings before expensive statistics."""
    budget = int(args.distill_budget)
    reservoir_multiplier = float(args.distill_quality_reservoir_multiplier)
    hard_core_ratio = float(args.distill_hard_matchability_core_ratio)
    reservoir_score = str(args.distill_quality_reservoir_score)
    wilson_z = float(args.distill_quality_reservoir_wilson_z)
    global_attractor_weight = float(
        getattr(args, "distill_global_attractor_weight", 0.0)
    )
    protected_ratio = float(getattr(args, "distill_protected_core_ratio", 0.0))
    rescue_weight = float(getattr(args, "distill_rescue_weight", 0.0))
    harmful_switch_weight = float(
        getattr(args, "distill_harmful_switch_weight", 0.0)
    )
    if budget < 0:
        raise ValueError("distill_budget must be non-negative")
    if not math.isfinite(reservoir_multiplier) or reservoir_multiplier < 0.0:
        raise ValueError("distill_quality_reservoir_multiplier must be finite and non-negative")
    if 0.0 < reservoir_multiplier < 1.0:
        raise ValueError(
            "distill_quality_reservoir_multiplier must be zero or at least one"
        )
    if reservoir_multiplier > 0.0 and budget <= 0:
        raise ValueError(
            "distill_quality_reservoir_multiplier requires a positive distill_budget"
        )
    if reservoir_score not in {"posterior_mean", "wilson_lower"}:
        raise ValueError(
            "distill_quality_reservoir_score must be posterior_mean or wilson_lower"
        )
    if not math.isfinite(wilson_z) or wilson_z <= 0.0:
        raise ValueError(
            "distill_quality_reservoir_wilson_z must be finite and positive"
        )
    if not math.isfinite(hard_core_ratio) or not 0.0 <= hard_core_ratio <= 1.0:
        raise ValueError(
            "distill_hard_matchability_core_ratio must lie in [0, 1]"
        )
    if (
        not math.isfinite(global_attractor_weight)
        or global_attractor_weight < 0.0
    ):
        raise ValueError(
            "distill_global_attractor_weight must be finite and non-negative"
        )
    if global_attractor_weight > 0.0 and budget <= 0:
        raise ValueError(
            "distill_global_attractor_weight requires a positive distill_budget"
        )
    if not math.isfinite(protected_ratio) or not 0.0 <= protected_ratio <= 1.0:
        raise ValueError("distill_protected_core_ratio must lie in [0, 1]")
    if protected_ratio > 0.0 and budget <= 0:
        raise ValueError(
            "distill_protected_core_ratio requires a positive distill_budget"
        )
    if int(getattr(args, "distill_rescue_max_positives", 4)) < 1:
        raise ValueError("distill_rescue_max_positives must be positive")
    if not math.isfinite(rescue_weight) or rescue_weight < 0.0:
        raise ValueError("distill_rescue_weight must be finite and non-negative")
    if not math.isfinite(harmful_switch_weight) or harmful_switch_weight < 0.0:
        raise ValueError(
            "distill_harmful_switch_weight must be finite and non-negative"
        )


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
    adaptive_trim = bool(args.ulf_fusion_adaptive_trim)
    if adaptive_trim and float(args.ulf_fusion_descriptor_trim_fraction) != 0.0:
        raise ValueError(
            "ulf_fusion_adaptive_trim requires ulf_fusion_descriptor_trim_fraction=0 "
            "so the per-landmark schedule has unambiguous semantics"
        )
    if not (
        0.0
        <= float(args.ulf_fusion_adaptive_trim_min_fraction)
        <= float(args.ulf_fusion_adaptive_trim_max_fraction)
        < 1.0
    ):
        raise ValueError(
            "adaptive GWFF trim fractions must satisfy 0 <= min <= max < 1"
        )
    if not -1.0 <= float(args.ulf_fusion_adaptive_trim_tail_cosine) <= 1.0:
        raise ValueError("ulf_fusion_adaptive_trim_tail_cosine must be in [-1, 1]")
    if int(args.ulf_fusion_adaptive_trim_min_observations) < 1:
        raise ValueError("ulf_fusion_adaptive_trim_min_observations must be positive")
    if str(args.ulf_fusion_adaptive_trim_mode) not in {"absolute", "relative_mad"}:
        raise ValueError(
            "ulf_fusion_adaptive_trim_mode must be absolute or relative_mad"
        )
    if float(args.ulf_fusion_adaptive_trim_mad_scale) <= 0.0:
        raise ValueError("ulf_fusion_adaptive_trim_mad_scale must be positive")
    if int(args.ulf_fusion_trim_histogram_bins) < 2:
        raise ValueError("ulf_fusion_trim_histogram_bins must be at least two")
    if str(args.ulf_fusion_reference_mode) not in {
        "mean",
        "weighted_cosine_medoid",
    }:
        raise ValueError(
            "ulf_fusion_reference_mode must be mean or weighted_cosine_medoid"
        )
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
    rank_reference_bank_features=None,
    rank_landmark_opportunities=None,
):
    """Return explicit native-candidate weights only for native proposals."""
    native = str(args.observation_source) in {"native", "native_plus_anchor"}
    # The global false-attractor prior is a train-split artifact.  Validation
    # must never build or consume it, so callers receive a zero-weight no-op
    # unless they explicitly supply the frozen training prior.
    global_attractor_enabled = global_attractor_scores is not None
    return {
        "native_outcome_mode": bool(args.native_outcome_mode and native),
        "native_nce_weight": float(args.native_nce_weight),
        "native_keep_weight": float(args.native_keep_weight),
        "native_keep_margin": float(args.native_keep_margin),
        "native_keep_loose_weight": float(args.native_keep_loose_weight),
        "native_keep_loose_radius_px": float(args.native_keep_loose_radius_px),
        "native_keep_loose_margin": float(args.native_keep_loose_margin),
        "native_swap_weight": float(args.native_swap_weight),
        "native_swap_margin": float(args.native_swap_margin),
        "native_miss_weight": float(args.native_miss_weight),
        "native_miss_margin": float(args.native_miss_margin),
        "native_reject_weight": float(args.native_reject_weight),
        "native_reject_threshold": float(args.native_reject_threshold),
        "native_attractor_weight": float(args.native_attractor_weight),
        "native_attractor_margin": float(args.native_attractor_margin),
        "native_global_attractor_weight": (
            float(args.native_global_attractor_weight)
            if global_attractor_enabled
            else 0.0
        ),
        "native_global_attractor_scores": global_attractor_scores,
        "native_rank_budget_mode": bool(args.native_rank_budget_mode and native),
        "native_rank_temperature": float(args.native_rank_temperature),
        "native_rank_margins": (
            float(args.native_rank_margin_at1),
            float(args.native_rank_margin_at4),
            float(args.native_rank_margin_at8),
            float(args.native_rank_margin_at32),
        ),
        "native_rank_top1_weight": float(args.native_rank_top1_weight),
        "native_rank_keep_weight": float(args.native_rank_keep_weight),
        "native_rank_band_proportions": (
            float(args.native_rank_band_rank1),
            float(args.native_rank_band_rank2_4),
            float(args.native_rank_band_rank5_32),
            float(args.native_rank_band_rank33_plus),
        ),
        "native_rank_landmark_opportunities": (
            rank_landmark_opportunities
            if bool(args.native_rank_landmark_balance)
            else None
        ),
        "native_rank_reference_bank_features": rank_reference_bank_features,
        "native_rank_reference_clean_weight": float(
            args.native_rank_reference_clean_weight
        ),
        "native_rank_reference_clean_margin": float(
            args.native_rank_reference_clean_margin
        ),
    }


@torch.no_grad()
def _native_geometry_support_mask(
    features,
    train_names,
    cache,
    base_bank_xyz,
    current_xyz,
    args,
    *,
    visibility_cache=None,
):
    """Require native GT-clean top-1 evidence from distinct support cameras.

    This pre-pass is deliberately descriptor-fixed and geometry-gated.  A
    landmark receives at most one support count per camera, so repeated
    keypoints in a single image cannot unlock a BA update on their own.
    """
    bank_count = int(features.shape[0])
    counts = torch.zeros(
        bank_count,
        dtype=torch.int32,
        device=features.device,
    )
    clean_observations = 0
    supported_views = 0
    support_observations = int(args.geometry_association_support_observations)
    if support_observations <= 0:
        support_observations = int(args.max_observations)
    for name in tqdm(train_names, desc="Native BA support qualification"):
        observations, _ = _primary_observations(
            cache[name],
            base_bank_xyz,
            args,
            max_observations=support_observations,
            bank_visibility_mask=(
                None if visibility_cache is None else visibility_cache[name]
            ),
            prediction_bank_xyz=current_xyz,
        )
        association = native_association_matches(
            current_xyz,
            features,
            observations,
            max_reprojection_error_px=args.geometry_association_max_reprojection_px,
            min_score_margin=args.geometry_association_min_margin,
            depth_abs_tolerance=args.geometry_association_depth_abs_tolerance,
            depth_rel_tolerance=args.geometry_association_depth_rel_tolerance,
            alpha_threshold=args.alpha_threshold,
        )
        clean_indices = association.top1_indices[association.clean]
        clean_observations += int(clean_indices.numel())
        if clean_indices.numel() == 0:
            continue
        unique_indices = torch.unique(clean_indices)
        counts[unique_indices] += 1
        supported_views += 1
    minimum = max(int(args.geometry_association_min_support_views), 1)
    eligible = counts >= minimum
    observed = counts > 0
    diagnostics = {
        "native_geometry_support_min_views": minimum,
        "native_geometry_support_observations_per_view": support_observations,
        "native_geometry_support_train_views": int(len(train_names)),
        "native_geometry_support_views_with_clean_match": int(supported_views),
        "native_geometry_support_clean_observations": int(clean_observations),
        "native_geometry_support_observed_landmarks": int(observed.sum().item()),
        "native_geometry_support_eligible_landmarks": int(eligible.sum().item()),
        "native_geometry_support_eligible_ratio": float(
            eligible.float().mean().item() if bank_count else 0.0
        ),
        "native_geometry_support_count_mean": float(counts.float().mean().item()),
        "native_geometry_support_count_p95": float(
            torch.quantile(counts.float(), 0.95).item() if bank_count else 0.0
        ),
        "native_geometry_support_count_max": int(counts.max().item()) if bank_count else 0,
    }
    return eligible, counts, diagnostics


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


def _ulf_parity_project_pixels(xyz, K, pose_w2c, width, height, *, rounding):
    """Project surfel centers with the original ULF-Loc KCS/GWFF convention."""
    xyz = torch.as_tensor(xyz, device=K.device, dtype=K.dtype)
    pose_w2c = torch.as_tensor(pose_w2c, device=K.device, dtype=K.dtype)
    homogeneous = torch.cat(
        [xyz, torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)],
        dim=1,
    )
    camera_xyz = (pose_w2c @ homogeneous.T)[:3].T
    depth = camera_xyz[:, 2]
    physical_uv = torch.empty((xyz.shape[0], 2), device=xyz.device, dtype=xyz.dtype)
    physical_uv[:, 0] = K[0, 0] * camera_xyz[:, 0] / depth + K[0, 2]
    physical_uv[:, 1] = K[1, 1] * camera_xyz[:, 1] / depth + K[1, 2]
    finite = torch.isfinite(physical_uv).all(dim=1) & torch.isfinite(depth)
    safe_uv = torch.where(finite[:, None], physical_uv, torch.zeros_like(physical_uv))
    if rounding == "trunc":
        pixel_uv = torch.trunc(safe_uv)
    elif rounding == "round":
        pixel_uv = torch.round(safe_uv)
    else:
        raise ValueError(f"Unsupported ULF parity projection rounding: {rounding}")
    in_image = (
        finite
        & (pixel_uv[:, 0] >= 0)
        & (pixel_uv[:, 0] < int(width))
        & (pixel_uv[:, 1] >= 0)
        & (pixel_uv[:, 1] < int(height))
    )
    return pixel_uv, in_image


def _ulf_parity_nearest_keypoint_squared_distance(projected_uv, keypoints):
    """Use FAISS IndexFlatL2, matching ULF-Loc's KCS distance semantics."""
    projected_uv = torch.as_tensor(projected_uv, dtype=torch.float32)
    keypoints = torch.as_tensor(
        keypoints, device=projected_uv.device, dtype=torch.float32
    )
    if projected_uv.numel() == 0:
        return projected_uv.new_zeros((0,))
    if keypoints.numel() == 0:
        return projected_uv.new_full((projected_uv.shape[0],), torch.inf)
    try:
        import faiss

        index = faiss.IndexFlatL2(2)
        index.add(keypoints.detach().cpu().contiguous().numpy())
        distances, _ = index.search(projected_uv.detach().cpu().contiguous().numpy(), 1)
        return torch.from_numpy(distances[:, 0]).to(device=projected_uv.device)
    except ImportError:
        return nearest_keypoint_distance(projected_uv, keypoints).square()


def _build_ulf_parity_consensus_landmark_indices(
    gaussians,
    cameras,
    masks,
    feature_extractor,
    args,
):
    """Strict raw-resolution KCS matching ULF-Loc's random-kNN procedure.

    This intentionally remains separate from ``ulf_consensus``.  The latter
    is a LaFGS extension with contribution-aware visibility and coverage
    balancing; it must not be labelled a ULF-parity bootstrap.
    """
    if str(feature_extractor.feature_type) != "sp":
        raise ValueError("ULF parity KCS requires SuperPoint")
    if int(args.longest_edge) > 0:
        raise ValueError(
            "ULF parity KCS requires --longest_edge 0 so support images stay native"
        )
    xyz = gaussians.get_xyz.detach().float()
    requested_budget = int(args.scaffold_budget)
    if requested_budget <= 0 or requested_budget > int(xyz.shape[0]):
        raise ValueError(
            "ULF parity KCS requires a positive scaffold budget no larger than "
            "the primitive count"
        )
    votes = torch.zeros(xyz.shape[0], dtype=torch.int32, device=xyz.device)
    kcs_mask_policy = _resolve_ulf_parity_kcs_mask_policy(
        args.ulf_parity_kcs_mask_policy
    )
    post_detection_mask_filter = kcs_mask_policy == "deployment_post_filter"
    cameras = _subsample_ulf_support_cameras(
        cameras,
        args.ulf_consensus_max_views,
        args.ulf_support_view_sampling,
    )
    distance_threshold_sq = float(args.ulf_consensus_radius_px)
    view_records = []
    for camera in tqdm(cameras, desc="ULF-parity keypoint-consensus sampling"):
        image, valid_mask = _ulf_parity_feature_input(camera, masks)
        height, width = image.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError(
                "ULF parity requires native image dimensions divisible by SuperPoint stride 8; "
                f"got {(height, width)} for {_camera_cache_key(camera)}"
            )
        sparse = feature_extractor.detectAndCompute(
            image[None], top_k=args.ulf_consensus_keypoints
        )[0]
        keypoints = sparse["keypoints"]
        if post_detection_mask_filter:
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
        pixel_uv, projected = _ulf_parity_project_pixels(
            xyz,
            K,
            camera.world_view_transform.transpose(0, 1).cuda(),
            width,
            height,
            rounding="trunc",
        )
        visible = projected & render_visible
        if post_detection_mask_filter:
            visible &= sample_mask_at_grid_uv(valid_mask, pixel_uv)
        visible_indices = torch.nonzero(visible, as_tuple=False).reshape(-1)
        matched_count = 0
        if visible_indices.numel() > 0 and keypoints.numel() > 0:
            squared_distance = _ulf_parity_nearest_keypoint_squared_distance(
                pixel_uv[visible_indices], keypoints
            )
            # ULF-Loc compares FAISS squared L2 output directly against
            # ``thre_dis``.  The formal parity configuration fixes it to 1.
            matched = squared_distance <= distance_threshold_sq
            if bool(matched.any().item()):
                votes[visible_indices[matched]] += 1
                matched_count = int(matched.sum().item())
        view_records.append(
            {
                "image_name": _camera_cache_key(camera),
                "processed_image_hw": [int(height), int(width)],
                "sparse_keypoints": int(keypoints.shape[0]),
                "input_valid_fraction": float(valid_mask.float().mean().item()),
                "contribution_visible_primitives": int(render_visible.sum().item()),
                "projected_visible_primitives": int(visible_indices.numel()),
                "consensus_votes": matched_count,
            }
        )

    selected = ulf_random_knn_vote_sample(
        xyz,
        requested_budget,
        votes,
        k=args.ulf_consensus_knn,
        seed=args.scaffold_seed,
    )
    if int(selected.numel()) != requested_budget:
        raise RuntimeError(
            "ULF parity random-kNN sampling produced fewer unique landmarks than "
            f"requested: requested={requested_budget} selected={selected.numel()}"
        )
    selected_votes = votes[selected]
    diagnostics = {
        "mode": "ulf_parity_kcs_random_knn_v1",
        "strict_ulf_parity": kcs_mask_policy == "rgb_only",
        "budget": int(selected.numel()),
        "requested_budget": requested_budget,
        "eligible_primitives": int(xyz.shape[0]),
        "selected_vote_mean": float(selected_votes.float().mean().item()),
        "selected_vote_max": int(selected_votes.max().item()),
        "selected_with_nonzero_vote": int((selected_votes > 0).sum().item()),
        "consensus_sparse_keypoints": int(args.ulf_consensus_keypoints),
        "consensus_distance_threshold_squared_px": distance_threshold_sq,
        "consensus_knn": int(args.ulf_consensus_knn),
        "consensus_random_seed": int(args.scaffold_seed),
        "consensus_view_count": int(len(cameras)),
        "support_view_sampling": str(args.ulf_support_view_sampling),
        "input_resolution": "native_camera_rgb_no_resize",
        "input_mask_policy": "rgb_only_object_and_sky_and_distortion_v1",
        "kcs_mask_policy": kcs_mask_policy,
        "post_detection_mask_filter": post_detection_mask_filter,
        "primitive_opacity_filter": False,
        "visibility": "2dgs_raster_contribution_gradient",
        "projection_rounding": "truncate_toward_zero",
        "selection": "uniform_random_seed_then_3d_knn_highest_vote_unique",
        "views": view_records,
    }
    return selected.detach().cpu(), diagnostics


def _sample_ulf_parity_dense_features(
    dense_features,
    pixel_uv,
    image_hw,
    *,
    channel_chunk,
):
    """Exactly sample ULF's native stride-8 descriptor map in channel chunks."""
    height, width = (int(image_hw[0]), int(image_hw[1]))
    pixel_uv = torch.as_tensor(
        pixel_uv, device=dense_features.device, dtype=dense_features.dtype
    ).reshape(-1, 2)
    if pixel_uv.numel() == 0:
        return dense_features.new_zeros((0, dense_features.shape[1]))
    # ULF samples its stride-8 map directly with physical pixel centers.  Do
    # not upsample first: interpolation uses edge replication at the border,
    # whereas the reference grid_sample uses zero padding there.
    grid = pixel_uv.clone()
    grid[:, 0] = 2.0 * (grid[:, 0] + 0.5) / float(width) - 1.0
    grid[:, 1] = 2.0 * (grid[:, 1] + 0.5) / float(height) - 1.0
    grid = grid.view(1, -1, 1, 2)
    channel_chunk = max(1, int(channel_chunk))
    sampled_parts = []
    for start in range(0, dense_features.shape[1], channel_chunk):
        sampled_parts.append(
            F.grid_sample(
                dense_features[:, start : start + channel_chunk],
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[0, :, :, 0].T
        )
    return torch.cat(sampled_parts, dim=1)


def _build_ulf_parity_geometry_features(
    cameras,
    gaussians,
    landmark_indices,
    masks,
    feature_extractor,
    fallback_features,
    args,
):
    """Strict native-resolution ULF-Loc Geometry-Weighted Feature Fusion."""
    if int(args.longest_edge) > 0:
        raise ValueError(
            "ULF parity GWFF requires --longest_edge 0 so support images stay native"
        )
    if float(args.ulf_fusion_min_cosine) != 0.0:
        raise ValueError(
            "ULF parity GWFF does not support post-hoc cosine trimming; use 0"
        )
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
    cameras = _subsample_ulf_support_cameras(
        cameras,
        args.ulf_fusion_max_views,
        args.ulf_support_view_sampling,
    )
    sampled_weight_sum = 0.0
    sampled_weight_count = 0
    view_records = []
    for camera in tqdm(cameras, desc="ULF-parity geometry-weighted feature fusion"):
        image, valid_mask = _ulf_parity_fusion_input(camera, masks)
        height, width = image.shape[-2:]
        if height % 8 or width % 8:
            raise ValueError(
                "ULF parity requires native image dimensions divisible by SuperPoint stride 8; "
                f"got {(height, width)} for {_camera_cache_key(camera)}"
            )
        dense_features, _ = feature_extractor.detectAndComputeDense(image[None])
        K = make_intrinsics_from_fov(
            camera.FoVx,
            camera.FoVy,
            width,
            height,
            device=bank_xyz.device,
            dtype=bank_xyz.dtype,
        )
        pixel_uv, projected = _ulf_parity_project_pixels(
            bank_xyz,
            K,
            camera.world_view_transform.transpose(0, 1).cuda(),
            width,
            height,
            rounding="round",
        )
        valid = projected & sample_mask_at_grid_uv(valid_mask, pixel_uv)
        compact_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        if compact_indices.numel() == 0:
            view_records.append(
                {
                    "image_name": _camera_cache_key(camera),
                    "processed_image_hw": [int(height), int(width)],
                    "valid_projected_landmarks": 0,
                }
            )
            continue
        sampled = _sample_ulf_parity_dense_features(
            dense_features,
            pixel_uv[compact_indices],
            (height, width),
            channel_chunk=args.ulf_parity_fusion_channel_chunk,
        )
        weights = geometry_view_weights(
            bank_xyz[compact_indices],
            normals[compact_indices],
            camera.camera_center.cuda(),
        ).float()
        useful = weights > 0.0
        compact_indices = compact_indices[useful]
        sampled = sampled[useful]
        weights = weights[useful]
        if compact_indices.numel() > 0:
            feature_sum.index_add_(0, compact_indices, sampled.float() * weights[:, None])
            weight_sum.index_add_(0, compact_indices, weights)
            observation_count.index_add_(
                0,
                compact_indices,
                torch.ones_like(compact_indices, dtype=observation_count.dtype),
            )
            sampled_weight_sum += float(weights.sum().item())
            sampled_weight_count += int(weights.numel())
        view_records.append(
            {
                "image_name": _camera_cache_key(camera),
                "processed_image_hw": [int(height), int(width)],
                "valid_projected_landmarks": int(valid.sum().item()),
                "geometry_weighted_samples": int(compact_indices.numel()),
            }
        )
        del dense_features, sampled

    observed = weight_sum > 1e-8
    # ULF-Loc leaves unobserved rows at zero.  Do not backfill them from a
    # rendered feature field in this parity mode.
    result = torch.zeros_like(fallback_features.float())
    if bool(observed.any().item()):
        result[observed] = F.normalize(
            feature_sum[observed] / weight_sum[observed, None], dim=-1
        )
    diagnostics = {
        "initialization_mode": "ulf_parity_geometry_weighted_fusion_v1",
        "strict_ulf_parity": True,
        "observed_landmarks": int(observed.sum().item()),
        "unobserved_landmarks": int((~observed).sum().item()),
        "observation_count_mean": float(observation_count.float().mean().item()),
        "observation_count_median": float(observation_count.float().median().item()),
        "observation_count_max": int(observation_count.max().item()),
        "geometry_weight_mean": sampled_weight_sum / max(sampled_weight_count, 1),
        "geometry_weighted_samples": int(sampled_weight_count),
        "fusion_view_count": int(len(cameras)),
        "support_view_sampling": str(args.ulf_support_view_sampling),
        "input_resolution": "native_camera_rgb_no_resize",
        "input_mask_policy": "raw_rgb_encoder_validity_mask_only_v1",
        "visibility": "projection_and_scene_mask_no_raster",
        "projection_rounding": "round_to_nearest_pixel",
        "dense_sampling": "direct_native_stride8_grid_sample_physical_pixel_center",
        "dense_channel_chunk": int(args.ulf_parity_fusion_channel_chunk),
        "unobserved_feature_policy": "zero_vector",
        "views": view_records,
    }
    return result, observation_count, diagnostics


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
        voxel_size = _automatic_ulf_voxel_size(
            xyz[candidate_eligible],
            requested_budget,
            extent_quantile=args.ulf_consensus_extent_quantile,
        )
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
        "voxel_extent_quantile": float(args.ulf_consensus_extent_quantile),
        "max_per_voxel": int(args.ulf_consensus_max_per_voxel),
        "visibility": "2dgs_raster_contribution_gradient",
        "visibility_resolution": "resized_feature_input_resolution",
        "query_feature_contract": _QUERY_FEATURE_CONTRACT_NATIVE,
        "coordinate_convention": "grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "views": view_records,
    }
    return selected.detach().cpu(), diagnostics


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
    if int(base_eligible.sum().item()) < int(args.scaffold_budget):
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
    fallback_to_non_consensus = False
    consensus_count = int(consensus_eligible.sum().item())
    if consensus_count < requested_budget:
        if not allow_fallback:
            raise RuntimeError(
                "Robust KCS gates produced too few consensus landmarks: "
                f"eligible={consensus_count} budget={requested_budget}. "
                "Relax a named gate explicitly or pass --ulf_consensus_allow_nonconsensus_fallback."
            )
        fallback_to_non_consensus = True
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
            requested_budget,
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
            requested_budget,
            vote_score,
            voxel_size=voxel_size,
            max_per_voxel=args.ulf_consensus_max_per_voxel,
            eligible=consensus_eligible,
            allow_overflow=True,
        )
    if selected.numel() != requested_budget:
        raise RuntimeError(
            "Robust ULF consensus scaffold could not satisfy the requested budget: "
            f"requested={requested_budget} selected={selected.numel()}"
        )
    selected_votes = consensus_votes[selected]
    selected_rates = consensus_rate[selected]
    diagnostics = {
        "mode": "ulf_robust_keypoint_consensus_v1",
        "strict_ulf_parity": False,
        "budget": int(selected.numel()),
        "requested_budget": requested_budget,
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

    reference_mode = str(args.ulf_fusion_reference_mode)
    reference = prototype
    reference_observed = prototype_observed
    medoid_observed = torch.zeros_like(prototype_observed)
    medoid_scores = None
    if reference_mode == "weighted_cosine_medoid":
        # The first-pass geometry-weighted mean is sufficient to score the
        # exact weighted cosine medoid in a second stream over observations.
        medoid_scores = torch.full(
            (bank_count,), -torch.inf, device=bank_xyz.device, dtype=torch.float32
        )
        medoid_features = prototype.clone()

        def accumulate_medoid(indices, sampled, _weights, _view_bin):
            update_weighted_cosine_medoid_state(
                medoid_scores,
                medoid_features,
                indices,
                sampled,
                prototype,
            )

        for_each_observation("Robust ULF GWFF weighted cosine medoid", accumulate_medoid)
        medoid_observed = torch.isfinite(medoid_scores)
        reference = prototype.clone()
        if bool(medoid_observed.any().item()):
            reference[medoid_observed] = F.normalize(
                medoid_features[medoid_observed], dim=-1
            )
        reference_observed = medoid_observed

    trim_fraction = float(args.ulf_fusion_descriptor_trim_fraction)
    descriptor_min_cosine = float(args.ulf_fusion_descriptor_min_cosine)
    adaptive_trim = bool(args.ulf_fusion_adaptive_trim)
    needs_trim = (
        trim_fraction > 0.0 or descriptor_min_cosine > -1.0 or adaptive_trim
    )
    thresholds = torch.full((bank_count,), -1.0, device=bank_xyz.device)
    posttrim_count = pretrim_count.clone()
    result = reference
    adaptive_trim_fractions = torch.full(
        (bank_count,), trim_fraction, dtype=torch.float32, device=bank_xyz.device
    )
    adaptive_tail_rates = None
    adaptive_tail_thresholds = None
    adaptive_medians = None
    adaptive_mads = None
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
        if adaptive_trim:
            (
                adaptive_trim_fractions,
                adaptive_tail_rates,
                adaptive_tail_thresholds,
                adaptive_medians,
                adaptive_mads,
            ) = adaptive_cosine_histogram_trim_schedule(
                histogram,
                tail_cosine=float(args.ulf_fusion_adaptive_trim_tail_cosine),
                min_fraction=float(args.ulf_fusion_adaptive_trim_min_fraction),
                max_fraction=float(args.ulf_fusion_adaptive_trim_max_fraction),
                min_observations=int(args.ulf_fusion_adaptive_trim_min_observations),
                mode=str(args.ulf_fusion_adaptive_trim_mode),
                mad_scale=float(args.ulf_fusion_adaptive_trim_mad_scale),
            )
        thresholds = cosine_histogram_trim_thresholds(
            histogram, adaptive_trim_fractions
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
        "adaptive_descriptor_trim_enabled": adaptive_trim,
        "adaptive_descriptor_trim_min_fraction": float(
            args.ulf_fusion_adaptive_trim_min_fraction
        ),
        "adaptive_descriptor_trim_max_fraction": float(
            args.ulf_fusion_adaptive_trim_max_fraction
        ),
        "adaptive_descriptor_trim_tail_cosine": float(
            args.ulf_fusion_adaptive_trim_tail_cosine
        ),
        "adaptive_descriptor_trim_mode": str(args.ulf_fusion_adaptive_trim_mode),
        "adaptive_descriptor_trim_mad_scale": float(
            args.ulf_fusion_adaptive_trim_mad_scale
        ),
        "adaptive_descriptor_trim_min_observations": int(
            args.ulf_fusion_adaptive_trim_min_observations
        ),
        "adaptive_descriptor_trim_fraction_mean_observed": (
            float(adaptive_trim_fractions[reference_observed].mean().item())
            if bool(reference_observed.any().item())
            else 0.0
        ),
        "adaptive_descriptor_trim_fraction_p95_observed": (
            float(torch.quantile(adaptive_trim_fractions[reference_observed], 0.95).item())
            if bool(reference_observed.any().item())
            else 0.0
        ),
        "adaptive_descriptor_trim_tail_rate_mean_observed": (
            float(adaptive_tail_rates[reference_observed].mean().item())
            if adaptive_tail_rates is not None and bool(reference_observed.any().item())
            else None
        ),
        "adaptive_descriptor_trim_threshold_mean_observed": (
            float(adaptive_tail_thresholds[reference_observed].mean().item())
            if adaptive_tail_thresholds is not None
            and bool(reference_observed.any().item())
            else None
        ),
        "adaptive_descriptor_trim_median_mean_observed": (
            float(adaptive_medians[reference_observed].mean().item())
            if adaptive_medians is not None and bool(reference_observed.any().item())
            else None
        ),
        "adaptive_descriptor_trim_mad_mean_observed": (
            float(adaptive_mads[reference_observed].mean().item())
            if adaptive_mads is not None and bool(reference_observed.any().item())
            else None
        ),
        "descriptor_min_cosine": descriptor_min_cosine,
        "descriptor_trim_histogram_bins": int(args.ulf_fusion_trim_histogram_bins),
        "fusion_reference_mode": reference_mode,
        "weighted_cosine_medoid_landmarks": int(medoid_observed.sum().item()),
        "weighted_cosine_medoid_score_mean": (
            float(medoid_scores[medoid_observed].mean().item())
            if medoid_scores is not None and bool(medoid_observed.any().item())
            else None
        ),
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
        metadata.setdefault("mode", f"{args.scaffold_mode}_cached")
        metadata.setdefault("budget", int(indices.numel()))
        return indices, path, metadata

    if args.scaffold_mode in {
        "ulf_consensus",
        "ulf_parity",
        "ulf_robust_consensus",
    }:
        if cameras is None or feature_extractor is None:
            raise ValueError("ULF consensus scaffold requires cameras and a feature extractor")
        if args.scaffold_mode == "ulf_parity":
            indices, diagnostics = _build_ulf_parity_consensus_landmark_indices(
                gaussians,
                cameras,
                masks,
                feature_extractor,
                args,
            )
        elif args.scaffold_mode == "ulf_robust_consensus":
            indices, diagnostics = _build_ulf_robust_consensus_landmark_indices(
                gaussians,
                cameras,
                masks,
                feature_extractor,
                args,
            )
        else:
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
    if torch.equal(state_indices, landmark_indices_cpu):
        if mvinit_count_valid:
            state["mvinit_observation_count"] = mvinit_observation_count
        state["_mvinit_observation_count_alignment_valid"] = mvinit_count_valid
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
    if mvinit_count_valid:
        aligned_mvinit_count = torch.zeros(
            landmark_indices.numel(), dtype=torch.long
        )
    if bool(matched.any().item()):
        source_position = order[positions[matched]]
        result[matched.to(device=result.device)] = F.normalize(
            features[source_position].to(device), dim=-1
        )
        if mvinit_count_valid:
            aligned_mvinit_count[matched] = mvinit_observation_count[
                source_position
            ]
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
    if mvinit_count_valid:
        state["mvinit_observation_count"] = aligned_mvinit_count
    state["_mvinit_observation_count_alignment_valid"] = mvinit_count_valid
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
def _collect_independent_geometry_teacher(
    features,
    query_names,
    cache,
    bank_xyz,
    args,
):
    """Build the explicit G0/G1 map-identity geometry-teacher controls."""
    teacher_mode = str(args.geometry_teacher_identity_mode)
    if teacher_mode not in {"map_top1", "gt_clean_map_top1"}:
        raise ValueError(
            "_collect_independent_geometry_teacher only supports map_top1 "
            "and gt_clean_map_top1"
        )
    normalized_features = F.normalize(features.detach(), dim=1)
    landmark_indices = []
    query_indices = []
    pixels = []
    confidences = []
    rendered_depths = []
    camera_intrinsics = []
    camera_poses = []
    query_support_offsets = [0]
    query_support_indices = []
    candidate_count = 0
    support_count = 0
    for query_index, name in enumerate(
        tqdm(query_names, desc="Independent descriptor-ray teacher")
    ):
        cached = cache[name]
        query = F.normalize(
            cached["native_descriptors"].cuda().float(), dim=1
        )
        keypoints = cached["native_keypoints"].cuda().float()
        K = cached["native_K"].cuda().float()
        pose_w2c = cached["pose_w2c"].cuda().float()
        height, width = map(int, cached["native_input_hw"])
        top_values, top_indices = torch.topk(
            query @ normalized_features.T, k=2, dim=1
        )
        top1 = top_indices[:, 0]
        margin = top_values[:, 0] - top_values[:, 1]
        association_valid = (
            (top_values[:, 0] >= float(args.geometry_teacher_min_similarity))
            & (margin >= float(args.geometry_teacher_min_margin))
        )
        detector_score = cached["native_scores"].cuda().float().clamp(0.0, 1.0)
        confidence = (
            torch.sigmoid(
                (top_values[:, 0] - float(args.geometry_teacher_min_similarity))
                / 0.05
            )
            * torch.sigmoid(
                (margin - float(args.geometry_teacher_min_margin)) / 0.02
            )
            * detector_score.clamp_min(0.05)
        )
        physical_keypoints = keypoints + float(PIXEL_CENTER_OFFSET)
        projected, _, projected_valid = project_landmarks_to_query(
            bank_xyz[top1],
            K,
            pose_w2c,
            height,
            width,
            pixel_center_offset=0.0,
        )
        clean = projected_valid & (
            torch.linalg.norm(projected - physical_keypoints, dim=1)
            <= float(args.positive_radius_px)
        )
        support = torch.unique(top1[clean]).detach().cpu()
        query_support_indices.append(support)
        support_count += int(support.numel())
        query_support_offsets.append(support_count)

        triangulation_valid = (
            association_valid & clean
            if teacher_mode == "gt_clean_map_top1"
            else association_valid
        )
        if bool(triangulation_valid.any()):
            selected = torch.nonzero(
                triangulation_valid, as_tuple=False
            ).reshape(-1)
            native_depth = cached["native_depth"].cuda().float()
            depth_xy = keypoints[selected].round().long()
            depth_xy[:, 0].clamp_(0, width - 1)
            depth_xy[:, 1].clamp_(0, height - 1)
            sampled_depth = native_depth[depth_xy[:, 1], depth_xy[:, 0]]
            landmark_indices.append(top1[selected].detach().cpu())
            query_indices.append(
                torch.full(
                    (selected.numel(),), query_index, dtype=torch.long
                )
            )
            pixels.append(physical_keypoints[selected].detach().cpu())
            confidences.append(confidence[selected].detach().cpu())
            rendered_depths.append(sampled_depth.detach().cpu())
            candidate_count += int(selected.numel())
        camera_intrinsics.append(cached["native_K"].float())
        camera_poses.append(cached["pose_w2c"].float())

    if not landmark_indices:
        raise RuntimeError(
            "Independent geometry teacher found no descriptor associations; "
            "relax its absolute similarity or margin threshold"
        )
    camera_intrinsics = torch.stack(camera_intrinsics)
    camera_poses = torch.stack(camera_poses)
    query_bins = camera_pose_bins(
        camera_poses,
        int(args.geometry_teacher_view_bins),
        direction_weight=float(args.geometry_teacher_view_direction_weight),
    )
    geometry = robust_triangulate_associations(
        landmark_count=int(bank_xyz.shape[0]),
        landmark_index=torch.cat(landmark_indices),
        query_index=torch.cat(query_indices),
        uv=torch.cat(pixels),
        confidence=torch.cat(confidences),
        camera_K=camera_intrinsics,
        pose_w2c=camera_poses,
        query_bin=query_bins,
        rendered_depth=torch.cat(rendered_depths),
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
    )
    support_indices = (
        torch.cat(query_support_indices)
        if query_support_indices
        else torch.zeros(0, dtype=torch.long)
    )
    statistics = {
        "query_support_offsets": torch.as_tensor(
            query_support_offsets, dtype=torch.long
        ),
        "query_support_indices": support_indices,
        "query_support_query_count": torch.as_tensor(
            len(query_names), dtype=torch.long
        ),
    }
    diagnostics = {
        "geometry_teacher_candidate_association_count": candidate_count,
        "geometry_teacher_query_support_edge_count": int(
            support_indices.numel()
        ),
        "geometry_teacher_triangulated_landmark_count": int(
            geometry["triangulated"].sum().item()
        ),
        "geometry_teacher_high_confidence_landmark_count": int(
            geometry["triangulation_high_confidence"].sum().item()
        ),
        "geometry_teacher_similarity_threshold": float(
            args.geometry_teacher_min_similarity
        ),
        "geometry_teacher_margin_threshold": float(
            args.geometry_teacher_min_margin
        ),
        "geometry_teacher_identity_mode": teacher_mode,
    }
    return statistics, geometry, diagnostics


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
    """Assign independent tracks using frozen 2DGS composition provenance."""
    track_count = int(track_geometry["triangulated_xyz"].shape[0])
    high_confidence = torch.as_tensor(
        track_geometry["triangulation_high_confidence"], dtype=torch.bool
    )
    track_candidates = [defaultdict(float) for _ in range(track_count)]
    track_candidate_views = [defaultdict(set) for _ in range(track_count)]
    observations_by_query = defaultdict(list)
    for observation, (track, query) in enumerate(
        zip(tracks["track_index"].tolist(), tracks["query_index"].tolist())
    ):
        if bool(high_confidence[track]):
            observations_by_query[query].append(observation)
    valid_observations = 0
    for query, observations in tqdm(
        sorted(observations_by_query.items()),
        desc="G3 frozen 2DGS provenance assignment",
    ):
        name = query_names[query]
        camera = cameras_by_name[name]
        cached = cache[name]
        height, width = map(int, cached["native_input_hw"])
        render_pkg = render_from_pose_gsplat(
            gaussians,
            cached["pose_w2c"].cuda().float(),
            camera.FoVx,
            camera.FoVy,
            width,
            height,
            bg_color=background,
            render_mode="RGB+ED",
            rgb_only=True,
            return_rgb_meta=True,
            rasterize_mode="antialiased",
        )
        observation_tensor = torch.as_tensor(
            observations, dtype=torch.long
        )
        local_keypoint_indices = tracks["keypoint_index"][observation_tensor]
        query_keypoints = (
            keypoints[query][local_keypoint_indices]
            - float(PIXEL_CENTER_OFFSET)
        ).cuda()
        local_ids, weights, valid = bank_splat_provenance_2dgs(
            query_keypoints,
            landmark_global_indices,
            render_pkg["rgb_meta"],
            rendered_depth=render_pkg.get("depth"),
            topk=args.geometry_teacher_provenance_topk,
            candidate_topk=max(
                int(args.geometry_teacher_provenance_topk) * 8, 32
            ),
            depth_abs_tolerance=(
                args.geometry_teacher_provenance_depth_abs_tolerance_m
            ),
            depth_rel_tolerance=(
                args.geometry_teacher_provenance_depth_rel_tolerance
            ),
        )
        for row, observation in enumerate(observations):
            if not bool(valid[row]):
                continue
            track = int(tracks["track_index"][observation])
            valid_observations += 1
            for landmark, weight in zip(
                local_ids[row].tolist(), weights[row].tolist()
            ):
                if weight <= 0.0:
                    continue
                track_candidates[track][landmark] += float(weight)
                track_candidate_views[track][landmark].add(query)
        del render_pkg, local_ids, weights, valid

    track_landmark = torch.full((track_count,), -1, dtype=torch.long)
    assignment_cost = torch.full(
        (track_count,), float("inf"), dtype=torch.float32
    )
    consensus_rate = torch.zeros(track_count, dtype=torch.float32)
    support_views = torch.zeros(track_count, dtype=torch.long)
    group_offsets = [0]
    group_landmarks = []
    group_costs = []
    group_rates = []
    group_support_views = []
    assigned = 0
    group_assigned_tracks = 0
    track_observation_counts = torch.bincount(
        torch.as_tensor(tracks["track_index"], dtype=torch.long),
        minlength=track_count,
    )
    for track in range(track_count):
        if not bool(high_confidence[track]):
            group_offsets.append(group_offsets[-1])
            continue
        candidates = track_candidates[track]
        if not candidates:
            group_offsets.append(group_offsets[-1])
            continue
        ordered_candidates = sorted(
            candidates.items(), key=lambda item: (-item[1], item[0])
        )
        landmark, mass = ordered_candidates[0]
        track_observations = int(track_observation_counts[track])
        rate = float(mass) / max(track_observations, 1)
        views = len(track_candidate_views[track][landmark])
        if (
            rate < float(args.geometry_teacher_provenance_min_consensus_rate)
            or views < int(args.geometry_teacher_provenance_min_views)
        ):
            group_offsets.append(group_offsets[-1])
            continue
        track_landmark[track] = int(landmark)
        consensus_rate[track] = rate
        support_views[track] = views
        assignment_cost[track] = 1.0 - min(rate, 1.0)
        assigned += 1

        accepted = []
        maximum_group = max(
            int(args.geometry_teacher_provenance_group_max_landmarks), 1
        )
        for candidate_landmark, candidate_mass in ordered_candidates:
            candidate_rate = float(candidate_mass) / max(
                track_observations, 1
            )
            candidate_views = len(
                track_candidate_views[track][candidate_landmark]
            )
            if (
                candidate_rate
                < float(
                    args.geometry_teacher_provenance_group_min_consensus_rate
                )
                or candidate_mass
                < float(
                    args.geometry_teacher_provenance_group_min_relative_mass
                )
                * float(mass)
                or candidate_views
                < int(args.geometry_teacher_provenance_min_views)
            ):
                continue
            accepted.append(
                (
                    int(candidate_landmark),
                    1.0 - min(candidate_rate, 1.0),
                    candidate_rate,
                    candidate_views,
                )
            )
            if len(accepted) >= maximum_group:
                break
        if not accepted:
            accepted.append((int(landmark), 1.0 - min(rate, 1.0), rate, views))
        group_landmarks.extend(item[0] for item in accepted)
        group_costs.extend(item[1] for item in accepted)
        group_rates.extend(item[2] for item in accepted)
        group_support_views.extend(item[3] for item in accepted)
        group_offsets.append(group_offsets[-1] + len(accepted))
        group_assigned_tracks += 1

    group_offsets = torch.as_tensor(group_offsets, dtype=torch.long)
    group_landmarks = torch.as_tensor(group_landmarks, dtype=torch.long)
    group_costs = torch.as_tensor(group_costs, dtype=torch.float32)
    group_rates = torch.as_tensor(group_rates, dtype=torch.float32)
    group_support_views = torch.as_tensor(
        group_support_views, dtype=torch.long
    )
    edge_tracks = torch.repeat_interleave(
        torch.arange(track_count, dtype=torch.long),
        group_offsets[1:] - group_offsets[:-1],
    )
    geometry, group_assignment = (
        transfer_triangulated_track_groups_to_landmarks(
            track_geometry,
            edge_track_index=edge_tracks,
            edge_landmark_index=group_landmarks,
            landmark_count=int(bank_xyz.shape[0]),
            edge_assignment_cost=group_costs,
        )
    )
    best_edge_all = group_assignment["landmark_best_edge_index"]
    selected = best_edge_all >= 0
    best_edge = best_edge_all[selected]
    best_track = edge_tracks[best_edge]
    assignment_distance = torch.full(
        (bank_xyz.shape[0],), float("inf"), dtype=torch.float32
    )
    assignment_distance[selected] = torch.linalg.norm(
        track_geometry["triangulated_xyz"][best_track]
        - bank_xyz.detach().cpu()[selected],
        dim=1,
    )
    geometry["track_assignment_distance_m"] = assignment_distance
    geometry["track_provenance_consensus_rate"] = torch.zeros(
        bank_xyz.shape[0], dtype=torch.float32
    )
    geometry["track_provenance_support_views"] = torch.zeros(
        bank_xyz.shape[0], dtype=torch.long
    )
    geometry["track_provenance_consensus_rate"][selected] = (
        group_rates[best_edge]
    )
    geometry["track_provenance_support_views"][selected] = (
        group_support_views[best_edge]
    )
    assignment = {
        "track_landmark_index": track_landmark,
        "track_assignment_cost": assignment_cost,
        "landmark_best_track_index": (
            group_assignment["landmark_best_track_index"]
        ),
        "track_landmark_offsets": group_offsets,
        "track_landmark_indices": group_landmarks,
        "track_landmark_costs": group_costs,
    }
    diagnostics = {
        "geometry_teacher_provenance_valid_observation_count": valid_observations,
        "geometry_teacher_provenance_assigned_track_count": assigned,
        "geometry_teacher_provenance_assigned_landmark_count": int(
            selected.sum().item()
        ),
        "geometry_teacher_provenance_group_assigned_track_count": (
            group_assigned_tracks
        ),
        "geometry_teacher_provenance_group_edge_count": int(
            group_landmarks.numel()
        ),
        "geometry_teacher_provenance_group_size_mean": (
            float(group_landmarks.numel()) / max(group_assigned_tracks, 1)
        ),
    }
    return geometry, assignment, diagnostics


@torch.no_grad()
def _collect_track_first_geometry_teacher(
    query_names,
    cache,
    bank_xyz,
    args,
    *,
    provenance_context=None,
):
    """Build G2 image-side tracks, triangulate them, then associate geometry."""
    descriptors = []
    keypoints = []
    detector_scores = []
    camera_intrinsics = []
    camera_poses = []
    depth_sources = []
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
    camera_intrinsics = torch.stack(camera_intrinsics)
    camera_poses = torch.stack(camera_poses)
    tracks, track_diagnostics = build_cycle_consistent_tracks(
        descriptors=descriptors,
        keypoints=keypoints,
        detector_scores=detector_scores,
        camera_K=camera_intrinsics,
        pose_w2c=camera_poses,
        pair_neighbors=args.geometry_teacher_track_pair_neighbors,
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
        local_geometry_filter=args.geometry_teacher_track_lgcv,
        local_geometry_neighbors=args.geometry_teacher_track_lgcv_neighbors,
        local_geometry_support_threshold=(
            args.geometry_teacher_track_lgcv_support_threshold
        ),
        local_geometry_angle_cosine=(
            args.geometry_teacher_track_lgcv_angle_cosine
        ),
        local_geometry_scale_threshold=(
            args.geometry_teacher_track_lgcv_scale_threshold
        ),
        local_geometry_scale_limit=(
            args.geometry_teacher_track_lgcv_scale_limit
        ),
        local_geometry_maximum_edge_px=(
            args.geometry_teacher_track_lgcv_maximum_edge_px
        ),
        local_geometry_minimum_matches=(
            args.geometry_teacher_track_lgcv_minimum_matches
        ),
        local_geometry_mode=args.geometry_teacher_track_lgcv_mode,
        local_geometry_confidence_floor=(
            args.geometry_teacher_track_lgcv_confidence_floor
        ),
        minimum_track_views=args.geometry_teacher_min_views,
        require_cycle=args.geometry_teacher_track_require_cycle,
        allow_chain_tracks=(
            args.geometry_teacher_track_allow_chain_tracks
        ),
        device="cuda",
    )
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
    rendered_depth_samples = []
    for query, keypoint in zip(
        observation_query.tolist(), observation_keypoint.tolist()
    ):
        depth_source = depth_sources[int(query)]
        if depth_source.ndim == 1:
            rendered_depth_samples.append(depth_source[int(keypoint)])
        else:
            keypoint_xy = (
                keypoints[int(query)][int(keypoint)]
                - float(PIXEL_CENTER_OFFSET)
            )
            x = min(
                max(int(round(float(keypoint_xy[0]))), 0),
                int(depth_source.shape[1]) - 1,
            )
            y = min(
                max(int(round(float(keypoint_xy[1]))), 0),
                int(depth_source.shape[0]) - 1,
            )
            rendered_depth_samples.append(depth_source[y, x])
    rendered_depth = torch.stack(rendered_depth_samples).float()
    query_bins = camera_pose_bins(
        camera_poses,
        int(args.geometry_teacher_view_bins),
        direction_weight=float(args.geometry_teacher_view_direction_weight),
    )
    track_geometry = robust_triangulate_associations(
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
    )
    track_geometry["track_confidence_level"] = tracks[
        "track_level"
    ].clone()
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
    return statistics, geometry, diagnostics


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
    identity_sources = []
    identity_predictions = []
    identity_weights = []
    benign_switch_count = torch.zeros(count, device=device)
    ambiguous_switch_count = torch.zeros(count, device=device)
    harmful_switch_count = torch.zeros(count, device=device)
    rescue_utility = torch.zeros(count, device=device)
    rescue_query_count = torch.zeros(count, device=device)
    target_correct_hit_count = torch.zeros(count, device=device)
    target_false_hit_count = torch.zeros(count, device=device)
    target_incoming_count = torch.zeros(count, device=device)
    recall_at_k_sum = {1: 0.0, 4: 0.0, 8: 0.0, 16: 0.0}
    recall_at_k_queries = 0
    for name in tqdm(query_names, desc="One-time landmark statistics"):
        observations, _ = _primary_observations(
            cache[name],
            bank_xyz if base_bank_xyz is None else base_bank_xyz,
            args,
            max_observations=args.statistics_observations,
            bank_visibility_mask=(
                None if visibility_cache is None else visibility_cache[name]
            ),
            prediction_bank_xyz=bank_xyz,
        )
        positive_offsets = observations.positive_offsets
        positive_indices = observations.positive_indices
        if positive_offsets is not None and positive_indices is not None:
            all_positive_counts = positive_offsets[1:] - positive_offsets[:-1]
            all_positive_rows = torch.repeat_interleave(
                torch.arange(
                    all_positive_counts.numel(),
                    device=device,
                ),
                all_positive_counts,
            )
            # A landmark that is the only, or one of very few, legal anchors
            # for a native keypoint is coverage reserve evidence. Split one
            # unit across the row's CSR positives so dense primitive clusters
            # cannot manufacture utility merely by duplicating anchors.
            scarce = (
                (all_positive_counts > 0)
                & (
                    all_positive_counts
                    <= int(getattr(args, "distill_rescue_max_positives", 4))
                )
            )
            if bool(scarce.any().item()):
                scarce_edges = scarce[all_positive_rows]
                scarce_landmarks = positive_indices[scarce_edges]
                scarce_rows = all_positive_rows[scarce_edges]
                edge_weight = torch.reciprocal(
                    all_positive_counts[scarce_rows].float()
                )
                rescue_utility.index_add_(
                    0,
                    scarce_landmarks,
                    edge_weight,
                )
                rescue_query_count.index_add_(
                    0,
                    torch.unique(scarce_landmarks),
                    torch.ones(
                        torch.unique(scarce_landmarks).numel(),
                        device=device,
                    ),
                )
        source = observations.source_indices
        source_valid = source >= 0
        if not bool(source_valid.all().item()):
            observations = _select_observation_rows(observations, source_valid)
            source = observations.source_indices
        if source.numel() == 0:
            continue
        query = F.normalize(observations.query_features, dim=1)
        scores = query @ normalized_features.T
        score_topk = min(max(int(args.statistics_hypothesis_topk), 2), count)
        top_values, top_indices = torch.topk(scores, score_topk, dim=1)
        top1 = top_indices[:, 0]
        positive_offsets = observations.positive_offsets
        positive_indices = observations.positive_indices
        if positive_offsets is not None and positive_indices is not None:
            positive_counts = positive_offsets[1:] - positive_offsets[:-1]
            positive_rows = torch.repeat_interleave(
                torch.arange(source.numel(), device=device),
                positive_counts,
            )
            positive_keys = positive_rows * count + positive_indices
            candidate_keys = (
                torch.arange(source.numel(), device=device)[:, None] * count
                + top_indices
            )
            candidate_positive = torch.isin(candidate_keys, positive_keys)
        else:
            candidate_positive = top_indices == source[:, None]
        top1_positive = candidate_positive[:, 0]
        if bool(top1_positive.any().item()):
            target_correct_hit_count.index_add_(
                0,
                top1[top1_positive],
                torch.ones_like(
                    top1[top1_positive],
                    dtype=target_correct_hit_count.dtype,
                ),
            )
        top1_distance = torch.linalg.norm(
            observations.bank_uv[top1] - observations.query_uv, dim=1
        )
        top1_projected = observations.bank_projected[top1]
        ambiguous_switch = (
            ~top1_positive
            & top1_projected
            & (top1_distance < float(args.negative_radius_px))
        )
        harmful_switch = ~top1_positive & ~ambiguous_switch
        target_incoming_count.index_add_(
            0,
            top1,
            torch.ones_like(top1, dtype=target_incoming_count.dtype),
        )
        target_false_hit_count.index_add_(
            0,
            top1[harmful_switch],
            torch.ones_like(
                top1[harmful_switch], dtype=target_false_hit_count.dtype
            ),
        )
        clean = observations.bank_visible[top1] & (
            top1_distance <= float(args.positive_radius_px)
        )
        if (
            positive_offsets is not None
            and positive_indices is not None
            and observations.positive_reprojection_errors is not None
        ):
            responsibility_rows, responsibility = (
                _csr_positive_responsibilities(
                    positive_offsets,
                    observations.positive_reprojection_errors,
                    sigma_px=getattr(
                        args, "statistics_responsibility_sigma_px", 1.0
                    ),
                )
            )
            responsibility_landmarks = positive_indices
        else:
            responsibility_rows = torch.arange(source.numel(), device=device)
            responsibility_landmarks = source
            responsibility = torch.ones(
                source.numel(), device=device, dtype=torch.float32
            )
        edge_top1 = top1[responsibility_rows]
        edge_same_identity = edge_top1 == responsibility_landmarks
        edge_benign_switch = (
            top1_positive[responsibility_rows] & ~edge_same_identity
        )
        edge_ambiguous_switch = ambiguous_switch[responsibility_rows]
        edge_harmful_switch = harmful_switch[responsibility_rows]
        edge_source_score = scores[
            responsibility_rows, responsibility_landmarks
        ]
        edge_competitor = torch.where(
            edge_same_identity,
            top_values[responsibility_rows, 1],
            top_values[responsibility_rows, 0],
        )
        edge_margin = edge_source_score - edge_competitor
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
        finite_top1_distance = top1_distance[
            top1_projected & torch.isfinite(top1_distance)
        ]
        observation_count.index_add_(
            0, responsibility_landmarks, responsibility
        )
        correct_count.index_add_(
            0,
            responsibility_landmarks,
            responsibility * top1_positive[responsibility_rows].float(),
        )
        source_top1_count.index_add_(
            0,
            responsibility_landmarks,
            responsibility * edge_same_identity.float(),
        )
        benign_switch_count.index_add_(
            0,
            responsibility_landmarks,
            responsibility * edge_benign_switch.float(),
        )
        ambiguous_switch_count.index_add_(
            0,
            responsibility_landmarks,
            responsibility * edge_ambiguous_switch.float(),
        )
        harmful_switch_count.index_add_(
            0,
            responsibility_landmarks,
            responsibility * edge_harmful_switch.float(),
        )
        identity_sources.append(responsibility_landmarks.detach().cpu())
        identity_predictions.append(edge_top1.detach().cpu())
        identity_weights.append(responsibility.detach().cpu())
        for recall_k in recall_at_k_sum:
            width_k = min(recall_k, candidate_positive.shape[1])
            recall_at_k_sum[recall_k] += float(
                candidate_positive[:, :width_k].any(dim=1).float().mean().item()
            )
        recall_at_k_queries += 1
        margin_sum.index_add_(
            0, responsibility_landmarks, responsibility * edge_margin
        )
        entropy_sum.index_add_(
            0,
            responsibility_landmarks,
            responsibility * entropy[responsibility_rows],
        )
        reprojection_sum.index_add_(
            0,
            responsibility_landmarks,
            responsibility * local_error[responsibility_rows],
        )
        height, width = observations.query_feature_map.shape[-2:]
        normalized_uv = observations.query_uv.clone()
        normalized_uv[:, 0] /= max(float(width - 1), 1.0)
        normalized_uv[:, 1] /= max(float(height - 1), 1.0)
        uv_sum.index_add_(
            0,
            responsibility_landmarks,
            responsibility[:, None] * normalized_uv[responsibility_rows],
        )
        depth_sum.index_add_(
            0,
            responsibility_landmarks,
            responsibility
            * observations.bank_depth[responsibility_landmarks],
        )
        if responsibility_landmarks.numel() >= 6:
            information = compute_pose_information(
                bank_xyz[responsibility_landmarks],
                observations.K,
                observations.pose_w2c,
                weights=(
                    responsibility
                    * top1_positive[responsibility_rows].float().clamp_min(0.05)
                ),
                translation_scale=args.pose_translation_scale_m,
                rotation_scale=float(
                    torch.deg2rad(torch.tensor(args.pose_rotation_scale_deg)).item()
                ),
            )
            translation_fim_sum.index_add_(
                0,
                responsibility_landmarks,
                information.translation_scores,
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
                "source_margin_mean": float(
                    (responsibility * edge_margin).sum().item()
                    / responsibility.sum().clamp_min(1e-8).item()
                ),
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
    identity_distinct_count = torch.zeros(count, device=device)
    identity_dominant_count = torch.zeros(count, device=device)
    if identity_sources:
        all_sources = torch.cat(identity_sources).long()
        all_predictions = torch.cat(identity_predictions).long()
        all_identity_weights = torch.cat(identity_weights).float()
        pair_keys = all_sources * count + all_predictions
        unique_keys, inverse = torch.unique(pair_keys, return_inverse=True)
        pair_counts = torch.zeros(
            unique_keys.numel(), dtype=torch.float32
        )
        pair_counts.index_add_(0, inverse, all_identity_weights)
        pair_sources = torch.div(unique_keys, count, rounding_mode="floor")
        distinct_cpu = torch.zeros(count, dtype=torch.float32)
        distinct_cpu.index_add_(
            0, pair_sources, torch.ones_like(pair_sources, dtype=torch.float32)
        )
        dominant_cpu = torch.zeros(count, dtype=torch.float32)
        dominant_cpu.scatter_reduce_(
            0,
            pair_sources,
            pair_counts,
            reduce="amax",
            include_self=True,
        )
        identity_distinct_count = distinct_cpu.to(device)
        identity_dominant_count = dominant_cpu.to(device)
    identity_dominance = identity_dominant_count / observation_count.clamp_min(1.0)
    identity_switch_rate = (1.0 - identity_dominance).clamp(0.0, 1.0)
    benign_switch_rate = benign_switch_count / observation_count.clamp_min(1.0)
    ambiguous_switch_rate = (
        ambiguous_switch_count / observation_count.clamp_min(1.0)
    )
    harmful_switch_rate = harmful_switch_count / observation_count.clamp_min(1.0)
    statistics = {
        "observation_count": observation_count,
        "correct_count": correct_count,
        "source_top1_count": source_top1_count,
        "source_identity_rate": (source_top1_count + 1.0)
        / (observation_count + 2.0),
        "cross_view_top1_identity_distinct_count": identity_distinct_count,
        "cross_view_top1_identity_dominance": identity_dominance,
        "cross_view_top1_identity_switch_rate": identity_switch_rate,
        "cross_view_top1_benign_positive_switch_rate": benign_switch_rate,
        "cross_view_top1_ambiguous_switch_rate": ambiguous_switch_rate,
        "cross_view_top1_harmful_switch_rate": harmful_switch_rate,
        "rescue_utility": rescue_utility,
        "rescue_query_count": rescue_query_count,
        "target_correct_hit_count": target_correct_hit_count,
        "target_false_hit_count": target_false_hit_count,
        "target_incoming_count": target_incoming_count,
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
            "rescue_landmark_count": int((rescue_utility > 0).sum().item()),
            "rescue_utility_sum": float(rescue_utility.sum().item()),
            "cross_view_top1_identity_switch_rate_mean": float(
                identity_switch_rate[observation_count > 0].mean().item()
                if bool((observation_count > 0).any())
                else 0.0
            ),
            "cross_view_top1_identity_dominance_mean": float(
                identity_dominance[observation_count > 0].mean().item()
                if bool((observation_count > 0).any())
                else 0.0
            ),
            "cross_view_top1_benign_positive_switch_rate_mean": float(
                benign_switch_rate[observation_count > 0].mean().item()
                if bool((observation_count > 0).any())
                else 0.0
            ),
            "cross_view_top1_ambiguous_switch_rate_mean": float(
                ambiguous_switch_rate[observation_count > 0].mean().item()
                if bool((observation_count > 0).any())
                else 0.0
            ),
            "cross_view_top1_harmful_switch_rate_mean": float(
                harmful_switch_rate[observation_count > 0].mean().item()
                if bool((observation_count > 0).any())
                else 0.0
            ),
            **{
                f"gt_recall_at_{recall_k}": (
                    recall_at_k_sum[recall_k] / max(recall_at_k_queries, 1)
                )
                for recall_k in recall_at_k_sum
            },
        }
    )
    return statistics, diagnostics


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
    global_attractor_statistics=None,
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
    global_attractor_weight = float(
        getattr(args, "distill_global_attractor_weight", 0.0)
    )
    global_attractor_score = torch.zeros_like(matchability)
    global_attractor_false_rate = torch.zeros_like(matchability)
    global_attractor_incoming = torch.zeros_like(matchability)
    global_attractor_reliability = torch.ones_like(matchability)
    global_attractor_active = False
    if global_attractor_weight > 0.0:
        if global_attractor_statistics is None:
            raise ValueError(
                "distill_global_attractor_weight requires train-only "
                "global_attractor_statistics"
            )

        def _global_stat(name):
            value = global_attractor_statistics.get(name)
            if value is None:
                return torch.zeros_like(matchability)
            value = torch.as_tensor(
                value, device=matchability.device, dtype=matchability.dtype
            ).reshape(-1)
            if value.numel() != matchability.numel():
                raise ValueError(
                    "global_attractor_statistics must align with the source "
                    "landmark bank"
                )
            return value

        global_attractor_score = _global_stat("score").clamp_min(0.0)
        global_attractor_false_rate = _global_stat("false_rate").clamp(0.0, 1.0)
        global_attractor_incoming = _global_stat("incoming_count").clamp_min(0.0)
        # This is a ranking prior, not a query-time filter.  A landmark with
        # no observed false-attractor evidence remains neutral, while a
        # repeatedly wrong target is less likely to enter the fixed bank.
        global_attractor_reliability = torch.reciprocal(
            1.0 + global_attractor_weight * global_attractor_score
        )
        utility = utility * global_attractor_reliability
        global_attractor_active = True
    identity_switch_rate = statistics.get(
        "cross_view_top1_harmful_switch_rate",
        statistics.get(
            "cross_view_top1_identity_switch_rate",
            (1.0 - statistics["source_identity_rate"]).clamp(0.0, 1.0),
        ),
    )
    harmful_reliability = torch.pow(
        (1.0 - identity_switch_rate).clamp_min(1e-3),
        float(getattr(args, "distill_harmful_switch_weight", 0.0)),
    )
    rescue_utility = statistics.get(
        "rescue_utility", torch.zeros_like(matchability)
    )
    rescue_quality = torch.sigmoid(robust_normalize(rescue_utility))
    utility = utility * (
        1.0 + float(getattr(args, "distill_rescue_weight", 0.0)) * rescue_quality
    )
    selection_score = (
        matchability * global_attractor_reliability * harmful_reliability
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
    requested_budget = int(args.distill_budget)
    if int(eligible.sum().item()) < int(args.distill_budget):
        observed_indices = torch.nonzero(
            observed_eligible, as_tuple=False
        ).reshape(-1)
        if observed_indices.numel() > 0:
            reliable_order = torch.argsort(
                selection_score[observed_indices], descending=True, stable=True
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
    # The legacy exact-budget path preserved every >=2-view landmark and then
    # let coverage fill the remaining one-view capacity.  When that primary
    # tier is already smaller than K, a hard core cannot change final
    # membership: both the strict tier and the coverage fill are exhausted.
    # Restrict all later choices to a matchability-first reservoir over the
    # genuinely observed source pool, then let the hard core and coverage
    # policy choose *within* that reservoir.
    quality_reservoir_multiplier = max(
        float(args.distill_quality_reservoir_multiplier), 0.0
    )
    quality_reservoir = torch.empty(
        0, dtype=torch.long, device=bank_xyz.device
    )
    quality_reservoir_observed_any = torch.zeros_like(eligible)
    quality_reservoir_active = False
    quality_reservoir_score_mode = str(
        args.distill_quality_reservoir_score
    )
    quality_reservoir_score = matchability
    if quality_reservoir_score_mode == "wilson_lower":
        effective_correct = (
            (1.0 - false_top1_rate).clamp(0.0, 1.0) * observation_count
        )
        quality_reservoir_score = wilson_lower_confidence(
            effective_correct,
            observation_count,
            z=float(args.distill_quality_reservoir_wilson_z),
        )
    quality_reservoir_score = quality_reservoir_score * global_attractor_reliability
    if quality_reservoir_multiplier > 0.0:
        quality_reservoir_observed_any = observation_count > 0
        quality_reservoir = top_score_reservoir(
            quality_reservoir_score,
            requested_budget,
            quality_reservoir_multiplier,
            eligible=quality_reservoir_observed_any,
        )
        if int(quality_reservoir.numel()) >= requested_budget:
            eligible = torch.zeros_like(eligible)
            eligible[quality_reservoir] = True
            rank_pool_size = int(quality_reservoir.numel())
            eligibility_relaxed = "matchability_quality_reservoir_observed_any"
            quality_reservoir_active = True
        else:
            # Keep the pre-existing explicit coverage-fill route when the
            # source has fewer observed primitives than the requested bank.
            quality_reservoir = torch.empty(
                0, dtype=torch.long, device=bank_xyz.device
            )
    strict_budget = min(int(args.distill_budget), int(eligible.sum().item()))
    if strict_budget <= 0:
        raise ValueError("No observed landmarks are available for final distillation")
    if strict_budget < int(args.distill_budget):
        eligibility_relaxed = "observed_shortfall"
        if bool(args.distill_require_exact_budget) and not bool(
            args.distill_allow_coverage_fill
        ):
            raise ValueError(
                "Final landmark distillation could not satisfy the requested "
                f"budget ({strict_budget}/{int(args.distill_budget)} observed "
                "eligible landmarks). Increase --statistics_observations or use "
                "a smaller fixed comparison budget, or explicitly enable the "
                "coverage-fill tier."
            )
    hard_core_ratio = min(
        max(float(args.distill_hard_matchability_core_ratio), 0.0), 1.0
    )
    hard_core_count = min(
        strict_budget, int(round(float(strict_budget) * hard_core_ratio))
    )
    protected_core_ratio = min(
        max(float(getattr(args, "distill_protected_core_ratio", 0.0)), 0.0), 1.0
    )
    protected_core_budget = min(
        strict_budget,
        int(round(float(strict_budget) * protected_core_ratio)),
    )
    protected_correct_count = statistics.get(
        "correct_count",
        (1.0 - false_top1_rate).clamp(0.0, 1.0) * observation_count,
    )
    protected_eligible = (
        eligible
        & (
            protected_correct_count
            >= float(getattr(args, "distill_protected_min_correct", 3))
        )
        & (
            matchability
            >= float(getattr(args, "distill_protected_matchability", 0.75))
        )
        & (
            identity_switch_rate
            <= float(
                getattr(args, "distill_protected_identity_switch_max", 0.25)
            )
        )
    )
    protected_score = (
        wilson_lower_confidence(
            protected_correct_count,
            observation_count,
            z=float(args.distill_quality_reservoir_wilson_z),
        )
        * (1.0 - identity_switch_rate).clamp(0.0, 1.0)
        * torch.log1p(protected_correct_count)
        * global_attractor_reliability
    )
    protected_core = hard_score_core(
        protected_score,
        protected_core_budget,
        eligible=protected_eligible,
    )
    regular_core_eligible = eligible.clone()
    if protected_core.numel() > 0:
        regular_core_eligible[protected_core] = False
    regular_hard_core = hard_score_core(
        selection_score,
        max(hard_core_count - int(protected_core.numel()), 0),
        eligible=regular_core_eligible,
    )
    hard_core = torch.cat([protected_core, regular_hard_core]).unique(sorted=True)
    coverage_eligible = eligible.clone()
    if hard_core.numel() > 0:
        coverage_eligible[hard_core] = False
    coverage_budget = strict_budget - int(hard_core.numel())
    coverage_selected, selection_meta = coverage_preserving_sample(
        bank_xyz,
        base_score=selection_score,
        utility=utility,
        num=coverage_budget,
        min_observations=coverage_eligible,
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
    strict_selected = torch.cat([hard_core, coverage_selected]).unique(
        sorted=True
    )
    if int(strict_selected.numel()) != strict_budget:
        raise RuntimeError(
            "Hard matchability core plus coverage selection did not satisfy "
            f"the strict distillation budget ({int(strict_selected.numel())}/"
            f"{strict_budget})"
        )
    selection_meta = dict(selection_meta)
    hard_core_mask = torch.zeros(
        bank_xyz.shape[0], dtype=torch.bool, device=bank_xyz.device
    )
    hard_core_mask[hard_core] = True
    protected_core_mask = torch.zeros_like(hard_core_mask)
    protected_core_mask[protected_core] = True
    selection_meta.update(
        {
            "hard_matchability_core_indices": hard_core.detach().clone(),
            "hard_matchability_core_ratio": torch.tensor(
                hard_core_ratio, device=bank_xyz.device
            ),
            "hard_matchability_core_count": torch.tensor(
                int(hard_core.numel()), device=bank_xyz.device
            ),
            "hard_matchability_core_cutoff": torch.tensor(
                float(matchability[hard_core].min().item())
                if hard_core.numel() > 0
                else 0.0,
                device=bank_xyz.device,
            ),
            "protected_core_indices": protected_core.detach().clone(),
            "protected_core_ratio": torch.tensor(
                protected_core_ratio, device=bank_xyz.device
            ),
            "protected_core_count": torch.tensor(
                int(protected_core.numel()), device=bank_xyz.device
            ),
        }
    )
    # Exact-bank comparisons must not quietly weaken the matchability gate.
    # When explicitly requested, retain every strict selection first and only
    # fill a capacity shortfall from the remaining source bank by coverage.
    # The tier provenance is persisted below so this cannot be confused with a
    # pure matchability-first distilled bank.
    selected = strict_selected
    selection_tier = torch.full(
        (bank_xyz.shape[0],), -1, dtype=torch.int8, device=bank_xyz.device
    )
    selection_tier[strict_selected] = 0
    coverage_fill_observed = torch.empty(
        0, dtype=torch.long, device=bank_xyz.device
    )
    coverage_fill_unobserved = torch.empty(
        0, dtype=torch.long, device=bank_xyz.device
    )
    remaining_budget = requested_budget - int(selected.numel())
    if remaining_budget > 0 and bool(args.distill_allow_coverage_fill):
        observed_any = observation_count > 0
        support_quality = torch.log1p(observation_count)
        if bool(observed_any.any().item()):
            support_quality = support_quality / support_quality[
                observed_any
            ].max().clamp_min(1.0)
        else:
            support_quality = torch.zeros_like(support_quality)
        coverage_fill_score = utility + 0.15 * support_quality
        selected_mask = torch.zeros_like(eligible)
        selected_mask[selected] = True
        observed_fill_eligible = observed_any & ~selected_mask
        coverage_fill_observed = coverage_ranked_fill(
            bank_xyz,
            coverage_fill_score,
            remaining_budget,
            observed_fill_eligible,
            voxel_size=voxel_size,
            max_per_voxel=args.distill_max_per_voxel,
            selected=selected,
            uv=statistics["mean_uv"] * 1000.0,
            image_size=(1000, 1000),
            grid_size=args.distill_grid_size,
            max_per_grid=args.distill_max_per_grid,
            depth=statistics["mean_depth"],
            depth_bins=args.distill_depth_bins,
            max_per_depth_bin=args.distill_max_per_depth_bin,
        )
        if coverage_fill_observed.numel() > 0:
            selected = torch.cat([selected, coverage_fill_observed]).unique(
                sorted=True
            )
            selection_tier[coverage_fill_observed] = 1
        remaining_budget = requested_budget - int(selected.numel())
        if remaining_budget > 0:
            selected_mask.zero_()
            selected_mask[selected] = True
            unobserved_fill_eligible = ~observed_any & ~selected_mask
            coverage_fill_unobserved = coverage_ranked_fill(
                bank_xyz,
                coverage_fill_score,
                remaining_budget,
                unobserved_fill_eligible,
                voxel_size=voxel_size,
                max_per_voxel=args.distill_max_per_voxel,
                selected=selected,
                uv=statistics["mean_uv"] * 1000.0,
                image_size=(1000, 1000),
                grid_size=args.distill_grid_size,
                max_per_grid=args.distill_max_per_grid,
                depth=statistics["mean_depth"],
                depth_bins=args.distill_depth_bins,
                max_per_depth_bin=args.distill_max_per_depth_bin,
            )
            if coverage_fill_unobserved.numel() > 0:
                selected = torch.cat(
                    [selected, coverage_fill_unobserved]
                ).unique(sorted=True)
                selection_tier[coverage_fill_unobserved] = 2
            remaining_budget = requested_budget - int(selected.numel())
        eligibility_relaxed = "matchability_then_explicit_coverage_fill"
    if remaining_budget > 0 and bool(args.distill_require_exact_budget):
        raise ValueError(
            "Final landmark distillation could not satisfy the requested "
            f"budget after explicit coverage fill ({int(selected.numel())}/"
            f"{requested_budget})."
        )
    selection_meta.update(
        {
            "strict_indices": strict_selected.detach().clone(),
            "final_indices": selected.detach().clone(),
            "strict_matchability_selected_count": torch.tensor(
                int(strict_selected.numel()), device=bank_xyz.device
            ),
            "coverage_fill_observed_count": torch.tensor(
                int(coverage_fill_observed.numel()), device=bank_xyz.device
            ),
            "coverage_fill_unobserved_count": torch.tensor(
                int(coverage_fill_unobserved.numel()), device=bank_xyz.device
            ),
            "final_selection_tier": selection_tier[selected].detach().clone(),
            "quality_reservoir_indices": quality_reservoir.detach().clone(),
            "quality_reservoir_active": torch.tensor(
                quality_reservoir_active, device=bank_xyz.device
            ),
            "quality_reservoir_multiplier": torch.tensor(
                quality_reservoir_multiplier, device=bank_xyz.device
            ),
            "quality_reservoir_score_mode": quality_reservoir_score_mode,
            "quality_reservoir_wilson_z": torch.tensor(
                float(args.distill_quality_reservoir_wilson_z),
                device=bank_xyz.device,
            ),
            "quality_reservoir_cutoff": torch.tensor(
                float(quality_reservoir_score[quality_reservoir].min().item())
                if quality_reservoir.numel() > 0
                else 0.0,
                device=bank_xyz.device,
            ),
            "quality_reservoir_matchability_cutoff": torch.tensor(
                float(matchability[quality_reservoir].min().item())
                if quality_reservoir.numel() > 0
                else 0.0,
                device=bank_xyz.device,
            ),
            "quality_reservoir_observed_count": torch.tensor(
                int(quality_reservoir_observed_any.sum().item()),
                device=bank_xyz.device,
            ),
            "global_attractor_selection_active": torch.tensor(
                global_attractor_active, device=bank_xyz.device
            ),
            "global_attractor_weight": torch.tensor(
                global_attractor_weight, device=bank_xyz.device
            ),
        }
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
        "global_attractor_selection": {
            "enabled": global_attractor_active,
            "weight": global_attractor_weight,
            "statistics_split": "train_only" if global_attractor_active else None,
            "role": "fixed_bank_ranking_prior_not_query_filter",
        },
        "uses_conditional_translation_fim": True,
        "uses_3d_image_depth_coverage": True,
        "eligibility_relaxed": eligibility_relaxed,
        "rank_pool_size": int(rank_pool_size),
        "rank_pool_multiplier": float(args.distill_rank_pool_multiplier),
        "quality_reservoir_active": quality_reservoir_active,
        "quality_reservoir_multiplier": quality_reservoir_multiplier,
        "quality_reservoir_score_mode": quality_reservoir_score_mode,
        "quality_reservoir_wilson_z": float(
            args.distill_quality_reservoir_wilson_z
        ),
        "quality_reservoir_count": int(quality_reservoir.numel()),
        "quality_reservoir_observed_count": int(
            quality_reservoir_observed_any.sum().item()
        ),
        "quality_reservoir_cutoff": float(
            quality_reservoir_score[quality_reservoir].min().item()
            if quality_reservoir.numel() > 0
            else 0.0
        ),
        "quality_reservoir_matchability_cutoff": float(
            matchability[quality_reservoir].min().item()
            if quality_reservoir.numel() > 0
            else 0.0
        ),
        "quality_reservoir_selection_score_cutoff": float(
            selection_score[quality_reservoir].min().item()
            if quality_reservoir.numel() > 0
            else 0.0
        ),
        "hard_matchability_core_ratio": hard_core_ratio,
        "hard_matchability_core_count": int(hard_core.numel()),
        "hard_matchability_core_cutoff": float(
            matchability[hard_core].min().item()
            if hard_core.numel() > 0
            else 0.0
        ),
        "protected_core_count": int(protected_core.numel()),
        "protected_core_ratio": protected_core_ratio,
        "protected_identity_switch_max": float(
            getattr(args, "distill_protected_identity_switch_max", 0.25)
        ),
        "strict_coverage_selected_count": int(coverage_selected.numel()),
        "requested_budget": requested_budget,
        "strict_matchability_selected_count": int(strict_selected.numel()),
        "coverage_fill_enabled": bool(args.distill_allow_coverage_fill),
        "coverage_fill_observed_count": int(coverage_fill_observed.numel()),
        "coverage_fill_unobserved_count": int(
            coverage_fill_unobserved.numel()
        ),
        "coverage_fill_count": int(selected.numel() - strict_selected.numel()),
        "observed_budget_shortfall": requested_budget - int(strict_selected.numel()),
        "final_budget_shortfall": requested_budget - int(selected.numel()),
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
        "distill_selection_tier": selection_tier[selected].detach().cpu(),
        "distill_support_tier": torch.where(
            observation_count[selected]
            >= float(args.distill_min_observations),
            torch.full_like(observation_count[selected], 2, dtype=torch.int8),
            torch.where(
                observation_count[selected] > 0,
                torch.ones_like(observation_count[selected], dtype=torch.int8),
                torch.zeros_like(observation_count[selected], dtype=torch.int8),
            ),
        ).detach().cpu(),
        "hard_matchability_core": hard_core_mask[selected].detach().cpu(),
        "protected_core": protected_core_mask[selected].detach().cpu(),
        "cross_view_top1_identity_switch_rate": identity_switch_rate[selected]
        .detach()
        .cpu(),
        "quality_reservoir_score": quality_reservoir_score[selected]
        .detach()
        .cpu(),
        "global_attractor_score": global_attractor_score[selected].detach().cpu(),
        "global_attractor_false_rate": global_attractor_false_rate[selected]
        .detach()
        .cpu(),
        "global_attractor_incoming_count": global_attractor_incoming[selected]
        .detach()
        .cpu(),
        "global_attractor_reliability": global_attractor_reliability[selected]
        .detach()
        .cpu(),
        "harmful_switch_reliability": harmful_reliability[selected]
        .detach()
        .cpu(),
        "rescue_utility": rescue_utility[selected].detach().cpu(),
        "rescue_quality": rescue_quality[selected].detach().cpu(),
        "selection_score": selection_score[selected].detach().cpu(),
        "quality_reservoir_member": torch.zeros_like(
            eligible, dtype=torch.bool
        )
        .scatter_(0, quality_reservoir, True)[selected]
        .detach()
        .cpu(),
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
        "hard_matchability_core_count": int(hard_core.numel()),
        "protected_core_count": int(protected_core.numel()),
        "protected_core_switch_rate_mean": float(
            identity_switch_rate[protected_core].mean().item()
            if protected_core.numel() > 0
            else 0.0
        ),
        "hard_matchability_core_matchability_mean": float(
            matchability[hard_core].mean().item()
            if hard_core.numel() > 0
            else 0.0
        ),
        "translation_fim_selected_mean": float(
            statistics["translation_fim"][selected].mean().item()
        ),
        "eligibility_relaxed": eligibility_relaxed,
        "rank_pool_size": int(rank_pool_size),
        "quality_reservoir_count": int(quality_reservoir.numel()),
        "quality_reservoir_active": quality_reservoir_active,
        "quality_reservoir_score_mode": quality_reservoir_score_mode,
        "global_attractor_selection_active": global_attractor_active,
        "global_attractor_selected_score_mean": float(
            global_attractor_score[selected].mean().item()
        ),
        "global_attractor_selected_reliability_mean": float(
            global_attractor_reliability[selected].mean().item()
        ),
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
            **_native_candidate_loss_kwargs(args),
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
                **_native_candidate_loss_kwargs(args),
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
        else:
            auxiliary_observations = anchor_auxiliary
        auxiliary_scale = _native_anchor_auxiliary_scale(args)
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
            local_loss = auxiliary_scale * local_loss
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
    *,
    initial_state=None,
    rgb_prior_contract=None,
):
    inherited_reject_contract = None
    if isinstance(initial_state, dict):
        inherited_config = initial_state.get("config", {})
        if isinstance(inherited_config, dict):
            candidate = inherited_config.get("native_reject_contract")
            if isinstance(candidate, dict) and bool(candidate.get("enabled", False)):
                inherited_reject_contract = dict(candidate)
            elif (
                bool(inherited_config.get("native_outcome_mode", False))
                and float(inherited_config.get("native_reject_weight", 0.0)) > 0.0
            ):
                inherited_reject_contract = {
                    "enabled": True,
                    "deployment_match_threshold": float(
                        inherited_config["native_reject_threshold"]
                    ),
                    "source": "legacy_initial_native_residual",
                }
    native_reject_enabled = (
        bool(args.native_outcome_mode)
        and str(args.observation_source) in {"native", "native_plus_anchor"}
        and float(args.native_reject_weight) > 0.0
    )
    if native_reject_enabled:
        native_reject_contract = {
            "enabled": True,
            "deployment_match_threshold": float(args.native_reject_threshold),
            "source": "current_native_residual",
        }
    elif inherited_reject_contract is not None:
        native_reject_contract = {
            **inherited_reject_contract,
            "source": "inherited_fixed_descriptor_stage",
        }
    else:
        native_reject_contract = {"enabled": False}
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
        "native_auxiliary_contract": _native_auxiliary_contract(args),
        "frozen_generic_proposal_head": bool(args.generic_proposal_count > 0),
        "geometry_frozen": not bool(args.geometry_weight > 0.0),
        "raw_xyz_trainable": False,
        "localization_base_from_initial_state": bool(
            args.initial_state_geometry_as_base
        ),
        "bounded_anchor_trainable": bool(args.geometry_weight > 0.0),
        "dynamic_landmark_selection": False,
        "one_time_landmark_distillation": bool(args.distill_budget > 0),
        "landmark_statistics_saved": bool(args.save_landmark_statistics),
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
        "native_semidense_weight": float(args.native_semidense_weight),
        "native_semidense_start_step": int(args.native_semidense_start_step),
        "native_semidense_interval": int(args.native_semidense_interval),
        "native_semidense_max_anchors": int(args.native_semidense_max_anchors),
        "native_semidense_neighbors": int(args.native_semidense_neighbors),
        "native_semidense_neighborhood_radius_m": float(
            args.native_semidense_neighborhood_radius_m
        ),
        "native_semidense_normal_cosine": float(
            args.native_semidense_normal_cosine
        ),
        "native_semidense_local_radius_px": int(
            args.native_semidense_local_radius_px
        ),
        "native_semidense_target_sigma_px": float(
            args.native_semidense_target_sigma_px
        ),
        "native_semidense_temperature": float(
            args.native_semidense_temperature
        ),
        "native_semidense_lgcv_weight": float(
            args.native_semidense_lgcv_weight
        ),
        "native_semidense_protected_v2": bool(
            args.native_semidense_protected_v2
        ),
        "native_semidense_measurement_min_reprojection_px": float(
            args.native_semidense_measurement_min_reprojection_px
        ),
        "native_semidense_measurement_max_reprojection_px": float(
            args.native_semidense_measurement_max_reprojection_px
        ),
        "native_semidense_surface_point_plane_m": float(
            args.native_semidense_surface_point_plane_m
        ),
        "native_semidense_surface_max_distance_m": float(
            args.native_semidense_surface_max_distance_m
        ),
        "native_semidense_surface_normal_cosine": float(
            args.native_semidense_surface_normal_cosine
        ),
        "native_semidense_projected_neighbor_radius_px": float(
            args.native_semidense_projected_neighbor_radius_px
        ),
        "native_semidense_local_identity_weight": float(
            args.native_semidense_local_identity_weight
        ),
        "native_semidense_margin_preservation_weight": float(
            args.native_semidense_margin_preservation_weight
        ),
        "native_semidense_gradient_audit": bool(
            args.native_semidense_gradient_audit
        ),
        "native_semidense_reference_refresh_steps": int(
            args.native_semidense_reference_refresh_steps
        ),
        "native_semidense_alternate_global": bool(
            args.native_semidense_alternate_global
        ),
        "native_semidense_max_gradient_ratio": float(
            args.native_semidense_max_gradient_ratio
        ),
        "native_protected_set_weight": float(
            args.native_protected_set_weight
        ),
        "native_protected_set_start_step": int(
            args.native_protected_set_start_step
        ),
        "native_protected_set_interval": int(
            args.native_protected_set_interval
        ),
        "native_protected_set_refresh_visits": int(
            args.native_protected_set_refresh_visits
        ),
        "native_protected_set_contract": {
            "fixed_seed_ransac": True,
            "ransac_seed": int(args.native_protected_set_ransac_seed),
            "strong_positive_radius_px": float(args.positive_radius_px),
            "neutral_radius_px": float(args.negative_radius_px),
            "max_useful": int(args.native_protected_set_max_useful),
            "max_harmful": int(args.native_protected_set_max_harmful),
            "grid_rows": int(args.native_protected_set_grid_rows),
            "grid_cols": int(args.native_protected_set_grid_cols),
            "depth_bins": int(args.native_protected_set_depth_bins),
            "surface_voxel_m": float(
                args.native_protected_set_surface_voxel_m
            ),
            "max_per_surface_group": int(
                args.native_protected_set_max_per_surface_group
            ),
        },
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
        "native_outcome_mode": bool(args.native_outcome_mode),
        "native_rank_budget_mode": bool(args.native_rank_budget_mode),
        "native_rank_stage_a_steps": int(args.native_rank_stage_a_steps),
        "native_rank_steps": int(args.native_rank_steps),
        "native_rank_temperature": float(args.native_rank_temperature),
        "native_rank_margins": {
            "at1": float(args.native_rank_margin_at1),
            "at4": float(args.native_rank_margin_at4),
            "at8": float(args.native_rank_margin_at8),
            "at32": float(args.native_rank_margin_at32),
        },
        "native_rank_top1_weight": float(args.native_rank_top1_weight),
        "native_rank_keep_weight": float(args.native_rank_keep_weight),
        "native_rank_reference_clean_weight": float(
            args.native_rank_reference_clean_weight
        ),
        "native_rank_reference_clean_margin": float(
            args.native_rank_reference_clean_margin
        ),
        "native_rank_landmark_balance": bool(
            args.native_rank_landmark_balance
        ),
        "native_rank_band_proportions": {
            "rank1": float(args.native_rank_band_rank1),
            "rank2_4": float(args.native_rank_band_rank2_4),
            "rank5_32": float(args.native_rank_band_rank5_32),
            "rank33_plus": float(args.native_rank_band_rank33_plus),
        },
        "native_nce_weight": float(args.native_nce_weight),
        "native_keep_weight": float(args.native_keep_weight),
        "native_keep_margin": float(args.native_keep_margin),
        "native_keep_loose_weight": float(args.native_keep_loose_weight),
        "native_keep_loose_radius_px": float(args.native_keep_loose_radius_px),
        "native_keep_loose_margin": float(args.native_keep_loose_margin),
        "native_swap_weight": float(args.native_swap_weight),
        "native_swap_margin": float(args.native_swap_margin),
        "native_miss_weight": float(args.native_miss_weight),
        "native_miss_margin": float(args.native_miss_margin),
        "native_reject_weight": float(args.native_reject_weight),
        "native_reject_threshold": float(args.native_reject_threshold),
        "native_attractor_weight": float(args.native_attractor_weight),
        "native_attractor_margin": float(args.native_attractor_margin),
        "native_global_attractor_weight": float(
            args.native_global_attractor_weight
        ),
        "native_global_attractor_min_incoming": int(
            args.native_global_attractor_min_incoming
        ),
        "native_global_attractor_support_power": float(
            args.native_global_attractor_support_power
        ),
        "native_global_attractor_max_score": float(
            args.native_global_attractor_max_score
        ),
        # A bounded-BA state inherits this from the residual state so the
        # evaluator can still enforce the descriptor-stage deployment score.
        "native_reject_contract": native_reject_contract,
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
        "geometry_association_depth_abs_tolerance": float(
            args.geometry_association_depth_abs_tolerance
        ),
        "geometry_association_depth_rel_tolerance": float(
            args.geometry_association_depth_rel_tolerance
        ),
        "geometry_association_min_support_views": int(
            args.geometry_association_min_support_views
        ),
        "geometry_association_support_observations": int(
            args.geometry_association_support_observations
        ),
        "surface_weight": float(args.surface_weight),
        "depth_weight": float(args.depth_weight),
        "reprojection_weight": float(args.reprojection_weight),
        "surface_anchor_parameterization": (
            "radial_tanh_tangent_plane_v1"
            if str(dataset.gaussian_type).lower() == "2dgs"
            else "covariance_bounded_tanh_v1"
        ),
        "tangent_bound_m": float(args.tangent_bound_m),
        "normal_bound_m": float(args.normal_bound_m),
        "covariance_anchor_scale": float(args.covariance_anchor_scale),
        "covariance_anchor_absolute_bound_m": float(
            args.covariance_anchor_absolute_bound_m
        ),
        "rgb_prior_contract": dict(rgb_prior_contract or {}),
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
        "scaffold_min_opacity": float(args.scaffold_min_opacity),
        "scaffold_opacity_keep_quantile": float(
            args.scaffold_opacity_keep_quantile
        ),
        "ulf_consensus_radius_px": float(args.ulf_consensus_radius_px),
        "ulf_consensus_min_votes": int(args.ulf_consensus_min_votes),
        "ulf_consensus_min_visible_views": int(
            args.ulf_consensus_min_visible_views
        ),
        "ulf_consensus_min_rate": float(args.ulf_consensus_min_rate),
        "ulf_consensus_view_bins": int(args.ulf_consensus_view_bins),
        "ulf_consensus_min_distinct_view_bins": int(
            args.ulf_consensus_min_distinct_view_bins
        ),
        "ulf_consensus_trajectory_bins": int(args.ulf_consensus_trajectory_bins),
        "ulf_consensus_min_distinct_trajectory_bins": int(
            args.ulf_consensus_min_distinct_trajectory_bins
        ),
        "ulf_consensus_independent_bin_scoring": bool(
            args.ulf_consensus_independent_bin_scoring
        ),
        "ulf_consensus_allow_nonconsensus_fallback": (
            None
            if args.ulf_consensus_allow_nonconsensus_fallback is None
            else bool(args.ulf_consensus_allow_nonconsensus_fallback)
        ),
        "ulf_consensus_knn": int(args.ulf_consensus_knn),
        "ulf_fusion_min_cosine": float(args.ulf_fusion_min_cosine),
        "ulf_fusion_descriptor_min_cosine": float(
            args.ulf_fusion_descriptor_min_cosine
        ),
        "ulf_fusion_descriptor_trim_fraction": float(
            args.ulf_fusion_descriptor_trim_fraction
        ),
        "ulf_fusion_adaptive_trim": bool(args.ulf_fusion_adaptive_trim),
        "ulf_fusion_adaptive_trim_min_fraction": float(
            args.ulf_fusion_adaptive_trim_min_fraction
        ),
        "ulf_fusion_adaptive_trim_max_fraction": float(
            args.ulf_fusion_adaptive_trim_max_fraction
        ),
        "ulf_fusion_adaptive_trim_tail_cosine": float(
            args.ulf_fusion_adaptive_trim_tail_cosine
        ),
        "ulf_fusion_adaptive_trim_mode": str(args.ulf_fusion_adaptive_trim_mode),
        "ulf_fusion_adaptive_trim_mad_scale": float(
            args.ulf_fusion_adaptive_trim_mad_scale
        ),
        "ulf_fusion_adaptive_trim_min_observations": int(
            args.ulf_fusion_adaptive_trim_min_observations
        ),
        "ulf_fusion_reference_mode": str(args.ulf_fusion_reference_mode),
        "ulf_fusion_trim_histogram_bins": int(
            args.ulf_fusion_trim_histogram_bins
        ),
        "ulf_fusion_view_bins": int(args.ulf_fusion_view_bins),
        "ulf_fusion_exact_bin_balance": bool(
            args.ulf_fusion_exact_bin_balance
        ),
        "ulf_support_mask_policy": str(args.ulf_support_mask_policy),
        "ulf_parity_fusion_channel_chunk": int(
            args.ulf_parity_fusion_channel_chunk
        ),
        "coordinate_convention": "feature_grid_index_plus_half_physical_v1",
        "valid_mask_policy": _VALID_MASK_POLICY,
        "distill_budget": int(args.distill_budget),
        "distill_require_exact_budget": bool(args.distill_require_exact_budget),
        "distill_allow_coverage_fill": bool(args.distill_allow_coverage_fill),
        "distill_rank_pool_multiplier": float(
            args.distill_rank_pool_multiplier
        ),
        "distill_matchability_preserve_ratio": float(
            args.distill_matchability_preserve_ratio
        ),
        "distill_hard_matchability_core_ratio": float(
            args.distill_hard_matchability_core_ratio
        ),
        "distill_quality_reservoir_multiplier": float(
            args.distill_quality_reservoir_multiplier
        ),
        "distill_quality_reservoir_score": str(
            args.distill_quality_reservoir_score
        ),
        "distill_quality_reservoir_wilson_z": float(
            args.distill_quality_reservoir_wilson_z
        ),
        "distill_utility_preserve_ratio": float(
            args.distill_utility_preserve_ratio
        ),
        "distill_high_confidence": float(args.distill_high_confidence),
        "distill_high_confidence_ratio": float(
            args.distill_high_confidence_ratio
        ),
        "distill_grid_size": int(args.distill_grid_size),
        "distill_max_per_grid": int(args.distill_max_per_grid),
        "distill_depth_bins": int(args.distill_depth_bins),
        "distill_max_per_depth_bin": int(args.distill_max_per_depth_bin),
        "distill_max_per_voxel": int(args.distill_max_per_voxel),
        "statistics_observations": int(args.statistics_observations),
        "statistics_responsibility_sigma_px": float(
            args.statistics_responsibility_sigma_px
        ),
        "distill_min_observations": int(args.distill_min_observations),
        "distill_matchability_threshold": float(
            args.distill_matchability_threshold
        ),
        "distill_false_top1_max": float(args.distill_false_top1_max),
        "distill_rescue_max_positives": int(
            getattr(args, "distill_rescue_max_positives", 4)
        ),
        "distill_rescue_weight": float(
            getattr(args, "distill_rescue_weight", 0.0)
        ),
        "distill_harmful_switch_weight": float(
            getattr(args, "distill_harmful_switch_weight", 0.0)
        ),
        "distill_proposal_weight": float(args.distill_proposal_weight),
        "distill_global_attractor_weight": float(
            args.distill_global_attractor_weight
        ),
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
    if bool(args.native_outcome_mode) and not native_observation_mode:
        raise ValueError(
            "native_outcome_mode requires --observation_source native or "
            "native_plus_anchor"
        )
    if bool(args.native_rank_budget_mode) and not native_observation_mode:
        raise ValueError(
            "native_rank_budget_mode requires --observation_source native or "
            "native_plus_anchor"
        )
    _validate_native_objective_semantics(args)
    _validate_distillation_semantics(args)
    _validate_ulf_initializer_semantics(args)
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
    ulf_mode = (
        args.scaffold_mode
        in {"ulf_consensus", "ulf_parity", "ulf_robust_consensus"}
        or args.initialization_mode
        in {"ulf_geometry", "ulf_parity", "ulf_robust_geometry"}
    )
    if ulf_mode and str(args.query_feature_contract) != _QUERY_FEATURE_CONTRACT_NATIVE:
        raise ValueError(
            "ULF KCS/GWFF requires --query_feature_contract "
            f"{_QUERY_FEATURE_CONTRACT_NATIVE}; otherwise its native sparse "
            "descriptors would be initialized against a different cache contract."
        )
    if (
        args.scaffold_mode == "ulf_parity"
        or args.initialization_mode == "ulf_parity"
    ) and int(dataset.longest_edge) > 0:
        raise ValueError(
            "Strict ULF parity requires --longest_edge 0; a resized input is "
            "an extension rather than a parity bootstrap"
        )
    if (
        args.scaffold_mode == "ulf_parity"
        or args.initialization_mode == "ulf_parity"
    ) and (
        float(args.ulf_fusion_descriptor_trim_fraction) != 0.0
        or bool(args.ulf_fusion_adaptive_trim)
        or float(args.ulf_fusion_descriptor_min_cosine) != -1.0
        or int(args.ulf_consensus_min_visible_views) != 0
        or float(args.ulf_consensus_min_rate) != 0.0
        or int(args.ulf_consensus_min_distinct_view_bins) != 0
        or int(args.ulf_consensus_min_distinct_trajectory_bins) != 0
        or bool(args.ulf_consensus_independent_bin_scoring)
    ):
        raise ValueError(
            "Strict ULF parity cannot be combined with robust KCS/GWFF gates; "
            "use ulf_robust_consensus / ulf_robust_geometry explicitly"
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
        rgb_prior_contract=rgb_prior_contract,
    )
    landmark_indices_cuda = landmark_indices.cuda()
    rgb_bank_xyz = gaussians.get_xyz[landmark_indices_cuda].detach().float()
    base_bank_xyz = rgb_bank_xyz
    if bool(args.initial_state_geometry_as_base):
        if not args.initial_state_path:
            raise ValueError(
                "--initial_state_geometry_as_base requires --initial_state_path"
            )
        geometry_state = torch.load(
            args.initial_state_path,
            map_location="cpu",
            weights_only=False,
        )
        geometry_indices = torch.as_tensor(
            geometry_state.get("landmark_indices"), dtype=torch.long
        ).reshape(-1)
        if not torch.equal(geometry_indices, landmark_indices.cpu()):
            raise ValueError(
                "Initial-state localization geometry is not exactly aligned "
                "with the requested landmark bank"
            )
        geometry_xyz = torch.as_tensor(
            geometry_state.get("landmark_xyz"), dtype=torch.float32
        )
        if geometry_xyz.shape != base_bank_xyz.shape:
            raise ValueError(
                "Initial-state landmark_xyz must match the landmark bank"
            )
        if not bool(torch.isfinite(geometry_xyz).all()):
            raise ValueError("Initial-state landmark_xyz contains non-finite values")
        base_bank_xyz = geometry_xyz.to(base_bank_xyz.device)
    base_bank_rotation = (
        gaussians.get_rotation[landmark_indices_cuda].detach().float()
    )
    base_bank_scaling = (
        gaussians.get_scaling[landmark_indices_cuda].detach().float()
    )
    base_bank_opacity = (
        gaussians.get_opacity[landmark_indices_cuda].detach().float().reshape(-1)
    )
    prior_geometry = GaussianPriorGeometry(
        str(dataset.gaussian_type),
        xyz=base_bank_xyz,
        rotation=base_bank_rotation,
        scaling=base_bank_scaling,
    )
    rgb_prior_geometry = GaussianPriorGeometry(
        str(dataset.gaussian_type),
        xyz=rgb_bank_xyz,
        rotation=base_bank_rotation,
        scaling=base_bank_scaling,
    )
    base_bank_normals = prior_geometry.proxy_normals.detach()

    def materialize_anchor(raw_offset):
        return prior_geometry.materialize_anchor(
            raw_offset,
            tangent_bound_m=args.tangent_bound_m,
            normal_bound_m=args.normal_bound_m,
            covariance_scale=args.covariance_anchor_scale,
            absolute_bound_m=args.covariance_anchor_absolute_bound_m,
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
    elif args.initialization_mode == "ulf_parity" and not state_only_initialization:
        mvinit_features, mvinit_observation_count, mvinit_diagnostics = (
            _build_ulf_parity_geometry_features(
                train_cameras,
                gaussians,
                landmark_indices,
                masks,
                feature_extractor,
                fallback,
                args,
            )
        )
    elif (
        args.initialization_mode == "ulf_robust_geometry"
        and not state_only_initialization
    ):
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
    elif args.initialization_mode in {
        "ulf_geometry",
        "ulf_parity",
        "ulf_robust_geometry",
    }:
        # A descriptor continuation with blend=1 reuses the bootstrap state
        # exactly. Avoid recomputing the frozen ULF fusion merely to construct
        # an unused fallback.
        mvinit_features = fallback
        mvinit_observation_count = torch.zeros(
            fallback.shape[0], dtype=torch.long, device=fallback.device
        )
        mvinit_diagnostics = {
            "initialization_mode": (
                "ulf_parity_geometry_weighted_fusion_v1"
                if args.initialization_mode == "ulf_parity"
                else (
                    "ulf_robust_geometry_weighted_fusion_v1"
                    if args.initialization_mode == "ulf_robust_geometry"
                    else "ulf_geometry_weighted_fusion"
                )
            ),
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
        validate_gaussian_anchor_resume(
            initial_state,
            gaussian_type=dataset.gaussian_type,
            tangent_bound_m=args.tangent_bound_m,
            normal_bound_m=args.normal_bound_m,
            covariance_scale=args.covariance_anchor_scale,
            absolute_bound_m=args.covariance_anchor_absolute_bound_m,
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

    initial_xyz = materialize_anchor(raw_anchor_offset)
    geometry_support_mask = None
    geometry_support_counts = None
    geometry_support_diagnostics = {}
    if (
        str(args.geometry_mode) == "native_association"
        and float(args.geometry_weight) > 0.0
    ):
        geometry_support_mask, geometry_support_counts, geometry_support_diagnostics = (
            _native_geometry_support_mask(
                initial_features,
                train_names,
                cache,
                base_bank_xyz,
                initial_xyz.detach(),
                args,
                visibility_cache=visibility_cache,
            )
        )
        if not bool(geometry_support_mask.any().item()):
            raise RuntimeError(
                "Native BA found no landmarks with the requested number of "
                "distinct GT-clean support views"
            )
        support_path = output_dir / "native_geometry_support.pt"
        torch.save(
            {
                "counts": geometry_support_counts.detach().cpu(),
                "eligible": geometry_support_mask.detach().cpu(),
                "diagnostics": dict(geometry_support_diagnostics),
            },
            support_path,
        )
        geometry_support_diagnostics["native_geometry_support_path"] = str(
            support_path.resolve()
        )
    native_global_attractor_scores = None
    native_global_attractor_diagnostics = {
        "native_global_attractor_prior_enabled": 0.0,
    }
    if (
        native_observation_mode
        and bool(args.native_outcome_mode)
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
        rank_reference_bank_features=(
            initial_features.detach()
            if float(args.native_rank_reference_clean_weight) > 0.0
            else None
        ),
        rank_landmark_opportunities=(
            mvinit_observation_count.detach()
            if bool(args.native_rank_landmark_balance)
            else None
        ),
    )
    functional_replay_bank = None
    functional_replay_rng = torch.Generator(device="cpu")
    functional_replay_rng.manual_seed(int(args.train_seed) + 17041)
    if float(args.functional_replay_weight) > 0.0:
        replay_path = (
            Path(args.functional_replay_cache_path).expanduser().resolve()
            if str(args.functional_replay_cache_path).strip()
            else output_dir / "functional_replay_bank.pt"
        )
        if replay_path.exists():
            functional_replay_bank = torch.load(
                replay_path, map_location="cpu"
            )
            expected_names = list(train_names)
            if functional_replay_bank.get("train_camera_names") != expected_names:
                raise ValueError(
                    "Functional replay cache train-camera contract does not "
                    "match the current all-train split"
                )
            if int(
                functional_replay_bank["reference_candidate_indices"].shape[1]
            ) != int(args.functional_replay_topm):
                raise ValueError(
                    "Functional replay cache top-M does not match "
                    "--functional_replay_topm"
                )
        else:
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            functional_replay_bank = _build_functional_replay_bank(
                initial_features,
                train_names,
                cache,
                initial_xyz,
                args,
                visibility_cache=visibility_cache,
                base_bank_xyz=base_bank_xyz,
            )
            torch.save(functional_replay_bank, replay_path)
        replay_diagnostics = dict(
            functional_replay_bank.get("diagnostics", {})
        )
        config["functional_replay"] = {
            "enabled": True,
            "path": str(replay_path),
            "row_count": int(
                functional_replay_bank["query_features"].shape[0]
            ),
            "core_rows_per_query": int(
                args.functional_replay_core_rows_per_query
            ),
            "weight": float(args.functional_replay_weight),
            "batch_size": int(args.functional_replay_batch_size),
            "topm": int(args.functional_replay_topm),
            "temperature": float(args.functional_replay_temperature),
            "margin_slack": float(args.functional_replay_margin_slack),
            "distribution_weight": float(
                args.functional_replay_distribution_weight
            ),
            "pnp_core_weight": float(
                args.functional_replay_pnp_core_weight
            ),
            "build_pnp_core": bool(
                args.functional_replay_build_pnp_core
            ),
            "gradient_projection": bool(
                args.functional_replay_gradient_projection
            ),
            **replay_diagnostics,
        }
    else:
        replay_diagnostics = {}
        config["functional_replay"] = {"enabled": False}
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
        {
            **mvinit_diagnostics,
            **geometry_support_diagnostics,
            **native_global_attractor_diagnostics,
            **replay_diagnostics,
            **initial_validation,
        },
        mvinit_observation_count,
        dustbin_score=dustbin_score,
        landmark_xyz=initial_xyz,
        raw_anchor_offset=raw_anchor_offset,
    )

    empty_observation_steps = 0
    empty_observation_checkpoint_steps = []
    semidense_reference_features = initial_features.detach().clone()
    protected_set_teacher = None
    if (
        native_observation_mode
        and float(args.native_protected_set_weight) > 0.0
    ):
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
            {
                **mvinit_diagnostics,
                **geometry_support_diagnostics,
                **native_global_attractor_diagnostics,
                **recent,
                **validation,
            },
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
        current_xyz = materialize_anchor(raw_anchor_offset)
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
        else:
            auxiliary_observations = anchor_auxiliary
        auxiliary_scale = _native_anchor_auxiliary_scale(args)
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
            step_native_loss_kwargs = dict(native_loss_kwargs)
            if (
                bool(args.native_rank_budget_mode)
                and bool(args.native_outcome_mode)
                and int(args.native_rank_stage_a_steps) > 0
            ):
                alternation_period = (
                    int(args.native_rank_stage_a_steps)
                    + int(args.native_rank_steps)
                )
                rank_step = (
                    (step - 1) % alternation_period
                    >= int(args.native_rank_stage_a_steps)
                )
                step_native_loss_kwargs["native_rank_budget_mode"] = rank_step
                step_native_loss_kwargs["native_outcome_mode"] = not rank_step
            else:
                rank_step = bool(args.native_rank_budget_mode)
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
                **step_native_loss_kwargs,
            )
            retrieval_loss = retrieval.loss
            retrieval_diagnostics = {
                **retrieval.diagnostics,
                "native_rank_alternation_rank_step": float(rank_step),
                "native_rank_alternation_stage_a_step": float(not rank_step),
            }
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
                **_native_candidate_loss_kwargs(args),
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
        if descriptor_active and functional_replay_bank is not None:
            replay_batch = _sample_functional_replay(
                functional_replay_bank,
                args.functional_replay_batch_size,
                functional_replay_rng,
                features.device,
            )
            functional_replay = protected_functional_replay_loss(
                features,
                replay_batch["query_features"],
                replay_batch["protected_landmark_indices"],
                replay_batch["reference_candidate_indices"],
                replay_batch["reference_candidate_logits"],
                replay_batch["reference_margins"],
                importance=replay_batch["importance"],
                temperature=args.functional_replay_temperature,
                margin_slack=args.functional_replay_margin_slack,
                distribution_weight=(
                    args.functional_replay_distribution_weight
                ),
            )
            functional_replay_loss = functional_replay.loss
            core_mask = replay_batch["pnp_core_mask"].bool()
            core_retention = (
                functional_replay.retained[core_mask].float().mean()
                if bool(core_mask.any().item())
                else functional_replay_loss.new_tensor(0.0)
            )
            functional_replay_diagnostics = {
                **functional_replay.diagnostics,
                "functional_replay_core_row_count": float(
                    core_mask.sum().item()
                ),
                "functional_replay_core_retention": float(
                    core_retention.detach().item()
                ),
            }
        else:
            functional_replay_loss = features.sum() * 0.0
            functional_replay_diagnostics = {
                "functional_replay_active": 0.0
            }
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
            and str(args.observation_source) in {"native", "native_plus_anchor"}
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
                pose_safe_max_delete_gain_m=(
                    args.native_semidense_pose_safe_max_delete_gain_m
                ),
                pose_safe_min_correspondences=(
                    args.native_semidense_pose_safe_min_correspondences
                ),
                pose_safe_teacher_pairs=(
                    args.native_semidense_pose_safe_teacher_pairs
                ),
                lgcv_weight=args.native_semidense_lgcv_weight,
                lgcv_minimum_edge_px=args.native_semidense_lgcv_minimum_edge_px,
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
        if args.local_weight > 0.0 and auxiliary_observations is not None:
            local_loss, local_diagnostics = local_correlation_peak_loss(
                features,
                auxiliary_observations,
                radius=args.local_radius,
                target_sigma=args.local_target_sigma,
                temperature=args.local_temperature,
            )
            local_loss = auxiliary_scale * local_loss
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
            dustbin_loss = auxiliary_scale * dustbin_loss
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
                    depth_abs_tolerance=args.geometry_association_depth_abs_tolerance,
                    depth_rel_tolerance=args.geometry_association_depth_rel_tolerance,
                    landmark_support_mask=geometry_support_mask,
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
            if str(dataset.gaussian_type).lower() == "3dgs":
                surface_loss = prior_geometry.mahalanobis_anchor_prior(
                    current_xyz
                ).mean()
                geometry_diagnostics.update(
                    {
                        "geometry_prior_kind": "gaussian_mahalanobis",
                        "geometry_mahalanobis_loss": float(
                            surface_loss.detach().item()
                        ),
                    }
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
                args.mv_weight * mv_loss
                + global_retrieval_scale
                * args.retrieval_weight
                * retrieval_loss
                + args.generic_proposal_weight * proposal_retrieval_loss
                + args.local_weight * local_loss
                + semidense_weight_scale
                * args.native_semidense_weight
                * native_semidense_loss
                + args.trust_weight * trust_loss
                + args.functional_replay_weight
                * functional_replay_loss
                + args.native_protected_set_weight
                * protected_set_loss
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
        loss_finite = bool(torch.isfinite(loss.detach()).item())
        optimizer.zero_grad(set_to_none=True)
        replay_projection_diagnostics = {
            "functional_replay_gradient_projection_active": 0.0,
            "functional_replay_gradient_conflict_landmarks": 0.0,
            "functional_replay_gradient_conflict_fraction": 0.0,
        }
        promotion_gradient = None
        protection_gradient = None
        replay_projection_active = bool(
            loss_finite
            and descriptor_active
            and functional_replay_bank is not None
            and args.functional_replay_gradient_projection
        )
        if replay_projection_active:
            protection_term = (
                descriptor_scale
                * args.functional_replay_weight
                * functional_replay_loss
            )
            promotion_gradient = torch.autograd.grad(
                loss - protection_term,
                residual,
                retain_graph=True,
                allow_unused=True,
            )[0]
            protection_gradient = torch.autograd.grad(
                protection_term,
                residual,
                retain_graph=True,
                allow_unused=True,
            )[0]
        if loss_finite:
            loss.backward()
        if (
            replay_projection_active
            and promotion_gradient is not None
            and protection_gradient is not None
        ):
            projected_gradient, conflict_mask = (
                per_landmark_gradient_conflict(
                    promotion_gradient.detach(),
                    protection_gradient.detach(),
                )
            )
            combined_gradient = projected_gradient + protection_gradient
            if residual.grad is None:
                residual.grad = combined_gradient
            else:
                residual.grad.copy_(combined_gradient)
            protection_active_landmarks = (
                protection_gradient.detach().reshape(
                    protection_gradient.shape[0], -1
                ).square().sum(dim=1) > 1e-20
            )
            conflict_count = int(conflict_mask.sum().item())
            replay_projection_diagnostics = {
                "functional_replay_gradient_projection_active": 1.0,
                "functional_replay_gradient_conflict_landmarks": float(
                    conflict_count
                ),
                "functional_replay_gradient_conflict_fraction": float(
                    conflict_count
                    / max(
                        int(protection_active_landmarks.sum().item()), 1
                    )
                ),
            }
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
            raw_anchor_offset.clamp_(-float(args.raw_offset_clip), float(args.raw_offset_clip))
            current_xyz = materialize_anchor(raw_anchor_offset)
            displacement = torch.linalg.norm(current_xyz - base_bank_xyz, dim=1)
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
            "geometry_update_active": float(geometry_update_active),
            "loss": loss_value if math.isfinite(loss_value) else 0.0,
            "loss_nonfinite": float(not math.isfinite(loss_value)),
            "mv_loss": float(mv_loss.detach().item()),
            "retrieval_loss": float(retrieval_loss.detach().item()),
            "generic_proposal_loss": float(
                proposal_retrieval_loss.detach().item()
            ),
            "local_loss": float(local_loss.detach().item()),
            "native_semidense_loss": float(
                native_semidense_loss.detach().item()
            ),
            "trust_loss": float(trust_loss.detach().item()),
            "functional_replay_loss": float(
                functional_replay_loss.detach().item()
            ),
            "native_protected_set_loss": float(
                protected_set_loss.detach().item()
            ),
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
            "native_candidate_observations": float(
                str(args.observation_source) != "anchor"
            ),
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
            **native_semidense_diagnostics,
            **protected_set_diagnostics,
            **functional_replay_diagnostics,
            **replay_projection_diagnostics,
            **semidense_gradient_diagnostics,
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
                semi=f"{recent.get('native_semidense_loss', 0.0):.4f}",
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
    final_xyz = materialize_anchor(raw_anchor_offset)
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
    if bool(args.save_landmark_statistics) and int(args.distill_budget) <= 0:
        statistics, landmark_statistics_summary = _collect_landmark_statistics(
            final_features,
            train_names,
            cache,
            final_xyz,
            args,
            visibility_cache=visibility_cache,
            base_bank_xyz=base_bank_xyz,
        )
        independent_geometry_evidence = {}
        if bool(args.save_independent_geometry_teacher):
            if str(args.geometry_teacher_identity_mode) in {
                "track_first",
                "track_first_provenance",
            }:
                provenance_context = None
                if (
                    str(args.geometry_teacher_identity_mode)
                    == "track_first_provenance"
                ):
                    if str(dataset.gaussian_type) != "2dgs":
                        raise ValueError(
                            "Exact splat-provenance geometry assignment is "
                            "currently defined only for 2DGS"
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
                (
                    independent_statistics,
                    independent_geometry_evidence,
                    independent_diagnostics,
                ) = _collect_track_first_geometry_teacher(
                    train_names,
                    cache,
                    final_xyz,
                    args,
                    provenance_context=provenance_context,
                )
            else:
                (
                    independent_statistics,
                    independent_geometry_evidence,
                    independent_diagnostics,
                ) = _collect_independent_geometry_teacher(
                    final_features,
                    train_names,
                    cache,
                    final_xyz,
                    args,
                )
            statistics.update(independent_statistics)
            triangulated_xyz = independent_geometry_evidence[
                "triangulated_xyz"
            ].to(device=final_xyz.device, dtype=final_xyz.dtype)
            triangulated = independent_geometry_evidence["triangulated"].to(
                device=final_xyz.device
            )
            safe_triangulated_xyz = torch.where(
                triangulated[:, None], triangulated_xyz, final_xyz
            )
            current_center_geometry = GaussianPriorGeometry(
                gaussian_type=str(dataset.gaussian_type),
                xyz=final_xyz,
                rotation=rgb_prior_geometry.rotation,
                scaling=rgb_prior_geometry.scaling,
            )
            current_surface_residual = (
                current_center_geometry.surface_residual_components(
                    safe_triangulated_xyz
                )
            )
            rgb_surface_residual = (
                rgb_prior_geometry.surface_residual_components(
                    safe_triangulated_xyz
                )
            )

            def _masked_surface_value(value):
                invalid = torch.full_like(value, float("inf"))
                mask = triangulated
                while mask.ndim < value.ndim:
                    mask = mask.unsqueeze(-1)
                return torch.where(mask, value, invalid).cpu()

            independent_geometry_evidence.update(
                {
                    "triangulation_current_center_offset_m": torch.where(
                        triangulated,
                        torch.linalg.norm(
                            safe_triangulated_xyz - final_xyz, dim=1
                        ),
                        torch.full_like(
                            triangulated_xyz[:, 0], float("inf")
                        ),
                    ).cpu(),
                    "triangulation_rgb_center_offset_m": torch.where(
                        triangulated,
                        torch.linalg.norm(
                            safe_triangulated_xyz - rgb_bank_xyz, dim=1
                        ),
                        torch.full_like(
                            triangulated_xyz[:, 0], float("inf")
                        ),
                    ).cpu(),
                    "triangulation_current_center_mahalanobis": torch.where(
                        triangulated,
                        current_center_geometry.mahalanobis_anchor_prior(
                            safe_triangulated_xyz
                        ),
                        torch.full_like(
                            triangulated_xyz[:, 0], float("inf")
                        ),
                    ).cpu(),
                    "triangulation_rgb_center_mahalanobis": torch.where(
                        triangulated,
                        rgb_prior_geometry.mahalanobis_anchor_prior(
                            safe_triangulated_xyz
                        ),
                        torch.full_like(
                            triangulated_xyz[:, 0], float("inf")
                        ),
                    ).cpu(),
                    **{
                        f"triangulation_current_{name}": (
                            _masked_surface_value(value)
                        )
                        for name, value in current_surface_residual.items()
                    },
                    **{
                        f"triangulation_rgb_{name}": (
                            _masked_surface_value(value)
                        )
                        for name, value in rgb_surface_residual.items()
                    },
                }
            )
            landmark_statistics_summary.update(independent_diagnostics)
        raster_visibility_count = torch.zeros_like(base_bank_opacity)
        if visibility_cache is not None:
            for name in train_names:
                raster_visibility_count.add_(
                    torch.as_tensor(
                        visibility_cache[name],
                        device=base_bank_opacity.device,
                        dtype=base_bank_opacity.dtype,
                    )
                )
        geometry_evidence = {
            "gaussian_type": str(dataset.gaussian_type),
            "opacity": base_bank_opacity.detach().cpu(),
            "scaling": base_bank_scaling.detach().cpu(),
            "rotation": rgb_prior_geometry.rotation.detach().cpu(),
            "planarity": prior_geometry.planarity.detach().cpu(),
            "raster_visibility_count": (
                raster_visibility_count.detach().cpu()
            ),
            "mvinit_observation_count": (
                mvinit_observation_count.detach().cpu()
            ),
            "rgb_center_offset_mahalanobis": (
                rgb_prior_geometry.mahalanobis_anchor_prior(base_bank_xyz)
                .detach()
                .cpu()
            ),
            "rgb_center_offset_m": (
                torch.linalg.norm(
                    base_bank_xyz - rgb_bank_xyz,
                    dim=1,
                )
                .detach()
                .cpu()
            ),
        }
        geometry_evidence.update(independent_geometry_evidence)
        torch.save(
            {
                "version": 2,
                "split": "train_only",
                "train_camera_names_sha256": _camera_names_sha256(train_names),
                "landmark_indices": landmark_indices.detach().cpu(),
                "statistics": {
                    name: value.detach().cpu()
                    for name, value in statistics.items()
                },
                "geometry_evidence": geometry_evidence,
                "diagnostics": dict(landmark_statistics_summary),
            },
            output_dir / "landmark_statistics_full.pt",
        )
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
        torch.save(
            {
                "version": 1,
                "split": "train_only",
                "train_camera_names_sha256": _camera_names_sha256(train_names),
                "landmark_indices": landmark_indices.detach().cpu(),
                "statistics": {
                    name: value.detach().cpu()
                    for name, value in statistics.items()
                },
                "diagnostics": dict(landmark_statistics_summary),
            },
            output_dir / "landmark_statistics_full.pt",
        )
        distill_global_attractor_statistics = None
        distill_global_attractor_diagnostics = {
            "distill_global_attractor_prior_enabled": 0.0,
        }
        if float(args.distill_global_attractor_weight) > 0.0:
            if not native_observation_mode:
                raise ValueError(
                    "distill_global_attractor_weight requires native sparse "
                    "observations"
                )
            (
                distill_global_attractor_statistics,
                distill_global_attractor_diagnostics,
            ) = _collect_native_global_attractor_statistics(
                final_features,
                train_names,
                cache,
                final_xyz,
                args,
                visibility_cache=visibility_cache,
                base_bank_xyz=base_bank_xyz,
                max_observations=args.statistics_observations,
            )
            distill_global_attractor_path = (
                output_dir / "distill_global_attractor_prior.pt"
            )
            torch.save(
                {
                    "version": 1,
                    "split": "train_only",
                    "train_camera_names_sha256": _camera_names_sha256(train_names),
                    "landmark_indices": landmark_indices.detach().cpu(),
                    "statistics": {
                        name: value.detach().cpu()
                        for name, value in distill_global_attractor_statistics.items()
                    },
                    "diagnostics": dict(distill_global_attractor_diagnostics),
                },
                distill_global_attractor_path,
            )
            config["distill_global_attractor_prior"] = {
                "enabled": True,
                "split": "train_only",
                "path": str(distill_global_attractor_path.resolve()),
                "train_camera_names_sha256": _camera_names_sha256(train_names),
            }
        else:
            config["distill_global_attractor_prior"] = {"enabled": False}
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
                global_attractor_statistics=distill_global_attractor_statistics,
            ),
            **distill_global_attractor_diagnostics,
        }
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
            "landmark_statistics": landmark_statistics_summary,
            "distillation": distillation_summary,
            "native_geometry_support": geometry_support_diagnostics,
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
            "dustbin": {
                "enabled": dustbin_score is not None,
                "score": (
                    None
                    if dustbin_score is None
                    else float(dustbin_score.detach().item())
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
                # raw_anchor_offset remains a Parameter so staged runs can
                # reuse one optimizer layout.  It is not trainable in a
                # descriptor-only phase unless a configured loss can update it.
                "bounded_anchor_parameter_requires_grad": bool(
                    raw_anchor_offset.requires_grad
                ),
                "localization_anchor_parameterization": (
                    config["surface_anchor_parameterization"]
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
            **geometry_support_diagnostics,
            **native_global_attractor_diagnostics,
            **_mean_diagnostics(history[-min(len(history), 200):]),
            **final_validation,
        },
        mvinit_observation_count,
        dustbin_score=dustbin_score,
        landmark_xyz=final_xyz,
        raw_anchor_offset=raw_anchor_offset,
    )
    # A fixed-size, non-distilled bank still needs an explicit identity
    # artifact for downstream evaluation.  Do not fabricate utility/prior
    # scores here: this map has not run landmark selection and consumers must
    # not treat it as if it had.  Distilled runs overwrite this file above
    # with their richer selection metadata.
    if int(args.distill_budget) <= 0:
        torch.save(
            {
                "version": 1,
                "landmark_indices": landmark_indices.detach().cpu(),
                "fixed_bank": True,
                "one_time_landmark_distillation": False,
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
        description="Detector-free localization descriptor reconstruction on a fixed Gaussian surface"
    )
    model_params = ModelParams(parser)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--rgb_prior_manifest_path",
        default="",
        help=(
            "Export manifest proving that the frozen Gaussian input does not "
            "contain a localization feature, detector, or prior landmark bank."
        ),
    )
    parser.add_argument(
        "--require_rgb_prior_manifest",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow_feature_stripped_prior",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow a compatibility prior whose geometry/topology was previously "
            "trained with feature loss. It must not be labelled rgb_only."
        ),
    )
    parser.add_argument(
        "--initial_state_geometry_as_base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use an exactly aligned initial state's localization-only xyz as "
            "the base anchor while leaving the frozen RGB Gaussian map intact."
        ),
    )
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
        choices=[
            "file",
            "pure_geometry",
            "protected_union",
            "ulf_consensus",
            "ulf_parity",
            "ulf_robust_consensus",
        ],
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
    parser.add_argument(
        "--scaffold_opacity_keep_quantile",
        type=float,
        default=0.0,
        help=(
            "Optional scene-normalized opacity floor. The effective floor is "
            "max(scaffold_min_opacity, this finite-opacity quantile)."
        ),
    )
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
        "--ulf_consensus_min_visible_views",
        type=int,
        default=0,
        help="Robust KCS: minimum raster-visible support views before selection.",
    )
    parser.add_argument(
        "--ulf_consensus_min_rate",
        type=float,
        default=0.0,
        help="Robust KCS: minimum keypoint-consensus / visible-view rate.",
    )
    parser.add_argument(
        "--ulf_consensus_view_bins",
        type=int,
        default=0,
        help="Robust KCS: camera-center coverage bins; zero disables the gate.",
    )
    parser.add_argument(
        "--ulf_consensus_min_distinct_view_bins",
        type=int,
        default=0,
        help="Robust KCS: distinct voting camera-center bins required per landmark.",
    )
    parser.add_argument(
        "--ulf_consensus_trajectory_bins",
        type=int,
        default=0,
        help="Robust KCS: chronological bins per camera trajectory; zero disables it.",
    )
    parser.add_argument(
        "--ulf_consensus_min_distinct_trajectory_bins",
        type=int,
        default=0,
        help="Robust KCS: distinct voting trajectory bins required per landmark.",
    )
    parser.add_argument(
        "--ulf_consensus_independent_bin_scoring",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use independent camera-center bins (or trajectory bins when camera "
            "bins are disabled) for KCS votes, visibility rates, and ranking "
            "instead of counting correlated frames repeatedly."
        ),
    )
    parser.add_argument(
        "--ulf_consensus_allow_nonconsensus_fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Allow robust KCS to fall back to opacity-only primitives when gates "
            "cannot fill the budget. Defaults to false for robust mode."
        ),
    )
    parser.add_argument(
        "--ulf_consensus_knn",
        type=int,
        default=32,
        help="3D kNN size for strict ULF-parity random seed selection.",
    )
    parser.add_argument(
        "--ulf_consensus_max_views",
        type=int,
        default=0,
        help="Zero uses every support camera; otherwise uniformly subsample views.",
    )
    parser.add_argument(
        "--ulf_support_view_sampling",
        choices=["uniform", "pose_diverse"],
        default="uniform",
        help=(
            "Select ULF-parity KCS/GWFF support views uniformly or by "
            "deterministic camera-center farthest-point sampling."
        ),
    )
    parser.add_argument("--ulf_consensus_distance_chunk", type=int, default=8192)
    parser.add_argument(
        "--ulf_consensus_max_candidates_per_view",
        type=int,
        default=0,
        help="Optional opacity-ranked cap for KCS; zero preserves all visible primitives.",
    )
    parser.add_argument("--ulf_consensus_voxel_size", type=float, default=0.0)
    parser.add_argument(
        "--ulf_consensus_extent_quantile",
        type=float,
        default=0.0,
        help=(
            "Symmetric quantile trimmed from each side when deriving the "
            "automatic KCS coverage voxel size. Zero preserves min/max extent."
        ),
    )
    parser.add_argument("--ulf_consensus_max_per_voxel", type=int, default=8)
    parser.add_argument(
        "--ulf_support_mask_policy",
        choices=["deployment_post_filter", "support_rgb_only"],
        default="deployment_post_filter",
        help=(
            "Support KCS/GWFF mask semantics: deployment filters post-detection "
            "keypoints, while support_rgb_only preserves ULF RGB-only KCS behavior."
        ),
    )
    parser.add_argument(
        "--ulf_parity_kcs_mask_policy",
        choices=["rgb_only", "deployment_post_filter"],
        default="rgb_only",
        help=(
            "Strict ULF KCS support-mask semantics: rgb_only preserves the "
            "reference behavior; deployment_post_filter applies the deployed "
            "valid mask to detected keypoints and projected primitives."
        ),
    )
    parser.add_argument(
        "--initialization_mode",
        choices=["mvinit", "ulf_geometry", "ulf_parity", "ulf_robust_geometry"],
        default="ulf_geometry",
        help="Descriptor initializer for a newly built landmark bank.",
    )
    parser.add_argument(
        "--ulf_fusion_max_views",
        type=int,
        default=0,
        help="Zero fuses every support camera; otherwise uniformly subsample views.",
    )
    parser.add_argument(
        "--ulf_fusion_view_bins",
        type=int,
        default=0,
        help=(
            "Equalize robust GWFF camera contribution across deterministic "
            "camera-center bins; zero preserves frame-weighted fusion."
        ),
    )
    parser.add_argument(
        "--ulf_fusion_exact_bin_balance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Fuse each landmark within each observed camera-center bin first, "
            "then average those per-bin prototypes equally."
        ),
    )
    parser.add_argument("--ulf_fusion_min_cosine", type=float, default=0.0)
    parser.add_argument(
        "--ulf_fusion_descriptor_min_cosine",
        type=float,
        default=-1.0,
        help=(
            "Robust GWFF: descriptor cosine lower bound relative to the first "
            "geometry-weighted prototype."
        ),
    )
    parser.add_argument(
        "--ulf_fusion_descriptor_trim_fraction",
        type=float,
        default=0.0,
        help="Robust GWFF: per-landmark bottom cosine fraction removed after prototype fusion.",
    )
    parser.add_argument(
        "--ulf_fusion_adaptive_trim",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use each landmark's streaming low-cosine tail rate to choose a "
            "bounded GWFF trim fraction instead of one global fraction."
        ),
    )
    parser.add_argument(
        "--ulf_fusion_adaptive_trim_min_fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--ulf_fusion_adaptive_trim_max_fraction", type=float, default=0.20
    )
    parser.add_argument(
        "--ulf_fusion_adaptive_trim_tail_cosine", type=float, default=0.75
    )
    parser.add_argument(
        "--ulf_fusion_adaptive_trim_min_observations", type=int, default=4
    )
    parser.add_argument(
        "--ulf_fusion_adaptive_trim_mode",
        choices=["absolute", "relative_mad"],
        default="absolute",
        help=(
            "Use an absolute cosine tail for ablations or each landmark's "
            "median/MAD-normalized tail for robust GWFF."
        ),
    )
    parser.add_argument("--ulf_fusion_adaptive_trim_mad_scale", type=float, default=2.5)
    parser.add_argument(
        "--ulf_fusion_reference_mode",
        choices=["mean", "weighted_cosine_medoid"],
        default="mean",
        help=(
            "Robust GWFF reference before descriptor trimming. "
            "weighted_cosine_medoid is the exact streaming medoid under the "
            "geometry-weighted cosine objective."
        ),
    )
    parser.add_argument(
        "--ulf_fusion_trim_histogram_bins",
        type=int,
        default=64,
        help="Robust GWFF: streaming cosine histogram resolution.",
    )
    parser.add_argument(
        "--ulf_parity_fusion_channel_chunk",
        type=int,
        default=32,
        help=(
            "Channels processed at once when exactly sampling ULF's full-resolution "
            "upsampled dense map; does not change the fusion formula."
        ),
    )
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
    parser.add_argument("--functional_replay_weight", type=float, default=0.0)
    parser.add_argument("--functional_replay_cache_path", default="")
    parser.add_argument(
        "--functional_replay_rows_per_query", type=int, default=64
    )
    parser.add_argument(
        "--functional_replay_core_rows_per_query", type=int, default=16
    )
    parser.add_argument("--functional_replay_batch_size", type=int, default=256)
    parser.add_argument("--functional_replay_topm", type=int, default=64)
    parser.add_argument(
        "--functional_replay_temperature", type=float, default=0.05
    )
    parser.add_argument(
        "--functional_replay_margin_slack", type=float, default=0.005
    )
    parser.add_argument(
        "--functional_replay_distribution_weight", type=float, default=1.0
    )
    parser.add_argument(
        "--functional_replay_pnp_core_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--functional_replay_build_pnp_core", action="store_true"
    )
    parser.add_argument(
        "--functional_replay_gradient_projection", action="store_true"
    )
    parser.add_argument("--functional_replay_grid_rows", type=int, default=4)
    parser.add_argument("--functional_replay_grid_cols", type=int, default=4)
    parser.add_argument("--functional_replay_depth_bins", type=int, default=4)
    parser.add_argument(
        "--functional_replay_surface_voxel_m", type=float, default=1.0
    )
    parser.add_argument(
        "--functional_replay_max_per_surface_group", type=int, default=4
    )
    parser.add_argument(
        "--functional_replay_ransac_reprojection_px",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--functional_replay_ransac_max_iterations", type=int, default=100000
    )
    parser.add_argument(
        "--functional_replay_ransac_min_iterations", type=int, default=1000
    )
    parser.add_argument(
        "--functional_replay_max_pose_error_cm", type=float, default=100.0
    )
    parser.add_argument("--trust_observation_power", type=float, default=0.5)
    parser.add_argument("--trust_weight_min", type=float, default=0.25)
    parser.add_argument("--trust_weight_max", type=float, default=4.0)
    parser.add_argument("--local_weight", type=float, default=0.05)
    parser.add_argument("--local_radius", type=int, default=3)
    parser.add_argument("--local_target_sigma", type=float, default=1.0)
    parser.add_argument("--local_temperature", type=float, default=0.07)
    parser.add_argument(
        "--native_semidense_weight",
        type=float,
        default=0.0,
        help=(
            "Training-only weight for local peak distillation seeded by "
            "GT-clean native top-1 matches. Deployment remains one sparse "
            "top-1 retrieval and one RANSAC/PnP."
        ),
    )
    parser.add_argument("--native_semidense_start_step", type=int, default=2500)
    parser.add_argument("--native_semidense_interval", type=int, default=1)
    parser.add_argument("--native_semidense_max_anchors", type=int, default=64)
    parser.add_argument("--native_semidense_neighbors", type=int, default=1)
    parser.add_argument(
        "--native_semidense_neighborhood_radius_m", type=float, default=0.25
    )
    parser.add_argument(
        "--native_semidense_normal_cosine", type=float, default=0.8
    )
    parser.add_argument(
        "--native_semidense_local_radius_px", type=int, default=8
    )
    parser.add_argument(
        "--native_semidense_target_sigma_px", type=float, default=2.0
    )
    parser.add_argument(
        "--native_semidense_temperature", type=float, default=0.07
    )
    parser.add_argument(
        "--native_semidense_pose_safe_max_delete_gain_m",
        type=float,
        default=-1.0,
        help=(
            "Reject a GT-clean semidense seed when deleting it improves the "
            "linearized translation bias by more than this many metres. A "
            "negative value disables the detached counterfactual gate."
        ),
    )
    parser.add_argument(
        "--native_semidense_pose_safe_min_correspondences",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--native_semidense_pose_safe_teacher_pairs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the counterfactual translation-bias gate again to every "
            "expanded semidense soft correspondence, not only its sparse seed."
        ),
    )
    parser.add_argument(
        "--native_semidense_lgcv_weight",
        type=float,
        default=0.0,
        help=(
            "Relative training-only B3 group geometry weight inside the "
            "semidense teacher."
        ),
    )
    parser.add_argument(
        "--native_semidense_lgcv_minimum_edge_px",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--native_semidense_protected_v2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Route local supervision only to measurement-limited, same-surface "
            "matches and preserve high-precision global assignment margins."
        ),
    )
    parser.add_argument(
        "--native_semidense_measurement_min_reprojection_px",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--native_semidense_measurement_max_reprojection_px",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--native_semidense_surface_point_plane_m",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--native_semidense_surface_max_distance_m",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--native_semidense_surface_normal_cosine",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--native_semidense_projected_neighbor_radius_px",
        type=float,
        default=64.0,
    )
    parser.add_argument(
        "--native_semidense_local_identity_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--native_semidense_margin_preservation_weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--native_semidense_gradient_audit",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--native_semidense_reference_refresh_steps",
        type=int,
        default=500,
        help="Refresh the frozen semidense descriptor teacher at this interval.",
    )
    parser.add_argument(
        "--native_semidense_alternate_global",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use semidense-only updates on its scheduled steps; the remaining "
            "steps retain the global assignment objective."
        ),
    )
    parser.add_argument(
        "--native_semidense_max_gradient_ratio",
        type=float,
        default=0.0,
        help=(
            "Cap the weighted semidense descriptor-gradient norm to this "
            "fraction of the global retrieval gradient; zero disables."
        ),
    )
    parser.add_argument("--native_protected_set_weight", type=float, default=0.0)
    parser.add_argument("--native_protected_set_start_step", type=int, default=1000)
    parser.add_argument("--native_protected_set_interval", type=int, default=5)
    parser.add_argument(
        "--native_protected_set_refresh_visits", type=int, default=1
    )
    parser.add_argument(
        "--native_protected_set_ransac_seed", type=int, default=0
    )
    parser.add_argument(
        "--native_protected_set_ransac_reprojection_px",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--native_protected_set_ransac_max_iterations",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--native_protected_set_ransac_min_iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--native_protected_set_max_pose_error_cm",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--native_protected_set_max_useful", type=int, default=96
    )
    parser.add_argument(
        "--native_protected_set_max_harmful", type=int, default=96
    )
    parser.add_argument(
        "--native_protected_set_grid_rows", type=int, default=4
    )
    parser.add_argument(
        "--native_protected_set_grid_cols", type=int, default=4
    )
    parser.add_argument(
        "--native_protected_set_depth_bins", type=int, default=4
    )
    parser.add_argument(
        "--native_protected_set_surface_voxel_m", type=float, default=0.25
    )
    parser.add_argument(
        "--native_protected_set_max_per_surface_group",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--native_protected_set_temperature", type=float, default=0.05
    )
    parser.add_argument(
        "--native_protected_set_margin", type=float, default=0.05
    )
    parser.add_argument(
        "--native_protected_set_score_target", type=float, default=0.5
    )
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
    parser.add_argument(
        "--native_outcome_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace the aggregate hard-candidate objective for native SuperPoint "
            "proposals with explicit keep/swap/miss/reject ranking losses."
        ),
    )
    parser.add_argument("--native_nce_weight", type=float, default=0.0)
    parser.add_argument("--native_keep_weight", type=float, default=1.0)
    parser.add_argument("--native_keep_margin", type=float, default=0.05)
    parser.add_argument("--native_keep_loose_weight", type=float, default=0.0)
    parser.add_argument("--native_keep_loose_radius_px", type=float, default=4.0)
    parser.add_argument("--native_keep_loose_margin", type=float, default=0.025)
    parser.add_argument("--native_swap_weight", type=float, default=1.0)
    parser.add_argument("--native_swap_margin", type=float, default=0.05)
    parser.add_argument("--native_miss_weight", type=float, default=1.0)
    parser.add_argument("--native_miss_margin", type=float, default=0.05)
    parser.add_argument("--native_reject_weight", type=float, default=0.0)
    parser.add_argument("--native_reject_threshold", type=float, default=0.5)
    parser.add_argument("--native_attractor_weight", type=float, default=0.0)
    parser.add_argument("--native_attractor_margin", type=float, default=0.05)
    parser.add_argument(
        "--native_global_attractor_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a frozen train-only per-landmark false-attractor prior; "
            "it preserves the deployed candidate set and only reweights ranking."
        ),
    )
    parser.add_argument(
        "--native_rank_budget_mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace the native residual objective with exact multi-positive "
            "rank-budget curriculum supervision."
        ),
    )
    parser.add_argument(
        "--native_rank_stage_a_steps",
        type=int,
        default=0,
        help=(
            "When rank and native outcome modes are both enabled, run this "
            "many Stage-A steps before each rank block."
        ),
    )
    parser.add_argument(
        "--native_rank_steps",
        type=int,
        default=1,
        help="Rank-promotion steps per Stage-A/rank alternation period.",
    )
    parser.add_argument("--native_rank_temperature", type=float, default=0.03)
    parser.add_argument("--native_rank_margin_at1", type=float, default=0.02)
    parser.add_argument("--native_rank_margin_at4", type=float, default=0.02)
    parser.add_argument("--native_rank_margin_at8", type=float, default=0.02)
    parser.add_argument("--native_rank_margin_at32", type=float, default=0.02)
    parser.add_argument("--native_rank_top1_weight", type=float, default=0.25)
    parser.add_argument("--native_rank_keep_weight", type=float, default=1.0)
    parser.add_argument(
        "--native_rank_reference_clean_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--native_rank_reference_clean_margin", type=float, default=0.02
    )
    parser.add_argument(
        "--native_rank_landmark_balance",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--native_rank_band_rank1", type=float, default=0.25)
    parser.add_argument("--native_rank_band_rank2_4", type=float, default=0.25)
    parser.add_argument("--native_rank_band_rank5_32", type=float, default=0.30)
    parser.add_argument("--native_rank_band_rank33_plus", type=float, default=0.20)
    parser.add_argument(
        "--native_global_attractor_min_incoming", type=int, default=4
    )
    parser.add_argument(
        "--native_global_attractor_support_power", type=float, default=0.5
    )
    parser.add_argument(
        "--native_global_attractor_max_score", type=float, default=4.0
    )
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
    parser.add_argument(
        "--covariance_anchor_scale",
        type=float,
        default=0.5,
        help="Per-axis 3DGS localization-anchor radius as a scale multiple.",
    )
    parser.add_argument(
        "--covariance_anchor_absolute_bound_m",
        type=float,
        default=0.03,
        help="Absolute per-axis cap for covariance-bounded 3DGS anchors.",
    )
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
    parser.add_argument(
        "--geometry_association_depth_abs_tolerance",
        type=float,
        default=0.0,
        help=(
            "Absolute metric-depth gate for native BA associations in metres; "
            "zero together with the relative tolerance preserves the legacy "
            "reprojection-only association gate."
        ),
    )
    parser.add_argument(
        "--geometry_association_depth_rel_tolerance",
        type=float,
        default=0.0,
        help=(
            "Relative metric-depth gate for native BA associations; enable "
            "with a positive absolute or relative tolerance."
        ),
    )
    parser.add_argument(
        "--geometry_association_min_support_views",
        type=int,
        default=3,
        help=(
            "Distinct GT-clean native support cameras required before a "
            "landmark may receive a bounded BA update."
        ),
    )
    parser.add_argument(
        "--geometry_association_support_observations",
        type=int,
        default=0,
        help=(
            "Native proposals inspected per support view in the fixed BA "
            "qualification pass; zero reuses --max_observations."
        ),
    )
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
        "--save_landmark_statistics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Persist train-only localization and Gaussian-geometry evidence "
            "without changing the deployed landmark bank."
        ),
    )
    parser.add_argument(
        "--save_independent_geometry_teacher",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "During an offline statistics sweep, triangulate descriptor-only "
            "cross-view native associations and save query-level coverage."
        ),
    )
    parser.add_argument(
        "--geometry_teacher_min_similarity", type=float, default=0.7
    )
    parser.add_argument(
        "--geometry_teacher_min_margin", type=float, default=0.03
    )
    parser.add_argument(
        "--geometry_teacher_max_observations_per_landmark",
        type=int,
        default=32,
    )
    parser.add_argument("--geometry_teacher_min_views", type=int, default=3)
    parser.add_argument(
        "--geometry_teacher_view_bins", type=int, default=8
    )
    parser.add_argument(
        "--geometry_teacher_min_view_bins", type=int, default=2
    )
    parser.add_argument(
        "--geometry_teacher_huber_delta_px", type=float, default=2.0
    )
    parser.add_argument(
        "--geometry_teacher_iterations", type=int, default=3
    )
    parser.add_argument(
        "--geometry_teacher_min_parallax_deg", type=float, default=1.0
    )
    parser.add_argument(
        "--geometry_teacher_max_reprojection_px", type=float, default=2.0
    )
    parser.add_argument(
        "--geometry_teacher_max_condition_number", type=float, default=1e6
    )
    parser.add_argument(
        "--geometry_teacher_identity_mode",
        choices=[
            "map_top1",
            "gt_clean_map_top1",
            "track_first",
            "track_first_provenance",
        ],
        default="map_top1",
        help=(
            "G0 map top-1, G1 GT-clean map top-1, or G2 reciprocal "
            "epipolar/cycle image tracks built before Gaussian assignment."
        ),
    )
    parser.add_argument(
        "--geometry_teacher_view_direction_weight", type=float, default=0.5
    )
    parser.add_argument(
        "--geometry_teacher_parallax_quantile", type=float, default=0.75
    )
    parser.add_argument(
        "--geometry_teacher_max_covariance_trace_m2",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--geometry_teacher_max_rendered_depth_residual_m",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--geometry_teacher_min_rendered_depth_observations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--geometry_teacher_track_pair_neighbors", type=int, default=6
    )
    parser.add_argument(
        "--geometry_teacher_track_min_baseline_m", type=float, default=0.03
    )
    parser.add_argument(
        "--geometry_teacher_track_max_baseline_m", type=float, default=5.0
    )
    parser.add_argument(
        "--geometry_teacher_track_max_axis_angle_deg",
        type=float,
        default=75.0,
    )
    parser.add_argument(
        "--geometry_teacher_track_min_similarity", type=float, default=0.65
    )
    parser.add_argument(
        "--geometry_teacher_track_min_margin", type=float, default=0.01
    )
    parser.add_argument(
        "--geometry_teacher_track_max_epipolar_error_px",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--geometry_teacher_track_epipolar_candidate_topk",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--geometry_teacher_track_epipolar_recovered_min_similarity",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--geometry_teacher_track_epipolar_recovered_min_margin",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--geometry_teacher_track_require_cycle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--geometry_teacher_track_allow_chain_tracks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Add reciprocal epipolar chain edges after cycle edges using a "
            "query-conflict-aware union. Cycle-seeded tracks are level A and "
            "pure-chain tracks are level B."
        ),
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv_neighbors", type=int, default=8
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv_support_threshold",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv_angle_cosine",
        type=float,
        default=0.9659,
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv_scale_threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv_scale_limit",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv_maximum_edge_px",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv_minimum_matches",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv_mode",
        choices=["hard", "soft"],
        default="hard",
    )
    parser.add_argument(
        "--geometry_teacher_track_lgcv_confidence_floor",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--geometry_teacher_track_assignment_max_distance_m",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--geometry_teacher_track_assignment_min_margin_m",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--geometry_teacher_provenance_topk", type=int, default=4
    )
    parser.add_argument(
        "--geometry_teacher_provenance_min_consensus_rate",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--geometry_teacher_provenance_min_views", type=int, default=2
    )
    parser.add_argument(
        "--geometry_teacher_provenance_group_max_landmarks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--geometry_teacher_provenance_group_min_relative_mass",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--geometry_teacher_provenance_group_min_consensus_rate",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--geometry_teacher_provenance_depth_abs_tolerance_m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--geometry_teacher_provenance_depth_rel_tolerance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--distill_require_exact_budget",
        action="store_true",
        help=(
            "Fail instead of silently shrinking the final distilled bank when "
            "the requested fixed landmark budget is not observed."
        ),
    )
    parser.add_argument(
        "--distill_allow_coverage_fill",
        action="store_true",
        help=(
            "When an exact final bank is requested but the strict "
            "matchability pool is short, retain that pool and explicitly fill "
            "only the shortage with coverage-ranked source-bank landmarks."
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
    parser.add_argument(
        "--statistics_responsibility_sigma_px",
        type=float,
        default=1.0,
        help=(
            "Gaussian reprojection scale used to distribute source-side "
            "landmark statistics across every legal CSR positive."
        ),
    )
    parser.add_argument("--distill_min_observations", type=int, default=2)
    parser.add_argument(
        "--distill_matchability_threshold", type=float, default=0.5
    )
    parser.add_argument("--distill_false_top1_max", type=float, default=0.5)
    parser.add_argument(
        "--distill_rescue_max_positives",
        type=int,
        default=4,
        help=(
            "Treat native rows with at most this many CSR positives as scarce "
            "coverage opportunities for the reserve bank."
        ),
    )
    parser.add_argument(
        "--distill_rescue_weight",
        type=float,
        default=0.0,
        help="Weight query-level scarce-positive rescue utility in coverage selection.",
    )
    parser.add_argument(
        "--distill_harmful_switch_weight",
        type=float,
        default=0.0,
        help=(
            "Exponent for cross-surface harmful-switch reliability in the "
            "distinctiveness core; zero disables the penalty."
        ),
    )
    parser.add_argument(
        "--distill_global_attractor_weight",
        type=float,
        default=0.0,
        help=(
            "Train-only target-side false-attractor penalty used only when "
            "ranking a distilled fixed bank; it never filters deployment pairs."
        ),
    )
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
    parser.add_argument(
        "--distill_hard_matchability_core_ratio",
        type=float,
        default=0.0,
        help=(
            "Reserve this fraction of the strict distilled bank as the exact "
            "top-matchability core before coverage/FIM selection."
        ),
    )
    parser.add_argument(
        "--distill_protected_core_ratio",
        type=float,
        default=0.0,
        help=(
            "Reserve this fraction of the final bank for train-only, "
            "high-precision landmarks with stable cross-view top-1 identity."
        ),
    )
    parser.add_argument("--distill_protected_min_correct", type=int, default=3)
    parser.add_argument(
        "--distill_protected_matchability", type=float, default=0.75
    )
    parser.add_argument(
        "--distill_protected_identity_switch_max", type=float, default=0.25
    )
    parser.add_argument(
        "--distill_quality_reservoir_multiplier",
        type=float,
        default=0.0,
        help=(
            "When positive, restrict hard-core and coverage selection to the "
            "top observed-native matchability reservoir of this many final "
            "bank budgets. Zero preserves the legacy explicit-fill policy."
        ),
    )
    parser.add_argument(
        "--distill_quality_reservoir_score",
        choices=["posterior_mean", "wilson_lower"],
        default="posterior_mean",
        help=(
            "Reliability score used to form the observed-native quality "
            "reservoir. Wilson lower confidence avoids one-view posterior "
            "inflation."
        ),
    )
    parser.add_argument(
        "--distill_quality_reservoir_wilson_z",
        type=float,
        default=1.96,
        help="Evidence calibration coefficient for the Wilson reservoir score.",
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
    parser.add_argument(
        "--max_observations",
        type=int,
        default=2048,
        help=(
            "Native proposals inspected per training view. The formal sparse "
            "protocol uses all 2048 deployment SuperPoint proposals."
        ),
    )
    parser.add_argument(
        "--validation_observations",
        type=int,
        default=2048,
        help="Native proposals inspected per held-out validation view.",
    )
    parser.add_argument("--grid_rows", type=int, default=8)
    parser.add_argument("--grid_cols", type=int, default=8)
    parser.add_argument("--depth_bins", type=int, default=4)
    parser.add_argument("--alpha_threshold", type=float, default=0.2)
    parser.add_argument("--depth_abs_tolerance", type=float, default=1e-3)
    parser.add_argument("--depth_rel_tolerance", type=float, default=0.01)
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument(
        "--split_mode",
        choices=[
            "random",
            "sequence_block",
            "temporal_block",
            "stratified_temporal_block",
        ],
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
