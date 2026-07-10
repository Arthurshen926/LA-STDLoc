from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from localization_training.correspondence import bilinear_sample_features
from localization_training.direct_landmark_teacher import (
    filter_depth_consistent_landmarks,
    gaussian_localization_xyz,
    make_intrinsics_from_fov,
    project_landmarks_to_query,
)
from localization_training.pose_refiner import project_points, weighted_gauss_newton_refine


@dataclass
class MultiViewInitResult:
    features: torch.Tensor
    reliability: torch.Tensor
    observation_count: torch.Tensor
    weight_sum: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


@dataclass
class MultiViewInitConfig:
    min_observations: int = 1
    alpha_threshold: float = 0.2
    depth_abs_tolerance: float = 1e-3
    depth_rel_tolerance: float = 0.01
    chunk_size: int = 0
    eps: float = 1e-6


def select_multiview_init_cameras(cameras, max_views=0, mode="uniform"):
    """Select support views for MVInit without biasing toward the first sequence."""
    cameras = list(cameras)
    max_views = int(max_views)
    if max_views == 0 or not cameras:
        return []
    if max_views < 0 or max_views >= len(cameras):
        return cameras
    mode = str(mode or "uniform").lower()
    if mode == "first":
        return cameras[:max_views]
    if mode != "uniform":
        raise ValueError(f"Unsupported MVInit view selection mode: {mode}")
    if max_views == 1:
        return [cameras[0]]
    last = len(cameras) - 1
    indices = [int((i * last) / (max_views - 1) + 0.5) for i in range(max_views)]
    return [cameras[index] for index in indices]


@dataclass
class SoftCorrespondenceOutput:
    uv: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    peak_probability: torch.Tensor
    margin: torch.Tensor
    probabilities: torch.Tensor
    valid: torch.Tensor


@dataclass
class DifferentiablePnPConfig:
    temperature: float = 0.07
    min_correspondences: int = 6
    confidence_threshold: float = 0.0
    pnp_iterations: int = 3
    damping: float = 1e-3
    pose_weight: float = 1.0
    reprojection_weight: float = 0.1
    gt_reprojection_weight: float = 1.0
    entropy_weight: float = 0.0
    translation_weight: float = 1.0
    rotation_weight: float = 1.0
    allow_geometry_grad: bool = False
    detach_query_feature_map: bool = True
    local_window_radius: float = 0.0
    max_correspondences: int = 0
    spatial_grid_size: int = 0
    min_spatial_span: float = 0.0
    min_spatial_area: float = 0.0
    point_weight_floor: float = 0.0
    max_condition_number: float = -1.0
    reprojection_loss_type: str = "smooth_l1"
    reprojection_loss_delta: float = 1.0
    geometry_reprojection_weight: float = 0.0
    geometry_depth_anchor_weight: float = 0.0
    geometry_confidence_threshold: float = 0.0
    geometry_margin_threshold: float = 0.0
    geometry_peak_probability_threshold: float = 0.0
    geometry_max_entropy: float = 0.0
    geometry_max_reprojection_error: float = 0.0
    geometry_use_all_correspondences: bool = False
    geometry_local_window_radius: float = 0.0
    geometry_match_reprojection_weight: float = 0.0
    geometry_match_confidence_threshold: float = -1.0
    geometry_match_margin_threshold: float = -1.0
    geometry_match_peak_probability_threshold: float = -1.0
    geometry_match_max_entropy: float = -1.0
    geometry_match_max_reprojection_error: float = -1.0
    geometry_pose_guard_max_loss_increase: float = -1.0
    geometry_pose_guard_max_loss: float = -1.0
    geometry_pose_guard_softness: float = 0.0
    geometry_pose_guard_min_scale: float = 0.0
    feedback_pose_guard_max_loss_increase: float = -1.0
    feedback_pose_guard_max_loss: float = -1.0
    feedback_pose_guard_softness: float = 0.0
    feedback_pose_guard_min_scale: float = 0.0
    feedback_pose_guard_keep_gt_reprojection: bool = False
    detach_gt_reprojection_points: bool = False
    detach_pnp_points: bool = False


@dataclass
class DifferentiablePnPOutput:
    loss: torch.Tensor
    pose_loss: torch.Tensor
    reprojection_loss: torch.Tensor
    gt_reprojection_loss: torch.Tensor
    geometry_reprojection_loss: torch.Tensor
    entropy_loss: torch.Tensor
    pose_w2c: torch.Tensor
    correspondences: SoftCorrespondenceOutput
    used_correspondences: int
    geometry_match_reprojection_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    geometry_depth_anchor_loss: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0))
    diagnostics: dict = field(default_factory=dict)


@dataclass
class LaFGSCurriculumConfig:
    mv_init_until: int = 0
    locrec_until: int = 5_000
    diff_pnp_until: int = 10_000
    geometry_until: int = 15_000


def _as_feature_matrix(features):
    if features.dim() == 1:
        features = features.reshape(1, -1)
    return features.reshape(features.shape[0], -1)


def _pose_guard_check(pose_loss, init_pose_loss, max_loss_increase=-1.0, max_loss=-1.0):
    max_loss_increase = float(max_loss_increase)
    max_loss = float(max_loss)
    enabled = max_loss_increase >= 0.0 or max_loss >= 0.0
    passed = True
    pose_loss_detached = pose_loss.detach()
    if max_loss_increase >= 0.0:
        passed = passed and bool(
            (
                pose_loss_detached
                <= init_pose_loss.detach() + pose_loss.new_tensor(max_loss_increase)
            ).item()
        )
    if max_loss >= 0.0:
        passed = passed and bool((pose_loss_detached <= pose_loss.new_tensor(max_loss)).item())
    return enabled, passed


def _pose_guard_violation(pose_loss, init_pose_loss, max_loss_increase=-1.0, max_loss=-1.0):
    max_loss_increase = float(max_loss_increase)
    max_loss = float(max_loss)
    enabled = max_loss_increase >= 0.0 or max_loss >= 0.0
    violation = pose_loss.detach().new_zeros(())
    pose_loss_detached = pose_loss.detach()
    if max_loss_increase >= 0.0:
        allowed = init_pose_loss.detach() + pose_loss.new_tensor(max_loss_increase)
        violation = torch.maximum(violation, pose_loss_detached - allowed.detach())
    if max_loss >= 0.0:
        allowed = pose_loss.new_tensor(max_loss)
        violation = torch.maximum(violation, pose_loss_detached - allowed)
    return torch.clamp_min(violation, 0.0), enabled


def _pose_guard_soft_scale(
    pose_loss,
    init_pose_loss,
    max_loss_increase=-1.0,
    max_loss=-1.0,
    softness=0.0,
    min_scale=0.0,
):
    enabled, passed = _pose_guard_check(
        pose_loss,
        init_pose_loss,
        max_loss_increase=max_loss_increase,
        max_loss=max_loss,
    )
    if (not enabled) or passed:
        return enabled, passed, pose_loss.new_tensor(1.0), pose_loss.new_tensor(0.0)
    softness = float(softness)
    if softness <= 0.0:
        return enabled, passed, pose_loss.new_tensor(0.0), pose_loss.new_tensor(0.0)
    violation, _ = _pose_guard_violation(
        pose_loss,
        init_pose_loss,
        max_loss_increase=max_loss_increase,
        max_loss=max_loss,
    )
    scale = torch.exp(-violation.detach() / pose_loss.new_tensor(max(softness, 1e-8)))
    min_scale = max(0.0, min(float(min_scale), 1.0))
    if min_scale > 0.0:
        scale = torch.maximum(scale, pose_loss.new_tensor(min_scale))
    return enabled, passed, torch.clamp(scale, 0.0, 1.0), violation


def aggregate_multiview_descriptors(observations, weights=None, valid=None, config=None):
    """Aggregate projected image descriptors into per-Gaussian descriptors.

    Args:
        observations: Tensor shaped [views, gaussians, channels].
        weights: Optional tensor shaped [views, gaussians].
        valid: Optional visibility/depth/alpha mask shaped [views, gaussians].
    """
    config = config or MultiViewInitConfig()
    observations = torch.as_tensor(observations).float()
    if observations.dim() != 3:
        raise ValueError("observations must have shape [views, gaussians, channels]")
    views, gaussians, _ = observations.shape
    device = observations.device
    dtype = observations.dtype

    if weights is None:
        weights = torch.ones((views, gaussians), dtype=dtype, device=device)
    else:
        weights = torch.as_tensor(weights, dtype=dtype, device=device).reshape(views, gaussians)
    if valid is None:
        valid = weights > 0
    else:
        valid = torch.as_tensor(valid, dtype=torch.bool, device=device).reshape(views, gaussians)

    weights = weights.clamp_min(0.0) * valid.to(dtype)
    observation_count = valid.sum(dim=0).to(dtype=torch.long)
    weight_sum = weights.sum(dim=0)
    weighted_sum = (observations * weights[..., None]).sum(dim=0)
    norm = torch.linalg.norm(weighted_sum, dim=-1, keepdim=True)
    features = torch.where(
        norm > float(config.eps),
        weighted_sum / norm.clamp_min(float(config.eps)),
        torch.zeros_like(weighted_sum),
    )

    obs_norm = F.normalize(observations, p=2, dim=-1)
    cosine = (obs_norm * features[None]).sum(dim=-1).clamp_min(0.0)
    reliability = (cosine * weights).sum(dim=0) / weight_sum.clamp_min(float(config.eps))
    reliability = torch.where(
        observation_count >= int(config.min_observations),
        reliability,
        torch.zeros_like(reliability),
    )
    reliability = torch.where(torch.isfinite(reliability), reliability, torch.zeros_like(reliability))
    diagnostics = {
        "view_count": int(views),
        "gaussian_count": int(gaussians),
        "observed_gaussians": int((observation_count > 0).sum().item()),
        "mean_observations": float(observation_count.float().mean().item()) if gaussians else 0.0,
    }
    return MultiViewInitResult(
        features=features,
        reliability=reliability,
        observation_count=observation_count,
        weight_sum=weight_sum,
        diagnostics=diagnostics,
    )


def _camera_pose_w2c(camera, device, dtype):
    pose = getattr(camera, "pose_w2c", None)
    if pose is None:
        pose = getattr(camera, "world_view_transform", None)
        if pose is None:
            raise ValueError("camera must expose pose_w2c or world_view_transform")
        pose = pose.transpose(0, 1)
    return torch.as_tensor(pose, device=device, dtype=dtype)


def _resolve_view_item(source, index, camera):
    if source is None:
        return None
    if callable(source):
        return source(camera)
    if isinstance(source, dict):
        key_candidates = [
            getattr(camera, "image_name", None),
            getattr(camera, "uid", None),
            index,
        ]
        for key in key_candidates:
            if key in source:
                return source[key]
        raise KeyError(f"No feature/depth item found for camera {index}")
    return source[index]


def _view_weight_for_chunk(view_weight, start, end, count, dtype, device):
    if view_weight is None:
        return torch.ones(count, dtype=dtype, device=device)
    view_weight = torch.as_tensor(view_weight, dtype=dtype, device=device).reshape(-1)
    if view_weight.numel() == 1:
        return view_weight.expand(count)
    return view_weight[start:end]


def _clear_cuda_cache_for_device(device):
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_multiview_initialization_chunked(
    gaussians,
    cameras,
    feature_maps,
    landmark_indices,
    depth_maps,
    alpha_maps,
    view_weights,
    config,
):
    xyz_all = gaussian_localization_xyz(gaussians)
    device = xyz_all.device
    dtype = xyz_all.dtype
    landmark_count = int(landmark_indices.numel())
    chunk_size = max(int(config.chunk_size), 1)
    chunks = [(start, min(start + chunk_size, landmark_count)) for start in range(0, landmark_count, chunk_size)]

    def _resolve_streamed_view(view_idx, camera):
        feature_map = _resolve_view_item(feature_maps, view_idx, camera)
        if feature_map is None:
            return None
        feature_map = torch.as_tensor(feature_map, device=device, dtype=dtype)
        height, width = feature_map.shape[-2:]
        K = make_intrinsics_from_fov(
            camera.FoVx,
            camera.FoVy,
            width,
            height,
            device=device,
            dtype=dtype,
        )
        return {
            "camera": camera,
            "feature_map": feature_map,
            "height": height,
            "width": width,
            "K": K,
            "pose_w2c": _camera_pose_w2c(camera, device=device, dtype=dtype),
            "depth_map": _resolve_view_item(depth_maps, view_idx, camera),
            "alpha_map": _resolve_view_item(alpha_maps, view_idx, camera),
            "view_weight": _resolve_view_item(view_weights, view_idx, camera),
        }

    weighted_sum = None
    weight_sum = xyz_all.new_zeros((landmark_count,))
    observation_count = torch.zeros(landmark_count, dtype=torch.long, device=device)
    view_count = 0

    for view_idx, camera in enumerate(cameras):
        view = _resolve_streamed_view(view_idx, camera)
        if view is None:
            continue
        view_count += 1
        if weighted_sum is None:
            weighted_sum = xyz_all.new_zeros((landmark_count, int(view["feature_map"].shape[0])))
        for start, end in chunks:
            chunk_indices = landmark_indices[start:end]
            xyz = xyz_all[chunk_indices]
            uv, depth, valid = project_landmarks_to_query(
                xyz,
                view["K"],
                view["pose_w2c"],
                height=view["height"],
                width=view["width"],
            )
            valid = filter_depth_consistent_landmarks(
                uv,
                depth,
                valid,
                target_depth=view["depth_map"],
                target_alpha=view["alpha_map"],
                alpha_threshold=config.alpha_threshold,
                abs_tolerance=config.depth_abs_tolerance,
                rel_tolerance=config.depth_rel_tolerance,
            )
            sampled = bilinear_sample_features(view["feature_map"], uv)
            chunk_weight = _view_weight_for_chunk(
                view["view_weight"],
                start,
                end,
                end - start,
                dtype,
                device,
            ).clamp_min(0.0)
            weighted = chunk_weight * valid.to(dtype)
            weighted_sum[start:end] += sampled * weighted[:, None]
            weight_sum[start:end] += weighted
            observation_count[start:end] += valid.to(torch.long)
        del view
        _clear_cuda_cache_for_device(device)

    if weighted_sum is None:
        return MultiViewInitResult(
            features=xyz_all.new_zeros((landmark_count, 0)),
            reliability=xyz_all.new_zeros((landmark_count,)),
            observation_count=torch.zeros(landmark_count, dtype=torch.long, device=device),
            weight_sum=xyz_all.new_zeros((landmark_count,)),
            diagnostics={
                "view_count": 0,
                "gaussian_count": landmark_count,
                "observed_gaussians": 0,
                "chunk_count": len(chunks),
                "chunk_size": chunk_size,
            },
        )

    norm = torch.linalg.norm(weighted_sum, dim=-1, keepdim=True)
    features = torch.where(
        norm > float(config.eps),
        weighted_sum / norm.clamp_min(float(config.eps)),
        torch.zeros_like(weighted_sum),
    )

    reliability_sum = xyz_all.new_zeros((landmark_count,))
    for view_idx, camera in enumerate(cameras):
        view = _resolve_streamed_view(view_idx, camera)
        if view is None:
            continue
        for start, end in chunks:
            chunk_indices = landmark_indices[start:end]
            xyz = xyz_all[chunk_indices]
            uv, depth, valid = project_landmarks_to_query(
                xyz,
                view["K"],
                view["pose_w2c"],
                height=view["height"],
                width=view["width"],
            )
            valid = filter_depth_consistent_landmarks(
                uv,
                depth,
                valid,
                target_depth=view["depth_map"],
                target_alpha=view["alpha_map"],
                alpha_threshold=config.alpha_threshold,
                abs_tolerance=config.depth_abs_tolerance,
                rel_tolerance=config.depth_rel_tolerance,
            )
            sampled = bilinear_sample_features(view["feature_map"], uv)
            obs_norm = F.normalize(sampled, p=2, dim=-1)
            cosine = (obs_norm * features[start:end]).sum(dim=-1).clamp_min(0.0)
            chunk_weight = _view_weight_for_chunk(
                view["view_weight"],
                start,
                end,
                end - start,
                dtype,
                device,
            ).clamp_min(0.0)
            reliability_sum[start:end] += cosine * chunk_weight * valid.to(dtype)
        del view
        _clear_cuda_cache_for_device(device)

    reliability = reliability_sum / weight_sum.clamp_min(float(config.eps))
    reliability = torch.where(
        observation_count >= int(config.min_observations),
        reliability,
        torch.zeros_like(reliability),
    )
    reliability = torch.where(torch.isfinite(reliability), reliability, torch.zeros_like(reliability))
    diagnostics = {
        "view_count": int(view_count),
        "gaussian_count": landmark_count,
        "observed_gaussians": int((observation_count > 0).sum().item()),
        "mean_observations": float(observation_count.float().mean().item()) if landmark_count else 0.0,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
    }
    return MultiViewInitResult(
        features=features,
        reliability=reliability,
        observation_count=observation_count,
        weight_sum=weight_sum,
        diagnostics=diagnostics,
    )


@torch.no_grad()
def build_multiview_initialization(
    gaussians,
    cameras,
    feature_maps,
    landmark_indices=None,
    depth_maps=None,
    alpha_maps=None,
    view_weights=None,
    config=None,
    min_observations=None,
):
    """Project Gaussians into training views and build MVInit descriptors."""
    if config is None:
        config = MultiViewInitConfig()
    if min_observations is not None:
        config = MultiViewInitConfig(
            min_observations=int(min_observations),
            alpha_threshold=config.alpha_threshold,
            depth_abs_tolerance=config.depth_abs_tolerance,
            depth_rel_tolerance=config.depth_rel_tolerance,
            chunk_size=config.chunk_size,
            eps=config.eps,
        )
    xyz_all = gaussian_localization_xyz(gaussians)
    device = xyz_all.device
    dtype = xyz_all.dtype
    if landmark_indices is None:
        landmark_indices = torch.arange(xyz_all.shape[0], dtype=torch.long, device=device)
    else:
        landmark_indices = torch.as_tensor(landmark_indices, dtype=torch.long, device=device).reshape(-1)
    xyz = xyz_all[landmark_indices]
    if landmark_indices.numel() == 0:
        return MultiViewInitResult(
            features=xyz.new_zeros((0, 0)),
            reliability=xyz.new_zeros((0,)),
            observation_count=torch.zeros(0, dtype=torch.long, device=device),
            weight_sum=xyz.new_zeros((0,)),
            diagnostics={"view_count": len(cameras), "gaussian_count": 0, "observed_gaussians": 0},
        )
    if int(config.chunk_size) > 0 and landmark_indices.numel() > int(config.chunk_size):
        return _build_multiview_initialization_chunked(
            gaussians,
            cameras,
            feature_maps,
            landmark_indices,
            depth_maps,
            alpha_maps,
            view_weights,
            config,
        )

    observations = []
    weights = []
    valid_masks = []
    for view_idx, camera in enumerate(cameras):
        feature_map = _resolve_view_item(feature_maps, view_idx, camera)
        if feature_map is None:
            continue
        feature_map = torch.as_tensor(feature_map, device=device, dtype=dtype)
        height, width = feature_map.shape[-2:]
        K = make_intrinsics_from_fov(
            camera.FoVx,
            camera.FoVy,
            width,
            height,
            device=device,
            dtype=dtype,
        )
        pose_w2c = _camera_pose_w2c(camera, device=device, dtype=dtype)
        uv, depth, valid = project_landmarks_to_query(xyz, K, pose_w2c, height=height, width=width)
        depth_map = _resolve_view_item(depth_maps, view_idx, camera)
        alpha_map = _resolve_view_item(alpha_maps, view_idx, camera)
        valid = filter_depth_consistent_landmarks(
            uv,
            depth,
            valid,
            target_depth=depth_map,
            target_alpha=alpha_map,
            alpha_threshold=config.alpha_threshold,
            abs_tolerance=config.depth_abs_tolerance,
            rel_tolerance=config.depth_rel_tolerance,
        )
        sampled = bilinear_sample_features(feature_map, uv)
        view_weight = _resolve_view_item(view_weights, view_idx, camera)
        if view_weight is None:
            view_weight = torch.ones(landmark_indices.numel(), dtype=dtype, device=device)
        else:
            view_weight = torch.as_tensor(view_weight, dtype=dtype, device=device).reshape(-1)
            if view_weight.numel() == 1:
                view_weight = view_weight.expand(landmark_indices.numel())
        observations.append(sampled)
        weights.append(view_weight[: landmark_indices.numel()])
        valid_masks.append(valid)

    if not observations:
        feature_dim = 0
        if isinstance(feature_maps, (list, tuple)) and feature_maps:
            feature_dim = int(torch.as_tensor(feature_maps[0]).shape[0])
        return MultiViewInitResult(
            features=xyz.new_zeros((landmark_indices.numel(), feature_dim)),
            reliability=xyz.new_zeros((landmark_indices.numel(),)),
            observation_count=torch.zeros(landmark_indices.numel(), dtype=torch.long, device=device),
            weight_sum=xyz.new_zeros((landmark_indices.numel(),)),
            diagnostics={"view_count": 0, "gaussian_count": int(landmark_indices.numel()), "observed_gaussians": 0},
        )

    return aggregate_multiview_descriptors(
        torch.stack(observations, dim=0),
        weights=torch.stack(weights, dim=0),
        valid=torch.stack(valid_masks, dim=0),
        config=config,
    )


@torch.no_grad()
def apply_multiview_initialization(gaussians, result: MultiViewInitResult, update_stats=True):
    """Write MVInit descriptors and reliability stats into a Gaussian model."""
    target = getattr(gaussians, "_loc_feature", None)
    if not torch.is_tensor(target):
        raise ValueError("gaussians must expose a _loc_feature tensor")
    features = F.normalize(result.features.to(device=target.device, dtype=target.dtype), p=2, dim=-1)
    if features.numel() != target.numel():
        raise ValueError(
            "MVInit feature shape does not match Gaussian loc_feature: "
            f"{tuple(features.shape)} vs {tuple(target.shape)}"
        )
    observed = result.observation_count.to(device=target.device).reshape(-1) > 0
    reshaped = features.reshape_as(target)
    if observed.numel() != target.shape[0]:
        raise ValueError(
            "MVInit observation count does not match Gaussian loc_feature rows: "
            f"{observed.numel()} vs {target.shape[0]}"
        )
    if bool(observed.any()):
        target[observed] = reshaped[observed]

    if not update_stats:
        return
    if hasattr(gaussians, "loc_prototype") and torch.is_tensor(gaussians.loc_prototype):
        if gaussians.loc_prototype.shape == features.shape:
            proto_observed = observed.to(device=gaussians.loc_prototype.device)
            if bool(proto_observed.any()):
                proto_features = features.to(gaussians.loc_prototype.device, gaussians.loc_prototype.dtype)
                gaussians.loc_prototype[proto_observed] = proto_features[proto_observed]
    if hasattr(gaussians, "loc_prototype_count") and torch.is_tensor(gaussians.loc_prototype_count):
        gaussians.loc_prototype_count.copy_(
            result.observation_count.to(gaussians.loc_prototype_count.device, gaussians.loc_prototype_count.dtype)
        )
    if hasattr(gaussians, "loc_observation_count") and torch.is_tensor(gaussians.loc_observation_count):
        gaussians.loc_observation_count.copy_(
            result.observation_count.to(gaussians.loc_observation_count.device, gaussians.loc_observation_count.dtype)
        )
    if hasattr(gaussians, "loc_repeatability_ema") and torch.is_tensor(gaussians.loc_repeatability_ema):
        gaussians.loc_repeatability_ema.copy_(
            result.reliability.to(gaussians.loc_repeatability_ema.device, gaussians.loc_repeatability_ema.dtype)
        )


@torch.no_grad()
def apply_multiview_localization_stats(gaussians, result: MultiViewInitResult, update_prototype=True):
    """Write MVInit observation stats without changing the Gaussian descriptor field."""
    if not hasattr(gaussians, "loc_observation_count") or not torch.is_tensor(gaussians.loc_observation_count):
        raise ValueError("gaussians must expose loc_observation_count")

    device = gaussians.loc_observation_count.device
    observed_count = result.observation_count.to(device=device, dtype=gaussians.loc_observation_count.dtype).reshape(-1)
    if observed_count.numel() != gaussians.loc_observation_count.numel():
        raise ValueError(
            "MVInit observation count does not match Gaussian localization rows: "
            f"{observed_count.numel()} vs {gaussians.loc_observation_count.numel()}"
        )
    gaussians.loc_observation_count.copy_(observed_count)

    if hasattr(gaussians, "loc_repeatability_ema") and torch.is_tensor(gaussians.loc_repeatability_ema):
        gaussians.loc_repeatability_ema.copy_(
            result.reliability.to(
                device=gaussians.loc_repeatability_ema.device,
                dtype=gaussians.loc_repeatability_ema.dtype,
            ).reshape_as(gaussians.loc_repeatability_ema)
        )

    if hasattr(gaussians, "loc_prototype_count") and torch.is_tensor(gaussians.loc_prototype_count):
        gaussians.loc_prototype_count.copy_(
            result.observation_count.to(
                device=gaussians.loc_prototype_count.device,
                dtype=gaussians.loc_prototype_count.dtype,
            ).reshape_as(gaussians.loc_prototype_count)
        )

    if not update_prototype:
        return
    if hasattr(gaussians, "loc_prototype") and torch.is_tensor(gaussians.loc_prototype):
        features = F.normalize(
            result.features.to(device=gaussians.loc_prototype.device, dtype=gaussians.loc_prototype.dtype),
            p=2,
            dim=-1,
        )
        if features.shape != gaussians.loc_prototype.shape:
            raise ValueError(
                "MVInit prototype shape does not match Gaussian localization prototype: "
                f"{tuple(features.shape)} vs {tuple(gaussians.loc_prototype.shape)}"
            )
        observed = result.observation_count.to(device=gaussians.loc_prototype.device).reshape(-1) > 0
        if bool(observed.any()):
            gaussians.loc_prototype[observed] = features[observed]


def _feature_grid(feature_map):
    if feature_map.dim() != 3:
        raise ValueError("query feature map must have shape [channels, height, width]")
    channels, height, width = feature_map.shape
    flat = feature_map.reshape(channels, height * width).T
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=feature_map.dtype, device=feature_map.device),
        torch.arange(width, dtype=feature_map.dtype, device=feature_map.device),
        indexing="ij",
    )
    coords = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)
    return flat, coords, height, width


def _local_window_mask(coords, projected_uv, radius):
    if projected_uv is None or float(radius) <= 0.0:
        return None
    projected_uv = torch.as_tensor(projected_uv, dtype=coords.dtype, device=coords.device).reshape(-1, 2)
    dist = torch.linalg.norm(coords[None] - projected_uv[:, None], dim=-1)
    mask = dist <= float(radius)
    empty = ~mask.any(dim=1)
    if empty.any():
        mask[empty] = True
    return mask


def _probability_peak_and_margin(probabilities):
    if probabilities.shape[1] == 1:
        peak = probabilities[:, 0]
        margin = peak
        return peak, margin
    top2 = torch.topk(probabilities, k=2, dim=1).values
    peak = top2[:, 0]
    margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
    return peak, margin


def _soft_correspondences_local_window(
    gaussian,
    query_feature_map,
    projected_uv,
    radius,
    temperature,
    valid_mask=None,
):
    channels, height, width = query_feature_map.shape
    projected_uv = torch.as_tensor(
        projected_uv,
        dtype=query_feature_map.dtype,
        device=query_feature_map.device,
    ).reshape(gaussian.shape[0], 2)
    radius = float(radius)
    window = max(int(torch.ceil(torch.tensor(radius)).item()), 0)
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-window, window + 1, dtype=query_feature_map.dtype, device=query_feature_map.device),
        torch.arange(-window, window + 1, dtype=query_feature_map.dtype, device=query_feature_map.device),
        indexing="ij",
    )
    offsets = torch.stack([offsets_x.reshape(-1), offsets_y.reshape(-1)], dim=-1)
    keep_offsets = torch.linalg.norm(offsets, dim=-1) <= radius
    offsets = offsets[keep_offsets]
    if offsets.numel() == 0:
        offsets = torch.zeros((1, 2), dtype=query_feature_map.dtype, device=query_feature_map.device)

    centers = projected_uv.round()
    coords = centers[:, None, :] + offsets[None, :, :]
    x = coords[..., 0]
    y = coords[..., 1]
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    x_long = x.clamp(0, width - 1).to(dtype=torch.long)
    y_long = y.clamp(0, height - 1).to(dtype=torch.long)

    if valid_mask is not None:
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=query_feature_map.device)
        valid_mask = valid_mask.reshape(height, width)
        valid = valid & valid_mask[y_long, x_long]

    empty = ~valid.any(dim=1)
    if empty.any():
        nearest = projected_uv[empty].round()
        x_near = nearest[:, 0].clamp(0, width - 1).to(dtype=torch.long)
        y_near = nearest[:, 1].clamp(0, height - 1).to(dtype=torch.long)
        x_long[empty, 0] = x_near
        y_long[empty, 0] = y_near
        coords[empty, 0, 0] = x_near.to(dtype=query_feature_map.dtype)
        coords[empty, 0, 1] = y_near.to(dtype=query_feature_map.dtype)
        valid[empty, 0] = True

    local_features = query_feature_map.permute(1, 2, 0)[y_long, x_long]
    local_features = F.normalize(local_features, p=2, dim=-1)
    logits = (local_features * gaussian[:, None, :]).sum(dim=-1) / temperature
    logits = logits.masked_fill(~valid, -torch.finfo(logits.dtype).max)

    probabilities = F.softmax(logits, dim=1)
    uv = (probabilities[..., None] * coords).sum(dim=1)
    entropy_raw = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
    support = valid.sum(dim=1).clamp_min(2).to(dtype=entropy_raw.dtype)
    entropy = (entropy_raw / torch.log(support)).clamp(0.0, 1.0)
    confidence = (1.0 - entropy).clamp(0.0, 1.0)
    peak_probability, margin = _probability_peak_and_margin(probabilities)
    is_valid = torch.isfinite(uv).all(dim=1) & torch.isfinite(confidence)
    return SoftCorrespondenceOutput(
        uv=uv.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        confidence=confidence.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        entropy=entropy.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        peak_probability=peak_probability.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        margin=margin.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        probabilities=probabilities.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        valid=is_valid.to(device=query_feature_map.device),
    )


def soft_3d_to_2d_correspondences(
    gaussian_features,
    query_feature_map,
    temperature=0.07,
    projected_uv=None,
    local_window_radius=0.0,
    valid_mask=None,
):
    """Match each Gaussian descriptor into a query feature map with soft-argmax."""
    gaussian = F.normalize(_as_feature_matrix(gaussian_features).float(), p=2, dim=-1)
    if projected_uv is not None and float(local_window_radius) > 0.0:
        return _soft_correspondences_local_window(
            gaussian,
            query_feature_map.float(),
            projected_uv,
            local_window_radius,
            max(float(temperature), 0.05),
            valid_mask=valid_mask,
        )
    fmap, coords, _, _ = _feature_grid(query_feature_map.float())
    fmap = F.normalize(fmap, p=2, dim=-1)
    temperature = max(float(temperature), 0.05)
    logits = gaussian @ fmap.T / temperature

    window_mask = _local_window_mask(coords, projected_uv, local_window_radius)
    if window_mask is not None:
        logits = logits.masked_fill(~window_mask, -torch.finfo(logits.dtype).max)
    if valid_mask is not None:
        valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=logits.device).reshape(-1)
        logits = logits.masked_fill(~valid_mask[None], -torch.finfo(logits.dtype).max)

    probabilities = F.softmax(logits, dim=1)
    uv = probabilities @ coords
    entropy_raw = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(dim=1)
    if window_mask is not None:
        support = window_mask.sum(dim=1).clamp_min(2).to(dtype=entropy_raw.dtype)
    elif valid_mask is not None:
        support = valid_mask.sum().reshape(1).clamp_min(2).expand_as(entropy_raw).to(dtype=entropy_raw.dtype)
    else:
        support = torch.full_like(entropy_raw, float(probabilities.shape[1]))
    entropy = entropy_raw / torch.log(support)
    entropy = entropy.clamp(0.0, 1.0)
    confidence = (1.0 - entropy).clamp(0.0, 1.0)
    peak_probability, margin = _probability_peak_and_margin(probabilities)
    valid = torch.isfinite(uv).all(dim=1) & torch.isfinite(confidence)
    return SoftCorrespondenceOutput(
        uv=uv.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        confidence=confidence.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        entropy=entropy.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        peak_probability=peak_probability.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        margin=margin.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        probabilities=probabilities.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
        valid=valid.to(device=query_feature_map.device),
    )


def _weighted_point_loss(pred_uv, target_uv, weights, loss_type="smooth_l1", robust_delta=1.0):
    loss_type = str(loss_type or "smooth_l1").lower()
    robust_delta = float(robust_delta)
    if loss_type == "smooth_l1":
        residual = F.smooth_l1_loss(pred_uv, target_uv, reduction="none").sum(dim=1)
    elif loss_type == "huber":
        beta = max(robust_delta, 1e-6)
        residual = F.smooth_l1_loss(pred_uv, target_uv, reduction="none", beta=beta).sum(dim=1)
    elif loss_type == "cauchy":
        delta = max(robust_delta, 1e-6)
        squared_norm = (pred_uv - target_uv).pow(2).sum(dim=1)
        residual = torch.log1p(squared_norm / (delta * delta))
    else:
        raise ValueError(
            "loss_type must be one of {'smooth_l1', 'huber', 'cauchy'}, "
            f"got {loss_type!r}"
        )
    weights = weights.to(device=residual.device, dtype=residual.dtype).reshape(-1)
    return (residual * weights).sum() / weights.sum().clamp_min(1e-6)


def _unproject_uv_depth_to_world(uv, depth, K, pose_w2c):
    depth = torch.as_tensor(depth, dtype=uv.dtype, device=uv.device).reshape(-1)
    x = (uv[:, 0] - K[0, 2]) * depth / K[0, 0].clamp_min(1e-8)
    y = (uv[:, 1] - K[1, 2]) * depth / K[1, 1].clamp_min(1e-8)
    points_cam = torch.stack([x, y, depth], dim=1)
    rotation = pose_w2c[:3, :3]
    translation = pose_w2c[:3, 3]
    return (points_cam - translation.reshape(1, 3)) @ rotation


def _weighted_xyz_loss(pred_xyz, target_xyz, weights):
    residual = F.smooth_l1_loss(pred_xyz, target_xyz, reduction="none").sum(dim=1)
    weights = weights.to(device=residual.device, dtype=residual.dtype).reshape(-1)
    return (residual * weights).sum() / weights.sum().clamp_min(1e-6)


def _masked_float_stat(values, mask, stat="mean"):
    if values is None or mask is None or values.numel() == 0 or mask.numel() == 0:
        return 0.0
    mask = mask.detach().to(dtype=torch.bool)
    if not bool(mask.any().item()):
        return 0.0
    selected = values.detach()[mask]
    if selected.numel() == 0:
        return 0.0
    if stat == "max":
        return float(selected.max().item())
    if stat == "min":
        return float(selected.min().item())
    return float(selected.mean().item())


def _override_or_fallback(value, fallback):
    value = float(value)
    return value if value >= 0.0 else float(fallback)


def _pose_l1_geodesic_loss(pose_w2c, pose_gt_w2c, translation_weight=1.0, rotation_weight=1.0):
    pose_gt_w2c = pose_gt_w2c.to(device=pose_w2c.device, dtype=pose_w2c.dtype)
    translation = (pose_w2c[:3, 3] - pose_gt_w2c[:3, 3]).abs().mean()
    rel = pose_w2c[:3, :3] @ pose_gt_w2c[:3, :3].T
    cos = ((torch.trace(rel) - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    rotation = torch.acos(cos)
    return float(translation_weight) * translation + float(rotation_weight) * rotation


def _pnp_spatial_diagnostics(reference_uv, image_size):
    reference_uv = torch.as_tensor(reference_uv)
    if reference_uv.numel() == 0:
        return {
            "spatial_span_x": 0.0,
            "spatial_span_y": 0.0,
            "spatial_min_span": 0.0,
            "spatial_area": 0.0,
        }
    height, width = image_size
    finite = torch.isfinite(reference_uv).all(dim=1)
    if not bool(finite.any().item()):
        return {
            "spatial_span_x": 0.0,
            "spatial_span_y": 0.0,
            "spatial_min_span": 0.0,
            "spatial_area": 0.0,
        }
    uv = reference_uv[finite].float()
    span = uv.max(dim=0).values - uv.min(dim=0).values
    span_x = float((span[0] / max(float(width), 1.0)).clamp_min(0.0).item())
    span_y = float((span[1] / max(float(height), 1.0)).clamp_min(0.0).item())
    return {
        "spatial_span_x": span_x,
        "spatial_span_y": span_y,
        "spatial_min_span": min(span_x, span_y),
        "spatial_area": span_x * span_y,
    }


def _cell_ids_for_uv(reference_uv, image_size, grid_size):
    height, width = image_size
    grid_size = max(int(grid_size), 1)
    uv = torch.as_tensor(reference_uv, dtype=torch.float32, device=reference_uv.device)
    cell_x = torch.floor(uv[:, 0].clamp(0, max(float(width) - 1.0, 0.0)) / max(float(width), 1.0) * grid_size)
    cell_y = torch.floor(uv[:, 1].clamp(0, max(float(height) - 1.0, 0.0)) / max(float(height), 1.0) * grid_size)
    cell_x = cell_x.to(dtype=torch.long).clamp(0, grid_size - 1)
    cell_y = cell_y.to(dtype=torch.long).clamp(0, grid_size - 1)
    return cell_y * grid_size + cell_x


def _select_pnp_correspondence_indices(
    valid_idx,
    confidence,
    reference_uv=None,
    image_size=None,
    max_correspondences=0,
    spatial_grid_size=0,
):
    valid_idx = torch.as_tensor(valid_idx, dtype=torch.long, device=confidence.device).reshape(-1)
    if valid_idx.numel() == 0:
        return valid_idx, {
            "selection_mode": "empty",
            "selected_spatial_cells": 0,
            "candidate_spatial_cells": 0,
        }

    max_correspondences = int(max_correspondences)
    cap = valid_idx.numel() if max_correspondences <= 0 else min(max_correspondences, valid_idx.numel())
    conf = confidence[valid_idx].detach()
    order = torch.argsort(conf, descending=True)
    selected_positions = []
    candidate_spatial_cells = 0
    selected_spatial_cells = 0
    selection_mode = "confidence"

    if (
        reference_uv is not None
        and image_size is not None
        and int(spatial_grid_size) > 1
        and valid_idx.numel() > cap
    ):
        ref = torch.as_tensor(reference_uv, dtype=torch.float32, device=confidence.device)
        ref_valid = ref[valid_idx]
        finite = torch.isfinite(ref_valid).all(dim=1)
        if bool(finite.any().item()):
            cell_ids = _cell_ids_for_uv(ref_valid, image_size, int(spatial_grid_size))
            candidate_spatial_cells = int(torch.unique(cell_ids[finite], sorted=True).numel())
            used_cells = set()
            for pos in order.tolist():
                if not bool(finite[pos].item()):
                    continue
                cell_id = int(cell_ids[pos].item())
                if cell_id in used_cells:
                    continue
                selected_positions.append(pos)
                used_cells.add(cell_id)
                if len(selected_positions) >= cap:
                    break
            if len(selected_positions) < cap:
                used_positions = set(selected_positions)
                for pos in order.tolist():
                    if pos in used_positions:
                        continue
                    selected_positions.append(pos)
                    if len(selected_positions) >= cap:
                        break
            selected_spatial_cells = len(used_cells)
            selection_mode = "grid"

    if not selected_positions:
        selected_positions = order[:cap].tolist()
        if reference_uv is not None and image_size is not None and int(spatial_grid_size) > 1:
            ref = torch.as_tensor(reference_uv, dtype=torch.float32, device=confidence.device)
            cells = _cell_ids_for_uv(ref[valid_idx], image_size, int(spatial_grid_size))
            candidate_spatial_cells = int(torch.unique(cells, sorted=True).numel())
            selected_spatial_cells = int(torch.unique(cells[selected_positions], sorted=True).numel())

    selected = valid_idx[torch.as_tensor(selected_positions, dtype=torch.long, device=valid_idx.device)]
    diagnostics = {
        "selection_mode": selection_mode,
        "selected_spatial_cells": int(selected_spatial_cells),
        "candidate_spatial_cells": int(candidate_spatial_cells),
    }
    return selected, diagnostics


def _zero_pnp_output(
    gaussian_features,
    query_feature_map,
    pose_gt_w2c,
    correspondences,
    used,
    point_weights=None,
    skipped_reason="min_correspondences",
    diagnostics=None,
):
    zero = gaussian_features.reshape(-1).sum() * 0.0 + query_feature_map.reshape(-1).sum() * 0.0
    if point_weights is not None:
        zero = zero + point_weights.reshape(-1).sum() * 0.0
    pose = pose_gt_w2c.to(device=query_feature_map.device, dtype=query_feature_map.dtype)
    diagnostics = dict(diagnostics or {})
    diagnostics.update({"skipped": 1.0, "skipped_reason": skipped_reason})
    return DifferentiablePnPOutput(
        loss=zero,
        pose_loss=zero,
        reprojection_loss=zero,
        gt_reprojection_loss=zero,
        geometry_reprojection_loss=zero,
        geometry_match_reprojection_loss=zero,
        geometry_depth_anchor_loss=zero,
        entropy_loss=zero,
        pose_w2c=pose,
        correspondences=correspondences,
        used_correspondences=int(used),
        diagnostics=diagnostics,
    )


def differentiable_pnp_pose_loss(
    gaussian_features,
    query_feature_map,
    points_world,
    K,
    pose_gt_w2c,
    pose_init_w2c=None,
    projected_uv=None,
    geometry_anchor_points_world=None,
    point_weights=None,
    config=None,
):
    """Pose-supervised LaFGS loss using soft 3D-to-2D matches and weighted PnP."""
    config = config or DifferentiablePnPConfig()
    feature_map = query_feature_map.detach() if config.detach_query_feature_map else query_feature_map
    if projected_uv is None and float(config.local_window_radius) > 0.0:
        points_for_projection = points_world.detach()
        projected_uv, _ = project_points(
            points_for_projection.to(device=feature_map.device, dtype=feature_map.dtype),
            K.to(device=feature_map.device, dtype=feature_map.dtype),
            pose_gt_w2c.to(device=feature_map.device, dtype=feature_map.dtype),
        )
    correspondences = soft_3d_to_2d_correspondences(
        gaussian_features,
        feature_map,
        temperature=config.temperature,
        projected_uv=projected_uv,
        local_window_radius=config.local_window_radius,
    )
    height, width = query_feature_map.shape[-2:]
    point_weights_tensor = None
    if point_weights is not None:
        point_weights_tensor = point_weights.to(
            device=correspondences.confidence.device,
            dtype=correspondences.confidence.dtype,
        ).reshape(-1)
        if point_weights_tensor.numel() != correspondences.confidence.numel():
            raise ValueError(
                "point_weights must have one value per Gaussian correspondence candidate: "
                f"got {point_weights_tensor.numel()} for {correspondences.confidence.numel()}"
            )
    valid = correspondences.valid & (correspondences.confidence >= float(config.confidence_threshold))
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    all_valid_idx = valid_idx
    reference_uv = projected_uv if projected_uv is not None else correspondences.uv.detach()
    selection_confidence = correspondences.confidence
    if point_weights_tensor is not None:
        point_weight_floor = float(config.point_weight_floor)
        point_weight_floor = min(max(point_weight_floor, 0.0), 1.0)
        selection_point_weights = point_weight_floor + (1.0 - point_weight_floor) * (
            point_weights_tensor.detach().clamp(0.0, 1.0)
        )
        selection_confidence = selection_confidence * selection_point_weights
    selection_diagnostics = {}
    if valid_idx.numel() > 0:
        valid_idx, selection_diagnostics = _select_pnp_correspondence_indices(
            valid_idx,
            selection_confidence,
            reference_uv=reference_uv,
            image_size=(height, width),
            max_correspondences=config.max_correspondences,
            spatial_grid_size=config.spatial_grid_size,
        )
    used = int(valid_idx.numel())
    if used < int(config.min_correspondences):
        return _zero_pnp_output(
            gaussian_features,
            query_feature_map,
            pose_gt_w2c,
            correspondences,
            0,
            point_weights=point_weights_tensor,
            skipped_reason="min_correspondences",
            diagnostics=selection_diagnostics,
        )
    spatial_diagnostics = _pnp_spatial_diagnostics(reference_uv[valid_idx].detach(), (height, width))
    selection_diagnostics.update(spatial_diagnostics)
    if float(config.min_spatial_span) > 0.0 and spatial_diagnostics["spatial_min_span"] < float(
        config.min_spatial_span
    ):
        return _zero_pnp_output(
            gaussian_features,
            query_feature_map,
            pose_gt_w2c,
            correspondences,
            0,
            point_weights=point_weights_tensor,
            skipped_reason="spatial_span",
            diagnostics=selection_diagnostics,
        )
    if float(config.min_spatial_area) > 0.0 and spatial_diagnostics["spatial_area"] < float(
        config.min_spatial_area
    ):
        return _zero_pnp_output(
            gaussian_features,
            query_feature_map,
            pose_gt_w2c,
            correspondences,
            0,
            point_weights=point_weights_tensor,
            skipped_reason="spatial_area",
            diagnostics=selection_diagnostics,
        )

    dtype = feature_map.dtype
    device = feature_map.device
    allow_geometry_grad = bool(config.allow_geometry_grad)
    detach_pnp_points = bool(config.detach_pnp_points)
    points = points_world.to(device=device, dtype=dtype)
    geometry_anchor_points = None
    if geometry_anchor_points_world is not None:
        geometry_anchor_points = torch.as_tensor(
            geometry_anchor_points_world,
            device=device,
            dtype=dtype,
        )
        if geometry_anchor_points.shape != points.shape:
            raise ValueError(
                "geometry_anchor_points_world must match points_world shape: "
                f"got {tuple(geometry_anchor_points.shape)} for {tuple(points.shape)}"
            )
    selected_points = points[valid_idx]
    pnp_points = selected_points
    if (not allow_geometry_grad) or detach_pnp_points:
        pnp_points = selected_points.detach()
    selected_uv = correspondences.uv[valid_idx].to(device=device, dtype=dtype)
    weights = correspondences.confidence[valid_idx].to(device=device, dtype=dtype).clamp_min(1e-4)
    if point_weights_tensor is not None:
        point_weight_floor = float(config.point_weight_floor)
        point_weight_floor = min(max(point_weight_floor, 0.0), 1.0)
        selected_point_weights_raw = point_weights_tensor[valid_idx].to(device=device, dtype=dtype).clamp(0.0, 1.0)
        selected_point_weights = point_weight_floor + (1.0 - point_weight_floor) * selected_point_weights_raw
        selected_point_weights = selected_point_weights.clamp_min(1e-4)
        weights = weights * selected_point_weights
        selection_diagnostics.update(
            {
                "point_weight_mean": float(selected_point_weights.detach().mean().item()),
                "point_weight_min": float(selected_point_weights.detach().min().item()),
                "point_weight_max": float(selected_point_weights.detach().max().item()),
            }
        )
    K = K.to(device=device, dtype=dtype)
    pose_gt_w2c = pose_gt_w2c.to(device=device, dtype=dtype)
    if pose_init_w2c is None:
        pose_init_w2c = pose_gt_w2c
    pose_init_w2c = pose_init_w2c.to(device=device, dtype=dtype)

    pose_pred, info = weighted_gauss_newton_refine(
        pnp_points,
        selected_uv,
        K,
        pose_init_w2c,
        weights=weights,
        num_iterations=int(config.pnp_iterations),
        damping=float(config.damping),
        detach_points=(not allow_geometry_grad) or detach_pnp_points,
    )
    pose_loss = _pose_l1_geodesic_loss(
        pose_pred,
        pose_gt_w2c,
        translation_weight=config.translation_weight,
        rotation_weight=config.rotation_weight,
    )
    init_pose_loss = _pose_l1_geodesic_loss(
        pose_init_w2c,
        pose_gt_w2c,
        translation_weight=config.translation_weight,
        rotation_weight=config.rotation_weight,
    )
    pose_guard_enabled, pose_guard_passed = _pose_guard_check(
        pose_loss,
        init_pose_loss,
        max_loss_increase=config.geometry_pose_guard_max_loss_increase,
        max_loss=config.geometry_pose_guard_max_loss,
    )
    (
        _,
        _,
        geometry_pose_guard_scale,
        geometry_pose_guard_violation,
    ) = _pose_guard_soft_scale(
        pose_loss,
        init_pose_loss,
        max_loss_increase=config.geometry_pose_guard_max_loss_increase,
        max_loss=config.geometry_pose_guard_max_loss,
        softness=config.geometry_pose_guard_softness,
        min_scale=config.geometry_pose_guard_min_scale,
    )
    (
        feedback_pose_guard_enabled,
        feedback_pose_guard_passed,
        feedback_scale,
        feedback_pose_guard_violation,
    ) = _pose_guard_soft_scale(
        pose_loss,
        init_pose_loss,
        max_loss_increase=config.feedback_pose_guard_max_loss_increase,
        max_loss=config.feedback_pose_guard_max_loss,
        softness=config.feedback_pose_guard_softness,
        min_scale=config.feedback_pose_guard_min_scale,
    )
    condition_number = info["condition_number"].detach()
    max_condition_number = float(config.max_condition_number)
    condition_guard_enabled = max_condition_number >= 0.0
    condition_guard_passed = True
    if condition_guard_enabled:
        condition_guard_passed = bool(
            torch.isfinite(condition_number).item()
            and (condition_number <= pose_loss.new_tensor(max_condition_number)).item()
        )
    condition_guard_scale = pose_loss.new_tensor(1.0 if condition_guard_passed else 0.0)
    feedback_loss_scale = feedback_scale * condition_guard_scale
    pred_uv, pred_valid = project_points(pnp_points, K, pose_pred)
    gt_uv, gt_valid = project_points(pnp_points, K, pose_gt_w2c)
    pred_weights = weights * pred_valid.to(dtype)
    gt_weights = weights * gt_valid.to(dtype)
    reprojection_loss = _weighted_point_loss(
        pred_uv,
        selected_uv,
        pred_weights,
        loss_type=config.reprojection_loss_type,
        robust_delta=config.reprojection_loss_delta,
    )
    gt_uv_for_loss = gt_uv.detach() if bool(config.detach_gt_reprojection_points) else gt_uv
    gt_reprojection_loss = _weighted_point_loss(
        gt_uv_for_loss,
        selected_uv,
        gt_weights,
        loss_type=config.reprojection_loss_type,
        robust_delta=config.reprojection_loss_delta,
    )
    geometry_reprojection_loss = (gt_uv.reshape(-1).sum() + selected_uv.reshape(-1).sum()) * 0.0
    geometry_correspondences = 0
    geometry_candidate_count = 0
    geometry_weight_sum = 0.0
    geometry_source = "pnp_selected"
    geometry_valid_candidate_count = 0
    geometry_filter_keep_ratio = 0.0
    geometry_candidate_confidence_mean = 0.0
    geometry_candidate_confidence_max = 0.0
    geometry_kept_confidence_mean = 0.0
    geometry_kept_confidence_max = 0.0
    geometry_candidate_margin_mean = 0.0
    geometry_candidate_margin_max = 0.0
    geometry_kept_margin_mean = 0.0
    geometry_kept_margin_max = 0.0
    geometry_candidate_peak_probability_mean = 0.0
    geometry_candidate_peak_probability_min = 0.0
    geometry_candidate_peak_probability_max = 0.0
    geometry_kept_peak_probability_mean = 0.0
    geometry_kept_peak_probability_min = 0.0
    geometry_kept_peak_probability_max = 0.0
    geometry_candidate_entropy_mean = 0.0
    geometry_candidate_entropy_max = 0.0
    geometry_kept_entropy_mean = 0.0
    geometry_kept_entropy_max = 0.0
    geometry_candidate_reprojection_error_mean = 0.0
    geometry_candidate_reprojection_error_max = 0.0
    geometry_kept_reprojection_error_mean = 0.0
    geometry_kept_reprojection_error_max = 0.0
    geometry_match_reprojection_loss = (gt_uv.reshape(-1).sum() + selected_uv.reshape(-1).sum()) * 0.0
    geometry_depth_anchor_loss = geometry_match_reprojection_loss * 0.0
    geometry_match_correspondences = 0
    geometry_match_candidate_count = 0
    geometry_match_weight_sum = 0.0
    geometry_match_filter_keep_ratio = 0.0
    geometry_depth_anchor_correspondences = 0
    geometry_depth_anchor_candidate_count = 0
    geometry_depth_anchor_weight_sum = 0.0
    geometry_depth_anchor_filter_keep_ratio = 0.0
    if allow_geometry_grad and float(config.geometry_reprojection_weight) > 0.0:
        if bool(config.geometry_use_all_correspondences):
            geometry_idx = all_valid_idx
            geometry_source = "all_valid"
        else:
            geometry_idx = valid_idx
        geometry_candidate_count = int(geometry_idx.numel())
        geometry_points = points[geometry_idx]
        geometry_uv = correspondences.uv[geometry_idx].to(device=device, dtype=dtype)
        geometry_confidence = correspondences.confidence[geometry_idx].to(device=device, dtype=dtype)
        geometry_margin = correspondences.margin[geometry_idx].to(device=device, dtype=dtype)
        geometry_peak_probability = correspondences.peak_probability[geometry_idx].to(device=device, dtype=dtype)
        geometry_entropy = correspondences.entropy[geometry_idx].to(device=device, dtype=dtype)
        geometry_valid = correspondences.valid[geometry_idx].to(device=device)
        geometry_local_window_radius = float(config.geometry_local_window_radius)
        if geometry_local_window_radius > 0.0 and projected_uv is not None and geometry_candidate_count > 0:
            gaussian_matrix = _as_feature_matrix(gaussian_features).to(device=device, dtype=dtype)
            geometry_projected_uv = torch.as_tensor(projected_uv, device=device, dtype=dtype)[geometry_idx]
            geometry_local = soft_3d_to_2d_correspondences(
                gaussian_matrix[geometry_idx].detach(),
                feature_map.detach(),
                temperature=config.temperature,
                projected_uv=geometry_projected_uv.detach(),
                local_window_radius=geometry_local_window_radius,
            )
            geometry_uv = geometry_local.uv.to(device=device, dtype=dtype)
            geometry_confidence = geometry_local.confidence.to(device=device, dtype=dtype)
            geometry_margin = geometry_local.margin.to(device=device, dtype=dtype)
            geometry_peak_probability = geometry_local.peak_probability.to(device=device, dtype=dtype)
            geometry_entropy = geometry_local.entropy.to(device=device, dtype=dtype)
            geometry_valid = geometry_local.valid.to(device=device)
            geometry_source = f"{geometry_source}_local"
        gt_uv_geometry, gt_valid_geometry = project_points(geometry_points, K, pose_gt_w2c)
        geometry_weights = geometry_confidence.detach().clone().clamp_min(1e-4) * gt_valid_geometry.detach().to(dtype)
        if point_weights_tensor is not None and geometry_candidate_count > 0:
            point_weight_floor = float(config.point_weight_floor)
            point_weight_floor = min(max(point_weight_floor, 0.0), 1.0)
            geometry_point_weights_raw = point_weights_tensor[geometry_idx].to(device=device, dtype=dtype).clamp(0.0, 1.0)
            geometry_point_weights = point_weight_floor + (1.0 - point_weight_floor) * geometry_point_weights_raw
            geometry_weights = geometry_weights * geometry_point_weights.clamp_min(1e-4)
        geometry_error = torch.linalg.norm(gt_uv_geometry.detach() - geometry_uv.detach(), dim=1)
        geometry_mask = gt_valid_geometry.detach().to(dtype=torch.bool)
        geometry_mask = geometry_mask & geometry_valid.detach().to(dtype=torch.bool)
        geometry_mask = geometry_mask & torch.isfinite(gt_uv_geometry.detach()).all(dim=1)
        geometry_mask = geometry_mask & torch.isfinite(geometry_uv.detach()).all(dim=1)
        geometry_mask = geometry_mask & torch.isfinite(geometry_error.detach())
        geometry_valid_candidate_count = int(geometry_mask.sum().detach().item())
        geometry_candidate_confidence_mean = _masked_float_stat(geometry_confidence, geometry_mask, "mean")
        geometry_candidate_confidence_max = _masked_float_stat(geometry_confidence, geometry_mask, "max")
        geometry_candidate_margin_mean = _masked_float_stat(geometry_margin, geometry_mask, "mean")
        geometry_candidate_margin_max = _masked_float_stat(geometry_margin, geometry_mask, "max")
        geometry_candidate_peak_probability_mean = _masked_float_stat(
            geometry_peak_probability, geometry_mask, "mean"
        )
        geometry_candidate_peak_probability_min = _masked_float_stat(
            geometry_peak_probability, geometry_mask, "min"
        )
        geometry_candidate_peak_probability_max = _masked_float_stat(
            geometry_peak_probability, geometry_mask, "max"
        )
        geometry_candidate_entropy_mean = _masked_float_stat(geometry_entropy, geometry_mask, "mean")
        geometry_candidate_entropy_max = _masked_float_stat(geometry_entropy, geometry_mask, "max")
        geometry_candidate_reprojection_error_mean = _masked_float_stat(geometry_error, geometry_mask, "mean")
        geometry_candidate_reprojection_error_max = _masked_float_stat(geometry_error, geometry_mask, "max")
        confidence_threshold = float(config.geometry_confidence_threshold)
        if confidence_threshold > 0.0:
            geometry_mask = geometry_mask & (geometry_confidence >= confidence_threshold)
        margin_threshold = float(config.geometry_margin_threshold)
        if margin_threshold > 0.0:
            geometry_mask = geometry_mask & (geometry_margin >= margin_threshold)
        peak_probability_threshold = float(config.geometry_peak_probability_threshold)
        if peak_probability_threshold > 0.0:
            geometry_mask = geometry_mask & (geometry_peak_probability >= peak_probability_threshold)
        max_entropy = float(config.geometry_max_entropy)
        if max_entropy > 0.0:
            geometry_mask = geometry_mask & (geometry_entropy <= max_entropy)
        max_reproj_error = float(config.geometry_max_reprojection_error)
        if max_reproj_error > 0.0:
            geometry_mask = geometry_mask & (geometry_error <= max_reproj_error)
        if pose_guard_enabled and (not pose_guard_passed) and float(config.geometry_pose_guard_softness) <= 0.0:
            geometry_mask = geometry_mask & torch.zeros_like(geometry_mask)
        geometry_weights = geometry_weights * geometry_mask.to(dtype)
        geometry_weights = geometry_weights * condition_guard_scale.detach().to(dtype)
        if pose_guard_enabled:
            geometry_weights = geometry_weights * geometry_pose_guard_scale.detach().to(dtype)
        geometry_kept_mask = geometry_weights.detach() > 0.0
        geometry_correspondences = int((geometry_weights > 0.0).sum().detach().item())
        geometry_weight_sum = float(geometry_weights.sum().detach().item())
        geometry_filter_keep_ratio = float(geometry_correspondences) / max(float(geometry_candidate_count), 1.0)
        geometry_kept_confidence_mean = _masked_float_stat(geometry_confidence, geometry_kept_mask, "mean")
        geometry_kept_confidence_max = _masked_float_stat(geometry_confidence, geometry_kept_mask, "max")
        geometry_kept_margin_mean = _masked_float_stat(geometry_margin, geometry_kept_mask, "mean")
        geometry_kept_margin_max = _masked_float_stat(geometry_margin, geometry_kept_mask, "max")
        geometry_kept_peak_probability_mean = _masked_float_stat(
            geometry_peak_probability, geometry_kept_mask, "mean"
        )
        geometry_kept_peak_probability_min = _masked_float_stat(
            geometry_peak_probability, geometry_kept_mask, "min"
        )
        geometry_kept_peak_probability_max = _masked_float_stat(
            geometry_peak_probability, geometry_kept_mask, "max"
        )
        geometry_kept_entropy_mean = _masked_float_stat(geometry_entropy, geometry_kept_mask, "mean")
        geometry_kept_entropy_max = _masked_float_stat(geometry_entropy, geometry_kept_mask, "max")
        geometry_kept_reprojection_error_mean = _masked_float_stat(geometry_error, geometry_kept_mask, "mean")
        geometry_kept_reprojection_error_max = _masked_float_stat(geometry_error, geometry_kept_mask, "max")
        geometry_reprojection_loss = _weighted_point_loss(
            gt_uv_geometry,
            geometry_uv.detach(),
            geometry_weights,
            loss_type=config.reprojection_loss_type,
            robust_delta=config.reprojection_loss_delta,
        )
    if float(config.geometry_match_reprojection_weight) > 0.0:
        if bool(config.geometry_use_all_correspondences):
            match_idx = all_valid_idx
        else:
            match_idx = valid_idx
        geometry_match_candidate_count = int(match_idx.numel())
        if geometry_match_candidate_count > 0:
            points_detached = points.detach()
            gt_uv_all, gt_valid_all = project_points(points_detached, K, pose_gt_w2c)
            match_gt_uv = gt_uv_all[match_idx].detach()
            match_gt_valid = gt_valid_all[match_idx].detach()
            match_local_window_radius = float(config.geometry_local_window_radius)
            if match_local_window_radius > 0.0:
                gaussian_matrix = _as_feature_matrix(gaussian_features).to(device=device, dtype=dtype)
                match_local = soft_3d_to_2d_correspondences(
                    gaussian_matrix[match_idx],
                    feature_map.detach(),
                    temperature=config.temperature,
                    projected_uv=match_gt_uv,
                    local_window_radius=match_local_window_radius,
                )
                match_uv = match_local.uv.to(device=device, dtype=dtype)
                match_confidence = match_local.confidence.to(device=device, dtype=dtype)
                match_margin = match_local.margin.to(device=device, dtype=dtype)
                match_peak_probability = match_local.peak_probability.to(device=device, dtype=dtype)
                match_entropy = match_local.entropy.to(device=device, dtype=dtype)
                match_valid = match_local.valid.to(device=device)
            else:
                match_uv = correspondences.uv[match_idx].to(device=device, dtype=dtype)
                match_confidence = correspondences.confidence[match_idx].to(device=device, dtype=dtype)
                match_margin = correspondences.margin[match_idx].to(device=device, dtype=dtype)
                match_peak_probability = correspondences.peak_probability[match_idx].to(device=device, dtype=dtype)
                match_entropy = correspondences.entropy[match_idx].to(device=device, dtype=dtype)
                match_valid = correspondences.valid[match_idx].to(device=device)

            match_error = torch.linalg.norm(match_gt_uv - match_uv.detach(), dim=1)
            match_mask = match_gt_valid.to(dtype=torch.bool)
            match_mask = match_mask & match_valid.detach().to(dtype=torch.bool)
            match_mask = match_mask & torch.isfinite(match_gt_uv).all(dim=1)
            match_mask = match_mask & torch.isfinite(match_uv.detach()).all(dim=1)
            match_mask = match_mask & torch.isfinite(match_error.detach())
            confidence_threshold = _override_or_fallback(
                config.geometry_match_confidence_threshold,
                config.geometry_confidence_threshold,
            )
            if confidence_threshold > 0.0:
                match_mask = match_mask & (match_confidence >= confidence_threshold)
            margin_threshold = _override_or_fallback(
                config.geometry_match_margin_threshold,
                config.geometry_margin_threshold,
            )
            if margin_threshold > 0.0:
                match_mask = match_mask & (match_margin >= margin_threshold)
            peak_probability_threshold = _override_or_fallback(
                config.geometry_match_peak_probability_threshold,
                config.geometry_peak_probability_threshold,
            )
            if peak_probability_threshold > 0.0:
                match_mask = match_mask & (match_peak_probability >= peak_probability_threshold)
            max_entropy = _override_or_fallback(
                config.geometry_match_max_entropy,
                config.geometry_max_entropy,
            )
            if max_entropy > 0.0:
                match_mask = match_mask & (match_entropy <= max_entropy)
            max_reproj_error = _override_or_fallback(
                config.geometry_match_max_reprojection_error,
                config.geometry_max_reprojection_error,
            )
            if max_reproj_error > 0.0:
                match_mask = match_mask & (match_error <= max_reproj_error)
            match_weights = match_confidence.detach().clone().clamp_min(1e-4)
            if point_weights_tensor is not None:
                point_weight_floor = float(config.point_weight_floor)
                point_weight_floor = min(max(point_weight_floor, 0.0), 1.0)
                match_point_weights_raw = point_weights_tensor[match_idx].to(device=device, dtype=dtype).clamp(0.0, 1.0)
                match_point_weights = point_weight_floor + (1.0 - point_weight_floor) * match_point_weights_raw
                match_weights = match_weights * match_point_weights.detach().clamp_min(1e-4)
            match_weights = match_weights * match_mask.to(dtype)
            match_weights = match_weights * condition_guard_scale.detach().to(dtype)
            if pose_guard_enabled:
                match_weights = match_weights * geometry_pose_guard_scale.detach().to(dtype)
            geometry_match_correspondences = int((match_weights > 0.0).sum().detach().item())
            geometry_match_weight_sum = float(match_weights.sum().detach().item())
            geometry_match_filter_keep_ratio = float(geometry_match_correspondences) / max(
                float(geometry_match_candidate_count),
                1.0,
            )
            geometry_match_reprojection_loss = _weighted_point_loss(
                match_uv,
                match_gt_uv.detach(),
                match_weights,
                loss_type=config.reprojection_loss_type,
                robust_delta=config.reprojection_loss_delta,
            )
    if (
        allow_geometry_grad
        and float(config.geometry_depth_anchor_weight) > 0.0
        and geometry_anchor_points is not None
    ):
        if bool(config.geometry_use_all_correspondences):
            depth_idx = all_valid_idx
        else:
            depth_idx = valid_idx
        geometry_depth_anchor_candidate_count = int(depth_idx.numel())
        if geometry_depth_anchor_candidate_count > 0:
            depth_anchor = geometry_anchor_points.detach()[depth_idx]
            depth_points = points[depth_idx]
            depth_anchor_uv, depth_anchor_valid = project_points(depth_anchor, K, pose_gt_w2c)
            depth_projected_uv = (
                torch.as_tensor(projected_uv, device=device, dtype=dtype)[depth_idx]
                if projected_uv is not None
                else depth_anchor_uv.detach()
            )
            if float(config.geometry_local_window_radius) > 0.0:
                gaussian_matrix = _as_feature_matrix(gaussian_features).to(device=device, dtype=dtype)
                depth_local = soft_3d_to_2d_correspondences(
                    gaussian_matrix[depth_idx].detach(),
                    feature_map.detach(),
                    temperature=config.temperature,
                    projected_uv=depth_projected_uv.detach(),
                    local_window_radius=float(config.geometry_local_window_radius),
                )
                depth_uv = depth_local.uv.to(device=device, dtype=dtype)
                depth_confidence = depth_local.confidence.to(device=device, dtype=dtype)
                depth_margin = depth_local.margin.to(device=device, dtype=dtype)
                depth_peak_probability = depth_local.peak_probability.to(device=device, dtype=dtype)
                depth_entropy = depth_local.entropy.to(device=device, dtype=dtype)
                depth_valid = depth_local.valid.to(device=device)
            else:
                depth_uv = correspondences.uv[depth_idx].to(device=device, dtype=dtype)
                depth_confidence = correspondences.confidence[depth_idx].to(device=device, dtype=dtype)
                depth_margin = correspondences.margin[depth_idx].to(device=device, dtype=dtype)
                depth_peak_probability = correspondences.peak_probability[depth_idx].to(device=device, dtype=dtype)
                depth_entropy = correspondences.entropy[depth_idx].to(device=device, dtype=dtype)
                depth_valid = correspondences.valid[depth_idx].to(device=device)
            depth_error = torch.linalg.norm(depth_anchor_uv.detach() - depth_uv.detach(), dim=1)
            depth_mask = depth_anchor_valid.detach().to(dtype=torch.bool)
            depth_mask = depth_mask & depth_valid.detach().to(dtype=torch.bool)
            depth_mask = depth_mask & torch.isfinite(depth_anchor_uv.detach()).all(dim=1)
            depth_mask = depth_mask & torch.isfinite(depth_uv.detach()).all(dim=1)
            depth_mask = depth_mask & torch.isfinite(depth_error.detach())
            confidence_threshold = float(config.geometry_confidence_threshold)
            if confidence_threshold > 0.0:
                depth_mask = depth_mask & (depth_confidence >= confidence_threshold)
            margin_threshold = float(config.geometry_margin_threshold)
            if margin_threshold > 0.0:
                depth_mask = depth_mask & (depth_margin >= margin_threshold)
            peak_probability_threshold = float(config.geometry_peak_probability_threshold)
            if peak_probability_threshold > 0.0:
                depth_mask = depth_mask & (depth_peak_probability >= peak_probability_threshold)
            max_entropy = float(config.geometry_max_entropy)
            if max_entropy > 0.0:
                depth_mask = depth_mask & (depth_entropy <= max_entropy)
            max_reproj_error = float(config.geometry_max_reprojection_error)
            if max_reproj_error > 0.0:
                depth_mask = depth_mask & (depth_error <= max_reproj_error)
            if pose_guard_enabled and (not pose_guard_passed) and float(config.geometry_pose_guard_softness) <= 0.0:
                depth_mask = depth_mask & torch.zeros_like(depth_mask)
            anchor_cam = (
                pose_gt_w2c[:3, :3] @ depth_anchor.detach().T + pose_gt_w2c[:3, 3:4]
            ).T
            depth_target = _unproject_uv_depth_to_world(
                depth_uv.detach(),
                anchor_cam[:, 2].detach(),
                K,
                pose_gt_w2c,
            )
            depth_weights = depth_confidence.detach().clone().clamp_min(1e-4)
            if point_weights_tensor is not None:
                point_weight_floor = float(config.point_weight_floor)
                point_weight_floor = min(max(point_weight_floor, 0.0), 1.0)
                depth_point_weights_raw = point_weights_tensor[depth_idx].to(device=device, dtype=dtype).clamp(0.0, 1.0)
                depth_point_weights = point_weight_floor + (1.0 - point_weight_floor) * depth_point_weights_raw
                depth_weights = depth_weights * depth_point_weights.detach().clamp_min(1e-4)
            depth_weights = depth_weights * depth_mask.to(dtype)
            depth_weights = depth_weights * condition_guard_scale.detach().to(dtype)
            if pose_guard_enabled:
                depth_weights = depth_weights * geometry_pose_guard_scale.detach().to(dtype)
            geometry_depth_anchor_correspondences = int((depth_weights > 0.0).sum().detach().item())
            geometry_depth_anchor_weight_sum = float(depth_weights.sum().detach().item())
            geometry_depth_anchor_filter_keep_ratio = float(geometry_depth_anchor_correspondences) / max(
                float(geometry_depth_anchor_candidate_count),
                1.0,
            )
            geometry_depth_anchor_loss = _weighted_xyz_loss(
                depth_points,
                depth_target.detach(),
                depth_weights,
            )
    entropy_loss = correspondences.entropy[valid_idx].mean()
    gt_reprojection_feedback_scale = pose_loss.new_tensor(
        1.0 if bool(config.feedback_pose_guard_keep_gt_reprojection) else float(feedback_scale.detach().item())
    ) * condition_guard_scale
    loss = (
        feedback_loss_scale * (float(config.pose_weight) * pose_loss)
        + feedback_loss_scale * (float(config.reprojection_weight) * reprojection_loss)
        + gt_reprojection_feedback_scale * (float(config.gt_reprojection_weight) * gt_reprojection_loss)
        + feedback_loss_scale * (float(config.entropy_weight) * entropy_loss)
        + float(config.geometry_reprojection_weight) * geometry_reprojection_loss
        + float(config.geometry_match_reprojection_weight) * geometry_match_reprojection_loss
        + float(config.geometry_depth_anchor_weight) * geometry_depth_anchor_loss
    )
    return DifferentiablePnPOutput(
        loss=loss,
        pose_loss=pose_loss,
        reprojection_loss=reprojection_loss,
        gt_reprojection_loss=gt_reprojection_loss,
        geometry_reprojection_loss=geometry_reprojection_loss,
        geometry_match_reprojection_loss=geometry_match_reprojection_loss,
        geometry_depth_anchor_loss=geometry_depth_anchor_loss,
        entropy_loss=entropy_loss,
        pose_w2c=pose_pred,
        correspondences=correspondences,
        used_correspondences=used,
        diagnostics={
            "skipped": 0.0,
            "condition_number": float(condition_number.item()),
            "condition_guard_max_condition_number": float(config.max_condition_number),
            "condition_guard_scale": float(condition_guard_scale.detach().item()),
            "condition_guard_enabled": 1.0 if condition_guard_enabled else 0.0,
            "condition_guard_passed": 1.0 if condition_guard_passed else 0.0,
            "initial_rmse": float(info["initial_rmse"].detach().item()),
            "final_rmse": float(info["final_rmse"].detach().item()),
            "reprojection_loss_type": str(config.reprojection_loss_type),
            "reprojection_loss_delta": float(config.reprojection_loss_delta),
            "geometry_correspondences": float(geometry_correspondences),
            "geometry_candidate_count": float(geometry_candidate_count),
            "geometry_weight_sum": float(geometry_weight_sum),
            "geometry_source": geometry_source,
            "geometry_valid_candidate_count": float(geometry_valid_candidate_count),
            "geometry_filter_keep_ratio": float(geometry_filter_keep_ratio),
            "geometry_candidate_confidence_mean": float(geometry_candidate_confidence_mean),
            "geometry_candidate_confidence_max": float(geometry_candidate_confidence_max),
            "geometry_kept_confidence_mean": float(geometry_kept_confidence_mean),
            "geometry_kept_confidence_max": float(geometry_kept_confidence_max),
            "geometry_candidate_margin_mean": float(geometry_candidate_margin_mean),
            "geometry_candidate_margin_max": float(geometry_candidate_margin_max),
            "geometry_kept_margin_mean": float(geometry_kept_margin_mean),
            "geometry_kept_margin_max": float(geometry_kept_margin_max),
            "geometry_candidate_peak_probability_mean": float(geometry_candidate_peak_probability_mean),
            "geometry_candidate_peak_probability_min": float(geometry_candidate_peak_probability_min),
            "geometry_candidate_peak_probability_max": float(geometry_candidate_peak_probability_max),
            "geometry_kept_peak_probability_mean": float(geometry_kept_peak_probability_mean),
            "geometry_kept_peak_probability_min": float(geometry_kept_peak_probability_min),
            "geometry_kept_peak_probability_max": float(geometry_kept_peak_probability_max),
            "geometry_candidate_entropy_mean": float(geometry_candidate_entropy_mean),
            "geometry_candidate_entropy_max": float(geometry_candidate_entropy_max),
            "geometry_kept_entropy_mean": float(geometry_kept_entropy_mean),
            "geometry_kept_entropy_max": float(geometry_kept_entropy_max),
            "geometry_candidate_reprojection_error_mean": float(geometry_candidate_reprojection_error_mean),
            "geometry_candidate_reprojection_error_max": float(geometry_candidate_reprojection_error_max),
            "geometry_kept_reprojection_error_mean": float(geometry_kept_reprojection_error_mean),
            "geometry_kept_reprojection_error_max": float(geometry_kept_reprojection_error_max),
            "geometry_reprojection_weight": float(config.geometry_reprojection_weight),
            "geometry_depth_anchor_loss": float(geometry_depth_anchor_loss.detach().item()),
            "geometry_depth_anchor_weight": float(config.geometry_depth_anchor_weight),
            "geometry_depth_anchor_correspondences": float(geometry_depth_anchor_correspondences),
            "geometry_depth_anchor_candidate_count": float(geometry_depth_anchor_candidate_count),
            "geometry_depth_anchor_weight_sum": float(geometry_depth_anchor_weight_sum),
            "geometry_depth_anchor_filter_keep_ratio": float(geometry_depth_anchor_filter_keep_ratio),
            "geometry_match_reprojection_weight": float(config.geometry_match_reprojection_weight),
            "geometry_match_confidence_threshold": _override_or_fallback(
                config.geometry_match_confidence_threshold,
                config.geometry_confidence_threshold,
            ),
            "geometry_match_margin_threshold": _override_or_fallback(
                config.geometry_match_margin_threshold,
                config.geometry_margin_threshold,
            ),
            "geometry_match_peak_probability_threshold": _override_or_fallback(
                config.geometry_match_peak_probability_threshold,
                config.geometry_peak_probability_threshold,
            ),
            "geometry_match_max_entropy": _override_or_fallback(
                config.geometry_match_max_entropy,
                config.geometry_max_entropy,
            ),
            "geometry_match_max_reprojection_error": _override_or_fallback(
                config.geometry_match_max_reprojection_error,
                config.geometry_max_reprojection_error,
            ),
            "geometry_match_correspondences": float(geometry_match_correspondences),
            "geometry_match_candidate_count": float(geometry_match_candidate_count),
            "geometry_match_weight_sum": float(geometry_match_weight_sum),
            "geometry_match_filter_keep_ratio": float(geometry_match_filter_keep_ratio),
            "geometry_confidence_threshold": float(config.geometry_confidence_threshold),
            "geometry_margin_threshold": float(config.geometry_margin_threshold),
            "geometry_peak_probability_threshold": float(config.geometry_peak_probability_threshold),
            "geometry_max_entropy": float(config.geometry_max_entropy),
            "geometry_max_reprojection_error": float(config.geometry_max_reprojection_error),
            "geometry_use_all_correspondences": 1.0
            if bool(config.geometry_use_all_correspondences)
            else 0.0,
            "geometry_local_window_radius": float(config.geometry_local_window_radius),
            "geometry_pose_guard_max_loss_increase": float(config.geometry_pose_guard_max_loss_increase),
            "geometry_pose_guard_max_loss": float(config.geometry_pose_guard_max_loss),
            "geometry_pose_guard_softness": float(config.geometry_pose_guard_softness),
            "geometry_pose_guard_min_scale": float(config.geometry_pose_guard_min_scale),
            "geometry_pose_guard_scale": float(geometry_pose_guard_scale.detach().item()),
            "geometry_pose_guard_violation": float(geometry_pose_guard_violation.detach().item()),
            "geometry_pose_guard_enabled": 1.0 if pose_guard_enabled else 0.0,
            "geometry_pose_guard_passed": 1.0 if pose_guard_passed else 0.0,
            "feedback_pose_guard_max_loss_increase": float(config.feedback_pose_guard_max_loss_increase),
            "feedback_pose_guard_max_loss": float(config.feedback_pose_guard_max_loss),
            "feedback_pose_guard_softness": float(config.feedback_pose_guard_softness),
            "feedback_pose_guard_min_scale": float(config.feedback_pose_guard_min_scale),
            "feedback_pose_guard_scale": float(feedback_scale.detach().item()),
            "feedback_pose_guard_violation": float(feedback_pose_guard_violation.detach().item()),
            "feedback_pose_guard_enabled": 1.0 if feedback_pose_guard_enabled else 0.0,
            "feedback_pose_guard_passed": 1.0 if feedback_pose_guard_passed else 0.0,
            "feedback_pose_guard_keep_gt_reprojection": 1.0
            if bool(config.feedback_pose_guard_keep_gt_reprojection)
            else 0.0,
            "feedback_gt_reprojection_scale": float(gt_reprojection_feedback_scale.detach().item()),
            "pose_loss": float(pose_loss.detach().item()),
            "init_pose_loss": float(init_pose_loss.detach().item()),
            "pose_loss_delta": float((pose_loss.detach() - init_pose_loss.detach()).item()),
            "selected_margin_mean": float(correspondences.margin[valid_idx].detach().mean().item()),
            "selected_peak_probability_mean": float(
                correspondences.peak_probability[valid_idx].detach().mean().item()
            ),
            "detach_gt_reprojection_points": 1.0
            if bool(config.detach_gt_reprojection_points)
            else 0.0,
            "detach_pnp_points": 1.0 if detach_pnp_points else 0.0,
            **selection_diagnostics,
        },
    )


def _landmark_stat_vector(value, count, device, dtype, default=None):
    if value is None:
        if default is None:
            return None
        value = default
    value = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if value.numel() == 0:
        if default is None:
            return None
        value = torch.as_tensor(default, device=device, dtype=dtype).reshape(-1)
    if value.numel() == 1:
        value = value.expand(count)
    if value.numel() < count:
        pad_value = value[-1] if value.numel() > 0 else torch.as_tensor(0.0, device=device, dtype=dtype)
        value = torch.cat([value, pad_value.expand(count - value.numel())], dim=0)
    return value[:count]


def _margin_quality(margin):
    return (0.5 + 0.5 * margin.clamp(-1.0, 1.0)).clamp(0.0, 1.0)


def pnp_output_to_landmark_stats(
    pnp_output,
    points_world,
    K,
    pose_gt_w2c,
    eps=1e-6,
    full_bank_positive_prob=None,
    full_bank_margin=None,
    pose_loss_scale=1.0,
    reprojection_error_scale=4.0,
):
    """Convert DiffPnP correspondences into Gaussian localization stats."""
    correspondences = pnp_output.correspondences
    uv = correspondences.uv
    count = int(uv.shape[0])
    dtype = uv.dtype
    device = uv.device
    points_world = points_world.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    pose_gt_w2c = pose_gt_w2c.to(device=device, dtype=dtype)
    gt_uv, gt_valid = project_points(points_world, K, pose_gt_w2c)
    reproj_error = torch.linalg.norm(gt_uv - uv, dim=-1)
    reproj_error = torch.where(gt_valid & torch.isfinite(reproj_error), reproj_error, torch.zeros_like(reproj_error))
    confidence = correspondences.confidence.detach().to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0)
    entropy = correspondences.entropy.detach().to(device=device, dtype=dtype).reshape(-1).clamp(0.0, 1.0)
    margin = correspondences.margin.detach().to(device=device, dtype=dtype).reshape(-1)
    if full_bank_positive_prob is None:
        full_bank_quality = torch.ones_like(confidence)
        full_bank_confidence = confidence
    else:
        full_bank_quality = _landmark_stat_vector(
            full_bank_positive_prob,
            count,
            device,
            dtype,
            default=torch.ones((), device=device, dtype=dtype),
        ).clamp(0.0, 1.0)
        full_bank_confidence = full_bank_quality
    full_bank_margin_quality = torch.ones_like(confidence)
    if full_bank_margin is not None:
        full_bank_margin_quality = _margin_quality(
            _landmark_stat_vector(full_bank_margin, count, device, dtype, default=0.0)
        )
    soft_margin_quality = _margin_quality(margin)
    reprojection_scale = max(float(reprojection_error_scale), float(eps))
    reprojection_quality = torch.exp(-reproj_error.detach().clamp_min(0.0) / reprojection_scale).clamp(0.0, 1.0)
    pose_loss = torch.as_tensor(
        getattr(pnp_output, "pose_loss", uv.new_tensor(0.0)),
        device=device,
        dtype=dtype,
    ).detach().reshape(-1)
    pose_loss_value = pose_loss[0] if pose_loss.numel() > 0 else uv.new_tensor(0.0)
    if float(pose_loss_scale) > 0.0:
        pose_quality = torch.exp(-pose_loss_value.clamp_min(0.0) / float(pose_loss_scale)).clamp(0.0, 1.0)
    else:
        pose_quality = uv.new_tensor(1.0)
    loc_utility = (
        confidence
        * soft_margin_quality
        * (1.0 - entropy).clamp(0.0, 1.0)
        * reprojection_quality
        * full_bank_quality
        * full_bank_margin_quality
        * pose_quality
    ).clamp(0.0, 1.0)
    return {
        "positive_prob": confidence,
        "full_bank_positive_prob": full_bank_confidence,
        "margin": margin,
        "entropy": entropy,
        "reproj_error": reproj_error.detach(),
        "information": loc_utility.detach(),
        "repeatability": loc_utility.detach(),
        "outlier": (1.0 - loc_utility).clamp(0.0, 1.0).detach(),
        "loc_utility": loc_utility.detach(),
        "update_mask": (correspondences.valid.to(device=device) & gt_valid).detach(),
    }


def update_diff_pnp_training_summary(summary, pnp_output, loss, allow_geometry_grad=False):
    used = int(getattr(pnp_output, "used_correspondences", 0))
    diagnostics = getattr(pnp_output, "diagnostics", {}) or {}
    if used <= 0 or bool(float(diagnostics.get("skipped", 0.0)) > 0.0):
        summary["diff_pnp_skipped_episodes"] = summary.get("diff_pnp_skipped_episodes", 0) + 1
        reason = str(diagnostics.get("skipped_reason", "unknown"))
        summary[f"diff_pnp_skipped_{reason}_episodes"] = (
            summary.get(f"diff_pnp_skipped_{reason}_episodes", 0) + 1
        )
        return summary

    summary["diff_pnp_episodes"] = summary.get("diff_pnp_episodes", 0) + 1
    summary["diff_pnp_used_correspondences_total"] = (
        summary.get("diff_pnp_used_correspondences_total", 0) + used
    )
    loss_value = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
    summary["diff_pnp_loss_total"] = summary.get("diff_pnp_loss_total", 0.0) + loss_value
    if loss_value > 0.0:
        summary["diff_pnp_nonzero_loss_episodes"] = (
            summary.get("diff_pnp_nonzero_loss_episodes", 0) + 1
        )
    if bool(allow_geometry_grad):
        summary["diff_pnp_allow_geometry_grad_episodes"] = (
            summary.get("diff_pnp_allow_geometry_grad_episodes", 0) + 1
        )
    for key in (
        "selected_spatial_cells",
        "candidate_spatial_cells",
        "spatial_min_span",
        "spatial_area",
        "point_weight_mean",
        "point_weight_min",
        "point_weight_max",
        "geometry_correspondences",
        "geometry_candidate_count",
        "geometry_weight_sum",
        "geometry_depth_anchor_loss",
        "geometry_depth_anchor_weight",
        "geometry_depth_anchor_correspondences",
        "geometry_depth_anchor_candidate_count",
        "geometry_depth_anchor_weight_sum",
        "geometry_depth_anchor_filter_keep_ratio",
        "geometry_match_reprojection_weight",
        "geometry_match_confidence_threshold",
        "geometry_match_margin_threshold",
        "geometry_match_peak_probability_threshold",
        "geometry_match_max_entropy",
        "geometry_match_max_reprojection_error",
        "geometry_match_correspondences",
        "geometry_match_candidate_count",
        "geometry_match_weight_sum",
        "geometry_match_filter_keep_ratio",
        "geometry_valid_candidate_count",
        "geometry_filter_keep_ratio",
        "geometry_candidate_confidence_mean",
        "geometry_candidate_confidence_max",
        "geometry_kept_confidence_mean",
        "geometry_kept_confidence_max",
        "geometry_candidate_margin_mean",
        "geometry_candidate_margin_max",
        "geometry_kept_margin_mean",
        "geometry_kept_margin_max",
        "geometry_candidate_peak_probability_mean",
        "geometry_candidate_peak_probability_min",
        "geometry_candidate_peak_probability_max",
        "geometry_kept_peak_probability_mean",
        "geometry_kept_peak_probability_min",
        "geometry_kept_peak_probability_max",
        "geometry_candidate_entropy_mean",
        "geometry_candidate_entropy_max",
        "geometry_kept_entropy_mean",
        "geometry_kept_entropy_max",
        "geometry_candidate_reprojection_error_mean",
        "geometry_candidate_reprojection_error_max",
        "geometry_kept_reprojection_error_mean",
        "geometry_kept_reprojection_error_max",
        "geometry_confidence_threshold",
        "geometry_margin_threshold",
        "geometry_peak_probability_threshold",
        "geometry_max_entropy",
        "geometry_max_reprojection_error",
        "geometry_use_all_correspondences",
        "geometry_local_window_radius",
        "geometry_pose_guard_max_loss_increase",
        "geometry_pose_guard_max_loss",
        "geometry_pose_guard_softness",
        "geometry_pose_guard_min_scale",
        "geometry_pose_guard_scale",
        "geometry_pose_guard_violation",
        "geometry_pose_guard_passed",
        "feedback_pose_guard_max_loss_increase",
        "feedback_pose_guard_max_loss",
        "feedback_pose_guard_softness",
        "feedback_pose_guard_min_scale",
        "feedback_pose_guard_scale",
        "feedback_pose_guard_violation",
        "feedback_pose_guard_passed",
        "feedback_pose_guard_keep_gt_reprojection",
        "feedback_gt_reprojection_scale",
        "condition_guard_max_condition_number",
        "condition_guard_scale",
        "condition_guard_enabled",
        "condition_guard_passed",
        "pose_loss",
        "init_pose_loss",
        "pose_loss_delta",
        "selected_margin_mean",
        "selected_peak_probability_mean",
        "detach_pnp_points",
    ):
        value = diagnostics.get(key)
        if value is None:
            continue
        metric_key = f"diff_pnp_{key}"
        value = float(value)
        summary[f"{metric_key}_total"] = summary.get(f"{metric_key}_total", 0.0) + value
        summary[f"{metric_key}_max"] = max(summary.get(f"{metric_key}_max", 0.0), value)
    return summary


def pose_aware_split_score(
    footprint,
    ambiguity,
    pnp_residual,
    repeatability,
    positive_prob=None,
    pose_information=None,
    min_footprint=0.0,
    min_repeatability=0.25,
    eps=1e-6,
):
    """Score split candidates using localization ambiguity and PnP residual."""
    footprint = torch.as_tensor(footprint).float()
    device = footprint.device
    ambiguity = torch.as_tensor(ambiguity, dtype=torch.float32, device=device).clamp_min(0.0)
    pnp_residual = torch.as_tensor(pnp_residual, dtype=torch.float32, device=device).clamp_min(0.0)
    repeatability = torch.as_tensor(repeatability, dtype=torch.float32, device=device).clamp(0.0, 1.0)
    if positive_prob is None:
        positive_prob = torch.ones_like(repeatability)
    else:
        positive_prob = torch.as_tensor(positive_prob, dtype=torch.float32, device=device).clamp(0.0, 1.0)
    if pose_information is None:
        pose_information = torch.ones_like(repeatability)
    else:
        pose_information = torch.as_tensor(pose_information, dtype=torch.float32, device=device).clamp(0.0, 1.0)
        if not bool((pose_information > 0).any()):
            pose_information = torch.ones_like(repeatability)
    eligible = (footprint >= float(min_footprint)) & (repeatability >= float(min_repeatability))
    residual_scale = pnp_residual / pnp_residual.detach().max().clamp_min(float(eps))
    score = ambiguity * residual_scale * repeatability * positive_prob * pose_information
    score = torch.where(eligible & torch.isfinite(score), score, torch.zeros_like(score))
    return score


def bounded_geometry_residual_loss(current_xyz, source_xyz, scaling=None, max_scale_ratio=0.2, point_weight=None):
    """Penalize Gaussian center movement beyond a scale-relative RGB anchor radius."""
    current_xyz = torch.as_tensor(current_xyz).float()
    source_xyz = torch.as_tensor(source_xyz, dtype=current_xyz.dtype, device=current_xyz.device)
    residual = current_xyz - source_xyz
    residual_norm = torch.linalg.norm(residual, dim=-1)
    if scaling is None:
        allowed = torch.full_like(residual_norm, float(max_scale_ratio))
    else:
        scaling = torch.as_tensor(scaling, dtype=current_xyz.dtype, device=current_xyz.device)
        allowed = scaling.reshape(scaling.shape[0], -1).amax(dim=1).abs() * float(max_scale_ratio)
    excess = (residual_norm - allowed).clamp_min(0.0)
    per_point = residual_norm.square() + excess.square()
    if point_weight is not None:
        point_weight = torch.as_tensor(point_weight, dtype=per_point.dtype, device=per_point.device).reshape(-1)
        loss = (per_point * point_weight).sum() / point_weight.sum().clamp_min(1e-6)
    else:
        loss = per_point.mean() if per_point.numel() else current_xyz.new_tensor(0.0)
    stats = {
        "max_residual_norm": float(residual_norm.detach().max().item()) if residual_norm.numel() else 0.0,
        "mean_residual_norm": float(residual_norm.detach().mean().item()) if residual_norm.numel() else 0.0,
        "max_allowed_norm": float(allowed.detach().max().item()) if allowed.numel() else 0.0,
        "over_limit_count": int((excess.detach() > 0).sum().item()),
    }
    return loss, stats


def lafgs_phase_for_iteration(iteration, config=None):
    config = config or LaFGSCurriculumConfig()
    iteration = int(iteration)
    if iteration <= int(config.mv_init_until):
        return "mv_init"
    if iteration <= int(config.locrec_until):
        return "locrec"
    if iteration <= int(config.diff_pnp_until):
        return "diff_pnp"
    if iteration <= int(config.geometry_until):
        return "geometry"
    return "topology"


def lafgs_phase_from_starts(iteration, locrec_start, diff_pnp_start, geometry_start, topology_start):
    iteration = int(iteration)
    if iteration < int(locrec_start):
        return "mv_init"
    if iteration < int(diff_pnp_start):
        return "locrec"
    if iteration < int(geometry_start):
        return "diff_pnp"
    if iteration < int(topology_start):
        return "geometry"
    return "topology"


def lafgs_curriculum_step(iteration, base_iteration=0):
    """Return the 1-based LaFGS step elapsed since the RGB/loaded-map iteration."""
    return max(1, int(iteration) - int(base_iteration))


def lafgs_should_sample_synthetic_view(loc_teacher, synthetic_view_ratio, query_camera_count, random_value):
    if str(loc_teacher) not in {"dense", "direct"}:
        return False
    return (
        float(synthetic_view_ratio) > 0.0
        and int(query_camera_count) >= 2
        and float(random_value) < float(synthetic_view_ratio)
    )


def lafgs_trainable_param_names(phase):
    phase = str(phase)
    if phase == "mv_init":
        return set()
    if phase in {"locrec", "diff_pnp"}:
        return {"loc_feature", "loc_opacity"}
    if phase == "geometry":
        return {"loc_feature", "loc_opacity", "xyz"}
    if phase == "topology":
        return {"loc_feature", "loc_opacity", "xyz", "scaling", "rotation"}
    raise ValueError(f"Unsupported LaFGS phase: {phase}")
