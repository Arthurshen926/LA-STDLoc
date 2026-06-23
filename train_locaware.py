import os
import pickle
import sys
import uuid
import argparse
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_from_pose_gsplat, render_gsplat
from localization_training.dense_teacher import dense_localization_teacher
from localization_training.direct_landmark_teacher import LandmarkObservationMemory, direct_landmark_teacher
from localization_training.episode_sampler import EpisodeSampler, SparsePoseCache, split_support_query_cameras
from localization_training.losses import (
    geometry_anchor_loss,
    hard_negative_ranking_loss,
    localization_opacity_regularizer,
    prototype_loss,
)
from localization_training.topology_controller import LocalizationTopologyController, TopologyConfig
from scene import Scene
from utils.general_utils import safe_state, seed_everything
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID", str(uuid.uuid4()))
        args.model_path = os.path.join("./output/", unique_str[0:10])
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    return SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None


def _load_masks(dataset):
    candidates = [
        os.path.join(dataset.source_path, dataset.images, "masks.pkl"),
        os.path.join(dataset.source_path, "masks.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            print("Loading masks from", path)
            return pickle.load(open(path, "rb"))
    return None


def _resize_bool_mask(mask, target_hw):
    if mask.shape[-2:] == target_hw:
        return mask.bool()
    return (
        F.interpolate(mask[None].float(), size=target_hw, mode="nearest")
        .squeeze(0)
        .bool()
    )


def _is_locaware_checkpoint(checkpoint_data):
    return isinstance(checkpoint_data, dict) and "model_params" in checkpoint_data


def _split_checkpoint_payload(checkpoint_data):
    if _is_locaware_checkpoint(checkpoint_data):
        return (
            checkpoint_data["model_params"],
            checkpoint_data.get("iteration", 0),
            checkpoint_data.get("localization_state"),
        )
    model_params, first_iter = checkpoint_data
    return model_params, first_iter, None


def _restore_checkpoint(gaussians, opt, checkpoint):
    first_iter = 0
    if not checkpoint:
        gaussians.training_setup(opt)
        return first_iter

    checkpoint_data = torch.load(checkpoint)
    model_params, first_iter, loc_state = _split_checkpoint_payload(checkpoint_data)
    gaussians.restore(model_params, opt)
    if loc_state is not None:
        gaussians.restore_localization_state(loc_state)
    else:
        gaussians.init_localization_state(from_rgb_opacity=True)
    return first_iter


def _restore_external_localization_state(gaussians, state_or_path):
    if not state_or_path:
        return False
    if isinstance(state_or_path, (str, os.PathLike)):
        point_tensor = getattr(gaussians, "get_xyz", None)
        device = point_tensor.device if torch.is_tensor(point_tensor) else "cpu"
        state = torch.load(os.fspath(state_or_path), map_location=device)
    else:
        state = state_or_path
    gaussians.restore_localization_state(state)
    return True


def _capture_geometry_anchor(gaussians):
    return {
        "xyz": gaussians._xyz.detach().clone(),
        "scaling": gaussians._scaling.detach().clone(),
        "rotation": gaussians._rotation.detach().clone(),
    }


def _refresh_geometry_anchor_if_point_count_changed(gaussians, geometry_anchor):
    if geometry_anchor["xyz"].shape[0] != gaussians.get_xyz.shape[0]:
        return _capture_geometry_anchor(gaussians)
    return geometry_anchor


def _current_geometry_state(gaussians):
    return {
        "xyz": gaussians._xyz,
        "scaling": gaussians._scaling,
        "rotation": gaussians._rotation,
    }


def _capture_feature_anchor(gaussians):
    features = gaussians.get_loc_feature.detach().clone()
    node_ids = getattr(gaussians, "loc_node_id", None)
    if torch.is_tensor(node_ids) and node_ids.numel() == features.shape[0]:
        node_ids = node_ids.detach().clone().to(dtype=torch.long, device=features.device)
    else:
        node_ids = torch.arange(features.shape[0], dtype=torch.long, device=features.device)
    return {"node_ids": node_ids, "features": features}


def _feature_anchor_tensor(feature_anchor):
    if feature_anchor is None:
        return None
    if isinstance(feature_anchor, dict):
        return feature_anchor["features"]
    return feature_anchor


def _refresh_feature_anchor_if_point_count_changed(gaussians, feature_anchor):
    if feature_anchor is None:
        return None
    current = gaussians.get_loc_feature.detach()
    if not isinstance(feature_anchor, dict):
        if feature_anchor.shape[0] == current.shape[0]:
            return feature_anchor
        if feature_anchor.shape[0] < current.shape[0]:
            return torch.cat([feature_anchor, current[feature_anchor.shape[0] :].clone()], dim=0)
        return feature_anchor[: current.shape[0]].clone()

    anchor_features = feature_anchor["features"].detach()
    anchor_node_ids = feature_anchor["node_ids"].detach().to(dtype=torch.long, device=current.device).reshape(-1)
    current_node_ids = getattr(gaussians, "loc_node_id", None)
    if torch.is_tensor(current_node_ids) and current_node_ids.numel() == current.shape[0]:
        current_node_ids = current_node_ids.detach().to(dtype=torch.long, device=current.device).reshape(-1)
    else:
        current_node_ids = torch.arange(current.shape[0], dtype=torch.long, device=current.device)
    if (
        anchor_features.shape[0] == current.shape[0]
        and anchor_node_ids.shape[0] == current_node_ids.shape[0]
        and torch.equal(anchor_node_ids, current_node_ids)
    ):
        return feature_anchor

    anchor_features = anchor_features.to(device=current.device, dtype=current.dtype)
    aligned = current.clone()
    anchor_pos = {int(node_id): idx for idx, node_id in enumerate(anchor_node_ids.detach().cpu().tolist())}
    for row, node_id in enumerate(current_node_ids.detach().cpu().tolist()):
        idx = anchor_pos.get(int(node_id))
        if idx is not None:
            aligned[row] = anchor_features[idx]
    return {"node_ids": current_node_ids.detach().clone(), "features": aligned.detach().clone()}


def _load_landmark_indices(model_path, landmark_path, device="cpu"):
    path = landmark_path
    if not os.path.isabs(path):
        path = os.path.join(model_path, path)
    with open(path, "rb") as f:
        indices = pickle.load(f)
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _current_landmark_indices_from_source_index(source_landmark_indices, gaussians):
    source_landmark_indices = torch.as_tensor(source_landmark_indices, dtype=torch.long).reshape(-1)
    source_index = getattr(gaussians, "loc_source_index", None)
    point_count = int(gaussians.get_xyz.shape[0])
    if not torch.is_tensor(source_index) or source_index.numel() != point_count:
        return source_landmark_indices.detach().cpu()
    if source_landmark_indices.numel() == 0:
        return source_landmark_indices.detach().cpu()
    source_index = source_index.to(dtype=torch.long)
    wanted = source_landmark_indices.to(device=source_index.device)
    current_mask = torch.isin(source_index, wanted)
    current = torch.nonzero(current_mask, as_tuple=False).squeeze(1).to(dtype=torch.long)
    if current.numel() == 0:
        return source_landmark_indices.detach().cpu()
    return current.detach().cpu()


def _flatten_render_map(value):
    if value is None:
        return None
    while value.dim() > 2:
        value = value.squeeze(0)
    return value


def _flatten_render_alpha(value):
    if value is None:
        return None
    value = value.squeeze()
    if value.dim() == 3:
        value = value[..., 0]
    return value


def _set_phase_lrs(gaussians, phase, args):
    for group in gaussians.optimizer.param_groups:
        group.setdefault("la_base_lr", group["lr"])
        group["lr"] = group["la_base_lr"]

    if phase == "feature":
        trainable = {"loc_feature"}
    elif phase in {"geometry", "topology", "closed_loop"}:
        trainable = {"xyz", "scaling", "rotation", "loc_feature", "loc_opacity"}
    else:
        trainable = {group["name"] for group in gaussians.optimizer.param_groups}

    for group in gaussians.optimizer.param_groups:
        if group["name"] not in trainable:
            group["lr"] = 0.0
        elif phase in {"geometry", "topology", "closed_loop"}:
            if group["name"] == "xyz":
                group["lr"] = group["la_base_lr"] * args.geometry_xyz_lr_mult
            elif group["name"] == "scaling":
                group["lr"] = group["la_base_lr"] * args.geometry_scale_lr_mult
            elif group["name"] == "rotation":
                group["lr"] = group["la_base_lr"] * args.geometry_rotation_lr_mult


def add_locaware_training_args(parser):
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--load_iteration", type=int, default=None)
    parser.add_argument("--localization_state_path", type=str, default=None)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--localization_enabled", action="store_true", default=True)
    parser.add_argument("--feature_only", action="store_true", default=False)
    parser.add_argument("--train_phase", type=str, default="feature", choices=["feature", "geometry", "topology", "closed_loop", "full"])
    parser.add_argument("--base_loss_weight", type=float, default=1.0)
    parser.add_argument("--base_feature_weight", type=float, default=1.0)
    parser.add_argument("--loc_loss_weight", type=float, default=1.0)
    parser.add_argument("--loc_start_iter", type=int, default=1)
    parser.add_argument("--loc_interval", type=int, default=8)
    parser.add_argument("--loc_anchors", type=int, default=512)
    parser.add_argument("--loc_alpha_threshold", type=float, default=0.2)
    parser.add_argument("--loc_desc_temperature", type=float, default=0.07)
    parser.add_argument("--loc_fine_temperature", type=float, default=0.05)
    parser.add_argument("--loc_fine_window_radius", type=int, default=4)
    parser.add_argument("--loc_desc_weight", type=float, default=1.0)
    parser.add_argument("--loc_reproj_weight", type=float, default=0.1)
    parser.add_argument("--loc_dense_kl_weight", type=float, default=0.0)
    parser.add_argument("--loc_dense_kl_temperature", type=float, default=0.07)
    parser.add_argument("--loc_responsibility_topk", type=int, default=32)
    parser.add_argument("--loc_responsibility_opacity_weight", type=float, default=0.0)
    parser.add_argument("--loc_responsibility_depth_weight", type=float, default=0.0)
    parser.add_argument("--loc_teacher", type=str, default="dense", choices=["dense", "direct"])
    parser.add_argument("--loc_direct_weight", type=float, default=0.1)
    parser.add_argument("--loc_multiview_weight", type=float, default=0.05)
    parser.add_argument("--loc_multiview_temperature", type=float, default=0.07)
    parser.add_argument("--loc_multiview_slots", type=int, default=4)
    parser.add_argument("--loc_multiview_ignore_radius", type=float, default=2.0)
    parser.add_argument("--loc_full_bank_weight", type=float, default=0.0)
    parser.add_argument("--loc_full_bank_temperature", type=float, default=0.07)
    parser.add_argument("--loc_full_bank_hard_negatives", type=int, default=32)
    parser.add_argument("--loc_full_bank_margin", type=float, default=0.2)
    parser.add_argument("--loc_anchor_weight", type=float, default=0.0)
    parser.add_argument("--landmark_path", type=str, default="detector/sampled_idx.pkl")
    parser.add_argument("--direct_depth_check", action="store_true", default=False)
    parser.add_argument("--direct_depth_abs_tolerance", type=float, default=1e-3)
    parser.add_argument("--direct_depth_rel_tolerance", type=float, default=0.01)
    parser.add_argument("--loc_proto_weight", type=float, default=0.0)
    parser.add_argument("--loc_rank_weight", type=float, default=0.0)
    parser.add_argument("--loc_rank_margin", type=float, default=0.2)
    parser.add_argument("--loc_opacity_weight", type=float, default=0.0)
    parser.add_argument("--loc_opacity_target", type=float, default=0.5)
    parser.add_argument("--loc_ema_decay", type=float, default=0.95)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--use_loc_opacity", action=argparse.BooleanOptionalAction, default=False)
    else:
        parser.add_argument("--use_loc_opacity", dest="use_loc_opacity", action="store_true")
        parser.add_argument("--no-use_loc_opacity", dest="use_loc_opacity", action="store_false")
        parser.set_defaults(use_loc_opacity=False)
    parser.add_argument("--query_mode", type=str, default="noise", choices=["noise", "sparse", "mixed"])
    parser.add_argument("--pose_noise_quantile", type=float, default=0.5)
    parser.add_argument("--pose_noise_sampling", type=str, default="empirical", choices=["empirical", "quantile"])
    parser.add_argument("--mixed_sparse_probability", type=float, default=0.5)
    parser.add_argument("--sparse_pose_cache", type=str, default=None)
    parser.add_argument("--support_query_split", action="store_true", default=False)
    parser.add_argument("--query_holdout_ratio", type=float, default=0.2)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--query_split_seed", type=int, default=2025)
    parser.add_argument("--query_split_mode", type=str, default="random", choices=["random", "sequence_block", "temporal_block"])
    parser.add_argument("--loc_anchor_grid_size", type=int, default=8)
    parser.add_argument("--geometry_anchor_weight", type=float, default=0.0)
    parser.add_argument("--geometry_anchor_scale_weight", type=float, default=0.1)
    parser.add_argument("--geometry_anchor_rotation_weight", type=float, default=0.1)
    parser.add_argument("--geometry_xyz_lr_mult", type=float, default=0.05)
    parser.add_argument("--geometry_scale_lr_mult", type=float, default=0.1)
    parser.add_argument("--geometry_rotation_lr_mult", type=float, default=0.1)
    parser.add_argument("--enable_topology", action="store_true", default=False)
    parser.add_argument("--topology_stats_warmup", type=int, default=1000)
    parser.add_argument("--topology_update_interval", type=int, default=200)
    parser.add_argument("--topology_min_observations", type=int, default=8)
    parser.add_argument("--topology_split_quantile", type=float, default=0.95)
    parser.add_argument("--topology_ambiguity_quantile", type=float, default=0.90)
    parser.add_argument("--topology_growth_cap_per_event", type=float, default=0.03)
    parser.add_argument("--topology_total_point_budget_ratio", type=float, default=1.25)
    parser.add_argument("--topology_cooldown_iterations", type=int, default=300)
    parser.add_argument("--topology_disable_split", action="store_true", default=False)
    parser.add_argument("--topology_min_repeatability", type=float, default=0.25)
    parser.add_argument("--topology_min_radius", type=float, default=4.0)
    parser.add_argument("--topology_enable_soft_prune", action="store_true", default=False)
    parser.add_argument("--topology_enable_physical_prune", action="store_true", default=False)
    parser.add_argument("--topology_protect_landmarks", action="store_true", default=False)
    parser.add_argument("--topology_soft_prune_threshold", type=float, default=-1.0)
    parser.add_argument("--topology_soft_prune_step", type=float, default=1.0)
    parser.add_argument("--topology_physical_rgb_threshold", type=float, default=0.005)
    parser.add_argument("--topology_physical_loc_threshold", type=float, default=0.005)
    parser.add_argument("--topology_physical_utility_threshold", type=float, default=-3.0)
    parser.add_argument("--topology_allow_untrained_loc_opacity_prune", action="store_true", default=False)
    return parser


def _base_losses(viewpoint_cam, render_pkg, feature_extractor, dataset, masks=None):
    image = render_pkg["render"]
    feature_map = render_pkg["feature_map"]
    original_image = viewpoint_cam.original_image.cuda()
    gt_image = F.interpolate(
        original_image.unsqueeze(0),
        size=(image.shape[1], image.shape[2]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    mask = None
    sky_mask = None
    if masks is not None:
        obj_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][0].cuda()[None], image.shape[-2:])
        sky_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][1].cuda()[None], image.shape[-2:])
        distort_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][2].cuda()[None], image.shape[-2:])
        mask = obj_mask & distort_mask
        image = image * mask
        gt_image = gt_image * mask
        gt_image[sky_mask.repeat(3, 1, 1) == False] = 1

    Ll1 = l1_loss(image, gt_image)
    if feature_map is None:
        Ll1_feature = image.new_tensor(0.0)
        gt_feature_map = None
    else:
        with torch.no_grad():
            gt_feature_map = feature_extractor(original_image[None])["feature_map"][0]
            gt_feature_map = F.interpolate(
                gt_feature_map.unsqueeze(0),
                size=(feature_map.shape[1], feature_map.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
        if mask is not None:
            feature_map_mask = _resize_bool_mask(mask, (gt_feature_map.shape[1], gt_feature_map.shape[2]))
            feature_map = feature_map * feature_map_mask
            gt_feature_map = gt_feature_map * feature_map_mask
        Ll1_feature = l1_loss(feature_map, gt_feature_map)

    return {
        "image": image,
        "gt_image": gt_image,
        "gt_feature_map": gt_feature_map,
        "Ll1": Ll1,
        "Ll1_feature": Ll1_feature,
    }


def _query_feature_map(viewpoint_cam, feature_extractor, target_hw, masks=None):
    original_image = viewpoint_cam.original_image.cuda()
    with torch.no_grad():
        gt_feature_map = feature_extractor(original_image[None])["feature_map"][0]
        gt_feature_map = F.interpolate(
            gt_feature_map.unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
    if masks is not None:
        obj_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][0].cuda()[None], target_hw)
        distort_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][2].cuda()[None], target_hw)
        gt_feature_map = gt_feature_map * (obj_mask & distort_mask)
    return gt_feature_map


def training(dataset, opt, args):
    print(opt)
    tb_writer = prepare_output_and_logger(dataset)
    print("Feature type:", dataset.feature_type)
    print("Gaussian type:", dataset.gaussian_type)
    if dataset.gaussian_type != "3dgs":
        raise ValueError("LA-STDLoc MVP currently supports gaussian_type=3dgs")

    from scene.gaussian_model import GaussianModel

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration)
    masks = _load_masks(dataset)
    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    first_iter = _restore_checkpoint(gaussians, opt, args.start_checkpoint)
    if args.localization_state_path:
        _restore_external_localization_state(gaussians, args.localization_state_path)
        print(f"Loaded external localization state from {args.localization_state_path}")
    if first_iter == 0 and scene.loaded_iter:
        first_iter = scene.loaded_iter
    geometry_anchor = _capture_geometry_anchor(gaussians)
    loc_feature_anchor = _capture_feature_anchor(gaussians) if args.loc_anchor_weight > 0 else None
    sparse_pose_cache = None
    if args.sparse_pose_cache:
        sparse_pose_cache = SparsePoseCache(args.sparse_pose_cache).load()
    episode_sampler = EpisodeSampler(
        sparse_pose_cache=sparse_pose_cache,
        query_mode=args.query_mode,
        noise_quantile=args.pose_noise_quantile,
        mixed_sparse_probability=args.mixed_sparse_probability,
        noise_sampling=args.pose_noise_sampling,
    )
    direct_landmark_indices = None
    direct_observation_memory = None
    if args.loc_teacher == "direct":
        direct_landmark_indices = _load_landmark_indices(dataset.model_path, args.landmark_path, device="cpu")
        print(f"Loaded {direct_landmark_indices.numel()} direct teacher landmarks from {args.landmark_path}")
        if args.loc_multiview_weight > 0:
            feature_dim = gaussians.get_loc_feature.reshape(gaussians.get_xyz.shape[0], -1).shape[1]
            direct_observation_memory = LandmarkObservationMemory(
                direct_landmark_indices,
                feature_dim=feature_dim,
                slots=args.loc_multiview_slots,
                device=gaussians.get_xyz.device,
            )
            print(
                "Initialized direct multi-view memory: "
                f"landmarks={direct_landmark_indices.numel()} slots={args.loc_multiview_slots}"
            )
    topology_controller = None
    if args.enable_topology or args.train_phase in {"topology", "closed_loop"}:
        protected_source_indices = None
        if args.topology_protect_landmarks:
            protected_source_indices = _load_landmark_indices(dataset.model_path, args.landmark_path, device="cpu")
            print(f"Protecting {protected_source_indices.numel()} sparse landmark source ids from physical prune")
        topology_controller = LocalizationTopologyController(
            TopologyConfig(
                stats_warmup=args.topology_stats_warmup,
                update_interval=args.topology_update_interval,
                min_observations=args.topology_min_observations,
                split_quantile=args.topology_split_quantile,
                ambiguity_quantile=args.topology_ambiguity_quantile,
                growth_cap_per_event=args.topology_growth_cap_per_event,
                total_point_budget_ratio=args.topology_total_point_budget_ratio,
                cooldown_iterations=args.topology_cooldown_iterations,
                enable_split=not args.topology_disable_split,
                min_repeatability=args.topology_min_repeatability,
                min_radius=args.topology_min_radius,
                enable_loc_clone=False,
                enable_soft_prune=args.topology_enable_soft_prune,
                enable_physical_prune=args.topology_enable_physical_prune,
                soft_prune_threshold=args.topology_soft_prune_threshold,
                soft_prune_step=args.topology_soft_prune_step,
                physical_rgb_threshold=args.topology_physical_rgb_threshold,
                physical_loc_threshold=args.topology_physical_loc_threshold,
                physical_utility_threshold=args.topology_physical_utility_threshold,
                require_loc_opacity_trained_for_physical_prune=not args.topology_allow_untrained_loc_opacity_prune,
            ),
            initial_points=gaussians.get_xyz.shape[0],
            protected_source_indices=protected_source_indices,
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_cameras = scene.getTrainCameras().copy()
    if args.support_query_split:
        support_cameras, query_cameras = split_support_query_cameras(
            train_cameras,
            query_ratio=args.query_holdout_ratio,
            seed=args.query_split_seed,
            mode=args.query_split_mode,
        )
        print(
            "Support/query split enabled: "
            f"support={len(support_cameras)} query={len(query_cameras)} "
            f"query_ratio={args.query_holdout_ratio} "
            f"query_split_seed={args.query_split_seed} "
            f"query_split_mode={args.query_split_mode}"
        )
    else:
        support_cameras = train_cameras
        query_cameras = train_cameras
    viewpoint_stack = None
    query_viewpoint_stack = None
    ema_loss_for_log = 0.0
    loc_opacity_grad_seen = False
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="LA Feature Gaussian")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        phase = "feature" if args.feature_only else args.train_phase
        _set_phase_lrs(gaussians, phase, args)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = support_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        render_pkg = render_gsplat(
            viewpoint_cam,
            gaussians,
            background,
            rgb_only=False,
            norm_feat_bf_render=dataset.norm_before_render,
            longest_edge=dataset.longest_edge,
            rasterize_mode="antialiased",
        )
        losses = _base_losses(viewpoint_cam, render_pkg, feature_extractor, dataset, masks=masks)
        image = losses["image"]
        base_loss = (
            (1.0 - opt.lambda_dssim) * losses["Ll1"]
            + opt.lambda_dssim * (1.0 - ssim(image, losses["gt_image"]))
            + args.base_feature_weight * losses["Ll1_feature"]
        )

        loc_loss = image.new_tensor(0.0)
        loc_desc_loss = image.new_tensor(0.0)
        loc_multiview_loss = image.new_tensor(0.0)
        loc_full_bank_loss = image.new_tensor(0.0)
        loc_anchor_loss = image.new_tensor(0.0)
        loc_reproj_loss = image.new_tensor(0.0)
        loc_dense_kl_loss = image.new_tensor(0.0)
        loc_proto_loss = image.new_tensor(0.0)
        loc_rank_loss = image.new_tensor(0.0)
        loc_opacity_loss = image.new_tensor(0.0)
        geom_anchor_loss = image.new_tensor(0.0)
        loc_grad = None
        teacher_out = None
        run_loc_episode = (
            args.localization_enabled
            and losses["gt_feature_map"] is not None
            and iteration >= args.loc_start_iter
            and iteration % args.loc_interval == 0
        )

        if run_loc_episode:
            query_cam = viewpoint_cam
            query_feature_map = losses["gt_feature_map"]
            if args.support_query_split:
                if not query_viewpoint_stack:
                    query_viewpoint_stack = query_cameras.copy()
                query_cam = query_viewpoint_stack.pop(randint(0, len(query_viewpoint_stack) - 1))
                query_feature_map = _query_feature_map(
                    query_cam,
                    feature_extractor,
                    target_hw=losses["gt_feature_map"].shape[-2:],
                    masks=masks,
                )
            episode = episode_sampler.sample(query_cam)
            pose_gt = episode.pose_gt_w2c.cuda()
            pose_init = episode.pose_init_w2c.cuda()
            if args.loc_teacher == "direct":
                current_direct_landmark_indices = _current_landmark_indices_from_source_index(
                    direct_landmark_indices,
                    gaussians,
                )
                target_depth = None
                target_alpha = None
                if args.direct_depth_check:
                    with torch.no_grad():
                        gt_render = render_from_pose_gsplat(
                            gaussians,
                            pose_gt,
                            query_cam.FoVx,
                            query_cam.FoVy,
                            query_feature_map.shape[2],
                            query_feature_map.shape[1],
                            bg_color=background,
                            render_mode="RGB+ED",
                            rgb_only=True,
                            norm_feat_bf_render=dataset.norm_before_render,
                            rasterize_mode="antialiased",
                        )
                    target_depth = _flatten_render_map(gt_render.get("depth"))
                    target_alpha = _flatten_render_alpha(gt_render.get("alphas"))
                teacher_out = direct_landmark_teacher(
                    gaussians,
                    query_feature_map,
                    pose_gt,
                    query_cam.FoVx,
                    query_cam.FoVy,
                    current_direct_landmark_indices,
                    target_depth=target_depth,
                    target_alpha=target_alpha,
                    alpha_threshold=args.loc_alpha_threshold,
                    depth_abs_tolerance=args.direct_depth_abs_tolerance,
                    depth_rel_tolerance=args.direct_depth_rel_tolerance,
                    max_landmarks=args.loc_anchors,
                    multiview_memory=direct_observation_memory,
                    multiview_temperature=args.loc_multiview_temperature,
                    multiview_ignore_radius=args.loc_multiview_ignore_radius,
                    full_bank_indices=current_direct_landmark_indices if args.loc_full_bank_weight > 0 else None,
                    full_bank_temperature=args.loc_full_bank_temperature,
                    full_bank_hard_negative_topk=args.loc_full_bank_hard_negatives,
                    full_bank_hard_negative_margin=args.loc_full_bank_margin,
                    sampling_grid_size=args.loc_anchor_grid_size,
                    anchor_features=_feature_anchor_tensor(loc_feature_anchor) if args.loc_anchor_weight > 0 else None,
                )
                loc_desc_loss = teacher_out.desc_loss
                loc_multiview_loss = teacher_out.multiview_loss
                loc_full_bank_loss = teacher_out.full_bank_loss
                loc_anchor_loss = teacher_out.anchor_loss
                loc_reproj_loss = teacher_out.reproj_loss
                loc_loss = (
                    args.loc_direct_weight * loc_desc_loss
                    + args.loc_multiview_weight * loc_multiview_loss
                    + args.loc_full_bank_weight * loc_full_bank_loss
                    + args.loc_anchor_weight * loc_anchor_loss
                )
            else:
                teacher_out = dense_localization_teacher(
                    gaussians,
                    query_feature_map,
                    pose_init,
                    pose_gt,
                    query_cam.FoVx,
                    query_cam.FoVy,
                    query_feature_map.shape[2],
                    query_feature_map.shape[1],
                    background,
                    anchor_count=args.loc_anchors,
                    alpha_threshold=args.loc_alpha_threshold,
                    desc_temperature=args.loc_desc_temperature,
                    fine_temperature=args.loc_fine_temperature,
                    fine_window_radius=args.loc_fine_window_radius,
                    dense_kl_weight=args.loc_dense_kl_weight,
                    dense_kl_temperature=args.loc_dense_kl_temperature,
                    responsibility_topk=args.loc_responsibility_topk,
                    responsibility_opacity_weight=args.loc_responsibility_opacity_weight,
                    responsibility_depth_weight=args.loc_responsibility_depth_weight,
                    norm_feat_bf_render=dataset.norm_before_render,
                    use_loc_opacity=args.use_loc_opacity,
                    rasterize_args={"rasterize_mode": "antialiased"},
                )
                loc_desc_loss = teacher_out.desc_loss
                loc_reproj_loss = teacher_out.reproj_loss
                loc_dense_kl_loss = teacher_out.kl_loss
                loc_loss = (
                    args.loc_desc_weight * loc_desc_loss
                    + args.loc_reproj_weight * loc_reproj_loss
                    + args.loc_dense_kl_weight * loc_dense_kl_loss
                )

            visible_idx = teacher_out.loc_visible_idx
            if visible_idx is not None and visible_idx.numel() > 0:
                seen = gaussians.loc_prototype_count[visible_idx] > 0
                if seen.any():
                    loc_features = gaussians.get_loc_feature[visible_idx][seen].reshape(seen.sum(), -1)
                    prototypes = gaussians.loc_prototype[visible_idx][seen]
                    loc_proto_loss = prototype_loss(loc_features, prototypes)
                    loc_loss = loc_loss + args.loc_proto_weight * loc_proto_loss
                    if args.loc_rank_weight > 0 and loc_features.shape[0] > 1:
                        loc_rank_loss = hard_negative_ranking_loss(loc_features, prototypes, margin=args.loc_rank_margin)
                        loc_loss = loc_loss + args.loc_rank_weight * loc_rank_loss

            if args.use_loc_opacity and args.loc_opacity_weight > 0:
                loc_opacity_loss = localization_opacity_regularizer(
                    gaussians.get_loc_opacity,
                    target_density=args.loc_opacity_target,
                    sparsity_weight=1.0,
                    density_weight=1.0,
                )
                loc_loss = loc_loss + args.loc_opacity_weight * loc_opacity_loss

            if teacher_out.loc_viewspace_points is not None and loc_loss.requires_grad:
                loc_grad = torch.autograd.grad(
                    loc_loss,
                    teacher_out.loc_viewspace_points,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

        if phase in {"geometry", "topology", "closed_loop"} and args.geometry_anchor_weight > 0:
            geometry_anchor = _refresh_geometry_anchor_if_point_count_changed(gaussians, geometry_anchor)
            geom_anchor_loss = geometry_anchor_loss(
                _current_geometry_state(gaussians),
                geometry_anchor,
                xyz_weight=1.0,
                scale_weight=args.geometry_anchor_scale_weight,
                rotation_weight=args.geometry_anchor_rotation_weight,
            )

        total_loss = (
            args.base_loss_weight * base_loss
            + args.loc_loss_weight * loc_loss
            + args.geometry_anchor_weight * geom_anchor_loss
        )
        total_loss.backward()
        loc_opacity_grad = getattr(getattr(gaussians, "_loc_opacity", None), "grad", None)
        if loc_opacity_grad is not None:
            loc_opacity_grad_seen = loc_opacity_grad_seen or bool(
                torch.isfinite(loc_opacity_grad).any().item()
                and (loc_opacity_grad.detach().abs().max() > 0).item()
            )
        gaussians.loc_opacity_grad_seen = loc_opacity_grad_seen

        with torch.no_grad():
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            gaussians.update_screen_radii(render_pkg.get("visibility_filter"), render_pkg.get("radii"))
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.6f}",
                    "Loc": f"{loc_loss.item():.6f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if tb_writer:
                tb_writer.add_scalar("train_loss/base", base_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc", loc_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_desc", loc_desc_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_multiview", loc_multiview_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_full_bank", loc_full_bank_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_anchor", loc_anchor_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_reproj", loc_reproj_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_dense_kl", loc_dense_kl_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_proto", loc_proto_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_rank", loc_rank_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_opacity", loc_opacity_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/geometry_anchor", geom_anchor_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/total", total_loss.item(), iteration)
                tb_writer.add_scalar("train/points", gaussians.get_xyz.shape[0], iteration)
                if teacher_out is not None:
                    for name, value in getattr(teacher_out, "diagnostics", {}).items():
                        if isinstance(value, (int, float)):
                            tb_writer.add_scalar(f"train_diagnostics/{name}", float(value), iteration)

            if teacher_out is not None and teacher_out.loc_visible_idx is not None:
                gaussians.add_localization_stats(
                    full_idx=teacher_out.loc_visible_idx,
                    means2d_grad=loc_grad,
                    radii=teacher_out.loc_radii,
                    episode_stats=teacher_out.stats,
                    ema_decay=args.loc_ema_decay,
                )

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

            if topology_controller is not None and topology_controller.should_update(iteration):
                topology_controller.update(gaussians, scene.cameras_extent, iteration)

            loc_feature_anchor = _refresh_feature_anchor_if_point_count_changed(gaussians, loc_feature_anchor)

            if iteration in args.save_iterations:
                print(f"\n[ITER {iteration}] Saving LA Gaussians")
                scene.save(iteration)
                torch.save(
                    {
                        "version": 2,
                        "iteration": iteration,
                        "model_params": gaussians.capture(),
                        "localization_state": gaussians.capture_localization_state(),
                        "config": vars(args),
                    },
                    os.path.join(dataset.model_path, f"chkpnt_locaware_{iteration}.pth"),
                )

            if iteration in args.test_iterations:
                psnr_val = psnr(image, losses["gt_image"]).mean().item()
                print(
                    f"\n[ITER {iteration}] base {base_loss.item():.6f} "
                    f"loc {loc_loss.item():.6f} psnr {psnr_val:.3f}"
                )

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="LA-STDLoc training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    add_locaware_training_args(parser)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    seed_everything(args.train_seed)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), args)
    print("\nLA-STDLoc training complete.")
