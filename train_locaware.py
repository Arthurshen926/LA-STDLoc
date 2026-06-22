import os
import pickle
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_gsplat
from localization_training.dense_teacher import dense_localization_teacher
from localization_training.episode_sampler import EpisodeSampler, SparsePoseCache
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


def _set_phase_lrs(gaussians, phase, args):
    if phase == "feature":
        trainable = {"loc_feature", "loc_opacity"}
    elif phase in {"geometry", "topology", "closed_loop"}:
        trainable = {"xyz", "scaling", "rotation", "loc_feature", "loc_opacity"}
    else:
        trainable = {group["name"] for group in gaussians.optimizer.param_groups}

    for group in gaussians.optimizer.param_groups:
        group.setdefault("la_base_lr", group["lr"])
        if group["name"] not in trainable:
            group["lr"] = 0.0
        elif phase in {"geometry", "topology", "closed_loop"}:
            if group["name"] == "xyz":
                group["lr"] *= args.geometry_xyz_lr_mult
            elif group["name"] == "scaling":
                group["lr"] = group["la_base_lr"] * args.geometry_scale_lr_mult
            elif group["name"] == "rotation":
                group["lr"] = group["la_base_lr"] * args.geometry_rotation_lr_mult


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
    if first_iter == 0 and scene.loaded_iter:
        first_iter = scene.loaded_iter
    geometry_anchor = _capture_geometry_anchor(gaussians)
    sparse_pose_cache = None
    if args.sparse_pose_cache:
        sparse_pose_cache = SparsePoseCache(args.sparse_pose_cache).load()
    episode_sampler = EpisodeSampler(
        sparse_pose_cache=sparse_pose_cache,
        query_mode=args.query_mode,
        noise_quantile=args.pose_noise_quantile,
    )
    topology_controller = None
    if args.enable_topology or args.train_phase in {"topology", "closed_loop"}:
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
                enable_loc_clone=False,
                soft_prune_threshold=args.topology_soft_prune_threshold,
                soft_prune_step=args.topology_soft_prune_step,
            ),
            initial_points=gaussians.get_xyz.shape[0],
        )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="LA Feature Gaussian")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):
        gaussians.update_learning_rate(iteration)
        phase = "feature" if args.feature_only else args.train_phase
        _set_phase_lrs(gaussians, phase, args)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
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
        loc_reproj_loss = image.new_tensor(0.0)
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
            episode = episode_sampler.sample(viewpoint_cam)
            pose_gt = episode.pose_gt_w2c.cuda()
            pose_init = episode.pose_init_w2c.cuda()
            teacher_out = dense_localization_teacher(
                gaussians,
                losses["gt_feature_map"],
                pose_init,
                pose_gt,
                viewpoint_cam.FoVx,
                viewpoint_cam.FoVy,
                losses["gt_feature_map"].shape[2],
                losses["gt_feature_map"].shape[1],
                background,
                anchor_count=args.loc_anchors,
                alpha_threshold=args.loc_alpha_threshold,
                desc_temperature=args.loc_desc_temperature,
                fine_temperature=args.loc_fine_temperature,
                fine_window_radius=args.loc_fine_window_radius,
                norm_feat_bf_render=dataset.norm_before_render,
                use_loc_opacity=args.use_loc_opacity,
                rasterize_args={"rasterize_mode": "antialiased"},
            )
            loc_desc_loss = teacher_out.desc_loss
            loc_reproj_loss = teacher_out.reproj_loss
            loc_loss = args.loc_desc_weight * loc_desc_loss + args.loc_reproj_weight * loc_reproj_loss

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
                tb_writer.add_scalar("train_loss/loc_reproj", loc_reproj_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_proto", loc_proto_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_rank", loc_rank_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/loc_opacity", loc_opacity_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/geometry_anchor", geom_anchor_loss.item(), iteration)
                tb_writer.add_scalar("train_loss/total", total_loss.item(), iteration)
                tb_writer.add_scalar("train/points", gaussians.get_xyz.shape[0], iteration)

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

            if topology_controller is not None and topology_controller.should_update(iteration):
                topology_controller.update(gaussians, scene.cameras_extent, iteration)

    if tb_writer:
        tb_writer.close()


if __name__ == "__main__":
    seed_everything(2025)
    parser = ArgumentParser(description="LA-STDLoc training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--load_iteration", type=int, default=None)
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
    parser.add_argument("--loc_proto_weight", type=float, default=0.1)
    parser.add_argument("--loc_rank_weight", type=float, default=0.05)
    parser.add_argument("--loc_rank_margin", type=float, default=0.2)
    parser.add_argument("--loc_opacity_weight", type=float, default=0.001)
    parser.add_argument("--loc_opacity_target", type=float, default=0.5)
    parser.add_argument("--loc_ema_decay", type=float, default=0.95)
    parser.add_argument("--use_loc_opacity", action="store_true", default=True)
    parser.add_argument("--query_mode", type=str, default="noise", choices=["noise", "sparse", "mixed"])
    parser.add_argument("--pose_noise_quantile", type=float, default=0.5)
    parser.add_argument("--sparse_pose_cache", type=str, default=None)
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
    parser.add_argument("--topology_soft_prune_threshold", type=float, default=-1.0)
    parser.add_argument("--topology_soft_prune_step", type=float, default=1.0)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), args)
    print("\nLA-STDLoc training complete.")
