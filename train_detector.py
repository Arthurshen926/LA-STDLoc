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
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
import math

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, get_combined_args
from gaussian_renderer import get_render_visible_mask, render_gsplat
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state, seed_everything
from utils.graphics_utils import focal2fov, fov2focal
from utils.image_utils import get_resolution_from_longest_edge
from utils.loss_utils import *

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import pickle

import torch.nn.functional as F

from encoders.feature_extractor import FeatureExtractor
from localization_training.landmark_distill import localization_aware_sample, save_landmark_meta
from scene.kpdetector import KpDetector


def extract_normalized_feature_map(feature_extractor, image, size):
    """Run the fixed image encoder without keeping gradients for detector targets."""
    with torch.no_grad():
        gt_feature_map = feature_extractor(image[None])["feature_map"]
        gt_feature_map = F.interpolate(
            gt_feature_map,
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=0)
    return gt_feature_map.detach()


def store_render_visible_mask(render_visible_masks, image_name, visible_mask):
    render_visible_masks[image_name] = visible_mask.detach().to(device="cpu", dtype=torch.bool)


def render_visible_mask_from_cache(render_visible_masks, image_name, device):
    visible_mask = render_visible_masks.get(image_name, None)
    if visible_mask is None:
        return None
    return visible_mask.to(device=device, dtype=torch.bool, non_blocking=True)


def get_sampled_gaussian(gaussians: GaussianModel, idx_sampled):
    sampled_gaussians = GaussianModel(gaussians.max_sh_degree)
    sampled_gaussians._xyz = gaussians._xyz[idx_sampled]
    sampled_gaussians._loc_feature = gaussians._loc_feature[idx_sampled]
    sampled_gaussians._scaling = gaussians._scaling[idx_sampled]
    sampled_gaussians._opacity = gaussians._opacity[idx_sampled]
    sampled_gaussians._rotation = gaussians._rotation[idx_sampled]
    sampled_gaussians._features_dc = gaussians._features_dc[idx_sampled]
    sampled_gaussians._features_rest = gaussians._features_rest[idx_sampled]
    return sampled_gaussians


@torch.no_grad()
def calculate_match_score(
    gaussians: GaussianModel,
    gt_feature_map,
    pose,
    K,
    render_visible_mask=None,
    img_mask=None,
):
    xyz = gaussians.get_xyz
    feat = gaussians.get_loc_feature.squeeze()

    # project gaussians to image space
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device)], dim=-1)
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_homo = xyz_cam / depths

    xy = (K @ xyz_cam_homo)[:2].long()

    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_feature_map.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_feature_map.shape[1])
    )

    if render_visible_mask is not None:
        visible_mask = in_mask & render_visible_mask
    else:
        visible_mask = in_mask

    if img_mask is not None:
        visible_xy = xy[:, in_mask]
        img_mask_expand = torch.zeros_like(visible_mask, dtype=torch.bool)
        img_mask_expand[in_mask] = img_mask[0, visible_xy[1], visible_xy[0]]
        visible_mask = visible_mask & img_mask_expand

    xy = xy[:, visible_mask]
    depths = depths[visible_mask]
    feat = feat[visible_mask]

    gs_feats = F.normalize(feat, p=2, dim=1)
    im_feats = gt_feature_map[:, xy[1], xy[0]].T
    score = (gs_feats * im_feats).sum(-1)
    return score, visible_mask


def generate_gt_map(
    gaussians: GaussianModel,
    gt_feature_map,
    idx_sampled,
    pose,
    K,
    render_visible_mask=None,
):
    if render_visible_mask is not None:
        render_visible_mask = render_visible_mask[idx_sampled]
        idx_sampled = idx_sampled[render_visible_mask]
    sampled_xyz = gaussians.get_xyz[idx_sampled]

    gt_map = torch.zeros(
        (1, gt_feature_map.shape[1], gt_feature_map.shape[2]),
        device=gt_feature_map.device,
    )
    
    xyz_homo = torch.cat(
        [sampled_xyz, torch.ones(sampled_xyz.shape[0], 1, device=sampled_xyz.device)],
        dim=-1,
    )
    xyz_cam = (pose @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths

    xy = (K @ xyz_cam_norm)[:2].long()

    in_mask = (
        (xy[0] >= 0)
        & (xy[0] < gt_feature_map.shape[2])
        & (xy[1] >= 0)
        & (xy[1] < gt_feature_map.shape[1])
    )

    xy_pos = xy[:, in_mask]

    gt_map[:, xy_pos[1], xy_pos[0]] = 1

    return gt_map


def _project_xyz_to_feature(xyz, pose, K, height, width):
    xyz_homo = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)], dim=-1)
    xyz_cam = (pose.to(device=xyz.device, dtype=xyz.dtype) @ xyz_homo.T)[:3]
    depths = xyz_cam[2]
    xyz_cam_norm = xyz_cam / depths.clamp_min(1e-8)
    xy = (K.to(device=xyz.device, dtype=xyz.dtype) @ xyz_cam_norm)[:2]
    valid = (
        (depths > 1e-8)
        & (xy[0] >= 0)
        & (xy[0] <= width - 1)
        & (xy[1] >= 0)
        & (xy[1] <= height - 1)
    )
    return xy, valid


def generate_soft_gt_map(
    xyz,
    gt_feature_map,
    pose,
    K,
    utility=None,
    sigma=1.5,
    render_visible_mask=None,
    max_landmarks=4096,
):
    """Generate a utility-weighted Gaussian detector target map."""
    height, width = gt_feature_map.shape[1], gt_feature_map.shape[2]
    device = gt_feature_map.device
    dtype = gt_feature_map.dtype
    xyz = xyz.to(device=device, dtype=dtype)
    xy, valid = _project_xyz_to_feature(xyz, pose, K, height, width)
    if render_visible_mask is not None:
        valid = valid & render_visible_mask.to(device=device, dtype=torch.bool)
    if utility is None:
        utility = torch.ones(xyz.shape[0], device=device, dtype=dtype)
    else:
        utility = utility.to(device=device, dtype=dtype).reshape(-1)
    if valid.sum() == 0:
        return torch.zeros((1, height, width), device=device, dtype=dtype)

    xy = xy[:, valid].T
    utility = utility[valid]
    utility = (utility - utility.min()) / (utility.max() - utility.min()).clamp_min(1e-6)
    utility = utility.clamp_min(0.05)
    if max_landmarks is not None and xy.shape[0] > max_landmarks:
        keep = torch.topk(utility, k=int(max_landmarks), largest=True).indices
        xy = xy[keep]
        utility = utility[keep]

    sigma2 = max(float(sigma), 1e-6) ** 2
    radius = max(1, int(math.ceil(3.0 * max(float(sigma), 1e-6))))
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.long)
    window = offsets.numel()
    offset_x = offsets.repeat(window)
    offset_y = offsets.repeat_interleave(window)

    base_x = torch.floor(xy[:, 0]).to(dtype=torch.long)[:, None]
    base_y = torch.floor(xy[:, 1]).to(dtype=torch.long)[:, None]
    px = base_x + offset_x[None]
    py = base_y + offset_y[None]
    pixel_valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)

    dist2 = (px.to(dtype=dtype) - xy[:, 0:1]).square() + (
        py.to(dtype=dtype) - xy[:, 1:2]
    ).square()
    values = utility[:, None] * torch.exp(-0.5 * dist2 / sigma2)
    flat_idx = py * width + px

    gt_flat = torch.zeros(height * width, device=device, dtype=dtype)
    gt_flat.scatter_reduce_(
        0,
        flat_idx[pixel_valid],
        values[pixel_valid],
        reduce="amax",
        include_self=True,
    )
    return gt_flat.view(height, width).clamp(0.0, 1.0)[None]


def utility_weighted_detector_loss(pred, target, gamma=2.0, alpha=0.25):
    """Focal BCE for soft utility targets; accepts probabilities or logits."""
    target = target.float()
    pred = pred.float()
    if pred.min() < 0 or pred.max() > 1:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
    else:
        prob = pred.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(prob, target, reduction="none")
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = alpha_t * (1.0 - pt).pow(gamma) * bce
    weights = 1.0 + target
    return (focal * weights).sum() / weights.sum().clamp_min(1e-6)


def random_knn_score(points, npoints, score, k=32):
    sampled_idx = torch.randperm(points.shape[0])[:npoints]
    sampled_points = points[sampled_idx]
    points = points.cpu()
    sampled_points = sampled_points.cpu()
    dist = torch.cdist(sampled_points, points)
    knn_idx = torch.topk(dist, k, largest=False, dim=-1)[1]
    knn_idx = knn_idx.cuda()

    # knn select
    knn_score = score[knn_idx]  # (npoints, k)
    score_knn_sort_idx = torch.argsort(
        knn_score, descending=True, dim=-1
    )  # (npoints, k)

    final_sampled_idx = set()

    for i in range(npoints):
        for j in score_knn_sort_idx[i]:
            idx = knn_idx[i, j].item()  
            if idx not in final_sampled_idx: 
                final_sampled_idx.add(idx)  
                break  

    return torch.tensor(list(final_sampled_idx)).cuda()


def matching_oriented_sample(
    scene,
    gaussians,
    feature_extractor,
    render_visible_masks,
    masks=None,
    num=16384,
    k=32,
):
    viewpoint_stack = scene.getTrainCameras().copy()
    score_sum = torch.zeros(
        gaussians.get_xyz.shape[0], dtype=torch.float32, device="cuda"
    )
    score_num = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.int, device="cuda")
    fine_resolution = (
        viewpoint_stack[0].original_image.shape[1],
        viewpoint_stack[0].original_image.shape[2],
    )

    for viewpoint_cam in tqdm(viewpoint_stack, desc="Match Score"):
        gt_image = viewpoint_cam.original_image.cuda()
        gt_feature_map = extract_normalized_feature_map(
            feature_extractor,
            gt_image,
            size=(fine_resolution[0], fine_resolution[1]),
        )

        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
        focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
        # print("focal:", focalX, focalY)
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )
        render_visible_mask = render_visible_mask_from_cache(
            render_visible_masks,
            viewpoint_cam.image_name,
            gt_feature_map.device,
        )
        if render_visible_mask is None:
            render_visible_mask = get_render_visible_mask(
                gaussians,
                viewpoint_cam,
                gt_feature_map.shape[2],
                gt_feature_map.shape[1],
            )
            store_render_visible_mask(
                render_visible_masks,
                viewpoint_cam.image_name,
                render_visible_mask,
            )
        if masks is not None:
            object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]
            mask = object_mask & distort_mask
            img_mask = (
                F.interpolate(
                    mask[None].float(),
                    size=(gt_feature_map.shape[1], gt_feature_map.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                > 0.5
            )
        else:
            img_mask = None

        score, mask = calculate_match_score(
            gaussians,
            gt_feature_map,
            viewmat,
            K,
            render_visible_mask=render_visible_mask,
            img_mask=img_mask,
        )
        score_num[mask] += 1
        score_sum[mask] += score

    score_num[score_num == 0] = 1  # avoid divide by zero
    score_avg = score_sum / score_num

    sampled_idx = random_knn_score(gaussians.get_xyz, num, score_avg, k=k)
    sampled_idx = torch.unique(sampled_idx)
    return sampled_idx, score_avg, score_num


def evaluate_detector(
    detector,
    feature_extractor,
    gaussians,
    sampled_idx,
    scene,
    masks=None,
    render_visible_masks=None,
    tb_writer=None,
    iteration=0,
):
    torch.cuda.empty_cache()

    landmarks = get_sampled_gaussian(gaussians, sampled_idx)

    bg_color = [1, 1, 1] if scene.args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    validation_configs = (
        {"name": "test", "cameras": scene.getTestCameras()},
        {
            "name": "train",
            "cameras": [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)
            ],
        },
    )

    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            fine_resolution = get_resolution_from_longest_edge(
                config["cameras"][0].original_image.shape[1],
                config["cameras"][0].original_image.shape[2],
                scene.longest_edge,
            )
            loss_sum = 0.0
            for idx, viewpoint_cam in enumerate(config["cameras"]):
                gt_image = viewpoint_cam.original_image.cuda()
                gt_feature_map = extract_normalized_feature_map(
                    feature_extractor,
                    gt_image,
                    size=(fine_resolution[0], fine_resolution[1]),
                )

                viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
                focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
                focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
                # print("focal:", focalX, focalY)
                K = torch.tensor(
                    [
                        [focalX, 0.0, gt_feature_map.shape[2] / 2],
                        [0.0, focalY, gt_feature_map.shape[1] / 2],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                    device="cuda",
                )
                visible_mask = render_visible_mask_from_cache(
                    render_visible_masks,
                    viewpoint_cam.image_name,
                    gt_feature_map.device,
                )
                if visible_mask is None:
                    visible_mask = get_render_visible_mask(
                        gaussians,
                        viewpoint_cam,
                        gt_feature_map.shape[2],
                        gt_feature_map.shape[1],
                    )
                    store_render_visible_mask(
                        render_visible_masks,
                        viewpoint_cam.image_name,
                        visible_mask,
                    )

                gt_map = generate_gt_map(
                    gaussians,
                    gt_feature_map,
                    sampled_idx,
                    viewmat,
                    K,
                    visible_mask,
                )

                if masks is not None:
                    object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
                    # sky_mask = masks[viewpoint_cam.image_name][1].cuda()[None]
                    distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]

                    mask = object_mask & distort_mask
                    gt_map_mask = (
                        F.interpolate(
                            mask[None].float(),
                            size=(gt_map.shape[1], gt_map.shape[2]),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                        > 0.5
                    )
                    gt_map *= gt_map_mask

                # Loss
                heat_map = detector(gt_feature_map)
                loss = score_map_bce_loss(heat_map, gt_map)

                loss_sum += loss.item()
                if tb_writer and idx < 5:
                    render = render_gsplat(
                        viewpoint_cam, gaussians, background, rgb_only=True
                    )["render"]
                    sampled_render = render_gsplat(
                        viewpoint_cam, landmarks, background, rgb_only=True
                    )["render"]
                    heat_map = (heat_map - heat_map.min()) / (
                        heat_map.max() - heat_map.min()
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/gt_map_{idx}",
                        gt_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/heat_map{idx}",
                        heat_map[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/render_{idx}",
                        render[None],
                        iteration,
                    )
                    tb_writer.add_images(
                        f"detector_vis_{config['name']}/sampled_render_{idx}",
                        sampled_render[None],
                        iteration,
                    )

            loss_sum /= len(config["cameras"])
            print(
                f"\n[ITER {iteration}] Evaluating detector: {config['name']} loss {loss_sum}"
            )
            if tb_writer:
                tb_writer.add_scalar(
                    f"detector_loss_patches/{config['name']}_loss",
                    loss_sum,
                    iteration,
                )


def training_detector(
    gaussians,
    scene: Scene,
    masks,
    testing_iterations,
    saving_iterations,
    tb_writer,
    train_iteration=30000,
    detector_folder="",
    landmark_num=16384,
    landmark_k=32,
    sampling_mode="baseline",
    utility_weight=1.0,
    min_loc_observations=1,
    detector_target_mode="hard",
    soft_sigma=1.5,
):
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
    feature_extractor = FeatureExtractor(scene.feature_type).cuda().eval()

    render_visible_masks = {}

    # M.O. sampling
    print("Matching oriented sampling...")
    sampled_idx, score_avg, score_num = matching_oriented_sample(
        scene,
        gaussians,
        feature_extractor,
        render_visible_masks,
        masks=masks,
        num=landmark_num,
        k=landmark_k,
    )
    save_path = os.path.join(scene.model_path, detector_folder)
    os.makedirs(save_path, exist_ok=True)
    landmark_meta = None
    if sampling_mode == "localization_aware":
        if not hasattr(gaussians, "compute_localization_utility"):
            raise ValueError("localization_aware sampling requires Gaussian localization state")
        utility = gaussians.compute_localization_utility(min_observations=min_loc_observations)
        observed = gaussians.loc_observation_count >= min_loc_observations
        sampled_idx, landmark_meta = localization_aware_sample(
            gaussians.get_xyz,
            score_avg,
            utility,
            num=landmark_num,
            k=landmark_k,
            min_observations=observed,
            utility_weight=utility_weight,
        )
        landmark_meta["repeatability"] = gaussians.loc_repeatability_ema[sampled_idx]
        landmark_meta["margin"] = gaussians.loc_margin_ema[sampled_idx]
        landmark_meta["information"] = gaussians.loc_information_ema[sampled_idx]
        landmark_meta["prototype"] = gaussians.loc_prototype[sampled_idx]
        save_landmark_meta(os.path.join(save_path, "landmark_meta.pt"), landmark_meta)
    elif sampling_mode != "baseline":
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
    pickle.dump(sampled_idx, open(os.path.join(save_path, "sampled_idx.pkl"), "wb"))
    del score_avg, score_num
    if "utility" in locals():
        del utility
    if "observed" in locals():
        del observed
    torch.cuda.empty_cache()

    # training scene-specific detector
    print("Training detector...")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    progress_bar = tqdm(range(0, train_iteration), desc="Scene-Specific Detector")
    first_iter = 1

    detector = KpDetector(feature_extractor.feature_dim).cuda().train()
    optimizer = torch.optim.AdamW(detector.parameters(), lr=0.001)
    grad_accum = 8
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_iteration // grad_accum, eta_min=0.0005
    )

    for iteration in range(first_iter, train_iteration + 1):
        iter_start.record()
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        fine_resolution = get_resolution_from_longest_edge(
            viewpoint_cam.original_image.shape[1],
            viewpoint_cam.original_image.shape[2],
            scene.longest_edge,
        )

        # generate gt_feature_map
        gt_image = viewpoint_cam.original_image.cuda()
        gt_feature_map = extract_normalized_feature_map(
            feature_extractor,
            gt_image,
            size=(fine_resolution[0], fine_resolution[1]),
        )

        # get viewmat and K
        viewmat = viewpoint_cam.world_view_transform.transpose(0, 1).cuda()  # [4, 4]
        focalX = fov2focal(viewpoint_cam.FoVx, gt_feature_map.shape[2])
        focalY = fov2focal(viewpoint_cam.FoVy, gt_feature_map.shape[1])
        K = torch.tensor(
            [
                [focalX, 0.0, gt_feature_map.shape[2] / 2],
                [0.0, focalY, gt_feature_map.shape[1] / 2],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device="cuda",
        )

        # get render visible mask
        render_visible_mask = render_visible_mask_from_cache(
            render_visible_masks,
            viewpoint_cam.image_name,
            gt_feature_map.device,
        )
        if render_visible_mask is None:
            render_visible_mask = get_render_visible_mask(
                gaussians,
                viewpoint_cam,
                gt_feature_map.shape[2],
                gt_feature_map.shape[1],
            )
            store_render_visible_mask(
                render_visible_masks,
                viewpoint_cam.image_name,
                render_visible_mask,
            )

        # generate gt_map
        if detector_target_mode == "soft" and landmark_meta is not None:
            gt_map = generate_soft_gt_map(
                gaussians.get_xyz[sampled_idx],
                gt_feature_map,
                viewmat,
                K,
                utility=landmark_meta.get("utility"),
                sigma=soft_sigma,
                render_visible_mask=render_visible_mask[sampled_idx],
            )
        else:
            gt_map = generate_gt_map(
                gaussians, gt_feature_map, sampled_idx, viewmat, K, render_visible_mask
            )

        # use mask to filter out object
        if masks is not None:
            object_mask = masks[viewpoint_cam.image_name][0].cuda()[None]
            distort_mask = masks[viewpoint_cam.image_name][2].cuda()[None]
            mask = object_mask & distort_mask
            gt_map_mask = (
                F.interpolate(
                    mask[None].float(),
                    size=(gt_map.shape[1], gt_map.shape[2]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                > 0.5
            )
            gt_map *= gt_map_mask

        # Loss
        heat_map = detector(gt_feature_map)
        if detector_target_mode == "soft" and landmark_meta is not None:
            loss = utility_weighted_detector_loss(heat_map, gt_map)
        else:
            loss = score_map_bce_loss(heat_map, gt_map)

        loss.backward()
        if iteration % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            loss_val = loss.item()
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss_val:.{7}f}",
                    }
                )
                progress_bar.update(10)
            if iteration == train_iteration:
                progress_bar.close()
            if tb_writer:
                tb_writer.add_scalar(
                    "detector_loss_patches/training_loss", loss_val, iteration
                )
                tb_writer.add_scalar(
                    "detector_loss_patches/lr",
                    optimizer.param_groups[0]["lr"],
                    iteration,
                )

        if iteration in testing_iterations:
            print("\n[ITER {}] Evaluating detector".format(iteration))
            detector.eval()
            evaluate_detector(
                detector,
                feature_extractor,
                gaussians,
                sampled_idx,
                scene,
                masks,
                render_visible_masks,
                tb_writer,
                iteration,
            )
            detector.train()

        if iteration in saving_iterations:
            print("\n[ITER {}] Saving detector".format(iteration))
            torch.save(detector.state_dict(), save_path + f"/{iteration}_detector.pth")


def prepare_output_and_logger(args, folder=None):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    if folder:
        output_path = os.path.join(args.model_path, folder)
    else:
        output_path = args.model_path
    print("Output folder: {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(output_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    seed_everything(2025)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[10000, 20000, 30000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--detector_folder", type=str, default="detector")
    parser.add_argument("--landmark_num", type=int, default=16384)
    parser.add_argument("--landmark_k", type=int, default=32)
    parser.add_argument("--sampling_mode", type=str, default="baseline", choices=["baseline", "localization_aware"])
    parser.add_argument("--utility_weight", type=float, default=1.0)
    parser.add_argument("--min_loc_observations", type=int, default=1)
    parser.add_argument("--detector_target_mode", type=str, default="hard", choices=["hard", "soft"])
    parser.add_argument("--soft_sigma", type=float, default=1.5)

    args = get_combined_args(parser)
    args.save_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dataset = lp.extract(args)
    if dataset.gaussian_type == "3dgs":
        from scene.gaussian_model import GaussianModel
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        from scene.gaussian_model import GaussianModel_2dgs
        gaussians = GaussianModel_2dgs(dataset.sh_degree)

    masks = None
    for mask_path in (
        os.path.join(dataset.source_path, dataset.images, "masks.pkl"),
        os.path.join(dataset.source_path, "masks.pkl"),
    ):
        if os.path.exists(mask_path):
            import pickle
            masks = pickle.load(open(mask_path, "rb"))
            break

    scene = Scene(dataset, gaussians, load_iteration=args.iteration)

    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(
            os.path.join(dataset.model_path, args.detector_folder)
        )
    else:
        tb_writer = None

    training_detector(
        gaussians,
        scene,
        masks,
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        tb_writer=tb_writer,
        train_iteration=args.iterations,
        detector_folder=args.detector_folder,
        landmark_num=args.landmark_num,
        landmark_k=args.landmark_k,
        sampling_mode=args.sampling_mode,
        utility_weight=args.utility_weight,
        min_loc_observations=args.min_loc_observations,
        detector_target_mode=args.detector_target_mode,
        soft_sigma=args.soft_sigma,
    )

    # All done
    print("\n Scene-specific detector training complete.")
