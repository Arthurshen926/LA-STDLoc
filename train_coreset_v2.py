import argparse
import json
import os
import pickle
import random
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_from_pose_gsplat
from localization_training.episode_sampler import sample_interpolated_novel_view
from localization_training.full_primitive_retrieval import chunked_exact_topk
from localization_training.lafgs_reconstruction import (
    MultiViewInitConfig,
    build_multiview_initialization,
)
from localization_training.pose_refiner import project_points
from localization_training.progressive_coreset import (
    active_group_representatives,
    active_set_diagnostics,
    build_surface_groups,
    coreset_soft_matching_loss,
    descriptor_trust_loss,
    group_representatives,
    make_progressive_budget_schedule,
    progressive_coreset_regularizers,
    project_active_set,
)
from localization_training.splat_provenance import bank_splat_provenance_2dgs
from scene import Scene
from scene.gaussian_model import GaussianModel_2dgs
from scene.kpdetector import KpDetector, simple_nms
from train_detector import extract_normalized_feature_map, fill_missing_model_defaults
from utils.general_utils import build_rotation, safe_state, seed_everything
from utils.graphics_utils import fov2focal
from utils.image_utils import get_resolution_from_longest_edge


def _camera_pose(camera, device="cuda"):
    return camera.world_view_transform.transpose(0, 1).to(device)


def _intrinsics(camera, width, height, device="cuda"):
    return torch.tensor(
        [
            [fov2focal(camera.FoVx, width), 0.0, width / 2.0],
            [0.0, fov2focal(camera.FoVy, height), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


@torch.no_grad()
def _render_teacher(gaussians, camera, width, height, background):
    return render_from_pose_gsplat(
        gaussians,
        _camera_pose(camera),
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


@torch.no_grad()
def _detect_query(feature_map, detector, count, nms_radius):
    heatmap = simple_nms(detector(feature_map), nms_radius).reshape(-1)
    count = min(int(count), heatmap.numel())
    scores, ids = torch.topk(heatmap, count)
    keep = scores > 0
    ids = ids[keep]
    height, width = feature_map.shape[-2:]
    xy = torch.stack([ids % width, ids // width], dim=1).float()
    descriptors = feature_map.reshape(feature_map.shape[0], -1)[:, ids].T
    return xy, F.normalize(descriptors, dim=1), scores[keep]


@torch.no_grad()
def _dominant_surface_labels(
    xyz,
    group_ids,
    keypoint_xy,
    K,
    pose_w2c,
    rendered_depth,
    visibility,
    height,
    width,
    *,
    depth_abs=0.05,
    depth_rel=0.02,
    neighbor_radius=1,
):
    uv, in_front = project_points(xyz, K, pose_w2c)
    rounded = uv.round().long()
    in_image = (
        in_front
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    if visibility is not None:
        in_image &= visibility.to(device=xyz.device, dtype=torch.bool)
    flat = rounded[:, 1].clamp(0, height - 1) * width + rounded[:, 0].clamp(0, width - 1)
    depth = rendered_depth.squeeze()
    sampled_depth = depth[
        rounded[:, 1].clamp(0, height - 1), rounded[:, 0].clamp(0, width - 1)
    ]
    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    camera_xyz = (pose_w2c @ torch.cat([xyz, ones], dim=1).T)[:3].T
    primitive_depth = camera_xyz[:, 2]
    tolerance = float(depth_abs) + float(depth_rel) * sampled_depth.abs()
    valid = in_image & (sampled_depth > 0) & ((primitive_depth - sampled_depth).abs() <= tolerance)
    subpixel = (uv - rounded.float()).square().sum(dim=1)
    score = subpixel + (primitive_depth - sampled_depth).abs() / tolerance.clamp_min(1e-6)
    score = score.masked_fill(~valid, torch.inf)
    pixel_count = height * width
    best_score = score.new_full((pixel_count,), torch.inf)
    best_score.scatter_reduce_(0, flat, score, reduce="amin", include_self=True)
    primitive_idx = torch.arange(xyz.shape[0], device=xyz.device, dtype=torch.long)
    sentinel = xyz.shape[0]
    candidate = torch.where(valid & (score == best_score[flat]), primitive_idx, sentinel)
    best_idx = torch.full((pixel_count,), sentinel, device=xyz.device, dtype=torch.long)
    best_idx.scatter_reduce_(0, flat, candidate, reduce="amin", include_self=True)

    keypoint = keypoint_xy.round().long()
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(-neighbor_radius, neighbor_radius + 1, device=xyz.device),
            torch.arange(-neighbor_radius, neighbor_radius + 1, device=xyz.device),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    neighbor_y = (keypoint[:, None, 1] + offsets[None, :, 0]).clamp(0, height - 1)
    neighbor_x = (keypoint[:, None, 0] + offsets[None, :, 1]).clamp(0, width - 1)
    neighbor_flat = neighbor_y * width + neighbor_x
    neighbor_idx = best_idx[neighbor_flat]
    neighbor_score = best_score[neighbor_flat] + offsets.float().square().sum(dim=1)[None]
    neighbor_score = neighbor_score.masked_fill(neighbor_idx == sentinel, torch.inf)
    choice = neighbor_score.argmin(dim=1)
    selected = neighbor_idx.gather(1, choice[:, None]).squeeze(1)
    reliable = selected != sentinel
    labels = torch.full(
        (keypoint.shape[0],), -1, device=xyz.device, dtype=torch.long
    )
    labels[reliable] = group_ids[selected[reliable]]
    selected = torch.where(reliable, selected, torch.full_like(selected, -1))
    return labels, selected, reliable


@torch.no_grad()
def _soft_splat_surface_labels(
    group_ids,
    keypoint_xy,
    render_pkg,
    *,
    topk=4,
):
    visible = torch.nonzero(
        render_pkg["visibility_filter"], as_tuple=False
    ).reshape(-1)
    if visible.numel() == 0:
        count = keypoint_xy.shape[0]
        return (
            torch.full((count, topk), -1, device=keypoint_xy.device, dtype=torch.long),
            torch.full((count, topk), -1, device=keypoint_xy.device, dtype=torch.long),
            torch.zeros((count, topk), device=keypoint_xy.device),
            torch.zeros(count, device=keypoint_xy.device, dtype=torch.bool),
        )
    local_ids, weights, reliable = bank_splat_provenance_2dgs(
        keypoint_xy,
        visible,
        render_pkg["rgb_meta"],
        rendered_depth=render_pkg.get("depth"),
        topk=topk,
        candidate_topk=max(32, topk * 8),
    )
    primitive_ids = visible[local_ids]
    groups = group_ids[primitive_ids]
    groups = groups.masked_fill(~reliable[:, None], -1)
    primitive_ids = primitive_ids.masked_fill(~reliable[:, None], -1)
    weights = weights.masked_fill(~reliable[:, None], 0.0)
    return groups, primitive_ids, weights, reliable


@torch.no_grad()
def _build_episode(
    gaussians,
    camera,
    image,
    feature_extractor,
    detector,
    group_ids,
    background,
    detect_num,
    nms_radius,
    longest_edge,
    provenance_topk,
):
    resolution = get_resolution_from_longest_edge(image.shape[-2], image.shape[-1], longest_edge)
    feature_map = extract_normalized_feature_map(
        feature_extractor, image.cuda(), size=resolution
    )
    height, width = feature_map.shape[-2:]
    keypoint_xy, descriptors, detector_scores = _detect_query(
        feature_map, detector, detect_num, nms_radius
    )
    render_pkg = _render_teacher(gaussians, camera, width, height, background)
    labels, primitive_ids, provenance_weights, reliable = _soft_splat_surface_labels(
        group_ids,
        keypoint_xy,
        render_pkg,
        topk=provenance_topk,
    )
    return {
        "xy": keypoint_xy[reliable].half().cpu(),
        "descriptors": descriptors[reliable].half().cpu(),
        "scores": detector_scores[reliable].half().cpu(),
        "groups": labels[reliable].int().cpu(),
        "primitive_ids": primitive_ids[reliable].int().cpu(),
        "provenance_weights": provenance_weights[reliable].half().cpu(),
        "query_count": int(keypoint_xy.shape[0]),
        "reliable_count": int(reliable.sum().item()),
    }


def _episode_to_device(episode):
    return {
        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
        for key, value in episode.items()
    }


def _save_state(
    output_dir,
    descriptors,
    initial_descriptors,
    gate_logits,
    group_ids,
    active_indices,
    config,
    history,
):
    os.makedirs(output_dir, exist_ok=True)
    active_indices = active_indices.detach().cpu().long().sort().values
    active_features = F.normalize(
        descriptors.detach()[active_indices.to(descriptors.device)].float(), dim=1
    ).cpu()
    state = {
        "version": 2,
        "architecture": "progressive_localization_coreset_v2",
        "landmark_indices": active_indices,
        "landmark_features": active_features,
        "gate_logits": gate_logits.detach().half().cpu(),
        "surface_group_ids": group_ids.detach().int().cpu(),
        "initial_active_features": initial_descriptors[active_indices].half().cpu(),
        "config": config,
        "history": history,
    }
    torch.save(state, os.path.join(output_dir, "coreset_state.pt"))
    torch.save(state, os.path.join(output_dir, "final_candidate_teacher_state.pt"))
    torch.save(active_features, os.path.join(output_dir, "localization_features.pt"))
    torch.save(
        state["surface_group_ids"][active_indices],
        os.path.join(output_dir, "surface_group_ids.pt"),
    )
    with open(os.path.join(output_dir, "sampled_idx.pkl"), "wb") as handle:
        pickle.dump(active_indices, handle)
    torch.save(
        {
            "landmark_indices": active_indices,
            "surface_group_ids": state["surface_group_ids"][active_indices],
            "selection_gate": torch.sigmoid(state["gate_logits"].float())[active_indices],
            "candidate_quality": torch.sigmoid(state["gate_logits"].float()),
        },
        os.path.join(output_dir, "landmark_meta.pt"),
    )
    with open(os.path.join(output_dir, "training_summary.json"), "w") as handle:
        json.dump({"config": config, "history": history}, handle, indent=2)


def train(args, dataset, scene, gaussians):
    output_dir = os.path.join(dataset.model_path, args.output_folder)
    os.makedirs(output_dir, exist_ok=True)
    cameras = scene.getTrainCameras().copy()
    if not cameras:
        raise ValueError("V2 coreset training requires training cameras")
    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    detector = KpDetector(feature_extractor.feature_dim)
    detector.load_state_dict(torch.load(args.proposal_detector, map_location="cpu"))
    detector.cuda().eval()
    for parameter in detector.parameters():
        parameter.requires_grad_(False)
    for parameter in feature_extractor.parameters():
        parameter.requires_grad_(False)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        device="cuda",
        dtype=torch.float32,
    )
    for parameter in gaussians.parameters():
        parameter.requires_grad_(False)

    normals = build_rotation(gaussians.get_rotation)[:, :, 2]
    group_ids, group_count = build_surface_groups(
        gaussians.get_xyz,
        normals,
        voxel_size=args.surface_voxel_size,
        normal_bins=args.surface_normal_bins,
    )
    point_count = int(gaussians.get_xyz.shape[0])
    all_indices = torch.arange(point_count, device="cuda", dtype=torch.long)

    mv_cameras = cameras
    if args.mvinit_views > 0 and len(cameras) > args.mvinit_views:
        positions = torch.linspace(0, len(cameras) - 1, args.mvinit_views).round().long()
        mv_cameras = [cameras[int(index)] for index in positions]
    depth_cache = {}
    for camera in tqdm(mv_cameras, desc="V2 MVInit depth"):
        resolution = get_resolution_from_longest_edge(
            camera.original_image.shape[-2], camera.original_image.shape[-1], dataset.longest_edge
        )
        pkg = _render_teacher(gaussians, camera, resolution[1], resolution[0], background)
        depth_cache[camera.image_name] = pkg["depth"].detach().cpu()

    def feature_source(camera):
        resolution = get_resolution_from_longest_edge(
            camera.original_image.shape[-2], camera.original_image.shape[-1], dataset.longest_edge
        )
        return extract_normalized_feature_map(
            feature_extractor, camera.original_image.cuda(), size=resolution
        )

    def depth_source(camera):
        return depth_cache[camera.image_name].cuda(non_blocking=True)

    print(
        f"[LaFGS V2] MVInit full map: primitives={point_count} views={len(mv_cameras)}"
    )
    mv_result = build_multiview_initialization(
        gaussians,
        mv_cameras,
        feature_source,
        landmark_indices=all_indices,
        depth_maps=depth_source,
        config=MultiViewInitConfig(
            min_observations=args.mvinit_min_observations,
            chunk_size=args.mvinit_chunk_size,
            alpha_threshold=0.0,
        ),
    )
    initial = F.normalize(mv_result.features.float(), dim=1)
    unobserved = mv_result.observation_count == 0
    if bool(unobserved.any()):
        fallback = F.normalize(gaussians.get_loc_feature.squeeze().float(), dim=1)
        initial[unobserved] = fallback[unobserved]
    gaussians._loc_feature = torch.nn.Parameter(
        initial[:, :, None].contiguous(), requires_grad=True
    )
    initial_cpu = initial.half().cpu()
    del mv_result, initial
    torch.cuda.empty_cache()

    gate_logits = torch.nn.Parameter(torch.full((point_count,), 6.0, device="cuda"))
    descriptor_optimizer = torch.optim.SGD([gaussians._loc_feature], lr=args.descriptor_lr)
    gate_optimizer = torch.optim.Adam([gate_logits], lr=args.gate_lr)
    schedule = make_progressive_budget_schedule(
        point_count, args.iterations, args.final_budget
    )
    active_indices = all_indices
    previous_active = None
    representatives = group_representatives(gate_logits.detach(), group_ids, group_count)
    active_group_count = torch.bincount(group_ids, minlength=group_count)
    active_group_rep = active_group_representatives(
        gate_logits.detach(), group_ids, active_indices, group_count
    )
    group_priority = torch.zeros(group_count, device="cuda")
    episode_cache = {}
    history = []
    running = {}
    stage_budget = point_count
    start_time = time.time()

    progress = tqdm(range(1, args.iterations + 1), desc="LaFGS V2 Coreset")
    for iteration in progress:
        target_budget = schedule.budget(iteration)
        refresh = (
            iteration == 1
            or target_budget != stage_budget
            or iteration % args.projection_interval == 0
        )
        if refresh:
            previous_active = active_indices
            active_indices = project_active_set(
                gate_logits.detach(),
                target_budget,
                previous_active=previous_active,
                hysteresis=args.hysteresis,
                group_ids=group_ids,
                group_priority=group_priority,
            )
            representatives = group_representatives(
                gate_logits.detach(), group_ids, group_count
            )
            active_group_count = torch.bincount(
                group_ids[active_indices], minlength=group_count
            )
            active_group_rep = active_group_representatives(
                gate_logits.detach(), group_ids, active_indices, group_count
            )
            stage_budget = target_budget

        add_synthetic = (
            args.synthetic_ratio > 0
            and random.random() < args.synthetic_ratio
            and len(cameras) >= 2
        )
        real_camera = cameras[(iteration - 1) % len(cameras)]
        if real_camera.image_name not in episode_cache:
            episode_cache[real_camera.image_name] = _build_episode(
                gaussians,
                real_camera,
                real_camera.original_image,
                feature_extractor,
                detector,
                group_ids,
                background,
                args.detect_num,
                args.nms_radius,
                dataset.longest_edge,
                args.provenance_topk,
            )
        episodes = [_episode_to_device(episode_cache[real_camera.image_name])]
        if add_synthetic:
            synthetic = sample_interpolated_novel_view(
                cameras, alpha_min=0.35, alpha_max=0.65
            )
            reference = cameras[max(0, int(synthetic.train_index_a))]
            resolution = get_resolution_from_longest_edge(
                reference.original_image.shape[-2],
                reference.original_image.shape[-1],
                dataset.longest_edge,
            )
            pkg = _render_teacher(
                gaussians, synthetic, resolution[1], resolution[0], background
            )
            episodes.append(
                _episode_to_device(
                    _build_episode(
                        gaussians,
                        synthetic,
                        pkg["render"].clamp(0.0, 1.0),
                        feature_extractor,
                        detector,
                        group_ids,
                        background,
                        args.detect_num,
                        args.nms_radius,
                        dataset.longest_edge,
                        args.provenance_topk,
                    )
                )
            )

        query_parts = []
        group_parts = []
        primitive_parts = []
        weight_parts = []
        for episode in episodes:
            episode_query = episode["descriptors"].float()
            episode_groups = episode["groups"].long()
            episode_primitives = episode["primitive_ids"].long()
            episode_weights = episode["provenance_weights"].float()
            if episode_query.shape[0] > args.train_keypoints:
                choice = torch.randperm(episode_query.shape[0], device="cuda")[
                    : args.train_keypoints
                ]
                episode_query = episode_query[choice]
                episode_groups = episode_groups[choice]
                episode_primitives = episode_primitives[choice]
                episode_weights = episode_weights[choice]
            query_parts.append(episode_query)
            group_parts.append(episode_groups)
            primitive_parts.append(episode_primitives)
            weight_parts.append(episode_weights)

        query = torch.cat(query_parts)
        target_groups = torch.cat(group_parts)
        target_primitives = torch.cat(primitive_parts)
        target_weights = torch.cat(weight_parts)
        valid_group = (target_groups >= 0) & (target_groups < group_count)
        valid_query = valid_group.any(dim=1) & (target_weights.sum(dim=1) > 0)
        query = query[valid_query]
        target_groups = target_groups[valid_query]
        target_primitives = target_primitives[valid_query]
        target_weights = target_weights[valid_query]
        valid_group = valid_group[valid_query]
        target_weights = target_weights.masked_fill(~valid_group, 0.0)
        target_weights = target_weights / target_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        if query.shape[0] < 4:
            continue

        with torch.no_grad():
            observed_count = torch.zeros(group_count, device="cuda")
            observed_count.scatter_add_(
                0, target_groups[valid_group], target_weights[valid_group]
            )
            group_priority.mul_(args.group_priority_decay).add_(observed_count)

        with torch.no_grad():
            active_features = gaussians._loc_feature[active_indices].squeeze(-1)
            retrieval = chunked_exact_topk(
                query,
                active_features,
                topk=args.retrieval_topk,
                chunk_size=args.retrieval_chunk_size,
            )
            negative_indices = active_indices[retrieval.indices]
            retrieved_groups = group_ids[negative_indices]
            positive_group_match = (
                retrieved_groups[:, :, None] == target_groups[:, None, :]
            ) & valid_group[:, None, :]
            negative_mask = ~positive_group_match.any(dim=2)
            recall = positive_group_match.flatten(1).any(dim=1)
            safe_groups = target_groups.clamp_min(0)
            positive_active = valid_group & (active_group_count[safe_groups] > 0)
            covered_mass = (target_weights * positive_active).sum(dim=1)
            coverage_miss = 1.0 - covered_mass

        covered = positive_active.any(dim=1)
        positive_indices = active_group_rep[safe_groups[covered]].clamp_min(0)
        positive_mask = positive_active[covered]
        positive_weights = target_weights[covered]
        match_query = query[covered]
        match_negative_indices = negative_indices[covered]
        match_negative_mask = negative_mask[covered]

        positive_features = F.embedding(
            positive_indices, gaussians._loc_feature.squeeze(-1), sparse=True
        )
        negative_features = F.embedding(
            match_negative_indices, gaussians._loc_feature.squeeze(-1), sparse=True
        )
        match_loss = coreset_soft_matching_loss(
            match_query,
            positive_features,
            positive_weights,
            gate_logits[positive_indices],
            positive_mask,
            negative_features,
            gate_logits[match_negative_indices],
            match_negative_mask,
            temperature=args.temperature,
        )
        regularizers = progressive_coreset_regularizers(
            gate_logits,
            group_ids,
            target_groups[valid_group],
            target_budget,
            observed_group_weights=target_weights[valid_group],
            redundancy_multiplier=args.redundancy_multiplier,
        )
        selected = torch.unique(
            torch.cat(
                [positive_indices[positive_mask], match_negative_indices.reshape(-1)]
            )
        )
        current_selected = F.embedding(
            selected, gaussians._loc_feature.squeeze(-1), sparse=True
        )
        initial_selected = initial_cpu[selected.detach().cpu()].float().cuda(non_blocking=True)
        trust_loss = descriptor_trust_loss(
            current_selected,
            initial_selected,
            torch.sigmoid(gate_logits[selected]).detach(),
        )
        redundancy_weight = args.redundancy_weight if target_budget < point_count else 0.0
        loss = (
            match_loss
            + args.coverage_weight * regularizers["coverage"]
            + args.budget_weight * regularizers["budget"]
            + redundancy_weight * regularizers["redundancy"]
            + args.trust_weight * trust_loss
        )
        descriptor_optimizer.zero_grad(set_to_none=True)
        gate_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        descriptor_optimizer.step()
        gate_optimizer.step()
        with torch.no_grad():
            gaussians._loc_feature[selected] = F.normalize(
                gaussians._loc_feature[selected].squeeze(-1), dim=1
            )[:, :, None]

        metrics = {
            "loss": float(loss.detach()),
            "match": float(match_loss.detach()),
            "coverage": float(regularizers["coverage"].detach()),
            "budget": float(regularizers["budget"].detach()),
            "redundancy": float(regularizers["redundancy"].detach()),
            "trust": float(trust_loss.detach()),
            "recall_at_k": float(recall.float().mean()),
            "coverage_miss": float(coverage_miss.mean()),
            "gate_sum": float(regularizers["probability_sum"]),
            "reliable_query_ratio": sum(x["reliable_count"] for x in episodes)
            / max(sum(x["query_count"] for x in episodes), 1),
            "synthetic": float(add_synthetic),
        }
        for key, value in metrics.items():
            running[key] = running.get(key, 0.0) * 0.98 + value * 0.02
        progress.set_postfix(
            K=target_budget,
            loss=f"{running['loss']:.3f}",
            recall=f"{running['recall_at_k']:.3f}",
            miss=f"{running['coverage_miss']:.3f}",
        )

        if iteration % args.log_interval == 0 or iteration == args.iterations:
            diagnostics = active_set_diagnostics(
                active_indices, group_ids, previous_active
            )
            record = {
                "iteration": iteration,
                "target_budget": target_budget,
                "elapsed_seconds": time.time() - start_time,
                **diagnostics,
                **running,
            }
            record.update({f"raw_{key}": value for key, value in metrics.items()})
            history.append(record)
            with open(os.path.join(output_dir, "training_log.jsonl"), "a") as handle:
                handle.write(json.dumps(record) + "\n")

    final_active = project_active_set(
        gate_logits.detach(),
        args.final_budget,
        active_indices,
        args.hysteresis,
        group_ids=group_ids,
        group_priority=group_priority,
    )
    config = vars(args).copy()
    config.update(
        {
            "point_count": point_count,
            "surface_group_count": group_count,
            "schedule_budgets": list(schedule.budgets),
            "schedule_boundaries": list(schedule.boundaries),
            "mvinit_observed_count": int((~unobserved).sum().item()),
        }
    )
    _save_state(
        output_dir,
        gaussians._loc_feature.squeeze(-1),
        initial_cpu,
        gate_logits,
        group_ids,
        final_active,
        config,
        history,
    )
    print(f"[LaFGS V2] Saved final coreset to {output_dir}")


def main():
    parser = argparse.ArgumentParser("LaFGS V2 progressive localization coreset")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--output_folder", default="lafgs_v2_coreset")
    parser.add_argument("--proposal_detector", required=True)
    parser.add_argument("--final_budget", type=int, default=16384)
    parser.add_argument("--surface_voxel_size", type=float, default=0.05)
    parser.add_argument("--surface_normal_bins", type=int, default=0)
    parser.add_argument("--mvinit_views", type=int, default=16)
    parser.add_argument("--mvinit_min_observations", type=int, default=1)
    parser.add_argument("--mvinit_chunk_size", type=int, default=32768)
    parser.add_argument("--detect_num", type=int, default=2048)
    parser.add_argument("--train_keypoints", type=int, default=512)
    parser.add_argument("--nms_radius", type=int, default=2)
    parser.add_argument("--retrieval_topk", type=int, default=16)
    parser.add_argument("--provenance_topk", type=int, default=4)
    parser.add_argument("--retrieval_chunk_size", type=int, default=8192)
    parser.add_argument("--projection_interval", type=int, default=500)
    parser.add_argument("--hysteresis", type=float, default=0.05)
    parser.add_argument("--group_priority_decay", type=float, default=0.999)
    parser.add_argument("--descriptor_lr", type=float, default=2e-3)
    parser.add_argument("--gate_lr", type=float, default=1e-2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--coverage_weight", type=float, default=0.5)
    parser.add_argument("--budget_weight", type=float, default=1.0)
    parser.add_argument("--redundancy_weight", type=float, default=0.1)
    parser.add_argument("--redundancy_multiplier", type=float, default=2.0)
    parser.add_argument("--trust_weight", type=float, default=0.01)
    parser.add_argument("--synthetic_ratio", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    fill_missing_model_defaults(args)
    seed_everything(args.seed)
    safe_state(args.quiet)
    dataset = model.extract(args)
    if dataset.gaussian_type != "2dgs":
        raise ValueError("LaFGS V2 requires a 2DGS render map")
    gaussians = GaussianModel_2dgs(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration)
    train(args, dataset, scene, gaussians)


if __name__ == "__main__":
    main()
