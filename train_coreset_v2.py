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
from localization_training.progressive_coreset import (
    active_set_diagnostics,
    build_surface_groups,
    build_surface_patch_atoms,
    deployment_soft_matching_loss,
    descriptor_trust_loss,
    discrete_select_atoms,
    make_gradual_budget_schedule,
)
from localization_training.splat_provenance import bank_splat_provenance_2dgs
from scene import Scene
from scene.gaussian_model import GaussianModel_2dgs
from scene.kpdetector import KpDetector, simple_nms
from train_detector import extract_normalized_feature_map, fill_missing_model_defaults
from utils.general_utils import build_rotation, safe_state, seed_everything
from utils.image_utils import get_resolution_from_longest_edge


def _camera_pose(camera, device="cuda"):
    return camera.world_view_transform.transpose(0, 1).to(device)


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


def _save_atom_state(
    output_dir,
    atom_features,
    initial_atom_features,
    atom_raw_indices,
    raw_to_atom,
    coverage_cell_ids,
    redundancy_group_ids,
    selection_utility,
    active_indices,
    config,
    history,
    atom_diagnostics,
    *,
    suffix="",
):
    os.makedirs(output_dir, exist_ok=True)
    active_indices = active_indices.detach().cpu().long().sort().values
    atom_raw_indices = atom_raw_indices.detach().cpu().long()
    landmark_indices = atom_raw_indices[active_indices]
    active_features = F.normalize(
        atom_features.detach()[active_indices.to(atom_features.device)].float(), dim=1
    ).cpu()
    state = {
        "version": 21,
        "architecture": "surface_patch_localization_coreset_v2_1",
        "landmark_indices": landmark_indices,
        "landmark_features": active_features,
        "active_atom_indices": active_indices,
        "atom_raw_indices": atom_raw_indices,
        "raw_to_atom": raw_to_atom.detach().int().cpu(),
        "coverage_cell_ids": coverage_cell_ids.detach().int().cpu(),
        "redundancy_group_ids": redundancy_group_ids.detach().int().cpu(),
        "selection_utility": selection_utility.detach().half().cpu(),
        "initial_active_features": initial_atom_features[active_indices].half().cpu(),
        "atom_diagnostics": atom_diagnostics,
        "config": config,
        "history": history,
    }
    tag = f"_{suffix}" if suffix else ""
    torch.save(state, os.path.join(output_dir, f"coreset_state{tag}.pt"))
    if not suffix:
        torch.save(state, os.path.join(output_dir, "final_candidate_teacher_state.pt"))
        torch.save(active_features, os.path.join(output_dir, "localization_features.pt"))
        torch.save(
            state["coverage_cell_ids"][active_indices],
            os.path.join(output_dir, "coverage_cell_ids.pt"),
        )
        with open(os.path.join(output_dir, "sampled_idx.pkl"), "wb") as handle:
            pickle.dump(landmark_indices, handle)
        torch.save(
            {
                "landmark_indices": landmark_indices,
                "active_atom_indices": active_indices,
                "coverage_cell_ids": state["coverage_cell_ids"][active_indices],
                "redundancy_group_ids": state["redundancy_group_ids"][active_indices],
                "candidate_quality": state["selection_utility"],
            },
            os.path.join(output_dir, "landmark_meta.pt"),
        )
        with open(os.path.join(output_dir, "training_summary.json"), "w") as handle:
            json.dump(
                {
                    "config": config,
                    "atom_diagnostics": atom_diagnostics,
                    "history": history,
                },
                handle,
                indent=2,
            )


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
            alpha_threshold=args.mvinit_alpha_threshold,
        ),
    )
    raw_initial = F.normalize(mv_result.features.float(), dim=1)
    observation_count = mv_result.observation_count.long()
    unobserved = observation_count == 0
    if bool(unobserved.any()):
        fallback = F.normalize(gaussians.get_loc_feature.squeeze().float(), dim=1)
        raw_initial[unobserved] = fallback[unobserved]
    if args.strong_feature_state:
        strong = torch.load(args.strong_feature_state, map_location="cpu")
        if "landmark_indices" in strong and "landmark_features" in strong:
            strong_indices = torch.as_tensor(
                strong["landmark_indices"], device="cuda", dtype=torch.long
            )
            strong_features = torch.as_tensor(
                strong["landmark_features"], device="cuda"
            ).float()
        elif "loc_prototype" in strong:
            strong_features = torch.as_tensor(
                strong["loc_prototype"], device="cuda"
            ).float()
            strong_indices = torch.arange(
                strong_features.shape[0], device="cuda", dtype=torch.long
            )
        else:
            raise ValueError(
                "Strong feature state must contain landmark_features or loc_prototype"
            )
        valid = (strong_indices >= 0) & (strong_indices < point_count)
        valid &= torch.linalg.norm(strong_features, dim=1) > 1e-6
        strong_features = F.normalize(strong_features, dim=1)
        blend = float(args.strong_feature_blend)
        raw_initial[strong_indices[valid]] = F.normalize(
            blend * strong_features[valid]
            + (1.0 - blend) * raw_initial[strong_indices[valid]],
            dim=1,
        )
        del strong, strong_features, strong_indices, valid
        torch.cuda.empty_cache()

    identity_ids, identity_count = build_surface_groups(
        gaussians.get_xyz,
        normals,
        voxel_size=args.identity_voxel_size,
        normal_bins=args.identity_normal_bins,
    )
    identity_priority = torch.zeros(identity_count, device="cuda")
    discovery_cache = {}
    discovery_cameras = cameras
    if args.atom_discovery_views > 0 and len(cameras) > args.atom_discovery_views:
        positions = torch.linspace(
            0, len(cameras) - 1, args.atom_discovery_views
        ).round().long()
        discovery_cameras = [cameras[int(index)] for index in positions]
    for camera in tqdm(discovery_cameras, desc="V2.1 query atom discovery"):
        episode = _build_episode(
            gaussians,
            camera,
            camera.original_image,
            feature_extractor,
            detector,
            identity_ids,
            background,
            args.detect_num,
            args.nms_radius,
            dataset.longest_edge,
            args.provenance_topk,
        )
        discovery_cache[camera.image_name] = episode
        groups = episode["groups"].long().cuda(non_blocking=True)
        weights = episode["provenance_weights"].float().cuda(non_blocking=True)
        valid_groups = (groups >= 0) & (groups < identity_count)
        identity_priority.scatter_add_(
            0, groups[valid_groups], weights[valid_groups]
        )

    atoms = build_surface_patch_atoms(
        gaussians.get_xyz,
        normals,
        observation_count,
        identity_voxel_size=args.identity_voxel_size,
        identity_normal_bins=args.identity_normal_bins,
        coverage_voxel_size=args.coverage_voxel_size,
        redundancy_voxel_size=args.redundancy_voxel_size,
        redundancy_normal_bins=args.redundancy_normal_bins,
        min_observations=args.mvinit_min_observations,
        max_atoms=args.max_atoms,
        identity_patch_priority=identity_priority,
    )
    atom_raw_indices = atoms.representative_raw_indices
    atom_count = int(atom_raw_indices.numel())
    if atom_count < args.final_budget:
        raise ValueError(
            f"Only {atom_count} observable surface atoms for final budget {args.final_budget}"
        )
    atom_initial = F.normalize(raw_initial[atom_raw_indices], dim=1)
    atom_features = torch.nn.Parameter(atom_initial.clone(), requires_grad=True)
    initial_cpu = atom_initial.half().cpu()
    descriptor_optimizer = torch.optim.SGD([atom_features], lr=args.descriptor_lr)
    schedule = make_gradual_budget_schedule(
        atom_count,
        args.iterations,
        args.final_budget,
        keep_ratio=args.stage_keep_ratio,
        warmup_ratio=args.warmup_ratio,
    )
    active_indices = torch.arange(atom_count, device="cuda", dtype=torch.long)
    previous_active = None
    active_mask = torch.ones(atom_count, device="cuda", dtype=torch.bool)
    observation_utility = torch.zeros(atom_count, device="cuda")
    match_utility = torch.zeros(atom_count, device="cuda")
    negative_risk = torch.zeros(atom_count, device="cuda")
    coverage_count = int(atoms.coverage_cell_ids.max().item()) + 1
    coverage_priority = torch.zeros(coverage_count, device="cuda")
    episode_cache = {}
    patch_to_atom_cpu = atoms.identity_patch_to_atom.long().cpu()
    for image_name, episode in discovery_cache.items():
        identity_group = episode["groups"].long()
        valid_identity = (identity_group >= 0) & (
            identity_group < patch_to_atom_cpu.numel()
        )
        atom_group = torch.full_like(identity_group, -1)
        atom_group[valid_identity] = patch_to_atom_cpu[
            identity_group[valid_identity]
        ]
        converted = dict(episode)
        converted["groups"] = atom_group.int()
        episode_cache[image_name] = converted
    history = []
    running = {}
    stage_budget = atom_count
    selection_diagnostics = {}
    start_time = time.time()
    del mv_result, raw_initial, atom_initial
    torch.cuda.empty_cache()

    progress = tqdm(range(1, args.iterations + 1), desc="LaFGS V2 Coreset")
    for iteration in progress:
        target_budget = schedule.budget(iteration)
        refresh = iteration == 1 or target_budget != stage_budget or (
            target_budget < atom_count and iteration % args.selection_interval == 0
        )
        if refresh:
            previous_active = active_indices
            selection_utility = (
                args.match_utility_weight * match_utility
                + args.observation_utility_weight * torch.log1p(observation_utility)
                - args.negative_risk_weight * negative_risk
            )
            active_indices, selection_diagnostics = discrete_select_atoms(
                selection_utility,
                atoms.coverage_cell_ids,
                atoms.redundancy_group_ids,
                target_budget,
                previous_active=previous_active,
                hysteresis=args.hysteresis,
                coverage_priority=coverage_priority,
                coverage_fraction=args.coverage_fraction,
                redundancy_penalty=args.selection_redundancy_penalty,
                return_diagnostics=True,
            )
            active_mask.zero_()
            active_mask[active_indices] = True
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
                atoms.raw_to_atom,
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
                        atoms.raw_to_atom,
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
        valid_group = (target_groups >= 0) & (target_groups < atom_count)
        pool_covered_mass = (target_weights * valid_group).sum(dim=1)
        pool_miss = 1.0 - pool_covered_mass
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

        mapped_raw = target_primitives[valid_group]
        mapped_atoms = target_groups[valid_group]
        mapping_distance = torch.linalg.norm(
            gaussians.get_xyz[mapped_raw]
            - gaussians.get_xyz[atom_raw_indices[mapped_atoms]],
            dim=1,
        )
        mapping_normal_cos = (
            normals[mapped_raw] * normals[atom_raw_indices[mapped_atoms]]
        ).sum(dim=1).clamp(-1, 1)
        mapping_normal_angle = torch.rad2deg(torch.acos(mapping_normal_cos))

        with torch.no_grad():
            active_features = atom_features[active_indices]
            retrieval = chunked_exact_topk(
                query,
                active_features,
                topk=args.retrieval_topk,
                chunk_size=args.retrieval_chunk_size,
            )
            negative_indices = active_indices[retrieval.indices]
            retrieved_groups = negative_indices
            positive_group_match = (
                retrieved_groups[:, :, None] == target_groups[:, None, :]
            ) & valid_group[:, None, :]
            negative_mask = ~positive_group_match.any(dim=2)
            recall = positive_group_match.flatten(1).any(dim=1)
            safe_groups = target_groups.clamp_min(0)
            positive_active = valid_group & active_mask[safe_groups]
            covered_mass = (target_weights * positive_active).sum(dim=1)
            coverage_miss = 1.0 - covered_mass

        # Shadow positives remain trainable even when their atom is inactive.
        positive_indices = safe_groups
        positive_mask = valid_group
        positive_weights = target_weights
        match_query = query
        match_negative_indices = negative_indices
        match_negative_mask = negative_mask

        positive_features = F.embedding(
            positive_indices, atom_features, sparse=True
        )
        negative_features = F.embedding(
            match_negative_indices, atom_features, sparse=True
        )
        match_loss = deployment_soft_matching_loss(
            match_query,
            positive_features,
            positive_weights,
            positive_mask,
            negative_features,
            match_negative_mask,
            temperature=args.temperature,
        )
        negative_similarity = F.cosine_similarity(
            query[:, None], negative_features, dim=2
        ).masked_fill(~match_negative_mask, -1.0)
        miss_mask = coverage_miss > 1e-4
        if bool(miss_mask.any()):
            miss_rejection = F.relu(
                negative_similarity[miss_mask].amax(dim=1) - args.miss_negative_margin
            ).mean()
        else:
            miss_rejection = match_loss * 0.0
        selected = torch.unique(
            torch.cat(
                [positive_indices[positive_mask], match_negative_indices.reshape(-1)]
            )
        )
        current_selected = F.embedding(
            selected, atom_features, sparse=True
        )
        initial_selected = initial_cpu[selected.detach().cpu()].float().cuda(non_blocking=True)
        trust_loss = descriptor_trust_loss(
            current_selected,
            initial_selected,
        )
        loss = (
            match_loss
            + args.miss_rejection_weight * miss_rejection
            + args.trust_weight * trust_loss
        )
        descriptor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        descriptor_optimizer.step()
        with torch.no_grad():
            atom_features[selected] = F.normalize(atom_features[selected], dim=1)
            decay = float(args.utility_decay)
            observation_utility.mul_(decay)
            match_utility.mul_(decay)
            negative_risk.mul_(decay)
            flat_atoms = target_groups[valid_group]
            flat_weights = target_weights[valid_group]
            observation_utility.scatter_add_(
                0, flat_atoms, (1.0 - decay) * flat_weights
            )
            positive_similarity = F.cosine_similarity(
                query[:, None], positive_features.detach(), dim=2
            )
            match_utility.scatter_add_(
                0,
                flat_atoms,
                (1.0 - decay) * (positive_similarity[valid_group] + 1.0) * 0.5 * flat_weights,
            )
            top_negative = match_negative_indices[:, 0]
            top_negative_risk = F.relu(negative_similarity[:, 0] - args.miss_negative_margin)
            negative_risk.scatter_add_(
                0, top_negative, (1.0 - decay) * top_negative_risk
            )
            observed_cells = atoms.coverage_cell_ids[flat_atoms]
            coverage_priority.mul_(decay)
            coverage_priority.scatter_add_(
                0, observed_cells, (1.0 - decay) * flat_weights
            )

        metrics = {
            "loss": float(loss.detach()),
            "match": float(match_loss.detach()),
            "miss_rejection": float(miss_rejection.detach()),
            "trust": float(trust_loss.detach()),
            "recall_at_k": float(recall.float().mean()),
            "coverage_miss": float(coverage_miss.mean()),
            "atom_pool_miss": float(pool_miss.mean()),
            "provenance_atom_distance_mean_m": float(mapping_distance.mean()),
            "provenance_atom_distance_p95_m": float(torch.quantile(mapping_distance, 0.95)),
            "provenance_atom_normal_angle_mean_deg": float(mapping_normal_angle.mean()),
            "selection_utility_mean": float(selection_utility.mean()),
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
                active_indices, atoms.coverage_cell_ids, previous_active
            )
            active_redundancy = torch.bincount(
                atoms.redundancy_group_ids[active_indices]
            )
            record = {
                "iteration": iteration,
                "target_budget": target_budget,
                "elapsed_seconds": time.time() - start_time,
                **diagnostics,
                **running,
                "active_redundancy_fraction": float(
                    (active_redundancy > 1).sum() / max(active_redundancy.numel(), 1)
                ),
                "selection_score_p50": float(torch.quantile(selection_utility, 0.5)),
                "selection_score_p95": float(torch.quantile(selection_utility, 0.95)),
                **selection_diagnostics,
            }
            record.update({f"raw_{key}": value for key, value in metrics.items()})
            history.append(record)
            with open(os.path.join(output_dir, "training_log.jsonl"), "a") as handle:
                handle.write(json.dumps(record) + "\n")

        if iteration in schedule.boundaries:
            checkpoint_config = vars(args).copy()
            checkpoint_config.update({"atom_count": atom_count})
            _save_atom_state(
                output_dir,
                atom_features,
                initial_cpu,
                atom_raw_indices,
                atoms.raw_to_atom,
                atoms.coverage_cell_ids,
                atoms.redundancy_group_ids,
                selection_utility,
                active_indices,
                checkpoint_config,
                history,
                atoms.diagnostics,
                suffix=f"iter{iteration}_k{target_budget}",
            )

    selection_utility = (
        args.match_utility_weight * match_utility
        + args.observation_utility_weight * torch.log1p(observation_utility)
        - args.negative_risk_weight * negative_risk
    )
    final_active, selection_diagnostics = discrete_select_atoms(
        selection_utility,
        atoms.coverage_cell_ids,
        atoms.redundancy_group_ids,
        args.final_budget,
        previous_active=active_indices,
        hysteresis=args.hysteresis,
        coverage_priority=coverage_priority,
        coverage_fraction=args.coverage_fraction,
        redundancy_penalty=args.selection_redundancy_penalty,
        return_diagnostics=True,
    )
    config = vars(args).copy()
    config.update(
        {
            "point_count": point_count,
            "atom_count": atom_count,
            "schedule_budgets": list(schedule.budgets),
            "schedule_boundaries": list(schedule.boundaries),
            "mvinit_observed_count": int((~unobserved).sum().item()),
        }
    )
    _save_atom_state(
        output_dir,
        atom_features,
        initial_cpu,
        atom_raw_indices,
        atoms.raw_to_atom,
        atoms.coverage_cell_ids,
        atoms.redundancy_group_ids,
        selection_utility,
        final_active,
        config,
        history,
        atoms.diagnostics,
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
    parser.add_argument("--identity_voxel_size", type=float, default=0.02)
    parser.add_argument("--identity_normal_bins", type=int, default=8)
    parser.add_argument("--coverage_voxel_size", type=float, default=0.20)
    parser.add_argument("--redundancy_voxel_size", type=float, default=0.05)
    parser.add_argument("--redundancy_normal_bins", type=int, default=4)
    parser.add_argument("--max_atoms", type=int, default=150000)
    parser.add_argument("--atom_discovery_views", type=int, default=64)
    parser.add_argument("--mvinit_views", type=int, default=16)
    parser.add_argument("--mvinit_min_observations", type=int, default=2)
    parser.add_argument("--mvinit_alpha_threshold", type=float, default=0.05)
    parser.add_argument("--mvinit_chunk_size", type=int, default=32768)
    parser.add_argument("--strong_feature_state", default="")
    parser.add_argument("--strong_feature_blend", type=float, default=0.75)
    parser.add_argument("--detect_num", type=int, default=4096)
    parser.add_argument("--train_keypoints", type=int, default=512)
    parser.add_argument("--nms_radius", type=int, default=2)
    parser.add_argument("--retrieval_topk", type=int, default=16)
    parser.add_argument("--provenance_topk", type=int, default=4)
    parser.add_argument("--retrieval_chunk_size", type=int, default=8192)
    parser.add_argument("--selection_interval", type=int, default=1000)
    parser.add_argument("--hysteresis", type=float, default=0.05)
    parser.add_argument("--stage_keep_ratio", type=float, default=0.75)
    parser.add_argument("--warmup_ratio", type=float, default=0.20)
    parser.add_argument("--coverage_fraction", type=float, default=0.50)
    parser.add_argument("--selection_redundancy_penalty", type=float, default=0.10)
    parser.add_argument("--utility_decay", type=float, default=0.995)
    parser.add_argument("--match_utility_weight", type=float, default=1.0)
    parser.add_argument("--observation_utility_weight", type=float, default=0.1)
    parser.add_argument("--negative_risk_weight", type=float, default=0.5)
    parser.add_argument("--descriptor_lr", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--miss_rejection_weight", type=float, default=0.25)
    parser.add_argument("--miss_negative_margin", type=float, default=0.40)
    parser.add_argument("--trust_weight", type=float, default=0.01)
    parser.add_argument("--synthetic_ratio", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    fill_missing_model_defaults(args)
    safe_state(args.quiet)
    seed_everything(args.seed)
    dataset = model.extract(args)
    if dataset.gaussian_type != "2dgs":
        raise ValueError("LaFGS V2 requires a 2DGS render map")
    gaussians = GaussianModel_2dgs(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration)
    train(args, dataset, scene, gaussians)


if __name__ == "__main__":
    main()
