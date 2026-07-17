import argparse
import hashlib
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
from localization_training.episode_sampler import (
    sample_interpolated_novel_view,
    split_support_query_cameras,
)
from localization_training.correspondence import project_world_to_pixels
from localization_training.full_primitive_retrieval import chunked_exact_topk
from localization_training.lafgs_reconstruction import (
    MultiViewInitConfig,
    build_multiview_initialization,
)
from localization_training.progressive_coreset import (
    active_set_diagnostics,
    aggregate_atom_features,
    append_reprojection_positive,
    build_surface_groups,
    build_surface_patch_atoms,
    deployment_soft_matching_loss,
    descriptor_trust_loss,
    discrete_select_atoms,
    make_gradual_budget_schedule,
    provenance_mass_partition,
)
from localization_training.splat_provenance import bank_splat_provenance_2dgs
from scene import Scene
from scene.gaussian_model import GaussianModel_2dgs
from scene.kpdetector import KpDetector, simple_nms
from train_detector import extract_normalized_feature_map, fill_missing_model_defaults
from utils.general_utils import build_rotation, safe_state, seed_everything
from utils.image_utils import get_resolution_from_longest_edge
from valid_support_mask import (
    NoReferenceValidSupportMaskBuilder,
    NoReferenceValidSupportMaskConfig,
)


def _camera_pose(camera, device="cuda"):
    return camera.world_view_transform.transpose(0, 1).to(device)


def _tensor_sha256(value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path, chunk_size=8 * 1024 * 1024):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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
def _detect_query(feature_map, detector, count, nms_radius, valid_feature_mask=None):
    heatmap = simple_nms(detector(feature_map), nms_radius).reshape(-1)
    if valid_feature_mask is not None:
        valid_feature_mask = torch.as_tensor(
            valid_feature_mask, device=heatmap.device, dtype=torch.bool
        ).reshape(-1)
        if valid_feature_mask.numel() != heatmap.numel():
            raise ValueError("valid_feature_mask must match the feature-map resolution")
        heatmap = heatmap.masked_fill(~valid_feature_mask, -torch.inf)
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
def _nearest_reprojection_labels(
    xyz,
    landmark_raw_ids,
    camera,
    keypoint_xy,
    render_pkg,
    width,
    height,
    radius,
    depth_abs_tolerance=0.05,
    depth_rel_tolerance=0.02,
    chunk_size=512,
):
    if landmark_raw_ids is None or landmark_raw_ids.numel() == 0 or keypoint_xy.numel() == 0:
        return (
            torch.full((keypoint_xy.shape[0],), -1, device=keypoint_xy.device, dtype=torch.long),
            torch.zeros(keypoint_xy.shape[0], device=keypoint_xy.device, dtype=torch.bool),
        )
    fx = 0.5 * float(width) / torch.tan(torch.tensor(float(camera.FoVx) * 0.5)).item()
    fy = 0.5 * float(height) / torch.tan(torch.tensor(float(camera.FoVy) * 0.5)).item()
    K = keypoint_xy.new_tensor(
        [[fx, 0.0, 0.5 * width], [0.0, fy, 0.5 * height], [0.0, 0.0, 1.0]]
    )
    pose = _camera_pose(camera, keypoint_xy.device)
    landmark_xyz = xyz[landmark_raw_ids]
    projected, front = project_world_to_pixels(landmark_xyz, K, pose)
    inside = (
        front
        & torch.isfinite(projected).all(dim=1)
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    visibility = render_pkg.get("visibility_filter")
    if visibility is not None:
        visibility = visibility.reshape(-1).to(device=inside.device, dtype=torch.bool)
        inside &= visibility[landmark_raw_ids]
    rendered_depth = render_pkg.get("depth")
    primitive_depth = render_pkg.get("rgb_meta", {}).get("depths")
    if rendered_depth is not None and primitive_depth is not None:
        rendered_depth = rendered_depth.squeeze().to(
            device=projected.device, dtype=projected.dtype
        )
        primitive_depth = primitive_depth.reshape(-1).to(
            device=projected.device, dtype=projected.dtype
        )[landmark_raw_ids]
        if rendered_depth.dim() == 2:
            px = projected[:, 0].long().clamp(0, rendered_depth.shape[1] - 1)
            py = projected[:, 1].long().clamp(0, rendered_depth.shape[0] - 1)
            surface_depth = rendered_depth[py, px]
            tolerance = float(depth_abs_tolerance) + (
                float(depth_rel_tolerance) * surface_depth.abs()
            )
            depth_visible = (
                torch.isfinite(surface_depth)
                & torch.isfinite(primitive_depth)
                & (surface_depth > 0)
                & ((primitive_depth - surface_depth).abs() <= tolerance)
            )
            inside &= depth_visible
    visible_ids = torch.nonzero(inside, as_tuple=False).reshape(-1)
    if visible_ids.numel() == 0:
        return (
            torch.full((keypoint_xy.shape[0],), -1, device=keypoint_xy.device, dtype=torch.long),
            torch.zeros(keypoint_xy.shape[0], device=keypoint_xy.device, dtype=torch.bool),
        )
    visible_uv = projected[visible_ids]
    best_distance = keypoint_xy.new_full((keypoint_xy.shape[0],), torch.inf)
    best_local = torch.full(
        (keypoint_xy.shape[0],), -1, device=keypoint_xy.device, dtype=torch.long
    )
    for start in range(0, visible_uv.shape[0], int(chunk_size)):
        distance = torch.cdist(keypoint_xy, visible_uv[start : start + int(chunk_size)])
        value, local = distance.min(dim=1)
        improve = value < best_distance
        best_distance[improve] = value[improve]
        best_local[improve] = start + local[improve]
    valid = (best_local >= 0) & (best_distance <= float(radius))
    raw_ids = torch.full_like(best_local, -1)
    raw_ids[valid] = landmark_raw_ids[visible_ids[best_local[valid]]]
    return raw_ids, valid


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
    valid_support=None,
    render_pkg=None,
    synthetic=False,
    reprojection_landmark_ids=None,
    reprojection_radius=2.0,
    reprojection_positive_weight=0.75,
    reprojection_depth_abs_tolerance=0.05,
    reprojection_depth_rel_tolerance=0.02,
):
    resolution = get_resolution_from_longest_edge(image.shape[-2], image.shape[-1], longest_edge)
    image = image.cuda()
    valid_feature_mask = None
    if valid_support is not None:
        rgb_valid = valid_support.valid_mask.to(device=image.device)
        if rgb_valid.shape != image.shape[-2:]:
            rgb_valid = F.interpolate(
                rgb_valid.float()[None, None],
                size=image.shape[-2:],
                mode="nearest",
            )[0, 0].bool()
        # Prevent invalid rendered pixels from influencing nearby encoder
        # activations, then strictly remove their feature cells from detection.
        neutral = image.mean(dim=(-2, -1), keepdim=True)
        image = torch.where(rgb_valid[None], image, neutral)
    feature_map = extract_normalized_feature_map(feature_extractor, image, size=resolution)
    height, width = feature_map.shape[-2:]
    if valid_support is not None:
        valid_feature_mask = valid_support.to_feature_mask(
            (height, width), min_valid_fraction=0.75, kind="valid"
        ).to(feature_map.device)
    keypoint_xy, descriptors, detector_scores = _detect_query(
        feature_map, detector, detect_num, nms_radius, valid_feature_mask
    )
    if render_pkg is None or render_pkg["render"].shape[-2:] != (height, width):
        render_pkg = _render_teacher(gaussians, camera, width, height, background)
    labels, primitive_ids, provenance_weights, reliable = _soft_splat_surface_labels(
        group_ids,
        keypoint_xy,
        render_pkg,
        topk=provenance_topk,
    )
    reprojection_ids, reprojection_valid = _nearest_reprojection_labels(
        gaussians.get_xyz,
        reprojection_landmark_ids,
        camera,
        keypoint_xy,
        render_pkg,
        width,
        height,
        reprojection_radius,
        depth_abs_tolerance=reprojection_depth_abs_tolerance,
        depth_rel_tolerance=reprojection_depth_rel_tolerance,
    )
    reprojection_groups = torch.full_like(reprojection_ids, -1)
    if bool(reprojection_valid.any()):
        reprojection_groups[reprojection_valid] = group_ids[
            reprojection_ids[reprojection_valid]
        ]
    # Keep a fixed provenance_topk+1 schema even when no seed reprojects into
    # a synthetic view; real and rendered episodes are concatenated later.
    labels, primitive_ids, provenance_weights = append_reprojection_positive(
        labels,
        primitive_ids,
        provenance_weights,
        reprojection_groups,
        reprojection_ids,
        reprojection_valid,
        positive_weight=reprojection_positive_weight,
    )
    reliable = reliable | reprojection_valid
    return {
        "xy": keypoint_xy[reliable].half().cpu(),
        "descriptors": descriptors[reliable].half().cpu(),
        "scores": detector_scores[reliable].half().cpu(),
        "groups": labels[reliable].int().cpu(),
        "primitive_ids": primitive_ids[reliable].int().cpu(),
        "provenance_weights": provenance_weights[reliable].half().cpu(),
        "query_count": int(keypoint_xy.shape[0]),
        "reliable_count": int(reliable.sum().item()),
        "synthetic": bool(synthetic),
        "valid_mask_fraction": float(
            valid_feature_mask.float().mean().item()
            if valid_feature_mask is not None else 1.0
        ),
        "reprojection_positive_ratio": float(
            reprojection_valid.float().mean() if reprojection_valid.numel() else 0.0
        ),
    }


def _episode_to_device(episode):
    return {
        key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
        for key, value in episode.items()
    }


def _load_landmark_tensor(path):
    with open(path, "rb") as handle:
        value = pickle.load(handle)
    return torch.as_tensor(value, device="cuda", dtype=torch.long).reshape(-1)


@torch.no_grad()
def _evaluate_fixed_probe(
    episodes,
    atom_features,
    active_indices,
    atom_count,
    *,
    min_pool_mass,
    retrieval_topk,
    retrieval_chunk_size,
    max_keypoints=256,
):
    correct = 0
    recalled = 0
    query_count = 0
    active_features = atom_features[active_indices]
    active_mask = torch.zeros(atom_count, device=atom_features.device, dtype=torch.bool)
    active_mask[active_indices] = True
    active_mass_sum = 0.0
    for cached in episodes:
        episode = _episode_to_device(cached)
        query = episode["descriptors"].float()[: int(max_keypoints)]
        groups = episode["groups"].long()[: int(max_keypoints)]
        weights = episode["provenance_weights"].float()[: int(max_keypoints)]
        mass = provenance_mass_partition(groups, weights, active_mask, atom_count)
        valid_query = mass["valid_mask"].any(dim=1) & (
            mass["pool_mass"] >= float(min_pool_mass)
        )
        if not bool(valid_query.any()):
            continue
        query = query[valid_query]
        groups = groups[valid_query]
        valid = mass["valid_mask"][valid_query]
        retrieval = chunked_exact_topk(
            query,
            active_features,
            topk=min(int(retrieval_topk), int(active_indices.numel())),
            chunk_size=int(retrieval_chunk_size),
        )
        retrieved = active_indices[retrieval.indices]
        matches = (retrieved[:, :, None] == groups[:, None, :]) & valid[:, None, :]
        correct += int(matches[:, 0].any(dim=1).sum())
        recalled += int(matches.flatten(1).any(dim=1).sum())
        query_count += int(query.shape[0])
        active_mass_sum += float(mass["active_mass"][valid_query].sum())
    denominator = max(query_count, 1)
    top1_precision = correct / denominator
    recall = recalled / denominator
    return {
        "query_count": query_count,
        "top1_precision": top1_precision,
        "recall_at_k": recall,
        "active_mass": active_mass_sum / denominator,
        "objective": top1_precision + 0.25 * recall,
    }


@torch.no_grad()
def _propose_shadow_swap(utility, active_indices, budget, swap_fraction):
    utility = torch.as_tensor(utility).reshape(-1)
    active_indices = torch.as_tensor(
        active_indices, device=utility.device, dtype=torch.long
    ).reshape(-1)
    budget = min(int(budget), int(utility.numel()))
    active_mask = torch.zeros(utility.numel(), device=utility.device, dtype=torch.bool)
    active_mask[active_indices] = True
    swap_count = min(
        max(1, int(round(budget * float(swap_fraction)))),
        int(active_indices.numel()),
        int((~active_mask).sum()),
    )
    remove = active_indices[torch.topk(utility[active_indices], swap_count, largest=False).indices]
    inactive = torch.nonzero(~active_mask, as_tuple=False).reshape(-1)
    add = inactive[torch.topk(utility[inactive], swap_count, largest=True).indices]
    active_mask[remove] = False
    active_mask[add] = True
    proposed = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
    if proposed.numel() != budget:
        proposed = torch.topk(utility, budget, sorted=False).indices
    return proposed.sort().values, remove, add


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
    active_quality = selection_utility.detach()[
        active_indices.to(selection_utility.device)
    ].float().cpu()
    if active_quality.numel() != landmark_indices.numel():
        raise RuntimeError("active candidate quality must align with active landmarks")
    if landmark_indices.unique().numel() != landmark_indices.numel():
        raise RuntimeError("active landmark raw IDs must be unique")
    artifact_identity = {
        "landmark_indices_sha256": _tensor_sha256(landmark_indices),
        "landmark_features_sha256": _tensor_sha256(active_features),
        "active_candidate_quality_sha256": _tensor_sha256(active_quality),
        "proposal_detector_sha256": config.get("proposal_detector_sha256"),
        "source_geometry_sha256": config.get("source_geometry_sha256"),
        "seed_landmark_sha256": config.get("seed_landmark_sha256"),
        "seed_feature_state_sha256": config.get("seed_feature_state_sha256"),
    }
    state = {
        "version": 30,
        "architecture": "seeded_candidate_aligned_map_refinement_v3",
        "landmark_indices": landmark_indices,
        "landmark_features": active_features,
        "active_atom_indices": active_indices,
        "atom_raw_indices": atom_raw_indices,
        "raw_to_atom": raw_to_atom.detach().int().cpu(),
        "coverage_cell_ids": coverage_cell_ids.detach().int().cpu(),
        "redundancy_group_ids": redundancy_group_ids.detach().int().cpu(),
        "selection_utility": selection_utility.detach().half().cpu(),
        "active_candidate_quality": active_quality.half(),
        "initial_active_features": initial_atom_features[active_indices].half().cpu(),
        "atom_diagnostics": atom_diagnostics,
        "artifact_identity": artifact_identity,
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
                "candidate_quality": state["active_candidate_quality"],
                "artifact_identity": artifact_identity,
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
    all_cameras = scene.getTrainCameras().copy()
    if not all_cameras:
        raise ValueError("V2 coreset training requires training cameras")
    probe_ratio = min(float(args.probe_views) / max(len(all_cameras), 1), 0.5)
    cameras, probe_cameras = split_support_query_cameras(
        all_cameras,
        query_ratio=probe_ratio,
        seed=args.probe_split_seed,
        mode=args.probe_split_mode,
    )
    support_names = {camera.image_name for camera in cameras}
    probe_names_set = {camera.image_name for camera in probe_cameras}
    if support_names & probe_names_set:
        raise RuntimeError("support/probe camera split must be disjoint")
    print(
        "[LaFGS V2] strict camera split: "
        f"support={len(cameras)} probe={len(probe_cameras)} "
        f"mode={args.probe_split_mode} seed={args.probe_split_seed}"
    )
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
    synthetic_mask_builder = None
    if args.synthetic_ratio > 0 and args.synthetic_valid_mask:
        synthetic_mask_builder = NoReferenceValidSupportMaskBuilder(
            NoReferenceValidSupportMaskConfig(
                support_threshold=args.synthetic_support_threshold,
                invalid_min_area=args.synthetic_invalid_min_area,
                invalid_dilate_radius=args.synthetic_invalid_dilate_radius,
                dark_threshold=args.synthetic_dark_threshold,
                bright_threshold=args.synthetic_bright_threshold,
            )
        )

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
    strong_covered = torch.zeros(point_count, device="cuda", dtype=torch.bool)
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
        strong_covered[strong_indices[valid]] = True
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
    seed_raw_indices = None
    if args.seed_landmark_path:
        seed_raw_indices = _load_landmark_tensor(args.seed_landmark_path)
        seed_raw_indices = seed_raw_indices[
            (seed_raw_indices >= 0) & (seed_raw_indices < point_count)
        ].unique(sorted=True)
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
            reprojection_landmark_ids=seed_raw_indices,
            reprojection_radius=args.reprojection_positive_radius,
            reprojection_positive_weight=args.reprojection_positive_weight,
            reprojection_depth_abs_tolerance=args.reprojection_depth_abs_tolerance,
            reprojection_depth_rel_tolerance=args.reprojection_depth_rel_tolerance,
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
    seed_atom_indices = None
    if seed_raw_indices is not None:
        # The proven 16k bank remains a set of exact, non-mergeable primitives.
        # Surface patches form a separate shadow pool used only for proposals.
        patch_atom_count = int(atoms.representative_raw_indices.numel())
        seed_atom_indices = torch.arange(
            patch_atom_count,
            patch_atom_count + seed_raw_indices.numel(),
            device="cuda",
            dtype=torch.long,
        )
        atoms.representative_raw_indices = torch.cat(
            [atoms.representative_raw_indices, seed_raw_indices]
        )
        atoms.identity_patch_ids = torch.cat(
            [atoms.identity_patch_ids, identity_ids[seed_raw_indices]]
        )
        atoms.raw_to_atom = atoms.raw_to_atom.clone()
        atoms.raw_to_atom[seed_raw_indices] = seed_atom_indices
        atom_xyz = gaussians.get_xyz[atoms.representative_raw_indices]
        atom_normals = normals[atoms.representative_raw_indices]
        atoms.coverage_cell_ids, _ = build_surface_groups(
            atom_xyz, voxel_size=args.coverage_voxel_size
        )
        atoms.redundancy_group_ids, _ = build_surface_groups(
            atom_xyz,
            atom_normals,
            voxel_size=args.redundancy_voxel_size,
            normal_bins=args.redundancy_normal_bins,
        )
        atoms.diagnostics["shadow_patch_atom_count"] = patch_atom_count
        atoms.diagnostics["exact_seed_atom_count"] = int(seed_atom_indices.numel())

    atom_raw_indices = atoms.representative_raw_indices.clone()
    atom_count = int(atom_raw_indices.numel())
    atoms.diagnostics["atom_count"] = atom_count
    if atom_count < args.final_budget:
        raise ValueError(
            f"Only {atom_count} observable surface atoms for final budget {args.final_budget}"
        )

    aggregation_weight = observation_count.float() + strong_covered.float()
    atom_initial, atom_feature_weight = aggregate_atom_features(
        raw_initial,
        atoms.raw_to_atom,
        atom_count,
        aggregation_weight,
        chunk_size=args.mvinit_chunk_size,
    )
    empty_atoms = atom_feature_weight <= 0
    if bool(empty_atoms.any()):
        atom_initial[empty_atoms] = F.normalize(
            raw_initial[atom_raw_indices[empty_atoms]], dim=1
        )
    seed_feature_coverage = 0
    if args.seed_feature_state:
        seed_state = torch.load(args.seed_feature_state, map_location="cpu")
        state_indices = torch.as_tensor(
            seed_state["landmark_indices"], device="cuda", dtype=torch.long
        )
        state_features = F.normalize(
            torch.as_tensor(seed_state["landmark_features"], device="cuda").float(),
            dim=1,
        )
        valid = (state_indices >= 0) & (state_indices < point_count)
        mapped = atoms.raw_to_atom[state_indices[valid]]
        valid_mapped = mapped >= 0
        mapped = mapped[valid_mapped]
        state_features = state_features[valid][valid_mapped]
        feature_sum = atom_initial.new_zeros(atom_initial.shape)
        feature_count = atom_initial.new_zeros(atom_count)
        feature_sum.index_add_(0, mapped, state_features)
        feature_count.index_add_(0, mapped, torch.ones_like(mapped, dtype=torch.float32))
        covered = feature_count > 0
        atom_initial[covered] = F.normalize(
            feature_sum[covered] / feature_count[covered, None], dim=1
        )
        seed_feature_coverage = int(covered.sum())
    atoms.diagnostics.update(
        {
            "descriptor_initialization": "weighted_patch_aggregation",
            "strong_feature_raw_coverage": float(strong_covered.float().mean()),
            "strong_feature_atom_coverage": float(
                (strong_covered[atom_raw_indices]).float().mean()
            ),
            "seed_landmark_count": int(seed_raw_indices.numel()) if seed_raw_indices is not None else 0,
            "seed_atom_count": int(seed_atom_indices.numel()) if seed_atom_indices is not None else 0,
            "seed_feature_atom_count": seed_feature_coverage,
        }
    )
    atom_features = torch.nn.Parameter(atom_initial.clone(), requires_grad=True)
    seed_initial_features = (
        atom_initial[seed_atom_indices].clone()
        if seed_atom_indices is not None else None
    )
    initial_cpu = atom_initial.half().cpu()
    descriptor_optimizer = torch.optim.SGD([atom_features], lr=args.descriptor_lr)
    schedule = make_gradual_budget_schedule(
        atom_count if args.selection_mode == "progressive" else args.final_budget,
        args.iterations,
        args.final_budget,
        keep_ratio=args.stage_keep_ratio,
        warmup_ratio=args.warmup_ratio,
    )
    if args.selection_mode == "shadow_swap":
        if seed_atom_indices is None or seed_atom_indices.numel() == 0:
            raise ValueError("shadow_swap requires --seed_landmark_path")
        active_indices = seed_atom_indices.unique(sorted=True)
        if active_indices.numel() > args.final_budget:
            active_indices = active_indices[: args.final_budget]
        if active_indices.numel() < args.final_budget:
            seed_mask = torch.zeros(atom_count, device="cuda", dtype=torch.bool)
            seed_mask[active_indices] = True
            atom_query_priority = identity_priority[atoms.identity_patch_ids]
            fill_score = atom_query_priority.masked_fill(seed_mask, -torch.inf)
            fill = torch.topk(
                fill_score, args.final_budget - active_indices.numel(), sorted=False
            ).indices
            active_indices = torch.cat([active_indices, fill]).sort().values
    else:
        active_indices = torch.arange(atom_count, device="cuda", dtype=torch.long)
    previous_active = None
    active_mask = torch.zeros(atom_count, device="cuda", dtype=torch.bool)
    active_mask[active_indices] = True
    if int(active_mask.sum()) != int(active_indices.numel()):
        raise RuntimeError("active_mask and active_indices disagree at initialization")
    observation_utility = torch.zeros(atom_count, device="cuda")
    match_utility = torch.zeros(atom_count, device="cuda")
    negative_risk = torch.zeros(atom_count, device="cuda")
    rescue_utility = torch.zeros(atom_count, device="cuda")
    selection_utility = torch.zeros(atom_count, device="cuda")
    coverage_count = int(atoms.coverage_cell_ids.max().item()) + 1
    coverage_priority = torch.zeros(coverage_count, device="cuda")
    episode_cache = {}
    raw_to_atom_cpu = atoms.raw_to_atom.long().cpu()
    for image_name, episode in discovery_cache.items():
        primitive_group = episode["primitive_ids"].long()
        valid_primitive = (primitive_group >= 0) & (
            primitive_group < raw_to_atom_cpu.numel()
        )
        atom_group = torch.full_like(primitive_group, -1)
        atom_group[valid_primitive] = raw_to_atom_cpu[
            primitive_group[valid_primitive]
        ]
        converted = dict(episode)
        converted["groups"] = atom_group.int()
        episode_cache[image_name] = converted
    probe_episodes = []
    for camera in tqdm(probe_cameras, desc="V2 held-out probe cache"):
        probe_episodes.append(
            _build_episode(
                gaussians,
                camera,
                camera.original_image,
                feature_extractor,
                detector,
                atoms.raw_to_atom,
                background,
                args.detect_num,
                args.nms_radius,
                dataset.longest_edge,
                args.provenance_topk,
                reprojection_landmark_ids=seed_raw_indices,
                reprojection_radius=args.reprojection_positive_radius,
                reprojection_positive_weight=args.reprojection_positive_weight,
                reprojection_depth_abs_tolerance=args.reprojection_depth_abs_tolerance,
                reprojection_depth_rel_tolerance=args.reprojection_depth_rel_tolerance,
            )
        )
    history = []
    running = {}
    stage_budget = int(active_indices.numel())
    selection_diagnostics = {}
    probe_diagnostics = _evaluate_fixed_probe(
        probe_episodes,
        atom_features,
        active_indices,
        atom_count,
        min_pool_mass=args.min_pool_mass,
        retrieval_topk=args.retrieval_topk,
        retrieval_chunk_size=args.retrieval_chunk_size,
        max_keypoints=args.probe_keypoints,
    )
    probe_diagnostics.update({"accepted": True, "reason": "initial"})
    start_time = time.time()
    del mv_result, raw_initial, atom_initial
    torch.cuda.empty_cache()

    progress = tqdm(range(1, args.iterations + 1), desc="LaFGS V2 Coreset")
    for iteration in progress:
        target_budget = (
            schedule.budget(iteration)
            if args.selection_mode == "progressive" else args.final_budget
        )
        warmup_iteration = max(1, int(round(args.iterations * args.warmup_ratio)))
        refresh = (
            iteration == 1
            if args.selection_mode == "progressive"
            else False
        ) or (
            args.selection_mode == "progressive"
            and (
                target_budget != stage_budget
                or (target_budget < atom_count and iteration % args.selection_interval == 0)
            )
        ) or (
            args.selection_mode == "shadow_swap"
            and iteration >= warmup_iteration
            and iteration % args.selection_interval == 0
        )
        if refresh:
            previous_active = active_indices
            selection_utility = (
                args.match_utility_weight * match_utility
                + args.rescue_utility_weight * rescue_utility
                + args.observation_utility_weight * torch.log1p(observation_utility)
                - args.negative_risk_weight * negative_risk
            )
            if args.selection_mode == "shadow_swap":
                proposed, removed, added = _propose_shadow_swap(
                    selection_utility,
                    active_indices,
                    args.final_budget,
                    args.swap_fraction,
                )
                current_probe = _evaluate_fixed_probe(
                    probe_episodes,
                    atom_features,
                    active_indices,
                    atom_count,
                    min_pool_mass=args.min_pool_mass,
                    retrieval_topk=args.retrieval_topk,
                    retrieval_chunk_size=args.retrieval_chunk_size,
                    max_keypoints=args.probe_keypoints,
                )
                proposed_probe = _evaluate_fixed_probe(
                    probe_episodes,
                    atom_features,
                    proposed,
                    atom_count,
                    min_pool_mass=args.min_pool_mass,
                    retrieval_topk=args.retrieval_topk,
                    retrieval_chunk_size=args.retrieval_chunk_size,
                    max_keypoints=args.probe_keypoints,
                )
                accepted = proposed_probe["objective"] >= (
                    current_probe["objective"] + args.swap_min_improvement
                )
                if accepted:
                    active_indices = proposed
                    probe_diagnostics = proposed_probe
                else:
                    probe_diagnostics = current_probe
                probe_diagnostics.update(
                    {
                        "accepted": bool(accepted),
                        "proposed_objective": proposed_probe["objective"],
                        "removed_count": int(removed.numel()),
                        "added_count": int(added.numel()),
                    }
                )
                selection_diagnostics = {
                    "coverage_reserved_count": 0,
                    "utility_fill_count": int(active_indices.numel()),
                    "coverage_reserved_fraction": 0.0,
                    "swap_accepted": bool(accepted),
                }
            else:
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
                if target_budget != stage_budget:
                    probe_diagnostics = _evaluate_fixed_probe(
                        probe_episodes,
                        atom_features,
                        active_indices,
                        atom_count,
                        min_pool_mass=args.min_pool_mass,
                        retrieval_topk=args.retrieval_topk,
                        retrieval_chunk_size=args.retrieval_chunk_size,
                        max_keypoints=args.probe_keypoints,
                    )
                    probe_diagnostics.update({"accepted": True, "reason": "progressive_stage"})
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
                reprojection_landmark_ids=seed_raw_indices,
                reprojection_radius=args.reprojection_positive_radius,
                reprojection_positive_weight=args.reprojection_positive_weight,
                reprojection_depth_abs_tolerance=args.reprojection_depth_abs_tolerance,
                reprojection_depth_rel_tolerance=args.reprojection_depth_rel_tolerance,
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
            valid_support = (
                synthetic_mask_builder.build(pkg["render"])
                if synthetic_mask_builder is not None else None
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
                        valid_support=valid_support,
                        synthetic=True,
                        reprojection_landmark_ids=seed_raw_indices,
                        reprojection_radius=args.reprojection_positive_radius,
                        reprojection_positive_weight=args.reprojection_positive_weight,
                        reprojection_depth_abs_tolerance=args.reprojection_depth_abs_tolerance,
                        reprojection_depth_rel_tolerance=args.reprojection_depth_rel_tolerance,
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
        mass = provenance_mass_partition(
            target_groups, target_weights, active_mask, atom_count
        )
        valid_group = mass["valid_mask"]
        pool_covered_mass = mass["pool_mass"]
        pool_miss = mass["missing_mass"]
        valid_query = (
            valid_group.any(dim=1)
            & (target_weights.sum(dim=1) > 0)
            & (pool_covered_mass >= float(args.min_pool_mass))
        )
        pool_ignored_ratio = 1.0 - valid_query.float().mean()
        query = query[valid_query]
        target_groups = target_groups[valid_query]
        target_primitives = target_primitives[valid_query]
        target_weights = target_weights[valid_query]
        valid_group = valid_group[valid_query]
        target_weights = target_weights.masked_fill(~valid_group, 0.0)
        # Keep absolute provenance mass. Renormalizing a partial 10% contributor
        # to a 100% positive was the main V2.1 supervision bug.
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
            shadow_mass = (target_weights * (valid_group & ~positive_active)).sum(dim=1)
            coverage_miss = (target_weights.sum(dim=1) - covered_mass).clamp_min(0.0)
            top1 = negative_indices[:, 0]
            top1_correct = (
                (top1[:, None] == target_groups) & valid_group
            ).any(dim=1)

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
            if args.freeze_seed_features and seed_atom_indices is not None:
                atom_features[seed_atom_indices] = seed_initial_features
            decay = float(args.utility_decay)
            observation_utility.mul_(decay)
            match_utility.mul_(decay)
            negative_risk.mul_(decay)
            rescue_utility.mul_(decay)
            flat_atoms = target_groups[valid_group]
            flat_weights = target_weights[valid_group]
            observation_utility.scatter_add_(
                0, flat_atoms, (1.0 - decay) * flat_weights
            )
            # Selection is driven by actual retrieval outcomes. Correct top-1
            # atoms earn utility; false top-1 atoms accumulate explicit risk;
            # missed queries promote their strongest shadow positive.
            match_utility.scatter_add_(
                0,
                top1[top1_correct],
                torch.full_like(top1[top1_correct], 1.0 - decay, dtype=torch.float32),
            )
            negative_risk.scatter_add_(
                0,
                top1[~top1_correct],
                torch.full_like(top1[~top1_correct], 1.0 - decay, dtype=torch.float32),
            )
            strongest_positive = target_weights.masked_fill(~valid_group, -1.0).argmax(dim=1)
            rescue_atoms = target_groups.gather(1, strongest_positive[:, None]).squeeze(1)
            needs_rescue = ~recall
            rescue_utility.scatter_add_(
                0,
                rescue_atoms[needs_rescue],
                torch.full_like(rescue_atoms[needs_rescue], 1.0 - decay, dtype=torch.float32),
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
            "shadow_mass": float(shadow_mass.mean()),
            "atom_pool_miss": float(pool_miss.mean()),
            "pool_ignored_ratio": float(pool_ignored_ratio),
            "top1_precision": float(top1_correct.float().mean()),
            "provenance_atom_distance_mean_m": float(mapping_distance.mean()),
            "provenance_atom_distance_p95_m": float(torch.quantile(mapping_distance, 0.95)),
            "provenance_atom_normal_angle_mean_deg": float(mapping_normal_angle.mean()),
            "selection_utility_mean": float(selection_utility.mean()),
            "reliable_query_ratio": sum(x["reliable_count"] for x in episodes)
            / max(sum(x["query_count"] for x in episodes), 1),
            "synthetic": float(add_synthetic),
            "synthetic_valid_fraction": float(
                sum(x["valid_mask_fraction"] for x in episodes if x["synthetic"])
                / max(sum(1 for x in episodes if x["synthetic"]), 1)
            ) if add_synthetic else 1.0,
            "reprojection_positive_ratio": float(
                sum(x["reprojection_positive_ratio"] for x in episodes)
                / max(len(episodes), 1)
            ),
            "synthetic_reprojection_positive_ratio": float(
                sum(
                    x["reprojection_positive_ratio"]
                    for x in episodes
                    if x["synthetic"]
                )
                / max(sum(1 for x in episodes if x["synthetic"]), 1)
            ) if add_synthetic else 0.0,
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
                **{f"probe_{key}": value for key, value in probe_diagnostics.items()},
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
        + args.rescue_utility_weight * rescue_utility
        + args.observation_utility_weight * torch.log1p(observation_utility)
        - args.negative_risk_weight * negative_risk
    )
    if args.selection_mode == "shadow_swap":
        final_active = active_indices
    else:
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
    source_geometry_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{args.iteration}",
        "point_cloud.ply",
    )
    config.update(
        {
            "point_count": point_count,
            "atom_count": atom_count,
            "schedule_budgets": list(schedule.budgets),
            "schedule_boundaries": list(schedule.boundaries),
            "mvinit_observed_count": int((~unobserved).sum().item()),
            "support_camera_names": sorted(support_names),
            "probe_camera_names": sorted(probe_names_set),
            "proposal_detector_sha256": _file_sha256(args.proposal_detector),
            "source_geometry_sha256": _file_sha256(source_geometry_path),
            "seed_landmark_sha256": _file_sha256(args.seed_landmark_path),
            "seed_feature_state_sha256": _file_sha256(args.seed_feature_state),
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
    parser.add_argument("--seed_landmark_path", default="")
    parser.add_argument("--seed_feature_state", default="")
    parser.add_argument(
        "--freeze_seed_features", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--selection_mode", choices=("progressive", "shadow_swap"), default="shadow_swap"
    )
    parser.add_argument("--final_budget", type=int, default=16384)
    parser.add_argument("--identity_voxel_size", type=float, default=0.02)
    parser.add_argument("--identity_normal_bins", type=int, default=8)
    parser.add_argument("--coverage_voxel_size", type=float, default=0.20)
    parser.add_argument("--redundancy_voxel_size", type=float, default=0.05)
    parser.add_argument("--redundancy_normal_bins", type=int, default=4)
    parser.add_argument(
        "--max_atoms", type=int, default=0,
        help="Maximum atom-pool size; 0 keeps every observable identity patch",
    )
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
    parser.add_argument("--swap_fraction", type=float, default=0.01)
    parser.add_argument("--swap_min_improvement", type=float, default=0.0)
    parser.add_argument("--probe_views", type=int, default=8)
    parser.add_argument("--probe_keypoints", type=int, default=256)
    parser.add_argument(
        "--probe_split_mode",
        choices=("random", "sequence_block", "temporal_block"),
        default="temporal_block",
    )
    parser.add_argument("--probe_split_seed", type=int, default=2026)
    parser.add_argument("--hysteresis", type=float, default=0.05)
    parser.add_argument("--stage_keep_ratio", type=float, default=0.75)
    parser.add_argument("--warmup_ratio", type=float, default=0.20)
    parser.add_argument("--coverage_fraction", type=float, default=0.50)
    parser.add_argument("--selection_redundancy_penalty", type=float, default=0.10)
    parser.add_argument("--utility_decay", type=float, default=0.995)
    parser.add_argument("--match_utility_weight", type=float, default=1.0)
    parser.add_argument("--observation_utility_weight", type=float, default=0.1)
    parser.add_argument("--negative_risk_weight", type=float, default=0.5)
    parser.add_argument("--rescue_utility_weight", type=float, default=1.0)
    parser.add_argument("--descriptor_lr", type=float, default=2e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--miss_rejection_weight", type=float, default=0.25)
    parser.add_argument("--miss_negative_margin", type=float, default=0.40)
    parser.add_argument("--min_pool_mass", type=float, default=0.7)
    parser.add_argument("--reprojection_positive_radius", type=float, default=2.0)
    parser.add_argument("--reprojection_positive_weight", type=float, default=0.75)
    parser.add_argument("--reprojection_depth_abs_tolerance", type=float, default=0.05)
    parser.add_argument("--reprojection_depth_rel_tolerance", type=float, default=0.02)
    parser.add_argument("--trust_weight", type=float, default=0.01)
    parser.add_argument("--synthetic_ratio", type=float, default=0.0)
    parser.add_argument("--synthetic_valid_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synthetic_support_threshold", type=float, default=0.22)
    parser.add_argument("--synthetic_invalid_min_area", type=int, default=96)
    parser.add_argument("--synthetic_invalid_dilate_radius", type=int, default=1)
    parser.add_argument("--synthetic_dark_threshold", type=float, default=0.035)
    parser.add_argument("--synthetic_bright_threshold", type=float, default=0.985)
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
