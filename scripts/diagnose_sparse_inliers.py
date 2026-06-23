#!/usr/bin/env python
import argparse
import itertools
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render_from_pose_gsplat
from localization_training.descriptor_diagnostics import calibrate_landmark_quality, summarize_landmark_value
from localization_training.direct_landmark_teacher import (
    _limit_valid_indices,
    filter_depth_consistent_landmarks,
    make_intrinsics_from_fov,
    project_landmarks_to_query,
)
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from scene.kpdetector import simple_nms
from stdloc import (
    STDLoc,
    dual_softmax,
    get_intrinsic,
    landmark_prior_from_meta,
    mnn_match,
    resolve_artifact_path,
    topk_match,
    validate_sampled_indices,
)
from utils.image_utils import get_resolution_from_longest_edge
from utils.pose_utils import cal_pose_error, solve_pose


LEVEL3_METRIC_KEYS = (
    "spearman_utility_inlier_rate",
    "spearman_utility_correct_rate",
    "top_quartile_inlier_rate",
    "bottom_quartile_inlier_rate",
    "spearman_calibrated_inlier_rate",
    "spearman_calibrated_correct_rate",
)


def _limit_cameras(cameras, max_images):
    if max_images is None:
        return list(cameras)
    return list(itertools.islice(cameras, max_images))


def _squeeze_scalar_map(value):
    if value is None:
        return None
    value = value.squeeze()
    if value.dim() == 3:
        value = value[..., 0]
    return value


def _load_landmark_indices(config, point_count):
    sparse = config["sparse"]
    path = resolve_artifact_path(
        config["model_path"],
        sparse["landmark_path"],
        sparse.get("landmark_model_path"),
    )
    with open(path, "rb") as f:
        sampled_idx = pickle.load(f)
    return validate_sampled_indices(sampled_idx, point_count).cpu()


def _apply_sparse_overrides(config, args):
    sparse = config.setdefault("sparse", {})
    for key in (
        "detector_path",
        "detector_model_path",
        "landmark_path",
        "landmark_model_path",
        "landmark_meta_path",
        "landmark_meta_model_path",
    ):
        value = getattr(args, key, None)
        if value is not None:
            sparse[key] = value
    reprojection_error = getattr(args, "reprojection_error", None)
    if reprojection_error is not None:
        sparse["reprojection_error"] = float(reprojection_error)
    use_landmark_prior = getattr(args, "use_landmark_prior", None)
    if use_landmark_prior is not None:
        sparse["use_landmark_prior"] = bool(use_landmark_prior)
    sparse["sparse_only"] = True
    return config


def _load_stdloc_config(dataset, args):
    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    config = _apply_sparse_overrides(config, args)
    config["dense"]["norm_before_render"] = dataset.norm_before_render
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    return config


def _load_gaussians(dataset):
    if dataset.gaussian_type == "3dgs":
        return GaussianModel(dataset.sh_degree)
    if dataset.gaussian_type == "2dgs":
        return GaussianModel_2dgs(dataset.sh_degree)
    raise ValueError(f"Unsupported gaussian_type: {dataset.gaussian_type}")


def _compute_utility(gaussians, config, landmark_indices, source, min_observations):
    point_count = gaussians.get_xyz.shape[0]
    if source == "localization":
        return gaussians.compute_localization_utility(min_observations).detach().cpu()
    if source == "reliability":
        return gaussians.compute_landmark_reliability(min_observations).detach().cpu()
    if source == "geometry":
        return gaussians.compute_pose_geometry_value(min_observations).detach().cpu()
    if source == "meta":
        utility = torch.zeros(point_count, dtype=torch.float32)
        sparse = config["sparse"]
        meta_path = resolve_artifact_path(
            config["model_path"],
            sparse.get("landmark_meta_path", "detector/landmark_meta.pt"),
            sparse.get("landmark_meta_model_path", sparse.get("landmark_model_path")),
        )
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            value = meta.get("score", meta.get("utility", None))
            if value is not None:
                value = value.detach().cpu().float().reshape(-1)
                count = min(value.numel(), landmark_indices.numel())
                utility[landmark_indices[:count]] = value[:count]
        return utility
    raise ValueError(f"Unsupported utility source: {source}")


def _collect_visible_landmarks(
    gaussians,
    pose_gt_w2c,
    fovx,
    fovy,
    landmark_indices,
    height,
    width,
    target_depth=None,
    target_alpha=None,
    alpha_threshold=0.2,
    depth_abs_tolerance=1e-3,
    depth_rel_tolerance=0.01,
    max_landmarks=None,
):
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    landmark_indices_cuda = landmark_indices.to(device=device, dtype=torch.long)
    xyz = gaussians.get_xyz[landmark_indices_cuda]
    K = make_intrinsics_from_fov(fovx, fovy, width, height, device=device, dtype=dtype)
    uv, depth, valid = project_landmarks_to_query(
        xyz,
        K,
        pose_gt_w2c.to(device=device, dtype=dtype),
        height,
        width,
    )
    valid = filter_depth_consistent_landmarks(
        uv,
        depth,
        valid,
        target_depth=target_depth,
        target_alpha=target_alpha,
        alpha_threshold=alpha_threshold,
        abs_tolerance=depth_abs_tolerance,
        rel_tolerance=depth_rel_tolerance,
    )
    keep = _limit_valid_indices(valid, max_landmarks)
    return landmark_indices_cuda[keep].detach().cpu()


def _detect_and_match(stdloc, query_feature_map):
    sparse = stdloc.config["sparse"]
    height, width = query_feature_map.shape[-2:]
    heat_map = stdloc.detector(query_feature_map)
    kp_scores_after_nms = simple_nms(heat_map, sparse.get("nms", 4)).flatten()
    detect_num = min(int(sparse.get("detect_num", 2048)), kp_scores_after_nms.numel())
    if detect_num <= 0:
        return None
    _, kp_ids = torch.topk(kp_scores_after_nms, detect_num)
    pos_mask = kp_scores_after_nms > 0
    kp_ids = kp_ids[pos_mask[kp_ids]]
    if kp_ids.numel() == 0:
        return None

    kp_mask = torch.zeros_like(kp_scores_after_nms, dtype=torch.bool)
    kp_mask[kp_ids] = True
    sampled_features = query_feature_map.reshape(query_feature_map.shape[0], -1)[:, kp_mask]
    landmark_features = F.normalize(stdloc.landmarks.get_loc_feature.squeeze(), dim=-1)
    corr_matrix = torch.matmul(sampled_features.T, landmark_features.T)

    if sparse["dual_softmax"] is True:
        corr_matrix = dual_softmax(corr_matrix=corr_matrix, temp=sparse["dual_softmax_temp"])

    if stdloc.landmark_meta is not None and sparse.get("use_landmark_prior", False):
        prior = landmark_prior_from_meta(
            stdloc.landmark_meta,
            landmark_features.shape[0],
            sampled_indices=getattr(stdloc, "landmark_indices", None),
        )
        if prior is not None:
            prior = prior.to(corr_matrix.device, dtype=corr_matrix.dtype)
            prior = (prior - prior.mean()) / prior.std().clamp_min(1e-6)
            corr_matrix = corr_matrix + sparse.get("landmark_prior_weight", 0.05) * prior[None]

    if sparse["mnn_match"] is True:
        _, im_idx, gs_ids = mnn_match(corr_matrix[None], thr=sparse["threshold"])
        val = corr_matrix[im_idx.reshape(-1), gs_ids.reshape(-1)] if im_idx.numel() else corr_matrix.new_empty(0)
    else:
        im_idx, gs_ids, val = topk_match(corr_matrix[None], sparse["topk"], thr=sparse["threshold"])

    im_idx = im_idx.reshape(-1).to(device=query_feature_map.device, dtype=torch.long)
    gs_ids = gs_ids.reshape(-1).to(device=query_feature_map.device, dtype=torch.long)
    val = val.reshape(-1).to(device=query_feature_map.device)
    if im_idx.numel() == 0:
        return None

    pixel_ids = torch.arange(height * width, device=query_feature_map.device)
    pixel_grid = torch.stack([pixel_ids % width, pixel_ids // width], dim=1).float()
    query_p2d = pixel_grid[kp_mask][im_idx]
    return {
        "query_p2d": query_p2d,
        "local_landmark_ids": gs_ids,
        "scores": val,
        "detected_keypoints": int(kp_ids.numel()),
    }


def _match_reprojection_correct(points3d, query_p2d, pose_gt_w2c, fovx, fovy, height, width, threshold):
    if query_p2d.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=query_p2d.device)
    K = make_intrinsics_from_fov(fovx, fovy, width, height, device=query_p2d.device, dtype=query_p2d.dtype)
    uv, _, valid = project_landmarks_to_query(
        points3d.to(device=query_p2d.device, dtype=query_p2d.dtype),
        K,
        pose_gt_w2c.to(device=query_p2d.device, dtype=query_p2d.dtype),
        height,
        width,
    )
    error = torch.linalg.norm((query_p2d + 0.5) - uv, dim=1)
    return valid & torch.isfinite(error) & (error <= float(threshold))


def _add_counts(counter, indices, weights=None):
    indices = indices.reshape(-1).cpu().long()
    if indices.numel() == 0:
        return
    if weights is None:
        weights = torch.ones(indices.numel(), dtype=counter.dtype)
    else:
        weights = weights.reshape(-1).cpu().to(dtype=counter.dtype)
    counter.index_add_(0, indices, weights)


def _calibration_image_roles(num_images, mode="none", eval_fraction=0.0):
    num_images = int(num_images)
    if num_images <= 0:
        return []
    mode = str(mode or "none")
    eval_fraction = float(eval_fraction)
    if mode == "none" or eval_fraction <= 0.0 or num_images < 2:
        return ["train"] * num_images

    eval_count = max(1, int(round(num_images * min(max(eval_fraction, 0.0), 1.0))))
    eval_count = min(eval_count, num_images - 1)
    if mode == "block":
        split = num_images - eval_count
        return ["train" if idx < split else "eval" for idx in range(num_images)]
    if mode == "alternate":
        period = max(2, int(round(1.0 / min(max(eval_fraction, 1e-6), 1.0))))
        roles = ["eval" if (idx + 1) % period == 0 else "train" for idx in range(num_images)]
        if "eval" not in roles:
            roles[-1] = "eval"
        if "train" not in roles:
            roles[0] = "train"
        return roles
    raise ValueError(f"Unsupported calibration split mode: {mode}")


def _calibration_target_counts(target, visible_count, matched_count, correct_count, inlier_count):
    target = str(target)
    if target == "correct":
        return correct_count, visible_count, visible_count > 0
    if target == "inlier":
        return inlier_count, matched_count, matched_count > 0
    raise ValueError(f"Unsupported calibration target: {target}")


def _new_count_buffers(point_count):
    return {
        "visible": torch.zeros(point_count, dtype=torch.long),
        "matched": torch.zeros(point_count, dtype=torch.long),
        "correct": torch.zeros(point_count, dtype=torch.long),
        "inlier": torch.zeros(point_count, dtype=torch.long),
    }


def _write_per_landmark_stats(path, landmark_indices, visible_count, matched_count, correct_count, inlier_count, utility):
    active = (visible_count > 0) | (matched_count > 0) | (utility.abs() > 0)
    rows = []
    for idx in torch.nonzero(active, as_tuple=False).squeeze(1).tolist():
        rows.append(
            {
                "landmark_id": int(idx),
                "is_sampled_landmark": bool((landmark_indices == idx).any().item()),
                "visible_count": int(visible_count[idx].item()),
                "matched_count": int(matched_count[idx].item()),
                "correct_count": int(correct_count[idx].item()),
                "inlier_count": int(inlier_count[idx].item()),
                "utility": float(utility[idx].item()),
            }
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
        f.write("\n")


def _calibration_features(gaussians, utility):
    gaussians._ensure_localization_state()
    return {
        "utility": utility.detach().cpu().float(),
        "repeatability": gaussians.loc_repeatability_ema.detach().cpu().float(),
        "positive_prob": gaussians.loc_positive_prob_ema.detach().cpu().float(),
        "margin": gaussians.loc_margin_ema.detach().cpu().float(),
        "entropy": gaussians.loc_entropy_ema.detach().cpu().float(),
        "outlier": gaussians.loc_outlier_ema.detach().cpu().float(),
        "information": gaussians.loc_information_ema.detach().cpu().float(),
        "reprojection_error": gaussians.loc_reproj_error_ema.detach().cpu().float(),
        "redundancy": gaussians.loc_redundancy_ema.detach().cpu().float(),
    }


def _write_calibrated_utility(path, landmark_indices, calibration):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    score = calibration["score"].detach().cpu().float()
    payload = {
        "score": score[landmark_indices].clone(),
        "full_score": score,
        "landmark_indices": landmark_indices.detach().cpu().long(),
        "feature_names": list(calibration["feature_names"]),
        "weights": calibration["weights"].detach().cpu().float(),
        "bias": float(calibration["bias"]),
        "calibration_target": calibration.get("calibration_target_name", "inlier"),
        "calibration_heldout": bool(calibration.get("calibration_heldout", False)),
    }
    torch.save(payload, path)


def diagnose(args, dataset, scene, gaussians):
    config = _load_stdloc_config(dataset, args)
    stdloc = STDLoc(gaussians, config)
    landmark_indices = _load_landmark_indices(config, gaussians.get_xyz.shape[0])
    point_count = gaussians.get_xyz.shape[0]

    visible_count = torch.zeros(point_count, dtype=torch.long)
    matched_count = torch.zeros(point_count, dtype=torch.long)
    correct_count = torch.zeros(point_count, dtype=torch.long)
    inlier_count = torch.zeros(point_count, dtype=torch.long)
    train_counts = _new_count_buffers(point_count)
    eval_counts = _new_count_buffers(point_count)
    utility = _compute_utility(
        gaussians,
        config,
        landmark_indices,
        args.utility_source,
        args.min_observations,
    )

    cameras = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    cameras = _limit_cameras(cameras, args.max_images)
    calibration_roles = _calibration_image_roles(
        len(cameras),
        mode=args.calibration_split_mode,
        eval_fraction=args.calibration_eval_fraction,
    )
    calibration_heldout = "eval" in calibration_roles
    correct_reprojection_error = getattr(args, "correct_reprojection_error", None)
    correct_threshold = (
        float(correct_reprojection_error)
        if correct_reprojection_error is not None
        else float(config["sparse"]["reprojection_error"])
    )

    image_summaries = []
    sparse_aes = []
    sparse_tes = []
    match_counts = []
    correct_counts = []
    inlier_counts = []
    for image_idx, camera in enumerate(tqdm(cameras, desc="Sparse inlier diagnostics")):
        calibration_role = calibration_roles[image_idx] if image_idx < len(calibration_roles) else "train"
        role_counts = eval_counts if calibration_role == "eval" else train_counts
        gt_w2c = camera.world_view_transform.transpose(0, 1).cuda()
        query_image = camera.original_image.cuda()
        fine_feature_map, _ = stdloc.get_feature_map(query_image)
        height, width = fine_feature_map.shape[-2:]

        target_depth = None
        target_alpha = None
        if args.depth_check:
            with torch.no_grad():
                rendered = render_from_pose_gsplat(
                    gaussians,
                    gt_w2c,
                    camera.FoVx,
                    camera.FoVy,
                    width,
                    height,
                    rgb_only=True,
                    render_mode="RGB+ED",
                )
            target_depth = _squeeze_scalar_map(rendered.get("depth"))
            target_alpha = _squeeze_scalar_map(rendered.get("alphas"))

        max_visible = args.max_visible_landmarks_per_image if args.max_visible_landmarks_per_image > 0 else None
        visible = _collect_visible_landmarks(
            gaussians,
            gt_w2c,
            camera.FoVx,
            camera.FoVy,
            landmark_indices,
            height,
            width,
            target_depth=target_depth,
            target_alpha=target_alpha,
            alpha_threshold=args.alpha_threshold,
            depth_abs_tolerance=args.depth_abs_tolerance,
            depth_rel_tolerance=args.depth_rel_tolerance,
            max_landmarks=max_visible,
        )
        _add_counts(visible_count, visible)
        _add_counts(role_counts["visible"], visible)

        matches = _detect_and_match(stdloc, fine_feature_map)
        if matches is None:
            image_summaries.append(
                {
                    "image_name": camera.image_name,
                    "visible_landmarks": int(visible.numel()),
                    "detected_keypoints": 0,
                    "matches": 0,
                    "correct_matches": 0,
                    "inliers": 0,
                    "sparse_AE": None,
                    "sparse_TE": None,
                }
            )
            match_counts.append(0)
            correct_counts.append(0)
            inlier_counts.append(0)
            continue

        local_landmark_ids = matches["local_landmark_ids"]
        full_landmark_ids = landmark_indices.to(device=local_landmark_ids.device)[local_landmark_ids]
        points3d = stdloc.landmarks.get_xyz[local_landmark_ids]
        query_p2d = matches["query_p2d"]
        correct_mask = _match_reprojection_correct(
            points3d,
            query_p2d,
            gt_w2c,
            camera.FoVx,
            camera.FoVy,
            height,
            width,
            correct_threshold,
        )

        K = get_intrinsic(camera.FoVx, camera.FoVy, width, height)
        pose_w2c, inliers = solve_pose(
            (query_p2d + 0.5).detach().cpu().numpy(),
            points3d.detach().cpu().numpy(),
            K,
            config["sparse"]["solver"],
            config["sparse"]["reprojection_error"],
            config["sparse"]["confidence"],
            config["sparse"]["max_iterations"],
            config["sparse"]["min_iterations"],
        )
        inliers = np.asarray(inliers).reshape(-1).astype(np.int64)
        inlier_mask = torch.zeros(full_landmark_ids.numel(), dtype=torch.bool)
        if inliers.size > 0:
            valid_inliers = inliers[(inliers >= 0) & (inliers < full_landmark_ids.numel())]
            inlier_mask[torch.as_tensor(valid_inliers, dtype=torch.long)] = True

        full_landmark_ids_cpu = full_landmark_ids.detach().cpu()
        _add_counts(matched_count, full_landmark_ids_cpu)
        _add_counts(correct_count, full_landmark_ids_cpu[correct_mask.detach().cpu()])
        _add_counts(inlier_count, full_landmark_ids_cpu[inlier_mask])
        _add_counts(role_counts["matched"], full_landmark_ids_cpu)
        _add_counts(role_counts["correct"], full_landmark_ids_cpu[correct_mask.detach().cpu()])
        _add_counts(role_counts["inlier"], full_landmark_ids_cpu[inlier_mask])

        gt_w2c_np = gt_w2c.detach().cpu().numpy()
        ae, te = cal_pose_error(pose_w2c, gt_w2c_np)
        sparse_aes.append(float(ae))
        sparse_tes.append(float(te))
        match_count = int(full_landmark_ids.numel())
        correct_count_image = int(correct_mask.sum().item())
        inlier_count_image = int(inlier_mask.sum().item())
        match_counts.append(match_count)
        correct_counts.append(correct_count_image)
        inlier_counts.append(inlier_count_image)
        image_summaries.append(
            {
                "image_name": camera.image_name,
                "visible_landmarks": int(visible.numel()),
                "detected_keypoints": int(matches["detected_keypoints"]),
                "matches": match_count,
                "correct_matches": correct_count_image,
                "inliers": inlier_count_image,
                "sparse_AE": float(ae),
                "sparse_TE": float(te),
            }
        )

    value_summary = summarize_landmark_value(
        visible_count=visible_count,
        matched_count=matched_count,
        correct_count=correct_count,
        inlier_count=inlier_count,
        utility=utility,
    )
    summary = {
        "model_path": dataset.model_path,
        "iteration": args.iteration,
        "cfg": args.cfg,
        "split": args.split,
        "image_count": len(cameras),
        "landmark_path": config["sparse"]["landmark_path"],
        "landmark_model_path": config["sparse"].get("landmark_model_path"),
        "detector_model_path": config["sparse"].get("detector_model_path"),
        "depth_check": bool(args.depth_check),
        "utility_source": args.utility_source,
        "min_observations": args.min_observations,
        "sparse_reprojection_error": float(config["sparse"]["reprojection_error"]),
        "correct_reprojection_error": correct_threshold,
        "avg_matches": float(np.mean(match_counts)) if match_counts else 0.0,
        "avg_correct_matches": float(np.mean(correct_counts)) if correct_counts else 0.0,
        "avg_inliers": float(np.mean(inlier_counts)) if inlier_counts else 0.0,
        "median_sparse_ae": float(np.median(sparse_aes)) if sparse_aes else None,
        "median_sparse_te": float(np.median(sparse_tes)) if sparse_tes else None,
        "calibration_target": args.calibration_target,
        "calibration_split_mode": args.calibration_split_mode,
        "calibration_eval_fraction": float(args.calibration_eval_fraction),
        "calibration_heldout": calibration_heldout,
    }
    summary.update(value_summary)

    if calibration_heldout:
        fit_counts = train_counts
        score_counts = eval_counts
    else:
        fit_counts = {
            "visible": visible_count,
            "matched": matched_count,
            "correct": correct_count,
            "inlier": inlier_count,
        }
        score_counts = None
    train_positive, train_trial, train_mask = _calibration_target_counts(
        args.calibration_target,
        fit_counts["visible"],
        fit_counts["matched"],
        fit_counts["correct"],
        fit_counts["inlier"],
    )
    if score_counts is not None:
        eval_positive, eval_trial, eval_mask = _calibration_target_counts(
            args.calibration_target,
            score_counts["visible"],
            score_counts["matched"],
            score_counts["correct"],
            score_counts["inlier"],
        )
    else:
        eval_positive = None
        eval_trial = None
        eval_mask = None

    calibration = calibrate_landmark_quality(
        features=_calibration_features(gaussians, utility),
        positive_count=train_positive,
        trial_count=train_trial,
        mask=train_mask,
        eval_positive_count=eval_positive,
        eval_trial_count=eval_trial,
        eval_mask=eval_mask,
        target_name=args.calibration_target,
        steps=args.calibration_steps,
        lr=args.calibration_lr,
        l2=args.calibration_l2,
    )
    target = args.calibration_target
    summary.update(
        {
            "calibration_feature_names": calibration["feature_names"],
            "calibration_landmark_count": calibration["calibration_landmark_count"],
            "calibration_train_landmark_count": calibration["calibration_train_landmark_count"],
            "calibration_eval_landmark_count": calibration["calibration_eval_landmark_count"],
            "calibration_trial_semantics": "visible_count" if target == "correct" else "matched_count",
            "calibrated_brier": calibration["calibrated_brier"],
            "calibrated_nll": calibration["calibrated_nll"],
            "spearman_calibrated_label_rate": calibration["spearman_calibrated_label_rate"],
            f"spearman_calibrated_{target}_rate": calibration[f"spearman_calibrated_{target}_rate"],
            f"top_quartile_calibrated_{target}_rate": calibration[f"top_quartile_calibrated_{target}_rate"],
            f"bottom_quartile_calibrated_{target}_rate": calibration[f"bottom_quartile_calibrated_{target}_rate"],
            "top_quartile_calibrated_label_rate": calibration["top_quartile_calibrated_label_rate"],
            "bottom_quartile_calibrated_label_rate": calibration["bottom_quartile_calibrated_label_rate"],
        }
    )
    if target == "inlier":
        summary["spearman_calibrated_inlier_rate"] = calibration["spearman_calibrated_inlier_rate"]
        summary["top_quartile_calibrated_inlier_rate"] = calibration["top_quartile_calibrated_inlier_rate"]
        summary["bottom_quartile_calibrated_inlier_rate"] = calibration["bottom_quartile_calibrated_inlier_rate"]

    per_landmark_output = getattr(args, "per_landmark_output", None)
    if per_landmark_output:
        _write_per_landmark_stats(
            per_landmark_output,
            landmark_indices,
            visible_count,
            matched_count,
            correct_count,
            inlier_count,
            utility,
        )

    label_state_output = getattr(args, "label_state_output", None)
    if label_state_output:
        if getattr(args, "label_state_reset", False):
            gaussians.init_localization_state(from_rgb_opacity=True)
        active = (visible_count > 0) | (matched_count > 0)
        full_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
        gaussians.add_sparse_match_label_stats(
            full_idx=full_idx,
            visible_count=visible_count[full_idx],
            matched_count=matched_count[full_idx],
            correct_count=correct_count[full_idx],
            inlier_count=inlier_count[full_idx],
            ema_decay=0.0,
        )
        os.makedirs(os.path.dirname(os.path.abspath(label_state_output)), exist_ok=True)
        torch.save(gaussians.capture_localization_state(), label_state_output)
        summary["label_state_output"] = label_state_output
        summary["label_state_reset"] = bool(getattr(args, "label_state_reset", False))

    calibrated_utility_output = getattr(args, "calibrated_utility_output", None)
    if calibrated_utility_output:
        _write_calibrated_utility(calibrated_utility_output, landmark_indices, calibration)
        summary["calibrated_utility_output"] = calibrated_utility_output

    return {"summary": summary, "images": image_summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse Level 3 landmark utility/inlier diagnostics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max_images", type=int, default=32)
    parser.add_argument("--detector_path", type=str, default=None)
    parser.add_argument("--detector_model_path", type=str, default=None)
    parser.add_argument("--landmark_path", type=str, default=None)
    parser.add_argument("--landmark_model_path", type=str, default=None)
    parser.add_argument("--landmark_meta_path", type=str, default=None)
    parser.add_argument("--landmark_meta_model_path", type=str, default=None)
    parser.add_argument("--use_landmark_prior", action="store_true", default=None)
    parser.add_argument("--no_use_landmark_prior", dest="use_landmark_prior", action="store_false")
    parser.add_argument("--reprojection_error", type=float, default=None)
    parser.add_argument("--correct_reprojection_error", type=float, default=None)
    parser.add_argument("--utility_source", type=str, default="reliability", choices=["reliability", "localization", "geometry", "meta"])
    parser.add_argument("--min_observations", type=int, default=8)
    parser.add_argument("--depth_check", action="store_true", default=False)
    parser.add_argument("--alpha_threshold", type=float, default=0.2)
    parser.add_argument("--depth_abs_tolerance", type=float, default=1e-3)
    parser.add_argument("--depth_rel_tolerance", type=float, default=0.01)
    parser.add_argument("--max_visible_landmarks_per_image", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--per_landmark_output", type=str, default=None)
    parser.add_argument("--label_state_output", type=str, default=None)
    parser.add_argument("--label_state_reset", action="store_true", default=False)
    parser.add_argument("--calibrated_utility_output", type=str, default=None)
    parser.add_argument("--calibration_target", type=str, default="inlier", choices=["inlier", "correct"])
    parser.add_argument("--calibration_split_mode", type=str, default="none", choices=["none", "block", "alternate"])
    parser.add_argument("--calibration_eval_fraction", type=float, default=0.0)
    parser.add_argument("--calibration_steps", type=int, default=200)
    parser.add_argument("--calibration_lr", type=float, default=0.1)
    parser.add_argument("--calibration_l2", type=float, default=1e-3)
    args = get_combined_args(parser)
    args.eval = args.split == "test"

    dataset = model.extract(args)
    gaussians = _load_gaussians(dataset)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, preload_cameras=False)

    result = diagnose(args, dataset, scene, gaussians)
    text = json.dumps(result, indent=2)
    output = getattr(args, "output", None)
    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w") as f:
            f.write(text)
            f.write("\n")
    print(text)
