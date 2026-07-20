#!/usr/bin/env python3
"""Compare the detector frontend with direct ULF-style SuperPoint matching.

Both paths use the same loaded landmark bank and active descriptor override.
The only intended difference is the query frontend: the current detector over
the deployment feature pyramid versus native ``detectAndCompute`` descriptors
followed by cosine top-k retrieval.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# Keep this standalone audit runnable from the repository root or any caller
# directory; evaluation scripts should not rely on an ambient PYTHONPATH.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arguments import ModelParams
from localization_training.full_primitive_retrieval import chunked_exact_topk
from localization_training.sparse_frontend import SparseMatchResult, select_match_candidates
from localization_training.ulf_initializer import sample_mask_at_grid_uv
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from stdloc import (
    STDLoc,
    evaluation_valid_mask,
    get_intrinsic,
    load_evaluation_masks,
    sparse_correspondence_diagnostics,
)
from utils.image_utils import get_resolution_from_longest_edge
from utils.pose_utils import cal_pose_error, solve_pose


def _summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p99": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def _project_residual(p2d, p3d, K, pose_w2c):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    if p2d.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    points_h = np.concatenate([p3d, np.ones((p3d.shape[0], 1))], axis=1)
    camera = (pose @ points_h.T)[:3].T
    projected = np.full((p3d.shape[0], 2), np.nan, dtype=np.float64)
    positive = camera[:, 2] > 1e-8
    projected[positive] = (
        (K @ camera[positive].T).T[:, :2] / camera[positive, 2:3]
    )
    return np.linalg.norm(projected - (p2d + 0.5), axis=1)


def _top1_per_keypoint(p2d, p3d, scores):
    """Select the descriptor-best candidate for every keypoint coordinate."""
    p2d = np.asarray(p2d, dtype=np.float32).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float32).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    best = {}
    for index, coordinate in enumerate(p2d):
        key = (float(coordinate[0]), float(coordinate[1]))
        if key not in best or scores[index] > scores[best[key]]:
            best[key] = index
    indices = np.asarray(sorted(best.values()), dtype=np.int64)
    return p2d[indices], p3d[indices], scores[indices]


def _recall_at_k_grouped(p2d, p3d, scores, K, gt_pose, ks):
    p2d = np.asarray(p2d, dtype=np.float32).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float32).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    groups = {}
    for index, coordinate in enumerate(p2d):
        groups.setdefault((float(coordinate[0]), float(coordinate[1])), []).append(index)
    output = {int(k): [] for k in ks}
    for indices in groups.values():
        indices = np.asarray(indices, dtype=np.int64)
        indices = indices[np.argsort(-scores[indices], kind="stable")]
        residual = _project_residual(p2d[indices], p3d[indices], K, gt_pose)
        for k in output:
            output[k].append(bool(np.any(residual[:k] <= 2.0)))
    return {
        f"recall_at_{k}": float(np.mean(values)) if values else 0.0
        for k, values in output.items()
    }


def _match_metrics(p2d, p3d, scores, K, gt_pose, pose_w2c, inliers):
    p2d = np.asarray(p2d, dtype=np.float32).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float32).reshape(-1, 3)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
    inliers = inliers[(inliers >= 0) & (inliers < p2d.shape[0])]
    residual = _project_residual(p2d, p3d, K, gt_pose)
    metrics = {
        "raw_match_count": int(p2d.shape[0]),
        "raw_precision_2px": float(np.mean(residual <= 2.0)) if residual.size else 0.0,
        "raw_precision_4px": float(np.mean(residual <= 4.0)) if residual.size else 0.0,
        "inlier_count": int(inliers.size),
        "inlier_gt_precision_2px": (
            float(np.mean(residual[inliers] <= 2.0)) if inliers.size else 0.0
        ),
        "inlier_gt_precision_4px": (
            float(np.mean(residual[inliers] <= 4.0)) if inliers.size else 0.0
        ),
    }
    diagnostics = sparse_correspondence_diagnostics(
        p2d,
        p3d,
        K,
        pose_w2c,
        inliers,
        int(K[0, 2] * 2),
        int(K[1, 2] * 2),
        gt_pose_w2c=gt_pose,
    )
    metrics["pose_info_logdet"] = diagnostics.get(
        "sparse_diag_inlier_pose_info_logdet", None
    )
    return metrics


def _rescale_grid_coordinates(p2d, source_hw, target_hw):
    """Map cell-index coordinates while preserving the explicit +0.5 center."""
    p2d = np.asarray(p2d, dtype=np.float32).reshape(-1, 2)
    source_height, source_width = map(int, source_hw)
    target_height, target_width = map(int, target_hw)
    if (source_height, source_width) == (target_height, target_width):
        return p2d.copy()
    physical = p2d + 0.5
    physical[:, 0] *= float(target_width) / float(source_width)
    physical[:, 1] *= float(target_height) / float(source_height)
    return physical - 0.5


def _query_input_hw(stdloc, camera):
    contract = stdloc.config["sparse"].get(
        "query_feature_contract", "legacy_full_then_resized_map"
    )
    source_hw = tuple(map(int, camera.original_image.shape[-2:]))
    if contract == "native_resized_input":
        return get_resolution_from_longest_edge(
            source_hw[0], source_hw[1], stdloc.longest_edge
        )
    if contract == "legacy_full_then_resized_map":
        return source_hw
    raise ValueError(f"Unknown sparse query feature contract: {contract!r}")


def _ulfloc_query_input(stdloc, camera, valid_mask):
    """Mirror the deployment RGB/mask contract before native SuperPoint."""
    contract = stdloc.config["sparse"].get(
        "query_feature_contract", "legacy_full_then_resized_map"
    )
    image = camera.original_image.cuda()
    target_hw = _query_input_hw(stdloc, camera)
    if tuple(image.shape[-2:]) != tuple(target_hw):
        image = F.interpolate(
            image[None], size=target_hw, mode="bilinear", align_corners=False
        )[0]
    if valid_mask is None:
        return image, None, contract
    mask = torch.as_tensor(valid_mask, dtype=torch.float32, device=image.device)
    if mask.ndim != 2:
        raise ValueError(f"Expected a two-dimensional valid mask, got {tuple(mask.shape)}")
    if tuple(mask.shape) != tuple(target_hw):
        mask = F.interpolate(mask[None, None], size=target_hw, mode="nearest")[0, 0]
    mask = mask.bool()
    # Native map-cache descriptors are extracted after masking this same RGB
    # tensor.  Preserve that condition in the direct sparse comparison.
    if contract == "native_resized_input":
        image = image * mask[None].to(dtype=image.dtype)
    return image, mask, contract


def _current_frontend(stdloc, camera, valid_mask, gt_pose, recall_ks, metric_hw):
    result = stdloc.localize(
        camera.original_image.cuda(),
        camera.FoVx,
        camera.FoVy,
        sparse_valid_mask=valid_mask,
    )["sparse"]
    debug = result.pop("_debug_sparse_matches", None)
    if debug is None:
        raise RuntimeError("Current frontend did not expose sparse diagnostics")
    feature_grid_hw = (int(debug["height"]), int(debug["width"]))
    raw_p2d = _rescale_grid_coordinates(
        debug["p2d_matcher_raw"], feature_grid_hw, metric_hw
    )
    raw_p3d = np.asarray(debug["p3d_matcher_raw"], dtype=np.float32)
    raw_scores = np.asarray(debug["scores_matcher_raw"], dtype=np.float32)
    K = get_intrinsic(camera.FoVx, camera.FoVy, metric_hw[1], metric_hw[0])
    top1_p2d, top1_p3d, top1_scores = _top1_per_keypoint(
        raw_p2d, raw_p3d, raw_scores
    )
    pose = np.asarray(result["pose_w2c"], dtype=np.float32)
    ae, te = cal_pose_error(pose, gt_pose)
    raw_metrics = _match_metrics(
        top1_p2d,
        top1_p3d,
        top1_scores,
        K,
        gt_pose,
        pose,
        [],
    )
    final_metrics = _match_metrics(
        _rescale_grid_coordinates(debug["p2d"], feature_grid_hw, metric_hw),
        debug["p3d"],
        debug["scores"],
        K,
        gt_pose,
        pose,
        debug["inliers"],
    )
    metrics = raw_metrics
    for key in (
        "inlier_count",
        "inlier_gt_precision_2px",
        "inlier_gt_precision_4px",
        "pose_info_logdet",
    ):
        metrics[key] = final_metrics[key]
    # PnP intentionally uses the deployment top-1 graph.  Recall@K is
    # measured from a diagnostic-only exact cosine retrieval, which is kept
    # separate so asking for Recall@16 cannot silently change the PnP graph.
    oracle = debug.get("discrete_oracle")
    if oracle is None:
        raise RuntimeError(
            "Current frontend parity requires diagnostics.dump_discrete_oracle"
        )
    oracle_keypoints = _rescale_grid_coordinates(
        oracle["keypoint_xy"], feature_grid_hw, metric_hw
    )
    oracle_indices = np.asarray(oracle["topk_landmark_idx"], dtype=np.int64)
    oracle_scores = np.asarray(oracle["topk_scores"], dtype=np.float32)
    if oracle_indices.ndim != 2 or oracle_scores.shape != oracle_indices.shape:
        raise RuntimeError(
            "Malformed diagnostic cosine retrieval in current frontend parity"
        )
    oracle_p2d = np.repeat(oracle_keypoints, oracle_indices.shape[1], axis=0)
    oracle_p3d = (
        stdloc.landmarks.get_xyz[torch.from_numpy(oracle_indices).to("cuda")]
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, 3)
    )
    metrics.update(
        _recall_at_k_grouped(
            oracle_p2d,
            oracle_p3d,
            oracle_scores.reshape(-1),
            K,
            gt_pose,
            recall_ks,
        )
    )
    metrics.update(
        {
            "pose_rotation_error_deg": float(ae),
            "pose_translation_error_cm": float(te),
            "detected_keypoints": int(result.get("detected_keypoints", 0)),
            "raw_candidate_keypoints": int(top1_p2d.shape[0]),
            "pnp_match_count": int(np.asarray(debug["p2d"]).shape[0]),
            "feature_grid_hw": [int(debug["height"]), int(debug["width"])],
            "metric_query_input_hw": [int(metric_hw[0]), int(metric_hw[1])],
            "frontend": (
                "stdloc_ulfloc_native_deployment"
                if getattr(stdloc, "sparse_frontend", "detector") == "ulfloc_native"
                else "stdloc_detector_plus_cosine"
            ),
        }
    )
    return metrics, raw_p2d


def _ulfloc_frontend(stdloc, camera, valid_mask, gt_pose, top_k, sparse_keypoints, recall_ks):
    image, input_mask, query_contract = _ulfloc_query_input(
        stdloc, camera, valid_mask
    )
    sparse = stdloc.feature_extractor.detectAndCompute(
        image[None], top_k=sparse_keypoints
    )[0]
    keypoints = sparse["keypoints"]
    descriptors = F.normalize(sparse["descriptors"], dim=1)
    if input_mask is not None:
        keep = sample_mask_at_grid_uv(input_mask, keypoints)
        keypoints = keypoints[keep]
        descriptors = descriptors[keep]
    bank_features = F.normalize(
        stdloc.landmarks.get_loc_feature.detach().reshape(
            stdloc.landmarks.get_loc_feature.shape[0], -1
        ),
        dim=1,
    )
    retrieval = chunked_exact_topk(descriptors, bank_features, topk=top_k)
    scores = retrieval.scores
    indices = retrieval.indices
    bank_xyz = stdloc.landmarks.get_xyz.detach()
    candidate_xyz = bank_xyz[indices].detach().cpu().numpy()
    candidate_p2d = keypoints[:, None, :].expand(-1, top_k, -1).detach().cpu().numpy()
    K = get_intrinsic(camera.FoVx, camera.FoVy, image.shape[-1], image.shape[-2])
    flat_p2d = candidate_p2d.reshape(-1, 2)
    flat_p3d = candidate_xyz.reshape(-1, 3)
    flat_scores = scores.detach().cpu().numpy().reshape(-1)
    recall = _recall_at_k_grouped(
        flat_p2d, flat_p3d, flat_scores, K, gt_pose, recall_ks
    )
    p2d = keypoints.detach().cpu().float()
    top1 = indices[:, 0].detach().cpu().long()
    top1_scores = scores[:, 0].detach().cpu().float()
    matches = select_match_candidates(
        SparseMatchResult(
            torch.arange(p2d.shape[0], dtype=torch.long),
            top1,
            top1_scores,
        ),
        threshold=float(stdloc.config["sparse"].get("threshold", 0.0)),
        max_matches_per_keypoint=int(
            stdloc.config["sparse"].get("max_matches_per_keypoint", 1)
        ),
        max_matches_per_landmark=int(
            stdloc.config["sparse"].get("max_matches_per_landmark", 2)
        ),
        min_match_count=int(stdloc.config["sparse"].get("min_candidate_matches", 0)),
        refill_trigger_count=int(
            stdloc.config["sparse"].get("candidate_refill_trigger_count", 0)
        ),
    )
    selected_p2d = p2d[matches.keypoint_idx].numpy()
    selected_p3d = bank_xyz[matches.landmark_idx.cuda()].detach().cpu().numpy()
    selected_scores = matches.scores.detach().cpu().numpy()
    pose, inliers = solve_pose(
        selected_p2d + 0.5,
        selected_p3d,
        K,
        solver=stdloc.config["sparse"].get("solver", "poselib"),
        reprojection_error=float(stdloc.config["sparse"].get("reprojection_error", 12.0)),
        confidence=float(stdloc.config["sparse"].get("confidence", 0.99999)),
        max_iterations=int(stdloc.config["sparse"].get("max_iterations", 100000)),
        min_iterations=int(stdloc.config["sparse"].get("min_iterations", 1000)),
        ransac_seed=int(stdloc.config["sparse"].get("ransac_seed", 0)),
    )
    ae, te = cal_pose_error(pose, gt_pose)
    raw_metrics = _match_metrics(
        p2d.numpy(),
        bank_xyz[top1.to(bank_xyz.device)].detach().cpu().numpy(),
        top1_scores.numpy(),
        K,
        gt_pose,
        pose,
        [],
    )
    final_metrics = _match_metrics(
        selected_p2d,
        selected_p3d,
        selected_scores,
        K,
        gt_pose,
        pose,
        inliers,
    )
    metrics = raw_metrics
    for key in (
        "inlier_count",
        "inlier_gt_precision_2px",
        "inlier_gt_precision_4px",
        "pose_info_logdet",
    ):
        metrics[key] = final_metrics[key]
    metrics.update(recall)
    metrics.update(
        {
            "pose_rotation_error_deg": float(ae),
            "pose_translation_error_cm": float(te),
            "detected_keypoints": int(keypoints.shape[0]),
            "raw_candidate_keypoints": int(keypoints.shape[0]),
            "pnp_match_count": int(selected_p2d.shape[0]),
            "feature_grid_hw": [int(image.shape[-2]), int(image.shape[-1])],
            "metric_query_input_hw": [int(image.shape[-2]), int(image.shape[-1])],
            "query_feature_contract": query_contract,
            "native_sparse_keypoints_before_mask": int(sparse["keypoints"].shape[0]),
            "native_sparse_keypoints_after_mask": int(keypoints.shape[0]),
            "frontend": "ulfloc_native_superpoint_cosine_top1",
        }
    )
    return metrics, keypoints.detach().cpu().numpy()


def _nearest_location_distance(current_p2d, direct_p2d):
    if current_p2d.size == 0 or direct_p2d.size == 0:
        return None
    current_physical = current_p2d.astype(np.float32) + 0.5
    direct_physical = direct_p2d.astype(np.float32) + 0.5
    distance = torch.cdist(
        torch.from_numpy(current_physical), torch.from_numpy(direct_physical)
    ).amin(dim=1).numpy()
    return _summary(distance)


def _aggregate(records, key):
    values = [record[key] for record in records if record.get(key) is not None]
    return _summary(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_params = ModelParams(parser)
    parser.add_argument("--load_iteration", type=int, default=30000)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--max_views", type=int, default=10)
    parser.add_argument("--sparse_keypoints", type=int, default=2048)
    parser.add_argument("--recall_ks", nargs="+", type=int, default=[1, 4, 16])
    args = parser.parse_args()
    dataset = model_params.extract(args)
    config = yaml.load(Path(args.cfg).read_text(), Loader=yaml.FullLoader)
    config = copy.deepcopy(config)
    config.setdefault("sparse", {})
    config.setdefault("dense", {})
    config["feature_type"] = dataset.feature_type
    config["longest_edge"] = dataset.longest_edge
    config["model_path"] = dataset.model_path
    config["sparse_only"] = True
    # The deployment graph and both PnP paths are explicitly top-1.  Current
    # frontend Recall@K uses ``dump_discrete_oracle`` below rather than
    # changing ``topk`` as a side effect of the requested reporting Ks.
    config["sparse"]["topk"] = 1
    config["sparse"]["use_candidate_pair_scorer"] = False
    config["sparse"]["use_pair_measurement"] = False
    config["sparse"].setdefault("diagnostics", {})
    config["sparse"]["diagnostics"].update(
        {
            "enabled": True,
            "gt_metrics": True,
            "dump_correspondences": False,
            "dump_discrete_oracle": True,
            "oracle_topk": max(args.recall_ks),
        }
    )

    if dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    elif dataset.gaussian_type == "3dgs":
        gaussians = GaussianModel(dataset.sh_degree)
    else:
        raise ValueError("Unsupported Gaussian type")
    scene = Scene(
        dataset,
        gaussians,
        load_iteration=args.load_iteration,
        shuffle=False,
        preload_cameras=False,
        load_test_cameras=True,
    )
    stdloc = STDLoc(gaussians, config)
    masks, masks_path = load_evaluation_masks(dataset)
    cameras = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    cameras = sorted(cameras, key=lambda camera: str(camera.image_name))
    if args.max_views > 0 and len(cameras) > args.max_views:
        positions = np.linspace(0, len(cameras) - 1, args.max_views).round().astype(int)
        cameras = [cameras[index] for index in positions]
    records = []
    for camera in cameras:
        gt_pose = camera.world_view_transform.transpose(0, 1).cpu().numpy()
        valid_mask = evaluation_valid_mask(masks, camera)
        metric_hw = _query_input_hw(stdloc, camera)
        current, current_p2d = _current_frontend(
            stdloc, camera, valid_mask, gt_pose, args.recall_ks, metric_hw
        )
        direct, direct_p2d = _ulfloc_frontend(
            stdloc,
            camera,
            valid_mask,
            gt_pose,
            max(args.recall_ks),
            args.sparse_keypoints,
            args.recall_ks,
        )
        location = _nearest_location_distance(
            current_p2d,
            direct_p2d,
        )
        records.append(
            {
                "image_name": str(camera.image_name),
                "stdloc": current,
                "ulfloc": direct,
                "nearest_stdloc_to_ulfloc_keypoint_distance_px": location,
            }
        )

    payload = {
        "schema_version": 1,
        "protocol": {
            "same_fixed_landmark_bank": True,
            "same_active_landmark_descriptors": True,
            "stdloc_pair_scorer_disabled": True,
            "stdloc_pair_measurement_disabled": True,
            "ulfloc_query_descriptor": "SuperPoint.detectAndCompute",
            "pnp_matching": "cosine top-1 in both frontends",
            "recall_matching": (
                "exact cosine top-k diagnostic retrieval; it does not change PnP"
            ),
            "metric_coordinate_space": "common_sparse_query_input_pixels_plus_half_v1",
            "query_feature_contract": stdloc.config["sparse"].get(
                "query_feature_contract", "legacy_full_then_resized_map"
            ),
            "valid_mask_policy": "object_and_sky_and_distortion_v1" if masks else "none",
            "valid_mask_path": masks_path,
        },
        "dataset": {
            "model_path": str(Path(dataset.model_path).resolve()),
            "source_path": str(Path(dataset.source_path).resolve()),
            "split": args.split,
            "views": int(len(records)),
            "cfg": str(Path(args.cfg).resolve()),
        },
        "summary": {
            frontend: {
                key: _aggregate([record[frontend] for record in records], key)
                for key in (
                    "detected_keypoints",
                    "raw_precision_2px",
                    "raw_precision_4px",
                    "inlier_count",
                    "inlier_gt_precision_2px",
                    "inlier_gt_precision_4px",
                    "pose_info_logdet",
                    "pose_translation_error_cm",
                    "pose_rotation_error_deg",
                    *[f"recall_at_{k}" for k in args.recall_ks],
                )
            }
            for frontend in ("stdloc", "ulfloc")
        },
        "per_view": records,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
