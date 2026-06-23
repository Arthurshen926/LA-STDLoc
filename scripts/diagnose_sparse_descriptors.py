#!/usr/bin/env python
import argparse
import itertools
import json
import os
import pickle

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from gaussian_renderer import render_from_pose_gsplat
from localization_training.descriptor_diagnostics import (
    collect_projected_descriptor_pairs,
    descriptor_alignment_metrics,
    full_bank_descriptor_metrics,
    summarize_descriptor_metric_batches,
    summarize_full_bank_metric_batches,
)
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from utils.image_utils import get_resolution_from_longest_edge


LEVEL1_METRIC_KEYS = ("positive_cosine_mean", "margin_mean", "top1_recall", "mnn_precision")
FULL_BANK_METRIC_KEYS = (
    "full_bank_recall_at_1",
    "full_bank_recall_at_5",
    "full_bank_recall_at_10",
    "full_bank_mnn_precision",
    "full_bank_margin_mean",
)


def _resolve_artifact_path(model_path, artifact_path, artifact_model_path=None):
    if os.path.isabs(artifact_path):
        return artifact_path
    return os.path.join(artifact_model_path or model_path, artifact_path)


def _load_landmark_indices(model_path, landmark_path, landmark_model_path=None):
    path = _resolve_artifact_path(model_path, landmark_path, landmark_model_path)
    with open(path, "rb") as f:
        indices = pickle.load(f)
    return torch.as_tensor(indices, dtype=torch.long)


def _load_gaussians_from_iteration(dataset, model_path, iteration):
    if dataset.gaussian_type == "3dgs":
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError(f"Unsupported gaussian_type: {dataset.gaussian_type}")
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    return gaussians


def _normalized_feature_map(feature_extractor, image, size):
    with torch.no_grad():
        feature_map = feature_extractor(image[None])["feature_map"]
        feature_map = F.interpolate(
            feature_map,
            size=size,
            mode="bilinear",
            align_corners=False,
        )[0]
        return F.normalize(feature_map, p=2, dim=0)


def _squeeze_scalar_map(value):
    if value is None:
        return None
    value = value.squeeze()
    if value.dim() == 3:
        value = value[..., 0]
    return value


def _limit_cameras(cameras, max_images):
    if max_images is None:
        return list(cameras)
    return list(itertools.islice(cameras, max_images))


def _ensure_optional_args(args):
    defaults = {
        "landmark_model_path": None,
        "baseline_model_path": None,
        "baseline_iteration": 30000,
        "full_bank": False,
        "full_bank_topk": [1, 5, 10],
        "full_bank_temperature": 0.07,
        "output": None,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def diagnose(args, dataset, scene, gaussians):
    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    landmark_indices = _load_landmark_indices(
        dataset.model_path,
        args.landmark_path,
        args.landmark_model_path,
    )
    point_count = gaussians.get_xyz.shape[0]
    if landmark_indices.numel() == 0 or int(landmark_indices.max().item()) >= point_count:
        raise ValueError(
            "Landmark indices are incompatible with the evaluated map: "
            f"landmarks={landmark_indices.numel()}, max={int(landmark_indices.max().item()) if landmark_indices.numel() else 'empty'}, "
            f"point_count={point_count}."
        )

    baseline_gaussians = None
    if args.baseline_model_path:
        baseline_gaussians = _load_gaussians_from_iteration(
            dataset,
            args.baseline_model_path,
            args.baseline_iteration,
        )
        if int(landmark_indices.max().item()) >= baseline_gaussians.get_xyz.shape[0]:
            baseline_gaussians = None

    cameras = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    cameras = _limit_cameras(cameras, args.max_images)

    bank_features = gaussians.get_loc_feature[
        landmark_indices.to(device=gaussians.get_xyz.device)
    ].reshape(landmark_indices.numel(), -1)
    full_to_bank = torch.full(
        (int(landmark_indices.max().item()) + 1,),
        -1,
        dtype=torch.long,
    )
    full_to_bank[landmark_indices.cpu()] = torch.arange(landmark_indices.numel(), dtype=torch.long)

    batch_metrics = []
    full_bank_batch_metrics = []
    image_summaries = []
    for camera in tqdm(cameras, desc="Descriptor diagnostics"):
        image = camera.original_image.cuda()
        height, width = get_resolution_from_longest_edge(
            image.shape[-2],
            image.shape[-1],
            dataset.longest_edge,
        )
        query_feature_map = _normalized_feature_map(feature_extractor, image, size=(height, width))
        pose = camera.world_view_transform.transpose(0, 1).cuda()
        target_depth = None
        target_alpha = None
        if args.depth_check:
            with torch.no_grad():
                rendered = render_from_pose_gsplat(
                    gaussians,
                    pose,
                    camera.FoVx,
                    camera.FoVy,
                    width,
                    height,
                    rgb_only=True,
                    render_mode="RGB+ED",
                )
            target_depth = _squeeze_scalar_map(rendered.get("depth"))
            target_alpha = _squeeze_scalar_map(rendered.get("alphas"))

        pairs = collect_projected_descriptor_pairs(
            gaussians,
            query_feature_map,
            pose,
            camera.FoVx,
            camera.FoVy,
            landmark_indices,
            target_depth=target_depth,
            target_alpha=target_alpha,
            alpha_threshold=args.alpha_threshold,
            depth_abs_tolerance=args.depth_abs_tolerance,
            depth_rel_tolerance=args.depth_rel_tolerance,
            max_landmarks=args.max_landmarks_per_image,
            baseline_gaussians=baseline_gaussians,
        )
        metrics = descriptor_alignment_metrics(
            pairs["gaussian_features"],
            pairs["query_features"],
            baseline_features=pairs["baseline_features"],
        )
        image_metrics = dict(metrics)
        if args.full_bank:
            full_idx = pairs["full_idx"].detach().cpu()
            positive_bank_indices = torch.full_like(full_idx, -1)
            valid = (full_idx >= 0) & (full_idx < full_to_bank.numel())
            positive_bank_indices[valid] = full_to_bank[full_idx[valid]]
            full_bank_metrics = full_bank_descriptor_metrics(
                pairs["query_features"],
                bank_features,
                positive_bank_indices.to(device=pairs["query_features"].device),
                topk=tuple(args.full_bank_topk),
                temperature=args.full_bank_temperature,
            )
            image_metrics.update(full_bank_metrics)
            full_bank_batch_metrics.append(full_bank_metrics)
        image_metrics["image_name"] = camera.image_name
        image_summaries.append(image_metrics)
        batch_metrics.append(metrics)

    summary = summarize_descriptor_metric_batches(batch_metrics)
    if args.full_bank:
        summary.update(summarize_full_bank_metric_batches(full_bank_batch_metrics))
    summary.update(
        {
            "model_path": dataset.model_path,
            "iteration": args.iteration,
            "landmark_path": args.landmark_path,
            "landmark_model_path": args.landmark_model_path,
            "baseline_model_path": args.baseline_model_path,
            "baseline_iteration": args.baseline_iteration if args.baseline_model_path else None,
            "split": args.split,
            "image_count": len(cameras),
            "depth_check": bool(args.depth_check),
            "max_landmarks_per_image": args.max_landmarks_per_image,
            "full_bank": bool(args.full_bank),
        }
    )
    return {"summary": summary, "images": image_summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sparse descriptor Level 1 diagnostics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--landmark_path", type=str, default="detector/sampled_idx.pkl")
    parser.add_argument("--landmark_model_path", type=str, default=None)
    parser.add_argument("--baseline_model_path", type=str, default=None)
    parser.add_argument("--baseline_iteration", type=int, default=30000)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max_images", type=int, default=32)
    parser.add_argument("--max_landmarks_per_image", type=int, default=1024)
    parser.add_argument("--depth_check", action="store_true", default=False)
    parser.add_argument("--alpha_threshold", type=float, default=0.2)
    parser.add_argument("--depth_abs_tolerance", type=float, default=1e-3)
    parser.add_argument("--depth_rel_tolerance", type=float, default=0.01)
    parser.add_argument("--full_bank", action="store_true", default=False)
    parser.add_argument("--full_bank_topk", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--full_bank_temperature", type=float, default=0.07)
    parser.add_argument("--output", type=str, default=None)
    args = get_combined_args(parser)
    args = _ensure_optional_args(args)
    args.eval = True

    dataset = model.extract(args)
    if dataset.gaussian_type == "3dgs":
        gaussians = GaussianModel(dataset.sh_degree)
    elif dataset.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(dataset.sh_degree)
    else:
        raise ValueError(f"Unsupported gaussian_type: {dataset.gaussian_type}")
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, preload_cameras=False)

    result = diagnose(args, dataset, scene, gaussians)
    text = json.dumps(result, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
            f.write("\n")
    print(text)
