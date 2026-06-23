#!/usr/bin/env python
import argparse
import itertools
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from encoders.feature_extractor import FeatureExtractor
from localization_training.dense_teacher import dense_localization_teacher
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import get_resolution_from_longest_edge


RESPONSIBILITY_KEYS = (
    "responsibility_reconstruction_mean_cosine",
    "responsibility_reconstruction_p10_cosine",
    "responsibility_reconstruction_min_cosine",
    "responsibility_reconstruction_valid_anchor_count",
)


def _limit_cameras(cameras, max_images):
    if max_images is None or int(max_images) <= 0:
        return list(cameras)
    return list(itertools.islice(cameras, int(max_images)))


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


def _weighted_mean(images, key, weight_key):
    weighted_total = 0.0
    weight_total = 0.0
    for image in images:
        weight = float(image.get(weight_key, 0.0))
        value = float(image.get(key, 0.0))
        if weight <= 0:
            continue
        weighted_total += value * weight
        weight_total += weight
    if weight_total > 0:
        return weighted_total / weight_total
    if not images:
        return 0.0
    return sum(float(image.get(key, 0.0)) for image in images) / len(images)


def _summarize_images(images):
    total_valid = int(
        sum(int(image.get("responsibility_reconstruction_valid_anchor_count", 0)) for image in images)
    )
    summary = {
        "image_count": len(images),
        "total_valid_anchor_count": total_valid,
        "mean_anchor_count": (
            sum(float(image.get("anchor_count", 0.0)) for image in images) / len(images) if images else 0.0
        ),
        "mean_responsibility_reconstruction_mean_cosine": _weighted_mean(
            images,
            "responsibility_reconstruction_mean_cosine",
            "responsibility_reconstruction_valid_anchor_count",
        ),
        "mean_responsibility_reconstruction_p10_cosine": _weighted_mean(
            images,
            "responsibility_reconstruction_p10_cosine",
            "responsibility_reconstruction_valid_anchor_count",
        ),
        "min_responsibility_reconstruction_min_cosine": (
            min(float(image.get("responsibility_reconstruction_min_cosine", 0.0)) for image in images)
            if images
            else 0.0
        ),
        "mean_desc_loss": _weighted_mean(images, "desc_loss", "anchor_count"),
        "mean_reproj_loss": _weighted_mean(images, "reproj_loss", "anchor_count"),
        "mean_kl_loss": _weighted_mean(images, "kl_loss", "anchor_count"),
        "mean_loss": _weighted_mean(images, "loss", "anchor_count"),
        "mean_loc_visible_count": (
            sum(float(image.get("loc_visible_count", 0.0)) for image in images) / len(images) if images else 0.0
        ),
    }
    return summary


def _ensure_optional_args(args):
    defaults = {
        "split": "train",
        "max_images": 8,
        "anchor_count": 128,
        "alpha_threshold": 0.2,
        "desc_temperature": 0.07,
        "fine_temperature": 0.05,
        "fine_window_radius": 4,
        "dense_kl_weight": 0.0,
        "dense_kl_temperature": 0.07,
        "responsibility_topk": 32,
        "responsibility_opacity_weight": 0.0,
        "responsibility_depth_weight": 0.0,
        "use_loc_opacity": False,
        "output": None,
        "eval": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def diagnose(args, dataset, scene, gaussians):
    if dataset.gaussian_type != "3dgs":
        raise ValueError(f"Dense responsibility diagnostics currently supports 3dgs only, got {dataset.gaussian_type}.")

    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    cameras = scene.getTestCameras() if args.split == "test" else scene.getTrainCameras()
    cameras = _limit_cameras(cameras, args.max_images)
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    image_summaries = []
    for camera in tqdm(cameras, desc="Dense responsibility diagnostics"):
        image = camera.original_image.cuda()
        height, width = get_resolution_from_longest_edge(
            image.shape[-2],
            image.shape[-1],
            dataset.longest_edge,
        )
        query_feature_map = _normalized_feature_map(feature_extractor, image, size=(height, width))
        pose = camera.world_view_transform.transpose(0, 1).cuda().float()

        with torch.no_grad():
            teacher_out = dense_localization_teacher(
                gaussians,
                query_feature_map,
                pose_init_w2c=pose,
                pose_gt_w2c=pose,
                fovx=camera.FoVx,
                fovy=camera.FoVy,
                width=width,
                height=height,
                background=background,
                anchor_count=args.anchor_count,
                alpha_threshold=args.alpha_threshold,
                desc_temperature=args.desc_temperature,
                fine_temperature=args.fine_temperature,
                fine_window_radius=args.fine_window_radius,
                dense_kl_weight=args.dense_kl_weight,
                dense_kl_temperature=args.dense_kl_temperature,
                responsibility_topk=args.responsibility_topk,
                responsibility_opacity_weight=args.responsibility_opacity_weight,
                responsibility_depth_weight=args.responsibility_depth_weight,
                use_loc_opacity=args.use_loc_opacity,
            )

        metrics = {
            "image_name": camera.image_name,
            "anchor_count": int(teacher_out.anchor_count),
            "loc_visible_count": int(
                teacher_out.loc_visible_idx.numel() if teacher_out.loc_visible_idx is not None else 0
            ),
            "loss": float(teacher_out.loss.detach().item()),
            "desc_loss": float(teacher_out.desc_loss.detach().item()),
            "reproj_loss": float(teacher_out.reproj_loss.detach().item()),
            "kl_loss": float(teacher_out.kl_loss.detach().item()),
        }
        for key in RESPONSIBILITY_KEYS:
            metrics[key] = float(teacher_out.diagnostics.get(key, 0.0))
        metrics["responsibility_reconstruction_valid_anchor_count"] = int(
            metrics["responsibility_reconstruction_valid_anchor_count"]
        )
        image_summaries.append(metrics)

    summary = _summarize_images(image_summaries)
    summary.update(
        {
            "model_path": dataset.model_path,
            "iteration": args.iteration,
            "split": args.split,
            "requested_max_images": args.max_images,
            "requested_anchor_count": args.anchor_count,
            "responsibility_topk": args.responsibility_topk,
            "responsibility_opacity_weight": args.responsibility_opacity_weight,
            "responsibility_depth_weight": args.responsibility_depth_weight,
            "use_loc_opacity": bool(args.use_loc_opacity),
            "dense_kl_weight": args.dense_kl_weight,
        }
    )
    return {"summary": summary, "images": image_summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dense anchor-to-Gaussian responsibility diagnostics")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    del pipeline
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--anchor_count", type=int, default=128)
    parser.add_argument("--alpha_threshold", type=float, default=0.2)
    parser.add_argument("--desc_temperature", type=float, default=0.07)
    parser.add_argument("--fine_temperature", type=float, default=0.05)
    parser.add_argument("--fine_window_radius", type=int, default=4)
    parser.add_argument("--dense_kl_weight", type=float, default=0.0)
    parser.add_argument("--dense_kl_temperature", type=float, default=0.07)
    parser.add_argument("--responsibility_topk", type=int, default=32)
    parser.add_argument("--responsibility_opacity_weight", type=float, default=0.0)
    parser.add_argument("--responsibility_depth_weight", type=float, default=0.0)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--use_loc_opacity", action=argparse.BooleanOptionalAction, default=False)
    else:
        parser.add_argument("--use_loc_opacity", dest="use_loc_opacity", action="store_true")
        parser.add_argument("--no-use_loc_opacity", dest="use_loc_opacity", action="store_false")
        parser.set_defaults(use_loc_opacity=False)
    parser.add_argument("--output", type=str, default=None)
    args = get_combined_args(parser)
    args = _ensure_optional_args(args)

    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False, preload_cameras=False)

    result = diagnose(args, dataset, scene, gaussians)
    text = json.dumps(result, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
            f.write("\n")
    print(text)
