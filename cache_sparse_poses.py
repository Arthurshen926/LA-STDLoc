import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import yaml
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from localization_training.episode_sampler import SparsePoseCache
from scene import Scene
from scene.gaussian_model import GaussianModel, GaussianModel_2dgs
from stdloc import STDLoc
from utils.pose_utils import cal_pose_error


def load_stdloc(args):
    model = ModelParams(ArgumentParser()).extract(args)
    if model.gaussian_type == "3dgs":
        gaussians = GaussianModel(model.sh_degree)
    elif model.gaussian_type == "2dgs":
        gaussians = GaussianModel_2dgs(model.sh_degree)
    else:
        raise ValueError("Gaussian type not supported")
    scene = Scene(model, gaussians, load_iteration=args.iteration, shuffle=False, preload_cameras=True)
    config = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    config.setdefault("sparse", {})["sparse_only"] = True
    config["dense"]["norm_before_render"] = model.norm_before_render
    config["feature_type"] = model.feature_type
    config["longest_edge"] = model.longest_edge
    config["model_path"] = model.model_path
    return scene, STDLoc(gaussians, config)


def cache_sparse_poses(scene, stdloc, output_path, split="train", max_queries=0):
    cameras = scene.getTrainCameras() if split == "train" else scene.getTestCameras()
    if not isinstance(cameras, list):
        cameras = list(cameras)
    if max_queries and max_queries > 0:
        cameras = cameras[:max_queries]
    cache = SparsePoseCache(output_path)
    ae_values = []
    te_values = []
    inliers = []
    failures = 0
    for camera in tqdm(cameras, desc=f"Cache sparse poses [{split}]"):
        gt_w2c = camera.world_view_transform.transpose(0, 1).cpu().numpy()
        query_image = camera.original_image.cuda()
        result = stdloc.localize(query_image, camera.FoVx, camera.FoVy)["sparse"]
        pose = result["pose_w2c"]
        ae, te = cal_pose_error(pose, gt_w2c)
        failed = int(result["inliers"]) < 4
        failures += int(failed)
        ae_values.append(float(ae))
        te_values.append(float(te))
        inliers.append(int(result["inliers"]))
        cache.update(
            camera.image_name,
            torch.as_tensor(pose, dtype=torch.float32),
            inliers=int(result["inliers"]),
            ae=float(ae),
            te=float(te),
            failed=failed,
        )
    cache.save()
    summary = {
        "split": split,
        "queries": len(cameras),
        "failures": failures,
        "median_ae": float(np.median(ae_values)) if ae_values else None,
        "median_te": float(np.median(te_values)) if te_values else None,
        "avg_inliers": float(np.mean(inliers)) if inliers else None,
    }
    with open(os.path.splitext(output_path)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Sparse pose cache summary:", summary)
    return summary


if __name__ == "__main__":
    parser = ArgumentParser(description="Cache sparse PnP poses for LA-STDLoc closed-loop training")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--cfg", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--max_queries", default=0, type=int)
    args = get_combined_args(parser)
    args.eval = args.split == "test"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scene, stdloc = load_stdloc(args)
    cache_sparse_poses(scene, stdloc, args.output, split=args.split, max_queries=args.max_queries)
