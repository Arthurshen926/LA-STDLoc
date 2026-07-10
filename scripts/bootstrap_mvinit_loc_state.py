#!/usr/bin/env python
import argparse
import json
import os
import pickle

import torch
import torch.nn.functional as F

from arguments import ModelParams
from encoders.feature_extractor import FeatureExtractor
from localization_training.lafgs_reconstruction import (
    MultiViewInitConfig,
    apply_multiview_localization_stats,
    build_multiview_initialization,
    select_multiview_init_cameras,
)
from scene import Scene
from utils.general_utils import safe_state, seed_everything


def _load_gaussians(gaussian_type, sh_degree):
    if gaussian_type == "3dgs":
        from scene.gaussian_model import GaussianModel

        return GaussianModel(sh_degree)
    if gaussian_type == "2dgs":
        from scene.gaussian_model import GaussianModel_2dgs

        return GaussianModel_2dgs(sh_degree)
    raise ValueError(f"Unsupported gaussian_type: {gaussian_type}")


def _resize_bool_mask(mask, target_hw):
    return F.interpolate(
        mask.float()[None],
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0) > 0.5


def _query_feature_map(viewpoint_cam, feature_extractor, target_hw, masks=None):
    original_image = viewpoint_cam.original_image.cuda()
    with torch.no_grad():
        feature_map = feature_extractor(original_image[None])["feature_map"][0]
        feature_map = F.interpolate(
            feature_map.unsqueeze(0),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        feature_map = F.normalize(feature_map, p=2, dim=0)
    if masks is not None and getattr(viewpoint_cam, "image_name", "") in masks:
        object_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][0].cuda()[None], target_hw)
        distort_mask = _resize_bool_mask(masks[viewpoint_cam.image_name][2].cuda()[None], target_hw)
        feature_map = feature_map * (object_mask & distort_mask)
    return feature_map


def bootstrap_mvinit_loc_state(args):
    seed_everything(args.seed)
    safe_state(args.quiet)
    dataset = args.model_params.extract(args)
    gaussians = _load_gaussians(dataset.gaussian_type, dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    masks = None
    mask_path = os.path.join(dataset.source_path, dataset.images, "masks.pkl")
    if os.path.exists(mask_path):
        print(f"Loading masks from {mask_path}")
        with open(mask_path, "rb") as handle:
            masks = pickle.load(handle)

    train_cameras = scene.getTrainCameras().copy()
    support_cameras = select_multiview_init_cameras(
        train_cameras,
        max_views=args.max_views,
        mode=args.view_selection,
    )
    print(
        "Bootstrapping MVInit localization stats: "
        f"views={len(support_cameras)} selection={args.view_selection} "
        f"min_observations={args.min_observations} stats_only=True"
    )
    feature_extractor = FeatureExtractor(dataset.feature_type).cuda().eval()
    feature_scale = max(float(args.feature_scale or 1.0), 1e-3)

    def feature_map_for_camera(camera):
        target_hw = (
            max(8, int(round(camera.image_height * feature_scale))),
            max(8, int(round(camera.image_width * feature_scale))),
        )
        return _query_feature_map(camera, feature_extractor, target_hw=target_hw, masks=masks)

    result = build_multiview_initialization(
        gaussians,
        support_cameras,
        feature_map_for_camera,
        config=MultiViewInitConfig(
            min_observations=args.min_observations,
            chunk_size=args.chunk_size,
        ),
    )
    apply_multiview_localization_stats(
        gaussians,
        result,
        update_prototype=not args.no_update_prototype,
    )
    point_cloud_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{args.iteration}")
    os.makedirs(point_cloud_path, exist_ok=True)
    loc_state_path = os.path.join(point_cloud_path, "loc_state.pt")
    gaussians.save_localization_state(loc_state_path)

    observed = gaussians.loc_observation_count
    eligible = int((observed >= args.detector_min_observations).sum().item())
    utility = gaussians.compute_localization_utility(min_observations=args.detector_min_observations)
    nonzero_utility = int((utility != 0).sum().item())
    summary = {
        "model_path": dataset.model_path,
        "iteration": int(args.iteration),
        "loc_state_path": loc_state_path,
        "views": int(len(support_cameras)),
        "view_selection": args.view_selection,
        "feature_scale": float(feature_scale),
        "min_observations": int(args.min_observations),
        "detector_min_observations": int(args.detector_min_observations),
        "eligible_gaussians": eligible,
        "nonzero_utility_gaussians": nonzero_utility,
        "diagnostics": result.diagnostics,
    }
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Bootstrap stats-only MVInit localization state for an existing Gaussian map.")
    model_params = ModelParams(parser)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--max_views", type=int, default=-1)
    parser.add_argument("--view_selection", choices=["first", "uniform"], default="uniform")
    parser.add_argument("--min_observations", type=int, default=1)
    parser.add_argument("--detector_min_observations", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--feature_scale", type=float, default=1.0)
    parser.add_argument("--no_update_prototype", action="store_true")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(model_params=model_params)
    return parser


if __name__ == "__main__":
    bootstrap_mvinit_loc_state(build_parser().parse_args())
