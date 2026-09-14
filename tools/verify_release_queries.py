"""Full-query numerical kernel regression against archived implementation.

This checks the deployed composition's extracted legacy kernels, not an
independent reproduction of the entire historical evaluation driver.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "research/legacy"))

from anygsloc.data.datasets import ColmapDataset
from anygsloc.data.calibration import resolve_scene_calibration
from anygsloc.online import runtime
from features.superpoint import SuperPoint as LegacySuperPoint
from localization.matcher import global_cosine_top1, retain_high_score_matches
from localization.pose_solver import solve_absolute_pose


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    dataset = ColmapDataset(a.dataset, images="processed")
    calibration = resolve_scene_calibration(dataset.split("mapping"))
    kwargs = dict(keypoint_count=calibration["keypoint_count"],
                  reprojection_error_px=calibration["reprojection_error_px"],
                  mask_policy="explicit_valid_mask", device="cuda:0", seed=2026)
    new = runtime.Localizer(a.map, **kwargs)
    old = runtime.Localizer(a.map, **kwargs)
    old.model = LegacySuperPoint().cuda().eval()
    old.model.nms_radius = 4
    names = ("global_cosine_top1", "retain_high_score_matches", "solve_absolute_pose")
    original = [getattr(runtime, n) for n in names]
    legacy = [global_cosine_top1, retain_high_score_matches, solve_absolute_pose]
    count = 0
    for camera in dataset.split("test"):
        image = dataset.load_image(camera)
        kw = dict(fov_x=camera.fov_x, fov_y=camera.fov_y,
                  valid_mask=dataset.valid_mask(camera))
        first = new.localize(image, **kw)
        try:
            for n, fn in zip(names, legacy):
                setattr(runtime, n, fn)
            second = old.localize(image, **kw)
        finally:
            for n, fn in zip(names, original):
                setattr(runtime, n, fn)
        for key in ("keypoints", "descriptors"):
            assert torch.equal(first[key], second[key]), (camera.image_name, key)
        for key in ("keypoint_indices", "anchor_indices", "scores"):
            assert torch.equal(getattr(first["matches"], key), getattr(second["matches"], key))
        np.testing.assert_array_equal(first["pose"].pose_w2c, second["pose"].pose_w2c)
        np.testing.assert_array_equal(first["pose"].inliers, second["pose"].inliers)
        count += 1
        print(json.dumps(dict(query=camera.image_name, exact=True, count=count)), flush=True)
    if not count:
        raise ValueError("empty test split")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(complete=True, queries=count, exact=True,
        scope="full_query_extracted_kernel_equivalence_not_historical_driver"), indent=2))


if __name__ == "__main__":
    main()
