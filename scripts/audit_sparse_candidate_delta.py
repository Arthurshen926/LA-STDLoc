#!/usr/bin/env python3
"""Audit how two discrete-oracle dumps change deployment correspondence graphs."""

import argparse
import json
import os

import numpy as np


def _load_dump(root):
    with open(os.path.join(root, "manifest.json")) as handle:
        manifest = json.load(handle)
    bank = dict(np.load(os.path.join(root, manifest["landmark_bank"])))
    queries = {}
    for filename in manifest["query_files"]:
        payload = dict(np.load(os.path.join(root, filename)))
        queries[str(payload["image_name"].item())] = payload
    return bank, queries


def _pair_arrays(payload, stage, bank_ids, *, inliers_only=False):
    keypoints = np.asarray(payload.get(f"{stage}_keypoint_idx", []), dtype=np.int64)
    landmarks = np.asarray(payload.get(f"{stage}_landmark_idx", []), dtype=np.int64)
    if inliers_only:
        keep = np.asarray(payload.get("hard_post_inliers", []), dtype=np.int64)
        keep = keep[(keep >= 0) & (keep < keypoints.size)]
        keypoints = keypoints[keep]
        landmarks = landmarks[keep]
    valid = (landmarks >= 0) & (landmarks < bank_ids.size)
    return keypoints[valid], landmarks[valid], bank_ids[landmarks[valid]]


def _pair_set(payload, stage, bank_ids, *, inliers_only=False):
    keypoints, _, raw_ids = _pair_arrays(
        payload, stage, bank_ids, inliers_only=inliers_only
    )
    return set(zip(keypoints.tolist(), raw_ids.tolist()))


def _jaccard(left, right):
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _camera_center(pose_w2c):
    rotation = np.asarray(pose_w2c, dtype=np.float64)[:3, :3]
    translation = np.asarray(pose_w2c, dtype=np.float64)[:3, 3]
    return -rotation.T @ translation


def _translation_error_cm(payload):
    return float(
        np.linalg.norm(
            _camera_center(payload["pred_pose_w2c"])
            - _camera_center(payload["gt_pose_w2c"])
        )
        * 100.0
    )


def _reprojection_errors(payload, local_ids, keypoint_ids, bank_xyz):
    if local_ids.size == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 2), dtype=np.float64), np.empty(0)
    keypoint_xy = np.asarray(payload["keypoint_xy"], dtype=np.float64)
    keypoint_ids = np.asarray(keypoint_ids, dtype=np.int64)
    valid = (keypoint_ids >= 0) & (keypoint_ids < keypoint_xy.shape[0])
    local_ids = local_ids[valid]
    keypoint_ids = keypoint_ids[valid]
    xyz = np.asarray(bank_xyz, dtype=np.float64)[local_ids]
    pose = np.asarray(payload["gt_pose_w2c"], dtype=np.float64)
    K = np.asarray(payload["K"], dtype=np.float64)
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = np.empty((camera.shape[0], 2), dtype=np.float64)
    projected[:, 0] = K[0, 0] * camera[:, 0] / np.maximum(depth, 1e-8) + K[0, 2]
    projected[:, 1] = K[1, 1] * camera[:, 1] / np.maximum(depth, 1e-8) + K[1, 2]
    residual = keypoint_xy[keypoint_ids] - projected
    error = np.linalg.norm(residual, axis=1)
    error[depth <= 0] = np.inf
    return error, residual, depth


def _pair_error_map(payload, stage, bank_ids, bank_xyz, *, inliers_only=False):
    keypoints, local_ids, raw_ids = _pair_arrays(
        payload, stage, bank_ids, inliers_only=inliers_only
    )
    errors, residuals, depths = _reprojection_errors(payload, local_ids, keypoints, bank_xyz)
    return {
        (int(keypoint), int(raw_id)): (float(error), residual, float(depth))
        for keypoint, raw_id, error, residual, depth in zip(
            keypoints, raw_ids, errors, residuals, depths
        )
    }


def _coverage(pair_info, payload):
    if not pair_info:
        return {"grid_occupancy": 0.0, "depth_bin_occupancy": 0.0}
    keypoint_xy = np.asarray(payload["keypoint_xy"], dtype=np.float64)
    width = max(int(np.asarray(payload["width"]).item()), 1)
    height = max(int(np.asarray(payload["height"]).item()), 1)
    keypoints = np.asarray([key for key, _ in pair_info], dtype=np.int64)
    xy = keypoint_xy[keypoints]
    grid_x = np.clip((xy[:, 0] * 4 / width).astype(np.int64), 0, 3)
    grid_y = np.clip((xy[:, 1] * 4 / height).astype(np.int64), 0, 3)
    grid_occupancy = np.unique(grid_y * 4 + grid_x).size / 16.0
    depths = np.asarray([value[2] for value in pair_info.values()], dtype=np.float64)
    finite = np.isfinite(depths) & (depths > 0)
    if finite.sum() < 2:
        depth_occupancy = float(finite.any()) / 4.0
    else:
        edges = np.quantile(depths[finite], [0.25, 0.5, 0.75])
        depth_bins = np.digitize(depths[finite], edges, right=False)
        depth_occupancy = np.unique(depth_bins).size / 4.0
    return {"grid_occupancy": float(grid_occupancy), "depth_bin_occupancy": float(depth_occupancy)}


def _mean_or_zero(values):
    return float(np.mean(values)) if values else 0.0


def _audit_query(old, new, bank_ids, bank_xyz, correct_radius_px):
    old_top1 = bank_ids[np.asarray(old["topk_landmark_idx"])[:, 0]]
    new_top1 = bank_ids[np.asarray(new["topk_landmark_idx"])[:, 0]]
    top1_change = (
        float(np.mean(old_top1 != new_top1))
        if old_top1.shape == new_top1.shape and old_top1.size
        else 1.0
    )
    old_pre = _pair_error_map(old, "hard_pre", bank_ids, bank_xyz)
    new_pre = _pair_error_map(new, "hard_pre", bank_ids, bank_xyz)
    old_inlier = _pair_error_map(
        old, "hard_post", bank_ids, bank_xyz, inliers_only=True
    )
    new_inlier = _pair_error_map(
        new, "hard_post", bank_ids, bank_xyz, inliers_only=True
    )
    old_pre_set, new_pre_set = set(old_pre), set(new_pre)
    old_inlier_set, new_inlier_set = set(old_inlier), set(new_inlier)
    added_pre = new_pre_set - old_pre_set
    removed_pre = old_pre_set - new_pre_set
    added_inlier = new_inlier_set - old_inlier_set
    removed_inlier = old_inlier_set - new_inlier_set
    correct = lambda values, pairs: sum(values[pair][0] <= correct_radius_px for pair in pairs)
    harmful = lambda values, pairs: sum(values[pair][0] > correct_radius_px for pair in pairs)
    old_coverage = _coverage(old_inlier, old)
    new_coverage = _coverage(new_inlier, new)
    old_residual = np.asarray([value[1] for value in old_inlier.values()])
    new_residual = np.asarray([value[1] for value in new_inlier.values()])
    old_te = _translation_error_cm(old)
    new_te = _translation_error_cm(new)
    return {
        "image_name": str(old["image_name"].item()),
        "top1_landmark_change_ratio": top1_change,
        "pre_pair_jaccard": _jaccard(old_pre_set, new_pre_set),
        "inlier_pair_jaccard": _jaccard(old_inlier_set, new_inlier_set),
        "added_pre_pairs": len(added_pre),
        "removed_pre_pairs": len(removed_pre),
        "added_gt_correct_pre_pairs": correct(new_pre, added_pre),
        "removed_gt_correct_pre_pairs": correct(old_pre, removed_pre),
        "added_inliers": len(added_inlier),
        "removed_inliers": len(removed_inlier),
        "added_gt_correct_inliers": correct(new_inlier, added_inlier),
        "removed_gt_correct_inliers": correct(old_inlier, removed_inlier),
        "added_harmful_inliers": harmful(new_inlier, added_inlier),
        "removed_harmful_inliers": harmful(old_inlier, removed_inlier),
        "old_inlier_gt_precision": (
            correct(old_inlier, old_inlier_set) / max(len(old_inlier_set), 1)
        ),
        "new_inlier_gt_precision": (
            correct(new_inlier, new_inlier_set) / max(len(new_inlier_set), 1)
        ),
        "old_signed_inlier_residual_x": _mean_or_zero(old_residual[:, 0].tolist()) if old_residual.size else 0.0,
        "old_signed_inlier_residual_y": _mean_or_zero(old_residual[:, 1].tolist()) if old_residual.size else 0.0,
        "new_signed_inlier_residual_x": _mean_or_zero(new_residual[:, 0].tolist()) if new_residual.size else 0.0,
        "new_signed_inlier_residual_y": _mean_or_zero(new_residual[:, 1].tolist()) if new_residual.size else 0.0,
        "old_grid_occupancy": old_coverage["grid_occupancy"],
        "new_grid_occupancy": new_coverage["grid_occupancy"],
        "old_depth_bin_occupancy": old_coverage["depth_bin_occupancy"],
        "new_depth_bin_occupancy": new_coverage["depth_bin_occupancy"],
        "old_te_cm": old_te,
        "new_te_cm": new_te,
        "te_delta_cm": new_te - old_te,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_dump")
    parser.add_argument("new_dump")
    parser.add_argument("--correct_radius_px", type=float, default=4.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    old_bank, old_queries = _load_dump(args.old_dump)
    new_bank, new_queries = _load_dump(args.new_dump)
    old_ids = np.asarray(old_bank["source_gaussian_idx"], dtype=np.int64)
    new_ids = np.asarray(new_bank["source_gaussian_idx"], dtype=np.int64)
    if not np.array_equal(old_ids, new_ids):
        raise ValueError("Candidate delta audit requires identical landmark raw-ID order")
    if not np.array_equal(old_bank["landmark_xyz"], new_bank["landmark_xyz"]):
        raise ValueError("Candidate delta audit requires identical landmark geometry")
    common = sorted(set(old_queries) & set(new_queries))
    rows = [
        _audit_query(
            old_queries[name],
            new_queries[name],
            old_ids,
            old_bank["landmark_xyz"],
            args.correct_radius_px,
        )
        for name in common
    ]
    numeric_keys = [key for key in rows[0] if key != "image_name"] if rows else []
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "median": float(np.median([row[key] for row in rows])),
        }
        for key in numeric_keys
    }
    worse = [row for row in rows if row["te_delta_cm"] > 0.0]
    better_or_equal = [row for row in rows if row["te_delta_cm"] <= 0.0]
    result = {
        "correct_radius_px": float(args.correct_radius_px),
        "common_query_count": len(common),
        "worse_query_count": len(worse),
        "better_or_equal_query_count": len(better_or_equal),
        "aggregate": aggregate,
        "worse_query_aggregate": {
            key: _mean_or_zero([row[key] for row in worse]) for key in numeric_keys
        },
        "better_or_equal_query_aggregate": {
            key: _mean_or_zero([row[key] for row in better_or_equal]) for key in numeric_keys
        },
        "per_query": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: result[key] for key in result if key != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
