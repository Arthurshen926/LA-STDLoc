#!/usr/bin/env python3
"""Compare two STDLoc discrete-oracle dumps layer by layer."""

import argparse
import hashlib
import json
import os

import numpy as np


def _sha256(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_dump(root):
    with open(os.path.join(root, "manifest.json")) as handle:
        manifest = json.load(handle)
    bank = dict(np.load(os.path.join(root, manifest["landmark_bank"])))
    queries = {}
    for filename in manifest["query_files"]:
        payload = dict(np.load(os.path.join(root, filename)))
        queries[str(payload["image_name"].item())] = payload
    return manifest, bank, queries


def _max_abs(left, right):
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _pair_set(payload, stage, bank_ids, inliers=False):
    key = f"{stage}_keypoint_idx"
    landmark_key = f"{stage}_landmark_idx"
    if key not in payload or landmark_key not in payload:
        return set()
    keypoints = payload[key].astype(np.int64)
    landmarks = payload[landmark_key].astype(np.int64)
    if inliers:
        selected = payload.get("hard_post_inliers", np.empty(0, dtype=np.int64))
        selected = selected.astype(np.int64)
        selected = selected[(selected >= 0) & (selected < keypoints.size)]
        keypoints = keypoints[selected]
        landmarks = landmarks[selected]
    return set(zip(keypoints.tolist(), bank_ids[landmarks].tolist()))


def _jaccard(left, right):
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def compare(left_root, right_root):
    _, left_bank, left_queries = _load_dump(left_root)
    _, right_bank, right_queries = _load_dump(right_root)
    left_ids = left_bank["source_gaussian_idx"].astype(np.int64)
    right_ids = right_bank["source_gaussian_idx"].astype(np.int64)
    common = sorted(set(left_queries) & set(right_queries))
    per_query = []
    for name in common:
        left = left_queries[name]
        right = right_queries[name]
        top1_left = left_ids[left["topk_landmark_idx"][:, 0]]
        top1_right = right_ids[right["topk_landmark_idx"][:, 0]]
        top1_equal = (
            float(np.mean(top1_left == top1_right))
            if top1_left.shape == top1_right.shape and top1_left.size else 0.0
        )
        raw_left = _pair_set(left, "matcher_raw", left_ids)
        raw_right = _pair_set(right, "matcher_raw", right_ids)
        pre_left = _pair_set(left, "hard_pre", left_ids)
        pre_right = _pair_set(right, "hard_pre", right_ids)
        post_left = _pair_set(left, "hard_post", left_ids)
        post_right = _pair_set(right, "hard_post", right_ids)
        inlier_left = _pair_set(left, "hard_post", left_ids, inliers=True)
        inlier_right = _pair_set(right, "hard_post", right_ids, inliers=True)
        per_query.append(
            {
                "image_name": name,
                "keypoint_xy_max_abs": _max_abs(left["keypoint_xy"], right["keypoint_xy"]),
                "keypoint_score_max_abs": _max_abs(
                    left["keypoint_detector_score"], right["keypoint_detector_score"]
                ),
                "top1_landmark_equal_ratio": top1_equal,
                "topk_score_max_abs": _max_abs(left["topk_scores"], right["topk_scores"]),
                "matcher_raw_jaccard": _jaccard(raw_left, raw_right),
                "pre_pnp_jaccard": _jaccard(pre_left, pre_right),
                "post_selector_jaccard": _jaccard(post_left, post_right),
                "ransac_inlier_jaccard": _jaccard(inlier_left, inlier_right),
                "pose_max_abs": _max_abs(left["pred_pose_w2c"], right["pred_pose_w2c"]),
            }
        )
    metric_keys = [key for key in per_query[0] if key != "image_name"] if per_query else []
    aggregate = {
        key: {
            "mean": float(np.mean([row[key] for row in per_query])),
            "max": float(np.max([row[key] for row in per_query])),
            "min": float(np.min([row[key] for row in per_query])),
        }
        for key in metric_keys
    }
    functional_identity = bool(
        len(left_queries) == len(right_queries) == len(common)
        and np.array_equal(left_ids, right_ids)
        and _max_abs(left_bank["landmark_xyz"], right_bank["landmark_xyz"]) == 0.0
        and all(
            row["keypoint_xy_max_abs"] == 0.0
            and row["keypoint_score_max_abs"] == 0.0
            and row["top1_landmark_equal_ratio"] == 1.0
            and row["matcher_raw_jaccard"] == 1.0
            and row["pre_pnp_jaccard"] == 1.0
            and row["post_selector_jaccard"] == 1.0
            and row["ransac_inlier_jaccard"] == 1.0
            and row["pose_max_abs"] == 0.0
            for row in per_query
        )
    )
    exact = bool(
        functional_identity
        and all(row["topk_score_max_abs"] == 0.0 for row in per_query)
    )
    return {
        "exact_identity": exact,
        "functional_identity": functional_identity,
        "left_query_count": len(left_queries),
        "right_query_count": len(right_queries),
        "common_query_count": len(common),
        "missing_from_left": sorted(set(right_queries) - set(left_queries)),
        "missing_from_right": sorted(set(left_queries) - set(right_queries)),
        "bank": {
            "left_ids_sha256": _sha256(left_ids),
            "right_ids_sha256": _sha256(right_ids),
            "ids_equal": bool(np.array_equal(left_ids, right_ids)),
            "xyz_max_abs": _max_abs(
                left_bank["landmark_xyz"], right_bank["landmark_xyz"]
            ),
            "render_xyz_max_abs": _max_abs(
                left_bank["render_xyz"], right_bank["render_xyz"]
            ),
        },
        "aggregate": aggregate,
        "per_query": per_query,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--output")
    parser.add_argument("--require_exact", action="store_true")
    args = parser.parse_args()
    result = compare(args.left, args.right)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(rendered + "\n")
    if args.require_exact and not result["exact_identity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
