#!/usr/bin/env python
"""Summarize deployment retrieval ranks from a discrete-oracle dump."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_discrete_decision_oracles import nearest_gt_targets, project_points


def load_rows(dump_dir, radius):
    dump_dir = Path(dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    with np.load(dump_dir / manifest["landmark_bank"]) as loaded:
        landmark_xyz = np.asarray(loaded["landmark_xyz"], dtype=np.float64)
    rows = {}
    for query_file in manifest["query_files"]:
        with np.load(dump_dir / query_file, allow_pickle=False) as loaded:
            query = {key: np.asarray(loaded[key]) for key in loaded.files}
        image_name = str(query["image_name"].item())
        keypoint_xy = np.asarray(query["keypoint_xy"], dtype=np.float64) + 0.5
        candidate = np.asarray(query["topk_landmark_idx"], dtype=np.int64)
        projected, _, projection_valid = project_points(
            landmark_xyz,
            np.asarray(query["K"], dtype=np.float64),
            np.asarray(query["gt_pose_w2c"], dtype=np.float64),
        )
        visible = np.asarray(query["render_visible_bank"], dtype=bool)
        width = int(query["width"])
        height = int(query["height"])
        valid = (
            visible
            & projection_valid
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        _, nearest_distance = nearest_gt_targets(
            keypoint_xy, projected, valid, float(radius)
        )
        matchable = np.isfinite(nearest_distance) & (nearest_distance <= float(radius))
        distance = np.linalg.norm(
            keypoint_xy[:, None] - projected[candidate], axis=2
        )
        positive = valid[candidate] & np.isfinite(distance) & (distance <= float(radius))
        first = np.argmax(positive, axis=1)
        found = positive.any(axis=1)
        rank = np.where(found, first + 1, candidate.shape[1] + 1)
        rows[image_name] = {
            "matchable": matchable,
            "rank": rank,
            "positive": positive,
        }
    return rows


def rank_band(rank):
    rank = np.asarray(rank)
    result = np.full(rank.shape, 3, dtype=np.int64)
    result[rank == 1] = 0
    result[(rank >= 2) & (rank <= 4)] = 1
    result[(rank >= 5) & (rank <= 32)] = 2
    return result


def summarize(rows):
    matchable = np.concatenate([value["matchable"] for value in rows.values()])
    rank = np.concatenate([value["rank"] for value in rows.values()])
    valid_rank = rank[matchable]
    topk = next(iter(rows.values()))["positive"].shape[1]
    summary = {
        "query_count": len(rows),
        "row_count": int(rank.size),
        "matchable_count": int(matchable.sum()),
        "matchable_fraction": float(matchable.mean()),
        "rank_observation_limit": int(topk),
    }
    for budget in (1, 2, 4, 8, 16, 32):
        summary[f"recall_at_{budget}"] = float((valid_rank <= budget).mean())
    found = valid_rank <= topk
    reciprocal = np.zeros(valid_rank.shape, dtype=np.float64)
    reciprocal[found] = 1.0 / valid_rank[found]
    summary["mrr_lower_bound"] = float(reciprocal.mean())
    summary["mrr_upper_bound"] = float(
        (reciprocal + (~found) / float(topk + 1)).mean()
    )
    bands = rank_band(valid_rank)
    summary["rank_band_counts"] = {
        "rank1": int((bands == 0).sum()),
        "rank2_4": int((bands == 1).sum()),
        "rank5_32": int((bands == 2).sum()),
        "rank33_plus": int((bands == 3).sum()),
    }
    return summary


def compare(reference, current):
    names = sorted(set(reference) & set(current))
    old_rank = []
    new_rank = []
    for name in names:
        old = reference[name]
        new = current[name]
        if old["rank"].shape != new["rank"].shape:
            raise ValueError(f"frontend row count changed for {name}")
        shared_matchable = old["matchable"] & new["matchable"]
        old_rank.append(old["rank"][shared_matchable])
        new_rank.append(new["rank"][shared_matchable])
    old_rank = np.concatenate(old_rank)
    new_rank = np.concatenate(new_rank)
    old_band = rank_band(old_rank)
    new_band = rank_band(new_rank)
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (old_band, new_band), 1)
    old_clean = old_rank == 1
    return {
        "shared_query_count": len(names),
        "shared_matchable_count": int(old_rank.size),
        "clean_top1_count": int(old_clean.sum()),
        "clean_top1_retention": float(
            (new_rank[old_clean] == 1).mean() if old_clean.any() else 0.0
        ),
        "rank_improved_fraction": float((new_rank < old_rank).mean()),
        "rank_worsened_fraction": float((new_rank > old_rank).mean()),
        "transition_rows": ["rank1", "rank2_4", "rank5_32", "rank33_plus"],
        "transition_cols": ["rank1", "rank2_4", "rank5_32", "rank33_plus"],
        "transition_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--reference_dump_dir")
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    current = load_rows(args.dump_dir, args.radius)
    result = {"current": summarize(current)}
    if args.reference_dump_dir:
        reference = load_rows(args.reference_dump_dir, args.radius)
        result["reference"] = summarize(reference)
        result["transition"] = compare(reference, current)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
