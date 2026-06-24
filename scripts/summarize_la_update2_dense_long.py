#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)


ITER_RE = re.compile(r"\[ITER\s+(?P<iter>\d+)\].*?\bloc\s+(?P<loc>[-+0-9.eE]+)")
PROGRESS_LOC_RE = re.compile(r"\bLoc=(?P<loc>[-+0-9.eE]+)")


def latest_result_for_model(results_root, model_path, iteration):
    model_path = os.path.realpath(model_path)
    prefix = f"phase-densekl-"
    best = None
    for summary_path in glob.glob(str(Path(results_root) / "**" / "summary.json"), recursive=True):
        parent = Path(summary_path).parent.name
        if not parent.startswith(prefix) or f"-{iteration}-" not in parent:
            continue
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        if os.path.realpath(summary.get("model_path", "")) != model_path:
            continue
        mtime = os.path.getmtime(summary_path)
        if best is None or mtime > best[0]:
            best = (mtime, summary_path, summary)
    if best is None:
        return None, None
    return best[1], best[2]


def latest_source_result(results_root, source_model, iteration):
    source_model = os.path.realpath(source_model)
    best = None
    for summary_path in glob.glob(str(Path(results_root) / "**" / "summary.json"), recursive=True):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception:
            continue
        if os.path.realpath(summary.get("model_path", "")) != source_model:
            continue
        parent = Path(summary_path).parent.name
        if f"-{iteration}-" not in parent:
            continue
        mtime = os.path.getmtime(summary_path)
        if best is None or mtime > best[0]:
            best = (mtime, summary_path, summary)
    if best is None:
        return None, None
    return best[1], best[2]


def parse_model_path(path, root):
    rel = Path(path).relative_to(root / "models")
    tag = rel.parts[0]
    scene = rel.parts[1]
    seed_part = rel.parts[2]
    if seed_part.startswith("seed_"):
        query_split_seed = int(seed_part.replace("seed_", ""))
        return {
            "tag": tag,
            "scene": scene,
            "seed": query_split_seed,
            "train_seed": None,
            "query_split_seed": query_split_seed,
            "legacy_seed_layout": True,
        }
    if seed_part.startswith("train_seed_") and len(rel.parts) >= 5 and rel.parts[3].startswith("query_split_"):
        train_seed = int(seed_part.replace("train_seed_", ""))
        query_split_seed = int(rel.parts[3].replace("query_split_", ""))
        return {
            "tag": tag,
            "scene": scene,
            "seed": query_split_seed,
            "train_seed": train_seed,
            "query_split_seed": query_split_seed,
            "legacy_seed_layout": False,
        }
    raise ValueError(f"Unsupported model path layout: {path}")


def iter_model_paths(root):
    patterns = [
        "models/*/*/seed_*/*",
        "models/*/*/train_seed_*/query_split_*/*",
    ]
    seen = set()
    for pattern in patterns:
        for model_path in sorted(root.glob(pattern)):
            real = os.path.realpath(model_path)
            if real in seen:
                continue
            seen.add(real)
            yield model_path


def source_model_for(v03_root, scene, train_seed, query_split_seed):
    legacy = Path(v03_root) / scene / f"seed_{query_split_seed}" / f"{scene}_v03"
    if train_seed is None:
        return legacy
    source = Path(v03_root) / scene / f"train_seed_{train_seed}" / f"query_split_{query_split_seed}" / f"{scene}_v03"
    if (source / "point_cloud").exists() or not legacy.exists():
        return source
    return legacy


def dense_log_path(root, scene, tag, train_seed, query_split_seed, legacy_seed_layout):
    if legacy_seed_layout:
        return root / "logs" / f"{scene}_seed{query_split_seed}_{tag}.log"
    return root / "logs" / f"{scene}_train{train_seed}_query{query_split_seed}_{tag}.log"


def dense_cache_path(root, scene, train_seed, query_split_seed, legacy_seed_layout, iteration):
    if legacy_seed_layout:
        return root / "cache" / f"{scene}_seed{query_split_seed}_dense_pose_cache_{iteration}.pt"
    return root / "cache" / f"{scene}_train{train_seed}_query{query_split_seed}_dense_pose_cache_{iteration}.pt"


def read_log_stats(log_path):
    stats = {
        "train_iter_count": 0,
        "loc_positive_iter_count": 0,
        "loc_loss_mean": None,
        "loc_loss_max": None,
    }
    if not log_path.exists():
        return stats
    loc_values = []
    for line in log_path.read_text(errors="replace").splitlines():
        m = ITER_RE.search(line)
        if m:
            loc_values.append(float(m.group("loc")))
        for progress_match in PROGRESS_LOC_RE.finditer(line.replace("\r", "\n")):
            loc_values.append(float(progress_match.group("loc")))
    if loc_values:
        arr = np.asarray(loc_values, dtype=float)
        stats.update(
            {
                "train_iter_count": int(arr.shape[0]),
                "loc_positive_iter_count": int((arr > 0).sum()),
                "loc_loss_mean": float(arr.mean()),
                "loc_loss_max": float(arr.max()),
            }
        )
    return stats


def read_pose_cache_stats(cache_path):
    if not cache_path.exists():
        return {}
    cache = torch.load(cache_path, map_location="cpu")
    rows = []
    for value in cache.values():
        rows.append(
            (
                value.get("te"),
                value.get("ae"),
                value.get("dense_te"),
                value.get("dense_ae"),
                value.get("inliers"),
                value.get("dense_inliers"),
            )
        )
    if not rows:
        return {"pose_cache_count": 0}
    arr = np.asarray(rows, dtype=float)
    te, ae, dense_te, dense_ae, inliers, dense_inliers = arr.T
    valid = np.isfinite(te) & np.isfinite(ae) & np.isfinite(dense_te) & np.isfinite(dense_ae)
    if not valid.any():
        return {"pose_cache_count": int(arr.shape[0]), "pose_cache_valid_count": 0}
    both = valid & (dense_te < te) & (dense_ae < ae)
    either = valid & ((dense_te < te) | (dense_ae < ae))
    return {
        "pose_cache_count": int(arr.shape[0]),
        "pose_cache_valid_count": int(valid.sum()),
        "dense_pose_both_better_count": int(both.sum()),
        "dense_pose_both_better_ratio": float(both.sum() / valid.sum()),
        "dense_pose_either_better_count": int(either.sum()),
        "dense_pose_either_better_ratio": float(either.sum() / valid.sum()),
        "dense_pose_te_improvement_median": float(np.median((te - dense_te)[valid])),
        "dense_pose_ae_improvement_median": float(np.median((ae - dense_ae)[valid])),
        "sparse_pose_te_median": float(np.median(te[valid])),
        "dense_pose_te_median": float(np.median(dense_te[valid])),
    }


def point_iterations(model_path, source_iteration):
    point_cloud = Path(model_path) / "point_cloud"
    if not point_cloud.exists():
        return []
    iterations = []
    for path in point_cloud.glob("iteration_*"):
        if not (path / "point_cloud.ply").exists():
            continue
        try:
            iteration = int(path.name.replace("iteration_", ""))
        except ValueError:
            continue
        if iteration > source_iteration:
            iterations.append(iteration)
    return sorted(iterations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/pool/sqy/stdloc_la_update2_dense_long_v1")
    parser.add_argument("--results_root", default="/root/STDLoc/results")
    parser.add_argument("--v03_root", default="/mnt/pool/sqy/stdloc_la_v03_full_length")
    parser.add_argument("--v03_iteration", type=int, default=30500)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for model_path in iter_model_paths(root):
        if not (model_path / "point_cloud").is_dir():
            continue
        parsed = parse_model_path(model_path, root)
        tag = parsed["tag"]
        scene = parsed["scene"]
        seed = parsed["seed"]
        train_seed = parsed["train_seed"]
        query_split_seed = parsed["query_split_seed"]
        legacy_seed_layout = parsed["legacy_seed_layout"]
        source_model = source_model_for(args.v03_root, scene, train_seed, query_split_seed)
        source_summary_path, source_summary = latest_source_result(
            args.results_root,
            str(source_model),
            args.v03_iteration,
        )
        source_sparse = (source_summary or {}).get("sparse", {})
        log_path = dense_log_path(root, scene, tag, train_seed, query_split_seed, legacy_seed_layout)
        cache_path = dense_cache_path(root, scene, train_seed, query_split_seed, legacy_seed_layout, args.v03_iteration)
        common = {
            "tag": tag,
            "scene": scene,
            "seed": seed,
            "train_seed": train_seed,
            "query_split_seed": query_split_seed,
            "legacy_seed_layout": legacy_seed_layout,
            "model_path": str(model_path),
            "source_model_path": str(source_model),
            "source_summary_path": source_summary_path,
            "source_recall_5cm_5d": source_sparse.get("recall_5cm_5d"),
            "source_recall_2cm_2d": source_sparse.get("recall_2cm_2d"),
            "source_median_te": source_sparse.get("median_te"),
            "source_median_ae": source_sparse.get("median_ae"),
        }
        common.update(read_log_stats(log_path))
        common.update(read_pose_cache_stats(cache_path))
        for iteration in point_iterations(model_path, args.v03_iteration):
            summary_path, summary = latest_result_for_model(args.results_root, str(model_path), iteration)
            sparse = (summary or {}).get("sparse", {})
            row = dict(common)
            row.update(
                {
                    "iteration": iteration,
                    "steps": iteration - args.v03_iteration,
                    "summary_path": summary_path,
                    "median_ae": sparse.get("median_ae"),
                    "median_te": sparse.get("median_te"),
                    "recall_5cm_5d": sparse.get("recall_5cm_5d"),
                    "recall_2cm_2d": sparse.get("recall_2cm_2d"),
                    "avg_inliers": sparse.get("avg_inliers"),
                }
            )
            if row["recall_5cm_5d"] is not None and row["source_recall_5cm_5d"] is not None:
                row["delta_source_recall_5cm_5d"] = row["recall_5cm_5d"] - row["source_recall_5cm_5d"]
            if row["recall_2cm_2d"] is not None and row["source_recall_2cm_2d"] is not None:
                row["delta_source_recall_2cm_2d"] = row["recall_2cm_2d"] - row["source_recall_2cm_2d"]
            if row["median_te"] is not None and row["source_median_te"] is not None:
                row["delta_source_median_te"] = row["median_te"] - row["source_median_te"]
            rows.append(row)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
