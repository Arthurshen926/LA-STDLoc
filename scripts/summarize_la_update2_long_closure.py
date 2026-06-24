#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import warnings
from pathlib import Path

import torch

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)


EVENT_RE = re.compile(
    r"\[Topology\]\s+iter=(?P<iter>\d+).*?"
    r"physical_prune=(?P<physical>\d+).*?"
    r"requested_split=(?P<requested>\d+).*?"
    r"children_added=(?P<children>\d+).*?"
    r"points=(?P<before>\d+)->(?P<after>\d+)"
)


def latest_result_for_model(results_root, model_path, iteration):
    model_path = os.path.realpath(model_path)
    prefix = f"phase-v03-topology-{iteration}-"
    best = None
    for summary_path in glob.glob(str(Path(results_root) / "**" / "summary.json"), recursive=True):
        parent = Path(summary_path).parent.name
        if not parent.startswith(prefix):
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


def parse_tag(tag):
    parts = tag.split("_")
    if tag.startswith("core_") and len(parts) >= 3:
        steps = int(parts[-1])
        mode = "_".join(parts[1:-1])
        return "core", mode, steps
    if tag.startswith("prune_") and len(parts) >= 3:
        steps = int(parts[-1])
        mode = "physical_prune_only"
        return "prune", "_".join(parts[1:-1]), steps
    return "unknown", tag, None


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


def topology_log_path(root, scene, tag, train_seed, query_split_seed, legacy_seed_layout):
    if legacy_seed_layout:
        return root / "logs" / f"{scene}_seed{query_split_seed}_{tag}.log"
    return root / "logs" / f"{scene}_train{train_seed}_query{query_split_seed}_{tag}.log"


def read_ply_vertex_count(path):
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return None


def read_loc_state_stats(model_path, iteration, initial_iteration=None):
    state_path = Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "loc_state.pt"
    if not state_path.exists():
        return {}
    state = torch.load(state_path, map_location="cpu", mmap=True)
    loc_opacity = state.get("loc_opacity")
    n = int(loc_opacity.shape[0]) if torch.is_tensor(loc_opacity) else None
    stats = {"point_count": n}
    parent = state.get("loc_parent_node_id")
    if torch.is_tensor(parent):
        stats["child_rows"] = int((parent >= 0).sum().item())
    source_xyz = state.get("loc_source_xyz")
    current_xyz = state.get("loc_current_xyz")
    if torch.is_tensor(source_xyz) and torch.is_tensor(current_xyz) and source_xyz.shape == current_xyz.shape:
        dist = torch.linalg.norm(current_xyz.float() - source_xyz.float(), dim=-1)
        stats["source_distance_mean"] = float(dist.mean().item())
        stats["source_distance_max"] = float(dist.max().item())
    if torch.is_tensor(loc_opacity):
        activated = torch.sigmoid(loc_opacity.float())
        stats["loc_opacity_mean"] = float(activated.mean().item())
        stats["loc_opacity_min"] = float(activated.min().item())
    if initial_iteration is not None:
        initial_count = read_ply_vertex_count(
            Path(model_path) / "point_cloud" / f"iteration_{initial_iteration}" / "point_cloud.ply"
        )
        if initial_count is not None and n is not None:
            stats["point_count_initial"] = initial_count
            stats["point_count_delta"] = int(n - initial_count)
    return stats


def read_remap_summary(model_path):
    path = Path(model_path) / "detector_topology" / "remap_summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        "remap_missing": data.get("missing_count"),
        "remap_source_distance_mean": data.get("remap_source_distance_mean"),
        "remap_source_distance_max": data.get("remap_source_distance_max"),
    }


def read_event_stats(log_path):
    stats = {
        "topology_events": 0,
        "physical_prune_total": 0,
        "physical_prune_max": 0,
        "requested_split_total": 0,
        "children_added_total": 0,
    }
    if not log_path.exists():
        return stats
    for line in log_path.read_text(errors="replace").splitlines():
        m = EVENT_RE.search(line)
        if not m:
            continue
        stats["topology_events"] += 1
        physical = int(m.group("physical"))
        requested = int(m.group("requested"))
        children = int(m.group("children"))
        stats["physical_prune_total"] += physical
        stats["physical_prune_max"] = max(stats["physical_prune_max"], physical)
        stats["requested_split_total"] += requested
        stats["children_added_total"] += children
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/pool/sqy/stdloc_la_update2_long_closure")
    parser.add_argument("--results_root", default="/root/STDLoc/results")
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
        family, mode, steps = parse_tag(tag)
        if steps is None:
            continue
        expected_iteration = args.v03_iteration + int(steps)
        iteration = expected_iteration
        state_path = model_path / "point_cloud" / f"iteration_{iteration}" / "loc_state.pt"
        if not state_path.exists():
            continue
        log_path = topology_log_path(root, scene, tag, train_seed, query_split_seed, legacy_seed_layout)
        result_path, result = latest_result_for_model(args.results_root, str(model_path), iteration)
        sparse = (result or {}).get("sparse", {})
        row = {
            "family": family,
            "mode": mode,
            "tag": tag,
            "scene": scene,
            "seed": seed,
            "train_seed": train_seed,
            "query_split_seed": query_split_seed,
            "legacy_seed_layout": legacy_seed_layout,
            "steps": steps,
            "iteration": iteration,
            "model_path": str(model_path),
            "summary_path": result_path,
            "median_ae": sparse.get("median_ae"),
            "median_te": sparse.get("median_te"),
            "recall_5cm_5d": sparse.get("recall_5cm_5d"),
            "recall_2cm_2d": sparse.get("recall_2cm_2d"),
            "avg_inliers": sparse.get("avg_inliers"),
        }
        row.update(read_loc_state_stats(model_path, iteration, initial_iteration=args.v03_iteration))
        row.update(read_remap_summary(model_path))
        row.update(read_event_stats(log_path))
        rows.append(row)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
