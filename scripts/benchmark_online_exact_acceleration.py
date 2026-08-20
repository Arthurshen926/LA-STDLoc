"""Benchmark exact online acceleration against the pre-acceleration oracle.

This is deliberately an artifact replay, not an accuracy experiment.  It
checks exact IDs, scores and PoseLib poses while measuring matcher-only latency
and a small real-image single-query panel.  The reported streaming number is
an overlap boundary, not a claim that the evaluator already pipelines frames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from data.datasets import ColmapDataset
import localization.localizer as localizer_module
from localization.localizer import SparseLocalizer
from localization.matcher import Top1Matches, global_cosine_top1


@torch.inference_mode()
def legacy_global_cosine_top1(
    query_descriptors: torch.Tensor,
    anchor_descriptors: torch.Tensor,
    *,
    chunk_size: int = 8192,
    **_ignored,
) -> Top1Matches:
    """The exact kernel used immediately before this acceleration change."""
    query = F.normalize(query_descriptors.float(), dim=1)
    best_scores = query.new_full((query.shape[0], 1), -torch.inf)
    best_indices = torch.zeros(
        (query.shape[0], 1), dtype=torch.long, device=query.device
    )
    for start in range(0, anchor_descriptors.shape[0], max(int(chunk_size), 1)):
        stop = min(start + max(int(chunk_size), 1), anchor_descriptors.shape[0])
        anchors = F.normalize(anchor_descriptors[start:stop].float(), dim=1)
        scores = query @ anchors.T
        indices = torch.arange(start, stop, device=query.device)[None].expand(
            query.shape[0], -1
        )
        merged_scores = torch.cat((best_scores, scores), dim=1)
        merged_indices = torch.cat((best_indices, indices), dim=1)
        best_scores, positions = torch.topk(merged_scores, 1, dim=1)
        best_indices = torch.gather(merged_indices, 1, positions)
    return Top1Matches(
        torch.arange(query.shape[0], device=query.device),
        best_indices[:, 0],
        best_scores[:, 0],
    )


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p90_ms": float(np.percentile(array, 90)),
    }


def _timed_matcher(function, queries, bank, **kwargs):
    outputs = []
    elapsed = []
    for query in queries:
        torch.cuda.synchronize(query.device)
        started = time.perf_counter()
        outputs.append(function(query, bank, **kwargs))
        torch.cuda.synchronize(query.device)
        elapsed.append((time.perf_counter() - started) * 1000.0)
    return outputs, elapsed


def _run_localizer(localizer, dataset, cameras):
    outputs = []
    for camera in cameras:
        outputs.append(
            localizer.localize(
                dataset.load_image(camera),
                fov_x=camera.fov_x,
                fov_y=camera.fov_y,
                valid_mask=dataset.valid_mask(camera),
            )
        )
    return outputs


def _runtime_summary(outputs):
    return {
        key: _stats([float(row.runtime_ms[key]) for row in outputs])
        for key in ("frontend_ms", "matching_ms", "ransac_ms", "total_ms")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--query-limit", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.query_limit < 1 or args.warmup < 1:
        parser.error("query-limit and warmup must be positive")

    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("the online acceleration benchmark requires CUDA")
    map_state = torch.load(args.map, map_location="cpu", weights_only=False)
    raw_bank = torch.as_tensor(map_state["anchor_features"], device=device).float()
    historical_bank = F.normalize(raw_bank, dim=1)
    cached_bank = F.normalize(historical_bank, dim=1)
    cache = torch.load(args.feature_cache, map_location="cpu", weights_only=False)
    cache_rows = list(cache["queries"].values())
    needed = args.warmup + args.query_limit
    if len(cache_rows) < needed:
        raise ValueError("feature cache has too few rows for the requested replay")
    cache_queries = [
        torch.as_tensor(row["native_descriptors"], device=device).float()
        for row in cache_rows[:needed]
    ]
    for query in cache_queries[: args.warmup]:
        legacy_global_cosine_top1(query, historical_bank)
        global_cosine_top1(
            query, cached_bank, anchor_descriptors_normalized=True
        )
    legacy_matches, legacy_ms = _timed_matcher(
        legacy_global_cosine_top1,
        cache_queries[args.warmup :],
        historical_bank,
    )
    exact_matches, exact_ms = _timed_matcher(
        global_cosine_top1,
        cache_queries[args.warmup :],
        cached_bank,
        anchor_descriptors_normalized=True,
    )
    matcher_ids_equal = all(
        torch.equal(old.anchor_indices, new.anchor_indices)
        for old, new in zip(legacy_matches, exact_matches)
    )
    matcher_scores_equal = all(
        torch.equal(old.scores, new.scores)
        for old, new in zip(legacy_matches, exact_matches)
    )

    dataset = ColmapDataset(args.dataset, images=args.images)
    cameras = dataset.split("test")
    if len(cameras) < needed:
        raise ValueError("test split has too few rows for the requested replay")
    warmup_cameras = cameras[: args.warmup]
    measured_cameras = cameras[args.warmup : needed]

    profile_localizer = SparseLocalizer(
        args.map, args.metric_state, device=device, profile_mode=True
    )
    profile_localizer.anchor_features = historical_bank
    localizer_module.global_cosine_top1 = legacy_global_cosine_top1
    _run_localizer(profile_localizer, dataset, warmup_cameras)
    legacy_outputs = _run_localizer(profile_localizer, dataset, measured_cameras)

    profile_localizer.anchor_features = cached_bank
    localizer_module.global_cosine_top1 = global_cosine_top1
    _run_localizer(profile_localizer, dataset, warmup_cameras)
    exact_outputs = _run_localizer(profile_localizer, dataset, measured_cameras)

    deployment_localizer = SparseLocalizer(
        args.map, args.metric_state, device=device, profile_mode=False
    )
    _run_localizer(deployment_localizer, dataset, warmup_cameras)
    deployment_outputs = _run_localizer(
        deployment_localizer, dataset, measured_cameras
    )
    localizer_module.global_cosine_top1 = global_cosine_top1

    def rows_equal(field, first, second):
        return all(field(a, b) for a, b in zip(first, second))

    ids_equal = rows_equal(
        lambda a, b: torch.equal(a.matches.anchor_indices, b.matches.anchor_indices),
        legacy_outputs,
        exact_outputs,
    ) and rows_equal(
        lambda a, b: torch.equal(a.matches.anchor_indices, b.matches.anchor_indices),
        legacy_outputs,
        deployment_outputs,
    )
    scores_equal = rows_equal(
        lambda a, b: torch.equal(a.matches.scores, b.matches.scores),
        legacy_outputs,
        exact_outputs,
    ) and rows_equal(
        lambda a, b: torch.equal(a.matches.scores, b.matches.scores),
        legacy_outputs,
        deployment_outputs,
    )
    poses_equal = rows_equal(
        lambda a, b: np.array_equal(a.pose.pose_w2c, b.pose.pose_w2c),
        legacy_outputs,
        exact_outputs,
    ) and rows_equal(
        lambda a, b: np.array_equal(a.pose.pose_w2c, b.pose.pose_w2c),
        legacy_outputs,
        deployment_outputs,
    )
    overlap_cycles = [
        max(
            float(row.runtime_ms["frontend_ms"])
            + float(row.runtime_ms["matching_ms"]),
            float(row.runtime_ms["ransac_ms"]),
        )
        for row in deployment_outputs
    ]
    overlap_mean = float(np.mean(overlap_cycles))
    payload = {
        "schema": "lafgs_online_exact_acceleration_benchmark",
        "version": 1,
        "scene": args.scene,
        "anchor_count": int(cached_bank.shape[0]),
        "descriptor_rows_per_query": int(cache_queries[0].shape[0]),
        "measured_query_count": int(args.query_limit),
        "matcher": {
            "legacy": _stats(legacy_ms),
            "exact_accelerated": _stats(exact_ms),
            "speedup_mean": float(np.mean(legacy_ms) / np.mean(exact_ms)),
            "serial_queries_per_second": float(1000.0 / np.mean(exact_ms)),
            "anchor_ids_exact": matcher_ids_equal,
            "scores_exact": matcher_scores_equal,
        },
        "single_query": {
            "legacy_profile": _runtime_summary(legacy_outputs),
            "exact_profile": _runtime_summary(exact_outputs),
            "exact_deployment": _runtime_summary(deployment_outputs),
            "anchor_ids_exact": ids_equal,
            "scores_exact": scores_equal,
            "pose_matrices_exact": poses_equal,
        },
        "streaming_overlap_boundary": {
            "definition": "max(frontend_plus_matching, ransac); not yet pipelined",
            "mean_cycle_ms": overlap_mean,
            "frames_per_second": float(1000.0 / overlap_mean),
        },
    }
    if not all((matcher_ids_equal, matcher_scores_equal, ids_equal, scores_equal, poses_equal)):
        raise RuntimeError("online acceleration changed artifact replay outputs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
