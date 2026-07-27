#!/usr/bin/env python3
"""Build query-specific functional statistics for every active LaFGS anchor."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.run_lafgs_alternating_structure import _deployment_valid_mask
from utils.pose_utils import solve_pose


def _project_candidates(
    xyz: torch.Tensor,
    indices: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
    keypoints: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = xyz[indices].float()
    pose = pose_w2c.float()
    camera = points @ pose[:3, :3].T + pose[:3, 3]
    projected = camera @ K.float().T
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    errors = torch.linalg.norm(uv - keypoints[:, None].float(), dim=-1)
    valid = camera[..., 2] > 1e-6
    errors = torch.where(valid, errors, torch.full_like(errors, 1e6))
    return errors, camera[..., 2]


def _increment_unique(target: torch.Tensor, indices: torch.Tensor) -> None:
    unique = torch.unique(indices.long())
    target.index_add_(0, unique, torch.ones_like(unique, dtype=target.dtype))


def _set_query_support(
    bitset: torch.Tensor, indices: torch.Tensor, query_index: int
) -> None:
    unique = torch.unique(indices.long())
    byte = int(query_index) // 8
    mask = 1 << (int(query_index) % 8)
    bitset[unique, byte] |= mask


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--deployment-mask-cache", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--clean-radius-px", type=float, default=4.0)
    parser.add_argument("--harm-radius-px", type=float, default=12.0)
    parser.add_argument("--grid-rows", type=int, default=4)
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Functional statistics require CUDA matching")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    anchor_map = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    query_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    query_cache = query_payload.get("queries", query_payload)
    names = list(query_cache)
    selected_queries = [
        index
        for index in range(len(names))
        if index % args.num_shards == args.shard_index
    ]
    deployment_masks = None
    if args.deployment_mask_cache:
        with Path(args.deployment_mask_cache).open("rb") as handle:
            deployment_masks = pickle.load(handle)

    xyz = torch.as_tensor(anchor_map["anchor_xyz"]).float()
    features = F.normalize(
        torch.as_tensor(anchor_map["anchor_features"]).float(), dim=1
    ).cuda()
    anchor_count = int(xyz.shape[0])
    sequence_names = sorted({name.split("/", 1)[0] for name in names})
    sequence_to_index = {
        sequence: index for index, sequence in enumerate(sequence_names)
    }
    counters = {
        key: torch.zeros(anchor_count, dtype=torch.int64)
        for key in (
            "winner_count",
            "clean_winner_count",
            "harm_winner_count",
            "inlier_count",
            "clean_topk_count",
            "clean_topk_query_count",
            "unique_topk_count",
            "image_bin_support_count",
        )
    }
    sequence_clean_support = torch.zeros(
        anchor_count, len(sequence_names), dtype=torch.int32
    )
    support_bits = torch.zeros(
        anchor_count, (len(names) + 7) // 8, dtype=torch.uint8
    )
    query_diagnostics = []

    for completed, query_index in enumerate(selected_queries, start=1):
        name = names[query_index]
        cached = query_cache[name]
        valid = _deployment_valid_mask(cached, name, deployment_masks)
        query_rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"])[query_rows].float(),
            dim=1,
        ).cuda()
        top_scores, top_indices = torch.topk(
            descriptors @ features.T,
            k=min(int(args.topk), anchor_count),
            dim=1,
        )
        top_scores = top_scores.cpu()
        top_indices = top_indices.cpu()
        keypoints = (
            torch.as_tensor(cached["native_keypoints"])[query_rows].float()
            + float(cached.get("pixel_center_offset", 0.5))
        )
        errors, depths = _project_candidates(
            xyz,
            top_indices,
            torch.as_tensor(cached["native_K"]),
            torch.as_tensor(cached["pose_w2c"]),
            keypoints,
        )
        clean = errors <= float(args.clean_radius_px)
        top1 = top_indices[:, 0]
        top1_error = errors[:, 0]
        counters["winner_count"].index_add_(
            0, top1, torch.ones_like(top1, dtype=torch.int64)
        )
        clean_top1 = top1_error <= float(args.clean_radius_px)
        harm_top1 = top1_error > float(args.harm_radius_px)
        counters["clean_winner_count"].index_add_(
            0,
            top1[clean_top1],
            torch.ones_like(top1[clean_top1], dtype=torch.int64),
        )
        counters["harm_winner_count"].index_add_(
            0,
            top1[harm_top1],
            torch.ones_like(top1[harm_top1], dtype=torch.int64),
        )
        clean_indices = top_indices[clean]
        counters["clean_topk_count"].index_add_(
            0,
            clean_indices,
            torch.ones_like(clean_indices, dtype=torch.int64),
        )
        _increment_unique(
            counters["clean_topk_query_count"], clean_indices
        )
        _set_query_support(support_bits, clean_indices, query_index)
        unique_rows = clean.sum(dim=1) == 1
        unique_indices = top_indices[unique_rows][
            clean[unique_rows]
        ]
        counters["unique_topk_count"].index_add_(
            0,
            unique_indices,
            torch.ones_like(unique_indices, dtype=torch.int64),
        )
        sequence_index = sequence_to_index[name.split("/", 1)[0]]
        unique_clean_query = torch.unique(clean_indices)
        sequence_clean_support[
            unique_clean_query, sequence_index
        ] += 1

        height, width = (
            int(value) for value in cached["native_input_hw"]
        )
        grid_x = torch.clamp(
            (keypoints[:, 0] * args.grid_cols / max(width, 1)).long(),
            0,
            args.grid_cols - 1,
        )
        grid_y = torch.clamp(
            (keypoints[:, 1] * args.grid_rows / max(height, 1)).long(),
            0,
            args.grid_rows - 1,
        )
        grid_id = grid_y * args.grid_cols + grid_x
        for cell in torch.unique(grid_id).tolist():
            rows = (grid_id == cell)[:, None] & clean
            _increment_unique(
                counters["image_bin_support_count"],
                top_indices[rows],
            )

        pose, inliers = solve_pose(
            keypoints.numpy(),
            xyz[top1].numpy(),
            torch.as_tensor(cached["native_K"]).numpy(),
            solver="poselib",
            reprojection_error=12.0,
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            scores=top_scores[:, 0].numpy(),
            ransac_seed=int(args.seed) + query_index,
        )
        inliers = torch.as_tensor(inliers, dtype=torch.long).reshape(-1)
        inliers = inliers[(inliers >= 0) & (inliers < top1.numel())]
        counters["inlier_count"].index_add_(
            0,
            top1[inliers],
            torch.ones_like(inliers, dtype=torch.int64),
        )
        query_diagnostics.append(
            {
                "query_index": query_index,
                "image_name": name,
                "keypoint_count": int(query_rows.numel()),
                "clean_top1_rate": float(clean_top1.float().mean()),
                "clean_topk_rate": float(clean.any(dim=1).float().mean()),
                "unique_topk_rate": float(unique_rows.float().mean()),
                "inlier_count": int(inliers.numel()),
                "mean_top1_score": float(top_scores[:, 0].mean()),
                "mean_top1_depth_m": float(depths[:, 0].mean()),
            }
        )
        if completed % 25 == 0 or completed == len(selected_queries):
            print(
                f"Shard {args.shard_index}: {completed}/"
                f"{len(selected_queries)} queries",
                flush=True,
            )

    output = {
        "schema": "lafgs_anchor_function_stats_shard",
        "version": 1,
        "anchor_map": str(Path(args.anchor_map).resolve()),
        "query_cache": str(Path(args.query_cache).resolve()),
        "anchor_count": anchor_count,
        "query_count_total": len(names),
        "query_indices": torch.as_tensor(selected_queries, dtype=torch.long),
        "sequence_names": sequence_names,
        "source_primitive_ids": torch.as_tensor(
            anchor_map["source_primitive_ids"]
        ).long(),
        "track_cluster_ids": torch.as_tensor(
            anchor_map["track_cluster_ids"]
        ).long(),
        "anchor_type": torch.as_tensor(anchor_map["anchor_type"]).to(
            torch.int8
        ),
        "sequence_clean_support": sequence_clean_support,
        "query_support_bits": support_bits,
        "query_diagnostics": query_diagnostics,
        "config": vars(args),
        **counters,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    summary = {
        "anchor_count": anchor_count,
        "query_count": len(selected_queries),
        "never_winner": int((counters["winner_count"] == 0).sum()),
        "used_but_never_clean": int(
            (
                (counters["winner_count"] > 0)
                & (counters["clean_winner_count"] == 0)
            ).sum()
        ),
        "zero_unique_support": int(
            (counters["unique_topk_count"] == 0).sum()
        ),
        "inlier_supported": int((counters["inlier_count"] > 0).sum()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
