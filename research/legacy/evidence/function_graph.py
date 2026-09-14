#!/usr/bin/env python3
"""Build a depth-legal keypoint-to-anchor graph for Active Map V4."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch
import torch.nn.functional as F

from features.raster_sampling import sample_raster_at_grid_uv

from evidence.projection import _project_candidates
from data.masks import deployment_valid_mask
from localization.pose_solver import solve_pose


def _sample_rendered_surface(cached: dict, keypoints: torch.Tensor):
    depth = sample_raster_at_grid_uv(
        torch.as_tensor(cached["native_depth"]).float(), keypoints
    )
    alpha = sample_raster_at_grid_uv(
        torch.as_tensor(cached["native_alpha"]).float(), keypoints
    )
    return depth, alpha


def _increment(target: torch.Tensor, indices: torch.Tensor) -> None:
    if indices.numel():
        target.index_add_(
            0,
            indices.long(),
            torch.ones_like(indices, dtype=target.dtype),
        )


def _load_raster_visibility(path: str):
    if not path:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("visibility", payload)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--deployment-mask-cache", default="")
    parser.add_argument("--raster-visibility-cache", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--depth-abs-tolerance-m", type=float, default=0.05)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.02)
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--strong-radius-px", type=float, default=2.0)
    parser.add_argument("--clean-radius-px", type=float, default=4.0)
    parser.add_argument("--ambiguous-radius-px", type=float, default=8.0)
    parser.add_argument("--pnp-reprojection-error-px", type=float, default=12.0)
    parser.add_argument("--harm-radius-px", type=float, default=12.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Functional graph construction requires CUDA")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    anchor_map = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    query_cache = payload.get("queries", payload)
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
    raster_visibility = _load_raster_visibility(
        args.raster_visibility_cache
    )

    xyz = torch.as_tensor(anchor_map["anchor_xyz"]).float()
    source_ids = torch.as_tensor(
        anchor_map["source_primitive_ids"]
    ).long()
    features = F.normalize(
        torch.as_tensor(anchor_map["anchor_features"]).float(), dim=1
    ).cuda()
    anchor_count = int(xyz.shape[0])
    topk = min(int(args.topk), anchor_count)
    counters = {
        key: torch.zeros(anchor_count, dtype=torch.int64)
        for key in (
            "candidate_opportunity_count",
            "winner_count",
            "legal_hit_2px_count",
            "legal_hit_4px_count",
            "legal_hit_8px_count",
            "legal_winner_2px_count",
            "legal_winner_4px_count",
            "solver_inlier_count",
            "solver_inlier_gtclean_2px_count",
            "solver_inlier_gtclean_4px_count",
            "harmful_solver_inlier_count",
        )
    }
    records = []
    diagnostics = []

    for completed, query_index in enumerate(selected_queries, start=1):
        name = names[query_index]
        cached = query_cache[name]
        valid = deployment_valid_mask(cached, name, deployment_masks)
        query_rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"])[
                query_rows
            ].float(),
            dim=1,
        ).cuda()
        scores, indices = torch.topk(
            descriptors @ features.T, k=topk, dim=1
        )
        scores = scores.cpu()
        indices = indices.cpu()
        keypoints = (
            torch.as_tensor(cached["native_keypoints"])[
                query_rows
            ].float()
            + float(cached.get("pixel_center_offset", 0.5))
        )
        errors, candidate_depth = _project_candidates(
            xyz,
            indices,
            torch.as_tensor(cached["native_K"]),
            torch.as_tensor(cached["pose_w2c"]),
            keypoints,
        )
        rendered_depth, rendered_alpha = _sample_rendered_surface(
            cached, keypoints
        )
        tolerance = float(args.depth_abs_tolerance_m) + (
            float(args.depth_rel_tolerance) * rendered_depth.abs()
        )
        surface_valid = (
            torch.isfinite(rendered_depth)
            & (rendered_depth > 0)
            & torch.isfinite(rendered_alpha)
            & (rendered_alpha >= float(args.alpha_min))
        )
        depth_legal = (
            surface_valid[:, None]
            & torch.isfinite(candidate_depth)
            & (
                (candidate_depth - rendered_depth[:, None]).abs()
                <= tolerance[:, None]
            )
        )
        raster_enabled = raster_visibility is not None
        if raster_enabled:
            query_visibility = torch.as_tensor(
                raster_visibility[name], dtype=torch.bool
            )
            if query_visibility.numel() == anchor_count:
                candidate_visible = query_visibility[indices]
            elif (
                source_ids.numel()
                and int(source_ids.max()) < query_visibility.numel()
            ):
                candidate_visible = query_visibility[source_ids[indices]]
            else:
                raise ValueError(
                    "raster visibility does not align with anchors or sources"
                )
            depth_legal &= candidate_visible
        legal2 = depth_legal & (errors <= float(args.strong_radius_px))
        legal4 = depth_legal & (errors <= float(args.clean_radius_px))
        legal8 = depth_legal & (errors <= float(args.ambiguous_radius_px))
        legal_flags = (
            depth_legal.to(torch.uint8)
            | (legal2.to(torch.uint8) << 1)
            | (legal4.to(torch.uint8) << 2)
            | (legal8.to(torch.uint8) << 3)
        )

        flat_indices = indices.reshape(-1)
        _increment(counters["candidate_opportunity_count"], flat_indices)
        _increment(counters["winner_count"], indices[:, 0])
        _increment(counters["legal_hit_2px_count"], indices[legal2])
        _increment(counters["legal_hit_4px_count"], indices[legal4])
        _increment(counters["legal_hit_8px_count"], indices[legal8])
        _increment(
            counters["legal_winner_2px_count"],
            indices[:, 0][legal2[:, 0]],
        )
        _increment(
            counters["legal_winner_4px_count"],
            indices[:, 0][legal4[:, 0]],
        )

        _, inliers = solve_pose(
            keypoints.numpy(),
            xyz[indices[:, 0]].numpy(),
            torch.as_tensor(cached["native_K"]).numpy(),
            solver="poselib",
            reprojection_error=float(args.pnp_reprojection_error_px),
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            scores=scores[:, 0].float().numpy(),
            ransac_seed=int(args.seed) + query_index,
        )
        inliers = torch.as_tensor(inliers, dtype=torch.long).reshape(-1)
        inliers = inliers[
            (inliers >= 0) & (inliers < query_rows.numel())
        ]
        solver_inlier = torch.zeros(
            query_rows.numel(), dtype=torch.bool
        )
        solver_inlier[inliers] = True
        top1 = indices[:, 0]
        _increment(counters["solver_inlier_count"], top1[inliers])
        _increment(
            counters["solver_inlier_gtclean_2px_count"],
            top1[solver_inlier & legal2[:, 0]],
        )
        _increment(
            counters["solver_inlier_gtclean_4px_count"],
            top1[solver_inlier & legal4[:, 0]],
        )
        harmful = solver_inlier & (
            (~depth_legal[:, 0])
            | (errors[:, 0] > float(args.harm_radius_px))
        )
        _increment(
            counters["harmful_solver_inlier_count"], top1[harmful]
        )
        global_keypoint_ids = (
            (torch.full_like(query_rows, query_index, dtype=torch.int64) << 16)
            | query_rows.to(torch.int64)
        )
        records.append(
            {
                "query_index": query_index,
                "query_rows": query_rows.to(torch.int32),
                "global_keypoint_ids": global_keypoint_ids,
                "top_indices": indices.to(torch.int32),
                "top_scores": scores.to(torch.float16),
                "legal_flags": legal_flags,
                "solver_inlier": solver_inlier,
                "harmful_solver_inlier": harmful,
            }
        )
        diagnostics.append(
            {
                "query_index": query_index,
                "image_name": name,
                "keypoint_count": int(query_rows.numel()),
                "surface_valid_rate": float(surface_valid.float().mean()),
                "depth_legal_top64_rate": float(
                    depth_legal.float().mean()
                ),
                "legal_top1_2px_rate": float(legal2[:, 0].float().mean()),
                "legal_top1_4px_rate": float(legal4[:, 0].float().mean()),
                "legal_top64_4px_rate": float(legal4.any(1).float().mean()),
                "solver_inlier_count": int(inliers.numel()),
                "harmful_solver_inlier_count": int(harmful.sum()),
            }
        )
        if completed % 25 == 0 or completed == len(selected_queries):
            print(
                f"Graph shard {args.shard_index}: {completed}/"
                f"{len(selected_queries)} queries",
                flush=True,
            )

    semantic_counters = {
        "legal_hit_strong_count": counters["legal_hit_2px_count"],
        "legal_hit_clean_count": counters["legal_hit_4px_count"],
        "legal_hit_ambiguous_count": counters["legal_hit_8px_count"],
        "legal_winner_strong_count": counters["legal_winner_2px_count"],
        "legal_winner_clean_count": counters["legal_winner_4px_count"],
        "solver_inlier_gtclean_strong_count": counters[
            "solver_inlier_gtclean_2px_count"
        ],
        "solver_inlier_gtclean_clean_count": counters[
            "solver_inlier_gtclean_4px_count"
        ],
    }
    output = {
        "schema": "lafgs_keypoint_function_graph_shard",
        "version": 2,
        "anchor_map": str(Path(args.anchor_map).resolve()),
        "query_cache": str(Path(args.query_cache).resolve()),
        "anchor_count": anchor_count,
        "query_count_total": len(names),
        "query_names": names,
        "query_indices": torch.as_tensor(
            selected_queries, dtype=torch.int32
        ),
        "source_primitive_ids": source_ids,
        "track_cluster_ids": torch.as_tensor(
            anchor_map["track_cluster_ids"]
        ).long(),
        "anchor_type": torch.as_tensor(anchor_map["anchor_type"]).to(
            torch.int8
        ),
        "records": records,
        "query_diagnostics": diagnostics,
        "config": vars(args),
        "raster_visibility_enabled": raster_visibility is not None,
        "resolved_thresholds": {
            "strong_radius_px": float(args.strong_radius_px),
            "clean_radius_px": float(args.clean_radius_px),
            "ambiguous_radius_px": float(args.ambiguous_radius_px),
            "pnp_reprojection_error_px": float(args.pnp_reprojection_error_px),
            "harm_radius_px": float(args.harm_radius_px),
            "depth_abs_tolerance_m": float(args.depth_abs_tolerance_m),
            "depth_rel_tolerance": float(args.depth_rel_tolerance),
        },
        **counters,
        **semantic_counters,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    opportunity = counters["candidate_opportunity_count"].clamp_min(1)
    summary = {
        "anchor_count": anchor_count,
        "query_count": len(selected_queries),
        "raster_visibility_enabled": raster_visibility is not None,
        "anchors_with_legal_2px": int(
            (counters["legal_hit_2px_count"] > 0).sum()
        ),
        "anchors_with_legal_4px": int(
            (counters["legal_hit_4px_count"] > 0).sum()
        ),
        "anchors_with_gtclean_solver_inlier": int(
            (counters["solver_inlier_gtclean_4px_count"] > 0).sum()
        ),
        "anchors_with_harmful_solver_inlier": int(
            (counters["harmful_solver_inlier_count"] > 0).sum()
        ),
        "harmful_consensus_rate_mean": float(
            (
                counters["harmful_solver_inlier_count"].float()
                / opportunity
            ).mean()
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
