#!/usr/bin/env python3
"""Cache sparse 2DGS composition provenance at native query keypoints."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import torch

from priors.rendering import render_from_pose_gsplat
from priors.rasterizer import (
    anchor_source_csr,
    bank_splat_provenance_2dgs,
    bank_splat_provenance_3dgs,
)
from priors.models import GaussianModel2D, GaussianModel3D
from data.masks import deployment_valid_mask


def _anchor_source_csr(
    state: dict,
    track_payload: dict | None,
    full_prior_pool: dict | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compatibility wrapper for callers and older tests."""
    return anchor_source_csr(state, track_payload, full_prior_pool)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument(
        "--gaussian-type", choices=("2dgs", "3dgs"), default="2dgs"
    )
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument(
        "--function-graph",
        default="",
        help=(
            "Optional top-candidate graph. When supplied, provenance is "
            "evaluated only over source families reachable by each query."
        ),
    )
    parser.add_argument("--track-payload", default="")
    parser.add_argument("--full-prior-pool", default="")
    parser.add_argument("--deployment-mask-cache", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk-primitives", type=int, default=16)
    parser.add_argument("--candidate-topk", type=int, default=64)
    parser.add_argument("--depth-abs-tolerance-m", type=float, default=0.05)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.02)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("raster provenance construction requires CUDA")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    state = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    query_cache = payload.get("queries", payload)
    track_payload = (
        torch.load(args.track_payload, map_location="cpu", weights_only=False)
        if args.track_payload
        else None
    )
    full_prior_pool = (
        torch.load(
            args.full_prior_pool, map_location="cpu", weights_only=False
        )
        if args.full_prior_pool
        else None
    )
    if full_prior_pool is not None:
        expected_ply = full_prior_pool.get("provenance", {}).get(
            "gaussian_ply_path"
        )
        if expected_ply and Path(expected_ply).resolve() != Path(
            args.gaussian_ply
        ).resolve():
            raise ValueError(
                "supplied Gaussian PLY does not match the anchor-map "
                f"provenance: expected {expected_ply}"
            )
    source_offsets, source_ids, source_weights = _anchor_source_csr(
        state, track_payload, full_prior_pool
    )
    source_universe = torch.unique(source_ids).sort().values
    if source_universe.numel() == 0:
        raise ValueError("anchor source universe is empty")

    deployment_masks = None
    if args.deployment_mask_cache:
        with Path(args.deployment_mask_cache).open("rb") as handle:
            deployment_masks = pickle.load(handle)
    gaussians = (
        GaussianModel2D(args.sh_degree)
        if args.gaussian_type == "2dgs"
        else GaussianModel3D(args.sh_degree)
    )
    gaussians.load_ply(args.gaussian_ply)
    gaussians = gaussians.cuda().eval()
    primitive_count = int(gaussians.get_xyz.shape[0])
    if int(source_universe.max()) >= primitive_count:
        raise ValueError(
            "anchor source IDs exceed frozen Gaussian primitive count"
        )

    names = list(query_cache)
    graph_records = None
    if args.function_graph:
        graph = torch.load(
            args.function_graph, map_location="cpu", weights_only=False
        )
        if graph["query_names"] != names:
            raise ValueError("function graph and query cache names differ")
        graph_records = {
            int(record["query_index"]): record
            for record in graph["records"]
        }
    records = []
    for query_index, name in enumerate(names):
        if query_index % args.num_shards != args.shard_index:
            continue
        cached = query_cache[name]
        valid = deployment_valid_mask(cached, name, deployment_masks)
        query_rows = torch.nonzero(valid, as_tuple=False).reshape(-1)
        query_source_universe = source_universe
        if graph_records is not None:
            graph_record = graph_records[query_index]
            if not torch.equal(
                torch.as_tensor(graph_record["query_rows"]).long(),
                query_rows,
            ):
                raise ValueError(
                    f"function graph rows differ for query {query_index}"
                )
            candidate_anchors = torch.unique(
                torch.as_tensor(graph_record["top_indices"]).long()
            )
            family_parts = [
                source_ids[
                    int(source_offsets[anchor]) : int(
                        source_offsets[anchor + 1]
                    )
                ]
                for anchor in candidate_anchors.tolist()
            ]
            query_source_universe = torch.unique(
                torch.cat(family_parts)
            ).sort().values
        keypoints = torch.as_tensor(
            cached["native_keypoints"]
        )[query_rows].float()
        height, width = map(int, cached["native_input_hw"])
        intrinsic = torch.as_tensor(cached["native_K"]).float()
        fovx = 2.0 * math.atan(width / (2.0 * float(intrinsic[0, 0])))
        fovy = 2.0 * math.atan(height / (2.0 * float(intrinsic[1, 1])))
        package = render_from_pose_gsplat(
            gaussians,
            torch.as_tensor(cached["pose_w2c"]).cuda().float(),
            fovx,
            fovy,
            width,
            height,
            bg_color=torch.zeros(3, device="cuda"),
            render_mode="RGB+ED",
            rgb_only=True,
            return_rgb_meta=True,
            rasterize_mode="antialiased",
        )
        provenance_function = (
            bank_splat_provenance_2dgs
            if args.gaussian_type == "2dgs"
            else bank_splat_provenance_3dgs
        )
        local_ids, weights, provenance_valid = (
            provenance_function(
                keypoints.cuda(),
                query_source_universe.cuda(),
                package["rgb_meta"],
                rendered_depth=package.get("depth"),
                topk=args.topk_primitives,
                candidate_topk=args.candidate_topk,
                depth_abs_tolerance=args.depth_abs_tolerance_m,
                depth_rel_tolerance=args.depth_rel_tolerance,
            )
        )
        records.append(
            {
                "query_index": query_index,
                "query_rows": query_rows.to(torch.int32),
                "primitive_ids": query_source_universe[
                    local_ids.cpu()
                ].to(torch.int32),
                "contribution_mass": weights.cpu().to(torch.float16),
                "valid": provenance_valid.cpu(),
            }
        )
        del package, local_ids, weights, provenance_valid
        if len(records) % 10 == 0:
            print(
                f"provenance shard {args.shard_index}: "
                f"{len(records)} queries",
                flush=True,
            )

    output = {
        "schema": "lafgs_native_keypoint_raster_provenance",
        "version": 1,
        "anchor_map": str(Path(args.anchor_map).resolve()),
        "query_cache": str(Path(args.query_cache).resolve()),
        "gaussian_ply": str(Path(args.gaussian_ply).resolve()),
        "query_names": names,
        "primitive_count": primitive_count,
        "source_universe": source_universe,
        "anchor_source_offsets": source_offsets,
        "anchor_source_primitive_ids": source_ids,
        "anchor_source_weights": source_weights,
        "records": records,
        "config": vars(args),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                "query_count": len(records),
                "source_primitive_count": int(source_universe.numel()),
                "anchor_source_edge_count": int(source_ids.numel()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
