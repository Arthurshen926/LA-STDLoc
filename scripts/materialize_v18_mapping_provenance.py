#!/usr/bin/env python3
"""Materialize exact Gaussian composition for Projective Anchor observations.

The output is mapping-only and descriptor-independent.  It preserves global
observation row IDs so separately scheduled query shards can be merged without
loading or copying the 24 GB source cache again.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from priors.models import GaussianModel2D, GaussianModel3D
from priors.rasterizer import bank_splat_provenance_2dgs, bank_splat_provenance_3dgs
from priors.rendering import render_from_pose_gsplat


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sample_raster(raster: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(raster).squeeze()
    pixels = torch.floor(keypoints).long()
    x = pixels[:, 0].clamp(0, value.shape[1] - 1)
    y = pixels[:, 1].clamp(0, value.shape[0] - 1)
    return value[y, x]


def _relative_depth_spread(
    depth_raster: torch.Tensor, keypoints: torch.Tensor
) -> torch.Tensor:
    depth = torch.as_tensor(depth_raster).squeeze()
    pixels = torch.floor(keypoints).long()
    samples = []
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            x = (pixels[:, 0] + offset_x).clamp(0, depth.shape[1] - 1)
            y = (pixels[:, 1] + offset_y).clamp(0, depth.shape[0] - 1)
            samples.append(depth[y, x])
    local = torch.stack(samples, dim=1)
    valid = torch.isfinite(local) & (local > 0)
    minimum = local.masked_fill(~valid, float("inf")).amin(1)
    maximum = local.masked_fill(~valid, -float("inf")).amax(1)
    center = _sample_raster(depth, keypoints).abs().clamp_min(1e-6)
    spread = (maximum - minimum) / center
    spread[~valid.any(1)] = float("inf")
    return spread


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--observation-cache", type=Path, required=True)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"), default="2dgs")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk-primitives", type=int, default=64)
    parser.add_argument(
        "--minimum-composition-mass", type=float, default=0.95
    )
    parser.add_argument(
        "--candidate-topk",
        type=int,
        default=0,
        help="0 performs full depth-ordered Gaussian compositing",
    )
    parser.add_argument(
        "--prefilter-topk",
        type=int,
        default=0,
        help="0 evaluates the complete Gaussian prior (required for formal truth)",
    )
    parser.add_argument("--depth-abs-tolerance-m", type=float, default=0.05)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.02)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        parser.error("invalid query shard")
    if int(args.topk_primitives) < 1 or (
        0 < int(args.candidate_topk) < int(args.topk_primitives)
    ):
        parser.error("invalid provenance Top-K")
    if not 0.0 < float(args.minimum_composition_mass) <= 1.0:
        parser.error("minimum composition mass must lie in (0, 1]")
    if int(args.candidate_topk) <= 0 and int(args.prefilter_topk) > 0:
        parser.error("full compositing cannot use a provenance prefilter")
    if 0 < int(args.prefilter_topk) < int(args.candidate_topk):
        parser.error("provenance prefilter must cover candidate Top-K")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V18 mapping provenance requires CUDA")

    map_path = args.anchor_map.resolve()
    cache_path = args.observation_cache.resolve()
    prior_path = args.gaussian_ply.resolve()
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache.get("queries", cache)
    names = list(state["v6_mapping_query_names"])
    if names != list(records):
        raise ValueError("V18 map and mapping observation cache registries differ")
    observations = state["projective_anchor_observations"]
    observation_offsets = torch.as_tensor(observations["observation_offsets"]).long()
    observation_queries = torch.as_tensor(observations["query_indices"]).long()
    observation_keypoints = torch.as_tensor(observations["keypoint_indices"]).long()
    anchor_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    if observation_offsets.shape != (anchor_count + 1,):
        raise ValueError("V18 observation offsets differ from the Anchor registry")
    if observation_queries.shape != observation_keypoints.shape or int(observation_offsets[-1]) != observation_queries.numel():
        raise ValueError("V18 observation CSR arrays do not align")
    if bool((observation_offsets[1:] <= observation_offsets[:-1]).any()):
        raise ValueError("every V18 Anchor must retain at least one observation")

    model_class = GaussianModel2D if args.gaussian_type == "2dgs" else GaussianModel3D
    gaussians = model_class(args.sh_degree, device=device)
    gaussians.load_ply(prior_path, loc_feature_dim=0)
    gaussians = gaussians.to(device).eval()
    primitive_count = int(gaussians.get_xyz.shape[0])
    # V2 Projective Anchors intentionally store ``source_primitive_ids=-1``:
    # their geometry comes from Tracks, not Gaussian centres.  Ground-truth
    # provenance must therefore be recomputed over the complete frozen prior
    # at every original observation instead of inventing a source ID.
    source_universe = torch.arange(primitive_count, dtype=torch.long)
    provenance_function = (
        bank_splat_provenance_2dgs
        if args.gaussian_type == "2dgs"
        else bank_splat_provenance_3dgs
    )

    order = torch.argsort(observation_queries, stable=True)
    sorted_queries = observation_queries[order]
    query_counts = torch.bincount(sorted_queries, minlength=len(names))
    query_offsets = torch.zeros(len(names) + 1, dtype=torch.long)
    query_offsets[1:] = query_counts.cumsum(0)
    selected_queries = [
        index
        for index in range(len(names))
        if index % int(args.num_shards) == int(args.shard_index)
        and int(query_counts[index]) > 0
    ]
    selected_edge_count = int(query_counts[selected_queries].sum())
    k = int(args.topk_primitives)
    output_rows = torch.empty(selected_edge_count, dtype=torch.long)
    output_ids = torch.full((selected_edge_count, k), -1, dtype=torch.int32)
    output_weights = torch.zeros((selected_edge_count, k), dtype=torch.float32)
    output_valid = torch.zeros(selected_edge_count, dtype=torch.bool)
    output_depth = torch.full((selected_edge_count,), float("nan"))
    output_alpha = torch.full((selected_edge_count,), float("nan"), dtype=torch.float16)
    output_entropy = torch.full((selected_edge_count,), float("nan"))
    output_depth_spread = torch.full((selected_edge_count,), float("nan"))
    output_retained_mass = torch.zeros(selected_edge_count)
    mapping_keypoints: list[torch.Tensor] = []
    mapping_intrinsics = torch.empty((len(names), 3, 3), dtype=torch.float32)
    mapping_poses = torch.empty((len(names), 4, 4), dtype=torch.float32)
    mapping_hw = torch.empty((len(names), 2), dtype=torch.long)
    mapping_offsets = torch.empty((len(names),), dtype=torch.float32)
    sequence_registry = sorted({str(name).split("/", 1)[0] for name in names})
    sequence_to_id = {
        sequence: index for index, sequence in enumerate(sequence_registry)
    }
    mapping_sequence_family_ids = torch.tensor(
        [sequence_to_id[str(name).split("/", 1)[0]] for name in names],
        dtype=torch.long,
    )
    for query_index, name in enumerate(names):
        record = records[name]
        mapping_keypoints.append(torch.as_tensor(record["native_keypoints"]).float())
        mapping_intrinsics[query_index] = torch.as_tensor(record["native_K"]).float()
        mapping_poses[query_index] = torch.as_tensor(record["pose_w2c"]).float()
        mapping_hw[query_index] = torch.as_tensor(record["native_input_hw"]).long()
        mapping_offsets[query_index] = float(record.get("pixel_center_offset", 0.5))

    cursor = 0
    source_device = source_universe.to(device)
    for completed, query_index in enumerate(selected_queries, start=1):
        begin, end = int(query_offsets[query_index]), int(query_offsets[query_index + 1])
        observation_rows = order[begin:end]
        keypoint_rows = observation_keypoints[observation_rows]
        keypoints = mapping_keypoints[query_index][keypoint_rows]
        height, width = map(int, mapping_hw[query_index].tolist())
        intrinsic = mapping_intrinsics[query_index]
        fov_x = 2.0 * math.atan(width / (2.0 * float(intrinsic[0, 0])))
        fov_y = 2.0 * math.atan(height / (2.0 * float(intrinsic[1, 1])))
        package = render_from_pose_gsplat(
            gaussians,
            mapping_poses[query_index].to(device),
            fov_x,
            fov_y,
            width,
            height,
            bg_color=torch.zeros(3, device=device),
            render_mode="RGB+ED",
            rgb_only=True,
            return_rgb_meta=True,
            rasterize_mode="antialiased",
        )
        provenance_result = provenance_function(
            keypoints.to(device),
            source_device,
            package["rgb_meta"],
            rendered_depth=package.get("depth"),
            topk=k,
            candidate_topk=int(args.candidate_topk),
            depth_abs_tolerance=float(args.depth_abs_tolerance_m),
            depth_rel_tolerance=float(args.depth_rel_tolerance),
            prefilter_topk=(
                int(args.prefilter_topk)
                if args.gaussian_type == "2dgs" and int(args.prefilter_topk) > 0
                else None
            ),
            **(
                {"return_diagnostics": True}
                if args.gaussian_type == "2dgs"
                else {}
            ),
            **(
                {"minimum_composition_mass": float(args.minimum_composition_mass)}
                if args.gaussian_type == "2dgs"
                else {}
            ),
        )
        if args.gaussian_type == "2dgs":
            local_ids, weights, valid, provenance_diagnostics = provenance_result
        else:
            local_ids, weights, valid = provenance_result
            provenance_diagnostics = {
                "retained_composition_fraction": torch.ones_like(valid).float()
            }
        count = int(observation_rows.numel())
        target = slice(cursor, cursor + count)
        output_rows[target] = observation_rows
        output_ids[target] = source_universe[local_ids.cpu()].to(torch.int32)
        output_weights[target] = weights.cpu().float()
        output_valid[target] = valid.cpu()
        output_depth[target] = _sample_raster(package["depth"], keypoints.to(device)).cpu()
        output_entropy[target] = (
            -(weights * weights.clamp_min(1e-12).log()).sum(1).cpu()
        )
        output_depth_spread[target] = _relative_depth_spread(
            package["depth"], keypoints.to(device)
        ).cpu()
        output_retained_mass[target] = provenance_diagnostics[
            "retained_composition_fraction"
        ].cpu()
        alpha = package.get("alphas", package.get("rend_alpha"))
        if alpha is None:
            raise RuntimeError("V18 renderer did not return alpha")
        output_alpha[target] = _sample_raster(alpha, keypoints.to(device)).cpu().to(torch.float16)
        cursor += count
        del package, local_ids, weights, valid
        if completed % max(int(args.progress_interval), 1) == 0 or completed == len(selected_queries):
            print(
                json.dumps(
                    {
                        "completed_queries": completed,
                        "shard_queries": len(selected_queries),
                        "completed_observations": cursor,
                    }
                ),
                flush=True,
            )
    if cursor != selected_edge_count:
        raise RuntimeError("V18 observation cursor did not cover the shard")
    artifact = {
        "schema": "lafgs_v18_mapping_observation_gaussian_provenance",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "loo_used": False,
        "descriptor_independent": True,
        "full_gaussian_prior_evaluated": int(args.prefilter_topk) <= 0,
        "full_depth_ordered_compositing": int(args.candidate_topk) <= 0,
        "minimum_retained_composition_mass": float(args.minimum_composition_mass),
        "anchor_count": anchor_count,
        "mapping_query_count": len(names),
        "global_observation_count": int(observation_queries.numel()),
        "observation_rows": output_rows,
        "observation_primitive_ids": output_ids,
        "observation_weights": output_weights,
        "observation_valid": output_valid,
        "observation_rendered_depth": output_depth,
        "observation_rendered_alpha": output_alpha,
        "observation_composition_entropy": output_entropy,
        "observation_relative_depth_spread_3x3": output_depth_spread,
        "observation_retained_composition_fraction": output_retained_mass,
        "mapping_query_names": names,
        "mapping_keypoints": mapping_keypoints,
        "mapping_intrinsics": mapping_intrinsics,
        "mapping_poses_w2c": mapping_poses,
        "mapping_image_hw": mapping_hw,
        "mapping_pixel_center_offset": mapping_offsets,
        "mapping_view_family_ids": mapping_sequence_family_ids,
        "mapping_view_family_registry": sequence_registry,
        "mapping_legacy_view_bins": torch.as_tensor(
            state["v6_mapping_query_bins"]
        ).long(),
        "mapping_view_family_policy": "source_mapping_sequence",
        "source_primitive_universe": source_universe,
        "primitive_count": primitive_count,
        "config": vars(args),
        "inputs": {
            "anchor_map": str(map_path),
            "anchor_map_sha256": sha256_file(map_path),
            "observation_cache": str(cache_path),
            "observation_cache_sha256": sha256_file(cache_path),
            "gaussian_ply": str(prior_path),
            "gaussian_ply_sha256": sha256_file(prior_path),
        },
    }
    _atomic_save(artifact, args.output.resolve())
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "query_count": len(selected_queries),
                "observation_count": selected_edge_count,
                "valid_observation_count": int(output_valid.sum()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
