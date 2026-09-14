#!/usr/bin/env python3
"""Build active-map-complete raster/depth legal native-keypoint positives."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from features.raster_sampling import sample_raster_at_grid_uv


def _source_anchor_lookup(source_ids: torch.Tensor):
    source_ids = torch.as_tensor(source_ids).long().reshape(-1)
    if source_ids.numel() == 0:
        return source_ids, torch.empty((0, 0), dtype=torch.long)
    order = torch.argsort(source_ids, stable=True)
    sorted_sources = source_ids[order]
    unique, counts = torch.unique_consecutive(
        sorted_sources, return_counts=True
    )
    width = int(counts.max())
    lookup = torch.full((unique.numel(), width), -1, dtype=torch.long)
    rows = torch.repeat_interleave(torch.arange(unique.numel()), counts)
    starts = torch.cumsum(counts, dim=0) - counts
    columns = torch.arange(source_ids.numel()) - torch.repeat_interleave(
        starts, counts
    )
    lookup[rows, columns] = order
    return unique, lookup


def _expand_provenance_candidates(
    primitive_ids: torch.Tensor,
    contribution_mass: torch.Tensor,
    source_ids: torch.Tensor,
    source_lookup: torch.Tensor,
    minimum_mass: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    primitive_ids = torch.as_tensor(
        primitive_ids, device=source_ids.device
    ).long()
    contribution_mass = torch.as_tensor(
        contribution_mass, device=source_ids.device
    ).float()
    position = torch.searchsorted(source_ids, primitive_ids)
    valid_source = position < source_ids.numel()
    safe = position.clamp(max=max(source_ids.numel() - 1, 0))
    valid_source &= source_ids[safe] == primitive_ids
    expanded = source_lookup[safe].reshape(
        primitive_ids.shape[0], -1
    )
    valid = (
        valid_source[:, :, None]
        & (contribution_mass[:, :, None] >= float(minimum_mass))
        & (source_lookup[safe] >= 0)
    ).reshape(primitive_ids.shape[0], -1)
    return expanded, valid


def _deduplicated_csr(
    values: torch.Tensor,
    valid: torch.Tensor,
    value_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(values).long()
    valid = torch.as_tensor(valid).bool()
    sentinel = int(value_count)
    ordered = torch.sort(
        torch.where(valid, values, torch.full_like(values, sentinel)),
        dim=1,
    ).values
    keep = ordered < sentinel
    if ordered.shape[1] > 1:
        keep[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    counts = keep.sum(dim=1)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), counts.cpu().cumsum(0)]
    )
    return offsets, ordered[keep].cpu()


def _exact_track_observations(
    payload: dict,
    active_track_indices: torch.Tensor,
    active_anchor_rows: torch.Tensor | None = None,
    query_index_remap: torch.Tensor | None = None,
) -> dict[int, dict[int, list[int]]]:
    active_track_indices = torch.as_tensor(active_track_indices).long()
    active_anchor_rows = (
        torch.arange(active_track_indices.numel(), dtype=torch.long)
        if active_anchor_rows is None
        else torch.as_tensor(active_anchor_rows).long()
    )
    if active_anchor_rows.numel() != active_track_indices.numel():
        raise ValueError("track identities and anchor rows must align")
    track_to_local: dict[int, list[int]] = defaultdict(list)
    for track, anchor_row in zip(
        active_track_indices.tolist(), active_anchor_rows.tolist()
    ):
        if int(track) >= 0:
            track_to_local[int(track)].append(int(anchor_row))
    exact: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    tracks = payload["tracks"]
    for track, query, keypoint in zip(
        tracks["track_index"].tolist(),
        tracks["query_index"].tolist(),
        tracks["keypoint_index"].tolist(),
    ):
        local = track_to_local.get(int(track), ())
        if local:
            target_query = (
                int(query_index_remap[int(query)])
                if query_index_remap is not None
                else int(query)
            )
            exact[target_query][int(keypoint)].extend(local)
    return exact


def _query_index_remap(
    source_names: list[str], target_names: list[str]
) -> torch.Tensor:
    if len(set(source_names)) != len(source_names):
        raise ValueError("source query names must be unique")
    if len(set(target_names)) != len(target_names):
        raise ValueError("target query names must be unique")
    target_by_name = {name: index for index, name in enumerate(target_names)}
    if set(source_names) != set(target_names):
        missing = sorted(set(source_names) - set(target_names))
        extra = sorted(set(target_names) - set(source_names))
        raise ValueError(
            f"query-name sets differ: missing={missing[:3]}, extra={extra[:3]}"
        )
    return torch.as_tensor(
        [target_by_name[name] for name in source_names], dtype=torch.long
    )


def _exact_dense(
    query_rows: torch.Tensor,
    exact: dict[int, list[int]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_rows = torch.as_tensor(query_rows).long()
    width = max((len(exact.get(int(row), ())) for row in query_rows), default=0)
    values = torch.full(
        (query_rows.numel(), width), -1, dtype=torch.long, device=device
    )
    for output_row, keypoint in enumerate(query_rows.tolist()):
        local = exact.get(int(keypoint), ())
        if local:
            values[output_row, : len(local)] = torch.as_tensor(
                local, dtype=torch.long, device=device
            )
    return values, values >= 0


def _sample_surface(
    cached: dict,
    query_rows: torch.Tensor,
    provenance_record: dict | None = None,
):
    keypoints = torch.as_tensor(cached["native_keypoints"]).float()[query_rows]
    if provenance_record is not None and {
        "rendered_depth",
        "rendered_alpha",
    } <= provenance_record.keys():
        depth = torch.as_tensor(provenance_record["rendered_depth"]).float()
        alpha = torch.as_tensor(provenance_record["rendered_alpha"]).float()
        if depth.shape != query_rows.shape or alpha.shape != query_rows.shape:
            raise ValueError(
                "raster-provenance surface samples must align with query_rows"
            )
        return keypoints, depth, alpha, "raster_provenance"
    depth = sample_raster_at_grid_uv(
        torch.as_tensor(cached["native_depth"]).float(), keypoints
    )
    if "native_alpha" not in cached:
        raise KeyError(
            "complete-positive teacher requires rendered_alpha in raster "
            "provenance or native_alpha in the query cache; rebuild raster "
            "provenance with the current pipeline"
        )
    alpha = sample_raster_at_grid_uv(
        torch.as_tensor(cached["native_alpha"]).float(), keypoints
    )
    return keypoints, depth, alpha, "query_cache"


@torch.no_grad()
def build_teacher(
    *,
    anchor_map: dict,
    query_cache: dict,
    provenance: dict,
    track_payload: dict,
    device: torch.device,
    strong_radius_px: float,
    ambiguous_radius_px: float,
    depth_abs_tolerance_m: float,
    depth_rel_tolerance: float,
    alpha_minimum: float,
    contribution_minimum: float,
    query_indices: list[int] | None = None,
) -> dict:
    xyz = torch.as_tensor(anchor_map["anchor_xyz"]).float()
    source = torch.as_tensor(anchor_map["source_primitive_ids"]).long()
    source_ids, source_lookup = _source_anchor_lookup(source)
    source_ids = source_ids.to(device)
    source_lookup = source_lookup.to(device)
    map_count = int(xyz.shape[0])
    anchor_type = torch.as_tensor(anchor_map["anchor_type"]).long()
    track_rows = torch.nonzero(anchor_type != 0, as_tuple=False).reshape(-1)
    active_track_indices = torch.as_tensor(
        anchor_map["track_cluster_ids"]
    ).long()[track_rows]
    names = provenance["query_names"]
    payload_to_teacher = _query_index_remap(
        track_payload["query_names"], names
    )
    exact = _exact_track_observations(
        track_payload,
        active_track_indices,
        track_rows,
        query_index_remap=payload_to_teacher,
    )
    cache = query_cache.get("queries", query_cache)
    records = []
    total_strong = 0
    total_ambiguous = 0
    positive_rows = 0
    exact_positive_count = 0
    surface_sample_sources: set[str] = set()

    selected_queries = (
        list(range(len(names)))
        if query_indices is None
        else [int(index) for index in query_indices]
    )
    for completed, query_index in enumerate(selected_queries, start=1):
        provenance_record = provenance["records"][query_index]
        name = names[query_index]
        cached = cache[name]
        query_rows = torch.as_tensor(
            provenance_record["query_rows"]
        ).long()
        if not torch.equal(
            query_rows,
            torch.as_tensor(provenance_record["query_rows"]).long(),
        ):
            raise RuntimeError("provenance query rows are not deterministic")
        candidates, candidate_valid = _expand_provenance_candidates(
            provenance_record["primitive_ids"],
            provenance_record["contribution_mass"],
            source_ids,
            source_lookup,
            contribution_minimum,
        )
        candidates = candidates.to(device)
        candidate_valid = candidate_valid.to(device)
        (
            keypoint_grid,
            rendered_depth,
            rendered_alpha,
            surface_sample_source,
        ) = _sample_surface(
            cached, query_rows, provenance_record
        )
        surface_sample_sources.add(surface_sample_source)
        keypoints = keypoint_grid + float(
            cached.get("pixel_center_offset", 0.5)
        )
        safe = candidates.clamp(min=0)
        candidate_xyz = xyz[safe.cpu()].to(device)
        pose = torch.as_tensor(cached["pose_w2c"]).float().to(device)
        K = torch.as_tensor(cached["native_K"]).float().to(device)
        camera = (
            candidate_xyz @ pose[:3, :3].T + pose[:3, 3]
        )
        depth = camera[:, :, 2]
        projected = camera @ K.T
        uv = projected[:, :, :2] / depth[:, :, None].clamp_min(1e-8)
        error = torch.linalg.norm(
            uv - keypoints.to(device)[:, None], dim=2
        )
        rendered_depth = rendered_depth.to(device)
        rendered_alpha = rendered_alpha.to(device)
        tolerance = float(depth_abs_tolerance_m) + (
            float(depth_rel_tolerance) * rendered_depth.abs()
        )
        surface_valid = (
            torch.isfinite(rendered_depth)
            & (rendered_depth > 0)
            & torch.isfinite(rendered_alpha)
            & (rendered_alpha >= float(alpha_minimum))
        )
        depth_legal = (
            candidate_valid
            & surface_valid[:, None]
            & torch.isfinite(depth)
            & (depth > 0)
            & (
                (depth - rendered_depth[:, None]).abs()
                <= tolerance[:, None]
            )
        )
        strong = depth_legal & (error <= float(strong_radius_px))
        ambiguous = (
            depth_legal
            & (error > float(strong_radius_px))
            & (error <= float(ambiguous_radius_px))
        )
        exact_values, exact_valid = _exact_dense(
            query_rows,
            exact.get(query_index, {}),
            device=device,
        )
        if exact_values.shape[1]:
            candidates = torch.cat([candidates, exact_values], dim=1)
            strong = torch.cat([strong, exact_valid], dim=1)
            ambiguous = torch.cat(
                [ambiguous, torch.zeros_like(exact_valid)], dim=1
            )
            exact_positive_count += int(exact_valid.sum())
        strong_offsets, strong_indices = _deduplicated_csr(
            candidates, strong, map_count
        )
        ambiguous_offsets, ambiguous_indices = _deduplicated_csr(
            candidates, ambiguous, map_count
        )
        strong_counts = strong_offsets[1:] - strong_offsets[:-1]
        positive_rows += int((strong_counts > 0).sum())
        total_strong += int(strong_indices.numel())
        total_ambiguous += int(ambiguous_indices.numel())
        records.append(
            {
                "query_index": int(query_index),
                "query_name": name,
                "query_rows": query_rows,
                "positive_offsets": strong_offsets,
                "positive_indices": strong_indices,
                "ambiguous_offsets": ambiguous_offsets,
                "ambiguous_indices": ambiguous_indices,
            }
        )
        if completed % 50 == 0 or completed == len(selected_queries):
            print(
                json.dumps(
                    {
                        "completed_queries": completed,
                        "positive_rows": positive_rows,
                        "strong_pairs": total_strong,
                    }
                ),
                flush=True,
            )

    return {
        "schema": "lafgs_v9_active_map_complete_positive_teacher",
        "version": 1,
        "anchor_count": map_count,
        "query_names": names,
        "records": records,
        "diagnostics": {
            "query_count": len(records),
            "positive_rows": positive_rows,
            "strong_pair_count": total_strong,
            "ambiguous_pair_count": total_ambiguous,
            "exact_track_positive_count": exact_positive_count,
        },
        "config": {
            "strong_radius_px": float(strong_radius_px),
            "ambiguous_radius_px": float(ambiguous_radius_px),
            "depth_abs_tolerance_m": float(depth_abs_tolerance_m),
            "depth_rel_tolerance": float(depth_rel_tolerance),
            "alpha_minimum": float(alpha_minimum),
            "contribution_minimum": float(contribution_minimum),
            "surface_sample_sources": sorted(surface_sample_sources),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--raster-provenance", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strong-radius-px", type=float, default=2.0)
    parser.add_argument("--ambiguous-radius-px", type=float, default=8.0)
    parser.add_argument("--depth-abs-tolerance-m", type=float, default=0.05)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.02)
    parser.add_argument("--alpha-minimum", type=float, default=0.01)
    parser.add_argument("--contribution-minimum", type=float, default=1e-4)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("complete positive teacher construction requires CUDA")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    anchor_map = torch.load(
        args.anchor_map, map_location="cpu", weights_only=False
    )
    query_cache = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    provenance = torch.load(
        args.raster_provenance, map_location="cpu", weights_only=False
    )
    track_payload = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    teacher = build_teacher(
        anchor_map=anchor_map,
        query_cache=query_cache,
        provenance=provenance,
        track_payload=track_payload,
        device=torch.device("cuda"),
        strong_radius_px=args.strong_radius_px,
        ambiguous_radius_px=args.ambiguous_radius_px,
        depth_abs_tolerance_m=args.depth_abs_tolerance_m,
        depth_rel_tolerance=args.depth_rel_tolerance,
        alpha_minimum=args.alpha_minimum,
        contribution_minimum=args.contribution_minimum,
        query_indices=[
            index
            for index in range(len(provenance["query_names"]))
            if index % args.num_shards == args.shard_index
        ],
    )
    teacher["config"].update(
        {
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        }
    )
    teacher.update(
        {
            "anchor_map": str(Path(args.anchor_map).resolve()),
            "query_cache": str(Path(args.query_cache).resolve()),
            "raster_provenance": str(
                Path(args.raster_provenance).resolve()
            ),
            "track_payload": str(Path(args.track_payload).resolve()),
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(teacher, output)
    print(json.dumps(teacher["diagnostics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
