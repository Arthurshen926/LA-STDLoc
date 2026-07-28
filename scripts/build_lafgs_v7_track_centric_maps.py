#!/usr/bin/env python3
"""Build Track-centric LaFGS maps without preserving a base-anchor prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from localization_training.micro_anchors import fuse_track_descriptors


def _normalized_log_score(value: torch.Tensor) -> torch.Tensor:
    value = torch.log1p(torch.as_tensor(value).float().clamp_min(0))
    scale = torch.quantile(value, 0.95).clamp_min(1e-6)
    return (value / scale).clamp_max(2.0)


def _track_quality(geometry: dict) -> torch.Tensor:
    observation = torch.as_tensor(
        geometry["triangulation_observation_count"]
    ).float()
    view_bins = torch.as_tensor(
        geometry["triangulation_distinct_view_bin_count"]
    ).float()
    reprojection = torch.as_tensor(
        geometry["triangulation_reprojection_median_px"]
    ).float()
    p90 = torch.as_tensor(
        geometry["triangulation_reprojection_p90_px"]
    ).float()
    parallax = torch.as_tensor(
        geometry["triangulation_parallax_deg"]
    ).float()
    covariance = torch.as_tensor(
        geometry["triangulation_covariance_trace"]
    ).float()
    confidence = torch.as_tensor(
        geometry["track_confidence_level"]
    ).float()
    return (
        1.5 * _normalized_log_score(observation)
        + 0.8 * view_bins.clamp_max(4) / 4.0
        + 0.8 * (confidence == 2).float()
        + 0.7 * torch.log1p(parallax.clamp_min(0)) / np.log(31.0)
        - 0.8 * torch.log1p(reprojection.clamp_min(0)) / np.log(11.0)
        - 0.4 * torch.log1p(p90.clamp_min(0)) / np.log(101.0)
        - 0.4 * torch.log1p(covariance.clamp_min(0)) / np.log(21.0)
    )


def _eligible_tracks(geometry: dict, quality_tier: str) -> torch.Tensor:
    tier = {
        "strict": (2.0, 8.0, 0.05, 1.0),
        "medium": (3.0, 15.0, 0.2, 1.0),
        "broad": (4.0, 25.0, 1.0, 0.5),
        "relaxed": (6.0, 40.0, 3.0, 0.25),
        "all": (float("inf"), float("inf"), float("inf"), 0.0),
    }[quality_tier]
    median_px, p90_px, covariance, parallax = tier
    xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()
    return (
        torch.as_tensor(geometry["triangulated"]).bool()
        & torch.isfinite(xyz).all(dim=1)
        & (
            torch.as_tensor(
                geometry["triangulation_distinct_view_bin_count"]
            )
            >= 2
        )
        & (
            torch.as_tensor(
                geometry["triangulation_reprojection_median_px"]
            )
            <= median_px
        )
        & (
            torch.as_tensor(
                geometry["triangulation_reprojection_p90_px"]
            )
            <= p90_px
        )
        & (
            torch.as_tensor(geometry["triangulation_covariance_trace"])
            <= covariance
        )
        & (
            torch.as_tensor(geometry["triangulation_parallax_deg"])
            >= parallax
        )
    )


def _base_utility(graph: dict, base_count: int) -> torch.Tensor:
    legal2 = _normalized_log_score(
        graph["provenance_legal_hit_2px_count"][:base_count]
    )
    legal4 = _normalized_log_score(
        graph["provenance_legal_hit_4px_count"][:base_count]
    )
    clean = _normalized_log_score(
        graph["provenance_solver_inlier_gtclean_2px_count"][:base_count]
    )
    harmful = _normalized_log_score(
        graph["provenance_harmful_solver_inlier_count"][:base_count]
    )
    opportunity = _normalized_log_score(
        graph["provenance_opportunity_count"][:base_count]
    )
    return 3.0 * clean + 1.5 * legal2 + 0.5 * legal4 + 0.3 * opportunity - 1.8 * harmful


def _group_balanced_base_utility(
    graph: dict,
    query_groups: torch.Tensor,
    base_count: int,
    utility: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Reward anchors with strong support in any stable mapping-view group."""
    query_groups = torch.as_tensor(query_groups).long().reshape(-1)
    group_count = int(query_groups.max()) + 1
    legal_hits = torch.zeros(group_count, base_count, dtype=torch.float32)
    for record in graph["records"]:
        query_index = int(record["query_index"])
        group = int(query_groups[query_index])
        indices = torch.as_tensor(record["top_indices"]).long()
        flags = torch.as_tensor(record["legal_flags"]).to(torch.uint8)
        valid = ((flags & 2) != 0) & (indices >= 0) & (indices < base_count)
        selected = indices[valid]
        if selected.numel():
            legal_hits[group].index_add_(
                0, selected, torch.ones(selected.numel())
            )
    normalized = torch.stack(
        [_normalized_log_score(row) for row in legal_hits], dim=0
    )
    group_peak = normalized.max(dim=0).values
    group_breadth = (legal_hits > 0).float().mean(dim=0)
    balanced = utility + 1.5 * group_peak + 0.5 * group_breadth
    report = {
        "group_count": group_count,
        "anchors_with_group_support": int((legal_hits.sum(0) > 0).sum()),
        "mean_supported_group_fraction": float(group_breadth.mean()),
    }
    return balanced, report


def _voxel_diverse_order(
    xyz: torch.Tensor, utility: torch.Tensor, voxel_size: float
) -> torch.Tensor:
    xyz = torch.as_tensor(xyz).float()
    utility = torch.as_tensor(utility).float()
    origin = xyz.amin(dim=0)
    voxel = torch.floor((xyz - origin) / float(voxel_size)).long()
    _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
    order = torch.argsort(utility, descending=True, stable=True)
    buckets: dict[int, list[int]] = {}
    for row in order.tolist():
        buckets.setdefault(int(inverse[row]), []).append(row)
    voxel_order = sorted(
        buckets,
        key=lambda key: (-float(utility[buckets[key][0]]), key),
    )
    selected = []
    depth = 0
    while True:
        progress = False
        for key in voxel_order:
            if depth < len(buckets[key]):
                selected.append(buckets[key][depth])
                progress = True
        if not progress:
            break
        depth += 1
    return torch.as_tensor(selected, dtype=torch.long)


def _track_source_ids(
    canonical: dict, payload: dict, track_indices: torch.Tensor
) -> torch.Tensor:
    base_count = int(canonical["base_anchor_count"])
    base_xyz = torch.as_tensor(canonical["anchor_xyz"][:base_count]).float()
    base_sources = torch.as_tensor(
        canonical["source_primitive_ids"][:base_count]
    ).long()
    track_xyz = torch.as_tensor(
        payload["track_geometry"]["triangulated_xyz"]
    ).float()
    assignment = torch.as_tensor(
        payload["assignment"]["track_landmark_index"]
    ).long()
    nearest = cKDTree(base_xyz.numpy()).query(
        track_xyz[track_indices].numpy(), k=1
    )[1]
    source = base_sources[torch.as_tensor(nearest).long()].clone()
    assigned = assignment[track_indices]
    valid = (assigned >= 0) & (assigned < base_count)
    source[valid] = base_sources[assigned[valid]]
    return source


def _materialize(
    canonical: dict,
    payload: dict,
    track_indices: torch.Tensor,
    track_features: torch.Tensor,
    base_rows: torch.Tensor,
    *,
    budget: int,
    quality_tier: str,
    source_map: Path,
    payload_path: Path,
    dependency_voxel_size: float,
) -> dict:
    base_rows = torch.as_tensor(base_rows).long()
    track_indices = torch.as_tensor(track_indices).long()
    geometry = payload["track_geometry"]
    track_xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()[
        track_indices
    ]
    track_sources = _track_source_ids(canonical, payload, track_indices)
    fields = {
        "source_primitive_ids": torch.cat(
            [
                track_sources,
                torch.as_tensor(canonical["source_primitive_ids"])[base_rows],
            ]
        ),
        "track_cluster_ids": torch.cat(
            [
                track_indices,
                torch.full((base_rows.numel(),), -1, dtype=torch.long),
            ]
        ),
        "anchor_xyz": torch.cat(
            [track_xyz, torch.as_tensor(canonical["anchor_xyz"])[base_rows]]
        ),
        "anchor_features": torch.cat(
            [
                track_features,
                torch.as_tensor(canonical["anchor_features"])[base_rows],
            ]
        ),
        "anchor_type": torch.cat(
            [
                torch.full((track_indices.numel(),), 1, dtype=torch.long),
                torch.zeros(base_rows.numel(), dtype=torch.long),
            ]
        ),
    }
    all_xyz = fields["anchor_xyz"]
    all_sources = fields["source_primitive_ids"].long()
    dependency_voxel = torch.floor(
        all_xyz / float(dependency_voxel_size)
    ).long()
    coarse_dependency = torch.unique(
        torch.cat([all_sources[:, None], dependency_voxel], dim=1),
        dim=0,
        return_inverse=True,
    )[1]
    fine_identity = torch.cat(
        (
            track_indices,
            torch.arange(base_rows.numel(), dtype=torch.long)
            + int(torch.as_tensor(geometry["triangulated"]).numel()),
        )
    )
    return {
        "version": 1,
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(budget, dtype=torch.long),
        **fields,
        "dependency_group_ids": coarse_dependency,
        "coarse_dependency_group_ids": coarse_dependency,
        "fine_identity_ids": fine_identity,
        "base_anchor_count": int(base_rows.numel()),
        "canonical_anchor_count": int(budget),
        "micro_anchor_count": int(track_indices.numel()),
        "requested_micro_anchor_budget": int(track_indices.numel()),
        "track_centric_reconstruction": {
            "schema": "lafgs_v7_track_centric_map",
            "version": 1,
            "budget": int(budget),
            "track_anchor_count": int(track_indices.numel()),
            "base_reserve_count": int(base_rows.numel()),
            "quality_tier": quality_tier,
            "dependency_voxel_size": float(dependency_voxel_size),
            "track_indices": track_indices,
            "base_canonical_rows": base_rows,
            "base_prefix_preserved": False,
        },
        "provenance": {
            **canonical.get("provenance", {}),
            "v7_source_map": str(source_map),
            "v7_track_payload": str(payload_path),
            "selection_split": "all_895_mapping_train",
            "base_prefix_preserved": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--specs",
        default="16000:6000:strict,20000:8400:medium,24000:10000:broad",
        help="Comma-separated total:track:tier specifications.",
    )
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--base-voxel-size", type=float, default=1.0)
    parser.add_argument(
        "--base-selection",
        choices=["global", "group_balanced"],
        default="global",
    )
    parser.add_argument("--dependency-voxel-size", type=float, default=0.5)
    args = parser.parse_args()
    source_map = Path(args.canonical_map).resolve()
    graph_path = Path(args.function_graph).resolve()
    payload_path = Path(args.track_payload).resolve()
    query_path = Path(args.query_cache).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    seen_specs = set()
    for raw in args.specs.split(","):
        budget, tracks, tier = raw.split(":")
        spec = (int(budget), int(tracks), tier)
        if spec in seen_specs:
            raise ValueError(f"duplicate map specification: {raw}")
        seen_specs.add(spec)
        specs.append(spec)

    canonical = torch.load(source_map, map_location="cpu", weights_only=False)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    query_cache = torch.load(
        query_path, map_location="cpu", weights_only=False
    )
    base_count = int(canonical["base_anchor_count"])
    if int(graph["anchor_count"]) != int(canonical["anchor_xyz"].shape[0]):
        raise ValueError("function graph and canonical map do not align")
    base_rows = torch.nonzero(
        torch.as_tensor(canonical["anchor_type"]) == 0,
        as_tuple=False,
    ).reshape(-1)
    if base_rows.numel() != base_count:
        raise ValueError("canonical base anchors must form the original base")
    utility = _base_utility(graph, base_count)
    group_balance_report = None
    if args.base_selection == "group_balanced":
        utility, group_balance_report = _group_balanced_base_utility(
            graph,
            torch.as_tensor(payload["query_bins"]),
            base_count,
            utility,
        )
    base_order_local = _voxel_diverse_order(
        torch.as_tensor(canonical["anchor_xyz"])[base_rows],
        utility,
        args.base_voxel_size,
    )
    base_order = base_rows[base_order_local]

    geometry = payload["track_geometry"]
    quality = _track_quality(geometry)
    tier_masks = {}
    for _, _, tier in specs:
        tier_masks[tier] = _eligible_tracks(geometry, tier)
    selected_by_spec = {}
    selected_union = []
    quality_order = torch.argsort(quality, descending=True, stable=True)
    for budget, track_budget, tier in specs:
        selected = quality_order[tier_masks[tier][quality_order]][:track_budget]
        if selected.numel() < track_budget:
            raise ValueError(
                f"{tier} provides {selected.numel()} tracks, "
                f"needs {track_budget}"
            )
        selected_by_spec[(budget, track_budget, tier)] = selected
        selected_union.append(selected)
    ordered_tracks = torch.unique(torch.cat(selected_union), sorted=False)
    print(
        f"Fusing {ordered_tracks.numel()} Track-First descriptors",
        flush=True,
    )
    fused = fuse_track_descriptors(
        payload=payload,
        query_cache=query_cache,
        track_indices=ordered_tracks,
        trim_fraction=args.descriptor_trim_fraction,
    )
    feature_by_track = {
        int(track): fused[row]
        for row, track in enumerate(ordered_tracks.tolist())
    }

    summary = {
        "schema": "lafgs_v7_track_centric_build",
        "canonical_map": str(source_map),
        "function_graph": str(graph_path),
        "track_payload": str(payload_path),
        "query_cache": str(query_path),
        "base_prefix_preserved": False,
        "base_selection": args.base_selection,
        "group_balance": group_balance_report,
        "maps": {},
    }
    for budget, track_budget, tier in specs:
        tracks = selected_by_spec[(budget, track_budget, tier)]
        features = torch.stack(
            [feature_by_track[int(track)] for track in tracks.tolist()]
        )
        reserve = budget - int(tracks.numel())
        selected_base = base_order[:reserve]
        state = _materialize(
            canonical,
            payload,
            tracks,
            features,
            selected_base,
            budget=budget,
            quality_tier=tier,
            source_map=source_map,
            payload_path=payload_path,
            dependency_voxel_size=args.dependency_voxel_size,
        )
        tag = (
            f"b{budget:05d}_t{track_budget:05d}_{tier}_"
            f"{args.base_selection}"
        )
        path = output_dir / f"track_centric_{tag}.pt"
        torch.save(state, path)
        summary["maps"][tag] = {
            "path": str(path),
            "track_count": int(tracks.numel()),
            "base_reserve_count": reserve,
            "base_selection": args.base_selection,
            "quality": {
                "median_reprojection_px": float(
                    torch.median(
                        torch.as_tensor(
                            geometry["triangulation_reprojection_median_px"]
                        )[tracks]
                    )
                ),
                "median_parallax_deg": float(
                    torch.median(
                        torch.as_tensor(
                            geometry["triangulation_parallax_deg"]
                        )[tracks]
                    )
                ),
            },
        }
    report = output_dir / "track_centric_build.json"
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
