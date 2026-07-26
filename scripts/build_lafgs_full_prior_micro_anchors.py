#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial import cKDTree
from tqdm import tqdm

from gaussian_renderer import render_from_pose_gsplat
from localization_training.micro_anchors import (
    compute_track_coverage_gain,
    fuse_track_descriptors,
)
from localization_training.splat_provenance import (
    bank_splat_provenance_2dgs,
)
from scene.gaussian_model import GaussianModel_2dgs


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_centers(path):
    vertex = PlyData.read(path)["vertex"].data
    return np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1).copy()


def _expand_visibility(base_visibility, canonical_source_ids, base_source_ids):
    source_to_row = {
        int(source): row for row, source in enumerate(base_source_ids.tolist())
    }
    row_lookup = torch.as_tensor(
        [source_to_row[int(source)] for source in canonical_source_ids.tolist()],
        dtype=torch.long,
    )
    return {
        name: torch.as_tensor(mask, dtype=torch.bool)[row_lookup]
        for name, mask in base_visibility.items()
    }


def _gather_candidate_csr(offsets, values, selected):
    selected = torch.as_tensor(selected, dtype=torch.long)
    offsets = torch.as_tensor(offsets, dtype=torch.long)
    values = torch.as_tensor(values)
    chunks = [
        values[int(offsets[row]) : int(offsets[row + 1])]
        for row in selected.tolist()
    ]
    lengths = torch.as_tensor(
        [chunk.shape[0] for chunk in chunks], dtype=torch.long
    )
    output_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), lengths.cumsum(dim=0))
    )
    output_values = (
        torch.cat(chunks)
        if chunks
        else values.new_empty((0,) + values.shape[1:])
    )
    return output_offsets, output_values


def _append_candidates(canonical, candidates, selected, *, config, provenance):
    selected = torch.as_tensor(selected, dtype=torch.long)
    base_rows = int(canonical["anchor_ids"].numel())
    output = {
        "version": 4,
        "schema": "lafgs_materialized_anchor_map",
        "anchor_ids": torch.arange(base_rows + selected.numel()),
        "source_primitive_ids": torch.cat(
            (
                canonical["source_primitive_ids"],
                candidates["source_primitive_ids"][selected],
            )
        ),
        "track_cluster_ids": torch.cat(
            (
                canonical["track_cluster_ids"],
                candidates["track_ids"][selected],
            )
        ),
        "anchor_xyz": torch.cat(
            (canonical["anchor_xyz"], candidates["anchor_xyz"][selected])
        ),
        "anchor_features": torch.cat(
            (
                canonical["anchor_features"],
                candidates["anchor_features"][selected],
            )
        ),
        "anchor_type": torch.cat(
            (
                canonical["anchor_type"],
                torch.full((selected.numel(),), 3, dtype=torch.int8),
            )
        ),
        "base_anchor_count": int(canonical["base_anchor_count"]),
        "canonical_anchor_count": base_rows,
        "requested_micro_anchor_budget": int(
            canonical["micro_anchor_count"] + selected.numel()
        ),
        "micro_anchor_count": int(
            canonical["micro_anchor_count"] + selected.numel()
        ),
        "config": config,
        "provenance": provenance,
        "full_prior_quality": {
            key: value[selected]
            for key, value in candidates["quality"].items()
        },
    }
    group_offsets, group_ids = _gather_candidate_csr(
        candidates["source_group_offsets"],
        candidates["source_group_primitive_ids"],
        selected,
    )
    _, group_responsibilities = _gather_candidate_csr(
        candidates["source_group_offsets"],
        candidates["source_group_responsibilities"],
        selected,
    )
    _, group_costs = _gather_candidate_csr(
        candidates["source_group_offsets"],
        candidates["source_group_costs"],
        selected,
    )
    output.update(
        {
            "full_prior_source_group_offsets": group_offsets,
            "full_prior_source_group_primitive_ids": group_ids,
            "full_prior_source_group_responsibilities": (
                group_responsibilities
            ),
            "full_prior_source_group_costs": group_costs,
        }
    )
    return output


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--base-visibility-cache", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", default="128,256,512,1000")
    parser.add_argument("--knn-primitives", type=int, default=64)
    parser.add_argument("--maximum-source-distance-m", type=float, default=0.5)
    parser.add_argument("--provenance-topk", type=int, default=4)
    parser.add_argument("--minimum-consensus-rate", type=float, default=0.35)
    parser.add_argument("--minimum-support-views", type=int, default=2)
    parser.add_argument("--group-max-primitives", type=int, default=4)
    parser.add_argument("--group-min-relative-mass", type=float, default=0.25)
    parser.add_argument("--group-min-consensus-rate", type=float, default=0.1)
    parser.add_argument("--descriptor-trim-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("full-prior provenance requires CUDA")

    canonical_path = Path(args.canonical_map).resolve()
    payload_path = Path(args.track_payload).resolve()
    query_path = Path(args.query_cache).resolve()
    visibility_path = Path(args.base_visibility_cache).resolve()
    ply_path = Path(args.gaussian_ply).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted(
        {int(value) for value in args.budgets.split(",") if value.strip()}
    )
    canonical = torch.load(
        canonical_path, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        payload_path, map_location="cpu", weights_only=False
    )
    print(f"Loading query cache: {query_path}", flush=True)
    query_payload = torch.load(
        query_path, map_location="cpu", weights_only=False
    )
    query_cache = query_payload.get("queries", query_payload)
    visibility_payload = torch.load(
        visibility_path, map_location="cpu", weights_only=False
    )
    base_visibility = visibility_payload.get(
        "visibility", visibility_payload
    )

    centers_np = _load_centers(ply_path)
    primitive_count = int(centers_np.shape[0])
    primitive_tree = cKDTree(centers_np)
    gaussians = GaussianModel_2dgs(3)
    gaussians.load_ply(str(ply_path))
    background = torch.zeros(3, device="cuda")

    geometry = payload["track_geometry"]
    tracks = payload["tracks"]
    assignment = payload["assignment"]
    high_confidence = torch.as_tensor(
        geometry["triangulation_high_confidence"], dtype=torch.bool
    )
    level = torch.as_tensor(
        geometry["track_confidence_level"], dtype=torch.int8
    )
    old_assigned = torch.as_tensor(
        assignment["track_landmark_index"], dtype=torch.long
    ) >= 0
    candidate_mask = high_confidence & (level == 2) & ~old_assigned
    candidate_tracks = torch.nonzero(
        candidate_mask, as_tuple=False
    ).reshape(-1)
    track_xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()

    canonical_base_count = int(canonical["base_anchor_count"])
    base_source_ids = canonical["source_primitive_ids"][
        :canonical_base_count
    ].long()
    canonical_source_ids = canonical["source_primitive_ids"].long()
    base_source_set = set(base_source_ids.tolist())
    distances, local_sources = primitive_tree.query(
        track_xyz[candidate_tracks].numpy(),
        k=min(int(args.knn_primitives), primitive_count),
    )
    if local_sources.ndim == 1:
        local_sources = local_sources[:, None]
        distances = distances[:, None]
    track_source_candidates = {}
    for row, track in enumerate(candidate_tracks.tolist()):
        sources = [
            int(source)
            for source, distance in zip(
                local_sources[row].tolist(), distances[row].tolist()
            )
            if (
                float(distance) <= float(args.maximum_source_distance_m)
                and int(source) not in base_source_set
            )
        ]
        track_source_candidates[int(track)] = set(sources)

    observations_by_query = defaultdict(list)
    for observation, (track, query) in enumerate(
        zip(tracks["track_index"].tolist(), tracks["query_index"].tolist())
    ):
        if bool(candidate_mask[int(track)]) and track_source_candidates[
            int(track)
        ]:
            observations_by_query[int(query)].append(observation)

    track_mass = [defaultdict(float) for _ in range(track_xyz.shape[0])]
    track_views = [defaultdict(set) for _ in range(track_xyz.shape[0])]
    valid_observations = 0
    query_names = payload["query_names"]
    for query, observations in tqdm(
        sorted(observations_by_query.items()),
        desc="M4 sparse full-prior provenance",
    ):
        name = query_names[query]
        cached = query_cache[name]
        height, width = map(int, cached["native_input_hw"])
        K = torch.as_tensor(cached["native_K"]).float()
        fov_x = 2.0 * math.atan(float(width) / (2.0 * float(K[0, 0])))
        fov_y = 2.0 * math.atan(float(height) / (2.0 * float(K[1, 1])))
        render_pkg = render_from_pose_gsplat(
            gaussians,
            torch.as_tensor(cached["pose_w2c"]).cuda().float(),
            fov_x,
            fov_y,
            width,
            height,
            bg_color=background,
            render_mode="RGB+ED",
            rgb_only=True,
            return_rgb_meta=True,
            rasterize_mode="antialiased",
        )
        query_sources = sorted(
            set().union(
                *(
                    track_source_candidates[
                        int(tracks["track_index"][observation])
                    ]
                    for observation in observations
                )
            )
        )
        source_tensor = torch.as_tensor(
            query_sources, device="cuda", dtype=torch.long
        )
        observation_tensor = torch.as_tensor(observations, dtype=torch.long)
        keypoint_indices = tracks["keypoint_index"][observation_tensor].long()
        keypoints = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[keypoint_indices].cuda()
        local_ids, weights, valid = bank_splat_provenance_2dgs(
            keypoints,
            source_tensor,
            render_pkg["rgb_meta"],
            rendered_depth=render_pkg.get("depth"),
            topk=args.provenance_topk,
            candidate_topk=max(int(args.provenance_topk) * 8, 32),
        )
        global_ids = source_tensor[local_ids]
        for row, observation in enumerate(observations):
            if not bool(valid[row]):
                continue
            track = int(tracks["track_index"][observation])
            allowed = track_source_candidates[track]
            accepted = False
            for source, weight in zip(
                global_ids[row].tolist(), weights[row].tolist()
            ):
                if float(weight) <= 0.0 or int(source) not in allowed:
                    continue
                track_mass[track][int(source)] += float(weight)
                track_views[track][int(source)].add(query)
                accepted = True
            valid_observations += int(accepted)
        del render_pkg, source_tensor, local_ids, weights, valid, global_ids

    observation_counts = torch.bincount(
        tracks["track_index"].long(), minlength=track_xyz.shape[0]
    )
    accepted_tracks = []
    accepted_groups = {}
    accepted_rates = {}
    accepted_views = {}
    for track in candidate_tracks.tolist():
        ordered = sorted(
            track_mass[track].items(), key=lambda item: (-item[1], item[0])
        )
        if not ordered:
            continue
        observations = max(int(observation_counts[track]), 1)
        best_source, best_mass = ordered[0]
        best_rate = float(best_mass) / observations
        best_views = len(track_views[track][best_source])
        if (
            best_rate < float(args.minimum_consensus_rate)
            or best_views < int(args.minimum_support_views)
        ):
            continue
        group = []
        for source, mass in ordered:
            rate = float(mass) / observations
            views = len(track_views[track][source])
            if (
                rate < float(args.group_min_consensus_rate)
                or float(mass)
                < float(args.group_min_relative_mass) * float(best_mass)
                or views < int(args.minimum_support_views)
            ):
                continue
            group.append(int(source))
            if len(group) >= int(args.group_max_primitives):
                break
        accepted_tracks.append(int(track))
        accepted_groups[int(track)] = group
        accepted_rates[int(track)] = best_rate
        accepted_views[int(track)] = best_views

    canonical_visibility = _expand_visibility(
        base_visibility, canonical_source_ids, base_source_ids
    )
    accepted_mask = torch.zeros(track_xyz.shape[0], dtype=torch.bool)
    accepted_mask[torch.as_tensor(accepted_tracks)] = True
    coverage = compute_track_coverage_gain(
        payload=payload,
        query_cache=query_cache,
        base_xyz=canonical["anchor_xyz"],
        visibility_cache=canonical_visibility,
        candidate_track_mask=accepted_mask,
    )
    accepted_tracks = [
        track
        for track in accepted_tracks
        if (
            int(coverage["coverage_gain"][track]) > 0
            and int(
                geometry["triangulation_distinct_view_bin_count"][track]
            )
            >= 2
        )
    ]
    accepted_tensor = torch.as_tensor(accepted_tracks, dtype=torch.long)
    features = fuse_track_descriptors(
        payload=payload,
        query_cache=query_cache,
        track_indices=accepted_tensor,
        trim_fraction=args.descriptor_trim_fraction,
    )
    source_ids = torch.as_tensor(
        [accepted_groups[track][0] for track in accepted_tracks],
        dtype=torch.long,
    )
    score = (
        coverage["coverage_gain"][accepted_tensor].float() * 1000.0
        + torch.as_tensor(
            geometry["triangulation_distinct_view_bin_count"]
        )[accepted_tensor].float()
        * 10.0
        + observation_counts[accepted_tensor].float()
        - torch.as_tensor(
            geometry["triangulation_reprojection_median_px"]
        )[accepted_tensor].float()
    )
    order = torch.argsort(score, descending=True, stable=True)
    ordered_tracks = accepted_tensor[order]
    source_group_offsets = [0]
    source_group_ids = []
    source_group_responsibilities = []
    source_group_costs = []
    for track in ordered_tracks.tolist():
        group = accepted_groups[int(track)]
        masses = torch.as_tensor(
            [track_mass[int(track)][source] for source in group]
        ).float()
        responsibilities = masses / masses.sum().clamp_min(1e-12)
        source_group_ids.extend(group)
        source_group_responsibilities.extend(responsibilities.tolist())
        source_group_costs.extend(
            (-responsibilities.clamp_min(1e-12).log()).tolist()
        )
        source_group_offsets.append(len(source_group_ids))
    candidates = {
        "track_ids": ordered_tracks,
        "source_primitive_ids": source_ids[order],
        "anchor_xyz": track_xyz[accepted_tensor][order],
        "anchor_features": features[order],
        "source_group_offsets": torch.as_tensor(
            source_group_offsets, dtype=torch.long
        ),
        "source_group_primitive_ids": torch.as_tensor(
            source_group_ids, dtype=torch.long
        ),
        "source_group_responsibilities": torch.as_tensor(
            source_group_responsibilities, dtype=torch.float32
        ),
        "source_group_costs": torch.as_tensor(
            source_group_costs, dtype=torch.float32
        ),
        "quality": {
            "coverage_gain": coverage["coverage_gain"][accepted_tensor][order],
            "consensus_rate": torch.as_tensor(
                [accepted_rates[track] for track in accepted_tracks]
            )[order],
            "support_views": torch.as_tensor(
                [accepted_views[track] for track in accepted_tracks]
            )[order],
            "triangulation_covariance_trace_m2": torch.as_tensor(
                geometry["triangulation_covariance_trace"]
            )[accepted_tensor][order],
            "triangulation_reprojection_median_px": torch.as_tensor(
                geometry["triangulation_reprojection_median_px"]
            )[accepted_tensor][order],
        },
    }
    provenance = {
        "canonical_map_path": str(canonical_path),
        "canonical_map_sha256": _sha256(canonical_path),
        "track_payload_path": str(payload_path),
        "track_payload_sha256": _sha256(payload_path),
        "query_cache_path": str(query_path),
        "query_cache_signature": query_payload.get("signature"),
        "base_visibility_cache_path": str(visibility_path),
        "base_visibility_cache_sha256": _sha256(visibility_path),
        "gaussian_ply_path": str(ply_path),
        "gaussian_ply_sha256": _sha256(ply_path),
        "full_prior_primitive_count": primitive_count,
        "statistics_split": "all_895_mapping_train",
    }
    summary = {
        "eligible_unassigned_level_a_track_count": int(
            candidate_tracks.numel()
        ),
        "provenance_query_count": len(observations_by_query),
        "valid_provenance_observation_count": valid_observations,
        "provenance_accepted_track_count": len(accepted_rates),
        "coverage_accepted_track_count": len(accepted_tracks),
        "outside_48k_source_count": int(torch.unique(source_ids).numel()),
        "candidate_count": int(accepted_tensor.numel()),
        "budgets": {},
    }
    for budget in budgets:
        selected = torch.arange(min(budget, accepted_tensor.numel()))
        config = {
            "method": "micro_anchor_v4_full_rgb_prior_provenance",
            "canonical_v1_frozen": True,
            "full_prior_only_outside_48k": True,
            "requested_new_anchor_budget": int(budget),
            "knn_primitives": int(args.knn_primitives),
            "maximum_source_distance_m": float(
                args.maximum_source_distance_m
            ),
            "minimum_consensus_rate": float(args.minimum_consensus_rate),
            "minimum_support_views": int(args.minimum_support_views),
        }
        output = _append_candidates(
            canonical,
            candidates,
            selected,
            config=config,
            provenance=provenance,
        )
        path = output_dir / f"full_prior_micro_anchor_{budget:04d}.pt"
        torch.save(output, path)
        summary["budgets"][str(budget)] = {
            "state": str(path),
            "selected_count": int(selected.numel()),
            "coverage_gain_sum": int(
                candidates["quality"]["coverage_gain"][selected].sum()
            ),
            "source_count": int(
                torch.unique(
                    candidates["source_primitive_ids"][selected]
                ).numel()
            ),
        }
    (output_dir / "full_prior_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
