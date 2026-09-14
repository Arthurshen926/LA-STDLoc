#!/usr/bin/env python3
"""Materialize a mapping-only virtual-render plan without rendering images."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from evidence.triangulation import camera_pose_bins
from evidence.virtual_render_planner import (
    PlannerPolicy,
    SCHEMA,
    VERSION,
    assign_pose_bins_from_reference,
    camera_registry_sha256,
    camera_centers,
    generate_candidate_poses,
    greedy_capped_coverage,
    validate_mapping_inputs,
    zbuffer_supported_mask,
)
from features.sampling import unproject_pixels


def _validate_formal_unified_map(
    selected_map: dict, query_path: Path, track_path: Path
) -> None:
    construction = selected_map.get("projective_anchor_construction", {})
    if construction.get("schema") != "lafgs_gaussian_supported_projective_anchor_construction":
        raise ValueError("planner requires the formal unified V4 Anchor map")
    kind = torch.as_tensor(selected_map.get("anchor_candidate_kind"))
    if kind.ndim != 1 or not bool((kind == 0).any()) or not bool((kind == 1).any()):
        raise ValueError("formal unified map must contain Track and completion Anchors")
    provenance = selected_map.get("provenance", {})
    expected = {
        "query_cache_sha256": sha256_file(query_path),
        "track_payload_sha256": sha256_file(track_path),
    }
    for field, value in expected.items():
        if provenance.get(field) != value:
            raise ValueError(f"formal unified map parent SHA mismatch: {field}")


@torch.inference_mode()
def render_candidate_coverage_support(
    *, args, field: dict, candidates: dict, proxy_cells: list[torch.Tensor],
    poses: torch.Tensor, intrinsics: torch.Tensor, query_payload: dict,
    track_payload: dict,
) -> tuple[list[torch.Tensor], dict]:
    """Filter projected voxels by real candidate alpha/depth z-buffer evidence."""
    from priors.models import GaussianModel2D
    from priors.rendering import render_from_pose_gsplat
    from features.raster_sampling import sample_raster_at_grid_uv

    model = GaussianModel2D(int(args.sh_degree))
    model.load_ply(args.gaussian_ply, loc_feature_dim=0)
    model = model.to(args.device).eval()
    names = list(track_payload["query_names"])
    cache = query_payload.get("queries", query_payload)
    voxel_xyz = torch.as_tensor(field["voxel_center_xyz"]).float()
    kept, audits = [], []
    for candidate_index, (pose, parent, projected) in enumerate(zip(
        candidates["pose_w2c"], candidates["parent_camera_index"], proxy_cells
    )):
        parent = int(parent)
        height, width = map(int, torch.as_tensor(cache[names[parent]]["native_input_hw"]).tolist())
        scale = float(args.candidate_support_scale)
        render_h, render_w = max(32, round(height * scale)), max(32, round(width * scale))
        K = intrinsics[parent].float()
        fov_x = 2.0 * math.atan(width / (2.0 * float(K[0, 0])))
        fov_y = 2.0 * math.atan(height / (2.0 * float(K[1, 1])))
        package = render_from_pose_gsplat(
            model, pose.to(args.device), fov_x, fov_y, render_w, render_h,
            bg_color=torch.zeros(3, device=args.device), render_mode="RGB+ED",
            rgb_only=True, rasterize_mode="antialiased",
        )
        alpha = torch.as_tensor(package.get("alphas", package.get("rend_alpha"))).float().squeeze().cpu()
        depth = torch.as_tensor(package["depth"]).float().squeeze().cpu()
        cells = torch.as_tensor(projected).long()
        if cells.numel():
            xyz = voxel_xyz[cells]
            camera = (pose[:3, :3].float() @ xyz.T).T + pose[:3, 3].float()
            uvw = (K @ camera.T).T
            uv = uvw[:, :2] / camera[:, 2, None].clamp_min(1e-8)
            uv[:, 0] *= render_w / width
            uv[:, 1] *= render_h / height
            visible = zbuffer_supported_mask(
                uv, camera[:, 2], alpha, depth,
                alpha_minimum=float(args.candidate_alpha_minimum),
                depth_absolute_m=float(args.candidate_depth_absolute_m),
                depth_relative=float(args.candidate_depth_relative),
            )
            cells = cells[visible]
        kept.append(cells)
        audits.append({
            "candidate_index": candidate_index,
            "projected_voxel_count": int(torch.as_tensor(projected).numel()),
            "zbuffer_supported_voxel_count": int(cells.numel()),
            "alpha_supported_fraction": float((alpha >= float(args.candidate_alpha_minimum)).float().mean()),
            "render_hw": [render_h, render_w],
        })
        if (candidate_index + 1) % 32 == 0:
            print(json.dumps({"candidate_support_rendered": candidate_index + 1,
                              "candidate_count": len(proxy_cells)}), flush=True)
    return kept, {
        "mode": "real_low_resolution_alpha_depth_zbuffer",
        "gaussian_ply": str(args.gaussian_ply.resolve()),
        "gaussian_ply_sha256": sha256_file(args.gaussian_ply.resolve()),
        "support_scale": float(args.candidate_support_scale),
        "alpha_minimum": float(args.candidate_alpha_minimum),
        "depth_absolute_m": float(args.candidate_depth_absolute_m),
        "depth_relative": float(args.candidate_depth_relative),
        "candidates": audits,
    }


def _load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pair_distinct_count(inverse: torch.Tensor, labels: torch.Tensor, voxel_count: int):
    width = max(int(labels.max()) + 1, 1) if labels.numel() else 1
    pair = torch.unique(inverse.long() * width + labels.long())
    count = torch.zeros(voxel_count, dtype=torch.long)
    if pair.numel():
        count.index_add_(0, torch.div(pair, width, rounding_mode="floor"),
                         torch.ones_like(pair))
    return count


def _pair_csr(inverse: torch.Tensor, labels: torch.Tensor, voxel_count: int):
    width = max(int(labels.max()) + 1, 1) if labels.numel() else 1
    pair = torch.unique(inverse.long() * width + labels.long()).sort().values
    voxel = torch.div(pair, width, rounding_mode="floor")
    counts = torch.bincount(voxel, minlength=voxel_count)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    return offsets, pair % width


def build_coverage_field(
    query_payload: dict,
    track_payload: dict,
    selected_map: dict | None,
    policy: PlannerPolicy,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = query_payload.get("queries", query_payload)
    names = list(track_payload["query_names"])
    if set(names) != set(cache):
        raise ValueError("Track and mapping-cache query registries differ")
    poses = torch.stack([torch.as_tensor(cache[name]["pose_w2c"]).float() for name in names])
    intrinsics = torch.stack([torch.as_tensor(cache[name]["native_K"]).float() for name in names])
    bins = torch.as_tensor(track_payload.get("query_bins", camera_pose_bins(poses, 8))).long()

    surface_xyz = []
    surface_family = []
    surface_bin = []
    stride = int(policy.surface_stride)
    for query, name in enumerate(names):
        record = cache[name]
        depth_source = record.get("native_depth", record.get("native_rendered_depth"))
        alpha_source = record.get("native_alpha", record.get("native_rendered_alpha"))
        if depth_source is None or alpha_source is None:
            raise ValueError("planner cache requires rendered alpha/depth rasters")
        depth = torch.as_tensor(depth_source).float()
        alpha = torch.as_tensor(alpha_source).float()
        if depth.shape != alpha.shape:
            raise ValueError("mapping alpha/depth rasters must align")
        y = torch.arange(stride // 2, depth.shape[0], stride)
        x = torch.arange(stride // 2, depth.shape[1], stride)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        value = depth[yy, xx].reshape(-1)
        valid = torch.isfinite(value) & (value > 0) & (
            alpha[yy, xx].reshape(-1) >= float(policy.alpha_minimum)
        )
        if bool(valid.any()):
            physical = grid[valid].float() + float(record.get("pixel_center_offset", 0.5))
            xyz = unproject_pixels(physical, value[valid], intrinsics[query], poses[query])
            finite = torch.isfinite(xyz).all(1)
            xyz = xyz[finite]
            surface_xyz.append(xyz)
            surface_family.append(torch.full((xyz.shape[0],), query, dtype=torch.long))
            surface_bin.append(torch.full((xyz.shape[0],), int(bins[query]), dtype=torch.long))

    geometry = track_payload["track_geometry"]
    track_xyz_all = torch.as_tensor(geometry["triangulated_xyz"]).float()
    triangulated = torch.as_tensor(geometry["triangulated"]).bool() & torch.isfinite(track_xyz_all).all(1)
    observations = track_payload["tracks"]
    observation_track = torch.as_tensor(observations["track_index"]).long()
    observation_query = torch.as_tensor(observations["query_index"]).long()
    keep = triangulated[observation_track]
    track_xyz = track_xyz_all[observation_track[keep]]
    track_family = observation_query[keep]
    track_bin = bins[track_family]
    confidence_level = torch.as_tensor(geometry["track_confidence_level"]).long()
    stable = confidence_level[observation_track[keep]] >= 2
    selected_tracks = torch.empty(0, dtype=torch.long)
    if selected_map is not None:
        selected_tracks = torch.as_tensor(selected_map["track_cluster_ids"]).long()
        selected_tracks = selected_tracks[selected_tracks >= 0].unique()
    selected_observation = torch.isin(observation_track[keep], selected_tracks)

    xyz_parts = [*surface_xyz, track_xyz]
    if not xyz_parts or sum(int(value.shape[0]) for value in xyz_parts) == 0:
        raise ValueError("coverage field has no finite mapping evidence")
    xyz = torch.cat(xyz_parts)
    family = torch.cat((*surface_family, track_family))
    view_bin = torch.cat((*surface_bin, track_bin))
    surface_rows = sum(int(value.shape[0]) for value in surface_xyz)
    is_surface = torch.arange(xyz.shape[0]) < surface_rows
    is_stable = torch.cat((torch.ones(surface_rows, dtype=torch.bool), stable))
    is_selected = torch.cat((torch.zeros(surface_rows, dtype=torch.bool), selected_observation))
    origin = xyz.amin(0)
    voxel_coord = torch.floor((xyz - origin) / float(policy.voxel_size_m)).long()
    unique_voxel, inverse = torch.unique(voxel_coord, dim=0, return_inverse=True)
    voxel_count = int(unique_voxel.shape[0])
    count = torch.bincount(inverse, minlength=voxel_count)
    centers = torch.zeros(voxel_count, 3)
    centers.index_add_(0, inverse, xyz)
    centers /= count[:, None].clamp_min(1)
    family_offsets, family_indices = _pair_csr(inverse, family, voxel_count)
    family_count = family_offsets[1:] - family_offsets[:-1]
    bin_count = _pair_distinct_count(inverse, view_bin, voxel_count)
    stable_count = _pair_distinct_count(
        inverse[is_stable], family[is_stable], voxel_count
    )
    surface_count = torch.zeros(voxel_count, dtype=torch.long)
    surface_count.index_add_(0, inverse[is_surface], torch.ones(int(is_surface.sum()), dtype=torch.long))
    selected_count = torch.zeros(voxel_count, dtype=torch.long)
    selected_count.index_add_(0, inverse[is_selected], torch.ones(int(is_selected.sum()), dtype=torch.long))
    track_count = torch.zeros(voxel_count, dtype=torch.long)
    track_mask = ~is_surface
    track_count.index_add_(0, inverse[track_mask], torch.ones(int(track_mask.sum()), dtype=torch.long))
    demand = (
        (int(policy.target_families) - family_count).clamp_min(0)
        + (int(policy.target_view_bins) - bin_count).clamp_min(0)
        + (int(policy.target_stable_observations) - stable_count).clamp_min(0)
        + ((track_count > 0) & (selected_count == 0)).long()
    ).float()
    field = {
        "voxel_coord": unique_voxel,
        "voxel_center_xyz": centers,
        "surface_support_count": surface_count,
        "camera_family_count": family_count,
        "camera_family_offsets": family_offsets,
        "camera_family_indices": family_indices,
        "view_bin_count": bin_count,
        "stable_observation_count": stable_count,
        "track_observation_count": track_count,
        "selected_track_observation_count": selected_count,
        "deficit_demand": demand,
    }
    return field, poses, intrinsics, bins


def candidate_coverage(
    field: dict,
    candidates: dict,
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    query_payload: dict,
    track_payload: dict,
    policy: PlannerPolicy,
):
    names = list(track_payload["query_names"])
    cache = query_payload.get("queries", query_payload)
    full_xyz = field["voxel_center_xyz"].double()
    demand = field["deficit_demand"]
    source_supported = field["surface_support_count"] > 0
    source_centers = camera_centers(poses)
    active_index = torch.nonzero(
        source_supported & (demand > 0), as_tuple=False
    ).reshape(-1)
    xyz = full_xyz[active_index]
    family_offsets = torch.as_tensor(field["camera_family_offsets"]).long()
    family_indices = torch.as_tensor(field["camera_family_indices"]).long()
    incidence_voxel = torch.repeat_interleave(
        torch.arange(full_xyz.shape[0]), family_offsets[1:] - family_offsets[:-1]
    )
    incidence_keep = source_supported[incidence_voxel] & (demand[incidence_voxel] > 0)
    incidence_voxel = incidence_voxel[incidence_keep]
    incidence_family = family_indices[incidence_keep]
    active_lookup = torch.full((full_xyz.shape[0],), -1, dtype=torch.long)
    active_lookup[active_index] = torch.arange(active_index.numel())
    incidence_local_voxel = active_lookup[incidence_voxel]
    incidence_source_ray = torch.nn.functional.normalize(
        full_xyz[incidence_voxel] - source_centers[incidence_family], dim=1
    )
    cells = []
    parallax = []
    appearance = []
    for candidate, parent in zip(candidates["pose_w2c"], candidates["parent_camera_index"]):
        parent = int(parent)
        K = intrinsics[parent].double()
        pose = candidate.double()
        camera = (pose[:3, :3] @ xyz.T).T + pose[:3, 3]
        depth = camera[:, 2]
        uvw = (K @ camera.T).T
        uv = uvw[:, :2] / depth[:, None].clamp_min(1e-8)
        height, width = map(int, cache[names[parent]]["native_input_hw"])
        center = camera_centers(pose[None])[0]
        parent_distance = torch.linalg.norm(xyz - source_centers[parent], dim=1)
        candidate_distance = torch.linalg.norm(xyz - center, dim=1)
        ratio = candidate_distance / parent_distance.clamp_min(1e-6)
        parent_ray = torch.nn.functional.normalize(xyz - source_centers[parent], dim=1)
        candidate_ray = torch.nn.functional.normalize(xyz - center, dim=1)
        angle = torch.rad2deg(torch.acos((parent_ray * candidate_ray).sum(1).clamp(-1, 1)))
        incidence_cosine = (
            incidence_source_ray * candidate_ray[incidence_local_voxel]
        ).sum(1).clamp(-1, 1)
        incidence_parallax = torch.rad2deg(torch.acos(incidence_cosine))
        maximum_parallax = torch.zeros(xyz.shape[0], dtype=torch.float64)
        maximum_parallax.scatter_reduce_(
            0, incidence_local_voxel, incidence_parallax,
            reduce="amax", include_self=True,
        )
        valid = (
            (depth > 0)
            & (uv[:, 0] >= 0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            & (ratio >= 0.5) & (ratio <= 2.0)
            & (angle <= float(policy.maximum_view_change_deg))
        )
        local_index = torch.nonzero(valid, as_tuple=False).reshape(-1)
        index = active_index[local_index]
        cells.append(index)
        parallax.append(
            float((maximum_parallax[local_index] / 30.0).clamp_max(1).mean())
            if index.numel() else 0.0
        )
        appearance.append(
            float(torch.cos(torch.deg2rad(angle[local_index])).clamp_min(0).mean())
            if index.numel() else 0.0
        )
    return cells, torch.tensor(parallax), torch.tensor(appearance)


def run(args) -> dict:
    torch.set_num_threads(max(1, int(getattr(args, "cpu_threads", 8))))
    query_path = args.query_cache.resolve()
    track_path = args.track_payload.resolve()
    query_payload = _load(query_path)
    track_payload = _load(track_path)
    validate_mapping_inputs(query_payload, track_payload)
    selected_map = _load(args.selected_map.resolve()) if args.selected_map else None
    if bool(getattr(args, "require_formal_unified_map", False)):
        if selected_map is None:
            raise ValueError("formal planner requires --selected-map")
        _validate_formal_unified_map(selected_map, query_path, track_path)
    policy = PlannerPolicy(**{
        "selected_view_budget": int(args.view_budget),
        "maximum_candidates": int(args.maximum_candidates),
        "voxel_size_m": float(args.voxel_size_m),
        "surface_stride": int(args.surface_stride),
    })
    field, poses, intrinsics, formal_bins = build_coverage_field(
        query_payload, track_payload, selected_map, policy
    )
    deficit = field["voxel_center_xyz"][field["deficit_demand"] > 0]
    candidates = generate_candidate_poses(poses, deficit, policy)
    candidates["formal_pose_bin"] = assign_pose_bins_from_reference(
        candidates["pose_w2c"], poses, formal_bins
    )
    coverage, parallax, appearance = candidate_coverage(
        field, candidates, poses, intrinsics, query_payload, track_payload, policy
    )
    candidate_support = None
    if getattr(args, "gaussian_ply", None) is not None:
        coverage, candidate_support = render_candidate_coverage_support(
            args=args, field=field, candidates=candidates, proxy_cells=coverage,
            poses=poses, intrinsics=intrinsics, query_payload=query_payload,
            track_payload=track_payload,
        )
    selected, trace = greedy_capped_coverage(
        coverage,
        field["deficit_demand"],
        candidates["pose_family"],
        budget=policy.selected_view_budget,
        maximum_per_family=policy.maximum_per_family,
        coverage_cap=policy.coverage_cap,
        parallax=parallax,
        appearance=appearance,
        artifact_risk=candidates["artifact_risk"],
        parallax_weight=policy.parallax_weight,
        appearance_weight=policy.appearance_weight,
        risk_weight=policy.risk_weight,
    )
    payload = {
        "schema": SCHEMA,
        "version": VERSION,
        "mapping_only": True,
        "uses_test_queries": False,
        "uses_source_mapping_rgb": False,
        "renders_images": candidate_support is not None,
        "default_pipeline_enabled": False,
        "policy": asdict(policy),
        "inputs": {
            "query_cache": str(query_path),
            "query_cache_sha256": sha256_file(query_path),
            "track_payload": str(track_path),
            "track_payload_sha256": sha256_file(track_path),
            "selected_map": str(args.selected_map.resolve()) if args.selected_map else None,
            "selected_map_sha256": sha256_file(args.selected_map.resolve()) if args.selected_map else None,
            "canonical_camera_registry_sha256": camera_registry_sha256(
                query_payload, list(track_payload["query_names"])
            ),
        },
        "coverage_field": field,
        "candidates": {
            **candidates,
            "covered_voxel_ids": coverage,
            "parallax_utility": parallax,
            "appearance_utility": appearance,
        },
        "selected_candidate_indices": selected,
        "greedy_trace": trace,
        "triangulation_family_contract": {
            "source_and_pose_proximity_components": True,
            "maximum_evidence_per_pose_family": 1,
        },
        "formal_pose_bin_frame": {
            "bin_count": int(torch.unique(formal_bins).numel()),
            "reference_query_count": int(formal_bins.numel()),
            "reference_query_bins": formal_bins,
            "assignment": "nearest_reference_bin_centroid_center_plus_direction",
        },
        "candidate_render_support": candidate_support,
        "gt_visible_diagnostic": None,
    }
    _atomic_save(payload, args.output.resolve())
    summary = {
        "schema": SCHEMA,
        "output": str(args.output.resolve()),
        "coverage_voxel_count": int(field["deficit_demand"].numel()),
        "deficit_voxel_count": int((field["deficit_demand"] > 0).sum()),
        "initial_deficit_demand": float(field["deficit_demand"].sum()),
        "candidate_count": int(candidates["pose_w2c"].shape[0]),
        "selected_candidate_count": int(selected.numel()),
        "selected_covered_voxel_count": int(torch.unique(torch.cat([
            coverage[int(index)] for index in selected
        ])).numel()) if selected.numel() else 0,
        "remaining_deficit_demand_proxy": trace[-1]["remaining_demand"] if trace else float(field["deficit_demand"].sum()),
        "uses_test_queries": False,
        "renders_images": candidate_support is not None,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--selected-map", type=Path)
    parser.add_argument("--gaussian-ply", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--candidate-support-scale", type=float, default=0.25)
    parser.add_argument("--candidate-alpha-minimum", type=float, default=0.2)
    parser.add_argument("--candidate-depth-absolute-m", type=float, default=0.10)
    parser.add_argument("--candidate-depth-relative", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-budget", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=512)
    parser.add_argument("--voxel-size-m", type=float, default=0.25)
    parser.add_argument("--surface-stride", type=int, default=24)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args(argv)
    args.require_formal_unified_map = True
    if min(args.view_budget, args.maximum_candidates, args.surface_stride) <= 0:
        parser.error("budgets and surface stride must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
