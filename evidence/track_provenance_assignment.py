"""Frozen-2DGS provenance assignment shared by bootstrap and factor replay."""

from __future__ import annotations

from collections import defaultdict

import torch
from tqdm import tqdm

from evidence.triangulation import transfer_triangulated_track_groups_to_landmarks
from features.multiview_fusion import PIXEL_CENTER_OFFSET
from priors.rasterizer import bank_splat_provenance_2dgs
from priors.rendering import render_from_pose_gsplat


@torch.no_grad()
def assign_tracks_by_splat_provenance(
    *,
    tracks,
    track_geometry,
    keypoints,
    query_names,
    cache,
    bank_xyz,
    gaussians,
    cameras_by_name,
    landmark_global_indices,
    background,
    topk: int,
    minimum_consensus_rate: float,
    minimum_views: int,
    group_maximum_landmarks: int,
    group_minimum_relative_mass: float,
    group_minimum_consensus_rate: float,
    depth_absolute_tolerance_m: float,
    depth_relative_tolerance: float,
):
    """Assign independent tracks using frozen 2DGS composition provenance.

    The routine is intentionally independent of bootstrap CLI state so the same
    frozen renderer contract can be replayed for a changed Track graph.
    """
    track_count = int(track_geometry["triangulated_xyz"].shape[0])
    high_confidence = torch.as_tensor(
        track_geometry["triangulation_high_confidence"], dtype=torch.bool
    )
    track_candidates = [defaultdict(float) for _ in range(track_count)]
    track_candidate_views = [defaultdict(set) for _ in range(track_count)]
    observations_by_query = defaultdict(list)
    render_device = torch.as_tensor(landmark_global_indices).device
    for observation, (track, query) in enumerate(
        zip(tracks["track_index"].tolist(), tracks["query_index"].tolist())
    ):
        if bool(high_confidence[track]):
            observations_by_query[query].append(observation)
    valid_observations = 0
    for query, observations in tqdm(
        sorted(observations_by_query.items()),
        desc="G3 frozen 2DGS provenance assignment",
    ):
        name = query_names[query]
        camera = cameras_by_name[name]
        cached = cache[name]
        height, width = map(int, cached["native_input_hw"])
        render_pkg = render_from_pose_gsplat(
            gaussians,
            cached["pose_w2c"].to(device=render_device).float(),
            camera.FoVx,
            camera.FoVy,
            width,
            height,
            bg_color=background,
            render_mode="RGB+ED",
            rgb_only=True,
            return_rgb_meta=True,
            rasterize_mode="antialiased",
        )
        observation_tensor = torch.as_tensor(observations, dtype=torch.long)
        local_keypoint_indices = tracks["keypoint_index"][observation_tensor]
        query_keypoints = (
            keypoints[query][local_keypoint_indices]
            - float(PIXEL_CENTER_OFFSET)
        ).to(device=render_device)
        local_ids, weights, valid = bank_splat_provenance_2dgs(
            query_keypoints,
            landmark_global_indices,
            render_pkg["rgb_meta"],
            rendered_depth=render_pkg.get("depth"),
            topk=int(topk),
            candidate_topk=max(int(topk) * 8, 32),
            depth_abs_tolerance=float(depth_absolute_tolerance_m),
            depth_rel_tolerance=float(depth_relative_tolerance),
        )
        for row, observation in enumerate(observations):
            if not bool(valid[row]):
                continue
            track = int(tracks["track_index"][observation])
            valid_observations += 1
            for landmark, weight in zip(
                local_ids[row].tolist(), weights[row].tolist()
            ):
                if weight <= 0.0:
                    continue
                track_candidates[track][landmark] += float(weight)
                track_candidate_views[track][landmark].add(query)
        del render_pkg, local_ids, weights, valid

    track_landmark = torch.full((track_count,), -1, dtype=torch.long)
    assignment_cost = torch.full(
        (track_count,), float("inf"), dtype=torch.float32
    )
    consensus_rate = torch.zeros(track_count, dtype=torch.float32)
    support_views = torch.zeros(track_count, dtype=torch.long)
    group_offsets = [0]
    group_landmarks = []
    group_costs = []
    group_rates = []
    group_support_views = []
    assigned = 0
    group_assigned_tracks = 0
    track_observation_counts = torch.bincount(
        torch.as_tensor(tracks["track_index"], dtype=torch.long),
        minlength=track_count,
    )
    for track in range(track_count):
        if not bool(high_confidence[track]):
            group_offsets.append(group_offsets[-1])
            continue
        candidates = track_candidates[track]
        if not candidates:
            group_offsets.append(group_offsets[-1])
            continue
        ordered_candidates = sorted(
            candidates.items(), key=lambda item: (-item[1], item[0])
        )
        landmark, mass = ordered_candidates[0]
        track_observations = int(track_observation_counts[track])
        rate = float(mass) / max(track_observations, 1)
        views = len(track_candidate_views[track][landmark])
        if (
            rate < float(minimum_consensus_rate)
            or views < int(minimum_views)
        ):
            group_offsets.append(group_offsets[-1])
            continue
        track_landmark[track] = int(landmark)
        consensus_rate[track] = rate
        support_views[track] = views
        assignment_cost[track] = 1.0 - min(rate, 1.0)
        assigned += 1

        accepted = []
        maximum_group = max(int(group_maximum_landmarks), 1)
        for candidate_landmark, candidate_mass in ordered_candidates:
            candidate_rate = float(candidate_mass) / max(track_observations, 1)
            candidate_views = len(
                track_candidate_views[track][candidate_landmark]
            )
            if (
                candidate_rate < float(group_minimum_consensus_rate)
                or candidate_mass
                < float(group_minimum_relative_mass) * float(mass)
                or candidate_views < int(minimum_views)
            ):
                continue
            accepted.append(
                (
                    int(candidate_landmark),
                    1.0 - min(candidate_rate, 1.0),
                    candidate_rate,
                    candidate_views,
                )
            )
            if len(accepted) >= maximum_group:
                break
        if not accepted:
            accepted.append((int(landmark), 1.0 - min(rate, 1.0), rate, views))
        group_landmarks.extend(item[0] for item in accepted)
        group_costs.extend(item[1] for item in accepted)
        group_rates.extend(item[2] for item in accepted)
        group_support_views.extend(item[3] for item in accepted)
        group_offsets.append(group_offsets[-1] + len(accepted))
        group_assigned_tracks += 1

    group_offsets = torch.as_tensor(group_offsets, dtype=torch.long)
    group_landmarks = torch.as_tensor(group_landmarks, dtype=torch.long)
    group_costs = torch.as_tensor(group_costs, dtype=torch.float32)
    group_rates = torch.as_tensor(group_rates, dtype=torch.float32)
    group_support_views = torch.as_tensor(group_support_views, dtype=torch.long)
    edge_tracks = torch.repeat_interleave(
        torch.arange(track_count, dtype=torch.long),
        group_offsets[1:] - group_offsets[:-1],
    )
    geometry, group_assignment = transfer_triangulated_track_groups_to_landmarks(
        track_geometry,
        edge_track_index=edge_tracks,
        edge_landmark_index=group_landmarks,
        landmark_count=int(bank_xyz.shape[0]),
        edge_assignment_cost=group_costs,
    )
    best_edge_all = group_assignment["landmark_best_edge_index"]
    selected = best_edge_all >= 0
    best_edge = best_edge_all[selected]
    best_track = edge_tracks[best_edge]
    assignment_distance = torch.full(
        (bank_xyz.shape[0],), float("inf"), dtype=torch.float32
    )
    assignment_distance[selected] = torch.linalg.norm(
        track_geometry["triangulated_xyz"][best_track]
        - bank_xyz.detach().cpu()[selected],
        dim=1,
    )
    geometry["track_assignment_distance_m"] = assignment_distance
    geometry["track_provenance_consensus_rate"] = torch.zeros(
        bank_xyz.shape[0], dtype=torch.float32
    )
    geometry["track_provenance_support_views"] = torch.zeros(
        bank_xyz.shape[0], dtype=torch.long
    )
    geometry["track_provenance_consensus_rate"][selected] = group_rates[best_edge]
    geometry["track_provenance_support_views"][selected] = (
        group_support_views[best_edge]
    )
    assignment = {
        "track_landmark_index": track_landmark,
        "track_assignment_cost": assignment_cost,
        "landmark_best_track_index": group_assignment[
            "landmark_best_track_index"
        ],
        "track_landmark_offsets": group_offsets,
        "track_landmark_indices": group_landmarks,
        "track_landmark_costs": group_costs,
    }
    diagnostics = {
        "geometry_teacher_provenance_valid_observation_count": valid_observations,
        "geometry_teacher_provenance_assigned_track_count": assigned,
        "geometry_teacher_provenance_assigned_landmark_count": int(
            selected.sum().item()
        ),
        "geometry_teacher_provenance_group_assigned_track_count": (
            group_assigned_tracks
        ),
        "geometry_teacher_provenance_group_edge_count": int(
            group_landmarks.numel()
        ),
        "geometry_teacher_provenance_group_size_mean": (
            float(group_landmarks.numel()) / max(group_assigned_tracks, 1)
        ),
    }
    return geometry, assignment, diagnostics


__all__ = ["assign_tracks_by_splat_provenance"]
