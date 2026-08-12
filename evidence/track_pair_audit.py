"""Mapping-only diagnostics for the image pair graph used to build Tracks.

The audit intentionally does not change :func:`candidate_camera_pairs` or any
downstream Track construction.  It reconstructs the frozen candidate graph,
then measures whether its finite pair budget is dominated by adjacent,
near-duplicate views or contains useful geometric baselines.
"""

from __future__ import annotations

from collections import defaultdict
import re

import torch
import torch.nn.functional as F

from evidence.triangulation import candidate_camera_pairs


_FRAME_PATTERN = re.compile(r"^(?P<prefix>.*?)(?P<frame>\d+)(?:\D*)$")


def _camera_centers_and_axes(
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = torch.as_tensor(pose_w2c, dtype=torch.float64)
    centers = -torch.einsum("qji,qj->qi", pose[:, :3, :3], pose[:, :3, 3])
    axes = torch.einsum(
        "qji,j->qi",
        pose[:, :3, :3],
        pose.new_tensor([0.0, 0.0, 1.0]),
    )
    return centers, F.normalize(axes, dim=1)


def _temporal_identity(name: str) -> tuple[str, int] | None:
    match = _FRAME_PATTERN.match(str(name))
    if match is None:
        return None
    return match.group("prefix"), int(match.group("frame"))


def _quantile(values: torch.Tensor, probability: float) -> float | None:
    values = torch.as_tensor(values, dtype=torch.float64)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return float(torch.quantile(values, float(probability)).item())


def _distribution(values: torch.Tensor) -> dict[str, float | int | None]:
    values = torch.as_tensor(values, dtype=torch.float64)
    values = values[torch.isfinite(values)]
    return {
        "count": int(values.numel()),
        "p10": _quantile(values, 0.10),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
    }


def _mapping_camera_table(
    query_names: list[str], query_cache: dict
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    records = query_cache.get("queries", query_cache)
    missing = [name for name in query_names if name not in records]
    if missing:
        raise ValueError(f"Query cache lacks mapping camera {missing[0]}")
    poses = torch.stack(
        [torch.as_tensor(records[name]["pose_w2c"]).double() for name in query_names]
    )
    intrinsics = torch.stack(
        [torch.as_tensor(records[name]["native_K"]).double() for name in query_names]
    )
    image_hw = []
    for name in query_names:
        record = records[name]
        hw = record.get("native_input_hw")
        if hw is None:
            depth = record.get("native_depth")
            if depth is None or torch.as_tensor(depth).ndim < 2:
                raise ValueError(
                    f"Mapping camera {name} lacks native_input_hw/native_depth"
                )
            hw = torch.as_tensor(depth).shape[-2:]
        image_hw.append(torch.as_tensor(hw, dtype=torch.long))
    return poses, intrinsics, torch.stack(image_hw)


def _sample_triangulated_points(
    track_payload: dict, maximum_points: int
) -> tuple[torch.Tensor, torch.Tensor]:
    geometry = track_payload["track_geometry"]
    xyz = torch.as_tensor(geometry["triangulated_xyz"]).float()
    valid = torch.as_tensor(geometry["triangulated"]).bool()
    valid &= torch.isfinite(xyz).all(dim=1)
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    if indices.numel() > int(maximum_points):
        offsets = torch.div(
            torch.arange(int(maximum_points), dtype=torch.long)
            * int(indices.numel()),
            int(maximum_points),
            rounding_mode="floor",
        )
        indices = indices[offsets]
    return xyz[indices], indices


def _point_visibility(
    xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    camera_K: torch.Tensor,
    image_hw: torch.Tensor,
    *,
    camera_chunk: int = 128,
) -> torch.Tensor:
    """Project a bounded mapping-derived point sample into every camera."""
    xyz = torch.as_tensor(xyz, dtype=torch.float32)
    pose = torch.as_tensor(pose_w2c, dtype=torch.float32)
    K = torch.as_tensor(camera_K, dtype=torch.float32)
    hw = torch.as_tensor(image_hw, dtype=torch.long)
    visibility = torch.zeros(
        (int(pose.shape[0]), int(xyz.shape[0])), dtype=torch.bool
    )
    for start in range(0, int(pose.shape[0]), int(camera_chunk)):
        end = min(start + int(camera_chunk), int(pose.shape[0]))
        camera = torch.einsum(
            "qij,pj->qpi", pose[start:end, :3, :3], xyz
        ) + pose[start:end, None, :3, 3]
        depth = camera[..., 2]
        projected = torch.einsum("qij,qpj->qpi", K[start:end], camera)
        uv = projected[..., :2] / depth[..., None].clamp_min(1e-8)
        height = hw[start:end, 0, None]
        width = hw[start:end, 1, None]
        visibility[start:end] = (
            (depth > 1e-6)
            & (uv[..., 0] >= 0.0)
            & (uv[..., 0] < width)
            & (uv[..., 1] >= 0.0)
            & (uv[..., 1] < height)
        )
    return visibility


def _query_track_sets(
    track_payload: dict, query_count: int
) -> list[set[int]]:
    tracks = track_payload["tracks"]
    query = torch.as_tensor(tracks["query_index"]).long()
    track = torch.as_tensor(tracks["track_index"]).long()
    result = [set() for _ in range(int(query_count))]
    for query_index, track_index in zip(query.tolist(), track.tolist()):
        result[int(query_index)].add(int(track_index))
    return result


def _temporal_reference_pairs(
    names: list[str], centers: torch.Tensor, axes: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, str]:
    by_sequence: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for query, name in enumerate(names):
        identity = _temporal_identity(name)
        if identity is not None:
            by_sequence[identity[0]].append((identity[1], query))
    pairs = []
    for sequence in by_sequence.values():
        ordered = sorted(sequence)
        pairs.extend(
            (left[1], right[1]) for left, right in zip(ordered[:-1], ordered[1:])
        )
    if not pairs:
        return torch.zeros(0), torch.zeros(0), "candidate_distribution_fallback"
    pair = torch.as_tensor(pairs, dtype=torch.long)
    baseline = torch.linalg.norm(centers[pair[:, 0]] - centers[pair[:, 1]], dim=1)
    cosine = (axes[pair[:, 0]] * axes[pair[:, 1]]).sum(dim=1).clamp(-1.0, 1.0)
    axis_angle = torch.rad2deg(torch.acos(cosine))
    return baseline, axis_angle, "consecutive_mapping_frames"


@torch.no_grad()
def audit_track_pair_graph(
    track_payload: dict,
    query_cache: dict,
    *,
    pair_neighbors: int = 6,
    minimum_baseline_m: float = 0.03,
    maximum_baseline_m: float = 5.0,
    maximum_axis_angle_deg: float = 75.0,
    minimum_effective_parallax_deg: float = 1.0,
    temporal_adjacency_gap: int = 1,
    maximum_visibility_points: int = 4096,
) -> dict:
    """Audit a frozen mapping pair graph without changing its pair selection."""
    names = [str(value) for value in track_payload["query_names"]]
    pose, K, image_hw = _mapping_camera_table(names, query_cache)
    centers, axes = _camera_centers_and_axes(pose)
    pairs_list = candidate_camera_pairs(
        pose,
        neighbors=int(pair_neighbors),
        minimum_baseline_m=float(minimum_baseline_m),
        maximum_baseline_m=float(maximum_baseline_m),
        maximum_axis_angle_deg=float(maximum_axis_angle_deg),
    )
    pairs = torch.as_tensor(pairs_list, dtype=torch.long).reshape(-1, 2)
    left = pairs[:, 0]
    right = pairs[:, 1]
    baseline = torch.linalg.norm(centers[left] - centers[right], dim=1)
    axis_cosine = (axes[left] * axes[right]).sum(dim=1).clamp(-1.0, 1.0)
    axis_angle = torch.rad2deg(torch.acos(axis_cosine))

    temporal = [_temporal_identity(name) for name in names]
    temporal_gap = torch.full((len(pairs_list),), -1, dtype=torch.long)
    same_sequence = torch.zeros(len(pairs_list), dtype=torch.bool)
    for pair_index, (a, b) in enumerate(pairs_list):
        if temporal[a] is None or temporal[b] is None:
            continue
        if temporal[a][0] == temporal[b][0]:
            same_sequence[pair_index] = True
            temporal_gap[pair_index] = abs(temporal[a][1] - temporal[b][1])
    temporal_adjacent = same_sequence & (
        temporal_gap <= int(temporal_adjacency_gap)
    )

    points, sampled_track = _sample_triangulated_points(
        track_payload, int(maximum_visibility_points)
    )
    visibility = _point_visibility(points, pose, K, image_hw)
    overlap_intersection = torch.zeros(len(pairs_list), dtype=torch.long)
    overlap_union = torch.zeros(len(pairs_list), dtype=torch.long)
    overlap_jaccard = torch.zeros(len(pairs_list), dtype=torch.float64)
    parallax = torch.full((len(pairs_list),), float("nan"), dtype=torch.float64)
    for pair_index, (a, b) in enumerate(pairs_list):
        jointly_visible = visibility[a] & visibility[b]
        union_visible = visibility[a] | visibility[b]
        overlap_intersection[pair_index] = int(jointly_visible.sum())
        overlap_union[pair_index] = int(union_visible.sum())
        if int(union_visible.sum()) > 0:
            overlap_jaccard[pair_index] = float(
                jointly_visible.sum() / union_visible.sum()
            )
        if bool(jointly_visible.any()):
            joint_xyz = points[jointly_visible].double()
            ray_a = F.normalize(joint_xyz - centers[a], dim=1)
            ray_b = F.normalize(joint_xyz - centers[b], dim=1)
            cosine = (ray_a * ray_b).sum(dim=1).clamp(-1.0, 1.0)
            parallax[pair_index] = torch.rad2deg(torch.acos(cosine)).median()

    query_tracks = _query_track_sets(track_payload, len(names))
    shared_final_track_count = torch.as_tensor(
        [
            len(query_tracks[a].intersection(query_tracks[b]))
            for a, b in pairs_list
        ],
        dtype=torch.long,
    )
    final_track_supported = shared_final_track_count > 0

    adjacent_baseline, adjacent_axis, calibration_source = _temporal_reference_pairs(
        names, centers, axes
    )
    adjacent_short_threshold = _quantile(adjacent_baseline, 0.75)
    candidate_short_threshold = _quantile(baseline, 0.25)
    available_short_thresholds = [
        value
        for value in (adjacent_short_threshold, candidate_short_threshold)
        if value is not None
    ]
    short_threshold = (
        max(available_short_thresholds)
        if available_short_thresholds
        else float(minimum_baseline_m)
    )
    adjacent_low_axis_threshold = _quantile(adjacent_axis, 0.75)
    candidate_low_axis_threshold = _quantile(axis_angle, 0.25)
    available_axis_thresholds = [
        value
        for value in (adjacent_low_axis_threshold, candidate_low_axis_threshold)
        if value is not None
    ]
    low_axis_threshold = max(available_axis_thresholds) if available_axis_thresholds else 0.0
    adjacent_positive_overlap = overlap_jaccard[
        temporal_adjacent & (overlap_jaccard > 0)
    ]
    high_overlap_threshold = _quantile(adjacent_positive_overlap, 0.50)
    if high_overlap_threshold is None:
        high_overlap_threshold = _quantile(
            overlap_jaccard[overlap_jaccard > 0], 0.50
        ) or 0.0
    short_baseline = baseline <= float(short_threshold)
    near_repeat_proxy = (
        short_baseline
        & (axis_angle <= float(low_axis_threshold))
        & (
            temporal_adjacent
            | (
                (overlap_jaccard > 0)
                & (overlap_jaccard >= float(high_overlap_threshold))
            )
        )
    )
    effective_geometry_proxy = (
        (~short_baseline)
        & (parallax >= float(minimum_effective_parallax_deg))
        & (overlap_intersection > 0)
    )
    insufficient_parallax = (~torch.isfinite(parallax)) | (
        parallax < float(minimum_effective_parallax_deg)
    )
    high_overlap_low_parallax = (
        insufficient_parallax
        & (overlap_jaccard >= float(high_overlap_threshold))
        & (overlap_jaccard > 0)
    )

    def count(mask: torch.Tensor) -> int:
        return int(torch.as_tensor(mask).sum().item())

    pair_count = max(len(pairs_list), 1)
    supported_count = max(count(final_track_supported), 1)
    diagnostics = track_payload.get("diagnostics", {})
    frozen_candidate_count = diagnostics.get("track_camera_pair_candidate_count")
    report = {
        "mapping_camera_count": len(names),
        "candidate_pair_count": len(pairs_list),
        "frozen_payload_candidate_pair_count": (
            None if frozen_candidate_count is None else int(frozen_candidate_count)
        ),
        "candidate_graph_exact_count_reconstructed": (
            None
            if frozen_candidate_count is None
            else int(frozen_candidate_count) == len(pairs_list)
        ),
        "frozen_matched_pair_count": (
            None
            if "track_camera_pair_matched_count" not in diagnostics
            else int(diagnostics["track_camera_pair_matched_count"])
        ),
        "baseline_m": _distribution(baseline),
        "optical_axis_change_deg": _distribution(axis_angle),
        "mapping_point_parallax_deg": _distribution(parallax),
        "mapping_fov_overlap_jaccard": _distribution(overlap_jaccard),
        "shared_final_track_count": _distribution(shared_final_track_count),
        "same_sequence_temporal_gap": _distribution(
            temporal_gap[same_sequence & (temporal_gap >= 0)]
        ),
        "temporal_adjacent_pair_count": count(temporal_adjacent),
        "temporal_adjacent_pair_fraction": count(temporal_adjacent) / pair_count,
        "cross_sequence_pair_count": count(~same_sequence),
        "final_track_supported_pair_count": count(final_track_supported),
        "short_baseline_pair_count": count(short_baseline),
        "short_baseline_pair_fraction": count(short_baseline) / pair_count,
        "short_baseline_near_repeat_proxy_count": count(near_repeat_proxy),
        "short_baseline_near_repeat_proxy_fraction": (
            count(near_repeat_proxy) / pair_count
        ),
        "supported_short_baseline_near_repeat_proxy_count": count(
            near_repeat_proxy & final_track_supported
        ),
        "supported_short_baseline_near_repeat_proxy_fraction": (
            count(near_repeat_proxy & final_track_supported) / supported_count
        ),
        "effective_geometry_proxy_count": count(effective_geometry_proxy),
        "effective_geometry_proxy_fraction": (
            count(effective_geometry_proxy) / pair_count
        ),
        "insufficient_parallax_pair_count": count(insufficient_parallax),
        "insufficient_parallax_pair_fraction": (
            count(insufficient_parallax) / pair_count
        ),
        "nonshort_insufficient_parallax_pair_count": count(
            (~short_baseline) & insufficient_parallax
        ),
        "high_overlap_low_parallax_pair_count": count(
            high_overlap_low_parallax
        ),
        "high_overlap_low_parallax_pair_fraction": (
            count(high_overlap_low_parallax) / pair_count
        ),
        "supported_effective_geometry_proxy_count": count(
            effective_geometry_proxy & final_track_supported
        ),
        "nonadjacent_effective_geometry_proxy_count": count(
            effective_geometry_proxy & ~temporal_adjacent
        ),
        "calibration": {
            "source": calibration_source,
            "consecutive_mapping_pair_count": int(adjacent_baseline.numel()),
            "short_baseline_threshold_m": float(short_threshold),
            "consecutive_frame_baseline_p75_m": adjacent_short_threshold,
            "candidate_baseline_p25_m": candidate_short_threshold,
            "low_axis_change_threshold_deg": float(low_axis_threshold),
            "consecutive_frame_axis_change_p75_deg": (
                adjacent_low_axis_threshold
            ),
            "candidate_axis_change_p25_deg": candidate_low_axis_threshold,
            "high_overlap_threshold_jaccard": float(high_overlap_threshold),
            "minimum_effective_parallax_deg": float(
                minimum_effective_parallax_deg
            ),
        },
        "provenance_contract": {
            "candidate_pairs_reconstructed_from_frozen_policy": True,
            "fov_overlap_uses_mapping_intrinsics_and_poses": True,
            "parallax_uses_sampled_triangulated_mapping_points": True,
            "final_track_support_is_downstream_connectivity_proxy": True,
            "original_pair_match_identity_available": False,
            "per_pair_match_edge_provenance_available": False,
            "all_candidate_pairs_matched_inferred_from_aggregate_count": (
                diagnostics.get("track_camera_pair_matched_count")
                == len(pairs_list)
            ),
            "exact_short_baseline_duplicate_texture_edge_fraction": None,
            "blocker": (
                "The frozen payload keeps aggregate pair/match counts and final "
                "Track components, but not per-camera-pair raw, accepted, cycle, "
                "or rejected match provenance."
            ),
            "minimum_future_interface": {
                "pair_left_query": "int64[P]",
                "pair_right_query": "int64[P]",
                "raw_match_count": "int64[P]",
                "accepted_match_count": "int64[P]",
                "cycle_supported_match_count": "int64[P]",
                "rejected_ambiguity_count": "int64[P]",
                "rejected_epipolar_count": "int64[P]",
            },
        },
    }
    return {
        "schema": "lafgs.track_pair_graph_audit",
        "version": 1,
        "uses_test_queries": False,
        "audit_only": True,
        "pair_selection_mutated": False,
        "deployment_mutated": False,
        "policy": {
            "pair_neighbors": int(pair_neighbors),
            "minimum_baseline_m": float(minimum_baseline_m),
            "maximum_baseline_m": float(maximum_baseline_m),
            "maximum_axis_angle_deg": float(maximum_axis_angle_deg),
            "maximum_visibility_points": int(maximum_visibility_points),
            "visibility_sample_policy": "uniform_over_valid_track_index_order",
            "near_repeat_proxy": (
                "mapping-calibrated short baseline + low optical-axis change + "
                "temporal adjacency or high mapping-FoV overlap"
            ),
            "effective_geometry_proxy": (
                "non-short baseline + sufficient mapping-point parallax + "
                "positive mapping-FoV overlap"
            ),
        },
        "candidate_pairs": {
            "query_index": pairs,
            "baseline_m": baseline.float(),
            "optical_axis_change_deg": axis_angle.float(),
            "temporal_gap": temporal_gap,
            "same_sequence": same_sequence,
            "temporal_adjacent": temporal_adjacent,
            "mapping_fov_overlap_jaccard": overlap_jaccard.float(),
            "mapping_point_parallax_deg": parallax.float(),
            "shared_final_track_count": shared_final_track_count,
            "short_baseline": short_baseline,
            "near_repeat_proxy": near_repeat_proxy,
            "effective_geometry_proxy": effective_geometry_proxy,
            "insufficient_parallax": insufficient_parallax,
            "high_overlap_low_parallax": high_overlap_low_parallax,
        },
        "sampled_mapping_track_index": sampled_track,
        "report": report,
    }
