"""Experimental input adapter of anygsloc/mapping/projective_reconstruction.py; numerical kernels unchanged."""

from __future__ import annotations


import torch


import torch.nn.functional as F


from .contracts import ANCHOR_CANDIDATE_SCHEMA, ASSOCIATION_GRAPH_SCHEMA, source_contract, require_association


from anygsloc.mapping.observation_provider import ObservationProvider


from anygsloc.mapping.parallel_triangulation import (
    robust_triangulate_associations_fresh_cpu,
)


from anygsloc.mapping.tracks import fuse_projective_anchor_observations


from anygsloc.mapping.triangulation import robust_triangulate_associations


from anygsloc.features.raster_sampling import sample_raster_at_grid_uv


def _gather(
    values: list[torch.Tensor], query: torch.Tensor, row: torch.Tensor
) -> torch.Tensor:
    counts = torch.tensor([item.shape[0] for item in values], dtype=torch.long)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return torch.cat(values, dim=0)[offsets[query] + row]


def _robust_residual_scale(
    residual: torch.Tensor, *, floor: float
) -> tuple[torch.Tensor, float, float]:
    finite = residual[torch.isfinite(residual)]
    if finite.numel() == 0:
        return residual.new_tensor(float(floor)), float("nan"), float("nan")
    median = finite.median()
    mad = (finite - median).abs().median()
    scale = torch.maximum(
        median + 2.5 * 1.4826 * mad,
        median.new_tensor(float(floor)),
    )
    return scale, float(median), float(mad)


@torch.no_grad()
def joint_track_observation_weights(
    *,
    landmark_count: int,
    landmark_index: torch.Tensor,
    query_index: torch.Tensor,
    uv: torch.Tensor,
    descriptors: torch.Tensor,
    confidence: torch.Tensor,
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    initial_xyz: torch.Tensor,
    rendered_depth: torch.Tensor | None,
    query_bin: torch.Tensor | None = None,
    minimum_weight: float = 0.05,
    maximum_adjusted_fraction_per_track: float = 0.20,
) -> tuple[torch.Tensor, dict]:
    """Downweight mapping observations inconsistent with their Track.

    Initial XYZ and Gaussian depth are diagnostic evidence only. The returned
    weights feed a second fixed-camera ray triangulation; neither source is
    copied into the final XYZ.
    """

    landmark_count = int(landmark_count)
    landmark_index = torch.as_tensor(landmark_index, dtype=torch.long).cpu()
    query_index = torch.as_tensor(query_index, dtype=torch.long).cpu()
    uv = torch.as_tensor(uv, dtype=torch.float32).cpu()
    descriptors = F.normalize(torch.as_tensor(descriptors).float().cpu(), dim=1)
    confidence = torch.as_tensor(confidence, dtype=torch.float32).cpu()
    camera_K = torch.as_tensor(camera_K, dtype=torch.float32).cpu()
    pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float32).cpu()
    initial_xyz = torch.as_tensor(initial_xyz, dtype=torch.float32).cpu()
    count = int(landmark_index.numel())
    if not (
        query_index.numel() == count
        and uv.shape == (count, 2)
        and descriptors.ndim == 2
        and descriptors.shape[0] == count
        and confidence.numel() == count
    ):
        raise ValueError("joint Track purification observations do not align")
    if landmark_count < 1 or initial_xyz.shape != (landmark_count, 3):
        raise ValueError("initial Track geometry does not align")
    if bool((landmark_index < 0).any()) or int(landmark_index.max()) >= landmark_count:
        raise ValueError("joint Track purification landmark index is invalid")
    if not 0.0 < float(minimum_weight) <= 1.0:
        raise ValueError("minimum purification weight must lie inside (0,1]")
    if not 0.0 <= float(maximum_adjusted_fraction_per_track) <= 0.5:
        raise ValueError("maximum adjusted Track fraction must lie inside [0,0.5]")

    point = initial_xyz[landmark_index]
    pose = pose_w2c[query_index]
    intrinsic = camera_K[query_index]
    camera = torch.einsum("nij,nj->ni", pose[:, :3, :3], point) + pose[:, :3, 3]
    homogeneous = torch.einsum("nij,nj->ni", intrinsic, camera)
    positive = camera[:, 2] > 1e-6
    projected = homogeneous[:, :2] / homogeneous[:, 2:3].clamp_min(1e-6)
    projection_residual = torch.linalg.norm(projected - uv, dim=1)

    prototype_sum = torch.zeros(
        (landmark_count, descriptors.shape[1]), dtype=torch.float32
    )
    prototype_weight = torch.zeros(landmark_count, dtype=torch.float32)
    base = confidence.clamp_min(1e-4)
    prototype_sum.index_add_(0, landmark_index, descriptors * base[:, None])
    prototype_weight.index_add_(0, landmark_index, base)
    # Each row must be judged by evidence from the other observations.  If the
    # inspected row contributes to its own Track prototype, an outlier can pull
    # the reference towards itself, especially in short weak-prior Tracks.
    leave_one_out_sum = prototype_sum[landmark_index] - descriptors * base[:, None]
    leave_one_out_weight = prototype_weight[landmark_index] - base
    prototype = F.normalize(
        leave_one_out_sum / leave_one_out_weight[:, None].clamp_min(1e-8), dim=1
    )
    descriptor_residual = 1.0 - (descriptors * prototype).sum(dim=1).clamp(-1.0, 1.0)

    valid_geometry = positive & torch.isfinite(point).all(dim=1)
    projection_scale, projection_median, projection_mad = _robust_residual_scale(
        projection_residual[valid_geometry], floor=0.5
    )
    descriptor_scale, descriptor_median, descriptor_mad = _robust_residual_scale(
        descriptor_residual[valid_geometry], floor=0.02
    )
    projection_factor = 1.0 / (
        1.0 + (projection_residual / projection_scale.clamp_min(1e-8)).square()
    )
    descriptor_factor = 1.0 / (
        1.0 + (descriptor_residual / descriptor_scale.clamp_min(1e-8)).square()
    )

    depth_factor = torch.ones(count, dtype=torch.float32)
    depth_outlier = torch.zeros(count, dtype=torch.bool)
    depth_median = float("nan")
    depth_mad = float("nan")
    depth_scale_value = float("nan")
    depth_valid_count = 0
    if rendered_depth is not None:
        rendered_depth = torch.as_tensor(rendered_depth).float().cpu().reshape(-1)
        if rendered_depth.numel() != count:
            raise ValueError("rendered depth does not align with Track observations")
        depth_valid = (
            valid_geometry & torch.isfinite(rendered_depth) & (rendered_depth > 1e-6)
        )
        relative = torch.zeros(count, dtype=torch.float32)
        relative[depth_valid] = (
            camera[depth_valid, 2] - rendered_depth[depth_valid]
        ) / rendered_depth[depth_valid].clamp_min(1e-6)
        track_sum = torch.zeros(landmark_count)
        track_count = torch.zeros(landmark_count)
        track_sum.index_add_(0, landmark_index[depth_valid], relative[depth_valid])
        track_count.index_add_(
            0,
            landmark_index[depth_valid],
            torch.ones(int(depth_valid.sum()), dtype=torch.float32),
        )
        leave_one_out_count = track_count[landmark_index[depth_valid]] - 1.0
        leave_one_out_mean = (
            track_sum[landmark_index[depth_valid]] - relative[depth_valid]
        ) / leave_one_out_count.clamp_min(1.0)
        depth_disagreement = torch.zeros(count)
        depth_disagreement[depth_valid] = (
            relative[depth_valid] - leave_one_out_mean
        ).abs()
        depth_scale, depth_median, depth_mad = _robust_residual_scale(
            depth_disagreement[depth_valid], floor=0.02
        )
        depth_scale_value = float(depth_scale)
        depth_factor[depth_valid] = 1.0 / (
            1.0
            + (depth_disagreement[depth_valid] / depth_scale.clamp_min(1e-8)).square()
        )
        depth_outlier[depth_valid] = depth_disagreement[depth_valid] > depth_scale
        depth_valid_count = int(depth_valid.sum())

    raw_joint = (
        (projection_factor * descriptor_factor * depth_factor)
        .clamp_min(0.0)
        .pow(1.0 / 3.0)
    )
    vote_count = (
        (projection_residual > projection_scale).long()
        + (descriptor_residual > descriptor_scale).long()
        + depth_outlier.long()
    )
    candidate = valid_geometry & (vote_count >= 2)
    singleton_bin_protected_count = 0
    if query_bin is not None:
        query_bin = torch.as_tensor(query_bin, dtype=torch.long).cpu().reshape(-1)
        if query_bin.numel() != camera_K.shape[0] or bool((query_bin < 0).any()):
            raise ValueError("query-bin registry does not align")
        observation_bin = query_bin[query_index]
        bin_count = int(query_bin.max()) + 1
        encoded = landmark_index * bin_count + observation_bin
        support = torch.bincount(encoded, minlength=landmark_count * bin_count)
        singleton = support[encoded] <= 1
        singleton_bin_protected_count = int((candidate & singleton).sum())
        candidate &= ~singleton

    joint = torch.ones(count, dtype=torch.float32)
    selected_count = 0
    adjusted_track_count = 0
    candidate_rows = torch.nonzero(candidate, as_tuple=False).reshape(-1)
    track_observation_count = torch.bincount(landmark_index, minlength=landmark_count)
    if candidate_rows.numel():
        candidate_tracks = landmark_index[candidate_rows]
        order = torch.argsort(candidate_tracks, stable=True)
        candidate_rows = candidate_rows[order]
        candidate_tracks = candidate_tracks[order]
        tracks, counts = torch.unique_consecutive(candidate_tracks, return_counts=True)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        for local, track in enumerate(tracks.tolist()):
            maximum = int(
                int(track_observation_count[track])
                * float(maximum_adjusted_fraction_per_track)
            )
            if maximum <= 0:
                continue
            rows = candidate_rows[offsets[local] : offsets[local + 1]]
            rows = rows[torch.argsort(raw_joint[rows], descending=False, stable=True)][
                :maximum
            ]
            if rows.numel() == 0:
                continue
            joint[rows] = raw_joint[rows].clamp(float(minimum_weight), 1.0)
            selected_count += int(rows.numel())
            adjusted_track_count += 1
    purified = confidence * joint
    finite_joint = joint[valid_geometry]
    report = {
        "schema": "anygsloc_weak_prior_track_purification",
        "version": 3,
        "observation_count": count,
        "geometry_valid_observation_count": int(valid_geometry.sum()),
        "rendered_depth_valid_observation_count": depth_valid_count,
        "projection_residual_median_px": projection_median,
        "projection_residual_mad_px": projection_mad,
        "projection_scale_px": float(projection_scale),
        "descriptor_residual_median": descriptor_median,
        "descriptor_residual_mad": descriptor_mad,
        "descriptor_scale": float(descriptor_scale),
        "relative_depth_disagreement_median": depth_median,
        "relative_depth_disagreement_mad": depth_mad,
        "relative_depth_disagreement_scale": depth_scale_value,
        "joint_weight_mean": (
            float(finite_joint.mean()) if finite_joint.numel() else float("nan")
        ),
        "joint_weight_p10": (
            float(torch.quantile(finite_joint, 0.1))
            if finite_joint.numel()
            else float("nan")
        ),
        "minimum_weight": float(minimum_weight),
        "two_of_three_outlier_candidate_count": int(candidate.sum()),
        "adjusted_observation_count": selected_count,
        "adjusted_observation_fraction": selected_count / max(count, 1),
        "adjusted_track_count": adjusted_track_count,
        "singleton_pose_bin_protected_count": singleton_bin_protected_count,
        "maximum_adjusted_fraction_per_track": float(
            maximum_adjusted_fraction_per_track
        ),
        "descriptor_reference": ("leave_one_observation_out_weighted_track_consensus"),
        "relative_depth_reference": "leave_one_observation_out_track_mean_bias",
        "gaussian_depth_role": "observation_consistency_only",
        "final_xyz_source": "second_fixed_camera_robust_ray_triangulation",
    }
    return purified, report


@torch.no_grad()
def reconstruct_projective_anchors(
    observations: ObservationProvider,
    association: dict,
    *,
    maximum_observations: int = 32,
    minimum_views: int = 3,
    minimum_view_bins: int = 1,
    huber_delta_px: float = 2.0,
    triangulation_iterations: int = 3,
    minimum_parallax_deg: float = 1.0,
    parallax_quantile: float = 0.75,
    maximum_reprojection_px: float = 2.0,
    maximum_condition_number: float = 1e6,
    descriptor_trim_fraction: float = 0.2,
    robust_reweighting: bool = True,
    parallel_workers: int = 2,
    parallel_minimum_tracks: int = 5000,
    joint_track_purification: bool = False,
    purification_minimum_weight: float = 0.05,
    purification_maximum_adjusted_fraction_per_track: float = 0.20,
) -> dict:
    if association.get("schema") != ASSOCIATION_GRAPH_SCHEMA:
        raise ValueError("V6 reconstruction requires projective association v2")
    require_association(association, observations)
    if list(observations.names) != list(association["query_names"]):
        raise ValueError("association and observation registries differ")
    tracks = association["tracks"]
    track_index = torch.as_tensor(tracks["track_index"]).long()
    query_index = torch.as_tensor(tracks["query_index"]).long()
    keypoint_index = torch.as_tensor(tracks["keypoint_index"]).long()
    confidence = torch.as_tensor(tracks["confidence"]).float()
    track_count = int(association["diagnostics"]["track_count"])
    views = [observations.build_view(index) for index in range(len(observations))]
    keypoints = [view.physical_keypoints for view in views]
    descriptors = [view.descriptors.float() for view in views]
    detector_scores = [view.detector_scores.float() for view in views]
    uv = _gather(keypoints, query_index, keypoint_index)
    camera_K = torch.stack([view.intrinsics.float() for view in views])
    pose_w2c = torch.stack([view.pose_w2c.float() for view in views])
    triangulator = (
        robust_triangulate_associations_fresh_cpu
        if track_count >= int(parallel_minimum_tracks) and int(parallel_workers) > 1
        else robust_triangulate_associations
    )
    triangulation_arguments = dict(
        landmark_count=track_count,
        landmark_index=track_index,
        query_index=query_index,
        uv=uv,
        confidence=confidence,
        camera_K=camera_K,
        pose_w2c=pose_w2c,
        query_bin=torch.as_tensor(association["query_bins"]).long(),
        rendered_depth=None,
        maximum_observations_per_landmark=int(maximum_observations),
        minimum_views=int(minimum_views),
        minimum_view_bins=int(minimum_view_bins),
        huber_delta_px=float(huber_delta_px),
        iterations=int(triangulation_iterations),
        robust_reweighting=bool(robust_reweighting),
        minimum_parallax_deg=float(minimum_parallax_deg),
        parallax_quantile=float(parallax_quantile),
        maximum_reprojection_px=float(maximum_reprojection_px),
        maximum_condition_number=float(maximum_condition_number),
        maximum_covariance_trace_m2=float("inf"),
        maximum_rendered_depth_residual_m=float("inf"),
        minimum_rendered_depth_observations=0,
        surface_support_enabled=False,
    )
    if triangulator is robust_triangulate_associations_fresh_cpu:
        geometry = triangulator(
            worker_count=int(parallel_workers), **triangulation_arguments
        )
    else:
        geometry = triangulator(**triangulation_arguments)
    purification_report = None
    if bool(joint_track_purification):
        packed_descriptors = _gather(descriptors, query_index, keypoint_index)
        sampled_depth = None
        if all(view.depth is not None for view in views):
            per_view_depth = [
                sample_raster_at_grid_uv(view.depth, view.keypoints)[:, None]
                for view in views
            ]
            sampled_depth = _gather(
                per_view_depth, query_index, keypoint_index
            ).reshape(-1)
        purified_confidence, purification_report = joint_track_observation_weights(
            landmark_count=track_count,
            landmark_index=track_index,
            query_index=query_index,
            uv=uv,
            descriptors=packed_descriptors,
            confidence=confidence,
            camera_K=camera_K,
            pose_w2c=pose_w2c,
            initial_xyz=geometry["triangulated_xyz"],
            rendered_depth=sampled_depth,
            query_bin=torch.as_tensor(association["query_bins"]).long(),
            minimum_weight=float(purification_minimum_weight),
            maximum_adjusted_fraction_per_track=float(
                purification_maximum_adjusted_fraction_per_track
            ),
        )
        triangulation_arguments["confidence"] = purified_confidence
        if triangulator is robust_triangulate_associations_fresh_cpu:
            geometry = triangulator(
                worker_count=int(parallel_workers), **triangulation_arguments
            )
        else:
            geometry = triangulator(**triangulation_arguments)
        confidence = purified_confidence
    eligible = torch.as_tensor(geometry["triangulated"]).bool()
    selected = torch.nonzero(eligible, as_tuple=False).reshape(-1)
    if selected.numel() == 0:
        raise ValueError("association graph contains no ray-triangulated Anchor")
    selected_lookup = torch.full((track_count,), -1, dtype=torch.long)
    selected_lookup[selected] = torch.arange(selected.numel())
    selected_observation = selected_lookup[track_index] >= 0
    selected_track = selected_lookup[track_index[selected_observation]]
    selected_query = query_index[selected_observation]
    selected_keypoint = keypoint_index[selected_observation]
    selected_confidence = confidence[selected_observation]
    selected_descriptors = _gather(descriptors, selected_query, selected_keypoint)
    selected_detector = _gather(
        [value[:, None] for value in detector_scores], selected_query, selected_keypoint
    ).reshape(-1)
    features = []
    offsets = [0]
    csr_query = []
    csr_keypoint = []
    ordered_rows = torch.argsort(selected_track, stable=True)
    selected_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.bincount(selected_track, minlength=int(selected.numel())).cumsum(0),
        )
    )
    for anchor in range(int(selected.numel())):
        rows = ordered_rows[selected_offsets[anchor] : selected_offsets[anchor + 1]]
        features.append(
            fuse_projective_anchor_observations(
                selected_descriptors[rows],
                torch.as_tensor(association["query_bins"])[selected_query[rows]],
                detector_weight=selected_detector[rows] * selected_confidence[rows],
                trim_fraction=float(descriptor_trim_fraction),
            )
        )
        csr_query.append(selected_query[rows])
        csr_keypoint.append(selected_keypoint[rows])
        offsets.append(offsets[-1] + int(rows.numel()))
    geometry_reliability = torch.exp(
        -torch.as_tensor(geometry["triangulation_reprojection_median_px"])[selected]
        / max(float(maximum_reprojection_px), 1e-6)
    ) * (
        torch.as_tensor(geometry["triangulation_parallax_deg"])[selected]
        / max(float(minimum_parallax_deg), 1e-6)
    ).clamp(max=1.0)
    identity_reliability = torch.as_tensor(
        association["component_statistics"]["identity_reliability"]
    )[selected]
    count = int(selected.numel())
    return {
        "schema": ANCHOR_CANDIDATE_SCHEMA,
        "version": 2,
        **source_contract(observations),
        "query_names": list(association["query_names"]),
        "query_bins": torch.as_tensor(association["query_bins"]).long(),
        "anchor_ids": torch.arange(count, dtype=torch.long),
        "source_component_ids": selected,
        "anchor_xyz": torch.as_tensor(geometry["triangulated_xyz"])[selected].float(),
        "anchor_features": torch.stack(features).float(),
        "anchor_position_covariance": torch.as_tensor(
            geometry["triangulation_covariance_matrix"]
        )[selected].float(),
        "identity_reliability": identity_reliability.float(),
        "geometry_reliability": geometry_reliability.float(),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor(offsets, dtype=torch.long),
            "query_indices": torch.cat(csr_query),
            "keypoint_indices": torch.cat(csr_keypoint),
        },
        "geometry_diagnostics": {
            key: torch.as_tensor(value)[selected].clone()
            for key, value in geometry.items()
            if isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and value.shape[0] == track_count
        },
        "joint_track_purification": purification_report,
        "contract": {
            "final_xyz_source": "fixed_camera_robust_ray_triangulation",
            "gaussian_depth_used_for_final_xyz": False,
            "gaussian_depth_used_for_observation_consistency": bool(
                joint_track_purification
                and purification_report is not None
                and purification_report["rendered_depth_valid_observation_count"] > 0
            ),
            "joint_track_purification": bool(joint_track_purification),
            "gaussian_primitive_center_used": False,
            "one_observation_per_camera": True,
            "continuous_identity_and_geometry_reliability": True,
            "global_pose_bin_is_not_a_base_geometry_hard_gate": (
                int(minimum_view_bins) == 1
            ),
            "triangulation_execution": (
                "fresh_cpu_fixed_contiguous_shards"
                if triangulator is robust_triangulate_associations_fresh_cpu
                else "serial_cpu"
            ),
        },
    }
