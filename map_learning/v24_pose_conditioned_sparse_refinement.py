"""Pose-conditioned sparse correspondence refinement for online localization.

The module is deliberately map read-only.  It preserves every first-pass PnP
inlier and only proposes an alternative Anchor for first-pass outliers.  Unlike
the V21 minimum-reprojection arm, descriptor evidence and geometric evidence
are ranked jointly and a newly selected Anchor can be owned by at most one
query keypoint.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from localization.matcher import (
    TopKMatches,
    global_cosine_topk,
    maximum_weight_anchor_assignment,
)


def spatial_jackknife_pose_stability(
    *,
    keypoints: np.ndarray,
    points_3d: np.ndarray,
    pose_w2c: np.ndarray,
    inlier_rows: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
    reprojection_error_px: float,
    grid_shape: tuple[int, int] = (4, 4),
    minimum_remaining_inliers: int = 16,
    camera: dict | None = None,
) -> dict:
    """Measure pose sensitivity to deleting one occupied image region.

    Each leave-one-cell-out pose is a local refinement initialized at the
    evaluated pose; no extra RANSAC hypothesis is generated.
    """

    from localization.pose_solver import refine_absolute_pose_from_initial

    xy = np.asarray(keypoints, dtype=np.float64)
    xyz = np.asarray(points_3d, dtype=np.float64)
    pose = np.asarray(pose_w2c, dtype=np.float64)
    rows = np.asarray(inlier_rows, dtype=np.int64).reshape(-1)
    height, width = int(image_hw[0]), int(image_hw[1])
    grid_y, grid_x = int(grid_shape[0]), int(grid_shape[1])
    if not (
        xy.ndim == 2
        and xy.shape[1] == 2
        and xyz.shape == (xy.shape[0], 3)
        and pose.shape == (4, 4)
        and np.asarray(intrinsic).shape == (3, 3)
        and rows.size >= int(minimum_remaining_inliers) + 1
        and np.unique(rows).size == rows.size
        and int(rows.min()) >= 0
        and int(rows.max()) < xy.shape[0]
        and height > 0
        and width > 0
        and grid_y > 1
        and grid_x > 1
        and int(minimum_remaining_inliers) >= 4
    ):
        raise ValueError("spatial jackknife pose inputs are invalid")
    inlier_xy = xy[rows]
    cell_x = np.clip((inlier_xy[:, 0] * grid_x / width).astype(np.int64), 0, grid_x - 1)
    cell_y = np.clip(
        (inlier_xy[:, 1] * grid_y / height).astype(np.int64), 0, grid_y - 1
    )
    cells = cell_y * grid_x + cell_x
    translation = []
    rotation = []
    removed_counts = []
    for cell in np.unique(cells):
        keep = cells != cell
        if int(np.count_nonzero(keep)) < int(minimum_remaining_inliers):
            continue
        refined = refine_absolute_pose_from_initial(
            xy,
            xyz,
            intrinsic,
            pose,
            rows[keep],
            reprojection_error_px=float(reprojection_error_px),
            camera=camera,
        )
        relative = refined.pose_w2c[:3, :3] @ pose[:3, :3].T
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        update_re = float(np.degrees(np.arccos(cosine)))
        original_center = np.linalg.inv(pose)[:3, 3]
        refined_center = np.linalg.inv(refined.pose_w2c)[:3, 3]
        update_te = float(np.linalg.norm(refined_center - original_center) * 100.0)
        translation.append(update_te)
        rotation.append(update_re)
        removed_counts.append(int(np.count_nonzero(~keep)))
    if len(translation) < 2:
        raise ValueError("spatial jackknife needs at least two valid occupied cells")
    translation_array = np.asarray(translation)
    rotation_array = np.asarray(rotation)
    normalized = np.sqrt((translation_array / 5.0) ** 2 + (rotation_array / 5.0) ** 2)
    return {
        "valid_group_count": len(translation),
        "removed_inlier_count_maximum": max(removed_counts),
        "translation_cm_p90": float(np.percentile(translation_array, 90)),
        "rotation_deg_p90": float(np.percentile(rotation_array, 90)),
        "normalized_task_update_p90": float(np.percentile(normalized, 90)),
    }


def default_config() -> dict:
    """Return the frozen correspondence-selection contract for the V24 arm."""

    return {
        "topk": 64,
        "projection_gate_px": 8.0,
        "maximum_score_drop_from_top1": 0.10,
        "reliability_adaptive_score_drop": False,
        "reliability_expanded_score_drop": 0.10,
        "reliability_minimum_matchability_quantile": 0.50,
        "reliability_maximum_uncertainty_quantile": 0.50,
        "reliability_maximum_geometry_cost": 0.50,
        "reliability_minimum_reprojection_improvement_px": 4.0,
        "reliability_expansion_requires_no_base_geometric_candidate": True,
        "minimum_reprojection_improvement_px": 1.0,
        "descriptor_cost_weight": 0.50,
        "geometry_cost_weight": 0.40,
        "mapping_reliability_cost_weight": 0.10,
        "view_direction_slack_deg": 20.0,
        "minimum_mapping_distance_ratio": 0.25,
        "maximum_mapping_distance_ratio": 4.0,
        "maximum_changed_rows": 256,
        "maximum_changed_to_baseline_inlier_ratio": 1.0,
        "allow_soft_inliers": False,
        "preserve_all_first_pass_inliers": True,
        "soft_inlier_minimum_baseline_residual_px": 6.0,
        "soft_inlier_maximum_score_drop": 0.02,
        "soft_inlier_minimum_reprojection_improvement_px": 2.0,
        "maximum_soft_inlier_changes": 16,
        "reserve_first_pass_inlier_anchors": True,
        "unique_new_anchor_owner": True,
        "pose_conditioned_mutual_matching": False,
        "set_level_reserve_selection": False,
        "set_level_diversity_weight": 0.15,
        "heldout_candidate_validation": False,
        "heldout_validation_grid_shape": [8, 8],
        "heldout_validation_modulus": 5,
        "heldout_validation_minimum_rows": 16,
        "uncertainty_aware_projection": False,
        "maximum_uncertainty_projection_gate_px": 12.0,
        "uncertainty_sigma_multiplier": 3.0,
        "keypoint_variance_px2": 1.0,
        "candidate_selection": "minimum_joint_descriptor_geometry_cost",
        "ground_truth_used": False,
        "map_mutated": False,
    }


def validate_config(value: Mapping) -> dict:
    config = dict(value)
    expected = default_config()
    frozen_keys = (
        "topk",
        "minimum_reprojection_improvement_px",
        "descriptor_cost_weight",
        "geometry_cost_weight",
        "mapping_reliability_cost_weight",
        "minimum_mapping_distance_ratio",
        "maximum_mapping_distance_ratio",
        "reserve_first_pass_inlier_anchors",
        "unique_new_anchor_owner",
        "heldout_validation_grid_shape",
        "heldout_validation_modulus",
        "heldout_validation_minimum_rows",
        "uncertainty_sigma_multiplier",
        "keypoint_variance_px2",
        "reliability_expansion_requires_no_base_geometric_candidate",
        "candidate_selection",
        "ground_truth_used",
        "map_mutated",
    )
    if set(config) != set(expected) or any(
        config.get(key) != expected[key] for key in frozen_keys
    ):
        raise ValueError("V24 pose-conditioned refinement configuration differs")
    if not (
        4.0 <= float(config["projection_gate_px"]) <= 24.0
        and float(config["projection_gate_px"])
        <= float(config["maximum_uncertainty_projection_gate_px"])
        <= 24.0
        and 0.0 < float(config["maximum_score_drop_from_top1"]) <= 0.2
        and float(config["maximum_score_drop_from_top1"])
        <= float(config["reliability_expanded_score_drop"])
        <= 0.2
        and 0.0
        <= float(config["reliability_minimum_matchability_quantile"])
        <= 1.0
        and 0.0
        <= float(config["reliability_maximum_uncertainty_quantile"])
        <= 1.0
        and 0.0 < float(config["reliability_maximum_geometry_cost"]) <= 1.0
        and float(config["reliability_minimum_reprojection_improvement_px"])
        >= float(config["minimum_reprojection_improvement_px"])
        and 0.0 <= float(config["view_direction_slack_deg"]) <= 60.0
        and 1 <= int(config["maximum_changed_rows"]) <= 512
        and 0.0 < float(config["maximum_changed_to_baseline_inlier_ratio"]) <= 2.0
        and bool(config["preserve_all_first_pass_inliers"])
        == (not bool(config["allow_soft_inliers"]))
        and 0.0
        < float(config["soft_inlier_minimum_baseline_residual_px"])
        <= float(config["projection_gate_px"])
        and 0.0 < float(config["soft_inlier_maximum_score_drop"]) <= 0.05
        and float(config["soft_inlier_minimum_reprojection_improvement_px"])
        >= float(config["minimum_reprojection_improvement_px"])
        and 1 <= int(config["maximum_soft_inlier_changes"]) <= 64
        and 0.0 <= float(config["set_level_diversity_weight"]) <= 0.5
    ):
        raise ValueError("V24 tunable correspondence-selection gate is invalid")
    return config


def runtime_config(
    *,
    projection_gate_px: float = 8.0,
    maximum_score_drop_from_top1: float = 0.10,
    reliability_adaptive_score_drop: bool = False,
    reliability_expanded_score_drop: float = 0.10,
    reliability_minimum_matchability_quantile: float = 0.50,
    reliability_maximum_uncertainty_quantile: float = 0.50,
    reliability_maximum_geometry_cost: float = 0.50,
    reliability_minimum_reprojection_improvement_px: float = 4.0,
    view_direction_slack_deg: float = 20.0,
    maximum_changed_rows: int = 256,
    maximum_changed_to_baseline_inlier_ratio: float = 1.0,
    allow_soft_inliers: bool = False,
    soft_inlier_minimum_baseline_residual_px: float = 6.0,
    soft_inlier_maximum_score_drop: float = 0.02,
    soft_inlier_minimum_reprojection_improvement_px: float = 2.0,
    maximum_soft_inlier_changes: int = 16,
    pose_conditioned_mutual_matching: bool = False,
    set_level_reserve_selection: bool = False,
    set_level_diversity_weight: float = 0.15,
    heldout_candidate_validation: bool = False,
    uncertainty_aware_projection: bool = False,
    maximum_uncertainty_projection_gate_px: float = 12.0,
) -> dict:
    config = default_config()
    config.update(
        {
            "projection_gate_px": float(projection_gate_px),
            "maximum_score_drop_from_top1": float(maximum_score_drop_from_top1),
            "reliability_adaptive_score_drop": bool(
                reliability_adaptive_score_drop
            ),
            "reliability_expanded_score_drop": float(
                reliability_expanded_score_drop
            ),
            "reliability_minimum_matchability_quantile": float(
                reliability_minimum_matchability_quantile
            ),
            "reliability_maximum_uncertainty_quantile": float(
                reliability_maximum_uncertainty_quantile
            ),
            "reliability_maximum_geometry_cost": float(
                reliability_maximum_geometry_cost
            ),
            "reliability_minimum_reprojection_improvement_px": float(
                reliability_minimum_reprojection_improvement_px
            ),
            "view_direction_slack_deg": float(view_direction_slack_deg),
            "maximum_changed_rows": int(maximum_changed_rows),
            "maximum_changed_to_baseline_inlier_ratio": float(
                maximum_changed_to_baseline_inlier_ratio
            ),
            "allow_soft_inliers": bool(allow_soft_inliers),
            "preserve_all_first_pass_inliers": not bool(allow_soft_inliers),
            "soft_inlier_minimum_baseline_residual_px": float(
                soft_inlier_minimum_baseline_residual_px
            ),
            "soft_inlier_maximum_score_drop": float(soft_inlier_maximum_score_drop),
            "soft_inlier_minimum_reprojection_improvement_px": float(
                soft_inlier_minimum_reprojection_improvement_px
            ),
            "maximum_soft_inlier_changes": int(maximum_soft_inlier_changes),
            "pose_conditioned_mutual_matching": bool(pose_conditioned_mutual_matching),
            "set_level_reserve_selection": bool(set_level_reserve_selection),
            "set_level_diversity_weight": float(set_level_diversity_weight),
            "heldout_candidate_validation": bool(heldout_candidate_validation),
            "uncertainty_aware_projection": bool(uncertainty_aware_projection),
            "maximum_uncertainty_projection_gate_px": float(
                maximum_uncertainty_projection_gate_px
            ),
        }
    )
    return validate_config(config)


def _anchor_view_support_mask(
    *,
    anchor_xyz: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    anchor_view_support: Mapping,
    view_direction_slack_deg: float,
    minimum_distance_ratio: float,
    maximum_distance_ratio: float,
) -> torch.Tensor:
    """Return the mapping-supported Anchor mask for one query pose."""

    xyz = torch.as_tensor(anchor_xyz).float()
    pose = torch.as_tensor(baseline_pose_w2c, device=xyz.device).float()
    support = dict(anchor_view_support)
    modes = torch.as_tensor(support.get("direction_modes"), device=xyz.device).float()
    radii = torch.as_tensor(
        support.get("direction_radius_deg"), device=xyz.device
    ).float()
    mode_count = torch.as_tensor(support.get("mode_count"), device=xyz.device).long()
    minimum_distance = torch.as_tensor(
        support.get("minimum_distance_m"), device=xyz.device
    ).float()
    maximum_distance = torch.as_tensor(
        support.get("maximum_distance_m"), device=xyz.device
    ).float()
    count = xyz.shape[0]
    structurally_valid = bool(
        support.get("schema") == "lafgs_v24_anchor_view_support"
        and support.get("uses_test_queries") is False
        and modes.shape == (count, 2, 3)
        and radii.shape == (count, 2)
        and mode_count.shape
        == minimum_distance.shape
        == maximum_distance.shape
        == (count,)
    )
    values_valid = bool(support.get("runtime_validated", False)) or bool(
        ((mode_count == 1) | (mode_count == 2)).all()
        and torch.isfinite(modes).all()
        and torch.isfinite(radii).all()
        and torch.isfinite(minimum_distance).all()
        and torch.isfinite(maximum_distance).all()
        and (minimum_distance > 0).all()
        and (maximum_distance >= minimum_distance).all()
    )
    if not (
        structurally_valid
        and values_valid
        and pose.shape == (4, 4)
        and bool(torch.isfinite(pose).all())
        and 0.0 <= float(view_direction_slack_deg) <= 60.0
        and 0.0 < float(minimum_distance_ratio) <= 1.0
        and float(maximum_distance_ratio) >= 1.0
    ):
        raise ValueError("V24 Anchor view-support artifact is invalid")
    camera_center = -(pose[:3, :3].T @ pose[:3, 3])
    view_ray = camera_center[None, :] - xyz
    query_distance = view_ray.norm(dim=1)
    query_direction = view_ray / query_distance.clamp_min(1e-12)[:, None]
    cosine = torch.einsum("nd,nmd->nm", query_direction, modes)
    angular_distance = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    mode_rows = torch.arange(2, device=xyz.device)[None, :]
    valid_mode = mode_rows < mode_count[:, None]
    direction_supported = (
        valid_mode & (angular_distance <= radii + float(view_direction_slack_deg))
    ).any(dim=1)
    distance_supported = (
        query_distance >= minimum_distance * float(minimum_distance_ratio)
    ) & (query_distance <= maximum_distance * float(maximum_distance_ratio))
    return (
        direction_supported
        & distance_supported
        & torch.isfinite(query_distance)
        & (query_distance > 1e-8)
    )


@torch.inference_mode()
def build_pose_visible_topk(
    *,
    query_descriptors: torch.Tensor,
    query_keypoints: torch.Tensor | None = None,
    normalized_anchor_features: torch.Tensor,
    baseline_anchor_rows: torch.Tensor,
    baseline_scores: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    image_hw: tuple[int, int],
    topk: int = 64,
    image_margin_px: float = 64.0,
    retrieval_query_rows: torch.Tensor | None = None,
    anchor_view_support: Mapping | None = None,
    prefilter_mapping_view_support: bool = False,
    view_direction_slack_deg: float = 15.0,
    minimum_mapping_distance_ratio: float = 0.25,
    maximum_mapping_distance_ratio: float = 4.0,
    view_conditioned_descriptor_state: Mapping | None = None,
    view_conditioned_minimum_concentration: float = 0.0,
    view_conditioned_residual_scale: float = 1.0,
    view_conditioned_require_two_valid_modes: bool = False,
    view_conditioned_score_fusion: str = "replace",
    projection_first_local_candidates: bool = False,
    projection_first_radius_px: float = 12.0,
    projection_first_query_chunk_size: int = 128,
) -> dict:
    """Retrieve exact Top-K inside the first-pose sparse 3D view frustum.

    The global Top-1 remains column zero. No image is rendered: visibility is
    a point projection and descriptor retrieval is exact inside that pool.
    """

    query = torch.as_tensor(query_descriptors).float()
    features = torch.as_tensor(normalized_anchor_features).float().to(query.device)
    baseline = torch.as_tensor(baseline_anchor_rows).long().to(query.device)
    scores = torch.as_tensor(baseline_scores).float().to(query.device)
    xyz = torch.as_tensor(anchor_xyz).float().to(query.device)
    calibration = torch.as_tensor(intrinsic).float().to(query.device)
    pose = torch.as_tensor(baseline_pose_w2c).float().to(query.device)
    query_xy = (
        None
        if query_keypoints is None
        else torch.as_tensor(query_keypoints, device=query.device).float()
    )
    height, width = (int(image_hw[0]), int(image_hw[1]))
    requested = int(topk)
    margin = float(image_margin_px)
    retrieval = (
        torch.arange(query.shape[0], device=query.device)
        if retrieval_query_rows is None
        else torch.as_tensor(retrieval_query_rows, device=query.device)
        .long()
        .reshape(-1)
    )
    if not (
        query.ndim == features.ndim == 2
        and query.shape[1] == features.shape[1]
        and baseline.shape == scores.shape == (query.shape[0],)
        and xyz.shape == (features.shape[0], 3)
        and calibration.shape == (3, 3)
        and pose.shape == (4, 4)
        and height > 0
        and width > 0
        and requested >= 2
        and requested <= features.shape[0]
        and margin >= 0.0
        and bool(torch.isfinite(query).all())
        and bool(torch.isfinite(features).all())
        and bool(torch.isfinite(xyz).all())
        and bool(torch.isfinite(calibration).all())
        and bool(torch.isfinite(pose).all())
        and (not baseline.numel() or int(baseline.min()) >= 0)
        and (not baseline.numel() or int(baseline.max()) < features.shape[0])
        and (
            not retrieval.numel()
            or (int(retrieval.min()) >= 0 and int(retrieval.max()) < query.shape[0])
        )
        and retrieval.unique().numel() == retrieval.numel()
        and (
            not bool(prefilter_mapping_view_support) or anchor_view_support is not None
        )
        and (
            view_conditioned_descriptor_state is None or anchor_view_support is not None
        )
        and 0.0 <= float(view_conditioned_minimum_concentration) <= 1.0
        and 0.0 < float(view_conditioned_residual_scale) <= 1.0
        and view_conditioned_score_fusion in {"replace", "max_with_base"}
        and (
            not bool(projection_first_local_candidates)
            or (
                query_xy is not None
                and query_xy.shape == (query.shape[0], 2)
                and bool(torch.isfinite(query_xy).all())
                and 4.0 <= float(projection_first_radius_px) <= 32.0
                and int(projection_first_query_chunk_size) >= 1
            )
        )
    ):
        raise ValueError("V24 pose-visible Top-K inputs are invalid")

    camera = (pose[:3, :3] @ xyz.T).T + pose[:3, 3]
    homogeneous = (calibration @ camera.T).T
    depth = homogeneous[:, 2]
    projected = homogeneous[:, :2] / depth.clamp_min(1e-12)[:, None]
    visible = (
        (depth > 1e-8)
        & (projected[:, 0] >= -margin)
        & (projected[:, 0] < width + margin)
        & (projected[:, 1] >= -margin)
        & (projected[:, 1] < height + margin)
        & torch.isfinite(projected).all(dim=1)
    )
    visible_rows = torch.nonzero(visible, as_tuple=False).reshape(-1)
    view_supported_count = 0
    prefilter_fallback = False
    supported_visible_rows = visible_rows
    if bool(prefilter_mapping_view_support):
        support_mask = _anchor_view_support_mask(
            anchor_xyz=xyz,
            baseline_pose_w2c=pose,
            anchor_view_support=anchor_view_support,
            view_direction_slack_deg=float(view_direction_slack_deg),
            minimum_distance_ratio=float(minimum_mapping_distance_ratio),
            maximum_distance_ratio=float(maximum_mapping_distance_ratio),
        )
        supported_visible_rows = torch.nonzero(
            visible & support_mask, as_tuple=False
        ).reshape(-1)
        view_supported_count = int(supported_visible_rows.numel())
        prefilter_fallback = bool(supported_visible_rows.numel() < requested)
    fallback = bool(visible_rows.numel() < requested)
    pool_rows = (
        torch.arange(features.shape[0], device=query.device)
        if fallback
        else visible_rows
        if prefilter_fallback or not bool(prefilter_mapping_view_support)
        else supported_visible_rows
    )
    base_retrieval_features = features[pool_rows]
    retrieval_features = base_retrieval_features
    selected_mode_count = 0
    base_fallback_count = int(pool_rows.numel())
    if view_conditioned_descriptor_state is not None:
        from map_learning.v27_view_conditioned_anchor_descriptor import (
            select_view_conditioned_anchor_features,
        )

        retrieval_features, mode_report = select_view_conditioned_anchor_features(
            base_anchor_features=features,
            base_anchor_features_normalized=True,
            anchor_xyz=xyz,
            direction_modes=view_conditioned_descriptor_state.get(
                "direction_modes", anchor_view_support["direction_modes"]
            ),
            baseline_pose_w2c=pose,
            mode_features=view_conditioned_descriptor_state["mode_features"],
            mode_valid=view_conditioned_descriptor_state["mode_valid"],
            mode_authorized=view_conditioned_descriptor_state.get(
                "mode_authorized", view_conditioned_descriptor_state["mode_valid"]
            ),
            mode_concentration=view_conditioned_descriptor_state["mode_concentration"],
            direction_radius_deg=view_conditioned_descriptor_state.get(
                "direction_radius_deg", anchor_view_support["direction_radius_deg"]
            ),
            minimum_concentration=float(view_conditioned_minimum_concentration),
            anchor_rows=pool_rows,
            residual_scale=float(view_conditioned_residual_scale),
            require_two_valid_modes=bool(view_conditioned_require_two_valid_modes),
        )
        selected_mode_count = int(mode_report["selected_mode_anchor_count"])
        base_fallback_count = int(mode_report["base_fallback_anchor_count"])
    output_rows = baseline[:, None].expand(-1, requested).clone()
    output_scores = scores[:, None].expand(-1, requested).clone()
    projection_local_edge_count = 0
    projection_local_nonempty_query_count = 0
    if retrieval.numel():
        if bool(projection_first_local_candidates):
            # Geometry-first ablation: descriptors only rank Anchors whose T0
            # projections already fall inside the sparse keypoint neighborhood.
            # Chunking bounds the temporary [query, visible Anchor] matrices.
            output_scores[:, 1:] -= 1.0
            normalized_query = torch.nn.functional.normalize(query[retrieval], dim=1)
            projected_pool = projected[pool_rows]
            alternative_count = min(requested - 1, int(pool_rows.numel()))
            chunk_size = int(projection_first_query_chunk_size)
            for start in range(0, retrieval.numel(), chunk_size):
                query_rows = retrieval[start : start + chunk_size]
                descriptors = normalized_query[start : start + chunk_size]
                delta = query_xy[query_rows, None, :] - projected_pool[None, :, :]
                local_mask = (
                    delta.square().sum(dim=2) <= float(projection_first_radius_px) ** 2
                )
                local_mask &= pool_rows[None, :] != baseline[query_rows, None]
                projection_local_edge_count += int(local_mask.sum().item())
                projection_local_nonempty_query_count += int(
                    local_mask.any(dim=1).sum().item()
                )
                local_scores = descriptors @ retrieval_features.T
                if (
                    view_conditioned_descriptor_state is not None
                    and view_conditioned_score_fusion == "max_with_base"
                ):
                    local_scores = torch.maximum(
                        local_scores, descriptors @ base_retrieval_features.T
                    )
                local_scores.masked_fill_(~local_mask, -torch.inf)
                best_scores, best_columns = torch.topk(
                    local_scores, k=alternative_count, dim=1
                )
                finite = torch.isfinite(best_scores)
                destination_rows = output_rows[query_rows, 1 : 1 + alternative_count]
                destination_scores = output_scores[
                    query_rows, 1 : 1 + alternative_count
                ]
                mapped_rows = pool_rows[best_columns]
                destination_rows[finite] = mapped_rows[finite]
                destination_scores[finite] = best_scores[finite]
                output_rows[query_rows, 1 : 1 + alternative_count] = destination_rows
                output_scores[query_rows, 1 : 1 + alternative_count] = (
                    destination_scores
                )
        else:
            if (
                view_conditioned_descriptor_state is not None
                and view_conditioned_score_fusion == "max_with_base"
            ):
                normalized_query = torch.nn.functional.normalize(
                    query[retrieval], dim=1
                )
                local_scores = torch.maximum(
                    normalized_query @ base_retrieval_features.T,
                    normalized_query @ retrieval_features.T,
                )
                best_scores, best_rows = torch.topk(local_scores, k=requested, dim=1)
                local = TopKMatches(
                    keypoint_indices=retrieval,
                    anchor_indices=best_rows,
                    scores=best_scores,
                )
            else:
                local = global_cosine_topk(
                    query[retrieval],
                    retrieval_features,
                    topk=requested,
                    anchor_descriptors_normalized=True,
                )
            mapped_rows = pool_rows[local.anchor_indices]
            alternative_scores = local.scores.masked_fill(
                mapped_rows == baseline[retrieval, None], -torch.inf
            )
            alt_scores, alt_columns = torch.topk(
                alternative_scores, k=requested - 1, dim=1
            )
            alt_rows = mapped_rows.gather(1, alt_columns)
            if not bool(torch.isfinite(alt_scores).all()):
                raise RuntimeError(
                    "V24 pose-visible pool has too few alternative Anchors"
                )
            output_rows[retrieval, 1:] = alt_rows
            output_scores[retrieval, 1:] = alt_scores
    result = TopKMatches(
        keypoint_indices=torch.arange(query.shape[0], device=query.device),
        anchor_indices=output_rows,
        scores=output_scores,
    )
    return {
        "matches": result,
        "visible_anchor_count": int(visible_rows.numel()),
        "view_supported_anchor_count": int(view_supported_count),
        "candidate_pool_anchor_count": int(pool_rows.numel()),
        "global_fallback": fallback,
        "view_support_prefilter_fallback": prefilter_fallback,
        "retrieval_query_count": int(retrieval.numel()),
        "view_conditioned_selected_mode_anchor_count": selected_mode_count,
        "view_conditioned_base_fallback_anchor_count": base_fallback_count,
        "view_conditioned_score_fusion": view_conditioned_score_fusion,
        "projection_first_local_candidates": bool(projection_first_local_candidates),
        "projection_first_radius_px": float(projection_first_radius_px),
        "projection_local_edge_count": int(projection_local_edge_count),
        "projection_local_nonempty_query_count": int(
            projection_local_nonempty_query_count
        ),
    }


def _validated_inputs(
    *,
    keypoints: torch.Tensor,
    topk_anchor_rows: torch.Tensor,
    topk_scores: torch.Tensor,
    baseline_inlier_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    config: Mapping,
    map_geometry_validated: bool,
) -> tuple[
    dict,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    cfg = validate_config(config)
    candidates = torch.as_tensor(topk_anchor_rows).long()
    scores = torch.as_tensor(topk_scores).float().to(candidates.device)
    xy = torch.as_tensor(keypoints).float().to(candidates.device)
    inliers = torch.as_tensor(baseline_inlier_rows).long().to(candidates.device)
    xyz = torch.as_tensor(anchor_xyz).float().to(candidates.device)
    calibration = torch.as_tensor(intrinsic).float().to(candidates.device)
    pose = torch.as_tensor(baseline_pose_w2c).float().to(candidates.device)
    if candidates.ndim != 2:
        raise ValueError("V24 candidate rows must have shape [N, K]")
    count, topk = candidates.shape
    if not (
        topk == int(cfg["topk"])
        and scores.shape == candidates.shape
        and xy.shape == (count, 2)
        and xyz.ndim == 2
        and xyz.shape[1] == 3
        and calibration.shape == (3, 3)
        and pose.shape == (4, 4)
        and bool(torch.isfinite(scores).all())
        and bool(torch.isfinite(xy).all())
        and (bool(map_geometry_validated) or bool(torch.isfinite(xyz).all()))
        and bool(torch.isfinite(calibration).all())
        and bool(torch.isfinite(pose).all())
        and (not candidates.numel() or int(candidates.min()) >= 0)
        and (not candidates.numel() or int(candidates.max()) < xyz.shape[0])
        and (
            not inliers.numel()
            or (int(inliers.min()) >= 0 and int(inliers.max()) < count)
        )
        and inliers.unique().numel() == inliers.numel()
    ):
        raise ValueError("V24 pose-conditioned refinement inputs are invalid")
    return cfg, xy, candidates, scores, inliers, xyz, calibration, pose


def _keep_unique_lowest_cost_owner(
    *,
    selected_rows: torch.Tensor,
    selected_costs: torch.Tensor,
    changed: torch.Tensor,
    anchor_count: int,
) -> torch.Tensor:
    """Keep only the cheapest changed query for every newly selected Anchor."""

    device = selected_rows.device
    changed_rows = torch.nonzero(changed, as_tuple=False).reshape(-1)
    if not changed_rows.numel():
        return changed
    changed_anchors = selected_rows[changed_rows]
    minimum_cost = torch.full(
        (anchor_count,), torch.inf, dtype=selected_costs.dtype, device=device
    )
    minimum_cost.scatter_reduce_(
        0,
        changed_anchors,
        selected_costs[changed_rows],
        reduce="amin",
        include_self=True,
    )
    cheapest = selected_costs[changed_rows] == minimum_cost[changed_anchors]
    # Deterministically break exact float ties by query-row index.
    tied_rows = changed_rows[cheapest]
    tied_anchors = selected_rows[tied_rows]
    minimum_query = torch.full(
        (anchor_count,), selected_rows.numel(), dtype=torch.long, device=device
    )
    minimum_query.scatter_reduce_(
        0, tied_anchors, tied_rows, reduce="amin", include_self=True
    )
    keep = torch.zeros_like(changed)
    keep[tied_rows] = tied_rows == minimum_query[tied_anchors]
    return keep


def _camera_projection_jacobian(
    camera_points: torch.Tensor, intrinsic: torch.Tensor
) -> torch.Tensor:
    """Return d(pixel xy)/d(camera xyz) for a standard pinhole K."""

    points = torch.as_tensor(camera_points).float()
    calibration = torch.as_tensor(intrinsic, device=points.device).float()
    if calibration.shape != (3, 3) or not torch.allclose(
        calibration[2],
        calibration.new_tensor([0.0, 0.0, 1.0]),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("uncertainty projection requires a standard pinhole K")
    x, y, z = points.unbind(dim=-1)
    inverse_z = z.clamp_min(1e-12).reciprocal()
    inverse_z2 = inverse_z.square()
    fx = calibration[0, 0]
    skew = calibration[0, 1]
    fy = calibration[1, 1]
    output = points.new_zeros((*points.shape[:-1], 2, 3))
    output[..., 0, 0] = fx * inverse_z
    output[..., 0, 1] = skew * inverse_z
    output[..., 0, 2] = -(fx * x + skew * y) * inverse_z2
    output[..., 1, 1] = fy * inverse_z
    output[..., 1, 2] = -(fy * y) * inverse_z2
    return output


def _skew_symmetric(points: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(points).float()
    x, y, z = values.unbind(dim=-1)
    output = values.new_zeros((*values.shape[:-1], 3, 3))
    output[..., 0, 1] = -z
    output[..., 0, 2] = y
    output[..., 1, 0] = z
    output[..., 1, 2] = -x
    output[..., 2, 0] = -y
    output[..., 2, 1] = x
    return output


def _pose_covariance_from_first_pass_inliers(
    *,
    keypoints: torch.Tensor,
    baseline_camera_points: torch.Tensor,
    baseline_inlier_rows: torch.Tensor,
    intrinsic: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    rows = torch.as_tensor(
        baseline_inlier_rows, device=baseline_camera_points.device
    ).long()
    if rows.numel() < 6:
        raise ValueError("pose covariance needs at least six first-pass inliers")
    camera = baseline_camera_points[rows]
    projection_jacobian = _camera_projection_jacobian(camera, intrinsic)
    pose_jacobian = torch.cat(
        (
            projection_jacobian,
            -torch.einsum("nij,njk->nik", projection_jacobian, _skew_symmetric(camera)),
        ),
        dim=2,
    )
    # Recompute with K so the robust image noise scale is in pixels.
    homogeneous = torch.einsum("ij,nj->ni", intrinsic, camera)
    projected = homogeneous[:, :2] / homogeneous[:, 2:].clamp_min(1e-12)
    residual = (projected - keypoints[rows]).norm(dim=1)
    sigma2 = residual.square().median().clamp(1.0, 144.0)
    robust_weight = (4.0 / residual.clamp_min(1e-6)).clamp_max(1.0)
    information = torch.einsum(
        "nri,nrj,n->ij", pose_jacobian, pose_jacobian, robust_weight
    )
    eigenvalues = torch.linalg.eigvalsh(information.double()).float()
    maximum = eigenvalues[-1].clamp_min(1e-12)
    minimum = eigenvalues[0].clamp_min(maximum * 1e-9)
    condition = float((maximum / minimum).item())
    damping = information.diagonal().mean().clamp_min(1e-9) * 1e-6
    covariance = sigma2 * torch.linalg.pinv(
        information + torch.eye(6, device=information.device) * damping,
        hermitian=True,
    )
    return covariance, condition


def select_pose_conditioned_rows(
    *,
    keypoints: torch.Tensor,
    topk_anchor_rows: torch.Tensor,
    topk_scores: torch.Tensor,
    baseline_inlier_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    anchor_view_support: Mapping | None = None,
    anchor_matchability: torch.Tensor | None = None,
    anchor_uncertainty: torch.Tensor | None = None,
    anchor_position_covariance: torch.Tensor | None = None,
    mapping_reliability_validated: bool = False,
    map_geometry_validated: bool = False,
    image_hw: tuple[int, int] | None = None,
    config: Mapping | None = None,
) -> dict:
    """Select a sparse, unique set of pose-conditioned outlier replacements.

    All tensors may stay on CUDA.  Only the compact returned diagnostics need
    to cross to CPU before the second PoseLib solve.
    """

    (
        cfg,
        xy,
        candidates,
        scores,
        inliers,
        xyz,
        calibration,
        pose,
    ) = _validated_inputs(
        keypoints=keypoints,
        topk_anchor_rows=topk_anchor_rows,
        topk_scores=topk_scores,
        baseline_inlier_rows=baseline_inlier_rows,
        anchor_xyz=anchor_xyz,
        intrinsic=intrinsic,
        baseline_pose_w2c=baseline_pose_w2c,
        config=default_config() if config is None else config,
        map_geometry_validated=bool(map_geometry_validated),
    )
    count, topk = candidates.shape
    baseline = candidates[:, 0]
    base_scores = scores[:, 0]

    points = xyz[candidates.reshape(-1)].reshape(count, topk, 3)
    camera = torch.einsum("ij,nkj->nki", pose[:3, :3], points) + pose[:3, 3]
    homogeneous = torch.einsum("ij,nkj->nki", calibration, camera)
    depth = homogeneous[:, :, 2]
    projected = homogeneous[:, :, :2] / depth.clamp_min(1e-12).unsqueeze(2)
    residual = (projected - xy[:, None, :]).norm(dim=2)
    residual[depth <= 1e-12] = torch.inf

    edge_projection_gate = torch.full_like(residual, float(cfg["projection_gate_px"]))
    pose_information_condition = 0.0
    expanded_projection_edges = 0
    if bool(cfg["uncertainty_aware_projection"]):
        if anchor_position_covariance is None:
            raise ValueError("uncertainty-aware projection requires Anchor covariance")
        position_covariance = torch.as_tensor(
            anchor_position_covariance, device=candidates.device
        ).float()
        if position_covariance.shape != (xyz.shape[0], 3, 3):
            raise ValueError("Anchor position covariance does not align with the map")
        if not map_geometry_validated and not bool(
            torch.isfinite(position_covariance).all()
        ):
            raise ValueError("Anchor position covariance is non-finite")
        pose_covariance, pose_information_condition = (
            _pose_covariance_from_first_pass_inliers(
                keypoints=xy,
                baseline_camera_points=camera[:, 0],
                baseline_inlier_rows=inliers,
                intrinsic=calibration,
            )
        )
        projection_jacobian = _camera_projection_jacobian(camera, calibration)
        pose_jacobian = torch.cat(
            (
                projection_jacobian,
                -torch.einsum(
                    "nkij,nkjl->nkil",
                    projection_jacobian,
                    _skew_symmetric(camera),
                ),
            ),
            dim=3,
        )
        pose_pixel_covariance = torch.einsum(
            "nkri,ij,nksj->nkrs",
            pose_jacobian,
            pose_covariance,
            pose_jacobian,
        )
        world_projection_jacobian = torch.einsum(
            "nkij,jl->nkil", projection_jacobian, pose[:3, :3]
        )
        candidate_covariance = position_covariance[candidates]
        anchor_pixel_covariance = torch.einsum(
            "nkri,nkij,nksj->nkrs",
            world_projection_jacobian,
            candidate_covariance,
            world_projection_jacobian,
        )
        pose_variance = (
            torch.diagonal(pose_pixel_covariance, dim1=2, dim2=3)
            .sum(dim=2)
            .mul(0.5)
            .clamp_min(0.0)
        )
        anchor_variance = (
            torch.diagonal(anchor_pixel_covariance, dim1=2, dim2=3)
            .sum(dim=2)
            .mul(0.5)
            .clamp_min(0.0)
        )
        projected_sigma = (
            float(cfg["keypoint_variance_px2"]) + pose_variance + anchor_variance
        ).sqrt()
        edge_projection_gate = (
            float(cfg["uncertainty_sigma_multiplier"]) * projected_sigma
        ).clamp(
            min=float(cfg["projection_gate_px"]),
            max=float(cfg["maximum_uncertainty_projection_gate_px"]),
        )
        edge_projection_gate[depth <= 1e-12] = float(cfg["projection_gate_px"])
        expanded_projection_edges = int(
            (edge_projection_gate > float(cfg["projection_gate_px"]) + 1e-6)
            .sum()
            .item()
        )

    view_supported = torch.ones_like(residual, dtype=torch.bool)
    view_support_available = anchor_view_support is not None
    if anchor_view_support is not None:
        support = dict(anchor_view_support)
        modes = torch.as_tensor(
            support.get("direction_modes"), device=candidates.device
        ).float()
        radii = torch.as_tensor(
            support.get("direction_radius_deg"), device=candidates.device
        ).float()
        mode_count = torch.as_tensor(
            support.get("mode_count"), device=candidates.device
        ).long()
        minimum_distance = torch.as_tensor(
            support.get("minimum_distance_m"), device=candidates.device
        ).float()
        maximum_distance = torch.as_tensor(
            support.get("maximum_distance_m"), device=candidates.device
        ).float()
        structurally_valid = bool(
            support.get("schema") == "lafgs_v24_anchor_view_support"
            and support.get("uses_test_queries") is False
            and modes.shape == (xyz.shape[0], 2, 3)
            and radii.shape == (xyz.shape[0], 2)
            and mode_count.shape
            == minimum_distance.shape
            == maximum_distance.shape
            == (xyz.shape[0],)
        )
        values_valid = bool(support.get("runtime_validated", False)) or bool(
            ((mode_count == 1) | (mode_count == 2)).all()
            and torch.isfinite(modes).all()
            and torch.isfinite(minimum_distance).all()
            and torch.isfinite(maximum_distance).all()
            and (minimum_distance > 0).all()
            and (maximum_distance >= minimum_distance).all()
        )
        if not structurally_valid or not values_valid:
            raise ValueError("V24 Anchor view-support artifact is invalid")
        camera_center = -(pose[:3, :3].T @ pose[:3, 3])
        view_ray = camera_center[None, None, :] - points
        query_distance = view_ray.norm(dim=2)
        query_direction = view_ray / query_distance.clamp_min(1e-12).unsqueeze(2)
        candidate_modes = modes[candidates]
        cosine = torch.einsum("nkd,nkmd->nkm", query_direction, candidate_modes)
        angular_distance = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
        candidate_radii = radii[candidates]
        mode_rows = torch.arange(2, device=candidates.device)[None, None, :]
        valid_mode = mode_rows < mode_count[candidates].unsqueeze(2)
        direction_supported = (
            valid_mode
            & (
                angular_distance
                <= candidate_radii + float(cfg["view_direction_slack_deg"])
            )
        ).any(dim=2)
        distance_supported = (
            query_distance
            >= minimum_distance[candidates]
            * float(cfg["minimum_mapping_distance_ratio"])
        ) & (
            query_distance
            <= maximum_distance[candidates]
            * float(cfg["maximum_mapping_distance_ratio"])
        )
        view_supported = (
            direction_supported
            & distance_supported
            & torch.isfinite(query_distance)
            & (query_distance > 1e-8)
        )

    first_pass_inlier = torch.zeros(count, dtype=torch.bool, device=candidates.device)
    first_pass_inlier[inliers] = True
    soft_inlier = (
        first_pass_inlier
        & bool(cfg["allow_soft_inliers"])
        & (residual[:, 0] >= float(cfg["soft_inlier_minimum_baseline_residual_px"]))
    )
    protected = first_pass_inlier & (~soft_inlier)
    reserved_anchors = torch.zeros(
        xyz.shape[0], dtype=torch.bool, device=candidates.device
    )
    if inliers.numel():
        reserved_anchors[baseline[inliers]] = True

    score_drop = base_scores[:, None] - scores
    maximum_drop = float(cfg["maximum_score_drop_from_top1"])
    descriptor_cost = score_drop.clamp_min(0.0) / maximum_drop
    geometry_cost = residual / edge_projection_gate
    reliability_cost = torch.zeros_like(geometry_cost)
    reliability_matchability_threshold = 0.0
    reliability_uncertainty_threshold = 0.0
    if (anchor_matchability is None) != (anchor_uncertainty is None):
        raise ValueError("V24 mapping reliability inputs must be paired")
    if anchor_matchability is not None:
        matchability = torch.as_tensor(
            anchor_matchability, device=candidates.device
        ).float()
        uncertainty = torch.as_tensor(
            anchor_uncertainty, device=candidates.device
        ).float()
        values_valid = bool(mapping_reliability_validated) or bool(
            torch.isfinite(matchability).all()
            and torch.isfinite(uncertainty).all()
            and (uncertainty >= 0).all()
        )
        if (
            matchability.shape != uncertainty.shape
            or matchability.shape != (xyz.shape[0],)
            or not values_valid
        ):
            raise ValueError("V24 mapping reliability inputs are invalid")
        reliability_cost = 0.5 * (1.0 - matchability[candidates].clamp(0.0, 1.0))
        reliability_cost += 0.5 * (
            uncertainty[candidates] / (1.0 + uncertainty[candidates])
        )
    elif bool(cfg["reliability_adaptive_score_drop"]):
        raise ValueError(
            "reliability-adaptive score drop requires mapping reliability"
        )
    joint_cost = (
        float(cfg["descriptor_cost_weight"]) * descriptor_cost
        + float(cfg["geometry_cost_weight"]) * geometry_cost
        + float(cfg["mapping_reliability_cost_weight"]) * reliability_cost
    )
    edge_maximum_drop = torch.full(
        (count, topk), maximum_drop, dtype=scores.dtype, device=candidates.device
    )
    reliability_authorized = torch.zeros_like(residual, dtype=torch.bool)
    expanded_budget_edge = torch.zeros_like(residual, dtype=torch.bool)
    if bool(cfg["reliability_adaptive_score_drop"]):
        bounded_matchability = matchability.clamp(0.0, 1.0)
        reliability_matchability_threshold = float(
            torch.quantile(
                bounded_matchability,
                float(cfg["reliability_minimum_matchability_quantile"]),
            ).item()
        )
        reliability_uncertainty_threshold = float(
            torch.quantile(
                uncertainty,
                float(cfg["reliability_maximum_uncertainty_quantile"]),
            ).item()
        )
        reliability_authorized = (
            (~first_pass_inlier[:, None])
            & (
                bounded_matchability[candidates]
                >= reliability_matchability_threshold
            )
            & (uncertainty[candidates] <= reliability_uncertainty_threshold)
            & (
                geometry_cost
                <= float(cfg["reliability_maximum_geometry_cost"])
            )
            & (
                residual
                <= residual[:, :1]
                - float(
                    cfg["reliability_minimum_reprojection_improvement_px"]
                )
            )
            & view_supported
        )
        base_geometric_candidate = (
            (~protected[:, None])
            & (score_drop <= maximum_drop)
            & (candidates != baseline[:, None])
            & (~reserved_anchors[candidates])
            & view_supported
            & (residual <= edge_projection_gate)
            & (
                residual
                <= residual[:, :1]
                - float(cfg["minimum_reprojection_improvement_px"])
            )
        )
        if bool(cfg["reliability_expansion_requires_no_base_geometric_candidate"]):
            reliability_authorized &= ~base_geometric_candidate.any(
                dim=1, keepdim=True
            )
        edge_maximum_drop[reliability_authorized] = float(
            cfg["reliability_expanded_score_drop"]
        )
        expanded_budget_edge = (
            reliability_authorized
            & (score_drop > maximum_drop)
            & (score_drop <= float(cfg["reliability_expanded_score_drop"]))
        )
    edge_maximum_drop[soft_inlier] = float(
        cfg["soft_inlier_maximum_score_drop"]
    )
    row_minimum_improvement = torch.full(
        (count, 1),
        float(cfg["minimum_reprojection_improvement_px"]),
        dtype=residual.dtype,
        device=candidates.device,
    )
    row_minimum_improvement[soft_inlier] = float(
        cfg["soft_inlier_minimum_reprojection_improvement_px"]
    )
    descriptor_view_eligible = (
        (~protected[:, None])
        & (score_drop <= edge_maximum_drop)
        & (candidates != baseline[:, None])
        & (~reserved_anchors[candidates])
        & view_supported
    )
    verification_rows = torch.empty(0, dtype=torch.long, device=candidates.device)
    verification_edges = torch.empty(
        (0, topk), dtype=torch.bool, device=candidates.device
    )
    if bool(cfg["heldout_candidate_validation"]):
        if image_hw is None or len(image_hw) != 2:
            raise ValueError("held-out validation requires image dimensions")
        height, width = map(int, image_hw)
        if height <= 0 or width <= 0:
            raise ValueError("held-out validation image dimensions are invalid")
        rows, columns = map(int, cfg["heldout_validation_grid_shape"])
        cell_x = torch.clamp((xy[:, 0] / width * columns).long(), 0, columns - 1)
        cell_y = torch.clamp((xy[:, 1] / height * rows).long(), 0, rows - 1)
        deterministic_holdout = (
            (cell_x + 3 * cell_y) % int(cfg["heldout_validation_modulus"])
        ) == 0
        has_alternative = descriptor_view_eligible.any(dim=1)
        verification_mask = deterministic_holdout & has_alternative
        tentative_rows = torch.nonzero(verification_mask, as_tuple=False).reshape(-1)
        if tentative_rows.numel() >= int(cfg["heldout_validation_minimum_rows"]):
            verification_rows = tentative_rows
            # Rank zero supplies the native correspondence.  The alternative
            # edges use descriptor + mapping support only; no T1 evidence and
            # no ground truth enters this held-out graph.
            validation_mask = descriptor_view_eligible.clone()
            validation_mask[:, 0] = True
            verification_edges = validation_mask[verification_rows]

    eligible = (
        descriptor_view_eligible
        & (residual <= edge_projection_gate)
        & (residual <= residual[:, :1] - row_minimum_improvement)
    )
    if verification_rows.numel():
        # These rows are excluded from both proposal selection and the second
        # pose solve.  They are consumed only after T1 has been estimated.
        eligible[verification_rows] = False
    # Rank zero is the baseline correspondence and is never a replacement.
    eligible[:, 0] = False
    mutual_rejected_edges = 0
    if bool(cfg["pose_conditioned_mutual_matching"]):
        # Sparse pose-conditioned mutual check: a geometrically feasible
        # Anchor accepts only the query row with the strongest descriptor
        # score in this query's candidate graph.  This is deliberately applied
        # after the geometric/view gates, so it neither scans the full image x
        # map graph nor introduces a dense matching stage.
        eligible_before_mutual = int(eligible.sum().item())
        edge_anchors = candidates[eligible]
        edge_scores = scores[eligible]
        maximum_score = torch.full(
            (xyz.shape[0],),
            -torch.inf,
            dtype=scores.dtype,
            device=candidates.device,
        )
        if edge_anchors.numel():
            maximum_score.scatter_reduce_(
                0,
                edge_anchors,
                edge_scores,
                reduce="amax",
                include_self=True,
            )
            eligible &= scores == maximum_score[candidates]
        mutual_rejected_edges = eligible_before_mutual - int(eligible.sum().item())
    ranked_cost = joint_cost.masked_fill(~eligible, torch.inf)
    best_cost, best_rank = ranked_cost.min(dim=1)
    selected = baseline.clone()
    has_candidate = torch.isfinite(best_cost)
    selected[has_candidate] = candidates.gather(1, best_rank[:, None]).reshape(-1)[
        has_candidate
    ]
    changed = has_candidate & (selected != baseline)
    changed = _keep_unique_lowest_cost_owner(
        selected_rows=selected,
        selected_costs=best_cost,
        changed=changed,
        anchor_count=xyz.shape[0],
    )
    soft_changed_rows = torch.nonzero(changed & soft_inlier, as_tuple=False).reshape(-1)
    soft_capacity_rejections = 0
    if soft_changed_rows.numel() > int(cfg["maximum_soft_inlier_changes"]):
        ranked_soft = soft_changed_rows[
            torch.argsort(best_cost[soft_changed_rows], stable=True)
        ]
        rejected_soft = ranked_soft[int(cfg["maximum_soft_inlier_changes"]) :]
        changed[rejected_soft] = False
        soft_capacity_rejections = int(rejected_soft.numel())
    changed_rows = torch.nonzero(changed, as_tuple=False).reshape(-1)
    maximum_changed = min(
        int(cfg["maximum_changed_rows"]),
        int(float(cfg["maximum_changed_to_baseline_inlier_ratio"]) * inliers.numel()),
    )
    capacity_rejections = 0
    if changed_rows.numel() > maximum_changed:
        if bool(cfg["set_level_reserve_selection"]):
            if image_hw is None:
                raise ValueError(
                    "set-level Reserve selection requires image dimensions"
                )
            height, width = map(int, image_hw)
            row_xy = xy[changed_rows]
            cell_x = (row_xy[:, 0] * 8 / width).long().clamp(0, 7)
            cell_y = (row_xy[:, 1] * 6 / height).long().clamp(0, 5)
            cells = cell_y * 8 + cell_x
            selected_depth = camera[changed_rows, best_rank[changed_rows], 2]
            depth_low = torch.quantile(selected_depth, 0.02)
            depth_high = torch.quantile(selected_depth, 0.98)
            depth_bin = (
                (
                    (selected_depth - depth_low)
                    / (depth_high - depth_low).clamp_min(1e-8)
                    * 4
                )
                .long()
                .clamp(0, 3)
            )
            cell_count = torch.zeros(48, device=changed_rows.device)
            depth_count = torch.zeros(4, device=changed_rows.device)
            available = torch.ones(
                changed_rows.numel(), dtype=torch.bool, device=changed_rows.device
            )
            chosen = []
            for _ in range(maximum_changed):
                utility = -best_cost[changed_rows]
                utility += float(cfg["set_level_diversity_weight"]) * (
                    (1.0 + cell_count[cells]).rsqrt()
                    + (1.0 + depth_count[depth_bin]).rsqrt()
                )
                utility[~available] = -torch.inf
                local = int(torch.argmax(utility))
                chosen.append(local)
                available[local] = False
                cell_count[cells[local]] += 1
                depth_count[depth_bin[local]] += 1
            ranked_rows = changed_rows[
                torch.as_tensor(chosen, device=changed_rows.device)
            ]
            rejected = changed_rows[available]
        else:
            ranked_rows = changed_rows[
                torch.argsort(best_cost[changed_rows], stable=True)
            ]
            rejected = ranked_rows[maximum_changed:]
        changed[rejected] = False
        capacity_rejections = int(rejected.numel())
    selected[~changed] = baseline[~changed]

    if bool((changed & protected).any()):
        raise RuntimeError("V24 refinement changed a protected first-pass inlier")
    changed_rows = torch.nonzero(changed, as_tuple=False).reshape(-1)
    ranks = best_rank[changed_rows]
    adaptive_selected = (
        expanded_budget_edge[changed_rows, ranks]
        if changed_rows.numel()
        else torch.empty(0, dtype=torch.bool, device=candidates.device)
    )
    return {
        "anchor_rows": selected,
        "changed_query_rows": changed_rows,
        "selected_candidate_ranks": ranks + 1,
        "selected_reprojection_residual_px": residual[changed_rows, ranks],
        "selected_score_drop": score_drop[changed_rows, ranks],
        "selected_joint_cost": best_cost[changed_rows],
        "eligible_edge_count": int(eligible.sum().item()),
        "reliability_adaptive_score_drop_enabled": bool(
            cfg["reliability_adaptive_score_drop"]
        ),
        "reliability_authorized_edge_count": int(
            reliability_authorized.sum().item()
        ),
        "reliability_expanded_budget_edge_count": int(
            expanded_budget_edge.sum().item()
        ),
        "reliability_expanded_selected_row_count": int(
            adaptive_selected.sum().item()
        ),
        "reliability_fallback_query_row_count": int(
            reliability_authorized.any(dim=1).sum().item()
        ),
        "reliability_matchability_threshold": float(
            reliability_matchability_threshold
        ),
        "reliability_uncertainty_threshold": float(
            reliability_uncertainty_threshold
        ),
        "mutual_candidate_matching_enabled": bool(
            cfg["pose_conditioned_mutual_matching"]
        ),
        "mutual_candidate_rejected_edge_count": mutual_rejected_edges,
        "heldout_candidate_validation_enabled": bool(
            cfg["heldout_candidate_validation"]
        ),
        "heldout_validation_query_rows": verification_rows,
        "heldout_validation_anchor_rows": candidates[verification_rows],
        "heldout_validation_scores": scores[verification_rows],
        "heldout_validation_edge_mask": verification_edges,
        "heldout_validation_edge_count": int(verification_edges.sum().item()),
        "uncertainty_aware_projection_enabled": bool(
            cfg["uncertainty_aware_projection"]
        ),
        "pose_information_condition": float(pose_information_condition),
        "expanded_projection_edge_count": expanded_projection_edges,
        "projection_gate_p50_px": float(edge_projection_gate.median().item()),
        "projection_gate_p90_px": float(
            torch.quantile(edge_projection_gate, 0.90).item()
        ),
        "pre_uniqueness_changed_row_count": int(has_candidate.sum().item()),
        "duplicate_candidate_owner_rejection_count": int(
            has_candidate.sum().item()
            - changed.sum().item()
            - capacity_rejections
            - soft_capacity_rejections
        ),
        "capacity_rejection_count": capacity_rejections,
        "set_level_reserve_selection_enabled": bool(cfg["set_level_reserve_selection"]),
        "soft_inlier_candidate_row_count": int(soft_inlier.sum().item()),
        "soft_inlier_changed_row_count": int((changed & soft_inlier).sum().item()),
        "soft_inlier_capacity_rejection_count": soft_capacity_rejections,
        "hard_core_inlier_row_count": int(protected.sum().item()),
        "view_support_available": bool(view_support_available),
        "view_support_rejected_edge_count": int((~view_supported).sum().item()),
    }


@torch.inference_mode()
def compare_poses_on_heldout_candidate_graph(
    *,
    keypoints: torch.Tensor,
    candidate_anchor_rows: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_edge_mask: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    candidate_pose_w2c: torch.Tensor,
    maximum_score_drop_from_top1: float,
    robust_scale_px: float,
    dustbin_energy: float = 2.0,
) -> dict:
    """Compare T0/T1 on correspondences excluded from the T1 solve.

    Both poses receive the same descriptor/view-supported sparse graph.  Each
    pose obtains its own strict one-query/one-Anchor optimum with a fixed
    dustbin cost.  The graph contains no ground-truth identity and is never
    used to estimate T1, which makes this a genuinely held-out self-consistency
    check rather than the former in-sample common-grid score.
    """

    xy = torch.as_tensor(keypoints).float()
    candidates = torch.as_tensor(candidate_anchor_rows, device=xy.device).long()
    scores = torch.as_tensor(candidate_scores, device=xy.device).float()
    valid = torch.as_tensor(candidate_edge_mask, device=xy.device).bool()
    xyz = torch.as_tensor(anchor_xyz, device=xy.device).float()
    calibration = torch.as_tensor(intrinsic, device=xy.device).float()
    poses = torch.stack(
        (
            torch.as_tensor(baseline_pose_w2c, device=xy.device).float(),
            torch.as_tensor(candidate_pose_w2c, device=xy.device).float(),
        )
    )
    maximum_drop = float(maximum_score_drop_from_top1)
    scale = float(robust_scale_px)
    dustbin = float(dustbin_energy)
    if not (
        xy.ndim == 2
        and xy.shape[1] == 2
        and candidates.ndim == 2
        and candidates.shape == scores.shape == valid.shape
        and candidates.shape[0] == xy.shape[0]
        and candidates.shape[1] >= 2
        and xyz.ndim == 2
        and xyz.shape[1] == 3
        and calibration.shape == (3, 3)
        and poses.shape == (2, 4, 4)
        and bool(torch.isfinite(xy).all())
        and bool(torch.isfinite(scores).all())
        and bool(torch.isfinite(xyz).all())
        and bool(torch.isfinite(calibration).all())
        and bool(torch.isfinite(poses).all())
        and (not candidates.numel() or int(candidates.min()) >= 0)
        and (not candidates.numel() or int(candidates.max()) < xyz.shape[0])
        and maximum_drop > 0.0
        and scale > 0.0
        and dustbin > 0.0
        and bool(valid[:, 0].all())
    ):
        raise ValueError("held-out candidate-graph inputs are invalid")
    if candidates.shape[0] < 4:
        raise ValueError("held-out candidate graph needs at least four rows")
    if candidates.shape[1] > 1:
        ordered = torch.sort(candidates, dim=1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("held-out candidate rows must be unique")

    points = xyz[candidates.reshape(-1)].reshape(*candidates.shape, 3)
    camera = torch.einsum("pij,nkj->pnki", poses[:, :3, :3], points)
    camera += poses[:, None, None, :3, 3]
    homogeneous = torch.einsum("ij,pnkj->pnki", calibration, camera)
    depth = homogeneous[..., 2]
    projected = homogeneous[..., :2] / depth.clamp_min(1e-12).unsqueeze(-1)
    residual = (projected - xy[None, :, None, :]).norm(dim=3)
    finite = torch.isfinite(residual) & (depth > 1e-12)
    descriptor_cost = (scores[:, :1] - scores).clamp_min(0.0) / maximum_drop
    geometry_cost = torch.log1p((residual / scale) ** 2)
    energy = 0.5 * descriptor_cost[None] + 0.5 * geometry_cost
    edge_valid = valid[None] & finite

    total_energy = []
    matched_count = []
    for pose_index in range(2):
        utility = -energy[pose_index]
        utility = utility.masked_fill(~edge_valid[pose_index], -dustbin)
        order = torch.argsort(utility, dim=1, descending=True, stable=True)
        ordered_utility = torch.gather(utility, 1, order)
        ordered_anchors = torch.gather(candidates, 1, order)
        assignment = maximum_weight_anchor_assignment(
            TopKMatches(
                keypoint_indices=torch.arange(
                    candidates.shape[0], device=candidates.device
                ),
                anchor_indices=ordered_anchors,
                scores=ordered_utility,
            ),
            dustbin_score=-dustbin,
        )
        assigned_rows = assignment.matches.keypoint_indices.long()
        assigned_anchors = assignment.matches.anchor_indices.long()
        assigned_utility = utility.new_empty((0,))
        if assigned_rows.numel():
            locations = ordered_anchors[assigned_rows] == assigned_anchors[:, None]
            if not bool((locations.sum(dim=1) == 1).all()):
                raise RuntimeError("held-out assignment edge is not unique")
            assigned_ranks = locations.float().argmax(dim=1)
            assigned_utility = ordered_utility[assigned_rows, assigned_ranks]
        utility_sum = utility.new_tensor(-dustbin * candidates.shape[0])
        if assigned_rows.numel():
            utility_sum += assigned_utility.sum() + dustbin * assigned_rows.numel()
        total_energy.append(-utility_sum / candidates.shape[0])
        matched_count.append(int(assigned_rows.numel()))

    baseline_energy, candidate_energy = total_energy
    relative_gain = (baseline_energy - candidate_energy) / baseline_energy.clamp_min(
        1e-12
    )
    return {
        "baseline_energy": baseline_energy,
        "candidate_energy": candidate_energy,
        "relative_energy_gain": relative_gain,
        "baseline_assignment_count": matched_count[0],
        "candidate_assignment_count": matched_count[1],
        "query_row_count": int(candidates.shape[0]),
        "candidate_edge_count": int(valid.sum().item()),
        "strict_one_to_one": True,
        "solver_rows_used": False,
        "ground_truth_used": False,
    }


@torch.inference_mode()
def compare_poses_on_common_candidate_grid(
    *,
    keypoints: torch.Tensor,
    topk_anchor_rows: torch.Tensor,
    topk_scores: torch.Tensor,
    baseline_inlier_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    candidate_pose_w2c: torch.Tensor,
    maximum_score_drop_from_top1: float,
    robust_scale_px: float,
    descriptor_penalty_weight: float = 0.25,
    duplicate_owner_penalty: float = 0.25,
) -> dict:
    """Compare two poses on exactly the same sparse correspondence grid.

    Each non-core query row may choose its best Top-K explanation under either
    pose. First-pass inliers remain hard core and can use only their original
    correspondence. A bounded duplicate-owner penalty discourages a pose from
    looking good by assigning the same 3D Anchor to several image points. The
    Cauchy reprojection term is robust, so this is a model-comparison score and
    not another PnP solve.
    """

    xy = torch.as_tensor(keypoints).float()
    candidates = torch.as_tensor(topk_anchor_rows, device=xy.device).long()
    scores = torch.as_tensor(topk_scores, device=xy.device).float()
    inliers = torch.as_tensor(baseline_inlier_rows, device=xy.device).long().reshape(-1)
    xyz = torch.as_tensor(anchor_xyz, device=xy.device).float()
    calibration = torch.as_tensor(intrinsic, device=xy.device).float()
    poses = torch.stack(
        (
            torch.as_tensor(baseline_pose_w2c, device=xy.device).float(),
            torch.as_tensor(candidate_pose_w2c, device=xy.device).float(),
        )
    )
    maximum_drop = float(maximum_score_drop_from_top1)
    scale = float(robust_scale_px)
    descriptor_weight = float(descriptor_penalty_weight)
    duplicate_penalty = float(duplicate_owner_penalty)
    if not (
        xy.ndim == 2
        and xy.shape[1] == 2
        and candidates.ndim == 2
        and candidates.shape == scores.shape
        and candidates.shape[0] == xy.shape[0]
        and candidates.shape[1] >= 2
        and xyz.ndim == 2
        and xyz.shape[1] == 3
        and calibration.shape == (3, 3)
        and poses.shape == (2, 4, 4)
        and bool(torch.isfinite(xy).all())
        and bool(torch.isfinite(scores).all())
        and bool(torch.isfinite(xyz).all())
        and bool(torch.isfinite(calibration).all())
        and bool(torch.isfinite(poses).all())
        and (not candidates.numel() or int(candidates.min()) >= 0)
        and (not candidates.numel() or int(candidates.max()) < xyz.shape[0])
        and (
            not inliers.numel()
            or (int(inliers.min()) >= 0 and int(inliers.max()) < xy.shape[0])
        )
        and inliers.unique().numel() == inliers.numel()
        and 0.0 < maximum_drop <= 0.2
        and scale > 0.0
        and descriptor_weight >= 0.0
        and duplicate_penalty >= 0.0
    ):
        raise ValueError("V25 common candidate-grid inputs are invalid")

    points = xyz[candidates.reshape(-1)].reshape(*candidates.shape, 3)
    camera = torch.einsum("pij,nkj->pnki", poses[:, :3, :3], points)
    camera = camera + poses[:, None, None, :3, 3]
    homogeneous = torch.einsum("ij,pnkj->pnki", calibration, camera)
    depth = homogeneous[..., 2]
    projected = homogeneous[..., :2] / depth.clamp_min(1e-12).unsqueeze(-1)
    residual = (projected - xy[None, :, None, :]).norm(dim=-1)
    residual = residual.masked_fill(depth <= 1e-12, torch.inf)

    score_drop = (scores[:, :1] - scores).clamp_min(0.0)
    valid = score_drop <= maximum_drop
    valid[:, 0] = True
    if inliers.numel():
        valid[inliers] = False
        valid[inliers, 0] = True
    descriptor_cost = score_drop / maximum_drop
    robust_cost = torch.log1p((residual / scale).square())
    # An Anchor behind the camera is a poor explanation, but it must remain a
    # finite model-comparison cost. Otherwise a globally bad first pose can
    # make an entire query impossible to score instead of being penalized.
    invalid_projection_cost = torch.log1p(
        torch.tensor(1e4, dtype=robust_cost.dtype, device=xy.device)
    )
    robust_cost = torch.where(
        torch.isfinite(robust_cost), robust_cost, invalid_projection_cost
    )
    edge_cost = robust_cost + descriptor_weight * descriptor_cost[None, :, :]
    edge_cost = edge_cost.masked_fill(~valid[None, :, :], torch.inf)
    best_cost, best_rank = edge_cost.min(dim=2)
    if not bool(torch.isfinite(best_cost).all()):
        raise RuntimeError("V25 common candidate grid has an unexplained query row")
    selected_owner = (
        candidates[None, :, :]
        .expand(2, -1, -1)
        .gather(2, best_rank.unsqueeze(2))
        .squeeze(2)
    )
    duplicate_count = torch.stack(
        [
            torch.tensor(
                selected_owner.shape[1] - selected_owner[index].unique().numel(),
                device=xy.device,
                dtype=best_cost.dtype,
            )
            for index in range(2)
        ]
    )
    duplicate_fraction = duplicate_count / max(int(xy.shape[0]), 1)
    reprojection_energy = best_cost.mean(dim=1)
    total_energy = reprojection_energy + duplicate_penalty * duplicate_fraction
    relative_gain = (total_energy[0] - total_energy[1]) / total_energy[0].clamp_min(
        1e-12
    )
    return {
        "baseline_energy": total_energy[0],
        "candidate_energy": total_energy[1],
        "relative_energy_gain": relative_gain,
        "baseline_reprojection_descriptor_energy": reprojection_energy[0],
        "candidate_reprojection_descriptor_energy": reprojection_energy[1],
        "baseline_duplicate_owner_count": int(duplicate_count[0].item()),
        "candidate_duplicate_owner_count": int(duplicate_count[1].item()),
    }


def changed_inlier_spatial_cell_count(
    *,
    keypoints: torch.Tensor,
    changed_query_rows: torch.Tensor,
    candidate_inlier_rows: torch.Tensor,
    image_hw: tuple[int, int],
    grid_shape: tuple[int, int] = (4, 4),
) -> int:
    """Count occupied image cells among changed rows accepted by the second PnP."""

    xy = torch.as_tensor(keypoints).float().cpu()
    changed = torch.as_tensor(changed_query_rows).long().cpu().reshape(-1)
    inliers = torch.as_tensor(candidate_inlier_rows).long().cpu().reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("V24 spatial-support keypoints must have shape [N, 2]")
    if changed.numel() and (
        int(changed.min()) < 0 or int(changed.max()) >= xy.shape[0]
    ):
        raise ValueError("V24 changed row is outside the keypoint registry")
    if inliers.numel() and (
        int(inliers.min()) < 0 or int(inliers.max()) >= xy.shape[0]
    ):
        raise ValueError("V24 candidate inlier is outside the keypoint registry")
    mask = torch.zeros(xy.shape[0], dtype=torch.bool)
    mask[changed] = True
    accepted = inliers[mask[inliers]]
    if not accepted.numel():
        return 0
    height, width = (int(image_hw[0]), int(image_hw[1]))
    rows, columns = (int(grid_shape[0]), int(grid_shape[1]))
    if height <= 0 or width <= 0 or rows <= 0 or columns <= 0:
        raise ValueError("V24 spatial-support grid is invalid")
    cell_x = (xy[accepted, 0] * columns / width).floor().long().clamp(0, columns - 1)
    cell_y = (xy[accepted, 1] * rows / height).floor().long().clamp(0, rows - 1)
    return int((cell_y * columns + cell_x).unique().numel())
