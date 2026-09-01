"""Pose-conditioned sparse correspondence refinement for online localization.

The module is deliberately map read-only.  It preserves every first-pass PnP
inlier and only proposes an alternative Anchor for first-pass outliers.  Unlike
the V21 minimum-reprojection arm, descriptor evidence and geometric evidence
are ranked jointly and a newly selected Anchor can be owned by at most one
query keypoint.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from localization.matcher import TopKMatches, global_cosine_topk


def default_config() -> dict:
    """Return the frozen correspondence-selection contract for the V24 arm."""

    return {
        "topk": 64,
        "projection_gate_px": 8.0,
        "maximum_score_drop_from_top1": 0.10,
        "minimum_reprojection_improvement_px": 1.0,
        "descriptor_cost_weight": 0.50,
        "geometry_cost_weight": 0.40,
        "mapping_reliability_cost_weight": 0.10,
        "view_direction_slack_deg": 20.0,
        "minimum_mapping_distance_ratio": 0.25,
        "maximum_mapping_distance_ratio": 4.0,
        "maximum_changed_rows": 256,
        "maximum_changed_to_baseline_inlier_ratio": 1.0,
        "preserve_all_first_pass_inliers": True,
        "reserve_first_pass_inlier_anchors": True,
        "unique_new_anchor_owner": True,
        "candidate_selection": "minimum_joint_descriptor_geometry_cost",
        "ground_truth_used": False,
        "map_mutated": False,
    }


def validate_config(value: Mapping) -> dict:
    config = dict(value)
    expected = default_config()
    frozen_keys = (
        "topk",
        "projection_gate_px",
        "minimum_reprojection_improvement_px",
        "descriptor_cost_weight",
        "geometry_cost_weight",
        "mapping_reliability_cost_weight",
        "minimum_mapping_distance_ratio",
        "maximum_mapping_distance_ratio",
        "preserve_all_first_pass_inliers",
        "reserve_first_pass_inlier_anchors",
        "unique_new_anchor_owner",
        "candidate_selection",
        "ground_truth_used",
        "map_mutated",
    )
    if set(config) != set(expected) or any(
        config.get(key) != expected[key] for key in frozen_keys
    ):
        raise ValueError("V24 pose-conditioned refinement configuration differs")
    if not (
        0.0 < float(config["maximum_score_drop_from_top1"]) <= 0.2
        and 0.0 <= float(config["view_direction_slack_deg"]) <= 60.0
        and 1 <= int(config["maximum_changed_rows"]) <= 512
        and 0.0
        < float(config["maximum_changed_to_baseline_inlier_ratio"])
        <= 2.0
    ):
        raise ValueError("V24 tunable correspondence-selection gate is invalid")
    return config


def runtime_config(
    *,
    maximum_score_drop_from_top1: float = 0.10,
    view_direction_slack_deg: float = 20.0,
    maximum_changed_rows: int = 256,
    maximum_changed_to_baseline_inlier_ratio: float = 1.0,
) -> dict:
    config = default_config()
    config.update(
        {
            "maximum_score_drop_from_top1": float(
                maximum_score_drop_from_top1
            ),
            "view_direction_slack_deg": float(view_direction_slack_deg),
            "maximum_changed_rows": int(maximum_changed_rows),
            "maximum_changed_to_baseline_inlier_ratio": float(
                maximum_changed_to_baseline_inlier_ratio
            ),
        }
    )
    return validate_config(config)


@torch.inference_mode()
def build_pose_visible_topk(
    *,
    query_descriptors: torch.Tensor,
    normalized_anchor_features: torch.Tensor,
    baseline_anchor_rows: torch.Tensor,
    baseline_scores: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    image_hw: tuple[int, int],
    topk: int = 64,
    image_margin_px: float = 64.0,
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
    height, width = (int(image_hw[0]), int(image_hw[1]))
    requested = int(topk)
    margin = float(image_margin_px)
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
    fallback = bool(visible_rows.numel() < requested)
    pool_rows = (
        torch.arange(features.shape[0], device=query.device)
        if fallback
        else visible_rows
    )
    local = global_cosine_topk(
        query,
        features[pool_rows],
        topk=requested,
        anchor_descriptors_normalized=True,
    )
    mapped_rows = pool_rows[local.anchor_indices]
    alternative_scores = local.scores.masked_fill(
        mapped_rows == baseline[:, None], -torch.inf
    )
    alt_scores, alt_columns = torch.topk(
        alternative_scores, k=requested - 1, dim=1
    )
    alt_rows = mapped_rows.gather(1, alt_columns)
    if not bool(torch.isfinite(alt_scores).all()):
        raise RuntimeError("V24 pose-visible pool has too few alternative Anchors")
    result = TopKMatches(
        keypoint_indices=torch.arange(query.shape[0], device=query.device),
        anchor_indices=torch.cat((baseline[:, None], alt_rows), dim=1),
        scores=torch.cat((scores[:, None], alt_scores), dim=1),
    )
    return {
        "matches": result,
        "visible_anchor_count": int(visible_rows.numel()),
        "candidate_pool_anchor_count": int(pool_rows.numel()),
        "global_fallback": fallback,
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
    mapping_reliability_validated: bool = False,
    map_geometry_validated: bool = False,
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
            and mode_count.shape == minimum_distance.shape == maximum_distance.shape
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

    protected = torch.zeros(count, dtype=torch.bool, device=candidates.device)
    protected[inliers] = True
    reserved_anchors = torch.zeros(
        xyz.shape[0], dtype=torch.bool, device=candidates.device
    )
    if inliers.numel():
        reserved_anchors[baseline[inliers]] = True

    score_drop = base_scores[:, None] - scores
    maximum_drop = float(cfg["maximum_score_drop_from_top1"])
    descriptor_cost = score_drop.clamp_min(0.0) / maximum_drop
    geometry_cost = residual / float(cfg["projection_gate_px"])
    reliability_cost = torch.zeros_like(geometry_cost)
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
        if matchability.shape != uncertainty.shape or matchability.shape != (
            xyz.shape[0],
        ) or not values_valid:
            raise ValueError("V24 mapping reliability inputs are invalid")
        reliability_cost = 0.5 * (1.0 - matchability[candidates].clamp(0.0, 1.0))
        reliability_cost += 0.5 * (
            uncertainty[candidates] / (1.0 + uncertainty[candidates])
        )
    joint_cost = (
        float(cfg["descriptor_cost_weight"]) * descriptor_cost
        + float(cfg["geometry_cost_weight"]) * geometry_cost
        + float(cfg["mapping_reliability_cost_weight"]) * reliability_cost
    )
    eligible = (
        (~protected[:, None])
        & (score_drop <= maximum_drop)
        & (residual <= float(cfg["projection_gate_px"]))
        & (
            residual
            <= residual[:, :1]
            - float(cfg["minimum_reprojection_improvement_px"])
        )
        & (candidates != baseline[:, None])
        & (~reserved_anchors[candidates])
        & view_supported
    )
    # Rank zero is the baseline correspondence and is never a replacement.
    eligible[:, 0] = False
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
    changed_rows = torch.nonzero(changed, as_tuple=False).reshape(-1)
    maximum_changed = min(
        int(cfg["maximum_changed_rows"]),
        int(
            float(cfg["maximum_changed_to_baseline_inlier_ratio"])
            * inliers.numel()
        ),
    )
    capacity_rejections = 0
    if changed_rows.numel() > maximum_changed:
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
    return {
        "anchor_rows": selected,
        "changed_query_rows": changed_rows,
        "selected_candidate_ranks": ranks + 1,
        "selected_reprojection_residual_px": residual[changed_rows, ranks],
        "selected_score_drop": score_drop[changed_rows, ranks],
        "selected_joint_cost": best_cost[changed_rows],
        "eligible_edge_count": int(eligible.sum().item()),
        "pre_uniqueness_changed_row_count": int(has_candidate.sum().item()),
        "duplicate_candidate_owner_rejection_count": int(
            has_candidate.sum().item() - changed.sum().item() - capacity_rejections
        ),
        "capacity_rejection_count": capacity_rejections,
        "view_support_available": bool(view_support_available),
        "view_support_rejected_edge_count": int((~view_supported).sum().item()),
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
    if changed.numel() and (int(changed.min()) < 0 or int(changed.max()) >= xy.shape[0]):
        raise ValueError("V24 changed row is outside the keypoint registry")
    if inliers.numel() and (int(inliers.min()) < 0 or int(inliers.max()) >= xy.shape[0]):
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
