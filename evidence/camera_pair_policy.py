"""Camera-graph policies with exact global budgets and mapping-only geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _camera_centers_and_axes(
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float64)
    camera_centers = -torch.einsum(
        "qji,qj->qi", pose_w2c[:, :3, :3], pose_w2c[:, :3, 3]
    )
    optical_axis = torch.einsum(
        "qji,j->qi",
        pose_w2c[:, :3, :3],
        pose_w2c.new_tensor([0.0, 0.0, 1.0]),
    )
    return camera_centers, F.normalize(optical_axis, dim=1)


def mapping_scene_points_from_depth_samples(
    keypoints: list[torch.Tensor],
    depth_at_keypoints: list[torch.Tensor],
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    *,
    points_per_camera: int = 8,
    maximum_points: int = 4096,
    voxel_size_m: float = 0.02,
) -> torch.Tensor:
    """Build a bounded mapping-only 3D support sample for pair selection."""
    if len(keypoints) != len(depth_at_keypoints):
        raise ValueError("Keypoint/depth camera tables must align")
    K = torch.as_tensor(camera_K, dtype=torch.float64).cpu()
    pose = torch.as_tensor(pose_w2c, dtype=torch.float64).cpu()
    if K.shape[0] != len(keypoints) or pose.shape[0] != len(keypoints):
        raise ValueError("Camera table does not align with depth samples")
    per_camera = max(int(points_per_camera), 1)
    collected = []
    for query, (uv_value, depth_value) in enumerate(
        zip(keypoints, depth_at_keypoints)
    ):
        uv = torch.as_tensor(uv_value, dtype=torch.float64).reshape(-1, 2)
        depth = torch.as_tensor(depth_value, dtype=torch.float64).reshape(-1)
        if uv.shape[0] != depth.numel():
            raise ValueError("Per-camera keypoints and depth samples must align")
        valid = torch.isfinite(depth) & (depth > 1e-6) & torch.isfinite(uv).all(dim=1)
        rows = torch.nonzero(valid, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        if rows.numel() > per_camera:
            offsets = torch.div(
                torch.arange(per_camera, dtype=torch.long) * int(rows.numel()),
                per_camera,
                rounding_mode="floor",
            )
            rows = rows[offsets]
        homogeneous = torch.cat(
            (uv[rows], torch.ones((rows.numel(), 1), dtype=torch.float64)), dim=1
        )
        camera = torch.linalg.solve(K[query], homogeneous.T).T * depth[rows, None]
        rotation = pose[query, :3, :3]
        translation = pose[query, :3, 3]
        world = (camera - translation) @ rotation
        collected.append(world)
    if not collected:
        raise ValueError("Mapping cache has no positive keypoint depth samples")
    points = torch.cat(collected)
    voxel = max(float(voxel_size_m), 1e-6)
    quantized = torch.round(points / voxel).long()
    # Stable first-occurrence voxel reduction avoids density domination by a
    # long near-static trajectory while remaining exactly reproducible.
    _, inverse = torch.unique(quantized, dim=0, return_inverse=True)
    first = torch.full(
        (int(inverse.max()) + 1,), points.shape[0], dtype=torch.long
    )
    row = torch.arange(points.shape[0], dtype=torch.long)
    if hasattr(first, "scatter_reduce_"):
        first.scatter_reduce_(0, inverse, row, reduce="amin", include_self=True)
    else:
        for index, group in enumerate(inverse.tolist()):
            first[group] = min(int(first[group]), index)
    first = first[first < points.shape[0]]
    first = torch.sort(first).values
    points = points[first]
    limit = max(int(maximum_points), 1)
    if points.shape[0] > limit:
        offsets = torch.div(
            torch.arange(limit, dtype=torch.long) * int(points.shape[0]),
            limit,
            rounding_mode="floor",
        )
        points = points[offsets]
    return points.float()


def candidate_camera_pairs(
    pose_w2c: torch.Tensor,
    *,
    neighbors: int = 6,
    minimum_baseline_m: float = 0.03,
    maximum_baseline_m: float = 5.0,
    maximum_axis_angle_deg: float = 75.0,
    policy: str = "nearest",
    pair_budget: int | None = None,
    camera_K: torch.Tensor | None = None,
    image_hw: torch.Tensor | None = None,
    scene_points_xyz: torch.Tensor | None = None,
    minimum_overlap_jaccard: float = 0.15,
    minimum_joint_visibility_points: int = 8,
    parallax_saturation_deg: float = 2.0,
    diversity_weight: float = 0.20,
    candidate_pool_per_camera: int = 48,
    scene_depth_m: torch.Tensor | None = None,
    minimum_expected_parallax_deg: float = 1.0,
    near_fraction: float = 1.0 / 3.0,
    maximum_baseline_depth_ratio: float = 0.5,
) -> list[tuple[int, int]]:
    """Build a deterministic camera graph without descriptor or map IDs.

    ``nearest`` is the frozen legacy policy.  ``parallax_diverse`` first builds
    a bounded pose-only proposal pool, then uses mapping-only scene points to
    require actual common field of view and to estimate pair parallax.  Its
    cardinality is an explicit hard contract: when ``pair_budget`` is omitted,
    the exact legacy graph cardinality is used.  This prevents a graph change
    from silently buying more descriptor comparisons.
    """
    if str(policy) not in {
        "nearest",
        "parallax_diverse",
        "parallax_stratified",
    }:
        raise ValueError(f"Unknown camera pair policy: {policy}")
    centers, axes = _camera_centers_and_axes(pose_w2c)
    count = int(centers.shape[0])
    if count < 2:
        return []
    distance = torch.cdist(centers, centers)
    axis_cosine = (axes @ axes.T).clamp(-1.0, 1.0)
    minimum_cosine = float(
        torch.cos(torch.deg2rad(torch.tensor(maximum_axis_angle_deg))).item()
    )
    valid = (
        (distance >= float(minimum_baseline_m))
        & (distance <= float(maximum_baseline_m))
        & (axis_cosine >= minimum_cosine)
    )
    valid.fill_diagonal_(False)
    positive = distance[valid]
    distance_scale = positive.median().clamp_min(1e-6) if positive.numel() else 1.0
    cost = distance / distance_scale + 0.5 * (1.0 - axis_cosine)
    cost = cost.masked_fill(~valid, torch.inf)
    legacy_pairs = set()
    width = min(max(int(neighbors), 1), max(count - 1, 1))
    for query in range(count):
        candidates = torch.topk(
            cost[query], width, largest=False, sorted=True
        ).indices
        for other in candidates.tolist():
            if not bool(torch.isfinite(cost[query, other])):
                continue
            legacy_pairs.add((min(query, other), max(query, other)))
    if str(policy) == "nearest":
        if pair_budget is not None and int(pair_budget) != len(legacy_pairs):
            raise ValueError(
                "nearest policy does not permit pair-budget cardinality changes"
            )
        return sorted(legacy_pairs)

    if str(policy) == "parallax_stratified":
        if pair_budget is not None and int(pair_budget) != len(legacy_pairs):
            raise ValueError(
                "parallax_stratified requires the exact nearest pair budget"
            )
        from evidence.parallax_stratified_pair_policy import (
            parallax_stratified_pairs,
        )

        return parallax_stratified_pairs(
            centers=centers,
            axes=axes,
            distance=distance,
            axis_cosine=axis_cosine,
            valid=valid,
            cost=cost,
            legacy_pairs=legacy_pairs,
            neighbors=neighbors,
            scene_depth_m=scene_depth_m,
            minimum_expected_parallax_deg=minimum_expected_parallax_deg,
            near_fraction=near_fraction,
            maximum_baseline_depth_ratio=maximum_baseline_depth_ratio,
        )

    if camera_K is None or image_hw is None or scene_points_xyz is None:
        raise ValueError(
            "parallax_diverse policy requires camera_K, image_hw and "
            "mapping-only scene_points_xyz"
        )
    budget = len(legacy_pairs) if pair_budget is None else int(pair_budget)
    if budget < 0:
        raise ValueError("pair_budget must be non-negative")
    if budget == 0:
        return []

    camera_K = torch.as_tensor(camera_K, dtype=torch.float64).cpu()
    image_hw = torch.as_tensor(image_hw, dtype=torch.long).cpu()
    scene_points_xyz = torch.as_tensor(
        scene_points_xyz, dtype=torch.float64
    ).reshape(-1, 3).cpu()
    if camera_K.shape[0] != count or image_hw.shape != (count, 2):
        raise ValueError("Camera intrinsics/image sizes must align with poses")
    finite_points = torch.isfinite(scene_points_xyz).all(dim=1)
    scene_points_xyz = scene_points_xyz[finite_points]
    if int(scene_points_xyz.shape[0]) < int(minimum_joint_visibility_points):
        raise ValueError("Too few finite mapping-only scene points")

    # Pose-only proposals include local-overlap candidates and candidates with
    # useful transverse baselines.  The expensive scene-point test is confined
    # to this pool and never observes descriptors or Track identities.
    pool_width = min(
        max(int(candidate_pool_per_camera), int(neighbors), 1), count - 1
    )
    camera = torch.einsum(
        "qij,pj->qpi", pose_w2c[:, :3, :3].double(), scene_points_xyz
    ) + pose_w2c[:, None, :3, 3].double()
    depth = camera[..., 2]
    representative_depth = torch.nanmedian(
        depth.masked_fill(depth <= 1e-6, torch.nan), dim=1
    ).values
    fallback_depth = representative_depth[torch.isfinite(representative_depth)]
    fallback_depth = (
        fallback_depth.median()
        if fallback_depth.numel()
        else distance_scale.new_tensor(1.0)
    )
    representative_depth = torch.where(
        torch.isfinite(representative_depth),
        representative_depth,
        fallback_depth,
    ).clamp_min(1e-3)
    displacement = centers[None, :, :] - centers[:, None, :]
    axial = (displacement * axes[:, None, :]).sum(dim=2, keepdim=True)
    transverse = torch.linalg.norm(
        displacement - axial * axes[:, None, :], dim=2
    )
    expected_parallax = torch.rad2deg(
        torch.atan2(transverse, representative_depth[:, None])
    )
    saturation = max(float(parallax_saturation_deg), 1e-6)
    # Hard saturation is deliberate: once a pair supplies enough angular
    # leverage, more baseline is not intrinsically better and can hurt
    # descriptor overlap.  FoV overlap and relative-pose diversity decide
    # between all pairs above the target.
    coarse_parallax = (expected_parallax / saturation).clamp(0.0, 1.0)
    coarse_overlap = ((axis_cosine - minimum_cosine) / (1.0 - minimum_cosine)).clamp(
        0.0, 1.0
    )
    coarse_score = 0.65 * coarse_parallax + 0.35 * coarse_overlap
    coarse_score = coarse_score.masked_fill(~valid, -torch.inf)
    # Retain half local-overlap proposals so a large baseline cannot crowd out
    # every feasible pair before the actual FoV test.
    local_width = min(max(pool_width // 2, 1), count - 1)
    geometric_width = min(pool_width - local_width, count - 1)
    proposal_pairs: set[tuple[int, int]] = set()
    for query in range(count):
        local = torch.topk(
            cost[query], local_width, largest=False, sorted=True
        ).indices
        geometric = (
            torch.topk(
                coarse_score[query], geometric_width, largest=True, sorted=True
            ).indices
            if geometric_width > 0
            else torch.empty(0, dtype=torch.long)
        )
        for other in torch.cat((local, geometric)).tolist():
            if not bool(valid[query, other]):
                continue
            proposal_pairs.add((min(query, other), max(query, other)))
    proposals = sorted(proposal_pairs)
    if not proposals:
        raise RuntimeError("No feasible parallax-diverse camera proposals")

    # Visibility and rays are exact for the supplied mapping-only scene-point
    # sample.  They are computed once, then gathered in bounded pair batches.
    projected = torch.einsum("qij,qpj->qpi", camera_K, camera)
    uv = projected[..., :2] / depth[..., None].clamp_min(1e-8)
    visibility = (
        (depth > 1e-6)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] < image_hw[:, 1, None])
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] < image_hw[:, 0, None])
    )
    rays = F.normalize(scene_points_xyz[None] - centers[:, None], dim=2)
    proposal_tensor = torch.as_tensor(proposals, dtype=torch.long)
    overlap_values = torch.zeros(len(proposals), dtype=torch.float64)
    parallax_values = torch.full(
        (len(proposals),), float("nan"), dtype=torch.float64
    )
    joint_counts = torch.zeros(len(proposals), dtype=torch.long)
    for start in range(0, len(proposals), 256):
        end = min(start + 256, len(proposals))
        left = proposal_tensor[start:end, 0]
        right = proposal_tensor[start:end, 1]
        joint = visibility[left] & visibility[right]
        union = visibility[left] | visibility[right]
        joint_count = joint.sum(dim=1)
        union_count = union.sum(dim=1)
        joint_counts[start:end] = joint_count
        overlap_values[start:end] = joint_count.double() / union_count.clamp_min(
            1
        ).double()
        cosine = (rays[left] * rays[right]).sum(dim=2).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine)).masked_fill(~joint, torch.nan)
        parallax_values[start:end] = torch.nanmedian(angles, dim=1).values
    feasible = (
        (joint_counts >= int(minimum_joint_visibility_points))
        & (overlap_values >= float(minimum_overlap_jaccard))
        & torch.isfinite(parallax_values)
    )
    proposal_tensor = proposal_tensor[feasible]
    overlap_values = overlap_values[feasible]
    parallax_values = parallax_values[feasible]
    if int(proposal_tensor.shape[0]) < budget:
        raise RuntimeError(
            "Overlap-constrained pair pool cannot satisfy the exact budget: "
            f"{int(proposal_tensor.shape[0])} < {budget}"
        )

    incident: list[list[int]] = [[] for _ in range(count)]
    for index, (left, right) in enumerate(proposal_tensor.tolist()):
        incident[left].append(index)
        incident[right].append(index)
    parallax_utility = (parallax_values / saturation).clamp(0.0, 1.0)
    base_utility = 0.70 * parallax_utility + 0.30 * overlap_values
    selected: set[int] = set()
    selected_features: list[list[torch.Tensor]] = [[] for _ in range(count)]
    degrees = torch.zeros(count, dtype=torch.long)

    def endpoint_feature(query: int, other: int, pair_index: int) -> torch.Tensor:
        direction_world = F.normalize(
            centers[other] - centers[query], dim=0
        )
        direction_camera = pose_w2c[query, :3, :3].double() @ direction_world
        return torch.cat(
            (
                direction_camera,
                parallax_values[pair_index : pair_index + 1] / saturation,
                torch.acos(axis_cosine[query, other]).reshape(1),
            )
        )

    def diversity(query: int, feature: torch.Tensor) -> float:
        previous = selected_features[query]
        if not previous:
            return 1.0
        stacked = torch.stack(previous)
        direction_cosine = (stacked[:, :3] * feature[:3]).sum(dim=1).clamp(
            -1.0, 1.0
        )
        direction_distance = torch.acos(direction_cosine) / torch.pi
        parallax_distance = (
            (stacked[:, 3] - feature[3]).abs().clamp_max(1.0)
        )
        axis_distance = (
            (stacked[:, 4] - feature[4]).abs() / torch.pi
        ).clamp_max(1.0)
        return float(
            (0.70 * direction_distance + 0.20 * parallax_distance + 0.10 * axis_distance)
            .min()
            .item()
        )

    # Round-robin proposal gives every mapping camera repeated opportunity;
    # an incoming edge contributes to its degree and diversity as well.
    while len(selected) < budget:
        progress = False
        for query in range(count):
            if len(selected) >= budget:
                break
            best = None
            best_key = None
            for pair_index in incident[query]:
                if pair_index in selected:
                    continue
                left, right = proposal_tensor[pair_index].tolist()
                other = right if query == left else left
                feature_query = endpoint_feature(query, other, pair_index)
                feature_other = endpoint_feature(other, query, pair_index)
                marginal = float(base_utility[pair_index]) + float(
                    diversity_weight
                ) * (
                    diversity(query, feature_query)
                    + 0.5 * diversity(other, feature_other)
                ) - 0.01 * float(degrees[other])
                key = (marginal, -pair_index)
                if best_key is None or key > best_key:
                    best = (pair_index, other, feature_query, feature_other)
                    best_key = key
            if best is None:
                continue
            pair_index, other, feature_query, feature_other = best
            selected.add(pair_index)
            selected_features[query].append(feature_query)
            selected_features[other].append(feature_other)
            degrees[query] += 1
            degrees[other] += 1
            progress = True
        if not progress:
            raise RuntimeError("Failed to fill exact parallax-diverse pair budget")
    result = [
        tuple(proposal_tensor[index].tolist()) for index in sorted(selected)
    ]
    if len(result) != budget:
        raise AssertionError("Pair-policy exact-budget contract failed")
    return sorted(result)


def _camera_pair_geometry_table(
    pairs: list[tuple[int, int]],
    pose_w2c: torch.Tensor,
    *,
    camera_K: torch.Tensor | None = None,
    image_hw: torch.Tensor | None = None,
    scene_points_xyz: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return auditable pose/FoV geometry for an already selected pair set."""
    pose = torch.as_tensor(pose_w2c, dtype=torch.float64).cpu()
    centers, axes = _camera_centers_and_axes(pose)
    pair = torch.as_tensor(pairs, dtype=torch.long).reshape(-1, 2)
    if pair.numel() == 0:
        return {
            "left_query_index": torch.zeros(0, dtype=torch.long),
            "right_query_index": torch.zeros(0, dtype=torch.long),
            "baseline_m": torch.zeros(0, dtype=torch.float64),
            "axis_angle_deg": torch.zeros(0, dtype=torch.float64),
            "mapping_point_joint_visibility_count": torch.zeros(
                0, dtype=torch.long
            ),
            "mapping_point_overlap_jaccard": torch.zeros(
                0, dtype=torch.float64
            ),
            "mapping_point_parallax_median_deg": torch.zeros(
                0, dtype=torch.float64
            ),
        }
    left, right = pair[:, 0], pair[:, 1]
    baseline = torch.linalg.norm(centers[left] - centers[right], dim=1)
    axis_cosine = (axes[left] * axes[right]).sum(dim=1).clamp(-1.0, 1.0)
    result = {
        "left_query_index": left,
        "right_query_index": right,
        "baseline_m": baseline,
        "axis_angle_deg": torch.rad2deg(torch.acos(axis_cosine)),
        "mapping_point_joint_visibility_count": torch.full(
            (len(pairs),), -1, dtype=torch.long
        ),
        "mapping_point_overlap_jaccard": torch.full(
            (len(pairs),), float("nan"), dtype=torch.float64
        ),
        "mapping_point_parallax_median_deg": torch.full(
            (len(pairs),), float("nan"), dtype=torch.float64
        ),
    }
    if camera_K is None or image_hw is None or scene_points_xyz is None:
        return result
    K = torch.as_tensor(camera_K, dtype=torch.float64).cpu()
    hw = torch.as_tensor(image_hw, dtype=torch.long).cpu()
    points = torch.as_tensor(scene_points_xyz, dtype=torch.float64).reshape(-1, 3)
    points = points[torch.isfinite(points).all(dim=1)].cpu()
    if points.numel() == 0:
        return result
    camera = torch.einsum("qij,pj->qpi", pose[:, :3, :3], points) + pose[
        :, None, :3, 3
    ]
    depth = camera[..., 2]
    projected = torch.einsum("qij,qpj->qpi", K, camera)
    uv = projected[..., :2] / depth[..., None].clamp_min(1e-8)
    visibility = (
        (depth > 1e-6)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] < hw[:, 1, None])
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] < hw[:, 0, None])
    )
    rays = F.normalize(points[None] - centers[:, None], dim=2)
    for start in range(0, len(pairs), 256):
        end = min(start + 256, len(pairs))
        chunk_left = left[start:end]
        chunk_right = right[start:end]
        joint = visibility[chunk_left] & visibility[chunk_right]
        union = visibility[chunk_left] | visibility[chunk_right]
        joint_count = joint.sum(dim=1)
        result["mapping_point_joint_visibility_count"][start:end] = joint_count
        result["mapping_point_overlap_jaccard"][start:end] = (
            joint_count.double() / union.sum(dim=1).clamp_min(1).double()
        )
        cosine = (rays[chunk_left] * rays[chunk_right]).sum(dim=2).clamp(
            -1.0, 1.0
        )
        angle = torch.rad2deg(torch.acos(cosine)).masked_fill(~joint, torch.nan)
        result["mapping_point_parallax_median_deg"][start:end] = torch.nanmedian(
            angle, dim=1
        ).values
    return result


__all__ = [
    "_camera_centers_and_axes",
    "_camera_pair_geometry_table",
    "candidate_camera_pairs",
    "mapping_scene_points_from_depth_samples",
]
