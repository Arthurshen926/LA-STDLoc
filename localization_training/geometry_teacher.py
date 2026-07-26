from __future__ import annotations

import torch


def camera_center_bins(pose_w2c: torch.Tensor, bin_count: int) -> torch.Tensor:
    """Assign cameras to deterministic pose-space bins."""
    pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float64)
    camera_centers = -torch.einsum(
        "qji,qj->qi", pose_w2c[:, :3, :3], pose_w2c[:, :3, 3]
    )
    count = int(camera_centers.shape[0])
    bin_count = min(max(int(bin_count), 1), count)
    if count == 0:
        return torch.zeros(0, dtype=torch.long)
    centroid = camera_centers.mean(dim=0)
    first = int(
        torch.argmax((camera_centers - centroid).square().sum(dim=1)).item()
    )
    selected = [first]
    selected_mask = torch.zeros(count, dtype=torch.bool)
    selected_mask[first] = True
    nearest = (camera_centers - camera_centers[first]).square().sum(dim=1)
    for _ in range(1, bin_count):
        index = int(
            torch.argmax(nearest.masked_fill(selected_mask, -torch.inf)).item()
        )
        selected.append(index)
        selected_mask[index] = True
        nearest = torch.minimum(
            nearest,
            (camera_centers - camera_centers[index]).square().sum(dim=1),
        )
    prototypes = camera_centers[torch.as_tensor(selected, dtype=torch.long)]
    return torch.cdist(camera_centers, prototypes).argmin(dim=1)


def _deduplicate_landmark_query(
    landmark_index: torch.Tensor,
    query_index: torch.Tensor,
    confidence: torch.Tensor,
    query_count: int,
) -> torch.Tensor:
    """Keep the highest-confidence observation for each landmark/query pair."""
    key = landmark_index.long() * int(query_count) + query_index.long()
    confidence_order = torch.argsort(
        confidence, descending=True, stable=True
    )
    key_order = torch.argsort(key[confidence_order], stable=True)
    ordered = confidence_order[key_order]
    ordered_key = key[ordered]
    keep = torch.ones(ordered.numel(), dtype=torch.bool)
    if ordered.numel() > 1:
        keep[1:] = ordered_key[1:] != ordered_key[:-1]
    return ordered[keep]


def _camera_rays(
    uv: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    uv = uv.to(dtype=torch.float64)
    K = K.to(dtype=torch.float64)
    pose_w2c = pose_w2c.to(dtype=torch.float64)
    homogeneous = torch.cat(
        (uv, torch.ones((uv.shape[0], 1), dtype=uv.dtype)), dim=1
    )
    direction_camera = torch.linalg.solve(K, homogeneous[:, :, None]).squeeze(2)
    rotation = pose_w2c[:, :3, :3]
    translation = pose_w2c[:, :3, 3]
    center = -torch.einsum("nji,nj->ni", rotation, translation)
    direction = torch.einsum("nji,nj->ni", rotation, direction_camera)
    direction = torch.nn.functional.normalize(direction, dim=1)
    return center, direction


def _weighted_ray_intersection(
    center: torch.Tensor,
    direction: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.eye(3, dtype=center.dtype)
    projector = identity[None] - direction[:, :, None] * direction[:, None, :]
    weighted = projector * weight[:, None, None]
    normal = weighted.sum(dim=0)
    rhs = torch.einsum("nij,nj->i", weighted, center)
    eigenvalues = torch.linalg.eigvalsh(normal)
    if (
        not bool(torch.isfinite(eigenvalues).all())
        or float(eigenvalues[0]) <= 1e-12
    ):
        raise torch.linalg.LinAlgError("Degenerate ray intersection")
    point = torch.linalg.solve(normal, rhs)
    condition = eigenvalues[-1] / eigenvalues[0]
    return point, normal, condition


def _project(
    point: torch.Tensor,
    K: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    point_h = torch.cat((point, point.new_ones(1)))
    camera = torch.einsum("nij,j->ni", pose_w2c, point_h)[:, :3]
    depth = camera[:, 2]
    projected_h = torch.einsum("nij,nj->ni", K, camera)
    uv = projected_h[:, :2] / projected_h[:, 2:3].clamp_min(1e-12)
    return uv, depth


def robust_triangulate_associations(
    *,
    landmark_count: int,
    landmark_index: torch.Tensor,
    query_index: torch.Tensor,
    uv: torch.Tensor,
    confidence: torch.Tensor,
    camera_K: torch.Tensor,
    pose_w2c: torch.Tensor,
    query_bin: torch.Tensor | None = None,
    rendered_depth: torch.Tensor | None = None,
    maximum_observations_per_landmark: int = 32,
    minimum_views: int = 3,
    minimum_view_bins: int = 2,
    huber_delta_px: float = 2.0,
    iterations: int = 3,
    minimum_parallax_deg: float = 1.0,
    maximum_reprojection_px: float = 2.0,
    maximum_condition_number: float = 1e6,
) -> dict[str, torch.Tensor]:
    """Robustly triangulate descriptor-only cross-view landmark associations."""
    previous_thread_count = torch.get_num_threads()
    # Thousands of independent 3x3 decompositions are substantially slower
    # when every call fans out across the host's full OpenMP thread pool.
    torch.set_num_threads(1)
    landmark_count = int(landmark_count)
    landmark_index = torch.as_tensor(landmark_index, dtype=torch.long).cpu()
    query_index = torch.as_tensor(query_index, dtype=torch.long).cpu()
    uv = torch.as_tensor(uv, dtype=torch.float64).cpu()
    confidence = torch.as_tensor(confidence, dtype=torch.float64).cpu()
    camera_K = torch.as_tensor(camera_K, dtype=torch.float64).cpu()
    pose_w2c = torch.as_tensor(pose_w2c, dtype=torch.float64).cpu()
    if query_bin is None:
        query_bin = camera_center_bins(pose_w2c, 8)
    query_bin = torch.as_tensor(query_bin, dtype=torch.long).cpu()
    if rendered_depth is not None:
        rendered_depth = torch.as_tensor(
            rendered_depth, dtype=torch.float64
        ).cpu()
    if not (
        landmark_index.numel()
        == query_index.numel()
        == uv.shape[0]
        == confidence.numel()
    ):
        raise ValueError("Association tensors must have the same leading size")
    if landmark_index.numel() == 0:
        raise ValueError("At least one association is required")
    if int(query_index.max()) >= int(camera_K.shape[0]):
        raise ValueError("query_index exceeds the supplied camera table")

    keep = _deduplicate_landmark_query(
        landmark_index, query_index, confidence, int(camera_K.shape[0])
    )
    landmark_index = landmark_index[keep]
    query_index = query_index[keep]
    uv = uv[keep]
    confidence = confidence[keep]
    if rendered_depth is not None:
        rendered_depth = rendered_depth[keep]
    order = torch.argsort(landmark_index, stable=True)
    landmark_index = landmark_index[order]
    query_index = query_index[order]
    uv = uv[order]
    confidence = confidence[order]
    if rendered_depth is not None:
        rendered_depth = rendered_depth[order]

    triangulated_xyz = torch.full(
        (landmark_count, 3), float("nan"), dtype=torch.float64
    )
    observation_count = torch.zeros(landmark_count, dtype=torch.long)
    distinct_view_count = torch.zeros(landmark_count, dtype=torch.long)
    distinct_view_bin_count = torch.zeros(landmark_count, dtype=torch.long)
    reprojection_median_px = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    reprojection_p90_px = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    parallax_deg = torch.zeros(landmark_count, dtype=torch.float64)
    condition_number = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    covariance_trace = torch.full(
        (landmark_count,), float("inf"), dtype=torch.float64
    )
    rendered_depth_signed_median_m = torch.full(
        (landmark_count,), float("nan"), dtype=torch.float64
    )
    triangulated = torch.zeros(landmark_count, dtype=torch.bool)

    unique_landmarks, counts = torch.unique_consecutive(
        landmark_index, return_counts=True
    )
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    for group, landmark in enumerate(unique_landmarks.tolist()):
        start = int(offsets[group])
        end = int(offsets[group + 1])
        selected = torch.arange(start, end)
        maximum = int(maximum_observations_per_landmark)
        if maximum > 0 and selected.numel() > maximum:
            rank = torch.argsort(
                confidence[selected], descending=True, stable=True
            )
            selected = selected[rank[:maximum]]
        queries = query_index[selected]
        if queries.numel() < int(minimum_views):
            continue
        bins = torch.unique(query_bin[queries])
        distinct_view_count[landmark] = int(queries.numel())
        distinct_view_bin_count[landmark] = int(bins.numel())
        if bins.numel() < int(minimum_view_bins):
            continue
        centers, directions = _camera_rays(
            uv[selected], camera_K[queries], pose_w2c[queries]
        )
        pair_cosine = (directions @ directions.T).clamp(-1.0, 1.0)
        pair_cosine.fill_diagonal_(1.0)
        maximum_angle = torch.rad2deg(torch.acos(pair_cosine.min()))
        parallax_deg[landmark] = maximum_angle
        base_weight = confidence[selected].clamp_min(1e-4)
        base_weight /= base_weight.mean().clamp_min(1e-8)
        weight = base_weight
        try:
            point, normal, condition = _weighted_ray_intersection(
                centers, directions, weight
            )
            for _ in range(max(int(iterations), 1)):
                projected, depth = _project(
                    point, camera_K[queries], pose_w2c[queries]
                )
                residual = torch.linalg.norm(projected - uv[selected], dim=1)
                robust = torch.where(
                    residual <= float(huber_delta_px),
                    torch.ones_like(residual),
                    float(huber_delta_px) / residual.clamp_min(1e-8),
                )
                robust = robust * (depth > 0).to(robust.dtype)
                point, normal, condition = _weighted_ray_intersection(
                    centers, directions, base_weight * robust
                )
        except (RuntimeError, torch.linalg.LinAlgError):
            continue
        projected, depth = _project(
            point, camera_K[queries], pose_w2c[queries]
        )
        residual = torch.linalg.norm(projected - uv[selected], dim=1)
        finite = torch.isfinite(residual) & (depth > 0)
        if int(finite.sum()) < int(minimum_views):
            continue
        residual = residual[finite]
        sigma2 = residual.square().median().clamp_min(1e-8)
        covariance = torch.linalg.inv(normal) * sigma2
        triangulated_xyz[landmark] = point
        observation_count[landmark] = int(finite.sum())
        reprojection_median_px[landmark] = residual.median()
        reprojection_p90_px[landmark] = torch.quantile(residual, 0.9)
        condition_number[landmark] = condition
        covariance_trace[landmark] = torch.trace(covariance)
        if rendered_depth is not None:
            valid_depth = (
                finite
                & torch.isfinite(rendered_depth[selected])
                & (rendered_depth[selected] > 0)
            )
            if bool(valid_depth.any()):
                rendered_depth_signed_median_m[landmark] = (
                    depth[valid_depth] - rendered_depth[selected][valid_depth]
                ).median()
        triangulated[landmark] = True

    high_confidence = (
        triangulated
        & (observation_count >= int(minimum_views))
        & (distinct_view_bin_count >= int(minimum_view_bins))
        & (parallax_deg >= float(minimum_parallax_deg))
        & (reprojection_median_px <= float(maximum_reprojection_px))
        & (condition_number <= float(maximum_condition_number))
    )
    result = {
        "triangulated_xyz": triangulated_xyz.float(),
        "triangulated": triangulated,
        "triangulation_high_confidence": high_confidence,
        "triangulation_observation_count": observation_count,
        "triangulation_distinct_view_count": distinct_view_count,
        "triangulation_distinct_view_bin_count": distinct_view_bin_count,
        "triangulation_reprojection_median_px": reprojection_median_px.float(),
        "triangulation_reprojection_p90_px": reprojection_p90_px.float(),
        "triangulation_parallax_deg": parallax_deg.float(),
        "triangulation_condition_number": condition_number.float(),
        "triangulation_covariance_trace": covariance_trace.float(),
        "triangulation_rendered_depth_signed_median_m": (
            rendered_depth_signed_median_m.float()
        ),
    }
    torch.set_num_threads(previous_thread_count)
    return result
