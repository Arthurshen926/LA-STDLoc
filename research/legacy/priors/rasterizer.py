from __future__ import annotations

import torch


def _flat_camera_tensor(value, trailing_dims):
    if value is None:
        return None
    return value.reshape(-1, *value.shape[-trailing_dims:]) if trailing_dims else value.reshape(-1)


def anchor_source_csr(
    state: dict,
    track_payload: dict | None = None,
    full_prior_pool: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recover the Gaussian source family supporting every localization anchor."""
    count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    source = torch.as_tensor(state["source_primitive_ids"]).long()
    track_ids = torch.as_tensor(state["track_cluster_ids"]).long()
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    families: list[dict[int, float]] = [
        {int(source[row]): 1.0} for row in range(count)
    ]

    assignment = track_payload.get("assignment") if track_payload is not None else None
    if assignment is not None and {
        "track_landmark_offsets",
        "track_landmark_indices",
    }.issubset(assignment):
        offsets = torch.as_tensor(assignment["track_landmark_offsets"]).long()
        landmark_rows = torch.as_tensor(assignment["track_landmark_indices"]).long()
        primitive_ids = torch.as_tensor(track_payload["landmark_indices"]).long()
        costs = torch.as_tensor(
            assignment.get(
                "track_landmark_costs",
                torch.zeros_like(landmark_rows, dtype=torch.float32),
            )
        ).float()
        for row in torch.nonzero(track_ids >= 0).reshape(-1).tolist():
            track = int(track_ids[row])
            if track + 1 >= offsets.numel():
                continue
            start, end = int(offsets[track]), int(offsets[track + 1])
            if end <= start:
                continue
            weights = torch.softmax(-costs[start:end], dim=0)
            families[row] = {
                int(primitive_ids[int(local)]): float(weight)
                for local, weight in zip(
                    landmark_rows[start:end].tolist(), weights.tolist()
                )
            }

    if full_prior_pool is not None:
        pool_ids = torch.as_tensor(full_prior_pool["anchor_ids"]).long()
        pool_lookup = {
            int(anchor_id): row for row, anchor_id in enumerate(pool_ids.tolist())
        }
        offsets = torch.as_tensor(
            full_prior_pool["full_prior_source_group_offsets"]
        ).long()
        primitive_ids = torch.as_tensor(
            full_prior_pool["full_prior_source_group_primitive_ids"]
        ).long()
        responsibilities = torch.as_tensor(
            full_prior_pool["full_prior_source_group_responsibilities"]
        ).float()
        canonical = int(full_prior_pool["canonical_anchor_count"])
        for row in range(count):
            pool_row = pool_lookup.get(int(anchor_ids[row]))
            if pool_row is None or pool_row < canonical:
                continue
            local = pool_row - canonical
            if local + 1 >= offsets.numel():
                continue
            start, end = int(offsets[local]), int(offsets[local + 1])
            if end <= start:
                continue
            weights = responsibilities[start:end].clamp_min(0)
            weights = weights / weights.sum().clamp_min(1e-8)
            families[row] = {
                int(primitive): float(weight)
                for primitive, weight in zip(
                    primitive_ids[start:end].tolist(), weights.tolist()
                )
            }

    csr_offsets = [0]
    csr_ids = []
    csr_weights = []
    for family in families:
        for primitive, weight in sorted(family.items()):
            csr_ids.append(primitive)
            csr_weights.append(weight)
        csr_offsets.append(len(csr_ids))
    return (
        torch.tensor(csr_offsets, dtype=torch.int64),
        torch.tensor(csr_ids, dtype=torch.int64),
        torch.tensor(csr_weights, dtype=torch.float32),
    )


@torch.no_grad()
def compress_2dgs_rgb_meta_to_bank(rgb_meta, landmark_global_indices):
    """Discard full-map raster metadata after retaining the localization bank."""
    indices = torch.as_tensor(
        landmark_global_indices,
        device=rgb_meta["means2d"].device,
        dtype=torch.long,
    ).reshape(-1)
    radii = rgb_meta.get("radii")
    if radii is None:
        raise ValueError("2DGS provenance metadata is missing radii")
    radii = (
        radii.reshape(-1, radii.shape[-1]).amax(dim=-1)
        if radii.dim() > 2
        else radii.reshape(-1)
    )
    return {
        "means2d": _flat_camera_tensor(rgb_meta.get("means2d"), 1)[indices],
        "depths": _flat_camera_tensor(rgb_meta.get("depths"), 0)[indices],
        "ray_transforms": _flat_camera_tensor(
            rgb_meta.get("ray_transforms"), 2
        )[indices],
        "opacities": _flat_camera_tensor(rgb_meta.get("opacities"), 0)[indices],
        "radii": radii[indices],
        "rendered_depth": rgb_meta.get("rendered_depth"),
    }


@torch.no_grad()
def bank_splat_provenance_2dgs(
    keypoint_xy,
    landmark_global_indices,
    rgb_meta,
    *,
    rendered_depth=None,
    topk=4,
    candidate_topk=32,
    depth_abs_tolerance=0.05,
    depth_rel_tolerance=0.02,
    chunk_size=128,
    prefilter_topk=None,
    return_diagnostics=False,
    minimum_composition_mass=None,
):
    """Compute bank-conditioned 2DGS composition weights at query keypoints.

    The kernel matches gsplat's 2DGS alpha evaluation. Composition is evaluated
    over the map's localization landmark bank, with full-render depth used as an
    occlusion guard for primitives outside that bank.
    """
    means2d = _flat_camera_tensor(rgb_meta.get("means2d"), 1)
    transforms = _flat_camera_tensor(rgb_meta.get("ray_transforms"), 2)
    depths = _flat_camera_tensor(rgb_meta.get("depths"), 0)
    opacities = _flat_camera_tensor(rgb_meta.get("opacities"), 0)
    radii = rgb_meta.get("radii")
    if any(value is None for value in (means2d, transforms, depths, opacities, radii)):
        raise ValueError("2DGS provenance requires means2d/ray_transforms/depths/opacities/radii")
    radii = radii.reshape(-1, radii.shape[-1]).amax(dim=-1) if radii.dim() > 2 else radii.reshape(-1)

    device = keypoint_xy.device
    dtype = keypoint_xy.dtype
    global_idx = torch.as_tensor(
        landmark_global_indices, device=device, dtype=torch.long
    ).reshape(-1)
    means2d = means2d.to(device=device, dtype=dtype)[global_idx]
    transforms = transforms.to(device=device, dtype=dtype)[global_idx]
    depths = depths.to(device=device, dtype=dtype)[global_idx]
    opacities = opacities.to(device=device, dtype=dtype)[global_idx]
    visible = radii.to(device=device)[global_idx] > 0

    count = int(keypoint_xy.shape[0])
    k = min(max(int(topk), 1), max(int(global_idx.numel()), 1))
    full_composition = candidate_topk is None or int(candidate_topk) <= 0
    candidate_k = (
        int(global_idx.numel())
        if full_composition
        else min(max(int(candidate_topk), k), max(int(global_idx.numel()), 1))
    )
    prefilter_k = (
        None
        if prefilter_topk is None
        else min(
            max(int(prefilter_topk), candidate_k),
            max(int(global_idx.numel()), 1),
        )
    )
    out_idx = torch.zeros((count, k), device=device, dtype=torch.long)
    out_weight = torch.zeros((count, k), device=device, dtype=dtype)
    out_valid = torch.zeros(count, device=device, dtype=torch.bool)
    out_retained_mass = torch.zeros(count, device=device, dtype=dtype)
    if count == 0 or global_idx.numel() == 0:
        result = (out_idx, out_weight, out_valid)
        return (
            (*result, {"retained_composition_fraction": out_retained_mass})
            if return_diagnostics
            else result
        )
    if full_composition and prefilter_k is not None:
        raise ValueError("full 2DGS composition cannot use a footprint prefilter")
    if minimum_composition_mass is not None and not 0.0 < float(
        minimum_composition_mass
    ) <= 1.0:
        raise ValueError("minimum composition mass must lie in (0, 1]")
    global_depth_order = depths.argsort() if full_composition else None

    depth_image = None
    if rendered_depth is not None:
        depth_image = rendered_depth.squeeze().to(device=device, dtype=dtype)

    for start in range(0, count, max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), count)
        xy = keypoint_xy[start:end] + 0.5
        if prefilter_k is not None and prefilter_k < global_idx.numel():
            # A conservative raster-footprint screen avoids evaluating the
            # full 2DGS plane equation for every primitive. The screen uses
            # radius-normalized image distance only; exact alpha, depth order,
            # transmittance, and composition are still evaluated below.
            all_delta = xy[:, None, :] - means2d[None]
            footprint_distance = all_delta.square().sum(dim=-1) / (
                radii.to(device=device, dtype=dtype)[global_idx][None]
                .clamp_min(1.0)
                .square()
            )
            footprint_distance = footprint_distance.masked_fill(
                ~visible[None], float("inf")
            )
            prefilter_idx = torch.topk(
                footprint_distance,
                prefilter_k,
                dim=1,
                largest=False,
            ).indices
            local_transforms = transforms[prefilter_idx]
            local_means2d = means2d[prefilter_idx]
            local_depths = depths[prefilter_idx]
            local_opacities = opacities[prefilter_idx]
            local_visible = visible[prefilter_idx]
        else:
            prefilter_idx = torch.arange(
                global_idx.numel(), device=device, dtype=torch.long
            )[None].expand(xy.shape[0], -1)
            local_transforms = transforms[None].expand(xy.shape[0], -1, -1, -1)
            local_means2d = means2d[None].expand(xy.shape[0], -1, -1)
            local_depths = depths[None].expand(xy.shape[0], -1)
            local_opacities = opacities[None].expand(xy.shape[0], -1)
            local_visible = visible[None].expand(xy.shape[0], -1)
        x = xy[:, 0, None, None]
        y = xy[:, 1, None, None]
        matrix = local_transforms
        h_u = -matrix[:, :, 0, :] + matrix[:, :, 2, :] * x
        h_v = -matrix[:, :, 1, :] + matrix[:, :, 2, :] * y
        cross = torch.cross(h_u, h_v, dim=-1)
        denominator = cross[..., 2]
        finite_plane = denominator.abs() > 1e-8
        u = cross[..., 0] / denominator.clamp(min=-1e30, max=1e30).where(
            finite_plane, torch.ones_like(denominator)
        )
        v = cross[..., 1] / denominator.clamp(min=-1e30, max=1e30).where(
            finite_plane, torch.ones_like(denominator)
        )
        delta = xy[:, None, :] - local_means2d
        sigma_3d = u.square() + v.square()
        sigma_2d = 2.0 * delta.square().sum(dim=-1)
        sigma = 0.5 * torch.minimum(sigma_3d, sigma_2d)
        alpha = (local_opacities * torch.exp(-sigma)).clamp(max=0.999)
        alpha = alpha.masked_fill(~local_visible | ~finite_plane, 0.0)

        if depth_image is not None and depth_image.dim() == 2:
            px = keypoint_xy[start:end, 0].long().clamp(0, depth_image.shape[1] - 1)
            py = keypoint_xy[start:end, 1].long().clamp(0, depth_image.shape[0] - 1)
            surface_depth = depth_image[py, px]
            tolerance = float(depth_abs_tolerance) + float(depth_rel_tolerance) * surface_depth.abs()
            depth_ok = (local_depths - surface_depth[:, None]).abs() <= tolerance[:, None]
            depth_ok = depth_ok | ~(surface_depth[:, None] > 0)
            alpha = alpha.masked_fill(~depth_ok, 0.0)

        if full_composition:
            depth_order = global_depth_order
            sorted_alpha = alpha[:, depth_order]
            sorted_idx = prefilter_idx[:, depth_order]
        else:
            candidate_alpha, candidate_idx = torch.topk(alpha, candidate_k, dim=1)
            candidate_depth = local_depths.gather(1, candidate_idx)
            candidate_bank_idx = prefilter_idx.gather(1, candidate_idx)
            depth_order = candidate_depth.argsort(dim=1)
            sorted_alpha = candidate_alpha.gather(1, depth_order)
            sorted_idx = candidate_bank_idx.gather(1, depth_order)
        transmittance = torch.cumprod(
            torch.cat(
                [torch.ones_like(sorted_alpha[:, :1]), 1.0 - sorted_alpha[:, :-1]],
                dim=1,
            ),
            dim=1,
        )
        composition = sorted_alpha * transmittance
        selected_weight, selected_order = torch.topk(composition, k, dim=1)
        selected_idx = sorted_idx.gather(1, selected_order)
        composition_sum = composition.sum(dim=1, keepdim=True)
        if minimum_composition_mass is not None:
            cumulative_before = (
                selected_weight.cumsum(1) - selected_weight
            ) / composition_sum.clamp_min(1e-8)
            selected_weight = selected_weight.masked_fill(
                cumulative_before >= float(minimum_composition_mass), 0.0
            )
        weight_sum = selected_weight.sum(dim=1, keepdim=True)
        retained_mass = weight_sum / composition_sum.clamp_min(1e-8)
        valid = weight_sum[:, 0] > 1e-8
        selected_weight = torch.where(
            valid[:, None],
            selected_weight / weight_sum.clamp_min(1e-8),
            torch.zeros_like(selected_weight),
        )
        out_idx[start:end] = selected_idx
        out_weight[start:end] = selected_weight
        out_valid[start:end] = valid
        out_retained_mass[start:end] = torch.where(
            valid, retained_mass[:, 0].clamp(0.0, 1.0), torch.zeros_like(valid, dtype=dtype)
        )
    result = (out_idx, out_weight, out_valid)
    return (
        (*result, {"retained_composition_fraction": out_retained_mass})
        if return_diagnostics
        else result
    )


@torch.no_grad()
def bank_splat_provenance_3dgs(
    keypoint_xy,
    landmark_global_indices,
    rgb_meta,
    *,
    rendered_depth=None,
    topk=4,
    candidate_topk=32,
    depth_abs_tolerance=0.05,
    depth_rel_tolerance=0.02,
    chunk_size=128,
):
    """Compute bank-conditioned 3DGS composition at query keypoints."""
    means2d = _flat_camera_tensor(rgb_meta.get("means2d"), 1)
    conics = _flat_camera_tensor(rgb_meta.get("conics"), 1)
    depths = _flat_camera_tensor(rgb_meta.get("depths"), 0)
    opacities = _flat_camera_tensor(rgb_meta.get("opacities"), 0)
    radii = rgb_meta.get("radii")
    if any(value is None for value in (means2d, conics, depths, opacities, radii)):
        raise ValueError("3DGS provenance requires means2d/conics/depths/opacities/radii")
    radii = (
        radii.reshape(-1, radii.shape[-1]).amax(dim=-1)
        if radii.dim() > 2
        else radii.reshape(-1)
    )
    device = keypoint_xy.device
    dtype = keypoint_xy.dtype
    global_idx = torch.as_tensor(
        landmark_global_indices, device=device, dtype=torch.long
    ).reshape(-1)
    means2d = means2d.to(device=device, dtype=dtype)[global_idx]
    conics = conics.to(device=device, dtype=dtype)[global_idx]
    depths = depths.to(device=device, dtype=dtype)[global_idx]
    opacities = opacities.to(device=device, dtype=dtype)[global_idx]
    visible = radii.to(device=device)[global_idx] > 0
    count = int(keypoint_xy.shape[0])
    k = min(max(int(topk), 1), max(int(global_idx.numel()), 1))
    candidate_k = min(max(int(candidate_topk), k), max(int(global_idx.numel()), 1))
    out_idx = torch.zeros((count, k), device=device, dtype=torch.long)
    out_weight = torch.zeros((count, k), device=device, dtype=dtype)
    out_valid = torch.zeros(count, device=device, dtype=torch.bool)
    if count == 0 or global_idx.numel() == 0:
        return out_idx, out_weight, out_valid
    depth_image = (
        rendered_depth.squeeze().to(device=device, dtype=dtype)
        if rendered_depth is not None
        else None
    )
    for start in range(0, count, max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), count)
        xy = keypoint_xy[start:end] + 0.5
        delta = xy[:, None, :] - means2d[None]
        exponent = -0.5 * (
            conics[None, :, 0] * delta[:, :, 0].square()
            + 2.0 * conics[None, :, 1] * delta[:, :, 0] * delta[:, :, 1]
            + conics[None, :, 2] * delta[:, :, 1].square()
        )
        alpha = (opacities[None] * torch.exp(exponent.clamp(max=0))).clamp(max=0.999)
        alpha = alpha.masked_fill(~visible[None] | ~torch.isfinite(alpha), 0.0)
        if depth_image is not None and depth_image.dim() == 2:
            px = keypoint_xy[start:end, 0].long().clamp(0, depth_image.shape[1] - 1)
            py = keypoint_xy[start:end, 1].long().clamp(0, depth_image.shape[0] - 1)
            surface_depth = depth_image[py, px]
            tolerance = float(depth_abs_tolerance) + float(depth_rel_tolerance) * surface_depth.abs()
            depth_ok = (depths[None] - surface_depth[:, None]).abs() <= tolerance[:, None]
            depth_ok = depth_ok | ~(surface_depth[:, None] > 0)
            alpha = alpha.masked_fill(~depth_ok, 0.0)
        candidate_alpha, candidate_idx = torch.topk(alpha, candidate_k, dim=1)
        depth_order = depths[candidate_idx].argsort(dim=1)
        sorted_alpha = candidate_alpha.gather(1, depth_order)
        sorted_idx = candidate_idx.gather(1, depth_order)
        transmittance = torch.cumprod(
            torch.cat(
                (torch.ones_like(sorted_alpha[:, :1]), 1.0 - sorted_alpha[:, :-1]),
                dim=1,
            ),
            dim=1,
        )
        composition = sorted_alpha * transmittance
        selected_weight, selected_order = torch.topk(composition, k, dim=1)
        selected_idx = sorted_idx.gather(1, selected_order)
        weight_sum = selected_weight.sum(dim=1, keepdim=True)
        valid = weight_sum[:, 0] > 1e-8
        selected_weight = torch.where(
            valid[:, None],
            selected_weight / weight_sum.clamp_min(1e-8),
            torch.zeros_like(selected_weight),
        )
        out_idx[start:end] = selected_idx
        out_weight[start:end] = selected_weight
        out_valid[start:end] = valid
    return out_idx, out_weight, out_valid
