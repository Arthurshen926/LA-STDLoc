import torch


def _flat_camera_tensor(value, trailing_dims):
    if value is None:
        return None
    return value.reshape(-1, *value.shape[-trailing_dims:]) if trailing_dims else value.reshape(-1)


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
    candidate_k = min(max(int(candidate_topk), k), max(int(global_idx.numel()), 1))
    out_idx = torch.zeros((count, k), device=device, dtype=torch.long)
    out_weight = torch.zeros((count, k), device=device, dtype=dtype)
    out_valid = torch.zeros(count, device=device, dtype=torch.bool)
    if count == 0 or global_idx.numel() == 0:
        return out_idx, out_weight, out_valid

    depth_image = None
    if rendered_depth is not None:
        depth_image = rendered_depth.squeeze().to(device=device, dtype=dtype)

    for start in range(0, count, max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), count)
        xy = keypoint_xy[start:end] + 0.5
        x = xy[:, 0, None, None]
        y = xy[:, 1, None, None]
        matrix = transforms[None]
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
        delta = xy[:, None, :] - means2d[None]
        sigma_3d = u.square() + v.square()
        sigma_2d = 2.0 * delta.square().sum(dim=-1)
        sigma = 0.5 * torch.minimum(sigma_3d, sigma_2d)
        alpha = (opacities[None] * torch.exp(-sigma)).clamp(max=0.999)
        alpha = alpha.masked_fill(~visible[None] | ~finite_plane, 0.0)

        if depth_image is not None and depth_image.dim() == 2:
            px = keypoint_xy[start:end, 0].long().clamp(0, depth_image.shape[1] - 1)
            py = keypoint_xy[start:end, 1].long().clamp(0, depth_image.shape[0] - 1)
            surface_depth = depth_image[py, px]
            tolerance = float(depth_abs_tolerance) + float(depth_rel_tolerance) * surface_depth.abs()
            depth_ok = (depths[None] - surface_depth[:, None]).abs() <= tolerance[:, None]
            depth_ok = depth_ok | ~(surface_depth[:, None] > 0)
            alpha = alpha.masked_fill(~depth_ok, 0.0)

        candidate_alpha, candidate_idx = torch.topk(alpha, candidate_k, dim=1)
        candidate_depth = depths[candidate_idx]
        depth_order = candidate_depth.argsort(dim=1)
        sorted_alpha = candidate_alpha.gather(1, depth_order)
        sorted_idx = candidate_idx.gather(1, depth_order)
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
