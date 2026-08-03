import torch


def _as_homogeneous(points):
    ones = torch.ones(*points.shape[:-1], 1, dtype=points.dtype, device=points.device)
    return torch.cat([points, ones], dim=-1)


def unproject_pixels(uv, depth, K, pose_w2c):
    """Unproject image pixels with depth from a camera pose into world points."""
    if uv.numel() == 0:
        return uv.new_zeros((0, 3))
    uv = uv.to(dtype=K.dtype, device=K.device)
    depth = depth.to(dtype=K.dtype, device=K.device).reshape(-1)
    pixels_h = torch.cat([uv, torch.ones(uv.shape[0], 1, dtype=K.dtype, device=K.device)], dim=1)
    rays = torch.linalg.solve(K, pixels_h.T).T
    points_cam = rays * depth[:, None]
    points_world_h = torch.linalg.inv(pose_w2c) @ _as_homogeneous(points_cam).T
    return points_world_h[:3].T


def project_world_to_pixels(points_world, K, pose_w2c, eps=1e-8):
    """Project world points into pixel coordinates for a world-to-camera pose."""
    if points_world.numel() == 0:
        return points_world.new_zeros((0, 2)), torch.zeros(0, dtype=torch.bool, device=points_world.device)
    points_world = points_world.to(dtype=K.dtype, device=K.device)
    points_cam = (pose_w2c @ _as_homogeneous(points_world).T)[:3].T
    z = points_cam[:, 2]
    valid = z > eps
    xy = torch.empty(points_world.shape[0], 2, dtype=K.dtype, device=K.device)
    xy[:, 0] = K[0, 0] * points_cam[:, 0] / z.clamp_min(eps) + K[0, 2]
    xy[:, 1] = K[1, 1] * points_cam[:, 1] / z.clamp_min(eps) + K[1, 2]
    return xy, valid


def build_target_correspondences(
    render_uv,
    render_depth,
    K,
    pose_init_w2c,
    pose_gt_w2c,
    pixel_center_offset=0.0,
):
    """Map rendered pixels from an initial pose to target feature-grid pixels.

    The localization evaluator treats an integer feature-grid coordinate as a
    pixel-cell index and performs PnP/lifting at ``index + 0.5``.  A dense
    teacher that perturbs the render pose must use the same convention:
    unproject at the source cell center, then convert the projected physical
    coordinate back to a feature-grid index.  The zero-offset default keeps
    legacy callers byte-for-byte compatible.
    """
    offset = torch.as_tensor(
        pixel_center_offset,
        dtype=render_uv.dtype,
        device=render_uv.device,
    )
    points_world = unproject_pixels(render_uv + offset, render_depth, K, pose_init_w2c)
    target_uv, valid = project_world_to_pixels(points_world, K, pose_gt_w2c)
    target_uv = target_uv - offset
    valid = valid & torch.isfinite(target_uv).all(dim=1) & torch.isfinite(points_world).all(dim=1)
    return {
        "points_world": points_world,
        "target_uv": target_uv,
        "valid": valid,
    }


def bilinear_sample_features(feature_map, uv):
    """Sample a CxHxW feature map at pixel-space uv coordinates."""
    if uv.numel() == 0:
        return feature_map.new_zeros((0, feature_map.shape[0]))
    c, h, w = feature_map.shape
    x = uv[:, 0] / max(w - 1, 1) * 2.0 - 1.0
    y = uv[:, 1] / max(h - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = torch.nn.functional.grid_sample(
        feature_map[None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0].T.reshape(-1, c)


def make_pixel_grid(height, width, device=None, dtype=torch.float32):
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)


def sample_valid_anchors(valid_mask, max_anchors, min_spacing=1):
    """Deterministically sample approximately uniform anchors from a boolean mask."""
    flat_idx = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).squeeze(1)
    if flat_idx.numel() <= max_anchors:
        return flat_idx
    step = max(flat_idx.numel() // max_anchors, min_spacing)
    sampled = flat_idx[::step][:max_anchors]
    if sampled.numel() < max_anchors:
        sampled = flat_idx[torch.linspace(0, flat_idx.numel() - 1, max_anchors, device=flat_idx.device).long()]
    return sampled
