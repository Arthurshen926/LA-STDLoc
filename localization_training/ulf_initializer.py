"""ULF-style sparse sampling and geometry-weighted descriptor fusion helpers.

The functions in this module are deliberately independent of ``Scene`` and
``GaussianModel``.  This keeps the coordinate convention testable and prevents
the ULF initialization path from silently inheriting the deployment feature
pyramid's resize semantics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


PIXEL_CENTER_OFFSET = 0.5


def physical_to_grid_index(uv: torch.Tensor) -> torch.Tensor:
    """Convert physical pixel coordinates to integer-cell grid coordinates."""
    return torch.as_tensor(uv) - PIXEL_CENTER_OFFSET


def grid_index_to_physical(uv: torch.Tensor) -> torch.Tensor:
    """Convert a feature-cell coordinate to the physical pixel center."""
    return torch.as_tensor(uv) + PIXEL_CENTER_OFFSET


def quaternion_to_rotation_matrix(rotation: torch.Tensor, eps: float = 1e-8):
    """Convert scalar-first quaternions to rotation matrices on any device."""
    rotation = torch.as_tensor(rotation)
    if rotation.shape[-1] != 4:
        raise ValueError("rotation must end in four quaternion components")
    q = F.normalize(rotation.float(), dim=-1, eps=eps)
    r, x, y, z = q.unbind(dim=-1)
    matrix = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - r * z),
            2 * (x * z + r * y),
            2 * (x * y + r * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - r * x),
            2 * (x * z - r * y),
            2 * (y * z + r * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    )
    return matrix.reshape(*rotation.shape[:-1], 3, 3).to(dtype=rotation.dtype)


def surface_normals_from_rotation(
    rotation: torch.Tensor,
    scaling: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return a stable normal axis for 2D surfels or 3D Gaussian ellipsoids."""
    matrix = quaternion_to_rotation_matrix(rotation, eps=eps)
    if scaling is not None and torch.as_tensor(scaling).shape[-1] >= 3:
        scaling = torch.as_tensor(scaling, device=matrix.device)
        normal_axis = scaling.argmin(dim=-1)
        normals = matrix.gather(
            2,
            normal_axis[..., None, None].expand(*normal_axis.shape, 3, 1),
        ).squeeze(-1)
    else:
        # A 2DGS primitive's local z-axis is its surface normal.
        normals = matrix[..., :, 2]
    return F.normalize(normals, dim=-1, eps=eps)


def geometry_view_weights(
    xyz: torch.Tensor,
    normals: torch.Tensor,
    camera_center: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ULF-Loc's fronto-parallel geometry fusion weights."""
    xyz = torch.as_tensor(xyz)
    normals = torch.as_tensor(normals, device=xyz.device, dtype=xyz.dtype)
    camera_center = torch.as_tensor(
        camera_center, device=xyz.device, dtype=xyz.dtype
    ).reshape(1, 3)
    view_direction = F.normalize(camera_center - xyz, dim=-1, eps=eps)
    return (normals * view_direction).sum(dim=-1).abs().clamp_min(0.0)


def sample_dense_descriptors_at_image_uv(
    dense_feature_map: torch.Tensor,
    physical_uv: torch.Tensor,
    image_hw,
    *,
    stride: int = 8,
) -> torch.Tensor:
    """Sample a stride-8 descriptor map at physical pixel-center coordinates.

    ``physical_uv`` follows the repository-wide sparse convention: image-grid
    index ``i`` is physically observed at ``i + 0.5``.  Passing
    ``grid_index_to_physical(keypoints)`` therefore reproduces ULF-Loc's
    ``sample_descriptors`` exactly.
    """
    dense_feature_map = torch.as_tensor(dense_feature_map)
    if dense_feature_map.ndim == 3:
        dense_feature_map = dense_feature_map[None]
    if dense_feature_map.ndim != 4 or dense_feature_map.shape[0] != 1:
        raise ValueError("dense_feature_map must have shape [C,H,W] or [1,C,H,W]")
    physical_uv = torch.as_tensor(
        physical_uv,
        device=dense_feature_map.device,
        dtype=dense_feature_map.dtype,
    ).reshape(-1, 2)
    if physical_uv.numel() == 0:
        return dense_feature_map.new_zeros((0, dense_feature_map.shape[1]))
    image_height, image_width = (int(image_hw[0]), int(image_hw[1]))
    dense_height, dense_width = dense_feature_map.shape[-2:]
    expected_height = int(dense_height) * int(stride)
    expected_width = int(dense_width) * int(stride)
    # SuperPoint's score decoder is defined on this effective stride-8 canvas.
    # Inputs in our formal protocol are multiples of eight; retaining the
    # explicit check makes an accidental resize/crop visible in diagnostics.
    if image_height != expected_height or image_width != expected_width:
        image_height, image_width = expected_height, expected_width
    grid = physical_uv.clone()
    grid[:, 0] = 2.0 * grid[:, 0] / float(image_width) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / float(image_height) - 1.0
    sampled = F.grid_sample(
        dense_feature_map,
        grid.view(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return F.normalize(sampled[0, :, :, 0].T, dim=-1)


def nearest_keypoint_distance(
    projected_grid_uv: torch.Tensor,
    keypoint_grid_uv: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Return exact nearest sparse-keypoint distances without a FAISS dependency."""
    projected_grid_uv = torch.as_tensor(projected_grid_uv)
    keypoint_grid_uv = torch.as_tensor(
        keypoint_grid_uv,
        device=projected_grid_uv.device,
        dtype=projected_grid_uv.dtype,
    )
    if projected_grid_uv.numel() == 0:
        return projected_grid_uv.new_zeros((0,))
    if keypoint_grid_uv.numel() == 0:
        return projected_grid_uv.new_full((projected_grid_uv.shape[0],), torch.inf)
    chunk_size = max(int(chunk_size), 1)
    distances = []
    for start in range(0, projected_grid_uv.shape[0], chunk_size):
        block = projected_grid_uv[start : start + chunk_size]
        distances.append(torch.cdist(block.float(), keypoint_grid_uv.float()).amin(dim=1))
    return torch.cat(distances, dim=0).to(dtype=projected_grid_uv.dtype)


def sample_mask_at_grid_uv(valid_mask: torch.Tensor, grid_uv: torch.Tensor) -> torch.Tensor:
    """Nearest-cell mask lookup for feature-grid coordinates."""
    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=grid_uv.device)
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must be two-dimensional")
    grid_uv = torch.as_tensor(grid_uv, device=valid_mask.device)
    height, width = valid_mask.shape
    index = grid_uv.round().long()
    in_bounds = (
        (index[:, 0] >= 0)
        & (index[:, 0] < width)
        & (index[:, 1] >= 0)
        & (index[:, 1] < height)
    )
    result = torch.zeros(index.shape[0], dtype=torch.bool, device=valid_mask.device)
    if bool(in_bounds.any().item()):
        result[in_bounds] = valid_mask[
            index[in_bounds, 1], index[in_bounds, 0]
        ]
    return result
