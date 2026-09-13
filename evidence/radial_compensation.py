"""Diagnostic correction for an attested extra image undistortion."""
import torch


def undo_extra_undistortion(grid_uv, K, radial):
    """Map output grid locations back to input grid via SIMPLE_RADIAL.

    OpenCV remap uses integer pixel centers. Apply the frontend's +0.5 physical
    coordinate convention only AFTER this image-domain mapping.
    """
    xy = (grid_uv - K[..., :2, 2]) / torch.stack([K[..., 0, 0], K[..., 1, 1]], -1)
    factor = 1 + radial * xy.square().sum(-1)
    return xy * factor[..., None] * torch.stack([K[..., 0, 0], K[..., 1, 1]], -1) + K[..., :2, 2]


def refit_fixed_tracks(xyz, row, uv, K, T, steps=10):
    """Batched fixed-camera Huber-2 Gauss-Newton, no test evidence.

    Same solver for corrected and unchanged observations. Singular/negative
    depth/nonfinite tracks fall back to their original XYZ, never get deleted.
    """
    x = xyz.double().clone()
    row = row.long()
    R, t = T[:, :3, :3].double(), T[:, :3, 3].double()
    K, uv = K.double(), uv.double()
    active = torch.ones(len(x), dtype=torch.bool)
    eye = torch.eye(3, dtype=torch.double)
    for _ in range(steps):
        cam = (R @ x[row, :, None])[..., 0] + t
        z = cam[:, 2].clamp_min(1e-6)
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        pred = torch.stack([fx * cam[:, 0] / z + K[:, 0, 2], fy * cam[:, 1] / z + K[:, 1, 2]], 1)
        residual = pred - uv
        j = torch.zeros((len(row), 2, 3), dtype=torch.double)
        j[:, 0, 0] = fx / z; j[:, 1, 1] = fy / z
        j[:, 0, 2] = -fx * cam[:, 0] / z.square()
        j[:, 1, 2] = -fy * cam[:, 1] / z.square()
        j = j @ R
        weight = (2 / torch.linalg.vector_norm(residual, dim=1).clamp_min(2))
        normal = torch.zeros((len(x), 3, 3), dtype=torch.double)
        rhs = torch.zeros_like(x)
        normal.index_add_(0, row, (j.transpose(1, 2) @ j) * weight[:, None, None])
        rhs.index_add_(0, row, (j.transpose(1, 2) @ residual[..., None])[..., 0] * weight[:, None])
        eig = torch.linalg.eigvalsh(normal)
        good = (eig[:, 0] > 1e-10) & (eig[:, -1] / eig[:, 0].clamp_min(1e-10) < 1e8)
        delta = torch.linalg.solve(normal + eye[None] * 1e-6, rhs[..., None])[..., 0]
        good &= torch.isfinite(delta).all(1)
        active &= good
        x -= torch.where(active[:, None], delta, torch.zeros_like(delta))
    cam = (R @ x[row, :, None])[..., 0] + t
    invalid = torch.zeros(len(x), dtype=torch.long)
    invalid.index_add_(0, row, ((cam[:, 2] <= 0) | ~torch.isfinite(cam).all(1)).long())
    active &= (invalid == 0) & torch.isfinite(x).all(1)
    return torch.where(active[:, None], x, xyz.double()).to(xyz.dtype), active
