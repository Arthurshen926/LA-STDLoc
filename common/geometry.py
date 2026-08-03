import torch


def _skew(v):
    z = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack(
        [
            torch.stack([z, -v[2], v[1]]),
            torch.stack([v[2], z, -v[0]]),
            torch.stack([-v[1], v[0], z]),
        ]
    )


def se3_exp(xi):
    """SE(3) exponential map for world-to-camera left updates."""
    xi = xi.reshape(6)
    t = xi[:3]
    w = xi[3:]
    theta = torch.linalg.norm(w)
    W = _skew(w)
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    if theta.item() < 1e-8:
        R = eye + W + 0.5 * (W @ W)
        V = eye + 0.5 * W + (1.0 / 6.0) * (W @ W)
    else:
        A = torch.sin(theta) / theta
        B = (1.0 - torch.cos(theta)) / theta.square()
        C = (theta - torch.sin(theta)) / theta.pow(3)
        R = eye + A * W + B * (W @ W)
        V = eye + B * W + C * (W @ W)
    out = torch.eye(4, dtype=xi.dtype, device=xi.device)
    out[:3, :3] = R
    out[:3, 3] = V @ t
    return out


def project_points(points_world, K, pose_w2c, eps=1e-8):
    if points_world.numel() == 0:
        return points_world.new_zeros((0, 2)), torch.zeros(0, dtype=torch.bool, device=points_world.device)
    ones = torch.ones(points_world.shape[0], 1, dtype=points_world.dtype, device=points_world.device)
    points_h = torch.cat([points_world, ones], dim=1)
    points_cam = (pose_w2c @ points_h.T)[:3].T
    z = points_cam[:, 2]
    valid = z > eps
    uv = torch.empty(points_world.shape[0], 2, dtype=points_world.dtype, device=points_world.device)
    uv[:, 0] = K[0, 0] * points_cam[:, 0] / z.clamp_min(eps) + K[0, 2]
    uv[:, 1] = K[1, 1] * points_cam[:, 1] / z.clamp_min(eps) + K[1, 2]
    return uv, valid
