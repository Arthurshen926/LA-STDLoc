from dataclasses import dataclass

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
    """Small SE(3) exponential map for world-to-camera left updates."""
    xi = xi.reshape(6)
    t = xi[:3]
    w = xi[3:]
    theta = torch.linalg.norm(w)
    W = _skew(w)
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
    if theta.item() < 1e-10:
        R = eye + W
    else:
        A = torch.sin(theta) / theta
        B = (1.0 - torch.cos(theta)) / theta.square()
        R = eye + A * W + B * (W @ W)
    out = torch.eye(4, dtype=xi.dtype, device=xi.device)
    out[:3, :3] = R
    out[:3, 3] = t
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


def reprojection_rmse(points_world, target_uv, K, pose_w2c, weights=None):
    uv, valid = project_points(points_world, K, pose_w2c)
    residual = uv - target_uv
    valid = valid & torch.isfinite(residual).all(dim=1)
    if weights is None:
        weights = torch.ones(points_world.shape[0], dtype=points_world.dtype, device=points_world.device)
    weights = weights.to(dtype=points_world.dtype, device=points_world.device) * valid.to(points_world.dtype)
    mse = (residual.square().sum(dim=1) * weights).sum() / weights.sum().clamp_min(1e-8)
    return torch.sqrt(mse)


def _numeric_pose_jacobian(points_world, K, pose_w2c, eps=1e-4):
    base_uv, _ = project_points(points_world, K, pose_w2c)
    jac = points_world.new_zeros((points_world.shape[0], 2, 6))
    for dim in range(6):
        delta = points_world.new_zeros(6)
        delta[dim] = eps
        plus_uv, _ = project_points(points_world, K, se3_exp(delta) @ pose_w2c)
        minus_uv, _ = project_points(points_world, K, se3_exp(-delta) @ pose_w2c)
        jac[:, :, dim] = (plus_uv - minus_uv) / (2.0 * eps)
    jac[~torch.isfinite(jac)] = 0
    return jac, base_uv


@dataclass
class RefinerInfo:
    iterations: int
    initial_rmse: torch.Tensor
    final_rmse: torch.Tensor
    condition_number: torch.Tensor


def weighted_gauss_newton_refine(
    points_world,
    target_uv,
    K,
    pose_init_w2c,
    weights=None,
    num_iterations=3,
    damping=1e-3,
    detach_points=True,
):
    """Refine a world-to-camera pose with weighted reprojection residuals."""
    points = points_world.detach() if detach_points else points_world
    dtype = points.dtype
    device = points.device
    K = K.to(device=device, dtype=dtype)
    pose = pose_init_w2c.to(device=device, dtype=dtype).clone()
    target_uv = target_uv.to(device=device, dtype=dtype)
    if weights is None:
        weights = torch.ones(points.shape[0], dtype=dtype, device=device)
    weights = weights.to(device=device, dtype=dtype).reshape(-1)

    initial_rmse = reprojection_rmse(points, target_uv, K, pose, weights)
    cond = points.new_tensor(float("inf"))
    for _ in range(num_iterations):
        jac, uv = _numeric_pose_jacobian(points, K, pose)
        residual = (uv - target_uv).reshape(-1)
        valid = torch.isfinite(residual.reshape(-1, 2)).all(dim=1)
        w = weights * valid.to(dtype)
        if w.sum() < 4:
            break
        J = jac.reshape(-1, 6)
        W = w.repeat_interleave(2)
        H = J.T @ (J * W[:, None]) + damping * torch.eye(6, dtype=dtype, device=device)
        b = J.T @ (residual * W)
        cond = torch.linalg.cond(H)
        if not torch.isfinite(cond):
            break
        delta = torch.linalg.solve(H, -b)
        if not torch.isfinite(delta).all():
            break
        pose = se3_exp(delta) @ pose

    final_rmse = reprojection_rmse(points, target_uv, K, pose, weights)
    info = {
        "iterations": num_iterations,
        "initial_rmse": initial_rmse.detach(),
        "final_rmse": final_rmse.detach(),
        "condition_number": cond.detach(),
    }
    return pose, info
