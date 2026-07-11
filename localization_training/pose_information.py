from dataclasses import dataclass

import torch

from localization_training.pose_refiner import project_points, se3_exp


def pose_jacobian_numeric(points_world, K, pose_w2c, eps=1e-4):
    points_world = points_world.to(dtype=K.dtype, device=K.device)
    pose_w2c = pose_w2c.to(dtype=K.dtype, device=K.device)
    jac = points_world.new_zeros((points_world.shape[0], 2, 6))
    for dim in range(6):
        delta = points_world.new_zeros(6)
        delta[dim] = eps
        plus, _ = project_points(points_world, K, se3_exp(delta) @ pose_w2c)
        minus, _ = project_points(points_world, K, se3_exp(-delta) @ pose_w2c)
        jac[:, :, dim] = (plus - minus) / (2.0 * eps)
    jac[~torch.isfinite(jac)] = 0
    return jac


def pose_jacobian_analytic(points_world, K, pose_w2c):
    """Pixel Jacobian for a left SE(3) update ordered as [t, rotation]."""
    points_world = points_world.to(dtype=K.dtype, device=K.device)
    pose_w2c = pose_w2c.to(dtype=K.dtype, device=K.device)
    ones = torch.ones(
        points_world.shape[0], 1, dtype=points_world.dtype, device=points_world.device
    )
    camera = (pose_w2c @ torch.cat([points_world, ones], dim=1).T)[:3].T
    x, y, z = camera.unbind(dim=1)
    z = z.clamp_min(1e-8)
    fx, fy = K[0, 0], K[1, 1]
    dproj = camera.new_zeros((camera.shape[0], 2, 3))
    dproj[:, 0, 0] = fx / z
    dproj[:, 0, 2] = -fx * x / z.square()
    dproj[:, 1, 1] = fy / z
    dproj[:, 1, 2] = -fy * y / z.square()
    skew = camera.new_zeros((camera.shape[0], 3, 3))
    skew[:, 0, 1] = -camera[:, 2]
    skew[:, 0, 2] = camera[:, 1]
    skew[:, 1, 0] = camera[:, 2]
    skew[:, 1, 2] = -camera[:, 0]
    skew[:, 2, 0] = -camera[:, 1]
    skew[:, 2, 1] = camera[:, 0]
    identity = torch.eye(3, dtype=camera.dtype, device=camera.device)
    camera_jacobian = torch.cat(
        [identity[None].expand(camera.shape[0], -1, -1), -skew], dim=2
    )
    jacobian = dproj @ camera_jacobian
    jacobian[~torch.isfinite(jacobian)] = 0
    return jacobian


@dataclass
class PoseInformation:
    scores: torch.Tensor
    matrix: torch.Tensor
    logdet: torch.Tensor
    condition_number: torch.Tensor


def compute_pose_information(points_world, K, pose_w2c, weights=None, damping=1e-4):
    dtype = points_world.dtype
    device = points_world.device
    K = K.to(device=device, dtype=dtype)
    pose_w2c = pose_w2c.to(device=device, dtype=dtype)
    if weights is None:
        weights = torch.ones(points_world.shape[0], dtype=dtype, device=device)
    weights = weights.to(device=device, dtype=dtype).reshape(-1).clamp_min(0)
    J = pose_jacobian_numeric(points_world, K, pose_w2c)
    H = torch.eye(6, dtype=dtype, device=device) * damping
    H = H + torch.einsum("n,nai,naj->ij", weights, J, J)
    sign, logabsdet = torch.linalg.slogdet(H)
    logdet = torch.where(sign > 0, logabsdet, torch.zeros_like(logabsdet))
    H_inv = torch.linalg.pinv(H)
    eye2 = torch.eye(2, dtype=dtype, device=device)
    scores = []
    for idx in range(points_world.shape[0]):
        Ji = J[idx]
        gain = eye2 + weights[idx] * (Ji @ H_inv @ Ji.T)
        sign_i, logdet_i = torch.linalg.slogdet(gain)
        scores.append(torch.where(sign_i > 0, logdet_i, torch.zeros_like(logdet_i)))
    scores = torch.stack(scores) if scores else points_world.new_zeros(0)
    return PoseInformation(
        scores=scores,
        matrix=H,
        logdet=logdet,
        condition_number=torch.linalg.cond(H),
    )
