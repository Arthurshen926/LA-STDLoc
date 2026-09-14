"""Frozen mainline kernel extracted from topology.pose_information."""

from __future__ import annotations


import torch


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


def task_scaled_pose_jacobian(
    jacobian,
    translation_scale=1.0,
    rotation_scale=1.0,
):
    """Express the pose Jacobian in dimensionless task-error coordinates."""
    scales = jacobian.new_tensor(
        [
            float(translation_scale),
            float(translation_scale),
            float(translation_scale),
            float(rotation_scale),
            float(rotation_scale),
            float(rotation_scale),
        ]
    )
    return jacobian * scales


def fisher_contributions(jacobian, weights=None, measurement_covariance=None, eps=1e-8):
    """Return one 6x6 Fisher contribution per 2D observation."""
    if jacobian.ndim != 3 or jacobian.shape[1:] != (2, 6):
        raise ValueError(
            f"Expected Jacobian shape [N, 2, 6], got {tuple(jacobian.shape)}"
        )
    count = int(jacobian.shape[0])
    if weights is None:
        weights = jacobian.new_ones((count,))
    else:
        weights = torch.as_tensor(weights, device=jacobian.device, dtype=jacobian.dtype)
        weights = weights.reshape(-1)
        if weights.numel() != count:
            raise ValueError(f"Expected {count} Fisher weights, got {weights.numel()}")
        weights = weights.clamp_min(0.0)

    if measurement_covariance is None:
        precision = torch.eye(2, dtype=jacobian.dtype, device=jacobian.device)
        precision = precision[None].expand(count, -1, -1)
    else:
        covariance = torch.as_tensor(
            measurement_covariance,
            device=jacobian.device,
            dtype=jacobian.dtype,
        )
        if covariance.ndim == 0:
            covariance = covariance.expand(count)
        if covariance.ndim == 1:
            if covariance.numel() != count:
                raise ValueError(
                    f"Expected {count} scalar variances, got {covariance.numel()}"
                )
            inverse = covariance.clamp_min(float(eps)).reciprocal()
            precision = torch.diag_embed(inverse[:, None].expand(-1, 2))
        elif covariance.ndim == 2 and covariance.shape == (count, 2):
            precision = torch.diag_embed(covariance.clamp_min(float(eps)).reciprocal())
        elif covariance.ndim == 3 and covariance.shape == (count, 2, 2):
            eye = torch.eye(2, dtype=jacobian.dtype, device=jacobian.device)
            precision = torch.linalg.pinv(covariance + float(eps) * eye)
        else:
            raise ValueError(
                "measurement_covariance must be scalar, [N], [N, 2], or [N, 2, 2], "
                f"got {tuple(covariance.shape)}"
            )
    precision = precision * weights[:, None, None]
    contributions = jacobian.transpose(1, 2) @ precision @ jacobian
    return 0.5 * (contributions + contributions.transpose(1, 2))


def translation_schur_complement(information_matrix, eps=1e-8):
    """Information on translation after conditioning/marginalizing rotation."""
    matrix = 0.5 * (information_matrix + information_matrix.transpose(-1, -2))
    H_tt = matrix[..., :3, :3]
    H_tr = matrix[..., :3, 3:]
    H_rt = matrix[..., 3:, :3]
    H_rr = matrix[..., 3:, 3:]
    eye = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
    solution = torch.linalg.solve(H_rr + float(eps) * eye, H_rt)
    schur = H_tt - H_tr @ solution
    return 0.5 * (schur + schur.transpose(-1, -2))


def _spd_eigenvalues(matrix, eps=1e-12):
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    return torch.linalg.eigvalsh(matrix).clamp_min(float(eps))


def _spd_logdet(matrix, eps=1e-12):
    return _spd_eigenvalues(matrix, eps=eps).log().sum(dim=-1)


def conditional_add_gain(base_information, contribution, objective="full", eps=1e-12):
    """Exact logdet gain from adding one or more Fisher contributions."""
    base_information = torch.as_tensor(base_information)
    contribution = torch.as_tensor(
        contribution,
        device=base_information.device,
        dtype=base_information.dtype,
    )
    if objective == "full":
        return _spd_logdet(base_information + contribution, eps) - _spd_logdet(
            base_information, eps
        )
    if objective == "translation":
        return _spd_logdet(
            translation_schur_complement(base_information + contribution, eps), eps
        ) - _spd_logdet(translation_schur_complement(base_information, eps), eps)
    raise ValueError(f"Unsupported Fisher objective: {objective}")
