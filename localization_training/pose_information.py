from dataclasses import dataclass
from typing import Optional

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
        raise ValueError(f"Expected Jacobian shape [N, 2, 6], got {tuple(jacobian.shape)}")
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
                raise ValueError(f"Expected {count} scalar variances, got {covariance.numel()}")
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


def _spd_condition_number(matrix, eps=1e-12):
    eigenvalues = _spd_eigenvalues(matrix, eps=eps)
    return eigenvalues[..., -1] / eigenvalues[..., 0]


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


def conditional_delete_loss(full_information, contribution, objective="full", eps=1e-12):
    """Exact logdet loss from deleting one or more Fisher contributions."""
    full_information = torch.as_tensor(full_information)
    contribution = torch.as_tensor(
        contribution,
        device=full_information.device,
        dtype=full_information.dtype,
    )
    without = full_information - contribution
    if objective == "full":
        return _spd_logdet(full_information, eps) - _spd_logdet(without, eps)
    if objective == "translation":
        return _spd_logdet(
            translation_schur_complement(full_information, eps), eps
        ) - _spd_logdet(translation_schur_complement(without, eps), eps)
    raise ValueError(f"Unsupported Fisher objective: {objective}")


def full_set_leverage_scores(full_information, contributions, eps=1e-12):
    """Legacy full-set leverage score; this is not a conditional marginal gain."""
    if contributions.numel() == 0:
        return contributions.new_zeros((0,))
    return conditional_add_gain(
        full_information,
        contributions,
        objective="full",
        eps=eps,
    )


def effective_sample_size(weights, eps=1e-12):
    weights = torch.as_tensor(weights).reshape(-1).clamp_min(0.0)
    if weights.numel() == 0:
        return weights.new_tensor(0.0)
    return weights.sum().square() / weights.square().sum().clamp_min(float(eps))


def normalize_information_scores(scores, floor=0.0, mode="max", eps=1e-8):
    """Map non-negative information scores to stable descriptor-loss weights."""
    scores = torch.as_tensor(scores)
    if scores.numel() == 0:
        return scores.reshape(-1)
    scores = scores.reshape(-1).clamp_min(0.0)
    finite = torch.isfinite(scores)
    normalized = torch.zeros_like(scores)
    if bool(finite.any().item()):
        selected = scores[finite]
        mode = str(mode).strip().lower()
        if mode == "max":
            normalized[finite] = selected / selected.max().clamp_min(float(eps))
        elif mode == "quantile":
            upper = torch.quantile(selected, 0.95).clamp_min(float(eps))
            normalized[finite] = (selected / upper).clamp_max(1.0)
        elif mode == "rank":
            if selected.numel() == 1:
                normalized[finite] = 1.0
            else:
                order = torch.argsort(selected, stable=True)
                ranks = torch.empty_like(order, dtype=scores.dtype)
                ranks[order] = torch.arange(
                    selected.numel(), device=scores.device, dtype=scores.dtype
                )
                normalized[finite] = ranks / float(selected.numel() - 1)
        else:
            raise ValueError(f"Unsupported information score normalization: {mode}")
    floor = max(0.0, min(float(floor), 1.0))
    if floor > 0.0:
        normalized = floor + (1.0 - floor) * normalized
    return normalized.clamp(0.0, 1.0)


@dataclass
class PoseInformation:
    # `scores` is the exact leave-one-out full logdet deletion loss.
    scores: torch.Tensor
    matrix: torch.Tensor
    logdet: torch.Tensor
    condition_number: torch.Tensor
    full_set_leverage_scores: Optional[torch.Tensor] = None
    translation_scores: Optional[torch.Tensor] = None
    translation_matrix: Optional[torch.Tensor] = None
    translation_logdet: Optional[torch.Tensor] = None
    translation_condition_number: Optional[torch.Tensor] = None
    translation_min_eigenvalue: Optional[torch.Tensor] = None
    translation_trace_covariance: Optional[torch.Tensor] = None
    translation_worst_std: Optional[torch.Tensor] = None
    effective_count: Optional[torch.Tensor] = None
    contributions: Optional[torch.Tensor] = None


def compute_pose_information(
    points_world,
    K,
    pose_w2c,
    weights=None,
    damping=1e-4,
    measurement_covariance=None,
    translation_scale=1.0,
    rotation_scale=1.0,
    use_analytic_jacobian=True,
    eps=1e-12,
):
    """Compute query-set Fisher diagnostics and exact leave-one-out gains."""
    dtype = points_world.dtype
    device = points_world.device
    K = K.to(device=device, dtype=dtype)
    pose_w2c = pose_w2c.to(device=device, dtype=dtype)
    if weights is None:
        weights = torch.ones(points_world.shape[0], dtype=dtype, device=device)
    weights = weights.to(device=device, dtype=dtype).reshape(-1).clamp_min(0)
    if use_analytic_jacobian:
        jacobian = pose_jacobian_analytic(points_world, K, pose_w2c)
    else:
        jacobian = pose_jacobian_numeric(points_world, K, pose_w2c)
    jacobian = task_scaled_pose_jacobian(
        jacobian,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
    )
    contributions = fisher_contributions(
        jacobian,
        weights=weights,
        measurement_covariance=measurement_covariance,
        eps=max(float(eps), 1e-12),
    )
    prior = torch.eye(6, dtype=dtype, device=device) * float(damping)
    H = prior + contributions.sum(dim=0)
    H = 0.5 * (H + H.T)

    if contributions.shape[0] > 0:
        full_scores = conditional_delete_loss(
            H[None].expand(contributions.shape[0], -1, -1),
            contributions,
            objective="full",
            eps=eps,
        ).clamp_min(0.0)
        translation_scores = conditional_delete_loss(
            H[None].expand(contributions.shape[0], -1, -1),
            contributions,
            objective="translation",
            eps=eps,
        ).clamp_min(0.0)
        leverage = full_set_leverage_scores(H, contributions, eps=eps).clamp_min(0.0)
    else:
        full_scores = points_world.new_zeros((0,))
        translation_scores = points_world.new_zeros((0,))
        leverage = points_world.new_zeros((0,))

    translation = translation_schur_complement(H, eps=max(float(eps), 1e-12))
    translation_eigenvalues = _spd_eigenvalues(translation, eps=eps)
    translation_covariance = torch.linalg.pinv(translation)
    translation_covariance_eigenvalues = _spd_eigenvalues(
        translation_covariance, eps=eps
    )
    return PoseInformation(
        scores=full_scores,
        matrix=H,
        logdet=_spd_logdet(H, eps=eps),
        condition_number=_spd_condition_number(H, eps=eps),
        full_set_leverage_scores=leverage,
        translation_scores=translation_scores,
        translation_matrix=translation,
        translation_logdet=_spd_logdet(translation, eps=eps),
        translation_condition_number=_spd_condition_number(translation, eps=eps),
        translation_min_eigenvalue=translation_eigenvalues[0],
        translation_trace_covariance=torch.trace(translation_covariance),
        translation_worst_std=translation_covariance_eigenvalues[-1].sqrt(),
        effective_count=effective_sample_size(weights, eps=eps),
        contributions=contributions,
    )
