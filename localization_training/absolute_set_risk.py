from dataclasses import dataclass

import torch

from localization_training.pose_information import (
    effective_sample_size,
    translation_schur_complement,
)


@dataclass
class AbsolutePoseSetRisk:
    translation_bias_loss: torch.Tensor
    translation_trace_loss: torch.Tensor
    translation_bias_m: torch.Tensor
    translation_covariance_trace_m2: torch.Tensor
    translation_covariance_logdet: torch.Tensor
    condition_number: torch.Tensor
    effective_pair_count: torch.Tensor


def absolute_pose_set_risk(
    jacobian,
    residual_offset,
    cholesky,
    inlier_logits,
    valid_mask=None,
    translation_prior_sigma_m=1.0,
    rotation_prior_sigma_rad=0.5,
    reference_translation_m=None,
    translation_task_scale_m=0.02,
    rotation_task_scale_rad=0.03490658503988659,
):
    """Absolute-scale pose bias and posterior uncertainty for one query set."""
    jacobian = jacobian.reshape(-1, 2, 6)
    residual_offset = residual_offset.reshape(-1, 2)
    cholesky = cholesky.reshape(-1, 2, 2)
    inlier_logits = inlier_logits.reshape(-1)
    if not (
        jacobian.shape[0]
        == residual_offset.shape[0]
        == cholesky.shape[0]
        == inlier_logits.shape[0]
    ):
        raise ValueError("set-risk inputs must contain the same pair count")
    finite = (
        torch.isfinite(jacobian).all(dim=2).all(dim=1)
        & torch.isfinite(residual_offset).all(dim=1)
        & torch.isfinite(cholesky).all(dim=2).all(dim=1)
        & torch.isfinite(inlier_logits)
    )
    if valid_mask is not None:
        finite = finite & torch.as_tensor(
            valid_mask, device=finite.device, dtype=torch.bool
        ).reshape(-1)
    jacobian = jacobian[finite]
    residual_offset = residual_offset[finite]
    cholesky = cholesky[finite]
    probability = torch.sigmoid(inlier_logits[finite]).clamp(1e-6, 1.0)

    dtype = residual_offset.dtype
    device = residual_offset.device
    prior_diagonal = residual_offset.new_tensor(
        [
            1.0 / max(float(translation_prior_sigma_m), 1e-6) ** 2,
            1.0 / max(float(translation_prior_sigma_m), 1e-6) ** 2,
            1.0 / max(float(translation_prior_sigma_m), 1e-6) ** 2,
            1.0 / max(float(rotation_prior_sigma_rad), 1e-6) ** 2,
            1.0 / max(float(rotation_prior_sigma_rad), 1e-6) ** 2,
            1.0 / max(float(rotation_prior_sigma_rad), 1e-6) ** 2,
        ]
    )
    information = torch.diag(prior_diagonal)
    bias = torch.zeros(6, dtype=dtype, device=device)
    if jacobian.shape[0] > 0:
        whitened_jacobian = torch.linalg.solve_triangular(
            cholesky, jacobian, upper=False
        )
        whitened_residual = torch.linalg.solve_triangular(
            cholesky, residual_offset.unsqueeze(-1), upper=False
        ).squeeze(-1)
        information = information + torch.einsum(
            "n,nai,naj->ij",
            probability,
            whitened_jacobian,
            whitened_jacobian,
        )
        bias = torch.einsum(
            "n,nai,na->i",
            probability,
            whitened_jacobian,
            whitened_residual,
        )
    information = 0.5 * (information + information.T)
    posterior = torch.linalg.inv(information)
    delta = posterior @ bias
    translation_information = translation_schur_complement(information)
    translation_covariance = torch.linalg.pinv(translation_information)
    translation_bias_m = torch.linalg.norm(delta[:3])
    translation_trace = torch.trace(translation_covariance)
    reference = max(
        float(
            translation_task_scale_m
            if reference_translation_m is None
            else reference_translation_m
        ),
        1e-6,
    )
    bias_loss = (translation_bias_m / reference).square()
    trace_loss = torch.log1p(translation_trace / (reference * reference))
    sign, logdet = torch.linalg.slogdet(translation_covariance)
    logdet = torch.where(sign > 0, logdet, logdet.new_tensor(float("inf")))
    task_scales = information.new_tensor(
        [
            float(translation_task_scale_m),
            float(translation_task_scale_m),
            float(translation_task_scale_m),
            float(rotation_task_scale_rad),
            float(rotation_task_scale_rad),
            float(rotation_task_scale_rad),
        ]
    )
    task_information = information * task_scales[:, None] * task_scales[None, :]
    eigenvalues = torch.linalg.eigvalsh(task_information).clamp_min(1e-12)
    return AbsolutePoseSetRisk(
        translation_bias_loss=bias_loss,
        translation_trace_loss=trace_loss,
        translation_bias_m=translation_bias_m,
        translation_covariance_trace_m2=translation_trace,
        translation_covariance_logdet=logdet,
        condition_number=eigenvalues[-1] / eigenvalues[0],
        effective_pair_count=effective_sample_size(probability),
    )
