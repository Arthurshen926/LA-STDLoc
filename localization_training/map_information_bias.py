from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from localization_training.pose_information import (
    effective_sample_size,
    task_scaled_pose_jacobian,
    translation_schur_complement,
)


@dataclass
class MapInformationBiasRisk:
    cleanliness_loss: torch.Tensor
    full_information_loss: torch.Tensor
    translation_information_loss: torch.Tensor
    translation_trace_loss: torch.Tensor
    translation_condition_loss: torch.Tensor
    bias_loss: torch.Tensor
    capacity_loss: torch.Tensor
    full_logdet_gain: torch.Tensor
    translation_logdet_gain: torch.Tensor
    translation_condition: torch.Tensor
    translation_trace_covariance: torch.Tensor
    translation_bias_task: torch.Tensor
    translation_bias_m: torch.Tensor
    expected_match_count: torch.Tensor
    clean_expected_match_count: torch.Tensor
    soft_inlier_expected_match_count: torch.Tensor
    target_match_count: torch.Tensor
    effective_clean_count: torch.Tensor
    effective_soft_inlier_count: torch.Tensor


@dataclass
class DirectionalCandidateSetRisk:
    loss: torch.Tensor
    translation_bias_task: torch.Tensor
    translation_bias_m: torch.Tensor
    score_energy: torch.Tensor
    score_rms: torch.Tensor
    expected_match_count: torch.Tensor
    robust_match_count: torch.Tensor
    target_budget: torch.Tensor
    effective_count: torch.Tensor
    weighted_residual_px: torch.Tensor
    translation_delta_task: torch.Tensor
    normalized_translation_score: torch.Tensor


@dataclass
class CounterfactualPoseSwapUtility:
    weights: torch.Tensor
    valid_mask: torch.Tensor
    bias_reduction_task2: torch.Tensor
    translation_logdet_gain: torch.Tensor
    counterfactual_translation_bias_task: torch.Tensor
    current_translation_bias_task: torch.Tensor
    current_translation_bias_m: torch.Tensor


def _balanced_logit_bce(logits, labels):
    if logits.numel() == 0:
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(
        logits,
        labels.to(dtype=logits.dtype),
        reduction="none",
    )
    positive = labels > 0.5
    parts = []
    if bool(positive.any().item()):
        parts.append(losses[positive].mean())
    if bool((~positive).any().item()):
        parts.append(losses[~positive].mean())
    return torch.stack(parts).mean() if parts else logits.sum() * 0.0


def _spd_logdet(matrix, eps=1e-9):
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    return torch.linalg.eigvalsh(matrix).clamp_min(float(eps)).log().sum(dim=-1)


def _soft_pose_terms(
    jacobian,
    residual,
    valid_mask,
    *,
    translation_scale,
    rotation_scale,
    measurement_sigma_px,
    residual_clip_px,
    inlier_sigma_px,
):
    finite = (
        torch.as_tensor(valid_mask, device=jacobian.device, dtype=torch.bool)
        & torch.isfinite(jacobian).flatten(start_dim=1).all(dim=1)
        & torch.isfinite(residual).all(dim=1)
    )
    jacobian = torch.where(finite[:, None, None], jacobian, torch.zeros_like(jacobian))
    residual = torch.where(finite[:, None], residual, torch.zeros_like(residual))
    scaled_jacobian = task_scaled_pose_jacobian(
        jacobian,
        translation_scale=float(translation_scale),
        rotation_scale=float(rotation_scale),
    )
    residual_norm = torch.linalg.norm(residual, dim=1)
    quality = torch.exp(
        -0.5
        * (residual_norm / max(float(inlier_sigma_px), 1e-4)).square()
    )
    quality = quality * finite.to(dtype=quality.dtype)
    sigma2 = max(float(measurement_sigma_px), 1e-4) ** 2
    information = torch.einsum(
        "n,nai,naj->nij",
        quality / sigma2,
        scaled_jacobian,
        scaled_jacobian,
    )
    residual_scale = (
        max(float(residual_clip_px), 0.0) / residual_norm.clamp_min(1e-8)
    ).clamp_max(1.0)
    clipped_residual = residual * residual_scale[:, None]
    gradient = torch.einsum(
        "n,nai,na->ni",
        quality / sigma2,
        scaled_jacobian,
        clipped_residual,
    )
    return information, gradient


def _indexed_or_zero(values, indices):
    zero = values.new_zeros((1,) + tuple(values.shape[1:]))
    padded = torch.cat([values, zero], dim=0)
    valid = (indices >= 0) & (indices < values.shape[0])
    safe_indices = torch.where(valid, indices, indices.new_full((), values.shape[0]))
    return padded[safe_indices]


def _normalize_positive_utility(values, valid_mask, quantile=0.9):
    positive = values[valid_mask & (values > 0)]
    if positive.numel() == 0:
        return torch.zeros_like(values)
    scale = torch.quantile(positive, float(quantile)).clamp_min(1e-8)
    return (values.clamp_min(0.0) / scale).clamp_max(1.0)


def directional_candidate_set_risk(
    jacobian,
    signed_reprojection_residual,
    candidate_similarity,
    candidate_valid_mask,
    clean_labels,
    dustbin_score,
    *,
    temperature=0.05,
    translation_scale=0.02,
    rotation_scale_degrees=2.0,
    measurement_sigma_px=1.0,
    damping=1e-4,
    residual_clip_px=24.0,
    robust_scale_px=12.0,
    robust_quality_floor=0.01,
    detach_dustbin=True,
):
    """Penalize coherent signed residuals in a sparse final matching matrix.

    Candidate indices and geometry are teacher signals. Each row contains the
    current top-K map candidates plus an optional GT candidate. A dustbin is
    included in the row softmax, while the non-dustbin weights are normalized to
    a fixed per-query GT budget. The normalization prevents minimizing the loss
    by uniformly rejecting every correspondence.
    """
    jacobian = torch.as_tensor(jacobian)
    if jacobian.ndim != 4 or jacobian.shape[-2:] != (2, 6):
        raise ValueError(
            "Directional candidate Jacobians must have shape rows x candidates x 2 x 6"
        )
    rows, candidates = jacobian.shape[:2]
    device = jacobian.device
    dtype = jacobian.dtype
    residual = torch.as_tensor(
        signed_reprojection_residual, device=device, dtype=dtype
    )
    similarity = torch.as_tensor(
        candidate_similarity, device=device, dtype=dtype
    )
    valid_mask = torch.as_tensor(
        candidate_valid_mask, device=device, dtype=torch.bool
    )
    labels = torch.as_tensor(clean_labels, device=device, dtype=torch.bool)
    expected_shape = (rows, candidates)
    if residual.shape != (rows, candidates, 2):
        raise ValueError(
            "Directional candidate residuals must match the Jacobian row/candidate shape"
        )
    if not (
        similarity.shape == valid_mask.shape == labels.shape == expected_shape
    ):
        raise ValueError(
            "Directional candidate similarities, masks, and labels must have "
            "shape rows x candidates"
        )

    zero = similarity.sum() * 0.0
    zero_vector = jacobian.new_zeros(3)
    finite = (
        torch.isfinite(jacobian).reshape(rows, candidates, 12).all(dim=2)
        & torch.isfinite(residual).all(dim=2)
        & torch.isfinite(similarity)
        & valid_mask
    )
    clean_row = (labels & finite).any(dim=1)
    target_budget = clean_row.to(dtype=dtype).sum().detach()
    if candidates == 0 or not bool(finite.any().item()) or target_budget.item() <= 0:
        return DirectionalCandidateSetRisk(
            loss=zero,
            translation_bias_task=zero,
            translation_bias_m=zero,
            score_energy=zero,
            score_rms=zero,
            expected_match_count=zero,
            robust_match_count=zero,
            target_budget=target_budget,
            effective_count=zero,
            weighted_residual_px=zero,
            translation_delta_task=zero_vector,
            normalized_translation_score=zero_vector,
        )

    temperature = max(float(temperature), 1e-4)
    masked_similarity = similarity.masked_fill(
        ~finite, torch.finfo(dtype).min
    )
    dustbin = torch.as_tensor(dustbin_score, device=device, dtype=dtype).reshape(())
    if bool(detach_dustbin):
        dustbin = dustbin.detach()
    row_logits = torch.cat(
        [
            masked_similarity / temperature,
            (dustbin / temperature).expand(rows, 1),
        ],
        dim=1,
    )
    candidate_probability = torch.softmax(row_logits, dim=1)[:, :candidates]
    candidate_probability = candidate_probability * finite.to(dtype=dtype)
    expected_match_count = candidate_probability.sum()

    jacobian = torch.where(
        finite[..., None, None], jacobian.detach(), torch.zeros_like(jacobian)
    )
    residual = torch.where(
        finite[..., None], residual.detach(), torch.zeros_like(residual)
    )
    residual_norm = torch.linalg.norm(residual, dim=2)
    robust_scale_px = max(float(robust_scale_px), 1e-4)
    robust_quality = 1.0 / (
        1.0 + (residual_norm / robust_scale_px).square()
    )
    quality_floor = max(0.0, min(float(robust_quality_floor), 1.0))
    robust_quality = quality_floor + (1.0 - quality_floor) * robust_quality
    robust_quality = robust_quality * finite.to(dtype=dtype)
    robust_probability = candidate_probability * robust_quality
    robust_match_count = robust_probability.sum()

    # Keep a fixed query budget so this loss changes relative candidate scores
    # instead of learning the trivial all-dustbin solution.
    normalized_weight = (
        robust_probability
        / robust_match_count.clamp_min(1e-8)
        * target_budget
    )
    flat_weight = normalized_weight.reshape(-1)
    flat_jacobian = jacobian.reshape(-1, 2, 6)
    flat_residual = residual.reshape(-1, 2)
    rotation_scale = math.radians(float(rotation_scale_degrees))
    scaled_jacobian = task_scaled_pose_jacobian(
        flat_jacobian,
        translation_scale=float(translation_scale),
        rotation_scale=rotation_scale,
    )
    accumulation_dtype = (
        torch.float64
        if dtype in (torch.float16, torch.bfloat16, torch.float32)
        else dtype
    )
    accumulation_weight = flat_weight.to(dtype=accumulation_dtype)
    accumulation_jacobian = scaled_jacobian.to(dtype=accumulation_dtype)
    sigma2 = max(float(measurement_sigma_px), 1e-4) ** 2
    information = (
        torch.eye(6, dtype=accumulation_dtype, device=device) * float(damping)
    )
    information = information + torch.einsum(
        "n,nai,naj->ij",
        accumulation_weight / sigma2,
        accumulation_jacobian,
        accumulation_jacobian,
    )
    information = 0.5 * (information + information.T)

    flat_residual_norm = torch.linalg.norm(flat_residual, dim=1).clamp_min(1e-8)
    residual_scale = (
        max(float(residual_clip_px), 0.0) / flat_residual_norm
    ).clamp_max(1.0)
    clipped_residual = flat_residual * residual_scale[:, None]
    accumulation_residual = clipped_residual.to(dtype=accumulation_dtype)
    gradient = torch.einsum(
        "n,nai,na->i",
        accumulation_weight / sigma2,
        accumulation_jacobian,
        accumulation_residual,
    )

    h_tt = information[:3, :3]
    h_tr = information[:3, 3:]
    h_rr = information[3:, 3:]
    g_t = gradient[:3]
    g_r = gradient[3:]
    solved_rt = torch.linalg.solve(h_rr, h_tr.T)
    solved_gr = torch.linalg.solve(h_rr, g_r)
    translation_information = h_tt - h_tr @ solved_rt
    translation_information = 0.5 * (
        translation_information + translation_information.T
    )
    translation_score = g_t - h_tr @ solved_gr
    translation_delta = -torch.linalg.solve(
        translation_information, translation_score
    )
    solved_score = torch.linalg.solve(
        translation_information, translation_score
    )
    score_energy = torch.dot(translation_score, solved_score).clamp_min(0.0)
    accumulation_budget = target_budget.to(dtype=accumulation_dtype).clamp_min(1.0)
    loss = score_energy / accumulation_budget
    score_rms = torch.sqrt(loss.clamp_min(0.0))
    normalized_score = translation_score / torch.sqrt(
        torch.diagonal(translation_information).clamp_min(1e-9)
        * accumulation_budget
    )
    weighted_residual_px = (
        flat_weight * flat_residual_norm
    ).sum() / flat_weight.sum().clamp_min(1e-8)
    return DirectionalCandidateSetRisk(
        loss=loss.to(dtype=dtype),
        translation_bias_task=torch.linalg.norm(translation_delta).to(dtype=dtype),
        translation_bias_m=(
            torch.linalg.norm(translation_delta).to(dtype=dtype)
            * float(translation_scale)
        ),
        score_energy=score_energy.to(dtype=dtype),
        score_rms=score_rms.to(dtype=dtype),
        expected_match_count=expected_match_count,
        robust_match_count=robust_match_count,
        target_budget=target_budget,
        effective_count=effective_sample_size(flat_weight),
        weighted_residual_px=weighted_residual_px,
        translation_delta_task=translation_delta.to(dtype=dtype),
        normalized_translation_score=normalized_score.to(dtype=dtype),
    )


@torch.no_grad()
def counterfactual_pose_swap_utility(
    current_jacobian,
    current_residual,
    current_valid_mask,
    remove_indices,
    add_jacobian,
    add_residual,
    swap_valid_mask,
    *,
    displaced_indices=None,
    refill_jacobian=None,
    refill_residual=None,
    refill_valid_mask=None,
    translation_scale=0.02,
    rotation_scale_degrees=2.0,
    measurement_sigma_px=1.0,
    damping=1e-4,
    residual_clip_px=12.0,
    inlier_sigma_px=4.0,
    bias_utility_weight=1.0,
    translation_utility_weight=0.0,
    utility_floor=0.1,
    require_positive_bias_gain=False,
    require_nonnegative_translation_gain=False,
):
    """Score exact top1-to-GT swaps against the current quota-limited set.

    The returned weights are detached teacher targets. A swap removes the current
    top1 pair, optionally removes the pair displaced by a per-landmark quota, and
    adds the GT-positive pair. This makes the supervision act on the discrete
    ranking decision used by localization instead of only calibrating its score.
    """
    current_jacobian = torch.as_tensor(current_jacobian).reshape(-1, 2, 6)
    device = current_jacobian.device
    dtype = current_jacobian.dtype
    current_residual = torch.as_tensor(
        current_residual, device=device, dtype=dtype
    ).reshape(-1, 2)
    current_valid_mask = torch.as_tensor(
        current_valid_mask, device=device, dtype=torch.bool
    ).reshape(-1)
    if not (
        current_jacobian.shape[0]
        == current_residual.shape[0]
        == current_valid_mask.shape[0]
    ):
        raise ValueError("Current swap-set tensors must have equal pair counts")

    add_jacobian = torch.as_tensor(
        add_jacobian, device=device, dtype=dtype
    ).reshape(-1, 2, 6)
    add_residual = torch.as_tensor(
        add_residual, device=device, dtype=dtype
    ).reshape(-1, 2)
    remove_indices = torch.as_tensor(
        remove_indices, device=device, dtype=torch.long
    ).reshape(-1)
    swap_valid_mask = torch.as_tensor(
        swap_valid_mask, device=device, dtype=torch.bool
    ).reshape(-1)
    swap_count = int(add_jacobian.shape[0])
    if not (
        add_residual.shape[0]
        == remove_indices.shape[0]
        == swap_valid_mask.shape[0]
        == swap_count
    ):
        raise ValueError("Counterfactual swap tensors must have equal swap counts")
    if displaced_indices is None:
        displaced_indices = remove_indices.new_full((swap_count,), -1)
    else:
        displaced_indices = torch.as_tensor(
            displaced_indices, device=device, dtype=torch.long
        ).reshape(-1)
        if displaced_indices.shape[0] != swap_count:
            raise ValueError("Displaced indices must match the swap count")
    displaced_indices = torch.where(
        displaced_indices == remove_indices,
        displaced_indices.new_full((), -1),
        displaced_indices,
    )
    if refill_jacobian is None:
        refill_jacobian = add_jacobian.new_zeros(add_jacobian.shape)
        refill_residual = add_residual.new_zeros(add_residual.shape)
        refill_valid_mask = torch.zeros_like(swap_valid_mask)
    else:
        refill_jacobian = torch.as_tensor(
            refill_jacobian, device=device, dtype=dtype
        ).reshape(-1, 2, 6)
        refill_residual = torch.as_tensor(
            refill_residual, device=device, dtype=dtype
        ).reshape(-1, 2)
        refill_valid_mask = torch.as_tensor(
            refill_valid_mask, device=device, dtype=torch.bool
        ).reshape(-1)
        if not (
            refill_jacobian.shape[0]
            == refill_residual.shape[0]
            == refill_valid_mask.shape[0]
            == swap_count
        ):
            raise ValueError("Counterfactual refill tensors must match the swap count")

    current_finite = (
        torch.isfinite(current_jacobian).reshape(
            current_jacobian.shape[0], 12
        ).all(dim=1)
        & torch.isfinite(current_residual).all(dim=1)
        & current_valid_mask
    )
    add_finite = (
        torch.isfinite(add_jacobian).reshape(add_jacobian.shape[0], 12).all(dim=1)
        & torch.isfinite(add_residual).all(dim=1)
    )
    valid = swap_valid_mask & add_finite
    rotation_scale = math.radians(float(rotation_scale_degrees))
    current_information, current_gradient = _soft_pose_terms(
        current_jacobian,
        current_residual,
        current_finite,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
        measurement_sigma_px=measurement_sigma_px,
        residual_clip_px=residual_clip_px,
        inlier_sigma_px=inlier_sigma_px,
    )
    add_information, add_gradient = _soft_pose_terms(
        add_jacobian,
        add_residual,
        valid,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
        measurement_sigma_px=measurement_sigma_px,
        residual_clip_px=residual_clip_px,
        inlier_sigma_px=inlier_sigma_px,
    )
    refill_information, refill_gradient = _soft_pose_terms(
        refill_jacobian,
        refill_residual,
        refill_valid_mask,
        translation_scale=translation_scale,
        rotation_scale=rotation_scale,
        measurement_sigma_px=measurement_sigma_px,
        residual_clip_px=residual_clip_px,
        inlier_sigma_px=inlier_sigma_px,
    )
    prior = torch.eye(6, dtype=dtype, device=device) * float(damping)
    full_information = prior + current_information.sum(dim=0)
    full_information = 0.5 * (full_information + full_information.T)
    full_gradient = current_gradient.sum(dim=0)
    current_delta = -torch.linalg.solve(full_information, full_gradient)
    current_bias_task = torch.linalg.norm(current_delta[:3])

    removed_information = _indexed_or_zero(current_information, remove_indices)
    removed_gradient = _indexed_or_zero(current_gradient, remove_indices)
    displaced_information = _indexed_or_zero(
        current_information, displaced_indices
    )
    displaced_gradient = _indexed_or_zero(current_gradient, displaced_indices)
    counterfactual_information = (
        full_information[None]
        - removed_information
        - displaced_information
        + add_information
        + refill_information
    )
    counterfactual_information = 0.5 * (
        counterfactual_information
        + counterfactual_information.transpose(-1, -2)
    )
    counterfactual_gradient = (
        full_gradient[None]
        - removed_gradient
        - displaced_gradient
        + add_gradient
        + refill_gradient
    )
    counterfactual_delta = -torch.linalg.solve(
        counterfactual_information,
        counterfactual_gradient.unsqueeze(-1),
    ).squeeze(-1)
    counterfactual_bias_task = torch.linalg.norm(
        counterfactual_delta[:, :3], dim=1
    )
    bias_reduction_task2 = (
        current_bias_task.square() - counterfactual_bias_task.square()
    )
    current_translation = translation_schur_complement(
        full_information, eps=1e-9
    )
    counterfactual_translation = translation_schur_complement(
        counterfactual_information, eps=1e-9
    )
    translation_logdet_gain = _spd_logdet(
        counterfactual_translation
    ) - _spd_logdet(current_translation)

    utility_valid = valid
    if bool(require_positive_bias_gain):
        utility_valid = utility_valid & (bias_reduction_task2 > 0)
    if bool(require_nonnegative_translation_gain):
        utility_valid = utility_valid & (translation_logdet_gain >= 0)

    normalized_bias = _normalize_positive_utility(
        bias_reduction_task2, utility_valid
    )
    normalized_translation = _normalize_positive_utility(
        translation_logdet_gain, utility_valid
    )
    bias_weight = max(float(bias_utility_weight), 0.0)
    translation_weight = max(float(translation_utility_weight), 0.0)
    utility_weight_sum = bias_weight + translation_weight
    if utility_weight_sum > 0.0:
        normalized_utility = (
            bias_weight * normalized_bias
            + translation_weight * normalized_translation
        ) / utility_weight_sum
    else:
        normalized_utility = torch.zeros_like(normalized_bias)
    floor = max(0.0, min(float(utility_floor), 1.0))
    weights = torch.where(
        utility_valid,
        floor + (1.0 - floor) * normalized_utility,
        torch.zeros_like(normalized_utility),
    )
    return CounterfactualPoseSwapUtility(
        weights=weights,
        valid_mask=utility_valid,
        bias_reduction_task2=bias_reduction_task2,
        translation_logdet_gain=translation_logdet_gain,
        counterfactual_translation_bias_task=counterfactual_bias_task,
        current_translation_bias_task=current_bias_task,
        current_translation_bias_m=(
            current_bias_task * float(translation_scale)
        ),
    )


def map_information_and_bias_risk(
    jacobian,
    signed_reprojection_residual,
    match_logits,
    clean_labels,
    *,
    valid_mask=None,
    classification_valid_mask=None,
    hard_acceptance_mask=None,
    translation_scale=0.02,
    rotation_scale_degrees=2.0,
    measurement_sigma_px=1.0,
    damping=1e-4,
    residual_clip_px=12.0,
    inlier_sigma_px=4.0,
    condition_target=100.0,
    probability_floor=1e-5,
    bias_huber_delta=1.0,
    bias_clip=4.0,
):
    """Differentiable set risk on the actual query/landmark candidate matrix.

    Geometry, candidate indices, labels, and residuals are teacher signals. Gradients
    flow through candidate match probabilities into the localization descriptors.
    """
    jacobian = torch.as_tensor(jacobian).reshape(-1, 2, 6)
    device = jacobian.device
    dtype = jacobian.dtype
    residual = torch.as_tensor(
        signed_reprojection_residual, device=device, dtype=dtype
    ).reshape(-1, 2)
    logits = torch.as_tensor(match_logits, device=device, dtype=dtype).reshape(-1)
    labels = torch.as_tensor(clean_labels, device=device, dtype=dtype).reshape(-1)
    count = int(jacobian.shape[0])
    if not (residual.shape[0] == logits.shape[0] == labels.shape[0] == count):
        raise ValueError("Map information-and-bias inputs must have equal pair counts")

    classification_finite = torch.isfinite(logits) & torch.isfinite(labels)
    if classification_valid_mask is not None:
        classification_finite = classification_finite & torch.as_tensor(
            classification_valid_mask, device=device, dtype=torch.bool
        ).reshape(-1)
    geometry_finite = (
        torch.isfinite(jacobian).all(dim=2).all(dim=1)
        & torch.isfinite(residual).all(dim=1)
        & classification_finite
    )
    if valid_mask is not None:
        geometry_finite = geometry_finite & torch.as_tensor(
            valid_mask, device=device, dtype=torch.bool
        ).reshape(-1)

    hard_acceptance = None
    if hard_acceptance_mask is not None:
        hard_acceptance = torch.as_tensor(
            hard_acceptance_mask, device=device, dtype=torch.bool
        ).reshape(-1)
        if hard_acceptance.numel() != count:
            raise ValueError("hard acceptance mask must match pair count")
    classification_logits = logits[classification_finite]
    classification_labels = labels[classification_finite].detach().clamp(0.0, 1.0)
    jacobian = jacobian[geometry_finite].detach()
    residual = residual[geometry_finite].detach()
    logits = logits[geometry_finite]
    labels = labels[geometry_finite].detach().clamp(0.0, 1.0)
    classification_hard_acceptance = (
        hard_acceptance[classification_finite] if hard_acceptance is not None else None
    )
    geometry_hard_acceptance = (
        hard_acceptance[geometry_finite] if hard_acceptance is not None else None
    )

    zero = match_logits.sum() * 0.0
    prior = torch.eye(6, dtype=dtype, device=device) * float(damping)
    prior_translation = translation_schur_complement(prior, eps=1e-9)
    probability_floor = max(0.0, min(float(probability_floor), 0.49))
    classification_probability = torch.sigmoid(classification_logits)
    classification_probability = probability_floor + (
        1.0 - 2.0 * probability_floor
    ) * classification_probability
    if classification_hard_acceptance is not None:
        hard = classification_hard_acceptance.to(dtype=dtype)
        classification_probability = (
            hard
            + classification_probability
            - classification_probability.detach()
        )
    classification_clean_probability = (
        classification_probability * classification_labels
    )
    expected_count = classification_probability.sum()
    clean_expected_count = classification_clean_probability.sum()
    target_count = classification_labels.sum()
    capacity_scale = classification_probability.new_tensor(
        max(int(classification_probability.numel()), 1)
    )
    capacity_loss = ((expected_count - target_count) / capacity_scale).square()
    cleanliness_loss = _balanced_logit_bce(
        classification_logits, classification_labels
    )
    effective_clean_count = effective_sample_size(
        classification_clean_probability
    )
    if logits.numel() == 0:
        one = zero + 1.0
        return MapInformationBiasRisk(
            cleanliness_loss=cleanliness_loss,
            full_information_loss=zero,
            translation_information_loss=zero,
            translation_trace_loss=zero,
            translation_condition_loss=zero,
            bias_loss=zero,
            capacity_loss=capacity_loss,
            full_logdet_gain=zero,
            translation_logdet_gain=zero,
            translation_condition=one,
            translation_trace_covariance=zero,
            translation_bias_task=zero,
            translation_bias_m=zero,
            expected_match_count=expected_count,
            clean_expected_match_count=clean_expected_count,
            soft_inlier_expected_match_count=zero,
            target_match_count=target_count,
            effective_clean_count=effective_clean_count,
            effective_soft_inlier_count=zero,
        )

    probability = torch.sigmoid(logits)
    probability = probability_floor + (1.0 - 2.0 * probability_floor) * probability
    if geometry_hard_acceptance is not None:
        hard = geometry_hard_acceptance.to(dtype=dtype)
        probability = hard + probability - probability.detach()
    inlier_sigma_px = max(float(inlier_sigma_px), 1e-4)
    soft_inlier_quality = torch.exp(
        -0.5
        * (
            torch.linalg.norm(residual, dim=1) / inlier_sigma_px
        ).square()
    ).detach()
    soft_inlier_probability = probability * soft_inlier_quality
    rotation_scale = math.radians(float(rotation_scale_degrees))
    scaled_jacobian = task_scaled_pose_jacobian(
        jacobian,
        translation_scale=float(translation_scale),
        rotation_scale=rotation_scale,
    )
    sigma = max(float(measurement_sigma_px), 1e-4)
    contributions = torch.einsum(
        "nai,naj->nij",
        scaled_jacobian,
        scaled_jacobian,
    ) / (sigma * sigma)

    clean_information = prior + torch.einsum(
        "n,nij->ij", soft_inlier_probability, contributions
    )
    clean_information = 0.5 * (clean_information + clean_information.T)
    clean_translation = translation_schur_complement(
        clean_information, eps=1e-9
    )
    full_gain = _spd_logdet(clean_information) - _spd_logdet(prior)
    translation_gain = _spd_logdet(clean_translation) - _spd_logdet(
        prior_translation
    )
    translation_eigenvalues = torch.linalg.eigvalsh(
        0.5 * (clean_translation + clean_translation.T)
    ).clamp_min(1e-9)
    translation_condition = (
        translation_eigenvalues[-1] / translation_eigenvalues[0]
    )
    translation_covariance = torch.linalg.pinv(clean_translation)
    translation_trace = torch.trace(translation_covariance).clamp_min(0.0)

    residual_norm = torch.linalg.norm(residual, dim=1, keepdim=True).clamp_min(1e-8)
    residual_scale = (
        max(float(residual_clip_px), 0.0) / residual_norm
    ).clamp_max(1.0)
    clipped_residual = residual * residual_scale
    all_information = prior + torch.einsum(
        "n,nij->ij", soft_inlier_probability, contributions
    )
    all_information = 0.5 * (all_information + all_information.T)
    bias_gradient = torch.einsum(
        "n,nai,na->i",
        soft_inlier_probability / (sigma * sigma),
        scaled_jacobian,
        clipped_residual,
    )
    delta_task = -torch.linalg.solve(all_information, bias_gradient)
    translation_bias_task = torch.linalg.norm(delta_task[:3])
    translation_bias_m = translation_bias_task * float(translation_scale)
    robust_bias = translation_bias_task
    if float(bias_clip) > 0.0:
        clip = robust_bias.new_tensor(float(bias_clip))
        robust_bias = clip * torch.tanh(robust_bias / clip)
    if float(bias_huber_delta) > 0.0:
        bias_loss = 2.0 * F.huber_loss(
            robust_bias,
            torch.zeros_like(robust_bias),
            delta=float(bias_huber_delta),
            reduction="sum",
        )
    else:
        bias_loss = robust_bias.square()

    condition_target = max(float(condition_target), 1.0)
    condition_loss = F.softplus(
        translation_condition.log() - math.log(condition_target)
    )
    return MapInformationBiasRisk(
        cleanliness_loss=cleanliness_loss,
        full_information_loss=-full_gain / 6.0,
        translation_information_loss=-translation_gain / 3.0,
        translation_trace_loss=torch.log1p(translation_trace),
        translation_condition_loss=condition_loss,
        bias_loss=bias_loss,
        capacity_loss=capacity_loss,
        full_logdet_gain=full_gain,
        translation_logdet_gain=translation_gain,
        translation_condition=translation_condition,
        translation_trace_covariance=translation_trace,
        translation_bias_task=translation_bias_task,
        translation_bias_m=translation_bias_m,
        expected_match_count=expected_count,
        clean_expected_match_count=clean_expected_count,
        soft_inlier_expected_match_count=soft_inlier_probability.sum(),
        target_match_count=target_count,
        effective_clean_count=effective_clean_count,
        effective_soft_inlier_count=effective_sample_size(
            soft_inlier_probability
        ),
    )
