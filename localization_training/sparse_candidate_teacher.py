from dataclasses import dataclass

import torch
import torch.nn.functional as F

from localization_training.correspondence import project_world_to_pixels
from localization_training.absolute_set_risk import absolute_pose_set_risk
from localization_training.pair_measurement import (
    PairMeasurementOutput,
    build_pair_geometry_features,
    gaussian_measurement_nll,
    sample_local_correlation_patch,
)
from localization_training.pose_information import (
    pose_jacobian_analytic,
    pose_jacobian_numeric,
)
from localization_training.sparse_frontend import (
    build_pair_context_features,
    build_score_matrix,
    match_score_matrix,
    select_keypoints,
)


@dataclass
class SparseCandidateBatch:
    keypoint_ids: torch.Tensor
    keypoint_xy: torch.Tensor
    detector_scores: torch.Tensor
    matching_detector_scores: torch.Tensor
    detector_targets: torch.Tensor
    detector_loss_weights: torch.Tensor
    detector_offset_predictions: torch.Tensor
    detector_offset_targets: torch.Tensor
    detector_offset_valid_mask: torch.Tensor
    pair_keypoint_idx: torch.Tensor
    pair_landmark_idx: torch.Tensor
    pair_logits: torch.Tensor
    pair_labels: torch.Tensor
    pair_valid_mask: torch.Tensor
    pair_reprojection_error: torch.Tensor
    pair_scorer_features: torch.Tensor
    pair_scorer_logits: torch.Tensor
    matcher_assignment_logits: torch.Tensor
    pair_scorer_keypoint_idx: torch.Tensor
    pair_scorer_landmark_idx: torch.Tensor
    pair_scorer_labels: torch.Tensor
    pair_scorer_valid_mask: torch.Tensor
    pair_scorer_reprojection_error: torch.Tensor
    pair_scorer_positive_jacobian: torch.Tensor
    pair_measurement_inlier_logits: torch.Tensor
    pair_measurement_offsets: torch.Tensor
    pair_measurement_cholesky: torch.Tensor
    pair_measurement_target_offsets: torch.Tensor
    pair_measurement_jacobian: torch.Tensor
    pair_measurement_geometry_valid_mask: torch.Tensor
    hard_negative_logits: torch.Tensor
    assignment_positive_similarity: torch.Tensor
    assignment_negative_similarity: torch.Tensor
    assignment_negative_mask: torch.Tensor
    multi_positive_similarity: torch.Tensor
    multi_positive_mask: torch.Tensor
    multi_negative_similarity: torch.Tensor
    multi_negative_mask: torch.Tensor
    multi_row_has_positive: torch.Tensor
    dustbin_score: torch.Tensor
    positive_pose_scores: torch.Tensor
    positive_grid_ids: torch.Tensor
    positive_depth_ids: torch.Tensor
    diagnostics: dict


@dataclass
class SparseCandidateLosses:
    pair: torch.Tensor
    hard_negative: torch.Tensor
    assignment: torch.Tensor
    dustbin_assignment: torch.Tensor
    matcher_assignment: torch.Tensor
    matcher_reprojection_assignment: torch.Tensor
    pair_scorer: torch.Tensor
    pair_scorer_assignment: torch.Tensor
    pair_measurement_inlier: torch.Tensor
    pair_measurement_nll: torch.Tensor
    pair_measurement_translation_bias: torch.Tensor
    pair_measurement_translation_covariance: torch.Tensor
    matcher_translation_info: torch.Tensor
    translation_info: torch.Tensor
    detector_match: torch.Tensor
    detector_offset: torch.Tensor
    geometry_set: torch.Tensor
    coverage: torch.Tensor


def _zero(reference):
    return reference.sum() * 0.0


def _match_logits(similarity, temperature=0.1, margin=0.5):
    return (similarity - float(margin)) / max(float(temperature), 1e-6)


def _balanced_focal_bce(logits, labels, gamma=2.0, valid_mask=None):
    if logits.numel() == 0:
        return _zero(logits)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
        logits = logits[valid_mask]
        labels = labels[valid_mask]
    if logits.numel() == 0:
        return _zero(logits)
    labels = labels.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * labels + (1.0 - prob) * (1.0 - labels)
    focal = (1.0 - pt).pow(float(gamma)) * bce
    positive = labels > 0.5
    negative = ~positive
    parts = []
    if bool(positive.any().item()):
        parts.append(focal[positive].mean())
    if bool(negative.any().item()):
        parts.append(focal[negative].mean())
    return torch.stack(parts).mean() if parts else _zero(logits)


def _balanced_probability_bce(probability, target, weights=None):
    if probability.numel() == 0:
        return _zero(probability)
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    target = target.to(dtype=probability.dtype).clamp(0.0, 1.0)
    loss = F.binary_cross_entropy(probability, target, reduction="none")
    weights = (
        torch.ones_like(loss)
        if weights is None
        else weights.to(device=loss.device, dtype=loss.dtype)
    )
    positive = target > 0.0
    negative = ~positive
    parts = []
    if bool(positive.any().item()):
        parts.append(
            (loss[positive] * weights[positive]).sum()
            / weights[positive].sum().clamp_min(1e-8)
        )
    if bool(negative.any().item()):
        parts.append(
            (loss[negative] * weights[negative]).sum()
            / weights[negative].sum().clamp_min(1e-8)
        )
    return torch.stack(parts).mean() if parts else _zero(probability)


def _row_assignment_loss(
    positive_similarity,
    negative_similarity,
    negative_mask,
    *,
    temperature=0.05,
    margin=0.05,
):
    """Rank one GT-valid landmark above the hardest false matches in each query row."""
    if positive_similarity.numel() == 0 or negative_similarity.numel() == 0:
        return _zero(positive_similarity)
    if negative_similarity.shape != negative_mask.shape:
        raise ValueError("assignment negative similarities and mask must have identical shapes")
    if negative_similarity.shape[0] != positive_similarity.shape[0]:
        raise ValueError("assignment positives and negatives must have the same row count")

    valid_row = negative_mask.any(dim=1)
    if not bool(valid_row.any().item()):
        return _zero(positive_similarity)

    temperature = max(float(temperature), 1e-6)
    positive_logit = (positive_similarity[valid_row] - float(margin)) / temperature
    negative_logit = negative_similarity[valid_row] / temperature
    valid_negative = negative_mask[valid_row]
    negative_logit = negative_logit.masked_fill(
        ~valid_negative,
        torch.finfo(negative_logit.dtype).min,
    )
    logits = torch.cat([positive_logit[:, None], negative_logit], dim=1)
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


def _grouped_multi_positive_assignment_loss(
    logits,
    labels,
    valid_mask,
    group_ids,
):
    """Rank any valid positive above competing hypotheses for the same keypoint."""
    if logits.numel() == 0:
        return _zero(logits)
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    group_ids = group_ids.to(device=logits.device, dtype=torch.long)
    if logits.shape != labels.shape or logits.shape != valid_mask.shape:
        raise ValueError("scorer logits, labels, and valid mask must have identical shapes")
    if group_ids.shape != logits.shape:
        raise ValueError("scorer group ids must match scorer logits")
    if not bool(valid_mask.any()):
        return _zero(logits)

    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask] > 0.5
    valid_groups = group_ids[valid_mask]
    group_count = int(valid_groups.max().item()) + 1
    group_max = valid_logits.new_full((group_count,), -torch.inf)
    group_max.scatter_reduce_(
        0,
        valid_groups,
        valid_logits,
        reduce="amax",
        include_self=True,
    )
    stabilized = torch.exp(valid_logits - group_max[valid_groups])
    denominator = valid_logits.new_zeros(group_count)
    denominator.scatter_add_(0, valid_groups, stabilized)
    positive_mass = valid_logits.new_zeros(group_count)
    if bool(valid_labels.any()):
        positive_mass.scatter_add_(
            0,
            valid_groups[valid_labels],
            stabilized[valid_labels],
        )
    positive_row = positive_mass > 0
    if not bool(positive_row.any()):
        return _zero(logits)
    return (
        torch.log(denominator[positive_row].clamp_min(1e-8))
        - torch.log(positive_mass[positive_row].clamp_min(1e-8))
    ).mean()


def _grouped_reprojection_assignment_loss(
    logits,
    labels,
    reprojection_error,
    valid_mask,
    group_ids,
    *,
    sigma_px=1.0,
):
    """Rank hypotheses using continuous GT reprojection quality within each row."""
    if logits.numel() == 0:
        return _zero(logits)
    if not (
        logits.shape
        == labels.shape
        == reprojection_error.shape
        == valid_mask.shape
        == group_ids.shape
    ):
        raise ValueError(
            "reprojection assignment logits, labels, errors, masks, and groups "
            "must have identical shapes"
        )
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    valid_mask = valid_mask & torch.isfinite(reprojection_error)
    if not bool(valid_mask.any()):
        return _zero(logits)

    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask] > 0.5
    valid_errors = reprojection_error[valid_mask].to(dtype=logits.dtype)
    valid_groups = group_ids[valid_mask].to(device=logits.device, dtype=torch.long)
    if not bool(valid_labels.any()):
        return _zero(logits)
    group_count = int(valid_groups.max().item()) + 1

    group_max = valid_logits.new_full((group_count,), -torch.inf)
    group_max.scatter_reduce_(
        0,
        valid_groups,
        valid_logits,
        reduce="amax",
        include_self=True,
    )
    stabilized = torch.exp(valid_logits - group_max[valid_groups])
    denominator = valid_logits.new_zeros(group_count)
    denominator.scatter_add_(0, valid_groups, stabilized)
    log_denominator = group_max + torch.log(denominator.clamp_min(1e-8))

    sigma_px = max(float(sigma_px), 1e-6)
    positive_quality = torch.exp(
        -0.5 * (valid_errors[valid_labels] / sigma_px).square()
    )
    positive_groups = valid_groups[valid_labels]
    quality_sum = valid_logits.new_zeros(group_count)
    quality_sum.scatter_add_(0, positive_groups, positive_quality)
    target_expectation = valid_logits.new_zeros(group_count)
    target_expectation.scatter_add_(
        0,
        positive_groups,
        positive_quality * valid_logits[valid_labels],
    )
    row_quality = valid_logits.new_zeros(group_count)
    row_quality.scatter_reduce_(
        0,
        positive_groups,
        positive_quality,
        reduce="amax",
        include_self=True,
    )
    positive_row = quality_sum > 0
    if not bool(positive_row.any()):
        return _zero(logits)
    target_expectation = target_expectation[positive_row] / quality_sum[
        positive_row
    ].clamp_min(1e-8)
    row_loss = log_denominator[positive_row] - target_expectation
    row_weight = row_quality[positive_row].clamp_min(1e-4)
    return (row_loss * row_weight).sum() / row_weight.sum().clamp_min(1e-8)


def _multi_positive_dustbin_loss(
    positive_similarity,
    positive_mask,
    negative_similarity,
    negative_mask,
    dustbin_score,
    *,
    temperature=0.05,
    margin=0.05,
):
    """Assign each query row to its geometric positive set or to unmatched."""
    if positive_similarity.shape != positive_mask.shape:
        raise ValueError("multi-positive similarities and mask must have identical shapes")
    if negative_similarity.shape != negative_mask.shape:
        raise ValueError("multi-negative similarities and mask must have identical shapes")
    if positive_similarity.shape[0] != negative_similarity.shape[0]:
        raise ValueError("multi-positive and multi-negative tensors must have the same row count")
    if positive_similarity.shape[0] == 0:
        return _zero(dustbin_score)

    temperature = max(float(temperature), 1e-6)
    minimum = torch.finfo(positive_similarity.dtype).min
    positive_logits = ((positive_similarity - float(margin)) / temperature).masked_fill(
        ~positive_mask,
        minimum,
    )
    negative_logits = (negative_similarity / temperature).masked_fill(
        ~negative_mask,
        minimum,
    )
    dustbin_logit = (dustbin_score.reshape(1) / temperature).expand(
        positive_similarity.shape[0], 1
    )
    denominator = torch.logsumexp(
        torch.cat([positive_logits, negative_logits, dustbin_logit], dim=1),
        dim=1,
    )
    has_positive = positive_mask.any(dim=1)
    positive_numerator = torch.logsumexp(positive_logits, dim=1)
    numerator = torch.where(has_positive, positive_numerator, dustbin_logit[:, 0])
    return (denominator - numerator).mean()


@torch.no_grad()
def _nearest_visible_landmark(keypoint_xy, projected_xy, visible, chunk_size=512):
    count = keypoint_xy.shape[0]
    nearest_distance = keypoint_xy.new_full((count,), float("inf"))
    nearest_idx = torch.full((count,), -1, dtype=torch.long, device=keypoint_xy.device)
    visible_idx = torch.nonzero(visible, as_tuple=False).reshape(-1)
    if count == 0 or visible_idx.numel() == 0:
        return nearest_distance, nearest_idx
    visible_xy = projected_xy[visible_idx]
    for start in range(0, count, max(int(chunk_size), 1)):
        end = min(count, start + max(int(chunk_size), 1))
        distance = torch.cdist(keypoint_xy[start:end], visible_xy)
        values, local_idx = distance.min(dim=1)
        nearest_distance[start:end] = values
        nearest_idx[start:end] = visible_idx[local_idx]
    return nearest_distance, nearest_idx


@torch.no_grad()
def _nearby_visible_landmarks(
    keypoint_xy,
    projected_xy,
    visible,
    radius,
    max_count=4,
    chunk_size=256,
):
    row_count = keypoint_xy.shape[0]
    max_count = max(int(max_count), 1)
    distances = keypoint_xy.new_full((row_count, max_count), float("inf"))
    indices = torch.full(
        (row_count, max_count),
        -1,
        dtype=torch.long,
        device=keypoint_xy.device,
    )
    visible_idx = torch.nonzero(visible, as_tuple=False).reshape(-1)
    if row_count == 0 or visible_idx.numel() == 0:
        return distances, indices, indices >= 0

    visible_xy = projected_xy[visible_idx]
    keep_count = min(max_count, int(visible_idx.numel()))
    for start in range(0, row_count, max(int(chunk_size), 1)):
        end = min(row_count, start + max(int(chunk_size), 1))
        pair_distance = torch.cdist(keypoint_xy[start:end], visible_xy)
        values, local_idx = torch.topk(
            pair_distance,
            keep_count,
            dim=1,
            largest=False,
            sorted=True,
        )
        distances[start:end, :keep_count] = values
        indices[start:end, :keep_count] = visible_idx[local_idx]
    mask = distances <= float(radius)
    indices = torch.where(mask, indices, torch.full_like(indices, -1))
    return distances, indices, mask


@torch.no_grad()
def _binary_classification_metrics(logits, labels, valid_mask, bins=10):
    valid_mask = valid_mask.to(dtype=torch.bool)
    logits = logits[valid_mask]
    labels = labels[valid_mask].to(dtype=torch.float32)
    if logits.numel() == 0:
        return {"pair_ap": 0.0, "pair_auroc": 0.0, "pair_ece": 0.0}
    probability = torch.sigmoid(logits.float())
    order = torch.argsort(probability, descending=True)
    sorted_labels = labels[order]
    positive_count = sorted_labels.sum()
    negative_count = sorted_labels.numel() - positive_count
    true_positive = torch.cumsum(sorted_labels, dim=0)
    false_positive = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = true_positive / torch.arange(
        1,
        sorted_labels.numel() + 1,
        device=logits.device,
        dtype=torch.float32,
    )
    ap = (
        (precision * sorted_labels).sum() / positive_count
        if positive_count > 0
        else probability.new_zeros(())
    )
    if positive_count > 0 and negative_count > 0:
        tpr = torch.cat([probability.new_zeros(1), true_positive / positive_count])
        fpr = torch.cat([probability.new_zeros(1), false_positive / negative_count])
        auroc = torch.trapz(tpr, fpr)
    else:
        auroc = probability.new_zeros(())

    ece = probability.new_zeros(())
    boundaries = torch.linspace(0.0, 1.0, int(bins) + 1, device=probability.device)
    for bin_idx in range(int(bins)):
        in_bin = (probability >= boundaries[bin_idx]) & (
            probability < boundaries[bin_idx + 1]
            if bin_idx + 1 < int(bins)
            else probability <= boundaries[bin_idx + 1]
        )
        if bool(in_bin.any()):
            ece = ece + in_bin.float().mean() * (
                probability[in_bin].mean() - labels[in_bin].mean()
            ).abs()
    return {
        "pair_ap": float(ap.item()),
        "pair_auroc": float(auroc.item()),
        "pair_ece": float(ece.item()),
    }


@torch.no_grad()
def calibrate_binary_threshold(logits, labels, valid_mask, min_recall=0.75):
    """Maximize held-out precision while retaining a minimum correct-pair recall."""
    valid_mask = valid_mask.to(device=logits.device, dtype=torch.bool)
    logits = logits[valid_mask].float()
    labels = labels[valid_mask].float() > 0.5
    positive_count = int(labels.sum().item())
    if logits.numel() == 0 or positive_count == 0:
        return {
            "threshold": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "accepted_count": 0,
            "correct_count": 0,
        }

    order = torch.argsort(logits, descending=True, stable=True)
    sorted_logits = logits[order]
    sorted_labels = labels[order].float()
    true_positive = torch.cumsum(sorted_labels, dim=0)
    count = torch.arange(
        1,
        sorted_labels.numel() + 1,
        device=logits.device,
        dtype=torch.float32,
    )
    precision = true_positive / count
    recall = true_positive / float(positive_count)
    eligible = recall >= max(0.0, min(float(min_recall), 1.0))
    if bool(eligible.any()):
        selected = int(torch.argmax(precision.masked_fill(~eligible, -1.0)).item())
    else:
        selected = int(sorted_labels.numel() - 1)

    boundary = sorted_logits[selected]
    if selected + 1 < sorted_logits.numel() and bool(
        (boundary > sorted_logits[selected + 1]).item()
    ):
        threshold = 0.5 * (boundary + sorted_logits[selected + 1])
    else:
        threshold = boundary - torch.finfo(boundary.dtype).eps
    return {
        "threshold": float(threshold.item()),
        "precision": float(precision[selected].item()),
        "recall": float(recall[selected].item()),
        "accepted_count": int(selected + 1),
        "correct_count": int(true_positive[selected].item()),
    }


@torch.no_grad()
def _pose_scores(points_world, K, pose_w2c, damping=1e-4):
    if points_world.numel() == 0:
        return points_world.new_empty(0), points_world.new_zeros((0, 2, 6))
    jacobian = pose_jacobian_numeric(points_world, K, pose_w2c)
    information = torch.eye(6, dtype=jacobian.dtype, device=jacobian.device) * float(damping)
    information = information + torch.einsum("nai,naj->ij", jacobian, jacobian)
    inverse = torch.linalg.pinv(information)
    gain = torch.eye(2, dtype=jacobian.dtype, device=jacobian.device)[None]
    gain = gain + torch.matmul(torch.matmul(jacobian, inverse), jacobian.transpose(1, 2))
    sign, leverage = torch.linalg.slogdet(gain)
    leverage = torch.where(sign > 0, leverage, torch.zeros_like(leverage)).clamp_min(0.0)
    translation_strength = jacobian[:, :, :3].square().sum(dim=(1, 2)).clamp_min(0.0)

    def normalize(values):
        positive = values[values > 0]
        if positive.numel() == 0:
            return torch.ones_like(values)
        scale = torch.quantile(positive, 0.9).clamp_min(1e-8)
        return (values / scale).clamp(0.0, 1.0)

    score = torch.sqrt(normalize(leverage) * normalize(translation_strength)).clamp(0.0, 1.0)
    return score, jacobian


def _inverse_frequency(ids, valid, bin_count):
    result = torch.zeros(ids.shape[0], dtype=torch.float32, device=ids.device)
    if not bool(valid.any().item()) or int(bin_count) <= 0:
        return result
    counts = torch.bincount(ids[valid], minlength=int(bin_count)).float().clamp_min(1.0)
    result[valid] = torch.rsqrt(counts[ids[valid]])
    maximum = result[valid].max().clamp_min(1e-8)
    result[valid] = result[valid] / maximum
    return result


def _weighted_uniform_kl(ids, weights, bin_count):
    if ids.numel() == 0 or int(bin_count) <= 1:
        return _zero(weights)
    mass = weights.new_zeros(int(bin_count))
    mass.scatter_add_(0, ids, weights)
    probability = (mass + 1e-8) / (mass.sum() + 1e-8 * int(bin_count))
    uniform_log = -torch.log(probability.new_tensor(float(bin_count)))
    return torch.sum(probability * (torch.log(probability.clamp_min(1e-8)) - uniform_log))


def _geometry_set_loss(jacobian, weights, damping=1e-4):
    if jacobian.shape[0] < 4:
        return _zero(weights), {
            "geometry_logdet": 0.0,
            "geometry_condition": float("inf"),
        }
    column_scale = jacobian.square().mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
    normalized_jacobian = jacobian / column_scale[None, None]
    per_pair = torch.einsum("nai,naj->nij", normalized_jacobian, normalized_jacobian)
    normalized_weights = weights / weights.sum().clamp_min(1e-8)
    information = torch.einsum("n,nij->ij", normalized_weights, per_pair)
    information = information + torch.eye(6, dtype=information.dtype, device=information.device) * float(damping)
    trace_scale = torch.trace(information).clamp_min(1e-8) / 6.0
    information = information / trace_scale
    eigenvalues = torch.linalg.eigvalsh(information).clamp_min(1e-6)
    loss = -torch.log(eigenvalues).mean()
    diagnostics = {
        "geometry_logdet": float(torch.log(eigenvalues).sum().detach().item()),
        "geometry_condition": float((eigenvalues[-1] / eigenvalues[0]).detach().item()),
    }
    return loss, diagnostics


def _translation_schur_loss(jacobian, weights, damping=1e-4):
    if jacobian.shape[0] < 4 or weights.numel() != jacobian.shape[0]:
        return _zero(weights), {
            "translation_logdet": 0.0,
            "translation_condition": float("inf"),
            "translation_min_eig": 0.0,
        }
    column_scale = jacobian.square().mean(dim=(0, 1)).sqrt().clamp_min(1e-6)
    normalized_jacobian = jacobian / column_scale[None, None]
    normalized_weights = weights / weights.sum().clamp_min(1e-8)
    information = torch.einsum(
        "n,nai,naj->ij",
        normalized_weights,
        normalized_jacobian,
        normalized_jacobian,
    )
    eye = torch.eye(3, dtype=information.dtype, device=information.device)
    h_tt = information[:3, :3]
    h_tr = information[:3, 3:]
    h_rr = information[3:, 3:]
    translation = h_tt - h_tr @ torch.linalg.pinv(h_rr + eye * float(damping)) @ h_tr.T
    translation = 0.5 * (translation + translation.T) + eye * float(damping)
    trace_scale = torch.trace(translation).clamp_min(1e-8) / 3.0
    translation = translation / trace_scale
    eigenvalues = torch.linalg.eigvalsh(translation).clamp_min(1e-6)
    loss = -torch.log(eigenvalues).mean()
    return loss, {
        "translation_logdet": float(torch.log(eigenvalues).sum().detach().item()),
        "translation_condition": float((eigenvalues[-1] / eigenvalues[0]).detach().item()),
        "translation_min_eig": float(eigenvalues[0].detach().item()),
    }


def build_sparse_candidate_batch(
    query_feature_map,
    detector_heatmap,
    landmark_features,
    landmark_xyz,
    K,
    pose_gt_w2c,
    *,
    visible_mask=None,
    detect_num=2048,
    nms_radius=2,
    match_mode="topk",
    match_topk=1,
    match_threshold=0.0,
    dual_softmax=False,
    dual_softmax_temperature=0.1,
    positive_radius_px=2.0,
    negative_radius_px=None,
    max_positives=1,
    hard_negatives=8,
    hard_negative_pool_multiplier=4,
    match_temperature=0.1,
    match_margin=0.5,
    grid_rows=4,
    grid_cols=4,
    depth_bins=4,
    detector_positive_floor=0.25,
    pose_damping=1e-4,
    dustbin_score=None,
    pair_scorer=None,
    pair_measurement_head=None,
    pair_context_topk=8,
    detector_supervision_heatmap=None,
    keypoint_offset_map=None,
    pair_measurement_accept_threshold=0.0,
    detector_offset_target_source="geometric_nearest",
    detector_target_source="geometric",
    detector_binary_target=False,
):
    """Build the actual query-keypoint/landmark candidates used by sparse localization."""
    if query_feature_map.dim() != 3:
        raise ValueError(f"query_feature_map must be CxHxW, got {tuple(query_feature_map.shape)}")
    height, width = query_feature_map.shape[-2:]
    device = query_feature_map.device
    dtype = query_feature_map.dtype
    negative_radius_px = (
        float(positive_radius_px)
        if negative_radius_px is None
        else max(float(negative_radius_px), float(positive_radius_px))
    )
    if detector_offset_target_source not in {"geometric_nearest", "matched_top1"}:
        raise ValueError(
            "detector_offset_target_source must be 'geometric_nearest' or "
            f"'matched_top1', got {detector_offset_target_source!r}"
        )
    landmark_xyz = landmark_xyz.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    pose_gt_w2c = pose_gt_w2c.to(device=device, dtype=dtype)

    keypoint_ids, matching_detector_scores = select_keypoints(
        detector_heatmap,
        detect_num,
        nms_radius,
    )
    detector_scores = (
        matching_detector_scores
        if detector_supervision_heatmap is None
        else detector_supervision_heatmap.reshape(-1)[keypoint_ids]
    )
    keypoint_xy = torch.stack(
        [keypoint_ids % width, torch.div(keypoint_ids, width, rounding_mode="floor")],
        dim=1,
    ).to(dtype=dtype)
    base_observed_xy = keypoint_xy + 0.5
    if keypoint_offset_map is not None:
        if keypoint_offset_map.shape != (2, height, width):
            raise ValueError(
                "keypoint_offset_map must be 2xHxW, got "
                f"{tuple(keypoint_offset_map.shape)}"
            )
        detector_offset_predictions = keypoint_offset_map.reshape(2, -1)[
            :, keypoint_ids
        ].T
    else:
        detector_offset_predictions = base_observed_xy.new_zeros(
            (keypoint_ids.numel(), 2)
        )
    observed_xy = base_observed_xy + detector_offset_predictions
    query_features = query_feature_map.reshape(query_feature_map.shape[0], -1)[:, keypoint_ids].T
    similarity, score_matrix = build_score_matrix(
        query_features,
        landmark_features.reshape(landmark_features.shape[0], -1),
        normalize=True,
        use_dual_softmax=dual_softmax,
        dual_softmax_temperature=dual_softmax_temperature,
    )

    with torch.no_grad():
        projected_xy, project_valid = project_world_to_pixels(landmark_xyz, K, pose_gt_w2c)
        in_image = (
            (projected_xy[:, 0] >= 0)
            & (projected_xy[:, 0] <= width - 1)
            & (projected_xy[:, 1] >= 0)
            & (projected_xy[:, 1] <= height - 1)
        )
        visible = project_valid & in_image & torch.isfinite(projected_xy).all(dim=1)
        if visible_mask is not None:
            visible = visible & visible_mask.to(device=device, dtype=torch.bool).reshape(-1)

        (
            base_positive_distance,
            base_positive_landmark_idx,
            base_positive_mask,
        ) = _nearby_visible_landmarks(
            base_observed_xy,
            projected_xy,
            visible,
            positive_radius_px,
            max_count=max_positives,
        )
        detector_offset_valid_mask = base_positive_mask.any(dim=1)
        detector_offset_targets = base_observed_xy.new_zeros(
            (keypoint_ids.numel(), 2)
        )
        if bool(detector_offset_valid_mask.any()):
            offset_landmark_idx = base_positive_landmark_idx[
                detector_offset_valid_mask, 0
            ]
            detector_offset_targets[detector_offset_valid_mask] = (
                projected_xy[offset_landmark_idx]
                - base_observed_xy[detector_offset_valid_mask]
            )

        predicted_matches = match_score_matrix(
            score_matrix.detach(),
            mode=match_mode,
            topk=match_topk,
            threshold=match_threshold,
        )
        predicted_keypoint_idx = predicted_matches.keypoint_idx
        predicted_landmark_idx = predicted_matches.landmark_idx
        if predicted_keypoint_idx.numel() > 0:
            predicted_error = torch.linalg.norm(
                projected_xy[predicted_landmark_idx] - observed_xy[predicted_keypoint_idx],
                dim=1,
            )
            predicted_label = visible[predicted_landmark_idx] & (
                predicted_error <= float(positive_radius_px)
            )
            predicted_ambiguous = visible[predicted_landmark_idx] & (
                predicted_error > float(positive_radius_px)
            ) & (predicted_error <= negative_radius_px)
            predicted_valid = ~predicted_ambiguous
        else:
            predicted_error = observed_xy.new_empty(0)
            predicted_label = torch.empty(0, dtype=torch.bool, device=device)
            predicted_ambiguous = torch.empty(0, dtype=torch.bool, device=device)
            predicted_valid = torch.empty(0, dtype=torch.bool, device=device)

        if keypoint_offset_map is None:
            positive_distance = base_positive_distance
            positive_landmark_idx = base_positive_landmark_idx
            positive_mask = base_positive_mask
        else:
            (
                positive_distance,
                positive_landmark_idx,
                positive_mask,
            ) = _nearby_visible_landmarks(
                observed_xy,
                projected_xy,
                visible,
                positive_radius_px,
                max_count=max_positives,
            )
        nearest_distance = positive_distance[:, 0]
        nearest_landmark_idx = positive_landmark_idx[:, 0]
        keypoint_has_gt = positive_mask.any(dim=1)
        predicted_positive_per_keypoint = torch.zeros(
            keypoint_ids.shape[0], dtype=torch.bool, device=device
        )
        if predicted_keypoint_idx.numel() > 0 and bool(predicted_label.any().item()):
            predicted_positive_per_keypoint[predicted_keypoint_idx[predicted_label]] = True
        top1_landmark_idx = score_matrix.detach().argmax(dim=1)
        top1_error = torch.linalg.norm(
            projected_xy[top1_landmark_idx] - observed_xy,
            dim=1,
        )
        top1_correct_per_keypoint = visible[top1_landmark_idx] & (
            top1_error <= float(positive_radius_px)
        )
        if detector_offset_target_source == "matched_top1":
            top1_base_error = torch.linalg.norm(
                projected_xy[top1_landmark_idx] - base_observed_xy,
                dim=1,
            )
            detector_offset_valid_mask = visible[top1_landmark_idx] & (
                top1_base_error <= float(positive_radius_px)
            )
            detector_offset_targets.zero_()
            detector_offset_targets[detector_offset_valid_mask] = (
                projected_xy[top1_landmark_idx[detector_offset_valid_mask]]
                - base_observed_xy[detector_offset_valid_mask]
            )
        topk_rescued_per_keypoint = (
            predicted_positive_per_keypoint & ~top1_correct_per_keypoint
        )
        false_negative = keypoint_has_gt & ~predicted_positive_per_keypoint
        recovered_keypoint_idx = torch.nonzero(false_negative, as_tuple=False).reshape(-1)
        recovered_landmark_idx = nearest_landmark_idx[recovered_keypoint_idx]

    pair_keypoint_idx = torch.cat([predicted_keypoint_idx, recovered_keypoint_idx], dim=0)
    pair_landmark_idx = torch.cat([predicted_landmark_idx, recovered_landmark_idx], dim=0)
    pair_labels = torch.cat(
        [
            predicted_label.to(dtype=dtype),
            torch.ones(recovered_keypoint_idx.shape[0], dtype=dtype, device=device),
        ],
        dim=0,
    )
    pair_valid_mask = torch.cat(
        [
            predicted_valid,
            torch.ones(recovered_keypoint_idx.shape[0], dtype=torch.bool, device=device),
        ],
        dim=0,
    )
    pair_reprojection_error = torch.cat(
        [predicted_error, nearest_distance[recovered_keypoint_idx]], dim=0
    )
    pair_logits = _match_logits(
        similarity[pair_keypoint_idx, pair_landmark_idx],
        temperature=match_temperature,
        margin=match_margin,
    )
    pair_scorer_features = build_pair_context_features(
        similarity,
        matching_detector_scores,
        predicted_matches,
        context_topk=pair_context_topk,
        entropy_temperature=match_temperature,
    )
    scorer_query_features = F.normalize(query_features, dim=1)
    scorer_landmark_features = F.normalize(landmark_features, dim=1)
    global_query_descriptor = (
        F.normalize(scorer_query_features.mean(dim=0), dim=0)
        if scorer_query_features.shape[0] > 0
        else scorer_query_features.new_zeros(scorer_query_features.shape[1])
    )
    pair_scorer_logits = (
        pair_scorer(
            pair_scorer_features,
            scorer_query_features[predicted_keypoint_idx],
            scorer_landmark_features[predicted_landmark_idx],
            global_query_descriptor,
        )
        if pair_scorer is not None
        else similarity.new_empty(0)
    )
    if pair_measurement_head is not None and predicted_keypoint_idx.numel() > 0:
        pair_measurement_patch = sample_local_correlation_patch(
            query_feature_map,
            keypoint_xy[predicted_keypoint_idx],
            scorer_landmark_features[predicted_landmark_idx],
            radius=pair_measurement_head.patch_radius,
        )
        pair_measurement_output = pair_measurement_head(
            pair_scorer_features,
            pair_measurement_patch,
            scorer_query_features[predicted_keypoint_idx],
            scorer_landmark_features[predicted_landmark_idx],
            geometry_features=(
                build_pair_geometry_features(
                    observed_xy[predicted_keypoint_idx],
                    landmark_xyz[predicted_landmark_idx],
                    landmark_xyz,
                    (height, width),
                )
                if pair_measurement_head.use_geometry_context
                else None
            ),
        )
    else:
        pair_measurement_output = PairMeasurementOutput(
            similarity.new_empty(0),
            similarity.new_empty((0, 2)),
            similarity.new_empty((0, 2, 2)),
        )
    pair_measurement_target_offsets = (
        projected_xy[predicted_landmark_idx]
        - observed_xy[predicted_keypoint_idx]
    )
    matcher_assignment_logits = _match_logits(
        similarity[predicted_keypoint_idx, predicted_landmark_idx],
        temperature=match_temperature,
        margin=match_margin,
    )

    assignment_keypoint_idx = torch.nonzero(keypoint_has_gt, as_tuple=False).reshape(-1)
    assignment_positive_similarity = (
        similarity[assignment_keypoint_idx, nearest_landmark_idx[assignment_keypoint_idx]]
        if assignment_keypoint_idx.numel() > 0
        else similarity.new_empty(0)
    )
    assignment_negative_similarity = similarity.new_empty((assignment_keypoint_idx.numel(), 0))
    assignment_negative_mask = torch.empty(
        assignment_keypoint_idx.numel(),
        0,
        dtype=torch.bool,
        device=device,
    )
    hard_negative_logits = similarity.new_empty(0)
    hard_negative_count = 0
    if similarity.numel() > 0 and int(hard_negatives) > 0:
        pool_size = min(
            similarity.shape[1],
            max(int(hard_negatives), int(hard_negatives) * int(hard_negative_pool_multiplier)),
        )
        _, hard_idx = torch.topk(score_matrix.detach(), pool_size, dim=1)
        hard_xy = projected_xy[hard_idx]
        hard_error = torch.linalg.norm(hard_xy - observed_xy[:, None], dim=2)
        hard_is_negative = (~visible[hard_idx]) | (hard_error > negative_radius_px)
        hard_similarity = similarity.gather(1, hard_idx)
        rank = torch.cumsum(hard_is_negative.to(torch.long), dim=1)
        keep = hard_is_negative & (rank <= int(hard_negatives))
        hard_keypoint_idx = torch.arange(similarity.shape[0], device=device)[:, None].expand_as(hard_idx)[keep]
        hard_landmark_idx = hard_idx[keep]
        hard_negative_logits = _match_logits(
            similarity[hard_keypoint_idx, hard_landmark_idx],
            temperature=match_temperature,
            margin=match_margin,
        )
        hard_negative_count = int(hard_negative_logits.numel())

        assignment_count = min(int(hard_negatives), int(hard_similarity.shape[1]))
        assignment_candidates = hard_similarity.masked_fill(
            ~hard_is_negative,
            torch.finfo(hard_similarity.dtype).min,
        )
        assignment_negative_similarity = torch.topk(
            assignment_candidates,
            assignment_count,
            dim=1,
        ).values[assignment_keypoint_idx]
        assignment_negative_mask = assignment_negative_similarity > (
            torch.finfo(assignment_negative_similarity.dtype).min / 2
        )
        assignment_negative_similarity = torch.where(
            assignment_negative_mask,
            assignment_negative_similarity,
            torch.zeros_like(assignment_negative_similarity),
        )

    safe_positive_idx = positive_landmark_idx.clamp_min(0)
    multi_positive_similarity = similarity.gather(1, safe_positive_idx)
    multi_positive_similarity = torch.where(
        positive_mask,
        multi_positive_similarity,
        torch.zeros_like(multi_positive_similarity),
    )
    if similarity.numel() > 0 and int(hard_negatives) > 0:
        multi_negative_similarity = torch.topk(
            assignment_candidates,
            assignment_count,
            dim=1,
        ).values
        multi_negative_mask = multi_negative_similarity > (
            torch.finfo(multi_negative_similarity.dtype).min / 2
        )
        multi_negative_similarity = torch.where(
            multi_negative_mask,
            multi_negative_similarity,
            torch.zeros_like(multi_negative_similarity),
        )
    else:
        multi_negative_similarity = similarity.new_empty((similarity.shape[0], 0))
        multi_negative_mask = torch.empty(
            similarity.shape[0], 0, dtype=torch.bool, device=device
        )
    if dustbin_score is None:
        dustbin_score = similarity.new_tensor(float(match_margin))
    else:
        dustbin_score = dustbin_score.to(device=device, dtype=dtype)

    detector_targets = detector_scores.new_zeros(detector_scores.shape)
    detector_loss_weights = detector_scores.new_ones(detector_scores.shape)
    positive_pose_scores = detector_scores.new_empty(0)
    positive_grid_ids = torch.empty(0, dtype=torch.long, device=device)
    positive_depth_ids = torch.empty(0, dtype=torch.long, device=device)
    geometry_jacobian = detector_scores.new_zeros((0, 2, 6))
    keypoint_pose_score = detector_scores.new_zeros(detector_scores.shape)
    keypoint_grid_ids = (
        torch.div(keypoint_xy[:, 1] * int(grid_rows), max(height, 1), rounding_mode="floor").long().clamp(0, int(grid_rows) - 1)
        * int(grid_cols)
        + torch.div(keypoint_xy[:, 0] * int(grid_cols), max(width, 1), rounding_mode="floor").long().clamp(0, int(grid_cols) - 1)
    )
    keypoint_depth_ids = torch.zeros(keypoint_ids.shape[0], dtype=torch.long, device=device)

    detector_positive = keypoint_has_gt
    if detector_target_source == "predicted_correct":
        detector_positive = predicted_positive_per_keypoint
    elif detector_target_source == "scorer_accepted_correct":
        if pair_scorer_logits.numel() == 0:
            raise ValueError(
                "scorer_accepted_correct detector targets require an active pair scorer"
            )
        detector_positive = torch.zeros_like(keypoint_has_gt)
        scorer_correct = predicted_label & (pair_scorer_logits.detach() > 0.0)
        detector_positive[predicted_keypoint_idx[scorer_correct]] = True
    elif detector_target_source == "measurement_accepted_correct":
        if pair_measurement_output.inlier_logits.numel() == 0:
            raise ValueError(
                "measurement_accepted_correct detector targets require an active "
                "pair measurement head"
            )
        detector_positive = torch.zeros_like(keypoint_has_gt)
        measurement_correct = (
            predicted_label
            & predicted_valid
            & (
                pair_measurement_output.inlier_logits.detach()
                > float(pair_measurement_accept_threshold)
            )
        )
        detector_positive[
            predicted_keypoint_idx[measurement_correct]
        ] = True
    elif detector_target_source != "geometric":
        raise ValueError(f"Unknown detector target source: {detector_target_source}")

    with torch.no_grad():
        valid_keypoint_idx = torch.nonzero(keypoint_has_gt, as_tuple=False).reshape(-1)
        if valid_keypoint_idx.numel() > 0:
            valid_landmark_idx = nearest_landmark_idx[valid_keypoint_idx]
            pose_scores, valid_jacobian = _pose_scores(
                landmark_xyz[valid_landmark_idx],
                K,
                pose_gt_w2c,
                damping=pose_damping,
            )
            keypoint_pose_score[valid_keypoint_idx] = pose_scores
            camera_points = (
                pose_gt_w2c
                @ torch.cat(
                    [landmark_xyz, torch.ones(landmark_xyz.shape[0], 1, device=device, dtype=dtype)],
                    dim=1,
                ).T
            )[:3].T
            visible_depth = camera_points[visible, 2]
            positive_depth = camera_points[valid_landmark_idx, 2]
            if int(depth_bins) > 1 and visible_depth.numel() > 0:
                quantiles = torch.linspace(0, 1, int(depth_bins) + 1, device=device, dtype=dtype)[1:-1]
                boundaries = torch.quantile(visible_depth, quantiles)
                keypoint_depth_ids[valid_keypoint_idx] = torch.bucketize(positive_depth, boundaries)

            spatial_balance = _inverse_frequency(
                keypoint_grid_ids,
                detector_positive,
                int(grid_rows) * int(grid_cols),
            ).to(dtype=dtype)
            depth_balance = _inverse_frequency(
                keypoint_depth_ids,
                detector_positive,
                max(int(depth_bins), 1),
            ).to(dtype=dtype)
            balance = torch.sqrt(spatial_balance * depth_balance).clamp(0.0, 1.0)
            positive_floor = max(0.0, min(float(detector_positive_floor), 1.0))
            positive_utility = (
                positive_floor + (1.0 - positive_floor) * keypoint_pose_score[detector_positive]
            ) * balance[detector_positive]
            if detector_binary_target:
                detector_targets[detector_positive] = 1.0
                detector_loss_weights[detector_positive] = positive_utility.clamp_min(0.05)
            else:
                detector_targets[detector_positive] = positive_utility

            positive_pair = pair_labels > 0.5
            if bool(positive_pair.any().item()):
                positive_kp = pair_keypoint_idx[positive_pair]
                positive_lm = pair_landmark_idx[positive_pair]
                positive_pose_scores = keypoint_pose_score[positive_kp]
                positive_grid_ids = keypoint_grid_ids[positive_kp]
                positive_depth_ids = keypoint_depth_ids[positive_kp]
                geometry_jacobian = pose_jacobian_numeric(
                    landmark_xyz[positive_lm], K, pose_gt_w2c
                ).detach()

    predicted_count = int(predicted_label.numel())
    predicted_correct = int(predicted_label.sum().item())
    valid_assignment_row = assignment_negative_mask.any(dim=1)
    if bool(valid_assignment_row.any().item()):
        hardest_negative = assignment_negative_similarity[valid_assignment_row].masked_fill(
            ~assignment_negative_mask[valid_assignment_row],
            torch.finfo(assignment_negative_similarity.dtype).min,
        ).max(dim=1).values
        assignment_margin = assignment_positive_similarity[valid_assignment_row] - hardest_negative
        assignment_top1_accuracy = float((assignment_margin > 0).float().mean().detach().item())
        assignment_margin_mean = float(assignment_margin.mean().detach().item())
        assignment_margin_median = float(assignment_margin.median().detach().item())
    else:
        assignment_top1_accuracy = 0.0
        assignment_margin_mean = 0.0
        assignment_margin_median = 0.0

    predicted_logits = _match_logits(
        similarity[predicted_keypoint_idx, predicted_landmark_idx],
        temperature=match_temperature,
        margin=match_margin,
    )
    pair_metrics = _binary_classification_metrics(
        predicted_logits.detach(),
        predicted_label,
        predicted_valid,
    )
    if pair_scorer_logits.numel() > 0:
        scorer_metrics = _binary_classification_metrics(
            pair_scorer_logits.detach(),
            predicted_label,
            predicted_valid,
        )
        scorer_accepted = (pair_scorer_logits.detach() > 0.0) & predicted_valid
        scorer_correct = scorer_accepted & predicted_label
        pair_metrics.update(
            {
                "pair_scorer_ap": scorer_metrics["pair_ap"],
                "pair_scorer_auroc": scorer_metrics["pair_auroc"],
                "pair_scorer_ece": scorer_metrics["pair_ece"],
                "pair_scorer_accepted_count": int(scorer_accepted.sum().item()),
                "pair_scorer_accepted_gt_precision": float(
                    scorer_correct.sum().item()
                    / max(int(scorer_accepted.sum().item()), 1)
                ),
                "pair_scorer_correct_accept_recall": float(
                    scorer_correct.sum().item() / max(predicted_correct, 1)
                ),
            }
        )
    if pair_measurement_output.inlier_logits.numel() > 0:
        measurement_metrics = _binary_classification_metrics(
            pair_measurement_output.inlier_logits.detach(),
            predicted_label,
            predicted_valid,
        )
        positive_measurement = predicted_label & predicted_valid
        measurement_endpoint_error = torch.linalg.norm(
            pair_measurement_output.offset.detach()[positive_measurement]
            - pair_measurement_target_offsets[positive_measurement],
            dim=1,
        )
        measurement_residual = (
            pair_measurement_output.offset.detach()[positive_measurement]
            - pair_measurement_target_offsets[positive_measurement]
        )
        measurement_cholesky = pair_measurement_output.cholesky.detach()[
            positive_measurement
        ]
        if measurement_residual.numel() > 0:
            whitened = torch.linalg.solve_triangular(
                measurement_cholesky,
                measurement_residual.unsqueeze(-1),
                upper=False,
            ).squeeze(-1)
            whitened_squared = whitened.square().sum(dim=1)
            covariance = measurement_cholesky @ measurement_cholesky.transpose(
                -1, -2
            )
            sigma = torch.diagonal(covariance, dim1=-2, dim2=-1).sqrt()
            signed_bias = measurement_residual.mean(dim=0)
        else:
            whitened_squared = measurement_endpoint_error
            sigma = measurement_residual
            signed_bias = measurement_residual.new_zeros(2)
        pair_metrics.update(
            {
                "pair_measurement_ap": measurement_metrics["pair_ap"],
                "pair_measurement_auroc": measurement_metrics["pair_auroc"],
                "pair_measurement_ece": measurement_metrics["pair_ece"],
                "pair_measurement_offset_epe_mean": float(
                    measurement_endpoint_error.mean().item()
                    if measurement_endpoint_error.numel()
                    else 0.0
                ),
                "pair_measurement_offset_epe_p95": float(
                    torch.quantile(measurement_endpoint_error, 0.95).item()
                    if measurement_endpoint_error.numel()
                    else 0.0
                ),
                "pair_measurement_corrected_signed_bias_x_px": float(
                    signed_bias[0].item()
                ),
                "pair_measurement_corrected_signed_bias_y_px": float(
                    signed_bias[1].item()
                ),
                "pair_measurement_corrected_signed_bias_norm_px": float(
                    torch.linalg.norm(signed_bias).item()
                ),
                "pair_measurement_whitened_squared_mean": float(
                    whitened_squared.mean().item()
                    if whitened_squared.numel()
                    else 0.0
                ),
                "pair_measurement_coverage_68": float(
                    (whitened_squared <= 2.30).float().mean().item()
                    if whitened_squared.numel()
                    else 0.0
                ),
                "pair_measurement_coverage_95": float(
                    (whitened_squared <= 5.99).float().mean().item()
                    if whitened_squared.numel()
                    else 0.0
                ),
                "pair_measurement_sigma_mean_px": float(
                    sigma.mean().item() if sigma.numel() else 0.0
                ),
            }
        )
    accepted = (
        similarity[predicted_keypoint_idx, predicted_landmark_idx].detach()
        > dustbin_score.detach()
    )
    accepted_valid = accepted & predicted_valid
    accepted_correct = accepted & predicted_label
    predicted_correct_error = predicted_error[predicted_label]
    top1_correct_error = top1_error[top1_correct_per_keypoint]
    valid_offset_prediction = detector_offset_predictions.detach()[
        detector_offset_valid_mask
    ]
    valid_offset_target = detector_offset_targets[detector_offset_valid_mask]
    offset_endpoint_error = torch.linalg.norm(
        valid_offset_prediction - valid_offset_target,
        dim=1,
    )
    offset_target_norm = torch.linalg.norm(valid_offset_target, dim=1)
    offset_prediction_norm = torch.linalg.norm(valid_offset_prediction, dim=1)
    row_best_score = score_matrix.detach().max(dim=1).values if score_matrix.shape[1] else detector_scores.new_empty(0)
    unmatched_row = ~keypoint_has_gt
    unmatched_rejected = row_best_score <= dustbin_score.detach()
    with torch.no_grad():
        pair_scorer_positive_jacobian = (
            pose_jacobian_numeric(
                landmark_xyz[predicted_landmark_idx[predicted_label]],
                K,
                pose_gt_w2c,
            ).detach()
            if bool(predicted_label.any())
            else similarity.new_zeros((0, 2, 6))
        )
        pair_measurement_jacobian = (
            pose_jacobian_analytic(
                landmark_xyz[predicted_landmark_idx],
                K,
                pose_gt_w2c,
            ).detach()
            if predicted_landmark_idx.numel() > 0
            else similarity.new_zeros((0, 2, 6))
        )
        pair_measurement_geometry_valid_mask = (
            predicted_valid
            & visible[predicted_landmark_idx]
            & torch.isfinite(pair_measurement_target_offsets).all(dim=1)
        )

    diagnostics = {
        "keypoint_count": int(keypoint_ids.numel()),
        "visible_landmark_count": int(visible.sum().item()),
        "predicted_pair_count": predicted_count,
        "predicted_correct_count": predicted_correct,
        "predicted_ambiguous_count": int(predicted_ambiguous.sum().item()),
        "predicted_valid_count": int(predicted_valid.sum().item()),
        "predicted_gt_precision": float(predicted_correct / max(predicted_count, 1)),
        "predicted_correct_reprojection_error_mean": float(
            predicted_correct_error.mean().item()
            if predicted_correct_error.numel() > 0
            else 0.0
        ),
        "predicted_correct_reprojection_error_p95": float(
            torch.quantile(predicted_correct_error, 0.95).item()
            if predicted_correct_error.numel() > 0
            else 0.0
        ),
        "predicted_correct_keypoint_count": int(
            predicted_positive_per_keypoint.sum().item()
        ),
        "predicted_correct_keypoint_recall": float(
            predicted_positive_per_keypoint.sum().item()
            / max(int(keypoint_has_gt.sum().item()), 1)
        ),
        "top1_correct_keypoint_count": int(top1_correct_per_keypoint.sum().item()),
        "top1_correct_reprojection_error_mean": float(
            top1_correct_error.mean().item() if top1_correct_error.numel() > 0 else 0.0
        ),
        "top1_correct_reprojection_error_p95": float(
            torch.quantile(top1_correct_error, 0.95).item()
            if top1_correct_error.numel() > 0
            else 0.0
        ),
        "topk_rescued_keypoint_count": int(topk_rescued_per_keypoint.sum().item()),
        "keypoint_with_gt_count": int(keypoint_has_gt.sum().item()),
        "keypoint_gt_coverage": float(
            keypoint_has_gt.sum().item() / max(int(keypoint_ids.numel()), 1)
        ),
        "detector_offset_target_count": int(detector_offset_valid_mask.sum().item()),
        "detector_offset_target_norm_mean": float(
            offset_target_norm.mean().item() if offset_target_norm.numel() > 0 else 0.0
        ),
        "detector_offset_prediction_norm_mean": float(
            offset_prediction_norm.mean().item()
            if offset_prediction_norm.numel() > 0
            else 0.0
        ),
        "detector_offset_endpoint_error_mean": float(
            offset_endpoint_error.mean().item()
            if offset_endpoint_error.numel() > 0
            else 0.0
        ),
        "detector_offset_endpoint_error_p95": float(
            torch.quantile(offset_endpoint_error, 0.95).item()
            if offset_endpoint_error.numel() > 0
            else 0.0
        ),
        "detector_target_positive_count": int(detector_positive.sum().item()),
        "detector_target_positive_rate": float(
            detector_positive.sum().item() / max(int(keypoint_ids.numel()), 1)
        ),
        "detector_positive_target_mean": float(
            detector_targets[detector_positive].mean().item()
            if bool(detector_positive.any())
            else 0.0
        ),
        "false_negative_count": int(false_negative.sum().item()),
        "false_negative_rate": float(
            false_negative.sum().item() / max(int(keypoint_has_gt.sum().item()), 1)
        ),
        "recovered_positive_count": int(recovered_keypoint_idx.numel()),
        "hard_negative_count": hard_negative_count,
        "multi_positive_row_count": int(keypoint_has_gt.sum().item()),
        "multi_positive_count": int(positive_mask.sum().item()),
        "multi_positive_per_positive_row": float(
            positive_mask.sum().item() / max(int(keypoint_has_gt.sum().item()), 1)
        ),
        "dustbin_score": float(dustbin_score.detach().item()),
        "dustbin_accepted_count": int(accepted_valid.sum().item()),
        "dustbin_accepted_gt_precision": float(
            accepted_correct.sum().item() / max(int(accepted_valid.sum().item()), 1)
        ),
        "dustbin_correct_accept_recall": float(
            accepted_correct.sum().item() / max(predicted_correct, 1)
        ),
        "dustbin_unmatched_reject_accuracy": float(
            unmatched_rejected[unmatched_row].float().mean().item()
            if bool(unmatched_row.any())
            else 0.0
        ),
        "assignment_row_count": int(valid_assignment_row.sum().item()),
        "assignment_top1_accuracy": assignment_top1_accuracy,
        "assignment_positive_hardest_margin_mean": assignment_margin_mean,
        "assignment_positive_hardest_margin_median": assignment_margin_median,
        "positive_grid_occupancy": int(torch.unique(positive_grid_ids).numel()),
        "positive_depth_occupancy": int(torch.unique(positive_depth_ids).numel()),
        "geometry_jacobian": geometry_jacobian,
        "grid_bin_count": int(grid_rows) * int(grid_cols),
        "depth_bin_count": max(int(depth_bins), 1),
    }
    diagnostics.update(pair_metrics)
    return SparseCandidateBatch(
        keypoint_ids=keypoint_ids,
        keypoint_xy=keypoint_xy,
        detector_scores=detector_scores,
        matching_detector_scores=matching_detector_scores,
        detector_targets=detector_targets,
        detector_loss_weights=detector_loss_weights,
        detector_offset_predictions=detector_offset_predictions,
        detector_offset_targets=detector_offset_targets,
        detector_offset_valid_mask=detector_offset_valid_mask,
        pair_keypoint_idx=pair_keypoint_idx,
        pair_landmark_idx=pair_landmark_idx,
        pair_logits=pair_logits,
        pair_labels=pair_labels,
        pair_valid_mask=pair_valid_mask,
        pair_reprojection_error=pair_reprojection_error,
        pair_scorer_features=pair_scorer_features,
        pair_scorer_logits=pair_scorer_logits,
        matcher_assignment_logits=matcher_assignment_logits,
        pair_scorer_keypoint_idx=predicted_keypoint_idx,
        pair_scorer_landmark_idx=predicted_landmark_idx,
        pair_scorer_labels=predicted_label.to(dtype=dtype),
        pair_scorer_valid_mask=predicted_valid,
        pair_scorer_reprojection_error=predicted_error,
        pair_scorer_positive_jacobian=pair_scorer_positive_jacobian,
        pair_measurement_inlier_logits=pair_measurement_output.inlier_logits,
        pair_measurement_offsets=pair_measurement_output.offset,
        pair_measurement_cholesky=pair_measurement_output.cholesky,
        pair_measurement_target_offsets=pair_measurement_target_offsets,
        pair_measurement_jacobian=pair_measurement_jacobian,
        pair_measurement_geometry_valid_mask=(
            pair_measurement_geometry_valid_mask
        ),
        hard_negative_logits=hard_negative_logits,
        assignment_positive_similarity=assignment_positive_similarity,
        assignment_negative_similarity=assignment_negative_similarity,
        assignment_negative_mask=assignment_negative_mask,
        multi_positive_similarity=multi_positive_similarity,
        multi_positive_mask=positive_mask,
        multi_negative_similarity=multi_negative_similarity,
        multi_negative_mask=multi_negative_mask,
        multi_row_has_positive=keypoint_has_gt,
        dustbin_score=dustbin_score,
        positive_pose_scores=positive_pose_scores,
        positive_grid_ids=positive_grid_ids,
        positive_depth_ids=positive_depth_ids,
        diagnostics=diagnostics,
    )


def sparse_candidate_losses(
    batch,
    pose_damping=1e-4,
    assignment_temperature=0.05,
    assignment_margin=0.05,
    reprojection_sigma_px=1.0,
    set_risk_residual_clip_px=32.0,
    set_risk_reference_translation_m=0.01,
):
    pair_loss = _balanced_focal_bce(
        batch.pair_logits,
        batch.pair_labels,
        valid_mask=batch.pair_valid_mask,
    )
    hard_negative_loss = (
        F.softplus(batch.hard_negative_logits).mean()
        if batch.hard_negative_logits.numel() > 0
        else _zero(batch.pair_logits)
    )
    assignment_loss = _row_assignment_loss(
        batch.assignment_positive_similarity,
        batch.assignment_negative_similarity,
        batch.assignment_negative_mask,
        temperature=assignment_temperature,
        margin=assignment_margin,
    )
    dustbin_assignment_loss = _multi_positive_dustbin_loss(
        batch.multi_positive_similarity,
        batch.multi_positive_mask,
        batch.multi_negative_similarity,
        batch.multi_negative_mask,
        batch.dustbin_score,
        temperature=assignment_temperature,
        margin=assignment_margin,
    )
    matcher_assignment_loss = _grouped_multi_positive_assignment_loss(
        batch.matcher_assignment_logits,
        batch.pair_scorer_labels,
        batch.pair_scorer_valid_mask,
        batch.pair_scorer_keypoint_idx,
    )
    matcher_reprojection_assignment_loss = _grouped_reprojection_assignment_loss(
        batch.matcher_assignment_logits,
        batch.pair_scorer_labels,
        batch.pair_scorer_reprojection_error,
        batch.pair_scorer_valid_mask,
        batch.pair_scorer_keypoint_idx,
        sigma_px=reprojection_sigma_px,
    )
    pair_scorer_loss = _balanced_focal_bce(
        batch.pair_scorer_logits,
        batch.pair_scorer_labels,
        valid_mask=batch.pair_scorer_valid_mask,
    )
    pair_scorer_assignment_loss = _grouped_multi_positive_assignment_loss(
        batch.pair_scorer_logits,
        batch.pair_scorer_labels,
        batch.pair_scorer_valid_mask,
        batch.pair_scorer_keypoint_idx,
    )
    pair_measurement_inlier_loss = _balanced_focal_bce(
        batch.pair_measurement_inlier_logits,
        batch.pair_scorer_labels,
        valid_mask=batch.pair_scorer_valid_mask,
    )
    pair_measurement_positive = (
        (batch.pair_scorer_labels > 0.5) & batch.pair_scorer_valid_mask
    )
    pair_measurement_nll_loss = gaussian_measurement_nll(
        batch.pair_measurement_target_offsets,
        PairMeasurementOutput(
            batch.pair_measurement_inlier_logits,
            batch.pair_measurement_offsets,
            batch.pair_measurement_cholesky,
        ),
        valid_mask=pair_measurement_positive,
    )
    pair_measurement_residual = (
        batch.pair_measurement_offsets - batch.pair_measurement_target_offsets
        if batch.pair_measurement_offsets.numel() > 0
        else batch.pair_measurement_offsets
    )
    if pair_measurement_residual.numel() > 0:
        residual_norm = torch.linalg.norm(
            pair_measurement_residual, dim=1, keepdim=True
        ).clamp_min(1e-8)
        residual_scale = (
            float(set_risk_residual_clip_px) / residual_norm
        ).clamp_max(1.0)
        pair_measurement_residual = pair_measurement_residual * residual_scale
    set_risk_pair_count = pair_measurement_residual.shape[0]
    set_risk = absolute_pose_set_risk(
        batch.pair_measurement_jacobian[:set_risk_pair_count],
        pair_measurement_residual,
        batch.pair_measurement_cholesky,
        batch.pair_measurement_inlier_logits,
        valid_mask=batch.pair_measurement_geometry_valid_mask[
            :set_risk_pair_count
        ],
        reference_translation_m=set_risk_reference_translation_m,
    )
    matcher_positive = batch.pair_scorer_labels > 0.5
    matcher_translation_weights = torch.sigmoid(
        batch.matcher_assignment_logits[matcher_positive]
    )
    matcher_translation_info_loss, matcher_translation_diagnostics = (
        _translation_schur_loss(
            batch.pair_scorer_positive_jacobian,
            matcher_translation_weights,
            damping=pose_damping,
        )
    )
    scorer_positive = batch.pair_scorer_labels > 0.5
    translation_weights = (
        torch.sigmoid(batch.pair_scorer_logits[scorer_positive])
        if batch.pair_scorer_logits.numel() > 0
        else batch.pair_scorer_logits
    )
    translation_info_loss, translation_diagnostics = _translation_schur_loss(
        batch.pair_scorer_positive_jacobian,
        translation_weights,
        damping=pose_damping,
    )
    detector_loss = _balanced_probability_bce(
        batch.detector_scores,
        batch.detector_targets,
        batch.detector_loss_weights,
    )
    detector_offset_loss = (
        F.smooth_l1_loss(
            batch.detector_offset_predictions[batch.detector_offset_valid_mask],
            batch.detector_offset_targets[batch.detector_offset_valid_mask],
            beta=0.25,
        )
        if bool(batch.detector_offset_valid_mask.any())
        else _zero(batch.detector_offset_predictions)
    )

    positive = batch.pair_labels > 0.5
    positive_weights = (
        torch.sigmoid(batch.pair_logits[positive])
        * batch.matching_detector_scores[batch.pair_keypoint_idx[positive]]
        * batch.positive_pose_scores.clamp_min(0.05)
    )
    geometry_loss, geometry_diagnostics = _geometry_set_loss(
        batch.diagnostics["geometry_jacobian"],
        positive_weights,
        damping=pose_damping,
    )
    spatial_loss = _weighted_uniform_kl(
        batch.positive_grid_ids,
        positive_weights,
        batch.diagnostics["grid_bin_count"],
    )
    depth_loss = _weighted_uniform_kl(
        batch.positive_depth_ids,
        positive_weights,
        batch.diagnostics["depth_bin_count"],
    )
    coverage_loss = 0.5 * (spatial_loss + depth_loss)
    batch.diagnostics.update(geometry_diagnostics)
    batch.diagnostics.update(translation_diagnostics)
    batch.diagnostics.update(
        {
            f"matcher_{key}": value
            for key, value in matcher_translation_diagnostics.items()
        }
    )
    batch.diagnostics.update(
        {
            "pair_measurement_set_bias_m": float(
                set_risk.translation_bias_m.detach().item()
            ),
            "pair_measurement_set_covariance_trace_m2": float(
                set_risk.translation_covariance_trace_m2.detach().item()
            ),
            "pair_measurement_set_covariance_logdet": float(
                set_risk.translation_covariance_logdet.detach().item()
            ),
            "pair_measurement_set_condition": float(
                set_risk.condition_number.detach().item()
            ),
            "pair_measurement_set_effective_count": float(
                set_risk.effective_pair_count.detach().item()
            ),
        }
    )
    return SparseCandidateLosses(
        pair=pair_loss,
        hard_negative=hard_negative_loss,
        assignment=assignment_loss,
        dustbin_assignment=dustbin_assignment_loss,
        matcher_assignment=matcher_assignment_loss,
        matcher_reprojection_assignment=matcher_reprojection_assignment_loss,
        pair_scorer=pair_scorer_loss,
        pair_scorer_assignment=pair_scorer_assignment_loss,
        pair_measurement_inlier=pair_measurement_inlier_loss,
        pair_measurement_nll=pair_measurement_nll_loss,
        pair_measurement_translation_bias=set_risk.translation_bias_loss,
        pair_measurement_translation_covariance=set_risk.translation_trace_loss,
        matcher_translation_info=matcher_translation_info_loss,
        translation_info=translation_info_loss,
        detector_match=detector_loss,
        detector_offset=detector_offset_loss,
        geometry_set=geometry_loss,
        coverage=coverage_loss,
    )
