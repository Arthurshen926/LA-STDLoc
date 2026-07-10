from dataclasses import dataclass

import torch
import torch.nn.functional as F

from localization_training.correspondence import project_world_to_pixels
from localization_training.pose_information import pose_jacobian_numeric


@dataclass
class SparseCandidateBatch:
    keypoint_ids: torch.Tensor
    keypoint_xy: torch.Tensor
    detector_scores: torch.Tensor
    detector_targets: torch.Tensor
    pair_keypoint_idx: torch.Tensor
    pair_landmark_idx: torch.Tensor
    pair_logits: torch.Tensor
    pair_labels: torch.Tensor
    pair_reprojection_error: torch.Tensor
    hard_negative_logits: torch.Tensor
    assignment_positive_similarity: torch.Tensor
    assignment_negative_similarity: torch.Tensor
    assignment_negative_mask: torch.Tensor
    positive_pose_scores: torch.Tensor
    positive_grid_ids: torch.Tensor
    positive_depth_ids: torch.Tensor
    diagnostics: dict


@dataclass
class SparseCandidateLosses:
    pair: torch.Tensor
    hard_negative: torch.Tensor
    assignment: torch.Tensor
    detector_match: torch.Tensor
    geometry_set: torch.Tensor
    coverage: torch.Tensor


def _zero(reference):
    return reference.sum() * 0.0


def _simple_nms(scores, nms_radius):
    nms_radius = int(nms_radius)
    if nms_radius < 0:
        raise ValueError("nms_radius must be non-negative")

    def max_pool(value):
        return F.max_pool2d(
            value,
            kernel_size=nms_radius * 2 + 1,
            stride=1,
            padding=nms_radius,
        )

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        suppression = max_pool(max_mask.float()) > 0
        suppressed_scores = torch.where(suppression, zeros, scores)
        new_max_mask = suppressed_scores == max_pool(suppressed_scores)
        max_mask = max_mask | (new_max_mask & ~suppression)
    return torch.where(max_mask, scores, zeros)


def _match_logits(similarity, temperature=0.1, margin=0.5):
    return (similarity - float(margin)) / max(float(temperature), 1e-6)


def _balanced_focal_bce(logits, labels, gamma=2.0):
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


def _balanced_probability_bce(probability, target):
    if probability.numel() == 0:
        return _zero(probability)
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    target = target.to(dtype=probability.dtype).clamp(0.0, 1.0)
    loss = F.binary_cross_entropy(probability, target, reduction="none")
    positive = target > 0.0
    negative = ~positive
    parts = []
    if bool(positive.any().item()):
        parts.append(loss[positive].mean())
    if bool(negative.any().item()):
        parts.append(loss[negative].mean())
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


def _select_keypoints(heatmap, height, width, count, nms_radius):
    scores = _simple_nms(heatmap, int(nms_radius)).reshape(-1)
    count = min(max(int(count), 0), int(scores.numel()))
    if count == 0:
        empty = torch.empty(0, dtype=torch.long, device=scores.device)
        return empty, scores.new_empty(0)
    values, ids = torch.topk(scores, count)
    keep = values > 0
    ids = ids[keep].sort().values
    return ids, scores[ids]


def _final_matches(score_matrix, mode="topk", topk=1, threshold=0.0):
    if score_matrix.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=score_matrix.device)
        return empty, empty
    if mode == "mnn":
        row_value, row_idx = score_matrix.max(dim=1)
        col_idx = score_matrix.max(dim=0).indices
        image_idx = torch.arange(score_matrix.shape[0], device=score_matrix.device)
        keep = (row_value > float(threshold)) & (col_idx[row_idx] == image_idx)
        return image_idx[keep], row_idx[keep]
    if mode != "topk":
        raise ValueError(f"Unknown sparse candidate match mode: {mode}")
    topk = min(max(int(topk), 1), int(score_matrix.shape[1]))
    values, landmark_idx = torch.topk(score_matrix, topk, dim=1)
    image_idx = torch.arange(score_matrix.shape[0], device=score_matrix.device)[:, None].expand_as(landmark_idx)
    keep = values > float(threshold)
    return image_idx[keep], landmark_idx[keep]


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
    hard_negatives=8,
    hard_negative_pool_multiplier=4,
    match_temperature=0.1,
    match_margin=0.5,
    grid_rows=4,
    grid_cols=4,
    depth_bins=4,
    detector_positive_floor=0.25,
    pose_damping=1e-4,
):
    """Build the actual query-keypoint/landmark candidates used by sparse localization."""
    if query_feature_map.dim() != 3:
        raise ValueError(f"query_feature_map must be CxHxW, got {tuple(query_feature_map.shape)}")
    height, width = query_feature_map.shape[-2:]
    device = query_feature_map.device
    dtype = query_feature_map.dtype
    landmark_xyz = landmark_xyz.to(device=device, dtype=dtype)
    K = K.to(device=device, dtype=dtype)
    pose_gt_w2c = pose_gt_w2c.to(device=device, dtype=dtype)

    keypoint_ids, detector_scores = _select_keypoints(
        detector_heatmap,
        height,
        width,
        detect_num,
        nms_radius,
    )
    keypoint_xy = torch.stack(
        [keypoint_ids % width, torch.div(keypoint_ids, width, rounding_mode="floor")],
        dim=1,
    ).to(dtype=dtype)
    observed_xy = keypoint_xy + 0.5
    query_features = query_feature_map.reshape(query_feature_map.shape[0], -1)[:, keypoint_ids].T
    query_features = F.normalize(query_features, dim=1)
    normalized_landmarks = F.normalize(landmark_features.reshape(landmark_features.shape[0], -1), dim=1)
    similarity = torch.matmul(query_features, normalized_landmarks.T)
    score_matrix = similarity
    if bool(dual_softmax) and similarity.numel() > 0:
        scaled = similarity / max(float(dual_softmax_temperature), 1e-6)
        score_matrix = torch.softmax(scaled, dim=0) * torch.softmax(scaled, dim=1)

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

        predicted_keypoint_idx, predicted_landmark_idx = _final_matches(
            score_matrix.detach(),
            mode=match_mode,
            topk=match_topk,
            threshold=match_threshold,
        )
        if predicted_keypoint_idx.numel() > 0:
            predicted_error = torch.linalg.norm(
                projected_xy[predicted_landmark_idx] - observed_xy[predicted_keypoint_idx],
                dim=1,
            )
            predicted_label = visible[predicted_landmark_idx] & (
                predicted_error <= float(positive_radius_px)
            )
        else:
            predicted_error = observed_xy.new_empty(0)
            predicted_label = torch.empty(0, dtype=torch.bool, device=device)

        nearest_distance, nearest_landmark_idx = _nearest_visible_landmark(
            observed_xy,
            projected_xy,
            visible,
        )
        keypoint_has_gt = (nearest_landmark_idx >= 0) & (
            nearest_distance <= float(positive_radius_px)
        )
        predicted_positive_per_keypoint = torch.zeros(
            keypoint_ids.shape[0], dtype=torch.bool, device=device
        )
        if predicted_keypoint_idx.numel() > 0 and bool(predicted_label.any().item()):
            predicted_positive_per_keypoint[predicted_keypoint_idx[predicted_label]] = True
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
    pair_reprojection_error = torch.cat(
        [predicted_error, nearest_distance[recovered_keypoint_idx]], dim=0
    )
    pair_logits = _match_logits(
        similarity[pair_keypoint_idx, pair_landmark_idx],
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
        hard_is_negative = (~visible[hard_idx]) | (hard_error > float(positive_radius_px))
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

    detector_targets = detector_scores.new_zeros(detector_scores.shape)
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
                keypoint_has_gt,
                int(grid_rows) * int(grid_cols),
            ).to(dtype=dtype)
            depth_balance = _inverse_frequency(
                keypoint_depth_ids,
                keypoint_has_gt,
                max(int(depth_bins), 1),
            ).to(dtype=dtype)
            balance = torch.sqrt(spatial_balance * depth_balance).clamp(0.0, 1.0)
            positive_floor = max(0.0, min(float(detector_positive_floor), 1.0))
            detector_targets[keypoint_has_gt] = (
                positive_floor + (1.0 - positive_floor) * keypoint_pose_score[keypoint_has_gt]
            ) * balance[keypoint_has_gt]

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

    diagnostics = {
        "keypoint_count": int(keypoint_ids.numel()),
        "visible_landmark_count": int(visible.sum().item()),
        "predicted_pair_count": predicted_count,
        "predicted_correct_count": predicted_correct,
        "predicted_gt_precision": float(predicted_correct / max(predicted_count, 1)),
        "keypoint_with_gt_count": int(keypoint_has_gt.sum().item()),
        "keypoint_gt_coverage": float(
            keypoint_has_gt.sum().item() / max(int(keypoint_ids.numel()), 1)
        ),
        "false_negative_count": int(false_negative.sum().item()),
        "false_negative_rate": float(
            false_negative.sum().item() / max(int(keypoint_has_gt.sum().item()), 1)
        ),
        "recovered_positive_count": int(recovered_keypoint_idx.numel()),
        "hard_negative_count": hard_negative_count,
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
    return SparseCandidateBatch(
        keypoint_ids=keypoint_ids,
        keypoint_xy=keypoint_xy,
        detector_scores=detector_scores,
        detector_targets=detector_targets,
        pair_keypoint_idx=pair_keypoint_idx,
        pair_landmark_idx=pair_landmark_idx,
        pair_logits=pair_logits,
        pair_labels=pair_labels,
        pair_reprojection_error=pair_reprojection_error,
        hard_negative_logits=hard_negative_logits,
        assignment_positive_similarity=assignment_positive_similarity,
        assignment_negative_similarity=assignment_negative_similarity,
        assignment_negative_mask=assignment_negative_mask,
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
):
    pair_loss = _balanced_focal_bce(batch.pair_logits, batch.pair_labels)
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
    detector_loss = _balanced_probability_bce(
        batch.detector_scores,
        batch.detector_targets,
    )

    positive = batch.pair_labels > 0.5
    positive_weights = (
        torch.sigmoid(batch.pair_logits[positive])
        * batch.detector_scores[batch.pair_keypoint_idx[positive]]
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
    return SparseCandidateLosses(
        pair=pair_loss,
        hard_negative=hard_negative_loss,
        assignment=assignment_loss,
        detector_match=detector_loss,
        geometry_set=geometry_loss,
        coverage=coverage_loss,
    )
