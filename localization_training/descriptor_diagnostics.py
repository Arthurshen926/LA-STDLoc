import math

import torch
import torch.nn.functional as F

from localization_training.correspondence import bilinear_sample_features
from localization_training.direct_landmark_teacher import (
    _limit_valid_indices,
    filter_depth_consistent_landmarks,
    make_intrinsics_from_fov,
    project_landmarks_to_query,
)


def _empty_metrics(device=None):
    return {
        "pair_count": 0,
        "positive_cosine_mean": 0.0,
        "positive_cosine_median": 0.0,
        "margin_mean": 0.0,
        "margin_median": 0.0,
        "top1_recall": 0.0,
        "mnn_precision": 0.0,
        "correct_match_count": 0,
        "mnn_correct_count": 0,
        "feature_drift_mean": 0.0,
        "feature_drift_median": 0.0,
    }


def descriptor_alignment_metrics(gaussian_features, query_features, baseline_features=None):
    if gaussian_features.shape[0] == 0 or query_features.shape[0] == 0:
        return _empty_metrics(device=gaussian_features.device)
    gaussian_features = gaussian_features.reshape(gaussian_features.shape[0], -1)
    query_features = query_features.reshape(query_features.shape[0], -1)
    if gaussian_features.numel() == 0 or query_features.numel() == 0:
        return _empty_metrics(device=gaussian_features.device)
    if gaussian_features.shape != query_features.shape:
        raise ValueError(
            "descriptor diagnostics require paired descriptors with the same shape: "
            f"got {tuple(gaussian_features.shape)} and {tuple(query_features.shape)}."
        )

    gaussian_n = F.normalize(gaussian_features.float(), p=2, dim=-1)
    query_n = F.normalize(query_features.float(), p=2, dim=-1)
    positive = (gaussian_n * query_n).sum(dim=-1)
    scores = gaussian_n @ query_n.T
    ids = torch.arange(scores.shape[0], device=scores.device)
    top1 = scores.argmax(dim=1)
    query_top1 = scores.argmax(dim=0)
    correct = top1 == ids
    mnn = correct & (query_top1 == ids)

    if scores.shape[0] > 1:
        neg = scores.masked_fill(torch.eye(scores.shape[0], dtype=torch.bool, device=scores.device), -torch.inf).max(dim=1).values
        margin = positive - neg
    else:
        margin = torch.ones_like(positive)

    if baseline_features is not None:
        baseline_features = baseline_features.reshape(baseline_features.shape[0], -1)
        if baseline_features.shape != gaussian_features.shape:
            raise ValueError(
                "baseline descriptors must match paired descriptor shape: "
                f"got {tuple(baseline_features.shape)} and {tuple(gaussian_features.shape)}."
            )
        baseline_n = F.normalize(baseline_features.float(), p=2, dim=-1)
        drift = 1.0 - (gaussian_n * baseline_n).sum(dim=-1)
    else:
        drift = torch.zeros_like(positive)

    return {
        "pair_count": int(scores.shape[0]),
        "positive_cosine_mean": float(positive.mean().item()),
        "positive_cosine_median": float(positive.median().item()),
        "margin_mean": float(margin.mean().item()),
        "margin_median": float(margin.median().item()),
        "top1_recall": float(correct.float().mean().item()),
        "mnn_precision": float(mnn.float().mean().item()),
        "correct_match_count": int(correct.sum().item()),
        "mnn_correct_count": int(mnn.sum().item()),
        "feature_drift_mean": float(drift.mean().item()),
        "feature_drift_median": float(drift.median().item()),
    }


def _empty_full_bank_metrics(topk=(1, 5, 10)):
    metrics = {
        "query_count": 0,
        "full_bank_mnn_precision": 0.0,
        "full_bank_margin_mean": 0.0,
        "full_bank_margin_median": 0.0,
        "full_bank_correct_count": 0,
        "full_bank_mnn_correct_count": 0,
        "full_bank_confidence_ece": 0.0,
    }
    for k in topk:
        metrics[f"full_bank_recall_at_{int(k)}"] = 0.0
    return metrics


def _expected_calibration_error(confidence, correct, bins=10):
    if confidence.numel() == 0:
        return 0.0
    confidence = confidence.float().clamp(0.0, 1.0)
    correct = correct.float()
    ece = confidence.new_tensor(0.0)
    edges = torch.linspace(0.0, 1.0, int(bins) + 1, device=confidence.device)
    for idx in range(int(bins)):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == int(bins) - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            weight = mask.float().mean()
            ece = ece + weight * (confidence[mask].mean() - correct[mask].mean()).abs()
    return float(ece.item())


def full_bank_descriptor_metrics(
    query_features,
    bank_features,
    positive_bank_indices,
    topk=(1, 5, 10),
    temperature=0.07,
    calibration_bins=10,
):
    """Evaluate query observations against the complete sparse landmark bank."""
    query_features = query_features.reshape(query_features.shape[0], -1)
    bank_features = bank_features.reshape(bank_features.shape[0], -1)
    positive_bank_indices = torch.as_tensor(
        positive_bank_indices,
        device=query_features.device,
        dtype=torch.long,
    ).reshape(-1)
    topk = tuple(int(k) for k in topk)
    if query_features.numel() == 0 or bank_features.numel() == 0 or positive_bank_indices.numel() == 0:
        return _empty_full_bank_metrics(topk)
    if query_features.shape[0] != positive_bank_indices.shape[0]:
        raise ValueError(
            "positive_bank_indices must contain one bank index per query descriptor: "
            f"got {positive_bank_indices.shape[0]} indices for {query_features.shape[0]} queries."
        )
    valid = (positive_bank_indices >= 0) & (positive_bank_indices < bank_features.shape[0])
    if not valid.any():
        return _empty_full_bank_metrics(topk)

    query_n = F.normalize(query_features[valid].float(), p=2, dim=-1)
    bank_n = F.normalize(bank_features.float().to(device=query_n.device), p=2, dim=-1)
    positive_bank_indices = positive_bank_indices[valid]
    scores = query_n @ bank_n.T
    query_ids = torch.arange(scores.shape[0], device=scores.device)
    positive_scores = scores[query_ids, positive_bank_indices]
    negative_scores = scores.clone()
    negative_scores[query_ids, positive_bank_indices] = -torch.inf
    hard_negative = negative_scores.max(dim=1).values
    margin = positive_scores - hard_negative

    max_k = max(1, min(max(topk), bank_n.shape[0]))
    top_indices = torch.topk(scores, k=max_k, dim=1).indices
    top1 = top_indices[:, 0]
    correct_top1 = top1 == positive_bank_indices
    bank_best_query = scores.argmax(dim=0)
    mnn = correct_top1 & (bank_best_query[positive_bank_indices] == query_ids)

    metrics = {
        "query_count": int(query_n.shape[0]),
        "full_bank_mnn_precision": float(mnn.float().mean().item()),
        "full_bank_margin_mean": float(margin.mean().item()),
        "full_bank_margin_median": float(margin.median().item()),
        "full_bank_correct_count": int(correct_top1.sum().item()),
        "full_bank_mnn_correct_count": int(mnn.sum().item()),
    }
    for k in topk:
        kk = max(1, min(int(k), bank_n.shape[0]))
        recall = (top_indices[:, :kk] == positive_bank_indices[:, None]).any(dim=1)
        metrics[f"full_bank_recall_at_{int(k)}"] = float(recall.float().mean().item())

    probs = F.softmax(scores / max(float(temperature), 1e-6), dim=1)
    top_conf = probs[query_ids, top1]
    metrics["full_bank_confidence_ece"] = _expected_calibration_error(
        top_conf,
        correct_top1,
        bins=calibration_bins,
    )
    return metrics


def summarize_descriptor_metric_batches(batch_metrics):
    total = sum(metric["pair_count"] for metric in batch_metrics)
    if total == 0:
        return _empty_metrics()

    weighted_keys = [
        "positive_cosine_mean",
        "positive_cosine_median",
        "margin_mean",
        "margin_median",
        "top1_recall",
        "mnn_precision",
        "feature_drift_mean",
        "feature_drift_median",
    ]
    summary = {"pair_count": int(total)}
    for key in weighted_keys:
        summary[key] = float(
            sum(metric[key] * metric["pair_count"] for metric in batch_metrics) / total
        )
    summary["correct_match_count"] = int(sum(metric["correct_match_count"] for metric in batch_metrics))
    summary["mnn_correct_count"] = int(sum(metric["mnn_correct_count"] for metric in batch_metrics))
    return summary


def summarize_full_bank_metric_batches(batch_metrics):
    total = sum(metric["query_count"] for metric in batch_metrics)
    if total == 0:
        keys = set()
        for metric in batch_metrics:
            keys.update(metric.keys())
        topk = []
        for key in keys:
            if key.startswith("full_bank_recall_at_"):
                topk.append(int(key.rsplit("_", 1)[-1]))
        return _empty_full_bank_metrics(tuple(sorted(topk)) or (1, 5, 10))

    weighted_keys = [
        key
        for key in batch_metrics[0].keys()
        if key.startswith("full_bank_recall_at_")
        or key
        in {
            "full_bank_mnn_precision",
            "full_bank_margin_mean",
            "full_bank_margin_median",
            "full_bank_confidence_ece",
        }
    ]
    summary = {"query_count": int(total)}
    for key in weighted_keys:
        summary[key] = float(sum(metric[key] * metric["query_count"] for metric in batch_metrics) / total)
    summary["full_bank_correct_count"] = int(sum(metric["full_bank_correct_count"] for metric in batch_metrics))
    summary["full_bank_mnn_correct_count"] = int(sum(metric["full_bank_mnn_correct_count"] for metric in batch_metrics))
    return summary


def _rankdata_average(values):
    values = values.float().reshape(-1)
    if values.numel() == 0:
        return values

    order = torch.argsort(values)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    n = values.numel()
    start = 0
    while start < n:
        end = start + 1
        while end < n and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = rank
        start = end
    return ranks


def spearman_rank_correlation(x, y, mask=None):
    x = x.double().reshape(-1)
    y = y.double().reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"Spearman inputs must have the same shape: got {tuple(x.shape)} and {tuple(y.shape)}.")
    valid = torch.isfinite(x) & torch.isfinite(y)
    if mask is not None:
        valid = valid & mask.to(device=x.device, dtype=torch.bool).reshape(-1)
    if int(valid.sum().item()) < 2:
        return 0.0

    x_rank = _rankdata_average(x[valid])
    y_rank = _rankdata_average(y[valid])
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denom = torch.linalg.norm(x_rank) * torch.linalg.norm(y_rank)
    if float(denom.item()) <= 1e-12:
        return 0.0
    corr = float(((x_rank * y_rank).sum() / denom).item())
    return max(-1.0, min(1.0, corr))


def _feature_matrix(features):
    if isinstance(features, dict):
        names = sorted(features)
        if not names:
            raise ValueError("features must contain at least one feature.")
        columns = [torch.as_tensor(features[name], dtype=torch.float32).reshape(-1) for name in names]
        length = columns[0].numel()
        if any(column.numel() != length for column in columns):
            raise ValueError("all feature tensors must have the same length.")
        return torch.stack(columns, dim=1), names
    matrix = torch.as_tensor(features, dtype=torch.float32)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {tuple(matrix.shape)}.")
    return matrix, [f"feature_{idx}" for idx in range(matrix.shape[1])]


def calibrate_landmark_quality(
    features,
    positive_count,
    trial_count,
    mask=None,
    eval_positive_count=None,
    eval_trial_count=None,
    eval_mask=None,
    target_name="inlier",
    steps=200,
    lr=0.1,
    l2=1e-3,
):
    matrix, names = _feature_matrix(features)
    positive_count = torch.as_tensor(positive_count, dtype=torch.float32).reshape(-1)
    trial_count = torch.as_tensor(trial_count, dtype=torch.float32).reshape(-1)
    if positive_count.shape != trial_count.shape or positive_count.numel() != matrix.shape[0]:
        raise ValueError("positive_count, trial_count, and features must describe the same number of landmarks.")
    if eval_positive_count is None:
        eval_positive_count = positive_count
    else:
        eval_positive_count = torch.as_tensor(eval_positive_count, dtype=torch.float32).reshape(-1)
    if eval_trial_count is None:
        eval_trial_count = trial_count
    else:
        eval_trial_count = torch.as_tensor(eval_trial_count, dtype=torch.float32).reshape(-1)
    if eval_positive_count.shape != eval_trial_count.shape or eval_positive_count.numel() != matrix.shape[0]:
        raise ValueError("eval_positive_count, eval_trial_count, and features must describe the same number of landmarks.")

    target_name = str(target_name or "label")
    train_valid = torch.isfinite(matrix).all(dim=1) & torch.isfinite(positive_count) & torch.isfinite(trial_count) & (trial_count > 0)
    if mask is not None:
        train_valid = train_valid & torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
    eval_valid = torch.isfinite(matrix).all(dim=1) & torch.isfinite(eval_positive_count) & torch.isfinite(eval_trial_count) & (eval_trial_count > 0)
    if eval_mask is not None:
        eval_valid = eval_valid & torch.as_tensor(eval_mask, dtype=torch.bool).reshape(-1)
    heldout = eval_positive_count is not positive_count or eval_trial_count is not trial_count or eval_mask is not None
    score = torch.zeros(matrix.shape[0], dtype=torch.float32)

    def _empty_result():
        payload = {
            "feature_names": names,
            "score": score,
            "weights": torch.zeros(matrix.shape[1], dtype=torch.float32),
            "bias": 0.0,
            "calibrated_brier": 0.0,
            "calibrated_nll": 0.0,
            "spearman_calibrated_label_rate": 0.0,
            "spearman_calibrated_inlier_rate": 0.0,
            f"spearman_calibrated_{target_name}_rate": 0.0,
            "top_quartile_calibrated_label_rate": 0.0,
            "top_quartile_calibrated_inlier_rate": 0.0,
            f"top_quartile_calibrated_{target_name}_rate": 0.0,
            "bottom_quartile_calibrated_label_rate": 0.0,
            "bottom_quartile_calibrated_inlier_rate": 0.0,
            f"bottom_quartile_calibrated_{target_name}_rate": 0.0,
            "calibration_landmark_count": int(train_valid.sum().item()),
            "calibration_train_landmark_count": int(train_valid.sum().item()),
            "calibration_eval_landmark_count": int(eval_valid.sum().item()),
            "calibration_target_name": target_name,
            "calibration_heldout": bool(heldout),
        }
        return payload

    if int(train_valid.sum().item()) < 2:
        return _empty_result()

    x = matrix[train_valid]
    trials = trial_count[train_valid].clamp_min(1.0)
    positives = torch.minimum(positive_count[train_valid].clamp_min(0.0), trials)
    train_target = positives / trials
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    x_norm = (x - mean) / scale

    global_rate = (positives.sum() / trials.sum().clamp_min(1.0)).clamp(1e-4, 1.0 - 1e-4)
    weights_param = torch.zeros(x_norm.shape[1], dtype=torch.float32, requires_grad=True)
    bias_param = torch.logit(global_rate.detach()).clone().requires_grad_(True)
    optimizer = torch.optim.Adam([weights_param, bias_param], lr=float(lr))
    weight_scale = trials / trials.mean().clamp_min(1.0)

    for _ in range(int(steps)):
        optimizer.zero_grad()
        logits = x_norm @ weights_param + bias_param
        loss = (
            F.binary_cross_entropy_with_logits(logits, train_target, reduction="none") * weight_scale
        ).mean()
        if float(l2) > 0.0:
            loss = loss + float(l2) * (weights_param * weights_param).sum()
        loss.backward()
        optimizer.step()

    full_norm = (matrix - mean) / scale
    score = torch.sigmoid(full_norm @ weights_param.detach() + bias_param.detach())
    score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
    if int(eval_valid.sum().item()) < 1:
        return _empty_result()

    eval_score = score[eval_valid]
    eval_trials = eval_trial_count[eval_valid].clamp_min(1.0)
    eval_positives = torch.minimum(eval_positive_count[eval_valid].clamp_min(0.0), eval_trials)
    eval_target = eval_positives / eval_trials
    eval_weight_scale = eval_trials / eval_trials.mean().clamp_min(1.0)
    brier = ((eval_score - eval_target) ** 2 * eval_weight_scale).sum() / eval_weight_scale.sum().clamp_min(1.0)
    nll = F.binary_cross_entropy(eval_score.clamp(1e-6, 1.0 - 1e-6), eval_target, reduction="none")
    nll = (nll * eval_weight_scale).sum() / eval_weight_scale.sum().clamp_min(1.0)
    quartile_count = max(1, int(math.ceil(eval_target.numel() * 0.25)))
    score_order = torch.argsort(eval_score)
    bottom = score_order[:quartile_count]
    top = score_order[-quartile_count:]
    spearman = spearman_rank_correlation(eval_score.detach(), eval_target.detach())
    top_rate = float(eval_target[top].mean().item())
    bottom_rate = float(eval_target[bottom].mean().item())

    payload = {
        "feature_names": names,
        "score": score.detach(),
        "weights": weights_param.detach(),
        "bias": float(bias_param.detach().item()),
        "calibrated_brier": float(brier.detach().item()),
        "calibrated_nll": float(nll.detach().item()),
        "spearman_calibrated_label_rate": spearman,
        "spearman_calibrated_inlier_rate": spearman,
        f"spearman_calibrated_{target_name}_rate": spearman,
        "top_quartile_calibrated_label_rate": top_rate,
        "top_quartile_calibrated_inlier_rate": top_rate,
        f"top_quartile_calibrated_{target_name}_rate": top_rate,
        "bottom_quartile_calibrated_label_rate": bottom_rate,
        "bottom_quartile_calibrated_inlier_rate": bottom_rate,
        f"bottom_quartile_calibrated_{target_name}_rate": bottom_rate,
        "calibration_landmark_count": int(train_valid.sum().item()),
        "calibration_train_landmark_count": int(train_valid.sum().item()),
        "calibration_eval_landmark_count": int(eval_valid.sum().item()),
        "calibration_target_name": target_name,
        "calibration_heldout": bool(heldout),
    }
    return payload


def summarize_landmark_value(visible_count, matched_count, correct_count, inlier_count, utility):
    visible_count = visible_count.reshape(-1).to(dtype=torch.long)
    matched_count = matched_count.reshape(-1).to(device=visible_count.device, dtype=torch.long)
    correct_count = correct_count.reshape(-1).to(device=visible_count.device, dtype=torch.long)
    inlier_count = inlier_count.reshape(-1).to(device=visible_count.device, dtype=torch.long)
    utility = utility.reshape(-1).to(device=visible_count.device, dtype=torch.float32)

    shape = visible_count.shape
    if not (matched_count.shape == correct_count.shape == inlier_count.shape == utility.shape == shape):
        raise ValueError("Landmark value tensors must all be one-dimensional tensors with the same length.")

    landmark_count = int(visible_count.numel())
    if landmark_count == 0:
        return {
            "landmark_count": 0,
            "visible_landmark_count": 0,
            "matched_landmark_count": 0,
            "inlier_landmark_count": 0,
            "visible_total": 0,
            "matched_total": 0,
            "correct_total": 0,
            "inlier_total": 0,
            "correct_rate_mean": 0.0,
            "correct_visible_rate_mean": 0.0,
            "inlier_rate_mean": 0.0,
            "inlier_rate_weighted": 0.0,
            "spearman_utility_correct_rate": 0.0,
            "spearman_utility_inlier_rate": 0.0,
            "top_quartile_correct_rate": 0.0,
            "bottom_quartile_correct_rate": 0.0,
            "top_quartile_inlier_rate": 0.0,
            "bottom_quartile_inlier_rate": 0.0,
        }

    visible_mask = visible_count > 0
    matched_mask = matched_count > 0
    visible_count_f = visible_count.float().clamp_min(1.0)
    matched_count_f = matched_count.float().clamp_min(1.0)
    correct_visible_rate = correct_count.float() / visible_count_f
    correct_rate = correct_count.float() / matched_count_f
    inlier_rate = inlier_count.float() / matched_count_f

    if bool(visible_mask.any()):
        visible_correct_rate = correct_visible_rate[visible_mask]
        visible_utility = utility[visible_mask]
        quartile_count = max(1, int(math.ceil(visible_correct_rate.numel() * 0.25)))
        utility_order = torch.argsort(visible_utility)
        bottom_correct_idx = utility_order[:quartile_count]
        top_correct_idx = utility_order[-quartile_count:]
        correct_visible_rate_mean = float(visible_correct_rate.mean().item())
        top_correct_quartile = float(visible_correct_rate[top_correct_idx].mean().item())
        bottom_correct_quartile = float(visible_correct_rate[bottom_correct_idx].mean().item())
    else:
        correct_visible_rate_mean = 0.0
        top_correct_quartile = 0.0
        bottom_correct_quartile = 0.0

    if bool(matched_mask.any()):
        matched_inlier_rate = inlier_rate[matched_mask]
        matched_correct_rate = correct_rate[matched_mask]
        matched_utility = utility[matched_mask]
        quartile_count = max(1, int(math.ceil(matched_inlier_rate.numel() * 0.25)))
        utility_order = torch.argsort(matched_utility)
        bottom_idx = utility_order[:quartile_count]
        top_idx = utility_order[-quartile_count:]
        correct_rate_mean = float(matched_correct_rate.mean().item())
        inlier_rate_mean = float(matched_inlier_rate.mean().item())
        top_quartile = float(matched_inlier_rate[top_idx].mean().item())
        bottom_quartile = float(matched_inlier_rate[bottom_idx].mean().item())
    else:
        correct_rate_mean = 0.0
        inlier_rate_mean = 0.0
        top_quartile = 0.0
        bottom_quartile = 0.0

    matched_total = int(matched_count.sum().item())
    return {
        "landmark_count": landmark_count,
        "visible_landmark_count": int((visible_count > 0).sum().item()),
        "matched_landmark_count": int(matched_mask.sum().item()),
        "inlier_landmark_count": int((inlier_count > 0).sum().item()),
        "visible_total": int(visible_count.sum().item()),
        "matched_total": matched_total,
        "correct_total": int(correct_count.sum().item()),
        "inlier_total": int(inlier_count.sum().item()),
        "correct_rate_mean": correct_rate_mean,
        "correct_visible_rate_mean": correct_visible_rate_mean,
        "inlier_rate_mean": inlier_rate_mean,
        "inlier_rate_weighted": float(inlier_count.sum().item() / matched_total) if matched_total > 0 else 0.0,
        "spearman_utility_correct_rate": spearman_rank_correlation(utility, correct_visible_rate, visible_mask),
        "spearman_utility_inlier_rate": spearman_rank_correlation(utility, inlier_rate, matched_mask),
        "top_quartile_correct_rate": top_correct_quartile,
        "bottom_quartile_correct_rate": bottom_correct_quartile,
        "top_quartile_inlier_rate": top_quartile,
        "bottom_quartile_inlier_rate": bottom_quartile,
    }


def collect_projected_descriptor_pairs(
    gaussians,
    query_feature_map,
    pose_gt_w2c,
    fovx,
    fovy,
    landmark_indices,
    target_depth=None,
    target_alpha=None,
    alpha_threshold=0.2,
    depth_abs_tolerance=1e-3,
    depth_rel_tolerance=0.01,
    max_landmarks=2048,
    baseline_gaussians=None,
):
    device = query_feature_map.device
    dtype = query_feature_map.dtype
    height, width = query_feature_map.shape[-2:]
    landmark_indices = landmark_indices.to(device=gaussians.get_xyz.device, dtype=torch.long).reshape(-1)
    xyz = gaussians.get_xyz[landmark_indices].to(device=device, dtype=dtype)
    K = make_intrinsics_from_fov(fovx, fovy, width, height, device=device, dtype=dtype)
    uv, depth, valid = project_landmarks_to_query(
        xyz,
        K,
        pose_gt_w2c.to(device=device, dtype=dtype),
        height,
        width,
    )
    valid = filter_depth_consistent_landmarks(
        uv,
        depth,
        valid,
        target_depth=target_depth,
        target_alpha=target_alpha,
        alpha_threshold=alpha_threshold,
        abs_tolerance=depth_abs_tolerance,
        rel_tolerance=depth_rel_tolerance,
    )
    keep = _limit_valid_indices(valid, max_landmarks)
    if keep.numel() == 0:
        empty = query_feature_map.new_empty((0, query_feature_map.shape[0]))
        return {
            "full_idx": landmark_indices.new_empty(0),
            "uv": uv.new_empty((0, 2)),
            "gaussian_features": empty,
            "query_features": empty,
            "baseline_features": empty if baseline_gaussians is not None else None,
        }

    full_idx = landmark_indices[keep]
    query_features = bilinear_sample_features(query_feature_map.detach(), uv[keep])
    gaussian_features = gaussians.get_loc_feature[full_idx].reshape(keep.numel(), -1)
    baseline_features = None
    if baseline_gaussians is not None:
        baseline_features = baseline_gaussians.get_loc_feature[full_idx].reshape(keep.numel(), -1)
    return {
        "full_idx": full_idx,
        "uv": uv[keep].detach(),
        "gaussian_features": gaussian_features,
        "query_features": query_features,
        "baseline_features": baseline_features,
    }
