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
            "inlier_rate_mean": 0.0,
            "inlier_rate_weighted": 0.0,
            "spearman_utility_inlier_rate": 0.0,
            "top_quartile_inlier_rate": 0.0,
            "bottom_quartile_inlier_rate": 0.0,
        }

    matched_mask = matched_count > 0
    matched_count_f = matched_count.float().clamp_min(1.0)
    correct_rate = correct_count.float() / matched_count_f
    inlier_rate = inlier_count.float() / matched_count_f

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
        "inlier_rate_mean": inlier_rate_mean,
        "inlier_rate_weighted": float(inlier_count.sum().item() / matched_total) if matched_total > 0 else 0.0,
        "spearman_utility_inlier_rate": spearman_rank_correlation(utility, inlier_rate, matched_mask),
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
