from __future__ import annotations

from dataclasses import dataclass

import torch


METRIC_ANCHOR_MIN_CONSISTENCY = 0.05


def _as_float(value):
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def percentile_rank(value):
    value = _as_float(value)
    if value.numel() == 0:
        return value
    finite = torch.isfinite(value)
    result = torch.zeros_like(value)
    if not bool(finite.any()):
        return result
    valid = value[finite]
    unique, inverse, counts = torch.unique(
        valid, sorted=True, return_inverse=True, return_counts=True
    )
    del unique
    starts = counts.cumsum(0) - counts
    # Equal evidence must receive equal reliability. Using stable ordinal
    # ranks here leaks the arbitrary landmark ordering into sanitization.
    average_positions = starts + 0.5 * (counts - 1)
    denominator = max(valid.numel() - 1, 1)
    rank = average_positions[inverse].to(valid.dtype) / float(denominator)
    result[finite] = rank
    return result


def wilson_lower_bound(successes, trials, z=1.96):
    successes = _as_float(successes)
    trials = _as_float(trials)
    probability = successes / trials.clamp_min(1.0)
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / trials.clamp_min(1.0)
    center = probability + z2 / (2.0 * trials.clamp_min(1.0))
    radius = float(z) * torch.sqrt(
        (
            probability * (1.0 - probability)
            + z2 / (4.0 * trials.clamp_min(1.0))
        )
        / trials.clamp_min(1.0)
    )
    lower = (center - radius) / denominator
    return torch.where(trials > 0, lower.clamp(0.0, 1.0), torch.zeros_like(lower))


def geometric_mean(components, eps=1e-4):
    values = torch.stack(
        [torch.as_tensor(value, dtype=torch.float32) for value in components],
        dim=0,
    ).clamp(min=float(eps), max=1.0)
    return values.log().mean(dim=0).exp()


@dataclass(frozen=True)
class SanitizationScores:
    localization_reliability: torch.Tensor
    geometry_reliability: torch.Tensor
    components: dict
    state: torch.Tensor


def binary_ranking_metrics(outlier_score, outlier_label):
    score = _as_float(outlier_score)
    label = torch.as_tensor(outlier_label, dtype=torch.bool).reshape(-1)
    if score.numel() != label.numel():
        raise ValueError("score and label sizes differ")
    positive_count = int(label.sum())
    negative_count = int((~label).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUROC/AUPRC require both positive and negative labels")
    order = torch.argsort(score, descending=True, stable=True)
    ordered_label = label[order].float()
    true_positive = ordered_label.cumsum(0)
    false_positive = (1.0 - ordered_label).cumsum(0)
    recall = true_positive / float(positive_count)
    false_positive_rate = false_positive / float(negative_count)
    precision = true_positive / torch.arange(
        1, score.numel() + 1, dtype=torch.float32
    )
    recall_with_origin = torch.cat((torch.zeros(1), recall))
    fpr_with_origin = torch.cat((torch.zeros(1), false_positive_rate))
    precision_with_origin = torch.cat((torch.ones(1), precision))
    auroc = torch.trapz(recall_with_origin, fpr_with_origin)
    auprc = (
        (recall_with_origin[1:] - recall_with_origin[:-1])
        * precision_with_origin[1:]
    ).sum()
    return {"auroc": float(auroc), "auprc": float(auprc)}


def build_sanitization_scores(statistics, geometry_evidence):
    observations = _as_float(
        statistics.get(
            "effective_observation_count", statistics["observation_count"]
        )
    )
    correct = _as_float(statistics["correct_count"])
    source_precision = wilson_lower_bound(correct, observations)
    harmful_switch = _as_float(
        statistics["cross_view_top1_harmful_switch_rate"]
    )
    identity_stability = (1.0 - harmful_switch).clamp(0.0, 1.0)
    target_incoming = _as_float(
        statistics.get("target_incoming_count", observations)
    )
    target_false = _as_float(
        statistics.get("target_false_hit_count", torch.zeros_like(observations))
    )
    target_precision = wilson_lower_bound(
        (target_incoming - target_false).clamp_min(0.0),
        target_incoming,
    )
    margin_quality = percentile_rank(statistics["margin"])
    observation_support = percentile_rank(torch.log1p(observations))
    localization_reliability = geometric_mean(
        (
            source_precision,
            identity_stability,
            target_precision,
            0.25 + 0.75 * margin_quality,
            0.25 + 0.75 * observation_support,
        )
    )

    visibility = _as_float(geometry_evidence["raster_visibility_count"])
    mvinit = _as_float(geometry_evidence["mvinit_observation_count"])
    opacity = _as_float(geometry_evidence["opacity"]).clamp(0.0, 1.0)
    planarity = _as_float(geometry_evidence["planarity"])
    scaling = torch.as_tensor(
        geometry_evidence["scaling"], dtype=torch.float32
    )
    log_scale = scaling.clamp_min(1e-8).log().mean(dim=1)
    median = log_scale.median()
    mad = (log_scale - median).abs().median().clamp_min(1e-3)
    scale_quality = torch.exp(
        -((log_scale - median).abs() / (3.0 * mad)).clamp_max(20.0)
    )
    reprojection_quality = (
        1.0 - percentile_rank(statistics["reprojection_error"])
    ).clamp(0.0, 1.0)
    # Statistics use zero-filled accumulators. A zero with no observations is
    # missing evidence, not perfect reprojection.
    reprojection_quality = torch.where(
        observations > 0,
        reprojection_quality,
        torch.zeros_like(reprojection_quality),
    )
    rgb_center_mahalanobis = _as_float(
        geometry_evidence.get(
            "rgb_center_offset_mahalanobis",
            torch.zeros_like(visibility),
        )
    )
    rgb_center_consistency = torch.exp(
        -0.5 * rgb_center_mahalanobis.clamp(0.0, 20.0)
    )
    rgb_center_offset_m = _as_float(
        geometry_evidence.get(
            "rgb_center_offset_m",
            torch.zeros_like(visibility),
        )
    )
    # A covariance-only score is unsafe for volumetric 3DGS: a large floater
    # can make a decimetre displacement look statistically small.
    rgb_center_metric_consistency = torch.exp(
        -0.5 * (rgb_center_offset_m / 0.02).square().clamp(0.0, 20.0)
    )
    visibility_support = percentile_rank(torch.log1p(visibility))
    mv_support = percentile_rank(torch.log1p(mvinit))
    opacity_quality = percentile_rank(opacity)
    if str(geometry_evidence.get("gaussian_type", "")).lower() == "3dgs":
        planarity_quality = 1.0 - percentile_rank(planarity)
    else:
        planarity_quality = torch.ones_like(planarity)
    geometry_reliability = geometric_mean(
        (
            0.1 + 0.9 * visibility_support,
            0.1 + 0.9 * mv_support,
            0.1 + 0.9 * opacity_quality,
            0.1 + 0.9 * scale_quality,
            0.1 + 0.9 * planarity_quality,
            0.1 + 0.9 * reprojection_quality,
            0.1 + 0.9 * rgb_center_consistency,
            0.1 + 0.9 * rgb_center_metric_consistency,
        )
    )

    loc_q20, loc_q50, loc_q70 = torch.quantile(
        localization_reliability, torch.tensor([0.2, 0.5, 0.7])
    )
    geo_q20, geo_q30 = torch.quantile(
        geometry_reliability, torch.tensor([0.2, 0.3])
    )
    # 0: localization-excluded, 1: keep, 2: repairable, 3: reject.
    state = torch.zeros_like(observations, dtype=torch.int64)
    state[
        (localization_reliability >= loc_q50)
        & (geometry_reliability >= geo_q20)
    ] = 1
    state[
        (localization_reliability >= loc_q70)
        & (geometry_reliability < geo_q30)
        & (visibility >= 4)
    ] = 2
    state[
        (localization_reliability < loc_q20)
        & (geometry_reliability < geo_q20)
    ] = 3
    return SanitizationScores(
        localization_reliability=localization_reliability,
        geometry_reliability=geometry_reliability,
        components={
            "source_precision_wilson": source_precision,
            "identity_stability": identity_stability,
            "target_precision_wilson": target_precision,
            "margin_quality": margin_quality,
            "observation_support": observation_support,
            "visibility_support": visibility_support,
            "mv_support": mv_support,
            "opacity_quality": opacity_quality,
            "scale_quality": scale_quality,
            "planarity_quality": planarity_quality,
            "reprojection_quality": reprojection_quality,
            "rgb_center_consistency": rgb_center_consistency,
            "rgb_center_metric_consistency": rgb_center_metric_consistency,
        },
        state=state,
    )


def _coverage_reserve(score, mean_uv, mean_depth, count):
    score = _as_float(score)
    mean_uv = torch.as_tensor(mean_uv, dtype=torch.float32)
    depth_rank = percentile_rank(mean_depth)
    row = (mean_uv[:, 1].clamp(0.0, 0.9999) * 4).long()
    col = (mean_uv[:, 0].clamp(0.0, 0.9999) * 4).long()
    depth_bin = (depth_rank.clamp(0.0, 0.9999) * 4).long()
    group = row * 16 + col * 4 + depth_bin
    selected = []
    order = torch.argsort(score, descending=True, stable=True)
    group_counts = torch.zeros(64, dtype=torch.int64)
    quota = max((int(count) + 63) // 64, 1)
    for index in order.tolist():
        group_index = int(group[index])
        if int(group_counts[group_index]) >= quota:
            continue
        selected.append(index)
        group_counts[group_index] += 1
        if len(selected) >= int(count):
            break
    return torch.as_tensor(selected, dtype=torch.long)


def select_sanitized_landmarks(scores, statistics, *, mode, budget):
    if mode not in {"loc", "loc_geo", "loc_geo_coverage"}:
        raise ValueError(f"Unsupported sanitization mode: {mode}")
    count = int(scores.localization_reliability.numel())
    budget = min(max(int(budget), 1), count)
    loc = scores.localization_reliability
    geo = scores.geometry_reliability
    if mode == "loc":
        utility = loc
        eligible = torch.ones(count, dtype=torch.bool)
        preferred = eligible
    else:
        utility = loc * (0.5 + 0.5 * geo)
        metric_anchor_consistency = scores.components.get(
            "rgb_center_metric_consistency",
            torch.ones(count, dtype=geo.dtype),
        )
        eligible = (
            metric_anchor_consistency
            >= float(METRIC_ANCHOR_MIN_CONSISTENCY)
        )
        preferred = eligible & (geo >= torch.quantile(geo, 0.1))
    if int(eligible.sum()) < budget:
        raise ValueError(
            "Hard localization-anchor eligibility leaves fewer landmarks "
            f"than the requested budget: eligible={int(eligible.sum())} "
            f"budget={budget}"
        )
    preferred_indices = torch.nonzero(
        preferred, as_tuple=False
    ).reshape(-1)
    preferred_order = torch.argsort(
        utility[preferred_indices], descending=True, stable=True
    )
    fallback_indices = torch.nonzero(
        eligible & ~preferred, as_tuple=False
    ).reshape(-1)
    fallback_order = torch.argsort(
        utility[fallback_indices], descending=True, stable=True
    )
    ranked = torch.cat(
        (
            preferred_indices[preferred_order],
            fallback_indices[fallback_order],
        )
    )
    if mode != "loc_geo_coverage":
        return ranked[:budget]

    reserve_count = min(max(int(round(0.1 * budget)), 64), budget)
    reserve = _coverage_reserve(
        utility,
        statistics["mean_uv"],
        statistics["mean_depth"],
        reserve_count,
    )
    reserve = reserve[eligible[reserve]]
    chosen = torch.zeros(count, dtype=torch.bool)
    chosen[reserve] = True
    fill = ranked[~chosen[ranked]][: budget - reserve.numel()]
    return torch.cat((reserve, fill))
