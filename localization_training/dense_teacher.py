from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from gaussian_renderer import render_from_pose_gsplat
from localization_training.correspondence import (
    bilinear_sample_features,
    build_target_correspondences,
    make_pixel_grid,
    sample_valid_anchors,
)
from localization_training.dense_distill import (
    dense_to_sparse_kl,
    dense_sparse_miss_hit_rank_loss,
    gaussian_teacher_distribution,
    responsibility_entropy,
    responsibility_reconstruction_cosine,
    responsibility_reconstruction_metrics,
)
from localization_training.losses import fine_reprojection_loss, symmetric_descriptor_loss
from localization_training.pose_information import compute_pose_information


@dataclass
class DenseTeacherOutput:
    loss: torch.Tensor
    desc_loss: torch.Tensor
    reproj_loss: torch.Tensor
    kl_loss: torch.Tensor
    stats: dict
    render_pkg: dict
    anchor_count: int
    diagnostics: dict = field(default_factory=dict)
    rank_loss: torch.Tensor = None

    @property
    def loc_viewspace_points(self):
        return self.render_pkg.get("loc_viewspace_points")

    @property
    def loc_visible_idx(self):
        return self.render_pkg.get("loc_visible_idx")

    @property
    def loc_radii(self):
        return self.render_pkg.get("loc_radii")


def _as_pose_tensor(pose, device):
    if isinstance(pose, torch.Tensor):
        return pose.to(device=device, dtype=torch.float32)
    return torch.tensor(pose, device=device, dtype=torch.float32)


def _flatten_depth(depth):
    if depth is None:
        return None
    while depth.dim() > 2:
        depth = depth.squeeze(0)
    return depth


def _flatten_alpha(alpha):
    if alpha is None:
        return None
    alpha = alpha.squeeze()
    if alpha.dim() == 3:
        alpha = alpha[..., 0]
    return alpha


def _normalize_responsibility(weights):
    weights = torch.as_tensor(weights, dtype=torch.float32)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _as_anchor_vector(value, count, device, dtype):
    value = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if value.numel() == 1:
        value = value.expand(count)
    return value[:count]


def aggregate_dense_anchor_stats(
    visible_idx,
    contributor_ids,
    responsibility_weights,
    query_features,
    fine_stats,
    margin,
    information,
    repeatability_threshold=0.25,
):
    visible_idx = torch.as_tensor(visible_idx, dtype=torch.long, device=query_features.device).reshape(-1)
    contributor_ids = torch.as_tensor(contributor_ids, dtype=torch.long, device=query_features.device)
    if contributor_ids.ndim == 1:
        contributor_ids = contributor_ids[:, None]
    weights = _normalize_responsibility(responsibility_weights).to(device=query_features.device, dtype=query_features.dtype)
    if weights.ndim == 1:
        weights = weights[:, None]
    if contributor_ids.shape != weights.shape:
        raise ValueError(
            "contributor_ids and responsibility_weights must have the same shape, "
            f"got {tuple(contributor_ids.shape)} and {tuple(weights.shape)}."
        )
    if contributor_ids.shape[0] != query_features.shape[0]:
        raise ValueError(
            "contributor_ids must have one row per dense anchor, "
            f"got {contributor_ids.shape[0]} rows for {query_features.shape[0]} anchors."
        )

    visible_count = visible_idx.numel()
    feature_dim = query_features.reshape(query_features.shape[0], -1).shape[1]
    stats = {
        "positive_prob": query_features.new_zeros(visible_count),
        "margin": query_features.new_zeros(visible_count),
        "entropy": query_features.new_zeros(visible_count),
        "reproj_error": query_features.new_zeros(visible_count),
        "information": query_features.new_zeros(visible_count),
        "repeatability": query_features.new_zeros(visible_count),
        "prototype": query_features.new_zeros((visible_count, feature_dim)),
        "update_mask": torch.zeros(visible_count, dtype=torch.bool, device=query_features.device),
        "responsibility_weight": query_features.new_zeros(visible_count),
    }
    if visible_count == 0 or contributor_ids.numel() == 0:
        return stats

    max_index = int(max(visible_idx.max().item(), contributor_ids.clamp_min(0).max().item())) if contributor_ids.numel() else int(visible_idx.max().item())
    lookup = torch.full((max_index + 1,), -1, dtype=torch.long, device=query_features.device)
    lookup[visible_idx] = torch.arange(visible_count, dtype=torch.long, device=query_features.device)
    in_range = (contributor_ids >= 0) & (contributor_ids < lookup.numel())
    local = torch.full_like(contributor_ids, -1)
    local[in_range] = lookup[contributor_ids[in_range]]
    valid = local >= 0
    if not valid.any():
        return stats

    weights = weights * valid.to(dtype=weights.dtype)
    anchor_count = query_features.shape[0]
    flat_local = local.reshape(-1)
    flat_weights = weights.reshape(-1)
    valid_flat = flat_local >= 0
    flat_local = flat_local[valid_flat]
    flat_weights = flat_weights[valid_flat]

    denom = query_features.new_zeros(visible_count)
    denom.scatter_add_(0, flat_local, flat_weights)
    update_mask = denom > 0
    stats["update_mask"] = update_mask
    stats["responsibility_weight"] = denom

    anchor_values = {
        "positive_prob": _as_anchor_vector(fine_stats.get("positive_prob", 0.0), anchor_count, query_features.device, query_features.dtype),
        "entropy": _as_anchor_vector(fine_stats.get("entropy", 0.0), anchor_count, query_features.device, query_features.dtype),
        "reproj_error": _as_anchor_vector(fine_stats.get("reproj_error", 0.0), anchor_count, query_features.device, query_features.dtype),
        "margin": _as_anchor_vector(margin, anchor_count, query_features.device, query_features.dtype),
        "information": _as_anchor_vector(information, anchor_count, query_features.device, query_features.dtype),
    }
    anchor_values["repeatability"] = (anchor_values["positive_prob"] > float(repeatability_threshold)).to(dtype=query_features.dtype)

    repeated_weights = weights
    for key, value in anchor_values.items():
        weighted = (value[:, None] * repeated_weights).reshape(-1)[valid_flat]
        stats[key].scatter_add_(0, flat_local, weighted)
        stats[key][update_mask] = stats[key][update_mask] / denom[update_mask].clamp_min(1e-8)

    flat_features = query_features.reshape(anchor_count, feature_dim)
    feature_contrib = (flat_features[:, None, :] * repeated_weights[..., None]).reshape(-1, feature_dim)[valid_flat]
    stats["prototype"].scatter_add_(0, flat_local[:, None].expand(-1, feature_dim), feature_contrib)
    stats["prototype"][update_mask] = F.normalize(stats["prototype"][update_mask] / denom[update_mask, None].clamp_min(1e-8), p=2, dim=-1)
    return stats


def dense_responsibility_kl_loss(
    query_features,
    rendered_features,
    bank_features,
    contributor_ids,
    responsibility_weights,
    dense_temperature=0.07,
    sparse_temperature=0.07,
    anchor_weights=None,
):
    query = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1)
    rendered = F.normalize(rendered_features.reshape(rendered_features.shape[0], -1), p=2, dim=-1)
    if query.numel() == 0 or rendered.numel() == 0 or bank_features.numel() == 0:
        return query_features.new_tensor(0.0)
    dense_logits = query @ rendered.T / max(float(dense_temperature), 1e-6)
    dense_probs = F.softmax(dense_logits, dim=1)
    teacher = gaussian_teacher_distribution(
        dense_probs,
        contributor_ids,
        responsibility_weights,
        bank_size=bank_features.reshape(bank_features.shape[0], -1).shape[0],
    )
    return dense_to_sparse_kl(
        query_features,
        bank_features,
        teacher,
        temperature=sparse_temperature,
        anchor_weights=anchor_weights,
    )


def dense_responsibility_rank_loss(
    query_features,
    rendered_features,
    bank_features,
    contributor_ids,
    responsibility_weights,
    dense_temperature=0.07,
    sparse_temperature=0.07,
    anchor_weights=None,
    teacher_confidence_threshold=0.0,
    miss_topk=1,
    margin=0.2,
):
    query = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1)
    rendered = F.normalize(rendered_features.reshape(rendered_features.shape[0], -1), p=2, dim=-1)
    if query.numel() == 0 or rendered.numel() == 0 or bank_features.numel() == 0:
        return query_features.new_tensor(0.0), {
            "dense_rank_anchor_count": 0,
            "dense_rank_eligible_anchor_count": 0,
            "dense_rank_sparse_hit_count": 0,
            "dense_rank_sparse_miss_count": 0,
            "dense_rank_low_confidence_count": 0,
            "dense_rank_teacher_confidence_mean": 0.0,
        }
    dense_logits = query @ rendered.T / max(float(dense_temperature), 1e-6)
    dense_probs = F.softmax(dense_logits, dim=1)
    teacher = gaussian_teacher_distribution(
        dense_probs,
        contributor_ids,
        responsibility_weights,
        bank_size=bank_features.reshape(bank_features.shape[0], -1).shape[0],
    )
    return dense_sparse_miss_hit_rank_loss(
        query_features,
        bank_features,
        teacher,
        temperature=sparse_temperature,
        anchor_weights=anchor_weights,
        teacher_confidence_threshold=teacher_confidence_threshold,
        miss_topk=miss_topk,
        margin=margin,
        return_diagnostics=True,
    )


def _selective_dense_anchor_weights(
    query_features,
    rendered_features,
    visible_bank_features,
    compact_contributor_ids,
    responsibility_weights,
    fine_stats,
    dense_pose_weight=1.0,
    attr_cosine_threshold=-1.0,
    attr_entropy_threshold=-1.0,
    min_positive_prob=-1.0,
    max_reproj_error=-1.0,
    min_eligible_anchors=1,
):
    weights = query_features.new_ones(query_features.shape[0]) * float(dense_pose_weight)
    if float(dense_pose_weight) <= 0.0:
        return weights.zero_(), {
            "dense_kl_pose_weight": float(dense_pose_weight),
            "dense_kl_eligible_anchor_count": 0,
            "dense_kl_anchor_count": int(query_features.shape[0]),
            "dense_kl_anchor_weight_mean": 0.0,
        }

    diagnostics = {
        "dense_kl_pose_weight": float(dense_pose_weight),
        "dense_kl_anchor_count": int(query_features.shape[0]),
    }
    if float(attr_cosine_threshold) >= 0.0:
        cosine = responsibility_reconstruction_cosine(
            rendered_features.detach(),
            visible_bank_features.detach().to(device=query_features.device, dtype=query_features.dtype),
            compact_contributor_ids,
            responsibility_weights,
        ).to(device=query_features.device, dtype=query_features.dtype)
        weights = weights * (cosine >= float(attr_cosine_threshold)).to(dtype=weights.dtype)
        valid_cos = cosine[torch.isfinite(cosine) & (cosine >= -1.0)]
        diagnostics["dense_kl_attr_cosine_mean"] = float(valid_cos.mean().item()) if valid_cos.numel() else 0.0
    if float(attr_entropy_threshold) >= 0.0:
        entropy = responsibility_entropy(responsibility_weights).to(device=query_features.device, dtype=query_features.dtype)
        weights = weights * (entropy <= float(attr_entropy_threshold)).to(dtype=weights.dtype)
        finite_entropy = entropy[torch.isfinite(entropy)]
        diagnostics["dense_kl_attr_entropy_mean"] = float(finite_entropy.mean().item()) if finite_entropy.numel() else 0.0
    if float(min_positive_prob) >= 0.0 and "positive_prob" in fine_stats:
        positive_prob = torch.as_tensor(fine_stats["positive_prob"], device=query_features.device, dtype=query_features.dtype).reshape(-1)
        weights = weights * (positive_prob[: weights.numel()] >= float(min_positive_prob)).to(dtype=weights.dtype)
    if float(max_reproj_error) >= 0.0 and "reproj_error" in fine_stats:
        reproj_error = torch.as_tensor(fine_stats["reproj_error"], device=query_features.device, dtype=query_features.dtype).reshape(-1)
        weights = weights * (reproj_error[: weights.numel()] <= float(max_reproj_error)).to(dtype=weights.dtype)

    eligible = int((weights > 0).sum().item())
    if eligible < int(min_eligible_anchors):
        weights = weights.zero_()
        eligible = 0
    diagnostics["dense_kl_eligible_anchor_count"] = eligible
    diagnostics["dense_kl_anchor_weight_mean"] = float(weights.mean().item()) if weights.numel() else 0.0
    return weights, diagnostics


def _compact_contributor_ids(contributor_ids, bank_indices):
    contributor_ids = torch.as_tensor(contributor_ids, dtype=torch.long, device=bank_indices.device)
    bank_indices = torch.as_tensor(bank_indices, dtype=torch.long, device=bank_indices.device).reshape(-1)
    compact = torch.full_like(contributor_ids, -1)
    if contributor_ids.numel() == 0 or bank_indices.numel() == 0:
        return compact
    max_index = int(max(contributor_ids.clamp_min(0).max().item(), bank_indices.max().item()))
    lookup = torch.full((max_index + 1,), -1, dtype=torch.long, device=bank_indices.device)
    lookup[bank_indices] = torch.arange(bank_indices.numel(), dtype=torch.long, device=bank_indices.device)
    in_range = (contributor_ids >= 0) & (contributor_ids < lookup.numel())
    compact[in_range] = lookup[contributor_ids[in_range]]
    return compact


def _sample_pixel_contributors(
    render_pkg,
    anchor_idx,
    render_uv,
    image_height,
    image_width,
    max_contributors=8,
    render_depth=None,
    opacity_weight=0.0,
    depth_consistency_weight=0.0,
):
    full = render_pkg.get("loc_contributor_full_idx")
    weights = render_pkg.get("loc_responsibility_weights")
    if full is not None and weights is not None:
        full = torch.as_tensor(full, device=render_uv.device, dtype=torch.long)
        weights = torch.as_tensor(weights, device=render_uv.device, dtype=render_uv.dtype)
        if full.dim() >= 3:
            full = full.reshape(-1, full.shape[-1])
        if weights.dim() >= 3:
            weights = weights.reshape(-1, weights.shape[-1])
        return full[anchor_idx], weights[anchor_idx]

    local = render_pkg.get("loc_contributor_idx")
    weights = render_pkg.get("loc_responsibility_weights")
    visible_idx = render_pkg.get("loc_visible_idx")
    if local is not None and weights is not None and visible_idx is not None:
        local = torch.as_tensor(local, device=render_uv.device, dtype=torch.long)
        weights = torch.as_tensor(weights, device=render_uv.device, dtype=render_uv.dtype)
        if local.dim() >= 3:
            local = local.reshape(-1, local.shape[-1])
        if weights.dim() >= 3:
            weights = weights.reshape(-1, weights.shape[-1])
        visible_idx = visible_idx.to(device=render_uv.device, dtype=torch.long).reshape(-1)
        local = local[anchor_idx].clamp(-1, max(visible_idx.numel() - 1, 0))
        full_idx = torch.full_like(local, -1)
        valid = local >= 0
        full_idx[valid] = visible_idx[local[valid]]
        return full_idx, weights[anchor_idx]

    means2d = render_pkg.get("loc_viewspace_points")
    radii = render_pkg.get("loc_radii")
    conics = render_pkg.get("loc_conics")
    depths = render_pkg.get("loc_depths")
    opacities = render_pkg.get("loc_opacities")
    if visible_idx is None or means2d is None or radii is None:
        return None, None
    visible_idx = visible_idx.to(device=render_uv.device, dtype=torch.long).reshape(-1)
    means2d = means2d.to(device=render_uv.device, dtype=render_uv.dtype)
    if means2d.dim() == 3:
        means2d = means2d.squeeze(0)
    means2d = means2d.reshape(-1, 2)
    radii = radii.to(device=render_uv.device, dtype=render_uv.dtype).reshape(-1)
    if means2d.shape[0] != visible_idx.numel() or radii.numel() != visible_idx.numel() or visible_idx.numel() == 0:
        return None, None

    k = min(int(max_contributors), visible_idx.numel())
    dist = torch.cdist(render_uv.to(dtype=means2d.dtype), means2d)
    score = None
    if conics is not None:
        conics = torch.as_tensor(conics, device=render_uv.device, dtype=render_uv.dtype)
        if conics.dim() >= 3:
            conics = conics.reshape(-1, conics.shape[-1])
        if conics.shape[0] == visible_idx.numel() and conics.shape[-1] >= 3:
            dx = render_uv[:, None, 0].to(dtype=conics.dtype) - means2d[None, :, 0].to(dtype=conics.dtype)
            dy = render_uv[:, None, 1].to(dtype=conics.dtype) - means2d[None, :, 1].to(dtype=conics.dtype)
            maha = conics[None, :, 0] * dx.square() + 2.0 * conics[None, :, 1] * dx * dy + conics[None, :, 2] * dy.square()
            valid_conic = torch.isfinite(maha) & torch.isfinite(conics).all(dim=-1)[None]
            score = torch.exp(-0.5 * maha.clamp_min(0.0)).masked_fill(~valid_conic, 0.0).to(dtype=render_uv.dtype)
    if score is None:
        sigma = radii.clamp_min(1.0)
        score = torch.exp(-0.5 * (dist / sigma[None]).square())
    score = score.masked_fill(radii[None] <= 0, 0.0)
    if opacities is not None and float(opacity_weight) > 0.0:
        opacities = torch.as_tensor(opacities, device=render_uv.device, dtype=render_uv.dtype).reshape(-1)
        if opacities.numel() == visible_idx.numel():
            score = score * opacities.clamp_min(0.0)[None].pow(float(opacity_weight))
    if depths is not None and render_depth is not None and float(depth_consistency_weight) > 0.0:
        depths = torch.as_tensor(depths, device=render_uv.device, dtype=render_uv.dtype).reshape(-1)
        anchor_depth = torch.as_tensor(render_depth, device=render_uv.device, dtype=render_uv.dtype).reshape(-1)
        if depths.numel() == visible_idx.numel() and anchor_depth.numel() == render_uv.shape[0]:
            sigma_depth = (anchor_depth.abs() * 0.05).clamp_min(0.05)
            depth_score = 1.0 / (1.0 + (depths[None] - anchor_depth[:, None]).abs() / sigma_depth[:, None])
            score = score * depth_score.pow(float(depth_consistency_weight))
    top_score, top_local = torch.topk(score, k=k, dim=1)
    empty = top_score.sum(dim=1) <= 1e-12
    if empty.any():
        nearest = dist[empty].argmin(dim=1)
        top_local[empty, 0] = nearest
        top_score[empty, 0] = 1.0
    return visible_idx[top_local], top_score


def dense_localization_teacher(
    gaussians,
    query_feature_map,
    pose_init_w2c,
    pose_gt_w2c,
    fovx,
    fovy,
    width,
    height,
    background,
    anchor_count=512,
    alpha_threshold=0.2,
    desc_temperature=0.07,
    fine_temperature=0.05,
    fine_window_radius=4,
    dense_kl_weight=0.0,
    dense_kl_temperature=0.07,
    dense_rank_weight=0.0,
    dense_rank_margin=0.2,
    dense_rank_teacher_confidence=0.0,
    dense_rank_miss_topk=1,
    responsibility_topk=32,
    responsibility_opacity_weight=0.0,
    responsibility_depth_weight=0.0,
    dense_pose_weight=1.0,
    attr_cosine_threshold=-1.0,
    attr_entropy_threshold=-1.0,
    min_positive_prob=-1.0,
    max_reproj_error=-1.0,
    min_eligible_anchors=1,
    norm_feat_bf_render=True,
    use_loc_opacity=True,
    min_anchors=8,
    rasterize_args=None,
):
    """Run one differentiable dense localization teacher episode."""
    rasterize_args = rasterize_args or {}
    device = query_feature_map.device
    pose_init_w2c = _as_pose_tensor(pose_init_w2c, device)
    pose_gt_w2c = _as_pose_tensor(pose_gt_w2c, device)

    render_pkg = render_from_pose_gsplat(
        gaussians,
        pose_init_w2c,
        fovx,
        fovy,
        width,
        height,
        bg_color=background,
        render_mode="RGB+ED",
        rgb_only=False,
        norm_feat_bf_render=norm_feat_bf_render,
        return_loc_meta=True,
        use_loc_opacity=use_loc_opacity,
        **rasterize_args,
    )
    rendered_feature_map = render_pkg["feature_map"]
    depth = _flatten_depth(render_pkg.get("depth"))
    alpha = _flatten_alpha(render_pkg.get("loc_alphas", render_pkg.get("alphas")))

    zero = query_feature_map.new_tensor(0.0)
    if rendered_feature_map is None or depth is None:
        return DenseTeacherOutput(zero, zero, zero, zero, {}, render_pkg, 0)

    h, w = depth.shape[-2:]
    grid = make_pixel_grid(h, w, device=device, dtype=torch.float32)
    valid = torch.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
    if alpha is not None and alpha.numel() == h * w:
        valid = valid & (alpha.reshape(-1) > alpha_threshold)

    anchor_idx = sample_valid_anchors(valid.reshape(h, w), anchor_count)
    if anchor_idx.numel() < min_anchors:
        return DenseTeacherOutput(zero, zero, zero, zero, {}, render_pkg, int(anchor_idx.numel()))

    render_uv = grid[anchor_idx]
    render_depth = depth.reshape(-1)[anchor_idx]
    targets = build_target_correspondences(render_uv, render_depth.detach(), render_pkg["loc_K"], pose_init_w2c, pose_gt_w2c)
    target_uv = targets["target_uv"].detach()
    target_valid = targets["valid"]
    target_valid = target_valid & (target_uv[:, 0] >= 0) & (target_uv[:, 0] <= query_feature_map.shape[2] - 1)
    target_valid = target_valid & (target_uv[:, 1] >= 0) & (target_uv[:, 1] <= query_feature_map.shape[1] - 1)

    if target_valid.sum() < min_anchors:
        return DenseTeacherOutput(zero, zero, zero, zero, {}, render_pkg, int(target_valid.sum().item()))

    anchor_idx = anchor_idx[target_valid]
    render_uv = render_uv[target_valid]
    target_uv = target_uv[target_valid]
    target_points_world = targets["points_world"][target_valid].detach()
    y = render_uv[:, 1].long().clamp(0, rendered_feature_map.shape[1] - 1)
    x = render_uv[:, 0].long().clamp(0, rendered_feature_map.shape[2] - 1)
    rendered_features = rendered_feature_map[:, y, x].T
    query_features = bilinear_sample_features(query_feature_map.detach(), target_uv)

    desc_loss = symmetric_descriptor_loss(rendered_features, query_features, temperature=desc_temperature)
    reproj_loss, fine_stats = fine_reprojection_loss(
        rendered_features,
        query_feature_map.detach(),
        target_uv,
        window_radius=fine_window_radius,
        temperature=fine_temperature,
    )
    visible_idx = render_pkg.get("loc_visible_idx")
    visible_count = 0 if visible_idx is None else visible_idx.numel()
    contributor_ids = None
    responsibility_weights = None
    compact_contributor_ids = None
    visible_bank_features = None
    diagnostics = {}
    if visible_count > 0:
        with torch.no_grad():
            contributor_ids, responsibility_weights = _sample_pixel_contributors(
                render_pkg,
                anchor_idx,
                render_uv,
                rendered_feature_map.shape[1],
                rendered_feature_map.shape[2],
                max_contributors=responsibility_topk,
                render_depth=render_depth,
                opacity_weight=responsibility_opacity_weight,
                depth_consistency_weight=responsibility_depth_weight,
            )
        if contributor_ids is not None and responsibility_weights is not None:
            visible_bank_features = gaussians.get_loc_feature[visible_idx].reshape(visible_count, -1)
            compact_contributor_ids = _compact_contributor_ids(
                contributor_ids.to(device=visible_idx.device),
                visible_idx.to(device=visible_idx.device),
            ).to(device=query_feature_map.device)
            with torch.no_grad():
                reconstruction = responsibility_reconstruction_metrics(
                    rendered_features.detach(),
                    visible_bank_features.detach().to(device=query_feature_map.device, dtype=query_feature_map.dtype),
                    compact_contributor_ids,
                    responsibility_weights,
                )
            diagnostics.update(
                {
                    "responsibility_reconstruction_mean_cosine": reconstruction["mean_cosine"],
                    "responsibility_reconstruction_min_cosine": reconstruction["min_cosine"],
                    "responsibility_reconstruction_p10_cosine": reconstruction["p10_cosine"],
                    "responsibility_reconstruction_valid_anchor_count": reconstruction["valid_anchor_count"],
                }
            )
    kl_loss = zero
    rank_loss = zero
    if (
        (float(dense_kl_weight) > 0 or float(dense_rank_weight) > 0)
        and compact_contributor_ids is not None
        and responsibility_weights is not None
        and visible_bank_features is not None
    ):
        anchor_weights, weight_diagnostics = _selective_dense_anchor_weights(
            query_features,
            rendered_features,
            visible_bank_features.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
            compact_contributor_ids,
            responsibility_weights,
            fine_stats,
            dense_pose_weight=dense_pose_weight,
            attr_cosine_threshold=attr_cosine_threshold,
            attr_entropy_threshold=attr_entropy_threshold,
            min_positive_prob=min_positive_prob,
            max_reproj_error=max_reproj_error,
            min_eligible_anchors=min_eligible_anchors,
        )
        diagnostics.update(weight_diagnostics)
        if float(dense_kl_weight) > 0:
            kl_loss = dense_responsibility_kl_loss(
                query_features,
                rendered_features,
                visible_bank_features.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
                compact_contributor_ids,
                responsibility_weights,
                dense_temperature=dense_kl_temperature,
                sparse_temperature=dense_kl_temperature,
                anchor_weights=anchor_weights,
            )
        if float(dense_rank_weight) > 0:
            rank_loss, rank_diagnostics = dense_responsibility_rank_loss(
                query_features,
                rendered_features,
                visible_bank_features.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
                compact_contributor_ids,
                responsibility_weights,
                dense_temperature=dense_kl_temperature,
                sparse_temperature=dense_kl_temperature,
                anchor_weights=anchor_weights,
                teacher_confidence_threshold=dense_rank_teacher_confidence,
                miss_topk=dense_rank_miss_topk,
                margin=dense_rank_margin,
            )
            diagnostics.update(rank_diagnostics)
    loss = desc_loss + reproj_loss + float(dense_kl_weight) * kl_loss + float(dense_rank_weight) * rank_loss

    with torch.no_grad():
        rendered_n = F.normalize(rendered_features, p=2, dim=-1)
        query_n = F.normalize(query_features, p=2, dim=-1)
        logits = rendered_n @ query_n.T
        labels = torch.arange(logits.shape[0], device=device)
        pos = logits[labels, labels]
        masked_logits = logits.masked_fill(torch.eye(logits.shape[0], dtype=torch.bool, device=device), -1e4)
        neg = masked_logits.max(dim=1).values
        margin = pos - neg

        try:
            pose_info = compute_pose_information(
                target_points_world,
                render_pkg["loc_K"],
                pose_gt_w2c,
                weights=fine_stats["positive_prob"].detach(),
            )
            information_value = pose_info.scores
        except Exception:
            information_value = torch.zeros_like(fine_stats["positive_prob"])
        if visible_count > 0:
            if contributor_ids is not None and responsibility_weights is not None:
                stats = aggregate_dense_anchor_stats(
                    visible_idx,
                    contributor_ids,
                    responsibility_weights,
                    query_features,
                    fine_stats,
                    margin,
                    information_value,
                )
                stats["contributor_count"] = torch.full_like(stats["positive_prob"], contributor_ids.shape[1])
            else:
                prototype = gaussians.get_loc_feature[visible_idx].reshape(visible_count, -1).detach()
                stats = {
                    "positive_prob": fine_stats["positive_prob"].mean().expand(visible_count),
                    "margin": margin.mean().expand(visible_count),
                    "entropy": fine_stats["entropy"].mean().expand(visible_count),
                    "reproj_error": fine_stats["reproj_error"].mean().expand(visible_count),
                    "information": information_value.mean().expand(visible_count),
                    "repeatability": (fine_stats["positive_prob"].mean() > 0.25).float().expand(visible_count),
                    "prototype": prototype,
                }
        else:
            stats = {}

    return DenseTeacherOutput(
        loss,
        desc_loss,
        reproj_loss,
        kl_loss,
        stats,
        render_pkg,
        int(target_valid.sum().item()),
        diagnostics=diagnostics,
        rank_loss=rank_loss,
    )
