from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gaussian_renderer import render_from_pose_gsplat
from localization_training.correspondence import (
    bilinear_sample_features,
    build_target_correspondences,
    make_pixel_grid,
    sample_valid_anchors,
)
from localization_training.losses import fine_reprojection_loss, symmetric_descriptor_loss
from localization_training.pose_information import compute_pose_information


@dataclass
class DenseTeacherOutput:
    loss: torch.Tensor
    desc_loss: torch.Tensor
    reproj_loss: torch.Tensor
    stats: dict
    render_pkg: dict
    anchor_count: int

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
        return DenseTeacherOutput(zero, zero, zero, {}, render_pkg, 0)

    h, w = depth.shape[-2:]
    grid = make_pixel_grid(h, w, device=device, dtype=torch.float32)
    valid = torch.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)
    if alpha is not None and alpha.numel() == h * w:
        valid = valid & (alpha.reshape(-1) > alpha_threshold)

    anchor_idx = sample_valid_anchors(valid.reshape(h, w), anchor_count)
    if anchor_idx.numel() < min_anchors:
        return DenseTeacherOutput(zero, zero, zero, {}, render_pkg, int(anchor_idx.numel()))

    render_uv = grid[anchor_idx]
    render_depth = depth.reshape(-1)[anchor_idx]
    targets = build_target_correspondences(render_uv, render_depth.detach(), render_pkg["loc_K"], pose_init_w2c, pose_gt_w2c)
    target_uv = targets["target_uv"].detach()
    target_valid = targets["valid"]
    target_valid = target_valid & (target_uv[:, 0] >= 0) & (target_uv[:, 0] <= query_feature_map.shape[2] - 1)
    target_valid = target_valid & (target_uv[:, 1] >= 0) & (target_uv[:, 1] <= query_feature_map.shape[1] - 1)

    if target_valid.sum() < min_anchors:
        return DenseTeacherOutput(zero, zero, zero, {}, render_pkg, int(target_valid.sum().item()))

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
    loss = desc_loss + reproj_loss

    with torch.no_grad():
        rendered_n = F.normalize(rendered_features, p=2, dim=-1)
        query_n = F.normalize(query_features, p=2, dim=-1)
        logits = rendered_n @ query_n.T
        labels = torch.arange(logits.shape[0], device=device)
        pos = logits[labels, labels]
        masked_logits = logits.masked_fill(torch.eye(logits.shape[0], dtype=torch.bool, device=device), -1e4)
        neg = masked_logits.max(dim=1).values
        margin = pos - neg

        visible_idx = render_pkg.get("loc_visible_idx")
        visible_count = 0 if visible_idx is None else visible_idx.numel()
        try:
            pose_info = compute_pose_information(
                target_points_world,
                render_pkg["loc_K"],
                pose_gt_w2c,
                weights=fine_stats["positive_prob"].detach(),
            )
            information_value = pose_info.scores.mean()
        except Exception:
            information_value = fine_stats["positive_prob"].new_tensor(0.0)
        if visible_count > 0:
            prototype = gaussians.get_loc_feature[visible_idx].reshape(visible_count, -1).detach()
            stats = {
                "positive_prob": fine_stats["positive_prob"].mean().expand(visible_count),
                "margin": margin.mean().expand(visible_count),
                "entropy": fine_stats["entropy"].mean().expand(visible_count),
                "reproj_error": fine_stats["reproj_error"].mean().expand(visible_count),
                "information": information_value.expand(visible_count),
                "repeatability": (fine_stats["positive_prob"].mean() > 0.25).float().expand(visible_count),
                "prototype": prototype,
            }
        else:
            stats = {}

    return DenseTeacherOutput(loss, desc_loss, reproj_loss, stats, render_pkg, int(target_valid.sum().item()))
