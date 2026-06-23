from dataclasses import dataclass

import torch
import torch.nn.functional as F

from localization_training.correspondence import bilinear_sample_features
from utils.graphics_utils import fov2focal


@dataclass
class DirectLandmarkTeacherOutput:
    loss: torch.Tensor
    desc_loss: torch.Tensor
    multiview_loss: torch.Tensor
    reproj_loss: torch.Tensor
    stats: dict
    loc_visible_idx: torch.Tensor
    target_uv: torch.Tensor
    anchor_count: int

    @property
    def loc_viewspace_points(self):
        return None

    @property
    def loc_radii(self):
        return None


def make_intrinsics_from_fov(fovx, fovy, width, height, device=None, dtype=torch.float32):
    return torch.tensor(
        [
            [fov2focal(fovx, width), 0.0, width / 2.0],
            [0.0, fov2focal(fovy, height), height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def project_landmarks_to_query(xyz, K, pose_w2c, height, width, eps=1e-8):
    if xyz.numel() == 0:
        empty_uv = xyz.new_zeros((0, 2))
        return empty_uv, xyz.new_zeros((0,)), torch.zeros(0, dtype=torch.bool, device=xyz.device)
    xyz = xyz.to(device=K.device, dtype=K.dtype)
    pose_w2c = pose_w2c.to(device=K.device, dtype=K.dtype)
    xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, dtype=K.dtype, device=K.device)], dim=1)
    xyz_cam = (pose_w2c @ xyz_h.T)[:3].T
    depth = xyz_cam[:, 2]
    uv = torch.empty(xyz.shape[0], 2, dtype=K.dtype, device=K.device)
    uv[:, 0] = K[0, 0] * xyz_cam[:, 0] / depth.clamp_min(eps) + K[0, 2]
    uv[:, 1] = K[1, 1] * xyz_cam[:, 1] / depth.clamp_min(eps) + K[1, 2]
    valid = (
        (depth > eps)
        & torch.isfinite(uv).all(dim=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= width - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= height - 1)
    )
    return uv, depth, valid


def _squeeze_map(value):
    if value is None:
        return None
    while value.dim() > 2:
        value = value.squeeze(0)
    return value


def _sample_scalar_map(scalar_map, uv):
    scalar_map = _squeeze_map(scalar_map)
    if scalar_map is None:
        return None
    return bilinear_sample_features(scalar_map[None], uv)[:, 0]


class LandmarkObservationMemory:
    """Compact per-landmark queue of query observations for multi-view positives."""

    def __init__(self, landmark_indices, feature_dim, slots=4, device=None, dtype=torch.float32):
        landmark_indices = torch.as_tensor(landmark_indices, dtype=torch.long, device=device).reshape(-1)
        self.landmark_indices = landmark_indices
        self.feature_dim = int(feature_dim)
        self.slots = max(1, int(slots))
        self.features = torch.zeros(
            (landmark_indices.numel(), self.slots, self.feature_dim),
            dtype=dtype,
            device=landmark_indices.device,
        )
        self.valid = torch.zeros(
            (landmark_indices.numel(), self.slots),
            dtype=torch.bool,
            device=landmark_indices.device,
        )
        self.next_slot = torch.zeros(landmark_indices.numel(), dtype=torch.long, device=landmark_indices.device)
        max_index = int(landmark_indices.max().item()) if landmark_indices.numel() else -1
        self.full_to_compact = torch.full(
            (max_index + 1,),
            -1,
            dtype=torch.long,
            device=landmark_indices.device,
        )
        if landmark_indices.numel():
            self.full_to_compact[landmark_indices] = torch.arange(
                landmark_indices.numel(),
                dtype=torch.long,
                device=landmark_indices.device,
            )

    def _compact_indices(self, full_idx):
        full_idx = torch.as_tensor(full_idx, dtype=torch.long, device=self.landmark_indices.device).reshape(-1)
        compact = torch.full_like(full_idx, -1)
        if self.full_to_compact.numel() == 0:
            return compact
        in_range = (full_idx >= 0) & (full_idx < self.full_to_compact.numel())
        compact[in_range] = self.full_to_compact[full_idx[in_range]]
        return compact

    def lookup(self, full_idx):
        compact = self._compact_indices(full_idx)
        features = self.features.new_zeros((compact.numel(), self.slots, self.feature_dim))
        valid = torch.zeros((compact.numel(), self.slots), dtype=torch.bool, device=self.features.device)
        known = compact >= 0
        if known.any():
            features[known] = self.features[compact[known]]
            valid[known] = self.valid[compact[known]]
        return features, valid

    def positive_count(self, full_idx):
        _, valid = self.lookup(full_idx)
        return valid.sum(dim=1)

    @torch.no_grad()
    def update(self, full_idx, query_features):
        compact = self._compact_indices(full_idx)
        query_features = torch.as_tensor(query_features, device=self.features.device, dtype=self.features.dtype)
        query_features = query_features.reshape(compact.numel(), -1)[:, : self.feature_dim]
        known = compact >= 0
        if not known.any():
            return
        rows = compact[known]
        slots = self.next_slot[rows]
        self.features[rows, slots] = F.normalize(query_features[known], p=2, dim=-1)
        self.valid[rows, slots] = True
        self.next_slot[rows] = (slots + 1) % self.slots


def filter_depth_consistent_landmarks(
    uv,
    projected_depth,
    valid,
    target_depth=None,
    target_alpha=None,
    alpha_threshold=0.2,
    abs_tolerance=1e-3,
    rel_tolerance=0.01,
):
    valid = valid.clone()
    if target_depth is not None:
        sampled_depth = _sample_scalar_map(target_depth.to(device=uv.device, dtype=uv.dtype), uv)
        tolerance = torch.maximum(
            uv.new_full(sampled_depth.shape, float(abs_tolerance)),
            sampled_depth.abs() * float(rel_tolerance),
        )
        valid = (
            valid
            & torch.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & ((projected_depth - sampled_depth).abs() <= tolerance)
        )
    if target_alpha is not None:
        sampled_alpha = _sample_scalar_map(target_alpha.to(device=uv.device, dtype=uv.dtype), uv)
        valid = valid & torch.isfinite(sampled_alpha) & (sampled_alpha >= alpha_threshold)
    return valid


def _limit_valid_indices(valid, max_landmarks):
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if max_landmarks is None or idx.numel() <= max_landmarks:
        return idx
    positions = torch.linspace(0, idx.numel() - 1, int(max_landmarks), device=idx.device).long()
    return idx[positions]


def direct_landmark_feature_loss(gaussian_features, query_features, weights=None):
    if gaussian_features.numel() == 0:
        return query_features.new_tensor(0.0)
    gaussian_features = F.normalize(gaussian_features.reshape(gaussian_features.shape[0], -1), p=2, dim=-1)
    query_features = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1)
    per_item = 1.0 - (gaussian_features * query_features.detach()).sum(dim=-1)
    if weights is None:
        return per_item.mean()
    weights = weights.to(device=per_item.device, dtype=per_item.dtype).reshape(-1)
    return (per_item * weights).sum() / weights.sum().clamp_min(1e-6)


def multiview_contrastive_landmark_loss(
    gaussian_features,
    query_features,
    landmark_indices,
    target_uv=None,
    memory=None,
    temperature=0.07,
    ignore_radius=2.0,
    weights=None,
):
    if gaussian_features.numel() == 0:
        return query_features.new_tensor(0.0)
    gaussian_features = F.normalize(gaussian_features.reshape(gaussian_features.shape[0], -1), p=2, dim=-1)
    query_features = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1).detach()
    landmark_indices = landmark_indices.to(device=gaussian_features.device, dtype=torch.long).reshape(-1)
    temperature = max(float(temperature), 1e-6)

    pos_features = [query_features[:, None, :]]
    pos_masks = [torch.ones((query_features.shape[0], 1), dtype=torch.bool, device=query_features.device)]
    if memory is not None:
        mem_features, mem_valid = memory.lookup(landmark_indices)
        pos_features.append(mem_features.to(device=query_features.device, dtype=query_features.dtype))
        pos_masks.append(mem_valid.to(device=query_features.device))
    pos_features = torch.cat(pos_features, dim=1)
    pos_mask = torch.cat(pos_masks, dim=1)
    pos_logits = torch.einsum("nd,nsd->ns", gaussian_features, pos_features) / temperature
    pos_logits = pos_logits.masked_fill(~pos_mask, -torch.inf)

    neg_logits = gaussian_features @ query_features.T / temperature
    neg_mask = landmark_indices[:, None] != landmark_indices[None, :]
    if target_uv is not None and target_uv.numel() > 0 and ignore_radius > 0:
        target_uv = target_uv.to(device=gaussian_features.device, dtype=gaussian_features.dtype)
        uv_dist = torch.cdist(target_uv, target_uv)
        neg_mask = neg_mask & (uv_dist > float(ignore_radius))
    neg_logits = neg_logits.masked_fill(~neg_mask, -torch.inf)

    pos_logsum = torch.logsumexp(pos_logits, dim=1)
    denom_logsum = torch.logsumexp(torch.cat([pos_logits, neg_logits], dim=1), dim=1)
    per_item = -(pos_logsum - denom_logsum)
    valid = torch.isfinite(per_item) & pos_mask.any(dim=1)
    if not valid.any():
        return query_features.new_tensor(0.0)
    per_item = per_item[valid]
    if weights is None:
        return per_item.mean()
    weights = weights.to(device=per_item.device, dtype=per_item.dtype).reshape(-1)[valid]
    return (per_item * weights).sum() / weights.sum().clamp_min(1e-6)


def _descriptor_stats(gaussian_features, query_features):
    gaussian_n = F.normalize(gaussian_features.reshape(gaussian_features.shape[0], -1), p=2, dim=-1)
    query_n = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1)
    pos = (gaussian_n * query_n.detach()).sum(dim=-1)
    if gaussian_n.shape[0] > 1:
        logits = gaussian_n.detach() @ query_n.detach().T
        eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        neg = logits.masked_fill(eye, -1e4).max(dim=1).values
        margin = pos.detach() - neg
    else:
        margin = torch.ones_like(pos)
    positive_prob = ((pos.detach() + 1.0) * 0.5).clamp(0.0, 1.0)
    entropy = -(positive_prob * positive_prob.clamp_min(1e-6).log())
    return positive_prob, margin, entropy


def direct_landmark_teacher(
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
    multiview_memory=None,
    multiview_temperature=0.07,
    multiview_ignore_radius=2.0,
    update_multiview_memory=True,
):
    device = query_feature_map.device
    dtype = query_feature_map.dtype
    height, width = query_feature_map.shape[-2:]
    pose_gt_w2c = pose_gt_w2c.to(device=device, dtype=dtype)
    landmark_indices = landmark_indices.to(device=gaussians.get_xyz.device, dtype=torch.long).reshape(-1)
    xyz = gaussians.get_xyz[landmark_indices].to(device=device, dtype=dtype)
    K = make_intrinsics_from_fov(fovx, fovy, width, height, device=device, dtype=dtype)

    uv, depth, valid = project_landmarks_to_query(xyz, K, pose_gt_w2c, height, width)
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
    zero = query_feature_map.new_tensor(0.0)
    if keep.numel() == 0:
        return DirectLandmarkTeacherOutput(zero, zero, zero, zero, {}, landmark_indices.new_empty(0), uv.new_zeros((0, 2)), 0)

    selected_full_idx = landmark_indices[keep].to(device=gaussians.get_xyz.device)
    selected_uv = uv[keep]
    query_features = bilinear_sample_features(query_feature_map.detach(), selected_uv)
    gaussian_features = gaussians.get_loc_feature[selected_full_idx].reshape(keep.numel(), -1)
    desc_loss = direct_landmark_feature_loss(gaussian_features, query_features)
    multiview_loss = zero
    multiview_positive_count = torch.zeros(keep.numel(), dtype=torch.float32, device=query_feature_map.device)
    if multiview_memory is not None:
        multiview_positive_count = multiview_memory.positive_count(selected_full_idx).to(
            device=query_feature_map.device,
            dtype=torch.float32,
        )
        multiview_loss = multiview_contrastive_landmark_loss(
            gaussian_features,
            query_features,
            selected_full_idx,
            target_uv=selected_uv,
            memory=multiview_memory,
            temperature=multiview_temperature,
            ignore_radius=multiview_ignore_radius,
        )
        if update_multiview_memory:
            multiview_memory.update(selected_full_idx, query_features.detach())
    positive_prob, margin, entropy = _descriptor_stats(gaussian_features, query_features)
    stats = {
        "positive_prob": positive_prob,
        "margin": margin,
        "entropy": entropy,
        "reproj_error": torch.zeros_like(positive_prob),
        "information": torch.zeros_like(positive_prob),
        "repeatability": (positive_prob > 0.25).float(),
        "prototype": F.normalize(query_features.detach(), p=2, dim=-1),
        "multiview_positive_count": multiview_positive_count,
    }
    return DirectLandmarkTeacherOutput(
        desc_loss,
        desc_loss,
        multiview_loss,
        zero,
        stats,
        selected_full_idx,
        selected_uv.detach(),
        int(keep.numel()),
    )
