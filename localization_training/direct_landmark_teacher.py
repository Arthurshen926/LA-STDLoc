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
    full_bank_loss: torch.Tensor
    anchor_loss: torch.Tensor
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
    """Compact per-landmark query observations with optional view-diverse replacement."""

    def __init__(
        self,
        landmark_indices,
        feature_dim,
        slots=4,
        device=None,
        dtype=torch.float32,
        view_similarity_threshold=0.95,
    ):
        landmark_indices = torch.as_tensor(landmark_indices, dtype=torch.long, device=device).reshape(-1)
        self.landmark_indices = landmark_indices
        self.feature_dim = int(feature_dim)
        self.slots = max(1, int(slots))
        self.view_similarity_threshold = float(view_similarity_threshold)
        self.features = torch.zeros(
            (landmark_indices.numel(), self.slots, self.feature_dim),
            dtype=dtype,
            device=landmark_indices.device,
        )
        self.view_directions = torch.zeros(
            (landmark_indices.numel(), self.slots, 3),
            dtype=dtype,
            device=landmark_indices.device,
        )
        self.confidences = torch.zeros(
            (landmark_indices.numel(), self.slots),
            dtype=dtype,
            device=landmark_indices.device,
        )
        self.camera_distances = torch.zeros(
            (landmark_indices.numel(), self.slots),
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
    def update(self, full_idx, query_features, view_directions=None, confidences=None, camera_distances=None):
        compact = self._compact_indices(full_idx)
        query_features = torch.as_tensor(query_features, device=self.features.device, dtype=self.features.dtype)
        query_features = query_features.reshape(compact.numel(), -1)[:, : self.feature_dim]
        known = compact >= 0
        if not known.any():
            return
        if view_directions is None:
            rows = compact[known]
            slots = self.next_slot[rows]
            self._write(rows, slots, query_features[known])
            self.next_slot[rows] = (slots + 1) % self.slots
            return

        view_directions = torch.as_tensor(view_directions, device=self.features.device, dtype=self.features.dtype)
        view_directions = F.normalize(view_directions.reshape(compact.numel(), 3), p=2, dim=-1)
        if confidences is None:
            confidences = torch.ones(compact.numel(), dtype=self.features.dtype, device=self.features.device)
        else:
            confidences = torch.as_tensor(confidences, device=self.features.device, dtype=self.features.dtype).reshape(-1)
        if camera_distances is None:
            camera_distances = torch.zeros(compact.numel(), dtype=self.features.dtype, device=self.features.device)
        else:
            camera_distances = torch.as_tensor(camera_distances, device=self.features.device, dtype=self.features.dtype).reshape(-1)

        for source_pos in torch.nonzero(known, as_tuple=False).squeeze(1).tolist():
            row = int(compact[source_pos].item())
            direction = view_directions[source_pos]
            confidence = confidences[source_pos]
            valid_slots = torch.nonzero(self.valid[row], as_tuple=False).squeeze(1)
            if valid_slots.numel() > 0:
                sims = self.view_directions[row, valid_slots] @ direction
                best_sim, best_offset = sims.max(dim=0)
                if best_sim.item() >= self.view_similarity_threshold:
                    slot = valid_slots[best_offset]
                    if confidence >= self.confidences[row, slot]:
                        self._write(
                            torch.tensor([row], dtype=torch.long, device=self.features.device),
                            slot.reshape(1),
                            query_features[source_pos : source_pos + 1],
                            direction.reshape(1, 3),
                            confidence.reshape(1),
                            camera_distances[source_pos : source_pos + 1],
                        )
                    continue
            invalid_slots = torch.nonzero(~self.valid[row], as_tuple=False).squeeze(1)
            if invalid_slots.numel() > 0:
                slot = invalid_slots[0]
            else:
                slot = self.confidences[row].argmin()
                if confidence < self.confidences[row, slot]:
                    continue
            self._write(
                torch.tensor([row], dtype=torch.long, device=self.features.device),
                slot.reshape(1),
                query_features[source_pos : source_pos + 1],
                direction.reshape(1, 3),
                confidence.reshape(1),
                camera_distances[source_pos : source_pos + 1],
            )

    def _write(self, rows, slots, features, view_directions=None, confidences=None, camera_distances=None):
        self.features[rows, slots] = F.normalize(features, p=2, dim=-1)
        if view_directions is not None:
            self.view_directions[rows, slots] = F.normalize(view_directions, p=2, dim=-1)
        if confidences is not None:
            self.confidences[rows, slots] = confidences
        else:
            self.confidences[rows, slots] = 1.0
        if camera_distances is not None:
            self.camera_distances[rows, slots] = camera_distances
        self.valid[rows, slots] = True


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


def _limit_valid_indices(valid, max_landmarks, uv=None, image_size=None, grid_size=8):
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if max_landmarks is None or idx.numel() <= max_landmarks:
        return idx
    if uv is not None and image_size is not None and int(grid_size) > 1:
        height, width = image_size
        grid_size = int(grid_size)
        uv = uv.to(device=valid.device, dtype=torch.float32)
        cell_x = torch.floor(uv[idx, 0].clamp(0, max(float(width) - 1.0, 0.0)) / max(float(width), 1.0) * grid_size)
        cell_y = torch.floor(uv[idx, 1].clamp(0, max(float(height) - 1.0, 0.0)) / max(float(height), 1.0) * grid_size)
        cell_x = cell_x.to(dtype=torch.long).clamp(0, grid_size - 1)
        cell_y = cell_y.to(dtype=torch.long).clamp(0, grid_size - 1)
        cell_ids = cell_y * grid_size + cell_x
        unique_cells = torch.unique(cell_ids, sorted=True)
        selected = []
        used = torch.zeros(idx.numel(), dtype=torch.bool, device=idx.device)
        while len(selected) < int(max_landmarks) and not bool(used.all().item()):
            progressed = False
            for cell_id in unique_cells.tolist():
                candidates = torch.nonzero((cell_ids == cell_id) & ~used, as_tuple=False).squeeze(1)
                if candidates.numel() == 0:
                    continue
                pos = int(candidates[0].item())
                selected.append(idx[pos])
                used[pos] = True
                progressed = True
                if len(selected) >= int(max_landmarks):
                    break
            if not progressed:
                break
        if selected:
            return torch.stack(selected)
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


def descriptor_anchor_loss(current_features, baseline_features, weights=None):
    if current_features.numel() == 0:
        return baseline_features.new_tensor(0.0)
    current = F.normalize(current_features.reshape(current_features.shape[0], -1), p=2, dim=-1)
    baseline = F.normalize(baseline_features.reshape(baseline_features.shape[0], -1).detach(), p=2, dim=-1)
    per_item = 1.0 - (current * baseline).sum(dim=-1)
    if weights is None:
        return per_item.mean()
    weights = weights.to(device=per_item.device, dtype=per_item.dtype).reshape(-1)
    return (per_item * weights).sum() / weights.sum().clamp_min(1e-6)


def full_bank_bimnn_loss(
    query_features,
    bank_features,
    positive_bank_indices,
    temperature=0.07,
    hard_negative_topk=0,
    hard_negative_margin=0.2,
    weights=None,
    ignore_bank_mask=None,
):
    query = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1)
    bank = F.normalize(bank_features.reshape(bank_features.shape[0], -1), p=2, dim=-1)
    positive_bank_indices = torch.as_tensor(
        positive_bank_indices,
        dtype=torch.long,
        device=query.device,
    ).reshape(-1)
    if query.numel() == 0 or bank.numel() == 0 or positive_bank_indices.numel() == 0:
        return query_features.new_tensor(0.0)
    valid = (positive_bank_indices >= 0) & (positive_bank_indices < bank.shape[0])
    if not valid.any():
        return query_features.new_tensor(0.0)
    query = query[valid]
    positive_bank_indices = positive_bank_indices[valid]
    if ignore_bank_mask is not None:
        ignore_bank_mask = torch.as_tensor(ignore_bank_mask, dtype=torch.bool, device=query.device)
        ignore_bank_mask = ignore_bank_mask.reshape(-1, bank.shape[0])[valid].clone()
    else:
        ignore_bank_mask = None
    temperature = max(float(temperature), 1e-6)

    query_to_bank = query @ bank.T / temperature
    query_ids = torch.arange(query.shape[0], dtype=torch.long, device=query.device)
    if ignore_bank_mask is not None:
        ignore_bank_mask[query_ids, positive_bank_indices] = False
        query_to_bank = query_to_bank.masked_fill(ignore_bank_mask, -torch.inf)
    query_loss = F.cross_entropy(query_to_bank, positive_bank_indices, reduction="none")

    positive_bank = bank[positive_bank_indices]
    bank_to_query = positive_bank @ query.T / temperature
    bank_loss = F.cross_entropy(bank_to_query, query_ids, reduction="none")
    per_item = query_loss + bank_loss

    if int(hard_negative_topk) > 0 and bank.shape[0] > 1:
        scores = (query @ bank.T).clone()
        scores[query_ids, positive_bank_indices] = -torch.inf
        if ignore_bank_mask is not None:
            scores = scores.masked_fill(ignore_bank_mask, -torch.inf)
        topk = min(int(hard_negative_topk), max(1, bank.shape[0] - 1))
        hard_neg = torch.topk(scores, k=topk, dim=1).values
        pos = (query * positive_bank).sum(dim=-1, keepdim=True)
        hard_loss = F.relu(float(hard_negative_margin) + hard_neg - pos).mean(dim=1)
        per_item = per_item + hard_loss

    if weights is not None:
        weights = weights.to(device=per_item.device, dtype=per_item.dtype).reshape(-1)[valid]
        return (per_item * weights).sum() / weights.sum().clamp_min(1e-6)
    return per_item.mean()


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


def _compact_positions(full_idx, reference_indices):
    full_idx = torch.as_tensor(full_idx, dtype=torch.long, device=reference_indices.device).reshape(-1)
    reference_indices = torch.as_tensor(reference_indices, dtype=torch.long, device=reference_indices.device).reshape(-1)
    positions = torch.full((full_idx.numel(),), -1, dtype=torch.long, device=reference_indices.device)
    if full_idx.numel() == 0 or reference_indices.numel() == 0:
        return positions
    max_index = int(max(full_idx.max().item(), reference_indices.max().item()))
    lookup = torch.full((max_index + 1,), -1, dtype=torch.long, device=reference_indices.device)
    lookup[reference_indices] = torch.arange(reference_indices.numel(), dtype=torch.long, device=reference_indices.device)
    in_range = (full_idx >= 0) & (full_idx < lookup.numel())
    positions[in_range] = lookup[full_idx[in_range]]
    return positions


def _select_reference_features(features, full_idx, reference_indices=None):
    features = features.reshape(features.shape[0], -1)
    full_idx = torch.as_tensor(full_idx, dtype=torch.long, device=features.device).reshape(-1)
    selected = features.new_zeros((full_idx.numel(), features.shape[1]))
    valid = torch.zeros(full_idx.numel(), dtype=torch.bool, device=features.device)
    if full_idx.numel() == 0 or features.shape[0] == 0:
        return selected, valid
    if int(full_idx.max().item()) < features.shape[0]:
        valid = full_idx >= 0
        selected[valid] = features[full_idx[valid]]
        return selected, valid
    if reference_indices is None:
        return selected, valid
    positions = _compact_positions(full_idx.to(device=features.device), reference_indices.to(device=features.device))
    valid = (positions >= 0) & (positions < features.shape[0])
    selected[valid] = features[positions[valid]]
    return selected, valid


def full_bank_descriptor_stats(query_features, bank_features, positive_bank_indices, temperature=0.07):
    query = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1)
    bank = F.normalize(bank_features.reshape(bank_features.shape[0], -1), p=2, dim=-1)
    positive_bank_indices = torch.as_tensor(positive_bank_indices, dtype=torch.long, device=query.device).reshape(-1)
    if query.numel() == 0 or bank.numel() == 0:
        zero = query.new_zeros((positive_bank_indices.numel(),))
        return zero, zero, zero
    valid = (positive_bank_indices >= 0) & (positive_bank_indices < bank.shape[0])
    positive_prob = query.new_zeros((positive_bank_indices.numel(),))
    margin = query.new_zeros((positive_bank_indices.numel(),))
    entropy = query.new_zeros((positive_bank_indices.numel(),))
    if not valid.any():
        return positive_prob, margin, entropy
    temperature = max(float(temperature), 1e-6)
    logits = query[valid] @ bank.T / temperature
    probs = torch.softmax(logits, dim=1)
    pos = positive_bank_indices[valid]
    pos_prob = probs[torch.arange(pos.numel(), device=query.device), pos]
    negative_logits = logits.clone()
    negative_logits[torch.arange(pos.numel(), device=query.device), pos] = -torch.inf
    margin_valid = logits[torch.arange(pos.numel(), device=query.device), pos] - negative_logits.max(dim=1).values
    entropy_valid = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
    entropy_valid = entropy_valid / torch.log(query.new_tensor(float(max(2, bank.shape[0]))))
    positive_prob[valid] = pos_prob.detach()
    margin[valid] = margin_valid.detach()
    entropy[valid] = entropy_valid.detach()
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
    full_bank_indices=None,
    full_bank_temperature=0.07,
    full_bank_hard_negative_topk=0,
    full_bank_hard_negative_margin=0.2,
    sampling_grid_size=8,
    anchor_features=None,
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
    keep = _limit_valid_indices(
        valid,
        max_landmarks,
        uv=uv,
        image_size=(height, width),
        grid_size=sampling_grid_size,
    )
    zero = query_feature_map.new_tensor(0.0)
    if keep.numel() == 0:
        return DirectLandmarkTeacherOutput(
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            {},
            landmark_indices.new_empty(0),
            uv.new_zeros((0, 2)),
            0,
        )

    selected_full_idx = landmark_indices[keep].to(device=gaussians.get_xyz.device)
    selected_uv = uv[keep]
    selected_xyz = xyz[keep]
    query_features = bilinear_sample_features(query_feature_map.detach(), selected_uv)
    gaussian_features = gaussians.get_loc_feature[selected_full_idx].reshape(keep.numel(), -1)
    desc_loss = direct_landmark_feature_loss(gaussian_features, query_features)
    multiview_loss = zero
    full_bank_loss = zero
    anchor_loss = zero
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
    positive_prob, margin, entropy = _descriptor_stats(gaussian_features, query_features)
    if full_bank_indices is not None:
        full_bank_indices = torch.as_tensor(
            full_bank_indices,
            dtype=torch.long,
            device=gaussians.get_xyz.device,
        ).reshape(-1)
        bank_features = gaussians.get_loc_feature[full_bank_indices].reshape(full_bank_indices.numel(), -1)
        positive_bank_indices = _compact_positions(
            selected_full_idx.to(device=full_bank_indices.device),
            full_bank_indices,
        ).to(device=query_feature_map.device)
        ignore_bank_mask = None
        source_index = getattr(gaussians, "loc_source_index", None)
        if torch.is_tensor(source_index) and source_index.numel() > 0:
            source_index = source_index.to(device=gaussians.get_xyz.device, dtype=torch.long).reshape(-1)
            max_required_idx = torch.cat([selected_full_idx.reshape(-1), full_bank_indices.reshape(-1)]).max()
            if max_required_idx.item() < source_index.numel():
                selected_source = source_index[selected_full_idx].to(device=query_feature_map.device)
                bank_source = source_index[full_bank_indices].to(device=query_feature_map.device)
                ignore_bank_mask = selected_source[:, None] == bank_source[None, :]
        full_bank_loss = full_bank_bimnn_loss(
            query_features,
            bank_features.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
            positive_bank_indices,
            temperature=full_bank_temperature,
            hard_negative_topk=full_bank_hard_negative_topk,
            hard_negative_margin=full_bank_hard_negative_margin,
            ignore_bank_mask=ignore_bank_mask,
        )
        positive_prob, margin, entropy = full_bank_descriptor_stats(
            query_features,
            bank_features.to(device=query_feature_map.device, dtype=query_feature_map.dtype),
            positive_bank_indices,
            temperature=full_bank_temperature,
        )
    if anchor_features is not None:
        anchor_selected, anchor_valid = _select_reference_features(
            anchor_features.to(device=gaussian_features.device, dtype=gaussian_features.dtype),
            selected_full_idx.to(device=gaussian_features.device),
            reference_indices=full_bank_indices if full_bank_indices is not None else None,
        )
        if anchor_valid.any():
            anchor_loss = descriptor_anchor_loss(
                gaussian_features[anchor_valid],
                anchor_selected[anchor_valid],
            )
    if multiview_memory is not None and update_multiview_memory:
        pose_c2w = torch.linalg.inv(pose_gt_w2c)
        camera_center = pose_c2w[:3, 3].to(device=selected_xyz.device, dtype=selected_xyz.dtype)
        view_delta = camera_center[None] - selected_xyz
        camera_distances = view_delta.norm(dim=-1)
        view_directions = F.normalize(view_delta, p=2, dim=-1)
        multiview_memory.update(
            selected_full_idx,
            query_features.detach(),
            view_directions=view_directions.detach(),
            confidences=positive_prob.detach(),
            camera_distances=camera_distances.detach(),
        )
    stats = {
        "positive_prob": positive_prob,
        "full_bank_positive_prob": positive_prob,
        "margin": margin,
        "entropy": entropy,
        "reproj_error": torch.zeros_like(positive_prob),
        "information": torch.zeros_like(positive_prob),
        "repeatability": (positive_prob > 0.25).float(),
        "prototype": F.normalize(query_features.detach(), p=2, dim=-1),
        "multiview_positive_count": multiview_positive_count,
        "anchor_loss": torch.full_like(positive_prob, float(anchor_loss.detach().item())),
    }
    return DirectLandmarkTeacherOutput(
        desc_loss + full_bank_loss + anchor_loss,
        desc_loss,
        multiview_loss,
        full_bank_loss,
        anchor_loss,
        zero,
        stats,
        selected_full_idx,
        selected_uv.detach(),
        int(keep.numel()),
    )
