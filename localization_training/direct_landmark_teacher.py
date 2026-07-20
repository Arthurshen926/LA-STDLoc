from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from localization_training.correspondence import bilinear_sample_features
from localization_training.pose_information import (
    compute_pose_information,
    normalize_information_scores,
    pose_jacobian_analytic,
)
from localization_training.render_artifacts import combine_artifact_confidence, sample_region_weight_map
from utils.graphics_utils import fov2focal


@dataclass
class DirectLandmarkTeacherOutput:
    loss: torch.Tensor
    desc_loss: torch.Tensor
    multiview_loss: torch.Tensor
    full_bank_loss: torch.Tensor
    anchor_loss: torch.Tensor
    reproj_loss: torch.Tensor
    clean_hard_negative_loss: torch.Tensor
    stats: dict
    loc_visible_idx: torch.Tensor
    target_uv: torch.Tensor
    anchor_count: int
    diagnostics: dict = field(default_factory=dict)

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


def gaussian_localization_xyz(gaussians):
    loc_xyz = getattr(gaussians, "get_loc_xyz", None)
    if torch.is_tensor(loc_xyz):
        return loc_xyz
    if callable(loc_xyz):
        loc_xyz = loc_xyz()
        if torch.is_tensor(loc_xyz):
            return loc_xyz
    return gaussians.get_xyz


def stable_landmark_memory_indices(gaussians, full_indices):
    """Map mutable Gaussian rows to stable source keys for multi-view memory."""
    full_indices = torch.as_tensor(full_indices, dtype=torch.long).reshape(-1)
    source_index = getattr(gaussians, "loc_source_index", None)
    if not torch.is_tensor(source_index) or source_index.numel() == 0:
        return full_indices
    source_index = source_index.detach().to(dtype=torch.long)
    full_indices = full_indices.to(device=source_index.device)
    in_range = (full_indices >= 0) & (full_indices < source_index.numel())
    stable = full_indices.clone()
    mapped = source_index[full_indices[in_range]]
    stable[in_range] = torch.where(
        mapped >= 0,
        mapped,
        full_indices[in_range],
    )
    return stable


def project_landmarks_to_query(
    xyz,
    K,
    pose_w2c,
    height,
    width,
    eps=1e-8,
    pixel_center_offset=0.0,
):
    """Project landmarks into a feature grid.

    ``pixel_center_offset=0`` retains the historical pixel-index convention.
    New sparse-compatible paths use ``0.5``: a grid cell at integer ``i`` is
    physically observed at pixel coordinate ``i + 0.5``, matching the PnP
    frontend's explicit ``+0.5`` conversion.
    """
    if xyz.numel() == 0:
        empty_uv = xyz.new_zeros((0, 2))
        return empty_uv, xyz.new_zeros((0,)), torch.zeros(0, dtype=torch.bool, device=xyz.device)
    xyz = xyz.to(device=K.device, dtype=K.dtype)
    pose_w2c = pose_w2c.to(device=K.device, dtype=K.dtype)
    xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, dtype=K.dtype, device=K.device)], dim=1)
    xyz_cam = (pose_w2c @ xyz_h.T)[:3].T
    depth = xyz_cam[:, 2]
    physical_uv = torch.empty(xyz.shape[0], 2, dtype=K.dtype, device=K.device)
    physical_uv[:, 0] = K[0, 0] * xyz_cam[:, 0] / depth.clamp_min(eps) + K[0, 2]
    physical_uv[:, 1] = K[1, 1] * xyz_cam[:, 1] / depth.clamp_min(eps) + K[1, 2]
    offset = torch.as_tensor(
        pixel_center_offset,
        dtype=K.dtype,
        device=K.device,
    )
    uv = physical_uv - offset
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


def _normalize_pose_information_scores(scores, floor=0.0, mode="max", eps=1e-8):
    return normalize_information_scores(scores, floor=floor, mode=mode, eps=eps)


def pose_information_weights(
    points_world,
    K,
    pose_w2c,
    floor=0.0,
    eps=1e-8,
    mode="point_jacobian",
    normalization="max",
    matchability=None,
    measurement_covariance=None,
    translation_scale=0.02,
    rotation_scale_degrees=2.0,
    damping=1e-4,
    return_diagnostics=False,
):
    """Build bounded descriptor weights from an explicitly named pose metric."""
    points_world = torch.as_tensor(points_world).detach()
    if points_world.numel() == 0:
        empty = points_world.new_zeros((0,))
        return (empty, {}) if return_diagnostics else empty
    device = points_world.device
    dtype = points_world.dtype if points_world.is_floating_point() else torch.float32
    points_world = points_world.to(device=device, dtype=dtype).reshape(-1, 3)
    K = torch.as_tensor(K, device=device, dtype=dtype)
    pose_w2c = torch.as_tensor(pose_w2c, device=device, dtype=dtype)
    xyz_h = torch.cat(
        [points_world, torch.ones(points_world.shape[0], 1, device=device, dtype=dtype)],
        dim=1,
    )
    xyz_cam = (pose_w2c @ xyz_h.T)[:3].T
    valid = xyz_cam[:, 2] > float(eps)
    mode = str(mode).strip().lower()
    aliases = {
        "current": "point_jacobian",
        "single_point": "point_jacobian",
        "exact_conditional_full": "conditional_full",
        "exact_conditional_translation": "conditional_translation",
    }
    mode = aliases.get(mode, mode)
    diagnostics = {"pose_information_valid_count": int(valid.sum().item())}

    if mode == "point_jacobian":
        jacobian = pose_jacobian_analytic(points_world, K, pose_w2c)
        raw_scores = jacobian.square().sum(dim=(1, 2)).clamp_min(0.0)
        raw_scores = torch.where(valid, raw_scores, torch.zeros_like(raw_scores))
    elif mode in {
        "full_set_leverage",
        "conditional_full",
        "conditional_translation",
    }:
        if matchability is None:
            fisher_weights = points_world.new_ones((points_world.shape[0],))
        else:
            fisher_weights = torch.as_tensor(
                matchability, device=device, dtype=dtype
            ).reshape(-1)
            if fisher_weights.numel() != points_world.shape[0]:
                raise ValueError(
                    f"Expected {points_world.shape[0]} matchability weights, "
                    f"got {fisher_weights.numel()}"
                )
            fisher_weights = fisher_weights.clamp_min(0.0)
        fisher_weights = torch.where(valid, fisher_weights, torch.zeros_like(fisher_weights))
        information = compute_pose_information(
            points_world,
            K,
            pose_w2c,
            weights=fisher_weights,
            damping=damping,
            measurement_covariance=measurement_covariance,
            translation_scale=translation_scale,
            rotation_scale=torch.deg2rad(points_world.new_tensor(float(rotation_scale_degrees))).item(),
            use_analytic_jacobian=True,
            eps=max(float(eps), 1e-12),
        )
        if mode == "full_set_leverage":
            raw_scores = information.full_set_leverage_scores
        elif mode == "conditional_full":
            raw_scores = information.scores
        else:
            raw_scores = information.translation_scores
        raw_scores = torch.where(valid, raw_scores, torch.zeros_like(raw_scores))
        diagnostics.update(
            {
                "pose_information_full_set_logdet": float(information.logdet.item()),
                "pose_information_full_set_condition": float(information.condition_number.item()),
                "pose_information_translation_logdet": float(information.translation_logdet.item()),
                "pose_information_translation_condition": float(
                    information.translation_condition_number.item()
                ),
                "pose_information_translation_min_eigenvalue": float(
                    information.translation_min_eigenvalue.item()
                ),
                "pose_information_translation_trace_covariance": float(
                    information.translation_trace_covariance.item()
                ),
                "pose_information_translation_worst_std": float(
                    information.translation_worst_std.item()
                ),
                "pose_information_effective_count": float(information.effective_count.item()),
            }
        )
    else:
        raise ValueError(f"Unsupported pose information mode: {mode}")

    finite_scores = raw_scores[torch.isfinite(raw_scores)]
    if finite_scores.numel() > 0:
        diagnostics.update(
            {
                "pose_information_raw_gain_min": float(finite_scores.min().item()),
                "pose_information_raw_gain_mean": float(finite_scores.mean().item()),
                "pose_information_raw_gain_max": float(finite_scores.max().item()),
            }
        )
    weights = _normalize_pose_information_scores(
        raw_scores,
        floor=floor,
        mode=normalization,
        eps=eps,
    )
    if return_diagnostics:
        diagnostics["pose_information_mode_id"] = float(
            {
                "point_jacobian": 1,
                "full_set_leverage": 2,
                "conditional_full": 3,
                "conditional_translation": 4,
            }[mode]
        )
        return weights, diagnostics
    return weights


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


def _bin_ids(values, bins, valid=None):
    values = torch.as_tensor(values, dtype=torch.float32)
    ids = torch.full(values.shape, -1, dtype=torch.long, device=values.device)
    bins = int(bins)
    if bins <= 1 or values.numel() == 0:
        return ids
    finite = torch.isfinite(values)
    if valid is not None:
        finite = finite & torch.as_tensor(valid, dtype=torch.bool, device=values.device).reshape(-1)
    if not bool(finite.any().item()):
        return ids
    selected = values[finite]
    span = (selected.max() - selected.min()).clamp_min(1e-6)
    ids[finite] = torch.floor((selected - selected.min()) / span * bins).to(dtype=torch.long).clamp(0, bins - 1)
    return ids


def _count_inverse_weights(ids, eps=1e-6):
    ids = torch.as_tensor(ids, dtype=torch.long)
    weights = torch.ones(ids.shape, dtype=torch.float32, device=ids.device)
    valid = ids >= 0
    if not bool(valid.any().item()):
        return weights
    unique, counts = torch.unique(ids[valid], return_counts=True)
    count_map = torch.zeros(int(unique.max().item()) + 1, dtype=torch.float32, device=ids.device)
    count_map[unique] = counts.to(dtype=torch.float32)
    weights[valid] = torch.rsqrt(count_map[ids[valid]].clamp_min(float(eps)))
    return weights


def geometry_balance_weights(
    uv,
    depth=None,
    image_size=None,
    grid_size=0,
    depth_bins=0,
    max_weight=4.0,
):
    """Return mean-normalized weights that reduce repeated 2D/depth concentration."""
    uv = torch.as_tensor(uv, dtype=torch.float32)
    if uv.numel() == 0:
        return uv.new_zeros((0,))
    uv = uv.reshape(-1, 2)
    device = uv.device
    weights = torch.ones(uv.shape[0], dtype=torch.float32, device=device)

    if image_size is not None and int(grid_size) > 1:
        height, width = int(image_size[0]), int(image_size[1])
        if height > 0 and width > 0:
            cell_x = torch.floor(uv[:, 0].clamp(0, width - 1) / max(float(width), 1.0) * int(grid_size))
            cell_y = torch.floor(uv[:, 1].clamp(0, height - 1) / max(float(height), 1.0) * int(grid_size))
            cell_x = cell_x.to(dtype=torch.long).clamp(0, int(grid_size) - 1)
            cell_y = cell_y.to(dtype=torch.long).clamp(0, int(grid_size) - 1)
            grid_ids = cell_y * int(grid_size) + cell_x
            weights = weights * _count_inverse_weights(grid_ids).to(device=device)

    if depth is not None and int(depth_bins) > 1:
        depth = torch.as_tensor(depth, dtype=torch.float32, device=device).reshape(-1)
        if depth.numel() == uv.shape[0]:
            depth_ids = _bin_ids(depth, int(depth_bins), valid=torch.isfinite(depth))
            weights = weights * _count_inverse_weights(depth_ids).to(device=device)

    weights = torch.where(torch.isfinite(weights), weights, torch.ones_like(weights))
    weights = weights / weights.mean().clamp_min(1e-6)
    max_weight = max(float(max_weight), 1.0)
    weights = weights.clamp(1.0 / max_weight, max_weight)
    return weights / weights.mean().clamp_min(1e-6)


def clean_reprojection_hard_negative_loss(
    query_features,
    bank_features,
    positive_bank_indices,
    query_uv,
    bank_uv,
    bank_valid=None,
    reprojection_radius=4.0,
    hard_negative_topk=16,
    margin=0.2,
    weights=None,
    ignore_bank_mask=None,
    positive_bank_mask=None,
    query_bank_scores=None,
    query_bank_uv_distances=None,
):
    """Penalize feature-similar negatives whose GT reprojection is far from the query anchor."""
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
    if not bool(valid.any().item()):
        return query_features.new_tensor(0.0)

    query = query[valid]
    positive_bank_indices = positive_bank_indices[valid]
    query_uv = torch.as_tensor(query_uv, dtype=torch.float32, device=query.device).reshape(-1, 2)[valid]
    bank_uv = torch.as_tensor(bank_uv, dtype=torch.float32, device=query.device).reshape(-1, 2)
    if bank_uv.shape[0] != bank.shape[0]:
        raise ValueError(f"bank_uv must have one row per bank feature, got {bank_uv.shape[0]} and {bank.shape[0]}.")

    if bank_valid is None:
        bank_valid = torch.isfinite(bank_uv).all(dim=1)
    else:
        bank_valid = torch.as_tensor(bank_valid, dtype=torch.bool, device=query.device).reshape(-1)
    if bank_valid.shape[0] != bank.shape[0]:
        raise ValueError(
            f"bank_valid must have one value per bank feature, got {bank_valid.shape[0]} and {bank.shape[0]}."
        )

    positive_mask = torch.zeros((query.shape[0], bank.shape[0]), dtype=torch.bool, device=query.device)
    query_ids = torch.arange(query.shape[0], dtype=torch.long, device=query.device)
    positive_mask[query_ids, positive_bank_indices] = True
    if positive_bank_mask is not None:
        extra_positive = torch.as_tensor(positive_bank_mask, dtype=torch.bool, device=query.device)
        extra_positive = extra_positive.reshape(-1, bank.shape[0])[valid]
        positive_mask = positive_mask | extra_positive

    if query_bank_uv_distances is None:
        uv_dist = torch.cdist(query_uv, bank_uv)
    else:
        uv_dist = torch.as_tensor(
            query_bank_uv_distances,
            dtype=query_uv.dtype,
            device=query.device,
        )
        expected_shape = (int(valid.numel()), int(bank.shape[0]))
        if tuple(uv_dist.shape) != expected_shape:
            raise ValueError(
                "query_bank_uv_distances must match the unfiltered query-bank matrix, "
                f"got {tuple(uv_dist.shape)} and {expected_shape}."
            )
        uv_dist = uv_dist[valid]
    clean_negative_mask = (uv_dist > float(reprojection_radius)) & bank_valid[None, :] & ~positive_mask
    if ignore_bank_mask is not None:
        ignore_bank_mask = torch.as_tensor(ignore_bank_mask, dtype=torch.bool, device=query.device)
        ignore_bank_mask = ignore_bank_mask.reshape(-1, bank.shape[0])[valid]
        clean_negative_mask = clean_negative_mask & ~ignore_bank_mask
    has_negative = clean_negative_mask.any(dim=1)
    if not bool(has_negative.any().item()):
        return query_features.new_tensor(0.0)

    if query_bank_scores is None:
        scores = query @ bank.T
    else:
        scores = torch.as_tensor(
            query_bank_scores,
            dtype=query.dtype,
            device=query.device,
        )
        expected_shape = (int(valid.numel()), int(bank.shape[0]))
        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                "query_bank_scores must match the unfiltered query-bank matrix, "
                f"got {tuple(scores.shape)} and {expected_shape}."
            )
        scores = scores[valid]
    positive_scores = scores.masked_fill(~positive_mask, -torch.inf).max(dim=1).values
    negative_scores = scores.masked_fill(~clean_negative_mask, -torch.inf)
    topk = min(max(int(hard_negative_topk), 1), bank.shape[0])
    hard_negatives = torch.topk(negative_scores, k=topk, dim=1).values
    hard_loss = F.relu(float(margin) + hard_negatives - positive_scores[:, None]).mean(dim=1)
    hard_loss = hard_loss[has_negative]
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=hard_loss.dtype, device=hard_loss.device).reshape(-1)[valid][has_negative]
        return (hard_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    return hard_loss.mean()


def full_bank_bimnn_loss(
    query_features,
    bank_features,
    positive_bank_indices,
    temperature=0.07,
    hard_negative_topk=0,
    hard_negative_margin=0.2,
    weights=None,
    ignore_bank_mask=None,
    positive_bank_mask=None,
    chunk_size=None,
    query_bank_scores=None,
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
    if not bool(valid.any().item()):
        return query_features.new_tensor(0.0)
    valid_rows = torch.nonzero(valid, as_tuple=False).squeeze(1)
    query = query[valid_rows]
    positive_bank_indices = positive_bank_indices[valid_rows]
    bank_count = int(bank.shape[0])
    shared_scores = None
    if query_bank_scores is not None:
        shared_scores = torch.as_tensor(
            query_bank_scores,
            dtype=query.dtype,
            device=query.device,
        )
        expected_shape = (int(valid.numel()), bank_count)
        if tuple(shared_scores.shape) != expected_shape:
            raise ValueError(
                "query_bank_scores must match the unfiltered query-bank matrix, "
                f"got {tuple(shared_scores.shape)} and {expected_shape}."
            )
        shared_scores = shared_scores[valid_rows]
    if ignore_bank_mask is not None:
        ignore_bank_mask = torch.as_tensor(
            ignore_bank_mask,
            dtype=torch.bool,
            device=query.device,
        ).reshape(-1, bank_count)
    if positive_bank_mask is not None:
        positive_bank_mask = torch.as_tensor(
            positive_bank_mask,
            dtype=torch.bool,
            device=query.device,
        ).reshape(-1, bank_count)
    temperature = max(float(temperature), 1e-6)
    if chunk_size is None or int(chunk_size) <= 0:
        chunk_size = int(query.shape[0])
    chunk_size = max(1, int(chunk_size))

    query_ids = torch.arange(query.shape[0], dtype=torch.long, device=query.device)
    if shared_scores is None:
        positive_bank = bank[positive_bank_indices]
        bank_to_query = positive_bank @ query.T / temperature
    else:
        bank_to_query = shared_scores[:, positive_bank_indices].T / temperature
    bank_loss = F.cross_entropy(bank_to_query, query_ids, reduction="none")

    query_losses = []
    for local_rows in query_ids.split(chunk_size):
        source_rows = valid_rows[local_rows]
        if positive_bank_mask is None:
            positive_mask = torch.zeros(
                (int(local_rows.numel()), bank_count),
                dtype=torch.bool,
                device=query.device,
            )
        else:
            positive_mask = positive_bank_mask[source_rows].clone()
        chunk_ids = torch.arange(local_rows.numel(), dtype=torch.long, device=query.device)
        positive_mask[chunk_ids, positive_bank_indices[local_rows]] = True

        raw_scores = (
            query[local_rows] @ bank.T
            if shared_scores is None
            else shared_scores[local_rows]
        )
        query_to_bank = raw_scores / temperature
        ignore_mask = None
        if ignore_bank_mask is not None:
            ignore_mask = ignore_bank_mask[source_rows] & ~positive_mask
            query_to_bank = query_to_bank.masked_fill(ignore_mask, -torch.inf)
        positive_logits = query_to_bank.masked_fill(~positive_mask, -torch.inf)
        chunk_loss = -(
            torch.logsumexp(positive_logits, dim=1)
            - torch.logsumexp(query_to_bank, dim=1)
        )

        if int(hard_negative_topk) > 0 and bank_count > 1:
            scores = raw_scores.masked_fill(positive_mask, -torch.inf)
            if ignore_mask is not None:
                scores = scores.masked_fill(ignore_mask, -torch.inf)
            topk = min(int(hard_negative_topk), max(1, bank_count - 1))
            hard_neg = torch.topk(scores, k=topk, dim=1).values
            pos = raw_scores.masked_fill(~positive_mask, -torch.inf).max(
                dim=1,
                keepdim=True,
            ).values
            chunk_loss = chunk_loss + F.relu(
                float(hard_negative_margin) + hard_neg - pos
            ).mean(dim=1)
        query_losses.append(chunk_loss)

    per_item = torch.cat(query_losses, dim=0) + bank_loss

    if weights is not None:
        weights = weights.to(device=per_item.device, dtype=per_item.dtype).reshape(-1)[valid_rows]
        return (per_item * weights).sum() / weights.sum().clamp_min(1e-6)
    return per_item.mean()


def _merge_ignore_bank_mask(existing, extra):
    if extra is None:
        return existing
    if existing is None:
        return extra
    return existing | extra


def _exact_positive_bank_mask(positive_bank_indices, bank_count, device):
    positive_bank_indices = torch.as_tensor(positive_bank_indices, dtype=torch.long, device=device).reshape(-1)
    mask = torch.zeros((positive_bank_indices.numel(), int(bank_count)), dtype=torch.bool, device=device)
    if positive_bank_indices.numel() == 0 or int(bank_count) <= 0:
        return mask
    rows = torch.arange(positive_bank_indices.numel(), dtype=torch.long, device=device)
    valid = (positive_bank_indices >= 0) & (positive_bank_indices < int(bank_count))
    if valid.any():
        mask[rows[valid], positive_bank_indices[valid]] = True
    return mask


def _full_bank_source_relation_diagnostics(
    positive_bank_indices,
    bank_count,
    source_bank_mask=None,
    ignore_bank_mask=None,
    positive_bank_mask=None,
):
    if source_bank_mask is None or int(bank_count) <= 0:
        return {
            "full_bank_source_related_count": 0,
            "full_bank_source_positive_count": 0,
            "full_bank_source_ignore_count": 0,
            "full_bank_source_negative_count": 0,
        }
    device = source_bank_mask.device
    source_bank_mask = torch.as_tensor(source_bank_mask, dtype=torch.bool, device=device).reshape(-1, int(bank_count))
    exact_positive_mask = _exact_positive_bank_mask(positive_bank_indices, bank_count, device=device)
    source_extra_mask = source_bank_mask & ~exact_positive_mask
    source_positive_mask = torch.zeros_like(source_extra_mask)
    source_ignore_mask = torch.zeros_like(source_extra_mask)
    if positive_bank_mask is not None:
        source_positive_mask = (
            torch.as_tensor(positive_bank_mask, dtype=torch.bool, device=device).reshape(-1, int(bank_count))
            & source_extra_mask
        )
    if ignore_bank_mask is not None:
        source_ignore_mask = (
            torch.as_tensor(ignore_bank_mask, dtype=torch.bool, device=device).reshape(-1, int(bank_count))
            & source_extra_mask
            & ~source_positive_mask
        )
    source_negative_mask = source_extra_mask & ~source_positive_mask & ~source_ignore_mask
    return {
        "full_bank_source_related_count": int(source_extra_mask.sum().item()),
        "full_bank_source_positive_count": int(source_positive_mask.sum().item()),
        "full_bank_source_ignore_count": int(source_ignore_mask.sum().item()),
        "full_bank_source_negative_count": int(source_negative_mask.sum().item()),
    }


def child_responsibility_keep_mask(
    selected_full_idx,
    source_index,
    gaussian_features,
    query_features,
    mode="none",
):
    selected_full_idx = torch.as_tensor(selected_full_idx, dtype=torch.long).reshape(-1)
    keep = torch.ones(selected_full_idx.numel(), dtype=torch.bool, device=selected_full_idx.device)
    if mode is None or str(mode) == "none" or selected_full_idx.numel() == 0:
        return keep
    if str(mode) != "feature":
        raise ValueError(f"Unsupported child responsibility mode: {mode}")
    if not torch.is_tensor(source_index) or source_index.numel() == 0:
        return keep

    source_index = source_index.to(device=selected_full_idx.device, dtype=torch.long).reshape(-1)
    in_range = (selected_full_idx >= 0) & (selected_full_idx < source_index.numel())
    if not bool(in_range.all().item()):
        return keep

    selected_source = source_index[selected_full_idx]
    known_source = selected_source >= 0
    if not bool(known_source.any().item()):
        return keep

    feature_device = gaussian_features.device
    gaussian = F.normalize(gaussian_features.reshape(selected_full_idx.numel(), -1).float(), p=2, dim=-1)
    query = F.normalize(
        query_features.to(device=feature_device).reshape(selected_full_idx.numel(), -1).float(),
        p=2,
        dim=-1,
    )
    scores = (gaussian * query).sum(dim=-1).to(device=selected_full_idx.device)
    keep = torch.zeros(selected_full_idx.numel(), dtype=torch.bool, device=selected_full_idx.device)
    keep[~known_source] = True
    for source in torch.unique(selected_source[known_source], sorted=True):
        rows = torch.nonzero(selected_source == source, as_tuple=False).squeeze(1)
        if rows.numel() == 1:
            keep[rows] = True
            continue
        best = rows[scores[rows].argmax()]
        keep[best] = True
    return keep


def _full_bank_ignore_diagnostics(positive_bank_indices, bank_count, ignore_bank_mask=None, positive_bank_mask=None):
    positive_bank_indices = torch.as_tensor(positive_bank_indices, dtype=torch.long).reshape(-1)
    bank_count = int(bank_count)
    valid = (positive_bank_indices >= 0) & (positive_bank_indices < bank_count)
    valid_positive_count = int(valid.sum().item())
    potential_negative_count = valid_positive_count * max(bank_count - 1, 0)
    ignore_negative_count = 0
    positive_count = valid_positive_count
    positive_mask = None
    if positive_bank_mask is not None and valid_positive_count > 0 and bank_count > 0:
        positive_mask = torch.as_tensor(
            positive_bank_mask,
            dtype=torch.bool,
            device=positive_bank_indices.device,
        ).reshape(-1, bank_count)[valid].clone()
        row_ids = torch.arange(valid_positive_count, dtype=torch.long, device=positive_bank_indices.device)
        positive_mask[row_ids, positive_bank_indices[valid]] = True
        positive_count = int(positive_mask.sum().item())
    if ignore_bank_mask is not None and valid_positive_count > 0 and bank_count > 0:
        ignore_bank_mask = torch.as_tensor(
            ignore_bank_mask,
            dtype=torch.bool,
            device=positive_bank_indices.device,
        ).reshape(-1, bank_count)[valid].clone()
        row_ids = torch.arange(valid_positive_count, dtype=torch.long, device=positive_bank_indices.device)
        ignore_bank_mask[row_ids, positive_bank_indices[valid]] = False
        if positive_mask is not None:
            ignore_bank_mask = ignore_bank_mask & ~positive_mask
        ignore_negative_count = int(ignore_bank_mask.sum().item())
    extra_positive_count = max(positive_count - valid_positive_count, 0)
    effective_negative_count = potential_negative_count - ignore_negative_count - extra_positive_count
    return {
        "full_bank_query_count": int(positive_bank_indices.numel()),
        "full_bank_bank_count": bank_count,
        "full_bank_valid_positive_count": valid_positive_count,
        "full_bank_positive_count": positive_count,
        "full_bank_extra_positive_count": extra_positive_count,
        "full_bank_potential_negative_count": potential_negative_count,
        "full_bank_ignore_negative_count": ignore_negative_count,
        "full_bank_effective_negative_count": effective_negative_count,
        "full_bank_ignore_negative_ratio": (
            float(ignore_negative_count) / float(potential_negative_count)
            if potential_negative_count > 0
            else 0.0
        ),
    }


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


def sample_stochastic_full_bank(
    full_bank_indices,
    required_positive_indices,
    max_landmarks=0,
):
    """Bound per-query matching memory while retaining every current positive."""
    bank = torch.as_tensor(full_bank_indices, dtype=torch.long).reshape(-1)
    required = torch.unique(
        torch.as_tensor(
            required_positive_indices,
            dtype=torch.long,
            device=bank.device,
        ).reshape(-1)
    )
    max_landmarks = int(max_landmarks or 0)
    if max_landmarks <= 0 or bank.numel() <= max_landmarks:
        return bank

    target_count = max(max_landmarks, int(required.numel()))
    remaining = max(0, target_count - int(required.numel()))
    if remaining == 0:
        return required

    # A full randperm over million-surfel maps is unnecessarily expensive every
    # iteration. Use it only when the requested bank is a large fraction of the
    # source; otherwise oversampled random positions have negligible collisions.
    if bank.numel() <= 4 * target_count:
        candidates = bank[torch.randperm(bank.numel(), device=bank.device)]
    else:
        draw_count = min(int(bank.numel()), max(2 * remaining, remaining + 1024))
        positions = torch.randint(
            int(bank.numel()),
            (draw_count,),
            device=bank.device,
        )
        candidates = torch.unique(bank[positions])
    if required.numel() > 0 and candidates.numel() > 0:
        candidates = candidates[~torch.isin(candidates, required)]
    sampled = candidates[:remaining]
    return torch.cat([required, sampled], dim=0)


def full_bank_descriptor_stats(
    query_features,
    bank_features,
    positive_bank_indices,
    temperature=0.07,
    ignore_bank_mask=None,
    positive_bank_mask=None,
    chunk_size=None,
    query_bank_scores=None,
):
    with torch.no_grad():
        query = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1)
        bank = F.normalize(bank_features.reshape(bank_features.shape[0], -1), p=2, dim=-1)
        positive_bank_indices = torch.as_tensor(
            positive_bank_indices,
            dtype=torch.long,
            device=query.device,
        ).reshape(-1)
        if query.numel() == 0 or bank.numel() == 0:
            zero = query.new_zeros((positive_bank_indices.numel(),))
            return zero, zero, zero
        valid = (positive_bank_indices >= 0) & (positive_bank_indices < bank.shape[0])
        positive_prob = query.new_zeros((positive_bank_indices.numel(),))
        margin = query.new_zeros((positive_bank_indices.numel(),))
        entropy = query.new_zeros((positive_bank_indices.numel(),))
        if not valid.any():
            return positive_prob, margin, entropy

        bank_count = int(bank.shape[0])
        shared_scores = None
        if query_bank_scores is not None:
            shared_scores = torch.as_tensor(
                query_bank_scores,
                dtype=query.dtype,
                device=query.device,
            )
            expected_shape = (int(positive_bank_indices.numel()), bank_count)
            if tuple(shared_scores.shape) != expected_shape:
                raise ValueError(
                    "query_bank_scores must match the query-bank matrix, "
                    f"got {tuple(shared_scores.shape)} and {expected_shape}."
                )
        valid_rows = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if chunk_size is None or int(chunk_size) <= 0:
            chunk_size = int(valid_rows.numel())
        chunk_size = max(1, int(chunk_size))
        temperature = max(float(temperature), 1e-6)
        normalizer = torch.log(query.new_tensor(float(max(2, bank_count))))
        full_positive_mask = None
        full_ignore_mask = None
        if positive_bank_mask is not None:
            full_positive_mask = torch.as_tensor(
                positive_bank_mask,
                dtype=torch.bool,
                device=query.device,
            ).reshape(-1, bank_count)
        if ignore_bank_mask is not None:
            full_ignore_mask = torch.as_tensor(
                ignore_bank_mask,
                dtype=torch.bool,
                device=query.device,
            ).reshape(-1, bank_count)

        for rows in valid_rows.split(chunk_size):
            raw_scores = (
                query[rows] @ bank.T
                if shared_scores is None
                else shared_scores[rows]
            )
            logits = raw_scores / temperature
            if full_positive_mask is not None:
                positive_mask = full_positive_mask[rows].clone()
            else:
                positive_mask = torch.zeros(
                    (int(rows.numel()), bank_count),
                    dtype=torch.bool,
                    device=query.device,
                )
            query_ids = torch.arange(rows.numel(), dtype=torch.long, device=query.device)
            row_positive_indices = positive_bank_indices[rows]
            positive_mask[query_ids, row_positive_indices] = True
            if full_ignore_mask is not None:
                ignore_mask = full_ignore_mask[rows].clone() & ~positive_mask
                logits = logits.masked_fill(ignore_mask, -torch.inf)

            probs = torch.softmax(logits, dim=1)
            pos_prob = probs.masked_fill(~positive_mask, 0.0).sum(dim=1)
            negative_logits = logits.masked_fill(positive_mask, -torch.inf)
            positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
            has_negative = torch.isfinite(negative_logits).any(dim=1)
            hard_negative = negative_logits.max(dim=1).values
            margin_valid = positive_logits.max(dim=1).values - hard_negative
            margin_valid = torch.where(has_negative, margin_valid, torch.ones_like(margin_valid))
            entropy_valid = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1) / normalizer

            positive_prob[rows] = pos_prob
            margin[rows] = margin_valid
            entropy[rows] = entropy_valid
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
    full_bank_ignore_3d_radius=0.0,
    full_bank_ignore_uv_radius=0.0,
    full_bank_source_mode="ignore",
    full_bank_nearby_as_positive=False,
    full_bank_stats_chunk_size=256,
    full_bank_max_landmarks=0,
    full_bank_pose_information_weight=0.0,
    full_bank_pose_information_floor=0.0,
    full_bank_pose_information_mode="point_jacobian",
    full_bank_pose_information_normalization="max",
    full_bank_fisher_translation_scale=0.02,
    full_bank_fisher_rotation_scale_degrees=2.0,
    full_bank_fisher_measurement_sigma=1.0,
    full_bank_fisher_damping=1e-4,
    full_bank_fisher_use_matchability=False,
    full_bank_fisher_matchability_floor=0.05,
    full_bank_fisher_matchability_power=1.0,
    full_bank_fisher_uncertainty_entropy_scale=0.0,
    full_bank_balance_weight=0.0,
    full_bank_balance_grid_size=0,
    full_bank_balance_depth_bins=0,
    full_bank_balance_max_weight=4.0,
    full_bank_clean_hard_negative_weight=0.0,
    full_bank_clean_reproj_radius=4.0,
    full_bank_clean_hard_negatives=16,
    sampling_grid_size=8,
    anchor_features=None,
    child_responsibility_mode="none",
    artifact_weight_map=None,
    artifact_image_weight=1.0,
    artifact_weight_combine_mode="product",
    artifact_loss_scale_mode="none",
):
    device = query_feature_map.device
    dtype = query_feature_map.dtype
    height, width = query_feature_map.shape[-2:]
    pose_gt_w2c = pose_gt_w2c.to(device=device, dtype=dtype)
    loc_xyz_all = gaussian_localization_xyz(gaussians)
    landmark_indices = landmark_indices.to(device=loc_xyz_all.device, dtype=torch.long).reshape(-1)
    xyz = loc_xyz_all[landmark_indices].to(device=device, dtype=dtype)
    K = make_intrinsics_from_fov(fovx, fovy, width, height, device=device, dtype=dtype)

    uv, depth, projected_valid = project_landmarks_to_query(
        xyz,
        K,
        pose_gt_w2c,
        height,
        width,
    )
    depth_valid = filter_depth_consistent_landmarks(
        uv,
        depth,
        projected_valid,
        target_depth=target_depth,
        target_alpha=None,
        alpha_threshold=alpha_threshold,
        abs_tolerance=depth_abs_tolerance,
        rel_tolerance=depth_rel_tolerance,
    )
    alpha_valid = filter_depth_consistent_landmarks(
        uv,
        depth,
        projected_valid,
        target_depth=None,
        target_alpha=target_alpha,
        alpha_threshold=alpha_threshold,
        abs_tolerance=depth_abs_tolerance,
        rel_tolerance=depth_rel_tolerance,
    )
    valid = depth_valid & alpha_valid
    diagnostics = {
        "projected_valid_count": int(projected_valid.sum().item()),
        "depth_valid_count": int(depth_valid.sum().item()),
        "alpha_valid_count": int(alpha_valid.sum().item()),
        "depth_alpha_valid_count": int(valid.sum().item()),
    }
    if child_responsibility_mode is not None and str(child_responsibility_mode) != "none":
        responsibility_candidates = torch.nonzero(valid, as_tuple=False).squeeze(1)
        if responsibility_candidates.numel() > 0:
            candidate_full_idx = landmark_indices[
                responsibility_candidates.to(device=landmark_indices.device)
            ].to(device=loc_xyz_all.device)
            candidate_uv = uv[responsibility_candidates]
            candidate_query_features = bilinear_sample_features(query_feature_map.detach(), candidate_uv)
            candidate_gaussian_features = gaussians.get_loc_feature[candidate_full_idx].reshape(
                responsibility_candidates.numel(),
                -1,
            )
            responsibility_keep = child_responsibility_keep_mask(
                candidate_full_idx,
                getattr(gaussians, "loc_source_index", None),
                candidate_gaussian_features,
                candidate_query_features,
                mode=child_responsibility_mode,
            )
            diagnostics.update(
                {
                    "child_responsibility_candidate_count": int(responsibility_keep.numel()),
                    "child_responsibility_kept_count": int(responsibility_keep.sum().item()),
                    "child_responsibility_dropped_count": int((~responsibility_keep).sum().item()),
                }
            )
            if not bool(responsibility_keep.all().item()):
                valid = valid.clone()
                drop_positions = responsibility_candidates[
                    ~responsibility_keep.to(device=responsibility_candidates.device)
                ]
                valid[drop_positions] = False
        else:
            diagnostics.update(
                {
                    "child_responsibility_candidate_count": 0,
                    "child_responsibility_kept_count": 0,
                    "child_responsibility_dropped_count": 0,
                }
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
        diagnostics.update(
            {
                "artifact_region_weight_min": 1.0,
                "artifact_region_weight_mean": 1.0,
                "artifact_region_weighted_count": 0,
                "artifact_image_weight": float(max(0.0, min(float(artifact_image_weight), 1.0))),
                "artifact_teacher_weight_min": 1.0,
                "artifact_teacher_weight_mean": 1.0,
                "artifact_teacher_weighted_count": 0,
                "artifact_teacher_loss_scale": 1.0,
            }
        )
        return DirectLandmarkTeacherOutput(
            zero,
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
            diagnostics=diagnostics,
        )

    selected_full_idx = landmark_indices[keep].to(device=loc_xyz_all.device)
    selected_uv = uv[keep]
    selected_xyz = xyz[keep]
    selected_depth = depth[keep]
    query_features = bilinear_sample_features(query_feature_map.detach(), selected_uv)
    gaussian_features = gaussians.get_loc_feature[selected_full_idx].reshape(keep.numel(), -1)
    artifact_region_weights = None
    artifact_weights = None
    if artifact_weight_map is not None:
        artifact_region_weights = sample_region_weight_map(
            artifact_weight_map,
            selected_uv,
            image_size=(height, width),
            default_weight=1.0,
        ).to(device=query_feature_map.device, dtype=query_feature_map.dtype)
        diagnostics.update(
            {
                "artifact_region_weight_min": float(artifact_region_weights.min().detach().item()),
                "artifact_region_weight_mean": float(artifact_region_weights.mean().detach().item()),
                "artifact_region_weighted_count": int(
                    (artifact_region_weights < 0.999).sum().detach().item()
                ),
            }
        )
    else:
        artifact_region_weights = torch.ones(
            keep.numel(),
            device=query_feature_map.device,
            dtype=query_feature_map.dtype,
        )
        diagnostics.update(
            {
                "artifact_region_weight_min": 1.0,
                "artifact_region_weight_mean": 1.0,
                "artifact_region_weighted_count": 0,
            }
        )
    artifact_image_weight = max(0.0, min(float(artifact_image_weight), 1.0))
    artifact_weights = combine_artifact_confidence(
        artifact_region_weights,
        image_weight=artifact_image_weight,
        mode=artifact_weight_combine_mode,
    ).to(device=query_feature_map.device, dtype=query_feature_map.dtype)
    artifact_loss_scale_mode = str(artifact_loss_scale_mode).strip().lower()
    if artifact_loss_scale_mode == "none":
        artifact_loss_scale = query_feature_map.new_tensor(1.0)
    elif artifact_loss_scale_mode == "region_mean":
        artifact_loss_scale = artifact_region_weights.mean().detach()
    elif artifact_loss_scale_mode == "combined_mean":
        artifact_loss_scale = artifact_weights.mean().detach()
    else:
        raise ValueError(f"Unsupported artifact loss scale mode: {artifact_loss_scale_mode}")
    diagnostics.update(
        {
            "artifact_image_weight": artifact_image_weight,
            "artifact_teacher_weight_min": float(artifact_weights.min().detach().item()),
            "artifact_teacher_weight_mean": float(artifact_weights.mean().detach().item()),
            "artifact_teacher_weighted_count": int((artifact_weights < 0.999).sum().detach().item()),
            "artifact_teacher_loss_scale": float(artifact_loss_scale.detach().item()),
        }
    )
    multiview_loss = zero
    full_bank_loss = zero
    clean_hn_loss = zero
    anchor_loss = zero
    desc_loss = direct_landmark_feature_loss(gaussian_features, query_features, weights=artifact_weights)
    selected_count = selected_full_idx.numel()
    multiview_positive_count = torch.zeros(selected_count, dtype=torch.float32, device=query_feature_map.device)
    fisher_measurement_sigma = max(float(full_bank_fisher_measurement_sigma), 1e-4)
    pose_info_weight, pose_info_diagnostics = pose_information_weights(
        selected_xyz,
        K,
        pose_gt_w2c,
        floor=full_bank_pose_information_floor,
        mode=full_bank_pose_information_mode,
        normalization=full_bank_pose_information_normalization,
        measurement_covariance=selected_xyz.new_tensor(fisher_measurement_sigma ** 2),
        translation_scale=full_bank_fisher_translation_scale,
        rotation_scale_degrees=full_bank_fisher_rotation_scale_degrees,
        damping=full_bank_fisher_damping,
        return_diagnostics=True,
    )
    pose_info_weight = pose_info_weight.to(
        device=query_feature_map.device, dtype=query_feature_map.dtype
    )
    full_bank_weights = artifact_weights
    pose_info_blend = max(0.0, min(float(full_bank_pose_information_weight), 1.0))
    fisher_use_matchability = bool(full_bank_fisher_use_matchability)
    if pose_info_blend > 0.0 and not fisher_use_matchability:
        pose_info_scale = (1.0 - pose_info_blend) + pose_info_blend * pose_info_weight
        full_bank_weights = artifact_weights * pose_info_scale.detach()
    if pose_info_blend > 0.0:
        diagnostics.update(pose_info_diagnostics)
        diagnostics.update(
            {
                "pose_information_weight_min": float(pose_info_weight.min().detach().item()),
                "pose_information_weight_mean": float(pose_info_weight.mean().detach().item()),
                "pose_information_weight_max": float(pose_info_weight.max().detach().item()),
                "pose_information_weight_blend": pose_info_blend,
                "pose_information_weight_floor": float(max(0.0, min(float(full_bank_pose_information_floor), 1.0))),
                "pose_information_uses_matchability": float(fisher_use_matchability),
                "pose_information_translation_task_scale_m": float(
                    full_bank_fisher_translation_scale
                ),
                "pose_information_rotation_task_scale_degrees": float(
                    full_bank_fisher_rotation_scale_degrees
                ),
                "pose_information_measurement_sigma_px": fisher_measurement_sigma,
            }
        )
    balance_blend = max(0.0, min(float(full_bank_balance_weight), 1.0))
    if balance_blend > 0.0:
        balance = geometry_balance_weights(
            selected_uv,
            depth=selected_depth,
            image_size=(height, width),
            grid_size=full_bank_balance_grid_size,
            depth_bins=full_bank_balance_depth_bins,
            max_weight=full_bank_balance_max_weight,
        ).to(device=query_feature_map.device, dtype=query_feature_map.dtype)
        balance_scale = (1.0 - balance_blend) + balance_blend * balance
        full_bank_weights = full_bank_weights * balance_scale.detach()
        diagnostics.update(
            {
                "full_bank_balance_weight_blend": balance_blend,
                "full_bank_balance_weight_min": float(balance.min().detach().item()),
                "full_bank_balance_weight_mean": float(balance.mean().detach().item()),
                "full_bank_balance_weight_max": float(balance.max().detach().item()),
            }
        )
    if multiview_memory is not None:
        memory_landmark_indices = stable_landmark_memory_indices(
            gaussians,
            selected_full_idx,
        )
        multiview_positive_count = multiview_memory.positive_count(
            memory_landmark_indices
        ).to(
            device=query_feature_map.device,
            dtype=torch.float32,
        )
        unique_memory_sources = torch.unique(memory_landmark_indices)
        diagnostics.update(
            {
                "multiview_memory_key_count": int(memory_landmark_indices.numel()),
                "multiview_memory_unique_source_count": int(unique_memory_sources.numel()),
                "multiview_memory_shared_source_count": int(
                    memory_landmark_indices.numel()
                    - unique_memory_sources.numel()
                ),
            }
        )
        multiview_loss = multiview_contrastive_landmark_loss(
            gaussian_features,
            query_features,
            memory_landmark_indices,
            target_uv=selected_uv,
            memory=multiview_memory,
            temperature=multiview_temperature,
            ignore_radius=multiview_ignore_radius,
            weights=artifact_weights,
        )
    positive_prob, margin, entropy = _descriptor_stats(gaussian_features, query_features)
    if full_bank_indices is not None:
        full_bank_source_mode = str(full_bank_source_mode)
        if full_bank_source_mode not in {"ignore", "positive", "responsibility"}:
            raise ValueError(f"Unsupported full_bank_source_mode: {full_bank_source_mode}")
        full_bank_indices = torch.as_tensor(
            full_bank_indices,
            dtype=torch.long,
            device=loc_xyz_all.device,
        ).reshape(-1)
        full_bank_source_count = int(full_bank_indices.numel())
        full_bank_indices = sample_stochastic_full_bank(
            full_bank_indices,
            selected_full_idx,
            max_landmarks=full_bank_max_landmarks,
        )
        diagnostics.update(
            {
                "full_bank_source_count": full_bank_source_count,
                "full_bank_sampled_count": int(full_bank_indices.numel()),
                "full_bank_sampling_fraction": float(
                    full_bank_indices.numel() / max(full_bank_source_count, 1)
                ),
                "full_bank_max_landmarks": int(full_bank_max_landmarks or 0),
            }
        )
        bank_features = gaussians.get_loc_feature[full_bank_indices].reshape(full_bank_indices.numel(), -1)
        bank_xyz = None
        bank_uv = None
        bank_valid = None
        uv_dist = None
        positive_bank_indices = _compact_positions(
            selected_full_idx.to(device=full_bank_indices.device),
            full_bank_indices,
        ).to(device=query_feature_map.device)
        ignore_bank_mask = None
        positive_bank_mask = None
        source_bank_mask = None
        spatial_bank_mask = None
        source_index = getattr(gaussians, "loc_source_index", None)
        if torch.is_tensor(source_index) and source_index.numel() > 0:
            source_index = source_index.to(device=loc_xyz_all.device, dtype=torch.long).reshape(-1)
            max_required_idx = torch.cat([selected_full_idx.reshape(-1), full_bank_indices.reshape(-1)]).max()
            if max_required_idx.item() < source_index.numel():
                selected_source = source_index[selected_full_idx].to(device=query_feature_map.device)
                bank_source = source_index[full_bank_indices].to(device=query_feature_map.device)
                source_bank_mask = selected_source[:, None] == bank_source[None, :]
        if float(full_bank_ignore_3d_radius) > 0.0:
            bank_xyz = loc_xyz_all[full_bank_indices].to(device=device, dtype=dtype)
            xyz_dist = torch.cdist(selected_xyz.float(), bank_xyz.float())
            nearby_xyz = xyz_dist <= float(full_bank_ignore_3d_radius)
            spatial_bank_mask = _merge_ignore_bank_mask(
                spatial_bank_mask,
                nearby_xyz.to(device=query_feature_map.device),
            )
        if float(full_bank_ignore_uv_radius) > 0.0:
            if bank_xyz is None:
                bank_xyz = loc_xyz_all[full_bank_indices].to(device=device, dtype=dtype)
            bank_uv, _, bank_valid = project_landmarks_to_query(bank_xyz, K, pose_gt_w2c, height, width)
            uv_dist = torch.cdist(selected_uv.float(), bank_uv.float())
            nearby_uv = (uv_dist <= float(full_bank_ignore_uv_radius)) & bank_valid[None, :]
            spatial_bank_mask = _merge_ignore_bank_mask(
                spatial_bank_mask,
                nearby_uv.to(device=query_feature_map.device),
            )
        if bool(full_bank_nearby_as_positive):
            positive_bank_mask = _merge_ignore_bank_mask(source_bank_mask, spatial_bank_mask)
        else:
            ignore_bank_mask = spatial_bank_mask
            if full_bank_source_mode == "ignore":
                ignore_bank_mask = _merge_ignore_bank_mask(ignore_bank_mask, source_bank_mask)
            elif full_bank_source_mode == "positive":
                positive_bank_mask = source_bank_mask
        bank_features_for_loss = bank_features.to(
            device=query_feature_map.device,
            dtype=query_feature_map.dtype,
        )
        shared_query_bank_scores = F.normalize(
            query_features.reshape(query_features.shape[0], -1),
            p=2,
            dim=-1,
        ) @ F.normalize(
            bank_features_for_loss.reshape(bank_features_for_loss.shape[0], -1),
            p=2,
            dim=-1,
        ).T
        positive_prob, margin, entropy = full_bank_descriptor_stats(
            query_features,
            bank_features_for_loss,
            positive_bank_indices,
            temperature=full_bank_temperature,
            ignore_bank_mask=ignore_bank_mask,
            positive_bank_mask=positive_bank_mask,
            chunk_size=full_bank_stats_chunk_size,
            query_bank_scores=shared_query_bank_scores.detach(),
        )
        if pose_info_blend > 0.0 and fisher_use_matchability:
            matchability_floor = max(
                0.0, min(float(full_bank_fisher_matchability_floor), 1.0)
            )
            matchability_power = max(float(full_bank_fisher_matchability_power), 0.0)
            fisher_matchability = positive_prob.detach().clamp(0.0, 1.0)
            fisher_matchability = fisher_matchability.pow(matchability_power)
            fisher_matchability = matchability_floor + (
                1.0 - matchability_floor
            ) * fisher_matchability
            entropy_scale = max(float(full_bank_fisher_uncertainty_entropy_scale), 0.0)
            fisher_sigma = fisher_measurement_sigma * (
                1.0 + entropy_scale * entropy.detach().clamp_min(0.0)
            )
            pose_info_weight, pose_info_diagnostics = pose_information_weights(
                selected_xyz,
                K,
                pose_gt_w2c,
                floor=full_bank_pose_information_floor,
                mode=full_bank_pose_information_mode,
                normalization=full_bank_pose_information_normalization,
                matchability=fisher_matchability,
                measurement_covariance=fisher_sigma.square(),
                translation_scale=full_bank_fisher_translation_scale,
                rotation_scale_degrees=full_bank_fisher_rotation_scale_degrees,
                damping=full_bank_fisher_damping,
                return_diagnostics=True,
            )
            pose_info_weight = pose_info_weight.to(
                device=query_feature_map.device, dtype=query_feature_map.dtype
            )
            pose_info_scale = (
                1.0 - pose_info_blend
            ) + pose_info_blend * pose_info_weight
            full_bank_weights = full_bank_weights * pose_info_scale.detach()
            diagnostics.update(pose_info_diagnostics)
            diagnostics.update(
                {
                    "pose_information_weight_min": float(pose_info_weight.min().item()),
                    "pose_information_weight_mean": float(pose_info_weight.mean().item()),
                    "pose_information_weight_max": float(pose_info_weight.max().item()),
                    "pose_information_matchability_min": float(fisher_matchability.min().item()),
                    "pose_information_matchability_mean": float(fisher_matchability.mean().item()),
                    "pose_information_matchability_max": float(fisher_matchability.max().item()),
                    "pose_information_measurement_sigma_mean": float(fisher_sigma.mean().item()),
                    "pose_information_uncertainty_entropy_scale": entropy_scale,
                }
            )
        full_bank_loss = full_bank_bimnn_loss(
            query_features,
            bank_features_for_loss,
            positive_bank_indices,
            temperature=full_bank_temperature,
            hard_negative_topk=full_bank_hard_negative_topk,
            hard_negative_margin=full_bank_hard_negative_margin,
            weights=full_bank_weights,
            ignore_bank_mask=ignore_bank_mask,
            positive_bank_mask=positive_bank_mask,
            chunk_size=full_bank_stats_chunk_size,
            query_bank_scores=shared_query_bank_scores,
        )
        clean_hn_weight = max(0.0, float(full_bank_clean_hard_negative_weight))
        if clean_hn_weight > 0.0:
            if bank_xyz is None:
                bank_xyz = loc_xyz_all[full_bank_indices].to(device=device, dtype=dtype)
            if bank_uv is None or bank_valid is None:
                bank_uv, _, bank_valid = project_landmarks_to_query(bank_xyz, K, pose_gt_w2c, height, width)
            clean_hn_loss = clean_reprojection_hard_negative_loss(
                query_features,
                bank_features_for_loss,
                positive_bank_indices,
                selected_uv,
                bank_uv,
                bank_valid=bank_valid,
                reprojection_radius=full_bank_clean_reproj_radius,
                hard_negative_topk=full_bank_clean_hard_negatives,
                margin=full_bank_hard_negative_margin,
                weights=full_bank_weights,
                ignore_bank_mask=ignore_bank_mask,
                positive_bank_mask=positive_bank_mask,
                query_bank_scores=shared_query_bank_scores,
                query_bank_uv_distances=uv_dist,
            )
        diagnostics.update(
            {
                "full_bank_clean_hard_negative_loss": float(clean_hn_loss.detach().item()),
                "full_bank_clean_hard_negative_weight": clean_hn_weight,
                "full_bank_clean_reproj_radius": float(full_bank_clean_reproj_radius),
                "full_bank_clean_hard_negatives": int(full_bank_clean_hard_negatives),
            }
        )
        diagnostics.update(
            _full_bank_ignore_diagnostics(
                positive_bank_indices,
                bank_features.shape[0],
                ignore_bank_mask=ignore_bank_mask,
                positive_bank_mask=positive_bank_mask,
            )
        )
        diagnostics.update(
            _full_bank_source_relation_diagnostics(
                positive_bank_indices,
                bank_features.shape[0],
                source_bank_mask=source_bank_mask,
                ignore_bank_mask=ignore_bank_mask,
                positive_bank_mask=positive_bank_mask,
            )
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
                weights=artifact_weights[anchor_valid.to(device=artifact_weights.device)] if artifact_weights is not None else None,
            )
    if artifact_loss_scale_mode != "none":
        artifact_loss_scale = artifact_loss_scale.to(device=query_feature_map.device, dtype=query_feature_map.dtype)
        desc_loss = desc_loss * artifact_loss_scale
        multiview_loss = multiview_loss * artifact_loss_scale
        full_bank_loss = full_bank_loss * artifact_loss_scale
        anchor_loss = anchor_loss * artifact_loss_scale
    if multiview_memory is not None and update_multiview_memory:
        pose_c2w = torch.linalg.inv(pose_gt_w2c)
        camera_center = pose_c2w[:3, 3].to(device=selected_xyz.device, dtype=selected_xyz.dtype)
        view_delta = camera_center[None] - selected_xyz
        camera_distances = view_delta.norm(dim=-1)
        view_directions = F.normalize(view_delta, p=2, dim=-1)
        multiview_memory.update(
            memory_landmark_indices,
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
        "information": pose_info_weight.detach(),
        "repeatability": (positive_prob > 0.25).float(),
        "prototype": F.normalize(query_features.detach(), p=2, dim=-1),
        "multiview_positive_count": multiview_positive_count,
        "anchor_loss": torch.full_like(positive_prob, float(anchor_loss.detach().item())),
        "artifact_weight": (
            artifact_weights.detach()
            if artifact_weights is not None
            else torch.ones_like(positive_prob)
        ),
    }
    return DirectLandmarkTeacherOutput(
        desc_loss + full_bank_loss + clean_hn_loss + anchor_loss,
        desc_loss,
        multiview_loss,
        full_bank_loss,
        anchor_loss,
        zero,
        clean_hn_loss,
        stats,
        selected_full_idx,
        selected_uv.detach(),
        int(selected_count),
        diagnostics=diagnostics,
    )
