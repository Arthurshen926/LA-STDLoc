from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from localization_training.correspondence import bilinear_sample_features
from localization_training.direct_landmark_teacher import (
    filter_depth_consistent_landmarks,
    project_landmarks_to_query,
)
from localization_training.sparse_frontend import simple_nms
from localization_training.ulf_initializer import (
    grid_index_to_physical,
    sample_dense_descriptors_at_image_uv,
)
from localization_training.pose_refiner import (
    camera_center_from_w2c,
    weighted_gauss_newton_refine,
)
from localization_training.hard_candidate_teacher import (
    linearized_translation_delete_gains,
)


@dataclass
class DetectorFreeObservationBatch:
    source_indices: torch.Tensor
    query_features: torch.Tensor
    query_uv: torch.Tensor
    source_depth: torch.Tensor
    bank_uv: torch.Tensor
    bank_depth: torch.Tensor
    bank_projected: torch.Tensor
    bank_visible: torch.Tensor
    base_bank_uv: Optional[torch.Tensor] = None
    base_bank_depth: Optional[torch.Tensor] = None
    base_bank_projected: Optional[torch.Tensor] = None
    base_bank_visible: Optional[torch.Tensor] = None
    query_feature_map: Optional[torch.Tensor] = None
    target_depth_map: Optional[torch.Tensor] = None
    target_alpha_map: Optional[torch.Tensor] = None
    query_valid_mask: Optional[torch.Tensor] = None
    K: Optional[torch.Tensor] = None
    pose_w2c: Optional[torch.Tensor] = None
    positive_offsets: Optional[torch.Tensor] = None
    positive_indices: Optional[torch.Tensor] = None
    positive_reprojection_errors: Optional[torch.Tensor] = None
    query_feature_image_size: Optional[tuple] = None
    native_input_count: Optional[int] = None
    native_valid_count: Optional[int] = None
    native_selected_count: Optional[int] = None
    configured_max_observations: Optional[int] = None


@dataclass
class DetectorFreeRetrievalOutput:
    loss: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


@dataclass
class NativeAssociationResult:
    """Detached-descriptor native top-1 associations for surface BA."""

    top1_indices: torch.Tensor
    top1_scores: torch.Tensor
    top1_margins: torch.Tensor
    projected_uv: torch.Tensor
    projected_depth: torch.Tensor
    projected: torch.Tensor
    visible: torch.Tensor
    reprojection_error: torch.Tensor
    depth_sample_valid: torch.Tensor
    depth_compatible: torch.Tensor
    depth_abs_error: torch.Tensor
    depth_gate_enabled: bool
    clean: torch.Tensor


@dataclass
class LocalSoftCorrespondenceOutput:
    expected_uv: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    valid: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


def _select_observation_rows(observations, rows):
    """Select query rows while preserving per-camera bank state."""
    query_count = int(observations.source_indices.numel())
    row_indices = torch.arange(
        query_count,
        device=observations.source_indices.device,
        dtype=torch.long,
    )[rows]
    positive_offsets = None
    positive_indices = None
    positive_reprojection_errors = None
    if observations.positive_offsets is not None:
        offsets = torch.as_tensor(
            observations.positive_offsets,
            device=observations.source_indices.device,
            dtype=torch.long,
        ).reshape(-1)
        if offsets.numel() != query_count + 1:
            raise ValueError("positive_offsets must have one entry per query plus one")
        counts = offsets[1:] - offsets[:-1]
        selected_counts = counts[row_indices]
        positive_offsets = torch.cat(
            [
                offsets.new_zeros(1),
                selected_counts.cumsum(dim=0),
            ]
        )
        edge_ranges = [
            torch.arange(
                int(offsets[index].item()),
                int(offsets[index + 1].item()),
                device=offsets.device,
                dtype=torch.long,
            )
            for index in row_indices.tolist()
            if int(counts[index].item()) > 0
        ]
        edge_indices = (
            torch.cat(edge_ranges)
            if edge_ranges
            else offsets.new_empty((0,))
        )
        positive_indices = observations.positive_indices[edge_indices]
        if observations.positive_reprojection_errors is not None:
            positive_reprojection_errors = (
                observations.positive_reprojection_errors[edge_indices]
            )
    return DetectorFreeObservationBatch(
        source_indices=observations.source_indices[row_indices],
        query_features=observations.query_features[row_indices],
        query_uv=observations.query_uv[row_indices],
        source_depth=observations.source_depth[row_indices],
        bank_uv=observations.bank_uv,
        bank_depth=observations.bank_depth,
        bank_projected=observations.bank_projected,
        bank_visible=observations.bank_visible,
        base_bank_uv=observations.base_bank_uv,
        base_bank_depth=observations.base_bank_depth,
        base_bank_projected=observations.base_bank_projected,
        base_bank_visible=observations.base_bank_visible,
        query_feature_map=observations.query_feature_map,
        target_depth_map=observations.target_depth_map,
        target_alpha_map=observations.target_alpha_map,
        query_valid_mask=observations.query_valid_mask,
        K=observations.K,
        pose_w2c=observations.pose_w2c,
        positive_offsets=positive_offsets,
        positive_indices=positive_indices,
        positive_reprojection_errors=positive_reprojection_errors,
        query_feature_image_size=observations.query_feature_image_size,
        native_input_count=observations.native_input_count,
        native_valid_count=observations.native_valid_count,
        native_selected_count=int(row_indices.numel()),
        configured_max_observations=observations.configured_max_observations,
    )


def descriptor_losses_active(step, descriptor_end_step):
    descriptor_end_step = int(descriptor_end_step)
    if descriptor_end_step < 0:
        return False
    if descriptor_end_step == 0:
        return True
    return int(step) <= descriptor_end_step


def observation_adaptive_trust_weights(
    observation_count,
    *,
    power=0.5,
    minimum=0.25,
    maximum=4.0,
    eps=1.0,
):
    """Protect weakly observed landmarks without freezing well-supported ones."""
    count = torch.as_tensor(observation_count, dtype=torch.float32).reshape(-1)
    positive = count[count > 0]
    reference = positive.median() if positive.numel() else count.new_tensor(1.0)
    weights = ((reference + float(eps)) / (count + float(eps))).pow(float(power))
    weights = weights.clamp(min=float(minimum), max=float(maximum))
    return weights / weights.mean().clamp_min(1e-8)


def materialize_descriptor_residual(
    initial_features,
    residual,
    *,
    residual_scale=1.0,
    max_residual_norm=0.0,
    eps=1e-8,
):
    initial = F.normalize(initial_features.reshape(initial_features.shape[0], -1), dim=-1)
    delta = residual.reshape_as(initial)
    max_residual_norm = float(max_residual_norm)
    if max_residual_norm > 0.0:
        norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
        delta = delta * torch.clamp(
            delta.new_tensor(max_residual_norm) / norm.clamp_min(float(eps)),
            max=1.0,
        )
    return F.normalize(initial + float(residual_scale) * delta, dim=-1)


def descriptor_trust_loss(current_features, initial_features, weights=None):
    current = F.normalize(current_features.reshape(current_features.shape[0], -1), dim=-1)
    initial = F.normalize(
        initial_features.reshape(initial_features.shape[0], -1).detach(), dim=-1
    )
    per_landmark = 1.0 - (current * initial).sum(dim=-1)
    if weights is None:
        return per_landmark.mean()
    weights = torch.as_tensor(
        weights, device=per_landmark.device, dtype=per_landmark.dtype
    ).reshape(-1)
    return (per_landmark * weights).sum() / weights.sum().clamp_min(1e-8)


def _balanced_visible_indices(
    valid,
    uv,
    depth,
    *,
    max_observations,
    image_size,
    grid_rows=8,
    grid_cols=8,
    depth_bins=4,
):
    indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    max_observations = int(max_observations)
    if max_observations <= 0 or indices.numel() <= max_observations:
        return indices

    height, width = image_size
    grid_rows = max(int(grid_rows), 1)
    grid_cols = max(int(grid_cols), 1)
    depth_bins = max(int(depth_bins), 1)
    selected_uv = uv[indices]
    x_bin = torch.floor(
        selected_uv[:, 0].clamp(0, max(float(width) - 1.0, 0.0))
        / max(float(width), 1.0)
        * grid_cols
    ).long().clamp(0, grid_cols - 1)
    y_bin = torch.floor(
        selected_uv[:, 1].clamp(0, max(float(height) - 1.0, 0.0))
        / max(float(height), 1.0)
        * grid_rows
    ).long().clamp(0, grid_rows - 1)

    selected_depth = depth[indices]
    if depth_bins > 1 and selected_depth.numel() > 1:
        quantiles = torch.linspace(
            0.0,
            1.0,
            depth_bins + 1,
            device=selected_depth.device,
            dtype=selected_depth.dtype,
        )[1:-1]
        boundaries = torch.quantile(selected_depth, quantiles)
        depth_bin = torch.bucketize(selected_depth, boundaries)
    else:
        depth_bin = torch.zeros_like(x_bin)
    group_id = (depth_bin * grid_rows + y_bin) * grid_cols + x_bin

    order = torch.argsort(group_id, stable=True)
    sorted_groups = group_id[order]
    unique_groups = torch.unique(sorted_groups, sorted=True)
    group_positions = [
        order[torch.nonzero(sorted_groups == group, as_tuple=False).squeeze(1)]
        for group in unique_groups
    ]
    chosen = []
    round_index = 0
    while len(chosen) < max_observations:
        progressed = False
        for positions in group_positions:
            if round_index >= positions.numel():
                continue
            chosen.append(indices[positions[round_index]])
            progressed = True
            if len(chosen) >= max_observations:
                break
        if not progressed:
            break
        round_index += 1
    return torch.stack(chosen) if chosen else indices[:max_observations]


def build_detector_free_observations(
    bank_xyz,
    query_feature_map,
    K,
    pose_w2c,
    *,
    prediction_bank_xyz=None,
    target_depth=None,
    target_alpha=None,
    bank_visibility_mask=None,
    query_valid_mask=None,
    alpha_threshold=0.2,
    depth_abs_tolerance=1e-3,
    depth_rel_tolerance=0.01,
    max_observations=2048,
    grid_rows=8,
    grid_cols=8,
    depth_bins=4,
    pixel_center_offset=0.0,
):
    feature_map = torch.as_tensor(query_feature_map)
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape [channels, height, width]")
    height, width = feature_map.shape[-2:]
    base_bank_xyz = torch.as_tensor(
        bank_xyz, device=feature_map.device, dtype=feature_map.dtype
    ).reshape(-1, 3).detach()
    if prediction_bank_xyz is None:
        prediction_bank_xyz = base_bank_xyz
    else:
        prediction_bank_xyz = torch.as_tensor(
            prediction_bank_xyz,
            device=feature_map.device,
            dtype=feature_map.dtype,
        ).reshape(-1, 3)
        if prediction_bank_xyz.shape != base_bank_xyz.shape:
            raise ValueError("prediction_bank_xyz must match bank_xyz")
    K = torch.as_tensor(K, device=feature_map.device, dtype=feature_map.dtype)
    pose_w2c = torch.as_tensor(
        pose_w2c, device=feature_map.device, dtype=feature_map.dtype
    )
    if query_valid_mask is not None:
        query_valid_mask = torch.as_tensor(
            query_valid_mask, device=feature_map.device, dtype=torch.bool
        ).squeeze()
        if tuple(query_valid_mask.shape) != (height, width):
            raise ValueError("query_valid_mask must match the query feature grid")
    base_bank_uv, base_bank_depth, base_bank_projected = project_landmarks_to_query(
        base_bank_xyz,
        K,
        pose_w2c,
        height,
        width,
        pixel_center_offset=pixel_center_offset,
    )
    bank_uv, bank_depth, bank_projected = project_landmarks_to_query(
        prediction_bank_xyz,
        K,
        pose_w2c,
        height,
        width,
        pixel_center_offset=pixel_center_offset,
    )
    if bank_visibility_mask is None:
        base_bank_visible = filter_depth_consistent_landmarks(
            base_bank_uv,
            base_bank_depth,
            base_bank_projected,
            target_depth=target_depth,
            target_alpha=target_alpha,
            alpha_threshold=alpha_threshold,
            abs_tolerance=depth_abs_tolerance,
            rel_tolerance=depth_rel_tolerance,
        )
    else:
        bank_visibility_mask = torch.as_tensor(
            bank_visibility_mask,
            device=feature_map.device,
            dtype=torch.bool,
        ).reshape(-1)
        if bank_visibility_mask.numel() != base_bank_xyz.shape[0]:
            raise ValueError("bank_visibility_mask must have one value per landmark")
        base_bank_visible = base_bank_projected & bank_visibility_mask
    if query_valid_mask is not None and bool(base_bank_visible.any().item()):
        rounded = base_bank_uv.detach().round().long()
        visible_indices = torch.nonzero(
            base_bank_visible, as_tuple=False
        ).reshape(-1)
        base_bank_visible = base_bank_visible.clone()
        base_bank_visible[visible_indices] &= query_valid_mask[
            rounded[visible_indices, 1],
            rounded[visible_indices, 0],
        ]
    bank_visible = base_bank_visible & bank_projected
    source_indices = _balanced_visible_indices(
        base_bank_visible,
        base_bank_uv,
        base_bank_depth,
        max_observations=max_observations,
        image_size=(height, width),
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        depth_bins=depth_bins,
    )
    if source_indices.numel() == 0:
        return DetectorFreeObservationBatch(
            source_indices=source_indices,
            query_features=feature_map.new_zeros((0, feature_map.shape[0])),
            query_uv=feature_map.new_zeros((0, 2)),
            source_depth=feature_map.new_zeros((0,)),
            bank_uv=bank_uv,
            bank_depth=bank_depth,
            bank_projected=bank_projected,
            bank_visible=bank_visible,
            base_bank_uv=base_bank_uv,
            base_bank_depth=base_bank_depth,
            base_bank_projected=base_bank_projected,
            base_bank_visible=base_bank_visible,
            query_feature_map=feature_map.detach(),
            target_depth_map=None if target_depth is None else torch.as_tensor(target_depth).detach(),
            target_alpha_map=None if target_alpha is None else torch.as_tensor(target_alpha).detach(),
            query_valid_mask=query_valid_mask,
            K=K,
            pose_w2c=pose_w2c,
        )
    query_uv = base_bank_uv[source_indices].detach()
    query_features = bilinear_sample_features(feature_map.detach(), query_uv)
    feature_valid = torch.isfinite(query_features).all(dim=1) & (
        torch.linalg.norm(query_features, dim=-1) > 1e-6
    )
    source_indices = source_indices[feature_valid]
    query_uv = query_uv[feature_valid]
    query_features = query_features[feature_valid]
    return DetectorFreeObservationBatch(
        source_indices=source_indices,
        query_features=F.normalize(query_features, dim=-1),
        query_uv=query_uv,
        source_depth=base_bank_depth[source_indices].detach(),
        bank_uv=bank_uv,
        bank_depth=bank_depth,
        bank_projected=bank_projected,
        bank_visible=bank_visible,
        base_bank_uv=base_bank_uv,
        base_bank_depth=base_bank_depth,
        base_bank_projected=base_bank_projected,
        base_bank_visible=base_bank_visible,
        query_feature_map=feature_map.detach(),
        target_depth_map=None if target_depth is None else torch.as_tensor(target_depth).detach(),
        target_alpha_map=None if target_alpha is None else torch.as_tensor(target_alpha).detach(),
        query_valid_mask=query_valid_mask,
        K=K,
        pose_w2c=pose_w2c,
    )


def _balanced_sparse_keypoint_indices(
    keypoints,
    scores,
    *,
    max_observations,
    image_size,
    grid_rows=8,
    grid_cols=8,
):
    """Select native sparse keypoints without replacing them by anchor samples.

    The output remains a subset of the detector's actual proposals.  Grid
    round-robin selection keeps a high-score facade from consuming the whole
    training batch while preserving the proposal distribution seen by sparse
    PnP at inference.
    """
    keypoints = torch.as_tensor(keypoints)
    scores = torch.as_tensor(scores, device=keypoints.device).reshape(-1)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("keypoints must have shape [N, 2]")
    if scores.numel() != keypoints.shape[0]:
        raise ValueError("scores must have one value per keypoint")
    count = int(keypoints.shape[0])
    maximum = int(max_observations)
    if maximum <= 0 or count <= maximum:
        return torch.arange(count, device=keypoints.device, dtype=torch.long)

    height, width = (int(image_size[0]), int(image_size[1]))
    rows = max(int(grid_rows), 1)
    cols = max(int(grid_cols), 1)
    x_bin = torch.floor(
        keypoints[:, 0].clamp(0.0, max(float(width) - 1.0, 0.0))
        / max(float(width), 1.0)
        * cols
    ).long().clamp(0, cols - 1)
    y_bin = torch.floor(
        keypoints[:, 1].clamp(0.0, max(float(height) - 1.0, 0.0))
        / max(float(height), 1.0)
        * rows
    ).long().clamp(0, rows - 1)
    group_id = y_bin * cols + x_bin
    # Stable sorting gives reproducible ties for native SuperPoint scores.
    score_order = torch.argsort(scores, descending=True, stable=True)
    groups = []
    for group in torch.unique(group_id, sorted=True):
        in_group = score_order[group_id[score_order] == group]
        if in_group.numel() > 0:
            groups.append(in_group)
    selected = []
    rank = 0
    while len(selected) < maximum:
        progressed = False
        for group in groups:
            if rank >= group.numel():
                continue
            selected.append(group[rank])
            progressed = True
            if len(selected) >= maximum:
                break
        if not progressed:
            break
        rank += 1
    if not selected:
        return score_order[:maximum]
    return torch.stack(selected)


def _mask_at_uv(valid_mask, uv):
    valid_mask = torch.as_tensor(valid_mask, device=uv.device, dtype=torch.bool)
    if valid_mask.ndim != 2:
        raise ValueError("query_valid_mask must be two-dimensional")
    height, width = valid_mask.shape
    rounded = uv.detach().round().long()
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    keep = torch.zeros(uv.shape[0], dtype=torch.bool, device=uv.device)
    if bool(in_bounds.any().item()):
        keep[in_bounds] = valid_mask[
            rounded[in_bounds, 1], rounded[in_bounds, 0]
        ]
    return keep


def build_native_sparse_observations(
    bank_xyz,
    native_keypoints,
    native_descriptors,
    native_scores,
    K,
    pose_w2c,
    *,
    image_size,
    prediction_bank_xyz=None,
    target_depth=None,
    target_alpha=None,
    bank_visibility_mask=None,
    query_valid_mask=None,
    query_feature_map=None,
    max_observations=2048,
    grid_rows=8,
    grid_cols=8,
    positive_radius_px=2.0,
    unmatched_fraction=0.25,
    sampling_mode="detector_grid",
    pixel_center_offset=0.5,
):
    """Build supervision from deployed native SuperPoint keypoints.

    ``native_keypoints`` uses the detector grid-index convention.  The sparse
    PnP frontend converts these coordinates with ``+0.5`` only when passing
    them to the solver; projections here therefore also use the corresponding
    grid-index convention through ``pixel_center_offset=0.5``.  Unlike
    ``build_detector_free_observations``, this function never creates a query
    by sampling the dense feature map at a landmark projection.

    ``source_indices`` is a GT-only association for diagnostics, missed-
    positive supervision, and optional multiview auxiliary losses.  It is not
    inserted into retrieval candidates by ``hard_hypothesis_retrieval_loss``.
    """
    native_keypoints = torch.as_tensor(native_keypoints)
    native_descriptors = torch.as_tensor(
        native_descriptors, device=native_keypoints.device
    )
    native_scores = torch.as_tensor(
        native_scores, device=native_keypoints.device, dtype=native_keypoints.dtype
    ).reshape(-1)
    native_input_count = int(native_keypoints.shape[0])
    if native_keypoints.ndim != 2 or native_keypoints.shape[1] != 2:
        raise ValueError("native_keypoints must have shape [N, 2]")
    if native_descriptors.ndim != 2 or native_descriptors.shape[0] != native_keypoints.shape[0]:
        raise ValueError("native_descriptors must have shape [N, D]")
    if native_scores.numel() != native_keypoints.shape[0]:
        raise ValueError("native_scores must have one value per keypoint")
    height, width = (int(image_size[0]), int(image_size[1]))
    device = native_keypoints.device
    dtype = native_keypoints.dtype
    base_bank_xyz = torch.as_tensor(
        bank_xyz, device=device, dtype=dtype
    ).reshape(-1, 3).detach()
    if prediction_bank_xyz is None:
        prediction_bank_xyz = base_bank_xyz
    else:
        prediction_bank_xyz = torch.as_tensor(
            prediction_bank_xyz, device=device, dtype=dtype
        ).reshape(-1, 3)
        if prediction_bank_xyz.shape != base_bank_xyz.shape:
            raise ValueError("prediction_bank_xyz must match bank_xyz")
    K = torch.as_tensor(K, device=device, dtype=dtype)
    pose_w2c = torch.as_tensor(pose_w2c, device=device, dtype=dtype)
    base_bank_uv, base_bank_depth, base_bank_projected = project_landmarks_to_query(
        base_bank_xyz,
        K,
        pose_w2c,
        height,
        width,
        pixel_center_offset=pixel_center_offset,
    )
    bank_uv, bank_depth, bank_projected = project_landmarks_to_query(
        prediction_bank_xyz,
        K,
        pose_w2c,
        height,
        width,
        pixel_center_offset=pixel_center_offset,
    )
    if bank_visibility_mask is None:
        base_bank_visible = base_bank_projected
    else:
        bank_visibility_mask = torch.as_tensor(
            bank_visibility_mask, device=device, dtype=torch.bool
        ).reshape(-1)
        if bank_visibility_mask.numel() != base_bank_xyz.shape[0]:
            raise ValueError("bank_visibility_mask must have one value per landmark")
        base_bank_visible = base_bank_projected & bank_visibility_mask
    if query_valid_mask is not None:
        query_valid_mask = torch.as_tensor(
            query_valid_mask, device=device, dtype=torch.bool
        ).squeeze()
        if tuple(query_valid_mask.shape) != (height, width):
            raise ValueError("query_valid_mask must match the native image size")
        base_bank_visible = base_bank_visible & _mask_at_uv(
            query_valid_mask, base_bank_uv
        )
    bank_visible = base_bank_visible & bank_projected

    descriptor_valid = torch.isfinite(native_descriptors).all(dim=1) & (
        torch.linalg.norm(native_descriptors, dim=-1) > 1e-6
    )
    coordinate_valid = (
        torch.isfinite(native_keypoints).all(dim=1)
        & (native_keypoints[:, 0] >= 0.0)
        & (native_keypoints[:, 0] <= float(width - 1))
        & (native_keypoints[:, 1] >= 0.0)
        & (native_keypoints[:, 1] <= float(height - 1))
    )
    valid = descriptor_valid & coordinate_valid
    if query_valid_mask is not None and bool(valid.any().item()):
        valid &= _mask_at_uv(query_valid_mask, native_keypoints)
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    empty_features = native_descriptors.new_zeros((0, native_descriptors.shape[1]))
    native_feature_map = None
    if query_feature_map is not None:
        native_feature_map = torch.as_tensor(
            query_feature_map,
            device=device,
            dtype=native_descriptors.dtype,
        ).squeeze(0)
        if native_feature_map.ndim != 3:
            raise ValueError("query_feature_map must have shape [C, H, W]")
        if native_feature_map.shape[0] != native_descriptors.shape[1]:
            raise ValueError(
                "query_feature_map channels must match native descriptor dimension"
            )
        effective_hw = (
            int(native_feature_map.shape[-2]) * 8,
            int(native_feature_map.shape[-1]) * 8,
        )
        if effective_hw != (height, width):
            raise ValueError(
                "native query feature map must be the stride-8 map from the "
                f"same resized RGB input: effective={effective_hw} image={(height, width)}"
            )
    if valid_indices.numel() == 0:
        return DetectorFreeObservationBatch(
            source_indices=torch.empty(0, dtype=torch.long, device=device),
            query_features=empty_features,
            query_uv=native_keypoints.new_zeros((0, 2)),
            source_depth=native_keypoints.new_zeros((0,)),
            bank_uv=bank_uv,
            bank_depth=bank_depth,
            bank_projected=bank_projected,
            bank_visible=bank_visible,
            base_bank_uv=base_bank_uv,
            base_bank_depth=base_bank_depth,
            base_bank_projected=base_bank_projected,
            base_bank_visible=base_bank_visible,
            target_depth_map=None if target_depth is None else torch.as_tensor(target_depth).detach(),
            target_alpha_map=None if target_alpha is None else torch.as_tensor(target_alpha).detach(),
            query_valid_mask=query_valid_mask,
            K=K,
            pose_w2c=pose_w2c,
            positive_offsets=torch.zeros(1, dtype=torch.long, device=device),
            positive_indices=torch.empty(0, dtype=torch.long, device=device),
            positive_reprojection_errors=native_keypoints.new_zeros((0,)),
            query_feature_map=(
                None if native_feature_map is None else native_feature_map.detach()
            ),
            query_feature_image_size=(
                None if native_feature_map is None else (height, width)
            ),
            native_input_count=native_input_count,
            native_valid_count=0,
            native_selected_count=0,
            configured_max_observations=int(max_observations),
        )

    sampling_mode = str(sampling_mode)
    if sampling_mode not in {"detector_grid", "label_balanced"}:
        raise ValueError(
            "sampling_mode must be detector_grid or label_balanced"
        )
    # ``detector_grid`` is the formal path: choose the actual deployment
    # proposals before inspecting geometry. ``label_balanced`` is retained
    # solely as an ablation for deliberately changing the train distribution.
    reservoir_limit = int(max_observations)
    if sampling_mode == "label_balanced" and reservoir_limit > 0:
        reservoir_limit *= 2
    selected_local = _balanced_sparse_keypoint_indices(
        native_keypoints[valid_indices],
        native_scores[valid_indices],
        max_observations=reservoir_limit,
        image_size=(height, width),
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    selected = valid_indices[selected_local]
    if sampling_mode == "detector_grid":
        expected_count = (
            min(int(valid_indices.numel()), int(max_observations))
            if int(max_observations) > 0
            else int(valid_indices.numel())
        )
        if int(selected.numel()) != expected_count:
            raise RuntimeError(
                "native detector-grid proposal coverage changed unexpectedly: "
                f"selected={int(selected.numel())} expected={expected_count}"
            )
    query_uv = native_keypoints[selected]
    query_features = F.normalize(native_descriptors[selected], dim=-1)
    query_scores = native_scores[selected]
    source_indices = torch.full(
        (selected.numel(),), -1, dtype=torch.long, device=device
    )
    visible_indices = torch.nonzero(base_bank_visible, as_tuple=False).reshape(-1)
    distance = None
    if visible_indices.numel() > 0 and selected.numel() > 0:
        distance = torch.cdist(query_uv.float(), base_bank_uv[visible_indices].float())
        nearest_distance, nearest_position = distance.min(dim=1)
        matched = nearest_distance <= float(positive_radius_px)
        source_indices[matched] = visible_indices[nearest_position[matched]]

    # This optional ablation rebalances labels after detector proposal
    # selection. It must never be used by the main candidate-aligned method,
    # because it changes the sparse frontend distribution seen at inference.
    maximum = int(max_observations)
    if (
        sampling_mode == "label_balanced"
        and maximum > 0
        and selected.numel() > maximum
    ):
        unmatched_limit = int(round(maximum * max(min(float(unmatched_fraction), 1.0), 0.0)))
        matched_positions = torch.nonzero(source_indices >= 0, as_tuple=False).reshape(-1)
        unmatched_positions = torch.nonzero(source_indices < 0, as_tuple=False).reshape(-1)
        selected_positions = []
        if matched_positions.numel() > 0:
            selected_positions.append(matched_positions[: max(maximum - unmatched_limit, 0)])
        if unmatched_limit > 0 and unmatched_positions.numel() > 0:
            selected_positions.append(unmatched_positions[:unmatched_limit])
        chosen = (
            torch.cat(selected_positions)
            if selected_positions
            else torch.empty(0, dtype=torch.long, device=device)
        )
        if chosen.numel() < maximum:
            remaining_mask = torch.ones(selected.numel(), dtype=torch.bool, device=device)
            remaining_mask[chosen] = False
            chosen = torch.cat(
                [chosen, torch.nonzero(remaining_mask, as_tuple=False).reshape(-1)[: maximum - chosen.numel()]]
            )
        chosen = chosen[:maximum]
        query_uv = query_uv[chosen]
        query_features = query_features[chosen]
        query_scores = query_scores[chosen]
        source_indices = source_indices[chosen]
        if distance is not None:
            distance = distance[chosen]

    if distance is None:
        positive_offsets = torch.zeros(
            source_indices.numel() + 1,
            dtype=torch.long,
            device=device,
        )
        positive_indices = torch.empty(0, dtype=torch.long, device=device)
        positive_reprojection_errors = query_uv.new_zeros((0,))
    else:
        positive_mask = distance <= float(positive_radius_px)
        positive_edges = torch.nonzero(positive_mask, as_tuple=False)
        positive_counts = positive_mask.sum(dim=1, dtype=torch.long)
        positive_offsets = torch.cat(
            [
                positive_counts.new_zeros(1),
                positive_counts.cumsum(dim=0),
            ]
        )
        positive_indices = visible_indices[positive_edges[:, 1]]
        positive_reprojection_errors = distance[
            positive_edges[:, 0], positive_edges[:, 1]
        ].to(dtype=query_uv.dtype)

    source_depth = query_uv.new_zeros(source_indices.shape[0])
    source_valid = source_indices >= 0
    if bool(source_valid.any().item()):
        source_depth[source_valid] = base_bank_depth[source_indices[source_valid]].detach()
    return DetectorFreeObservationBatch(
        source_indices=source_indices,
        query_features=query_features,
        query_uv=query_uv,
        source_depth=source_depth,
        bank_uv=bank_uv,
        bank_depth=bank_depth,
        bank_projected=bank_projected,
        bank_visible=bank_visible,
        base_bank_uv=base_bank_uv,
        base_bank_depth=base_bank_depth,
        base_bank_projected=base_bank_projected,
        base_bank_visible=base_bank_visible,
        query_feature_map=(
            None if native_feature_map is None else native_feature_map.detach()
        ),
        target_depth_map=None if target_depth is None else torch.as_tensor(target_depth).detach(),
        target_alpha_map=None if target_alpha is None else torch.as_tensor(target_alpha).detach(),
        query_valid_mask=query_valid_mask,
        K=K,
        pose_w2c=pose_w2c,
        positive_offsets=positive_offsets,
        positive_indices=positive_indices,
        positive_reprojection_errors=positive_reprojection_errors,
        query_feature_image_size=(
            None if native_feature_map is None else (height, width)
        ),
        native_input_count=native_input_count,
        native_valid_count=int(valid_indices.numel()),
        native_selected_count=int(query_uv.shape[0]),
        configured_max_observations=int(max_observations),
    )


def jitter_detector_free_observations(
    observations,
    *,
    standard_deviation=0.0,
    maximum=0.0,
    generator=None,
):
    feature_map = observations.query_feature_map
    query_count = int(observations.source_indices.numel())
    if (
        feature_map is None
        or query_count == 0
        or float(standard_deviation) <= 0.0
        or float(maximum) <= 0.0
    ):
        return observations
    offsets = torch.randn(
        (query_count, 2),
        device=observations.query_uv.device,
        dtype=observations.query_uv.dtype,
        generator=generator,
    ) * float(standard_deviation)
    offset_norm = torch.linalg.norm(offsets, dim=-1, keepdim=True)
    offsets = offsets * torch.clamp(
        offsets.new_tensor(float(maximum)) / offset_norm.clamp_min(1e-8),
        max=1.0,
    )
    query_uv = observations.query_uv + offsets
    height, width = _query_feature_domain(observations)
    query_uv[:, 0].clamp_(0.0, float(width - 1))
    query_uv[:, 1].clamp_(0.0, float(height - 1))
    query_features = bilinear_sample_features(feature_map.detach(), query_uv)
    valid = torch.isfinite(query_features).all(dim=1) & (
        torch.linalg.norm(query_features, dim=-1) > 1e-6
    )
    return DetectorFreeObservationBatch(
        source_indices=observations.source_indices[valid],
        query_features=F.normalize(query_features[valid], dim=-1),
        query_uv=query_uv[valid],
        source_depth=observations.source_depth[valid],
        bank_uv=observations.bank_uv,
        bank_depth=observations.bank_depth,
        bank_projected=observations.bank_projected,
        bank_visible=observations.bank_visible,
        base_bank_uv=observations.base_bank_uv,
        base_bank_depth=observations.base_bank_depth,
        base_bank_projected=observations.base_bank_projected,
        base_bank_visible=observations.base_bank_visible,
        query_feature_map=feature_map,
        target_depth_map=observations.target_depth_map,
        target_alpha_map=observations.target_alpha_map,
        query_valid_mask=observations.query_valid_mask,
        K=observations.K,
        pose_w2c=observations.pose_w2c,
    )


def build_score_proposal_observations(
    observations,
    proposal_score_map,
    *,
    max_proposals=512,
    nms_radius=2,
    score_threshold=0.0,
    positive_search_radius_px=2.0,
    include_unmatched=False,
):
    """Build detector-independent query observations from a frozen score head."""
    feature_map = observations.query_feature_map
    if feature_map is None:
        raise ValueError("Score proposals require the frozen query feature map")
    height, width = feature_map.shape[-2:]
    score_map = torch.as_tensor(
        proposal_score_map,
        device=feature_map.device,
        dtype=feature_map.dtype,
    ).squeeze()
    if score_map.ndim != 2:
        raise ValueError("proposal_score_map must be two-dimensional")
    if tuple(score_map.shape) != (height, width):
        score_map = F.adaptive_max_pool2d(
            score_map[None, None],
            output_size=(height, width),
        )[0, 0]

    nms_scores = simple_nms(score_map[None, None], int(nms_radius))[0, 0]
    flat_scores = nms_scores.reshape(-1)
    proposal_count = min(max(int(max_proposals), 0), int(flat_scores.numel()))
    if proposal_count == 0:
        selected = torch.empty(0, device=feature_map.device, dtype=torch.long)
    else:
        values, selected = torch.topk(flat_scores, proposal_count)
        selected = selected[
            torch.isfinite(values) & (values > float(score_threshold))
        ]
    query_uv = torch.stack(
        [selected % width, torch.div(selected, width, rounding_mode="floor")],
        dim=1,
    ).to(dtype=feature_map.dtype)

    teacher_uv = (
        observations.base_bank_uv
        if observations.base_bank_uv is not None
        else observations.bank_uv
    )
    teacher_depth = (
        observations.base_bank_depth
        if observations.base_bank_depth is not None
        else observations.bank_depth
    )
    teacher_visible = (
        observations.base_bank_visible
        if observations.base_bank_visible is not None
        else observations.bank_visible
    )
    visible_indices = torch.nonzero(
        teacher_visible, as_tuple=False
    ).squeeze(1)
    if query_uv.numel() == 0 or visible_indices.numel() == 0:
        source_indices = visible_indices.new_full(
            (query_uv.shape[0],), -1
        )
        if not include_unmatched:
            source_indices = visible_indices.new_empty(0)
            query_uv = feature_map.new_zeros((0, 2))
    else:
        distances = torch.cdist(query_uv, teacher_uv[visible_indices])
        nearest_distance, nearest_position = distances.min(dim=1)
        matched = nearest_distance <= float(positive_search_radius_px)
        if include_unmatched:
            source_indices = visible_indices.new_full(
                (query_uv.shape[0],), -1
            )
            source_indices[matched] = visible_indices[nearest_position[matched]]
        else:
            query_uv = query_uv[matched]
            source_indices = visible_indices[nearest_position[matched]]

    query_features = bilinear_sample_features(feature_map.detach(), query_uv)
    feature_valid = torch.isfinite(query_features).all(dim=1) & (
        torch.linalg.norm(query_features, dim=-1) > 1e-6
    )
    source_indices = source_indices[feature_valid]
    query_uv = query_uv[feature_valid]
    query_features = query_features[feature_valid]
    source_depth = feature_map.new_zeros(source_indices.shape[0])
    matched = source_indices >= 0
    if bool(matched.any().item()):
        source_depth[matched] = teacher_depth[source_indices[matched]].detach()
    return DetectorFreeObservationBatch(
        source_indices=source_indices,
        query_features=F.normalize(query_features, dim=-1),
        query_uv=query_uv,
        source_depth=source_depth,
        bank_uv=observations.bank_uv,
        bank_depth=observations.bank_depth,
        bank_projected=observations.bank_projected,
        bank_visible=observations.bank_visible,
        base_bank_uv=observations.base_bank_uv,
        base_bank_depth=observations.base_bank_depth,
        base_bank_projected=observations.base_bank_projected,
        base_bank_visible=observations.base_bank_visible,
        query_feature_map=feature_map,
        target_depth_map=observations.target_depth_map,
        target_alpha_map=observations.target_alpha_map,
        query_valid_mask=observations.query_valid_mask,
        K=observations.K,
        pose_w2c=observations.pose_w2c,
    )


def multiview_descriptor_loss(bank_features, observations):
    valid = observations.source_indices >= 0
    if not bool(valid.any().item()):
        return bank_features.sum() * 0.0
    source = F.normalize(bank_features[observations.source_indices[valid]], dim=-1)
    query = F.normalize(observations.query_features[valid].detach(), dim=-1)
    return (1.0 - (source * query).sum(dim=-1)).mean()


def _query_feature_domain(observations):
    feature_map = observations.query_feature_map
    if feature_map is None:
        raise ValueError("A frozen query feature map is required")
    if observations.query_feature_image_size is None:
        return int(feature_map.shape[-2]), int(feature_map.shape[-1])
    height, width = observations.query_feature_image_size
    return int(height), int(width)


def _sample_observation_features(observations, uv):
    feature_map = observations.query_feature_map
    if feature_map is None:
        raise ValueError("A frozen query feature map is required")
    if observations.query_feature_image_size is None:
        return bilinear_sample_features(feature_map.detach(), uv)
    return sample_dense_descriptors_at_image_uv(
        feature_map.detach(),
        grid_index_to_physical(uv),
        observations.query_feature_image_size,
    )


def local_correlation_peak_loss(
    bank_features,
    observations,
    *,
    radius=3,
    target_sigma=1.0,
    temperature=0.07,
):
    """Concentrate each anchor's frozen-image correlation peak at its GT projection."""
    feature_map = observations.query_feature_map
    if feature_map is None:
        raise ValueError("Local correlation loss requires the frozen query feature map")
    source_valid = observations.source_indices >= 0
    observations = _select_observation_rows(observations, source_valid)
    query_count = int(observations.source_indices.numel())
    radius = max(int(radius), 0)
    if query_count == 0:
        return bank_features.sum() * 0.0, {
            "local_query_count": 0,
            "local_center_top1_ratio": 0.0,
            "local_peak_offset_mean_px": 0.0,
            "local_expected_offset_mean_px": 0.0,
        }

    axis = torch.arange(
        -radius,
        radius + 1,
        device=bank_features.device,
        dtype=observations.query_uv.dtype,
    )
    offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack([offset_x.reshape(-1), offset_y.reshape(-1)], dim=-1)
    patch_uv = observations.query_uv[:, None, :] + offsets[None, :, :]
    height, width = _query_feature_domain(observations)
    patch_valid = (
        (patch_uv[..., 0] >= 0.0)
        & (patch_uv[..., 0] <= float(width - 1))
        & (patch_uv[..., 1] >= 0.0)
        & (patch_uv[..., 1] <= float(height - 1))
    )
    patch_features = _sample_observation_features(
        observations,
        patch_uv.reshape(-1, 2),
    ).reshape(query_count, offsets.shape[0], -1)
    patch_features = F.normalize(patch_features, dim=-1)
    source_features = F.normalize(
        bank_features[observations.source_indices], dim=-1
    )
    logits = torch.einsum("qd,qpd->qp", source_features, patch_features)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~patch_valid, -torch.inf)

    sigma = max(float(target_sigma), 1e-6)
    target_logits = -offsets.square().sum(dim=-1) / (2.0 * sigma * sigma)
    target_logits = target_logits[None].expand(query_count, -1)
    target_logits = target_logits.masked_fill(~patch_valid, -torch.inf)
    target = torch.softmax(target_logits, dim=-1)
    log_probability = torch.log_softmax(logits, dim=-1)
    log_probability = log_probability.masked_fill(~patch_valid, 0.0)
    loss = -(target * log_probability).sum(dim=-1).mean()

    with torch.no_grad():
        probability = torch.softmax(logits, dim=-1)
        peak_index = logits.argmax(dim=-1)
        peak_offsets = offsets[peak_index]
        expected_offsets = probability @ offsets
        center_index = radius * (2 * radius + 1) + radius
        diagnostics = {
            "local_query_count": query_count,
            "local_center_top1_ratio": float(
                (peak_index == center_index).float().mean().item()
            ),
            "local_peak_offset_mean_px": float(
                torch.linalg.norm(peak_offsets, dim=-1).mean().item()
            ),
            "local_expected_offset_mean_px": float(
                torch.linalg.norm(expected_offsets, dim=-1).mean().item()
            ),
        }
    return loss, diagnostics


def group_geometric_consistency_loss(
    predicted_uv,
    target_uv,
    group_ids,
    *,
    valid=None,
    minimum_edge_px=1.0,
):
    """Preserve local direction, scale, and orientation within teacher groups."""
    predicted_uv = torch.as_tensor(predicted_uv)
    target_uv = torch.as_tensor(
        target_uv, device=predicted_uv.device, dtype=predicted_uv.dtype
    )
    group_ids = torch.as_tensor(
        group_ids, device=predicted_uv.device, dtype=torch.long
    ).reshape(-1)
    if predicted_uv.shape != target_uv.shape or predicted_uv.shape[-1] != 2:
        raise ValueError("predicted_uv and target_uv must be aligned Nx2 tensors")
    if group_ids.numel() != predicted_uv.shape[0]:
        raise ValueError("group_ids must align with predicted_uv")
    if valid is None:
        valid = torch.ones(
            predicted_uv.shape[0], device=predicted_uv.device, dtype=torch.bool
        )
    else:
        valid = torch.as_tensor(
            valid, device=predicted_uv.device, dtype=torch.bool
        ).reshape(-1)
    valid = (
        valid
        & torch.isfinite(predicted_uv).all(dim=1)
        & torch.isfinite(target_uv).all(dim=1)
    )

    direction_terms = []
    scale_terms = []
    orientation_terms = []
    group_count = 0
    edge_count = 0
    triangle_count = 0
    for group_id in torch.unique(group_ids[valid]).tolist():
        members = torch.nonzero(
            valid & (group_ids == int(group_id)), as_tuple=False
        ).reshape(-1)
        if members.numel() < 3:
            continue
        pair_indices = torch.combinations(members, r=2)
        target_edges = target_uv[pair_indices[:, 1]] - target_uv[pair_indices[:, 0]]
        predicted_edges = (
            predicted_uv[pair_indices[:, 1]] - predicted_uv[pair_indices[:, 0]]
        )
        target_lengths = torch.linalg.norm(target_edges, dim=1)
        predicted_lengths = torch.linalg.norm(predicted_edges, dim=1)
        edge_valid = (
            torch.isfinite(target_lengths)
            & torch.isfinite(predicted_lengths)
            & (target_lengths >= float(minimum_edge_px))
            & (predicted_lengths > 1e-6)
        )
        if int(edge_valid.sum().item()) < 2:
            continue
        target_edges = target_edges[edge_valid]
        predicted_edges = predicted_edges[edge_valid]
        target_lengths = target_lengths[edge_valid]
        predicted_lengths = predicted_lengths[edge_valid]
        direction_terms.append(
            1.0
            - (
                F.normalize(predicted_edges, dim=1)
                * F.normalize(target_edges, dim=1)
            ).sum(dim=1)
        )
        scale_terms.append(
            F.smooth_l1_loss(
                torch.log(predicted_lengths / target_lengths.clamp_min(1e-6)),
                torch.zeros_like(predicted_lengths),
                reduction="none",
                beta=0.1,
            )
        )
        edge_count += int(edge_valid.sum().item())

        root = members[0]
        other = members[1:]
        triangle_pairs = torch.combinations(other, r=2)
        if triangle_pairs.numel() > 0:
            target_a = target_uv[triangle_pairs[:, 0]] - target_uv[root]
            target_b = target_uv[triangle_pairs[:, 1]] - target_uv[root]
            predicted_a = predicted_uv[triangle_pairs[:, 0]] - predicted_uv[root]
            predicted_b = predicted_uv[triangle_pairs[:, 1]] - predicted_uv[root]
            target_denominator = (
                torch.linalg.norm(target_a, dim=1)
                * torch.linalg.norm(target_b, dim=1)
            ).clamp_min(1e-6)
            predicted_denominator = (
                torch.linalg.norm(predicted_a, dim=1)
                * torch.linalg.norm(predicted_b, dim=1)
            ).clamp_min(1e-6)
            target_sine = (
                target_a[:, 0] * target_b[:, 1]
                - target_a[:, 1] * target_b[:, 0]
            ) / target_denominator
            predicted_sine = (
                predicted_a[:, 0] * predicted_b[:, 1]
                - predicted_a[:, 1] * predicted_b[:, 0]
            ) / predicted_denominator
            triangle_valid = (
                torch.isfinite(target_sine)
                & torch.isfinite(predicted_sine)
                & (target_sine.abs() >= 0.05)
            )
            if bool(triangle_valid.any()):
                orientation_terms.append(
                    F.smooth_l1_loss(
                        predicted_sine[triangle_valid],
                        target_sine[triangle_valid],
                        reduction="none",
                        beta=0.1,
                    )
                )
                triangle_count += int(triangle_valid.sum().item())
        group_count += 1

    zero = predicted_uv.sum() * 0.0
    direction_loss = (
        torch.cat(direction_terms).mean() if direction_terms else zero
    )
    scale_loss = torch.cat(scale_terms).mean() if scale_terms else zero
    orientation_loss = (
        torch.cat(orientation_terms).mean() if orientation_terms else zero
    )
    loss = direction_loss + 0.5 * scale_loss + 0.25 * orientation_loss
    diagnostics = {
        "native_semidense_lgcv_group_count": group_count,
        "native_semidense_lgcv_edge_count": edge_count,
        "native_semidense_lgcv_triangle_count": triangle_count,
        "native_semidense_lgcv_direction_loss": float(
            direction_loss.detach().item()
        ),
        "native_semidense_lgcv_scale_loss": float(scale_loss.detach().item()),
        "native_semidense_lgcv_orientation_loss": float(
            orientation_loss.detach().item()
        ),
        "native_semidense_lgcv_loss": float(loss.detach().item()),
    }
    return loss, diagnostics


@torch.no_grad()
def counterfactual_pose_safe_mask(
    bank_xyz,
    landmark_indices,
    observed_uv,
    K,
    pose_w2c,
    *,
    maximum_delete_gain_m=0.0,
    minimum_correspondences=6,
):
    """Keep observations whose deletion cannot improve linearized pose bias."""
    landmark_indices = torch.as_tensor(
        landmark_indices, device=bank_xyz.device, dtype=torch.long
    ).reshape(-1)
    count = int(landmark_indices.numel())
    safe = torch.ones(count, device=bank_xyz.device, dtype=torch.bool)
    diagnostics = {
        "native_semidense_pose_safe_evaluable": 0.0,
        "native_semidense_pose_safe_input_count": count,
        "native_semidense_pose_safe_count": count,
        "native_semidense_pose_harmful_count": 0,
        "native_semidense_pose_delete_gain_mean_m": 0.0,
        "native_semidense_pose_delete_gain_p50_m": 0.0,
        "native_semidense_pose_delete_gain_p90_m": 0.0,
        "native_semidense_pose_delete_gain_max_m": 0.0,
    }
    if (
        count < max(int(minimum_correspondences), 4)
        or K is None
        or pose_w2c is None
    ):
        return safe, diagnostics
    gains = linearized_translation_delete_gains(
        bank_xyz[landmark_indices].detach().double(),
        torch.as_tensor(observed_uv).detach().double() + 0.5,
        torch.as_tensor(K).detach().double(),
        torch.as_tensor(pose_w2c).detach().double(),
        torch.as_tensor(pose_w2c).detach().double(),
    ).to(device=bank_xyz.device, dtype=bank_xyz.dtype)
    safe = gains <= float(maximum_delete_gain_m)
    diagnostics.update(
        {
            "native_semidense_pose_safe_evaluable": 1.0,
            "native_semidense_pose_safe_count": int(safe.sum().item()),
            "native_semidense_pose_harmful_count": int((~safe).sum().item()),
            "native_semidense_pose_delete_gain_mean_m": float(
                gains.mean().item()
            ),
            "native_semidense_pose_delete_gain_p50_m": float(
                gains.quantile(0.5).item()
            ),
            "native_semidense_pose_delete_gain_p90_m": float(
                gains.quantile(0.9).item()
            ),
            "native_semidense_pose_delete_gain_max_m": float(gains.max().item()),
        }
    )
    return safe, diagnostics


def local_group_identity_assignment_loss(
    bank_features,
    teacher_features,
    teacher_indices,
    teacher_uv,
    group_ids,
    *,
    temperature=0.07,
    positive_radius_px=2.0,
):
    """Preserve one-to-one local identity inside each projected surface group."""
    losses = []
    group_count = 0
    row_count = 0
    for group_id in torch.unique(group_ids).tolist():
        rows = torch.nonzero(
            group_ids == int(group_id), as_tuple=False
        ).reshape(-1)
        if rows.numel() < 2:
            continue
        descriptors = F.normalize(bank_features[teacher_indices[rows]], dim=1)
        targets = F.normalize(teacher_features[rows].detach(), dim=1)
        logits = descriptors @ targets.T
        logits = logits / max(float(temperature), 1e-6)
        positive = torch.cdist(
            teacher_uv[rows].float(), teacher_uv[rows].float()
        ) <= float(positive_radius_px)
        positive.fill_diagonal_(True)
        positive_logits = logits.masked_fill(~positive, -torch.inf)
        row_loss = torch.logsumexp(logits, dim=1) - torch.logsumexp(
            positive_logits, dim=1
        )
        # Symmetric supervision prevents several landmarks from collapsing
        # onto the same query location.
        column_loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(
            positive_logits, dim=0
        )
        losses.append(0.5 * (row_loss + column_loss))
        group_count += 1
        row_count += int(rows.numel())
    zero = bank_features.sum() * 0.0
    loss = torch.cat(losses).mean() if losses else zero
    return loss, {
        "native_semidense_local_identity_group_count": group_count,
        "native_semidense_local_identity_row_count": row_count,
        "native_semidense_local_identity_loss": float(loss.detach().item()),
    }


def global_margin_preservation_loss(
    bank_features,
    reference_bank_features,
    observations,
    rows,
    positive_landmarks,
):
    """Prevent a protected local step from reducing an existing clean margin."""
    rows = torch.as_tensor(
        rows, device=bank_features.device, dtype=torch.long
    ).reshape(-1)
    if rows.numel() == 0 or reference_bank_features is None:
        zero = bank_features.sum() * 0.0
        return zero, {
            "native_semidense_margin_preserve_count": 0,
            "native_semidense_margin_preserve_violation_rate": 0.0,
            "native_semidense_margin_preserve_loss": 0.0,
        }
    query = F.normalize(observations.query_features[rows].detach(), dim=1)
    current = F.normalize(bank_features, dim=1)
    reference = F.normalize(reference_bank_features.detach(), dim=1)
    current_scores = query @ current.T
    with torch.no_grad():
        reference_scores = query @ reference.T
    positive_landmarks = torch.as_tensor(
        positive_landmarks, device=bank_features.device, dtype=torch.long
    ).reshape(-1)
    negative_mask = torch.ones_like(current_scores, dtype=torch.bool)
    negative_mask[
        torch.arange(rows.numel(), device=bank_features.device),
        positive_landmarks,
    ] = False
    if (
        observations.positive_offsets is not None
        and observations.positive_indices is not None
    ):
        offsets = observations.positive_offsets
        for output_row, observation_row in enumerate(rows.tolist()):
            start = int(offsets[observation_row].item())
            end = int(offsets[observation_row + 1].item())
            if end > start:
                negative_mask[
                    output_row,
                    observations.positive_indices[start:end],
                ] = False
    current_positive = current_scores.gather(
        1, positive_landmarks[:, None]
    ).squeeze(1)
    reference_positive = reference_scores.gather(
        1, positive_landmarks[:, None]
    ).squeeze(1)
    current_negative = current_scores.masked_fill(
        ~negative_mask, -torch.inf
    ).max(dim=1).values
    reference_negative = reference_scores.masked_fill(
        ~negative_mask, -torch.inf
    ).max(dim=1).values
    current_margin = current_positive - current_negative
    reference_margin = reference_positive - reference_negative
    degradation = reference_margin - current_margin
    loss = F.relu(degradation).mean()
    return loss, {
        "native_semidense_margin_preserve_count": int(rows.numel()),
        "native_semidense_margin_preserve_violation_rate": float(
            (degradation > 0).float().mean().detach().item()
        ),
        "native_semidense_margin_reference_mean": float(
            reference_margin.mean().detach().item()
        ),
        "native_semidense_margin_current_mean": float(
            current_margin.mean().detach().item()
        ),
        "native_semidense_margin_preserve_loss": float(loss.detach().item()),
    }


def native_semidense_neighborhood_loss(
    bank_features,
    bank_xyz,
    bank_normals,
    observations,
    *,
    positive_radius_px=2.0,
    max_anchors=64,
    neighbors_per_anchor=1,
    neighborhood_radius_m=0.25,
    normal_cosine=0.8,
    local_radius_px=8,
    target_sigma_px=2.0,
    temperature=0.07,
    pose_safe_max_delete_gain_m=-1.0,
    pose_safe_min_correspondences=6,
    pose_safe_teacher_pairs=False,
    lgcv_weight=0.0,
    lgcv_minimum_edge_px=1.0,
    protected_v2=False,
    measurement_min_reprojection_px=2.0,
    measurement_max_reprojection_px=8.0,
    surface_point_plane_m=0.03,
    surface_max_distance_m=0.15,
    surface_normal_cosine=0.95,
    projected_neighbor_radius_px=64.0,
    local_identity_weight=0.0,
    margin_preservation_weight=0.0,
    reference_bank_features=None,
):
    """Distill local surface structure from real native-query proposals.

    Candidate mining uses the current deployment descriptor field. Only
    current top-1 pairs that are already GT-reprojection clean seed the
    teacher. Their surface neighbors are projected with the training GT pose
    and supervised against the frozen dense SuperPoint map from the exact same
    resized RGB input. This branch is training-only and never changes the
    deployment candidate graph.
    """
    zero = bank_features.sum() * 0.0
    if observations.query_feature_map is None:
        return zero, {"native_semidense_active": 0.0}
    query_count = int(observations.query_features.shape[0])
    bank_count = int(bank_features.shape[0])
    if query_count == 0 or bank_count == 0:
        return zero, {
            "native_semidense_active": 1.0,
            "native_semidense_clean_anchor_count": 0,
            "native_semidense_teacher_pair_count": 0,
        }

    bank_xyz = torch.as_tensor(
        bank_xyz, device=bank_features.device, dtype=bank_features.dtype
    ).reshape(-1, 3)
    bank_normals = F.normalize(
        torch.as_tensor(
            bank_normals,
            device=bank_features.device,
            dtype=bank_features.dtype,
        ).reshape(-1, 3),
        dim=-1,
    )
    if bank_xyz.shape[0] != bank_count or bank_normals.shape[0] != bank_count:
        raise ValueError("native semidense geometry must match the descriptor bank")

    pose_safe_diagnostics = {
        "native_semidense_pose_safe_enabled": float(
            float(pose_safe_max_delete_gain_m) >= 0.0
        )
    }
    with torch.no_grad():
        query = F.normalize(observations.query_features.detach(), dim=-1)
        bank = F.normalize(bank_features.detach(), dim=-1)
        top_count = min(2, bank_count)
        top_scores, top_indices = torch.topk(query @ bank.T, k=top_count, dim=1)
        top1 = top_indices[:, 0]
        margin = (
            top_scores[:, 0] - top_scores[:, 1]
            if top_count > 1
            else torch.full_like(top_scores[:, 0], torch.inf)
        )
        top1_distance = torch.linalg.norm(
            observations.bank_uv[top1] - observations.query_uv,
            dim=-1,
        )
        clean = observations.bank_visible[top1] & (
            top1_distance <= float(positive_radius_px)
        )
        protected_rows = torch.nonzero(clean, as_tuple=False).reshape(-1)
        if bool(protected_v2):
            source = observations.source_indices
            source_valid = source >= 0
            safe_source = source.clamp_min(0)
            delta = bank_xyz[top1] - bank_xyz[safe_source]
            distance_3d = torch.linalg.norm(delta, dim=1)
            normal_agreement = (
                bank_normals[top1] * bank_normals[safe_source]
            ).sum(dim=1).abs()
            source_plane = (
                delta * bank_normals[safe_source]
            ).sum(dim=1).abs()
            target_plane = (
                delta * bank_normals[top1]
            ).sum(dim=1).abs()
            same_surface = (
                source_valid
                & (distance_3d <= float(surface_max_distance_m))
                & (normal_agreement >= float(surface_normal_cosine))
                & (source_plane <= float(surface_point_plane_m))
                & (target_plane <= float(surface_point_plane_m))
            )
            measurement_limited = (
                observations.bank_visible[top1]
                & same_surface
                & (
                    top1_distance
                    > float(measurement_min_reprojection_px)
                )
                & (
                    top1_distance
                    <= float(measurement_max_reprojection_px)
                )
            )
            clean_rows = torch.nonzero(
                measurement_limited, as_tuple=False
            ).reshape(-1)
        else:
            clean_rows = protected_rows
        if clean_rows.numel() == 0:
            margin_diagnostics = {
                "native_semidense_margin_preserve_count": 0,
                "native_semidense_margin_preserve_violation_rate": 0.0,
                "native_semidense_margin_preserve_loss": 0.0,
            }
            margin_only_loss = zero
            if (
                bool(protected_v2)
                and protected_rows.numel() > 0
                and float(margin_preservation_weight) > 0.0
            ):
                margin_rows = protected_rows[: max(int(max_anchors), 1)]
                # The routing decision is detached, but the preservation loss
                # must remain differentiable even inside this no-grad block.
                with torch.enable_grad():
                    margin_only_loss, margin_diagnostics = (
                        global_margin_preservation_loss(
                            bank_features,
                            reference_bank_features,
                            observations,
                            margin_rows,
                            top1[margin_rows],
                        )
                    )
                    margin_only_loss = (
                        float(margin_preservation_weight) * margin_only_loss
                    )
            return margin_only_loss, {
                "native_semidense_active": 1.0,
                "native_semidense_clean_anchor_count": 0,
                "native_semidense_teacher_pair_count": 0,
                "native_semidense_measurement_limited_count": 0,
                "native_semidense_protected_high_precision_count": int(
                    protected_rows.numel()
                ),
                "native_semidense_protected_v2": float(bool(protected_v2)),
                **margin_diagnostics,
            }
        if float(pose_safe_max_delete_gain_m) >= 0.0:
            safe, pose_safe_diagnostics = counterfactual_pose_safe_mask(
                bank_xyz,
                top1[clean_rows],
                observations.query_uv[clean_rows],
                observations.K,
                observations.pose_w2c,
                maximum_delete_gain_m=pose_safe_max_delete_gain_m,
                minimum_correspondences=pose_safe_min_correspondences,
            )
            pose_safe_diagnostics["native_semidense_pose_safe_enabled"] = 1.0
            clean_rows = clean_rows[safe]
            if clean_rows.numel() == 0:
                return zero, {
                    "native_semidense_active": 1.0,
                    "native_semidense_clean_anchor_count": 0,
                    "native_semidense_teacher_pair_count": 0,
                    **pose_safe_diagnostics,
                }
        order = clean_rows[torch.argsort(margin[clean_rows], descending=True)]
        anchor_rows = []
        seen_landmarks = set()
        for row in order.tolist():
            landmark = int(top1[row].item())
            if landmark in seen_landmarks:
                continue
            seen_landmarks.add(landmark)
            anchor_rows.append(row)
            if 0 < int(max_anchors) <= len(anchor_rows):
                break
        anchor_rows = torch.as_tensor(
            anchor_rows, device=bank_features.device, dtype=torch.long
        )
        anchor_indices = (
            observations.source_indices[anchor_rows]
            if bool(protected_v2)
            else top1[anchor_rows]
        )

        neighbors_per_anchor = max(int(neighbors_per_anchor), 1)
        if neighbors_per_anchor == 1:
            teacher_indices = anchor_indices
            teacher_uv = observations.query_uv[anchor_rows]
            teacher_anchor_ids = torch.arange(
                anchor_indices.numel(),
                device=bank_features.device,
                dtype=torch.long,
            )
        else:
            distance = torch.cdist(
                bank_xyz[anchor_indices].float(), bank_xyz.float()
            )
            normal_support = (
                bank_normals[anchor_indices] @ bank_normals.T
            ).abs()
            if bool(protected_v2):
                delta = (
                    bank_xyz[None]
                    - bank_xyz[anchor_indices, None]
                )
                anchor_plane = (
                    delta * bank_normals[anchor_indices, None]
                ).sum(dim=2).abs()
                target_plane = (
                    delta * bank_normals[None]
                ).sum(dim=2).abs()
                projected_distance = torch.cdist(
                    observations.bank_uv[anchor_indices].float(),
                    observations.bank_uv.float(),
                )
                eligible = (
                    observations.bank_visible[None]
                    & (normal_support >= float(surface_normal_cosine))
                    & (distance <= float(surface_max_distance_m))
                    & (anchor_plane <= float(surface_point_plane_m))
                    & (target_plane <= float(surface_point_plane_m))
                    & (
                        projected_distance
                        <= float(projected_neighbor_radius_px)
                    )
                )
                ranking_distance = projected_distance
            else:
                eligible = (
                    observations.bank_visible[None]
                    & (normal_support >= float(normal_cosine))
                    & (distance <= float(neighborhood_radius_m))
                )
                ranking_distance = distance
            masked_distance = ranking_distance.masked_fill(
                ~eligible, torch.inf
            )
            neighbor_count = min(neighbors_per_anchor, bank_count)
            neighbor_distance, neighbor_indices = torch.topk(
                masked_distance,
                k=neighbor_count,
                dim=1,
                largest=False,
            )
            neighbor_valid = torch.isfinite(neighbor_distance)
            teacher_indices = neighbor_indices[neighbor_valid]
            teacher_anchor_ids = (
                torch.arange(
                    anchor_indices.numel(),
                    device=bank_features.device,
                    dtype=torch.long,
                )[:, None]
                .expand_as(neighbor_indices)[neighbor_valid]
            )
            teacher_uv = observations.bank_uv[teacher_indices].detach().clone()
            anchor_member = teacher_indices == anchor_indices[teacher_anchor_ids]
            if bool(anchor_member.any().item()):
                teacher_uv[anchor_member] = observations.query_uv[
                    anchor_rows[teacher_anchor_ids[anchor_member]]
                ]

        domain_height, domain_width = _query_feature_domain(observations)
        teacher_valid = (
            torch.isfinite(teacher_uv).all(dim=1)
            & (teacher_uv[:, 0] >= 0.0)
            & (teacher_uv[:, 0] <= float(domain_width - 1))
            & (teacher_uv[:, 1] >= 0.0)
            & (teacher_uv[:, 1] <= float(domain_height - 1))
        )
        if observations.query_valid_mask is not None and bool(teacher_valid.any()):
            teacher_valid &= _mask_at_uv(
                observations.query_valid_mask,
                teacher_uv,
            )
        teacher_indices = teacher_indices[teacher_valid]
        teacher_uv = teacher_uv[teacher_valid]
        teacher_anchor_ids = teacher_anchor_ids[teacher_valid]
        protected_neighbor_excluded_count = 0
        if bool(protected_v2) and protected_rows.numel() > 0:
            protected_landmarks = torch.unique(top1[protected_rows])
            protected_teacher = torch.isin(
                teacher_indices, protected_landmarks
            )
            protected_neighbor_excluded_count = int(
                protected_teacher.sum().item()
            )
            teacher_indices = teacher_indices[~protected_teacher]
            teacher_uv = teacher_uv[~protected_teacher]
            teacher_anchor_ids = teacher_anchor_ids[~protected_teacher]

    if teacher_indices.numel() == 0:
        return zero, {
            "native_semidense_active": 1.0,
            "native_semidense_clean_anchor_count": int(anchor_indices.numel()),
            "native_semidense_teacher_pair_count": 0,
            "native_semidense_protected_neighbor_excluded_count": int(
                protected_neighbor_excluded_count
            ),
        }
    teacher_observations = DetectorFreeObservationBatch(
        source_indices=teacher_indices,
        query_features=_sample_observation_features(
            observations, teacher_uv
        ).detach(),
        query_uv=teacher_uv,
        source_depth=observations.bank_depth[teacher_indices].detach(),
        bank_uv=observations.bank_uv,
        bank_depth=observations.bank_depth,
        bank_projected=observations.bank_projected,
        bank_visible=observations.bank_visible,
        base_bank_uv=observations.base_bank_uv,
        base_bank_depth=observations.base_bank_depth,
        base_bank_projected=observations.base_bank_projected,
        base_bank_visible=observations.base_bank_visible,
        query_feature_map=observations.query_feature_map,
        target_depth_map=observations.target_depth_map,
        target_alpha_map=observations.target_alpha_map,
        query_valid_mask=observations.query_valid_mask,
        K=observations.K,
        pose_w2c=observations.pose_w2c,
        query_feature_image_size=observations.query_feature_image_size,
    )
    pair_pose_safe_diagnostics = {
        "native_semidense_pair_pose_safe_enabled": float(
            bool(pose_safe_teacher_pairs)
            and float(pose_safe_max_delete_gain_m) >= 0.0
        )
    }
    if (
        bool(pose_safe_teacher_pairs)
        and float(pose_safe_max_delete_gain_m) >= 0.0
    ):
        current_local = local_soft_correspondences(
            bank_features.detach(),
            teacher_observations,
            radius=local_radius_px,
            temperature=temperature,
        )
        pair_safe, raw_pair_diagnostics = counterfactual_pose_safe_mask(
            bank_xyz,
            teacher_indices,
            current_local.expected_uv.detach(),
            observations.K,
            observations.pose_w2c,
            maximum_delete_gain_m=pose_safe_max_delete_gain_m,
            minimum_correspondences=pose_safe_min_correspondences,
        )
        pair_safe &= current_local.valid
        pair_pose_safe_diagnostics.update(
            {
                key.replace(
                    "native_semidense_pose_",
                    "native_semidense_pair_pose_",
                ): value
                for key, value in raw_pair_diagnostics.items()
            }
        )
        pair_pose_safe_diagnostics[
            "native_semidense_pair_pose_safe_enabled"
        ] = 1.0
        teacher_observations = _select_observation_rows(
            teacher_observations, pair_safe
        )
        teacher_indices = teacher_indices[pair_safe]
        teacher_uv = teacher_uv[pair_safe]
        teacher_anchor_ids = teacher_anchor_ids[pair_safe]
        if teacher_indices.numel() == 0:
            return zero, {
                "native_semidense_active": 1.0,
                "native_semidense_clean_anchor_count": int(
                    anchor_indices.numel()
                ),
                "native_semidense_teacher_pair_count": 0,
                **pose_safe_diagnostics,
                **pair_pose_safe_diagnostics,
            }
    loss, diagnostics = local_correlation_peak_loss(
        bank_features,
        teacher_observations,
        radius=local_radius_px,
        target_sigma=target_sigma_px,
        temperature=temperature,
    )
    teacher_features = teacher_observations.query_features
    if float(local_identity_weight) > 0.0:
        identity_loss, identity_diagnostics = (
            local_group_identity_assignment_loss(
                bank_features,
                teacher_features,
                teacher_indices,
                teacher_uv,
                teacher_anchor_ids,
                temperature=temperature,
                positive_radius_px=positive_radius_px,
            )
        )
        loss = loss + float(local_identity_weight) * identity_loss
    else:
        identity_diagnostics = {
            "native_semidense_local_identity_group_count": 0,
            "native_semidense_local_identity_row_count": 0,
            "native_semidense_local_identity_loss": 0.0,
        }
    if float(margin_preservation_weight) > 0.0:
        margin_loss, margin_diagnostics = global_margin_preservation_loss(
            bank_features,
            reference_bank_features,
            observations,
            protected_rows[: max(int(max_anchors), 1)],
            top1[protected_rows[: max(int(max_anchors), 1)]],
        )
        loss = loss + float(margin_preservation_weight) * margin_loss
    else:
        margin_diagnostics = {
            "native_semidense_margin_preserve_count": 0,
            "native_semidense_margin_preserve_violation_rate": 0.0,
            "native_semidense_margin_preserve_loss": 0.0,
        }
    if float(lgcv_weight) > 0.0:
        local = local_soft_correspondences(
            bank_features,
            teacher_observations,
            radius=local_radius_px,
            temperature=temperature,
        )
        lgcv_loss, lgcv_diagnostics = group_geometric_consistency_loss(
            local.expected_uv,
            teacher_uv,
            teacher_anchor_ids,
            valid=local.valid,
            minimum_edge_px=lgcv_minimum_edge_px,
        )
        loss = loss + float(lgcv_weight) * lgcv_loss
    else:
        lgcv_diagnostics = {
            "native_semidense_lgcv_group_count": 0,
            "native_semidense_lgcv_edge_count": 0,
            "native_semidense_lgcv_triangle_count": 0,
            "native_semidense_lgcv_loss": 0.0,
        }
    diagnostics.update(
        {
            "native_semidense_active": 1.0,
            "native_semidense_clean_anchor_count": int(anchor_indices.numel()),
            "native_semidense_teacher_pair_count": int(teacher_indices.numel()),
            "native_semidense_unique_landmarks": int(
                torch.unique(teacher_indices).numel()
            ),
            "native_semidense_neighbors_per_anchor_mean": float(
                teacher_indices.numel() / max(anchor_indices.numel(), 1)
            ),
            "native_semidense_anchor_group_count": int(
                torch.unique(teacher_anchor_ids).numel()
            ),
            "native_semidense_measurement_limited_count": int(
                anchor_indices.numel() if bool(protected_v2) else 0
            ),
            "native_semidense_protected_high_precision_count": int(
                protected_rows.numel()
            ),
            "native_semidense_protected_v2": float(bool(protected_v2)),
            "native_semidense_protected_neighbor_excluded_count": int(
                protected_neighbor_excluded_count
            ),
            "native_semidense_neighborhood_overlap_mean": float(
                torch.bincount(
                    teacher_indices, minlength=bank_count
                ).float()[torch.unique(teacher_indices)].mean().item()
            ),
            "native_semidense_neighborhood_overlap_max": int(
                torch.bincount(
                    teacher_indices, minlength=bank_count
                ).max().item()
            ),
            **pose_safe_diagnostics,
            **pair_pose_safe_diagnostics,
            **identity_diagnostics,
            **margin_diagnostics,
            **lgcv_diagnostics,
        }
    )
    return loss, diagnostics


def local_soft_correspondences(
    bank_features,
    observations,
    *,
    radius=3,
    temperature=0.07,
):
    """Return differentiable local soft-argmax measurements around GT projections."""
    feature_map = observations.query_feature_map
    if feature_map is None:
        raise ValueError("Local correspondences require the frozen query feature map")
    query_count = int(observations.source_indices.numel())
    radius = max(int(radius), 0)
    if query_count == 0:
        empty_uv = bank_features.new_zeros((0, 2))
        empty = bank_features.new_zeros((0,))
        return LocalSoftCorrespondenceOutput(
            expected_uv=empty_uv,
            confidence=empty,
            entropy=empty,
            valid=torch.zeros(0, device=bank_features.device, dtype=torch.bool),
            diagnostics={"soft_correspondence_count": 0},
        )
    axis = torch.arange(
        -radius,
        radius + 1,
        device=bank_features.device,
        dtype=observations.query_uv.dtype,
    )
    offset_y, offset_x = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack([offset_x.reshape(-1), offset_y.reshape(-1)], dim=-1)
    patch_uv = observations.query_uv[:, None, :] + offsets[None]
    height, width = _query_feature_domain(observations)
    patch_valid = (
        (patch_uv[..., 0] >= 0.0)
        & (patch_uv[..., 0] <= float(width - 1))
        & (patch_uv[..., 1] >= 0.0)
        & (patch_uv[..., 1] <= float(height - 1))
    )
    patch_features = _sample_observation_features(
        observations, patch_uv.reshape(-1, 2)
    ).reshape(query_count, offsets.shape[0], -1)
    feature_valid = torch.isfinite(patch_features).all(dim=-1) & (
        torch.linalg.norm(patch_features, dim=-1) > 1e-6
    )
    # Keep the mask immutable after masked_fill records it for backward.
    patch_valid = patch_valid & feature_valid
    patch_features = F.normalize(
        torch.nan_to_num(patch_features, nan=0.0, posinf=0.0, neginf=0.0), dim=-1
    )
    source_valid = observations.source_indices >= 0
    safe_source_indices = observations.source_indices.clamp_min(0)
    source = F.normalize(bank_features[safe_source_indices], dim=-1)
    logits = torch.einsum("qd,qpd->qp", source, patch_features)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~patch_valid, -torch.inf)
    valid = patch_valid.any(dim=1) & source_valid
    safe_logits = torch.where(valid[:, None], logits, torch.zeros_like(logits))
    probability = torch.softmax(safe_logits, dim=-1) * patch_valid
    probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
    expected_offset = probability @ offsets
    expected_uv = observations.query_uv + expected_offset
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=1)
    entropy = entropy / max(float(torch.log(torch.tensor(offsets.shape[0])).item()), 1.0)
    confidence = probability.max(dim=1).values * (1.0 - entropy).clamp_min(0.0)
    confidence = confidence * valid.to(confidence.dtype)
    diagnostics = {
        "soft_correspondence_count": int(valid.sum().detach().item()),
        "soft_correspondence_confidence_mean": float(
            confidence[valid].mean().detach().item() if bool(valid.any()) else 0.0
        ),
        "soft_correspondence_entropy_mean": float(
            entropy[valid].mean().detach().item() if bool(valid.any()) else 0.0
        ),
        "soft_correspondence_offset_mean_px": float(
            torch.linalg.norm(expected_offset[valid], dim=1).mean().detach().item()
            if bool(valid.any())
            else 0.0
        ),
    }
    return LocalSoftCorrespondenceOutput(
        expected_uv=expected_uv,
        confidence=confidence,
        entropy=entropy,
        valid=valid,
        diagnostics=diagnostics,
    )


def bounded_geometry_losses(
    current_xyz,
    raw_offset,
    bank_features,
    observations,
    *,
    local_radius=3,
    local_temperature=0.07,
    depth_scale_floor=0.25,
):
    """Surface, rendered-depth, and local feature reprojection supervision."""
    # The image measurement is produced by a frozen descriptor teacher around
    # the immutable base projection. Only the predicted anchor projection is
    # allowed to move.
    local = local_soft_correspondences(
        bank_features.detach(),
        observations,
        radius=local_radius,
        temperature=local_temperature,
    )
    source_valid = observations.source_indices >= 0
    valid = local.valid & (local.confidence > 0) & source_valid
    if bool(valid.any()):
        source_indices = observations.source_indices[valid]
        predicted_uv = observations.bank_uv[source_indices]
        reprojection_error = torch.linalg.norm(
            local.expected_uv[valid].detach() - predicted_uv, dim=1
        )
        reprojection_loss = (
            F.smooth_l1_loss(
                reprojection_error,
                torch.zeros_like(reprojection_error),
                reduction="none",
                beta=1.0,
            )
            * local.confidence[valid].detach()
        ).sum() / local.confidence[valid].detach().sum().clamp_min(1e-8)
    else:
        reprojection_loss = current_xyz.sum() * 0.0

    depth_loss = current_xyz.sum() * 0.0
    depth_valid_count = 0
    if observations.target_depth_map is not None and bool(valid.any()):
        depth_map = torch.as_tensor(
            observations.target_depth_map,
            device=current_xyz.device,
            dtype=current_xyz.dtype,
        ).squeeze()
        rendered_depth = bilinear_sample_features(
            depth_map[None], local.expected_uv[valid].detach()
        )[:, 0]
        predicted_depth = observations.bank_depth[
            observations.source_indices[valid]
        ]
        depth_valid = (
            torch.isfinite(rendered_depth)
            & (rendered_depth > 0)
            & torch.isfinite(predicted_depth)
            & (predicted_depth > 0)
        )
        depth_valid_count = int(depth_valid.sum().detach().item())
        if bool(depth_valid.any()):
            scale = rendered_depth[depth_valid].detach().abs().clamp_min(
                float(depth_scale_floor)
            )
            normalized_error = (
                predicted_depth[depth_valid] - rendered_depth[depth_valid].detach()
            ) / scale
            depth_loss = F.smooth_l1_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
                beta=0.01,
            )
    surface_loss = torch.tanh(raw_offset).square().mean()
    diagnostics = {
        **local.diagnostics,
        "geometry_reprojection_loss": float(reprojection_loss.detach().item()),
        "geometry_depth_loss": float(depth_loss.detach().item()),
        "geometry_depth_count": depth_valid_count,
    }
    return surface_loss, depth_loss, reprojection_loss, local, diagnostics


def _native_observation_image_size(observations):
    """Recover the native SuperPoint grid size carried by an observation batch."""
    for value in (
        observations.query_valid_mask,
        observations.target_depth_map,
        observations.target_alpha_map,
    ):
        if value is None:
            continue
        value = torch.as_tensor(value)
        if value.ndim >= 2:
            return int(value.shape[-2]), int(value.shape[-1])
    raise ValueError(
        "Native association BA requires a query valid/depth/alpha map to "
        "recover the native image resolution"
    )


def native_association_matches(
    current_xyz,
    bank_features,
    observations,
    *,
    max_reprojection_error_px=2.0,
    min_score_margin=0.0,
    depth_abs_tolerance=0.0,
    depth_rel_tolerance=0.0,
    alpha_threshold=0.2,
):
    """Retrieve native top-1 matches and gate them with the GT geometry.

    The descriptor bank is deliberately detached.  The returned projection is
    recomputed from ``current_xyz`` rather than borrowed from a cached
    observation batch, so the BA residual has an explicit, auditable gradient
    path to the bounded surface anchor.  An optional rendered-depth gate
    rejects a reprojection-clean association when its primitive is not on the
    front surface at the native query proposal.  It is opt-in to preserve the
    historical position-only BA protocol.
    """
    query_count = int(observations.query_features.shape[0])
    bank_count = int(bank_features.shape[0])
    if query_count == 0 or bank_count == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=current_xyz.device)
        empty_float = current_xyz.new_zeros((0,))
        empty_uv = current_xyz.new_zeros((0, 2))
        empty_bool = torch.empty(0, dtype=torch.bool, device=current_xyz.device)
        return NativeAssociationResult(
            top1_indices=empty_long,
            top1_scores=empty_float,
            top1_margins=empty_float,
            projected_uv=empty_uv,
            projected_depth=empty_float,
            projected=empty_bool,
            visible=empty_bool,
            reprojection_error=empty_float,
            depth_sample_valid=empty_bool,
            depth_compatible=empty_bool,
            depth_abs_error=empty_float,
            depth_gate_enabled=False,
            clean=empty_bool,
        )
    if observations.K is None or observations.pose_w2c is None:
        raise ValueError("Native association BA requires camera intrinsics and pose")

    query = F.normalize(observations.query_features.detach(), dim=-1)
    bank = F.normalize(bank_features.detach(), dim=-1)
    scores = query @ bank.T
    top_count = min(2, bank_count)
    top_scores, top_indices = torch.topk(scores, k=top_count, dim=1)
    top1 = top_indices[:, 0]
    top1_score = top_scores[:, 0]
    if top_count > 1:
        score_margin = top_scores[:, 0] - top_scores[:, 1]
    else:
        score_margin = torch.full_like(top1_score, torch.inf)

    height, width = _native_observation_image_size(observations)
    projected_uv, projected_depth, projected = project_landmarks_to_query(
        current_xyz,
        observations.K,
        observations.pose_w2c,
        height,
        width,
        # Native sparse coordinates are grid indices. The frontend adds 0.5
        # only for PnP, so projections use the matching grid-index convention.
        pixel_center_offset=0.5,
    )
    base_visible = observations.base_bank_visible
    if base_visible is None:
        base_visible = observations.bank_visible
    base_visible = torch.as_tensor(
        base_visible, device=current_xyz.device, dtype=torch.bool
    ).reshape(-1)
    if base_visible.numel() != bank_count:
        raise ValueError("native association visibility must match bank size")
    visible = base_visible & projected
    if observations.query_valid_mask is not None and bool(visible.any().item()):
        visible = visible & _mask_at_uv(
            observations.query_valid_mask,
            projected_uv,
        )

    predicted_uv = projected_uv[top1]
    reprojection_error = torch.linalg.norm(
        predicted_uv - observations.query_uv.detach(), dim=1
    )
    depth_gate_enabled = (
        float(depth_abs_tolerance) > 0.0 or float(depth_rel_tolerance) > 0.0
    )
    depth_sample_valid = torch.ones(
        query_count, dtype=torch.bool, device=current_xyz.device
    )
    depth_compatible = depth_sample_valid
    depth_abs_error = torch.zeros_like(top1_score)
    if depth_gate_enabled:
        if observations.target_depth_map is None:
            raise ValueError(
                "Native association depth gating requires target_depth_map"
            )
        depth_map = torch.as_tensor(
            observations.target_depth_map,
            device=current_xyz.device,
            dtype=current_xyz.dtype,
        ).squeeze()
        rendered_depth = bilinear_sample_features(
            depth_map[None], observations.query_uv.detach()
        )[:, 0]
        predicted_depth = projected_depth[top1]
        depth_sample_valid = (
            torch.isfinite(rendered_depth)
            & (rendered_depth > 0)
            & torch.isfinite(predicted_depth)
            & (predicted_depth > 0)
        )
        if observations.target_alpha_map is not None:
            alpha_map = torch.as_tensor(
                observations.target_alpha_map,
                device=current_xyz.device,
                dtype=current_xyz.dtype,
            ).squeeze()
            alpha = bilinear_sample_features(
                alpha_map[None], observations.query_uv.detach()
            )[:, 0]
            depth_sample_valid &= torch.isfinite(alpha) & (
                alpha >= float(alpha_threshold)
            )
        depth_abs_error = (predicted_depth - rendered_depth.detach()).abs()
        depth_tolerance = torch.maximum(
            depth_abs_error.new_full(
                depth_abs_error.shape, float(depth_abs_tolerance)
            ),
            rendered_depth.detach().abs() * float(depth_rel_tolerance),
        )
        depth_compatible = depth_sample_valid & (
            depth_abs_error.detach() <= depth_tolerance
        )
    source_valid = observations.source_indices >= 0
    clean = (
        source_valid
        & visible[top1]
        & torch.isfinite(reprojection_error)
        & (reprojection_error.detach() <= float(max_reprojection_error_px))
        & (score_margin.detach() >= float(min_score_margin))
        & depth_compatible
    )
    if observations.query_valid_mask is not None and bool(clean.any().item()):
        clean = clean & _mask_at_uv(
            observations.query_valid_mask,
            observations.query_uv,
        )
    return NativeAssociationResult(
        top1_indices=top1,
        top1_scores=top1_score,
        top1_margins=score_margin,
        projected_uv=projected_uv,
        projected_depth=projected_depth,
        projected=projected,
        visible=visible,
        reprojection_error=reprojection_error,
        depth_sample_valid=depth_sample_valid,
        depth_compatible=depth_compatible,
        depth_abs_error=depth_abs_error,
        depth_gate_enabled=depth_gate_enabled,
        clean=clean,
    )


def native_association_geometry_losses(
    current_xyz,
    raw_offset,
    bank_features,
    observations,
    *,
    max_reprojection_error_px=2.0,
    min_score_margin=0.0,
    alpha_threshold=0.2,
    depth_scale_floor=0.25,
    depth_abs_tolerance=0.0,
    depth_rel_tolerance=0.0,
    landmark_support_mask=None,
):
    """Fixed-descriptor, native-match-driven bounded surface BA loss.

    The association is produced by the same native SuperPoint descriptor and
    full-bank cosine retrieval used at deployment.  Only current top-1 matches
    that are already GT-reprojection clean are accepted.  Thus the GT pose is
    a geometric gate and target measurement, not a shortcut that inserts a
    landmark into the candidate set.  ``bank_features`` is detached so this
    loss can update only the bounded surface anchor parameterization.
    """
    query_count = int(observations.query_features.shape[0])
    bank_count = int(bank_features.shape[0])
    zero = current_xyz.sum() * 0.0
    if query_count == 0 or bank_count == 0:
        return zero, zero, zero, {
            "native_geometry_active": 0.0,
            "native_geometry_candidate_count": 0,
            "native_geometry_clean_correspondences": 0,
        }
    association = native_association_matches(
        current_xyz,
        bank_features,
        observations,
        max_reprojection_error_px=max_reprojection_error_px,
        min_score_margin=min_score_margin,
        depth_abs_tolerance=depth_abs_tolerance,
        depth_rel_tolerance=depth_rel_tolerance,
        alpha_threshold=alpha_threshold,
    )
    top1 = association.top1_indices
    top1_score = association.top1_scores
    score_margin = association.top1_margins
    predicted_uv = association.projected_uv[top1]
    predicted_depth_all = association.projected_depth
    reprojection_error = association.reprojection_error
    clean_before_support = association.clean
    clean = clean_before_support
    support_eligible_count = bank_count
    if landmark_support_mask is not None:
        landmark_support_mask = torch.as_tensor(
            landmark_support_mask,
            device=current_xyz.device,
            dtype=torch.bool,
        ).reshape(-1)
        if landmark_support_mask.numel() != bank_count:
            raise ValueError("landmark_support_mask must match bank size")
        support_eligible_count = int(landmark_support_mask.sum().item())
        clean = clean & landmark_support_mask[top1]

    reprojection_loss = zero
    if bool(clean.any().item()):
        reprojection_loss = F.smooth_l1_loss(
            predicted_uv[clean],
            observations.query_uv[clean].detach(),
            reduction="none",
            beta=1.0,
        ).sum(dim=1).mean()

    depth_loss = zero
    depth_count = 0
    if observations.target_depth_map is not None and bool(clean.any().item()):
        depth_map = torch.as_tensor(
            observations.target_depth_map,
            device=current_xyz.device,
            dtype=current_xyz.dtype,
        ).squeeze()
        rendered_depth = bilinear_sample_features(
            depth_map[None], observations.query_uv[clean].detach()
        )[:, 0]
        predicted_depth = predicted_depth_all[top1[clean]]
        depth_valid = (
            torch.isfinite(rendered_depth)
            & (rendered_depth > 0)
            & torch.isfinite(predicted_depth)
            & (predicted_depth > 0)
        )
        if observations.target_alpha_map is not None and bool(depth_valid.any().item()):
            alpha_map = torch.as_tensor(
                observations.target_alpha_map,
                device=current_xyz.device,
                dtype=current_xyz.dtype,
            ).squeeze()
            alpha = bilinear_sample_features(
                alpha_map[None], observations.query_uv[clean].detach()
            )[:, 0]
            depth_valid &= torch.isfinite(alpha) & (alpha >= float(alpha_threshold))
        depth_count = int(depth_valid.sum().detach().item())
        if bool(depth_valid.any().item()):
            scale = rendered_depth[depth_valid].detach().abs().clamp_min(
                float(depth_scale_floor)
            )
            normalized_error = (
                predicted_depth[depth_valid] - rendered_depth[depth_valid].detach()
            ) / scale
            depth_loss = F.smooth_l1_loss(
                normalized_error,
                torch.zeros_like(normalized_error),
                beta=0.01,
            )
    surface_loss = torch.tanh(raw_offset).square().mean()
    source_identity = (
        (top1[clean] == observations.source_indices[clean]).float().mean()
        if bool(clean.any().item())
        else current_xyz.new_zeros(())
    )
    diagnostics = {
        "native_geometry_active": 1.0,
        "native_geometry_candidate_count": query_count,
        "native_geometry_clean_before_support": int(
            clean_before_support.sum().detach().item()
        ),
        "native_geometry_clean_correspondences": int(clean.sum().detach().item()),
        "native_geometry_clean_ratio": float(clean.float().mean().detach().item()),
        "native_geometry_support_eligible_landmarks": support_eligible_count,
        "native_geometry_reprojection_median_px": float(
            reprojection_error[clean].detach().median().item()
            if bool(clean.any().item())
            else 0.0
        ),
        "native_geometry_reprojection_loss": float(reprojection_loss.detach().item()),
        "native_geometry_depth_loss": float(depth_loss.detach().item()),
        "native_geometry_depth_count": depth_count,
        "native_geometry_depth_gate_enabled": float(association.depth_gate_enabled),
        "native_geometry_depth_gate_valid_count": int(
            association.depth_sample_valid.sum().detach().item()
        ),
        "native_geometry_depth_gate_pass_count": int(
            association.depth_compatible.sum().detach().item()
        ),
        "native_geometry_depth_gate_rejected_count": int(
            (
                association.depth_sample_valid & ~association.depth_compatible
            ).sum().detach().item()
        ),
        "native_geometry_depth_gate_abs_error_median_m": float(
            association.depth_abs_error[association.depth_sample_valid]
            .detach()
            .median()
            .item()
            if bool(association.depth_sample_valid.any().item())
            else 0.0
        ),
        "native_geometry_depth_gate_abs_error_p95_m": float(
            torch.quantile(
                association.depth_abs_error[association.depth_sample_valid].detach(),
                0.95,
            ).item()
            if bool(association.depth_sample_valid.any().item())
            else 0.0
        ),
        "native_geometry_depth_gate_abs_error_max_m": float(
            association.depth_abs_error[association.depth_sample_valid]
            .detach()
            .max()
            .item()
            if bool(association.depth_sample_valid.any().item())
            else 0.0
        ),
        "native_geometry_top1_score_mean": float(top1_score.detach().mean().item()),
        "native_geometry_top1_margin_mean": float(score_margin.detach().mean().item()),
        "native_geometry_source_identity_ratio": float(source_identity.detach().item()),
        "native_geometry_descriptor_frozen": 1.0,
    }
    return surface_loss, depth_loss, reprojection_loss, diagnostics


def pose_layer_loss(
    current_xyz,
    observations,
    local_correspondences,
    pose_init_w2c,
    *,
    num_iterations=3,
    damping=1e-3,
    min_points=24,
    max_points=128,
    translation_scale_m=0.02,
    rotation_scale_degrees=2.0,
    max_condition_number=5e4,
    gradient_mode="geometry",
):
    if observations.K is None or observations.pose_w2c is None:
        raise ValueError("PoseLayer requires K and the GT world-to-camera pose")
    valid = (
        local_correspondences.valid
        & (local_correspondences.confidence > 0)
        & (observations.source_indices >= 0)
    )
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if valid_indices.numel() < int(min_points):
        zero = current_xyz.sum() * 0.0
        return zero, {"pose_layer_active": 0.0, "pose_layer_points": int(valid_indices.numel())}
    if valid_indices.numel() > int(max_points):
        confidence = local_correspondences.confidence[valid_indices]
        valid_indices = valid_indices[torch.topk(confidence, int(max_points)).indices]
    source_indices = observations.source_indices[valid_indices]
    points = current_xyz[source_indices]
    gradient_mode = str(gradient_mode)
    if gradient_mode == "feature":
        points = points.detach()
        target_uv = local_correspondences.expected_uv[valid_indices]
        weights = local_correspondences.confidence[valid_indices].clamp_min(1e-4)
        zero_reference = target_uv.sum() + weights.sum()
        detach_points = True
    elif gradient_mode == "geometry":
        target_uv = local_correspondences.expected_uv[valid_indices].detach()
        weights = local_correspondences.confidence[valid_indices].detach().clamp_min(1e-4)
        zero_reference = current_xyz.sum()
        detach_points = False
    else:
        raise ValueError("gradient_mode must be 'feature' or 'geometry'")
    parameter_scale = points.new_tensor(
        [float(translation_scale_m)] * 3
        + [float(torch.deg2rad(torch.tensor(rotation_scale_degrees)).item())] * 3
    )
    refined_pose, info = weighted_gauss_newton_refine(
        points,
        target_uv,
        observations.K,
        pose_init_w2c,
        weights=weights,
        num_iterations=int(num_iterations),
        damping=float(damping),
        detach_points=detach_points,
        parameter_scale=parameter_scale,
    )
    gt_pose = observations.pose_w2c.to(device=points.device, dtype=points.dtype)
    center_error = torch.linalg.norm(
        camera_center_from_w2c(refined_pose) - camera_center_from_w2c(gt_pose)
    )
    relative_rotation = refined_pose[:3, :3] @ gt_pose[:3, :3].T
    cosine = ((torch.trace(relative_rotation) - 1.0) * 0.5).clamp(
        -1.0 + 1e-6, 1.0 - 1e-6
    )
    rotation_error = torch.acos(cosine)
    rotation_chord_error = torch.linalg.norm(
        refined_pose[:3, :3] - gt_pose[:3, :3]
    ) / (2.0**0.5)
    rotation_scale = max(
        float(torch.deg2rad(torch.tensor(rotation_scale_degrees)).item()), 1e-8
    )
    accepted = (
        bool(torch.isfinite(info["condition_number"]).item())
        and float(info["condition_number"].item()) <= float(max_condition_number)
        and bool(torch.isfinite(info["final_rmse"]).item())
        and float(info["final_rmse"].item()) < float(info["initial_rmse"].item())
    )
    loss = F.smooth_l1_loss(
        center_error / max(float(translation_scale_m), 1e-8),
        center_error.new_zeros(()),
        beta=1.0,
    ) + F.smooth_l1_loss(
        rotation_chord_error / rotation_scale,
        rotation_chord_error.new_zeros(()),
        beta=1.0,
    )
    if not accepted:
        loss = zero_reference * 0.0
    diagnostics = {
        "pose_layer_active": 1.0,
        "pose_layer_accepted": float(accepted),
        "pose_layer_points": int(points.shape[0]),
        "pose_layer_translation_error_m": float(center_error.detach().item()),
        "pose_layer_rotation_error_deg": float(
            torch.rad2deg(rotation_error.detach()).item()
        ),
        "pose_layer_initial_rmse_px": float(info["initial_rmse"].item()),
        "pose_layer_final_rmse_px": float(info["final_rmse"].item()),
        "pose_layer_condition_number": float(info["condition_number"].item()),
        "pose_layer_feature_gradient_mode": float(gradient_mode == "feature"),
        "pose_layer_geometry_gradient_mode": float(gradient_mode == "geometry"),
    }
    return loss, diagnostics


def _candidate_geometry_masks(
    observations,
    candidate_indices,
    *,
    positive_radius_px,
    negative_radius_px,
):
    candidate_uv = observations.bank_uv[candidate_indices]
    distance = torch.linalg.norm(
        candidate_uv - observations.query_uv[:, None, :], dim=-1
    )
    candidate_visible = observations.bank_visible[candidate_indices]
    candidate_projected = observations.bank_projected[candidate_indices]
    positive = candidate_visible & (distance <= float(positive_radius_px))
    nearby = candidate_projected & (distance < float(negative_radius_px))
    negative = ~positive & ~nearby
    return positive, negative, distance


def _masked_multi_positive_nce(
    logits,
    positive_mask,
    negative_mask,
    *,
    dustbin_logit=None,
):
    included = positive_mask | negative_mask
    valid_rows = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not bool(valid_rows.any().item()):
        return logits.sum() * 0.0, valid_rows
    logits = logits[valid_rows]
    positive_mask = positive_mask[valid_rows]
    included = included[valid_rows]
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    denominator_logits = logits.masked_fill(~included, -torch.inf)
    if dustbin_logit is not None:
        dustbin_logit = torch.as_tensor(
            dustbin_logit,
            device=logits.device,
            dtype=logits.dtype,
        ).reshape(1, 1)
        denominator_logits = torch.cat(
            [denominator_logits, dustbin_logit.expand(logits.shape[0], 1)],
            dim=1,
        )
    loss = -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(denominator_logits, dim=1)
    )
    return loss.mean(), valid_rows


def _retrieval_diagnostics(scores, observations, candidate_indices=None):
    if scores.numel() == 0:
        return {
            "retrieval_query_count": 0,
            "retrieval_top1_gt_precision_2px": 0.0,
            "retrieval_top1_gt_precision_4px": 0.0,
            "retrieval_source_top1_ratio": 0.0,
        }
    if candidate_indices is None:
        top1 = scores.argmax(dim=1)
    else:
        top1 = candidate_indices.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)
    top1_uv = observations.bank_uv[top1]
    distance = torch.linalg.norm(top1_uv - observations.query_uv, dim=-1)
    top1_visible = observations.bank_visible[top1]
    top1_projected = observations.bank_projected[top1]
    valid_distance = top1_projected & torch.isfinite(distance)
    source_valid = observations.source_indices >= 0
    source_top1 = (
        (top1[source_valid] == observations.source_indices[source_valid])
        .float()
        .mean()
        if bool(source_valid.any().item())
        else scores.new_zeros(())
    )
    diagnostics = {
        "retrieval_query_count": int(scores.shape[0]),
        "retrieval_top1_gt_precision_2px": float(
            (top1_visible & (distance <= 2.0)).float().mean().detach().item()
        ),
        "retrieval_top1_gt_precision_4px": float(
            (top1_visible & (distance <= 4.0)).float().mean().detach().item()
        ),
        "retrieval_source_top1_ratio": float(
            source_top1.detach().item()
        ),
        "retrieval_top1_reprojection_median_px": float(
            distance[valid_distance].median().detach().item()
            if bool(valid_distance.any().item())
            else 0.0
        ),
        "retrieval_top1_projected_ratio": float(
            valid_distance.float().mean().detach().item()
        ),
    }
    if observations.positive_offsets is not None:
        offsets = torch.as_tensor(
            observations.positive_offsets,
            device=scores.device,
            dtype=torch.long,
        ).reshape(-1)
        positive_count = offsets[1:] - offsets[:-1]
        matched_count = positive_count[positive_count > 0].float()
        diagnostics.update(
            {
                "retrieval_positive_multiplicity_mean": float(
                    matched_count.mean().item() if matched_count.numel() else 0.0
                ),
                "retrieval_positive_multiplicity_p50": float(
                    torch.quantile(matched_count, 0.50).item()
                    if matched_count.numel()
                    else 0.0
                ),
                "retrieval_positive_multiplicity_p90": float(
                    torch.quantile(matched_count, 0.90).item()
                    if matched_count.numel()
                    else 0.0
                ),
                "retrieval_positive_multiplicity_p99": float(
                    torch.quantile(matched_count, 0.99).item()
                    if matched_count.numel()
                    else 0.0
                ),
                "retrieval_multi_positive_query_fraction": float(
                    (positive_count > 1).float().mean().item()
                    if positive_count.numel()
                    else 0.0
                ),
            }
        )
    if candidate_indices is None:
        max_k = min(64, int(scores.shape[1]))
        top_indices = torch.topk(scores, k=max_k, dim=1).indices
        for requested_k in (1, 4, 16, 64):
            effective_k = min(requested_k, max_k)
            retrieved = top_indices[:, :effective_k]
            if bool(source_valid.any().item()):
                source_recall = (
                    retrieved[source_valid]
                    == observations.source_indices[source_valid, None]
                ).any(dim=1).float().mean()
            else:
                source_recall = scores.new_zeros(())
            retrieved_uv = observations.bank_uv[retrieved]
            distance_k = torch.linalg.norm(
                retrieved_uv - observations.query_uv[:, None], dim=-1
            )
            visible_k = observations.bank_visible[retrieved]
            geometric_recall = (
                visible_k & (distance_k <= 2.0)
            ).any(dim=1).float().mean()
            diagnostics[
                f"retrieval_source_recall_at_{requested_k}"
            ] = float(source_recall.detach().item())
            diagnostics[
                f"retrieval_geometric_recall_at_{requested_k}"
            ] = float(geometric_recall.detach().item())
    return diagnostics


def random_negative_retrieval_loss(
    bank_features,
    observations,
    *,
    negative_count=64,
    temperature=0.07,
    positive_radius_px=2.0,
    negative_radius_px=6.0,
    dustbin_score=None,
    generator=None,
):
    unmatched_query_count = int((observations.source_indices < 0).sum().item())
    observations = _select_observation_rows(
        observations,
        observations.source_indices >= 0,
    )
    query_count = int(observations.source_indices.numel())
    bank_count = int(bank_features.shape[0])
    if query_count == 0 or bank_count < 2:
        return DetectorFreeRetrievalOutput(
            bank_features.sum() * 0.0,
            {
                "retrieval_mode_random": 1.0,
                "retrieval_query_count": 0,
                "retrieval_unmatched_query_count": unmatched_query_count,
            },
        )
    negative_count = max(1, min(int(negative_count), bank_count - 1))
    random_indices = torch.randint(
        bank_count,
        (query_count, negative_count),
        device=bank_features.device,
        generator=generator,
    )
    candidate_indices = torch.cat(
        [observations.source_indices[:, None], random_indices], dim=1
    )
    query = F.normalize(observations.query_features.detach(), dim=-1)
    bank = F.normalize(bank_features, dim=-1)
    candidate_features = bank[candidate_indices]
    logits = torch.einsum("qd,qcd->qc", query, candidate_features) / max(
        float(temperature), 1e-6
    )
    positive, negative, _ = _candidate_geometry_masks(
        observations,
        candidate_indices,
        positive_radius_px=positive_radius_px,
        negative_radius_px=negative_radius_px,
    )
    dustbin_logit = (
        None
        if dustbin_score is None
        else torch.as_tensor(dustbin_score, device=logits.device, dtype=logits.dtype)
        / max(float(temperature), 1e-6)
    )
    loss, valid_rows = _masked_multi_positive_nce(
        logits,
        positive,
        negative,
        dustbin_logit=dustbin_logit,
    )
    diagnostics = _retrieval_diagnostics(logits.detach(), observations, candidate_indices)
    diagnostics.update(
        {
            "retrieval_mode_random": 1.0,
            "retrieval_candidate_count_mean": float(candidate_indices.shape[1]),
            "retrieval_valid_loss_rows": int(valid_rows.sum().item()),
            "retrieval_positive_count_mean": float(
                positive.float().sum(dim=1).mean().detach().item()
            ),
            "retrieval_negative_count_mean": float(
                negative.float().sum(dim=1).mean().detach().item()
            ),
            "retrieval_unmatched_query_count": unmatched_query_count,
        }
    )
    return DetectorFreeRetrievalOutput(loss, diagnostics)


def hard_hypothesis_retrieval_loss(
    bank_features,
    observations,
    *,
    hypothesis_topk=32,
    temperature=0.07,
    positive_radius_px=2.0,
    negative_radius_px=6.0,
    margin=0.0,
    missed_positive_weight=1.0,
    missed_positive_margin=0.05,
    unmatched_rejection_weight=0.0,
    unmatched_max_similarity=0.5,
    dustbin_score=None,
    native_outcome_mode=False,
    native_nce_weight=0.0,
    native_keep_weight=1.0,
    native_keep_margin=0.05,
    native_keep_loose_weight=0.0,
    native_keep_loose_radius_px=4.0,
    native_keep_loose_margin=0.025,
    native_swap_weight=1.0,
    native_swap_margin=0.05,
    native_miss_weight=1.0,
    native_miss_margin=0.05,
    native_reject_weight=0.1,
    native_reject_threshold=0.5,
    native_attractor_weight=0.0,
    native_attractor_margin=0.05,
    native_global_attractor_weight=0.0,
    native_global_attractor_scores=None,
):
    query_count = int(observations.source_indices.numel())
    bank_count = int(bank_features.shape[0])
    if query_count == 0 or bank_count < 2:
        return DetectorFreeRetrievalOutput(bank_features.sum() * 0.0)
    query = F.normalize(observations.query_features.detach(), dim=-1)
    bank = F.normalize(bank_features, dim=-1)
    full_scores = query @ bank.T
    topk = max(1, min(int(hypothesis_topk), bank_count))
    top_indices = torch.topk(full_scores.detach(), k=topk, dim=1).indices
    # This is the deployment candidate set. Ground-truth/source landmarks are
    # never injected into it.
    candidate_indices = top_indices
    candidate_features = bank[candidate_indices]
    raw_logits = torch.einsum("qd,qcd->qc", query, candidate_features)
    logits = raw_logits / max(float(temperature), 1e-6)
    positive, negative, _ = _candidate_geometry_masks(
        observations,
        candidate_indices,
        positive_radius_px=positive_radius_px,
        negative_radius_px=negative_radius_px,
    )
    dustbin_logit = (
        None
        if dustbin_score is None
        else torch.as_tensor(dustbin_score, device=logits.device, dtype=logits.dtype)
        / max(float(temperature), 1e-6)
    )
    nce_loss, valid_rows = _masked_multi_positive_nce(
        logits,
        positive,
        negative,
        dustbin_logit=dustbin_logit,
    )
    source_valid = observations.source_indices >= 0
    unmatched = ~source_valid
    source_retrieved = torch.zeros(
        query_count, dtype=torch.bool, device=full_scores.device
    )
    if bool(source_valid.any().item()):
        source_retrieved[source_valid] = (
            top_indices[source_valid]
            == observations.source_indices[source_valid, None]
        ).any(dim=1)
    # A projected surface location can be represented by several overlapping
    # surfels. Retrieval supervision therefore uses the full GT-consistent
    # landmark set, while exact source-ID recall remains a diagnostic only.
    geometric_retrieved = positive.any(dim=1)
    missed_positive = source_valid & ~geometric_retrieved

    # The deployed candidate set is the detached full-bank top-K above.  Its
    # four outcomes are mutually exclusive for a native proposal with a valid
    # GT association: a correct top-1 is kept, a wrong top-1 with a retained
    # positive is swapped, and a positive outside top-K is a miss.  Proposals
    # without a valid association are rejected only when the candidate set is
    # likewise geometrically empty.  This protects already clean pairs while
    # retaining a direct gradient for false positives and false negatives.
    top1_positive = positive[:, 0]
    keep = source_valid & top1_positive
    swap = source_valid & ~top1_positive & geometric_retrieved
    reject = ~source_valid & ~geometric_retrieved
    zero = full_scores.sum() * 0.0

    candidate_positive_score = raw_logits.masked_fill(
        ~positive, -torch.inf
    ).max(dim=1).values
    candidate_negative_score = raw_logits.masked_fill(
        ~negative, -torch.inf
    ).max(dim=1).values
    keep_valid = keep & torch.isfinite(candidate_negative_score)
    keep_loss = (
        F.relu(
            float(native_keep_margin)
            + candidate_negative_score[keep_valid]
            - raw_logits[keep_valid, 0]
        ).mean()
        if bool(keep_valid.any().item())
        else zero
    )
    # A 2--4px top-1 is not accurate enough to be a high-precision positive,
    # but it is still a useful correspondence that a later residual stage
    # should not casually destroy while repairing false top-1 assignments.
    # This tier is optional so historical native objectives remain bitwise
    # equivalent when its weight is zero.
    loose_keep = torch.zeros_like(top1_positive)
    loose_keep_loss = zero
    if float(native_keep_loose_weight) > 0.0:
        loose_radius = float(native_keep_loose_radius_px)
        if loose_radius < float(positive_radius_px):
            raise ValueError(
                "native_keep_loose_radius_px must be at least positive_radius_px"
            )
        loose_positive, _, _ = _candidate_geometry_masks(
            observations,
            candidate_indices,
            positive_radius_px=loose_radius,
            negative_radius_px=negative_radius_px,
        )
        loose_keep = (~top1_positive) & loose_positive[:, 0]
        loose_keep_valid = loose_keep & torch.isfinite(candidate_negative_score)
        if bool(loose_keep_valid.any().item()):
            loose_keep_loss = F.relu(
                float(native_keep_loose_margin)
                + candidate_negative_score[loose_keep_valid]
                - raw_logits[loose_keep_valid, 0]
            ).mean()
    swap_loss = (
        F.relu(
            float(native_swap_margin)
            + raw_logits[swap, 0]
            - candidate_positive_score[swap]
        ).mean()
        if bool(swap.any().item())
        else zero
    )

    full_positive_scores = None
    def _full_positive_best(rows):
        nonlocal full_positive_scores
        if not bool(rows.any().item()):
            return full_scores.new_zeros((0,)), rows
        bank_uv = observations.bank_uv
        query_uv = observations.query_uv[rows]
        distance_squared = (
            query_uv.square().sum(dim=1, keepdim=True)
            + bank_uv.square().sum(dim=1)[None]
            - 2.0 * (query_uv @ bank_uv.T)
        ).clamp_min(0.0)
        full_positive_scores = (
            observations.bank_visible[None]
            & torch.isfinite(distance_squared)
            & (distance_squared <= float(positive_radius_px) ** 2)
        )
        return (
            full_scores[rows].masked_fill(~full_positive_scores, -torch.inf)
            .max(dim=1)
            .values,
            rows,
        )

    missed_positive_loss = zero
    if bool(missed_positive.any().item()):
        positive_scores, missed_rows = _full_positive_best(missed_positive)
        retrieved_best = raw_logits[missed_rows, 0]
        valid_miss = torch.isfinite(positive_scores)
        if bool(valid_miss.any().item()):
            missed_positive_loss = F.relu(
                float(native_miss_margin)
                + retrieved_best[valid_miss]
                - positive_scores[valid_miss]
            ).mean()

    unmatched_loss = zero
    if bool(reject.any().item()):
        unmatched_loss = F.relu(
            raw_logits[reject, 0] - float(native_reject_threshold)
        ).mean()

    # Repeated facade elements can become a false-attractor landmark: many
    # valid native keypoints choose the same wrong top-1 in one query.  The
    # normal swap/miss objective sees these rows independently.  Weight their
    # ranking violations by log(1 + per-landmark false-attractor count), while
    # keeping candidate IDs detached and unchanged.
    attractor_loss = zero
    global_attractor_loss = zero
    attractor_count = 0
    attractor_unique_count = 0
    attractor_max_count = 0
    global_attractor_count = 0
    global_attractor_unique_count = 0
    global_attractor_score_mean = 0.0
    use_local_attractor = float(native_attractor_weight) > 0.0
    use_global_attractor = float(native_global_attractor_weight) > 0.0
    if bool(native_outcome_mode) and (use_local_attractor or use_global_attractor):
        global_scores = None
        if use_global_attractor:
            if native_global_attractor_scores is None:
                raise ValueError(
                    "native_global_attractor_weight requires "
                    "native_global_attractor_scores"
                )
            global_scores = torch.as_tensor(
                native_global_attractor_scores,
                device=raw_logits.device,
                dtype=raw_logits.dtype,
            ).reshape(-1)
            if int(global_scores.numel()) != bank_count:
                raise ValueError(
                    "native_global_attractor_scores must have one entry per "
                    "landmark"
                )
            global_scores = global_scores.detach().clamp_min(0.0)
        false_attractor = source_valid & ~top1_positive
        if bool(false_attractor.any().item()):
            attractor_positive_score, attractor_rows = _full_positive_best(
                false_attractor
            )
            valid_attractor = torch.isfinite(attractor_positive_score)
            if bool(valid_attractor.any().item()):
                attractor_landmarks = top_indices[attractor_rows, 0][
                    valid_attractor
                ]
                attractor_scores = raw_logits[attractor_rows, 0][
                    valid_attractor
                ]
                attractor_positive_score = attractor_positive_score[
                    valid_attractor
                ]
                violation = F.relu(
                    float(native_attractor_margin)
                    + attractor_scores
                    - attractor_positive_score
                )
                if use_local_attractor:
                    counts = torch.zeros(
                        bank_count,
                        device=raw_logits.device,
                        dtype=raw_logits.dtype,
                    )
                    counts.scatter_add_(
                        0,
                        attractor_landmarks,
                        torch.ones_like(attractor_scores),
                    )
                    row_counts = counts[attractor_landmarks]
                    row_weights = torch.log1p(row_counts)
                    row_weights = row_weights / row_weights.mean().detach().clamp_min(1e-8)
                    attractor_loss = (
                        (row_weights * violation).sum()
                        / row_weights.sum().clamp_min(1e-8)
                    )
                    attractor_count = int(attractor_landmarks.numel())
                    attractor_unique_count = int(
                        torch.unique(attractor_landmarks).numel()
                    )
                    attractor_max_count = int(row_counts.max().item())
                if use_global_attractor:
                    global_row_weights = global_scores[attractor_landmarks]
                    globally_supported = global_row_weights > 0.0
                    if bool(globally_supported.any().item()):
                        global_row_weights = global_row_weights[globally_supported]
                        global_violation = violation[globally_supported]
                        global_attractor_loss = (
                            (global_row_weights * global_violation).sum()
                            / global_row_weights.sum().clamp_min(1e-8)
                        )
                        global_landmarks = attractor_landmarks[globally_supported]
                        global_attractor_count = int(global_landmarks.numel())
                        global_attractor_unique_count = int(
                            torch.unique(global_landmarks).numel()
                        )
                        global_attractor_score_mean = float(
                            global_row_weights.mean().detach().item()
                        )

    if native_outcome_mode:
        loss = (
            float(native_nce_weight) * nce_loss
            + float(native_keep_weight) * keep_loss
            + float(native_keep_loose_weight) * loose_keep_loss
            + float(native_swap_weight) * swap_loss
            + float(native_miss_weight) * missed_positive_loss
            + float(native_reject_weight) * unmatched_loss
            + float(native_attractor_weight) * attractor_loss
            + float(native_global_attractor_weight) * global_attractor_loss
        )
    else:
        # Preserve the original hard candidate objective for existing
        # experiments. The explicit outcome terms above are diagnostics until
        # the native-outcome curriculum is selected by the caller.
        loss = nce_loss
        if float(margin) > 0.0 and bool(valid_rows.any().item()):
            positive_score = raw_logits.masked_fill(~positive, -torch.inf).max(dim=1).values
            negative_score = raw_logits.masked_fill(~negative, -torch.inf).max(dim=1).values
            finite = torch.isfinite(positive_score) & torch.isfinite(negative_score)
            if bool(finite.any().item()):
                loss = loss + F.relu(
                    float(margin) + negative_score[finite] - positive_score[finite]
                ).mean()
        if float(missed_positive_weight) > 0.0 and bool(missed_positive.any().item()):
            positive_scores, missed_rows = _full_positive_best(missed_positive)
            retrieved_best = raw_logits[missed_rows, 0]
            valid_miss = torch.isfinite(positive_scores)
            if bool(valid_miss.any().item()):
                legacy_missed_loss = F.relu(
                    float(missed_positive_margin)
                    + retrieved_best[valid_miss]
                    - positive_scores[valid_miss]
                ).mean()
                loss = loss + float(missed_positive_weight) * legacy_missed_loss
                missed_positive_loss = legacy_missed_loss

        legacy_unmatched_loss = zero
        if float(unmatched_rejection_weight) > 0.0 and bool(unmatched.any().item()):
            unmatched_positive = positive[unmatched].any(dim=1)
            truly_unmatched = unmatched.clone()
            truly_unmatched[unmatched] = ~unmatched_positive
            if bool(truly_unmatched.any().item()):
                best_similarity = raw_logits[truly_unmatched, 0]
                legacy_unmatched_loss = F.relu(
                    best_similarity - float(unmatched_max_similarity)
                ).mean()
                loss = loss + float(unmatched_rejection_weight) * legacy_unmatched_loss
                unmatched_loss = legacy_unmatched_loss

    diagnostics = _retrieval_diagnostics(
        full_scores.detach(), observations, candidate_indices=None
    )
    topk_positive = positive.any(dim=1)
    diagnostics.update(
        {
            "retrieval_mode_hard": 1.0,
            "retrieval_candidate_count_mean": float(candidate_indices.shape[1]),
            "retrieval_valid_loss_rows": int(valid_rows.sum().item()),
            "retrieval_positive_count_mean": float(
                positive.float().sum(dim=1).mean().detach().item()
            ),
            "retrieval_negative_count_mean": float(
                negative.float().sum(dim=1).mean().detach().item()
            ),
            "retrieval_gt_recall_at_hypothesis_k": float(
                topk_positive.float().mean().detach().item()
            ),
            "retrieval_source_recall_at_hypothesis_k": float(
                source_retrieved[source_valid].float().mean().detach().item()
                if bool(source_valid.any().item())
                else 0.0
            ),
            "retrieval_missed_source_count": int(
                (source_valid & ~source_retrieved).sum().item()
            ),
            "retrieval_missed_positive_count": int(
                missed_positive.sum().item()
            ),
            "retrieval_missed_positive_loss": float(
                missed_positive_loss.detach().item()
            ),
            "retrieval_unmatched_query_count": int(unmatched.sum().item()),
            "retrieval_unmatched_rejection_loss": float(
                unmatched_loss.detach().item()
            ),
            "retrieval_candidate_source_injected": 0.0,
            "retrieval_native_outcome_mode": float(bool(native_outcome_mode)),
            "retrieval_native_nce_loss": float(nce_loss.detach().item()),
            "retrieval_native_keep_count": int(keep.sum().item()),
            "retrieval_native_keep_loss": float(keep_loss.detach().item()),
            "retrieval_native_keep_loose_count": int(loose_keep.sum().item()),
            "retrieval_native_keep_loose_loss": float(
                loose_keep_loss.detach().item()
            ),
            "retrieval_native_swap_count": int(swap.sum().item()),
            "retrieval_native_swap_loss": float(swap_loss.detach().item()),
            "retrieval_native_miss_count": int(missed_positive.sum().item()),
            "retrieval_native_miss_loss": float(missed_positive_loss.detach().item()),
            "retrieval_native_reject_count": int(reject.sum().item()),
            "retrieval_native_reject_loss": float(unmatched_loss.detach().item()),
            "retrieval_native_attractor_count": attractor_count,
            "retrieval_native_attractor_unique_count": attractor_unique_count,
            "retrieval_native_attractor_max_count": attractor_max_count,
            "retrieval_native_attractor_loss": float(attractor_loss.detach().item()),
            "retrieval_native_global_attractor_count": global_attractor_count,
            "retrieval_native_global_attractor_unique_count": (
                global_attractor_unique_count
            ),
            "retrieval_native_global_attractor_score_mean": (
                global_attractor_score_mean
            ),
            "retrieval_native_global_attractor_loss": float(
                global_attractor_loss.detach().item()
            ),
        }
    )
    return DetectorFreeRetrievalOutput(loss, diagnostics)


def background_dustbin_loss(
    bank_features,
    observations,
    dustbin_score,
    *,
    sample_count=128,
    exclusion_radius_px=6.0,
    background_alpha_max=0.05,
    allow_no_anchor=False,
    hypothesis_topk=32,
    temperature=0.07,
    generator=None,
):
    feature_map = observations.query_feature_map
    if feature_map is None or int(sample_count) <= 0:
        return bank_features.sum() * 0.0, {"dustbin_background_count": 0}
    height, width = feature_map.shape[-2:]
    used_no_anchor = False
    valid_region = (
        observations.query_valid_mask
        if observations.query_valid_mask is not None
        else torch.ones((height, width), dtype=torch.bool, device=feature_map.device)
    )
    if observations.target_alpha_map is not None:
        alpha = torch.as_tensor(
            observations.target_alpha_map,
            device=feature_map.device,
            dtype=feature_map.dtype,
        ).squeeze()
        background = torch.isfinite(alpha) & (alpha <= float(background_alpha_max))
        background &= valid_region
        if not bool(background.any().item()) and bool(allow_no_anchor):
            background = valid_region.clone()
            used_no_anchor = True
        if not bool(background.any().item()):
            return bank_features.sum() * 0.0, {
                "dustbin_background_count": 0,
                "dustbin_coverage_miss_ignored": 1.0,
            }
    else:
        background = valid_region.clone()
        used_no_anchor = True

    visible_uv = observations.bank_uv[observations.bank_visible]
    if visible_uv.numel() > 0:
        rounded = visible_uv.round().long()
        in_bounds = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < height)
        )
        rounded = rounded[in_bounds]
        occupied = torch.zeros(
            (height, width),
            dtype=feature_map.dtype,
            device=feature_map.device,
        )
        occupied[rounded[:, 1], rounded[:, 0]] = 1.0
        radius = max(int(torch.ceil(torch.tensor(float(exclusion_radius_px))).item()), 0)
        if radius > 0:
            occupied = F.max_pool2d(
                occupied[None, None],
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )[0, 0]
        background &= occupied <= 0
    coordinates = torch.nonzero(background, as_tuple=False)
    if coordinates.numel() == 0:
        return bank_features.sum() * 0.0, {"dustbin_background_count": 0}
    order = torch.randperm(
        coordinates.shape[0], device=coordinates.device, generator=generator
    )[: int(sample_count)]
    coordinates = coordinates[order]
    uv = torch.stack([coordinates[:, 1], coordinates[:, 0]], dim=1).to(
        dtype=observations.query_uv.dtype
    )
    if uv.numel() == 0:
        return bank_features.sum() * 0.0, {"dustbin_background_count": 0}
    query = bilinear_sample_features(feature_map.detach(), uv)
    valid = torch.isfinite(query).all(dim=1) & (torch.linalg.norm(query, dim=-1) > 1e-6)
    query = F.normalize(query[valid], dim=-1)
    if query.numel() == 0:
        return bank_features.sum() * 0.0, {"dustbin_background_count": 0}
    scores = query @ F.normalize(bank_features, dim=-1).T
    topk = max(1, min(int(hypothesis_topk), scores.shape[1]))
    candidate_logits = torch.topk(scores, k=topk, dim=1).values / max(
        float(temperature), 1e-6
    )
    dustbin_logit = (
        torch.as_tensor(dustbin_score, device=scores.device, dtype=scores.dtype)
        / max(float(temperature), 1e-6)
    )
    denominator = torch.logsumexp(
        torch.cat(
            [
                candidate_logits,
                dustbin_logit.reshape(1, 1).expand(query.shape[0], 1),
            ],
            dim=1,
        ),
        dim=1,
    )
    loss = (denominator - dustbin_logit).mean()
    with torch.no_grad():
        best = scores.max(dim=1).values
        diagnostics = {
            "dustbin_background_count": int(query.shape[0]),
            "dustbin_no_anchor_count": int(query.shape[0]) if used_no_anchor else 0,
            "dustbin_score": float(dustbin_logit.mul(float(temperature)).item()),
            "dustbin_background_reject_ratio": float(
                (best <= dustbin_logit * float(temperature)).float().mean().item()
            ),
            "dustbin_background_best_score_mean": float(best.mean().item()),
        }
    return loss, diagnostics
