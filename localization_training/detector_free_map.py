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
from localization_training.pose_refiner import (
    camera_center_from_w2c,
    weighted_gauss_newton_refine,
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
    query_feature_map: Optional[torch.Tensor] = None
    target_depth_map: Optional[torch.Tensor] = None
    target_alpha_map: Optional[torch.Tensor] = None
    query_valid_mask: Optional[torch.Tensor] = None
    K: Optional[torch.Tensor] = None
    pose_w2c: Optional[torch.Tensor] = None


@dataclass
class DetectorFreeRetrievalOutput:
    loss: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


@dataclass
class LocalSoftCorrespondenceOutput:
    expected_uv: torch.Tensor
    confidence: torch.Tensor
    entropy: torch.Tensor
    valid: torch.Tensor
    diagnostics: dict = field(default_factory=dict)


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
    target_depth=None,
    target_alpha=None,
    bank_visibility_mask=None,
    query_valid_mask=None,
    alpha_threshold=0.2,
    depth_abs_tolerance=1e-3,
    depth_rel_tolerance=0.01,
    max_observations=512,
    grid_rows=8,
    grid_cols=8,
    depth_bins=4,
):
    feature_map = torch.as_tensor(query_feature_map)
    if feature_map.ndim != 3:
        raise ValueError("query_feature_map must have shape [channels, height, width]")
    height, width = feature_map.shape[-2:]
    bank_xyz = torch.as_tensor(
        bank_xyz, device=feature_map.device, dtype=feature_map.dtype
    ).reshape(-1, 3)
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
    bank_uv, bank_depth, bank_projected = project_landmarks_to_query(
        bank_xyz,
        K,
        pose_w2c,
        height,
        width,
    )
    if bank_visibility_mask is None:
        bank_visible = filter_depth_consistent_landmarks(
            bank_uv,
            bank_depth,
            bank_projected,
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
        if bank_visibility_mask.numel() != bank_xyz.shape[0]:
            raise ValueError("bank_visibility_mask must have one value per landmark")
        bank_visible = bank_projected & bank_visibility_mask
    source_indices = _balanced_visible_indices(
        bank_visible,
        bank_uv,
        bank_depth,
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
            query_feature_map=feature_map.detach(),
            target_depth_map=None if target_depth is None else torch.as_tensor(target_depth).detach(),
            target_alpha_map=None if target_alpha is None else torch.as_tensor(target_alpha).detach(),
            query_valid_mask=query_valid_mask,
            K=K,
            pose_w2c=pose_w2c,
        )
    query_uv = bank_uv[source_indices]
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
        source_depth=bank_depth[source_indices],
        bank_uv=bank_uv,
        bank_depth=bank_depth,
        bank_projected=bank_projected,
        bank_visible=bank_visible,
        query_feature_map=feature_map.detach(),
        target_depth_map=None if target_depth is None else torch.as_tensor(target_depth).detach(),
        target_alpha_map=None if target_alpha is None else torch.as_tensor(target_alpha).detach(),
        query_valid_mask=query_valid_mask,
        K=K,
        pose_w2c=pose_w2c,
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
    height, width = feature_map.shape[-2:]
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

    visible_indices = torch.nonzero(
        observations.bank_visible, as_tuple=False
    ).squeeze(1)
    if query_uv.numel() == 0 or visible_indices.numel() == 0:
        source_indices = visible_indices.new_empty(0)
        query_uv = feature_map.new_zeros((0, 2))
    else:
        distances = torch.cdist(query_uv, observations.bank_uv[visible_indices])
        nearest_distance, nearest_position = distances.min(dim=1)
        keep = nearest_distance <= float(positive_search_radius_px)
        query_uv = query_uv[keep]
        source_indices = visible_indices[nearest_position[keep]]

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
        source_depth=observations.bank_depth[source_indices],
        bank_uv=observations.bank_uv,
        bank_depth=observations.bank_depth,
        bank_projected=observations.bank_projected,
        bank_visible=observations.bank_visible,
        query_feature_map=feature_map,
        target_depth_map=observations.target_depth_map,
        target_alpha_map=observations.target_alpha_map,
        query_valid_mask=observations.query_valid_mask,
        K=observations.K,
        pose_w2c=observations.pose_w2c,
    )


def multiview_descriptor_loss(bank_features, observations):
    if observations.source_indices.numel() == 0:
        return bank_features.sum() * 0.0
    source = F.normalize(bank_features[observations.source_indices], dim=-1)
    query = F.normalize(observations.query_features.detach(), dim=-1)
    return (1.0 - (source * query).sum(dim=-1)).mean()


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
    height, width = feature_map.shape[-2:]
    patch_valid = (
        (patch_uv[..., 0] >= 0.0)
        & (patch_uv[..., 0] <= float(width - 1))
        & (patch_uv[..., 1] >= 0.0)
        & (patch_uv[..., 1] <= float(height - 1))
    )
    patch_features = bilinear_sample_features(
        feature_map.detach(),
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
    height, width = feature_map.shape[-2:]
    patch_valid = (
        (patch_uv[..., 0] >= 0.0)
        & (patch_uv[..., 0] <= float(width - 1))
        & (patch_uv[..., 1] >= 0.0)
        & (patch_uv[..., 1] <= float(height - 1))
    )
    patch_features = bilinear_sample_features(
        feature_map.detach(), patch_uv.reshape(-1, 2)
    ).reshape(query_count, offsets.shape[0], -1)
    feature_valid = torch.isfinite(patch_features).all(dim=-1) & (
        torch.linalg.norm(patch_features, dim=-1) > 1e-6
    )
    patch_valid &= feature_valid
    patch_features = F.normalize(
        torch.nan_to_num(patch_features, nan=0.0, posinf=0.0, neginf=0.0), dim=-1
    )
    source = F.normalize(bank_features[observations.source_indices], dim=-1)
    logits = torch.einsum("qd,qpd->qp", source, patch_features)
    logits = logits / max(float(temperature), 1e-6)
    logits = logits.masked_fill(~patch_valid, -torch.inf)
    valid = patch_valid.any(dim=1)
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
    local = local_soft_correspondences(
        bank_features,
        observations,
        radius=local_radius,
        temperature=local_temperature,
    )
    valid = local.valid & (local.confidence > 0)
    if bool(valid.any()):
        reprojection_error = torch.linalg.norm(
            local.expected_uv.detach() - observations.query_uv, dim=1
        )
        reprojection_loss = (
            F.smooth_l1_loss(
                reprojection_error[valid],
                torch.zeros_like(reprojection_error[valid]),
                reduction="none",
                beta=1.0,
            )
            * local.confidence[valid].detach()
        ).sum() / local.confidence[valid].detach().sum().clamp_min(1e-8)
    else:
        reprojection_loss = current_xyz.sum() * 0.0

    depth_loss = current_xyz.sum() * 0.0
    depth_valid_count = 0
    if observations.target_depth_map is not None and observations.query_uv.numel() > 0:
        depth_map = torch.as_tensor(
            observations.target_depth_map,
            device=current_xyz.device,
            dtype=current_xyz.dtype,
        ).squeeze()
        rendered_depth = bilinear_sample_features(
            depth_map[None], observations.query_uv
        )[:, 0]
        depth_valid = (
            torch.isfinite(rendered_depth)
            & (rendered_depth > 0)
            & torch.isfinite(observations.source_depth)
            & (observations.source_depth > 0)
        )
        depth_valid_count = int(depth_valid.sum().detach().item())
        if bool(depth_valid.any()):
            scale = rendered_depth[depth_valid].detach().abs().clamp_min(
                float(depth_scale_floor)
            )
            normalized_error = (
                observations.source_depth[depth_valid] - rendered_depth[depth_valid].detach()
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
):
    if observations.K is None or observations.pose_w2c is None:
        raise ValueError("PoseLayer requires K and the GT world-to-camera pose")
    valid = local_correspondences.valid & (local_correspondences.confidence > 0)
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if valid_indices.numel() < int(min_points):
        zero = current_xyz.sum() * 0.0
        return zero, {"pose_layer_active": 0.0, "pose_layer_points": int(valid_indices.numel())}
    if valid_indices.numel() > int(max_points):
        confidence = local_correspondences.confidence[valid_indices]
        valid_indices = valid_indices[torch.topk(confidence, int(max_points)).indices]
    source_indices = observations.source_indices[valid_indices]
    points = current_xyz[source_indices]
    target_uv = local_correspondences.expected_uv[valid_indices].detach()
    weights = local_correspondences.confidence[valid_indices].detach().clamp_min(1e-4)
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
        detach_points=False,
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
        loss = current_xyz.sum() * 0.0
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
    return {
        "retrieval_query_count": int(scores.shape[0]),
        "retrieval_top1_gt_precision_2px": float(
            (top1_visible & (distance <= 2.0)).float().mean().detach().item()
        ),
        "retrieval_top1_gt_precision_4px": float(
            (top1_visible & (distance <= 4.0)).float().mean().detach().item()
        ),
        "retrieval_source_top1_ratio": float(
            (top1 == observations.source_indices).float().mean().detach().item()
        ),
        "retrieval_top1_reprojection_median_px": float(
            distance.median().detach().item()
        ),
    }


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
    query_count = int(observations.source_indices.numel())
    bank_count = int(bank_features.shape[0])
    if query_count == 0 or bank_count < 2:
        return DetectorFreeRetrievalOutput(bank_features.sum() * 0.0)
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
    dustbin_score=None,
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
    candidate_indices = torch.cat(
        [observations.source_indices[:, None], top_indices], dim=1
    )
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
    loss, valid_rows = _masked_multi_positive_nce(
        logits,
        positive,
        negative,
        dustbin_logit=dustbin_logit,
    )
    if float(margin) > 0.0 and bool(valid_rows.any().item()):
        positive_score = raw_logits.masked_fill(~positive, -torch.inf).max(dim=1).values
        negative_score = raw_logits.masked_fill(~negative, -torch.inf).max(dim=1).values
        finite = torch.isfinite(positive_score) & torch.isfinite(negative_score)
        if bool(finite.any().item()):
            loss = loss + F.relu(
                float(margin) + negative_score[finite] - positive_score[finite]
            ).mean()
    diagnostics = _retrieval_diagnostics(
        full_scores.detach(), observations, candidate_indices=None
    )
    topk_positive = positive[:, 1:].any(dim=1)
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
