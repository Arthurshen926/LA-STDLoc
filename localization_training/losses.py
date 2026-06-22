import torch
import torch.nn.functional as F


def _normalize_rows(features):
    return F.normalize(features, p=2, dim=-1)


def symmetric_descriptor_loss(rendered_features, query_features, temperature=0.07):
    """Symmetric cross-entropy for paired rendered/query descriptors."""
    if rendered_features.numel() == 0:
        return rendered_features.new_tensor(0.0)
    rendered_features = _normalize_rows(rendered_features)
    query_features = _normalize_rows(query_features)
    logits = rendered_features @ query_features.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _window_coordinates(target_uv, height, width, radius):
    offsets = torch.arange(-radius, radius + 1, device=target_uv.device, dtype=target_uv.dtype)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    coords = target_uv[:, None, None, :] + torch.stack([dx, dy], dim=-1)[None]
    valid = (
        (coords[..., 0] >= 0)
        & (coords[..., 0] <= width - 1)
        & (coords[..., 1] >= 0)
        & (coords[..., 1] <= height - 1)
    )
    return coords.reshape(target_uv.shape[0], -1, 2), valid.reshape(target_uv.shape[0], -1)


def _sample_many(feature_map, coords):
    c, h, w = feature_map.shape
    flat = coords.reshape(-1, 2)
    x = flat[:, 0] / max(w - 1, 1) * 2.0 - 1.0
    y = flat[:, 1] / max(h - 1, 1) * 2.0 - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(feature_map[None], grid, align_corners=True, padding_mode="zeros")
    return sampled[0, :, :, 0].T.reshape(coords.shape[0], coords.shape[1], c)


def fine_reprojection_loss(
    rendered_features,
    query_feature_map,
    target_uv,
    window_radius=4,
    temperature=0.05,
    confidence=None,
):
    """Local soft-argmax reprojection loss around target pixels."""
    if rendered_features.numel() == 0:
        zero = query_feature_map.new_tensor(0.0)
        return zero, {
            "pred_uv": target_uv.new_zeros((0, 2)),
            "positive_prob": target_uv.new_zeros((0,)),
            "entropy": target_uv.new_zeros((0,)),
            "reproj_error": target_uv.new_zeros((0,)),
        }

    _, height, width = query_feature_map.shape
    coords, valid = _window_coordinates(target_uv, height, width, window_radius)
    window_features = _normalize_rows(_sample_many(query_feature_map, coords))
    rendered_features = _normalize_rows(rendered_features)
    logits = (window_features * rendered_features[:, None, :]).sum(dim=-1) / temperature
    logits = logits.masked_fill(~valid, -torch.finfo(logits.dtype).max)
    prob = F.softmax(logits, dim=-1)
    pred_uv = (prob[..., None] * coords).sum(dim=1)
    reproj_error = torch.linalg.norm(pred_uv - target_uv, dim=-1)

    weights = confidence if confidence is not None else torch.ones_like(reproj_error)
    loss = F.smooth_l1_loss(pred_uv, target_uv, reduction="none").sum(dim=-1)
    loss = (loss * weights).sum() / weights.sum().clamp_min(1e-6)

    center = (coords - target_uv[:, None]).abs().sum(dim=-1).argmin(dim=-1)
    positive_prob = prob.gather(1, center[:, None]).squeeze(1)
    entropy = -(prob.clamp_min(1e-12).log() * prob).sum(dim=1)
    return loss, {
        "pred_uv": pred_uv,
        "positive_prob": positive_prob,
        "entropy": entropy,
        "reproj_error": reproj_error,
    }


def prototype_loss(loc_features, prototypes, weight=None):
    if loc_features.numel() == 0:
        return loc_features.new_tensor(0.0)
    loc_features = _normalize_rows(loc_features)
    prototypes = _normalize_rows(prototypes.detach())
    loss = 1.0 - (loc_features * prototypes).sum(dim=-1)
    if weight is None:
        return loss.mean()
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def _bounded_indices(count, max_count, device):
    if max_count is None or max_count <= 0 or count <= max_count:
        return torch.arange(count, device=device)
    return torch.linspace(0, count - 1, max_count, device=device).round().long()


def hard_negative_ranking_loss(loc_features, prototypes, margin=0.2, max_samples=2048, max_negatives=4096):
    """Separate each descriptor from its hardest non-matching prototype."""
    if loc_features.numel() == 0 or loc_features.shape[0] < 2:
        return loc_features.new_tensor(0.0)
    loc_features = _normalize_rows(loc_features)
    prototypes = _normalize_rows(prototypes.detach())
    count = min(loc_features.shape[0], prototypes.shape[0])
    sample_idx = _bounded_indices(count, max_samples, loc_features.device)
    negative_idx = _bounded_indices(count, max_negatives, loc_features.device)

    query_features = loc_features[sample_idx]
    query_prototypes = prototypes[sample_idx]
    negative_prototypes = prototypes[negative_idx]

    sim = query_features @ negative_prototypes.T
    self_mask = sample_idx[:, None] == negative_idx[None, :]
    neg = sim.masked_fill(self_mask, -torch.inf).max(dim=1).values
    valid = torch.isfinite(neg)
    if not valid.any():
        return loc_features.new_tensor(0.0)

    pos = (query_features * query_prototypes).sum(dim=-1)
    return F.relu(margin + neg[valid] - pos[valid]).mean()


def geometry_anchor_loss(current, anchor, xyz_weight=1.0, scale_weight=0.1, rotation_weight=0.1, point_weight=None):
    """Keep geometry close to a saved baseline map when geometry is unlocked."""
    xyz = (current["xyz"] - anchor["xyz"].to(current["xyz"].device, current["xyz"].dtype)).square().sum(dim=-1)
    scaling = (current["scaling"] - anchor["scaling"].to(current["scaling"].device, current["scaling"].dtype)).square().sum(dim=-1)
    rot_anchor = anchor["rotation"].to(current["rotation"].device, current["rotation"].dtype)
    rot_cur = F.normalize(current["rotation"], p=2, dim=-1)
    rot_ref = F.normalize(rot_anchor, p=2, dim=-1)
    rot_dist = (1.0 - (rot_cur * rot_ref).sum(dim=-1).abs()).clamp_min(0.0)
    loss = xyz_weight * xyz + scale_weight * scaling + rotation_weight * rot_dist
    if point_weight is not None:
        point_weight = point_weight.to(loss.device, loss.dtype).reshape(-1)
        return (loss * point_weight).sum() / point_weight.sum().clamp_min(1e-6)
    return loss.mean()


def localization_opacity_regularizer(loc_opacity, target_density=0.5, sparsity_weight=1.0, density_weight=1.0):
    sparsity = loc_opacity.mean()
    density = (loc_opacity.mean() - target_density).abs()
    return sparsity_weight * sparsity + density_weight * density
