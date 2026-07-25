import torch
import torch.nn.functional as F


def spatial_pyramid_global_descriptor(
    descriptors,
    keypoints,
    scores,
    image_size,
    *,
    grid_rows=2,
    grid_cols=2,
):
    """Pool native sparse descriptors into one global spatial context vector."""
    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    keypoints = torch.as_tensor(
        keypoints, device=descriptors.device, dtype=descriptors.dtype
    )
    scores = torch.as_tensor(
        scores, device=descriptors.device, dtype=descriptors.dtype
    ).reshape(-1)
    if descriptors.shape[0] != keypoints.shape[0] or scores.numel() != descriptors.shape[0]:
        raise ValueError("descriptor, keypoint, and score rows must align")
    if descriptors.shape[0] == 0:
        return descriptors.new_zeros(
            descriptors.shape[1] * (1 + int(grid_rows) * int(grid_cols))
        )
    height, width = map(int, image_size)
    weights = scores.clamp_min(0.0) + 1e-6

    def weighted_pool(mask):
        if not bool(mask.any().item()):
            return descriptors.new_zeros(descriptors.shape[1])
        selected_weights = weights[mask]
        pooled = (
            descriptors[mask] * selected_weights[:, None]
        ).sum(dim=0) / selected_weights.sum().clamp_min(1e-8)
        return F.normalize(pooled, dim=0)

    parts = [weighted_pool(torch.ones_like(weights, dtype=torch.bool))]
    x_bin = torch.floor(
        keypoints[:, 0] / max(float(width), 1.0) * int(grid_cols)
    ).long().clamp(0, int(grid_cols) - 1)
    y_bin = torch.floor(
        keypoints[:, 1] / max(float(height), 1.0) * int(grid_rows)
    ).long().clamp(0, int(grid_rows) - 1)
    bins = y_bin * int(grid_cols) + x_bin
    for index in range(int(grid_rows) * int(grid_cols)):
        parts.append(weighted_pool(bins == index))
    return F.normalize(torch.cat(parts), dim=0)


def visibility_context_bias(
    query_embedding,
    support_embeddings,
    support_visibility,
    *,
    nearest_views=8,
    temperature=0.05,
    delta_max=0.01,
    prior_center=0.1,
    prior_scale=0.1,
    normalization="absolute",
):
    """Convert nearest support-view visibility into a bounded map-side bias."""
    query_embedding = F.normalize(
        torch.as_tensor(query_embedding).float(), dim=0
    )
    support_embeddings = F.normalize(
        torch.as_tensor(
            support_embeddings, device=query_embedding.device
        ).float(),
        dim=1,
    )
    support_visibility = torch.as_tensor(
        support_visibility, device=query_embedding.device, dtype=torch.bool
    )
    if support_embeddings.ndim != 2 or support_embeddings.shape[0] == 0:
        raise ValueError("support embeddings must contain at least one view")
    if support_visibility.ndim != 2:
        raise ValueError("support visibility must be 2D")
    if support_visibility.shape[0] != support_embeddings.shape[0]:
        raise ValueError("support embeddings and visibility rows must align")
    count = min(max(int(nearest_views), 1), support_embeddings.shape[0])
    similarities, indices = torch.topk(
        support_embeddings @ query_embedding, count
    )
    weights = torch.softmax(
        similarities / max(float(temperature), 1e-6), dim=0
    )
    prior = (
        support_visibility[indices].float() * weights[:, None]
    ).sum(dim=0)
    global_prior = support_visibility.float().mean(dim=0)
    if normalization == "absolute":
        signal = prior - float(prior_center)
        signal_scale = prior.new_tensor(
            max(float(prior_scale), 1e-6)
        )
    elif normalization == "global_lift":
        signal = prior - global_prior
        signal_scale = torch.quantile(signal.abs(), 0.75).clamp_min(
            1e-4
        ) * max(float(prior_scale), 1e-6)
    else:
        raise ValueError(
            f"unsupported visibility context normalization: {normalization}"
        )
    bias = float(delta_max) * torch.tanh(signal / signal_scale)
    entropy = -(
        weights * weights.clamp_min(1e-8).log()
    ).sum() / max(torch.log(weights.new_tensor(float(count))).item(), 1e-8)
    return bias, {
        "nearest_similarity_mean": float(similarities.mean().item()),
        "nearest_similarity_max": float(similarities.max().item()),
        "nearest_weight_entropy": float(entropy.item()),
        "visible_prior_mean": float(prior.mean().item()),
        "visible_prior_positive_fraction": float(
            (prior > 0).float().mean().item()
        ),
        "global_visible_prior_mean": float(global_prior.mean().item()),
        "context_signal_abs_q75": float(
            torch.quantile(signal.abs(), 0.75).item()
        ),
        "context_bias_std": float(bias.std().item()),
        "context_bias_saturation_fraction": float(
            (bias.abs() >= 0.95 * max(float(delta_max), 1e-8))
            .float()
            .mean()
            .item()
        ),
    }
