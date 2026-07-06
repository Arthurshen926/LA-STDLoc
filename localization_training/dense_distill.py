import torch
import torch.nn.functional as F


def _normalized_responsibility(weights):
    weights = torch.as_tensor(weights, dtype=torch.float32)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def responsibility_weighted_features(gaussian_features, contributor_ids, responsibility_weights):
    gaussian_features = torch.as_tensor(gaussian_features, dtype=torch.float32)
    contributor_ids = torch.as_tensor(contributor_ids, dtype=torch.long, device=gaussian_features.device)
    weights = _normalized_responsibility(responsibility_weights).to(device=gaussian_features.device)
    if contributor_ids.numel() == 0:
        return gaussian_features.new_zeros((0, gaussian_features.reshape(gaussian_features.shape[0], -1).shape[1]))
    flat_features = gaussian_features.reshape(gaussian_features.shape[0], -1)
    safe_ids = contributor_ids.clamp(0, max(flat_features.shape[0] - 1, 0))
    gathered = flat_features[safe_ids]
    valid = (contributor_ids >= 0) & (contributor_ids < flat_features.shape[0])
    gathered = gathered * valid[..., None].to(dtype=gathered.dtype)
    weighted = (gathered * weights[..., None]).sum(dim=1)
    return F.normalize(weighted, p=2, dim=-1)


def responsibility_reconstruction_metrics(rendered_features, gaussian_features, contributor_ids, responsibility_weights):
    rendered_features = torch.as_tensor(rendered_features, dtype=torch.float32)
    gaussian_features = torch.as_tensor(gaussian_features, dtype=torch.float32, device=rendered_features.device)
    contributor_ids = torch.as_tensor(contributor_ids, dtype=torch.long, device=rendered_features.device)
    weights = torch.as_tensor(responsibility_weights, dtype=torch.float32, device=rendered_features.device)
    if rendered_features.numel() == 0 or contributor_ids.numel() == 0:
        return {
            "mean_cosine": 0.0,
            "min_cosine": 0.0,
            "p10_cosine": 0.0,
            "valid_anchor_count": 0,
        }

    rendered = F.normalize(rendered_features.reshape(rendered_features.shape[0], -1), p=2, dim=-1)
    reconstructed = responsibility_weighted_features(
        gaussian_features,
        contributor_ids,
        weights,
    ).to(device=rendered.device, dtype=rendered.dtype)
    valid = (
        (contributor_ids >= 0)
        & (contributor_ids < gaussian_features.reshape(gaussian_features.shape[0], -1).shape[0])
        & (weights > 0)
    ).any(dim=1)
    cosine = (rendered * reconstructed).sum(dim=-1)
    valid = valid & torch.isfinite(cosine)
    if not bool(valid.any()):
        return {
            "mean_cosine": 0.0,
            "min_cosine": 0.0,
            "p10_cosine": 0.0,
            "valid_anchor_count": 0,
        }
    valid_cosine = cosine[valid]
    return {
        "mean_cosine": float(valid_cosine.mean().item()),
        "min_cosine": float(valid_cosine.min().item()),
        "p10_cosine": float(torch.quantile(valid_cosine, 0.10).item()),
        "valid_anchor_count": int(valid_cosine.numel()),
    }


def responsibility_reconstruction_cosine(rendered_features, gaussian_features, contributor_ids, responsibility_weights):
    rendered_features = torch.as_tensor(rendered_features, dtype=torch.float32)
    gaussian_features = torch.as_tensor(gaussian_features, dtype=torch.float32, device=rendered_features.device)
    contributor_ids = torch.as_tensor(contributor_ids, dtype=torch.long, device=rendered_features.device)
    weights = torch.as_tensor(responsibility_weights, dtype=torch.float32, device=rendered_features.device)
    if rendered_features.numel() == 0 or contributor_ids.numel() == 0:
        return rendered_features.new_zeros((0,))

    rendered = F.normalize(rendered_features.reshape(rendered_features.shape[0], -1), p=2, dim=-1)
    reconstructed = responsibility_weighted_features(
        gaussian_features,
        contributor_ids,
        weights,
    ).to(device=rendered.device, dtype=rendered.dtype)
    valid = (
        (contributor_ids >= 0)
        & (contributor_ids < gaussian_features.reshape(gaussian_features.shape[0], -1).shape[0])
        & (weights > 0)
    ).any(dim=1)
    cosine = (rendered * reconstructed).sum(dim=-1)
    return torch.where(valid & torch.isfinite(cosine), cosine, cosine.new_full(cosine.shape, -1.0))


def responsibility_entropy(responsibility_weights):
    weights = _normalized_responsibility(responsibility_weights)
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1)
    valid = torch.as_tensor(responsibility_weights, dtype=torch.float32).clamp_min(0.0).sum(dim=-1) > 0
    return torch.where(valid, entropy, entropy.new_full(entropy.shape, float("inf")))


def gaussian_teacher_distribution(dense_probs, contributor_ids, responsibility_weights, bank_size):
    dense_probs = torch.as_tensor(dense_probs, dtype=torch.float32)
    contributor_ids = torch.as_tensor(contributor_ids, dtype=torch.long, device=dense_probs.device)
    weights = _normalized_responsibility(responsibility_weights).to(device=dense_probs.device)
    if dense_probs.ndim != 2:
        raise ValueError(f"dense_probs must have shape [Q, M], got {tuple(dense_probs.shape)}.")
    if contributor_ids.ndim != 2 or contributor_ids.shape[0] != dense_probs.shape[1]:
        raise ValueError(
            "contributor_ids must have shape [M, K] and match dense_probs anchors, "
            f"got {tuple(contributor_ids.shape)} for dense_probs {tuple(dense_probs.shape)}."
        )
    if weights.shape != contributor_ids.shape:
        raise ValueError(f"responsibility_weights must match contributor_ids, got {tuple(weights.shape)}.")

    teacher = dense_probs.new_zeros((dense_probs.shape[0], int(bank_size)))
    for slot in range(contributor_ids.shape[1]):
        ids = contributor_ids[:, slot]
        valid = (ids >= 0) & (ids < int(bank_size))
        if not valid.any():
            continue
        contribution = dense_probs[:, valid] * weights[valid, slot][None]
        teacher.scatter_add_(1, ids[valid][None].expand(dense_probs.shape[0], -1), contribution)
    return teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)


def dense_to_sparse_kl(query_features, bank_features, teacher_distribution, temperature=0.07, anchor_weights=None):
    query = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1)
    bank = F.normalize(bank_features.reshape(bank_features.shape[0], -1), p=2, dim=-1)
    teacher = torch.as_tensor(teacher_distribution, device=query.device, dtype=query.dtype)
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    logits = query @ bank.T / max(float(temperature), 1e-6)
    log_probs = F.log_softmax(logits, dim=1)
    per_anchor = F.kl_div(log_probs, teacher.detach(), reduction="none").sum(dim=1)
    if anchor_weights is None:
        return per_anchor.mean()
    weights = torch.as_tensor(anchor_weights, device=query.device, dtype=query.dtype).reshape(-1)[: per_anchor.numel()]
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    denom = weights.sum()
    if denom <= 0:
        return per_anchor.new_tensor(0.0)
    return (per_anchor * weights).sum() / denom.clamp_min(1e-8)


def dense_sparse_miss_hit_rank_loss(
    query_features,
    bank_features,
    teacher_distribution,
    temperature=0.07,
    anchor_weights=None,
    teacher_confidence_threshold=0.0,
    miss_topk=1,
    margin=0.2,
    return_diagnostics=False,
):
    query = F.normalize(query_features.reshape(query_features.shape[0], -1), p=2, dim=-1)
    bank = F.normalize(bank_features.reshape(bank_features.shape[0], -1), p=2, dim=-1)
    teacher = torch.as_tensor(teacher_distribution, device=query.device, dtype=query.dtype)
    if query.numel() == 0 or bank.numel() == 0 or teacher.numel() == 0:
        loss = query.new_tensor(0.0)
        diagnostics = {
            "dense_rank_anchor_count": 0,
            "dense_rank_eligible_anchor_count": 0,
            "dense_rank_sparse_hit_count": 0,
            "dense_rank_sparse_miss_count": 0,
            "dense_rank_low_confidence_count": 0,
            "dense_rank_teacher_confidence_mean": 0.0,
        }
        return (loss, diagnostics) if return_diagnostics else loss

    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-8)
    rows = min(query.shape[0], teacher.shape[0])
    query = query[:rows]
    teacher = teacher[:rows]
    logits = query @ bank.T / max(float(temperature), 1e-6)
    teacher_confidence, teacher_index = teacher.max(dim=1)

    weights = query.new_ones(rows)
    if anchor_weights is not None:
        weights = torch.as_tensor(anchor_weights, device=query.device, dtype=query.dtype).reshape(-1)[:rows]
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    active = weights > 0
    high_confidence = (teacher_confidence >= float(teacher_confidence_threshold)) & active

    topk = int(miss_topk)
    if topk > 0 and logits.shape[1] > 0:
        topk = min(topk, logits.shape[1])
        sparse_topk = logits.topk(k=topk, dim=1).indices
        sparse_hit = (sparse_topk == teacher_index[:, None]).any(dim=1)
    else:
        sparse_hit = torch.zeros(rows, dtype=torch.bool, device=query.device)
    sparse_miss = high_confidence & (~sparse_hit)

    positive = logits.gather(1, teacher_index[:, None]).squeeze(1)
    negative_logits = logits.scatter(1, teacher_index[:, None], float("-inf"))
    hardest_negative = negative_logits.max(dim=1).values
    per_anchor = F.relu(float(margin) + hardest_negative - positive)

    rank_weights = weights * sparse_miss.to(dtype=weights.dtype)
    denom = rank_weights.sum()
    if denom <= 0:
        loss = per_anchor.new_tensor(0.0)
    else:
        loss = (per_anchor * rank_weights).sum() / denom.clamp_min(1e-8)

    diagnostics = {
        "dense_rank_anchor_count": int(rows),
        "dense_rank_eligible_anchor_count": int(sparse_miss.sum().item()),
        "dense_rank_sparse_hit_count": int((high_confidence & sparse_hit).sum().item()),
        "dense_rank_sparse_miss_count": int(sparse_miss.sum().item()),
        "dense_rank_low_confidence_count": int((active & (~high_confidence)).sum().item()),
        "dense_rank_teacher_confidence_mean": float(teacher_confidence[active].mean().item()) if bool(active.any()) else 0.0,
    }
    return (loss, diagnostics) if return_diagnostics else loss
