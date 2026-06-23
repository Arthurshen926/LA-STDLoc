import torch


def robust_normalize(values, mask=None, eps=1e-6):
    values = values.float()
    out = torch.zeros_like(values)
    if mask is None:
        mask = torch.ones_like(values, dtype=torch.bool)
    if mask.sum() == 0:
        return out
    data = values[mask]
    median = data.median()
    mad = (data - median).abs().median().clamp_min(eps)
    out[mask] = ((values[mask] - median) / (1.4826 * mad)).clamp(-5.0, 5.0)
    return out


def spatial_knn_score(points, npoints, score, k=32, eligible=None):
    points = points.float()
    score = score.float().to(points.device)
    if eligible is None:
        eligible = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    else:
        eligible = eligible.to(device=points.device, dtype=torch.bool)
    valid_idx = torch.nonzero(eligible, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0 or npoints <= 0:
        return torch.empty(0, dtype=torch.long, device=points.device)

    points_valid = points[valid_idx]
    score_valid = score[valid_idx]
    sample_count = min(int(npoints), points_valid.shape[0])
    seed_idx = torch.randperm(points_valid.shape[0], device=points_valid.device)[:sample_count]
    sampled_points = points_valid[seed_idx]

    k_eff = max(1, min(int(k), points_valid.shape[0]))
    dist = torch.cdist(sampled_points.detach().cpu(), points_valid.detach().cpu())
    knn_idx = torch.topk(dist, k_eff, largest=False, dim=-1).indices.to(points.device)
    knn_score = score_valid[knn_idx]
    score_order = torch.argsort(knn_score, descending=True, dim=-1)

    selected = []
    selected_set = set()
    for row in range(score_order.shape[0]):
        for col in score_order[row]:
            local_idx = int(knn_idx[row, col].item())
            full_idx = int(valid_idx[local_idx].item())
            if full_idx not in selected_set:
                selected_set.add(full_idx)
                selected.append(full_idx)
                break

    return torch.tensor(selected, dtype=torch.long, device=points.device).sort().values


def localization_aware_sample(
    xyz,
    base_score,
    utility,
    num=16384,
    k=32,
    min_observations=None,
    base_weight=1.0,
    utility_weight=1.0,
    spatial=True,
):
    """Select landmarks using baseline match score plus localization utility."""
    base_score = base_score.float()
    utility = utility.float().to(base_score.device)
    if min_observations is None:
        eligible = torch.ones_like(base_score, dtype=torch.bool)
    else:
        eligible = min_observations.to(device=base_score.device, dtype=torch.bool)

    combined = base_weight * robust_normalize(base_score, eligible) + utility_weight * robust_normalize(utility, eligible)
    combined = combined.masked_fill(~eligible, -torch.inf)
    sample_num = min(num, int(eligible.sum().item()))
    if sample_num == 0:
        sampled = torch.empty(0, dtype=torch.long, device=base_score.device)
    elif not spatial:
        sampled = torch.topk(combined, sample_num).indices.sort().values
    else:
        sampled = spatial_knn_score(
            xyz.to(device=base_score.device),
            sample_num,
            combined,
            k=k,
            eligible=eligible,
        )

    meta = {
        "indices": sampled,
        "score": combined[sampled],
        "base_score": base_score[sampled],
        "utility": utility[sampled],
        "version": torch.tensor(1, device=base_score.device),
    }
    return sampled, meta


def save_landmark_meta(path, meta):
    torch.save(meta, path)
