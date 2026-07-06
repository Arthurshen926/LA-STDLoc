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


def voxel_balanced_score(
    points,
    npoints,
    score,
    voxel_size=0.25,
    max_per_voxel=8,
    eligible=None,
    seed_indices=None,
):
    points = points.float()
    score = score.float().to(points.device)
    if eligible is None:
        eligible = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    else:
        eligible = eligible.to(device=points.device, dtype=torch.bool)
    valid_idx = torch.nonzero(eligible, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0 or npoints <= 0:
        return torch.empty(0, dtype=torch.long, device=points.device)

    sample_count = min(int(npoints), int(valid_idx.numel()))
    order = valid_idx[torch.argsort(score[valid_idx], descending=True)]
    voxel_size = max(float(voxel_size), 1e-8)
    max_per_voxel = max(int(max_per_voxel), 1)
    voxel_counts = {}
    if seed_indices is not None:
        for seed_idx in seed_indices.to(device=points.device, dtype=torch.long).tolist():
            voxel = torch.floor(points[seed_idx] / voxel_size).to(dtype=torch.long)
            key = tuple(int(v) for v in voxel.tolist())
            voxel_counts[key] = voxel_counts.get(key, 0) + 1
    selected = []
    overflow = []

    for full_idx in order.tolist():
        voxel = torch.floor(points[full_idx] / voxel_size).to(dtype=torch.long)
        key = tuple(int(v) for v in voxel.tolist())
        if voxel_counts.get(key, 0) < max_per_voxel:
            selected.append(full_idx)
            voxel_counts[key] = voxel_counts.get(key, 0) + 1
            if len(selected) >= sample_count:
                break
        else:
            overflow.append(full_idx)

    if len(selected) < sample_count:
        selected_set = set(selected)
        for full_idx in overflow:
            if full_idx in selected_set:
                continue
            selected.append(full_idx)
            selected_set.add(full_idx)
            if len(selected) >= sample_count:
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
    pnp_balance=False,
    pnp_voxel_size=0.25,
    pnp_max_per_voxel=8,
    pnp_preserve_ratio=0.5,
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
    elif pnp_balance:
        if spatial:
            target_seed = spatial_knn_score(
                xyz.to(device=base_score.device),
                sample_num,
                combined,
                k=k,
                eligible=eligible,
            )
            pnp_sample_num = int(target_seed.numel())
        else:
            pnp_sample_num = sample_num
        pnp_sample_num = min(pnp_sample_num, sample_num)
        pnp_preserve_ratio = min(max(float(pnp_preserve_ratio), 0.0), 1.0)
        preserve_num = min(pnp_sample_num, int(round(pnp_sample_num * pnp_preserve_ratio)))
        if preserve_num > 0:
            preserved = torch.topk(combined, preserve_num).indices
            fill_eligible = eligible.clone()
            fill_eligible[preserved] = False
        else:
            preserved = torch.empty(0, dtype=torch.long, device=base_score.device)
            fill_eligible = eligible
        fill_num = pnp_sample_num - int(preserved.numel())
        if fill_num > 0:
            filled = voxel_balanced_score(
                xyz.to(device=base_score.device),
                fill_num,
                combined,
                voxel_size=pnp_voxel_size,
                max_per_voxel=pnp_max_per_voxel,
                eligible=fill_eligible,
                seed_indices=preserved,
            )
            sampled = torch.cat([preserved, filled]).sort().values
        else:
            sampled = preserved.sort().values
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
        "pnp_balance": torch.tensor(bool(pnp_balance), device=base_score.device),
        "pnp_voxel_size": torch.tensor(float(pnp_voxel_size), device=base_score.device),
        "pnp_max_per_voxel": torch.tensor(int(pnp_max_per_voxel), device=base_score.device),
        "pnp_preserve_ratio": torch.tensor(float(pnp_preserve_ratio), device=base_score.device),
        "version": torch.tensor(1, device=base_score.device),
    }
    return sampled, meta


def save_landmark_meta(path, meta):
    torch.save(meta, path)
