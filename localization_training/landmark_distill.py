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


def _depth_bin_ids(depth, eligible, bins):
    depth = torch.as_tensor(depth, dtype=torch.float32, device=eligible.device).reshape(-1)
    bins = int(bins)
    ids = torch.full(depth.shape, -1, dtype=torch.long, device=eligible.device)
    if bins <= 1 or depth.numel() == 0:
        return ids
    valid = torch.isfinite(depth)
    if not bool(valid.any().item()):
        return ids
    d_min = depth[valid].min()
    d_max = depth[valid].max()
    span = (d_max - d_min).clamp_min(1e-6)
    ids[valid] = torch.floor((depth[valid] - d_min) / span * bins).to(dtype=torch.long).clamp(0, bins - 1)
    return ids


def _uv_cell_ids(uv, eligible, image_size, grid_size):
    uv = torch.as_tensor(uv, dtype=torch.float32, device=eligible.device).reshape(-1, 2)
    grid_size = int(grid_size)
    ids = torch.full((uv.shape[0],), -1, dtype=torch.long, device=eligible.device)
    if grid_size <= 1 or image_size is None or uv.numel() == 0:
        return ids
    height, width = int(image_size[0]), int(image_size[1])
    valid = torch.isfinite(uv).all(dim=1) & (width > 0) & (height > 0)
    if not bool(valid.any().item()):
        return ids
    cell_x = torch.floor(uv[valid, 0].clamp(0, max(float(width) - 1.0, 0.0)) / max(float(width), 1.0) * grid_size)
    cell_y = torch.floor(uv[valid, 1].clamp(0, max(float(height) - 1.0, 0.0)) / max(float(height), 1.0) * grid_size)
    cell_x = cell_x.to(dtype=torch.long).clamp(0, grid_size - 1)
    cell_y = cell_y.to(dtype=torch.long).clamp(0, grid_size - 1)
    ids[valid] = cell_y * grid_size + cell_x
    return ids


def coverage_balanced_score(
    points,
    npoints,
    score,
    voxel_size=0.25,
    max_per_voxel=8,
    eligible=None,
    seed_indices=None,
    uv=None,
    image_size=None,
    grid_size=0,
    max_per_grid=0,
    depth=None,
    depth_bins=0,
    max_per_depth_bin=0,
    allow_overflow=True,
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
    max_per_grid = int(max_per_grid)
    max_per_depth_bin = int(max_per_depth_bin)
    use_grid = uv is not None and image_size is not None and int(grid_size) > 1 and max_per_grid > 0
    use_depth = depth is not None and int(depth_bins) > 1 and max_per_depth_bin > 0
    uv_ids = _uv_cell_ids(uv, eligible, image_size, grid_size) if use_grid else None
    depth_ids = _depth_bin_ids(depth, eligible, depth_bins) if use_depth else None

    voxel_counts = {}
    grid_counts = {}
    depth_counts = {}

    def _seed(full_idx):
        voxel = torch.floor(points[full_idx] / voxel_size).to(dtype=torch.long)
        v_key = tuple(int(v) for v in voxel.tolist())
        voxel_counts[v_key] = voxel_counts.get(v_key, 0) + 1
        if use_grid:
            g_key = int(uv_ids[full_idx].item())
            if g_key >= 0:
                grid_counts[g_key] = grid_counts.get(g_key, 0) + 1
        if use_depth:
            d_key = int(depth_ids[full_idx].item())
            if d_key >= 0:
                depth_counts[d_key] = depth_counts.get(d_key, 0) + 1

    def _allowed(full_idx):
        voxel = torch.floor(points[full_idx] / voxel_size).to(dtype=torch.long)
        v_key = tuple(int(v) for v in voxel.tolist())
        if voxel_counts.get(v_key, 0) >= max_per_voxel:
            return False
        if use_grid:
            g_key = int(uv_ids[full_idx].item())
            if g_key >= 0 and grid_counts.get(g_key, 0) >= max_per_grid:
                return False
        if use_depth:
            d_key = int(depth_ids[full_idx].item())
            if d_key >= 0 and depth_counts.get(d_key, 0) >= max_per_depth_bin:
                return False
        return True

    if seed_indices is not None:
        for seed_idx in seed_indices.to(device=points.device, dtype=torch.long).tolist():
            _seed(int(seed_idx))

    selected = []
    overflow = []
    for full_idx in order.tolist():
        full_idx = int(full_idx)
        if _allowed(full_idx):
            selected.append(full_idx)
            _seed(full_idx)
            if len(selected) >= sample_count:
                break
        else:
            overflow.append(full_idx)

    if allow_overflow and len(selected) < sample_count:
        selected_set = set(selected)
        for full_idx in overflow:
            if full_idx in selected_set:
                continue
            selected.append(full_idx)
            selected_set.add(full_idx)
            if len(selected) >= sample_count:
                break

    return torch.tensor(selected, dtype=torch.long, device=points.device).sort().values


def _topk_eligible(score, npoints, eligible, exclude=None):
    score = score.float()
    eligible = eligible.to(device=score.device, dtype=torch.bool).clone()
    if exclude is not None and exclude.numel() > 0:
        eligible[exclude.to(device=score.device, dtype=torch.long)] = False
    valid_idx = torch.nonzero(eligible, as_tuple=False).squeeze(1)
    if valid_idx.numel() == 0 or npoints <= 0:
        return torch.empty(0, dtype=torch.long, device=score.device)
    count = min(int(npoints), int(valid_idx.numel()))
    order = torch.argsort(score[valid_idx], descending=True)
    return valid_idx[order[:count]]


def _unique_cat(parts, device):
    selected = []
    seen = set()
    for part in parts:
        if part is None:
            continue
        for idx in part.detach().cpu().reshape(-1).tolist():
            idx = int(idx)
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)
    return torch.tensor(selected, dtype=torch.long, device=device)


def coverage_preserving_sample(
    xyz,
    base_score,
    utility,
    num=16384,
    k=32,
    min_observations=None,
    base_weight=1.0,
    utility_weight=1.0,
    base_preserve_ratio=0.5,
    utility_preserve_ratio=0.25,
    high_confidence=None,
    high_confidence_ratio=0.0,
    voxel_size=0.25,
    max_per_voxel=8,
    uv=None,
    image_size=None,
    grid_size=0,
    max_per_grid=0,
    depth=None,
    depth_bins=0,
    max_per_depth_bin=0,
    allow_unbalanced_fallback=False,
):
    """Union visible-stable, pose-useful, and coverage-balanced landmarks."""
    base_score = base_score.float()
    utility = utility.float().to(base_score.device)
    xyz = xyz.to(device=base_score.device)
    if min_observations is None:
        eligible = torch.ones_like(base_score, dtype=torch.bool)
    else:
        eligible = min_observations.to(device=base_score.device, dtype=torch.bool)

    sample_num = min(int(num), int(eligible.sum().item()))
    if sample_num <= 0:
        sampled = torch.empty(0, dtype=torch.long, device=base_score.device)
        empty_score = base_score.new_empty(0)
        return sampled, {
            "indices": sampled,
            "score": empty_score,
            "base_score": empty_score,
            "utility": empty_score,
            "full_score": base_score.new_zeros(base_score.shape),
            "source_visible_stable_count": torch.tensor(0, device=base_score.device),
            "source_pose_useful_count": torch.tensor(0, device=base_score.device),
            "source_high_confidence_count": torch.tensor(0, device=base_score.device),
            "source_fill_count": torch.tensor(0, device=base_score.device),
            "source_relaxed_fill_count": torch.tensor(0, device=base_score.device),
            "source_fallback_count": torch.tensor(0, device=base_score.device),
            "coverage_requested_count": torch.tensor(0, device=base_score.device),
            "coverage_underfill_count": torch.tensor(0, device=base_score.device),
            "coverage_allow_unbalanced_fallback": torch.tensor(
                bool(allow_unbalanced_fallback), device=base_score.device
            ),
            "version": torch.tensor(2, device=base_score.device),
        }

    base_preserve_ratio = min(max(float(base_preserve_ratio), 0.0), 1.0)
    utility_preserve_ratio = min(max(float(utility_preserve_ratio), 0.0), 1.0)
    high_confidence_ratio = min(max(float(high_confidence_ratio), 0.0), 1.0)
    base_norm = robust_normalize(base_score, eligible)
    utility_norm = robust_normalize(utility, eligible)
    if high_confidence is not None:
        high_confidence = high_confidence.float().to(base_score.device)
        high_confidence_norm = robust_normalize(high_confidence, eligible)
    else:
        high_confidence_norm = None
    combined = base_weight * base_norm + utility_weight * utility_norm
    if high_confidence_norm is not None:
        combined = combined + high_confidence_norm
    combined = combined.masked_fill(~eligible, -torch.inf)

    base_num = min(sample_num, int(round(sample_num * base_preserve_ratio)))
    high_confidence_num = min(
        sample_num - base_num,
        int(round(sample_num * high_confidence_ratio)) if high_confidence_norm is not None else 0,
    )
    utility_num = min(sample_num - base_num - high_confidence_num, int(round(sample_num * utility_preserve_ratio)))

    pose_idx = coverage_balanced_score(
        xyz,
        utility_num,
        utility_norm,
        voxel_size=voxel_size,
        max_per_voxel=max_per_voxel,
        eligible=eligible,
        uv=uv,
        image_size=image_size,
        grid_size=grid_size,
        max_per_grid=max_per_grid,
        depth=depth,
        depth_bins=depth_bins,
        max_per_depth_bin=max_per_depth_bin,
        allow_overflow=False,
    )

    high_idx = torch.empty(0, dtype=torch.long, device=base_score.device)
    if high_confidence_num > 0:
        high_eligible = eligible.clone()
        if pose_idx.numel() > 0:
            high_eligible[pose_idx] = False
        high_idx = coverage_balanced_score(
            xyz,
            high_confidence_num,
            high_confidence_norm,
            voxel_size=voxel_size,
            max_per_voxel=max_per_voxel,
            eligible=high_eligible,
            seed_indices=pose_idx,
            uv=uv,
            image_size=image_size,
            grid_size=grid_size,
            max_per_grid=max_per_grid,
            depth=depth,
            depth_bins=depth_bins,
            max_per_depth_bin=max_per_depth_bin,
            allow_overflow=False,
        )

    stable_eligible = eligible.clone()
    selected_seed = _unique_cat([pose_idx, high_idx], device=base_score.device)
    if selected_seed.numel() > 0:
        stable_eligible[selected_seed] = False
    stable_idx = coverage_balanced_score(
        xyz,
        base_num,
        base_norm,
        voxel_size=voxel_size,
        max_per_voxel=max_per_voxel,
        eligible=stable_eligible,
        seed_indices=selected_seed,
        uv=uv,
        image_size=image_size,
        grid_size=grid_size,
        max_per_grid=max_per_grid,
        depth=depth,
        depth_bins=depth_bins,
        max_per_depth_bin=max_per_depth_bin,
        allow_overflow=False,
    )

    selected = _unique_cat([stable_idx, high_idx, pose_idx], device=base_score.device)
    fill_idx = torch.empty(0, dtype=torch.long, device=base_score.device)
    fill_num = sample_num - int(selected.numel())
    if fill_num > 0:
        fill_eligible = eligible.clone()
        if selected.numel() > 0:
            fill_eligible[selected] = False
        fill_idx = coverage_balanced_score(
            xyz,
            fill_num,
            combined,
            voxel_size=voxel_size,
            max_per_voxel=max_per_voxel,
            eligible=fill_eligible,
            seed_indices=selected,
            uv=uv,
            image_size=image_size,
            grid_size=grid_size,
            max_per_grid=max_per_grid,
            depth=depth,
            depth_bins=depth_bins,
            max_per_depth_bin=max_per_depth_bin,
            allow_overflow=False,
        )
        selected = _unique_cat([selected, fill_idx], device=base_score.device)

    relaxed_fill = torch.empty(0, dtype=torch.long, device=base_score.device)
    if selected.numel() < sample_num:
        relaxed_eligible = eligible.clone()
        if selected.numel() > 0:
            relaxed_eligible[selected] = False
        relaxed_fill = coverage_balanced_score(
            xyz,
            sample_num - int(selected.numel()),
            combined,
            voxel_size=voxel_size,
            max_per_voxel=max(int(max_per_voxel), int(sample_num), 1),
            eligible=relaxed_eligible,
            seed_indices=selected,
            uv=uv,
            image_size=image_size,
            grid_size=grid_size,
            max_per_grid=max_per_grid,
            depth=depth,
            depth_bins=depth_bins,
            max_per_depth_bin=max_per_depth_bin,
            allow_overflow=False,
        )
        selected = _unique_cat([selected, relaxed_fill], device=base_score.device)

    fallback = torch.empty(0, dtype=torch.long, device=base_score.device)
    grid_limited = uv is not None and image_size is not None and int(grid_size) > 1 and int(max_per_grid) > 0
    depth_limited = depth is not None and int(depth_bins) > 1 and int(max_per_depth_bin) > 0
    unbalanced_fallback_allowed = bool(allow_unbalanced_fallback) or not (grid_limited or depth_limited)
    if selected.numel() < sample_num and unbalanced_fallback_allowed:
        fallback = _topk_eligible(combined, sample_num - int(selected.numel()), eligible, exclude=selected)
        selected = _unique_cat([selected, fallback], device=base_score.device)

    sampled = selected[:sample_num].sort().values
    meta = {
        "indices": sampled,
        "score": combined[sampled],
        "base_score": base_score[sampled],
        "utility": utility[sampled],
        "full_score": combined.detach().clone(),
        "source_visible_stable_count": torch.tensor(int(stable_idx.numel()), device=base_score.device),
        "source_pose_useful_count": torch.tensor(int(pose_idx.numel()), device=base_score.device),
        "source_high_confidence_count": torch.tensor(int(high_idx.numel()), device=base_score.device),
        "source_fill_count": torch.tensor(int(fill_idx.numel()), device=base_score.device),
        "source_relaxed_fill_count": torch.tensor(int(relaxed_fill.numel()), device=base_score.device),
        "source_fallback_count": torch.tensor(int(fallback.numel()), device=base_score.device),
        "coverage_requested_count": torch.tensor(int(sample_num), device=base_score.device),
        "coverage_underfill_count": torch.tensor(
            max(0, int(sample_num) - int(sampled.numel())),
            device=base_score.device,
        ),
        "coverage_allow_unbalanced_fallback": torch.tensor(
            bool(allow_unbalanced_fallback),
            device=base_score.device,
        ),
        "coverage_preserve_ratio": torch.tensor(float(base_preserve_ratio), device=base_score.device),
        "coverage_utility_ratio": torch.tensor(float(utility_preserve_ratio), device=base_score.device),
        "coverage_high_confidence_ratio": torch.tensor(float(high_confidence_ratio), device=base_score.device),
        "coverage_voxel_size": torch.tensor(float(voxel_size), device=base_score.device),
        "coverage_max_per_voxel": torch.tensor(int(max_per_voxel), device=base_score.device),
        "version": torch.tensor(2, device=base_score.device),
    }
    if high_confidence is not None:
        meta["high_confidence"] = high_confidence[sampled]
    if uv is not None:
        meta["coverage_uv"] = torch.as_tensor(uv, dtype=torch.float32, device=base_score.device)[sampled]
        meta["coverage_grid_size"] = torch.tensor(int(grid_size), device=base_score.device)
        meta["coverage_max_per_grid"] = torch.tensor(int(max_per_grid), device=base_score.device)
        if image_size is not None:
            meta["coverage_image_size"] = torch.tensor(
                [int(image_size[0]), int(image_size[1])],
                dtype=torch.long,
                device=base_score.device,
            )
    if depth is not None:
        full_depth = torch.as_tensor(depth, dtype=torch.float32, device=base_score.device).reshape(-1)
        meta["coverage_depth"] = full_depth[sampled]
        meta["coverage_depth_bins"] = torch.tensor(int(depth_bins), device=base_score.device)
        meta["coverage_max_per_depth_bin"] = torch.tensor(int(max_per_depth_bin), device=base_score.device)
        finite_depth = full_depth[torch.isfinite(full_depth)]
        if finite_depth.numel() > 0:
            meta["coverage_depth_min"] = finite_depth.min()
            meta["coverage_depth_max"] = finite_depth.max()
    return sampled, meta


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
