import torch


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
