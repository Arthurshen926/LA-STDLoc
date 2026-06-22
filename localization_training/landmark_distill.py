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


def localization_aware_sample(
    xyz,
    base_score,
    utility,
    num=16384,
    k=32,
    min_observations=None,
    base_weight=1.0,
    utility_weight=1.0,
):
    """Select landmarks using baseline match score plus localization utility."""
    del xyz, k
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
    else:
        sampled = torch.topk(combined, sample_num, largest=True).indices
        sampled = sampled.sort().values

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
