import torch


def robust_zscore(values, mask=None, eps=1e-6):
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


def summarize_localization_state(gaussians):
    observed = gaussians.loc_observation_count > 0
    return {
        "points": int(gaussians.get_xyz.shape[0]),
        "observed": int(observed.sum().item()),
        "mean_observations": float(gaussians.loc_observation_count.float().mean().item()),
        "mean_loc_opacity": float(gaussians.get_loc_opacity.mean().item()),
    }
