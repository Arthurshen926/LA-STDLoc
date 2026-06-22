from dataclasses import dataclass

import torch


@dataclass
class TopologyConfig:
    stats_warmup: int = 1000
    update_interval: int = 200
    min_observations: int = 8
    split_quantile: float = 0.95
    ambiguity_quantile: float = 0.90
    growth_cap_per_event: float = 0.03
    total_point_budget_ratio: float = 1.25
    cooldown_iterations: int = 300
    enable_loc_clone: bool = False
    min_repeatability: float = 0.25
    min_radius: float = 4.0
    soft_prune_threshold: float = -1.0
    soft_prune_step: float = 1.0


def _quantile(values, q):
    if values.numel() == 0:
        return values.new_tensor(float("inf"))
    return torch.quantile(values.float(), max(0.0, min(1.0, q)))


def select_localization_splits(gaussians, config: TopologyConfig, iteration):
    n = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    if hasattr(gaussians, "_ensure_screen_radius_state"):
        gaussians._ensure_screen_radius_state()
    observed = gaussians.loc_observation_count.to(device=device) >= config.min_observations
    cooldown = (iteration - gaussians.last_topology_iteration.to(device=device)) >= config.cooldown_iterations
    repeatable = gaussians.loc_repeatability_ema.to(device=device) >= config.min_repeatability
    radii = gaussians.max_radii2D.to(device=device)
    large = radii >= config.min_radius
    grad = (gaussians.loc_grad_accum.to(device=device).squeeze(-1) / gaussians.loc_grad_denom.to(device=device).squeeze(-1).clamp_min(1.0))
    entropy = gaussians.loc_entropy_ema.to(device=device)
    eligible = observed & cooldown & repeatable & large
    if eligible.sum() == 0:
        return torch.zeros(n, dtype=torch.bool, device=device)
    grad_thr = _quantile(grad[eligible], config.split_quantile)
    entropy_thr = _quantile(entropy[eligible], config.ambiguity_quantile)
    split = eligible & ((grad >= grad_thr) | (entropy >= entropy_thr))
    cap = max(1, int(n * config.growth_cap_per_event))
    if split.sum() > cap:
        score = grad + entropy
        selected = torch.topk(score.masked_fill(~split, -torch.inf), cap).indices
        capped = torch.zeros_like(split)
        capped[selected] = True
        split = capped
    return split


def apply_localization_soft_prune(gaussians, utility=None, threshold=-1.0, step=1.0):
    if utility is None:
        utility = gaussians.compute_localization_utility()
    utility = utility.to(device=gaussians._loc_opacity.device)
    mask = utility < threshold
    with torch.no_grad():
        gaussians._loc_opacity[mask] -= step
    return mask


def joint_physical_prune_mask(gaussians, utility=None, rgb_threshold=0.005, loc_threshold=0.005, utility_threshold=-3.0):
    if utility is None:
        utility = gaussians.compute_localization_utility()
    utility = utility.to(device=gaussians.get_xyz.device)
    return (
        (gaussians.get_opacity.squeeze(-1) < rgb_threshold)
        & (gaussians.get_loc_opacity.squeeze(-1) < loc_threshold)
        & (utility < utility_threshold)
    )


class LocalizationTopologyController:
    def __init__(self, config: TopologyConfig, initial_points=None):
        self.config = config
        self.initial_points = initial_points

    def should_update(self, iteration):
        return iteration >= self.config.stats_warmup and iteration % self.config.update_interval == 0

    def update(self, gaussians, scene_extent, iteration):
        utility = gaussians.compute_localization_utility(self.config.min_observations)
        apply_localization_soft_prune(
            gaussians,
            utility=utility,
            threshold=self.config.soft_prune_threshold,
            step=self.config.soft_prune_step,
        )
        physical = joint_physical_prune_mask(gaussians, utility=utility)
        if physical.any():
            gaussians.prune_points(physical)
        split = select_localization_splits(gaussians, self.config, iteration)
        budget = int((self.initial_points or gaussians.get_xyz.shape[0]) * self.config.total_point_budget_ratio)
        if split.any() and gaussians.get_xyz.shape[0] < budget:
            point_count_before = gaussians.get_xyz.shape[0]
            split_count = int(split.sum().item())
            grads = torch.zeros_like(gaussians.xyz_gradient_accum)
            grads[split] = gaussians.loc_grad_accum[split] / gaussians.loc_grad_denom[split].clamp_min(1.0)
            gaussians.densify_and_split(grads, grad_threshold=0.0, scene_extent=scene_extent, N=2)
            new_clone_count = max(0, gaussians.get_xyz.shape[0] - (point_count_before - split_count))
            if new_clone_count > 0:
                gaussians.last_topology_iteration[-new_clone_count:] = iteration
