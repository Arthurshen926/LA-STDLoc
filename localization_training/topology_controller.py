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
    enable_split: bool = True
    enable_loc_clone: bool = False
    enable_soft_prune: bool = False
    enable_physical_prune: bool = False
    min_repeatability: float = 0.25
    min_radius: float = 4.0
    soft_prune_threshold: float = -1.0
    soft_prune_step: float = 1.0
    physical_rgb_threshold: float = 0.005
    physical_loc_threshold: float = 0.005
    physical_utility_threshold: float = -3.0
    require_loc_opacity_trained_for_physical_prune: bool = True


def _quantile(values, q):
    if values.numel() == 0:
        return values.new_tensor(float("inf"))
    return torch.quantile(values.float(), max(0.0, min(1.0, q)))


_LOCALIZATION_BUFFER_NAMES = (
    "_loc_feature",
    "_loc_opacity",
    "loc_grad_accum",
    "loc_grad_denom",
    "loc_observation_count",
    "loc_repeatability_ema",
    "loc_positive_prob_ema",
    "loc_margin_ema",
    "loc_entropy_ema",
    "loc_outlier_ema",
    "loc_reproj_error_ema",
    "loc_information_ema",
    "loc_redundancy_ema",
    "loc_prototype",
    "loc_prototype_count",
    "loc_birth_iteration",
    "last_topology_iteration",
    "loc_node_id",
    "loc_parent_node_id",
    "loc_source_index",
    "loc_source_xyz",
)


def _assert_localization_buffers_match_point_count(gaussians):
    n = int(gaussians.get_xyz.shape[0])
    mismatches = []
    for name in _LOCALIZATION_BUFFER_NAMES:
        value = getattr(gaussians, name, None)
        if torch.is_tensor(value) and value.numel() > 0 and value.shape[0] != n:
            mismatches.append(f"{name}:{value.shape[0]}")
    if mismatches:
        raise RuntimeError(
            "Localization topology buffer size mismatch after mutation: "
            f"point_count={n}, " + ", ".join(mismatches)
        )


def _utility_quantiles(utility):
    finite = utility.detach().float()
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        return {"q25": 0.0, "q50": 0.0, "q75": 0.0}
    return {
        "q25": float(torch.quantile(finite, 0.25).item()),
        "q50": float(torch.quantile(finite, 0.50).item()),
        "q75": float(torch.quantile(finite, 0.75).item()),
    }


def localization_split_eligible_mask(gaussians, config: TopologyConfig, iteration):
    n = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    if hasattr(gaussians, "_ensure_screen_radius_state"):
        gaussians._ensure_screen_radius_state()
    observed = gaussians.loc_observation_count.to(device=device) >= config.min_observations
    cooldown = (iteration - gaussians.last_topology_iteration.to(device=device)) >= config.cooldown_iterations
    repeatable = gaussians.loc_repeatability_ema.to(device=device) >= config.min_repeatability
    radii = gaussians.max_radii2D.to(device=device)
    large = radii >= config.min_radius
    return observed & cooldown & repeatable & large


def _localization_split_ambiguity(gaussians):
    if hasattr(gaussians, "loc_entropy_ema"):
        return gaussians.loc_entropy_ema.to(device=gaussians.get_xyz.device).float().clamp_min(0.0)
    return torch.ones(gaussians.get_xyz.shape[0], dtype=torch.float32, device=gaussians.get_xyz.device)


def _localization_split_score(gaussians, config: TopologyConfig):
    device = gaussians.get_xyz.device
    if hasattr(gaussians, "compute_split_necessity"):
        return gaussians.compute_split_necessity(
            min_observations=config.min_observations,
            min_radius=config.min_radius,
            min_repeatability=config.min_repeatability,
        ).to(device=device)
    grad = (
        gaussians.loc_grad_accum.to(device=device).squeeze(-1)
        / gaussians.loc_grad_denom.to(device=device).squeeze(-1).clamp_min(1.0)
    )
    entropy = _localization_split_ambiguity(gaussians).to(device=device)
    return grad.clamp_min(0.0) * entropy.clamp_min(0.0) * gaussians.loc_repeatability_ema.to(device=device).clamp(0.0, 1.0)


def _cap_split_mask(split, score, cap):
    cap = int(cap)
    if cap <= 0:
        return torch.zeros_like(split)
    if int(split.sum().item()) <= cap:
        return split
    selected = torch.topk(score.masked_fill(~split, -torch.inf), cap).indices
    capped = torch.zeros_like(split)
    capped[selected] = True
    return capped


def select_localization_splits(gaussians, config: TopologyConfig, iteration):
    n = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    eligible = localization_split_eligible_mask(gaussians, config, iteration).to(device=device)
    if eligible.sum() == 0:
        return torch.zeros(n, dtype=torch.bool, device=device)
    ambiguity = _localization_split_ambiguity(gaussians).to(device=device)
    ambiguity_thr = _quantile(ambiguity[eligible], config.ambiguity_quantile)
    eligible = eligible & (ambiguity >= ambiguity_thr)
    if eligible.sum() == 0:
        return torch.zeros(n, dtype=torch.bool, device=device)
    split_score = _localization_split_score(gaussians, config)
    split_score[~eligible] = 0.0
    score_thr = _quantile(split_score[eligible], config.split_quantile)
    split = eligible & (split_score > 0) & (split_score >= score_thr)
    cap = max(1, int(n * config.growth_cap_per_event))
    return _cap_split_mask(split, split_score, cap)


def apply_localization_soft_prune(gaussians, utility=None, threshold=-1.0, step=1.0):
    if utility is None:
        utility = gaussians.compute_localization_utility()
    utility = utility.to(device=gaussians._loc_opacity.device)
    mask = utility < threshold
    with torch.no_grad():
        gaussians._loc_opacity[mask] -= step
    return mask


def joint_physical_prune_mask(
    gaussians,
    utility=None,
    rgb_threshold=0.005,
    loc_threshold=0.005,
    utility_threshold=-3.0,
    protected_source_indices=None,
):
    if utility is None:
        utility = gaussians.compute_localization_utility()
    utility = utility.to(device=gaussians.get_xyz.device)
    mask = (
        (gaussians.get_opacity.squeeze(-1) < rgb_threshold)
        & (gaussians.get_loc_opacity.squeeze(-1) < loc_threshold)
        & (utility < utility_threshold)
    )
    if protected_source_indices is not None:
        protected_source_indices = torch.as_tensor(
            protected_source_indices,
            dtype=torch.long,
            device=gaussians.get_xyz.device,
        ).reshape(-1)
        if protected_source_indices.numel() > 0:
            source_index = getattr(gaussians, "loc_source_index", None)
            if source_index is None:
                source_index = torch.arange(gaussians.get_xyz.shape[0], dtype=torch.long, device=gaussians.get_xyz.device)
            else:
                source_index = source_index.to(device=gaussians.get_xyz.device, dtype=torch.long)
            mask = mask & ~torch.isin(source_index, protected_source_indices)
    return mask


class LocalizationTopologyController:
    def __init__(self, config: TopologyConfig, initial_points=None, protected_source_indices=None):
        self.config = config
        self.initial_points = initial_points
        self.last_event = None
        self.protected_source_indices = protected_source_indices

    def should_update(self, iteration):
        return iteration >= self.config.stats_warmup and iteration % self.config.update_interval == 0

    def update(self, gaussians, scene_extent, iteration):
        _assert_localization_buffers_match_point_count(gaussians)
        point_count_start = int(gaussians.get_xyz.shape[0])
        if hasattr(gaussians, "compute_landmark_reliability"):
            reliability = gaussians.compute_landmark_reliability(self.config.min_observations)
            geometry = gaussians.compute_pose_geometry_value(self.config.min_observations)
            utility = reliability + geometry
        else:
            utility = gaussians.compute_localization_utility(self.config.min_observations)
        if self.config.enable_soft_prune:
            apply_localization_soft_prune(
                gaussians,
                utility=utility,
                threshold=self.config.soft_prune_threshold,
                step=self.config.soft_prune_step,
            )
        if self.config.enable_physical_prune:
            if (
                self.config.require_loc_opacity_trained_for_physical_prune
                and not bool(getattr(gaussians, "loc_opacity_grad_seen", False))
            ):
                raise RuntimeError(
                    "Physical topology prune requires trained loc opacity evidence. "
                    "Enable loc opacity in the loss path until a non-zero loc opacity gradient is observed, "
                    "or pass the explicit legacy override for untrained loc opacity pruning."
                )
            physical = joint_physical_prune_mask(
                gaussians,
                utility=utility,
                rgb_threshold=self.config.physical_rgb_threshold,
                loc_threshold=self.config.physical_loc_threshold,
                utility_threshold=self.config.physical_utility_threshold,
                protected_source_indices=self.protected_source_indices,
            )
        else:
            physical = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
        physical_count = int(physical.sum().item())
        if self.config.enable_physical_prune and physical.any():
            gaussians.prune_points(physical)
            _assert_localization_buffers_match_point_count(gaussians)
        if self.config.enable_split:
            candidate_count = int(localization_split_eligible_mask(gaussians, self.config, iteration).sum().item())
            split = select_localization_splits(gaussians, self.config, iteration)
            budget = int((self.initial_points or gaussians.get_xyz.shape[0]) * self.config.total_point_budget_ratio)
            num_children_per_parent = 2
            net_growth_per_split = max(1, num_children_per_parent - 1)
            remaining_growth_budget = max(0, budget - int(gaussians.get_xyz.shape[0]))
            max_splits_by_budget = remaining_growth_budget // net_growth_per_split
            if split.any():
                split = _cap_split_mask(
                    split,
                    _localization_split_score(gaussians, self.config),
                    max_splits_by_budget,
                )
        else:
            candidate_count = 0
            split = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
            budget = int(gaussians.get_xyz.shape[0])
            num_children_per_parent = 2
        event = {
            "iteration": int(iteration),
            "candidate_count": candidate_count,
            "requested_split_count": int(split.sum().item()),
            "actual_parent_removed": 0,
            "actual_children_added": 0,
            "physical_prune_count": physical_count,
            "point_count_start": point_count_start,
            "point_count_before": int(gaussians.get_xyz.shape[0]),
            "point_count_after": int(gaussians.get_xyz.shape[0]),
            "utility_quantiles": _utility_quantiles(utility),
        }
        if split.any() and gaussians.get_xyz.shape[0] < budget:
            point_count_before = gaussians.get_xyz.shape[0]
            split_count = int(split.sum().item())
            if not hasattr(gaussians, "densify_and_split_selected"):
                raise RuntimeError("Localization topology requires densify_and_split_selected(selected_mask=...).")
            gaussians.densify_and_split_selected(split, scene_extent=scene_extent, N=num_children_per_parent)
            point_count_after = int(gaussians.get_xyz.shape[0])
            expected_after = int(point_count_before + split_count * (num_children_per_parent - 1))
            if point_count_after != expected_after:
                raise RuntimeError(
                    "Localization split requested "
                    f"{split_count} splits with N={num_children_per_parent}, "
                    f"expected {expected_after} points after mutation, got {point_count_after}."
                )
            _assert_localization_buffers_match_point_count(gaussians)
            new_clone_count = split_count * num_children_per_parent
            if new_clone_count > 0:
                gaussians.last_topology_iteration[-new_clone_count:] = iteration
            event.update(
                {
                    "actual_parent_removed": split_count,
                    "actual_children_added": new_clone_count,
                    "point_count_before": int(point_count_before),
                    "point_count_after": point_count_after,
                }
            )
            print(
                "[Topology] "
                f"iter={iteration} candidates={event['candidate_count']} "
                f"physical_prune={event['physical_prune_count']} "
                f"requested_split={split_count} parent_removed={split_count} "
                f"children_added={new_clone_count} points={point_count_before}->{point_count_after} "
                f"utility_q25={event['utility_quantiles']['q25']:.4f} "
                f"utility_q50={event['utility_quantiles']['q50']:.4f} "
                f"utility_q75={event['utility_quantiles']['q75']:.4f}"
            )
        self.last_event = event
        return event
