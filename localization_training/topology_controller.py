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
    max_mutation_events: int = 0
    risk_commit_policy: str = "off"
    pose_information_floor: float = 0.0
    residual_score_floor: float = 0.0


@dataclass
class TopologyMutationProposal:
    iteration: int
    split_mask: torch.Tensor
    physical_prune_mask: torch.Tensor
    soft_prune_mask: torch.Tensor
    utility: torch.Tensor
    candidate_count: int
    point_count_start: int
    point_count_before: int
    budget: int
    num_children_per_parent: int = 2


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


def _coerce_risk_decision(decision):
    if isinstance(decision, bool):
        return {"accepted": bool(decision), "reason": "callback_bool"}
    if isinstance(decision, dict):
        out = dict(decision)
        out["accepted"] = bool(out.get("accepted", False))
        out.setdefault("reason", "callback")
        return out
    raise TypeError("risk evaluator must return bool or dict with an 'accepted' field")


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
        kwargs = dict(
            min_observations=config.min_observations,
            min_radius=config.min_radius,
            min_repeatability=config.min_repeatability,
        )
        if float(config.pose_information_floor) > 0.0:
            kwargs["pose_information_floor"] = config.pose_information_floor
        if float(config.residual_score_floor) > 0.0:
            kwargs["residual_score_floor"] = config.residual_score_floor
        return gaussians.compute_split_necessity(**kwargs).to(device=device)
    grad = (
        gaussians.loc_grad_accum.to(device=device).squeeze(-1)
        / gaussians.loc_grad_denom.to(device=device).squeeze(-1).clamp_min(1.0)
    )
    entropy = _localization_split_ambiguity(gaussians).to(device=device)
    repeatability = gaussians.loc_repeatability_ema.to(device=device).clamp(0.0, 1.0)
    reprojection = getattr(gaussians, "loc_reproj_error_ema", None)
    if torch.is_tensor(reprojection):
        reprojection = reprojection.to(device=device).float().clamp_min(0.0)
        if bool((reprojection > 0).any()):
            reprojection = reprojection * (1.0 + grad.clamp_min(0.0))
        else:
            reprojection = 1.0 + grad.clamp_min(0.0)
    else:
        reprojection = 1.0 + grad.clamp_min(0.0)
    positive_prob = getattr(gaussians, "loc_positive_prob_ema", None)
    if torch.is_tensor(positive_prob):
        positive_prob = positive_prob.to(device=device).float()
        positive_prob = torch.where(
            positive_prob > 0,
            positive_prob.clamp(0.0, 1.0),
            torch.ones_like(positive_prob),
        )
    pose_information = getattr(gaussians, "loc_information_ema", None)
    footprint = getattr(gaussians, "max_radii2D", None)
    if not torch.is_tensor(footprint):
        footprint = torch.zeros_like(entropy)
    from localization_training.lafgs_reconstruction import pose_aware_split_score

    return pose_aware_split_score(
        footprint=footprint,
        ambiguity=entropy,
        pnp_residual=reprojection,
        repeatability=repeatability,
        positive_prob=positive_prob,
        pose_information=pose_information,
        pose_information_floor=config.pose_information_floor,
        residual_score_floor=config.residual_score_floor,
        min_footprint=config.min_radius,
        min_repeatability=config.min_repeatability,
    )


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


def select_localization_splits(
    gaussians, config: TopologyConfig, iteration, candidate_scope_mask=None
):
    n = gaussians.get_xyz.shape[0]
    device = gaussians.get_xyz.device
    eligible = localization_split_eligible_mask(gaussians, config, iteration).to(device=device)
    if candidate_scope_mask is not None:
        eligible = eligible & candidate_scope_mask.to(device=device, dtype=torch.bool)
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
    def __init__(
        self,
        config: TopologyConfig,
        initial_points=None,
        protected_source_indices=None,
        split_source_indices=None,
        risk_evaluator=None,
    ):
        self.config = config
        self.initial_points = initial_points
        self.last_event = None
        self.protected_source_indices = protected_source_indices
        self.split_source_indices = split_source_indices
        self.mutation_event_count = 0
        self.risk_evaluator = risk_evaluator

    def should_update(self, iteration):
        if int(self.config.max_mutation_events) > 0 and self.mutation_event_count >= int(self.config.max_mutation_events):
            return False
        return iteration >= self.config.stats_warmup and iteration % self.config.update_interval == 0

    def _risk_commit_enabled(self):
        return self.risk_evaluator is not None or self.config.risk_commit_policy != "off"

    def _evaluate_risk_commit(self, proposal, gaussians):
        policy = self.config.risk_commit_policy
        if self.risk_evaluator is not None:
            return _coerce_risk_decision(self.risk_evaluator(proposal, gaussians))
        if policy == "accept_all":
            return {"accepted": True, "reason": "accept_all"}
        if policy == "reject_all":
            return {"accepted": False, "reason": "reject_all"}
        if policy == "callback":
            raise RuntimeError("risk_commit_policy='callback' requires a risk_evaluator")
        if policy == "off":
            return {"accepted": True, "reason": "off"}
        raise ValueError(f"Unknown risk_commit_policy: {policy}")

    def update(self, gaussians, scene_extent, iteration):
        _assert_localization_buffers_match_point_count(gaussians)
        point_count_start = int(gaussians.get_xyz.shape[0])
        risk_decision = None
        risk_commit_active = self._risk_commit_enabled()
        if hasattr(gaussians, "compute_landmark_reliability"):
            reliability = gaussians.compute_landmark_reliability(self.config.min_observations)
            geometry = gaussians.compute_pose_geometry_value(self.config.min_observations)
            utility = reliability + geometry
        else:
            utility = gaussians.compute_localization_utility(self.config.min_observations)
        if self.config.enable_soft_prune:
            if risk_commit_active:
                raise RuntimeError(
                    "Risk commit does not yet support rollback-safe soft prune. "
                    "Disable soft prune for risk-commit experiments."
                )
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
        if self.config.enable_physical_prune and physical.any() and not risk_commit_active:
            gaussians.prune_points(physical)
            _assert_localization_buffers_match_point_count(gaussians)
        if self.config.enable_split:
            split_scope = None
            if self.split_source_indices is not None:
                source_index = getattr(gaussians, "loc_source_index", None)
                if not torch.is_tensor(source_index):
                    raise RuntimeError(
                        "bank-restricted topology requires loc_source_index lineage"
                    )
                split_scope = torch.isin(
                    source_index.to(device=gaussians.get_xyz.device, dtype=torch.long),
                    torch.as_tensor(
                        self.split_source_indices,
                        device=gaussians.get_xyz.device,
                        dtype=torch.long,
                    ).reshape(-1),
                )
            split_scope_count = int(split_scope.sum().item()) if split_scope is not None else point_count_start
            eligible = localization_split_eligible_mask(
                gaussians, self.config, iteration
            )
            if split_scope is not None:
                eligible = eligible & split_scope
            candidate_count = int(eligible.sum().item())
            split = select_localization_splits(
                gaussians,
                self.config,
                iteration,
                candidate_scope_mask=split_scope,
            )
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
            split_scope_count = 0
            candidate_count = 0
            split = torch.zeros(gaussians.get_xyz.shape[0], dtype=torch.bool, device=gaussians.get_xyz.device)
            budget = int(gaussians.get_xyz.shape[0])
            num_children_per_parent = 2
        proposed_split_count = int(split.sum().item())
        if risk_commit_active and (split.any() or physical.any()):
            proposal = TopologyMutationProposal(
                iteration=int(iteration),
                split_mask=split.detach().clone(),
                physical_prune_mask=physical.detach().clone(),
                soft_prune_mask=torch.zeros_like(split, dtype=torch.bool),
                utility=utility.detach().clone(),
                candidate_count=int(candidate_count),
                point_count_start=int(point_count_start),
                point_count_before=int(gaussians.get_xyz.shape[0]),
                budget=int(budget),
                num_children_per_parent=int(num_children_per_parent),
            )
            risk_decision = self._evaluate_risk_commit(proposal, gaussians)
            if not risk_decision["accepted"]:
                split = torch.zeros_like(split, dtype=torch.bool)
                physical = torch.zeros_like(physical, dtype=torch.bool)
                physical_count = 0
            elif physical.any() and split.any():
                raise RuntimeError(
                    "Risk commit currently requires physical prune and split proposals to be evaluated separately. "
                    "Disable physical prune for split-risk experiments."
                )
        if risk_commit_active and self.config.enable_physical_prune and physical.any():
            gaussians.prune_points(physical)
            _assert_localization_buffers_match_point_count(gaussians)
        event = {
            "iteration": int(iteration),
            "candidate_count": candidate_count,
            "split_scope_count": split_scope_count,
            "requested_split_count": proposed_split_count,
            "actual_parent_removed": 0,
            "actual_children_added": 0,
            "physical_prune_count": physical_count,
            "budget": int(budget),
            "point_count_start": point_count_start,
            "point_count_before": int(gaussians.get_xyz.shape[0]),
            "point_count_after": int(gaussians.get_xyz.shape[0]),
            "utility_quantiles": _utility_quantiles(utility),
        }
        if risk_decision is not None:
            event["risk_commit"] = risk_decision
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
                if hasattr(gaussians, "loc_birth_iteration"):
                    gaussians.loc_birth_iteration[-new_clone_count:] = iteration
            event.update(
                {
                    "actual_parent_removed": split_count,
                    "actual_children_added": new_clone_count,
                    "point_count_before": int(point_count_before),
                    "point_count_after": point_count_after,
                }
            )
        if self.config.enable_split or self.config.enable_physical_prune or "risk_commit" in event:
            risk_text = ""
            if "risk_commit" in event:
                risk_parts = [
                    f"risk_accepted={event['risk_commit']['accepted']}",
                    f"risk_reason={str(event['risk_commit'].get('reason', '')).replace(' ', '_')}",
                ]
                for key, label in (
                    ("baseline_risk", "risk_baseline"),
                    ("trial_risk", "risk_trial"),
                    ("delta_risk", "risk_delta"),
                    ("delta_risk_ucb", "risk_delta_ucb"),
                    ("epsilon", "risk_epsilon"),
                    ("ci_z", "risk_ci_z"),
                    ("risk_sample_count", "risk_samples"),
                    ("risk_metric_count", "risk_metric_count"),
                    ("risk_r5_delta", "risk_r5_delta"),
                    ("risk_r2_delta", "risk_r2_delta"),
                    ("risk_tail_fail_delta", "risk_tail_fail_delta"),
                    ("risk_r5_rate_delta", "risk_r5_rate_delta"),
                    ("risk_r2_rate_delta", "risk_r2_rate_delta"),
                    ("risk_tail_fail_rate_delta", "risk_tail_fail_rate_delta"),
                ):
                    value = event["risk_commit"].get(key)
                    if isinstance(value, (int, float)):
                        if key in {"risk_sample_count", "risk_metric_count", "risk_r5_delta", "risk_r2_delta", "risk_tail_fail_delta"}:
                            risk_parts.append(f"{label}={int(value)}")
                        else:
                            risk_parts.append(f"{label}={float(value):.6f}")
                risk_text = (
                    " "
                    + " ".join(risk_parts)
                )
            print(
                "[Topology] "
                f"iter={iteration} scope={event['split_scope_count']} "
                f"eligible={event['candidate_count']} budget={event['budget']} "
                f"physical_prune={event['physical_prune_count']} "
                f"requested_split={event['requested_split_count']} "
                f"parent_removed={event['actual_parent_removed']} "
                f"children_added={event['actual_children_added']} "
                f"points={event['point_count_start']}->{event['point_count_after']} "
                f"utility_q25={event['utility_quantiles']['q25']:.4f} "
                f"utility_q50={event['utility_quantiles']['q50']:.4f} "
                f"utility_q75={event['utility_quantiles']['q75']:.4f}"
                f"{risk_text}"
            )
        if event["actual_children_added"] > 0 or event["physical_prune_count"] > 0:
            self.mutation_event_count += 1
        self.last_event = event
        return event
