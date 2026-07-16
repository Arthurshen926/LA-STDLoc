from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CoresetSchedule:
    budgets: tuple
    boundaries: tuple

    def budget(self, iteration):
        for boundary, budget in zip(self.boundaries, self.budgets):
            if int(iteration) <= int(boundary):
                return int(budget)
        return int(self.budgets[-1])


def make_progressive_budget_schedule(point_count, total_steps, final_budget):
    point_count = int(point_count)
    total_steps = int(total_steps)
    final_budget = min(max(int(final_budget), 1), point_count)
    candidates = [
        point_count,
        max(point_count // 4, final_budget),
        max(point_count // 16, final_budget),
        max(65536, final_budget),
        max(32768, final_budget),
        final_budget,
    ]
    budgets = []
    for value in candidates:
        value = min(value, point_count)
        if not budgets or value < budgets[-1]:
            budgets.append(value)
    warmup = max(int(round(total_steps * 0.1)), 1)
    if len(budgets) == 1:
        boundaries = [total_steps]
    else:
        remaining = max(total_steps - warmup, len(budgets) - 1)
        boundaries = [warmup]
        for stage in range(1, len(budgets)):
            boundaries.append(
                warmup + int(round(remaining * stage / (len(budgets) - 1)))
            )
        boundaries[-1] = total_steps
    return CoresetSchedule(tuple(budgets), tuple(boundaries))


@dataclass
class SurfacePatchAtoms:
    raw_to_atom: torch.Tensor
    identity_patch_to_atom: torch.Tensor
    representative_raw_indices: torch.Tensor
    identity_patch_ids: torch.Tensor
    coverage_cell_ids: torch.Tensor
    redundancy_group_ids: torch.Tensor
    diagnostics: dict


@torch.no_grad()
def aggregate_atom_features(raw_features, raw_to_atom, atom_count, raw_weights=None, chunk_size=65536):
    """Aggregate primitive descriptors into normalized patch descriptors."""
    raw_features = torch.as_tensor(raw_features).float()
    raw_to_atom = torch.as_tensor(
        raw_to_atom, device=raw_features.device, dtype=torch.long
    ).reshape(-1)
    if raw_features.shape[0] != raw_to_atom.numel():
        raise ValueError("raw_features and raw_to_atom must have equal first dimensions")
    if raw_weights is None:
        raw_weights = raw_features.new_ones(raw_to_atom.shape)
    else:
        raw_weights = torch.as_tensor(
            raw_weights, device=raw_features.device, dtype=raw_features.dtype
        ).reshape(-1)
    feature_sum = raw_features.new_zeros((int(atom_count), raw_features.shape[1]))
    weight_sum = raw_features.new_zeros(int(atom_count))
    for start in range(0, raw_to_atom.numel(), int(chunk_size)):
        end = min(start + int(chunk_size), raw_to_atom.numel())
        atom_ids = raw_to_atom[start:end]
        valid = (atom_ids >= 0) & (raw_weights[start:end] > 0)
        if not bool(valid.any()):
            continue
        atom_ids = atom_ids[valid]
        weights = raw_weights[start:end][valid]
        feature_sum.index_add_(
            0, atom_ids, raw_features[start:end][valid] * weights[:, None]
        )
        weight_sum.index_add_(0, atom_ids, weights)
    aggregated = feature_sum / weight_sum[:, None].clamp_min(1e-8)
    return F.normalize(aggregated, dim=1), weight_sum


def provenance_mass_partition(target_groups, target_weights, active_mask, atom_count):
    """Keep active, shadow, and unrepresentable provenance mass distinct."""
    target_groups = torch.as_tensor(target_groups, dtype=torch.long)
    target_weights = torch.as_tensor(
        target_weights, device=target_groups.device, dtype=torch.float32
    )
    active_mask = torch.as_tensor(active_mask, device=target_groups.device, dtype=torch.bool)
    valid = (target_groups >= 0) & (target_groups < int(atom_count))
    safe = target_groups.clamp(0, max(int(atom_count) - 1, 0))
    active = valid & active_mask[safe]
    pool_mass = (target_weights * valid).sum(dim=1)
    active_mass = (target_weights * active).sum(dim=1)
    shadow_mass = (pool_mass - active_mass).clamp_min(0.0)
    missing_mass = (target_weights.sum(dim=1) - pool_mass).clamp_min(0.0)
    return {
        "valid_mask": valid,
        "active_mask": active,
        "pool_mass": pool_mass,
        "active_mass": active_mass,
        "shadow_mass": shadow_mass,
        "missing_mass": missing_mass,
    }


def append_reprojection_positive(
    labels,
    primitive_ids,
    provenance_weights,
    reprojection_groups,
    reprojection_ids,
    reprojection_valid,
    positive_weight=0.75,
):
    """Append a fixed GT-reprojection channel to a provenance distribution."""
    mix = float(positive_weight)
    reprojection_valid = torch.as_tensor(
        reprojection_valid, device=provenance_weights.device, dtype=torch.bool
    )
    provenance_weights = provenance_weights * torch.where(
        reprojection_valid[:, None],
        provenance_weights.new_tensor(1.0 - mix),
        provenance_weights.new_tensor(1.0),
    )
    return (
        torch.cat([labels, reprojection_groups[:, None]], dim=1),
        torch.cat([primitive_ids, reprojection_ids[:, None]], dim=1),
        torch.cat(
            [
                provenance_weights,
                reprojection_valid[:, None].to(provenance_weights.dtype) * mix,
            ],
            dim=1,
        ),
    )


@torch.no_grad()
def make_gradual_budget_schedule(atom_count, total_steps, final_budget, keep_ratio=0.75, warmup_ratio=0.2):
    atom_count = int(atom_count)
    final_budget = min(max(int(final_budget), 1), atom_count)
    budgets = [atom_count]
    while budgets[-1] > final_budget:
        next_budget = max(final_budget, int(round(budgets[-1] * float(keep_ratio))))
        if next_budget == budgets[-1]:
            next_budget = final_budget
        budgets.append(next_budget)
    warmup = max(1, int(round(int(total_steps) * float(warmup_ratio))))
    if len(budgets) == 1:
        return CoresetSchedule(tuple(budgets), (int(total_steps),))
    remaining = max(int(total_steps) - warmup, len(budgets) - 1)
    boundaries = [warmup]
    for stage in range(1, len(budgets)):
        boundaries.append(warmup + int(round(remaining * stage / (len(budgets) - 1))))
    boundaries[-1] = int(total_steps)
    return CoresetSchedule(tuple(budgets), tuple(boundaries))


@torch.no_grad()
def build_surface_patch_atoms(
    xyz,
    normals,
    observation_count,
    *,
    identity_voxel_size=0.02,
    identity_normal_bins=8,
    coverage_voxel_size=0.20,
    redundancy_voxel_size=0.05,
    redundancy_normal_bins=4,
    min_observations=2,
    max_atoms=0,
    identity_patch_priority=None,
):
    """Create observable localization atoms while retaining stable raw IDs."""
    xyz = torch.as_tensor(xyz).float()
    normals = F.normalize(torch.as_tensor(normals, device=xyz.device).float(), dim=1)
    observation_count = torch.as_tensor(
        observation_count, device=xyz.device, dtype=torch.float32
    ).reshape(-1)
    identity_ids, identity_count = build_surface_groups(
        xyz, normals, identity_voxel_size, identity_normal_bins
    )
    if identity_patch_priority is None:
        identity_patch_priority = observation_count.new_zeros(identity_count)
    else:
        identity_patch_priority = torch.as_tensor(
            identity_patch_priority, device=xyz.device, dtype=torch.float32
        ).reshape(-1)
        if identity_patch_priority.numel() != identity_count:
            raise ValueError("identity_patch_priority must have one value per identity patch")
    observed = observation_count >= float(min_observations)
    observed_mass = observation_count.new_zeros(identity_count)
    observed_mass.scatter_add_(0, identity_ids, observation_count * observed)
    # A patch anchor is a weighted geometric medoid, not merely its most
    # frequently observed member. Query-only patches receive uniform weight.
    member_weight = observation_count + (
        identity_patch_priority[identity_ids] > 0
    ).to(observation_count.dtype)
    centroid_sum = xyz.new_zeros((identity_count, 3))
    centroid_weight = observation_count.new_zeros(identity_count)
    centroid_sum.index_add_(0, identity_ids, xyz * member_weight[:, None])
    centroid_weight.index_add_(0, identity_ids, member_weight)
    centroid = centroid_sum / centroid_weight[:, None].clamp_min(1e-8)
    medoid_distance = torch.linalg.norm(xyz - centroid[identity_ids], dim=1)
    representative_score = torch.where(
        member_weight > 0,
        -medoid_distance,
        observation_count.new_full((), -torch.inf),
    )
    representatives = group_representatives(representative_score, identity_ids, identity_count)
    observable_patches = torch.nonzero(
        (observed_mass > 0) | (identity_patch_priority > 0), as_tuple=False
    ).reshape(-1)
    selected_patches = observable_patches
    if int(max_atoms) > 0 and selected_patches.numel() > int(max_atoms):
        candidate_raw = representatives[selected_patches]
        candidate_coverage, candidate_coverage_count = build_surface_groups(
            xyz[candidate_raw], None, coverage_voxel_size, 0
        )
        selected_priority = identity_patch_priority[selected_patches]
        selected_observation = observed_mass[selected_patches]
        observation_scale = selected_observation.max().clamp_min(1.0)
        priority_scale = selected_priority.max().clamp_min(1.0)
        # Query-observed identities are lexically preferred; MV observations fill
        # only the capacity left after preserving the actual candidate domain.
        selection_mass = (
            (selected_priority > 0).float() * 2.0
            + selected_priority / priority_scale
            + 0.5 * selected_observation / observation_scale
        )
        coverage_representatives = group_representatives(
            selection_mass,
            candidate_coverage,
            candidate_coverage_count,
        )
        reserved_local = coverage_representatives[
            coverage_representatives < selected_patches.numel()
        ]
        if reserved_local.numel() > int(max_atoms):
            reserve_score = selection_mass[reserved_local]
            reserved_local = reserved_local[
                torch.topk(reserve_score, int(max_atoms), sorted=False).indices
            ]
        keep_mask = torch.zeros(
            selected_patches.numel(), device=xyz.device, dtype=torch.bool
        )
        keep_mask[reserved_local] = True
        remaining = int(max_atoms) - int(reserved_local.numel())
        if remaining > 0:
            fill_score = selection_mass.clone()
            fill_score[keep_mask] = -torch.inf
            fill_local = torch.topk(fill_score, remaining, sorted=False).indices
            keep_mask[fill_local] = True
        selected_patches = selected_patches[keep_mask]
    representative_raw = representatives[selected_patches]
    patch_to_atom = torch.full(
        (identity_count,), -1, device=xyz.device, dtype=torch.long
    )
    patch_to_atom[selected_patches] = torch.arange(
        selected_patches.numel(), device=xyz.device
    )
    raw_to_atom = patch_to_atom[identity_ids]
    atom_xyz = xyz[representative_raw]
    atom_normals = normals[representative_raw]
    coverage_ids, _ = build_surface_groups(
        atom_xyz, None, coverage_voxel_size, 0
    )
    redundancy_ids, _ = build_surface_groups(
        atom_xyz,
        atom_normals,
        redundancy_voxel_size,
        redundancy_normal_bins,
    )

    mapped = raw_to_atom >= 0
    distance = torch.linalg.norm(
        xyz[mapped] - atom_xyz[raw_to_atom[mapped]], dim=1
    )
    normal_cos = (normals[mapped] * atom_normals[raw_to_atom[mapped]]).sum(dim=1).clamp(-1, 1)
    normal_angle = torch.rad2deg(torch.acos(normal_cos))

    def quantiles(value):
        if value.numel() == 0:
            return {"p50": 0.0, "p95": 0.0, "max": 0.0}
        q = torch.quantile(value.float(), torch.tensor([0.5, 0.95], device=value.device))
        return {"p50": float(q[0]), "p95": float(q[1]), "max": float(value.max())}

    return SurfacePatchAtoms(
        raw_to_atom=raw_to_atom,
        identity_patch_to_atom=patch_to_atom,
        representative_raw_indices=representative_raw,
        identity_patch_ids=selected_patches,
        coverage_cell_ids=coverage_ids,
        redundancy_group_ids=redundancy_ids,
        diagnostics={
            "raw_primitive_count": int(xyz.shape[0]),
            "identity_patch_count_all": int(identity_count),
            "observable_identity_patch_count": int(observable_patches.numel()),
            "query_observed_identity_patch_count": int(
                (identity_patch_priority > 0).sum()
            ),
            "atom_count": int(representative_raw.numel()),
            "mapped_raw_count": int(mapped.sum()),
            "identity_distance_m": quantiles(distance),
            "identity_normal_angle_deg": quantiles(normal_angle),
            "representative_mode": "weighted_geometric_medoid",
            "coverage_cell_count": int(torch.unique(coverage_ids).numel()),
            "redundancy_group_count": int(torch.unique(redundancy_ids).numel()),
        },
    )


@torch.no_grad()
def discrete_select_atoms(
    utility,
    coverage_cell_ids,
    redundancy_group_ids,
    budget,
    *,
    coverage_priority=None,
    previous_active=None,
    hysteresis=0.0,
    coverage_fraction=0.5,
    redundancy_penalty=0.1,
    return_diagnostics=False,
):
    """Exact-budget hard selection; coverage frequency is only a tie-break."""
    utility = torch.as_tensor(utility).reshape(-1)
    coverage_cell_ids = torch.as_tensor(
        coverage_cell_ids, device=utility.device, dtype=torch.long
    )
    redundancy_group_ids = torch.as_tensor(
        redundancy_group_ids, device=utility.device, dtype=torch.long
    )
    budget = min(max(int(budget), 1), utility.numel())
    score = utility.clone()
    if previous_active is not None and float(hysteresis) > 0:
        score[torch.as_tensor(previous_active, device=score.device, dtype=torch.long)] += float(hysteresis)

    redundancy_size = torch.bincount(redundancy_group_ids).float().clamp_min(1)
    score -= float(redundancy_penalty) * torch.log1p(
        redundancy_size[redundancy_group_ids]
    )
    selected = []
    reserve_count = min(
        budget,
        int(round(budget * max(0.0, min(1.0, float(coverage_fraction))))),
    )
    if reserve_count > 0:
        cell_count = int(coverage_cell_ids.max()) + 1 if coverage_cell_ids.numel() else 0
        representatives = group_representatives(score, coverage_cell_ids, cell_count)
        valid_cells = torch.nonzero(representatives < utility.numel(), as_tuple=False).reshape(-1)
        cell_score = score[representatives[valid_cells]]
        if coverage_priority is not None:
            priority = torch.as_tensor(
                coverage_priority, device=score.device, dtype=score.dtype
            )
            # Frequency breaks near-ties but cannot override learned utility.
            cell_score = cell_score + 1e-3 * torch.log1p(priority[valid_cells])
        take = min(reserve_count, int(valid_cells.numel()))
        chosen_cells = valid_cells[torch.topk(cell_score, take, sorted=False).indices]
        selected.append(representatives[chosen_cells])
    reserved = torch.cat(selected) if selected else torch.empty(0, device=score.device, dtype=torch.long)
    remainder_score = score.clone()
    remainder_score[reserved] = -torch.inf
    remainder = torch.topk(remainder_score, budget - reserved.numel(), sorted=False).indices
    result = torch.cat([reserved, remainder]).sort().values
    if not return_diagnostics:
        return result
    selected_score = score[result]
    unselected_mask = torch.ones(score.numel(), device=score.device, dtype=torch.bool)
    unselected_mask[result] = False
    best_unselected = score[unselected_mask].max() if bool(unselected_mask.any()) else selected_score.min()
    return result, {
        "coverage_reserved_count": int(reserved.numel()),
        "utility_fill_count": int(remainder.numel()),
        "coverage_reserved_fraction": float(reserved.numel() / max(budget, 1)),
        "selected_score_min": float(selected_score.min()),
        "selected_score_median": float(torch.quantile(selected_score, 0.5)),
        "best_unselected_score": float(best_unselected),
        "topk_cutoff_margin": float(selected_score.min() - best_unselected),
    }


@torch.no_grad()
def build_surface_groups(xyz, normals=None, voxel_size=0.05, normal_bins=0):
    """Build stable surface groups from quantized position and optional normal."""
    xyz = torch.as_tensor(xyz).float()
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    voxel_size = max(float(voxel_size), 1e-6)
    keys = [torch.floor(xyz / voxel_size).to(torch.int64)]
    if normals is not None and int(normal_bins) > 1:
        normals = F.normalize(torch.as_tensor(normals, device=xyz.device).float(), dim=1)
        bins = torch.floor((normals + 1.0) * (int(normal_bins) * 0.5)).long()
        bins = bins.clamp(0, int(normal_bins) - 1)
        keys.append(bins)
    key = torch.cat(keys, dim=1)
    _, inverse = torch.unique(key, dim=0, sorted=True, return_inverse=True)
    return inverse.long(), int(inverse.max().item()) + 1 if inverse.numel() else 0


@torch.no_grad()
def group_representatives(scores, group_ids, group_count=None):
    scores = torch.as_tensor(scores).reshape(-1)
    group_ids = torch.as_tensor(group_ids, device=scores.device, dtype=torch.long).reshape(-1)
    if scores.numel() != group_ids.numel():
        raise ValueError("scores and group_ids must have equal length")
    if group_count is None:
        group_count = int(group_ids.max().item()) + 1 if group_ids.numel() else 0
    maximum = scores.new_full((group_count,), -torch.inf)
    maximum.scatter_reduce_(0, group_ids, scores, reduce="amax", include_self=True)
    indices = torch.arange(scores.numel(), device=scores.device, dtype=torch.long)
    sentinel = scores.numel()
    candidates = torch.where(scores == maximum[group_ids], indices, sentinel)
    representatives = torch.full(
        (group_count,), sentinel, device=scores.device, dtype=torch.long
    )
    representatives.scatter_reduce_(
        0, group_ids, candidates, reduce="amin", include_self=True
    )
    return representatives


@torch.no_grad()
def project_active_set(
    gate_logits,
    budget,
    previous_active=None,
    hysteresis=0.0,
    *,
    group_ids=None,
    group_priority=None,
):
    scores = torch.as_tensor(gate_logits).reshape(-1)
    budget = min(max(int(budget), 1), scores.numel())
    projected = scores.clone()
    if previous_active is not None and float(hysteresis) > 0.0:
        previous_active = torch.as_tensor(
            previous_active, device=scores.device, dtype=torch.long
        ).reshape(-1)
        projected[previous_active] += float(hysteresis)
    if group_ids is None or group_priority is None:
        return torch.topk(projected, budget, sorted=False).indices.sort().values

    group_ids = torch.as_tensor(group_ids, device=scores.device, dtype=torch.long)
    group_priority = torch.as_tensor(
        group_priority, device=scores.device, dtype=scores.dtype
    ).reshape(-1)
    eligible_groups = torch.nonzero(group_priority > 0, as_tuple=False).reshape(-1)
    if eligible_groups.numel() == 0:
        return torch.topk(projected, budget, sorted=False).indices.sort().values

    representatives = group_representatives(projected, group_ids, group_priority.numel())
    eligible_groups = eligible_groups[representatives[eligible_groups] < scores.numel()]
    reserve_count = min(budget, int(eligible_groups.numel()))
    if eligible_groups.numel() > reserve_count:
        priority = group_priority[eligible_groups]
        chosen_groups = eligible_groups[
            torch.topk(priority, reserve_count, sorted=False).indices
        ]
    else:
        chosen_groups = eligible_groups
    reserved = representatives[chosen_groups]
    if reserved.numel() == budget:
        return reserved.sort().values

    projected[reserved] = -torch.inf
    remaining = torch.topk(
        projected, budget - reserved.numel(), sorted=False
    ).indices
    return torch.cat([reserved, remaining]).sort().values


@torch.no_grad()
def active_group_representatives(scores, group_ids, active_indices, group_count=None):
    """Return a global primitive index per represented group and -1 otherwise."""
    scores = torch.as_tensor(scores).reshape(-1)
    group_ids = torch.as_tensor(group_ids, device=scores.device, dtype=torch.long)
    active_indices = torch.as_tensor(
        active_indices, device=scores.device, dtype=torch.long
    ).reshape(-1)
    if group_count is None:
        group_count = int(group_ids.max().item()) + 1 if group_ids.numel() else 0
    local = group_representatives(
        scores[active_indices], group_ids[active_indices], group_count
    )
    valid = local < active_indices.numel()
    result = torch.full((group_count,), -1, device=scores.device, dtype=torch.long)
    result[valid] = active_indices[local[valid]]
    return result


def progressive_coreset_regularizers(
    gate_logits,
    group_ids,
    observed_groups,
    target_budget,
    *,
    observed_group_weights=None,
    redundancy_multiplier=2.0,
):
    probability = torch.sigmoid(gate_logits.reshape(-1))
    group_ids = group_ids.to(device=probability.device, dtype=torch.long)
    group_count = int(group_ids.max().item()) + 1 if group_ids.numel() else 0
    group_sum = probability.new_zeros(group_count)
    group_sum.scatter_add_(0, group_ids, probability)
    observed_groups = torch.as_tensor(
        observed_groups, device=probability.device, dtype=torch.long
    ).reshape(-1)
    valid_observed = (observed_groups >= 0) & (observed_groups < group_count)
    observed_groups = observed_groups[valid_observed]
    if observed_group_weights is None:
        observed_group_weights = probability.new_ones(observed_groups.shape)
    else:
        observed_group_weights = torch.as_tensor(
            observed_group_weights, device=probability.device, dtype=probability.dtype
        ).reshape(-1)[valid_observed]
    if observed_groups.numel():
        # Probability that at least one member of a surface group remains active.
        log_not_active = torch.log1p(-probability.clamp(max=1.0 - 1e-6))
        group_log_not_active = probability.new_zeros(group_count)
        group_log_not_active.scatter_add_(0, group_ids, log_not_active)
        coverage_probability = -torch.expm1(group_log_not_active[observed_groups])
        coverage_terms = -torch.log(coverage_probability.clamp_min(1e-6))
        coverage = (coverage_terms * observed_group_weights).sum() / observed_group_weights.sum().clamp_min(1e-6)
    else:
        coverage = probability.sum() * 0.0
    target = probability.new_tensor(float(target_budget))
    budget = ((probability.sum() - target) / target.clamp_min(1.0)).square()
    capacity = max(
        1.0,
        float(target_budget) / max(group_count, 1) * float(redundancy_multiplier),
    )
    redundancy = F.relu(group_sum - capacity).square().mean() if group_count else budget * 0.0
    return {
        "coverage": coverage,
        "budget": budget,
        "redundancy": redundancy,
        "probability_sum": probability.sum().detach(),
    }


def coreset_matching_loss(
    query_descriptors,
    positive_descriptors,
    negative_descriptors,
    positive_gate_logits,
    negative_gate_logits,
    negative_mask,
    *,
    temperature=0.07,
):
    query = F.normalize(query_descriptors.float(), dim=1)
    positive = F.normalize(positive_descriptors.float(), dim=1)
    negative = F.normalize(negative_descriptors.float(), dim=2)
    positive_score = (query * positive).sum(dim=1)
    negative_score = (query[:, None] * negative).sum(dim=2)
    positive_logit = positive_score / max(float(temperature), 1e-6)
    negative_logit = negative_score / max(float(temperature), 1e-6)
    positive_logit = positive_logit + F.logsigmoid(positive_gate_logits)
    negative_logit = negative_logit + F.logsigmoid(negative_gate_logits)
    negative_logit = negative_logit.masked_fill(~negative_mask, -torch.inf)
    logits = torch.cat([positive_logit[:, None], negative_logit], dim=1)
    target = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
    valid = negative_mask.any(dim=1) & torch.isfinite(logits[:, 0])
    if not bool(valid.any().item()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], target[valid])


def coreset_soft_matching_loss(
    query_descriptors,
    positive_descriptors,
    positive_weights,
    positive_gate_logits,
    positive_mask,
    negative_descriptors,
    negative_gate_logits,
    negative_mask,
    *,
    temperature=0.07,
):
    """Multi-positive InfoNCE with soft splat provenance mass."""
    query = F.normalize(query_descriptors.float(), dim=1)
    positive = F.normalize(positive_descriptors.float(), dim=2)
    negative = F.normalize(negative_descriptors.float(), dim=2)
    positive_logits = (query[:, None] * positive).sum(dim=2) / max(
        float(temperature), 1e-6
    )
    positive_logits = positive_logits + F.logsigmoid(positive_gate_logits)
    positive_logits = positive_logits + torch.log(positive_weights.clamp_min(1e-8))
    positive_logits = positive_logits.masked_fill(~positive_mask, -torch.inf)
    negative_logits = (query[:, None] * negative).sum(dim=2) / max(
        float(temperature), 1e-6
    )
    negative_logits = negative_logits + F.logsigmoid(negative_gate_logits)
    negative_logits = negative_logits.masked_fill(~negative_mask, -torch.inf)
    numerator = torch.logsumexp(positive_logits, dim=1)
    denominator = torch.logsumexp(
        torch.cat([positive_logits, negative_logits], dim=1), dim=1
    )
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not bool(valid.any().item()):
        return (positive_logits.sum() + negative_logits.sum()) * 0.0
    return (denominator[valid] - numerator[valid]).mean()


def deployment_soft_matching_loss(
    query_descriptors,
    positive_descriptors,
    positive_weights,
    positive_mask,
    negative_descriptors,
    negative_mask,
    *,
    temperature=0.07,
):
    """Soft-positive matching with exactly the cosine score used at inference."""
    query = F.normalize(query_descriptors.float(), dim=1)
    positive = F.normalize(positive_descriptors.float(), dim=2)
    negative = F.normalize(negative_descriptors.float(), dim=2)
    scale = max(float(temperature), 1e-6)
    positive_logits = (query[:, None] * positive).sum(dim=2) / scale
    positive_logits = positive_logits + torch.log(positive_weights.clamp_min(1e-8))
    positive_logits = positive_logits.masked_fill(~positive_mask, -torch.inf)
    negative_logits = (query[:, None] * negative).sum(dim=2) / scale
    negative_logits = negative_logits.masked_fill(~negative_mask, -torch.inf)
    numerator = torch.logsumexp(positive_logits, dim=1)
    denominator = torch.logsumexp(
        torch.cat([positive_logits, negative_logits], dim=1), dim=1
    )
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not bool(valid.any()):
        return (positive_logits.sum() + negative_logits.sum()) * 0.0
    return (denominator[valid] - numerator[valid]).mean()


def descriptor_trust_loss(current, initial, weights=None):
    current = F.normalize(current.float(), dim=-1)
    initial = F.normalize(initial.float(), dim=-1)
    loss = 1.0 - (current * initial).sum(dim=-1)
    if weights is None:
        return loss.mean() if loss.numel() else current.sum() * 0.0
    weights = torch.as_tensor(weights, device=loss.device, dtype=loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-6)


def active_set_diagnostics(active_indices, group_ids, previous_active=None):
    active_indices = torch.as_tensor(active_indices, dtype=torch.long)
    group_ids = torch.as_tensor(group_ids, dtype=torch.long, device=active_indices.device)
    active_groups = torch.unique(group_ids[active_indices])
    result = {
        "active_count": int(active_indices.numel()),
        "active_group_count": int(active_groups.numel()),
        "group_coverage": float(
            active_groups.numel() / max(int(torch.unique(group_ids).numel()), 1)
        ),
    }
    if previous_active is not None:
        previous = torch.as_tensor(previous_active, device=active_indices.device, dtype=torch.long)
        intersection = torch.isin(active_indices, previous).sum()
        union = active_indices.numel() + previous.numel() - intersection
        result["active_jaccard"] = float(intersection / max(int(union), 1))
        result["active_churn"] = float(1.0 - intersection / max(int(active_indices.numel()), 1))
    return result
