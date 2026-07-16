from dataclasses import dataclass
import math

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
