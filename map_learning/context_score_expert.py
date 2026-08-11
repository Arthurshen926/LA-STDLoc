"""Independent context score expert layered on a frozen A1 metric.

The local A1 score is never rewritten.  A compact single-image context code
contributes an additive score whose map-side magnitude is the cross-view code
concentration.  The sum is exactly representable by one normalized descriptor
bank through one orthogonal compensation coordinate.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from map_learning.alias_teacher import protected_clean_margin_loss
from map_learning.context_booster_crossfit import (
    DEFAULT_TOPKS,
    _empty_retrieval,
    _update_descriptor_counts,
    accumulate_view_descriptors,
    summarize_retrieval,
)
from map_learning.context_metric import (
    CONTEXT_MODES,
    DEFAULT_CONTEXT_KERNELS,
    context_from_cached_query,
)
from map_learning.metric import SharedLowRankMetric
from map_learning.repeated_assignment_audit import (
    _pose_summary,
    _selected_csr_edges,
    _solve_assignments,
)
from map_learning.trainer import _multi_positive_list_loss


class ContextScoreExpert(nn.Module):
    """Produce an independent unit context code from frozen image evidence."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 256,
        code_dim: int = 32,
        hidden_dim: int = 256,
        context_kernels: Sequence[int] = DEFAULT_CONTEXT_KERNELS,
        context_mode: str = "local_only",
        input_scope: str = "base_and_tokens",
        learned_query_gate: bool = False,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.code_dim = int(code_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_kernels = tuple(int(value) for value in context_kernels)
        self.context_mode = str(context_mode)
        self.input_scope = str(input_scope)
        self.learned_query_gate = bool(learned_query_gate)
        if min(self.descriptor_dim, self.code_dim, self.hidden_dim) < 1:
            raise ValueError("descriptor, code, and hidden dimensions must be positive")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"unsupported context mode: {self.context_mode}")
        if self.input_scope not in ("base_and_tokens", "shared_global"):
            raise ValueError(f"unsupported context input scope: {self.input_scope}")
        if self.learned_query_gate and self.input_scope != "shared_global":
            raise ValueError("the learned query gate must be image-shared")
        if self.input_scope == "shared_global" and self.context_mode in (
            "local_only",
            "zero",
        ):
            raise ValueError("shared-global input requires a non-zero global token")
        token_count = len(self.context_kernels) + 1
        input_dim = (
            self.descriptor_dim
            if self.input_scope == "shared_global"
            else self.descriptor_dim * (token_count + 1)
        )
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.context_head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.code_dim),
        )
        if self.learned_query_gate:
            self.query_gate_head = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
            nn.init.zeros_(self.query_gate_head[-1].weight)
            nn.init.zeros_(self.query_gate_head[-1].bias)
        else:
            self.query_gate_head = None

    def _inputs(
        self,
        base_descriptors: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate inputs and return normalized base plus head inputs."""
        base = F.normalize(torch.as_tensor(base_descriptors).float(), dim=1)
        context = torch.as_tensor(
            context_tokens, device=base.device, dtype=base.dtype
        )
        expected = (base.shape[0], len(self.context_kernels) + 1)
        if base.ndim != 2 or base.shape[1] != self.descriptor_dim:
            raise ValueError("base descriptors must have shape [N, D]")
        if context.ndim != 3 or context.shape[:2] != expected:
            raise ValueError(
                "context tokens must have shape "
                f"[N, {len(self.context_kernels) + 1}, D]"
            )
        if context.shape[2] != self.descriptor_dim:
            raise ValueError("context token dimension must match descriptor")
        if self.input_scope == "shared_global":
            inputs = context[:, -1]
        else:
            inputs = torch.cat((base, context.flatten(start_dim=1)), dim=1)
        return base, inputs

    def forward(
        self,
        base_descriptors: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> torch.Tensor:
        _, inputs = self._inputs(base_descriptors, context_tokens)
        return F.normalize(self.context_head(self.input_norm(inputs)), dim=1)

    def query(
        self,
        base_descriptors: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a query code scaled by one image-shared confidence gate."""
        _, inputs = self._inputs(base_descriptors, context_tokens)
        normalized_inputs = self.input_norm(inputs)
        codes = F.normalize(self.context_head(normalized_inputs), dim=1)
        if self.query_gate_head is None:
            gates = codes.new_ones((codes.shape[0], 1))
        else:
            gates = torch.sigmoid(self.query_gate_head(normalized_inputs))
        return codes * gates, gates

    def export_config(self) -> dict:
        return {
            "descriptor_dim": self.descriptor_dim,
            "code_dim": self.code_dim,
            "hidden_dim": self.hidden_dim,
            "context_kernels": list(self.context_kernels),
            "context_mode": self.context_mode,
            "input_scope": self.input_scope,
            "learned_query_gate": self.learned_query_gate,
            "output_dim": self.code_dim,
            "role": "independent_additive_context_score",
            "map_gate": "cross_view_unit_code_concentration",
            "query_gate": (
                "learned_image_shared_sigmoid"
                if self.learned_query_gate
                else "constant_one"
            ),
        }


def protocol_name(weight: float) -> str:
    """Return a stable JSON/checkpoint name for a context score weight."""
    value = format(float(weight), ".8g").replace("-", "m").replace(".", "p")
    return f"context_lambda_{value}"


def concatenate_dual_expert_descriptors(
    base_descriptors: torch.Tensor,
    context_codes: torch.Tensor,
    *,
    context_weight: float,
    map_side: bool,
) -> torch.Tensor:
    """Encode ``base + lambda * context`` as one normalized descriptor.

    Query and map context codes may have norm below one, representing their
    confidence gates.  Two mutually orthogonal compensation coordinates make
    both sides constant-norm without contributing to their dot product.
    """
    weight = float(context_weight)
    if weight < 0.0:
        raise ValueError("context weight must be non-negative")
    base = torch.as_tensor(base_descriptors).float()
    context = torch.as_tensor(context_codes, device=base.device, dtype=base.dtype)
    if base.ndim != 2 or context.ndim != 2 or base.shape[0] != context.shape[0]:
        raise ValueError("base descriptors and context codes must align by row")
    if weight == 0.0:
        return base
    base = F.normalize(base, dim=1)
    norms = torch.linalg.norm(context, dim=1, keepdim=True)
    tolerance = 2e-5
    if bool((norms > 1.0 + tolerance).any()):
        raise ValueError("context code norm exceeds one")
    if not map_side and torch.allclose(
        norms, torch.ones_like(norms), atol=tolerance, rtol=0.0
    ):
        # Exact unit query codes need no compensation. Avoid differentiating
        # sqrt(1 - ||c||^2) at its singular zero endpoint.
        complement = torch.zeros_like(norms)
    else:
        complement = torch.sqrt((1.0 - norms.square()).clamp_min(1e-12))
    zero = torch.zeros_like(complement)
    query_complement, map_complement = (
        (zero, complement) if map_side else (complement, zero)
    )
    scale = math.sqrt(weight)
    joint = torch.cat(
        (
            base,
            scale * context,
            scale * query_complement,
            scale * map_complement,
        ),
        dim=1,
    )
    return joint / math.sqrt(1.0 + weight)


@torch.inference_mode()
def build_context_score_bank(
    *,
    expert: ContextScoreExpert,
    metric: SharedLowRankMetric,
    teacher: dict,
    query_cache: dict,
    support_query_indices: Sequence[int],
    anchor_indices: torch.Tensor,
    expected_view_counts: torch.Tensor,
    device: torch.device,
    progress_interval: int = 0,
    return_observation_state: bool = False,
) -> tuple[torch.Tensor, dict] | tuple[torch.Tensor, dict, dict]:
    """Fuse one unit context code per observing image without renormalizing."""
    names = list(teacher["query_names"])
    cache = query_cache.get("queries", query_cache)
    support = [int(value) for value in support_query_indices]
    anchor_count = int(teacher["anchor_count"])
    code_sum = torch.zeros((anchor_count, expert.code_dim), device=device)
    view_counts = torch.zeros(anchor_count, dtype=torch.long, device=device)
    query_codes = {}
    observation_count = 0
    for completed, query_index in enumerate(support, start=1):
        record = teacher["records"][query_index]
        rows = torch.as_tensor(record["query_rows"]).long()
        _, edge_rows, edge_anchors = _selected_csr_edges(
            record, "positive", torch.arange(rows.numel())
        )
        if edge_rows.numel():
            raw, tokens = context_from_cached_query(
                cache[names[query_index]],
                rows[edge_rows],
                device=device,
                kernels=expert.context_kernels,
                context_mode=expert.context_mode,
            )
            base, _ = metric(raw)
            codes = expert(base, tokens)
            if return_observation_state:
                if expert.input_scope != "shared_global":
                    raise ValueError(
                        "LOO observation state requires shared-global context"
                    )
                query_codes[query_index] = codes[:1].detach().clone()
            accumulate_view_descriptors(
                code_sum,
                view_counts,
                edge_anchors.to(device),
                codes,
            )
            observation_count += int(codes.shape[0])
        if progress_interval > 0 and (
            completed % int(progress_interval) == 0 or completed == len(support)
        ):
            print(
                {
                    "event": "context_score_bank",
                    "queries_complete": completed,
                    "query_count": len(support),
                },
                flush=True,
            )
    selected = torch.as_tensor(anchor_indices, device=device).long()
    if not torch.equal(view_counts[selected], expected_view_counts.to(device)):
        raise AssertionError("context and raw support view counts diverged")
    counts = view_counts[selected].float().clamp_min(1.0)
    bank = code_sum[selected] / counts[:, None]
    concentration = bank.norm(dim=1).clamp(0.0, 1.0)
    values = concentration.cpu().numpy().astype(np.float64)
    report = {
        "adapted_observation_count": int(observation_count),
        "map_concentration_mean": float(values.mean()) if values.size else 0.0,
        "map_concentration_p10": (
            float(np.percentile(values, 10)) if values.size else 0.0
        ),
        "map_concentration_median": (
            float(np.median(values)) if values.size else 0.0
        ),
        "map_concentration_p90": (
            float(np.percentile(values, 90)) if values.size else 0.0
        ),
        "map_concentration_maximum": float(values.max()) if values.size else 0.0,
    }
    if return_observation_state:
        return bank, report, {
            "view_counts": view_counts[selected].detach().clone(),
            "query_codes": query_codes,
        }
    return bank, report


def leave_one_query_out_anchor_codes(
    context_bank: torch.Tensor,
    view_counts: torch.Tensor,
    anchor_indices: torch.Tensor,
    query_code: torch.Tensor,
) -> torch.Tensor:
    """Remove one image's exact contribution from selected anchor prototypes."""
    indices = torch.as_tensor(anchor_indices, device=context_bank.device).long()
    counts = torch.as_tensor(
        view_counts, device=context_bank.device, dtype=context_bank.dtype
    )[indices]
    if bool((counts <= 1.0).any()):
        raise ValueError("LOO context supervision requires at least two views")
    code = torch.as_tensor(
        query_code, device=context_bank.device, dtype=context_bank.dtype
    ).reshape(1, -1)
    return (context_bank[indices] * counts[:, None] - code) / (
        counts[:, None] - 1.0
    )


def balanced_unique_anchor_prior_loss(
    query_code: torch.Tensor,
    positive_codes: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_codes: torch.Tensor,
    negative_indices: torch.Tensor,
    anchor_type: torch.Tensor,
    family_ids: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """Train one image prior with unique, type- and family-balanced anchors."""
    if float(temperature) <= 0.0:
        raise ValueError("anchor-prior temperature must be positive")
    code = torch.as_tensor(query_code).reshape(1, -1)
    positive_indices = torch.as_tensor(
        positive_indices, device=code.device
    ).long().reshape(-1)
    negative_indices = torch.as_tensor(
        negative_indices, device=code.device
    ).long().reshape(-1)
    types = torch.as_tensor(anchor_type, device=code.device).long()
    families = torch.as_tensor(family_ids, device=code.device).long()

    def balanced_mean(
        values: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, int]:
        category_means = []
        family_count = 0
        for is_reserve in (False, True):
            selected = (types[indices] == 0) == is_reserve
            if not bool(selected.any()):
                continue
            selected_values = values[selected]
            selected_families = families[indices[selected]]
            unique, inverse = torch.unique(
                selected_families, sorted=True, return_inverse=True
            )
            sums = selected_values.new_zeros(unique.numel())
            counts = selected_values.new_zeros(unique.numel())
            sums.index_add_(0, inverse, selected_values)
            counts.index_add_(0, inverse, torch.ones_like(selected_values))
            category_means.append((sums / counts.clamp_min(1.0)).mean())
            family_count += int(unique.numel())
        if not category_means:
            return code.sum() * 0.0, 0
        return torch.stack(category_means).mean(), family_count

    positive_scores = (code @ positive_codes.T)[0] / float(temperature)
    negative_scores = (code @ negative_codes.T)[0] / float(temperature)
    positive_loss, positive_families = balanced_mean(
        F.softplus(-positive_scores), positive_indices
    )
    negative_loss, negative_families = balanced_mean(
        F.softplus(negative_scores), negative_indices
    )
    return positive_loss + negative_loss, {
        "unique_positive_anchor_count": int(positive_indices.numel()),
        "unique_negative_anchor_count": int(negative_indices.numel()),
        "positive_family_count": positive_families,
        "negative_family_count": negative_families,
        "positive_loss": positive_loss.detach(),
        "negative_loss": negative_loss.detach(),
    }


@torch.no_grad()
def select_incompatible_anchor_negatives(
    query_code: torch.Tensor,
    context_bank: torch.Tensor,
    anchor_xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    input_hw: Sequence[int] | torch.Tensor,
    excluded_indices: torch.Tensor,
    *,
    maximum_count: int = 256,
    minimum_depth: float = 1e-3,
) -> torch.Tensor:
    """Select context-hard anchors that are provably outside the query camera."""
    if int(maximum_count) < 1 or float(minimum_depth) <= 0.0:
        raise ValueError("incompatible-negative count and depth must be positive")
    device = context_bank.device
    xyz = torch.as_tensor(anchor_xyz, device=device, dtype=context_bank.dtype)
    pose = torch.as_tensor(pose_w2c, device=device, dtype=context_bank.dtype)
    camera_k = torch.as_tensor(intrinsic, device=device, dtype=context_bank.dtype)
    height, width = (int(value) for value in torch.as_tensor(input_hw).tolist())
    if xyz.shape != (context_bank.shape[0], 3):
        raise ValueError("anchor xyz must align with the context bank")
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = camera @ camera_k.T
    safe_depth = depth.clamp_min(float(minimum_depth))
    u = projected[:, 0] / safe_depth
    v = projected[:, 1] / safe_depth
    visible = (
        torch.isfinite(camera).all(dim=1)
        & (depth > float(minimum_depth))
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )
    candidates = ~visible
    excluded = torch.as_tensor(excluded_indices, device=device).long()
    candidates[excluded] = False
    indices = torch.nonzero(candidates, as_tuple=False).reshape(-1)
    if not indices.numel():
        return indices
    scores = torch.as_tensor(query_code, device=device).reshape(1, -1) @ (
        context_bank[indices].T
    )
    count = min(int(maximum_count), int(indices.numel()))
    return indices[torch.topk(scores[0], k=count).indices]


def _legal_top1_log_probability(
    query: torch.Tensor,
    bank: torch.Tensor,
    positives: torch.Tensor,
    ignored: torch.Tensor,
    *,
    topk: int = 64,
    temperature: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return legal top-1 log probabilities and winning positive indices."""
    scores = query @ bank.T
    top_scores, top_indices = torch.topk(
        scores, k=min(int(topk), bank.shape[0]), dim=1
    )
    positive_valid = positives >= 0
    positive_scores = torch.einsum(
        "bd,bpd->bp", query, bank[positives.clamp_min(0)]
    )
    top_is_positive = (
        (top_indices[:, :, None] == positives[:, None, :])
        & positive_valid[:, None, :]
    ).any(dim=2)
    ignored_valid = ignored >= 0
    top_is_ignored = (
        (top_indices[:, :, None] == ignored[:, None, :])
        & ignored_valid[:, None, :]
    ).any(dim=2)
    valid_top = ~top_is_positive & ~top_is_ignored
    # A binary top-1 probability against the hardest legal negative is used
    # instead of a masked full softmax. It is the event needed by one-winner
    # matching and avoids undefined gradients from all-masked softmax slices.
    finite_floor = -1e4
    best_positive, best_positive_slot = positive_scores.masked_fill(
        ~positive_valid, finite_floor
    ).max(dim=1)
    best_positive_indices = positives.gather(
        1, best_positive_slot[:, None]
    )[:, 0]
    hardest_negative = top_scores.masked_fill(
        ~valid_top, finite_floor
    ).max(dim=1).values
    log_legal_mass = F.logsigmoid(
        (best_positive - hardest_negative) / float(temperature)
    )
    return log_legal_mass, best_positive_indices, positive_valid.any(dim=1)


def expected_clean_inlier_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    positives: torch.Tensor,
    ignored: torch.Tensor,
    *,
    topk: int = 64,
    temperature: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query-level risk and per-row legal probability mass.

    Complete-positive teacher edges are the candidates already certified by
    the mapping GT pose, depth, and visibility gates. Ambiguous edges are
    removed from the denominator rather than trained as negatives.
    """
    log_legal_mass, _, _ = _legal_top1_log_probability(
        query,
        bank,
        positives,
        ignored,
        topk=topk,
        temperature=temperature,
    )
    log_query_mass = torch.logsumexp(log_legal_mass, dim=0) - math.log(
        max(log_legal_mass.numel(), 1)
    )
    risk = -log_query_mass
    legal_mass = torch.exp(log_legal_mass.detach().clamp_min(-80.0))
    return risk, legal_mass


def expected_clean_pose_information_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    positives: torch.Tensor,
    ignored: torch.Tensor,
    anchor_xyz: torch.Tensor,
    pose_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    *,
    topk: int = 64,
    temperature: float = 0.04,
    damping: float = 1e-3,
    minimum_depth: float = 1e-3,
) -> tuple[torch.Tensor, dict]:
    """Penalize loss of clean six-DoF PnP information.

    Every legal row contributes its pixel reprojection Jacobian at the mapping
    GT pose, weighted by the probability that a legal anchor wins top-1.  The
    expected 6x6 Fisher matrix is whitened by the all-legal reference matrix,
    so translation/rotation units and focal scale cannot dominate the loss.
    The log-determinant gap is monotone: suppressing any legal probability can
    never improve the objective merely by changing the denominator.
    """
    if float(damping) <= 0.0 or float(minimum_depth) <= 0.0:
        raise ValueError("information damping and minimum depth must be positive")
    xyz = torch.as_tensor(anchor_xyz, device=query.device, dtype=query.dtype)
    pose = torch.as_tensor(pose_w2c, device=query.device, dtype=query.dtype)
    camera_k = torch.as_tensor(intrinsic, device=query.device, dtype=query.dtype)
    if xyz.ndim != 2 or xyz.shape != (bank.shape[0], 3):
        raise ValueError("anchor xyz must align with the descriptor bank")
    if pose.shape != (4, 4) or camera_k.shape != (3, 3):
        raise ValueError("pose and intrinsic must have shapes [4,4] and [3,3]")

    log_probability, best_positive, positive_valid = _legal_top1_log_probability(
        query,
        bank,
        positives,
        ignored,
        topk=topk,
        temperature=temperature,
    )
    safe_positive = best_positive.clamp_min(0)
    world = xyz[safe_positive]
    camera = world @ pose[:3, :3].T + pose[:3, 3]
    x, y, z = camera.unbind(dim=1)
    geometry_valid = (
        positive_valid
        & torch.isfinite(camera).all(dim=1)
        & (z > float(minimum_depth))
    )
    valid_count = int(geometry_valid.sum())
    if valid_count < 4:
        zero = query.sum() * 0.0
        return zero, {
            "valid_row_count": valid_count,
            "information_retention": zero.detach(),
            "reference_min_eigenvalue": zero.detach(),
            "expected_min_eigenvalue": zero.detach(),
        }

    inverse_z = z.clamp_min(float(minimum_depth)).reciprocal()
    projection = query.new_zeros((query.shape[0], 2, 3))
    projection[:, 0, 0] = camera_k[0, 0] * inverse_z
    projection[:, 0, 2] = -camera_k[0, 0] * x * inverse_z.square()
    projection[:, 1, 1] = camera_k[1, 1] * inverse_z
    projection[:, 1, 2] = -camera_k[1, 1] * y * inverse_z.square()

    skew = query.new_zeros((query.shape[0], 3, 3))
    skew[:, 0, 1] = -z
    skew[:, 0, 2] = y
    skew[:, 1, 0] = z
    skew[:, 1, 2] = -x
    skew[:, 2, 0] = -y
    skew[:, 2, 1] = x
    translation = torch.eye(3, device=query.device, dtype=query.dtype)[None]
    motion = torch.cat((translation.expand(query.shape[0], -1, -1), -skew), dim=2)
    jacobian = projection @ motion
    row_information = jacobian.transpose(1, 2) @ jacobian
    valid_weight = geometry_valid.to(query.dtype)
    denominator = valid_weight.sum().clamp_min(1.0)
    reference = (
        row_information * valid_weight[:, None, None]
    ).sum(dim=0) / denominator
    clean_probability = torch.exp(log_probability).clamp(0.0, 1.0)
    expected = (
        row_information
        * (valid_weight * clean_probability)[:, None, None]
    ).sum(dim=0) / denominator

    diagonal_scale = torch.rsqrt(reference.diagonal().clamp_min(1e-8))
    whitener = torch.diag(diagonal_scale)
    reference = whitener @ reference @ whitener
    expected = whitener @ expected @ whitener
    reference = 0.5 * (reference + reference.T)
    expected = 0.5 * (expected + expected.T)
    identity = torch.eye(6, device=query.device, dtype=query.dtype)
    reference_regularized = reference + float(damping) * identity
    expected_regularized = expected + float(damping) * identity
    reference_logdet = torch.linalg.slogdet(reference_regularized).logabsdet
    expected_logdet = torch.linalg.slogdet(expected_regularized).logabsdet
    risk = (reference_logdet - expected_logdet) / 6.0
    with torch.no_grad():
        reference_minimum = torch.linalg.eigvalsh(reference_regularized).min()
        expected_minimum = torch.linalg.eigvalsh(expected_regularized).min()
        retention = torch.exp(-risk.detach()).clamp(0.0, 1.0)
    return risk, {
        "valid_row_count": valid_count,
        "information_retention": retention,
        "reference_min_eigenvalue": reference_minimum,
        "expected_min_eigenvalue": expected_minimum,
    }


def train_context_score_stage(
    *,
    expert: ContextScoreExpert,
    metric: SharedLowRankMetric,
    teacher: dict,
    query_cache: dict,
    support_query_indices: Sequence[int],
    records: dict[int, dict],
    base_reference_bank: torch.Tensor,
    context_task_bank: torch.Tensor,
    anchor_xyz: torch.Tensor | None,
    context_view_counts: torch.Tensor | None = None,
    context_query_codes: dict[int, torch.Tensor] | None = None,
    anchor_type: torch.Tensor | None = None,
    anchor_family_ids: torch.Tensor | None = None,
    device: torch.device,
    context_weight: float = 0.05,
    epochs: int = 1,
    batch_size: int = 256,
    repair_topk: int = 32,
    training_topk: int = 64,
    learning_rate: float = 5e-4,
    temperature: float = 0.04,
    collision_weight: float = 1.0,
    clean_weight: float = 2.0,
    clean_margin_slack: float = 0.01,
    clean_task_scale: float = 0.25,
    gate_supervision_weight: float = 1.0,
    consensus_weight: float = 0.0,
    query_tail_weight: float = 0.0,
    query_tail_alpha: float = 0.75,
    query_batch_size: int = 1,
    observability_weight: float = 0.0,
    observability_tail_weight: float = 0.0,
    observability_damping: float = 1e-3,
    training_objective: str = "rowwise",
    prior_weight: float = 0.1,
    prior_temperature: float = 0.1,
    prior_incompatible_negatives: int = 256,
    seed: int = 2026,
    stage_name: str = "context_score_target",
    progress_interval: int = 100,
) -> dict:
    """Train the exact deployable dual score on recoverable A1 errors."""
    if epochs < 1 or batch_size < 1 or repair_topk < 2 or query_batch_size < 1:
        raise ValueError("epochs/batch must be positive and repair top-K >= 2")
    if float(context_weight) <= 0.0:
        raise ValueError("training context weight must be positive")
    if min(
        float(consensus_weight),
        float(query_tail_weight),
        float(observability_weight),
        float(observability_tail_weight),
    ) < 0.0:
        raise ValueError("query-level objective weights must be non-negative")
    if not 0.0 <= float(query_tail_alpha) < 1.0:
        raise ValueError("query-tail alpha must lie in [0, 1)")
    if float(observability_damping) <= 0.0:
        raise ValueError("observability damping must be positive")
    if training_objective not in ("rowwise", "anchor_prior_loo"):
        raise ValueError(f"unsupported context training objective: {training_objective}")
    if min(float(prior_weight), float(prior_temperature)) <= 0.0:
        raise ValueError("anchor-prior weight and temperature must be positive")
    if int(prior_incompatible_negatives) < 1:
        raise ValueError("incompatible-negative count must be positive")
    prior_enabled = training_objective == "anchor_prior_loo"
    if prior_enabled and (
        expert.input_scope != "shared_global"
        or expert.learned_query_gate
        or context_view_counts is None
        or context_query_codes is None
        or anchor_xyz is None
        or anchor_type is None
        or anchor_family_ids is None
    ):
        raise ValueError(
            "LOO anchor-prior training requires ungated shared-global codes, "
            "observation state, and anchor metadata"
        )
    observability_enabled = bool(
        float(observability_weight) > 0.0
        or float(observability_tail_weight) > 0.0
    )
    if observability_enabled and anchor_xyz is None:
        raise ValueError("anchor xyz is required by the observability objective")
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    support = [int(value) for value in support_query_indices]
    reference = base_reference_bank.detach().to(device).clone()
    context_bank = context_task_bank.detach().to(device).clone()
    geometry_bank = (
        None
        if anchor_xyz is None
        else torch.as_tensor(anchor_xyz, device=device).float().clone()
    )
    view_counts = (
        None
        if context_view_counts is None
        else torch.as_tensor(context_view_counts, device=device).float().clone()
    )
    type_bank = (
        None
        if anchor_type is None
        else torch.as_tensor(anchor_type, device=device).long().clone()
    )
    family_bank = (
        None
        if anchor_family_ids is None
        else torch.as_tensor(anchor_family_ids, device=device).long().clone()
    )
    joint_bank = concatenate_dual_expert_descriptors(
        reference,
        context_bank,
        context_weight=float(context_weight),
        map_side=True,
    ).detach()
    optimizer = torch.optim.AdamW(
        expert.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(int(seed))
    history = []
    global_step = 0
    for epoch in range(int(epochs)):
        order = torch.randperm(len(support), generator=generator).tolist()
        totals = {
            "loss": 0.0,
            "list": 0.0,
            "clean_loss": 0.0,
            "gate_loss": 0.0,
            "consensus_loss": 0.0,
            "legal_mass": 0.0,
            "optimizer_objective": 0.0,
            "tail_objective": 0.0,
            "observability_loss": 0.0,
            "observability_retention": 0.0,
            "observability_min_eigenvalue": 0.0,
            "observability_valid_rows": 0,
            "observability_tail_objective": 0.0,
            "prior_loss": 0.0,
            "prior_positive_anchors": 0,
            "prior_negative_anchors": 0,
            "prior_positive_families": 0,
            "prior_negative_families": 0,
            "rows": 0,
            "supervised": 0,
            "recoverable_false": 0,
            "unrecoverable_false": 0,
            "clean_rows": 0,
            "clean_violations": 0,
            "query_gate_sum": 0.0,
            "safe_gate_targets": 0,
            "steps": 0,
        }
        pending_losses = []
        pending_consensus = []
        pending_observability = []

        def step_pending() -> None:
            nonlocal global_step
            if not pending_losses:
                return
            mean_objective = torch.stack(pending_losses).mean()
            risks = torch.stack(pending_consensus)
            observability_risks = torch.stack(pending_observability)
            tail_count = max(
                1,
                int(math.ceil((1.0 - float(query_tail_alpha)) * risks.numel())),
            )
            tail_objective = torch.topk(risks, k=tail_count).values.mean()
            observability_tail = torch.topk(
                observability_risks, k=tail_count
            ).values.mean()
            objective = (
                mean_objective
                + float(query_tail_weight) * tail_objective
                + float(observability_tail_weight) * observability_tail
            )
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError("non-finite query-level context objective")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            nonfinite_gradients = [
                name
                for name, parameter in expert.named_parameters()
                if parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
            ]
            if nonfinite_gradients:
                raise FloatingPointError(
                    "non-finite query-level gradients in "
                    f"{nonfinite_gradients}; objective={float(objective.detach())}, "
                    f"risks={risks.detach().cpu().tolist()}"
                )
            torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
            optimizer.step()
            query_count = len(pending_losses)
            totals["optimizer_objective"] += float(objective.detach()) * query_count
            totals["tail_objective"] += (
                float(tail_objective.detach()) * query_count
            )
            totals["observability_tail_objective"] += (
                float(observability_tail.detach()) * query_count
            )
            totals["steps"] += 1
            global_step += 1
            pending_losses.clear()
            pending_consensus.clear()
            pending_observability.clear()

        for completed, position in enumerate(order, start=1):
            query_index = support[position]
            record = records[query_index]
            eligible = record["matchable_rows"]
            if not eligible.numel():
                continue
            if eligible.numel() > int(batch_size):
                selection = torch.randperm(
                    eligible.numel(), generator=generator
                )[: int(batch_size)]
                local_rows = eligible[selection]
            else:
                local_rows = eligible
            native_rows = record["native_rows"][local_rows]
            raw, tokens = context_from_cached_query(
                cache[names[query_index]],
                native_rows,
                device=device,
                kernels=expert.context_kernels,
                context_mode=expert.context_mode,
            )
            with torch.no_grad():
                base, _ = metric(raw)
            positives = record["positives"][local_rows].to(device)
            ignored = record["ignored"][local_rows].to(device)
            with torch.no_grad():
                base_scores = base @ reference.T
                ranked_scores, ranked_indices = torch.topk(
                    base_scores,
                    k=min(int(repair_topk), reference.shape[0]),
                    dim=1,
                )
                base_top1 = ranked_indices[:, 0]
                top1_positive = (
                    (positives == base_top1[:, None]) & (positives >= 0)
                ).any(dim=1)
                top1_ignored = (
                    (ignored == base_top1[:, None]) & (ignored >= 0)
                ).any(dim=1)
                false_top1 = ~top1_positive & ~top1_ignored
                positive_in_repair_topk = (
                    (
                        ranked_indices[:, :, None] == positives[:, None, :]
                    )
                    & (positives[:, None, :] >= 0)
                ).any(dim=2).any(dim=1)
                recoverable_false = false_top1 & positive_in_repair_topk
                unrecoverable_false = false_top1 & ~positive_in_repair_topk
                harmful = torch.where(
                    recoverable_false,
                    base_top1,
                    torch.full_like(base_top1, -1),
                )[:, None]
                clean_anchor = torch.where(
                    top1_positive,
                    base_top1,
                    torch.full_like(base_top1, -1),
                )
                base_margin = ranked_scores[:, 0] - ranked_scores[:, 1]
                clean_floor = (
                    base_margin - float(clean_margin_slack)
                ) / (1.0 + float(context_weight))
                clean_floor[~top1_positive] = torch.nan

            query_codes, query_gates = expert.query(base, tokens)
            unit_query_codes = F.normalize(query_codes, dim=1)
            gate_loss = query_codes.new_zeros(())
            if expert.learned_query_gate:
                with torch.no_grad():
                    full_scores = base_scores + float(context_weight) * (
                        unit_query_codes @ context_bank.T
                    )
                    full_winner = full_scores.argmax(dim=1)
                    full_positive = (
                        (positives == full_winner[:, None]) & (positives >= 0)
                    ).any(dim=1)
                    full_ignored = (
                        (ignored == full_winner[:, None]) & (ignored >= 0)
                    ).any(dim=1)
                    repair_count = int(
                        (recoverable_false & full_positive).sum()
                    )
                    harm_count = int(
                        (
                            top1_positive & ~full_positive & ~full_ignored
                        ).sum()
                    )
                    safe_target = float(repair_count > 0 and harm_count == 0)
                gate_target = query_gates.new_full((1, 1), safe_target)
                gate_loss = F.binary_cross_entropy(
                    query_gates[:1], gate_target
                )
                query_codes = unit_query_codes * query_gates.detach()
            joint_query = concatenate_dual_expert_descriptors(
                base,
                query_codes,
                context_weight=float(context_weight),
                map_side=False,
            )
            per_row = _multi_positive_list_loss(
                joint_query,
                joint_bank,
                positives,
                ignored,
                harmful,
                topk=int(training_topk),
                temperature=float(temperature),
                harmful_weight=float(collision_weight),
            )
            row_weights = torch.zeros_like(per_row)
            row_weights[recoverable_false] = 1.0
            row_weights[top1_positive] = float(clean_task_scale)
            list_loss = (per_row * row_weights).sum() / row_weights.sum().clamp_min(
                1e-8
            )
            clean_loss, clean_diagnostics = protected_clean_margin_loss(
                joint_query,
                joint_bank,
                clean_anchor,
                clean_floor,
            )
            consensus_loss, legal_mass = expected_clean_inlier_loss(
                joint_query,
                joint_bank,
                positives,
                ignored,
                topk=int(training_topk),
                temperature=float(temperature),
            )
            prior_loss = joint_query.new_zeros(())
            prior_diagnostics = {
                "unique_positive_anchor_count": 0,
                "unique_negative_anchor_count": 0,
                "positive_family_count": 0,
                "negative_family_count": 0,
            }
            if prior_enabled:
                all_positive = record["positives"][record["matchable_rows"]]
                unique_positive = torch.unique(
                    all_positive[all_positive >= 0].to(device), sorted=True
                )
                unique_negative = torch.unique(base_top1[false_top1], sorted=True)
                cached = cache[names[query_index]]
                incompatible = select_incompatible_anchor_negatives(
                    unit_query_codes[0],
                    context_bank,
                    geometry_bank,
                    cached["pose_w2c"],
                    cached["native_K"],
                    cached["native_input_hw"],
                    unique_positive,
                    maximum_count=int(prior_incompatible_negatives),
                )
                unique_negative = torch.unique(
                    torch.cat((unique_negative, incompatible)), sorted=True
                )
                if unique_negative.numel():
                    unique_negative = unique_negative[
                        ~torch.isin(unique_negative, unique_positive)
                    ]
                snapshot_code = context_query_codes[query_index].to(device)[0]
                positive_codes = leave_one_query_out_anchor_codes(
                    context_bank,
                    view_counts,
                    unique_positive,
                    snapshot_code,
                )
                prior_loss, prior_diagnostics = balanced_unique_anchor_prior_loss(
                    unit_query_codes[0],
                    positive_codes,
                    unique_positive,
                    context_bank[unique_negative],
                    unique_negative,
                    type_bank,
                    family_bank,
                    temperature=float(prior_temperature),
                )
            observability_loss = joint_query.new_zeros(())
            observability_diagnostics = {
                "valid_row_count": 0,
                "information_retention": joint_query.new_zeros(()),
                "expected_min_eigenvalue": joint_query.new_zeros(()),
            }
            if observability_enabled:
                cached = cache[names[query_index]]
                observability_loss, observability_diagnostics = (
                    expected_clean_pose_information_loss(
                        joint_query,
                        joint_bank,
                        positives,
                        ignored,
                        geometry_bank,
                        cached["pose_w2c"],
                        cached["native_K"],
                        topk=int(training_topk),
                        temperature=float(temperature),
                        damping=float(observability_damping),
                    )
                )
            task_loss = (
                float(prior_weight) * prior_loss if prior_enabled else list_loss
            )
            loss = (
                task_loss
                + float(clean_weight) * clean_loss
                + float(gate_supervision_weight) * gate_loss
                + float(consensus_weight) * consensus_loss
                + float(observability_weight) * observability_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    "non-finite context-score loss: "
                    f"list={float(list_loss.detach())}, "
                    f"clean={float(clean_loss.detach())}, "
                    f"gate={float(gate_loss.detach())}, "
                    f"consensus={float(consensus_loss.detach())}, "
                    f"observability={float(observability_loss.detach())}, "
                    f"prior={float(prior_loss.detach())}"
                )
            pending_losses.append(loss)
            pending_consensus.append(consensus_loss)
            pending_observability.append(observability_loss)
            if len(pending_losses) >= int(query_batch_size):
                step_pending()
            row_count = int(local_rows.numel())
            totals["loss"] += float(loss.detach()) * row_count
            totals["list"] += float(list_loss.detach()) * row_count
            totals["clean_loss"] += float(clean_loss.detach()) * row_count
            totals["gate_loss"] += float(gate_loss.detach()) * row_count
            totals["consensus_loss"] += (
                float(consensus_loss.detach()) * row_count
            )
            totals["legal_mass"] += float(legal_mass.mean()) * row_count
            totals["observability_loss"] += (
                float(observability_loss.detach()) * row_count
            )
            totals["observability_retention"] += (
                float(observability_diagnostics["information_retention"])
                * row_count
            )
            totals["observability_min_eigenvalue"] += (
                float(observability_diagnostics["expected_min_eigenvalue"])
                * row_count
            )
            totals["observability_valid_rows"] += int(
                observability_diagnostics["valid_row_count"]
            )
            totals["prior_loss"] += float(prior_loss.detach()) * row_count
            totals["prior_positive_anchors"] += int(
                prior_diagnostics["unique_positive_anchor_count"]
            )
            totals["prior_negative_anchors"] += int(
                prior_diagnostics["unique_negative_anchor_count"]
            )
            totals["prior_positive_families"] += int(
                prior_diagnostics["positive_family_count"]
            )
            totals["prior_negative_families"] += int(
                prior_diagnostics["negative_family_count"]
            )
            totals["rows"] += row_count
            totals["supervised"] += int((row_weights > 0).sum())
            totals["recoverable_false"] += int(recoverable_false.sum())
            totals["unrecoverable_false"] += int(unrecoverable_false.sum())
            totals["clean_rows"] += int(top1_positive.sum())
            totals["clean_violations"] += int(
                clean_diagnostics["protected_clean_violations"]
            )
            totals["query_gate_sum"] += float(query_gates.detach().sum())
            totals["safe_gate_targets"] += int(
                expert.learned_query_gate and safe_target > 0.0
            )
            if progress_interval > 0 and (
                completed % int(progress_interval) == 0
                or completed == len(order)
            ):
                print(
                    {
                        "event": "context_score_train",
                        "stage": stage_name,
                        "epoch": epoch + 1,
                        "queries_complete": completed,
                        "query_count": len(order),
                    },
                    flush=True,
                )
        step_pending()
        denominator = max(int(totals["rows"]), 1)
        query_denominator = max(len(order), 1)
        row = {
            "stage": stage_name,
            "epoch": epoch + 1,
            "global_step": int(global_step),
            "step_count": int(totals["steps"]),
            "row_count": int(totals["rows"]),
            "supervised_row_count": int(totals["supervised"]),
            "recoverable_false_top1_row_count": int(totals["recoverable_false"]),
            "unrecoverable_false_top1_row_count": int(
                totals["unrecoverable_false"]
            ),
            "clean_top1_row_count": int(totals["clean_rows"]),
            "clean_violation_count": int(totals["clean_violations"]),
            "mean_loss": float(totals["loss"] / denominator),
            "mean_list_loss": float(totals["list"] / denominator),
            "mean_clean_loss": float(totals["clean_loss"] / denominator),
            "mean_gate_loss": float(totals["gate_loss"] / denominator),
            "mean_consensus_loss": float(
                totals["consensus_loss"] / denominator
            ),
            "mean_expected_clean_inlier_mass": float(
                totals["legal_mass"] / denominator
            ),
            "mean_optimizer_objective": float(
                totals["optimizer_objective"] / query_denominator
            ),
            "mean_tail_objective": float(
                totals["tail_objective"] / query_denominator
            ),
            "mean_observability_loss": float(
                totals["observability_loss"] / denominator
            ),
            "mean_observability_information_retention": float(
                totals["observability_retention"] / denominator
            ),
            "mean_observability_min_eigenvalue": float(
                totals["observability_min_eigenvalue"] / denominator
            ),
            "observability_valid_row_count": int(
                totals["observability_valid_rows"]
            ),
            "mean_observability_tail_objective": float(
                totals["observability_tail_objective"] / query_denominator
            ),
            "mean_prior_loss": float(totals["prior_loss"] / denominator),
            "mean_unique_positive_anchors_per_query": float(
                totals["prior_positive_anchors"] / query_denominator
            ),
            "mean_unique_negative_anchors_per_query": float(
                totals["prior_negative_anchors"] / query_denominator
            ),
            "mean_positive_families_per_query": float(
                totals["prior_positive_families"] / query_denominator
            ),
            "mean_negative_families_per_query": float(
                totals["prior_negative_families"] / query_denominator
            ),
            "mean_query_gate": float(
                totals["query_gate_sum"] / denominator
            ),
            "safe_gate_target_query_count": int(totals["safe_gate_targets"]),
        }
        history.append(row)
        print({"event": "context_score_train_epoch_complete", **row}, flush=True)
    return {
        "stage": stage_name,
        "config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "repair_topk": int(repair_topk),
            "training_topk": int(training_topk),
            "learning_rate": float(learning_rate),
            "temperature": float(temperature),
            "collision_weight": float(collision_weight),
            "clean_weight": float(clean_weight),
            "clean_margin_slack": float(clean_margin_slack),
            "clean_task_scale": float(clean_task_scale),
            "gate_supervision_weight": float(gate_supervision_weight),
            "consensus_weight": float(consensus_weight),
            "query_tail_weight": float(query_tail_weight),
            "query_tail_alpha": float(query_tail_alpha),
            "query_batch_size": int(query_batch_size),
            "observability_weight": float(observability_weight),
            "observability_tail_weight": float(observability_tail_weight),
            "observability_damping": float(observability_damping),
            "training_objective": training_objective,
            "prior_weight": float(prior_weight),
            "prior_temperature": float(prior_temperature),
            "prior_incompatible_negatives": int(prior_incompatible_negatives),
            "context_weight": float(context_weight),
            "seed": int(seed),
        },
        "history": history,
    }


def _empty_clean() -> dict:
    return {
        "a1_clean_row_count": 0,
        "positive_retained_count": 0,
        "exact_winner_retained_count": 0,
        "new_false_attractor_count": 0,
        "clean_margin_violation_count": 0,
    }


def summarize_clean_counts(counts: dict) -> dict:
    denominator = max(int(counts["a1_clean_row_count"]), 1)
    return {
        **{key: int(value) for key, value in counts.items()},
        "positive_retention_percent": float(
            100.0 * counts["positive_retained_count"] / denominator
        ),
        "exact_winner_retention_percent": float(
            100.0 * counts["exact_winner_retained_count"] / denominator
        ),
        "new_false_attractor_percent": float(
            100.0 * counts["new_false_attractor_count"] / denominator
        ),
        "clean_margin_violation_percent": float(
            100.0 * counts["clean_margin_violation_count"] / denominator
        ),
    }


@torch.inference_mode()
def evaluate_context_score_banks(
    *,
    state: dict,
    teacher: dict,
    query_cache: dict,
    gate_query_indices: Sequence[int],
    pose_query_indices: Sequence[int],
    anchor_indices: torch.Tensor,
    base_bank: torch.Tensor,
    context_bank: torch.Tensor,
    metric: SharedLowRankMetric,
    expert: ContextScoreExpert,
    context_weights: Sequence[float],
    device: torch.device,
    topks: Sequence[int] = DEFAULT_TOPKS,
    deployment_row_limit: int = 0,
    ransac_reprojection_px: float = 12.0,
    clean_margin_slack: float = 0.01,
    seed: int = 2026,
    progress_interval: int = 25,
) -> tuple[dict, list[dict]]:
    """Evaluate A1 and all preregistered context weights on identical rows."""
    topks = tuple(sorted(set(int(value) for value in topks)))
    weights = tuple(float(value) for value in context_weights)
    if not weights or any(value <= 0.0 for value in weights):
        raise ValueError("context weights must be non-empty and positive")
    gate = [int(value) for value in gate_query_indices]
    pose_set = {int(value) for value in pose_query_indices}
    if not gate or not pose_set.issubset(set(gate)):
        raise ValueError("invalid gate or pose query partition")
    names = list(teacher["query_names"])
    cache = query_cache.get("queries", query_cache)
    anchor_indices = torch.as_tensor(anchor_indices).long().to(device)
    base_bank = torch.as_tensor(base_bank, device=device).float()
    context_bank = torch.as_tensor(context_bank, device=device).float()
    supported = torch.zeros(int(teacher["anchor_count"]), dtype=torch.bool)
    supported[anchor_indices.cpu()] = True
    anchor_type = torch.as_tensor(state["anchor_type"]).long().cpu()
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    max_k = min(max(topks), anchor_indices.numel())
    protocols = ["a1", *(protocol_name(value) for value in weights)]
    retrieval = {name: _empty_retrieval(topks) for name in protocols}
    clean = {name: _empty_clean() for name in protocols if name != "a1"}
    pose_rows = []
    query_norms = []
    query_gate_values = []
    for completed, query_index in enumerate(gate, start=1):
        name = names[query_index]
        record = teacher["records"][query_index]
        all_rows = torch.as_tensor(record["query_rows"]).long()
        selected_local = torch.arange(all_rows.numel())
        if deployment_row_limit > 0:
            selected_local = selected_local[all_rows < int(deployment_row_limit)]
        rows = all_rows[selected_local]
        if not rows.numel():
            continue
        raw, tokens = context_from_cached_query(
            cache[name],
            rows,
            device=device,
            kernels=expert.context_kernels,
            context_mode=expert.context_mode,
        )
        base, _ = metric(raw)
        query_codes, query_gates = expert.query(base, tokens)
        query_norms.extend(query_codes.norm(dim=1).cpu().tolist())
        query_gate_values.extend(query_gates[:, 0].cpu().tolist())
        base_scores = base @ base_bank.T
        context_scores = query_codes @ context_bank.T
        scores = {"a1": base_scores}
        for weight in weights:
            scores[protocol_name(weight)] = base_scores + weight * context_scores

        _, positive_rows, positive_anchors = _selected_csr_edges(
            record, "positive", selected_local
        )
        positive_keep = supported[positive_anchors]
        positive_rows = positive_rows[positive_keep]
        positive_anchors = positive_anchors[positive_keep]
        _, ambiguous_rows, ambiguous_anchors = _selected_csr_edges(
            record, "ambiguous", selected_local
        )
        ambiguous_keep = supported[ambiguous_anchors]
        ambiguous_rows = ambiguous_rows[ambiguous_keep]
        ambiguous_anchors = ambiguous_anchors[ambiguous_keep]

        positive_sets = [set() for _ in range(rows.numel())]
        ambiguous_sets = [set() for _ in range(rows.numel())]
        for edge_row, anchor in zip(
            positive_rows.tolist(), positive_anchors.tolist()
        ):
            positive_sets[edge_row].add(anchor)
        for edge_row, anchor in zip(
            ambiguous_rows.tolist(), ambiguous_anchors.tolist()
        ):
            ambiguous_sets[edge_row].add(anchor)

        winners = {}
        local_winners = {}
        for descriptor_name, score in scores.items():
            local_ranked = torch.topk(score, k=max_k, dim=1).indices
            ranked = anchor_indices[local_ranked].cpu()
            winners[descriptor_name] = ranked[:, 0]
            local_winners[descriptor_name] = local_ranked[:, 0]
            _update_descriptor_counts(
                retrieval[descriptor_name],
                ranked=ranked,
                positive_edge_rows=positive_rows,
                positive_edge_anchors=positive_anchors,
                ambiguous_edge_rows=ambiguous_rows,
                ambiguous_edge_anchors=ambiguous_anchors,
                anchor_type=anchor_type,
                topks=topks,
            )

        a1_winners = winners["a1"].tolist()
        a1_clean_rows = [
            row_index
            for row_index, winner in enumerate(a1_winners)
            if winner in positive_sets[row_index]
        ]
        if a1_clean_rows:
            clean_index = torch.as_tensor(a1_clean_rows, device=device).long()
            clean_anchor = local_winners["a1"][clean_index]
            a1_top2 = torch.topk(base_scores[clean_index], k=2, dim=1).values
            base_margin = a1_top2[:, 0] - a1_top2[:, 1]
            for descriptor_name in clean:
                counts = clean[descriptor_name]
                protocol_winners = winners[descriptor_name].tolist()
                counts["a1_clean_row_count"] += len(a1_clean_rows)
                for row_index in a1_clean_rows:
                    winner = protocol_winners[row_index]
                    if winner in positive_sets[row_index]:
                        counts["positive_retained_count"] += 1
                    elif winner not in ambiguous_sets[row_index]:
                        counts["new_false_attractor_count"] += 1
                    if winner == a1_winners[row_index]:
                        counts["exact_winner_retained_count"] += 1
                protocol_scores = scores[descriptor_name][clean_index]
                clean_score = protocol_scores.gather(1, clean_anchor[:, None])[:, 0]
                masked = protocol_scores.clone()
                masked.scatter_(1, clean_anchor[:, None], -torch.inf)
                protocol_margin = clean_score - masked.max(dim=1).values
                counts["clean_margin_violation_count"] += int(
                    (
                        protocol_margin
                        < base_margin - float(clean_margin_slack)
                    ).sum()
                )

        if query_index in pose_set:
            cached = cache[name]
            keypoints = (
                torch.as_tensor(cached["native_keypoints"]).float()[rows]
                + float(cached.get("pixel_center_offset", 0.5))
            ).cpu()
            intrinsic = torch.as_tensor(cached["native_K"]).float().cpu()
            gt_pose = torch.as_tensor(cached["pose_w2c"]).float().cpu()
            pose_row = {"query_index": query_index, "image_name": name}
            for descriptor_name, assignments in winners.items():
                result = _solve_assignments(
                    assignments,
                    keypoints=keypoints,
                    xyz=xyz,
                    intrinsic=intrinsic,
                    gt_pose=gt_pose,
                    reprojection_error_px=ransac_reprojection_px,
                    seed=seed,
                )
                pose_row.update(
                    {
                        f"{descriptor_name}_{key}": value
                        for key, value in result.items()
                    }
                )
            pose_rows.append(pose_row)
        if progress_interval > 0 and (
            completed % int(progress_interval) == 0 or completed == len(gate)
        ):
            print(
                {
                    "event": "context_score_gate",
                    "queries_complete": completed,
                    "query_count": len(gate),
                },
                flush=True,
            )
    norms = np.asarray(query_norms, dtype=np.float64)
    gates = np.asarray(query_gate_values, dtype=np.float64)
    report = {
        name: summarize_retrieval(counts, topks)
        for name, counts in retrieval.items()
    }
    report["clean_preservation"] = {
        name: summarize_clean_counts(counts) for name, counts in clean.items()
    }
    report["expert_diagnostics"] = {
        "gated_query_code_norm_mean": (
            float(norms.mean()) if norms.size else 0.0
        ),
        "query_gate_mean": float(gates.mean()) if gates.size else 0.0,
        "query_gate_p10": (
            float(np.percentile(gates, 10)) if gates.size else 0.0
        ),
        "query_gate_p90": (
            float(np.percentile(gates, 90)) if gates.size else 0.0
        ),
    }
    report["additive_counts"] = retrieval
    report["additive_clean_counts"] = clean
    return report, pose_rows


def summarize_context_score_pose(
    pose_rows: list[dict], context_weights: Sequence[float]
) -> dict:
    protocols = ["a1", *(protocol_name(value) for value in context_weights)]
    output = {name: _pose_summary(pose_rows, name) for name in protocols}
    for name in protocols:
        valid = [row for row in pose_rows if not row[f"{name}_failed"]]
        for threshold in (1.0, 2.0, 5.0):
            count = sum(
                row[f"{name}_te_cm"] <= threshold
                and row[f"{name}_ae_deg"] <= threshold
                for row in valid
            )
            output[name][
                f"recall_{int(threshold)}cm_{int(threshold)}deg_percent"
            ] = 100.0 * count / max(len(pose_rows), 1)
    return output


def compare_context_score_protocols(
    retrieval: dict,
    pose: dict,
    context_weights: Sequence[float],
) -> dict:
    """Apply the preregistered safety gate and select only among passing λ."""
    base_r1 = float(retrieval["a1"]["positive_recall_at_k"]["1"])
    base_pose = pose["a1"]
    comparisons = {}
    passing = []
    for weight in context_weights:
        name = protocol_name(weight)
        candidate_r1 = float(retrieval[name]["positive_recall_at_k"]["1"])
        candidate_pose = pose[name]
        if int(base_pose.get("query_count", 0)) == 0:
            comparison = {
                "context_weight": float(weight),
                "top1_positive_recall_delta_percentage_points": float(
                    100.0 * (candidate_r1 - base_r1)
                ),
                "mapping_gate_pass": False,
                "routing_verdict": "pose_replay_skipped_no_final_verdict",
            }
        else:
            def relative(key: str) -> float:
                base = float(base_pose.get(key, 0.0))
                return float(
                    (float(candidate_pose.get(key, 0.0)) - base)
                    / max(base, 1e-12)
                )

            recall_delta = float(
                candidate_pose["recall_2cm_2deg_percent"]
                - base_pose["recall_2cm_2deg_percent"]
            )
            tail_improves = all(
                float(candidate_pose[key]) <= float(base_pose[key])
                for key in (
                    "mean_te_cm",
                    "p90_te_cm",
                    "cvar95_te_cm",
                    "mean_hypotheses",
                )
            )
            no_new_catastrophe = int(
                candidate_pose["catastrophic_100cm_count"]
            ) <= int(base_pose["catastrophic_100cm_count"])
            passed = recall_delta >= 0.0 and tail_improves and no_new_catastrophe
            comparison = {
                "context_weight": float(weight),
                "top1_positive_recall_delta_percentage_points": float(
                    100.0 * (candidate_r1 - base_r1)
                ),
                "recall_2cm_2deg_delta_percentage_points": recall_delta,
                "relative_mean_te": relative("mean_te_cm"),
                "relative_p90_te": relative("p90_te_cm"),
                "relative_cvar95_te": relative("cvar95_te_cm"),
                "relative_mean_hypotheses": relative("mean_hypotheses"),
                "no_new_catastrophic_pose": bool(no_new_catastrophe),
                "mapping_gate_pass": bool(passed),
                "routing_verdict": (
                    "eligible_for_sentinel_scenes"
                    if passed
                    else "hold_before_sentinel_scenes"
                ),
            }
            if passed:
                passing.append(name)
        comparisons[name] = comparison
    # Safety-first selection: tight pose recall dominates, followed by the
    # translation tail and mean.  Retrieval R@1 breaks only otherwise equal
    # pose outcomes; the smaller weight is the final conservative tie-break.
    selected = max(
        passing,
        key=lambda name: (
            float(pose[name]["recall_2cm_2deg_percent"]),
            -float(pose[name]["cvar95_te_cm"]),
            -float(pose[name]["mean_te_cm"]),
            float(retrieval[name]["positive_recall_at_k"]["1"]),
            -float(comparisons[name]["context_weight"]),
        ),
        default=None,
    )
    return {
        "by_protocol": comparisons,
        "selected_protocol": selected,
        "mapping_gate_pass": selected is not None,
        "routing_verdict": (
            "advance_selected_context_weight_to_sentinel_scenes"
            if selected is not None
            else "stop_context_score_branch_before_sentinel_scenes"
        ),
    }
