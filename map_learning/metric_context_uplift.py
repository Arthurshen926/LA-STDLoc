"""Metric-preserving, map-consistent contextual residual uplift.

The frozen A1 shared metric and learned anchor descriptors remain the local
expert.  A shared single-image context head predicts only a bounded tangent
angle update.  Mapping observations fuse those updates, never replacement
descriptors, so the zero-initialized protocol exactly reproduces A1.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
from torch import nn

from map_learning.alias_teacher import protected_clean_margin_loss
from map_learning.context_booster_crossfit import (
    DEFAULT_TOPKS,
    _empty_retrieval,
    _update_descriptor_counts,
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


class MetricPreservingContextUplift(nn.Module):
    """Predict a gated tangent update while preserving A1 at initialization."""

    def __init__(
        self,
        *,
        descriptor_dim: int = 256,
        hidden_dim: int = 256,
        context_kernels: Sequence[int] = DEFAULT_CONTEXT_KERNELS,
        context_mode: str = "multi_scale_global",
        maximum_angle_rad: float = 0.05,
    ) -> None:
        super().__init__()
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_kernels = tuple(int(value) for value in context_kernels)
        self.context_mode = str(context_mode)
        self.maximum_angle_rad = float(maximum_angle_rad)
        if self.descriptor_dim < 1 or self.hidden_dim < 1:
            raise ValueError("descriptor and hidden dimensions must be positive")
        if self.context_mode not in CONTEXT_MODES:
            raise ValueError(f"unsupported context mode: {self.context_mode}")
        if not 0.0 <= self.maximum_angle_rad < math.pi / 2:
            raise ValueError("maximum descriptor angle must lie in [0, pi/2)")
        token_count = len(self.context_kernels) + 1
        input_dim = self.descriptor_dim * (token_count + 1)
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.context_head = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.descriptor_dim + 1),
        )
        final = self.context_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    @staticmethod
    def apply_angle_vector(
        base_descriptors: torch.Tensor,
        angle_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Apply tangent vectors with the unit-sphere exponential map."""
        base = torch.as_tensor(base_descriptors).float()
        update = torch.as_tensor(
            angle_vectors, device=base.device, dtype=base.dtype
        )
        if base.shape != update.shape or base.ndim != 2:
            raise ValueError("base descriptors and angle vectors must align")
        update = update - (update * base).sum(dim=1, keepdim=True) * base
        angle = torch.linalg.norm(update, dim=1, keepdim=True)
        sinc = torch.sin(angle) / angle.clamp_min(1e-8)
        sinc = torch.where(angle <= 1e-8, torch.ones_like(sinc), sinc)
        return torch.cos(angle) * base + sinc * update

    def forward(
        self,
        base_descriptors: torch.Tensor,
        context_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.as_tensor(base_descriptors).float()
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
        inputs = torch.cat((base, context.flatten(start_dim=1)), dim=1)
        prediction = self.context_head(self.input_norm(inputs))
        raw_direction = prediction[:, : self.descriptor_dim]
        gate = torch.sigmoid(prediction[:, self.descriptor_dim :])
        tangent = raw_direction - (
            raw_direction * base
        ).sum(dim=1, keepdim=True) * base
        norm = torch.linalg.norm(tangent, dim=1, keepdim=True)
        if self.maximum_angle_rad == 0.0:
            angle_vector = torch.zeros_like(tangent)
        else:
            maximum = tangent.new_tensor(self.maximum_angle_rad)
            angle_vector = tangent * gate * (maximum / (maximum + norm))
        return self.apply_angle_vector(base, angle_vector), angle_vector, gate

    def export_config(self) -> dict:
        return {
            "descriptor_dim": self.descriptor_dim,
            "hidden_dim": self.hidden_dim,
            "context_kernels": list(self.context_kernels),
            "context_mode": self.context_mode,
            "maximum_angle_rad": self.maximum_angle_rad,
            "maximum_angle_deg": math.degrees(self.maximum_angle_rad),
            "identity_initialization": "zero_final_projection",
            "geometry": "unit_sphere_tangent_exponential_map",
            "query_gate": "learned_sigmoid",
            "map_gate": "cross_view_residual_direction_coherence",
            "output_dim": self.descriptor_dim,
        }


def load_frozen_metric_state(
    state: dict,
    *,
    anchor_ids: torch.Tensor,
    device: torch.device,
) -> SharedLowRankMetric:
    """Load and strictly align the frozen A1 metric used by the uplift."""
    if state.get("schema") != "lafgs_shared_metric_state":
        raise ValueError("unsupported shared metric schema")
    metric_ids = torch.as_tensor(state["landmark_indices"]).long().reshape(-1)
    if not torch.equal(metric_ids.cpu(), torch.as_tensor(anchor_ids).long().cpu()):
        raise ValueError("shared metric does not align with the base anchor map")
    metric = SharedLowRankMetric(**state["metric_config"]).to(device)
    metric.load_state_dict(state["metric_state_dict"], strict=True)
    metric.requires_grad_(False)
    return metric.eval()


def _accumulate_view_vectors(
    vector_sum: torch.Tensor,
    magnitude_sum: torch.Tensor,
    view_counts: torch.Tensor,
    anchor_indices: torch.Tensor,
    vectors: torch.Tensor,
) -> int:
    """Average repeated rows inside one image, then add one vector per view."""
    anchors = torch.as_tensor(anchor_indices, device=vector_sum.device).long()
    values = torch.as_tensor(vectors, device=vector_sum.device).float()
    if anchors.ndim != 1 or values.ndim != 2:
        raise ValueError("anchors and vectors must have shapes [E] and [E, D]")
    if anchors.numel() != values.shape[0]:
        raise ValueError("anchor and vector edge counts must agree")
    if not anchors.numel():
        return 0
    unique, inverse = torch.unique(anchors, sorted=True, return_inverse=True)
    per_view = values.new_zeros((unique.numel(), values.shape[1]))
    edge_counts = values.new_zeros((unique.numel(), 1))
    per_view.index_add_(0, inverse, values)
    edge_counts.index_add_(0, inverse, values.new_ones((values.shape[0], 1)))
    per_view = per_view / edge_counts.clamp_min(1.0)
    vector_sum.index_add_(0, unique, per_view)
    magnitude_sum.index_add_(0, unique, per_view.norm(dim=1))
    view_counts[unique] += 1
    return int(unique.numel())


@torch.inference_mode()
def build_metric_uplift_bank(
    *,
    adapter: MetricPreservingContextUplift,
    metric: SharedLowRankMetric,
    base_anchor_bank: torch.Tensor,
    teacher: dict,
    query_cache: dict,
    support_query_indices: Sequence[int],
    anchor_indices: torch.Tensor,
    expected_view_counts: torch.Tensor,
    device: torch.device,
    progress_interval: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Fuse only observation tangent updates onto frozen learned anchors."""
    names = list(teacher["query_names"])
    cache = query_cache.get("queries", query_cache)
    support = [int(value) for value in support_query_indices]
    anchor_count = int(teacher["anchor_count"])
    vector_sum = torch.zeros((anchor_count, adapter.descriptor_dim), device=device)
    magnitude_sum = torch.zeros(anchor_count, device=device)
    view_counts = torch.zeros(anchor_count, dtype=torch.long, device=device)
    observation_angles = []
    observation_gates = []
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
                kernels=adapter.context_kernels,
                context_mode=adapter.context_mode,
            )
            base_observations, _ = metric(raw)
            _, angle_vectors, gates = adapter(base_observations, tokens)
            _accumulate_view_vectors(
                vector_sum,
                magnitude_sum,
                view_counts,
                edge_anchors.to(device),
                angle_vectors,
            )
            observation_angles.extend(angle_vectors.norm(dim=1).cpu().tolist())
            observation_gates.extend(gates[:, 0].cpu().tolist())
        if progress_interval > 0 and (
            completed % int(progress_interval) == 0 or completed == len(support)
        ):
            print(
                {
                    "event": "metric_context_uplift_bank",
                    "queries_complete": completed,
                    "query_count": len(support),
                },
                flush=True,
            )
    selected = torch.as_tensor(anchor_indices, device=device).long()
    if not torch.equal(view_counts[selected], expected_view_counts.to(device)):
        raise AssertionError("uplift and support view counts diverged")
    counts = view_counts[selected].float().clamp_min(1.0)
    mean_update = vector_sum[selected] / counts[:, None]
    coherence = vector_sum[selected].norm(dim=1) / magnitude_sum[selected].clamp_min(
        1e-8
    )
    coherence = torch.where(
        magnitude_sum[selected] > 1e-8,
        coherence.clamp(0.0, 1.0),
        torch.zeros_like(coherence),
    )
    base = torch.as_tensor(base_anchor_bank, device=device).float()
    if base.shape != mean_update.shape:
        raise ValueError("base anchor bank does not align with supported anchors")
    map_update = mean_update * coherence[:, None]
    map_update = map_update - (map_update * base).sum(dim=1, keepdim=True) * base
    uplifted = adapter.apply_angle_vector(base, map_update)
    obs_angles = np.asarray(observation_angles, dtype=np.float64)
    obs_gates = np.asarray(observation_gates, dtype=np.float64)
    map_angles = map_update.norm(dim=1).cpu().numpy()
    coherence_values = coherence.cpu().numpy()
    return uplifted, {
        "adapted_observation_count": int(obs_angles.size),
        "observation_angle_rad_mean": (
            float(obs_angles.mean()) if obs_angles.size else 0.0
        ),
        "observation_angle_rad_p90": (
            float(np.percentile(obs_angles, 90)) if obs_angles.size else 0.0
        ),
        "observation_angle_rad_maximum": (
            float(obs_angles.max()) if obs_angles.size else 0.0
        ),
        "query_gate_mean": float(obs_gates.mean()) if obs_gates.size else 0.0,
        "map_angle_rad_mean": float(map_angles.mean()) if map_angles.size else 0.0,
        "map_angle_rad_p90": (
            float(np.percentile(map_angles, 90)) if map_angles.size else 0.0
        ),
        "map_angle_rad_maximum": (
            float(map_angles.max()) if map_angles.size else 0.0
        ),
        "map_coherence_mean": (
            float(coherence_values.mean()) if coherence_values.size else 0.0
        ),
        "map_coherence_p10": (
            float(np.percentile(coherence_values, 10))
            if coherence_values.size
            else 0.0
        ),
    }


def train_metric_context_stage(
    *,
    adapter: MetricPreservingContextUplift,
    metric: SharedLowRankMetric,
    teacher: dict,
    query_cache: dict,
    support_query_indices: Sequence[int],
    records: dict[int, dict],
    base_reference_bank: torch.Tensor,
    task_bank: torch.Tensor,
    device: torch.device,
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
    trust_weight: float = 1.0,
    seed: int = 2026,
    stage_name: str = "a1_anchor_target",
    progress_interval: int = 100,
) -> dict:
    """Train only clean A1 rows and A1-false rows recoverable within top-K."""
    if epochs < 1 or batch_size < 1 or repair_topk < 2:
        raise ValueError("epochs/batch must be positive and repair top-K >= 2")
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    support = [int(value) for value in support_query_indices]
    reference = base_reference_bank.detach().to(device).clone()
    task = task_bank.detach().to(device).clone()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(learning_rate), weight_decay=1e-4
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
            "trust": 0.0,
            "rows": 0,
            "supervised": 0,
            "recoverable_false": 0,
            "unrecoverable_false": 0,
            "clean_rows": 0,
            "clean_violations": 0,
            "steps": 0,
        }
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
                kernels=adapter.context_kernels,
                context_mode=adapter.context_mode,
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
                clean_floor = (
                    ranked_scores[:, 0]
                    - ranked_scores[:, 1]
                    - float(clean_margin_slack)
                )
                clean_floor[~top1_positive] = torch.nan

            uplifted, angle_vector, _ = adapter(base, tokens)
            per_row = _multi_positive_list_loss(
                uplifted,
                task,
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
                uplifted,
                task,
                clean_anchor,
                clean_floor,
            )
            trust_loss = angle_vector.square().sum(dim=1).mean()
            loss = (
                list_loss
                + float(clean_weight) * clean_loss
                + float(trust_weight) * trust_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite metric-context loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            row_count = int(local_rows.numel())
            totals["loss"] += float(loss.detach()) * row_count
            totals["list"] += float(list_loss.detach()) * row_count
            totals["clean_loss"] += float(clean_loss.detach()) * row_count
            totals["trust"] += float(trust_loss.detach()) * row_count
            totals["rows"] += row_count
            totals["supervised"] += int((row_weights > 0).sum())
            totals["recoverable_false"] += int(recoverable_false.sum())
            totals["unrecoverable_false"] += int(unrecoverable_false.sum())
            totals["clean_rows"] += int(top1_positive.sum())
            totals["clean_violations"] += int(
                clean_diagnostics["protected_clean_violations"]
            )
            totals["steps"] += 1
            if progress_interval > 0 and (
                completed % int(progress_interval) == 0
                or completed == len(order)
            ):
                print(
                    {
                        "event": "metric_context_train",
                        "stage": stage_name,
                        "epoch": epoch + 1,
                        "queries_complete": completed,
                        "query_count": len(order),
                    },
                    flush=True,
                )
        denominator = max(int(totals["rows"]), 1)
        row = {
            "stage": stage_name,
            "epoch": epoch + 1,
            "global_step": int(global_step),
            "step_count": int(totals["steps"]),
            "row_count": int(totals["rows"]),
            "supervised_row_count": int(totals["supervised"]),
            "recoverable_false_top1_row_count": int(
                totals["recoverable_false"]
            ),
            "unrecoverable_false_top1_row_count": int(
                totals["unrecoverable_false"]
            ),
            "clean_top1_row_count": int(totals["clean_rows"]),
            "clean_violation_count": int(totals["clean_violations"]),
            "mean_loss": float(totals["loss"] / denominator),
            "mean_list_loss": float(totals["list"] / denominator),
            "mean_clean_loss": float(totals["clean_loss"] / denominator),
            "mean_trust_loss": float(totals["trust"] / denominator),
        }
        history.append(row)
        print({"event": "metric_context_train_epoch_complete", **row}, flush=True)
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
            "trust_weight": float(trust_weight),
            "seed": int(seed),
        },
        "history": history,
    }


@torch.inference_mode()
def evaluate_metric_context_banks(
    *,
    state: dict,
    teacher: dict,
    query_cache: dict,
    gate_query_indices: Sequence[int],
    pose_query_indices: Sequence[int],
    banks: dict[str, torch.Tensor],
    metric: SharedLowRankMetric,
    adapter: MetricPreservingContextUplift,
    device: torch.device,
    topks: Sequence[int] = DEFAULT_TOPKS,
    deployment_row_limit: int = 0,
    ransac_reprojection_px: float = 12.0,
    clean_margin_slack: float = 0.01,
    seed: int = 2026,
    progress_interval: int = 25,
) -> tuple[dict, list[dict]]:
    """Evaluate frozen A1 and its uplift on identical support/gate rows."""
    topks = tuple(sorted(set(int(value) for value in topks)))
    gate = [int(value) for value in gate_query_indices]
    pose_set = {int(value) for value in pose_query_indices}
    if not gate or not pose_set.issubset(set(gate)):
        raise ValueError("invalid gate or pose query partition")
    names = list(teacher["query_names"])
    cache = query_cache.get("queries", query_cache)
    anchor_indices = banks["anchor_indices"].long().to(device)
    supported = torch.zeros(int(teacher["anchor_count"]), dtype=torch.bool)
    supported[anchor_indices.cpu()] = True
    anchor_type = torch.as_tensor(state["anchor_type"]).long().cpu()
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    max_k = min(max(topks), anchor_indices.numel())
    map_banks = {"a1": banks["a1"].to(device), "uplift": banks["uplift"].to(device)}
    retrieval = {name: _empty_retrieval(topks) for name in map_banks}
    pose_rows = []
    query_angles = []
    query_gates = []
    clean = {
        "a1_clean_row_count": 0,
        "uplift_positive_retained_count": 0,
        "exact_winner_retained_count": 0,
        "new_false_attractor_count": 0,
        "clean_margin_violation_count": 0,
    }
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
            kernels=adapter.context_kernels,
            context_mode=adapter.context_mode,
        )
        base, _ = metric(raw)
        uplifted, angle_vectors, gates = adapter(base, tokens)
        query_angles.extend(angle_vectors.norm(dim=1).cpu().tolist())
        query_gates.extend(gates[:, 0].cpu().tolist())
        query_features = {"a1": base, "uplift": uplifted}

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

        scores = {key: value @ map_banks[key].T for key, value in query_features.items()}
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
        uplift_winners = winners["uplift"].tolist()
        clean_rows = []
        for row_index, (a1_winner, uplift_winner) in enumerate(
            zip(a1_winners, uplift_winners)
        ):
            if a1_winner not in positive_sets[row_index]:
                continue
            clean_rows.append(row_index)
            clean["a1_clean_row_count"] += 1
            if uplift_winner in positive_sets[row_index]:
                clean["uplift_positive_retained_count"] += 1
            elif uplift_winner not in ambiguous_sets[row_index]:
                clean["new_false_attractor_count"] += 1
            if uplift_winner == a1_winner:
                clean["exact_winner_retained_count"] += 1
        if clean_rows:
            clean_index = torch.as_tensor(clean_rows, device=device).long()
            clean_anchor = local_winners["a1"][clean_index]
            a1_top2 = torch.topk(scores["a1"][clean_index], k=2, dim=1).values
            base_margin = a1_top2[:, 0] - a1_top2[:, 1]
            uplift_scores = scores["uplift"][clean_index]
            clean_score = uplift_scores.gather(1, clean_anchor[:, None])[:, 0]
            masked = uplift_scores.clone()
            masked.scatter_(1, clean_anchor[:, None], -torch.inf)
            uplift_margin = clean_score - masked.max(dim=1).values
            clean["clean_margin_violation_count"] += int(
                (
                    uplift_margin
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
                    "event": "metric_context_gate",
                    "queries_complete": completed,
                    "query_count": len(gate),
                },
                flush=True,
            )
    angles = np.asarray(query_angles, dtype=np.float64)
    gates = np.asarray(query_gates, dtype=np.float64)
    report = {
        name: summarize_retrieval(counts, topks)
        for name, counts in retrieval.items()
    }
    clean_count = max(int(clean["a1_clean_row_count"]), 1)
    report["clean_preservation"] = {
        **clean,
        "positive_retention_percent": float(
            100.0 * clean["uplift_positive_retained_count"] / clean_count
        ),
        "exact_winner_retention_percent": float(
            100.0 * clean["exact_winner_retained_count"] / clean_count
        ),
        "new_false_attractor_percent": float(
            100.0 * clean["new_false_attractor_count"] / clean_count
        ),
        "clean_margin_violation_percent": float(
            100.0 * clean["clean_margin_violation_count"] / clean_count
        ),
    }
    report["adapter_diagnostics"] = {
        "query_angle_rad_mean": float(angles.mean()) if angles.size else 0.0,
        "query_angle_rad_p90": (
            float(np.percentile(angles, 90)) if angles.size else 0.0
        ),
        "query_angle_rad_maximum": float(angles.max()) if angles.size else 0.0,
        "query_gate_mean": float(gates.mean()) if gates.size else 0.0,
        "query_gate_p10": (
            float(np.percentile(gates, 10)) if gates.size else 0.0
        ),
    }
    report["additive_counts"] = retrieval
    return report, pose_rows


def summarize_metric_context_pose(pose_rows: list[dict]) -> dict:
    output = {name: _pose_summary(pose_rows, name) for name in ("a1", "uplift")}
    for name in output:
        valid = [
            row
            for row in pose_rows
            if not row[f"{name}_failed"]
        ]
        for threshold in (1.0, 2.0, 5.0):
            count = sum(
                row[f"{name}_te_cm"] <= threshold
                and row[f"{name}_ae_deg"] <= threshold
                for row in valid
            )
            output[name][f"recall_{int(threshold)}cm_{int(threshold)}deg_percent"] = (
                100.0 * count / max(len(pose_rows), 1)
            )
    return output


def compare_metric_context_protocols(retrieval: dict, pose: dict) -> dict:
    base_r1 = float(retrieval["a1"]["positive_recall_at_k"]["1"])
    uplift_r1 = float(retrieval["uplift"]["positive_recall_at_k"]["1"])
    base_pose = pose["a1"]
    uplift_pose = pose["uplift"]
    if int(base_pose.get("query_count", 0)) == 0:
        return {
            "uplift_minus_a1_top1_positive_recall_percentage_points": float(
                100.0 * (uplift_r1 - base_r1)
            ),
            "mapping_gate_pass": False,
            "routing_verdict": "pose_replay_skipped_no_final_routing_verdict",
        }

    def relative(key: str) -> float:
        base = float(base_pose.get(key, 0.0))
        return float((float(uplift_pose.get(key, 0.0)) - base) / max(base, 1e-12))

    recall_delta = float(
        uplift_pose["recall_2cm_2deg_percent"]
        - base_pose["recall_2cm_2deg_percent"]
    )
    tail_improves = all(
        float(uplift_pose[key]) <= float(base_pose[key])
        for key in ("mean_te_cm", "cvar95_te_cm", "mean_hypotheses")
    )
    no_new_catastrophe = int(uplift_pose["catastrophic_100cm_count"]) <= int(
        base_pose["catastrophic_100cm_count"]
    )
    go = recall_delta >= 0.0 and tail_improves and no_new_catastrophe
    return {
        "uplift_minus_a1_top1_positive_recall_percentage_points": float(
            100.0 * (uplift_r1 - base_r1)
        ),
        "uplift_minus_a1_recall_2cm_2deg_percentage_points": recall_delta,
        "uplift_relative_mean_te": relative("mean_te_cm"),
        "uplift_relative_p90_te": relative("p90_te_cm"),
        "uplift_relative_cvar95_te": relative("cvar95_te_cm"),
        "uplift_relative_mean_hypotheses": relative("mean_hypotheses"),
        "no_new_catastrophic_pose": bool(no_new_catastrophe),
        "mapping_gate_pass": bool(go),
        "routing_verdict": (
            "advance_metric_context_to_sentinel_scenes"
            if go
            else "hold_metric_context_before_topology_changes"
        ),
    }
