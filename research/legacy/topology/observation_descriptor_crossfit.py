"""Trajectory-block cross-fit gate for observation descriptor materialization.

The P6.0 audit measures observation consistency on the same observations used
to build the descriptor.  This module instead builds a descriptor from one
set of mapping trajectories and evaluates exact global retrieval on disjoint
held-out trajectories.  It is deliberately audit-only: no map tensor is
mutated and no descriptor is authorized for deployment by this module.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from topology.observation_descriptor import (
    _quantiles,
    _trajectory_ids,
    _validate_registry,
    robust_observation_fusion,
)


SCHEMA = "lafgs_observation_descriptor_crossfit_audit"
VERSION = 1


def _trajectory_labels(query_names: list[str]) -> list[str]:
    return [
        str(name).replace("\\", "/").split("/", 1)[0]
        for name in query_names
    ]


def _extract_observations(
    registry: Mapping, query_cache: Mapping, feature_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = query_cache.get("queries", query_cache)
    if not isinstance(cache, Mapping):
        raise ValueError("query cache must be a mapping")
    names = list(registry["query_names"])
    queries = torch.as_tensor(registry["observation_query_indices"]).long()
    keypoints = torch.as_tensor(registry["observation_keypoint_indices"]).long()
    descriptors = torch.zeros((queries.numel(), feature_dim), dtype=torch.float16)
    confidence = torch.ones(queries.numel(), dtype=torch.float32)
    valid = torch.zeros(queries.numel(), dtype=torch.bool)
    for query in torch.unique(queries, sorted=True).tolist():
        positions = torch.nonzero(queries == int(query), as_tuple=False).flatten()
        name = names[int(query)]
        if name not in cache:
            raise ValueError(f"query cache is missing mapping query: {name}")
        payload = cache[name]
        source = torch.as_tensor(payload["native_descriptors"])
        if source.ndim != 2 or int(source.shape[1]) != int(feature_dim):
            raise ValueError("query descriptor dimension does not match Anchor map")
        rows = keypoints[positions]
        if rows.numel() and (
            int(rows.min()) < 0 or int(rows.max()) >= int(source.shape[0])
        ):
            raise ValueError("Anchor observation references an invalid keypoint row")
        selected = source[rows].float()
        finite = torch.isfinite(selected).all(dim=1) & (selected.norm(dim=1) > 0)
        descriptors[positions] = selected.half()
        valid[positions] = finite
        if "native_scores" in payload:
            scores = torch.as_tensor(payload["native_scores"])[rows].float()
            confidence[positions] = torch.where(
                torch.isfinite(scores) & (scores > 0), scores, torch.ones_like(scores)
            )
    return descriptors, confidence, valid


@torch.inference_mode()
def _metric_descriptor(metric, descriptor: torch.Tensor) -> torch.Tensor:
    transformed = metric(descriptor)
    if isinstance(transformed, tuple):
        transformed = transformed[0]
    return F.normalize(torch.as_tensor(transformed).float(), dim=1)


def _aggregate_mean(
    anchor: torch.Tensor, value: torch.Tensor, anchor_count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(anchor_count, dtype=torch.float64)
    count = torch.zeros(anchor_count, dtype=torch.long)
    total.index_add_(0, anchor.cpu(), value.double().cpu())
    count.index_add_(0, anchor.cpu(), torch.ones_like(anchor.cpu()))
    mean = torch.full((anchor_count,), float("nan"), dtype=torch.float32)
    populated = count > 0
    mean[populated] = (total[populated] / count[populated]).float()
    return mean, count


@torch.inference_mode()
def _evaluate_direction(
    *,
    support_fold: int,
    edge_fold: torch.Tensor,
    edge_anchor: torch.Tensor,
    observation_descriptor: torch.Tensor,
    observation_valid: torch.Tensor,
    deployment_bank: torch.Tensor,
    fold_bank: torch.Tensor,
    support_eligible: torch.Tensor,
    metric,
    device: torch.device,
    score_chunk: int,
) -> dict:
    held_out = 1 - int(support_fold)
    evaluate_edge = (
        observation_valid
        & (edge_fold == held_out)
        & support_eligible[edge_anchor]
    )
    selected_edges = torch.nonzero(evaluate_edge, as_tuple=False).flatten()
    selected_anchor = edge_anchor[selected_edges]
    anchor_count = int(deployment_bank.shape[0])
    variant_bank = deployment_bank.clone()
    variant_bank[support_eligible] = fold_bank[support_fold, support_eligible]
    baseline_margin_parts = []
    variant_margin_parts = []
    baseline_correct_parts = []
    variant_correct_parts = []
    positive_delta_parts = []
    for begin in range(0, int(selected_edges.numel()), int(score_chunk)):
        rows = selected_edges[begin : begin + int(score_chunk)]
        target = edge_anchor[rows].to(device)
        raw = observation_descriptor[rows].float().to(device)
        query = _metric_descriptor(metric, raw)
        baseline_score = query @ deployment_bank.T
        variant_score = query @ variant_bank.T
        row = torch.arange(target.numel(), device=device)
        baseline_positive = baseline_score[row, target]
        variant_positive = variant_score[row, target]
        baseline_competitor = baseline_score.clone()
        variant_competitor = variant_score.clone()
        baseline_competitor[row, target] = -torch.inf
        variant_competitor[row, target] = -torch.inf
        baseline_margin_parts.append(
            (baseline_positive - baseline_competitor.max(dim=1).values).cpu()
        )
        variant_margin_parts.append(
            (variant_positive - variant_competitor.max(dim=1).values).cpu()
        )
        baseline_correct_parts.append((baseline_score.argmax(dim=1) == target).cpu())
        variant_correct_parts.append((variant_score.argmax(dim=1) == target).cpu())
        positive_delta_parts.append((variant_positive - baseline_positive).cpu())
    if selected_edges.numel():
        baseline_margin = torch.cat(baseline_margin_parts)
        variant_margin = torch.cat(variant_margin_parts)
        baseline_correct = torch.cat(baseline_correct_parts).float()
        variant_correct = torch.cat(variant_correct_parts).float()
        positive_delta = torch.cat(positive_delta_parts)
    else:
        baseline_margin = torch.empty(0)
        variant_margin = torch.empty(0)
        baseline_correct = torch.empty(0)
        variant_correct = torch.empty(0)
        positive_delta = torch.empty(0)
    baseline_margin_by_anchor, evaluation_count = _aggregate_mean(
        selected_anchor, baseline_margin, anchor_count
    )
    variant_margin_by_anchor, _ = _aggregate_mean(
        selected_anchor, variant_margin, anchor_count
    )
    baseline_r1_by_anchor, _ = _aggregate_mean(
        selected_anchor, baseline_correct, anchor_count
    )
    variant_r1_by_anchor, _ = _aggregate_mean(
        selected_anchor, variant_correct, anchor_count
    )
    margin_delta = variant_margin - baseline_margin
    return {
        "support_fold": int(support_fold),
        "held_out_fold": int(held_out),
        "eligible_anchor_count": int(support_eligible.sum()),
        "held_out_observation_count": int(selected_edges.numel()),
        "baseline_r1": (
            float(baseline_correct.mean()) if baseline_correct.numel() else None
        ),
        "variant_r1": (
            float(variant_correct.mean()) if variant_correct.numel() else None
        ),
        "r1_delta": (
            float(variant_correct.mean() - baseline_correct.mean())
            if baseline_correct.numel()
            else None
        ),
        "false_winner_delta": (
            int((1.0 - variant_correct).sum() - (1.0 - baseline_correct).sum())
            if baseline_correct.numel()
            else 0
        ),
        "positive_cosine_delta": _quantiles(positive_delta),
        "positive_margin_delta": _quantiles(margin_delta),
        "anchors_with_mean_margin_improvement": int(
            (
                torch.isfinite(variant_margin_by_anchor)
                & (variant_margin_by_anchor > baseline_margin_by_anchor)
            ).sum()
        ),
        "evaluation_count_by_anchor": evaluation_count,
        "baseline_margin_by_anchor": baseline_margin_by_anchor,
        "variant_margin_by_anchor": variant_margin_by_anchor,
        "baseline_r1_by_anchor": baseline_r1_by_anchor,
        "variant_r1_by_anchor": variant_r1_by_anchor,
    }


@torch.inference_mode()
def audit_crossfit_observation_descriptors(
    registry: Mapping,
    query_cache: Mapping,
    metric,
    *,
    trim_fraction: float = 0.2,
    minimum_support_queries: int = 2,
    minimum_support_strata: int = 2,
    minimum_direction_cosine: float = 0.65,
    fold_a_trajectories: tuple[str, ...] | None = None,
    score_chunk: int = 256,
    device: str | torch.device = "cpu",
) -> dict:
    """Evaluate support-fold fused descriptors on held-out trajectories."""
    anchor_count, offsets = _validate_registry(registry)
    if int(minimum_support_queries) < 1 or int(minimum_support_strata) < 1:
        raise ValueError("minimum cross-fit support must be positive")
    if not -1.0 <= float(minimum_direction_cosine) <= 1.0:
        raise ValueError("minimum direction cosine must lie in [-1, 1]")
    if int(score_chunk) < 1:
        raise ValueError("score_chunk must be positive")
    names = list(registry["query_names"])
    trajectory = _trajectory_ids(names)
    unique_trajectory = torch.unique(trajectory[trajectory >= 0], sorted=True)
    labels = _trajectory_labels(names)
    label_by_id = {
        int(value): labels[int(torch.nonzero(trajectory == value)[0])]
        for value in unique_trajectory.tolist()
    }
    base = {
        "schema": SCHEMA,
        "version": VERSION,
        "uses_test_queries": False,
        "audit_only": True,
        "deployment_descriptor_mutated": False,
        "replacement_scope": "surface_only_track_unchanged",
        "descriptor_space": {
            "support_descriptor": "frozen_metric(robust_fuse(raw_native_observations))",
            "held_out_query_descriptor": "frozen_metric(raw_native_observation)",
            "baseline_anchor_descriptor": "current_deployment_anchor_feature",
            "retrieval": "exact_global_cosine_top1",
        },
        "fold_policy": (
            "explicit_trajectory_partition"
            if fold_a_trajectories is not None
            else "sorted_trajectory_round_robin_ab"
        ),
        "trajectory_count": int(unique_trajectory.numel()),
    }
    if unique_trajectory.numel() < 2:
        return {
            **base,
            "crossfit_available": False,
            "blocker": "requires_at_least_two_mapping_trajectories",
            "trajectory_labels": sorted(set(labels)),
            "report": {
                "anchor_count": anchor_count,
                "surface_anchor_count": int(
                    (torch.as_tensor(registry["anchor_type"]).long() == 0).sum()
                ),
            },
        }
    trajectory_fold = torch.full_like(trajectory, -1)
    fold_labels = {0: [], 1: []}
    available_labels = {
        label_by_id[int(value)] for value in unique_trajectory.tolist()
    }
    requested_fold_a = set(fold_a_trajectories or ())
    if fold_a_trajectories is not None:
        unknown = sorted(requested_fold_a - available_labels)
        if unknown:
            raise ValueError(f"unknown fold-A trajectory: {unknown[0]}")
        if not requested_fold_a or requested_fold_a == available_labels:
            raise ValueError(
                "explicit trajectory partition must leave both folds non-empty"
            )
    for order, value in enumerate(unique_trajectory.tolist()):
        label = label_by_id[int(value)]
        fold = (
            int(label not in requested_fold_a)
            if fold_a_trajectories is not None
            else int(order % 2)
        )
        trajectory_fold[trajectory == int(value)] = fold
        fold_labels[fold].append(label)
    feature_dim = int(torch.as_tensor(registry["anchor_features"]).shape[1])
    observation_descriptor, observation_confidence, observation_valid = (
        _extract_observations(registry, query_cache, feature_dim)
    )
    edge_query = torch.as_tensor(registry["observation_query_indices"]).long()
    edge_fold = trajectory_fold[edge_query]
    edge_anchor = torch.repeat_interleave(
        torch.arange(anchor_count), offsets[1:] - offsets[:-1]
    )
    query_groups = torch.as_tensor(
        registry.get("query_group_ids", torch.full((len(names),), -1))
    ).long()
    device = torch.device(device)
    metric = metric.to(device).eval()
    deployment_bank = F.normalize(
        torch.as_tensor(registry["anchor_features"], device=device).float(), dim=1
    )
    fold_bank = torch.zeros((2, anchor_count, feature_dim), device=device)
    fold_valid = torch.zeros((2, anchor_count), dtype=torch.bool)
    fold_query_count = torch.zeros((2, anchor_count), dtype=torch.long)
    fold_stratum_count = torch.zeros((2, anchor_count), dtype=torch.long)
    fold_observation_count = torch.zeros((2, anchor_count), dtype=torch.long)
    for anchor in range(anchor_count):
        begin, end = int(offsets[anchor]), int(offsets[anchor + 1])
        for fold in (0, 1):
            keep = observation_valid[begin:end] & (edge_fold[begin:end] == fold)
            fold_observation_count[fold, anchor] = int(keep.sum())
            if not bool(keep.any()):
                continue
            rows = torch.arange(begin, end)[keep]
            queries = edge_query[rows]
            fused, diagnostics = robust_observation_fusion(
                observation_descriptor[rows].float(),
                queries,
                query_groups[queries],
                trajectory[queries],
                observation_confidence[rows],
                trim_fraction=float(trim_fraction),
            )
            fold_bank[fold, anchor] = _metric_descriptor(
                metric, fused[None].to(device)
            )[0]
            fold_valid[fold, anchor] = True
            fold_query_count[fold, anchor] = int(torch.unique(queries).numel())
            fold_stratum_count[fold, anchor] = int(diagnostics["stratum_count"])
    surface = torch.as_tensor(registry["anchor_type"]).long() == 0
    support_eligible = []
    directions = []
    for fold in (0, 1):
        eligible = (
            surface
            & fold_valid[fold]
            & (fold_query_count[fold] >= int(minimum_support_queries))
            & (fold_stratum_count[fold] >= int(minimum_support_strata))
            & (fold_observation_count[1 - fold] > 0)
        )
        support_eligible.append(eligible)
        directions.append(
            _evaluate_direction(
                support_fold=fold,
                edge_fold=edge_fold,
                edge_anchor=edge_anchor,
                observation_descriptor=observation_descriptor,
                observation_valid=observation_valid,
                deployment_bank=deployment_bank,
                fold_bank=fold_bank,
                support_eligible=eligible,
                metric=metric,
                device=device,
                score_chunk=int(score_chunk),
            )
        )
    crossfold_cosine = torch.full((anchor_count,), float("nan"))
    both_valid = fold_valid[0] & fold_valid[1]
    crossfold_cosine[both_valid] = (
        fold_bank[0, both_valid] * fold_bank[1, both_valid]
    ).sum(dim=1).cpu()
    bidirectional = support_eligible[0] & support_eligible[1]
    stable = bidirectional & (crossfold_cosine >= float(minimum_direction_cosine))
    for direction in directions:
        evaluated = direction["evaluation_count_by_anchor"] > 0
        stable &= evaluated
        stable &= direction["variant_margin_by_anchor"] >= direction[
            "baseline_margin_by_anchor"
        ]
        stable &= direction["variant_r1_by_anchor"] >= direction[
            "baseline_r1_by_anchor"
        ]
    stable_cosine = crossfold_cosine[bidirectional]
    direction_non_degrading = []
    for direction in directions:
        direction_non_degrading.append(
            (direction["evaluation_count_by_anchor"] > 0)
            & (
                direction["variant_margin_by_anchor"]
                >= direction["baseline_margin_by_anchor"]
            )
            & (
                direction["variant_r1_by_anchor"]
                >= direction["baseline_r1_by_anchor"]
            )
        )
    report_directions = []
    for direction in directions:
        report_directions.append(
            {
                key: value
                for key, value in direction.items()
                if not isinstance(value, torch.Tensor)
            }
        )
    return {
        **base,
        "crossfit_available": True,
        "blocker": None,
        "trajectory_labels": sorted(set(labels)),
        "fold_trajectory_labels": fold_labels,
        "minimum_support_queries": int(minimum_support_queries),
        "minimum_support_strata": int(minimum_support_strata),
        "minimum_direction_cosine": float(minimum_direction_cosine),
        "trim_fraction": float(trim_fraction),
        "fold_metric_descriptor": fold_bank.cpu().half(),
        "fold_descriptor_valid": fold_valid,
        "fold_observation_count": fold_observation_count,
        "fold_query_count": fold_query_count,
        "fold_stratum_count": fold_stratum_count,
        "crossfold_descriptor_cosine": crossfold_cosine,
        "bidirectional_eligible_mask": bidirectional,
        "stable_surface_mask": stable,
        "directions": directions,
        "report": {
            "anchor_count": anchor_count,
            "surface_anchor_count": int(surface.sum()),
            "bidirectional_eligible_surface_count": int(bidirectional.sum()),
            "direction_consistent_surface_count": int(
                (
                    bidirectional
                    & (
                        crossfold_cosine
                        >= float(minimum_direction_cosine)
                    )
                ).sum()
            ),
            "fold0_to_fold1_non_degrading_surface_count": int(
                (bidirectional & direction_non_degrading[0]).sum()
            ),
            "fold1_to_fold0_non_degrading_surface_count": int(
                (bidirectional & direction_non_degrading[1]).sum()
            ),
            "both_directions_non_degrading_surface_count": int(
                (
                    bidirectional
                    & direction_non_degrading[0]
                    & direction_non_degrading[1]
                ).sum()
            ),
            "stable_surface_count": int(stable.sum()),
            "stable_fraction_of_bidirectional": (
                float(stable.sum() / bidirectional.sum())
                if bool(bidirectional.any())
                else None
            ),
            "crossfold_descriptor_cosine": _quantiles(stable_cosine),
            "directions": report_directions,
        },
    }
