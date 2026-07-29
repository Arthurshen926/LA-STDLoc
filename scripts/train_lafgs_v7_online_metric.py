#!/usr/bin/env python3
"""Online-refreshed shared-metric reconstruction for a Track-centric map."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.shared_metric import (
    NativeNullHead,
    SharedLowRankMetric,
    build_native_null_features,
    select_native_matchable_rows,
)
from utils.pose_utils import cal_pose_error, solve_pose


def _first_k(values: torch.Tensor, mask: torch.Tensor, width: int) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("candidate values and mask must align")
    width = min(int(width), values.shape[1])
    output = torch.full(
        (values.shape[0], width), -1, dtype=values.dtype, device=values.device
    )
    rank = mask.to(torch.int64).cumsum(dim=1) - 1
    selected = mask & (rank < width)
    rows, columns = torch.nonzero(selected, as_tuple=True)
    output[rows, rank[rows, columns]] = values[rows, columns]
    return output


def _query_index_remap(
    source_names: list[str], target_names: list[str]
) -> torch.Tensor:
    if len(set(source_names)) != len(source_names):
        raise ValueError("source query names must be unique")
    if len(set(target_names)) != len(target_names):
        raise ValueError("target query names must be unique")
    target_by_name = {name: index for index, name in enumerate(target_names)}
    if set(source_names) != set(target_names):
        raise ValueError("source and target query-name sets differ")
    return torch.as_tensor(
        [target_by_name[name] for name in source_names], dtype=torch.long
    )


def _track_observations(
    payload: dict,
    track_to_local: torch.Tensor,
    query_index_remap: torch.Tensor | None = None,
):
    by_query: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    tracks = payload["tracks"]
    for track, query, keypoint in zip(
        tracks["track_index"].tolist(),
        tracks["query_index"].tolist(),
        tracks["keypoint_index"].tolist(),
    ):
        local = int(track_to_local[int(track)])
        if local >= 0:
            target_query = (
                int(query_index_remap[int(query)])
                if query_index_remap is not None
                else int(query)
            )
            by_query[target_query][int(keypoint)].append(local)
    return by_query


def _csr_first_k(
    offsets: torch.Tensor,
    indices: torch.Tensor,
    width: int,
) -> torch.Tensor:
    offsets = torch.as_tensor(offsets).long().reshape(-1)
    indices = torch.as_tensor(indices).long().reshape(-1)
    row_count = offsets.numel() - 1
    output = torch.full((row_count, int(width)), -1, dtype=torch.long)
    counts = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(torch.arange(row_count), counts)
    rank = torch.arange(indices.numel()) - offsets[rows]
    keep = rank < int(width)
    output[rows[keep], rank[keep]] = indices[keep]
    return output


def _csr_first_k_values(
    offsets: torch.Tensor,
    values: torch.Tensor,
    width: int,
    *,
    fill: float,
) -> torch.Tensor:
    offsets = torch.as_tensor(offsets).long().reshape(-1)
    values = torch.as_tensor(values).float().reshape(-1)
    row_count = offsets.numel() - 1
    output = torch.full((row_count, int(width)), float(fill))
    counts = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(torch.arange(row_count), counts)
    rank = torch.arange(values.numel()) - offsets[rows]
    keep = rank < int(width)
    output[rows[keep], rank[keep]] = values[keep]
    return output


def _csr_topk_by_values(
    offsets: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.as_tensor(offsets).long().reshape(-1)
    indices = torch.as_tensor(indices).long().reshape(-1)
    values = torch.as_tensor(values).float().reshape(-1)
    if indices.numel() != values.numel():
        raise ValueError("CSR indices and ranking values must align")
    row_count = offsets.numel() - 1
    output_indices = torch.full((row_count, int(width)), -1, dtype=torch.long)
    output_values = torch.ones((row_count, int(width)), dtype=torch.float32)
    counts = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(torch.arange(row_count), counts)
    value_order = torch.argsort(values, descending=True, stable=True)
    row_order = torch.argsort(rows[value_order], stable=True)
    order = value_order[row_order]
    sorted_rows = rows[order]
    starts = torch.cumsum(counts, dim=0) - counts
    ranks = torch.arange(indices.numel()) - torch.repeat_interleave(starts, counts)
    keep = ranks < int(width)
    output_indices[sorted_rows[keep], ranks[keep]] = indices[order[keep]]
    output_values[sorted_rows[keep], ranks[keep]] = values[order[keep]]
    return output_indices, output_values


def _replace_refreshed_pairs(
    clean_pairs: dict,
    harmful_pairs: dict,
    query_indices: list[int],
    refreshed_clean: dict,
    refreshed_harmful: dict,
) -> dict:
    old_clean = sum(len(clean_pairs.get(int(query), {})) for query in query_indices)
    old_harmful = sum(
        len(harmful_pairs.get(int(query), {})) for query in query_indices
    )
    for query in query_indices:
        clean_pairs.pop(int(query), None)
        harmful_pairs.pop(int(query), None)
    for query, values in refreshed_clean.items():
        clean_pairs[int(query)] = dict(values)
    for query, values in refreshed_harmful.items():
        harmful_pairs[int(query)] = dict(values)
    new_clean = sum(len(clean_pairs.get(int(query), {})) for query in query_indices)
    new_harmful = sum(
        len(harmful_pairs.get(int(query), {})) for query in query_indices
    )
    return {
        "old_clean_pair_count": old_clean,
        "old_harmful_pair_count": old_harmful,
        "new_clean_pair_count": new_clean,
        "new_harmful_pair_count": new_harmful,
    }


def _build_rotating_shards(
    groups: torch.Tensor, shard_count: int
) -> list[list[int]]:
    """Round-robin every stable query group across deterministic shards."""
    groups = torch.as_tensor(groups).long().reshape(-1)
    shard_count = max(min(int(shard_count), groups.numel()), 1)
    shards: list[list[int]] = [[] for _ in range(shard_count)]
    for group in torch.unique(groups, sorted=True).tolist():
        indices = torch.nonzero(
            groups == int(group), as_tuple=False
        ).reshape(-1)
        for offset, query_index in enumerate(indices.tolist()):
            shards[offset % shard_count].append(int(query_index))
    for shard in shards:
        shard.sort()
    if sorted(index for shard in shards for index in shard) != list(
        range(groups.numel())
    ):
        raise RuntimeError("rotating query shards must cover every query once")
    return shards


def _group_pose_risk(errors_cm: list[float]) -> float:
    errors = torch.as_tensor(errors_cm, dtype=torch.float32)
    if errors.numel() == 0:
        return 0.0
    smooth_mean = torch.log1p(errors / 10.0).mean()
    tail_count = max(int(math.ceil(0.2 * errors.numel())), 1)
    tail = torch.topk(errors, k=tail_count).values.mean() / 20.0
    near_five = F.softplus((errors - 5.0) / 2.0).mean() / 5.0
    return float(smooth_mean + 0.5 * tail + 0.5 * near_five)


def _build_training_records(
    graph: dict,
    payload: dict,
    state: dict,
    max_positives: int,
    *,
    device: torch.device | str = "cpu",
    query_chunk_size: int = 32,
    positive_teacher: dict | None = None,
    critical_teacher: dict | None = None,
    critical_pair_power: float = 1.0,
    critical_row_power: float = 1.0,
):
    metadata = state["track_centric_reconstruction"]
    track_indices = torch.as_tensor(metadata["track_indices"]).long()
    base_rows = torch.as_tensor(metadata["base_canonical_rows"]).long()
    track_count = int(track_indices.numel())
    canonical_count = int(graph["anchor_count"])
    canonical_to_local = torch.full(
        (canonical_count,), -1, dtype=torch.long
    )
    canonical_to_local[base_rows] = (
        torch.arange(base_rows.numel()) + track_count
    )
    payload_track_count = int(
        payload["track_geometry"]["triangulated_xyz"].shape[0]
    )
    track_to_local = torch.full(
        (payload_track_count,), -1, dtype=torch.long
    )
    track_to_local[track_indices] = torch.arange(track_count)
    graph_records = graph["records"]
    payload_to_graph = _query_index_remap(
        payload["query_names"], graph["query_names"]
    )
    exact = (
        {}
        if positive_teacher is not None
        else _track_observations(
            payload, track_to_local, query_index_remap=payload_to_graph
        )
    )
    row_counts = [
        int(torch.as_tensor(record["query_rows"]).numel())
        for record in graph_records
    ]
    build_device = torch.device(device)
    canonical_to_local_build = canonical_to_local.to(build_device)
    positive_blocks = []
    positive_weight_blocks = []
    row_weight_blocks = []
    legal4_blocks = []
    if positive_teacher is not None:
        if int(positive_teacher["anchor_count"]) != int(
            state["anchor_xyz"].shape[0]
        ):
            raise ValueError("complete positive teacher does not align with map")
        if list(positive_teacher["query_names"]) != list(graph["query_names"]):
            raise ValueError("complete positive teacher query order mismatch")
        if len(positive_teacher["records"]) != len(graph_records):
            raise ValueError("complete positive teacher query count mismatch")
        for graph_record, teacher_record in zip(
            graph_records, positive_teacher["records"]
        ):
            graph_rows = torch.as_tensor(
                graph_record["query_rows"]
            ).long()
            teacher_rows = torch.as_tensor(
                teacher_record["query_rows"]
            ).long()
            if not torch.equal(graph_rows, teacher_rows):
                raise ValueError("complete positive teacher row mismatch")
            if critical_teacher is not None:
                critical_record = critical_teacher["records"][
                    len(positive_blocks)
                ]
                if not torch.equal(
                    teacher_rows,
                    torch.as_tensor(critical_record["query_rows"]).long(),
                ):
                    raise ValueError("pose-critical teacher row mismatch")
                ranked_indices, ranked_weights = _csr_topk_by_values(
                    teacher_record["positive_offsets"],
                    teacher_record["positive_indices"],
                    critical_record["positive_weights"],
                    max_positives,
                )
                positive_blocks.append(ranked_indices)
                positive_weight_blocks.append(
                    ranked_weights.clamp_min(1e-8).pow(
                        float(critical_pair_power)
                    )
                )
                row_weight_blocks.append(
                    torch.as_tensor(critical_record["row_weights"])
                    .float()
                    .clamp_min(1e-8)
                    .pow(float(critical_row_power))
                )
            else:
                positive_blocks.append(
                    _csr_first_k(
                        teacher_record["positive_offsets"],
                        teacher_record["positive_indices"],
                        max_positives,
                    )
                )
            ambiguous_offsets = torch.as_tensor(
                teacher_record["ambiguous_offsets"]
            ).long()
            legal4_blocks.append(
                (ambiguous_offsets[1:] - ambiguous_offsets[:-1]) > 0
            )
    else:
        query_chunk_size = max(int(query_chunk_size), 1)
        for start in range(0, len(graph_records), query_chunk_size):
            chunk = graph_records[start : start + query_chunk_size]
            chunk_counts = row_counts[start : start + query_chunk_size]
            candidates = torch.cat(
                [
                    torch.as_tensor(record["top_indices"]).long()
                    for record in chunk
                ],
                dim=0,
            ).to(build_device)
            flags = torch.cat(
                [
                    torch.as_tensor(record["legal_flags"]).to(torch.uint8)
                    for record in chunk
                ],
                dim=0,
            ).to(build_device)
            candidate_valid = (candidates >= 0) & (
                candidates < canonical_count
            )
            local = canonical_to_local_build[
                candidates.clamp(min=0, max=canonical_count - 1)
            ]
            local = torch.where(
                candidate_valid, local, torch.full_like(local, -1)
            )
            chunk_positives = _first_k(
                local,
                (local >= 0) & candidate_valid & ((flags & 2) != 0),
                max_positives,
            ).cpu()
            chunk_legal4 = (
                candidate_valid & ((flags & 4) != 0)
            ).any(dim=1).cpu()
            positive_blocks.extend(chunk_positives.split(chunk_counts))
            legal4_blocks.extend(chunk_legal4.split(chunk_counts))
    query_bins = torch.empty_like(torch.as_tensor(payload["query_bins"]).long())
    query_bins[payload_to_graph] = torch.as_tensor(payload["query_bins"]).long()
    del canonical_to_local_build
    records = []
    positive_rows = 0
    for query_index, record in enumerate(graph_records):
        cache_rows = torch.as_tensor(record["query_rows"]).long()
        positives = positive_blocks[query_index].clone()
        positive_weights = (
            positive_weight_blocks[query_index].clone()
            if positive_weight_blocks
            else torch.ones_like(positives, dtype=torch.float32)
        )
        row_weights = (
            row_weight_blocks[query_index].clone()
            if row_weight_blocks
            else torch.ones(positives.shape[0], dtype=torch.float32)
        )
        query_exact = exact.get(query_index, {})
        if (
            positive_teacher is None
            and query_exact
            and cache_rows.numel()
        ):
            row_lookup = {
                int(keypoint): row
                for row, keypoint in enumerate(cache_rows.tolist())
            }
            for keypoint, tracks in query_exact.items():
                output_row = row_lookup.get(int(keypoint))
                if output_row is None:
                    continue
                for track in tracks:
                    current = positives[output_row]
                    if bool((current == int(track)).any()):
                        continue
                    empty = torch.nonzero(
                        current < 0, as_tuple=False
                    ).reshape(-1)
                    column = int(empty[0]) if empty.numel() else -1
                    current[column] = int(track)
        valid = (positives >= 0).any(dim=1)
        canonical_legal4 = legal4_blocks[query_index]
        null_weight = torch.where(
            valid,
            torch.ones_like(valid, dtype=torch.float32),
            torch.where(
                canonical_legal4,
                torch.full_like(valid, 0.25, dtype=torch.float32),
                torch.ones_like(valid, dtype=torch.float32),
            ),
        )
        positive_rows += int(valid.sum())
        records.append(
            {
                "deployment_rows": cache_rows,
                "cache_rows": cache_rows,
                "positives": positives,
                "positive_weights": positive_weights,
                "critical_row_weights": row_weights,
                "matchable": valid,
                "null_weight": null_weight,
                "group": int(query_bins[query_index]),
            }
        )
    return records, {
        "positive_rows": positive_rows,
        "track_anchor_count": track_count,
        "base_anchor_count": int(base_rows.numel()),
        "complete_positive_teacher_enabled": positive_teacher is not None,
        "pose_critical_teacher_enabled": critical_teacher is not None,
        "complete_positive_pair_count": int(
            positive_teacher["diagnostics"]["strong_pair_count"]
            if positive_teacher is not None
            else 0
        ),
        "query_groups": query_bins.tolist(),
    }


def _bounded_anchor_features(
    raw: torch.Tensor, residual: torch.Tensor, maximum: float
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.norm(residual, dim=1, keepdim=True)
    bounded = residual * torch.clamp(
        float(maximum) / norm.clamp_min(1e-8), max=1.0
    )
    return F.normalize(raw + bounded, dim=1), bounded


def _initial_anchor_features(state: dict, mode: str) -> tuple[torch.Tensor, str]:
    if mode not in {"current", "pre_metric_raw"}:
        raise ValueError("unsupported feature initialization mode")
    key = (
        "v7_metric_raw_features"
        if mode == "pre_metric_raw" and "v7_metric_raw_features" in state
        else "anchor_features"
    )
    return F.normalize(torch.as_tensor(state[key]).float(), dim=1), key


def _multi_positive_list_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    positives: torch.Tensor,
    topk: int,
    temperature: float,
    harmful_prior: torch.Tensor | None,
    harmful_weight: float,
    harmful_indices: torch.Tensor | None = None,
    positive_weights: torch.Tensor | None = None,
):
    scores = query @ bank.T
    top_scores, top_indices = torch.topk(
        scores, k=min(int(topk), bank.shape[0]), dim=1
    )
    safe = positives.clamp_min(0)
    positive_scores = torch.einsum(
        "bd,bpd->bp", query, bank[safe]
    )
    positive_mask = positives >= 0
    numerator = torch.logsumexp(
        (positive_scores / temperature).masked_fill(
            ~positive_mask, -torch.inf
        ),
        dim=1,
    )
    top_is_positive = (
        (top_indices[:, :, None] == positives[:, None, :])
        & positive_mask[:, None, :]
    ).any(dim=2)
    denominator_scores = torch.cat([top_scores, positive_scores], dim=1)
    denominator_mask = torch.cat(
        [
            ~top_is_positive,
            positive_mask,
        ],
        dim=1,
    )
    denominator = torch.logsumexp(
        (denominator_scores / temperature).masked_fill(
            ~denominator_mask, -torch.inf
        ),
        dim=1,
    )
    if positive_weights is None:
        positive_aggregate = numerator * float(temperature)
        list_loss = denominator * float(temperature) - positive_aggregate
    else:
        target = positive_weights.clamp_min(0) * positive_mask
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
        positive_aggregate = (target * positive_scores).sum(dim=1)
        list_loss = denominator * float(temperature) - positive_aggregate
    harmful_loss = torch.zeros_like(list_loss)
    if harmful_prior is not None:
        retrieved_harm = harmful_prior[top_indices]
        harmful_loss = harmful_loss + (
            torch.softmax(top_scores / temperature, dim=1) * retrieved_harm
        ).sum(dim=1)
    if harmful_indices is not None:
        harmful_valid = harmful_indices >= 0
        safe_harmful = harmful_indices.clamp_min(0)
        harmful_scores = torch.einsum(
            "bd,bhd->bh", query, bank[safe_harmful]
        )
        harmful_scores = harmful_scores.masked_fill(
            ~harmful_valid, -torch.inf
        )
        hardest_harmful = harmful_scores.max(dim=1).values
        has_harmful = harmful_valid.any(dim=1)
        harmful_loss = harmful_loss + torch.where(
            has_harmful,
            F.softplus(
                (hardest_harmful - positive_aggregate)
                / float(temperature)
            )
            * float(temperature),
            torch.zeros_like(harmful_loss),
        )
    return (
        list_loss + float(harmful_weight) * harmful_loss,
        top_indices,
        top_scores,
    )


def _basin_good_set_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    anchors: torch.Tensor,
    propensities: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    maximum_inverse_propensity: float,
) -> torch.Tensor:
    """Joint log-probability of the three assignments in each good basis."""
    if anchors.numel() == 0:
        return torch.zeros((), device=bank.device)
    set_count = int(anchors.shape[0])
    edge_loss = _multi_positive_list_loss(
        query.reshape(-1, query.shape[-1]),
        bank,
        anchors.reshape(-1, 1),
        topk,
        temperature,
        None,
        0.0,
    )[0].reshape(set_count, 3)
    propensity = propensities.float().clamp_min(1e-12)
    inverse = (
        propensity.median().clamp_min(1e-12) / propensity
    ).clamp_max(float(maximum_inverse_propensity))
    inverse = inverse / inverse.mean().clamp_min(1e-8)
    return (edge_loss.sum(dim=1) * inverse).sum() / inverse.sum().clamp_min(1e-8)


def _basin_counterfactual_blame_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    harmful_anchors: torch.Tensor,
    positive_anchors: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float,
    margin: float,
) -> torch.Tensor:
    """Suppress only the edge whose legal replacement fixes the pose basin."""
    if harmful_anchors.numel() == 0:
        return torch.zeros((), device=bank.device)
    harmful_score = torch.einsum("bd,bd->b", query, bank[harmful_anchors])
    positive_score = torch.einsum("bd,bd->b", query, bank[positive_anchors])
    normalized = weights.float().clamp_min(0)
    normalized = normalized / normalized.mean().clamp_min(1e-8)
    loss = F.softplus(
        (harmful_score - positive_score + float(margin))
        / float(temperature)
    ) * float(temperature)
    return (loss * normalized).sum() / normalized.sum().clamp_min(1e-8)


def _basin_good_margin_guard_loss(
    raw_query: torch.Tensor,
    raw_bank: torch.Tensor,
    adapted_query: torch.Tensor,
    adapted_bank: torch.Tensor,
    anchors: torch.Tensor,
    propensities: torch.Tensor,
    *,
    maximum_inverse_propensity: float,
    slack: float = 0.0,
) -> torch.Tensor:
    """Prevent a correct basis edge from losing its initial ranking margin."""
    if anchors.numel() == 0:
        return torch.zeros((), device=adapted_bank.device)
    flat_anchor = anchors.reshape(-1)

    def margins(query, bank):
        score = query.reshape(-1, query.shape[-1]) @ bank.T
        positive = score.gather(1, flat_anchor[:, None]).reshape(-1)
        score = score.scatter(1, flat_anchor[:, None], -torch.inf)
        negative = score.max(dim=1).values
        return (positive - negative).reshape(anchors.shape)

    baseline = margins(raw_query, raw_bank).detach()
    current = margins(adapted_query, adapted_bank)
    set_loss = F.relu(baseline - current + float(slack)).sum(dim=1)
    propensity = propensities.float().clamp_min(1e-12)
    inverse = (
        propensity.median().clamp_min(1e-12) / propensity
    ).clamp_max(float(maximum_inverse_propensity))
    inverse = inverse / inverse.mean().clamp_min(1e-8)
    return (set_loss * inverse).sum() / inverse.sum().clamp_min(1e-8)


def _basin_set_log_scores(
    query: torch.Tensor,
    bank: torch.Tensor,
    anchors: torch.Tensor,
    *,
    assignment_temperature: float,
) -> torch.Tensor:
    """Log-probability of each three-edge assignment under the full bank."""
    if anchors.numel() == 0:
        return torch.empty(0, device=bank.device)
    flat_query = query.reshape(-1, query.shape[-1])
    flat_anchor = anchors.reshape(-1)
    logits = (flat_query @ bank.T) / float(assignment_temperature)
    selected = logits.gather(1, flat_anchor[:, None]).reshape(-1)
    edge_log_probability = selected - torch.logsumexp(logits, dim=1)
    return edge_log_probability.reshape(anchors.shape).sum(dim=1)


def _basin_hyperedge_losses(
    query: torch.Tensor,
    bank: torch.Tensor,
    anchors: torch.Tensor,
    set_types: torch.Tensor,
    correct_basin: torch.Tensor,
    te_cm: torch.Tensor,
    re_deg: torch.Tensor,
    parent_set_index: torch.Tensor,
    propensities: torch.Tensor,
    *,
    assignment_temperature: float,
    basin_temperature: float,
    counterfactual_temperature: float,
    counterfactual_margin: float,
    translation_reward_scale_cm: float,
    rotation_reward_scale_deg: float,
    maximum_inverse_propensity: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Contrast successful P3P hyperedges and rank repaired near-miss sets."""
    if anchors.numel() == 0:
        zero = torch.zeros((), device=bank.device)
        return zero, zero, {"coarse": 0, "precision": 0, "strict": 0}
    set_scores = _basin_set_log_scores(
        query,
        bank,
        anchors,
        assignment_temperature=assignment_temperature,
    )
    types = set_types.long()
    correct = correct_basin.bool()
    good = correct & ((types == 0) | (types == 2))
    harmful = types == 1
    coarse = good & (te_cm <= 50.0) & (re_deg <= 5.0)
    precision = good & (te_cm <= 15.0) & (re_deg <= 2.0)
    strict = good & (te_cm <= 5.0) & (re_deg <= 5.0)
    propensity = propensities.float().clamp_min(1e-12)
    inverse = (
        propensity.median().clamp_min(1e-12) / propensity
    ).clamp_max(float(maximum_inverse_propensity))
    inverse = inverse / inverse.mean().clamp_min(1e-8)
    continuous_reward = torch.exp(
        -te_cm.clamp_min(0) / float(translation_reward_scale_cm)
        - re_deg.clamp_min(0) / float(rotation_reward_scale_deg)
    )
    quality = (
        0.25
        + continuous_reward
        + precision.float()
        + strict.float()
    )
    debiased = torch.log(inverse.clamp_min(1e-8))
    good_logits = (
        set_scores[good]
        + torch.log(quality[good].clamp_min(1e-8))
        + debiased[good]
    ) / float(basin_temperature)
    harmful_logits = (
        set_scores[harmful] + debiased[harmful]
    ) / float(basin_temperature)
    if good_logits.numel() and harmful_logits.numel():
        numerator = torch.logsumexp(good_logits, dim=0)
        denominator = torch.logsumexp(
            torch.cat((good_logits, harmful_logits)), dim=0
        )
        contrastive = denominator - numerator
    else:
        contrastive = torch.zeros((), device=bank.device)

    near = torch.nonzero(
        (types == 2) & correct & (parent_set_index >= 0), as_tuple=False
    ).reshape(-1)
    valid_near = near[
        parent_set_index[near].long().clamp_min(0) < set_scores.numel()
    ]
    if valid_near.numel():
        parents = parent_set_index[valid_near].long()
        valid_parent = types[parents] == 1
        valid_near = valid_near[valid_parent]
        parents = parents[valid_parent]
    if valid_near.numel():
        counterfactual = F.softplus(
            (
                set_scores[parents]
                - set_scores[valid_near]
                + float(counterfactual_margin)
            )
            / float(counterfactual_temperature)
        ) * float(counterfactual_temperature)
        counterfactual = (
            counterfactual * inverse[valid_near]
        ).sum() / inverse[valid_near].sum().clamp_min(1e-8)
    else:
        counterfactual = torch.zeros((), device=bank.device)
    return contrastive, counterfactual, {
        "coarse": int(coarse.sum()),
        "precision": int(precision.sum()),
        "strict": int(strict.sum()),
    }


def _verify_basin_teacher_prefix(state: dict, teacher: dict) -> None:
    teacher_count = int(teacher["anchor_count"])
    state_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if teacher_count > state_count:
        raise ValueError("basin teacher has more anchors than the target map")
    source_record = teacher.get("artifacts", {}).get("map", {})
    source_path = Path(str(source_record.get("path", "")))
    if not source_path.is_file():
        raise ValueError("basin teacher source map is unavailable")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    for key in ("anchor_ids", "anchor_xyz", "source_primitive_ids"):
        expected = torch.as_tensor(source[key])[:teacher_count]
        actual = torch.as_tensor(state[key])[:teacher_count]
        if expected.shape != actual.shape or not torch.equal(expected, actual):
            raise ValueError(f"target map does not preserve basin-teacher {key} prefix")


def _project_errors(xyz, keypoints, K, pose):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = torch.empty_like(keypoints)
    projected[:, 0] = K[0, 0] * camera[:, 0] / depth.clamp_min(1e-8) + K[0, 2]
    projected[:, 1] = K[1, 1] * camera[:, 1] / depth.clamp_min(1e-8) + K[1, 2]
    return torch.linalg.norm(projected - keypoints, dim=1)


@torch.no_grad()
def _refresh_ransac_outcomes(
    *,
    metric,
    null_head,
    null_temperature,
    null_threshold,
    null_minimum_total,
    null_grid_rows,
    null_grid_cols,
    null_minimum_per_cell,
    raw_features,
    anchor_residual,
    maximum_anchor_residual,
    state,
    cache,
    names,
    groups,
    training_records,
    device,
    query_limit,
    query_indices,
    seed,
):
    anchor, _ = _bounded_anchor_features(
        raw_features, anchor_residual, maximum_anchor_residual
    )
    bank, _ = metric(anchor)
    xyz_cpu = torch.as_tensor(state["anchor_xyz"]).float()
    harmful = torch.zeros(bank.shape[0])
    clean = torch.zeros(bank.shape[0])
    harmful_pairs: dict[int, dict[int, int]] = defaultdict(dict)
    clean_pairs: dict[int, dict[int, int]] = defaultdict(dict)
    group_error: dict[int, list[float]] = defaultdict(list)
    order = (
        np.asarray(query_indices, dtype=int)
        if query_indices is not None
        else np.linspace(
            0,
            len(names) - 1,
            min(int(query_limit), len(names)),
            dtype=int,
        )
    )
    records = []
    for query_index in order.tolist():
        cached = cache[names[query_index]]
        deployment_rows = training_records[query_index][
            "deployment_rows"
        ].long()
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[
                deployment_rows
            ],
            dim=1,
        ).to(device)
        adapted, _ = metric(descriptors)
        score_matrix = adapted @ bank.T
        null_top_scores, null_top_indices = torch.topk(
            score_matrix, k=min(8, score_matrix.shape[1]), dim=1
        )
        keypoint_grid = torch.as_tensor(
            cached["native_keypoints"]
        ).float()[deployment_rows]
        null_features = build_native_null_features(
            null_top_scores,
            torch.as_tensor(cached["native_scores"]).float()[
                deployment_rows
            ].to(device),
            temperature=float(null_temperature),
        )
        matchable_probability = torch.sigmoid(null_head(null_features))
        native_height, native_width = cached["native_input_hw"]
        keep = select_native_matchable_rows(
            matchable_probability,
            keypoint_grid.to(device),
            width=int(native_width),
            height=int(native_height),
            threshold=float(null_threshold),
            minimum_total=int(null_minimum_total),
            grid_rows=int(null_grid_rows),
            grid_cols=int(null_grid_cols),
            minimum_per_cell=int(null_minimum_per_cell),
        )
        score = null_top_scores[keep, 0]
        index = null_top_indices[keep, 0]
        deployment_rows = deployment_rows[keep.cpu()]
        keypoint = keypoint_grid[keep.cpu()] + float(
            cached.get("pixel_center_offset", 0.5)
        )
        K = torch.as_tensor(cached["native_K"]).float()
        pose, inliers, diagnostics = solve_pose(
            keypoint.numpy(),
            xyz_cpu[index.cpu()].numpy(),
            K.numpy(),
            solver="poselib",
            reprojection_error=12.0,
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            scores=score.cpu().numpy(),
            ransac_seed=int(seed),
            return_diagnostics=True,
        )
        _, te_cm = cal_pose_error(
            pose, torch.as_tensor(cached["pose_w2c"]).numpy()
        )
        group = int(groups[query_index])
        group_error[group].append(float(te_cm))
        inliers = torch.as_tensor(inliers).long().reshape(-1)
        if inliers.numel():
            gt_pose = torch.as_tensor(cached["pose_w2c"]).float()
            errors = _project_errors(
                xyz_cpu[index.cpu()[inliers]],
                keypoint[inliers],
                K,
                gt_pose,
            )
            clean_mask = errors <= 4.0
            clean.index_add_(
                0,
                index.cpu()[inliers[clean_mask]],
                torch.ones(int(clean_mask.sum())),
            )
            harmful.index_add_(
                0,
                index.cpu()[inliers[~clean_mask]],
                torch.ones(int((~clean_mask).sum())),
            )
            inlier_cache_rows = deployment_rows[inliers]
            inlier_anchors = index.cpu()[inliers]
            for cache_row, anchor, is_clean in zip(
                inlier_cache_rows.tolist(),
                inlier_anchors.tolist(),
                clean_mask.tolist(),
            ):
                target = clean_pairs if is_clean else harmful_pairs
                target[query_index][int(cache_row)] = int(anchor)
        records.append(
            {
                "query_index": query_index,
                "group": group,
                "te_cm": float(te_cm),
                "inliers": int(inliers.numel()),
                "candidate_count": int(keep.numel()),
                "hypotheses": diagnostics.get("ransac_actual_hypotheses"),
            }
        )
    prior = harmful / (harmful + clean + 1.0)
    risks = {
        group: _group_pose_risk(values)
        for group, values in group_error.items()
    }
    return (
        prior.to(device),
        risks,
        records,
        dict(clean_pairs),
        dict(harmful_pairs),
    )


def _save_checkpoint(
    *,
    output_dir,
    step,
    state,
    metric,
    null_head,
    raw_features,
    anchor_residual,
    maximum_anchor_residual,
    history,
    config,
):
    with torch.no_grad():
        anchor, bounded = _bounded_anchor_features(
            raw_features, anchor_residual, maximum_anchor_residual
        )
        transformed, _ = metric(anchor)
    output = dict(state)
    output["v7_metric_raw_features"] = anchor.detach().cpu()
    output["anchor_features"] = transformed.detach().cpu()
    output["v7_online_metric"] = {
        "schema": "lafgs_v7_online_shared_metric",
        "version": 1,
        "step": int(step),
        "anchor_residual_mean": float(bounded.norm(dim=1).mean()),
        "anchor_residual_max": float(bounded.norm(dim=1).max()),
        "config": config,
        "history": history,
    }
    map_path = output_dir / f"anchor_map_step_{step:04d}.pt"
    metric_path = output_dir / f"metric_state_step_{step:04d}.pt"
    torch.save(output, map_path)
    torch.save(
        {
            "schema": "lafgs_v7_shared_metric_state",
            "version": 1,
            "landmark_indices": torch.arange(
                transformed.shape[0], dtype=torch.long
            ),
            "metric_config": metric.export_config(),
            "metric_state_dict": {
                key: value.detach().cpu()
                for key, value in metric.state_dict().items()
            },
            "null_head_config": {
                "feature_dim": int(null_head.feature_dim),
                "temperature": float(config["null_temperature"]),
                "threshold": float(config["null_threshold"]),
                "minimum_total": int(config["null_minimum_total"]),
                "grid_rows": int(config["null_grid_rows"]),
                "grid_cols": int(config["null_grid_cols"]),
                "minimum_per_cell": int(config["null_minimum_per_cell"]),
            },
            "null_head_state_dict": {
                key: value.detach().cpu()
                for key, value in null_head.state_dict().items()
            },
            "map_path": str(map_path),
            "step": int(step),
        },
        metric_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", default="")
    parser.add_argument("--pose-critical-teacher", default="")
    parser.add_argument("--basin-teacher", default="")
    parser.add_argument("--basin-weight", type=float, default=0.0)
    parser.add_argument("--basin-good-weight", type=float, default=1.0)
    parser.add_argument("--basin-blame-weight", type=float, default=1.0)
    parser.add_argument("--basin-sets-per-step", type=int, default=8)
    parser.add_argument("--basin-blame-per-step", type=int, default=16)
    parser.add_argument("--basin-maximum-inverse-propensity", type=float, default=100.0)
    parser.add_argument("--basin-blame-margin", type=float, default=0.02)
    parser.add_argument(
        "--basin-good-mode",
        choices=("joint_ce", "margin_guard", "hyperedge"),
        default="joint_ce",
    )
    parser.add_argument("--basin-good-margin-slack", type=float, default=0.0)
    parser.add_argument("--basin-hyperedge-counterfactual-weight", type=float, default=1.0)
    parser.add_argument("--basin-hyperedge-temperature", type=float, default=1.0)
    parser.add_argument("--basin-counterfactual-temperature", type=float, default=0.25)
    parser.add_argument("--basin-counterfactual-margin", type=float, default=0.1)
    parser.add_argument("--basin-translation-reward-scale-cm", type=float, default=15.0)
    parser.add_argument("--basin-rotation-reward-scale-deg", type=float, default=2.0)
    parser.add_argument("--critical-pair-power", type=float, default=1.0)
    parser.add_argument("--critical-row-power", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--checkpoint-steps", default="100,250,500")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max-positives", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--metric-residual", type=float, default=0.10)
    parser.add_argument("--anchor-residual", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--harmful-weight", type=float, default=0.05)
    parser.add_argument("--trust-weight", type=float, default=0.1)
    parser.add_argument("--group-dro-eta", type=float, default=0.03)
    parser.add_argument("--refresh-interval", type=int, default=100)
    parser.add_argument("--refresh-query-limit", type=int, default=128)
    parser.add_argument("--refresh-shards", type=int, default=7)
    parser.add_argument("--null-weight", type=float, default=0.2)
    parser.add_argument("--null-temperature", type=float, default=0.05)
    parser.add_argument("--null-threshold", type=float, default=0.5)
    parser.add_argument("--null-minimum-total", type=int, default=384)
    parser.add_argument("--null-grid-rows", type=int, default=4)
    parser.add_argument("--null-grid-cols", type=int, default=4)
    parser.add_argument("--null-minimum-per-cell", type=int, default=8)
    parser.add_argument(
        "--training-mode",
        choices=("joint", "metric_only", "anchor_only", "sequential"),
        default="joint",
    )
    parser.add_argument("--metric-only-steps", type=int, default=250)
    parser.add_argument(
        "--conflict-anchor-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--feature-initialization",
        choices=("current", "pre_metric_raw"),
        default="current",
        help="A fresh metric must normally start from the deployed current descriptors.",
    )
    parser.add_argument("--initial-metric-state", default="")
    args = parser.parse_args()
    if not 0.0 <= args.critical_pair_power <= 1.0:
        raise ValueError("critical_pair_power must be in [0, 1]")
    if not 0.0 <= args.critical_row_power <= 1.0:
        raise ValueError("critical_row_power must be in [0, 1]")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    graph = torch.load(
        args.function_graph, map_location="cpu", weights_only=False
    )
    payload = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    positive_teacher = (
        torch.load(
            args.complete_positive_teacher,
            map_location="cpu",
            weights_only=False,
        )
        if args.complete_positive_teacher
        else None
    )
    critical_teacher = (
        torch.load(
            args.pose_critical_teacher,
            map_location="cpu",
            weights_only=False,
        )
        if args.pose_critical_teacher
        else None
    )
    basin_teacher = (
        torch.load(
            args.basin_teacher,
            map_location="cpu",
            weights_only=False,
        )
        if args.basin_teacher
        else None
    )
    if critical_teacher is not None:
        if positive_teacher is None:
            raise ValueError("pose-critical teacher requires complete positives")
        if int(critical_teacher["anchor_count"]) != int(
            torch.as_tensor(state["anchor_xyz"]).shape[0]
        ):
            raise ValueError("pose-critical teacher anchor count mismatch")
        if list(critical_teacher["query_names"]) != list(
            positive_teacher["query_names"]
        ):
            raise ValueError("pose-critical teacher query order mismatch")
    cache = cache_payload.get("queries", cache_payload)
    names = graph["query_names"]
    basin_records = [None] * len(names)
    basin_conflict_mask = torch.zeros(
        int(torch.as_tensor(state["anchor_xyz"]).shape[0]), dtype=torch.bool
    )
    if basin_teacher is not None:
        if basin_teacher.get("schema") != "lafgs_basin_teacher":
            raise ValueError("unsupported basin teacher schema")
        _verify_basin_teacher_prefix(state, basin_teacher)
        name_to_query = {name: index for index, name in enumerate(names)}
        for basin_record in basin_teacher["records"]:
            name = basin_record["query_name"]
            if name not in name_to_query:
                raise ValueError(f"basin teacher contains an unknown query: {name}")
            query_index = name_to_query[name]
            if basin_records[query_index] is not None:
                raise ValueError(f"duplicate basin teacher query: {name}")
            basin_records[query_index] = basin_record
            harmful = torch.as_tensor(
                basin_record["blame_harmful_anchors"]
            ).long()
            if harmful.numel():
                if int(harmful.min()) < 0 or int(harmful.max()) >= len(
                    basin_conflict_mask
                ):
                    raise ValueError("basin blame references an invalid anchor")
                basin_conflict_mask[harmful] = True
    records, data_report = _build_training_records(
        graph,
        payload,
        state,
        args.max_positives,
        device=device,
        positive_teacher=positive_teacher,
        critical_teacher=critical_teacher,
        critical_pair_power=args.critical_pair_power,
        critical_row_power=args.critical_row_power,
    )
    data_report.update(
        {
            "basin_teacher_enabled": basin_teacher is not None,
            "basin_teacher_query_count": int(
                sum(record is not None for record in basin_records)
            ),
            "basin_conflict_anchor_count": int(basin_conflict_mask.sum()),
        }
    )
    del graph
    feature_initialization = args.feature_initialization
    initial_metric_payload = None
    if args.initial_metric_state:
        initial_metric_payload = torch.load(
            args.initial_metric_state, map_location="cpu", weights_only=False
        )
        feature_initialization = "pre_metric_raw"
    raw_features, feature_key = _initial_anchor_features(
        state, feature_initialization
    )
    raw_features = raw_features.to(device)
    data_report["feature_initialization_key"] = feature_key
    anchor_residual = torch.nn.Parameter(torch.zeros_like(raw_features))
    metric = SharedLowRankMetric(
        **(
            initial_metric_payload["metric_config"]
            if initial_metric_payload is not None
            else {
                "descriptor_dim": raw_features.shape[1],
                "rank": args.rank,
                "max_residual_norm": args.metric_residual,
            }
        )
    ).to(device)
    if initial_metric_payload is not None:
        metric.load_state_dict(initial_metric_payload["metric_state_dict"])
        data_report["initial_metric_state"] = str(
            Path(args.initial_metric_state).resolve()
        )
    null_head = NativeNullHead().to(device)
    optimizer = torch.optim.AdamW(
        [*metric.parameters(), *null_head.parameters(), anchor_residual],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    groups = torch.as_tensor(data_report["query_groups"]).long()
    group_count = int(groups.max()) + 1
    group_weights = torch.ones(group_count, device=device) / group_count
    harmful_prior = torch.zeros(raw_features.shape[0], device=device)
    clean_pairs = {}
    harmful_pairs = {}
    generator = torch.Generator().manual_seed(args.seed + 1)
    checkpoints = {
        int(value)
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    refresh_shards = _build_rotating_shards(groups, args.refresh_shards)
    refresh_index = 0
    for step in range(1, args.steps + 1):
        if step == 1 or (
            args.refresh_interval > 0
            and (step - 1) % args.refresh_interval == 0
        ):
            active_shard = refresh_index % len(refresh_shards)
            (
                harmful_prior,
                group_risks,
                outcome,
                refreshed_clean_pairs,
                refreshed_harmful_pairs,
            ) = _refresh_ransac_outcomes(
                metric=metric,
                null_head=null_head,
                null_temperature=args.null_temperature,
                null_threshold=args.null_threshold,
                null_minimum_total=args.null_minimum_total,
                null_grid_rows=args.null_grid_rows,
                null_grid_cols=args.null_grid_cols,
                null_minimum_per_cell=args.null_minimum_per_cell,
                raw_features=raw_features,
                anchor_residual=anchor_residual,
                maximum_anchor_residual=args.anchor_residual,
                state=state,
                cache=cache,
                names=names,
                groups=groups,
                training_records=records,
                device=device,
                query_limit=args.refresh_query_limit,
                query_indices=refresh_shards[active_shard],
                seed=args.seed,
            )
            pair_churn = _replace_refreshed_pairs(
                clean_pairs,
                harmful_pairs,
                refresh_shards[active_shard],
                refreshed_clean_pairs,
                refreshed_harmful_pairs,
            )
            risk = torch.zeros_like(group_weights)
            for group, value in group_risks.items():
                risk[group] = float(value)
            group_weights *= torch.exp(float(args.group_dro_eta) * risk)
            group_weights /= group_weights.sum().clamp_min(1e-8)
            history.append(
                {
                    "step": step - 1,
                    "event": "deployment_refresh",
                    "shard": int(active_shard),
                    "shard_query_count": len(refresh_shards[active_shard]),
                    "covered_query_count": int(
                        sum(
                            len(refresh_shards[index])
                            for index in range(
                                min(refresh_index + 1, len(refresh_shards))
                            )
                        )
                    ),
                    "mean_te_cm": float(
                        np.mean([row["te_cm"] for row in outcome])
                    ),
                    "mean_candidate_count": float(
                        np.mean(
                            [row["candidate_count"] for row in outcome]
                        )
                    ),
                    "mean_hypotheses": float(
                        np.mean(
                            [
                                row["hypotheses"]
                                for row in outcome
                                if row["hypotheses"] is not None
                            ]
                        )
                    ),
                    "harmful_anchor_fraction": float(
                        (harmful_prior > 0).float().mean()
                    ),
                    "group_weight_max": float(group_weights.max()),
                    **pair_churn,
                }
            )
            print(json.dumps(history[-1]), flush=True)
            refresh_index += 1

        query_index = int(
            torch.randint(len(records), (1,), generator=generator)
        )
        record = records[query_index]
        count = int(record["cache_rows"].numel())
        if count == 0:
            continue
        rows = torch.randint(
            count,
            (min(args.batch_size, count),),
            generator=generator,
        )
        cache_rows = record["cache_rows"][rows]
        query = F.normalize(
            torch.as_tensor(
                cache[names[query_index]]["native_descriptors"]
            ).float()[cache_rows],
            dim=1,
        ).to(device)
        positives = record["positives"][rows].to(device)
        positive_weights = record["positive_weights"][rows].to(device)
        critical_row_weights = record["critical_row_weights"][rows].to(device)
        matchable = record["matchable"][rows].to(device)
        null_weight = record["null_weight"][rows].to(device)
        current_clean = clean_pairs.get(query_index, {})
        clean_survivors = torch.as_tensor(
            [current_clean.get(int(row), -1) for row in cache_rows],
            dtype=torch.long,
            device=device,
        )
        add_clean = (clean_survivors >= 0) & ~(
            positives == clean_survivors[:, None]
        ).any(dim=1)
        if bool(add_clean.any()):
            positives = positives.clone()
            replace = torch.where(
                (positives < 0).any(dim=1),
                (positives < 0).to(torch.int64).argmax(dim=1),
                torch.full(
                    (positives.shape[0],),
                    positives.shape[1] - 1,
                    device=device,
                ),
            )
            positives[
                torch.arange(positives.shape[0], device=device)[add_clean],
                replace[add_clean],
            ] = clean_survivors[add_clean]
            matchable = matchable | add_clean
        current_harmful = harmful_pairs.get(query_index, {})
        harmful_survivors = torch.as_tensor(
            [current_harmful.get(int(row), -1) for row in cache_rows],
            dtype=torch.long,
            device=device,
        )[:, None]
        anchor, bounded_anchor = _bounded_anchor_features(
            raw_features, anchor_residual, args.anchor_residual
        )
        adapted_query, query_metric_residual = metric(query)
        adapted_anchor, anchor_metric_residual = metric(anchor)
        score_matrix = adapted_query @ adapted_anchor.T
        top_scores = torch.topk(
            score_matrix, k=min(args.topk, score_matrix.shape[1]), dim=1
        ).values
        list_loss = torch.zeros(
            adapted_query.shape[0], device=device, dtype=adapted_query.dtype
        )
        if bool(matchable.any()):
            list_loss[matchable] = _multi_positive_list_loss(
                adapted_query[matchable],
                adapted_anchor,
                positives[matchable],
                args.topk,
                args.temperature,
                None,
                args.harmful_weight,
                harmful_indices=harmful_survivors[matchable],
                positive_weights=positive_weights[matchable],
            )[0]
        keypoint_score = torch.as_tensor(
            cache[names[query_index]]["native_scores"]
        ).float()[cache_rows].to(device)
        null_features = build_native_null_features(
            top_scores,
            keypoint_score,
            temperature=args.null_temperature,
        )
        null_logits = null_head(null_features)
        null_loss = F.binary_cross_entropy_with_logits(
            null_logits,
            matchable.float(),
            weight=null_weight,
            reduction="mean",
        )
        group_weight = (
            group_weights[int(record["group"])] * float(group_count)
        )
        task_loss = (
            (
                list_loss[matchable] * critical_row_weights[matchable]
            ).sum()
            / critical_row_weights[matchable].sum().clamp_min(1e-8)
            if bool(matchable.any())
            else torch.zeros((), device=device)
        ) * group_weight
        basin_good_loss = torch.zeros((), device=device)
        basin_blame_loss = torch.zeros((), device=device)
        basin_counterfactual_loss = torch.zeros((), device=device)
        basin_tiers = {"coarse": 0, "precision": 0, "strict": 0}
        basin_record = basin_records[query_index]
        if basin_record is not None and float(args.basin_weight) > 0:
            set_types = torch.as_tensor(basin_record["set_types"]).long()
            correct_basin = torch.as_tensor(
                basin_record["correct_basin"]
            ).bool()
            if args.basin_good_mode == "hyperedge":
                basin_rows = torch.as_tensor(
                    basin_record["set_query_rows"]
                ).long()
                basin_raw_query = F.normalize(
                    torch.as_tensor(
                        cache[names[query_index]]["native_descriptors"]
                    ).float()[basin_rows.reshape(-1)],
                    dim=1,
                ).to(device)
                basin_query, _ = metric(basin_raw_query)
                basin_good_loss, basin_counterfactual_loss, basin_tiers = (
                    _basin_hyperedge_losses(
                        basin_query.reshape(len(basin_rows), 3, -1),
                        adapted_anchor,
                        torch.as_tensor(
                            basin_record["set_anchor_indices"]
                        ).long().to(device),
                        set_types.to(device),
                        correct_basin.to(device),
                        torch.as_tensor(basin_record["te_cm"]).float().to(device),
                        torch.as_tensor(basin_record["re_deg"]).float().to(device),
                        torch.as_tensor(
                            basin_record["parent_set_index"]
                        ).long().to(device),
                        torch.as_tensor(
                            basin_record["sampling_propensity"]
                        ).float().to(device),
                        assignment_temperature=args.temperature,
                        basin_temperature=args.basin_hyperedge_temperature,
                        counterfactual_temperature=(
                            args.basin_counterfactual_temperature
                        ),
                        counterfactual_margin=args.basin_counterfactual_margin,
                        translation_reward_scale_cm=(
                            args.basin_translation_reward_scale_cm
                        ),
                        rotation_reward_scale_deg=(
                            args.basin_rotation_reward_scale_deg
                        ),
                        maximum_inverse_propensity=(
                            args.basin_maximum_inverse_propensity
                        ),
                    )
                )
            good_sets = torch.nonzero(
                correct_basin & ((set_types == 0) | (set_types == 2)),
                as_tuple=False,
            ).reshape(-1)
            if good_sets.numel() and args.basin_good_mode != "hyperedge":
                take = min(int(args.basin_sets_per_step), int(good_sets.numel()))
                selected = good_sets[
                    torch.randperm(
                        good_sets.numel(), generator=generator
                    )[:take]
                ]
                basin_rows = torch.as_tensor(
                    basin_record["set_query_rows"]
                ).long()[selected]
                basin_raw_query = F.normalize(
                    torch.as_tensor(
                        cache[names[query_index]]["native_descriptors"]
                    ).float()[basin_rows.reshape(-1)],
                    dim=1,
                ).to(device)
                basin_query, _ = metric(basin_raw_query)
                basin_anchors = torch.as_tensor(
                    basin_record["set_anchor_indices"]
                ).long()[selected].to(device)
                basin_propensity = torch.as_tensor(
                    basin_record["sampling_propensity"]
                ).float()[selected].to(device)
                if args.basin_good_mode == "margin_guard":
                    basin_good_loss = _basin_good_margin_guard_loss(
                        basin_raw_query.reshape(take, 3, -1),
                        raw_features,
                        basin_query.reshape(take, 3, -1),
                        adapted_anchor,
                        basin_anchors,
                        basin_propensity,
                        maximum_inverse_propensity=args.basin_maximum_inverse_propensity,
                        slack=args.basin_good_margin_slack,
                    )
                else:
                    basin_good_loss = _basin_good_set_loss(
                        basin_query.reshape(take, 3, -1),
                        adapted_anchor,
                        basin_anchors,
                        basin_propensity,
                        topk=args.topk,
                        temperature=args.temperature,
                        maximum_inverse_propensity=args.basin_maximum_inverse_propensity,
                    )
            blame_rows = torch.as_tensor(
                basin_record["blame_rows"]
            ).long()
            if blame_rows.numel() and float(args.basin_blame_weight) > 0:
                take = min(
                    int(args.basin_blame_per_step), int(blame_rows.numel())
                )
                selected = torch.randperm(
                    blame_rows.numel(), generator=generator
                )[:take]
                blame_query = F.normalize(
                    torch.as_tensor(
                        cache[names[query_index]]["native_descriptors"]
                    ).float()[blame_rows[selected]],
                    dim=1,
                ).to(device)
                blame_query, _ = metric(blame_query)
                basin_blame_loss = _basin_counterfactual_blame_loss(
                    blame_query,
                    adapted_anchor,
                    torch.as_tensor(
                        basin_record["blame_harmful_anchors"]
                    ).long()[selected].to(device),
                    torch.as_tensor(
                        basin_record["blame_positive_anchors"]
                    ).long()[selected].to(device),
                    torch.as_tensor(
                        basin_record["blame_weights"]
                    ).float()[selected].to(device),
                    temperature=args.temperature,
                    margin=args.basin_blame_margin,
                )
        trust = (
            query_metric_residual.square().sum(dim=1).mean()
            + anchor_metric_residual.square().sum(dim=1).mean()
            + bounded_anchor.square().sum(dim=1).mean()
        )
        loss = (
            task_loss
            + float(args.null_weight) * null_loss
            + float(args.trust_weight) * trust
            + float(args.basin_weight)
            * (
                float(args.basin_good_weight) * basin_good_loss
                + float(args.basin_blame_weight) * basin_blame_loss
                + float(args.basin_hyperedge_counterfactual_weight)
                * basin_counterfactual_loss
            )
        )
        if args.training_mode == "sequential":
            phase = (
                "metric_only"
                if step <= int(args.metric_only_steps)
                else "anchor_only"
            )
        else:
            phase = args.training_mode
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if phase == "metric_only":
            anchor_residual.grad = None
        elif phase == "anchor_only":
            for parameter in metric.parameters():
                parameter.grad = None
            if (
                bool(args.conflict_anchor_only)
                and anchor_residual.grad is not None
            ):
                conflict = harmful_prior > 0
                conflict = conflict | basin_conflict_mask.to(device)
                anchor_residual.grad[~conflict] = 0
        torch.nn.utils.clip_grad_norm_(
            [*metric.parameters(), anchor_residual], 1.0
        )
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "trust_loss": float(trust.detach()),
                "null_loss": float(null_loss.detach()),
                "basin_good_loss": float(basin_good_loss.detach()),
                "basin_blame_loss": float(basin_blame_loss.detach()),
                "basin_counterfactual_loss": float(
                    basin_counterfactual_loss.detach()
                ),
                "basin_coarse_sets": basin_tiers["coarse"],
                "basin_precision_sets": basin_tiers["precision"],
                "basin_strict_sets": basin_tiers["strict"],
                "matchable_fraction": float(matchable.float().mean()),
                "group": int(record["group"]),
                "phase": phase,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if step in checkpoints or step == args.steps:
            _save_checkpoint(
                output_dir=output_dir,
                step=step,
                state=state,
                metric=metric,
                null_head=null_head,
                raw_features=raw_features,
                anchor_residual=anchor_residual,
                maximum_anchor_residual=args.anchor_residual,
                history=history,
                config={**vars(args), **data_report},
            )
    (output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "schema": "lafgs_v7_online_metric_training",
                "config": {**vars(args), **data_report},
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
