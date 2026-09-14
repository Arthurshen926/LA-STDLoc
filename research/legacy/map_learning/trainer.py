"""Self-localization-guided descriptor reconstruction for a compact map.

The trainer contains only the frozen paper path: complete-positive retrieval,
current-map hard outcomes, trajectory-group DRO, and a bounded shared metric.
All geometry and anchor identities remain fixed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common.hashing import sha256_file
from evidence.tracks import (
    LeaveOneQueryOutProjectiveAnchorDescriptorBank,
    LeaveOneQueryOutTrackDescriptorBank,
)
from map_learning.metric import SharedLowRankMetric
from map_learning.alias_teacher import (
    RecurrentAliasTeacher,
    alias_group_ranking_loss,
    build_recurrent_alias_teacher,
    protected_clean_margin_loss,
    replace_query_assignments,
)
from map_learning.soft_pose import soft_pose_bias_loss
from localization.pose_solver import solve_absolute_pose


def _query_index_remap(source: list[str], target: list[str]) -> torch.Tensor:
    if len(set(source)) != len(source) or len(set(target)) != len(target):
        raise ValueError("query registries must be unique")
    target_by_name = {name: index for index, name in enumerate(target)}
    if set(source) != set(target):
        raise ValueError("query registries differ")
    return torch.as_tensor([target_by_name[name] for name in source]).long()


def _csr_first_k(offsets, indices, width: int) -> torch.Tensor:
    offsets = torch.as_tensor(offsets).long().reshape(-1)
    indices = torch.as_tensor(indices).long().reshape(-1)
    output = torch.full((offsets.numel() - 1, int(width)), -1, dtype=torch.long)
    counts = offsets[1:] - offsets[:-1]
    rows = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    rank = torch.arange(indices.numel()) - offsets[rows]
    keep = rank < int(width)
    output[rows[keep], rank[keep]] = indices[keep]
    return output


def _build_training_records(
    graph: dict,
    track_payload: dict,
    state: dict,
    teacher: dict,
    max_positives: int,
) -> tuple[list[dict], dict]:
    names = list(graph["query_names"])
    if list(teacher["query_names"]) != names:
        raise ValueError("complete-positive teacher query order mismatch")
    if int(teacher["anchor_count"]) != int(state["anchor_xyz"].shape[0]):
        raise ValueError("complete-positive teacher anchor count mismatch")
    graph_records = graph["records"]
    if len(teacher["records"]) != len(graph_records):
        raise ValueError("complete-positive teacher query count mismatch")

    payload_to_graph = _query_index_remap(track_payload["query_names"], names)
    payload_bins = torch.as_tensor(track_payload["query_bins"]).long()
    query_bins = torch.empty_like(payload_bins)
    query_bins[payload_to_graph] = payload_bins

    records = []
    positive_rows = 0
    ambiguous_pairs = 0
    ignored_ambiguous_pairs = 0
    for graph_record, teacher_record, group in zip(
        graph_records, teacher["records"], query_bins.tolist()
    ):
        query_rows = torch.as_tensor(graph_record["query_rows"]).long()
        teacher_rows = torch.as_tensor(teacher_record["query_rows"]).long()
        if not torch.equal(query_rows, teacher_rows):
            raise ValueError("complete-positive teacher row mismatch")
        positives = _csr_first_k(
            teacher_record["positive_offsets"],
            teacher_record["positive_indices"],
            max_positives,
        )
        ambiguous = _csr_first_k(
            teacher_record["ambiguous_offsets"],
            teacher_record["ambiguous_indices"],
            max_positives,
        )
        ignore_ambiguous = graph_record.get("ambiguous_training_policy") == "ignore"
        ignored = ambiguous if ignore_ambiguous else torch.full_like(ambiguous, -1)
        matchable = (positives >= 0).any(dim=1)
        positive_rows += int(matchable.sum())
        ambiguous_pairs += int((ambiguous >= 0).sum())
        if ignore_ambiguous:
            ignored_ambiguous_pairs += int((ambiguous >= 0).sum())
        records.append(
            {
                "deployment_rows": query_rows,
                "cache_rows": query_rows,
                "positives": positives,
                "ignored_anchors": ignored,
                "matchable": matchable,
                "group": int(group),
            }
        )
    metadata = state["track_centric_reconstruction"]
    return records, {
        "positive_rows": positive_rows,
        "complete_positive_pair_count": int(
            teacher["diagnostics"]["strong_pair_count"]
        ),
        "ambiguous_pair_count": ambiguous_pairs,
        "ignored_ambiguous_pair_count": ignored_ambiguous_pairs,
        "track_anchor_count": int(torch.as_tensor(metadata["track_indices"]).numel()),
        "base_anchor_count": int(
            torch.as_tensor(metadata["base_canonical_rows"]).numel()
        ),
        "query_groups": query_bins.tolist(),
    }


def limit_training_records(
    records: list[dict], maximum_cache_row: int
) -> tuple[list[dict], dict]:
    """Restrict an evidence-rich cache to a nested deployment detector prefix."""
    maximum_cache_row = int(maximum_cache_row)
    if maximum_cache_row <= 0:
        return records, {
            "deployment_row_limit": 0,
            "deployment_rows_before": int(
                sum(record["cache_rows"].numel() for record in records)
            ),
            "deployment_rows_after": int(
                sum(record["cache_rows"].numel() for record in records)
            ),
        }
    limited = []
    before = 0
    after = 0
    for record in records:
        rows = torch.as_tensor(record["cache_rows"]).long()
        keep = rows < maximum_cache_row
        before += int(rows.numel())
        after += int(keep.sum())
        revised = dict(record)
        for key in (
            "deployment_rows",
            "cache_rows",
            "positives",
            "ignored_anchors",
            "matchable",
        ):
            revised[key] = torch.as_tensor(record[key])[keep]
        limited.append(revised)
    if after == 0:
        raise ValueError("deployment row limit removed every mapping observation")
    return limited, {
        "deployment_row_limit": maximum_cache_row,
        "deployment_rows_before": before,
        "deployment_rows_after": after,
    }


def resolve_density_prefixes(
    records: list[dict],
    deployment_row_limit: int,
    fractions: tuple[float, ...],
) -> tuple[int, ...]:
    """Resolve scene-independent nested prefixes in native detector rank space."""
    if not fractions:
        raise ValueError("density prefix fractions cannot be empty")
    values = sorted(set(float(value) for value in fractions))
    if values[-1] != 1.0 or values[0] <= 0.0 or values[-1] > 1.0:
        raise ValueError("density prefix fractions must lie in (0, 1] and include 1")
    inferred = max(
        int(torch.as_tensor(record["cache_rows"]).max().item()) + 1
        for record in records
        if torch.as_tensor(record["cache_rows"]).numel()
    )
    maximum = int(deployment_row_limit) if int(deployment_row_limit) > 0 else inferred
    prefixes = tuple(
        sorted(set(max(int(round(maximum * value)), 1) for value in values))
    )
    if prefixes[-1] != maximum:
        raise RuntimeError("density prefix resolution lost the full deployment density")
    return prefixes


def _build_rotating_shards(groups: torch.Tensor, count: int) -> list[list[int]]:
    groups = torch.as_tensor(groups).long().reshape(-1)
    count = max(min(int(count), groups.numel()), 1)
    shards: list[list[int]] = [[] for _ in range(count)]
    for group in torch.unique(groups, sorted=True).tolist():
        indices = torch.nonzero(groups == int(group), as_tuple=False).reshape(-1)
        for offset, query_index in enumerate(indices.tolist()):
            shards[offset % count].append(int(query_index))
    for shard in shards:
        shard.sort()
    if sorted(index for shard in shards for index in shard) != list(
        range(groups.numel())
    ):
        raise RuntimeError("refresh shards do not cover each mapping query once")
    return shards


def _replace_refreshed_pairs(
    clean_pairs: dict,
    harmful_pairs: dict,
    query_indices: list[int],
    refreshed_clean: dict,
    refreshed_harmful: dict,
) -> dict[str, int]:
    old_clean = sum(len(clean_pairs.get(int(query), {})) for query in query_indices)
    old_harmful = sum(len(harmful_pairs.get(int(query), {})) for query in query_indices)
    for query in query_indices:
        clean_pairs.pop(int(query), None)
        harmful_pairs.pop(int(query), None)
    clean_pairs.update(
        {int(key): dict(value) for key, value in refreshed_clean.items()}
    )
    harmful_pairs.update(
        {int(key): dict(value) for key, value in refreshed_harmful.items()}
    )
    return {
        "old_clean_pair_count": old_clean,
        "old_harmful_pair_count": old_harmful,
        "new_clean_pair_count": sum(
            len(clean_pairs.get(int(query), {})) for query in query_indices
        ),
        "new_harmful_pair_count": sum(
            len(harmful_pairs.get(int(query), {})) for query in query_indices
        ),
    }


def _group_pose_risk(errors_cm: list[float]) -> float:
    errors = torch.as_tensor(errors_cm, dtype=torch.float32)
    if errors.numel() == 0:
        return 0.0
    smooth_mean = torch.log1p(errors / 10.0).mean()
    tail_count = max(int(math.ceil(0.2 * errors.numel())), 1)
    tail = torch.topk(errors, k=tail_count).values.mean() / 20.0
    near_five = F.softplus((errors - 5.0) / 2.0).mean() / 5.0
    return float(smooth_mean + 0.5 * tail + 0.5 * near_five)


def full_refresh_interval(steps: int, shards: int) -> int:
    """Space refreshes so every rotating shard is replayed at least once."""
    steps = int(steps)
    shards = int(shards)
    if steps < 1 or shards < 1:
        raise ValueError("steps and refresh shards must be positive")
    if shards == 1:
        return max(steps, 1)
    return max((steps - 1) // (shards - 1), 1)


def _update_group_dro_weights(
    weights: torch.Tensor,
    risk: torch.Tensor,
    *,
    eta: float,
    maximum_uniform_ratio: float,
) -> torch.Tensor:
    """Apply a centered entropic DRO update with a capped simplex safeguard."""
    weights = torch.as_tensor(weights)
    risk = torch.as_tensor(risk, device=weights.device, dtype=weights.dtype)
    if weights.ndim != 1 or risk.shape != weights.shape:
        raise ValueError("group weights and risks must be aligned vectors")
    if float(maximum_uniform_ratio) < 1.0:
        raise ValueError("maximum group-weight ratio must be at least one")
    centered = risk - risk.mean()
    updated = weights * torch.exp((float(eta) * centered).clamp(min=-20.0, max=20.0))
    updated /= updated.sum().clamp_min(1e-8)
    uniform = 1.0 / max(updated.numel(), 1)
    cap = min(float(maximum_uniform_ratio) * uniform, 1.0)
    maximum = float(updated.max())
    if maximum > cap and maximum > uniform:
        alpha = (cap - uniform) / (maximum - uniform)
        updated = uniform + float(alpha) * (updated - uniform)
        updated /= updated.sum().clamp_min(1e-8)
    return updated


def _project_errors(xyz, keypoints, intrinsic, pose):
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    projected = torch.empty_like(keypoints)
    projected[:, 0] = (
        intrinsic[0, 0] * camera[:, 0] / depth.clamp_min(1e-8) + intrinsic[0, 2]
    )
    projected[:, 1] = (
        intrinsic[1, 1] * camera[:, 1] / depth.clamp_min(1e-8) + intrinsic[1, 2]
    )
    return torch.linalg.norm(projected - keypoints, dim=1)


def _pose_error_cm(predicted: np.ndarray, target: torch.Tensor) -> float:
    target = torch.as_tensor(target).cpu().numpy()
    predicted_center = np.linalg.inv(predicted)[:3, -1]
    target_center = np.linalg.inv(target)[:3, -1]
    return float(np.linalg.norm(predicted_center - target_center) * 100.0)


@torch.no_grad()
def _refresh_ransac_outcomes(
    *,
    metric: SharedLowRankMetric,
    raw_features: torch.Tensor,
    state: dict,
    cache: dict,
    names: list[str],
    groups: torch.Tensor,
    training_records: list[dict],
    device: torch.device,
    query_indices: list[int],
    seed: int,
    ransac_reprojection_px: float,
    clean_reprojection_px: float,
    deployment_row_limit: int = 0,
    anchor_residual_parameter: torch.Tensor | None = None,
    anchor_residual_max_norm: float = 0.0,
    loo_descriptor_bank: LeaveOneQueryOutTrackDescriptorBank | None = None,
):
    reference_bank, _, _ = bounded_anchor_bank(
        metric,
        raw_features,
        anchor_residual_parameter,
        anchor_residual_max_norm,
    )
    xyz_cpu = torch.as_tensor(state["anchor_xyz"]).float()
    harmful = torch.zeros(reference_bank.shape[0])
    clean = torch.zeros(reference_bank.shape[0])
    harmful_pairs: dict[int, dict[int, int]] = defaultdict(dict)
    clean_pairs: dict[int, dict[int, int]] = defaultdict(dict)
    false_top1_pairs: dict[int, dict[int, int]] = defaultdict(dict)
    clean_margin_pairs: dict[int, dict[int, float]] = defaultdict(dict)
    group_error: dict[int, list[float]] = defaultdict(list)
    rows = []
    for query_index in np.asarray(query_indices, dtype=int).tolist():
        bank = reference_bank
        if loo_descriptor_bank is not None:
            update_rows, update_features = loo_descriptor_bank.query_update(query_index)
            if update_rows.numel():
                device_rows = update_rows.to(device)
                row_residual = (
                    None
                    if anchor_residual_parameter is None
                    else anchor_residual_parameter[device_rows]
                )
                update_bank, _, _ = bounded_anchor_bank(
                    metric,
                    update_features.to(device),
                    row_residual,
                    anchor_residual_max_norm,
                )
                bank = reference_bank.index_copy(0, device_rows, update_bank)
        cached = cache[names[query_index]]
        record = training_records[query_index]
        deployment_rows = record["deployment_rows"].long()
        if int(deployment_row_limit) > 0:
            deployment_rows = deployment_rows[
                deployment_rows < int(deployment_row_limit)
            ]
        if deployment_rows.numel() == 0:
            continue
        record_rows = torch.searchsorted(record["cache_rows"].long(), deployment_rows)
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[deployment_rows],
            dim=1,
        ).to(device)
        adapted, _ = metric(descriptors)
        top_scores, top_indices = torch.topk(
            adapted @ bank.T, k=min(8, bank.shape[0]), dim=1
        )
        index = top_indices[:, 0]
        positives = record["positives"][record_rows]
        ignored = record["ignored_anchors"][record_rows]
        has_positive = (positives >= 0).any(dim=1)
        top1_positive = ((positives == index.cpu()[:, None]) & (positives >= 0)).any(
            dim=1
        )
        top1_ignored = ((ignored == index.cpu()[:, None]) & (ignored >= 0)).any(dim=1)
        false_top1 = has_positive & ~top1_positive & ~top1_ignored
        for cache_row, anchor in zip(
            deployment_rows[false_top1].tolist(), index.cpu()[false_top1].tolist()
        ):
            false_top1_pairs[query_index][int(cache_row)] = int(anchor)
        keypoint = torch.as_tensor(cached["native_keypoints"]).float()[
            deployment_rows
        ] + float(cached.get("pixel_center_offset", 0.5))
        intrinsic = torch.as_tensor(cached["native_K"]).float()
        estimate = solve_absolute_pose(
            keypoint.numpy(),
            xyz_cpu[index.cpu()].numpy(),
            intrinsic.numpy(),
            reprojection_error_px=float(ransac_reprojection_px),
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            seed=int(seed),
        )
        te_cm = _pose_error_cm(estimate.pose_w2c, torch.as_tensor(cached["pose_w2c"]))
        group = int(groups[query_index])
        group_error[group].append(te_cm)
        inliers = torch.as_tensor(estimate.inliers).long().reshape(-1)
        if inliers.numel():
            errors = _project_errors(
                xyz_cpu[index.cpu()[inliers]],
                keypoint[inliers],
                intrinsic,
                torch.as_tensor(cached["pose_w2c"]).float(),
            )
            clean_mask = errors <= float(clean_reprojection_px)
            clean.index_add_(
                0, index.cpu()[inliers[clean_mask]], torch.ones(int(clean_mask.sum()))
            )
            harmful.index_add_(
                0,
                index.cpu()[inliers[~clean_mask]],
                torch.ones(int((~clean_mask).sum())),
            )
            for cache_row, anchor, is_clean in zip(
                deployment_rows[inliers].tolist(),
                index.cpu()[inliers].tolist(),
                clean_mask.tolist(),
            ):
                target = clean_pairs if is_clean else harmful_pairs
                target[query_index][int(cache_row)] = int(anchor)
            if top_scores.shape[1] >= 2:
                margins = (top_scores[:, 0] - top_scores[:, 1]).detach().cpu()
                for local in inliers[clean_mask].tolist():
                    clean_margin_pairs[query_index][int(deployment_rows[local])] = (
                        float(margins[local])
                    )
        rows.append(
            {
                "query_index": query_index,
                "group": group,
                "te_cm": te_cm,
                "inliers": int(inliers.numel()),
                "candidate_count": int(deployment_rows.numel()),
                "hypotheses": estimate.diagnostics.get("iterations"),
            }
        )
    return (
        (harmful / (harmful + clean + 1.0)).to(device),
        {group: _group_pose_risk(value) for group, value in group_error.items()},
        rows,
        dict(clean_pairs),
        dict(harmful_pairs),
        dict(false_top1_pairs),
        dict(clean_margin_pairs),
    )


def alias_repair_anchor_indices(
    teacher: RecurrentAliasTeacher,
    records: list[dict],
    *,
    include_positives: bool,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Return the bounded parameter rows allowed to repair alias rankings.

    A ranking repair has two map-side participants: the recurrent false winner
    and the legal positive it beat.  Suppressing only the false winner discards
    half of that gradient and can damage views where the winner is legitimate.
    """
    false_anchors = set(int(value) for value in teacher.active_anchors.tolist())
    positive_anchors: set[int] = set()
    if include_positives:
        for query, assignments in teacher.row_anchors.items():
            cache_rows = torch.as_tensor(records[int(query)]["cache_rows"]).long()
            requested = torch.as_tensor(sorted(assignments), dtype=torch.long)
            positions = torch.searchsorted(cache_rows, requested)
            if bool((positions >= cache_rows.numel()).any()) or not torch.equal(
                cache_rows[positions], requested
            ):
                raise ValueError("alias teacher row is absent from training record")
            positives = torch.as_tensor(records[int(query)]["positives"]).long()[
                positions
            ]
            positive_anchors.update(
                int(value) for value in positives[positives >= 0].tolist()
            )
    combined = torch.as_tensor(
        sorted(false_anchors | positive_anchors), dtype=torch.long
    )
    return combined, {
        "alias_false_trainable_anchor_count": int(len(false_anchors)),
        "alias_positive_trainable_anchor_count": int(len(positive_anchors)),
        "alias_pair_trainable_anchor_count": int(combined.numel()),
    }


def _multi_positive_list_loss(
    query: torch.Tensor,
    bank: torch.Tensor,
    positives: torch.Tensor,
    ignored: torch.Tensor,
    harmful: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    harmful_weight: float,
) -> torch.Tensor:
    scores = query @ bank.T
    top_scores, top_indices = torch.topk(scores, k=min(int(topk), bank.shape[0]), dim=1)
    positive_mask = positives >= 0
    positive_scores = torch.einsum("bd,bpd->bp", query, bank[positives.clamp_min(0)])
    top_is_positive = (
        (top_indices[:, :, None] == positives[:, None, :]) & positive_mask[:, None, :]
    ).any(dim=2)
    ignored_valid = ignored >= 0
    top_is_ignored = (
        (top_indices[:, :, None] == ignored[:, None, :]) & ignored_valid[:, None, :]
    ).any(dim=2)
    denominator = torch.logsumexp(
        (
            torch.cat((top_scores, positive_scores), dim=1) / float(temperature)
        ).masked_fill(
            ~torch.cat((~top_is_positive & ~top_is_ignored, positive_mask), dim=1),
            -torch.inf,
        ),
        dim=1,
    )
    target = positive_mask.float()
    target /= target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    positive_aggregate = (target * positive_scores).sum(dim=1)
    list_loss = denominator * float(temperature) - positive_aggregate

    harmful_valid = harmful >= 0
    harmful_scores = torch.einsum(
        "bd,bhd->bh", query, bank[harmful.clamp_min(0)]
    ).masked_fill(~harmful_valid, -torch.inf)
    hardest = harmful_scores.max(dim=1).values
    harmful_loss = torch.where(
        harmful_valid.any(dim=1),
        F.softplus((hardest - positive_aggregate) / float(temperature))
        * float(temperature),
        torch.zeros_like(list_loss),
    )
    return list_loss + float(harmful_weight) * harmful_loss


def bounded_anchor_bank(
    metric: SharedLowRankMetric,
    raw_features: torch.Tensor,
    anchor_residual_parameter: torch.Tensor | None,
    maximum_norm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a shared query metric plus a bounded anchor-specific residual."""
    shared, shared_residual = metric(raw_features)
    if anchor_residual_parameter is None or float(maximum_norm) <= 0.0:
        return shared, shared_residual, torch.zeros_like(shared)
    residual = torch.as_tensor(anchor_residual_parameter)
    norm = torch.linalg.norm(residual, dim=1, keepdim=True)
    residual = residual * torch.clamp(
        float(maximum_norm) / norm.clamp_min(1e-8), max=1.0
    )
    return F.normalize(shared + residual, dim=1), shared_residual, residual


def bounded_query_anchor_bank(
    *,
    metric: SharedLowRankMetric,
    raw_features: torch.Tensor,
    query_index: int,
    loo_descriptor_bank: LeaveOneQueryOutTrackDescriptorBank | None,
    anchor_residual_parameter: torch.Tensor | None,
    maximum_norm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build the training bank, replacing only rows affected by one query.

    The deployed checkpoint still contains the full-observation Track bank.
    This helper only removes the current mapping query's own descriptor
    observations while that query supplies self-localization feedback.
    """
    bank, shared_residual, anchor_residual = bounded_anchor_bank(
        metric,
        raw_features,
        anchor_residual_parameter,
        maximum_norm,
    )
    if loo_descriptor_bank is None:
        return bank, shared_residual, anchor_residual, 0
    rows, features = loo_descriptor_bank.query_update(int(query_index))
    if rows.numel() == 0:
        return bank, shared_residual, anchor_residual, 0
    device_rows = rows.to(raw_features.device)
    row_residual = (
        None
        if anchor_residual_parameter is None
        else anchor_residual_parameter[device_rows]
    )
    updates, _, _ = bounded_anchor_bank(
        metric,
        features.to(device=raw_features.device, dtype=raw_features.dtype),
        row_residual,
        maximum_norm,
    )
    return (
        bank.index_copy(0, device_rows, updates),
        shared_residual,
        anchor_residual,
        int(rows.numel()),
    )


def track_descriptor_payload_for_loo(payload: dict) -> dict:
    """Restore pose-view bins after training replaces them with DRO groups."""
    if "pose_view_bins" not in payload:
        return payload
    pose_view_bins = torch.as_tensor(payload["pose_view_bins"]).long().reshape(-1)
    if pose_view_bins.numel() != len(payload["query_names"]):
        raise ValueError("LOO pose-view bins do not align with the query registry")
    return {**payload, "query_bins": pose_view_bins}


def _save_checkpoint(
    output_dir: Path,
    step: int,
    state: dict,
    metric: SharedLowRankMetric,
    raw_features: torch.Tensor,
    history: list[dict],
    config: dict,
    anchor_residual_parameter: torch.Tensor | None = None,
    anchor_residual_max_norm: float = 0.0,
) -> None:
    with torch.no_grad():
        transformed, _, anchor_residual = bounded_anchor_bank(
            metric,
            raw_features,
            anchor_residual_parameter,
            anchor_residual_max_norm,
        )
    output = dict(state)
    output["v7_metric_raw_features"] = raw_features.detach().cpu()
    output["anchor_features"] = transformed.detach().cpu()
    if anchor_residual_parameter is not None:
        output["v7_anchor_residual_parameter"] = (
            anchor_residual_parameter.detach().cpu()
        )
        output["v7_anchor_residual"] = anchor_residual.detach().cpu()
    output["v7_online_metric"] = {
        "schema": "lafgs_self_localization_descriptor_reconstruction",
        "version": 1,
        "step": int(step),
        "config": config,
        "history": history,
    }
    map_path = (output_dir / f"anchor_map_step_{step:04d}.pt").resolve()
    torch.save(output, map_path)
    torch.save(
        {
            "schema": "lafgs_shared_metric_state",
            "version": 1,
            "landmark_indices": torch.as_tensor(output["anchor_ids"]).long().clone(),
            "metric_config": metric.export_config(),
            "metric_state_dict": {
                key: value.detach().cpu() for key, value in metric.state_dict().items()
            },
            "map_path": str(map_path),
            "map_sha256": sha256_file(map_path),
            "step": int(step),
        },
        output_dir / f"metric_state_step_{step:04d}.pt",
    )


def train(
    *,
    map_path: str | Path,
    function_graph_path: str | Path,
    track_payload_path: str | Path,
    query_cache_path: str | Path,
    positive_teacher_path: str | Path,
    output_dir: str | Path,
    initial_metric_state_path: str | Path | None = None,
    steps: int = 175,
    checkpoint_steps: tuple[int, ...] = (175,),
    batch_size: int = 512,
    topk: int = 64,
    max_positives: int = 8,
    rank: int = 16,
    metric_residual: float = 0.05,
    learning_rate: float = 2e-4,
    temperature: float = 0.04,
    harmful_weight: float = 0.1,
    trust_weight: float = 1.0,
    group_dro_eta: float = 0.03,
    group_dro_max_weight_ratio: float = 1e9,
    refresh_interval: int = 0,
    refresh_shards: int = 7,
    initial_ransac_refresh: bool = True,
    deployment_row_limit: int = 0,
    density_prefix_fractions: tuple[float, ...] = (1.0,),
    density_dro_eta: float = 0.03,
    density_dro_max_weight_ratio: float = 3.0,
    alias_weight: float = 0.0,
    alias_margin: float = 0.05,
    alias_minimum_distinct_groups: int = 2,
    alias_minimum_queries: int = 3,
    alias_minimum_occurrences: int = 6,
    alias_minimum_rows_per_query: int = 2,
    alias_query_replay_fraction: float = 0.5,
    alias_require_harmful_inlier: bool = False,
    protected_clean_weight: float = 0.0,
    protected_clean_minimum_margin: float = 0.05,
    protected_clean_margin_slack: float = 0.01,
    protected_clean_task_scale: float = 0.25,
    anchor_feature_residual_max_norm: float = 0.0,
    anchor_feature_residual_trust_weight: float = 1.0,
    anchor_feature_residual_alias_only: bool = False,
    anchor_feature_residual_include_alias_positives: bool = False,
    freeze_shared_metric: bool = False,
    soft_pose_weight: float = 0.0,
    soft_pose_topk: int = 8,
    soft_pose_temperature: float = 0.05,
    soft_pose_inlier_softness_px: float = 1.0,
    soft_pose_minimum_depth: float = 1e-4,
    soft_pose_miss_weight: float = 0.05,
    task_translation_m: float = 0.05,
    task_rotation_deg: float = 5.0,
    ransac_reprojection_px: float = 12.0,
    clean_reprojection_px: float = 4.0,
    seed: int = 2026,
    leave_one_query_out_track_descriptors: bool = False,
    loo_descriptor_trim_fraction: float = 0.2,
    cpu_threads: int = 1,
) -> dict:
    if not 0.0 <= float(alias_query_replay_fraction) <= 1.0:
        raise ValueError("alias_query_replay_fraction must lie in [0, 1]")
    if float(alias_weight) < 0.0 or float(protected_clean_weight) < 0.0:
        raise ValueError("alias and protected-clean weights must be non-negative")
    if not 0.0 <= float(protected_clean_task_scale) <= 1.0:
        raise ValueError("protected_clean_task_scale must lie in [0, 1]")
    if float(anchor_feature_residual_max_norm) < 0.0:
        raise ValueError("anchor feature residual bound must be non-negative")
    if int(cpu_threads) <= 0:
        raise ValueError("CPU thread count must be positive")
    torch.set_num_threads(int(cpu_threads))
    torch.manual_seed(int(seed))
    device = torch.device("cuda")
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    graph = torch.load(function_graph_path, map_location="cpu", weights_only=False)
    payload = torch.load(track_payload_path, map_location="cpu", weights_only=False)
    cache_payload = torch.load(query_cache_path, map_location="cpu", weights_only=False)
    teacher = torch.load(positive_teacher_path, map_location="cpu", weights_only=False)
    cache = cache_payload.get("queries", cache_payload)
    names = list(graph["query_names"])
    records, data_report = _build_training_records(
        graph, payload, state, teacher, max_positives
    )
    records, row_limit_report = limit_training_records(records, deployment_row_limit)
    data_report.update(row_limit_report)
    density_prefixes = resolve_density_prefixes(
        records, deployment_row_limit, density_prefix_fractions
    )
    # Fused Track descriptors are already the exact normalized deployment raw
    # bank.  Keep their serialized values bitwise intact for the LOO replay;
    # SharedLowRankMetric performs the required normalization internally.
    raw_features_cpu = torch.as_tensor(
        state.get("v7_metric_raw_features", state["anchor_features"])
    ).float()
    loo_descriptor_bank = None
    if leave_one_query_out_track_descriptors:
        if (
            payload.get("rendered_rgb_only") is not True
            or cache_payload.get("uses_source_mapping_rgb") is not False
            or cache_payload.get("uses_test_queries") is not False
        ):
            raise ValueError(
                "LOO Track descriptor training requires source-image-free mapping inputs"
            )
        loo_payload = track_descriptor_payload_for_loo(payload)
        if bool((torch.as_tensor(state["track_cluster_ids"]) < 0).any()):
            loo_descriptor_bank = LeaveOneQueryOutProjectiveAnchorDescriptorBank(
                state=state,
                payload=loo_payload,
                query_cache=cache_payload,
                reference_features=raw_features_cpu,
                trim_fraction=float(loo_descriptor_trim_fraction),
            )
        else:
            loo_descriptor_bank = LeaveOneQueryOutTrackDescriptorBank(
                payload=loo_payload,
                query_cache=cache_payload,
                track_indices=state["track_cluster_ids"],
                reference_features=raw_features_cpu,
                trim_fraction=float(loo_descriptor_trim_fraction),
            )
    raw_features = raw_features_cpu.to(device)
    metric = SharedLowRankMetric(
        descriptor_dim=raw_features.shape[1],
        rank=rank,
        max_residual_norm=metric_residual,
    ).to(device)
    if initial_metric_state_path is not None:
        initial_metric = torch.load(
            initial_metric_state_path, map_location="cpu", weights_only=False
        )
        expected_ids = torch.as_tensor(state["anchor_ids"]).long().reshape(-1)
        metric_ids = (
            torch.as_tensor(initial_metric["landmark_indices"]).long().reshape(-1)
        )
        if not torch.equal(metric_ids, expected_ids):
            raise ValueError("initial metric state does not align with the anchor map")
        if dict(initial_metric["metric_config"]) != metric.export_config():
            raise ValueError(
                "initial metric configuration differs from requested training"
            )
        metric.load_state_dict(initial_metric["metric_state_dict"])
    if freeze_shared_metric:
        for parameter in metric.parameters():
            parameter.requires_grad_(False)
    anchor_residual_parameter = None
    if float(anchor_feature_residual_max_norm) > 0.0:
        initial_anchor_residual = state.get(
            "v7_anchor_residual_parameter", torch.zeros_like(raw_features.cpu())
        )
        anchor_residual_parameter = nn.Parameter(
            torch.as_tensor(initial_anchor_residual).float().to(device)
        )
        if anchor_residual_parameter.shape != raw_features.shape:
            raise ValueError("anchor residual state does not align with compact map")
    parameters_to_optimize = [
        parameter for parameter in metric.parameters() if parameter.requires_grad
    ]
    if anchor_residual_parameter is not None:
        parameters_to_optimize.append(anchor_residual_parameter)
    if not parameters_to_optimize:
        raise ValueError("descriptor reconstruction has no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters_to_optimize, lr=learning_rate, weight_decay=1e-4
    )
    groups = torch.as_tensor(data_report["query_groups"]).long()
    group_count = int(groups.max()) + 1
    group_weights = torch.ones(group_count, device=device) / group_count
    clean_pairs: dict = {}
    harmful_pairs: dict = {}
    false_top1_pairs: dict = {}
    clean_margin_pairs: dict = {}
    alias_teacher = RecurrentAliasTeacher(
        row_anchors={},
        row_weights={},
        active_anchors=torch.empty(0, dtype=torch.long),
        diagnostics={
            "observed_false_assignment_count": 0,
            "observed_false_anchor_count": 0,
            "solver_conditioned_alias": bool(alias_require_harmful_inlier),
            "solver_conditioned_false_assignment_count": 0,
            "solver_conditioned_false_anchor_count": 0,
            "recurrent_alias_anchor_count": 0,
            "active_alias_anchor_count": 0,
            "active_alias_query_group_count": 0,
            "active_alias_row_count": 0,
            "active_alias_anchor_fraction": 0.0,
        },
    )
    alias_repair_anchors = torch.empty(0, dtype=torch.long)
    alias_repair_diagnostics = {
        "alias_false_trainable_anchor_count": 0,
        "alias_positive_trainable_anchor_count": 0,
        "alias_pair_trainable_anchor_count": 0,
    }
    generator = torch.Generator().manual_seed(int(seed) + 1)
    shards = _build_rotating_shards(groups, refresh_shards)
    refresh_index = 0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    density_weights = torch.ones(len(density_prefixes), device=device)
    density_weights /= density_weights.sum()
    config = {
        "steps": int(steps),
        "batch_size": int(batch_size),
        "topk": int(topk),
        "max_positives": int(max_positives),
        "rank": int(rank),
        "metric_residual": float(metric_residual),
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "harmful_weight": float(harmful_weight),
        "trust_weight": float(trust_weight),
        "group_dro_eta": float(group_dro_eta),
        "group_dro_max_weight_ratio": float(group_dro_max_weight_ratio),
        "refresh_interval": int(refresh_interval),
        "refresh_shards": int(refresh_shards),
        "initial_ransac_refresh": bool(initial_ransac_refresh),
        "deployment_row_limit": int(deployment_row_limit),
        "density_prefix_fractions": [
            float(value) for value in density_prefix_fractions
        ],
        "density_prefixes": list(density_prefixes),
        "density_dro_eta": float(density_dro_eta),
        "density_dro_max_weight_ratio": float(density_dro_max_weight_ratio),
        "alias_weight": float(alias_weight),
        "alias_margin": float(alias_margin),
        "alias_minimum_distinct_groups": int(alias_minimum_distinct_groups),
        "alias_minimum_queries": int(alias_minimum_queries),
        "alias_minimum_occurrences": int(alias_minimum_occurrences),
        "alias_minimum_rows_per_query": int(alias_minimum_rows_per_query),
        "alias_query_replay_fraction": float(alias_query_replay_fraction),
        "alias_require_harmful_inlier": bool(alias_require_harmful_inlier),
        "protected_clean_weight": float(protected_clean_weight),
        "protected_clean_minimum_margin": float(protected_clean_minimum_margin),
        "protected_clean_margin_slack": float(protected_clean_margin_slack),
        "protected_clean_task_scale": float(protected_clean_task_scale),
        "anchor_feature_residual_max_norm": float(anchor_feature_residual_max_norm),
        "anchor_feature_residual_trust_weight": float(
            anchor_feature_residual_trust_weight
        ),
        "anchor_feature_residual_alias_only": bool(anchor_feature_residual_alias_only),
        "anchor_feature_residual_include_alias_positives": bool(
            anchor_feature_residual_include_alias_positives
        ),
        "freeze_shared_metric": bool(freeze_shared_metric),
        "soft_pose_weight": float(soft_pose_weight),
        "soft_pose_topk": int(soft_pose_topk),
        "soft_pose_temperature": float(soft_pose_temperature),
        "soft_pose_inlier_softness_px": float(soft_pose_inlier_softness_px),
        "soft_pose_minimum_depth": float(soft_pose_minimum_depth),
        "soft_pose_miss_weight": float(soft_pose_miss_weight),
        "task_translation_m": float(task_translation_m),
        "task_rotation_deg": float(task_rotation_deg),
        "ransac_reprojection_px": float(ransac_reprojection_px),
        "clean_reprojection_px": float(clean_reprojection_px),
        "seed": int(seed),
        "leave_one_query_out_track_descriptors": bool(
            leave_one_query_out_track_descriptors
        ),
        "loo_descriptor_trim_fraction": float(loo_descriptor_trim_fraction),
        "formal_method_uses_crossfit": False,
        "cpu_threads": int(cpu_threads),
        "initial_metric_state": (
            str(Path(initial_metric_state_path).resolve())
            if initial_metric_state_path is not None
            else None
        ),
        **data_report,
    }
    checkpoints = set(int(value) for value in checkpoint_steps)
    for step in range(1, int(steps) + 1):
        if (step == 1 and initial_ransac_refresh) or (
            step > 1
            and refresh_interval > 0
            and (step - 1) % int(refresh_interval) == 0
        ):
            shard_index = refresh_index % len(shards)
            refresh_results = []
            for density_prefix in density_prefixes:
                refresh_results.append(
                    _refresh_ransac_outcomes(
                        metric=metric,
                        raw_features=raw_features,
                        state=state,
                        cache=cache,
                        names=names,
                        groups=groups,
                        training_records=records,
                        device=device,
                        query_indices=shards[shard_index],
                        seed=seed,
                        ransac_reprojection_px=ransac_reprojection_px,
                        clean_reprojection_px=clean_reprojection_px,
                        deployment_row_limit=density_prefix,
                        anchor_residual_parameter=anchor_residual_parameter,
                        anchor_residual_max_norm=anchor_feature_residual_max_norm,
                        loo_descriptor_bank=loo_descriptor_bank,
                    )
                )
            (
                harmful_prior,
                _,
                outcomes,
                clean,
                harmful,
                refreshed_false_top1,
                refreshed_clean_margins,
            ) = refresh_results[-1]
            churn = _replace_refreshed_pairs(
                clean_pairs, harmful_pairs, shards[shard_index], clean, harmful
            )
            replace_query_assignments(
                false_top1_pairs,
                shards[shard_index],
                refreshed_false_top1,
            )
            replace_query_assignments(
                clean_margin_pairs,
                shards[shard_index],
                refreshed_clean_margins,
            )
            alias_teacher = build_recurrent_alias_teacher(
                false_top1_pairs,
                groups,
                anchor_count=int(raw_features.shape[0]),
                minimum_distinct_groups=alias_minimum_distinct_groups,
                minimum_queries=alias_minimum_queries,
                minimum_occurrences=alias_minimum_occurrences,
                minimum_rows_per_query=alias_minimum_rows_per_query,
                solver_harmful_assignments=(
                    harmful_pairs if alias_require_harmful_inlier else None
                ),
            )
            alias_repair_anchors, alias_repair_diagnostics = (
                alias_repair_anchor_indices(
                    alias_teacher,
                    records,
                    include_positives=(anchor_feature_residual_include_alias_positives),
                )
            )
            risk = torch.zeros_like(group_weights)
            for _, group_risks, *_ in refresh_results:
                for group, value in group_risks.items():
                    risk[group] = max(float(risk[group]), float(value))
            group_weights = _update_group_dro_weights(
                group_weights,
                risk,
                eta=group_dro_eta,
                maximum_uniform_ratio=group_dro_max_weight_ratio,
            )
            density_risk = torch.as_tensor(
                [
                    _group_pose_risk([value["te_cm"] for value in result[2]])
                    for result in refresh_results
                ],
                device=device,
                dtype=density_weights.dtype,
            )
            density_weights = _update_group_dro_weights(
                density_weights,
                density_risk,
                eta=density_dro_eta,
                maximum_uniform_ratio=density_dro_max_weight_ratio,
            )
            density_outcomes = []
            for prefix, result, risk_value, weight in zip(
                density_prefixes,
                refresh_results,
                density_risk.tolist(),
                density_weights.tolist(),
            ):
                values = result[2]
                density_outcomes.append(
                    {
                        "prefix": int(prefix),
                        "mean_te_cm": float(
                            np.mean([value["te_cm"] for value in values])
                        ),
                        "risk": float(risk_value),
                        "weight": float(weight),
                        "mean_hypotheses": float(
                            np.mean(
                                [
                                    value["hypotheses"]
                                    for value in values
                                    if value["hypotheses"] is not None
                                ]
                            )
                        ),
                    }
                )
            row = {
                "step": step - 1,
                "event": "self_localization_refresh",
                "shard": shard_index,
                "shard_query_count": len(shards[shard_index]),
                "mean_te_cm": float(np.mean([value["te_cm"] for value in outcomes])),
                "mean_candidate_count": float(
                    np.mean([value["candidate_count"] for value in outcomes])
                ),
                "mean_hypotheses": float(
                    np.mean(
                        [
                            value["hypotheses"]
                            for value in outcomes
                            if value["hypotheses"] is not None
                        ]
                    )
                ),
                "harmful_anchor_fraction": float((harmful_prior > 0).float().mean()),
                "group_weight_max": float(group_weights.max()),
                "group_weight_effective_count": float(
                    group_weights.sum().square()
                    / group_weights.square().sum().clamp_min(1e-8)
                ),
                "density_outcomes": density_outcomes,
                "density_weight_max": float(density_weights.max()),
                **alias_teacher.diagnostics,
                **alias_repair_diagnostics,
                **churn,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            refresh_index += 1

        active_alias_queries = tuple(sorted(alias_teacher.row_anchors))
        replay_alias_query = (
            float(alias_weight) > 0.0
            and bool(active_alias_queries)
            and float(torch.rand((), generator=generator))
            < float(alias_query_replay_fraction)
        )
        if replay_alias_query:
            query_index = active_alias_queries[
                int(torch.randint(len(active_alias_queries), (1,), generator=generator))
            ]
        else:
            query_index = int(torch.randint(len(records), (1,), generator=generator))
        record = records[query_index]
        query_alias_anchors = alias_teacher.row_anchors.get(query_index, {})
        query_alias_weights = alias_teacher.row_weights.get(query_index, {})
        density_index = (step - 1) % len(density_prefixes)
        density_prefix = density_prefixes[density_index]
        eligible_rows = torch.nonzero(
            record["cache_rows"] < int(density_prefix), as_tuple=False
        ).reshape(-1)
        count = int(eligible_rows.numel())
        if count == 0:
            continue
        sample_count = min(int(batch_size), count)
        if float(alias_weight) > 0.0 and query_alias_anchors:
            mandatory_mask = torch.as_tensor(
                [
                    int(record["cache_rows"][row]) in query_alias_anchors
                    for row in eligible_rows
                ],
                dtype=torch.bool,
            )
            mandatory = eligible_rows[mandatory_mask]
            optional = eligible_rows[~mandatory_mask]
            if mandatory.numel() > sample_count:
                mandatory = mandatory[
                    torch.randperm(mandatory.numel(), generator=generator)[
                        :sample_count
                    ]
                ]
            remaining = sample_count - int(mandatory.numel())
            optional = optional[
                torch.randperm(optional.numel(), generator=generator)[:remaining]
            ]
            rows = torch.cat((mandatory, optional))
        elif float(protected_clean_weight) > 0.0:
            rows = eligible_rows[
                torch.randperm(count, generator=generator)[:sample_count]
            ]
        else:
            rows = eligible_rows[
                torch.randint(count, (sample_count,), generator=generator)
            ]
        cache_rows = record["cache_rows"][rows]
        query = F.normalize(
            torch.as_tensor(cache[names[query_index]]["native_descriptors"]).float()[
                cache_rows
            ],
            dim=1,
        ).to(device)
        positives = record["positives"][rows].to(device)
        ignored = record["ignored_anchors"][rows].to(device)
        matchable = record["matchable"][rows].to(device)

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
                    (positives.shape[0],), positives.shape[1] - 1, device=device
                ),
            )
            positives[
                torch.arange(positives.shape[0], device=device)[add_clean],
                replace[add_clean],
            ] = clean_survivors[add_clean]
            matchable |= add_clean
        current_harmful = harmful_pairs.get(query_index, {})
        harmful_survivors = torch.as_tensor(
            [current_harmful.get(int(row), -1) for row in cache_rows],
            dtype=torch.long,
            device=device,
        )[:, None]
        harmful_survivors = torch.where(
            ((harmful_survivors == ignored) & (ignored >= 0)).any(dim=1, keepdim=True),
            torch.full_like(harmful_survivors, -1),
            harmful_survivors,
        )

        alias_anchors = torch.as_tensor(
            [query_alias_anchors.get(int(row), -1) for row in cache_rows],
            dtype=torch.long,
            device=device,
        )
        alias_row_weights = torch.as_tensor(
            [query_alias_weights.get(int(row), 0.0) for row in cache_rows],
            dtype=torch.float32,
            device=device,
        )
        current_clean_margins = clean_margin_pairs.get(query_index, {})
        protected_margin = torch.as_tensor(
            [current_clean_margins.get(int(row), float("nan")) for row in cache_rows],
            dtype=torch.float32,
            device=device,
        )
        protected_anchor = clean_survivors.clone()
        protect = (
            (protected_anchor >= 0)
            & torch.isfinite(protected_margin)
            & (protected_margin >= float(protected_clean_minimum_margin))
        )
        protected_anchor[~protect] = -1
        protected_floor = protected_margin - float(protected_clean_margin_slack)
        protected_floor[~protect] = torch.nan

        adapted_query, query_residual = metric(query)
        (
            adapted_anchor,
            shared_anchor_residual,
            anchor_feature_residual,
            loo_affected_anchor_count,
        ) = bounded_query_anchor_bank(
            metric=metric,
            raw_features=raw_features,
            query_index=query_index,
            loo_descriptor_bank=loo_descriptor_bank,
            anchor_residual_parameter=anchor_residual_parameter,
            maximum_norm=anchor_feature_residual_max_norm,
        )
        list_loss = torch.zeros(adapted_query.shape[0], device=device)
        if bool(matchable.any()):
            list_loss[matchable] = _multi_positive_list_loss(
                adapted_query[matchable],
                adapted_anchor,
                positives[matchable],
                ignored[matchable],
                harmful_survivors[matchable],
                topk=topk,
                temperature=temperature,
                harmful_weight=harmful_weight,
            )
        if bool(matchable.any()):
            all_row_weights = torch.ones_like(list_loss)
            if float(protected_clean_weight) > 0.0:
                all_row_weights[protect] = float(protected_clean_task_scale)
            row_weights = all_row_weights[matchable]
            task_loss = (
                list_loss[matchable] * row_weights
            ).sum() / row_weights.sum().clamp_min(1e-8)
        else:
            task_loss = torch.zeros((), device=device)
        alias_loss, alias_diagnostics = alias_group_ranking_loss(
            adapted_query,
            adapted_anchor,
            positives,
            alias_anchors,
            alias_row_weights,
            margin=alias_margin,
            temperature=temperature,
        )
        protected_loss, protected_diagnostics = protected_clean_margin_loss(
            adapted_query,
            adapted_anchor,
            protected_anchor,
            protected_floor,
        )
        task_scale = (
            group_weights[int(record["group"])]
            * float(group_count)
            * density_weights[density_index]
            * float(len(density_prefixes))
        )
        task_loss *= task_scale
        alias_loss *= task_scale
        protected_loss *= task_scale
        trust_loss = (
            query_residual.square().sum(dim=1).mean()
            + shared_anchor_residual.square().sum(dim=1).mean()
            + torch.zeros((), device=device)
        )
        anchor_feature_trust_loss = anchor_feature_residual.square().sum(dim=1).mean()
        if float(soft_pose_weight) > 0.0:
            cached = cache[names[query_index]]
            keypoint_xy = torch.as_tensor(cached["native_keypoints"]).float()[
                cache_rows
            ] + float(cached.get("pixel_center_offset", 0.5))
            soft_pose_loss, soft_pose_diagnostics = soft_pose_bias_loss(
                query_features=adapted_query,
                anchor_features=adapted_anchor,
                anchor_xyz=torch.as_tensor(state["anchor_xyz"]).float(),
                keypoint_xy=keypoint_xy,
                intrinsic=torch.as_tensor(cached["native_K"]).float(),
                pose_gt_w2c=torch.as_tensor(cached["pose_w2c"]).float(),
                topk=soft_pose_topk,
                temperature=soft_pose_temperature,
                inlier_threshold_px=clean_reprojection_px,
                inlier_softness_px=soft_pose_inlier_softness_px,
                minimum_depth=soft_pose_minimum_depth,
                miss_weight=soft_pose_miss_weight,
                task_translation_m=task_translation_m,
                task_rotation_deg=task_rotation_deg,
            )
        else:
            soft_pose_loss = task_loss.new_zeros(())
            soft_pose_diagnostics = {"soft_pose_active": 0.0}
        loss = (
            task_loss
            + float(alias_weight) * alias_loss
            + float(protected_clean_weight) * protected_loss
            + float(trust_weight) * trust_loss
            + float(anchor_feature_residual_trust_weight) * anchor_feature_trust_loss
            + float(soft_pose_weight) * soft_pose_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                "Non-finite descriptor reconstruction loss at "
                f"step={step}, query_index={query_index}, "
                f"task={float(task_loss.detach())}, "
                f"alias={float(alias_loss.detach())}, "
                f"protected={float(protected_loss.detach())}, "
                f"trust={float(trust_loss.detach())}, "
                f"anchor_feature_trust={float(anchor_feature_trust_loss.detach())}, "
                f"soft_pose={float(soft_pose_loss.detach())}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable_anchor_count = int(raw_features.shape[0])
        if anchor_feature_residual_alias_only and anchor_residual_parameter is not None:
            trainable = torch.zeros(
                raw_features.shape[0], dtype=torch.bool, device=device
            )
            active = alias_repair_anchors.to(device)
            if active.numel():
                trainable[active] = True
            if anchor_residual_parameter.grad is not None:
                anchor_residual_parameter.grad.mul_(trainable[:, None])
            trainable_anchor_count = int(trainable.sum())
        torch.nn.utils.clip_grad_norm_(parameters_to_optimize, 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "task_loss": float(task_loss.detach()),
                "alias_loss": float(alias_loss.detach()),
                "protected_clean_loss": float(protected_loss.detach()),
                "trust_loss": float(trust_loss.detach()),
                "anchor_feature_trust_loss": float(anchor_feature_trust_loss.detach()),
                "anchor_feature_residual_mean": float(
                    torch.linalg.norm(anchor_feature_residual.detach(), dim=1).mean()
                ),
                "anchor_feature_residual_max": float(
                    torch.linalg.norm(anchor_feature_residual.detach(), dim=1).max()
                ),
                "trainable_anchor_count": int(trainable_anchor_count),
                "soft_pose_loss": float(soft_pose_loss.detach()),
                "matchable_fraction": float(matchable.float().mean()),
                "group": int(record["group"]),
                "density_prefix": int(density_prefix),
                "density_weight": float(density_weights[density_index]),
                "alias_query_replay": bool(replay_alias_query),
                "active_alias_query_count": int(len(active_alias_queries)),
                "loo_affected_anchor_count": int(loo_affected_anchor_count),
                **alias_diagnostics,
                **protected_diagnostics,
                **soft_pose_diagnostics,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
        if step in checkpoints or step == steps:
            _save_checkpoint(
                output_dir,
                step,
                state,
                metric,
                raw_features,
                history,
                config,
                anchor_residual_parameter=anchor_residual_parameter,
                anchor_residual_max_norm=anchor_feature_residual_max_norm,
            )
    report = {
        "schema": "lafgs_self_localization_training",
        "version": 1,
        "config": config,
        "history": history,
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--function-graph", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--complete-positive-teacher", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-metric-state")
    parser.add_argument("--steps", type=int, default=175)
    parser.add_argument("--checkpoint-steps", default="175")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max-positives", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--metric-residual", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--harmful-weight", type=float, default=0.1)
    parser.add_argument("--trust-weight", type=float, default=1.0)
    parser.add_argument("--group-dro-eta", type=float, default=0.03)
    parser.add_argument("--group-dro-max-weight-ratio", type=float, default=1e9)
    parser.add_argument("--refresh-interval", type=int, default=0)
    parser.add_argument("--refresh-shards", type=int, default=7)
    parser.add_argument(
        "--initial-ransac-refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--density-prefix-fractions", default="1.0")
    parser.add_argument("--density-dro-eta", type=float, default=0.03)
    parser.add_argument("--density-dro-max-weight-ratio", type=float, default=3.0)
    parser.add_argument("--alias-weight", type=float, default=0.0)
    parser.add_argument("--alias-margin", type=float, default=0.05)
    parser.add_argument("--alias-minimum-distinct-groups", type=int, default=2)
    parser.add_argument("--alias-minimum-queries", type=int, default=3)
    parser.add_argument("--alias-minimum-occurrences", type=int, default=6)
    parser.add_argument("--alias-minimum-rows-per-query", type=int, default=2)
    parser.add_argument("--alias-query-replay-fraction", type=float, default=0.5)
    parser.add_argument(
        "--alias-require-harmful-inlier",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--protected-clean-weight", type=float, default=0.0)
    parser.add_argument("--protected-clean-minimum-margin", type=float, default=0.05)
    parser.add_argument("--protected-clean-margin-slack", type=float, default=0.01)
    parser.add_argument("--protected-clean-task-scale", type=float, default=0.25)
    parser.add_argument("--anchor-feature-residual-max-norm", type=float, default=0.0)
    parser.add_argument(
        "--anchor-feature-residual-trust-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--anchor-feature-residual-alias-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--anchor-feature-residual-include-alias-positives",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--freeze-shared-metric",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--soft-pose-weight", type=float, default=0.0)
    parser.add_argument("--soft-pose-topk", type=int, default=8)
    parser.add_argument("--soft-pose-temperature", type=float, default=0.05)
    parser.add_argument("--soft-pose-inlier-softness-px", type=float, default=1.0)
    parser.add_argument("--soft-pose-minimum-depth", type=float, default=1e-4)
    parser.add_argument("--soft-pose-miss-weight", type=float, default=0.05)
    parser.add_argument("--task-translation-m", type=float, default=0.05)
    parser.add_argument("--task-rotation-deg", type=float, default=5.0)
    parser.add_argument("--ransac-reprojection-px", type=float, default=12.0)
    parser.add_argument("--clean-reprojection-px", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--leave-one-query-out-track-descriptors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--loo-descriptor-trim-fraction", type=float, default=0.2)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    train(
        map_path=args.map,
        function_graph_path=args.function_graph,
        track_payload_path=args.track_payload,
        query_cache_path=args.query_cache,
        positive_teacher_path=args.complete_positive_teacher,
        output_dir=args.output_dir,
        initial_metric_state_path=args.initial_metric_state,
        steps=args.steps,
        checkpoint_steps=tuple(
            int(value) for value in args.checkpoint_steps.split(",") if value
        ),
        batch_size=args.batch_size,
        topk=args.topk,
        max_positives=args.max_positives,
        rank=args.rank,
        metric_residual=args.metric_residual,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        harmful_weight=args.harmful_weight,
        trust_weight=args.trust_weight,
        group_dro_eta=args.group_dro_eta,
        group_dro_max_weight_ratio=args.group_dro_max_weight_ratio,
        refresh_interval=args.refresh_interval,
        refresh_shards=args.refresh_shards,
        initial_ransac_refresh=args.initial_ransac_refresh,
        deployment_row_limit=args.deployment_row_limit,
        density_prefix_fractions=tuple(
            float(value) for value in args.density_prefix_fractions.split(",") if value
        ),
        density_dro_eta=args.density_dro_eta,
        density_dro_max_weight_ratio=args.density_dro_max_weight_ratio,
        alias_weight=args.alias_weight,
        alias_margin=args.alias_margin,
        alias_minimum_distinct_groups=args.alias_minimum_distinct_groups,
        alias_minimum_queries=args.alias_minimum_queries,
        alias_minimum_occurrences=args.alias_minimum_occurrences,
        alias_minimum_rows_per_query=args.alias_minimum_rows_per_query,
        alias_query_replay_fraction=args.alias_query_replay_fraction,
        alias_require_harmful_inlier=args.alias_require_harmful_inlier,
        protected_clean_weight=args.protected_clean_weight,
        protected_clean_minimum_margin=args.protected_clean_minimum_margin,
        protected_clean_margin_slack=args.protected_clean_margin_slack,
        protected_clean_task_scale=args.protected_clean_task_scale,
        anchor_feature_residual_max_norm=args.anchor_feature_residual_max_norm,
        anchor_feature_residual_trust_weight=(
            args.anchor_feature_residual_trust_weight
        ),
        anchor_feature_residual_alias_only=(args.anchor_feature_residual_alias_only),
        anchor_feature_residual_include_alias_positives=(
            args.anchor_feature_residual_include_alias_positives
        ),
        freeze_shared_metric=args.freeze_shared_metric,
        soft_pose_weight=args.soft_pose_weight,
        soft_pose_topk=args.soft_pose_topk,
        soft_pose_temperature=args.soft_pose_temperature,
        soft_pose_inlier_softness_px=args.soft_pose_inlier_softness_px,
        soft_pose_minimum_depth=args.soft_pose_minimum_depth,
        soft_pose_miss_weight=args.soft_pose_miss_weight,
        task_translation_m=args.task_translation_m,
        task_rotation_deg=args.task_rotation_deg,
        ransac_reprojection_px=args.ransac_reprojection_px,
        clean_reprojection_px=args.clean_reprojection_px,
        seed=args.seed,
        leave_one_query_out_track_descriptors=(
            args.leave_one_query_out_track_descriptors
        ),
        loo_descriptor_trim_fraction=args.loo_descriptor_trim_fraction,
        cpu_threads=args.cpu_threads,
    )


if __name__ == "__main__":
    main()
