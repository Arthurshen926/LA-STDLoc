"""Cross-fit training and evaluation for the bounded context adapter."""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np
import torch
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
    MapConsistentContextAdapter,
    context_from_cached_query,
)
from map_learning.repeated_assignment_audit import (
    _pose_summary,
    _selected_csr_edges,
    _solve_assignments,
)
from map_learning.trainer import _multi_positive_list_loss


def _csr_first_k(offsets, indices, width: int) -> torch.Tensor:
    offsets = torch.as_tensor(offsets).long().reshape(-1)
    indices = torch.as_tensor(indices).long().reshape(-1)
    output = torch.full((offsets.numel() - 1, int(width)), -1, dtype=torch.long)
    counts = offsets[1:] - offsets[:-1]
    if not indices.numel():
        return output
    rows = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    rank = torch.arange(indices.numel()) - offsets[rows]
    keep = rank < int(width)
    output[rows[keep], rank[keep]] = indices[keep]
    return output


def _remap_dense_anchors(values: torch.Tensor, lookup: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values).long()
    valid = values >= 0
    output = torch.full_like(values, -1)
    if bool(valid.any()):
        output[valid] = lookup[values[valid]]
    return output


@torch.inference_mode()
def build_raw_observation_bank(
    *,
    teacher: dict,
    query_cache: dict,
    support_query_indices: Sequence[int],
    device: torch.device,
    minimum_support_views: int = 2,
    progress_interval: int = 0,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Fuse a support-only raw-SP bank with equal per-image view weights."""
    if minimum_support_views < 1:
        raise ValueError("minimum support views must be positive")
    names = list(teacher["query_names"])
    cache = query_cache.get("queries", query_cache)
    support = [int(value) for value in support_query_indices]
    anchor_count = int(teacher["anchor_count"])
    raw_sum = torch.zeros((anchor_count, 256), device=device)
    view_counts = torch.zeros(anchor_count, dtype=torch.long, device=device)
    positive_edge_count = 0
    anchor_view_count = 0
    for completed, query_index in enumerate(support, start=1):
        record = teacher["records"][query_index]
        rows = torch.as_tensor(record["query_rows"]).long()
        _, edge_rows, edge_anchors = _selected_csr_edges(
            record, "positive", torch.arange(rows.numel())
        )
        if edge_rows.numel():
            cached = cache[names[query_index]]
            raw = F.normalize(
                torch.as_tensor(cached["native_descriptors"])
                .float()[rows[edge_rows]]
                .to(device),
                dim=1,
            )
            observed = accumulate_view_descriptors(
                raw_sum, view_counts, edge_anchors.to(device), raw
            )
            positive_edge_count += int(edge_rows.numel())
            anchor_view_count += observed
        if progress_interval > 0 and (
            completed % int(progress_interval) == 0 or completed == len(support)
        ):
            print(
                {
                    "event": "mccd_raw_support",
                    "queries_complete": completed,
                    "query_count": len(support),
                },
                flush=True,
            )
    supported = view_counts >= int(minimum_support_views)
    if not bool(supported.any()):
        raise ValueError("support partition did not observe any deployable anchors")
    anchor_indices = torch.nonzero(supported, as_tuple=False).reshape(-1)
    return {
        "anchor_indices": anchor_indices,
        "raw_superpoint": F.normalize(raw_sum[anchor_indices], dim=1),
        "view_counts": view_counts[anchor_indices],
    }, {
        "support_query_count": len(support),
        "positive_edge_count": int(positive_edge_count),
        "anchor_view_count": int(anchor_view_count),
        "minimum_support_views": int(minimum_support_views),
        "supported_anchor_count": int(supported.sum()),
        "unsupported_anchor_count": int((~supported).sum()),
        "supported_view_count_minimum": int(view_counts[supported].min()),
        "supported_view_count_median": float(view_counts[supported].float().median()),
        "supported_view_count_maximum": int(view_counts[supported].max()),
    }


@torch.inference_mode()
def build_context_observation_bank(
    *,
    adapter: MapConsistentContextAdapter,
    teacher: dict,
    query_cache: dict,
    support_query_indices: Sequence[int],
    anchor_indices: torch.Tensor,
    expected_view_counts: torch.Tensor,
    device: torch.device,
    progress_interval: int = 0,
) -> tuple[torch.Tensor, dict]:
    """Fuse adapted descriptors from the identical raw support observations."""
    names = list(teacher["query_names"])
    cache = query_cache.get("queries", query_cache)
    support = [int(value) for value in support_query_indices]
    anchor_count = int(teacher["anchor_count"])
    context_sum = torch.zeros((anchor_count, adapter.descriptor_dim), device=device)
    view_counts = torch.zeros(anchor_count, dtype=torch.long, device=device)
    residual_norms = []
    for completed, query_index in enumerate(support, start=1):
        record = teacher["records"][query_index]
        rows = torch.as_tensor(record["query_rows"]).long()
        _, edge_rows, edge_anchors = _selected_csr_edges(
            record, "positive", torch.arange(rows.numel())
        )
        if edge_rows.numel():
            native_rows = rows[edge_rows]
            raw, tokens = context_from_cached_query(
                cache[names[query_index]],
                native_rows,
                device=device,
                kernels=adapter.context_kernels,
                context_mode=adapter.context_mode,
            )
            adapted, residual = adapter(raw, tokens)
            accumulate_view_descriptors(
                context_sum,
                view_counts,
                edge_anchors.to(device),
                adapted,
            )
            residual_norms.extend(
                torch.linalg.norm(residual, dim=1).detach().cpu().tolist()
            )
        if progress_interval > 0 and (
            completed % int(progress_interval) == 0 or completed == len(support)
        ):
            print(
                {
                    "event": "mccd_context_support",
                    "queries_complete": completed,
                    "query_count": len(support),
                },
                flush=True,
            )
    selected = torch.as_tensor(anchor_indices, device=device).long()
    if not torch.equal(view_counts[selected], expected_view_counts.to(device)):
        raise AssertionError("raw and contextual support view counts diverged")
    values = np.asarray(residual_norms, dtype=np.float64)
    return F.normalize(context_sum[selected], dim=1), {
        "adapted_observation_count": int(values.size),
        "residual_norm_mean": float(values.mean()) if values.size else 0.0,
        "residual_norm_median": float(np.median(values)) if values.size else 0.0,
        "residual_norm_p90": float(np.percentile(values, 90)) if values.size else 0.0,
        "residual_norm_maximum": float(values.max()) if values.size else 0.0,
    }


def prepare_training_records(
    *,
    teacher: dict,
    support_query_indices: Sequence[int],
    anchor_indices: torch.Tensor,
    maximum_positives: int = 8,
    maximum_ignored: int = 16,
) -> tuple[dict[int, dict], dict]:
    """Materialize support-only positives in the filtered bank index space."""
    anchor_count = int(teacher["anchor_count"])
    lookup = torch.full((anchor_count,), -1, dtype=torch.long)
    lookup[torch.as_tensor(anchor_indices).long().cpu()] = torch.arange(
        torch.as_tensor(anchor_indices).numel()
    )
    records = {}
    matchable_rows = 0
    positive_pairs = 0
    for query_index in support_query_indices:
        record = teacher["records"][int(query_index)]
        positives = _remap_dense_anchors(
            _csr_first_k(
                record["positive_offsets"],
                record["positive_indices"],
                maximum_positives,
            ),
            lookup,
        )
        ignored = _remap_dense_anchors(
            _csr_first_k(
                record["ambiguous_offsets"],
                record["ambiguous_indices"],
                maximum_ignored,
            ),
            lookup,
        )
        matchable = (positives >= 0).any(dim=1)
        records[int(query_index)] = {
            "native_rows": torch.as_tensor(record["query_rows"]).long(),
            "positives": positives,
            "ignored": ignored,
            "matchable_rows": torch.nonzero(matchable, as_tuple=False).reshape(-1),
        }
        matchable_rows += int(matchable.sum())
        positive_pairs += int((positives >= 0).sum())
    return records, {
        "training_query_count": len(records),
        "training_matchable_row_count": int(matchable_rows),
        "training_positive_pair_count": int(positive_pairs),
        "maximum_positives": int(maximum_positives),
        "maximum_ignored": int(maximum_ignored),
    }


def train_context_adapter_stage(
    *,
    adapter: MapConsistentContextAdapter,
    teacher: dict,
    query_cache: dict,
    support_query_indices: Sequence[int],
    records: dict[int, dict],
    raw_reference_bank: torch.Tensor,
    task_bank: torch.Tensor,
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 256,
    topk: int = 64,
    learning_rate: float = 5e-4,
    temperature: float = 0.04,
    collision_weight: float = 1.0,
    clean_weight: float = 2.0,
    clean_margin_slack: float = 0.01,
    clean_task_scale: float = 0.25,
    trust_weight: float = 1.0,
    seed: int = 2026,
    stage_name: str = "raw_target",
    progress_interval: int = 100,
) -> dict:
    """Train query-side residuals against a fixed support-only map bank."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    support = [int(value) for value in support_query_indices]
    # Banks are constructed under inference_mode.  Cloning outside that mode
    # materializes ordinary frozen tensors that autograd may save while it
    # differentiates the query adapter.
    raw_reference_bank = raw_reference_bank.detach().to(device).clone()
    task_bank = task_bank.detach().to(device).clone()
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
            "clean": 0.0,
            "trust": 0.0,
            "rows": 0,
            "false_rows": 0,
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
            positives = record["positives"][local_rows].to(device)
            ignored = record["ignored"][local_rows].to(device)
            with torch.no_grad():
                raw_scores = raw @ raw_reference_bank.T
                raw_top_scores, raw_top_indices = torch.topk(
                    raw_scores, k=min(2, raw_reference_bank.shape[0]), dim=1
                )
                raw_top1 = raw_top_indices[:, 0]
                top1_positive = (
                    (positives == raw_top1[:, None]) & (positives >= 0)
                ).any(dim=1)
                top1_ignored = (
                    (ignored == raw_top1[:, None]) & (ignored >= 0)
                ).any(dim=1)
                false_top1 = ~top1_positive & ~top1_ignored
                harmful = torch.where(
                    false_top1, raw_top1, torch.full_like(raw_top1, -1)
                )[:, None]
                clean_anchor = torch.where(
                    top1_positive, raw_top1, torch.full_like(raw_top1, -1)
                )
                if raw_top_scores.shape[1] > 1:
                    clean_floor = (
                        raw_top_scores[:, 0]
                        - raw_top_scores[:, 1]
                        - float(clean_margin_slack)
                    )
                else:
                    clean_floor = torch.zeros_like(raw_top_scores[:, 0])
                clean_floor[~top1_positive] = torch.nan

            adapted, residual = adapter(raw, tokens)
            per_row = _multi_positive_list_loss(
                adapted,
                task_bank,
                positives,
                ignored,
                harmful,
                topk=int(topk),
                temperature=float(temperature),
                harmful_weight=float(collision_weight),
            )
            row_weights = torch.ones_like(per_row)
            row_weights[top1_positive] = float(clean_task_scale)
            list_loss = (per_row * row_weights).sum() / row_weights.sum().clamp_min(
                1e-8
            )
            clean_loss, clean_diagnostics = protected_clean_margin_loss(
                adapted,
                task_bank,
                clean_anchor,
                clean_floor,
            )
            trust_loss = residual.square().sum(dim=1).mean()
            loss = (
                list_loss
                + float(clean_weight) * clean_loss
                + float(trust_weight) * trust_loss
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite MCCD training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            row_count = int(local_rows.numel())
            totals["loss"] += float(loss.detach()) * row_count
            totals["list"] += float(list_loss.detach()) * row_count
            totals["clean"] += float(clean_loss.detach()) * row_count
            totals["trust"] += float(trust_loss.detach()) * row_count
            totals["rows"] += row_count
            totals["false_rows"] += int(false_top1.sum())
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
                        "event": "mccd_train",
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
            "false_top1_row_count": int(totals["false_rows"]),
            "clean_top1_row_count": int(totals["clean_rows"]),
            "clean_violation_count": int(totals["clean_violations"]),
            "mean_loss": float(totals["loss"] / denominator),
            "mean_list_loss": float(totals["list"] / denominator),
            "mean_clean_loss": float(totals["clean"] / denominator),
            "mean_trust_loss": float(totals["trust"] / denominator),
        }
        history.append(row)
        print({"event": "mccd_train_epoch_complete", **row}, flush=True)
    return {
        "stage": stage_name,
        "config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "topk": int(topk),
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
def evaluate_mccd_banks(
    *,
    state: dict,
    teacher: dict,
    query_cache: dict,
    gate_query_indices: Sequence[int],
    pose_query_indices: Sequence[int],
    banks: dict[str, torch.Tensor],
    adapter: MapConsistentContextAdapter,
    device: torch.device,
    topks: Sequence[int] = DEFAULT_TOPKS,
    deployment_row_limit: int = 0,
    ransac_reprojection_px: float = 12.0,
    seed: int = 2026,
    progress_interval: int = 25,
) -> tuple[dict, list[dict]]:
    """Evaluate raw and MCCD with identical support and online protocols."""
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
    map_banks = {
        "raw_superpoint": banks["raw_superpoint"].to(device),
        "mccd": banks["mccd"].to(device),
    }
    retrieval = {name: _empty_retrieval(topks) for name in map_banks}
    pose_rows = []
    residual_norms = []
    query_cosines = []
    for completed, query_index in enumerate(gate, start=1):
        name = names[query_index]
        record = teacher["records"][query_index]
        all_rows = torch.as_tensor(record["query_rows"]).long()
        selected_local = torch.arange(all_rows.numel())
        if deployment_row_limit > 0:
            selected_local = selected_local[
                all_rows < int(deployment_row_limit)
            ]
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
        adapted, residual = adapter(raw, tokens)
        residual_norms.extend(torch.linalg.norm(residual, dim=1).cpu().tolist())
        query_cosines.extend((raw * adapted).sum(dim=1).cpu().tolist())
        query_features = {"raw_superpoint": raw, "mccd": adapted}

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

        winners = {}
        for descriptor_name, bank in map_banks.items():
            local_ranked = torch.topk(
                query_features[descriptor_name] @ bank.T, k=max_k, dim=1
            ).indices
            ranked = anchor_indices[local_ranked].cpu()
            winners[descriptor_name] = ranked[:, 0]
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
                    "event": "mccd_gate",
                    "queries_complete": completed,
                    "query_count": len(gate),
                },
                flush=True,
            )
    norms = np.asarray(residual_norms, dtype=np.float64)
    cosines = np.asarray(query_cosines, dtype=np.float64)
    report = {
        name: summarize_retrieval(counts, topks)
        for name, counts in retrieval.items()
    }
    report["adapter_diagnostics"] = {
        "query_residual_norm_mean": float(norms.mean()) if norms.size else 0.0,
        "query_residual_norm_p90": (
            float(np.percentile(norms, 90)) if norms.size else 0.0
        ),
        "query_residual_norm_maximum": float(norms.max()) if norms.size else 0.0,
        "raw_to_mccd_cosine_mean": (
            float(cosines.mean()) if cosines.size else 1.0
        ),
        "raw_to_mccd_cosine_p10": (
            float(np.percentile(cosines, 10)) if cosines.size else 1.0
        ),
    }
    report["additive_counts"] = retrieval
    return report, pose_rows


def summarize_mccd_pose(pose_rows: list[dict]) -> dict:
    return {
        name: _pose_summary(pose_rows, name)
        for name in ("raw_superpoint", "mccd")
    }


def compare_mccd_protocols(retrieval: dict, pose: dict) -> dict:
    raw_r1 = float(retrieval["raw_superpoint"]["positive_recall_at_k"]["1"])
    mccd_r1 = float(retrieval["mccd"]["positive_recall_at_k"]["1"])
    raw_pose = pose["raw_superpoint"]
    mccd_pose = pose["mccd"]

    def relative(key: str) -> float:
        raw = float(raw_pose.get(key, 0.0))
        return float((float(mccd_pose.get(key, 0.0)) - raw) / max(raw, 1e-12))

    retrieval_gain_pp = 100.0 * (mccd_r1 - raw_r1)
    pose_non_regressive = all(
        float(mccd_pose.get(key, math.inf)) <= float(raw_pose.get(key, -math.inf))
        for key in ("mean_te_cm", "p90_te_cm", "cvar95_te_cm")
    )
    if retrieval_gain_pp >= 1.0 and pose_non_regressive:
        verdict = "advance_mccd_to_sentinel_scenes"
    elif retrieval_gain_pp > 0.0 or any(
        relative(key) < -0.02
        for key in ("mean_te_cm", "p90_te_cm", "cvar95_te_cm")
    ):
        verdict = "mixed_signal_refine_bounded_context_objective"
    else:
        verdict = "reject_dense_context_adapter"
    return {
        "mccd_minus_raw_top1_positive_recall_percentage_points": float(
            retrieval_gain_pp
        ),
        "mccd_relative_mean_te": relative("mean_te_cm"),
        "mccd_relative_p90_te": relative("p90_te_cm"),
        "mccd_relative_cvar95_te": relative("cvar95_te_cm"),
        "mccd_relative_mean_hypotheses": relative("mean_hypotheses"),
        "pose_non_regressive_on_mean_p90_cvar95": bool(pose_non_regressive),
        "routing_verdict": verdict,
    }
