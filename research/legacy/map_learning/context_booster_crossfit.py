"""Mapping-only cross-fit diagnostic for contextual local descriptors."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F

from map_learning.context_booster import (
    FeatureBooster,
    normalize_keypoint_properties,
)
from map_learning.repeated_assignment_audit import (
    _pose_summary,
    _rank_membership,
    _selected_csr_edges,
    _solve_assignments,
)


DEFAULT_TOPKS = (1, 4, 8, 16, 32)


def accumulate_view_descriptors(
    accumulator: torch.Tensor,
    view_counts: torch.Tensor,
    anchor_indices: torch.Tensor,
    descriptors: torch.Tensor,
) -> int:
    """Fuse one image with equal weight per anchor-observing view.

    Multiple legal detector rows for an anchor are averaged inside the image,
    normalized, and then contribute one unit view.  This prevents feature-rich
    views from dominating the fused map descriptor.
    """
    anchors = torch.as_tensor(anchor_indices, device=accumulator.device).long()
    features = torch.as_tensor(descriptors, device=accumulator.device).float()
    if anchors.ndim != 1 or features.ndim != 2:
        raise ValueError("anchors and descriptors must have shapes [E] and [E, D]")
    if anchors.numel() != features.shape[0]:
        raise ValueError("anchor and descriptor edge counts must agree")
    if features.shape[1] != accumulator.shape[1]:
        raise ValueError("descriptor dimensions do not agree")
    if anchors.numel() == 0:
        return 0
    unique, inverse = torch.unique(anchors, sorted=True, return_inverse=True)
    per_view = features.new_zeros((unique.numel(), features.shape[1]))
    per_view.index_add_(0, inverse, features)
    per_view = F.normalize(per_view, dim=1)
    accumulator.index_add_(0, unique, per_view)
    view_counts[unique] += 1
    return int(unique.numel())


def _boost_query(
    model: FeatureBooster,
    cached: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = F.normalize(
        torch.as_tensor(cached["native_descriptors"]).float().to(device), dim=1
    )
    properties = normalize_keypoint_properties(
        cached["native_keypoints"],
        cached["native_scores"],
        cached["native_input_hw"],
    ).to(device)
    return raw, model(raw, properties)


@torch.inference_mode()
def build_observation_fused_banks(
    *,
    teacher: dict,
    query_cache: dict,
    support_query_indices: Sequence[int],
    model: FeatureBooster,
    device: torch.device,
    minimum_support_views: int = 2,
    progress_interval: int = 25,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build raw and boosted banks from exactly the same support observations."""
    if minimum_support_views < 1:
        raise ValueError("minimum support views must be positive")
    anchor_count = int(teacher["anchor_count"])
    names = list(teacher["query_names"])
    cache = query_cache.get("queries", query_cache)
    support = [int(value) for value in support_query_indices]
    if not support:
        raise ValueError("support partition is empty")
    descriptor_dim = int(model.config["output_dim"])
    raw_sum = torch.zeros((anchor_count, descriptor_dim), device=device)
    boost_sum = torch.zeros_like(raw_sum)
    view_counts = torch.zeros(anchor_count, dtype=torch.long, device=device)
    positive_edge_count = 0
    anchor_view_count = 0

    for completed, query_index in enumerate(support, start=1):
        name = names[query_index]
        record = teacher["records"][query_index]
        cached = cache[name]
        rows = torch.as_tensor(record["query_rows"]).long()
        selected = torch.arange(rows.numel())
        _, edge_rows, edge_anchors = _selected_csr_edges(
            record, "positive", selected
        )
        if edge_rows.numel():
            raw, boosted = _boost_query(model, cached, device)
            native_edge_rows = rows[edge_rows].to(device)
            edge_anchors_device = edge_anchors.to(device)
            observed = accumulate_view_descriptors(
                raw_sum,
                view_counts,
                edge_anchors_device,
                raw[native_edge_rows],
            )
            boost_counts = torch.zeros_like(view_counts)
            accumulate_view_descriptors(
                boost_sum,
                boost_counts,
                edge_anchors_device,
                boosted[native_edge_rows],
            )
            if int(boost_counts.sum()) != observed:
                raise AssertionError("raw and boosted support observations diverged")
            positive_edge_count += int(edge_rows.numel())
            anchor_view_count += observed
        if progress_interval > 0 and (
            completed % int(progress_interval) == 0 or completed == len(support)
        ):
            print(
                {
                    "event": "context_booster_support",
                    "queries_complete": completed,
                    "query_count": len(support),
                },
                flush=True,
            )

    supported = view_counts >= int(minimum_support_views)
    if not bool(supported.any()):
        raise ValueError("support partition did not observe any deployable anchors")
    anchor_indices = torch.nonzero(supported, as_tuple=False).reshape(-1)
    banks = {
        "anchor_indices": anchor_indices,
        "raw_superpoint": F.normalize(raw_sum[anchor_indices], dim=1),
        "superpoint_boost_f": F.normalize(boost_sum[anchor_indices], dim=1),
        "view_counts": view_counts[anchor_indices],
    }
    return banks, {
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


def _empty_retrieval(topks: Sequence[int]) -> dict:
    return {
        "replayed_rows": 0,
        "positive_eligible_rows": 0,
        "track_positive_eligible_rows": 0,
        "reserve_positive_eligible_rows": 0,
        "top1_correct": 0,
        "top1_false": 0,
        "top1_ambiguous": 0,
        "recall_hits": {str(k): 0 for k in topks},
        "track_recall_hits": {str(k): 0 for k in topks},
        "reserve_recall_hits": {str(k): 0 for k in topks},
        "false_recoverable_hits": {str(k): 0 for k in topks},
    }


def summarize_retrieval(counts: dict, topks: Sequence[int]) -> dict:
    """Turn additive retrieval counts into JSON-facing rates."""
    positive = int(counts["positive_eligible_rows"])
    track = int(counts["track_positive_eligible_rows"])
    reserve = int(counts["reserve_positive_eligible_rows"])
    false = int(counts["top1_false"])
    return {
        "row_counts": {
            key: int(counts[key])
            for key in (
                "replayed_rows",
                "positive_eligible_rows",
                "track_positive_eligible_rows",
                "reserve_positive_eligible_rows",
                "top1_correct",
                "top1_false",
                "top1_ambiguous",
            )
        },
        "positive_recall_at_k": {
            str(k): float(counts["recall_hits"][str(k)] / max(positive, 1))
            for k in topks
        },
        "positive_recall_at_k_by_anchor_kind": {
            "track_core": {
                str(k): float(
                    counts["track_recall_hits"][str(k)] / max(track, 1)
                )
                for k in topks
            },
            "gaussian_reserve": {
                str(k): float(
                    counts["reserve_recall_hits"][str(k)] / max(reserve, 1)
                )
                for k in topks
            },
        },
        "false_top1_recoverable_at_k": {
            str(k): float(
                counts["false_recoverable_hits"][str(k)] / max(false, 1)
            )
            for k in topks
        },
    }


def _update_descriptor_counts(
    counts: dict,
    *,
    ranked: torch.Tensor,
    positive_edge_rows: torch.Tensor,
    positive_edge_anchors: torch.Tensor,
    ambiguous_edge_rows: torch.Tensor,
    ambiguous_edge_anchors: torch.Tensor,
    anchor_type: torch.Tensor,
    topks: Sequence[int],
) -> None:
    positive_membership = _rank_membership(
        ranked, positive_edge_rows, positive_edge_anchors
    )
    ambiguous = _rank_membership(
        ranked[:, :1], ambiguous_edge_rows, ambiguous_edge_anchors
    )[:, 0]
    row_count = ranked.shape[0]
    has_positive = torch.zeros(row_count, dtype=torch.bool)
    if positive_edge_rows.numel():
        has_positive[positive_edge_rows] = True
    current_correct = positive_membership[:, 0]
    current_false = has_positive & ~current_correct & ~ambiguous

    positive_types = anchor_type[positive_edge_anchors]
    track_edges = positive_types != 0
    reserve_edges = positive_types == 0
    track_membership = _rank_membership(
        ranked,
        positive_edge_rows[track_edges],
        positive_edge_anchors[track_edges],
    )
    reserve_membership = _rank_membership(
        ranked,
        positive_edge_rows[reserve_edges],
        positive_edge_anchors[reserve_edges],
    )
    track_eligible = torch.zeros(row_count, dtype=torch.bool)
    reserve_eligible = torch.zeros(row_count, dtype=torch.bool)
    if bool(track_edges.any()):
        track_eligible[positive_edge_rows[track_edges]] = True
    if bool(reserve_edges.any()):
        reserve_eligible[positive_edge_rows[reserve_edges]] = True

    counts["replayed_rows"] += int(row_count)
    counts["positive_eligible_rows"] += int(has_positive.sum())
    counts["track_positive_eligible_rows"] += int(track_eligible.sum())
    counts["reserve_positive_eligible_rows"] += int(reserve_eligible.sum())
    counts["top1_correct"] += int(current_correct.sum())
    counts["top1_false"] += int(current_false.sum())
    counts["top1_ambiguous"] += int((~current_correct & ambiguous).sum())
    for requested_k in topks:
        effective_k = min(int(requested_k), ranked.shape[1])
        recovered = positive_membership[:, :effective_k].any(dim=1)
        counts["recall_hits"][str(requested_k)] += int(recovered.sum())
        counts["track_recall_hits"][str(requested_k)] += int(
            track_membership[:, :effective_k].any(dim=1).sum()
        )
        counts["reserve_recall_hits"][str(requested_k)] += int(
            reserve_membership[:, :effective_k].any(dim=1).sum()
        )
        counts["false_recoverable_hits"][str(requested_k)] += int(
            (current_false & recovered).sum()
        )


@torch.inference_mode()
def evaluate_context_banks(
    *,
    state: dict,
    teacher: dict,
    query_cache: dict,
    gate_query_indices: Sequence[int],
    pose_query_indices: Sequence[int],
    banks: dict[str, torch.Tensor],
    model: FeatureBooster,
    device: torch.device,
    topks: Sequence[int] = DEFAULT_TOPKS,
    deployment_row_limit: int = 0,
    ransac_reprojection_px: float = 12.0,
    seed: int = 2026,
    progress_interval: int = 25,
) -> tuple[dict, list[dict]]:
    """Evaluate raw-control and Boost-F banks under one global-top1 protocol."""
    topks = tuple(sorted(set(int(value) for value in topks)))
    if not topks or topks[0] < 1:
        raise ValueError("top-K values must be positive")
    gate = [int(value) for value in gate_query_indices]
    if not gate:
        raise ValueError("gate partition is empty")
    pose_set = {int(value) for value in pose_query_indices}
    if not pose_set.issubset(set(gate)):
        raise ValueError("pose queries must be a subset of gate queries")

    names = list(teacher["query_names"])
    cache = query_cache.get("queries", query_cache)
    anchor_indices = banks["anchor_indices"].long().to(device)
    supported = torch.zeros(int(teacher["anchor_count"]), dtype=torch.bool)
    supported[anchor_indices.cpu()] = True
    anchor_type = torch.as_tensor(state["anchor_type"]).long().cpu()
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    max_k = min(max(topks), anchor_indices.numel())
    descriptors = {
        "raw_superpoint": banks["raw_superpoint"].to(device),
        "superpoint_boost_f": banks["superpoint_boost_f"].to(device),
    }
    retrieval = {name: _empty_retrieval(topks) for name in descriptors}
    pose_rows: list[dict] = []

    for completed, query_index in enumerate(gate, start=1):
        name = names[query_index]
        record = teacher["records"][query_index]
        cached = cache[name]
        all_rows = torch.as_tensor(record["query_rows"]).long()
        selected_local = torch.arange(all_rows.numel())
        if deployment_row_limit > 0:
            selected_local = selected_local[
                all_rows < int(deployment_row_limit)
            ]
        rows = all_rows[selected_local]
        if not rows.numel():
            continue
        raw, boosted = _boost_query(model, cached, device)
        query_features = {
            "raw_superpoint": raw[rows.to(device)],
            "superpoint_boost_f": boosted[rows.to(device)],
        }

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

        winners: dict[str, torch.Tensor] = {}
        for descriptor_name, bank in descriptors.items():
            scores = query_features[descriptor_name] @ bank.T
            local_ranked = torch.topk(scores, k=max_k, dim=1).indices
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
                    "event": "context_booster_gate",
                    "queries_complete": completed,
                    "query_count": len(gate),
                },
                flush=True,
            )

    report = {
        name: summarize_retrieval(counts, topks)
        for name, counts in retrieval.items()
    }
    report["additive_counts"] = retrieval
    return report, pose_rows


def combine_additive_counts(folds: Sequence[dict], topks: Sequence[int]) -> dict:
    """Combine per-direction additive retrieval statistics."""
    combined = _empty_retrieval(topks)
    for counts in folds:
        for key in (
            "replayed_rows",
            "positive_eligible_rows",
            "track_positive_eligible_rows",
            "reserve_positive_eligible_rows",
            "top1_correct",
            "top1_false",
            "top1_ambiguous",
        ):
            combined[key] += int(counts[key])
        for group in (
            "recall_hits",
            "track_recall_hits",
            "reserve_recall_hits",
            "false_recoverable_hits",
        ):
            for requested_k in topks:
                combined[group][str(requested_k)] += int(
                    counts[group][str(requested_k)]
                )
    return combined


def summarize_pose_rows(pose_rows: list[dict]) -> dict:
    """Summarize both descriptor protocols over pooled cross-fit pose rows."""
    return {
        name: _pose_summary(pose_rows, name)
        for name in ("raw_superpoint", "superpoint_boost_f")
    }


def compare_protocols(retrieval: dict, pose: dict) -> dict:
    """Report signed Boost-F minus raw-control deltas and a routing verdict."""
    raw_r1 = float(retrieval["raw_superpoint"]["positive_recall_at_k"]["1"])
    boost_r1 = float(
        retrieval["superpoint_boost_f"]["positive_recall_at_k"]["1"]
    )
    raw_pose = pose["raw_superpoint"]
    boost_pose = pose["superpoint_boost_f"]

    def relative(key: str) -> float:
        raw = float(raw_pose.get(key, 0.0))
        return float((float(boost_pose.get(key, 0.0)) - raw) / max(raw, 1e-12))

    retrieval_gain_pp = 100.0 * (boost_r1 - raw_r1)
    pose_non_regressive = all(
        float(boost_pose.get(key, math.inf)) <= float(raw_pose.get(key, -math.inf))
        for key in ("mean_te_cm", "p90_te_cm", "cvar95_te_cm")
    )
    if retrieval_gain_pp >= 1.0 and pose_non_regressive:
        verdict = "advance_context_distillation"
    elif retrieval_gain_pp > 0.0 or any(
        relative(key) < -0.02
        for key in ("mean_te_cm", "p90_te_cm", "cvar95_te_cm")
    ):
        verdict = "mixed_signal_run_scene_conditioned_ablation"
    else:
        verdict = "reject_official_boost_f_direction"
    return {
        "boost_minus_raw_top1_positive_recall_percentage_points": float(
            retrieval_gain_pp
        ),
        "boost_relative_mean_te": relative("mean_te_cm"),
        "boost_relative_p90_te": relative("p90_te_cm"),
        "boost_relative_cvar95_te": relative("cvar95_te_cm"),
        "boost_relative_mean_hypotheses": relative("mean_hypotheses"),
        "pose_non_regressive_on_mean_p90_cvar95": bool(pose_non_regressive),
        "routing_verdict": verdict,
    }
