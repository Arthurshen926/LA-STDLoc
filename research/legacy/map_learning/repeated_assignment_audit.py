"""Mapping-only audit for repeated-structure descriptor assignments.

The audit measures whether a legal 2D--3D positive is already present in the
deployed descriptor ranking.  It deliberately leaves topology, geometry, and
the PoseLib protocol unchanged so it can route later representation work
without consulting test images.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import numpy as np
import torch
import torch.nn.functional as F

from localization.pose_solver import pose_error, solve_absolute_pose


DEFAULT_TOPKS = (1, 2, 4, 8, 16, 32)
_QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _selected_csr_edges(
    record: dict,
    prefix: str,
    selected_local_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected-row CSR counts plus flattened local-row/anchor edges."""
    offsets = torch.as_tensor(record[f"{prefix}_offsets"]).long().reshape(-1)
    indices = torch.as_tensor(record[f"{prefix}_indices"]).long().reshape(-1)
    selected = torch.as_tensor(selected_local_rows).long().reshape(-1)
    row_count = offsets.numel() - 1
    if selected.numel() and (
        int(selected.min()) < 0 or int(selected.max()) >= row_count
    ):
        raise ValueError(f"{prefix} selected row is outside the CSR registry")
    counts = offsets[1:] - offsets[:-1]
    selected_counts = counts[selected]
    if indices.numel() == 0 or selected.numel() == 0:
        return (
            selected_counts,
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
    source_rows = torch.repeat_interleave(torch.arange(row_count), counts)
    selected_lookup = torch.full((row_count,), -1, dtype=torch.long)
    selected_lookup[selected] = torch.arange(selected.numel())
    local_rows = selected_lookup[source_rows]
    keep = local_rows >= 0
    return selected_counts, local_rows[keep], indices[keep]


def _rank_membership(
    ranked_indices: torch.Tensor,
    edge_rows: torch.Tensor,
    edge_anchors: torch.Tensor,
) -> torch.Tensor:
    """Return one positive-membership flag per row and retrieved rank."""
    ranked = torch.as_tensor(ranked_indices).long().cpu()
    output = torch.zeros(ranked.shape, dtype=torch.bool)
    if edge_rows.numel() == 0:
        return output
    rows = torch.as_tensor(edge_rows).long().cpu()
    anchors = torch.as_tensor(edge_anchors).long().cpu()
    for rank in range(ranked.shape[1]):
        hits = anchors == ranked[rows, rank]
        if bool(hits.any()):
            output[rows[hits], rank] = True
    return output


def _edge_maximum(
    scores: torch.Tensor,
    edge_rows: torch.Tensor,
    edge_anchors: torch.Tensor,
) -> torch.Tensor:
    output = scores.new_full((scores.shape[0],), -torch.inf)
    if edge_rows.numel():
        rows = edge_rows.to(scores.device)
        anchors = edge_anchors.to(scores.device)
        output.scatter_reduce_(
            0,
            rows,
            scores[rows, anchors],
            reduce="amax",
            include_self=True,
        )
    return output


def _quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    output: dict[str, float | int] = {"count": int(array.size)}
    if not array.size:
        output.update({f"q{int(q * 100):02d}": 0.0 for q in _QUANTILES})
        return output
    for quantile in _QUANTILES:
        output[f"q{int(quantile * 100):02d}"] = float(
            np.quantile(array, quantile)
        )
    output["mean"] = float(array.mean())
    return output


def _pose_summary(rows: list[dict], prefix: str) -> dict:
    if not rows:
        return {"query_count": 0}
    te = np.asarray([row[f"{prefix}_te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row[f"{prefix}_ae_deg"] for row in rows], dtype=np.float64)
    tail_count = max(int(math.ceil(0.05 * te.size)), 1)
    return {
        "query_count": int(te.size),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "cvar95_te_cm": float(np.sort(te)[-tail_count:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "p90_ae_deg": float(np.percentile(ae, 90)),
        "recall_5cm_5deg_percent": float(
            100.0 * np.mean((te <= 5.0) & (ae <= 5.0))
        ),
        "catastrophic_100cm_count": int(np.count_nonzero(te >= 100.0)),
        "failure_count": int(sum(row[f"{prefix}_failed"] for row in rows)),
        "mean_hypotheses": float(
            np.mean([row[f"{prefix}_hypotheses"] for row in rows])
        ),
    }


def _solve_assignments(
    assignments: torch.Tensor,
    *,
    keypoints: torch.Tensor,
    xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    gt_pose: torch.Tensor,
    reprojection_error_px: float,
    seed: int,
) -> dict:
    estimate = solve_absolute_pose(
        keypoints.numpy(),
        xyz[assignments.long()].numpy(),
        intrinsic.numpy(),
        reprojection_error_px=float(reprojection_error_px),
        confidence=0.99999,
        max_iterations=100000,
        min_iterations=1000,
        seed=int(seed),
    )
    ae_deg, te_cm = pose_error(estimate.pose_w2c, gt_pose.numpy())
    return {
        "te_cm": float(te_cm),
        "ae_deg": float(ae_deg),
        "failed": bool(int(estimate.inliers.size) < 4),
        "inliers": int(estimate.inliers.size),
        "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
    }


def _type_summary(
    *,
    name: str,
    mask: torch.Tensor,
    winner_count: torch.Tensor,
    correct_count: torch.Tensor,
    false_count: torch.Tensor,
) -> dict:
    mask = torch.as_tensor(mask).bool().reshape(-1)
    wins = int(winner_count[mask].sum())
    correct = int(correct_count[mask].sum())
    false = int(false_count[mask].sum())
    return {
        "name": name,
        "anchor_count": int(mask.sum()),
        "winner_count": wins,
        "winner_fraction": float(wins / max(int(winner_count.sum()), 1)),
        "correct_winner_count": correct,
        "false_winner_count": false,
        "precision_among_labeled_winners": float(correct / max(correct + false, 1)),
    }


@torch.inference_mode()
def audit_repeated_assignments(
    *,
    state: dict,
    metric: torch.nn.Module,
    teacher: dict,
    query_cache: dict,
    device: torch.device,
    topks: Sequence[int] = DEFAULT_TOPKS,
    query_indices: Sequence[int] | torch.Tensor | None = None,
    oracle_query_indices: Sequence[int] | torch.Tensor | None = None,
    deployment_row_limit: int = 0,
    ransac_reprojection_px: float = 12.0,
    seed: int = 2026,
    sharpness_by_name: Mapping[str, float] | None = None,
    top_attractor_count: int = 32,
    progress_interval: int = 25,
) -> dict:
    """Audit exact global descriptor ranking on mapping observations.

    Recall is reported both over every replayed detector row and over only rows
    with at least one legal positive.  The latter isolates descriptor ranking
    from evidence coverage.  Oracle PnP replaces a top-1 winner by the
    highest-ranked legal positive within top-K and leaves unrecoverable rows
    unchanged.
    """
    topks = tuple(sorted(set(int(value) for value in topks)))
    if not topks or topks[0] < 1:
        raise ValueError("top-K values must be positive")
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long().reshape(-1)
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    anchor_type = torch.as_tensor(state["anchor_type"]).long().reshape(-1).cpu()
    anchor_count = int(anchor_ids.numel())
    if not (
        xyz.shape[0] == bank.shape[0] == anchor_type.numel() == anchor_count
    ):
        raise ValueError("compact map rows do not align")
    if int(teacher["anchor_count"]) != anchor_count:
        raise ValueError("teacher and compact map anchor counts differ")
    max_k = min(max(topks), anchor_count)
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    if set(names) - set(cache):
        raise ValueError("query cache misses mapping queries from the teacher")
    selected_queries = (
        list(range(len(names)))
        if query_indices is None
        else [int(value) for value in torch.as_tensor(query_indices).tolist()]
    )
    if not selected_queries:
        raise ValueError("repeated-assignment audit query subset is empty")
    if min(selected_queries) < 0 or max(selected_queries) >= len(names):
        raise ValueError("repeated-assignment audit query index is out of range")
    oracle_queries = (
        set(selected_queries)
        if oracle_query_indices is None
        else {int(value) for value in torch.as_tensor(oracle_query_indices).tolist()}
    )
    if not oracle_queries.issubset(set(selected_queries)):
        raise ValueError("oracle queries must be a subset of retrieval audit queries")

    winner_count = torch.zeros(anchor_count, dtype=torch.long)
    correct_count = torch.zeros(anchor_count, dtype=torch.long)
    false_count = torch.zeros(anchor_count, dtype=torch.long)
    false_query_count = torch.zeros(anchor_count, dtype=torch.long)
    recall_hits = {k: 0 for k in topks}
    track_recall_hits = {k: 0 for k in topks}
    reserve_recall_hits = {k: 0 for k in topks}
    recoverable_false_hits = {k: 0 for k in topks}
    positive_rows_total = 0
    track_positive_rows_total = 0
    reserve_positive_rows_total = 0
    replayed_rows_total = 0
    ambiguous_winner_total = 0
    no_positive_winner_total = 0
    positive_scores: list[float] = []
    wrong_scores: list[float] = []
    margins: list[float] = []
    false_incoming_per_query: list[int] = []
    query_rows: list[dict] = []
    pose_rows: list[dict] = []

    for completed, query_index in enumerate(selected_queries, start=1):
        name = names[query_index]
        record = teacher["records"][query_index]
        cached = cache[name]
        all_rows = torch.as_tensor(record["query_rows"]).long()
        selected_local = torch.arange(all_rows.numel())
        if int(deployment_row_limit) > 0:
            selected_local = selected_local[
                all_rows < int(deployment_row_limit)
            ]
        rows = all_rows[selected_local]
        if rows.numel() == 0:
            continue
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptors)
        scores = adapted @ bank.T
        ranked_scores, ranked_indices = torch.topk(scores, k=max_k, dim=1)
        del ranked_scores
        ranked = ranked_indices.cpu()
        winners = ranked[:, 0]

        positive_counts, positive_edge_rows, positive_edge_anchors = (
            _selected_csr_edges(record, "positive", selected_local)
        )
        _, ambiguous_edge_rows, ambiguous_edge_anchors = _selected_csr_edges(
            record, "ambiguous", selected_local
        )
        positive_membership = _rank_membership(
            ranked, positive_edge_rows, positive_edge_anchors
        )
        ambiguous_membership = _rank_membership(
            ranked[:, :1], ambiguous_edge_rows, ambiguous_edge_anchors
        )[:, 0]
        has_positive = positive_counts > 0
        current_correct = positive_membership[:, 0]
        current_false = has_positive & ~current_correct & ~ambiguous_membership
        no_positive = ~has_positive

        replayed_rows_total += int(rows.numel())
        positive_rows_total += int(has_positive.sum())
        ambiguous_winner_total += int((~current_correct & ambiguous_membership).sum())
        no_positive_winner_total += int(no_positive.sum())
        winner_count += torch.bincount(winners, minlength=anchor_count)
        if bool(current_correct.any()):
            correct_count += torch.bincount(
                winners[current_correct], minlength=anchor_count
            )
        if bool(current_false.any()):
            false_winners = winners[current_false]
            false_count += torch.bincount(false_winners, minlength=anchor_count)
            false_query_count[torch.unique(false_winners)] += 1
            per_query = torch.bincount(false_winners, minlength=anchor_count)
            false_incoming_per_query.extend(
                per_query[per_query > 0].tolist()
            )

        positive_anchor_types = anchor_type[positive_edge_anchors]
        track_edges = positive_anchor_types != 0
        reserve_edges = positive_anchor_types == 0
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
        track_eligible = torch.zeros(rows.numel(), dtype=torch.bool)
        reserve_eligible = torch.zeros(rows.numel(), dtype=torch.bool)
        if bool(track_edges.any()):
            track_eligible[positive_edge_rows[track_edges]] = True
        if bool(reserve_edges.any()):
            reserve_eligible[positive_edge_rows[reserve_edges]] = True
        track_positive_rows_total += int(track_eligible.sum())
        reserve_positive_rows_total += int(reserve_eligible.sum())
        for requested_k in topks:
            effective_k = min(requested_k, max_k)
            recovered = positive_membership[:, :effective_k].any(dim=1)
            recall_hits[requested_k] += int(recovered.sum())
            recoverable_false_hits[requested_k] += int(
                (current_false & recovered).sum()
            )
            track_recall_hits[requested_k] += int(
                track_membership[:, :effective_k].any(dim=1).sum()
            )
            reserve_recall_hits[requested_k] += int(
                reserve_membership[:, :effective_k].any(dim=1).sum()
            )

        best_positive = _edge_maximum(
            scores, positive_edge_rows, positive_edge_anchors
        )
        competing = scores.clone()
        if positive_edge_rows.numel():
            competing[
                positive_edge_rows.to(device), positive_edge_anchors.to(device)
            ] = -torch.inf
        if ambiguous_edge_rows.numel():
            competing[
                ambiguous_edge_rows.to(device), ambiguous_edge_anchors.to(device)
            ] = -torch.inf
        best_wrong = competing.max(dim=1).values
        valid_margin = has_positive.to(device) & torch.isfinite(best_wrong)
        positive_scores.extend(best_positive[valid_margin].cpu().tolist())
        wrong_scores.extend(best_wrong[valid_margin].cpu().tolist())
        margins.extend(
            (best_positive[valid_margin] - best_wrong[valid_margin]).cpu().tolist()
        )
        del competing, scores, descriptors, adapted

        per_query_hits = {
            str(k): int(positive_membership[:, : min(k, max_k)].any(dim=1).sum())
            for k in topks
        }
        query_report = {
            "query_index": int(query_index),
            "image_name": name,
            "row_count": int(rows.numel()),
            "positive_row_count": int(has_positive.sum()),
            "top1_correct_count": int(current_correct.sum()),
            "top1_false_count": int(current_false.sum()),
            "top1_ambiguous_count": int(
                (~current_correct & ambiguous_membership).sum()
            ),
            "positive_recall_at_k": {
                key: float(value / max(int(has_positive.sum()), 1))
                for key, value in per_query_hits.items()
            },
        }
        if sharpness_by_name is not None and name in sharpness_by_name:
            query_report["laplacian_sharpness"] = float(sharpness_by_name[name])

        if query_index in oracle_queries:
            keypoints = (
                torch.as_tensor(cached["native_keypoints"]).float()[rows]
                + float(cached.get("pixel_center_offset", 0.5))
            ).cpu()
            intrinsic = torch.as_tensor(cached["native_K"]).float().cpu()
            gt_pose = torch.as_tensor(cached["pose_w2c"]).float().cpu()
            current = _solve_assignments(
                winners,
                keypoints=keypoints,
                xyz=xyz,
                intrinsic=intrinsic,
                gt_pose=gt_pose,
                reprojection_error_px=ransac_reprojection_px,
                seed=seed,
            )
            pose_report = {
                "query_index": int(query_index),
                "image_name": name,
                **{f"current_{key}": value for key, value in current.items()},
            }
            first_positive_rank = positive_membership.float().argmax(dim=1)
            row_index = torch.arange(rows.numel())
            for requested_k in topks:
                effective_k = min(requested_k, max_k)
                recoverable = positive_membership[:, :effective_k].any(dim=1)
                oracle = winners.clone()
                oracle[recoverable] = ranked[
                    row_index[recoverable], first_positive_rank[recoverable]
                ]
                result = _solve_assignments(
                    oracle,
                    keypoints=keypoints,
                    xyz=xyz,
                    intrinsic=intrinsic,
                    gt_pose=gt_pose,
                    reprojection_error_px=ransac_reprojection_px,
                    seed=seed,
                )
                pose_report[f"oracle_top{requested_k}_replaced_rows"] = int(
                    (oracle != winners).sum()
                )
                for key, value in result.items():
                    pose_report[f"oracle_top{requested_k}_{key}"] = value
            pose_rows.append(pose_report)
            query_report["current_te_cm"] = current["te_cm"]
            query_report["current_ae_deg"] = current["ae_deg"]
            query_report["current_hypotheses"] = current["hypotheses"]
        query_rows.append(query_report)

        if progress_interval > 0 and (
            completed % int(progress_interval) == 0
            or completed == len(selected_queries)
        ):
            print(
                {
                    "event": "repeated_assignment_audit",
                    "queries_complete": completed,
                    "query_count": len(selected_queries),
                },
                flush=True,
            )

    if replayed_rows_total == 0:
        raise ValueError("deployment row limit removed every mapping observation")

    if pose_rows:
        tail_threshold = float(
            np.quantile(
                np.asarray(
                    [row["current_te_cm"] for row in pose_rows], dtype=np.float64
                ),
                0.95,
            )
        )
        query_by_index = {row["query_index"]: row for row in query_rows}
        for row in pose_rows:
            is_tail = bool(float(row["current_te_cm"]) >= tail_threshold)
            row["current_tail_95"] = is_tail
            query_by_index[row["query_index"]]["current_tail_95"] = is_tail

    recall = {}
    recall_by_type = {"track_core": {}, "gaussian_reserve": {}}
    for requested_k in topks:
        recall[str(requested_k)] = {
            "hit_count": int(recall_hits[requested_k]),
            "all_rows_recall": float(
                recall_hits[requested_k] / max(replayed_rows_total, 1)
            ),
            "positive_eligible_recall": float(
                recall_hits[requested_k] / max(positive_rows_total, 1)
            ),
        }
        recall_by_type["track_core"][str(requested_k)] = {
            "hit_count": int(track_recall_hits[requested_k]),
            "eligible_row_count": int(track_positive_rows_total),
            "eligible_recall": float(
                track_recall_hits[requested_k]
                / max(track_positive_rows_total, 1)
            ),
        }
        recall_by_type["gaussian_reserve"][str(requested_k)] = {
            "hit_count": int(reserve_recall_hits[requested_k]),
            "eligible_row_count": int(reserve_positive_rows_total),
            "eligible_recall": float(
                reserve_recall_hits[requested_k]
                / max(reserve_positive_rows_total, 1)
            ),
        }

    attractor_rows = []
    false_order = torch.argsort(false_count, descending=True, stable=True)
    for anchor in false_order[: max(int(top_attractor_count), 0)].tolist():
        if int(false_count[anchor]) == 0:
            break
        attractor_rows.append(
            {
                "anchor_index": int(anchor),
                "anchor_id": int(anchor_ids[anchor]),
                "anchor_kind": (
                    "track_core"
                    if int(anchor_type[anchor]) != 0
                    else "gaussian_reserve"
                ),
                "false_incoming_row_count": int(false_count[anchor]),
                "affected_query_count": int(false_query_count[anchor]),
                "all_winner_row_count": int(winner_count[anchor]),
                "correct_winner_row_count": int(correct_count[anchor]),
            }
        )

    pose_summaries = {"current": _pose_summary(pose_rows, "current")}
    for requested_k in topks:
        pose_summaries[f"oracle_top{requested_k}"] = _pose_summary(
            pose_rows, f"oracle_top{requested_k}"
        )

    blur = {"status": "unavailable", "reason": "dataset images not supplied"}
    sharp_rows = [row for row in query_rows if "laplacian_sharpness" in row]
    if sharp_rows:
        values = np.asarray(
            [row["laplacian_sharpness"] for row in sharp_rows], dtype=np.float64
        )
        lower, upper = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        strata = {}
        for label, predicate in (
            ("blurred", values <= lower),
            ("middle", (values > lower) & (values < upper)),
            ("sharp", values >= upper),
        ):
            selected = [row for row, keep in zip(sharp_rows, predicate) if keep]
            correct = sum(row["top1_correct_count"] for row in selected)
            eligible = sum(row["positive_row_count"] for row in selected)
            false = sum(row["top1_false_count"] for row in selected)
            strata[label] = {
                "query_count": len(selected),
                "positive_row_count": int(eligible),
                "top1_positive_recall": float(correct / max(eligible, 1)),
                "false_winner_fraction": float(false / max(eligible, 1)),
            }
        blur = {
            "status": "available",
            "metric": "variance_of_grayscale_laplacian",
            "interpretation": "lower_is_blurrier_or_lower_texture",
            "thresholds": {
                "lower_tertile": float(lower),
                "upper_tertile": float(upper),
            },
            "strata": strata,
        }

    return {
        "schema": "lafgs_repeated_assignment_audit",
        "version": 1,
        "uses_test_queries": False,
        "protocol": {
            "matching": "shared_metric_then_exact_global_cosine_topk",
            "oracle": "replace_top1_with_highest_ranked_legal_positive_within_k",
            "unrecoverable_or_unlabeled_rows": "retain_deployed_top1",
            "positive_recall_denominators": "all_rows_and_positive_eligible_rows",
        },
        "query_count": len(query_rows),
        "oracle_query_count": len(pose_rows),
        "deployment_row_limit": int(deployment_row_limit),
        "anchor_count": anchor_count,
        "row_counts": {
            "replayed": int(replayed_rows_total),
            "positive_eligible": int(positive_rows_total),
            "track_positive_eligible": int(track_positive_rows_total),
            "gaussian_reserve_positive_eligible": int(reserve_positive_rows_total),
            "top1_ambiguous": int(ambiguous_winner_total),
            "no_positive": int(no_positive_winner_total),
        },
        "positive_recall_at_k": recall,
        "false_top1_recoverable_at_k": {
            str(k): {
                "hit_count": int(recoverable_false_hits[k]),
                "false_top1_count": int(false_count.sum()),
                "fraction": float(
                    recoverable_false_hits[k] / max(int(false_count.sum()), 1)
                ),
            }
            for k in topks
        },
        "positive_recall_at_k_by_anchor_kind": recall_by_type,
        "score_distributions": {
            "best_positive": _quantiles(positive_scores),
            "best_wrong_excluding_ambiguous": _quantiles(wrong_scores),
            "positive_minus_wrong_margin": _quantiles(margins),
        },
        "winner_breakdown": {
            "track_core": _type_summary(
                name="track_core",
                mask=anchor_type != 0,
                winner_count=winner_count,
                correct_count=correct_count,
                false_count=false_count,
            ),
            "gaussian_reserve": _type_summary(
                name="gaussian_reserve",
                mask=anchor_type == 0,
                winner_count=winner_count,
                correct_count=correct_count,
                false_count=false_count,
            ),
        },
        "false_attractors": {
            "anchor_count": int((false_count > 0).sum()),
            "incoming_rows_per_attractor": _quantiles(
                false_count[false_count > 0].numpy()
            ),
            "incoming_rows_per_attractor_per_query": _quantiles(
                false_incoming_per_query
            ),
            "top": attractor_rows,
        },
        "blur_conditioning": blur,
        "oracle_pose_summaries": pose_summaries,
        "queries": query_rows,
        "oracle_queries": pose_rows,
    }
