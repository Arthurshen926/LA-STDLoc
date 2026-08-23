"""Formal mapping-only query-local feedback for the V6 closed loop."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from evidence.observation_provider import ObservationProvider
from evidence.projective_loo import LeaveOneQueryOutProjectiveMap
from localization.matcher import global_cosine_topk
from localization.pose_solver import solve_absolute_pose
from map_learning.self_localization_feedback import build_self_localization_feedback
from topology.pose_information import (
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
)


def _maximum_matching(edges: list[list[int]]) -> tuple[int, list[tuple[int, int]]]:
    row_for_anchor: dict[int, int] = {}

    def augment(row: int, seen: set[int]) -> bool:
        for anchor in edges[row]:
            if anchor in seen:
                continue
            seen.add(anchor)
            previous = row_for_anchor.get(anchor)
            if previous is None or augment(previous, seen):
                row_for_anchor[anchor] = row
                return True
        return False

    for row in range(len(edges)):
        augment(row, set())
    pairs = [(row, anchor) for anchor, row in sorted(row_for_anchor.items())]
    return len(pairs), pairs


def _project(
    xyz: torch.Tensor, K: torch.Tensor, pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:].clamp_min(1e-8), camera[:, 2]


def _layer_edges(
    keypoints: torch.Tensor,
    projected: torch.Tensor,
    visible_rows: torch.Tensor,
    radius_px: float,
) -> list[list[int]]:
    result = [[] for _ in range(int(keypoints.shape[0]))]
    if keypoints.numel() == 0 or visible_rows.numel() == 0:
        return result
    for start in range(0, int(visible_rows.numel()), 2048):
        anchors = visible_rows[start : start + 2048]
        distance = torch.cdist(keypoints.float(), projected[anchors].float())
        query, local = torch.nonzero(
            distance <= float(radius_px), as_tuple=True
        )
        for row, candidate in zip(query.tolist(), anchors[local].tolist()):
            result[row].append(int(candidate))
    for values in result:
        values.sort()
    return result


def _positive_score_statistics(
    dense_scores: torch.Tensor,
    positives_by_row: list[list[int]],
    *,
    chunk_size: int = 64,
) -> dict[int, tuple[float, float, int, int]]:
    """Return exact stable-rank statistics with one bank scan per query row.

    Stable descending argsort breaks equal-score ties by the original Anchor
    row. Counting larger scores plus equal scores at smaller rows is exactly
    equivalent. Rows are processed in bounded batches so rank and best-wrong
    share one vectorized scan rather than separately scanning the full bank
    for every positive keypoint.
    """

    if dense_scores.ndim != 2 or len(positives_by_row) != dense_scores.shape[0]:
        raise ValueError("positive edges and dense score rows differ")
    if int(chunk_size) < 1:
        raise ValueError("positive statistics chunk size must be positive")
    positive_query_rows = [
        row for row, positives in enumerate(positives_by_row) if positives
    ]
    result: dict[int, tuple[float, float, int, int]] = {}
    anchor_count = int(dense_scores.shape[1])
    anchor_rows = torch.arange(anchor_count, device=dense_scores.device)
    for start in range(0, len(positive_query_rows), int(chunk_size)):
        rows_list = positive_query_rows[start : start + int(chunk_size)]
        rows = torch.tensor(rows_list, dtype=torch.long, device=dense_scores.device)
        maximum_degree = max(len(positives_by_row[row]) for row in rows_list)
        padded = torch.full(
            (len(rows_list), maximum_degree),
            -1,
            dtype=torch.long,
            device=dense_scores.device,
        )
        for local, row in enumerate(rows_list):
            values = positives_by_row[row]
            padded[local, : len(values)] = torch.tensor(
                values, dtype=torch.long, device=dense_scores.device
            )
        valid = padded >= 0
        gathered = dense_scores[rows[:, None], padded.clamp_min(0)]
        gathered = gathered.masked_fill(~valid, -torch.inf)
        best_positive = gathered.max(1).values
        best_anchor = torch.where(
            valid & (gathered == best_positive[:, None]),
            padded,
            anchor_count,
        ).min(1).values
        scores = dense_scores[rows]
        ranks = 1 + (
            (scores > best_positive[:, None])
            | (
                (scores == best_positive[:, None])
                & (anchor_rows[None] < best_anchor[:, None])
            )
        ).sum(1)
        wrong = scores.clone()
        local_rows = torch.arange(len(rows_list), device=dense_scores.device)
        local_rows = local_rows[:, None].expand_as(padded)
        wrong[local_rows[valid], padded[valid]] = -torch.inf
        best_wrong = wrong.max(1).values
        packed = torch.stack(
            [best_positive, best_wrong, ranks, best_anchor], dim=1
        ).cpu()
        for row, values in zip(rows_list, packed.tolist()):
            result[row] = (
                float(values[0]),
                float(values[1]),
                int(values[2]),
                int(values[3]),
            )
    return result


def _summary(rows: list[dict]) -> dict:
    te = np.asarray([row["te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row["ae_deg"] for row in rows], dtype=np.float64)
    tail = max(int(math.ceil(0.05 * len(rows))), 1)
    correspondence_count = sum(int(row["correspondences"]) for row in rows)
    correct_count = sum(int(row["correct_winners"]) for row in rows)
    inlier_count = sum(int(row["inliers"]) for row in rows)
    clean_inlier_count = sum(int(row["clean_inliers"]) for row in rows)
    positive_rows = sum(int(row["positive_rows"]) for row in rows)
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "cvar95_te_cm": float(np.sort(te)[-tail:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "recall_5cm_5deg_percent": float(np.mean((te < 5.0) & (ae < 5.0)) * 100.0),
        "catastrophic_100cm_count": int((te >= 100.0).sum()),
        "raw_gt_precision_percent": 100.0 * correct_count / max(correspondence_count, 1),
        "inlier_gt_precision_percent": 100.0 * clean_inlier_count / max(inlier_count, 1),
        "correct_anchor_recall_at_1_percent": 100.0
        * sum(int(row["correct_anchor_rank_le_1"]) for row in rows)
        / max(positive_rows, 1),
        "correct_anchor_recall_at_16_percent": 100.0
        * sum(int(row["correct_anchor_rank_le_16"]) for row in rows)
        / max(positive_rows, 1),
        "mean_poselib_iterations": float(
            np.mean([row["poselib_iterations"] for row in rows])
        ),
        "online_latency_ms": float(
            np.mean([row["online_latency_ms"] for row in rows])
        ),
        "loo_feedback_latency_ms": float(
            np.mean([row["loo_feedback_latency_ms"] for row in rows])
        ),
    }


@torch.inference_mode()
def evaluate_query_local_feedback(
    *,
    state: dict,
    observations: ObservationProvider,
    source_map_sha256: str,
    query_cache_sha256: str,
    device: torch.device,
    positive_radius_px: float,
    alpha_minimum: float,
    required_rank: int,
    ransac_reprojection_px: float,
    seed: int,
) -> dict:
    """One global Top-1 and one standard PoseLib solve per mapping query."""

    if state.get("provenance", {}).get("uses_test_queries") is not False:
        raise ValueError("V6 feedback requires a test-free map")
    replay = LeaveOneQueryOutProjectiveMap(state, observations)
    base_xyz = torch.as_tensor(state["anchor_xyz"]).float()
    base_bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    query_rows = []
    feedback_records = []
    for query_index, name in enumerate(observations.names):
        view = observations.build_view(query_index)
        loo_started = time.perf_counter()
        update = replay.query_update(query_index)
        loo_latency_ms = (time.perf_counter() - loo_started) * 1000.0
        online_started = time.perf_counter()
        xyz = base_xyz.clone()
        bank = base_bank.clone()
        active = torch.ones(xyz.shape[0], dtype=torch.bool)
        affected = update["anchor_rows"]
        if affected.numel():
            xyz[affected] = update["anchor_xyz"]
            bank[affected] = F.normalize(update["anchor_features"], dim=1)
            active[affected] = update["valid"]
        projected, depth = _project(xyz, view.intrinsics.float(), view.pose_w2c.float())
        height, width = view.image_hw
        visible = (
            active
            & torch.isfinite(projected).all(1)
            & torch.isfinite(depth)
            & (depth > 0)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
        )
        if view.alpha is not None:
            x = projected[:, 0].round().long().clamp(0, width - 1)
            y = projected[:, 1].round().long().clamp(0, height - 1)
            visible &= torch.isfinite(view.alpha[y, x]) & (
                view.alpha[y, x] >= float(alpha_minimum)
            )
        visible_rows = torch.nonzero(visible, as_tuple=False).reshape(-1)
        keypoints = view.physical_keypoints.float()
        positive_edges = _layer_edges(
            keypoints, projected, visible_rows, float(positive_radius_px)
        )
        detectable_rank, detectable_pairs = _maximum_matching(positive_edges)
        query_descriptor = F.normalize(view.descriptors.float(), dim=1).to(device)
        active_rows = torch.nonzero(active, as_tuple=False).reshape(-1)
        if active_rows.numel() < 4:
            raise ValueError(f"query {name} has fewer than four LOO-valid Anchors")
        matches = global_cosine_topk(
            query_descriptor,
            bank[active_rows].to(device),
            topk=1,
            anchor_descriptors_normalized=True,
        )
        winners = active_rows[matches.anchor_indices[:, 0].cpu()]
        correct = torch.tensor(
            [int(winner) in positive_edges[row] for row, winner in enumerate(winners)],
            dtype=torch.bool,
        )
        matching_rank, matching_pairs = _maximum_matching(
            [[int(winners[row])] if bool(correct[row]) else [] for row in range(len(winners))]
        )
        estimate = solve_absolute_pose(
            keypoints.numpy(),
            xyz[winners].numpy(),
            view.intrinsics.float().numpy(),
            reprojection_error_px=float(ransac_reprojection_px),
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            seed=int(seed),
        )
        inliers = torch.as_tensor(estimate.inliers).long().reshape(-1)
        pose = torch.as_tensor(estimate.pose_w2c).float()
        rotation = pose[:3, :3] @ view.pose_w2c[:3, :3].float().T
        cosine = ((torch.trace(rotation) - 1.0) / 2.0).clamp(-1, 1)
        ae_deg = float(torch.rad2deg(torch.acos(cosine)))
        center_est = -(pose[:3, :3].T @ pose[:3, 3])
        gt = view.pose_w2c.float()
        center_gt = -(gt[:3, :3].T @ gt[:3, 3])
        te_cm = float(torch.linalg.norm(center_est - center_gt) * 100.0)
        online_latency_ms = (time.perf_counter() - online_started) * 1000.0
        clean_rows = inliers[correct[inliers]] if inliers.numel() else torch.empty(0, dtype=torch.long)
        clean_ids = winners[clean_rows]
        harmful_ids = winners[inliers][~correct[inliers]] if inliers.numel() else torch.empty(0, dtype=torch.long)
        dense_scores = query_descriptor @ bank.to(device).T
        dense_scores[:, ~active.to(device)] = -torch.inf
        best_positive = []
        best_wrong = []
        correct_anchor_ranks = []
        confusion_pairs = []
        positive_statistics = _positive_score_statistics(
            dense_scores, positive_edges
        )
        for row, positives in enumerate(positive_edges):
            if positives:
                positive_score, wrong_score, rank, best = positive_statistics[row]
                best_positive.append(positive_score)
                best_wrong.append(wrong_score)
                correct_anchor_ranks.append(rank)
                if not bool(correct[row]):
                    confusion_pairs.append((int(winners[row]), best))
        if clean_ids.numel():
            clean_jacobian = pose_jacobian_analytic(
                xyz[clean_ids].double(),
                view.intrinsics.double(),
                view.pose_w2c.double(),
            )
            clean_jacobian = task_scaled_pose_jacobian(
                clean_jacobian,
                translation_scale=0.05,
                rotation_scale=math.radians(5.0),
            )
            clean_information = fisher_contributions(clean_jacobian)
            total_information = clean_information.sum(0)
            information_rank = int(torch.linalg.matrix_rank(total_information))
            information_logdet = float(
                torch.linalg.slogdet(
                    total_information + torch.eye(6, dtype=torch.float64) * 1e-9
                )[1]
            )
        else:
            clean_information = torch.empty((0, 6, 6), dtype=torch.float64)
            information_rank = 0
            information_logdet = float("-inf")
        feedback_records.append(
            {
                "image_name": name,
                "visible_rank": int(visible_rows.numel()),
                "detectable_rank": int(detectable_rank),
                "correct_anchor_rank": min(correct_anchor_ranks, default=0),
                "matching_rank": int(matching_rank),
                "winner_anchor": int(winners[0]) if winners.numel() else -1,
                "best_positive_score": max(best_positive, default=-1.0),
                "best_wrong_score": max(best_wrong, default=-1.0),
                "clean_inlier_anchor_ids": torch.unique(clean_ids),
                "harmful_inlier_anchor_ids": torch.unique(harmful_ids),
                "query_rows": torch.arange(keypoints.shape[0], dtype=torch.long),
                "winner_anchor_ids": winners,
                "winner_scores": matches.scores[:, 0].cpu(),
                "inlier_query_rows": inliers,
                "inlier_clean_mask": correct[inliers] if inliers.numel() else torch.empty(0, dtype=torch.bool),
                "visible_anchor_ids": visible_rows,
                "detectable_pairs": torch.tensor(detectable_pairs, dtype=torch.long).reshape(-1, 2),
                "matching_pairs": torch.tensor(matching_pairs, dtype=torch.long).reshape(-1, 2),
                "confusion_pairs": torch.tensor(
                    confusion_pairs, dtype=torch.long
                ).reshape(-1, 2),
                "dependency_group_ids": torch.unique(
                    torch.as_tensor(state["dependency_group_ids"])[winners]
                ),
                "clean_inlier_pose_anchor_ids": clean_ids,
                "clean_inlier_pose_information": clean_information,
                "pose_information_rank": information_rank,
                "pose_information_logdet": information_logdet,
                "pose_information_contribution": information_logdet,
                "pose_information_sufficient": information_rank >= 6,
                "pose_success": bool(te_cm < 5.0 and ae_deg < 5.0),
                "query_geometry_loo": True,
            }
        )
        query_rows.append(
            {
                "query_index": query_index,
                "image_name": name,
                "te_cm": te_cm,
                "ae_deg": ae_deg,
                "inliers": int(inliers.numel()),
                "clean_inliers": int(correct[inliers].sum()) if inliers.numel() else 0,
                "correct_winners": int(correct.sum()),
                "positive_rows": int(len(correct_anchor_ranks)),
                "correct_anchor_rank_le_1": int(
                    sum(rank <= 1 for rank in correct_anchor_ranks)
                ),
                "correct_anchor_rank_le_16": int(
                    sum(rank <= 16 for rank in correct_anchor_ranks)
                ),
                "correspondences": int(winners.numel()),
                "pose_solves": 1,
                "poselib_iterations": int(estimate.diagnostics.get("iterations", 0)),
                "online_latency_ms": online_latency_ms,
                "loo_feedback_latency_ms": loo_latency_ms,
                "detectable_matching_pairs": detectable_pairs,
                "top1_correct_pairs": matching_pairs,
            }
        )
    feedback = build_self_localization_feedback(
        query_names=list(observations.names),
        records=feedback_records,
        required_rank=int(required_rank),
        source_map_sha256=source_map_sha256,
        query_cache_sha256=query_cache_sha256,
    )
    summary = _summary(query_rows)
    summary["anchor_count"] = int(base_xyz.shape[0])
    return {
        "schema": "lafgs_v6_query_local_feedback_evaluation",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": query_rows,
        "summary": summary,
        "feedback": feedback,
        "contract": {
            "query_descriptor_loo": True,
            "query_geometry_loo": True,
            "global_top1": True,
            "pose_solves_per_query": 1,
            "retrieval": False,
            "refinement": False,
        },
    }
