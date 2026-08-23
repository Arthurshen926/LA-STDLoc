"""Formal mapping-only query-local feedback for the V6 closed loop."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from common.v6_contracts import require_mapping_only
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
    if not float(radius_px) > 0:
        raise ValueError("positive radius must be positive")
    visible_xy = projected[visible_rows].float().numpy()
    tree = cKDTree(visible_xy)
    neighbors = tree.query_ball_point(
        keypoints.float().numpy(), r=float(radius_px), return_sorted=True
    )
    for row, local_rows in enumerate(neighbors):
        if len(local_rows):
            result[row] = visible_rows[
                torch.as_tensor(local_rows, dtype=torch.long)
            ].tolist()
    return result


def _visible_spatial_rank(
    projected: torch.Tensor,
    visible_rows: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    grid_rows: int = 4,
    grid_cols: int = 4,
) -> int:
    """Count independently occupied image cells, not raw visible Anchors."""

    if visible_rows.numel() == 0:
        return 0
    height, width = image_hw
    xy = projected[visible_rows]
    columns = torch.floor(xy[:, 0] / max(float(width), 1.0) * grid_cols).long()
    rows = torch.floor(xy[:, 1] / max(float(height), 1.0) * grid_rows).long()
    columns = columns.clamp(0, grid_cols - 1)
    rows = rows.clamp(0, grid_rows - 1)
    return int(torch.unique(rows * grid_cols + columns).numel())


def _pose_neighborhoods(
    observations: ObservationProvider,
    neighbor_count: int,
) -> list[torch.Tensor]:
    """Return deterministic query-local pose neighborhoods.

    Translation and rotation are normalized by their mapping-trajectory
    nearest-neighbor scales.  This avoids filename/sequence heuristics while
    preventing adjacent trajectory frames from leaking into LOO maps.
    """

    count = len(observations)
    neighbor_count = min(max(int(neighbor_count), 1), count)
    if neighbor_count == 1:
        return [torch.tensor([index], dtype=torch.long) for index in range(count)]
    poses = torch.stack(
        [observations.build_view(index).pose_w2c.float() for index in range(count)]
    )
    rotations = poses[:, :3, :3]
    centers = -(rotations.transpose(1, 2) @ poses[:, :3, 3, None]).squeeze(-1)
    translation = torch.cdist(centers, centers)
    trace = torch.einsum("aij,bij->ab", rotations, rotations)
    rotation = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    diagonal = torch.eye(count, dtype=torch.bool)

    def nearest_scale(distance: torch.Tensor, fallback: float) -> torch.Tensor:
        masked = distance.masked_fill(diagonal | (distance <= 1e-8), torch.inf)
        nearest = masked.min(1).values
        finite = nearest[torch.isfinite(nearest)]
        if finite.numel() == 0:
            return torch.tensor(float(fallback), dtype=distance.dtype)
        return finite.median().clamp_min(1e-6)

    translation_scale = nearest_scale(translation, 1.0)
    rotation_scale = nearest_scale(rotation, math.radians(15.0))
    distance = translation / translation_scale + rotation / rotation_scale
    # Stable sorting makes repeated/same-pose cameras deterministic.
    order = torch.argsort(distance, dim=1, stable=True)
    return [torch.sort(order[index, :neighbor_count]).values for index in range(count)]


def _positive_score_statistics(
    dense_scores: torch.Tensor,
    positives_by_row: list[list[int]],
    *,
    chunk_size: int = 64,
) -> dict[int, tuple[float, float, int, int, int]]:
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
        padded_cpu = torch.full(
            (len(rows_list), maximum_degree),
            -1,
            dtype=torch.long,
        )
        lengths = torch.tensor(
            [len(positives_by_row[row]) for row in rows_list], dtype=torch.long
        )
        flat = torch.tensor(
            [anchor for row in rows_list for anchor in positives_by_row[row]],
            dtype=torch.long,
        )
        local = torch.repeat_interleave(torch.arange(len(rows_list)), lengths)
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        within = torch.arange(flat.numel()) - torch.repeat_interleave(
            offsets[:-1], lengths
        )
        padded_cpu[local, within] = flat
        padded = padded_cpu.to(dense_scores.device)
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
        best_wrong_anchor = torch.argmax(wrong, dim=1)
        packed = torch.stack(
            [best_positive, best_wrong, ranks, best_anchor, best_wrong_anchor], dim=1
        ).cpu()
        for row, values in zip(rows_list, packed.tolist()):
            result[row] = (
                float(values[0]),
                float(values[1]),
                int(values[2]),
                int(values[3]),
                int(values[4]),
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
        "oracle_feedback_localization_latency_ms": float(
            np.mean([row["online_latency_ms"] for row in rows])
        ),
        "loo_feedback_latency_ms": float(
            np.mean([row["loo_feedback_latency_ms"] for row in rows])
        ),
    }


def _descriptor_training_query_masks(state: dict, query_count: int) -> dict:
    """Return declared training-split and direct-gradient query masks.

    Older residual checkpoints did not serialize this registry.  When a
    non-zero residual exists without a registry, fail closed and treat every
    mapping query as training-reused rather than claiming query LOO.
    """

    training_split = torch.zeros(int(query_count), dtype=torch.bool)
    gradient_reused = torch.zeros(int(query_count), dtype=torch.bool)
    explicit = False
    report = state.get("v6_descriptor_distillation")
    if isinstance(report, dict):
        training_value = report.get("training_query_indices")
        selected_value = report.get("selected_query_indices", training_value)
        if training_value is not None:
            rows = torch.as_tensor(training_value, dtype=torch.long).reshape(-1)
            if rows.numel() and (
                int(rows.min()) < 0 or int(rows.max()) >= int(query_count)
            ):
                raise ValueError("descriptor training query registry is invalid")
            training_split[rows] = True
        if selected_value is not None:
            rows = torch.as_tensor(selected_value, dtype=torch.long).reshape(-1)
            if rows.numel() and (
                int(rows.min()) < 0 or int(rows.max()) >= int(query_count)
            ):
                raise ValueError("descriptor gradient query registry is invalid")
            gradient_reused[rows] = True
        explicit = bool(report.get("training_query_registry_explicit", False))
        if training_value is not None or selected_value is not None:
            return {
                "training_split": training_split,
                "gradient_reused": gradient_reused,
                "training_registry_explicit": explicit,
                "descriptor_dependency_present": True,
            }
        training_split[:] = True
        gradient_reused[:] = True
        return {
            "training_split": training_split,
            "gradient_reused": gradient_reused,
            "training_registry_explicit": False,
            "descriptor_dependency_present": True,
        }
    residual = state.get("anchor_descriptor_residual")
    if residual is not None and bool(torch.as_tensor(residual).abs().max() > 0):
        training_split[:] = True
        gradient_reused[:] = True
        dependency_present = True
    else:
        dependency_present = False
    return {
        "training_split": training_split,
        "gradient_reused": gradient_reused,
        "training_registry_explicit": False,
        "descriptor_dependency_present": dependency_present,
    }


def _descriptor_training_query_mask(state: dict, query_count: int) -> torch.Tensor:
    """Backward-compatible direct-gradient mask used by focused tests/tools."""

    return _descriptor_training_query_masks(state, query_count)["gradient_reused"]


def _reconstruction_target_query_mask(
    state: dict, query_count: int
) -> torch.Tensor:
    """Queries whose rendered depth selected a reconstruction seed region."""

    mask = torch.zeros(int(query_count), dtype=torch.bool)
    report = state.get("v6_reconstruction_distillation")
    if not isinstance(report, dict):
        # Initial mapping completion Anchors share the same candidate kind as
        # later feedback-targeted reconstruction, so kind alone is not a
        # dependency signal.  Legacy targeted rounds did persist this exact
        # feedback lineage even before the target registry was added.
        legacy_reconstruction = (
            state.get("provenance", {}).get("v6_reconstruction_feedback_sha256")
            is not None
        )
        if legacy_reconstruction:
            mask[:] = True
        return mask
    rows = torch.as_tensor(
        report.get("target_query_indices", ()), dtype=torch.long
    ).reshape(-1)
    if rows.numel() and (
        int(rows.min()) < 0 or int(rows.max()) >= int(query_count)
    ):
        raise ValueError("reconstruction target query registry is invalid")
    mask[rows] = True
    return mask


def _selection_training_query_mask(state: dict, query_count: int) -> torch.Tensor:
    """Queries whose feedback directly determined Anchor selection."""

    mask = torch.zeros(int(query_count), dtype=torch.bool)
    report = state.get("v6_selection_distillation")
    if not isinstance(report, dict):
        return mask
    rows = torch.as_tensor(
        report.get("training_query_indices", ()), dtype=torch.long
    ).reshape(-1)
    # Older selection maps did not serialize dependencies.  Fail closed.
    if rows.numel() == 0:
        mask[:] = True
        return mask
    if int(rows.min()) < 0 or int(rows.max()) >= int(query_count):
        raise ValueError("selection training query registry is invalid")
    mask[rows] = True
    return mask


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
    loo_pose_neighbors: int = 1,
    required_visibility_rank: int = 4,
    required_detectable_rank: int | None = None,
    loo_affected_anchor_policy: str = "rebuild",
) -> dict:
    """One global Top-1 and one standard PoseLib solve per mapping query."""

    evaluation_started = time.perf_counter()
    require_mapping_only(state.get("provenance", {}), label="V6 feedback map")
    # Purging every Anchor touched by the held-out pose neighborhood is a
    # conservative, leakage-free holdout that scales to full maps.  It is not
    # equivalent to rebuilding Anchors that retain enough support.
    replay = LeaveOneQueryOutProjectiveMap(
        state,
        observations,
        affected_anchor_policy=loo_affected_anchor_policy,
    )
    base_xyz = torch.as_tensor(state["anchor_xyz"]).float()
    base_bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    query_rows = []
    feedback_records = []
    descriptor_masks = _descriptor_training_query_masks(
        state, len(observations)
    )
    descriptor_training_split = descriptor_masks["training_split"]
    descriptor_gradient_reused = descriptor_masks["gradient_reused"]
    reconstruction_target_reused = _reconstruction_target_query_mask(
        state, len(observations)
    )
    selection_training_reused = _selection_training_query_mask(
        state, len(observations)
    )
    pose_neighborhoods = _pose_neighborhoods(observations, loo_pose_neighbors)
    print(
        f"[v6-feedback] prepared {loo_affected_anchor_policy} "
        "pose-neighborhood LOO "
        f"for {len(observations)} queries and {base_xyz.shape[0]} Anchors "
        f"in {time.perf_counter() - evaluation_started:.1f}s",
        flush=True,
    )
    for query_index, name in enumerate(observations.names):
        if query_index % 25 == 0:
            print(
                f"[v6-feedback] query {query_index + 1}/{len(observations)}: {name}",
                flush=True,
            )
        view = observations.build_view(query_index)
        loo_started = time.perf_counter()
        excluded_queries = pose_neighborhoods[query_index]
        update = replay.query_update(
            query_index, excluded_queries=excluded_queries
        )
        loo_latency_ms = (time.perf_counter() - loo_started) * 1000.0
        online_started = time.perf_counter()
        xyz = base_xyz
        bank = base_bank
        active = torch.ones(xyz.shape[0], dtype=torch.bool)
        affected = update["anchor_rows"]
        if affected.numel():
            if bool(update["valid"].any()):
                xyz = base_xyz.clone()
                bank = base_bank.clone()
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
        visible_rank = _visible_spatial_rank(
            projected,
            visible_rows,
            image_hw=(height, width),
        )
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
        descriptor_triplets = []
        positive_statistics = _positive_score_statistics(
            dense_scores, positive_edges
        )
        for row, positives in enumerate(positive_edges):
            if positives:
                (
                    positive_score,
                    wrong_score,
                    rank,
                    best,
                    best_wrong_anchor,
                ) = positive_statistics[row]
                best_positive.append(positive_score)
                best_wrong.append(wrong_score)
                correct_anchor_ranks.append(rank)
                if not bool(correct[row]):
                    confusion_pairs.append((row, int(winners[row]), best))
                descriptor_triplets.append(
                    (row, best, best_wrong_anchor, int(bool(correct[row])))
                )
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
                "visible_rank": int(visible_rank),
                "visible_anchor_count": int(visible_rows.numel()),
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
                ).reshape(-1, 3),
                "descriptor_triplets": torch.tensor(
                    descriptor_triplets, dtype=torch.long
                ).reshape(-1, 4),
                "excluded_query_indices": excluded_queries,
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
                "query_descriptor_loo": not bool(
                    descriptor_gradient_reused[query_index]
                ),
                "descriptor_training_query_reused": bool(
                    descriptor_gradient_reused[query_index]
                ),
                "descriptor_training_split_member": bool(
                    descriptor_training_split[query_index]
                ),
                "query_geometry_loo": not bool(
                    reconstruction_target_reused[query_index]
                ),
                "query_raw_geometry_observation_loo": True,
                "query_candidate_topology_loo": not bool(
                    reconstruction_target_reused[query_index]
                    or selection_training_reused[query_index]
                ),
                "reconstruction_target_query_reused": bool(
                    reconstruction_target_reused[query_index]
                ),
                "selection_training_query_reused": bool(
                    selection_training_reused[query_index]
                ),
                "independent_mapping_validation_query": bool(
                    (
                        not descriptor_masks["descriptor_dependency_present"]
                        or (
                            descriptor_masks["training_registry_explicit"]
                            and not descriptor_training_split[query_index]
                        )
                    )
                    and not reconstruction_target_reused[query_index]
                    and not selection_training_reused[query_index]
                ),
                "pose_neighborhood_loo": int(excluded_queries.numel()) > 1,
                "affected_anchor_policy": update["contract"][
                    "affected_anchor_policy"
                ],
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
                "descriptor_training_query_reused": bool(
                    descriptor_gradient_reused[query_index]
                ),
                "descriptor_training_split_member": bool(
                    descriptor_training_split[query_index]
                ),
                "reconstruction_target_query_reused": bool(
                    reconstruction_target_reused[query_index]
                ),
                "selection_training_query_reused": bool(
                    selection_training_reused[query_index]
                ),
                "independent_mapping_validation_query": bool(
                    (
                        not descriptor_masks["descriptor_dependency_present"]
                        or (
                            descriptor_masks["training_registry_explicit"]
                            and not descriptor_training_split[query_index]
                        )
                    )
                    and not reconstruction_target_reused[query_index]
                    and not selection_training_reused[query_index]
                ),
                "detectable_matching_pairs": detectable_pairs,
                "top1_correct_pairs": matching_pairs,
            }
        )
    feedback = build_self_localization_feedback(
        query_names=list(observations.names),
        records=feedback_records,
        required_rank=int(required_rank),
        required_visibility_rank=int(required_visibility_rank),
        required_detectable_rank=(
            int(required_rank)
            if required_detectable_rank is None
            else int(required_detectable_rank)
        ),
        source_map_sha256=source_map_sha256,
        query_cache_sha256=query_cache_sha256,
    )
    summary = _summary(query_rows)
    summary["anchor_count"] = int(base_xyz.shape[0])
    validation_rows = []
    if bool(descriptor_masks["training_registry_explicit"]):
        validation_rows = [
            row
            for row in query_rows
            if not bool(row["descriptor_training_split_member"])
        ]
    training_replay_rows = [
        row for row in query_rows if bool(row["descriptor_training_split_member"])
    ]
    gradient_reuse_rows = [
        row for row in query_rows if bool(row["descriptor_training_query_reused"])
    ]
    validation_summary = None if not validation_rows else _summary(validation_rows)
    independent_validation_rows = [
        row
        for row in query_rows
        if bool(row["independent_mapping_validation_query"])
    ]
    independent_validation_summary = (
        None
        if not independent_validation_rows
        else _summary(independent_validation_rows)
    )
    training_replay_summary = (
        None if not training_replay_rows else _summary(training_replay_rows)
    )
    gradient_reuse_summary = (
        None if not gradient_reuse_rows else _summary(gradient_reuse_rows)
    )
    reconstruction_replay_rows = [
        row
        for row in query_rows
        if bool(row["reconstruction_target_query_reused"])
    ]
    reconstruction_replay_summary = (
        None
        if not reconstruction_replay_rows
        else _summary(reconstruction_replay_rows)
    )
    selection_replay_rows = [
        row for row in query_rows if bool(row["selection_training_query_reused"])
    ]
    selection_replay_summary = (
        None if not selection_replay_rows else _summary(selection_replay_rows)
    )
    descriptor_query_loo = not bool(descriptor_gradient_reused.any())
    return {
        "schema": "lafgs_v6_query_local_feedback_evaluation",
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": query_rows,
        "summary": summary,
        "descriptor_validation_summary": validation_summary,
        "independent_mapping_validation_summary": independent_validation_summary,
        "descriptor_training_replay_summary": training_replay_summary,
        "descriptor_gradient_reuse_summary": gradient_reuse_summary,
        "reconstruction_target_replay_summary": reconstruction_replay_summary,
        "selection_training_replay_summary": selection_replay_summary,
        "feedback": feedback,
        "contract": {
            "query_descriptor_loo": descriptor_query_loo,
            "descriptor_training_split_query_count": int(
                descriptor_training_split.sum()
            ),
            "descriptor_gradient_reuse_query_count": int(
                descriptor_gradient_reused.sum()
            ),
            "descriptor_validation_query_count": len(validation_rows),
            "independent_mapping_validation_query_count": len(
                independent_validation_rows
            ),
            "descriptor_training_registry_explicit": bool(
                descriptor_masks["training_registry_explicit"]
            ),
            "descriptor_dependency_present": bool(
                descriptor_masks["descriptor_dependency_present"]
            ),
            "independent_mapping_validation_available": bool(
                independent_validation_rows
            ),
            "query_geometry_loo": not bool(reconstruction_target_reused.any()),
            "query_raw_geometry_observation_loo": True,
            "query_candidate_topology_loo": not bool(
                reconstruction_target_reused.any()
                or selection_training_reused.any()
            ),
            "reconstruction_target_reuse_query_count": int(
                reconstruction_target_reused.sum()
            ),
            "selection_training_reuse_query_count": int(
                selection_training_reused.sum()
            ),
            "pose_neighborhood_loo": int(loo_pose_neighbors) > 1,
            "loo_pose_neighbors": int(loo_pose_neighbors),
            "affected_anchor_policy": loo_affected_anchor_policy,
            "positive_radius_px": float(positive_radius_px),
            "alpha_minimum": float(alpha_minimum),
            "required_matching_rank": int(required_rank),
            "required_visibility_rank": int(required_visibility_rank),
            "required_detectable_rank": int(
                required_rank
                if required_detectable_rank is None
                else required_detectable_rank
            ),
            "ransac_reprojection_px": float(ransac_reprojection_px),
            "ransac_seed": int(seed),
            "evaluation_device": str(device),
            "affected_anchor_holdout_is_exact_rebuild": (
                loo_affected_anchor_policy == "rebuild"
            ),
            "purged_holdout_is_exact_rebuild": (
                False if loo_affected_anchor_policy == "purge" else None
            ),
            "reported_online_latency_is_oracle_feedback_localization": True,
            "global_top1": True,
            "pose_solves_per_query": 1,
            "retrieval": False,
            "refinement": False,
        },
    }
