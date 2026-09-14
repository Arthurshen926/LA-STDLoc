"""Reusable exact Top-K mapping candidates and parallel one-shot pose replay."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable
import math

import torch
import torch.nn.functional as F

from localization.localizer import load_shared_metric
from localization.matcher import TopKMatches, maximum_weight_anchor_assignment
from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.trainer import _pose_error_cm, _project_errors
from topology.deployment_revision import _csr_contains_at_rows, _csr_values, _summary
from topology.pose_information import (
    conditional_delete_loss,
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
)


COUNTER_NAMES = (
    "winner_count",
    "correct_winner_count",
    "false_attractor_count",
    "ambiguous_winner_count",
    "clean_inlier_count",
    "harmful_inlier_count",
    "counterfactual_clean_gain",
    "information_deletion_loss",
)


def validate_mapping_topk(sidecar: dict) -> None:
    if sidecar.get("schema") != "lafgs_v4_mapping_loo_topk_sidecar":
        raise ValueError("unsupported mapping Top-K sidecar")
    if sidecar.get("uses_test_queries") is not False:
        raise ValueError("mapping Top-K sidecar unexpectedly uses test queries")
    topk = int(sidecar.get("topk", 0))
    anchor_count = int(sidecar.get("anchor_count", 0))
    names = list(sidecar.get("query_names", ()))
    records = list(sidecar.get("records", ()))
    if topk < 1 or anchor_count < topk or len(names) != len(records):
        raise ValueError("mapping Top-K sidecar registry is invalid")
    for query_index, (name, record) in enumerate(zip(names, records)):
        if (
            int(record.get("query_index", -1)) != query_index
            or record.get("image_name") != name
        ):
            raise ValueError("mapping Top-K query order is invalid")
        indices = torch.as_tensor(record.get("anchor_indices"))
        scores = torch.as_tensor(record.get("scores"))
        local_rows = torch.as_tensor(record.get("teacher_local_rows"))
        cache_rows = torch.as_tensor(record.get("cache_rows"))
        keypoints = torch.as_tensor(record.get("keypoints"))
        intrinsic = torch.as_tensor(record.get("intrinsic"))
        pose = torch.as_tensor(record.get("pose_w2c"))
        count = int(indices.shape[0]) if indices.ndim == 2 else -1
        if (
            indices.dtype != torch.int64
            or indices.shape != (count, topk)
            or scores.dtype != torch.float32
            or scores.shape != indices.shape
            or local_rows.dtype != torch.int64
            or local_rows.shape != (count,)
            or cache_rows.dtype != torch.int64
            or cache_rows.shape != (count,)
            or keypoints.dtype != torch.float32
            or keypoints.shape != (count, 2)
            or intrinsic.dtype != torch.float32
            or intrinsic.shape != (3, 3)
            or pose.dtype != torch.float32
            or pose.shape != (4, 4)
        ):
            raise ValueError("mapping Top-K tensor contract is invalid")
        if count < 1 or bool((indices < 0).any()) or int(indices.max()) >= anchor_count:
            raise ValueError("mapping Top-K Anchor index is invalid")
        if not bool(torch.isfinite(scores).all()):
            raise ValueError("mapping Top-K score is not finite")
        if topk > 1 and bool((scores[:, 1:] > scores[:, :-1]).any()):
            raise ValueError("mapping Top-K scores are not rank sorted")
        if topk > 1:
            ordered = torch.sort(indices, dim=1).values
            if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
                raise ValueError("mapping Top-K row contains a duplicate Anchor")


@torch.inference_mode()
def materialize_mapping_topk(
    *,
    state: dict,
    metric_state_path: str | Path,
    teacher: dict,
    query_cache: dict,
    device: torch.device,
    anchor_bank_updater: Callable[[int, torch.Tensor], None],
    topk: int = 8,
    deployment_row_limit: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Materialize exact LOO candidates once, without running a pose solver."""
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    topk = int(topk)
    if topk < 1 or topk > anchor_count:
        raise ValueError("sidecar Top-K must be within the Anchor count")
    if int(teacher["anchor_count"]) != anchor_count:
        raise ValueError("teacher and deployment map anchor counts differ")
    metric = load_shared_metric(
        metric_state_path,
        anchor_ids=torch.as_tensor(state["anchor_ids"]).long(),
        device=device,
    )
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float().to(device), dim=1
    )
    cache = query_cache.get("queries", query_cache)
    names = list(teacher["query_names"])
    if names != list(cache):
        raise ValueError("query cache order differs from the teacher")
    records = []
    for query_index, name in enumerate(names):
        anchor_bank_updater(query_index, bank)
        teacher_record = teacher["records"][query_index]
        all_rows = torch.as_tensor(teacher_record["query_rows"]).long()
        local_rows = torch.arange(all_rows.numel())
        rows = all_rows
        if int(deployment_row_limit) > 0:
            keep = rows < int(deployment_row_limit)
            rows = rows[keep]
            local_rows = local_rows[keep]
        if rows.numel() == 0:
            raise ValueError(f"query {name} has no retained deployment rows")
        cached = cache[name]
        descriptors = F.normalize(
            torch.as_tensor(cached["native_descriptors"]).float()[rows], dim=1
        ).to(device)
        adapted, _ = metric(descriptors)
        scores, indices = torch.topk(adapted @ bank.T, k=topk, dim=1)
        records.append(
            {
                "query_index": int(query_index),
                "image_name": name,
                "teacher_local_rows": local_rows.cpu(),
                "cache_rows": rows.cpu(),
                "keypoints": (
                    torch.as_tensor(cached["native_keypoints"]).float()[rows]
                    + float(cached.get("pixel_center_offset", 0.5))
                ).cpu(),
                "intrinsic": torch.as_tensor(cached["native_K"]).float().cpu(),
                "pose_w2c": torch.as_tensor(cached["pose_w2c"]).float().cpu(),
                "anchor_indices": indices.cpu(),
                "scores": scores.cpu(),
            }
        )
        if progress is not None:
            progress(query_index + 1, len(names))
    return {
        "schema": "lafgs_v4_mapping_loo_topk_sidecar",
        "version": 1,
        "uses_test_queries": False,
        "topk": topk,
        "anchor_count": anchor_count,
        "query_names": names,
        "records": records,
    }


def _solve_pose(task: tuple) -> object:
    points_2d, points_3d, intrinsic, reprojection_px, seed = task
    return solve_absolute_pose(
        points_2d,
        points_3d,
        intrinsic,
        reprojection_error_px=float(reprojection_px),
        confidence=0.99999,
        max_iterations=100000,
        min_iterations=1000,
        seed=int(seed),
    )


def _ordered_pose_results(tasks: list[tuple], workers: int) -> Iterable[object]:
    workers = int(workers)
    if workers < 1:
        raise ValueError("pose worker count must be positive")
    if workers == 1:
        return [_solve_pose(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="poselib") as pool:
        return list(pool.map(_solve_pose, tasks))


@torch.inference_mode()
def replay_mapping_topk(
    *,
    sidecar: dict,
    state: dict,
    teacher: dict,
    assignment_topk: int,
    assignment_dustbin_score: float = -1.0,
    assignment_maximum_regret: float | None = None,
    ransac_reprojection_px: float,
    clean_reprojection_px: float,
    task_translation_m: float,
    task_rotation_deg: float,
    seed: int,
    pose_workers: int = 1,
) -> dict:
    """Replay Top-1 or a regret-bounded partial assignment from one sidecar."""
    validate_mapping_topk(sidecar)
    records = list(sidecar["records"])
    names = list(teacher["query_names"])
    if names != list(sidecar["query_names"]) or len(records) != len(names):
        raise ValueError("sidecar and teacher query registries differ")
    xyz = torch.as_tensor(state["anchor_xyz"]).float().cpu()
    anchor_count = int(xyz.shape[0])
    if int(sidecar["anchor_count"]) != anchor_count:
        raise ValueError("sidecar Anchor registry differs from the map")
    assignment_topk = int(assignment_topk)
    if assignment_topk < 0 or assignment_topk > int(sidecar["topk"]):
        raise ValueError("assignment Top-K is outside the sidecar")
    if assignment_maximum_regret is not None and float(assignment_maximum_regret) < 0:
        raise ValueError("maximum assignment regret must be non-negative")

    counters = {
        name: torch.zeros(anchor_count, dtype=torch.float64) for name in COUNTER_NAMES
    }
    prepared = []
    tasks = []
    for query_index, candidate in enumerate(records):
        if (
            int(candidate["query_index"]) != query_index
            or candidate["image_name"] != names[query_index]
        ):
            raise ValueError("sidecar query record order is invalid")
        indices = torch.as_tensor(candidate["anchor_indices"]).long()
        scores = torch.as_tensor(candidate["scores"]).float()
        local_rows = torch.as_tensor(candidate["teacher_local_rows"]).long()
        keypoints = torch.as_tensor(candidate["keypoints"]).float()
        intrinsic = torch.as_tensor(candidate["intrinsic"]).float()
        if indices.shape != scores.shape or indices.ndim != 2:
            raise ValueError("sidecar candidate matrices do not align")
        assignment = None
        if assignment_topk:
            selected_scores = scores[:, :assignment_topk].clone()
            if assignment_maximum_regret is not None:
                eligible = selected_scores >= (
                    selected_scores[:, :1] - float(assignment_maximum_regret)
                )
                selected_scores[~eligible] = float(assignment_dustbin_score)
            assignment = maximum_weight_anchor_assignment(
                TopKMatches(
                    keypoint_indices=torch.arange(indices.shape[0]),
                    anchor_indices=indices[:, :assignment_topk],
                    scores=selected_scores,
                ),
                dustbin_score=float(assignment_dustbin_score),
            )
            positions = assignment.matches.keypoint_indices.cpu()
            winners = assignment.matches.anchor_indices.cpu()
            selected_local_rows = local_rows[positions]
            selected_keypoints = keypoints[positions]
        else:
            positions = torch.arange(indices.shape[0])
            winners = indices[:, 0]
            selected_local_rows = local_rows
            selected_keypoints = keypoints

        record = teacher["records"][query_index]
        counters["winner_count"].index_add_(
            0, winners, torch.ones(winners.numel(), dtype=torch.float64)
        )
        correct, has_positive = _csr_contains_at_rows(
            record, "positive", selected_local_rows, winners
        )
        ambiguous, _ = _csr_contains_at_rows(
            record, "ambiguous", selected_local_rows, winners
        )
        for counter_name, selected in (
            ("correct_winner_count", winners[correct]),
            ("ambiguous_winner_count", winners[~correct & ambiguous]),
            ("false_attractor_count", winners[~correct & ~ambiguous & has_positive]),
        ):
            counters[counter_name].index_add_(
                0, selected, torch.ones(selected.numel(), dtype=torch.float64)
            )
        if not assignment_topk:
            for local, winner in enumerate(winners.tolist()):
                positives = _csr_values(
                    record, "positive", int(selected_local_rows[local])
                )
                replacement_correct = any(
                    bool((positives == alternative).any())
                    for alternative in indices[positions[local], 1:].tolist()
                    if alternative != winner
                )
                counters["counterfactual_clean_gain"][winner] += float(
                    replacement_correct
                ) - float(correct[local])

        task = (
            selected_keypoints.numpy(),
            xyz[winners].numpy(),
            intrinsic.numpy(),
            float(ransac_reprojection_px),
            int(seed),
        )
        tasks.append(task)
        prepared.append(
            {
                "query_index": query_index,
                "image_name": names[query_index],
                "winners": winners,
                "keypoints": selected_keypoints,
                "intrinsic": intrinsic,
                "pose_w2c": torch.as_tensor(candidate["pose_w2c"]).float(),
                "correct": correct,
                "assignment": assignment,
            }
        )

    estimates = _ordered_pose_results(tasks, pose_workers)
    query_rows = []
    for prepared_query, estimate in zip(prepared, estimates):
        winners = prepared_query["winners"]
        keypoints = prepared_query["keypoints"]
        intrinsic = prepared_query["intrinsic"]
        pose_w2c = prepared_query["pose_w2c"]
        inliers = torch.as_tensor(estimate.inliers).long().reshape(-1)
        clean_mask = torch.zeros(inliers.numel(), dtype=torch.bool)
        if inliers.numel():
            errors = _project_errors(
                xyz[winners[inliers]], keypoints[inliers], intrinsic, pose_w2c
            )
            clean_mask = errors <= float(clean_reprojection_px)
            clean_anchor = winners[inliers[clean_mask]]
            harmful_anchor = winners[inliers[~clean_mask]]
            counters["clean_inlier_count"].index_add_(
                0, clean_anchor, torch.ones(clean_anchor.numel(), dtype=torch.float64)
            )
            counters["harmful_inlier_count"].index_add_(
                0,
                harmful_anchor,
                torch.ones(harmful_anchor.numel(), dtype=torch.float64),
            )
            if clean_anchor.numel():
                clean_points = xyz[clean_anchor].double()
                jacobian = task_scaled_pose_jacobian(
                    pose_jacobian_analytic(
                        clean_points, intrinsic.double(), pose_w2c.double()
                    ),
                    translation_scale=float(task_translation_m),
                    rotation_scale=math.radians(float(task_rotation_deg)),
                )
                contribution = fisher_contributions(
                    jacobian,
                    measurement_covariance=torch.full(
                        (clean_points.shape[0],),
                        max(float(clean_reprojection_px), 0.5) ** 2,
                        dtype=torch.float64,
                    ),
                )
                full = (
                    contribution.sum(dim=0) + torch.eye(6, dtype=torch.float64) * 1e-4
                )
                for anchor in torch.unique(clean_anchor).tolist():
                    selected = clean_anchor == int(anchor)
                    loss = conditional_delete_loss(
                        full, contribution[selected].sum(dim=0), objective="full"
                    )
                    counters["information_deletion_loss"][anchor] += float(
                        loss.clamp_min(0)
                    )
        ae_deg, _ = pose_error(estimate.pose_w2c, pose_w2c.numpy())
        te_cm = _pose_error_cm(estimate.pose_w2c, pose_w2c)
        assignment = prepared_query["assignment"]
        query_rows.append(
            {
                "query_index": prepared_query["query_index"],
                "image_name": prepared_query["image_name"],
                "te_cm": float(te_cm),
                "ae_deg": float(ae_deg),
                "inliers": int(inliers.numel()),
                "clean_inliers": int(clean_mask.sum()),
                "hypotheses": int(estimate.diagnostics.get("iterations", 0)),
                "group_diverse_selected": False,
                "correspondences": int(winners.numel()),
                "assignment_topk": assignment_topk,
                "assignment_unmatched_queries": int(assignment.unmatched_query_count)
                if assignment
                else 0,
                "assignment_reassigned_queries": int(assignment.reassigned_query_count)
                if assignment
                else 0,
                "assignment_top1_collisions": int(assignment.top1_collision_count)
                if assignment
                else 0,
            }
        )
    return {
        "counters": counters,
        "queries": query_rows,
        "summary": _summary(query_rows, counters),
    }
