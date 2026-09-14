"""Sparse LGCV filtering for query-specific Top-K geometric feedback.

This arm does not render an image and does not perform dense matching.  It
uses the first PoseLib estimate only to project sparse Top-K Anchor candidates.
Local triangles whose two neighbours are frozen first-pass PoseLib inliers
provide a query-level confidence gate.  The coupled Top-K proposal is kept
whole rather than pruned row-by-row.  Proposed replacements never support one
another, and first-pass inliers are never modified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
import re
import time
import uuid

import torch
import torch.nn.functional as F

from localization.matcher import global_cosine_topk
from map_learning.v21_pose_feedback_transductive import (
    replay_pose_with_contract,
    validate_complete_cache_payloads,
)
from map_learning.v21_topk_geometric_feedback import (
    default_config as topk_default_config,
    select_topk_geometry_rows,
)


SCHEMA = "lafgs_v22_sparse_lgcv_topk_feedback_evaluation"
FINAL_SCHEMA = "lafgs_v22_sparse_lgcv_topk_feedback_final_decision"
VERSION = 2
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def default_config() -> dict:
    """Return the predeclared sparse fusion configuration.

    LGCV constants follow the public ULF-Loc Cambridge configuration.  The
    only deliberate semantic change is that neighbourhoods contain frozen
    first-pass inliers rather than other unverified proposals.
    """

    return {
        "topk_geometry": topk_default_config(),
        "lgcv_neighbors": 8,
        "lgcv_angle_threshold_cosine": 0.9659,
        "lgcv_scale_threshold": 0.1,
        "lgcv_scale_limit": 3.0,
        "lgcv_maximum_neighbor_distance_px": 50.0,
        "lgcv_support_threshold_exclusive": 4,
        "minimum_query_supported_proposal_count": 8,
        "minimum_query_supported_proposal_fraction": 0.05,
        "lgcv_reference_rows": "first_pass_poselib_inliers_only",
        "lgcv_proposals_cannot_support_each_other": True,
        "lgcv_decision_granularity": "query_bundle_gate_not_row_filter",
        "preserve_all_first_pass_inliers": True,
        "second_pose_estimate": "one_standard_poselib_replay_after_lgcv",
        "ground_truth_used_by_selection": False,
    }


def validate_config(value: Mapping) -> dict:
    config = dict(value)
    expected = default_config()
    if config != expected:
        raise ValueError("V22 sparse LGCV configuration differs from the frozen arm")
    return config


def _source(value: object, *, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"V22 sparse LGCV {label} source is missing")
    path = str(Path(str(value.get("path", ""))).expanduser().resolve())
    digest = str(value.get("sha256", ""))
    size = int(value.get("size_bytes", 0))
    if not path or SHA256_PATTERN.fullmatch(digest) is None or size <= 0:
        raise ValueError(f"V22 sparse LGCV {label} source is invalid")
    return {"path": path, "sha256": digest, "size_bytes": size}


def _baseline_outcome(record: Mapping) -> dict:
    return {
        "pose_w2c": torch.as_tensor(record["baseline_pose_w2c"]).float().cpu(),
        "translation_error_cm": float(record["baseline_translation_error_cm"]),
        "rotation_error_deg": float(record["baseline_rotation_error_deg"]),
        "task_error": float(record["baseline_task_error"]),
        "r5_success": bool(record["baseline_r5"]),
        "inlier_count": int(record["baseline_inlier_count"]),
        "inlier_query_rows": torch.as_tensor(record["baseline_inliers"])
        .long()
        .cpu(),
    }


def _project_rows(
    anchor_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    rows = torch.as_tensor(anchor_rows).long().cpu().reshape(-1)
    xyz = torch.as_tensor(anchor_xyz).float().cpu()
    calibration = torch.as_tensor(intrinsic).float().cpu()
    pose = torch.as_tensor(pose_w2c).float().cpu()
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or calibration.shape != (3, 3)
        or pose.shape != (4, 4)
        or (rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= xyz.shape[0]))
    ):
        raise ValueError("V22 sparse LGCV projection inputs are invalid")
    camera = (pose[:3, :3] @ xyz[rows].T).T + pose[:3, 3]
    homogeneous = (calibration @ camera.T).T
    depth = homogeneous[:, 2]
    projected = homogeneous[:, :2] / depth.clamp_min(1e-12)[:, None]
    projected[depth <= 1e-12] = torch.inf
    return projected


def sparse_lgcv_candidate_support(
    *,
    candidate_query_xy: torch.Tensor,
    candidate_projected_xy: torch.Tensor,
    reference_query_xy: torch.Tensor,
    reference_projected_xy: torch.Tensor,
    config: Mapping,
) -> dict:
    """Compute directed-triangle LGCV support from trusted reference rows.

    Each candidate is the centre of a local graph.  Its K nearest first-pass
    inliers supply the other two triangle vertices.  The returned score follows
    ULF-Loc and counts directed neighbour pairs, so ``score > 4`` means at least
    three consistent undirected triangles.
    """

    cfg = validate_config(config)
    center_x = torch.as_tensor(candidate_query_xy).float().cpu()
    center_y = torch.as_tensor(candidate_projected_xy).float().cpu()
    reference_x = torch.as_tensor(reference_query_xy).float().cpu()
    reference_y = torch.as_tensor(reference_projected_xy).float().cpu()
    if (
        center_x.ndim != 2
        or center_x.shape[1] != 2
        or center_y.shape != center_x.shape
        or reference_x.ndim != 2
        or reference_x.shape[1] != 2
        or reference_y.shape != reference_x.shape
        or not bool(torch.isfinite(center_x).all())
        or not bool(torch.isfinite(center_y).all())
        or not bool(torch.isfinite(reference_x).all())
        or not bool(torch.isfinite(reference_y).all())
    ):
        raise ValueError("V22 sparse LGCV coordinate inputs are invalid")
    candidate_count = center_x.shape[0]
    requested_k = int(cfg["lgcv_neighbors"])
    effective_k = min(requested_k, int(reference_x.shape[0]))
    if candidate_count == 0:
        return {
            "support_scores": torch.empty(0),
            "valid_directed_triangle_counts": torch.empty(0, dtype=torch.long),
            "effective_neighbor_count": effective_k,
        }
    if effective_k < 2:
        return {
            "support_scores": torch.zeros(candidate_count),
            "valid_directed_triangle_counts": torch.zeros(
                candidate_count, dtype=torch.long
            ),
            "effective_neighbor_count": effective_k,
        }

    distances = torch.cdist(center_x, reference_x)
    neighbor_rows = distances.topk(effective_k, dim=1, largest=False).indices
    x_neighbors = reference_x[neighbor_rows]
    y_neighbors = reference_y[neighbor_rows]
    x_relative = x_neighbors - center_x[:, None, :]
    y_relative = y_neighbors - center_y[:, None, :]
    eps = 1e-8

    x_norm = x_relative.norm(dim=2)
    y_norm = y_relative.norm(dim=2)
    maximum_distance = float(cfg["lgcv_maximum_neighbor_distance_px"])
    valid_neighbor = (
        (x_norm > 1e-6)
        & (y_norm > 1e-6)
        & (x_norm <= maximum_distance)
        & (y_norm <= maximum_distance)
    )
    x_unit = x_relative / (x_norm[:, :, None] + eps)
    y_unit = y_relative / (y_norm[:, :, None] + eps)
    x_dot = x_unit @ x_unit.transpose(1, 2)
    y_dot = y_unit @ y_unit.transpose(1, 2)
    angular = (x_dot - y_dot).abs() < (
        1.0 - float(cfg["lgcv_angle_threshold_cosine"])
    )

    x_j = x_relative[:, :, None, :]
    x_k = x_relative[:, None, :, :]
    y_j = y_relative[:, :, None, :]
    y_k = y_relative[:, None, :, :]
    orientation = (
        (x_j[..., 0] * x_k[..., 1] - x_j[..., 1] * x_k[..., 0])
        * (y_j[..., 0] * y_k[..., 1] - y_j[..., 1] * y_k[..., 0])
    ) > 0

    x_jk = (x_j - x_k).norm(dim=3)
    y_jk = (y_j - y_k).norm(dim=3)
    triangle = (
        valid_neighbor[:, :, None]
        & valid_neighbor[:, None, :]
        & (x_jk > 1e-6)
        & (y_jk > 1e-6)
    )
    diagonal = torch.eye(effective_k, dtype=torch.bool)[None]
    triangle &= ~diagonal

    scale_a = y_norm[:, :, None] / (x_norm[:, :, None] + eps)
    scale_b = y_norm[:, None, :] / (x_norm[:, None, :] + eps)
    scale_c = y_jk / (x_jk + eps)
    scale_threshold = float(cfg["lgcv_scale_threshold"])
    scale_similar = (
        ((scale_a - scale_b).abs() < scale_threshold)
        & ((scale_a - scale_c).abs() < scale_threshold)
        & ((scale_b - scale_c).abs() < scale_threshold)
    )
    lower = 1.0 / float(cfg["lgcv_scale_limit"])
    upper = float(cfg["lgcv_scale_limit"])
    scale_valid = (
        (scale_a > lower)
        & (scale_a < upper)
        & (scale_b > lower)
        & (scale_b < upper)
        & (scale_c > lower)
        & (scale_c < upper)
    )
    support = triangle & angular & orientation & scale_similar & scale_valid
    return {
        "support_scores": support.float().sum(dim=(1, 2)),
        "valid_directed_triangle_counts": triangle.sum(dim=(1, 2)).long(),
        "effective_neighbor_count": effective_k,
    }


def filter_provisional_assignment_with_sparse_lgcv(
    *,
    keypoints: torch.Tensor,
    baseline_anchor_rows: torch.Tensor,
    provisional_anchor_rows: torch.Tensor,
    provisional_changed_query_rows: torch.Tensor,
    baseline_inlier_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    baseline_pose_w2c: torch.Tensor,
    config: Mapping,
) -> dict:
    """Restore proposed replacements that lack first-pass-inlier support."""

    cfg = validate_config(config)
    xy = torch.as_tensor(keypoints).float().cpu()
    baseline = torch.as_tensor(baseline_anchor_rows).long().cpu().reshape(-1)
    provisional = torch.as_tensor(provisional_anchor_rows).long().cpu().reshape(-1)
    proposed_rows = (
        torch.as_tensor(provisional_changed_query_rows).long().cpu().reshape(-1)
    )
    inlier_rows = torch.as_tensor(baseline_inlier_rows).long().cpu().reshape(-1)
    if (
        xy.shape != (baseline.numel(), 2)
        or provisional.shape != baseline.shape
        or torch.unique(proposed_rows).numel() != proposed_rows.numel()
        or torch.unique(inlier_rows).numel() != inlier_rows.numel()
        or (
            proposed_rows.numel()
            and (int(proposed_rows.min()) < 0 or int(proposed_rows.max()) >= xy.shape[0])
        )
        or (
            inlier_rows.numel()
            and (int(inlier_rows.min()) < 0 or int(inlier_rows.max()) >= xy.shape[0])
        )
        or bool(torch.isin(proposed_rows, inlier_rows).any())
        or not torch.equal(
            torch.nonzero(provisional != baseline, as_tuple=False).reshape(-1),
            proposed_rows,
        )
    ):
        raise ValueError("V22 sparse LGCV provisional assignment is invalid")
    if not proposed_rows.numel():
        return {
            "anchor_rows": baseline.clone(),
            "supported_changed_query_rows": torch.empty(0, dtype=torch.long),
            "rejected_changed_query_rows": torch.empty(0, dtype=torch.long),
            "proposal_support_scores": torch.empty(0),
            "proposal_valid_directed_triangle_counts": torch.empty(
                0, dtype=torch.long
            ),
            "effective_neighbor_count": min(
                int(cfg["lgcv_neighbors"]), int(inlier_rows.numel())
            ),
        }

    candidate_projection = _project_rows(
        provisional[proposed_rows], anchor_xyz, intrinsic, baseline_pose_w2c
    )
    reference_projection = _project_rows(
        baseline[inlier_rows], anchor_xyz, intrinsic, baseline_pose_w2c
    )
    if not bool(torch.isfinite(candidate_projection).all()) or not bool(
        torch.isfinite(reference_projection).all()
    ):
        raise ValueError("V22 sparse LGCV projected coordinates are invalid")
    support = sparse_lgcv_candidate_support(
        candidate_query_xy=xy[proposed_rows],
        candidate_projected_xy=candidate_projection,
        reference_query_xy=xy[inlier_rows],
        reference_projected_xy=reference_projection,
        config=cfg,
    )
    accepted_mask = support["support_scores"] > float(
        cfg["lgcv_support_threshold_exclusive"]
    )
    supported_rows = proposed_rows[accepted_mask]
    rejected_rows = proposed_rows[~accepted_mask]
    selected = baseline.clone()
    selected[supported_rows] = provisional[supported_rows]
    if bool((selected[inlier_rows] != baseline[inlier_rows]).any()):
        raise RuntimeError("V22 sparse LGCV changed a protected first-pass inlier")
    return {
        "anchor_rows": selected,
        "supported_changed_query_rows": supported_rows,
        "rejected_changed_query_rows": rejected_rows,
        "proposal_support_scores": support["support_scores"],
        "proposal_valid_directed_triangle_counts": support[
            "valid_directed_triangle_counts"
        ],
        "effective_neighbor_count": int(support["effective_neighbor_count"]),
    }


def _synchronize(device: str | torch.device) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


def evaluate_record(
    *,
    record: Mapping,
    normalized_anchor_features: torch.Tensor,
    anchor_xyz: torch.Tensor,
    baseline_contract: Mapping,
    config: Mapping,
    device: str | torch.device,
    matcher_chunk_size: int,
) -> dict:
    """Run one sparse online feedback pass and a Top-K-only diagnostic ablation."""

    cfg = validate_config(config)
    topk_cfg = cfg["topk_geometry"]
    baseline = _baseline_outcome(record)
    eligible = (
        int(topk_cfg["minimum_baseline_inlier_count_inclusive"])
        <= int(baseline["inlier_count"])
        < int(topk_cfg["maximum_baseline_inlier_count_exclusive"])
    )
    baseline_rows = torch.as_tensor(record["winner_anchor_rows"]).long().cpu()
    empty_long = torch.empty(0, dtype=torch.long)
    provisional = {
        "anchor_rows": baseline_rows,
        "changed_query_rows": empty_long,
        "selected_candidate_ranks": empty_long,
        "selected_reprojection_residual_px": torch.empty(0),
    }
    filtered = {
        "anchor_rows": baseline_rows,
        "supported_changed_query_rows": empty_long,
        "rejected_changed_query_rows": empty_long,
        "proposal_support_scores": torch.empty(0),
        "proposal_valid_directed_triangle_counts": empty_long,
        "effective_neighbor_count": 0,
    }
    fused_candidate = baseline
    topk_only_candidate = baseline
    fused_accepted = False
    topk_only_accepted = False
    query_lgcv_gate_passed = False
    feedback_ms = 0.0
    second_pnp_ms = 0.0
    diagnostic_topk_only_pnp_ms = 0.0
    pixel_offset = float(baseline_contract["pixel_center_offset"])
    keypoints = torch.as_tensor(record["keypoints"]).float() + pixel_offset

    if eligible:
        _synchronize(device)
        feedback_start = time.perf_counter()
        matches = global_cosine_topk(
            torch.as_tensor(record["descriptors"], device=device).float(),
            normalized_anchor_features,
            topk=int(topk_cfg["topk"]),
            chunk_size=int(matcher_chunk_size),
            anchor_descriptors_normalized=True,
        )
        provisional = select_topk_geometry_rows(
            keypoints=keypoints,
            topk_anchor_rows=matches.anchor_indices.cpu(),
            topk_scores=matches.scores.cpu(),
            baseline_anchor_rows=record["winner_anchor_rows"],
            baseline_scores=record["winner_scores"],
            baseline_inlier_rows=record["baseline_inliers"],
            anchor_xyz=anchor_xyz,
            intrinsic=record["intrinsics"],
            baseline_pose_w2c=record["baseline_pose_w2c"],
            config=topk_cfg,
        )
        filtered = filter_provisional_assignment_with_sparse_lgcv(
            keypoints=keypoints,
            baseline_anchor_rows=record["winner_anchor_rows"],
            provisional_anchor_rows=provisional["anchor_rows"],
            provisional_changed_query_rows=provisional["changed_query_rows"],
            baseline_inlier_rows=record["baseline_inliers"],
            anchor_xyz=anchor_xyz,
            intrinsic=record["intrinsics"],
            baseline_pose_w2c=record["baseline_pose_w2c"],
            config=cfg,
        )
        _synchronize(device)
        feedback_ms = (time.perf_counter() - feedback_start) * 1000.0

        supported_count = int(filtered["supported_changed_query_rows"].numel())
        proposed_count = int(provisional["changed_query_rows"].numel())
        supported_fraction = (
            float(supported_count / proposed_count) if proposed_count else 0.0
        )
        query_lgcv_gate_passed = bool(
            supported_count >= int(cfg["minimum_query_supported_proposal_count"])
            and supported_fraction
            >= float(cfg["minimum_query_supported_proposal_fraction"])
        )
        if query_lgcv_gate_passed:
            pnp_start = time.perf_counter()
            replay = replay_pose_with_contract(
                keypoints=keypoints,
                anchor_rows=provisional["anchor_rows"],
                anchor_xyz=anchor_xyz,
                intrinsic=record["intrinsics"],
                ground_truth_w2c=record["pose_w2c"],
                baseline_contract=baseline_contract,
            )
            second_pnp_ms = (time.perf_counter() - pnp_start) * 1000.0
            fused_accepted = int(replay["inlier_count"]) >= (
                int(baseline["inlier_count"])
                + int(topk_cfg["minimum_candidate_inlier_gain"])
            )
            if fused_accepted:
                fused_candidate = replay

        if provisional["changed_query_rows"].numel():
            if query_lgcv_gate_passed:
                topk_replay = replay
            else:
                diagnostic_start = time.perf_counter()
                topk_replay = replay_pose_with_contract(
                    keypoints=keypoints,
                    anchor_rows=provisional["anchor_rows"],
                    anchor_xyz=anchor_xyz,
                    intrinsic=record["intrinsics"],
                    ground_truth_w2c=record["pose_w2c"],
                    baseline_contract=baseline_contract,
                )
                diagnostic_topk_only_pnp_ms = (
                    time.perf_counter() - diagnostic_start
                ) * 1000.0
            topk_only_accepted = int(topk_replay["inlier_count"]) >= (
                int(baseline["inlier_count"])
                + int(topk_cfg["minimum_candidate_inlier_gain"])
            )
            if topk_only_accepted:
                topk_only_candidate = topk_replay

    gain = bool(not baseline["r5_success"] and fused_candidate["r5_success"])
    loss = bool(baseline["r5_success"] and not fused_candidate["r5_success"])
    topk_gain = bool(
        not baseline["r5_success"] and topk_only_candidate["r5_success"]
    )
    topk_loss = bool(
        baseline["r5_success"] and not topk_only_candidate["r5_success"]
    )
    return {
        "query_index": int(record["query_index"]),
        "image_name": str(record["image_name"]),
        "sequence_id": str(record["sequence_id"]),
        "block_id": str(record["block_id"]),
        "source_record_sha256": str(record["source_record_sha256"]),
        "eligible_by_baseline_inlier_band": eligible,
        "provisional_changed_query_rows": provisional["changed_query_rows"],
        "provisional_selected_candidate_ranks": provisional[
            "selected_candidate_ranks"
        ],
        "provisional_selected_reprojection_residual_px": provisional[
            "selected_reprojection_residual_px"
        ],
        "lgcv_supported_changed_query_rows": filtered[
            "supported_changed_query_rows"
        ],
        "lgcv_rejected_changed_query_rows": filtered["rejected_changed_query_rows"],
        "lgcv_proposal_support_scores": filtered["proposal_support_scores"],
        "lgcv_proposal_valid_directed_triangle_counts": filtered[
            "proposal_valid_directed_triangle_counts"
        ],
        "lgcv_effective_neighbor_count": int(filtered["effective_neighbor_count"]),
        "query_lgcv_gate_passed": query_lgcv_gate_passed,
        "query_lgcv_supported_proposal_fraction": float(
            filtered["supported_changed_query_rows"].numel()
            / max(int(provisional["changed_query_rows"].numel()), 1)
        ),
        "candidate_accepted": fused_accepted,
        "topk_only_candidate_accepted": topk_only_accepted,
        "baseline": baseline,
        "candidate": fused_candidate,
        "topk_only_candidate": topk_only_candidate,
        "r5_gain": gain,
        "r5_loss": loss,
        "topk_only_r5_gain": topk_gain,
        "topk_only_r5_loss": topk_loss,
        "catastrophe": loss,
        "paired_delta_translation_error_cm": float(
            fused_candidate["translation_error_cm"]
            - baseline["translation_error_cm"]
        ),
        "paired_delta_rotation_error_deg": float(
            fused_candidate["rotation_error_deg"] - baseline["rotation_error_deg"]
        ),
        "paired_delta_task_error": float(
            fused_candidate["task_error"] - baseline["task_error"]
        ),
        "online_timing_ms": {
            "topk_and_sparse_lgcv": float(feedback_ms),
            "second_poselib": float(second_pnp_ms),
            "total_feedback_after_first_pose": float(feedback_ms + second_pnp_ms),
            "excluded_diagnostic_topk_only_poselib": float(
                diagnostic_topk_only_pnp_ms
            ),
        },
    }


def _distribution(values: Sequence[float]) -> dict:
    if not values:
        return {}
    tensor = torch.tensor(list(values), dtype=torch.float64)
    quantiles = torch.quantile(
        tensor, torch.tensor([0.5, 0.9, 0.95], dtype=torch.float64)
    )
    return {
        "mean": float(tensor.mean()),
        "median": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "max": float(tensor.max()),
    }


def summarize(records: Sequence[Mapping]) -> dict:
    count = len(records)
    baseline_success = sum(bool(value["baseline"]["r5_success"]) for value in records)
    candidate_success = sum(bool(value["candidate"]["r5_success"]) for value in records)
    topk_success = sum(
        bool(value["topk_only_candidate"]["r5_success"]) for value in records
    )

    def errors(key: str, field: str) -> dict:
        return _distribution([float(value[key][field]) for value in records])

    triggered = [
        value for value in records if bool(value["eligible_by_baseline_inlier_band"])
    ]
    locally_supported = [
        value
        for value in records
        if int(value["lgcv_supported_changed_query_rows"].numel()) > 0
    ]
    second_pose = [value for value in records if bool(value["query_lgcv_gate_passed"])]
    return {
        "query_count": count,
        "eligible_query_count": len(triggered),
        "query_with_provisional_change_count": sum(
            int(value["provisional_changed_query_rows"].numel()) > 0
            for value in records
        ),
        "query_with_lgcv_supported_change_count": len(locally_supported),
        "query_passing_lgcv_bundle_gate_count": sum(
            bool(value["query_lgcv_gate_passed"]) for value in records
        ),
        "accepted_query_count": sum(bool(value["candidate_accepted"]) for value in records),
        "topk_only_accepted_query_count": sum(
            bool(value["topk_only_candidate_accepted"]) for value in records
        ),
        "provisional_changed_row_count_total": sum(
            int(value["provisional_changed_query_rows"].numel()) for value in records
        ),
        "lgcv_supported_changed_row_count_total": sum(
            int(value["lgcv_supported_changed_query_rows"].numel()) for value in records
        ),
        "lgcv_rejected_changed_row_count_total": sum(
            int(value["lgcv_rejected_changed_query_rows"].numel()) for value in records
        ),
        "baseline_r5_success_count": baseline_success,
        "candidate_r5_success_count": candidate_success,
        "paired_r5_gain_count": sum(bool(value["r5_gain"]) for value in records),
        "paired_r5_loss_count": sum(bool(value["r5_loss"]) for value in records),
        "paired_r5_net_count": candidate_success - baseline_success,
        "topk_only_r5_success_count": topk_success,
        "topk_only_paired_r5_gain_count": sum(
            bool(value["topk_only_r5_gain"]) for value in records
        ),
        "topk_only_paired_r5_loss_count": sum(
            bool(value["topk_only_r5_loss"]) for value in records
        ),
        "catastrophe_count": sum(bool(value["catastrophe"]) for value in records),
        "continuous_pose_metrics": {
            "translation_error_cm": {
                "baseline": errors("baseline", "translation_error_cm"),
                "candidate": errors("candidate", "translation_error_cm"),
                "topk_only_candidate": errors(
                    "topk_only_candidate", "translation_error_cm"
                ),
                "paired_delta": _distribution(
                    [float(value["paired_delta_translation_error_cm"]) for value in records]
                ),
            },
            "rotation_error_deg": {
                "baseline": errors("baseline", "rotation_error_deg"),
                "candidate": errors("candidate", "rotation_error_deg"),
                "topk_only_candidate": errors(
                    "topk_only_candidate", "rotation_error_deg"
                ),
                "paired_delta": _distribution(
                    [float(value["paired_delta_rotation_error_deg"]) for value in records]
                ),
            },
        },
        "online_latency_ms": {
            "triggered_topk_and_sparse_lgcv": _distribution(
                [
                    float(value["online_timing_ms"]["topk_and_sparse_lgcv"])
                    for value in triggered
                ]
            ),
            "triggered_total_after_first_pose": _distribution(
                [
                    float(value["online_timing_ms"]["total_feedback_after_first_pose"])
                    for value in triggered
                ]
            ),
            "second_poselib_when_run": _distribution(
                [
                    float(value["online_timing_ms"]["second_poselib"])
                    for value in second_pose
                ]
            ),
            "amortized_total_after_first_pose": float(
                sum(
                    float(value["online_timing_ms"]["total_feedback_after_first_pose"])
                    for value in records
                )
                / count
            )
            if count
            else 0.0,
        },
    }


def build_evaluation(
    *,
    stable_map: Mapping,
    cache_payloads: Sequence[Mapping],
    stable_map_source: Mapping,
    cache_sources: Sequence[Mapping],
    producer_sources: Sequence[Mapping],
    device: str | torch.device = "cpu",
    matcher_chunk_size: int = 8192,
) -> dict:
    ordered, records, baseline_contract = validate_complete_cache_payloads(cache_payloads)
    role = str(ordered[0]["role"])
    if role not in {"adaptation", "control", "confirmation"}:
        raise ValueError("V22 sparse LGCV evaluation role is unsupported")
    stable_source = _source(stable_map_source, label="stable map")
    sources = [_source(value, label="frontend cache") for value in cache_sources]
    if len(sources) != len(cache_payloads):
        raise ValueError("V22 sparse LGCV cache source registry differs")
    for payload in cache_payloads:
        declared = _source(payload["inputs"]["stable_map"], label="cache stable map")
        if (declared["path"], declared["sha256"]) != (
            stable_source["path"],
            stable_source["sha256"],
        ):
            raise ValueError("V22 sparse LGCV cache/stable-map lineage differs")
    features = torch.as_tensor(stable_map.get("anchor_features")).float()
    xyz = torch.as_tensor(stable_map.get("anchor_xyz")).float().cpu()
    if features.ndim != 2 or xyz.shape != (features.shape[0], 3):
        raise ValueError("V22 sparse LGCV stable map is invalid")
    bank = F.normalize(features.to(device), dim=1)
    cfg = default_config()
    evaluated = [
        evaluate_record(
            record=record,
            normalized_anchor_features=bank,
            anchor_xyz=xyz,
            baseline_contract=baseline_contract,
            config=cfg,
            device=device,
            matcher_chunk_size=matcher_chunk_size,
        )
        for record in records
    ]
    output = {
        "schema": SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted",
        "uses_test_queries": True,
        "test_adapted": True,
        "evaluation_role": role,
        "configuration_origin": "ULF-Loc Cambridge LGCV constants plus frozen V21 Top-K geometry",
        "configuration_frozen_before_control": True,
        "control_outcomes_used_to_form_configuration": False,
        "confirmation_outcomes_used_to_form_configuration": False,
        "confirmation_consumed": role == "confirmation",
        "writes_map_or_metric": False,
        "controller_authorized": False,
        "deployment_authorized": False,
        "uses_rendering": False,
        "uses_dense_matching": False,
        "uses_ground_truth_for_selection_or_acceptance": False,
        "uses_ground_truth_for_pose_metrics_only": True,
        "first_pass_inliers_are_bit_exact_protected": True,
        "provisional_candidates_cannot_support_each_other": True,
        "lgcv_is_a_query_bundle_gate_not_a_row_filter": True,
        "online_pose_count": "one existing first pass plus at most one feedback PoseLib pass",
        "diagnostic_topk_only_replay_excluded_from_online_latency": True,
        "timing_contract": {
            "artifact_measurement": "standalone_cache_replay_recomputes_exact_top64",
            "deployment_path": "SparseLocalizer retains exact top64 during first matching pass",
            "artifact_timing_is_not_the_integrated_deployment_benchmark": True,
        },
        "configuration": cfg,
        "baseline_contract": baseline_contract,
        "matcher_chunk_size": int(matcher_chunk_size),
        "inputs": {
            "stable_map": stable_source,
            "frontend_caches": sources,
            "producer_sources": [
                _source(value, label="producer") for value in producer_sources
            ],
            "split_manifest": dict(ordered[0]["inputs"]["split_manifest"]),
        },
        "records": evaluated,
        "summary": summarize(evaluated),
    }
    validate_evaluation(output)
    return output


def validate_evaluation(payload: Mapping) -> None:
    records = payload.get("records")
    if not (
        payload.get("schema") == SCHEMA
        and payload.get("version") == VERSION
        and payload.get("protocol") == "test_adapted"
        and payload.get("evaluation_role") in {"adaptation", "control", "confirmation"}
        and payload.get("uses_rendering") is False
        and payload.get("uses_dense_matching") is False
        and payload.get("uses_ground_truth_for_selection_or_acceptance") is False
        and payload.get("writes_map_or_metric") is False
        and payload.get("controller_authorized") is False
        and payload.get("deployment_authorized") is False
        and payload.get("first_pass_inliers_are_bit_exact_protected") is True
        and payload.get("provisional_candidates_cannot_support_each_other") is True
        and payload.get("lgcv_is_a_query_bundle_gate_not_a_row_filter") is True
        and payload.get("timing_contract", {}).get(
            "artifact_timing_is_not_the_integrated_deployment_benchmark"
        )
        is True
        and isinstance(records, list)
        and records
    ):
        raise ValueError("unsupported V22 sparse LGCV evaluation")
    cfg = validate_config(payload.get("configuration", {}))
    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("V22 sparse LGCV input lineage is missing")
    _source(inputs.get("stable_map"), label="stable map")
    caches = inputs.get("frontend_caches")
    producers = inputs.get("producer_sources")
    if not isinstance(caches, list) or not caches or not isinstance(producers, list) or not producers:
        raise ValueError("V22 sparse LGCV source registries are empty")
    [_source(value, label="frontend cache") for value in caches]
    [_source(value, label="producer") for value in producers]
    seen = set()
    for record in records:
        query = int(record.get("query_index", -1))
        proposed = torch.as_tensor(record.get("provisional_changed_query_rows")).long().reshape(-1)
        ranks = torch.as_tensor(record.get("provisional_selected_candidate_ranks")).long().reshape(-1)
        residual = torch.as_tensor(
            record.get("provisional_selected_reprojection_residual_px")
        ).float().reshape(-1)
        supported = torch.as_tensor(record.get("lgcv_supported_changed_query_rows")).long().reshape(-1)
        rejected = torch.as_tensor(record.get("lgcv_rejected_changed_query_rows")).long().reshape(-1)
        support = torch.as_tensor(record.get("lgcv_proposal_support_scores")).float().reshape(-1)
        triangles = torch.as_tensor(
            record.get("lgcv_proposal_valid_directed_triangle_counts")
        ).long().reshape(-1)
        baseline = record.get("baseline")
        candidate = record.get("candidate")
        topk_candidate = record.get("topk_only_candidate")
        timing = record.get("online_timing_ms")
        if not all(
            isinstance(value, Mapping)
            for value in (baseline, candidate, topk_candidate, timing)
        ):
            raise ValueError("V22 sparse LGCV outcomes are missing")
        baseline_inliers = torch.as_tensor(baseline.get("inlier_query_rows")).long().reshape(-1)
        eligible = (
            int(cfg["topk_geometry"]["minimum_baseline_inlier_count_inclusive"])
            <= int(baseline.get("inlier_count", -1))
            < int(cfg["topk_geometry"]["maximum_baseline_inlier_count_exclusive"])
        )
        accepted = bool(record.get("candidate_accepted"))
        topk_accepted = bool(record.get("topk_only_candidate_accepted"))
        expected_fraction = float(supported.numel() / max(int(proposed.numel()), 1))
        expected_gate = bool(
            supported.numel() >= int(cfg["minimum_query_supported_proposal_count"])
            and expected_fraction
            >= float(cfg["minimum_query_supported_proposal_fraction"])
        )
        if (
            query < 0
            or query in seen
            or proposed.shape != ranks.shape
            or ranks.shape != residual.shape
            or support.shape != proposed.shape
            or triangles.shape != proposed.shape
            or torch.unique(proposed).numel() != proposed.numel()
            or not torch.equal(torch.sort(torch.cat((supported, rejected))).values, torch.sort(proposed).values)
            or bool(torch.isin(supported, rejected).any())
            or bool(torch.isin(proposed, baseline_inliers).any())
            or (proposed.numel() and (int(ranks.min()) < 1 or int(ranks.max()) > int(cfg["topk_geometry"]["topk"])))
            or not bool(torch.isfinite(residual).all())
            or not bool(torch.isfinite(support).all())
            or bool(
                (
                    support[torch.isin(proposed, supported)]
                    <= float(cfg["lgcv_support_threshold_exclusive"])
                ).any()
            )
            or bool(
                (
                    support[torch.isin(proposed, rejected)]
                    > float(cfg["lgcv_support_threshold_exclusive"])
                ).any()
            )
            or bool(record.get("eligible_by_baseline_inlier_band")) != eligible
            or bool(record.get("query_lgcv_gate_passed")) != expected_gate
            or not math.isclose(
                float(record.get("query_lgcv_supported_proposal_fraction", math.nan)),
                expected_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or (not eligible and (proposed.numel() or accepted or topk_accepted))
            or (accepted and not expected_gate)
            or (topk_accepted and not proposed.numel())
            or (
                accepted
                and int(candidate.get("inlier_count", -1))
                < int(baseline.get("inlier_count", -1))
                + int(cfg["topk_geometry"]["minimum_candidate_inlier_gain"])
            )
            or (
                topk_accepted
                and int(topk_candidate.get("inlier_count", -1))
                < int(baseline.get("inlier_count", -1))
                + int(cfg["topk_geometry"]["minimum_candidate_inlier_gain"])
            )
            or any(not math.isfinite(float(value)) or float(value) < 0 for value in timing.values())
        ):
            raise ValueError("V22 sparse LGCV record is invalid")
        gain = bool(not baseline.get("r5_success") and candidate.get("r5_success"))
        loss = bool(baseline.get("r5_success") and not candidate.get("r5_success"))
        if (
            bool(record.get("r5_gain")) != gain
            or bool(record.get("r5_loss")) != loss
            or bool(record.get("catastrophe")) != loss
        ):
            raise ValueError("V22 sparse LGCV R5 flags differ")
        seen.add(query)
    if payload.get("summary") != summarize(records):
        raise ValueError("V22 sparse LGCV summary differs")


def atomic_torch_save_fresh(payload: Mapping, output: str | Path) -> Path:
    path = Path(output).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"V22 sparse LGCV output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(dict(payload), temporary)
        validate_evaluation(torch.load(temporary, map_location="cpu", weights_only=False))
        os.link(temporary, path)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return path


def finalize_evaluations(
    evaluations: Sequence[Mapping], evaluation_sources: Sequence[Mapping]
) -> dict:
    """Apply the continuous-pose primary gate after one confirmation replay."""

    if len(evaluations) != 3 or len(evaluation_sources) != 3:
        raise ValueError("V22 finalizer requires adaptation/control/confirmation")
    by_role = {}
    sources = {}
    for payload, source in zip(evaluations, evaluation_sources):
        validate_evaluation(payload)
        role = str(payload["evaluation_role"])
        if role in by_role:
            raise ValueError("V22 finalizer role is duplicated")
        by_role[role] = payload
        sources[role] = _source(source, label=f"{role} evaluation")
    if set(by_role) != {"adaptation", "control", "confirmation"}:
        raise ValueError("V22 finalizer role coverage differs")
    first = by_role["adaptation"]
    stable_identity = (
        first["inputs"]["stable_map"]["path"],
        first["inputs"]["stable_map"]["sha256"],
    )
    split_identity = (
        first["inputs"]["split_manifest"]["path"],
        first["inputs"]["split_manifest"]["sha256"],
    )
    seen_queries = set()
    for payload in by_role.values():
        if (
            payload["configuration"] != first["configuration"]
            or payload["baseline_contract"] != first["baseline_contract"]
            or (
                payload["inputs"]["stable_map"]["path"],
                payload["inputs"]["stable_map"]["sha256"],
            )
            != stable_identity
            or (
                payload["inputs"]["split_manifest"]["path"],
                payload["inputs"]["split_manifest"]["sha256"],
            )
            != split_identity
        ):
            raise ValueError("V22 finalizer protocol lineage differs")
        queries = {int(record["query_index"]) for record in payload["records"]}
        if seen_queries & queries:
            raise ValueError("V22 finalizer query registries overlap")
        seen_queries |= queries

    phase_gates = {}
    for role in ("adaptation", "control", "confirmation"):
        summary = by_role[role]["summary"]
        metrics = summary["continuous_pose_metrics"]
        te = metrics["translation_error_cm"]
        re = metrics["rotation_error_deg"]
        median_te_delta = float(
            te["candidate"]["median"] - te["baseline"]["median"]
        )
        median_re_delta = float(
            re["candidate"]["median"] - re["baseline"]["median"]
        )
        p90_te_delta = float(te["candidate"]["p90"] - te["baseline"]["p90"])
        p90_re_delta = float(re["candidate"]["p90"] - re["baseline"]["p90"])
        no_r5_loss = int(summary["paired_r5_loss_count"]) == 0
        tail_safe = p90_te_delta <= 1e-9 and p90_re_delta <= 1e-9
        median_safe = median_te_delta <= 1e-9 and median_re_delta <= 0.005
        primary_improvement = median_te_delta < -1e-9 or median_re_delta < -1e-9
        phase_gates[role] = {
            "median_translation_delta_cm": median_te_delta,
            "median_rotation_delta_deg": median_re_delta,
            "p90_translation_delta_cm": p90_te_delta,
            "p90_rotation_delta_deg": p90_re_delta,
            "paired_r5_gain_count": int(summary["paired_r5_gain_count"]),
            "paired_r5_loss_count": int(summary["paired_r5_loss_count"]),
            "no_r5_loss": no_r5_loss,
            "tail_not_worse": tail_safe,
            "median_not_materially_worse": median_safe,
            "continuous_primary_improvement": primary_improvement,
            "passed": bool(
                no_r5_loss and tail_safe and median_safe and primary_improvement
            ),
        }

    records = [
        record
        for role in ("adaptation", "control", "confirmation")
        for record in by_role[role]["records"]
    ]
    combined = summarize(records)
    combined_metrics = combined["continuous_pose_metrics"]
    combined_median_te_delta = float(
        combined_metrics["translation_error_cm"]["candidate"]["median"]
        - combined_metrics["translation_error_cm"]["baseline"]["median"]
    )
    combined_median_re_delta = float(
        combined_metrics["rotation_error_deg"]["candidate"]["median"]
        - combined_metrics["rotation_error_deg"]["baseline"]["median"]
    )
    combined_primary_passed = bool(
        combined_median_te_delta < -1e-9 and combined_median_re_delta < -1e-9
    )
    deployment = bool(
        all(value["passed"] for value in phase_gates.values())
        and combined_primary_passed
    )
    return {
        "schema": FINAL_SCHEMA,
        "version": VERSION,
        "protocol": "test_adapted_forward_adaptation_control_confirmation",
        "uses_test_queries": True,
        "test_adapted": True,
        "configuration_frozen_before_control": True,
        "confirmation_consumed_once": True,
        "continuous_pose_metrics_are_primary": True,
        "r5_is_a_safety_and_auxiliary_metric": True,
        "decision": (
            "GO_DEPLOYMENT_CANDIDATE"
            if deployment
            else "STOP_CONFIRMATION_CONTINUOUS_GAIN_NOT_REPRODUCED"
        ),
        "deployment_authorized": deployment,
        "controller_authorized": deployment,
        "writes_map_or_metric": False,
        "inputs": {"evaluations": sources},
        "configuration": first["configuration"],
        "phase_gates": phase_gates,
        "combined_summary": combined,
        "combined_median_translation_delta_cm": combined_median_te_delta,
        "combined_median_rotation_delta_deg": combined_median_re_delta,
        "combined_primary_passed": combined_primary_passed,
        "excluded_embargo_queries_are_not_scored": True,
        "reason": (
            "all forward continuous-pose gates passed"
            if deployment
            else "confirmation had no strict median TE/RE improvement"
        ),
    }


def atomic_torch_save_final_fresh(payload: Mapping, output: str | Path) -> Path:
    if payload.get("schema") != FINAL_SCHEMA or payload.get("version") != VERSION:
        raise ValueError("unsupported V22 final decision")
    path = Path(output).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"V22 final output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        torch.save(dict(payload), temporary)
        loaded = torch.load(temporary, map_location="cpu", weights_only=False)
        if (
            loaded.get("schema") != FINAL_SCHEMA
            or loaded.get("version") != VERSION
            or loaded.get("decision") != payload.get("decision")
            or loaded.get("deployment_authorized")
            != payload.get("deployment_authorized")
            or loaded.get("configuration") != payload.get("configuration")
        ):
            raise ValueError("V22 final decision reload differs")
        os.link(temporary, path)
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return path
