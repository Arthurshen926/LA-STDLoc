#!/usr/bin/env python
"""Evaluate exact hard-frontend oracles from a discrete decision dump.

The script is intentionally CPU-only. It replays the same top-1, landmark quota,
score rejection, and PnP path used by STDLoc before measuring O1--O6.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization_training.pose_information import pose_jacobian_analytic
from utils.pose_utils import cal_pose_error, solve_pose


@dataclass
class CandidateSet:
    keypoint_idx: np.ndarray
    landmark_idx: np.ndarray
    scores: np.ndarray
    source_idx: np.ndarray

    def subset(self, keep):
        keep = np.asarray(keep)
        return CandidateSet(
            self.keypoint_idx[keep],
            self.landmark_idx[keep],
            self.scores[keep],
            self.source_idx[keep],
        )


def project_points(points, K, pose_w2c):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    pose_w2c = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera = (pose_w2c @ points_h.T)[:3].T
    depth = camera[:, 2]
    valid = np.isfinite(camera).all(axis=1) & (depth > 1e-8)
    uv = np.full((points.shape[0], 2), np.nan, dtype=np.float64)
    uv[valid, 0] = K[0, 0] * camera[valid, 0] / depth[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * camera[valid, 1] / depth[valid] + K[1, 2]
    return uv, depth, valid


def _group_limit_mask(groups, scores, limit, priority=None):
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    limit = int(limit)
    if limit <= 0:
        return np.ones(groups.shape[0], dtype=bool)
    priority = (
        np.zeros(groups.shape[0], dtype=np.int64)
        if priority is None
        else np.asarray(priority, dtype=np.int64).reshape(-1)
    )
    keep = np.zeros(groups.shape[0], dtype=bool)
    positions = np.arange(groups.shape[0], dtype=np.int64)
    for group in np.unique(groups):
        idx = positions[groups == group]
        order = np.lexsort((positions[idx], -scores[idx], -priority[idx]))
        keep[idx[order[:limit]]] = True
    return keep


def select_candidates(
    keypoint_idx,
    landmark_idx,
    scores,
    *,
    threshold,
    max_matches_per_keypoint=0,
    max_matches_per_landmark=0,
    min_match_count=0,
    refill_trigger_count=0,
    correctness_priority=None,
):
    """Numpy replay of ``select_match_candidates`` with optional GT hardcap."""
    candidates = CandidateSet(
        np.asarray(keypoint_idx, dtype=np.int64).reshape(-1),
        np.asarray(landmark_idx, dtype=np.int64).reshape(-1),
        np.asarray(scores, dtype=np.float64).reshape(-1),
        np.arange(np.asarray(scores).size, dtype=np.int64),
    )
    priority = (
        None
        if correctness_priority is None
        else np.asarray(correctness_priority, dtype=np.int64).reshape(-1)
    )
    keep = _group_limit_mask(
        candidates.keypoint_idx,
        candidates.scores,
        max_matches_per_keypoint,
    )
    candidates = candidates.subset(keep)
    if priority is not None:
        priority = priority[keep]
    keep = _group_limit_mask(
        candidates.landmark_idx,
        candidates.scores,
        max_matches_per_landmark,
        priority=priority,
    )
    candidates = candidates.subset(keep)
    source_after_limits = np.flatnonzero(
        _group_limit_mask(
            np.asarray(keypoint_idx, dtype=np.int64).reshape(-1),
            np.asarray(scores, dtype=np.float64).reshape(-1),
            max_matches_per_keypoint,
        )
    )[keep]
    candidates.source_idx = source_after_limits

    accepted = candidates.scores > float(threshold)
    target = min(max(int(min_match_count), 0), candidates.scores.size)
    accepted_count = int(accepted.sum())
    trigger = max(int(refill_trigger_count), 0) or target
    if accepted_count < trigger and accepted_count < target:
        order = np.argsort(-candidates.scores, kind="stable")[:target]
        accepted[order] = True
    return candidates.subset(accepted)


def pair_is_correct(keypoint_xy, landmark_idx, projected, valid, radius):
    keypoint_xy = np.asarray(keypoint_xy, dtype=np.float64).reshape(-1, 2)
    landmark_idx = np.asarray(landmark_idx, dtype=np.int64).reshape(-1)
    distance = np.linalg.norm(keypoint_xy - projected[landmark_idx], axis=1)
    correct = valid[landmark_idx] & np.isfinite(distance) & (distance <= radius)
    return correct, distance


def nearest_gt_targets(keypoint_xy, projected, valid, radius):
    valid_ids = np.flatnonzero(valid)
    target = np.full(len(keypoint_xy), -1, dtype=np.int64)
    distance = np.full(len(keypoint_xy), np.inf, dtype=np.float64)
    if valid_ids.size == 0:
        return target, distance
    tree = cKDTree(projected[valid_ids])
    distance, local_idx = tree.query(np.asarray(keypoint_xy, dtype=np.float64), k=1)
    matchable = np.isfinite(distance) & (distance <= float(radius))
    target[matchable] = valid_ids[np.asarray(local_idx)[matchable]]
    return target, np.asarray(distance, dtype=np.float64)


def provenance_gt_targets(
    keypoint_xy,
    projected,
    projection_valid,
    provenance_sources,
    provenance_weights,
    provenance_valid,
    source_to_landmarks,
    radius,
    *,
    allowed_landmarks=None,
):
    """Find the nearest reprojection-valid anchor in a pixel's splat families."""
    keypoint_xy = np.asarray(keypoint_xy, dtype=np.float64).reshape(-1, 2)
    provenance_sources = np.asarray(provenance_sources, dtype=np.int64)
    provenance_weights = np.asarray(provenance_weights, dtype=np.float64)
    provenance_valid = np.asarray(provenance_valid, dtype=bool).reshape(-1)
    allowed = (
        np.ones(len(projected), dtype=bool)
        if allowed_landmarks is None
        else np.asarray(allowed_landmarks, dtype=bool).reshape(-1)
    )
    target = np.full(len(keypoint_xy), -1, dtype=np.int64)
    distance = np.full(len(keypoint_xy), np.inf, dtype=np.float64)
    for row in np.flatnonzero(provenance_valid):
        candidates = []
        for source, weight in zip(
            provenance_sources[row], provenance_weights[row]
        ):
            if weight <= 0.0:
                continue
            candidates.extend(source_to_landmarks.get(int(source), ()))
        if not candidates:
            continue
        candidates = np.unique(np.asarray(candidates, dtype=np.int64))
        candidates = candidates[
            projection_valid[candidates] & allowed[candidates]
        ]
        if candidates.size == 0:
            continue
        residual = np.linalg.norm(
            projected[candidates] - keypoint_xy[row], axis=1
        )
        best = int(np.argmin(residual))
        if np.isfinite(residual[best]) and residual[best] <= float(radius):
            target[row] = int(candidates[best])
            distance[row] = float(residual[best])
    return target, distance


def oracle_assignment_candidates(raw_rows, targets, scores):
    """Construct the attachment's O1 oracle without changing native 2D points.

    Each matchable native keypoint receives the nearest visible landmark under
    the ground-truth pose.  Rows without a valid nearby landmark are excluded:
    assigning an arbitrary 3D point to them would turn an assignment oracle
    into an artificial outlier-contamination experiment.
    """
    raw_rows = np.asarray(raw_rows, dtype=np.int64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not (raw_rows.shape == targets.shape == scores.shape):
        raise ValueError("oracle assignment inputs must have equal lengths")
    keep = targets >= 0
    return CandidateSet(
        keypoint_idx=raw_rows[keep],
        landmark_idx=targets[keep],
        scores=scores[keep],
        source_idx=raw_rows[keep],
    )


def oracle_topk_candidates(topk_landmark_idx, topk_scores, candidate_correct):
    """Select at most one GT-valid hypothesis per native keypoint.

    This is the upper bound for the proposed one-of-K reranker. It never
    duplicates a 2D measurement and emits null for a row whose retrieved
    hypotheses contain no geometrically valid landmark.
    """
    topk_landmark_idx = np.asarray(topk_landmark_idx, dtype=np.int64)
    topk_scores = np.asarray(topk_scores, dtype=np.float64)
    candidate_correct = np.asarray(candidate_correct, dtype=bool)
    if not (
        topk_landmark_idx.ndim == 2
        and topk_landmark_idx.shape == topk_scores.shape
        and topk_landmark_idx.shape == candidate_correct.shape
    ):
        raise ValueError("top-K oracle arrays must have identical NxK shapes")
    has_positive = candidate_correct.any(axis=1)
    rows = np.flatnonzero(has_positive)
    # Retrieval is score sorted. argmax on the boolean mask therefore returns
    # the highest-ranked valid hypothesis and preserves deployment ordering.
    positions = candidate_correct[rows].argmax(axis=1)
    return CandidateSet(
        keypoint_idx=rows,
        landmark_idx=topk_landmark_idx[rows, positions],
        scores=topk_scores[rows, positions],
        source_idx=rows,
    )


def run_pose(candidates, keypoint_xy, landmark_xyz, K, query, seed):
    cv2.setRNGSeed(int(seed))
    p2d = np.asarray(keypoint_xy, dtype=np.float64)[candidates.keypoint_idx]
    p3d = np.asarray(landmark_xyz, dtype=np.float64)[candidates.landmark_idx]
    pose, inliers = solve_pose(
        p2d,
        p3d,
        np.asarray(K, dtype=np.float64),
        str(query["solver"].item()),
        float(query["reprojection_error"].item()),
        float(query["confidence"].item()),
        int(query["max_iterations"].item()),
        int(query["min_iterations"].item()),
    )
    return np.asarray(pose, dtype=np.float64), np.asarray(inliers).reshape(-1)


def deterministic_pnp(p2d, p3d, K):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if p2d.shape[0] < 4:
        return np.eye(4, dtype=np.float64), False
    success, rvec, tvec = cv2.solvePnP(
        p3d,
        p2d,
        K,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success:
        return np.eye(4, dtype=np.float64), False
    success, rvec, tvec = cv2.solvePnP(
        p3d,
        p2d,
        K,
        np.zeros((4, 1), dtype=np.float64),
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    pose = np.eye(4, dtype=np.float64)
    if success:
        pose[:3, :3] = cv2.Rodrigues(rvec)[0]
        pose[:3, 3] = tvec.reshape(3)
    return pose, bool(success)


def pose_error(pose, gt_pose):
    ae, te = cal_pose_error(
        np.asarray(pose, dtype=np.float64), np.asarray(gt_pose, dtype=np.float64)
    )
    return float(ae), float(te)


def _task_pose_terms(
    p2d,
    p3d,
    K,
    gt_pose,
    translation_scale_m,
    rotation_scale_degrees,
):
    p2d = np.asarray(p2d, dtype=np.float64).reshape(-1, 2)
    p3d = np.asarray(p3d, dtype=np.float64).reshape(-1, 3)
    projected, _, valid = project_points(p3d, K, gt_pose)
    residual = p2d - projected
    residual_norm = np.linalg.norm(residual, axis=1)
    finite = valid & np.isfinite(residual).all(axis=1) & np.isfinite(residual_norm)
    residual[~finite] = 0.0
    residual_norm[~finite] = np.inf
    residual_scale = np.minimum(1.0, 12.0 / np.maximum(residual_norm, 1e-12))
    residual = residual * residual_scale[:, None]
    weight = np.exp(-np.square(residual_norm) / (2.0 * 4.0**2))
    weight[~finite] = 0.0
    jacobian = pose_jacobian_analytic(
        torch.as_tensor(p3d, dtype=torch.float64),
        torch.as_tensor(K, dtype=torch.float64),
        torch.as_tensor(gt_pose, dtype=torch.float64),
    ).numpy()
    scales = np.asarray(
        [
            translation_scale_m,
            translation_scale_m,
            translation_scale_m,
            math.radians(rotation_scale_degrees),
            math.radians(rotation_scale_degrees),
            math.radians(rotation_scale_degrees),
        ],
        dtype=np.float64,
    )
    jacobian = jacobian * scales[None, None, :]
    information = weight[:, None, None] * np.einsum(
        "nai,naj->nij", jacobian, jacobian
    )
    gradient = weight[:, None] * np.einsum("nai,na->ni", jacobian, residual)
    return information, gradient


def _pose_term_metrics(information, gradient, translation_scale_m):
    information = np.asarray(information, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    information = 0.5 * (information + information.T)
    try:
        delta = -np.linalg.solve(information, gradient)
    except np.linalg.LinAlgError:
        delta = -np.linalg.pinv(information) @ gradient
    h_tt = information[:3, :3]
    h_tr = information[:3, 3:]
    h_rr = information[3:, 3:]
    translation = h_tt - h_tr @ np.linalg.pinv(h_rr) @ h_tr.T
    translation = 0.5 * (translation + translation.T)
    eig = np.linalg.eigvalsh(translation).clip(1e-12, None)
    return {
        "bias_m": float(np.linalg.norm(delta[:3]) * translation_scale_m),
        "translation_logdet": float(np.log(eig).sum()),
    }


def set_bias_metrics(
    candidates,
    keypoint_xy,
    landmark_xyz,
    K,
    gt_pose,
    translation_scale_m,
    rotation_scale_degrees,
):
    p2d = np.asarray(keypoint_xy)[candidates.keypoint_idx]
    p3d = np.asarray(landmark_xyz)[candidates.landmark_idx]
    information, gradient = _task_pose_terms(
        p2d,
        p3d,
        K,
        gt_pose,
        translation_scale_m,
        rotation_scale_degrees,
    )
    H = np.eye(6, dtype=np.float64) * 1e-4 + information.sum(axis=0)
    g = gradient.sum(axis=0)
    return _pose_term_metrics(H, g, translation_scale_m)


def _selected_group_sources(raw_lm, scores, landmark, quota, exclude=-1, add=-1):
    source = np.flatnonzero(np.asarray(raw_lm) == int(landmark)).tolist()
    if exclude >= 0:
        source = [idx for idx in source if idx != int(exclude)]
    if add >= 0 and int(add) not in source:
        source.append(int(add))
    source.sort(key=lambda idx: (-float(scores[idx]), int(idx)))
    return source if int(quota) <= 0 else source[: int(quota)]


def counterfactual_gain_distribution(
    raw_rows,
    raw_lm,
    raw_scores,
    selected,
    target_lm,
    keypoint_xy,
    landmark_xyz,
    K,
    gt_pose,
    threshold,
    landmark_quota,
    translation_scale_m,
    rotation_scale_degrees,
):
    """Strict one-row swap gains, including old-group refill and displacement."""
    raw_rows = np.asarray(raw_rows, dtype=np.int64)
    raw_lm = np.asarray(raw_lm, dtype=np.int64)
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    selected_sources = set(np.asarray(selected.source_idx, dtype=np.int64).tolist())
    selected_by_source = {
        int(src): int(lm)
        for src, lm in zip(selected.source_idx, selected.landmark_idx)
    }
    base_p2d = np.asarray(keypoint_xy)[selected.keypoint_idx]
    base_p3d = np.asarray(landmark_xyz)[selected.landmark_idx]
    base_info, base_grad = _task_pose_terms(
        base_p2d,
        base_p3d,
        K,
        gt_pose,
        translation_scale_m,
        rotation_scale_degrees,
    )
    base_H = np.eye(6, dtype=np.float64) * 1e-4 + base_info.sum(axis=0)
    base_g = base_grad.sum(axis=0)
    base_metrics = _pose_term_metrics(base_H, base_g, translation_scale_m)
    base_terms = {
        int(src): (base_info[idx], base_grad[idx])
        for idx, src in enumerate(selected.source_idx)
    }

    raw_p2d = np.asarray(keypoint_xy)[raw_rows]
    raw_p3d = np.asarray(landmark_xyz)[raw_lm]
    raw_info, raw_grad = _task_pose_terms(
        raw_p2d,
        raw_p3d,
        K,
        gt_pose,
        translation_scale_m,
        rotation_scale_degrees,
    )
    gt_valid = np.asarray(target_lm) >= 0
    gt_info = np.zeros_like(raw_info)
    gt_grad = np.zeros_like(raw_grad)
    if gt_valid.any():
        info, grad = _task_pose_terms(
            raw_p2d[gt_valid],
            np.asarray(landmark_xyz)[np.asarray(target_lm)[gt_valid]],
            K,
            gt_pose,
            translation_scale_m,
            rotation_scale_degrees,
        )
        gt_info[gt_valid] = info
        gt_grad[gt_valid] = grad

    records = []
    for source in sorted(selected_sources):
        row = int(raw_rows[source])
        old_lm = int(raw_lm[source])
        new_lm = int(target_lm[row])
        if new_lm < 0 or new_lm == old_lm:
            continue
        old_sources = _selected_group_sources(
            raw_lm, raw_scores, old_lm, landmark_quota, exclude=source
        )
        new_sources = _selected_group_sources(
            raw_lm, raw_scores, new_lm, landmark_quota, add=source
        )
        if source not in new_sources or raw_scores[source] <= float(threshold):
            continue
        old_sources = [idx for idx in old_sources if raw_scores[idx] > threshold]
        new_sources = [idx for idx in new_sources if raw_scores[idx] > threshold]
        current_local = {
            src: lm
            for src, lm in selected_by_source.items()
            if lm in {old_lm, new_lm}
        }
        candidate_local = {idx: old_lm for idx in old_sources}
        candidate_local.update(
            {idx: (new_lm if idx == source else int(raw_lm[idx])) for idx in new_sources}
        )

        H = base_H.copy()
        g = base_g.copy()
        for src in current_local:
            info, grad = base_terms[src]
            H -= info
            g -= grad
        for src, lm in candidate_local.items():
            if src == source and lm == new_lm:
                H += gt_info[row]
                g += gt_grad[row]
            else:
                H += raw_info[src]
                g += raw_grad[src]
        candidate_metrics = _pose_term_metrics(H, g, translation_scale_m)
        bias_gain_m2 = base_metrics["bias_m"] ** 2 - candidate_metrics["bias_m"] ** 2
        translation_gain = (
            candidate_metrics["translation_logdet"]
            - base_metrics["translation_logdet"]
        )
        strict = bool(bias_gain_m2 > 0.0 and translation_gain >= -1e-10)
        utility = (
            bias_gain_m2 / max(base_metrics["bias_m"] ** 2, 1e-12)
            + 0.05 * max(translation_gain, 0.0)
        )
        records.append(
            {
                "source_idx": source,
                "row": row,
                "target_landmark": new_lm,
                "bias_gain_m2": float(bias_gain_m2),
                "translation_logdet_gain": float(translation_gain),
                "strict_positive": strict,
                "utility": float(utility),
            }
        )
    return base_metrics, records


def summarize_pose_errors(ae, te):
    ae = np.asarray(ae, dtype=np.float64)
    te = np.asarray(te, dtype=np.float64)
    return {
        "count": int(te.size),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "recall_2cm_2deg": float(np.mean((te <= 2.0) & (ae <= 2.0))),
        "recall_5cm_5deg": float(np.mean((te <= 5.0) & (ae <= 5.0))),
    }


def paired_summary(base_te, method_te, seed=2026, bootstrap_samples=10000):
    base = np.asarray(base_te, dtype=np.float64)
    method = np.asarray(method_te, dtype=np.float64)
    delta = method - base
    rng = np.random.default_rng(int(seed))
    samples = rng.integers(0, delta.size, size=(int(bootstrap_samples), delta.size))
    mean_boot = delta[samples].mean(axis=1)
    median_boot = np.median(method[samples], axis=1) - np.median(base[samples], axis=1)
    return {
        "mean_delta_cm": float(delta.mean()),
        "median_of_paired_delta_cm": float(np.median(delta)),
        "paired_wins": int((delta < -1e-9).sum()),
        "paired_losses": int((delta > 1e-9).sum()),
        "paired_ties": int((np.abs(delta) <= 1e-9).sum()),
        "win_rate": float(np.mean(delta < -1e-9)),
        "mean_delta_bootstrap_95ci_cm": np.percentile(mean_boot, [2.5, 97.5]).tolist(),
        "median_delta_bootstrap_95ci_cm": np.percentile(
            median_boot, [2.5, 97.5]
        ).tolist(),
    }


def _json_scalar(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def evaluate(
    dump_dir,
    output,
    radius=2.0,
    seed=2026,
    bootstrap_samples=10000,
    translation_scale_m=None,
    rotation_scale_degrees=None,
    skip_counterfactual=False,
    track_payload=None,
):
    dump_dir = Path(dump_dir)
    manifest = json.loads((dump_dir / "manifest.json").read_text())
    translation_scale_m = float(
        manifest.get("task_translation_scale_m", 0.02)
        if translation_scale_m is None
        else translation_scale_m
    )
    rotation_scale_degrees = float(
        manifest.get("task_rotation_scale_degrees", 2.0)
        if rotation_scale_degrees is None
        else rotation_scale_degrees
    )
    if translation_scale_m <= 0.0 or rotation_scale_degrees <= 0.0:
        raise ValueError("task-space pose scales must be positive")
    with np.load(dump_dir / manifest["landmark_bank"]) as bank_file:
        landmark_xyz = np.asarray(bank_file["landmark_xyz"], dtype=np.float64)
        source_gaussian_idx = np.asarray(
            bank_file["source_gaussian_idx"], dtype=np.int64
        )
        anchor_type = np.asarray(
            (
                bank_file["anchor_type"]
                if "anchor_type" in bank_file.files
                else np.zeros(len(landmark_xyz), dtype=np.int64)
            ),
            dtype=np.int64,
        )
        track_cluster_id = np.asarray(
            (
                bank_file["track_cluster_id"]
                if "track_cluster_id" in bank_file.files
                else np.full(len(landmark_xyz), -1, dtype=np.int64)
            ),
            dtype=np.int64,
        )
    geometry_oracle_xyz = landmark_xyz.copy()
    geometry_oracle_replaced = np.zeros(len(landmark_xyz), dtype=bool)
    if track_payload:
        payload = torch.load(
            track_payload, map_location="cpu", weights_only=False
        )
        geometry = payload["track_geometry"]
        track_xyz = torch.as_tensor(
            geometry["triangulated_xyz"]
        ).cpu().numpy()
        track_high_confidence = torch.as_tensor(
            geometry["triangulation_high_confidence"]
        ).cpu().numpy().astype(bool)
        valid_track = (
            (track_cluster_id >= 0)
            & (track_cluster_id < len(track_xyz))
        )
        valid_track[valid_track] &= track_high_confidence[
            track_cluster_id[valid_track]
        ]
        geometry_oracle_xyz[valid_track] = track_xyz[
            track_cluster_id[valid_track]
        ]
        geometry_oracle_replaced = valid_track
    source_to_landmarks = {}
    for landmark, source in enumerate(source_gaussian_idx):
        source_to_landmarks.setdefault(int(source), []).append(int(landmark))

    oracle_assignment_ks = (1, 2, 4, 8, 16)
    methods = (
        "actual",
        "replay",
        "O1_oracle_3d_assignment",
        "OP_oracle_provenance_assignment",
        "OG_oracle_track_geometry",
        "O2_oracle_candidate_filter",
        "O3_oracle_2d_measurement",
        "O2_top1_swap",
        "O3_hardcap",
        "O2O3_swap_hardcap",
        "O4_gt_clean_hard_set",
        "O5_fixed_inliers_base",
        "O5_signed_coordinate",
        "O6_best_single_swap",
        "O6_all_strict_swaps",
    ) + tuple(f"OK{topk}_one_of_k" for topk in oracle_assignment_ks)
    errors = {name: {"ae": [], "te": []} for name in methods}
    retrieval_ks = (1, 2, 4, 8, 16, 32)
    retrieval = {
        2.0: {k: [] for k in retrieval_ks},
        4.0: {k: [] for k in retrieval_ks},
    }
    provenance_coverage = {
        radius: {
            split: {"all": [], "base": [], "micro": []}
            for split in ("all", "seq4", "seq8")
        }
        for radius in (2.0, 4.0, 8.0)
    }
    query_records = []
    hard_precision = []
    inlier_precision = []
    hard_bias = []
    final_bias = []
    clean_hard_bias = []
    quota_displacements = []
    dustbin_stats = []
    cf_positive_queries = 0
    cf_positive_rows = []
    selector_replay_failures = []

    for query_index, query_file in enumerate(manifest["query_files"]):
        with np.load(dump_dir / query_file, allow_pickle=False) as loaded:
            query = {key: np.asarray(loaded[key]) for key in loaded.files}
        image_name = str(query["image_name"].item())
        gt_pose = np.asarray(query["gt_pose_w2c"], dtype=np.float64)
        K = np.asarray(query["K"], dtype=np.float64)
        width, height = int(query["width"]), int(query["height"])
        keypoint_xy = np.asarray(query["keypoint_xy"], dtype=np.float64) + 0.5
        topk_lm = np.asarray(query["topk_landmark_idx"], dtype=np.int64)
        topk_scores = np.asarray(query["topk_scores"], dtype=np.float64)
        visible = np.asarray(query["render_visible_bank"], dtype=bool)
        projected, _, projection_valid = project_points(landmark_xyz, K, gt_pose)
        valid = (
            visible
            & projection_valid
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        has_provenance = all(
            key in query
            for key in (
                "splat_provenance_source_gaussian_idx",
                "splat_provenance_weight",
                "splat_provenance_valid",
            )
        )
        provenance_targets = {}
        if has_provenance:
            provenance_sources = np.asarray(
                query["splat_provenance_source_gaussian_idx"],
                dtype=np.int64,
            )
            provenance_weights = np.asarray(
                query["splat_provenance_weight"], dtype=np.float64
            )
            provenance_valid = np.asarray(
                query["splat_provenance_valid"], dtype=bool
            )
            split = (
                "seq4"
                if image_name.replace("\\", "/").startswith("seq4/")
                else "seq8"
                if image_name.replace("\\", "/").startswith("seq8/")
                else "all"
            )
            for eval_radius in (2.0, 4.0, 8.0):
                target, _ = provenance_gt_targets(
                    keypoint_xy,
                    projected,
                    projection_valid,
                    provenance_sources,
                    provenance_weights,
                    provenance_valid,
                    source_to_landmarks,
                    eval_radius,
                )
                provenance_targets[eval_radius] = target
                target_base, _ = provenance_gt_targets(
                    keypoint_xy,
                    projected,
                    projection_valid,
                    provenance_sources,
                    provenance_weights,
                    provenance_valid,
                    source_to_landmarks,
                    eval_radius,
                    allowed_landmarks=anchor_type == 0,
                )
                target_micro, _ = provenance_gt_targets(
                    keypoint_xy,
                    projected,
                    projection_valid,
                    provenance_sources,
                    provenance_weights,
                    provenance_valid,
                    source_to_landmarks,
                    eval_radius,
                    allowed_landmarks=anchor_type != 0,
                )
                for group in {"all", split}:
                    provenance_coverage[eval_radius][group]["all"].append(
                        float(np.mean(target >= 0))
                    )
                    provenance_coverage[eval_radius][group]["base"].append(
                        float(np.mean(target_base >= 0))
                    )
                    provenance_coverage[eval_radius][group]["micro"].append(
                        float(np.mean(target_micro >= 0))
                    )

        targets = {}
        matchable = {}
        for eval_radius in (2.0, 4.0):
            target, nearest_distance = nearest_gt_targets(
                keypoint_xy, projected, valid, eval_radius
            )
            targets[eval_radius] = target
            matchable[eval_radius] = target >= 0
            candidate_distance = np.linalg.norm(
                keypoint_xy[:, None, :] - projected[topk_lm], axis=2
            )
            candidate_correct = valid[topk_lm] & (candidate_distance <= eval_radius)
            denom = max(int(matchable[eval_radius].sum()), 1)
            for recall_k in retrieval_ks:
                k = min(recall_k, topk_lm.shape[1])
                retrieval[eval_radius][recall_k].append(
                    float(candidate_correct[:, :k].any(axis=1)[matchable[eval_radius]].sum() / denom)
                )

        strict_target = targets[float(radius)]
        strict_matchable = matchable[float(radius)]
        raw_rows = np.asarray(query["matcher_raw_keypoint_idx"], dtype=np.int64)
        raw_lm = np.asarray(query["matcher_raw_landmark_idx"], dtype=np.int64)
        raw_scores = np.asarray(query["matcher_raw_scores"], dtype=np.float64)
        if not (
            raw_rows.size == keypoint_xy.shape[0]
            and np.array_equal(raw_rows, np.arange(raw_rows.size))
        ):
            raise ValueError(f"{image_name}: oracle currently requires one top1 per row")
        threshold = float(query["candidate_threshold"].item())
        max_per_keypoint = int(query["max_matches_per_keypoint"].item())
        max_per_landmark = int(query["max_matches_per_landmark"].item())
        min_matches = int(query["min_candidate_matches"].item())
        refill_trigger = int(query["candidate_refill_trigger_count"].item())

        raw_correct, _ = pair_is_correct(
            keypoint_xy[raw_rows], raw_lm, projected, valid, float(radius)
        )
        baseline = select_candidates(
            raw_rows,
            raw_lm,
            raw_scores,
            threshold=threshold,
            max_matches_per_keypoint=max_per_keypoint,
            max_matches_per_landmark=max_per_landmark,
            min_match_count=min_matches,
            refill_trigger_count=refill_trigger,
        )
        dumped_rows = np.asarray(query["hard_pre_keypoint_idx"], dtype=np.int64)
        dumped_lm = np.asarray(query["hard_pre_landmark_idx"], dtype=np.int64)
        dumped_scores = np.asarray(query["hard_pre_scores"], dtype=np.float64)
        replay_ok = (
            np.array_equal(baseline.keypoint_idx, dumped_rows)
            and np.array_equal(baseline.landmark_idx, dumped_lm)
            and np.allclose(baseline.scores, dumped_scores, atol=1e-7, rtol=0.0)
        )
        if not replay_ok:
            selector_replay_failures.append(image_name)
            raise AssertionError(f"{image_name}: hard selector replay differs from dump")
        if bool(query["geometry_selector_enabled"].item()):
            raise ValueError("oracle gate requires geometry selector to be disabled")
        if not (
            np.array_equal(
                np.asarray(query["hard_post_keypoint_idx"], dtype=np.int64),
                baseline.keypoint_idx,
            )
            and np.array_equal(
                np.asarray(query["hard_post_landmark_idx"], dtype=np.int64),
                baseline.landmark_idx,
            )
        ):
            raise AssertionError(f"{image_name}: post-selector hard set differs")

        hard_correct, _ = pair_is_correct(
            keypoint_xy[baseline.keypoint_idx],
            baseline.landmark_idx,
            projected,
            valid,
            float(radius),
        )
        hard_precision.append(float(hard_correct.mean()))
        hardcap = select_candidates(
            raw_rows,
            raw_lm,
            raw_scores,
            threshold=threshold,
            max_matches_per_keypoint=max_per_keypoint,
            max_matches_per_landmark=max_per_landmark,
            min_match_count=min_matches,
            refill_trigger_count=refill_trigger,
            correctness_priority=raw_correct,
        )
        normal_sources = set(baseline.source_idx.tolist())
        hardcap_sources = set(hardcap.source_idx.tolist())
        quota_displacements.append(len(normal_sources - hardcap_sources))

        swapped_lm = raw_lm.copy()
        swap_mask = strict_matchable[raw_rows] & ~raw_correct
        swapped_lm[swap_mask] = strict_target[raw_rows[swap_mask]]
        swapped_correct, _ = pair_is_correct(
            keypoint_xy[raw_rows], swapped_lm, projected, valid, float(radius)
        )
        swapped = select_candidates(
            raw_rows,
            swapped_lm,
            raw_scores,
            threshold=threshold,
            max_matches_per_keypoint=max_per_keypoint,
            max_matches_per_landmark=max_per_landmark,
            min_match_count=min_matches,
            refill_trigger_count=refill_trigger,
        )
        swapped_hardcap = select_candidates(
            raw_rows,
            swapped_lm,
            raw_scores,
            threshold=threshold,
            max_matches_per_keypoint=max_per_keypoint,
            max_matches_per_landmark=max_per_landmark,
            min_match_count=min_matches,
            refill_trigger_count=refill_trigger,
            correctness_priority=swapped_correct,
        )
        clean_hard = baseline.subset(hard_correct)
        assignment_oracle = oracle_assignment_candidates(
            raw_rows,
            strict_target[raw_rows],
            raw_scores,
        )
        provenance_assignment_oracle = oracle_assignment_candidates(
            raw_rows,
            (
                provenance_targets[float(radius)][raw_rows]
                if has_provenance
                else strict_target[raw_rows]
            ),
            raw_scores,
        )
        strict_candidate_distance = np.linalg.norm(
            keypoint_xy[:, None, :] - projected[topk_lm], axis=2
        )
        strict_candidate_correct = (
            valid[topk_lm] & (strict_candidate_distance <= float(radius))
        )
        topk_oracle_poses = {}
        topk_oracle_counts = {}
        for topk in oracle_assignment_ks:
            width_k = min(topk, topk_lm.shape[1])
            oracle_candidates = oracle_topk_candidates(
                topk_lm[:, :width_k],
                topk_scores[:, :width_k],
                strict_candidate_correct[:, :width_k],
            )
            topk_oracle_counts[topk] = int(oracle_candidates.scores.size)
            topk_oracle_poses[topk] = run_pose(
                oracle_candidates,
                keypoint_xy,
                landmark_xyz,
                K,
                query,
                seed + query_index,
            )[0]

        actual_pose = np.asarray(query["pred_pose_w2c"], dtype=np.float64)
        replay_pose, replay_inliers = run_pose(
            baseline, keypoint_xy, landmark_xyz, K, query, seed + query_index
        )
        swap_pose, _ = run_pose(
            swapped, keypoint_xy, landmark_xyz, K, query, seed + query_index
        )
        hardcap_pose, _ = run_pose(
            hardcap, keypoint_xy, landmark_xyz, K, query, seed + query_index
        )
        swap_hardcap_pose, _ = run_pose(
            swapped_hardcap, keypoint_xy, landmark_xyz, K, query, seed + query_index
        )
        clean_pose, _ = run_pose(
            clean_hard, keypoint_xy, landmark_xyz, K, query, seed + query_index
        )
        geometry_oracle_pose, _ = run_pose(
            clean_hard,
            keypoint_xy,
            geometry_oracle_xyz,
            K,
            query,
            seed + query_index,
        )

        dumped_inliers = np.asarray(query["hard_post_inliers"], dtype=np.int64)
        dumped_inliers = dumped_inliers[
            (dumped_inliers >= 0) & (dumped_inliers < len(baseline.scores))
        ]
        inlier_set = baseline.subset(dumped_inliers)
        inlier_correct, _ = pair_is_correct(
            keypoint_xy[inlier_set.keypoint_idx],
            inlier_set.landmark_idx,
            projected,
            valid,
            float(radius),
        )
        inlier_precision.append(float(inlier_correct.mean()))
        inlier_p2d = keypoint_xy[inlier_set.keypoint_idx].copy()
        inlier_p3d = landmark_xyz[inlier_set.landmark_idx]
        fixed_base_pose, _ = deterministic_pnp(inlier_p2d, inlier_p3d, K)
        signed_p2d = inlier_p2d.copy()
        signed_p2d[inlier_correct] = projected[inlier_set.landmark_idx[inlier_correct]]
        signed_pose, _ = deterministic_pnp(signed_p2d, inlier_p3d, K)
        clean_inliers = inlier_set.subset(inlier_correct)
        measurement_pose, _ = deterministic_pnp(
            projected[clean_inliers.landmark_idx],
            landmark_xyz[clean_inliers.landmark_idx],
            K,
        )

        base_bias_metrics = set_bias_metrics(
            baseline,
            keypoint_xy,
            landmark_xyz,
            K,
            gt_pose,
            translation_scale_m,
            rotation_scale_degrees,
        )
        final_bias_metrics = set_bias_metrics(
            inlier_set,
            keypoint_xy,
            landmark_xyz,
            K,
            gt_pose,
            translation_scale_m,
            rotation_scale_degrees,
        )
        clean_bias_metrics = set_bias_metrics(
            clean_hard,
            keypoint_xy,
            landmark_xyz,
            K,
            gt_pose,
            translation_scale_m,
            rotation_scale_degrees,
        )
        hard_bias.append(base_bias_metrics["bias_m"] * 100.0)
        final_bias.append(final_bias_metrics["bias_m"] * 100.0)
        clean_hard_bias.append(clean_bias_metrics["bias_m"] * 100.0)

        if skip_counterfactual:
            cf_records = []
        else:
            _, cf_records = counterfactual_gain_distribution(
                raw_rows,
                raw_lm,
                raw_scores,
                baseline,
                strict_target,
                keypoint_xy,
                landmark_xyz,
                K,
                gt_pose,
                threshold,
                max_per_landmark,
                translation_scale_m,
                rotation_scale_degrees,
            )
        strict_cf = [record for record in cf_records if record["strict_positive"]]
        cf_positive_rows.append(len(strict_cf))
        if strict_cf:
            cf_positive_queries += 1
            best_cf = max(strict_cf, key=lambda record: record["utility"])
            best_lm = raw_lm.copy()
            best_lm[best_cf["source_idx"]] = best_cf["target_landmark"]
            best_set = select_candidates(
                raw_rows,
                best_lm,
                raw_scores,
                threshold=threshold,
                max_matches_per_keypoint=max_per_keypoint,
                max_matches_per_landmark=max_per_landmark,
                min_match_count=min_matches,
                refill_trigger_count=refill_trigger,
            )
            best_pose, _ = run_pose(
                best_set, keypoint_xy, landmark_xyz, K, query, seed + query_index
            )
            all_lm = raw_lm.copy()
            for record in strict_cf:
                all_lm[record["source_idx"]] = record["target_landmark"]
            all_set = select_candidates(
                raw_rows,
                all_lm,
                raw_scores,
                threshold=threshold,
                max_matches_per_keypoint=max_per_keypoint,
                max_matches_per_landmark=max_per_landmark,
                min_match_count=min_matches,
                refill_trigger_count=refill_trigger,
            )
            all_pose, _ = run_pose(
                all_set, keypoint_xy, landmark_xyz, K, query, seed + query_index
            )
        else:
            best_pose = actual_pose
            all_pose = actual_pose

        accepted = raw_scores > threshold
        rejected = ~accepted
        unmatchable = ~strict_matchable[raw_rows]
        dustbin_stats.append(
            {
                "reject_precision": float(
                    unmatchable[rejected].mean() if rejected.any() else 0.0
                ),
                "reject_recall": float(
                    rejected[unmatchable].mean() if unmatchable.any() else 0.0
                ),
                "accept_matchable_precision": float(
                    strict_matchable[raw_rows][accepted].mean()
                    if accepted.any()
                    else 0.0
                ),
            }
        )

        poses = {
            "actual": actual_pose,
            "replay": replay_pose,
            "O1_oracle_3d_assignment": run_pose(
                assignment_oracle,
                keypoint_xy,
                landmark_xyz,
                K,
                query,
                seed + query_index,
            )[0],
            "OP_oracle_provenance_assignment": run_pose(
                provenance_assignment_oracle,
                keypoint_xy,
                landmark_xyz,
                K,
                query,
                seed + query_index,
            )[0],
            "OG_oracle_track_geometry": geometry_oracle_pose,
            "O2_oracle_candidate_filter": clean_pose,
            "O3_oracle_2d_measurement": measurement_pose,
            "O2_top1_swap": swap_pose,
            "O3_hardcap": hardcap_pose,
            "O2O3_swap_hardcap": swap_hardcap_pose,
            "O4_gt_clean_hard_set": clean_pose,
            "O5_fixed_inliers_base": fixed_base_pose,
            "O5_signed_coordinate": signed_pose,
            "O6_best_single_swap": best_pose,
            "O6_all_strict_swaps": all_pose,
            **{
                f"OK{topk}_one_of_k": topk_oracle_poses[topk]
                for topk in oracle_assignment_ks
            },
        }
        per_query_errors = {}
        for method, pose in poses.items():
            ae, te = pose_error(pose, gt_pose)
            errors[method]["ae"].append(ae)
            errors[method]["te"].append(te)
            per_query_errors[method] = {"ae_deg": ae, "te_cm": te}

        query_records.append(
            {
                "image_name": image_name,
                "matchable_rows_2px": int(strict_matchable.sum()),
                "oracle_assignment_matches": int(len(assignment_oracle.scores)),
                "provenance_assignment_matches": int(
                    len(provenance_assignment_oracle.scores)
                ),
                "one_of_k_oracle_matches": {
                    str(topk): topk_oracle_counts[topk]
                    for topk in oracle_assignment_ks
                },
                "hard_matches": int(len(baseline.scores)),
                "hard_gt_precision_2px": float(hard_correct.mean()),
                "ransac_inliers": int(len(inlier_set.scores)),
                "ransac_inlier_gt_precision_2px": float(inlier_correct.mean()),
                "quota_oracle_displacements": int(quota_displacements[-1]),
                "harmful_consensus_count": int((~inlier_correct).sum()),
                "geometry_oracle_replaced_clean_matches": int(
                    geometry_oracle_replaced[
                        clean_hard.landmark_idx
                    ].sum()
                ),
                "counterfactual_eligible_rows": int(len(cf_records)),
                "counterfactual_strict_positive_rows": int(len(strict_cf)),
                "hard_bias_cm": hard_bias[-1],
                "final_inlier_bias_cm": final_bias[-1],
                "clean_hard_bias_cm": clean_hard_bias[-1],
                "pose": per_query_errors,
            }
        )

    summaries = {
        method: summarize_pose_errors(values["ae"], values["te"])
        for method, values in errors.items()
    }
    paired = {
        method: paired_summary(
            errors["actual"]["te"],
            values["te"],
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        for method, values in errors.items()
        if method != "actual"
    }
    retrieval_summary = {
        f"radius_{int(eval_radius)}px": {
            f"recall_at_{k}": float(np.mean(values))
            for k, values in by_k.items()
        }
        for eval_radius, by_k in retrieval.items()
    }
    provenance_coverage_summary = {
        f"radius_{int(eval_radius)}px": {
            split: {
                anchor_class: (
                    float(np.mean(values)) if values else None
                )
                for anchor_class, values in classes.items()
            }
            for split, classes in splits.items()
        }
        for eval_radius, splits in provenance_coverage.items()
    }
    base_te = np.asarray(errors["actual"]["te"])
    oracle_te = np.asarray(errors["O2O3_swap_hardcap"]["te"])
    base_summary = summaries["actual"]
    oracle_summary = summaries["O2O3_swap_hardcap"]
    median_gain = base_summary["median_te_cm"] - oracle_summary["median_te_cm"]
    r2_gain = oracle_summary["recall_2cm_2deg"] - base_summary["recall_2cm_2deg"]
    r5_gain = oracle_summary["recall_5cm_5deg"] - base_summary["recall_5cm_5deg"]
    positive_cf_fraction = cf_positive_queries / max(len(query_records), 1)
    improved_queries = int((oracle_te < base_te - 1e-9).sum())
    clean_bias_improvement = float(np.median(final_bias) - np.median(clean_hard_bias))

    stop_reasons = []
    if median_gain < 0.1:
        stop_reasons.append("O2+O3 median TE gain is below 0.1 cm")
    if max(r2_gain, r5_gain) < 0.01:
        stop_reasons.append("O2+O3 R2/R5 gain is below 1 percentage point")
    if positive_cf_fraction < 0.15:
        stop_reasons.append("strict-positive counterfactual queries are below 15%")
    if clean_bias_improvement <= 0.0:
        stop_reasons.append("GT-clean hard set does not lower median bias")
    if improved_queries <= 3:
        stop_reasons.append("pose gain is concentrated in at most three queries")
    continue_checks = {
        "median_gain_over_0_2cm": bool(median_gain > 0.2),
        "paired_mean_decreases": bool(np.mean(oracle_te - base_te) < 0.0),
        "positive_counterfactual_queries_over_30pct": bool(
            positive_cf_fraction > 0.30
        ),
        "r2_or_r5_gain_at_least_1pp": bool(max(r2_gain, r5_gain) >= 0.01),
        "more_than_three_improved_queries": bool(improved_queries > 3),
    }
    recommendation = (
        "STOP_EXACT_TEACHER"
        if stop_reasons or not all(continue_checks.values())
        else "CONTINUE_WITH_D1_EXACT_SWAP"
    )
    corr = spearmanr(final_bias, errors["actual"]["te"])
    report = {
        "schema_version": 1,
        "dump_dir": str(dump_dir),
        "strict_gt_radius_px": float(radius),
        "task_translation_scale_m": translation_scale_m,
        "task_rotation_scale_degrees": rotation_scale_degrees,
        "query_count": len(query_records),
        "selector_replay_failures": selector_replay_failures,
        "O1_retrieval": retrieval_summary,
        "strict_splat_provenance_coverage": provenance_coverage_summary,
        "geometry_oracle": {
            "track_payload": str(track_payload or ""),
            "replaced_anchor_count": int(
                geometry_oracle_replaced.sum()
            ),
            "pose": summaries["OG_oracle_track_geometry"],
            "paired_vs_gt_clean_current_geometry": paired_summary(
                errors["O2_oracle_candidate_filter"]["te"],
                errors["OG_oracle_track_geometry"]["te"],
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            ),
        },
        "P0_oracles": {
            "O1_oracle_3d_assignment": summaries["O1_oracle_3d_assignment"],
            "O2_oracle_candidate_filter": summaries[
                "O2_oracle_candidate_filter"
            ],
            "OP_oracle_provenance_assignment": summaries[
                "OP_oracle_provenance_assignment"
            ],
            "O3_oracle_2d_measurement": summaries["O3_oracle_2d_measurement"],
            "counterfactual_enabled": bool(not skip_counterfactual),
            "one_of_k": {
                str(topk): summaries[f"OK{topk}_one_of_k"]
                for topk in oracle_assignment_ks
            },
        },
        "discrete_diagnostics": {
            "hard_top1_gt_precision_mean": float(np.mean(hard_precision)),
            "ransac_inlier_gt_precision_mean": float(np.mean(inlier_precision)),
            "quota_displacement_count_mean": float(np.mean(quota_displacements)),
            "harmful_consensus_count_mean": float(
                np.mean([row["harmful_consensus_count"] for row in query_records])
            ),
            "dustbin_reject_precision_mean": float(
                np.mean([row["reject_precision"] for row in dustbin_stats])
            ),
            "dustbin_reject_recall_mean": float(
                np.mean([row["reject_recall"] for row in dustbin_stats])
            ),
            "accepted_row_matchable_precision_mean": float(
                np.mean([row["accept_matchable_precision"] for row in dustbin_stats])
            ),
        },
        "bias_diagnostics": {
            "hard_set_bias_cm_median": float(np.median(hard_bias)),
            "final_inlier_bias_cm_median": float(np.median(final_bias)),
            "gt_clean_hard_set_bias_cm_median": float(np.median(clean_hard_bias)),
            "gt_clean_bias_improvement_cm": clean_bias_improvement,
            "final_bias_vs_actual_te_spearman": {
                "rho": float(corr.statistic),
                "pvalue": float(corr.pvalue),
            },
        },
        "O6_counterfactual": {
            "positive_query_count": int(cf_positive_queries),
            "positive_query_fraction": float(positive_cf_fraction),
            "strict_positive_rows_mean": float(np.mean(cf_positive_rows)),
            "strict_positive_rows_median": float(np.median(cf_positive_rows)),
        },
        "pose": summaries,
        "paired_vs_actual": paired,
        "oracle_gate": {
            "O2O3_median_te_gain_cm": float(median_gain),
            "O2O3_r2_gain": float(r2_gain),
            "O2O3_r5_gain": float(r5_gain),
            "O2O3_improved_query_count": improved_queries,
            "continue_checks": continue_checks,
            "stop_reasons": stop_reasons,
            "recommendation": recommendation,
        },
        "queries": query_records,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=_json_scalar) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius", type=float, default=2.0, choices=[2.0, 4.0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--translation_scale_m", type=float, default=None)
    parser.add_argument("--rotation_scale_degrees", type=float, default=None)
    parser.add_argument(
        "--skip_counterfactual",
        action="store_true",
        help="Skip expensive O6 single-swap analysis when only P0 oracle bounds are needed.",
    )
    parser.add_argument("--track_payload")
    args = parser.parse_args()
    report = evaluate(
        args.dump_dir,
        args.output,
        radius=args.radius,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        translation_scale_m=args.translation_scale_m,
        rotation_scale_degrees=args.rotation_scale_degrees,
        skip_counterfactual=args.skip_counterfactual,
        track_payload=args.track_payload,
    )
    print(json.dumps({
        "O1_retrieval": report["O1_retrieval"],
        "P0_oracles": report["P0_oracles"],
        "discrete_diagnostics": report["discrete_diagnostics"],
        "bias_diagnostics": report["bias_diagnostics"],
        "O6_counterfactual": report["O6_counterfactual"],
        "pose": report["pose"],
        "oracle_gate": report["oracle_gate"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
