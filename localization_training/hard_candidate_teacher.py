"""Discrete RANSAC teachers for inference-aligned candidate preservation.

The sparse candidate losses operate on a soft candidate graph.  This module
adds a deliberately non-differentiable teacher that replays the deployed hard
top-1/quota candidate set, runs PnP/RANSAC, and returns labels aligned back to
the current candidate tensor.  Gradients flow only through the selected
descriptor similarities, never through RANSAC or pose estimation.
"""

from dataclasses import dataclass
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.pose_information import (
    compute_pose_information,
    normalize_information_scores,
    pose_jacobian_analytic,
)
from localization_training.pose_refiner import project_points
from localization_training.sparse_frontend import (
    SparseMatchResult,
    match_candidate_selection_mask,
)
from utils.pose_utils import cal_pose_error, solve_pose


@dataclass
class HardCandidateTargets:
    """Teacher labels aligned with the predicted-candidate axis."""

    useful_mask: torch.Tensor
    harmful_mask: torch.Tensor
    useful_weights: torch.Tensor
    harmful_weights: torch.Tensor
    diagnostics: dict


def _empty_targets(reference, diagnostics=None):
    count = int(reference.numel())
    return HardCandidateTargets(
        useful_mask=torch.zeros(count, dtype=torch.bool, device=reference.device),
        harmful_mask=torch.zeros(count, dtype=torch.bool, device=reference.device),
        useful_weights=torch.zeros(count, dtype=reference.dtype, device=reference.device),
        harmful_weights=torch.zeros(count, dtype=reference.dtype, device=reference.device),
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )


def _ranked_subset(indices, scores, limit):
    """Keep a stable highest-score subset without introducing random sampling."""
    if indices.numel() == 0 or int(limit) <= 0 or indices.numel() <= int(limit):
        return indices
    order = torch.argsort(scores[indices], descending=True, stable=True)
    return indices[order[: int(limit)]]


def _candidate_keys(keypoint_ids, candidate_keypoint_idx, candidate_landmark_idx, landmark_count):
    keypoint_ids = torch.as_tensor(keypoint_ids, dtype=torch.long).reshape(-1)
    candidate_keypoint_idx = torch.as_tensor(
        candidate_keypoint_idx, dtype=torch.long
    ).reshape(-1)
    candidate_landmark_idx = torch.as_tensor(
        candidate_landmark_idx, dtype=torch.long
    ).reshape(-1)
    if candidate_keypoint_idx.numel() != candidate_landmark_idx.numel():
        raise ValueError("candidate keypoint and landmark index counts must match")
    if candidate_keypoint_idx.numel() == 0:
        return np.empty(0, dtype=np.int64)
    if bool((candidate_keypoint_idx < 0).any()) or bool(
        (candidate_keypoint_idx >= keypoint_ids.numel()).any()
    ):
        raise ValueError("candidate keypoint indices are out of range")
    if bool((candidate_landmark_idx < 0).any()) or bool(
        (candidate_landmark_idx >= int(landmark_count)).any()
    ):
        raise ValueError("candidate landmark indices are out of range")
    base = int(landmark_count) + 1
    absolute_keypoints = keypoint_ids[candidate_keypoint_idx]
    packed = absolute_keypoints * base + candidate_landmark_idx
    return packed.detach().cpu().numpy().astype(np.int64, copy=False)


def _batched_camera_centers_after_left_updates(pose_w2c, updates):
    """Return camera centers after batched left SE(3) updates.

    The hard teacher only uses this detached calculation to decide whether
    deleting a RANSAC inlier lowers the current translation bias.  Keeping it
    batched makes the test inexpensive even when a query has many inliers.
    """
    updates = torch.as_tensor(updates)
    if updates.ndim != 2 or updates.shape[1] != 6:
        raise ValueError("updates must have shape [N, 6]")
    pose_w2c = torch.as_tensor(
        pose_w2c, device=updates.device, dtype=updates.dtype
    ).reshape(4, 4)
    count = int(updates.shape[0])
    if count == 0:
        return updates.new_empty((0, 3))

    translation = updates[:, :3]
    rotation = updates[:, 3:]
    theta = torch.linalg.norm(rotation, dim=1)
    theta2 = theta.square()
    theta_safe = theta.clamp_min(1e-8)
    small = theta < 1e-4

    skew = updates.new_zeros((count, 3, 3))
    skew[:, 0, 1] = -rotation[:, 2]
    skew[:, 0, 2] = rotation[:, 1]
    skew[:, 1, 0] = rotation[:, 2]
    skew[:, 1, 2] = -rotation[:, 0]
    skew[:, 2, 0] = -rotation[:, 1]
    skew[:, 2, 1] = rotation[:, 0]
    skew_squared = skew @ skew

    # Stable Rodrigues and left-Jacobian coefficients near zero rotation.
    a_small = 1.0 - theta2 / 6.0 + theta2.square() / 120.0
    b_small = 0.5 - theta2 / 24.0 + theta2.square() / 720.0
    c_small = 1.0 / 6.0 - theta2 / 120.0 + theta2.square() / 5040.0
    a = torch.where(small, a_small, torch.sin(theta) / theta_safe)
    b = torch.where(small, b_small, (1.0 - torch.cos(theta)) / theta_safe.square())
    c = torch.where(
        small,
        c_small,
        (theta - torch.sin(theta)) / theta_safe.pow(3),
    )
    eye = torch.eye(3, dtype=updates.dtype, device=updates.device)[None]
    rotation_delta = eye + a[:, None, None] * skew + b[:, None, None] * skew_squared
    translation_delta = (
        (eye + b[:, None, None] * skew + c[:, None, None] * skew_squared)
        @ translation[:, :, None]
    )[:, :, 0]

    rotation_new = rotation_delta @ pose_w2c[:3, :3]
    translation_new = (
        rotation_delta @ pose_w2c[:3, 3][None, :, None]
    )[:, :, 0] + translation_delta
    return -(rotation_new.transpose(1, 2) @ translation_new[:, :, None])[:, :, 0]


def _linearized_translation_delete_gains(
    points_world,
    observed_xy,
    K,
    pose_w2c,
    gt_pose_w2c,
    damping=1e-4,
):
    """Approximate translation-error reduction from deleting each observation.

    This is a non-differentiable label calculation.  Around the current
    RANSAC pose, it solves the full and leave-one-out Gauss-Newton systems and
    compares their camera-center error against the known GT pose.  Positive
    values mean that deleting that observation reduces translation error.
    """
    points_world = torch.as_tensor(points_world, dtype=torch.float64)
    observed_xy = torch.as_tensor(
        observed_xy, device=points_world.device, dtype=points_world.dtype
    )
    K = torch.as_tensor(K, device=points_world.device, dtype=points_world.dtype)
    pose_w2c = torch.as_tensor(
        pose_w2c, device=points_world.device, dtype=points_world.dtype
    ).reshape(4, 4)
    gt_pose_w2c = torch.as_tensor(
        gt_pose_w2c, device=points_world.device, dtype=points_world.dtype
    ).reshape(4, 4)
    count = int(points_world.shape[0])
    gains = points_world.new_zeros((count,))
    if count < 4:
        return gains

    projected_xy, positive_depth = project_points(points_world, K, pose_w2c)
    residual = observed_xy - projected_xy
    valid = (
        positive_depth
        & torch.isfinite(projected_xy).all(dim=1)
        & torch.isfinite(residual).all(dim=1)
    )
    if int(valid.sum().item()) < 4:
        return gains

    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    points = points_world[valid_indices]
    residual = residual[valid_indices]
    jacobian = pose_jacobian_analytic(points, K, pose_w2c)
    contribution = jacobian.transpose(1, 2) @ jacobian
    rhs = (jacobian.transpose(1, 2) @ residual[:, :, None])[:, :, 0]
    eye = torch.eye(6, dtype=points.dtype, device=points.device)
    full_information = contribution.sum(dim=0) + float(damping) * eye
    full_rhs = rhs.sum(dim=0)
    try:
        full_update = torch.linalg.solve(full_information, full_rhs)
        without_update = torch.linalg.solve(
            full_information[None] - contribution,
            (full_rhs[None] - rhs)[:, :, None],
        )[:, :, 0]
    except RuntimeError:
        full_update = torch.linalg.pinv(full_information) @ full_rhs
        without_update = (
            torch.linalg.pinv(full_information[None] - contribution)
            @ (full_rhs[None] - rhs)[:, :, None]
        )[:, :, 0]

    full_center = _batched_camera_centers_after_left_updates(
        pose_w2c, full_update[None]
    )[0]
    without_centers = _batched_camera_centers_after_left_updates(
        pose_w2c, without_update
    )
    gt_center = -(
        gt_pose_w2c[:3, :3].transpose(0, 1) @ gt_pose_w2c[:3, 3]
    )
    full_error = torch.linalg.norm(full_center - gt_center)
    delete_gains = full_error - torch.linalg.norm(
        without_centers - gt_center[None], dim=1
    )
    delete_gains[~torch.isfinite(delete_gains)] = 0.0
    gains[valid_indices] = delete_gains
    return gains


def _default_pose_solver(
    p2d,
    p3d,
    K,
    *,
    solver,
    reprojection_error,
    confidence,
    max_iterations,
    min_iterations,
    ransac_seed=0,
):
    return solve_pose(
        p2d,
        p3d,
        K,
        solver=str(solver),
        reprojection_error=float(reprojection_error),
        confidence=float(confidence),
        max_iterations=int(max_iterations),
        min_iterations=int(min_iterations),
        ransac_seed=int(ransac_seed),
    )


def _replay_candidate_selection_mask(
    candidate_keypoint_idx,
    candidate_landmark_idx,
    candidate_scores,
    *,
    threshold,
    max_matches_per_keypoint,
    max_matches_per_landmark,
):
    """Reapply the deployed score/quota rule to a detached candidate graph."""
    matches = SparseMatchResult(
        keypoint_idx=torch.as_tensor(candidate_keypoint_idx, dtype=torch.long),
        landmark_idx=torch.as_tensor(candidate_landmark_idx, dtype=torch.long),
        scores=torch.as_tensor(candidate_scores),
    )
    return match_candidate_selection_mask(
        matches,
        threshold=float(threshold),
        max_matches_per_keypoint=int(max_matches_per_keypoint),
        max_matches_per_landmark=int(max_matches_per_landmark),
    )


def derive_hard_candidate_targets(
    *,
    keypoint_xy,
    keypoint_ids,
    candidate_keypoint_idx,
    candidate_landmark_idx,
    candidate_scores,
    deployment_mask,
    gt_correct_mask,
    landmark_xyz,
    K,
    pose_gt_w2c,
    solver="poselib",
    reprojection_error=8.0,
    confidence=0.99999,
    max_iterations=100000,
    min_iterations=1000,
    ransac_seed=0,
    min_inliers=4,
    max_pose_error_cm=100.0,
    max_useful=96,
    max_harmful=96,
    translation_scale=0.02,
    rotation_scale_degrees=2.0,
    harmful_mode="all_false",
    harmful_min_translation_delete_gain_m=0.0,
    exact_replay_max_candidates=8,
    exact_replay_min_pose_gain_cm=0.0,
    exact_replay_rotation_weight_cm_per_degree=0.0,
    exact_replay_selection_threshold=-float("inf"),
    exact_replay_max_matches_per_keypoint=0,
    exact_replay_max_matches_per_landmark=0,
    pose_solver=None,
):
    """Label useful and harmful hard candidates from one deployed PnP replay.

    ``deployment_mask`` must already encode the same candidate graph used by
    deployment (top-k/matching threshold and per-landmark quota).  The result
    is aligned to all predicted candidates, not merely the candidates passed to
    PnP, so it can be applied directly to differentiable similarities.
    """
    harmful_mode = str(harmful_mode)
    if harmful_mode not in {
        "all_false",
        "translation_delete",
        "exact_pose_delete",
    }:
        raise ValueError(
            "harmful_mode must be 'all_false', 'translation_delete', or "
            "'exact_pose_delete'"
        )
    if float(harmful_min_translation_delete_gain_m) < 0.0:
        raise ValueError(
            "harmful_min_translation_delete_gain_m must be non-negative"
        )
    if int(exact_replay_max_candidates) < 0:
        raise ValueError("exact_replay_max_candidates must be non-negative")
    if float(exact_replay_min_pose_gain_cm) < 0.0:
        raise ValueError("exact_replay_min_pose_gain_cm must be non-negative")
    if float(exact_replay_rotation_weight_cm_per_degree) < 0.0:
        raise ValueError(
            "exact_replay_rotation_weight_cm_per_degree must be non-negative"
        )
    if int(exact_replay_max_matches_per_keypoint) < 0:
        raise ValueError(
            "exact_replay_max_matches_per_keypoint must be non-negative"
        )
    if int(exact_replay_max_matches_per_landmark) < 0:
        raise ValueError(
            "exact_replay_max_matches_per_landmark must be non-negative"
        )
    candidate_scores = torch.as_tensor(candidate_scores)
    device = candidate_scores.device
    dtype = candidate_scores.dtype
    count = int(candidate_scores.numel())
    tensors = {
        "candidate_keypoint_idx": candidate_keypoint_idx,
        "candidate_landmark_idx": candidate_landmark_idx,
        "deployment_mask": deployment_mask,
        "gt_correct_mask": gt_correct_mask,
    }
    for name, value in tensors.items():
        if torch.as_tensor(value).numel() != count:
            raise ValueError(f"{name} must have one entry per candidate")
    base_diagnostics = {
        "hard_teacher_candidate_count": float(count),
        "hard_teacher_deployment_count": 0.0,
        "hard_teacher_ransac_inlier_count": 0.0,
        "hard_teacher_ransac_inlier_gt_precision": 0.0,
        "hard_teacher_useful_count": 0.0,
        "hard_teacher_false_ransac_inlier_count": 0.0,
        "hard_teacher_harmful_count": 0.0,
        "hard_teacher_selected_useful_count": 0.0,
        "hard_teacher_selected_harmful_count": 0.0,
        "hard_teacher_pose_te_cm": float("inf"),
        "hard_teacher_pose_ae_deg": float("inf"),
        "hard_teacher_translation_delete_gain_mean": 0.0,
        "hard_teacher_translation_delete_gain_max": 0.0,
        "hard_teacher_harmful_translation_delete_evaluable": 0.0,
        "hard_teacher_harmful_translation_delete_gain_mean": 0.0,
        "hard_teacher_harmful_translation_delete_gain_max": 0.0,
        "hard_teacher_selected_harmful_translation_delete_gain_mean": 0.0,
        "hard_teacher_selected_harmful_translation_delete_gain_median": 0.0,
        "hard_teacher_selected_harmful_translation_delete_gain_max": 0.0,
        "hard_teacher_harmful_bias_improving_count": 0.0,
        "hard_teacher_exact_replay_graph_aligned": 0.0,
        "hard_teacher_exact_replay_graph_checked": 0.0,
        "hard_teacher_exact_replay_graph_mismatch_count": 0.0,
        "hard_teacher_exact_replay_exception": 0.0,
        "hard_teacher_exact_replay_baseline_pose_valid": 0.0,
        "hard_teacher_exact_replay_candidate_cap": float(
            max(int(exact_replay_max_candidates), 0)
        ),
        "hard_teacher_exact_replay_evaluated_count": 0.0,
        "hard_teacher_exact_replay_valid_count": 0.0,
        "hard_teacher_exact_replay_positive_count": 0.0,
        "hard_teacher_exact_replay_refill_count": 0.0,
        "hard_teacher_exact_replay_pose_gain_cm_mean": 0.0,
        "hard_teacher_exact_replay_pose_gain_cm_max": 0.0,
        "hard_teacher_exact_replay_te_gain_cm_mean": 0.0,
        "hard_teacher_exact_replay_ae_gain_deg_mean": 0.0,
        "hard_teacher_valid": 0.0,
    }
    if count == 0:
        return _empty_targets(candidate_scores, base_diagnostics)

    candidate_keypoint_idx = torch.as_tensor(
        candidate_keypoint_idx, device=device, dtype=torch.long
    ).reshape(-1)
    candidate_landmark_idx = torch.as_tensor(
        candidate_landmark_idx, device=device, dtype=torch.long
    ).reshape(-1)
    deployment_mask = torch.as_tensor(
        deployment_mask, device=device, dtype=torch.bool
    ).reshape(-1)
    gt_correct_mask = torch.as_tensor(
        gt_correct_mask, device=device, dtype=torch.bool
    ).reshape(-1)
    keypoint_xy = torch.as_tensor(keypoint_xy, device=device, dtype=dtype)
    landmark_xyz = torch.as_tensor(landmark_xyz, device=device, dtype=dtype)
    if keypoint_xy.ndim != 2 or keypoint_xy.shape[1] != 2:
        raise ValueError("keypoint_xy must be [N, 2]")
    if landmark_xyz.ndim != 2 or landmark_xyz.shape[1] != 3:
        raise ValueError("landmark_xyz must be [K, 3]")

    candidate_valid = (candidate_keypoint_idx >= 0) & (
        candidate_keypoint_idx < keypoint_xy.shape[0]
    )
    candidate_valid &= (candidate_landmark_idx >= 0) & (
        candidate_landmark_idx < landmark_xyz.shape[0]
    )
    if bool(candidate_valid.any()):
        candidate_xy = keypoint_xy[candidate_keypoint_idx[candidate_valid]]
        candidate_xyz = landmark_xyz[candidate_landmark_idx[candidate_valid]]
        finite = torch.isfinite(candidate_xy).all(dim=1) & torch.isfinite(
            candidate_xyz
        ).all(dim=1)
        valid_indices = torch.nonzero(candidate_valid, as_tuple=False).reshape(-1)
        candidate_valid.zero_()
        candidate_valid[valid_indices[finite]] = True
    valid = deployment_mask & candidate_valid
    selected = torch.nonzero(valid, as_tuple=False).reshape(-1)
    base_diagnostics["hard_teacher_deployment_count"] = float(selected.numel())
    if selected.numel() < max(int(min_inliers), 4):
        return _empty_targets(candidate_scores, base_diagnostics)

    K_cpu = torch.as_tensor(K).detach().cpu().double().numpy()
    gt_pose_cpu = torch.as_tensor(pose_gt_w2c).detach().cpu().double().numpy()
    solver_fn = _default_pose_solver if pose_solver is None else pose_solver

    def replay_pose(replay_selected):
        replay_selected = torch.as_tensor(
            replay_selected, device=device, dtype=torch.long
        ).reshape(-1)
        if replay_selected.numel() < max(int(min_inliers), 4):
            return None
        p2d = (
            keypoint_xy[candidate_keypoint_idx[replay_selected]]
            .detach()
            .cpu()
            .double()
            .numpy()
            + 0.5
        )
        p3d = (
            landmark_xyz[candidate_landmark_idx[replay_selected]]
            .detach()
            .cpu()
            .double()
            .numpy()
        )
        try:
            pose, inliers = solver_fn(
                p2d,
                p3d,
                K_cpu,
                solver=solver,
                reprojection_error=reprojection_error,
                confidence=confidence,
                max_iterations=max_iterations,
                min_iterations=min_iterations,
                ransac_seed=ransac_seed,
            )
        except Exception:
            return None
        pose = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)
        inliers = inliers[(inliers >= 0) & (inliers < replay_selected.numel())]
        if inliers.size < max(int(min_inliers), 4) or not np.isfinite(pose).all():
            return None
        return pose, inliers

    replay = replay_pose(selected)
    if replay is None:
        return _empty_targets(candidate_scores, base_diagnostics)
    pose, inliers = replay

    pose_ae, pose_te = cal_pose_error(pose, gt_pose_cpu)
    base_diagnostics["hard_teacher_pose_te_cm"] = float(pose_te)
    base_diagnostics["hard_teacher_pose_ae_deg"] = float(pose_ae)
    if not np.isfinite(pose_te) or float(pose_te) > float(max_pose_error_cm):
        return _empty_targets(candidate_scores, base_diagnostics)
    base_diagnostics["hard_teacher_exact_replay_baseline_pose_valid"] = 1.0

    inlier_selected = torch.as_tensor(inliers, device=device, dtype=torch.long)
    inlier_indices = selected[inlier_selected]
    inlier_correct = gt_correct_mask[inlier_indices]
    useful_indices = inlier_indices[inlier_correct]
    false_inlier_indices = inlier_indices[~inlier_correct]
    harmful_indices = false_inlier_indices
    useful_weights = torch.zeros(count, dtype=dtype, device=device)
    harmful_weights = torch.zeros(count, dtype=dtype, device=device)

    translation_scores = torch.zeros(
        inlier_indices.numel(), dtype=dtype, device=device
    )
    try:
        info = compute_pose_information(
            landmark_xyz[candidate_landmark_idx[inlier_indices]]
            .detach()
            .cpu()
            .double(),
            torch.as_tensor(K_cpu, dtype=torch.float64),
            torch.as_tensor(pose, dtype=torch.float64),
            damping=1e-4,
            translation_scale=float(translation_scale),
            rotation_scale=math.radians(float(rotation_scale_degrees)),
            use_analytic_jacobian=True,
        )
        raw_scores = info.translation_scores.to(dtype=torch.float32)
        normalized = normalize_information_scores(
            raw_scores, floor=0.0, mode="quantile"
        ).to(dtype=dtype, device=device)
        translation_scores = raw_scores.to(dtype=dtype, device=device)
        # Every GT-clean RANSAC inlier is preserved, while high leave-one-out
        # translation impact receives up to four times the baseline weight.
        inlier_weights = 0.25 + 0.75 * normalized
        useful_weights[inlier_indices[inlier_correct]] = inlier_weights[
            inlier_correct
        ]
    except Exception:
        useful_weights[useful_indices] = 1.0

    harmful_translation_gains = torch.zeros(
        inlier_indices.numel(), dtype=dtype, device=device
    )
    harmful_delete_evaluable = False
    exact_replay_gains = candidate_scores.new_empty((0,))
    exact_replay_te_gains = candidate_scores.new_empty((0,))
    exact_replay_ae_gains = candidate_scores.new_empty((0,))
    exact_replay_refill_count = 0
    exact_replay_evaluated_count = 0
    exact_replay_valid_count = 0
    exact_replay_graph_aligned = False
    exact_replay_graph_checked = False
    exact_replay_graph_mismatch_count = 0
    exact_replay_exception = None
    if harmful_mode == "all_false":
        if harmful_indices.numel() > 0:
            harmful_weights[harmful_indices] = 1.0
        harmful_rank_scores = candidate_scores.detach()
    elif harmful_mode == "translation_delete":
        harmful_indices = inlier_indices.new_empty(0)
        try:
            harmful_translation_gains = _linearized_translation_delete_gains(
                landmark_xyz[candidate_landmark_idx[inlier_indices]]
                .detach()
                .cpu()
                .double(),
                keypoint_xy[candidate_keypoint_idx[inlier_indices]]
                .detach()
                .cpu()
                .double()
                + 0.5,
                torch.as_tensor(K_cpu, dtype=torch.float64),
                torch.as_tensor(pose, dtype=torch.float64),
                torch.as_tensor(gt_pose_cpu, dtype=torch.float64),
            ).to(dtype=dtype, device=device)
            harmful_delete_evaluable = True
            false_gains = harmful_translation_gains[~inlier_correct]
            bias_improving = false_gains > float(
                harmful_min_translation_delete_gain_m
            )
            harmful_indices = false_inlier_indices[bias_improving]
            if harmful_indices.numel() > 0:
                normalized_harm = normalize_information_scores(
                    false_gains[bias_improving], floor=0.0, mode="quantile"
                ).to(dtype=dtype, device=device)
                harmful_weights[harmful_indices] = 0.25 + 0.75 * normalized_harm
            harmful_rank_scores = harmful_weights
        except Exception:
            # A strict teacher must not silently fall back to all false inliers.
            harmful_rank_scores = harmful_weights
    else:
        harmful_indices = inlier_indices.new_empty(0)
        try:
            baseline_replay_mask = _replay_candidate_selection_mask(
                candidate_keypoint_idx,
                candidate_landmark_idx,
                candidate_scores.detach(),
                threshold=exact_replay_selection_threshold,
                max_matches_per_keypoint=exact_replay_max_matches_per_keypoint,
                max_matches_per_landmark=exact_replay_max_matches_per_landmark,
            ).to(device=device, dtype=torch.bool)
            baseline_replay_mask &= candidate_valid
            exact_replay_graph_mismatch_count = int(
                (baseline_replay_mask ^ valid).sum().item()
            )
            exact_replay_graph_checked = True
            exact_replay_graph_aligned = exact_replay_graph_mismatch_count == 0
            false_local = torch.nonzero(~inlier_correct, as_tuple=False).reshape(-1)
            if exact_replay_graph_aligned and false_local.numel() > 0:
                # Candidate score only bounds the replay budget. The label
                # itself comes from a full deterministic deployment-graph PnP
                # re-solve, not from a Fisher surrogate.
                replay_candidates = _ranked_subset(
                    false_inlier_indices,
                    candidate_scores.detach(),
                    exact_replay_max_candidates,
                )
                replay_gains = []
                replay_te_gains = []
                replay_ae_gains = []
                positive_candidates = []
                positive_gains = []
                base_pose_cost = float(pose_te) + float(
                    exact_replay_rotation_weight_cm_per_degree
                ) * float(pose_ae)
                for candidate_index in replay_candidates.detach().cpu().tolist():
                    replay_scores = candidate_scores.detach().clone()
                    replay_scores[int(candidate_index)] = -torch.inf
                    replay_mask = _replay_candidate_selection_mask(
                        candidate_keypoint_idx,
                        candidate_landmark_idx,
                        replay_scores,
                        threshold=exact_replay_selection_threshold,
                        max_matches_per_keypoint=(
                            exact_replay_max_matches_per_keypoint
                        ),
                        max_matches_per_landmark=(
                            exact_replay_max_matches_per_landmark
                        ),
                    ).to(device=device, dtype=torch.bool)
                    replay_mask &= candidate_valid
                    if bool(replay_mask[int(candidate_index)].item()):
                        continue
                    replay_selected = torch.nonzero(
                        replay_mask, as_tuple=False
                    ).reshape(-1)
                    exact_replay_evaluated_count += 1
                    exact_replay_refill_count += int(
                        (replay_mask & ~valid).sum().item()
                    )
                    replay = replay_pose(replay_selected)
                    if replay is None:
                        continue
                    replay_pose_w2c, _ = replay
                    replay_ae, replay_te = cal_pose_error(
                        replay_pose_w2c, gt_pose_cpu
                    )
                    if not np.isfinite(replay_te) or not np.isfinite(replay_ae):
                        continue
                    exact_replay_valid_count += 1
                    te_gain = float(pose_te) - float(replay_te)
                    ae_gain = float(pose_ae) - float(replay_ae)
                    pose_gain = base_pose_cost - (
                        float(replay_te)
                        + float(exact_replay_rotation_weight_cm_per_degree)
                        * float(replay_ae)
                    )
                    replay_gains.append(pose_gain)
                    replay_te_gains.append(te_gain)
                    replay_ae_gains.append(ae_gain)
                    if pose_gain > float(exact_replay_min_pose_gain_cm):
                        positive_candidates.append(int(candidate_index))
                        positive_gains.append(pose_gain)
                if positive_candidates:
                    harmful_indices = torch.as_tensor(
                        positive_candidates, device=device, dtype=torch.long
                    )
                    positive_gains = torch.as_tensor(
                        positive_gains,
                        dtype=dtype,
                        device=device,
                    )
                    normalized_harm = normalize_information_scores(
                        positive_gains, floor=0.0, mode="quantile"
                    ).to(dtype=dtype, device=device)
                    harmful_weights[harmful_indices] = 0.25 + 0.75 * normalized_harm
                exact_replay_gains = torch.as_tensor(
                    replay_gains, dtype=dtype, device=device
                )
                exact_replay_te_gains = torch.as_tensor(
                    replay_te_gains, dtype=dtype, device=device
                )
                exact_replay_ae_gains = torch.as_tensor(
                    replay_ae_gains, dtype=dtype, device=device
                )
            harmful_rank_scores = harmful_weights
        except Exception as exc:
            # Exact mode never falls back to all false RANSAC inliers.
            exact_replay_exception = exc
            harmful_rank_scores = harmful_weights
    harmful_eligible_count = int(harmful_indices.numel())
    useful_indices = _ranked_subset(
        useful_indices, useful_weights, max_useful
    )
    harmful_indices = _ranked_subset(
        harmful_indices, harmful_rank_scores, max_harmful
    )
    useful_mask = torch.zeros(count, dtype=torch.bool, device=device)
    harmful_mask = torch.zeros(count, dtype=torch.bool, device=device)
    useful_mask[useful_indices] = True
    harmful_mask[harmful_indices] = True
    useful_weights = useful_weights * useful_mask.to(dtype=dtype)
    harmful_weights = harmful_weights * harmful_mask.to(dtype=dtype)
    selected_harmful_gains = harmful_translation_gains.new_empty((0,))
    if harmful_mode == "translation_delete" and bool(harmful_mask.any()):
        candidate_gains = torch.zeros(count, dtype=dtype, device=device)
        candidate_gains[inlier_indices] = harmful_translation_gains
        selected_harmful_gains = candidate_gains[harmful_mask]

    base_diagnostics.update(
        {
            "hard_teacher_ransac_inlier_count": float(inlier_indices.numel()),
            "hard_teacher_ransac_inlier_gt_precision": float(
                inlier_correct.float().mean().item()
            ),
            "hard_teacher_useful_count": float((inlier_correct).sum().item()),
            "hard_teacher_false_ransac_inlier_count": float(
                (~inlier_correct).sum().item()
            ),
            "hard_teacher_harmful_count": float(harmful_eligible_count),
            "hard_teacher_selected_useful_count": float(useful_mask.sum().item()),
            "hard_teacher_selected_harmful_count": float(harmful_mask.sum().item()),
            "hard_teacher_translation_delete_gain_mean": float(
                translation_scores[inlier_correct].mean().item()
                if bool(inlier_correct.any())
                else 0.0
            ),
            "hard_teacher_translation_delete_gain_max": float(
                translation_scores[inlier_correct].max().item()
                if bool(inlier_correct.any())
                else 0.0
            ),
            "hard_teacher_harmful_translation_delete_evaluable": float(
                harmful_delete_evaluable
            ),
            "hard_teacher_harmful_translation_delete_gain_mean": float(
                harmful_translation_gains[~inlier_correct].mean().item()
                if harmful_mode == "translation_delete"
                and bool((~inlier_correct).any())
                else 0.0
            ),
            "hard_teacher_harmful_translation_delete_gain_max": float(
                harmful_translation_gains[~inlier_correct].max().item()
                if harmful_mode == "translation_delete"
                and bool((~inlier_correct).any())
                else 0.0
            ),
            "hard_teacher_selected_harmful_translation_delete_gain_mean": float(
                selected_harmful_gains.mean().item()
                if selected_harmful_gains.numel()
                else 0.0
            ),
            "hard_teacher_selected_harmful_translation_delete_gain_median": float(
                selected_harmful_gains.median().item()
                if selected_harmful_gains.numel()
                else 0.0
            ),
            "hard_teacher_selected_harmful_translation_delete_gain_max": float(
                selected_harmful_gains.max().item()
                if selected_harmful_gains.numel()
                else 0.0
            ),
            "hard_teacher_harmful_bias_improving_count": float(
                (
                    harmful_translation_gains[~inlier_correct]
                    > float(harmful_min_translation_delete_gain_m)
                )
                .sum()
                .item()
                if harmful_mode == "translation_delete"
                else 0.0
            ),
            "hard_teacher_exact_replay_graph_aligned": float(
                exact_replay_graph_aligned
            ),
            "hard_teacher_exact_replay_graph_checked": float(
                exact_replay_graph_checked
            ),
            "hard_teacher_exact_replay_graph_mismatch_count": float(
                exact_replay_graph_mismatch_count
            ),
            "hard_teacher_exact_replay_exception": float(
                exact_replay_exception is not None
            ),
            "hard_teacher_exact_replay_exception_type": (
                type(exact_replay_exception).__name__
                if exact_replay_exception is not None
                else ""
            ),
            "hard_teacher_exact_replay_exception_message": (
                str(exact_replay_exception)
                if exact_replay_exception is not None
                else ""
            ),
            "hard_teacher_exact_replay_evaluated_count": float(
                exact_replay_evaluated_count
            ),
            "hard_teacher_exact_replay_valid_count": float(
                exact_replay_valid_count
            ),
            "hard_teacher_exact_replay_positive_count": float(
                harmful_eligible_count
                if harmful_mode == "exact_pose_delete"
                else 0
            ),
            "hard_teacher_exact_replay_refill_count": float(
                exact_replay_refill_count
            ),
            "hard_teacher_exact_replay_pose_gain_cm_mean": float(
                exact_replay_gains.mean().item()
                if exact_replay_gains.numel()
                else 0.0
            ),
            "hard_teacher_exact_replay_pose_gain_cm_max": float(
                exact_replay_gains.max().item()
                if exact_replay_gains.numel()
                else 0.0
            ),
            "hard_teacher_exact_replay_te_gain_cm_mean": float(
                exact_replay_te_gains.mean().item()
                if exact_replay_te_gains.numel()
                else 0.0
            ),
            "hard_teacher_exact_replay_ae_gain_deg_mean": float(
                exact_replay_ae_gains.mean().item()
                if exact_replay_ae_gains.numel()
                else 0.0
            ),
            "hard_teacher_valid": 1.0,
        }
    )
    return HardCandidateTargets(
        useful_mask=useful_mask,
        harmful_mask=harmful_mask,
        useful_weights=useful_weights,
        harmful_weights=harmful_weights,
        diagnostics=base_diagnostics,
    )


def hard_candidate_preservation_loss(
    candidate_logits,
    targets,
    *,
    temperature=0.05,
    margin=0.05,
    score_target=0.5,
    require_harmful=False,
):
    """Separate final-RANSAC preservation loss over fixed hard teacher labels."""
    candidate_logits = torch.as_tensor(candidate_logits)
    if candidate_logits.ndim != 1:
        raise ValueError("candidate_logits must be one-dimensional")
    if candidate_logits.numel() != targets.useful_mask.numel():
        raise ValueError("candidate logits and hard teacher targets must align")
    temperature = max(float(temperature), 1e-6)
    useful = candidate_logits[targets.useful_mask]
    harmful = candidate_logits[targets.harmful_mask]
    useful_weights = targets.useful_weights[targets.useful_mask].clamp_min(0.0)
    harmful_weights = targets.harmful_weights[targets.harmful_mask].clamp_min(0.0)
    zero = candidate_logits.sum() * 0.0
    if bool(require_harmful) and not harmful.numel():
        # Exact replay is a sparse counterfactual teacher. In the absence of a
        # proven harmful edge, leave the normal candidate/anchor objectives in
        # charge instead of creating a useful-only feature drift force.
        return zero, {
            "hard_teacher_loss_useful": 0.0,
            "hard_teacher_loss_harmful": 0.0,
            "hard_teacher_loss_pairwise": 0.0,
            "hard_teacher_loss_skipped_no_harmful": 1.0,
        }
    useful_loss = (
        (F.softplus((float(score_target) - useful) / temperature) * useful_weights)
        .sum()
        / useful_weights.sum().clamp_min(1e-8)
        if useful.numel()
        else zero
    )
    harmful_loss = (
        (F.softplus((harmful - float(score_target)) / temperature) * harmful_weights)
        .sum()
        / harmful_weights.sum().clamp_min(1e-8)
        if harmful.numel()
        else zero
    )
    if useful.numel() and harmful.numel():
        pair_weights = useful_weights[:, None] * harmful_weights[None, :]
        pair_loss = (
            F.softplus(
                (float(margin) + harmful[None, :] - useful[:, None])
                / temperature
            )
            * pair_weights
        ).sum() / pair_weights.sum().clamp_min(1e-8)
    else:
        pair_loss = zero
    # Individual terms preserve correct RANSAC support even in a clean query;
    # the pairwise term directly suppresses false consensus when it exists.
    loss = 0.25 * useful_loss + 0.25 * harmful_loss + 0.5 * pair_loss
    diagnostics = {
        "hard_teacher_loss_useful": float(useful_loss.detach().item()),
        "hard_teacher_loss_harmful": float(harmful_loss.detach().item()),
        "hard_teacher_loss_pairwise": float(pair_loss.detach().item()),
        "hard_teacher_loss_skipped_no_harmful": 0.0,
    }
    return loss, diagnostics


class HardCandidateTeacherCache:
    """Refresh a discrete teacher every N visits while preserving edge identity."""

    def __init__(self, refresh_visits=2, **teacher_kwargs):
        self.refresh_visits = max(int(refresh_visits), 1)
        self.teacher_kwargs = dict(teacher_kwargs)
        self.records = {}
        self.stats = {
            "queries": 0,
            "refreshes": 0,
            "cache_hits": 0,
            "mapped_target_count": 0,
            "candidate_count": 0,
        }

    def _record_from_targets(self, keys, targets):
        useful = targets.useful_weights.detach().cpu().numpy().astype(np.float32)
        harmful = targets.harmful_weights.detach().cpu().numpy().astype(np.float32)
        selected = (useful > 0.0) | (harmful > 0.0)
        record_keys = np.asarray(keys[selected], dtype=np.int64)
        order = np.argsort(record_keys, kind="stable")
        return {
            "keys": record_keys[order],
            "useful": useful[selected][order],
            "harmful": harmful[selected][order],
            "diagnostics": dict(targets.diagnostics),
        }

    def _remap(self, keys, record, reference):
        targets = _empty_targets(reference, record["diagnostics"])
        if keys.size == 0 or record["keys"].size == 0:
            return targets
        position = np.searchsorted(record["keys"], keys)
        valid = position < record["keys"].size
        valid[valid] &= record["keys"][position[valid]] == keys[valid]
        useful = np.zeros(keys.size, dtype=np.float32)
        harmful = np.zeros(keys.size, dtype=np.float32)
        useful[valid] = record["useful"][position[valid]]
        harmful[valid] = record["harmful"][position[valid]]
        targets.useful_weights = torch.as_tensor(
            useful, dtype=reference.dtype, device=reference.device
        )
        targets.harmful_weights = torch.as_tensor(
            harmful, dtype=reference.dtype, device=reference.device
        )
        targets.useful_mask = targets.useful_weights > 0.0
        targets.harmful_mask = targets.harmful_weights > 0.0
        return targets

    def build(self, query_key, *, keypoint_ids, candidate_keypoint_idx, candidate_landmark_idx, candidate_scores, **kwargs):
        keys = _candidate_keys(
            keypoint_ids,
            candidate_keypoint_idx,
            candidate_landmark_idx,
            torch.as_tensor(kwargs["landmark_xyz"]).shape[0],
        )
        query_key = str(query_key)
        record = self.records.get(query_key)
        visits = 1 if record is None else int(record["visits"]) + 1
        # Visit one is always fresh; every following ``refresh_visits`` visits
        # must refresh as well.  The previous modulo form accidentally cached
        # forever when ``refresh_visits == 1``.
        refresh = record is None or (visits - 1) % self.refresh_visits == 0
        self.stats["queries"] += 1
        self.stats["candidate_count"] += int(keys.size)
        if refresh:
            fresh = derive_hard_candidate_targets(
                keypoint_ids=keypoint_ids,
                candidate_keypoint_idx=candidate_keypoint_idx,
                candidate_landmark_idx=candidate_landmark_idx,
                candidate_scores=candidate_scores,
                **self.teacher_kwargs,
                **kwargs,
            )
            record = self._record_from_targets(keys, fresh)
            record["visits"] = visits
            self.records[query_key] = record
            self.stats["refreshes"] += 1
            targets = fresh
            replay_exception = fresh.diagnostics.get(
                "hard_teacher_exact_replay_exception", 0.0
            )
            if float(replay_exception) > 0.0:
                exception_type = fresh.diagnostics.get(
                    "hard_teacher_exact_replay_exception_type", "Exception"
                )
                exception_message = fresh.diagnostics.get(
                    "hard_teacher_exact_replay_exception_message", ""
                )
                warnings.warn(
                    "Exact hard-candidate replay skipped for "
                    f"{query_key}: {exception_type}: {exception_message}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            targets.diagnostics["hard_teacher_refreshed"] = 1.0
        else:
            record["visits"] = visits
            targets = self._remap(keys, record, candidate_scores)
            self.stats["cache_hits"] += 1
            targets.diagnostics["hard_teacher_refreshed"] = 0.0
        mapped = int((targets.useful_mask | targets.harmful_mask).sum().item())
        self.stats["mapped_target_count"] += mapped
        targets.diagnostics["hard_teacher_cached_target_count"] = float(mapped)
        return targets

    def diagnostics(self):
        queries = max(int(self.stats["queries"]), 1)
        return {
            "hard_teacher_cache_queries": float(self.stats["queries"]),
            "hard_teacher_cache_refreshes": float(self.stats["refreshes"]),
            "hard_teacher_cache_hits": float(self.stats["cache_hits"]),
            "hard_teacher_cache_hit_rate": float(self.stats["cache_hits"]) / queries,
            "hard_teacher_cache_mapped_target_mean": float(
                self.stats["mapped_target_count"]
            )
            / queries,
            "hard_teacher_cache_candidate_mean": float(self.stats["candidate_count"])
            / queries,
        }
