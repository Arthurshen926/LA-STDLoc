"""Offline oracle for structured-outlier-aware absolute pose consensus.

This module is intentionally not wired into deployment.  It measures whether
the existing correspondence set contains a correct hypothesis that standard
inlier-count scoring loses, and whether group-diverse minimal sets add such a
hypothesis.  A positive oracle result is required before replacing PoseLib.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from localization.pose_solver import pose_error


@dataclass(frozen=True)
class HypothesisScores:
    standard_inlier_count: np.ndarray
    group_inlier_count: np.ndarray
    inlier_residual_sum: np.ndarray
    group_min_residual_sum: np.ndarray


def _validated_residuals_and_groups(
    residuals: np.ndarray, group_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.asarray(residuals, dtype=np.float64)
    groups = np.asarray(group_ids)
    if residuals.ndim != 2:
        raise ValueError("hypothesis residuals must have shape [H,N]")
    if groups.dtype.kind not in "iu" or groups.ndim != 1:
        raise ValueError("correlation group IDs must be an integer vector")
    if residuals.shape[1] != groups.shape[0]:
        raise ValueError("residual and correlation-group columns do not align")
    if not np.isfinite(residuals).all() or np.any(residuals < 0):
        raise ValueError("hypothesis residuals must be finite and non-negative")
    if groups.size and np.any(groups < 0):
        raise ValueError("correlation group IDs must be non-negative")
    return residuals, groups.astype(np.int64, copy=False)


def score_hypothesis_residuals(
    residuals: np.ndarray,
    group_ids: np.ndarray,
    *,
    threshold_px: float,
) -> HypothesisScores:
    """Score hypotheses with standard and one-vote-per-group consensus."""

    residuals, groups = _validated_residuals_and_groups(residuals, group_ids)
    threshold = float(threshold_px)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("reprojection threshold must be positive and finite")
    inlier = residuals <= threshold
    standard_count = inlier.sum(axis=1, dtype=np.int64)
    clipped = np.minimum(residuals, threshold)
    inlier_residual_sum = (clipped * inlier).sum(axis=1)
    unique_groups = np.unique(groups)
    group_count = np.zeros(residuals.shape[0], dtype=np.int64)
    group_min_sum = np.zeros(residuals.shape[0], dtype=np.float64)
    for group in unique_groups:
        group_residual = residuals[:, groups == group].min(axis=1)
        group_inlier = group_residual <= threshold
        group_count += group_inlier.astype(np.int64)
        group_min_sum += np.where(group_inlier, group_residual, threshold)
    return HypothesisScores(
        standard_inlier_count=standard_count,
        group_inlier_count=group_count,
        inlier_residual_sum=inlier_residual_sum,
        group_min_residual_sum=group_min_sum,
    )


def select_standard_hypothesis(scores: HypothesisScores) -> int:
    """Standard RANSAC ordering: inlier count, then lower inlier residual."""

    if scores.standard_inlier_count.size == 0:
        raise ValueError("cannot rank an empty hypothesis set")
    order = np.lexsort(
        (
            np.arange(scores.standard_inlier_count.size),
            scores.inlier_residual_sum,
            -scores.standard_inlier_count,
        )
    )
    return int(order[0])


def select_group_capped_hypothesis(scores: HypothesisScores) -> int:
    """Rank by independent groups before raw correspondence multiplicity."""

    if scores.group_inlier_count.size == 0:
        raise ValueError("cannot rank an empty hypothesis set")
    order = np.lexsort(
        (
            np.arange(scores.group_inlier_count.size),
            scores.inlier_residual_sum,
            -scores.standard_inlier_count,
            scores.group_min_residual_sum,
            -scores.group_inlier_count,
        )
    )
    return int(order[0])


def reprojection_residuals(
    hypotheses_w2c: np.ndarray,
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    hypotheses = np.asarray(hypotheses_w2c, dtype=np.float64)
    points_2d = np.asarray(points_2d, dtype=np.float64)
    points_3d = np.asarray(points_3d, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if hypotheses.ndim != 3 or hypotheses.shape[1:] != (4, 4):
        raise ValueError("pose hypotheses must have shape [H,4,4]")
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError("2D points must have shape [N,2]")
    if points_3d.shape != (points_2d.shape[0], 3):
        raise ValueError("3D points must align with 2D points")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic matrix must have shape [3,3]")
    camera = (
        np.einsum("hij,nj->hni", hypotheses[:, :3, :3], points_3d)
        + hypotheses[:, None, :3, 3]
    )
    projected = np.einsum("ij,hnj->hni", intrinsic, camera)
    valid = projected[:, :, 2] > 1e-12
    uv = projected[:, :, :2] / np.maximum(
        projected[
            :,
            :,
            2:,
        ],
        1e-12,
    )
    residual = np.linalg.norm(uv - points_2d[None], axis=2)
    residual[~valid] = np.finfo(np.float64).max / 4.0
    return residual


def group_diverse_minimal_sets(
    group_ids: np.ndarray,
    *,
    sample_size: int = 4,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Deterministically sample minimal sets with distinct correlation groups."""

    groups = np.asarray(group_ids)
    if groups.dtype.kind not in "iu" or groups.ndim != 1:
        raise ValueError("correlation group IDs must be an integer vector")
    if groups.size and np.any(groups < 0):
        raise ValueError("correlation group IDs must be non-negative")
    unique = np.unique(groups)
    if int(sample_size) < 4 or unique.size < int(sample_size):
        return np.empty((0, int(sample_size)), dtype=np.int64)
    if int(sample_count) < 0:
        raise ValueError("sample count cannot be negative")
    rows_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    generator = np.random.default_rng(int(seed))
    samples = []
    seen: set[tuple[int, ...]] = set()
    maximum_attempts = max(int(sample_count) * 20, 100)
    for _ in range(maximum_attempts):
        chosen_groups = generator.choice(unique, size=int(sample_size), replace=False)
        rows = tuple(
            sorted(
                int(generator.choice(rows_by_group[int(group)]))
                for group in chosen_groups
            )
        )
        if rows in seen:
            continue
        seen.add(rows)
        samples.append(rows)
        if len(samples) == int(sample_count):
            break
    return np.asarray(samples, dtype=np.int64).reshape(-1, int(sample_size))


def _pnp_hypotheses_for_samples(
    samples: np.ndarray,
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    hypotheses = []
    distortion = np.zeros(4, dtype=np.float64)
    for rows in np.asarray(samples, dtype=np.int64):
        success, rotations, translations, _ = cv2.solvePnPGeneric(
            np.asarray(points_3d[rows], dtype=np.float64),
            np.asarray(points_2d[rows], dtype=np.float64),
            np.asarray(intrinsic, dtype=np.float64),
            distortion,
            flags=cv2.SOLVEPNP_AP3P,
        )
        if not success:
            continue
        for rotation, translation in zip(rotations, translations):
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = cv2.Rodrigues(rotation)[0]
            matrix[:3, 3] = np.asarray(translation).reshape(3)
            if np.isfinite(matrix).all():
                hypotheses.append(matrix)
    return np.stack(hypotheses) if hypotheses else np.empty((0, 4, 4), dtype=np.float64)


def build_standard_and_group_diverse_hypotheses(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    group_ids: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build matched-size random and group-diverse PnP hypothesis pools."""

    points_2d = np.asarray(points_2d)
    points_3d = np.asarray(points_3d)
    count = int(points_2d.shape[0])
    if count < 4:
        empty = np.empty((0, 4, 4), dtype=np.float64)
        return empty, empty.copy()
    generator = np.random.default_rng(int(seed))
    standard_sets = np.stack(
        [generator.choice(count, size=4, replace=False) for _ in range(sample_count)]
    )
    diverse_sets = group_diverse_minimal_sets(
        group_ids, sample_size=4, sample_count=sample_count, seed=seed
    )
    return (
        _pnp_hypotheses_for_samples(standard_sets, points_2d, points_3d, intrinsic),
        _pnp_hypotheses_for_samples(diverse_sets, points_2d, points_3d, intrinsic),
    )


def build_group_diverse_hypotheses(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    group_ids: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Build only the bounded, distinct-group AP3P supplement."""

    samples = group_diverse_minimal_sets(
        group_ids, sample_size=4, sample_count=sample_count, seed=seed
    )
    return _pnp_hypotheses_for_samples(samples, points_2d, points_3d, intrinsic)


def correlation_groups_from_map(
    state: dict, anchor_rows: np.ndarray, *, field: str | None = None
) -> np.ndarray:
    """Resolve one auditable structured-outlier group per matched Anchor row."""

    rows = np.asarray(anchor_rows)
    if rows.dtype.kind not in "iu" or rows.ndim != 1:
        raise ValueError("anchor rows must be an integer vector")
    count = int(np.asarray(state["anchor_ids"]).size)
    if rows.size and (np.any(rows < 0) or np.any(rows >= count)):
        raise ValueError("anchor row is outside map")
    if field is not None and field not in state:
        raise ValueError(f"requested map correlation field is missing: {field}")
    field = field or next(
        (
            name
            for name in (
                "anchor_correlation_group_ids",
                "parent_source_track_ids",
                "source_dependency_group_ids",
                "coarse_dependency_group_ids",
                "dependency_group_ids",
            )
            if name in state
        ),
        None,
    )
    groups = (
        np.arange(count, dtype=np.int64)
        if field is None
        else np.asarray(state[field], dtype=np.int64)
    )
    if groups.shape != (count,):
        raise ValueError(f"map correlation field {field} does not align")
    # Unknown lineage is independent, never one giant artificial group.
    groups = groups.copy()
    unknown = groups < 0
    offset = int(groups[~unknown].max()) + 1 if np.any(~unknown) else 0
    groups[unknown] = offset + np.arange(count, dtype=np.int64)[unknown]
    return groups[rows.astype(np.int64, copy=False)]


def classify_hypothesis_oracle(
    *,
    standard_hypotheses_w2c: np.ndarray,
    group_diverse_hypotheses_w2c: np.ndarray,
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    group_ids: np.ndarray,
    ground_truth_w2c: np.ndarray,
    reprojection_threshold_px: float,
    correct_translation_cm: float = 5.0,
    correct_rotation_deg: float = 5.0,
) -> dict:
    """Classify scorer headroom (A), sampler headroom (B), or no headroom (C)."""

    standard = np.asarray(standard_hypotheses_w2c, dtype=np.float64)
    diverse = np.asarray(group_diverse_hypotheses_w2c, dtype=np.float64)

    def evaluate(hypotheses: np.ndarray) -> tuple[np.ndarray, HypothesisScores]:
        if hypotheses.shape[0] == 0:
            empty = np.empty((0, points_2d.shape[0]))
            return np.empty(0, dtype=bool), score_hypothesis_residuals(
                empty, group_ids, threshold_px=reprojection_threshold_px
            )
        errors = np.asarray([pose_error(pose, ground_truth_w2c) for pose in hypotheses])
        correct = (errors[:, 0] <= float(correct_rotation_deg)) & (
            errors[:, 1] <= float(correct_translation_cm)
        )
        scores = score_hypothesis_residuals(
            reprojection_residuals(hypotheses, points_2d, points_3d, intrinsic),
            group_ids,
            threshold_px=reprojection_threshold_px,
        )
        return correct, scores

    standard_correct, standard_scores = evaluate(standard)
    diverse_correct, diverse_scores = evaluate(diverse)
    standard_has_correct = bool(standard_correct.any())
    diverse_has_correct = bool(diverse_correct.any())
    if standard.shape[0]:
        standard_winner = select_standard_hypothesis(standard_scores)
        group_winner = select_group_capped_hypothesis(standard_scores)
        standard_winner_correct = bool(standard_correct[standard_winner])
        group_winner_correct = bool(standard_correct[group_winner])
    else:
        standard_winner = group_winner = None
        standard_winner_correct = group_winner_correct = False

    def winner_correct(
        correct: np.ndarray, scores: HypothesisScores, *, group_capped: bool
    ) -> bool:
        if correct.size == 0:
            return False
        selector = (
            select_group_capped_hypothesis
            if group_capped
            else select_standard_hypothesis
        )
        return bool(correct[selector(scores)])

    diverse_standard_winner_correct = winner_correct(
        diverse_correct, diverse_scores, group_capped=False
    )
    diverse_group_winner_correct = winner_correct(
        diverse_correct, diverse_scores, group_capped=True
    )
    combined = np.concatenate((standard, diverse), axis=0)
    combined_correct, combined_scores = evaluate(combined)
    combined_standard_winner_correct = winner_correct(
        combined_correct, combined_scores, group_capped=False
    )
    combined_group_winner_correct = winner_correct(
        combined_correct, combined_scores, group_capped=True
    )
    if standard_has_correct and not standard_winner_correct and group_winner_correct:
        category = "A_GROUP_SCORER_HEADROOM"
    elif not standard_has_correct and diverse_has_correct:
        category = "B_GROUP_DIVERSE_SAMPLING_HEADROOM"
    elif not standard_has_correct and not diverse_has_correct:
        category = "C_NO_HYPOTHESIS_UPPER_BOUND"
    else:
        category = "NO_ACTIONABLE_GROUP_ORACLE_GAIN"
    return {
        "category": category,
        "standard_hypothesis_count": int(standard.shape[0]),
        "group_diverse_hypothesis_count": int(diverse.shape[0]),
        "standard_has_correct_hypothesis": standard_has_correct,
        "group_diverse_has_correct_hypothesis": diverse_has_correct,
        "standard_winner_index": standard_winner,
        "group_capped_winner_index": group_winner,
        "standard_winner_correct": standard_winner_correct,
        "group_capped_winner_correct": group_winner_correct,
        "diverse_standard_winner_correct": diverse_standard_winner_correct,
        "diverse_group_capped_winner_correct": diverse_group_winner_correct,
        "combined_standard_winner_correct": combined_standard_winner_correct,
        "combined_group_capped_winner_correct": combined_group_winner_correct,
        "authorizes_deployment_solver_change": False,
    }
