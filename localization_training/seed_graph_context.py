"""Candidate-conditioned context utilities for sparse landmark assignment."""

import numpy as np
from scipy import sparse


def camera_center_and_direction(pose_w2c):
    pose = np.asarray(pose_w2c, dtype=np.float64).reshape(4, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    center = -rotation.T @ translation
    direction = rotation.T @ np.array([0.0, 0.0, 1.0])
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return center, direction


def build_landmark_outcomes(records, landmark_count):
    """Build per-view selected correct/false landmark outcomes."""
    correct = sparse.lil_matrix(
        (len(records), int(landmark_count)), dtype=np.int8
    )
    false = sparse.lil_matrix(
        (len(records), int(landmark_count)), dtype=np.int8
    )
    correct_sets = []
    false_sets = []
    for view_index, record in enumerate(records):
        candidate = np.asarray(record["top_indices"], dtype=np.int64)
        positive = np.asarray(record["positive_mask"], dtype=bool)
        probability = np.asarray(
            record["teacher_probability"], dtype=np.float64
        )
        selected = probability.argmax(axis=1)
        rows = np.arange(selected.size)
        landmark = candidate[rows, selected]
        selected_correct = positive[rows, selected]
        correct_ids = np.unique(landmark[selected_correct])
        false_ids = np.unique(landmark[~selected_correct])
        # A correct observation is stronger evidence than another repeated row
        # selecting the same identity incorrectly in the same support image.
        false_ids = np.setdiff1d(false_ids, correct_ids, assume_unique=True)
        correct[view_index, correct_ids] = 1
        false[view_index, false_ids] = 1
        correct_sets.append(correct_ids)
        false_sets.append(false_ids)
    return correct.tocsr(), false.tocsr(), correct_sets, false_sets


def build_positive_pmi_graph(correct_incidence, minimum_cohits=2):
    """Return a sparse positive-PMI graph from correct support co-hits."""
    incidence = sparse.csr_matrix(correct_incidence, dtype=np.int32)
    view_count, landmark_count = incidence.shape
    if view_count <= 0:
        raise ValueError("at least one support view is required")
    marginal = np.asarray(incidence.sum(axis=0)).reshape(-1).astype(np.float64)
    cohit = (incidence.T @ incidence).tocoo()
    off_diagonal = (
        (cohit.row != cohit.col)
        & (cohit.data >= int(minimum_cohits))
        & (marginal[cohit.row] > 0)
        & (marginal[cohit.col] > 0)
    )
    row = cohit.row[off_diagonal]
    col = cohit.col[off_diagonal]
    count = cohit.data[off_diagonal].astype(np.float64)
    value = np.log(
        (count * float(view_count))
        / (marginal[row] * marginal[col])
    )
    positive = value > 0.0
    graph = sparse.csr_matrix(
        (
            value[positive].astype(np.float32),
            (row[positive], col[positive]),
        ),
        shape=(landmark_count, landmark_count),
    )
    graph.sum_duplicates()
    return graph


def graph_candidate_scores(graph, candidate_ids, seed_ids, seed_weights=None):
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    seed_ids = np.asarray(seed_ids, dtype=np.int64).reshape(-1)
    if seed_ids.size == 0:
        return np.zeros(candidate_ids.shape, dtype=np.float64)
    weights = (
        np.ones(seed_ids.size, dtype=np.float64)
        if seed_weights is None
        else np.asarray(seed_weights, dtype=np.float64).reshape(-1)
    )
    if weights.size != seed_ids.size:
        raise ValueError("seed weights must align with seed IDs")
    unique_candidates, inverse = np.unique(
        candidate_ids.reshape(-1), return_inverse=True
    )
    scores = np.asarray(
        graph[unique_candidates][:, seed_ids] @ weights
    ).reshape(-1)
    return scores[inverse].reshape(candidate_ids.shape)


def confusion_candidate_scores(
    correct_incidence,
    false_incidence,
    candidate_ids,
    seed_ids,
    seed_weights=None,
):
    """Positive PMI that a candidate is false when a seed is correct."""
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    seed_ids = np.asarray(seed_ids, dtype=np.int64).reshape(-1)
    if seed_ids.size == 0:
        return np.zeros(candidate_ids.shape, dtype=np.float64)
    weights = (
        np.ones(seed_ids.size, dtype=np.float64)
        if seed_weights is None
        else np.asarray(seed_weights, dtype=np.float64).reshape(-1)
    )
    unique_candidates, inverse = np.unique(
        candidate_ids.reshape(-1), return_inverse=True
    )
    correct = sparse.csr_matrix(correct_incidence, dtype=np.int32)
    false = sparse.csr_matrix(false_incidence, dtype=np.int32)
    cohit = (
        false[:, unique_candidates].T @ correct[:, seed_ids]
    ).toarray().astype(np.float64)
    candidate_count = np.asarray(
        false[:, unique_candidates].sum(axis=0)
    ).reshape(-1)
    seed_count = np.asarray(
        correct[:, seed_ids].sum(axis=0)
    ).reshape(-1)
    denominator = candidate_count[:, None] * seed_count[None]
    pmi = np.zeros_like(cohit)
    valid = (cohit > 0.0) & (denominator > 0.0)
    pmi[valid] = np.log(
        cohit[valid] * float(correct.shape[0]) / denominator[valid]
    )
    pmi = np.maximum(pmi, 0.0)
    score = pmi @ weights
    return score[inverse].reshape(candidate_ids.shape)


def normalized_candidate_context(context):
    """Normalize only within a candidate row, preserving identity ordering."""
    context = np.asarray(context, dtype=np.float64)
    centered = context - np.median(context, axis=1, keepdims=True)
    scale = np.max(np.abs(centered), axis=1, keepdims=True)
    return np.divide(
        centered,
        np.maximum(scale, 1e-12),
        out=np.zeros_like(centered),
        where=np.isfinite(scale),
    )


def apply_bounded_context(
    candidate_logits,
    context,
    *,
    delta_max,
    protected_rows=None,
):
    logits = np.asarray(candidate_logits, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    if logits.shape != context.shape or logits.ndim != 2:
        raise ValueError("candidate logits and context must be equal NxK arrays")
    adjusted = logits + float(delta_max) * normalized_candidate_context(
        context
    )
    selected = adjusted.argmax(axis=1)
    if protected_rows is not None:
        protected = np.asarray(protected_rows, dtype=bool).reshape(-1)
        if protected.size != logits.shape[0]:
            raise ValueError("protected rows do not align with logits")
        selected[protected] = logits[protected].argmax(axis=1)
    return selected, adjusted


def assignment_metrics(
    candidate_positive,
    baseline_selected,
    selected,
    matchable_rows=None,
):
    positive = np.asarray(candidate_positive, dtype=bool)
    baseline = np.asarray(baseline_selected, dtype=np.int64).reshape(-1)
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if positive.ndim != 2 or baseline.size != positive.shape[0]:
        raise ValueError("assignment arrays do not align")
    rows = np.arange(positive.shape[0])
    has_positive = positive.any(axis=1)
    matchable = (
        has_positive
        if matchable_rows is None
        else np.asarray(matchable_rows, dtype=bool).reshape(-1)
    )
    if matchable.size != positive.shape[0]:
        raise ValueError("matchable rows do not align with assignments")
    baseline_correct = positive[rows, baseline]
    selected_correct = positive[rows, selected]
    beneficial = ~baseline_correct & selected_correct
    harmful = baseline_correct & ~selected_correct
    clean_identity_retained = baseline_correct & (selected == baseline)
    return {
        "rows": int(rows.size),
        "positive_in_topk_rate_all_rows": float(has_positive.mean()),
        "positive_in_topk_given_matchable": float(
            has_positive[matchable].mean() if matchable.any() else 0.0
        ),
        "topk_conditional_selection_accuracy": float(
            selected_correct[has_positive].mean()
            if has_positive.any()
            else 0.0
        ),
        "baseline_topk_conditional_selection_accuracy": float(
            baseline_correct[has_positive].mean()
            if has_positive.any()
            else 0.0
        ),
        "conditional_recall_at_1_given_matchable": float(
            selected_correct[matchable].mean()
            if matchable.any()
            else 0.0
        ),
        "baseline_conditional_recall_at_1_given_matchable": float(
            baseline_correct[matchable].mean()
            if matchable.any()
            else 0.0
        ),
        "clean_top1_retention": float(
            clean_identity_retained.sum() / max(int(baseline_correct.sum()), 1)
        ),
        "beneficial_swaps": int(beneficial.sum()),
        "harmful_swaps": int(harmful.sum()),
        "beneficial_harmful_ratio": float(
            beneficial.sum() / max(int(harmful.sum()), 1)
        ),
        "changed_rate": float((selected != baseline).mean()),
    }
