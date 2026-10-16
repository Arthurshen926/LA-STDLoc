"""Support-preserving cleanup after pose-aware completion (research v2).

The old track/coverage/Fisher selection stages are unchanged. This additional
final substep prunes redundant stage-2 (matching-completion) landmarks in
ascending original landmark matchability, retaining all stage-1 core and
stage-3 pose-completion landmarks. It preserves, for every mapping camera,
the attained distinct feature-row support capped at the preexisting target.

No query pose, image, test-GT result, or online response is accessed.
This is NOT a guarantee about exact maximum bipartite matching rank or query
localization. Never overwrite frozen v1.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def prune_redundant_coverage(
    state: dict[str, Any],
    selected_source_ids: torch.Tensor,
    selector_trace: dict[str, Any],
    *,
    support_target: int = 256,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if state["provenance"].get("uses_test_queries") is not False:
        raise ValueError("Only mapping-side evidence is valid for selector cleanup")
    if support_target < 1:
        raise ValueError("support_target must be positive")
    selected = torch.as_tensor(selected_source_ids, dtype=torch.long).cpu().numpy()
    if selected.ndim != 1 or selected.size != np.unique(selected).size:
        raise ValueError("invalid selected source IDs")
    reasons = selector_trace["trace"]
    reason_names = {"precision", "matching_completion", "observability_completion"}
    assert {r["primary_reason"] for r in reasons} <= reason_names
    recorded = np.asarray([r["candidate_universe_id"] for r in reasons], dtype=np.int64)
    if len(recorded) != len(selected) or not np.array_equal(
        np.sort(recorded), np.sort(selected)
    ):
        raise ValueError("trace-selected point identity mismatch")
    coverage_ids = np.asarray(
        [r["candidate_universe_id"] for r in reasons
         if r["primary_reason"] == "matching_completion"],
        dtype=np.int64,
    )
    pose_ids = np.asarray(
        [r["candidate_universe_id"] for r in reasons
         if r["primary_reason"] == "observability_completion"],
        dtype=np.int64,
    )
    if np.intersect1d(coverage_ids, pose_ids).size:
        raise ValueError("trace stages have duplicate source points")
    quality = np.asarray(state["anchor_matchability"], dtype=np.float64)
    if not np.isfinite(quality).all():
        raise ValueError("quality contains non-finite scores")
    csr = state["projective_anchor_observations"]
    offsets = np.asarray(csr["observation_offsets"], dtype=np.int64)
    view_ids = np.asarray(csr["query_indices"], dtype=np.int64)
    keypoint_ids = np.asarray(csr["keypoint_indices"], dtype=np.int64)
    anchor_count = len(offsets) - 1
    if selected.size == 0 or selected.min() < 0 or selected.max() >= anchor_count:
        raise ValueError("selected rows are outside candidate pool")
    if len(quality) != anchor_count or offsets[-1] != len(view_ids) or len(keypoint_ids) != len(view_ids):
        raise ValueError("mapping observation metadata is inconsistent")
    if len(view_ids) == 0 or view_ids.min() < 0 or keypoint_ids.min() < 0:
        raise ValueError("invalid observation rows")
    num_views = len(state["v6_mapping_query_names"])
    if view_ids.max() >= num_views:
        raise ValueError("unknown reference view")
    # The original mapping cache stores at most 2048 native keypoint rows per
    # view. This dynamically sized matrix also works for reduced-detector maps.
    row_count = int(keypoint_ids.max()) + 1
    counts = np.zeros((num_views, row_count), dtype=np.int32)
    observations = np.concatenate(
        [np.arange(offsets[int(i)], offsets[int(i) + 1], dtype=np.int64) for i in selected]
    )
    np.add.at(counts, (view_ids[observations], keypoint_ids[observations]), 1)
    unique_counts = np.count_nonzero(counts, axis=1).astype(np.int32)
    target = np.minimum(unique_counts, support_target)
    original_support = unique_counts.copy()
    # Comparable to the existing pilot, no score learning or test tuning.
    ordered = coverage_ids[
        np.lexsort((coverage_ids, quality[coverage_ids]))
    ]
    removed: list[int] = []
    for item in ordered:
        begin, end = int(offsets[item]), int(offsets[item + 1])
        views, rows = view_ids[begin:end], keypoint_ids[begin:end]
        if np.unique(views).size != views.size:
            raise ValueError("multiple mapping observations of one candidate in one camera")
        active = counts[views, rows]
        if not np.all(active >= 1):
            raise ValueError("an observation was subtracted more than once")
        loss = (active == 1).astype(np.int32)
        if np.all(unique_counts[views] - loss >= target[views]):
            counts[views, rows] -= 1
            unique_counts[views] -= loss
            removed.append(int(item))
    result = np.setdiff1d(
        selected, np.asarray(removed, dtype=np.int64), assume_unique=True
    )
    if not np.array_equal(unique_counts, np.count_nonzero(counts, axis=1)):
        raise AssertionError("distinct reference support accounting drifted")
    if np.any(unique_counts < target):
        raise AssertionError("reference-view support target decreased")
    if np.intersect1d(result, pose_ids).size != len(pose_ids):
        raise AssertionError("pose completion rows must remain intact")
    summary = dict(
        version="compact-fast-v2-support-preserving-cleanup",
        stage="post_pose_completion_redundancy_cleanup",
        quality_order="ascending_anchor_matchability",
        source_row_tie_break=True,
        target_support_per_reference=int(support_target),
        achieved_support_at_start_capped_to_target=True,
        original_selected_count=int(len(selected)),
        removed_coverage_count=int(len(removed)),
        retained_count=int(len(result)),
        original_under_supported_reference_views=int(
            np.sum(original_support < support_target)
        ),
        retained_under_supported_reference_views=int(
            np.sum(unique_counts < support_target)
        ),
        pose_completion_retained_count=int(len(pose_ids)),
        removed_source_ids=torch.as_tensor(removed, dtype=torch.long),
        no_test_query_information=True,
        does_not_recompute_pose_completion=True,
        metric_boundary="unique_mapping_2D_feature_rows_not_exact_bipartite_rank",
    )
    return torch.as_tensor(result, dtype=torch.long), summary
