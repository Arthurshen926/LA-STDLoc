"""Read-only best-of-two pose audit for V26 online sparse refinement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from evaluation.metrics import pose_error, summarize_pose_errors


def _summary(rows: Sequence[tuple[float, float]]) -> dict:
    rotation = [value[0] for value in rows]
    translation = [value[1] for value in rows]
    result = summarize_pose_errors(rotation, translation)
    result["catastrophic_100cm_count"] = int(
        np.count_nonzero(np.asarray(translation) >= 100.0)
    )
    return result


def audit_best_of_two(
    *,
    first_pass_records: Sequence[Mapping],
    refinement_records: Sequence[Mapping],
) -> dict:
    """Compare deployed selection with a GT-only T0/T1 upper bound.

    The oracle minimizes ``max(TE/5cm, RE/5deg)`` and keeps T0 on exact ties.
    Queries without a materialized T1 necessarily retain T0.
    """

    first = {str(record["image_name"]): record for record in first_pass_records}
    if len(first) != len(first_pass_records) or len(first) != len(refinement_records):
        raise ValueError("V28 T0/refinement query registries differ")
    t0_errors: list[tuple[float, float]] = []
    actual_errors: list[tuple[float, float]] = []
    oracle_errors: list[tuple[float, float]] = []
    candidate_count = 0
    oracle_t1_count = 0
    t1_dominates = 0
    t0_dominates = 0
    tradeoff = 0
    actual_disagrees = 0
    records = []
    for refined in refinement_records:
        name = str(refined["image_name"])
        if name not in first:
            raise ValueError("V28 refinement query is missing from T0")
        baseline = first[name]
        gt0 = np.asarray(baseline["gt_pose_w2c"], dtype=np.float64)
        gt1 = np.asarray(refined["gt_pose_w2c"], dtype=np.float64)
        if not np.array_equal(gt0, gt1):
            raise ValueError("V28 ground-truth pose binding differs")
        t0 = (
            float(baseline["rotation_error_deg"]),
            float(baseline["translation_error_cm"]),
        )
        actual = (
            float(refined["rotation_error_deg"]),
            float(refined["translation_error_cm"]),
        )
        candidate_pose = refined.get("sparse_feedback_candidate_pose_w2c")
        selected = "T0"
        t1 = None
        if candidate_pose is not None:
            candidate_count += 1
            t1 = pose_error(np.asarray(candidate_pose, dtype=np.float64), gt1)
            if t1[0] < t0[0] and t1[1] < t0[1]:
                t1_dominates += 1
            elif t0[0] < t1[0] and t0[1] < t1[1]:
                t0_dominates += 1
            else:
                tradeoff += 1
            task0 = max(t0[0] / 5.0, t0[1] / 5.0)
            task1 = max(t1[0] / 5.0, t1[1] / 5.0)
            if task1 < task0:
                selected = "T1"
                oracle_t1_count += 1
        oracle = t1 if selected == "T1" else t0
        actual_selected = "T1" if bool(refined["sparse_feedback_accepted"]) else "T0"
        actual_disagrees += int(actual_selected != selected)
        t0_errors.append(t0)
        actual_errors.append(actual)
        oracle_errors.append(oracle)
        records.append(
            {
                "image_name": name,
                "t0_re_deg": t0[0],
                "t0_te_cm": t0[1],
                "t1_available": t1 is not None,
                "t1_re_deg": None if t1 is None else float(t1[0]),
                "t1_te_cm": None if t1 is None else float(t1[1]),
                "actual_selection": actual_selected,
                "oracle_selection": selected,
            }
        )
    return {
        "query_count": len(records),
        "candidate_pose_count": candidate_count,
        "oracle_t1_selection_count": oracle_t1_count,
        "t1_strictly_improves_te_and_re_count": t1_dominates,
        "t0_strictly_improves_te_and_re_count": t0_dominates,
        "te_re_tradeoff_or_tie_count": tradeoff,
        "actual_oracle_selection_disagreement_count": actual_disagrees,
        "first_pass": _summary(t0_errors),
        "actual_v26": _summary(actual_errors),
        "best_of_two_oracle": _summary(oracle_errors),
        "records": records,
    }

