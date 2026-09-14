from __future__ import annotations

import numpy as np

from map_learning.v28_best_of_two_oracle import audit_best_of_two


def _pose(x: float) -> list[list[float]]:
    value = np.eye(4)
    value[0, 3] = x
    return value.tolist()


def test_oracle_selects_candidate_by_final_pose_task_error() -> None:
    gt = _pose(0.0)
    first = [
        {
            "image_name": "a",
            "gt_pose_w2c": gt,
            "rotation_error_deg": 0.0,
            "translation_error_cm": 10.0,
        }
    ]
    refined = [
        {
            "image_name": "a",
            "gt_pose_w2c": gt,
            "rotation_error_deg": 0.0,
            "translation_error_cm": 10.0,
            "sparse_feedback_candidate_pose_w2c": _pose(0.01),
            "sparse_feedback_accepted": False,
        }
    ]
    result = audit_best_of_two(
        first_pass_records=first, refinement_records=refined
    )
    assert result["oracle_t1_selection_count"] == 1
    assert result["actual_oracle_selection_disagreement_count"] == 1
    assert result["best_of_two_oracle"]["median_te_cm"] == 1.0

