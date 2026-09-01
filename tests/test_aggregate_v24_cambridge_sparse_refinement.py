from __future__ import annotations

import numpy as np

from scripts.aggregate_v24_cambridge_sparse_refinement import (
    _candidate_rows,
    _feedback_timing_summary,
    _passes_gate,
)


def _pose(tx: float = 0.0) -> list[list[float]]:
    value = np.eye(4, dtype=np.float32)
    value[0, 3] = tx
    return value.tolist()


def test_candidate_pose_is_remeasured_against_ground_truth() -> None:
    baseline = [
        {
            "image_name": "seq1/frame00001.png",
            "gt_pose_w2c": _pose(),
            "pose_w2c": _pose(0.02),
            "translation_error_cm": 2.0,
            "rotation_error_deg": 0.0,
            "inliers": 20,
        }
    ]
    refinement = [
        {
            **baseline[0],
            "sparse_feedback_candidate_pose_w2c": _pose(0.01),
            "sparse_feedback_candidate_inliers": 24,
            "sparse_feedback_candidate_inlier_gain": 4,
            "sparse_feedback_gate_passed": 1,
        }
    ]

    candidate = _candidate_rows(baseline, refinement)[0]

    assert candidate is not None
    assert np.isclose(candidate["translation_error_cm"], 1.0)
    assert np.isclose(candidate["rotation_error_deg"], 0.0)


def test_gate_uses_only_runtime_diagnostics() -> None:
    record = {
        "inliers": 100,
        "sparse_feedback_candidate_inliers": 105,
        "sparse_feedback_candidate_pose_w2c": _pose(),
        "sparse_feedback_support_passed": 1,
        "sparse_feedback_candidate_inlier_gain": 5,
        "sparse_feedback_baseline_inlier_retention_fraction": 0.97,
        "sparse_feedback_protected_median_residual_increase_px": 0.1,
        "sparse_feedback_protected_p90_residual_increase_px": 0.4,
        "sparse_feedback_pose_update_translation_cm": 1.5,
        "sparse_feedback_pose_update_rotation_deg": 0.04,
    }
    config = {
        "minimum_candidate_inlier_gain": 4,
        "minimum_candidate_relative_inlier_gain": 0.05,
        "minimum_baseline_inlier_retention": 0.95,
        "maximum_protected_median_residual_increase_px": 0.25,
        "maximum_protected_p90_residual_increase_px": 0.5,
        "maximum_pose_update_translation_cm": 2.0,
        "maximum_pose_update_rotation_deg": 0.05,
        "maximum_candidate_ransac_iterations": 2_000,
    }

    assert _passes_gate(record, config)
    record["translation_error_cm"] = 1e9
    record["rotation_error_deg"] = 1e9
    record["gt_pose_w2c"] = _pose(123.0)
    assert _passes_gate(record, config)
    record["sparse_feedback_candidate_inlier_gain"] = 3
    assert not _passes_gate(record, config)


def test_feedback_timing_summary_reports_true_marginal_cost() -> None:
    rows = [
        {
            "feedback_geometry_ms": 2.0,
            "feedback_ransac_ms": 3.0,
            "sparse_feedback_candidate_pose_w2c": [[1.0]],
        },
        {
            "feedback_geometry_ms": 4.0,
            "feedback_ransac_ms": 0.0,
            "sparse_feedback_candidate_pose_w2c": None,
        },
    ]

    summary = _feedback_timing_summary(rows)

    assert summary["geometry_ms"]["mean"] == 3.0
    assert summary["second_ransac_ms"]["mean"] == 1.5
    assert summary["combined_marginal_ms"]["mean"] == 4.5
    assert summary["second_solve_count"] == 1
    assert summary["second_solve_percent"] == 50.0
