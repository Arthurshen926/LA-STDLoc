from __future__ import annotations

import json

from common.hashing import sha256_file
from scripts.aggregate_v23_cambridge_online_feedback import (
    _apply_test_calibrated_acceptance,
    _feedback_summary,
    _paired_summary,
    _pose_summary,
    _validate_base_report,
)


def _row(image: str, te: float, re: float, **extra: int | float) -> dict:
    return {
        "image_name": image,
        "translation_error_cm": te,
        "rotation_error_deg": re,
        **extra,
    }


def test_paired_summary_tracks_continuous_and_r5_changes() -> None:
    baseline = [_row("a", 5.1, 0.1), _row("b", 4.9, 0.1)]
    candidate = [_row("a", 4.8, 0.1), _row("b", 5.2, 0.1)]
    paired = _paired_summary(baseline, candidate)
    assert paired["translation_improved_query_count"] == 1
    assert paired["translation_worsened_query_count"] == 1
    assert paired["r5_gain_count"] == 1
    assert paired["r5_loss_count"] == 1


def test_feedback_summary_reports_trigger_and_second_solve_rates() -> None:
    rows = [
        _row(
            "a",
            1.0,
            0.1,
            sparse_feedback_eligible=1,
            sparse_feedback_gate_passed=1,
            sparse_feedback_accepted=1,
            sparse_feedback_proposed_rows=10,
            sparse_feedback_supported_rows=4,
        ),
        _row("b", 2.0, 0.2),
    ]
    summary = _feedback_summary(rows)
    assert summary["eligible_query_percent"] == 50.0
    assert summary["supported_row_fraction"] == 0.4
    assert summary["pose_solve_count_per_query_mean"] == 1.5


def test_calibrated_gate_uses_only_online_solver_diagnostics() -> None:
    pose = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    baseline = [
        {
            **_row("a", 5.1, 0.1),
            "inliers": 400,
            "ransac_iterations": 1000,
            "pose_w2c": pose,
        },
        {
            **_row("b", 5.1, 0.1),
            "inliers": 400,
            "ransac_iterations": 1000,
            "pose_w2c": pose,
        },
    ]
    candidate = [
        {
            **_row("a", 4.5, 0.1),
            "inliers": 600,
            "ransac_iterations": 9000,
            "pose_w2c": pose,
        },
        {
            **_row("b", 4.5, 0.1),
            "inliers": 550,
            "ransac_iterations": 9000,
            "pose_w2c": pose,
        },
    ]
    selected, accepted = _apply_test_calibrated_acceptance(
        baseline,
        candidate,
        {
            "minimum_candidate_inlier_gain": 160,
            "minimum_candidate_relative_inlier_gain": 0.2,
            "maximum_candidate_ransac_iterations": 10000,
            "maximum_pose_update_translation_cm": 20.0,
            "maximum_pose_update_rotation_deg": 0.5,
        },
    )
    assert accepted == [True, False]
    assert selected[0] is candidate[0]
    assert selected[1] is baseline[1]


def test_pose_summary_uses_median_te_re_as_primary_continuous_metrics() -> None:
    summary = _pose_summary([_row("a", 1.0, 0.1), _row("b", 3.0, 0.3)])
    assert summary["translation_error_cm"]["median"] == 2.0
    assert summary["rotation_error_deg"]["median"] == 0.2


def test_base_report_rejects_test_consumption(tmp_path) -> None:
    map_path = tmp_path / "map.pt"
    metric_path = tmp_path / "metric.pt"
    map_path.write_bytes(b"map")
    metric_path.write_bytes(b"metric")
    report_path = tmp_path / "report.json"
    report = {
        "schema": "lafgs_rendered_rgb_only_track_probe_report",
        "scientific_scope": {
            "test_queries_used_for_map_construction": False,
            "mapping_source_rgb_loaded": False,
            "mapping_source_rgb_used": False,
            "gaussian_rendered_rgb_used": True,
        },
        "artifacts": {
            "anchor_map": str(map_path),
            "identity_metric": str(metric_path),
        },
        "artifacts_sha256": {
            "anchor_map": sha256_file(map_path),
            "identity_metric": sha256_file(metric_path),
        },
        "selected_map_track_count": 4,
        "query_count": 2,
    }
    report_path.write_text(json.dumps(report))
    assert _validate_base_report(report_path, map_path, metric_path)[
        "selected_anchor_count"
    ] == 4
    report["scientific_scope"]["test_queries_used_for_map_construction"] = True
    report_path.write_text(json.dumps(report))
    try:
        _validate_base_report(report_path, map_path, metric_path)
    except ValueError:
        pass
    else:
        raise AssertionError("test-consuming base map must be rejected")
