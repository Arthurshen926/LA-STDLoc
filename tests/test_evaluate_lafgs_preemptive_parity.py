import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_lafgs_preemptive_parity.py"
)
SPEC = importlib.util.spec_from_file_location("preemptive_parity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(name, *, ransac_ms, total_ms, reduction=0.0):
    return {
        "image_name": name,
        "sparse_TE": 1.0,
        "sparse_AE": 0.1,
        "sparse": {
            "pose_w2c": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "inliers": 42,
            "sparse_diag_ransac_actual_hypotheses": 1000,
            "sparse_diag_ransac_refinements": 3,
            "sparse_diag_runtime_ransac_ms": ransac_ms,
            "sparse_diag_runtime_total_ms": total_ms,
            "sparse_diag_preemptive_residual_reduction": reduction,
        },
    }


def test_reports_exact_parity_and_runtime_improvement(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps([_row("q1", ransac_ms=10, total_ms=20)]))
    candidate.write_text(
        json.dumps(
            [_row("q1", ransac_ms=8, total_ms=18, reduction=0.2)]
        )
    )
    result = MODULE.compare_results(baseline, candidate)
    assert result["success"] == {
        "exact_pose_solver_parity": True,
        "runtime_improved": True,
        "deployable": True,
    }
    assert result["candidate"]["residual_evaluation_reduction_mean"] == 0.2
    assert result["ransac_runtime_delta_percent"] == pytest.approx(-20.0)
