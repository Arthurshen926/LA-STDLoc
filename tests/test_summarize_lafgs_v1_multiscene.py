import numpy as np

from scripts.summarize_lafgs_v1_multiscene import (
    _metric_vector,
    _paired_bootstrap,
)


def test_metric_vector_reports_pose_error_and_recall():
    values = np.asarray([1.0, 2.0, 4.0, 10.0])
    metrics = _metric_vector(values)
    np.testing.assert_allclose(
        metrics,
        np.asarray([3.0, 4.25, 8.2, 50.0, 75.0]),
    )


def test_paired_bootstrap_preserves_improvement_direction():
    baseline = np.linspace(2.0, 20.0, 64)
    candidate = baseline - 1.0
    result = _paired_bootstrap(
        baseline,
        candidate,
        samples=500,
        seed=2026,
    )
    assert result["median_te_cm"]["delta"] < 0
    assert result["mean_te_cm"]["improvement_probability"] == 1.0
    assert result["r5_pp"]["delta"] > 0
