import numpy as np

from map_learning.v13_risk_supervisor import (
    block_bootstrap_improvement_probability,
    paired_effect,
    supervise_candidate,
)


def _pose(task, translation=None, rotation=0.1):
    return {
        "task_error": float(task),
        "translation_error_cm": float(translation if translation is not None else task * 5),
        "rotation_error_deg": float(rotation),
    }


def _rows(base, candidate):
    return [
        {
            "query_index": index,
            "pose_family_id": index // 2,
            "baseline": _pose(before),
            "candidate": _pose(after),
        }
        for index, (before, after) in enumerate(zip(base, candidate))
    ]


def test_effect_measures_magnitude_instead_of_only_worsening_count():
    rows = _rows([10, 10, 1, 1], [1, 1, 1.01, 1.01])
    effect = paired_effect(rows, "candidate")
    assert effect["worsening_fraction"] == 0.5
    assert effect["net_gain"] > 17.9
    assert effect["harm_ratio"] < 0.01


def test_tail_improvement_can_be_a_pareto_candidate_with_small_median_tradeoff():
    base = [0.10] * 14 + [1.0, 2.0, 4.0, 8.0, 12.0, 20.0]
    candidate = [0.11] * 14 + [0.8, 1.5, 2.0, 3.0, 5.0, 8.0]
    decision = supervise_candidate(
        _rows(base, candidate), "candidate", bootstrap_samples=300, seed=7
    )
    assert decision["hard_safety"]["passed"]
    assert decision["candidate"]["q50_task"] > decision["baseline"]["q50_task"]
    assert decision["candidate"]["q90_task"] < decision["baseline"]["q90_task"]
    assert decision["classification"] in {"PARETO_CANDIDATE", "DEFAULT_CANDIDATE"}


def test_pose_family_bootstrap_is_deterministic():
    rows = _rows(np.linspace(0.1, 3.0, 20), np.linspace(0.09, 2.5, 20))
    one = block_bootstrap_improvement_probability(
        rows, "candidate", samples=100, seed=11
    )
    two = block_bootstrap_improvement_probability(
        rows, "candidate", samples=100, seed=11
    )
    assert one == two


def test_extreme_single_query_regression_fails_hard_safety():
    base = [0.2] * 20
    candidate = [0.1] * 19 + [4.0]
    decision = supervise_candidate(
        _rows(base, candidate), "candidate", bootstrap_samples=100, seed=3
    )
    assert not decision["hard_safety"]["checks"]["single_query"]
    assert decision["classification"] == "REJECTED"
