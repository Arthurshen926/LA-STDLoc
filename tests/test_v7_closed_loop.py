import pytest

from map_learning.v7_closed_loop import (
    confirm_or_rollback_v7_round,
    next_v7_round_action,
)


def _result(query, te, *, rgb="a"):
    return {
        "query_index": query,
        "rgb_sha256": rgb,
        "certificate_decision": "ACCEPT",
        "translation_error_cm": te,
        "rotation_error_deg": 0.1,
        "runtime_ms": 10.0,
        "uses_test_queries": False,
    }


def test_fresh_better_proposal_is_accepted() -> None:
    report = confirm_or_rollback_v7_round(
        round_index=0,
        baseline_map_sha256="base",
        proposal_map_sha256="proposal",
        control_query_ids=["q0"],
        confirmation_query_ids=["q1", "q2"],
        baseline_results=[_result(1, 2.0), _result(2, 3.0)],
        proposal_results=[_result(1, 1.0), _result(2, 1.5)],
    )
    assert report["decision"] == "ACCEPT"
    assert report["chosen_map_sha256"] == "proposal"


def test_failed_proposal_rolls_back_exactly() -> None:
    report = confirm_or_rollback_v7_round(
        round_index=0,
        baseline_map_sha256="base",
        proposal_map_sha256="proposal",
        control_query_ids=["q0"],
        confirmation_query_ids=["q1", "q2"],
        baseline_results=[_result(1, 1.0), _result(2, 1.5)],
        proposal_results=[_result(1, 2.0), _result(2, 3.0)],
    )
    assert report["decision"] == "ROLLBACK"
    assert report["chosen_map_sha256"] == "base"
    assert report["atomic_rollback_exact"] is True


def test_confirmation_must_be_fresh_and_same_rgb() -> None:
    with pytest.raises(ValueError, match="fresh"):
        confirm_or_rollback_v7_round(
            round_index=0,
            baseline_map_sha256="base",
            proposal_map_sha256="proposal",
            control_query_ids=["q1"],
            confirmation_query_ids=["q1"],
            baseline_results=[_result(1, 1.0)],
            proposal_results=[_result(1, 0.5)],
        )
    with pytest.raises(ValueError, match="same RGB"):
        confirm_or_rollback_v7_round(
            round_index=0,
            baseline_map_sha256="base",
            proposal_map_sha256="proposal",
            control_query_ids=["q0"],
            confirmation_query_ids=["q1"],
            baseline_results=[_result(1, 1.0, rgb="a")],
            proposal_results=[_result(1, 0.5, rgb="b")],
        )


def test_p7_stop_order_and_two_round_limit() -> None:
    assert (
        next_v7_round_action(
            completed_rounds=0,
            previous_proposal_accepted=True,
            executable_representation_deficit_count=0,
            median_task_improvement=1.0,
        )
        == "no_executable_representation_deficit"
    )
    assert (
        next_v7_round_action(
            completed_rounds=2,
            previous_proposal_accepted=True,
            executable_representation_deficit_count=3,
            median_task_improvement=1.0,
        )
        == "maximum_two_rounds_reached"
    )
