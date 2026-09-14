from __future__ import annotations

import pytest

from map_learning.v18_control_gate import gate_closed_loop_action


def _decision(decision: str, selected_arm: str | None) -> dict:
    return {
        "uses_test_queries": False,
        "decision": decision,
        "selected_arm": selected_arm,
    }


def test_control_rejection_retains_stable_map_without_confirmation() -> None:
    result = gate_closed_loop_action(
        candidate_arm="reactivate_15459",
        control_decision=_decision("NO_ACTION", None),
    )
    assert result["decision"] == "REJECT_CONTROL"
    assert result["formal_deployment_authorized"] is False


def test_confirmation_is_required_after_control_advance() -> None:
    result = gate_closed_loop_action(
        candidate_arm="candidate",
        control_decision=_decision("ADVANCE_TO_CONFIRMATION", "candidate"),
    )
    assert result["decision"] == "AWAIT_CONFIRMATION"
    assert result["formal_deployment_authorized"] is False


def test_only_explicit_confirmation_can_deploy_frozen_arm() -> None:
    result = gate_closed_loop_action(
        candidate_arm="candidate",
        control_decision=_decision("ADVANCE_TO_CONFIRMATION", "candidate"),
        confirmation_decision=_decision("PARETO_CONFIRMED", "candidate"),
    )
    assert result["decision"] == "DEPLOY_CANDIDATE"
    assert result["formal_deployment_authorized"] is True


def test_not_confirmed_suffix_cannot_be_mistaken_for_confirmation() -> None:
    result = gate_closed_loop_action(
        candidate_arm="candidate",
        control_decision=_decision("ADVANCE_TO_CONFIRMATION", "candidate"),
        confirmation_decision=_decision("NOT_CONFIRMED", None),
    )
    assert result["decision"] == "REJECT_CONFIRMATION"
    assert result["formal_deployment_authorized"] is False


def test_test_query_decision_is_rejected() -> None:
    control = _decision("ADVANCE_TO_CONFIRMATION", "candidate")
    control["uses_test_queries"] = True
    with pytest.raises(ValueError, match="must not use test"):
        gate_closed_loop_action(candidate_arm="candidate", control_decision=control)
