"""Fail-closed deployment gate for closed-loop map actions.

Design evidence may propose an action, but it cannot authorize deployment.
Only a disjoint control decision may advance the frozen action to confirmation,
and only a disjoint confirmation decision may replace the stable map.
"""

from __future__ import annotations

from collections.abc import Mapping


_CONFIRMED_DECISIONS = {"DEFAULT_CONFIRMED", "PARETO_CONFIRMED"}


def gate_closed_loop_action(
    *,
    candidate_arm: str,
    control_decision: Mapping,
    confirmation_decision: Mapping | None = None,
) -> dict:
    """Return a fail-closed deployment decision for one frozen action arm."""

    arm = str(candidate_arm)
    if not arm:
        raise ValueError("closed-loop deployment requires a named frozen arm")
    if control_decision.get("uses_test_queries") is not False:
        raise ValueError("closed-loop control must not use test queries")
    control_selected = control_decision.get("selected_arm") == arm
    control_advanced = control_decision.get("decision") == "ADVANCE_TO_CONFIRMATION"
    if not (control_selected and control_advanced):
        return {
            "schema": "lafgs_v18_closed_loop_deployment_gate",
            "version": 1,
            "uses_test_queries": False,
            "candidate_arm": arm,
            "control_passed": False,
            "confirmation_passed": False,
            "formal_deployment_authorized": False,
            "decision": "REJECT_CONTROL",
            "stable_map_policy": "retain_previous_stable_active_set",
        }
    if confirmation_decision is None:
        return {
            "schema": "lafgs_v18_closed_loop_deployment_gate",
            "version": 1,
            "uses_test_queries": False,
            "candidate_arm": arm,
            "control_passed": True,
            "confirmation_passed": False,
            "formal_deployment_authorized": False,
            "decision": "AWAIT_CONFIRMATION",
            "stable_map_policy": "retain_previous_stable_active_set",
        }
    if confirmation_decision.get("uses_test_queries") is not False:
        raise ValueError("closed-loop confirmation must not use test queries")
    confirmation_passed = bool(
        confirmation_decision.get("selected_arm") == arm
        and confirmation_decision.get("decision") in _CONFIRMED_DECISIONS
    )
    return {
        "schema": "lafgs_v18_closed_loop_deployment_gate",
        "version": 1,
        "uses_test_queries": False,
        "candidate_arm": arm,
        "control_passed": True,
        "confirmation_passed": confirmation_passed,
        "formal_deployment_authorized": confirmation_passed,
        "decision": "DEPLOY_CANDIDATE" if confirmation_passed else "REJECT_CONFIRMATION",
        "stable_map_policy": (
            "deploy_confirmed_candidate"
            if confirmation_passed
            else "retain_previous_stable_active_set"
        ),
    }


__all__ = ["gate_closed_loop_action"]
