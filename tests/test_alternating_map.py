import numpy as np
import torch

from localization_training.alternating_map import (
    PoseRiskConfig,
    affected_query_mask,
    evaluate_structure_proposal,
    summarize_pose_risk,
)


def test_pose_risk_rewards_joint_tail_improvement():
    config = PoseRiskConfig(complexity_weight=0.0)
    current = np.array([0.02, 0.08, 0.20, 0.60])
    proposal = np.array([0.02, 0.07, 0.15, 0.30])
    decision = evaluate_structure_proposal(
        current,
        proposal,
        current_anchor_count=100,
        proposal_anchor_count=104,
        config=config,
    )
    assert decision["accepted"]
    assert decision["objective_gain"] > 0
    assert decision["proposal"]["tail_cvar"] < decision["current"]["tail_cvar"]


def test_pose_risk_protects_existing_five_centimeter_query():
    config = PoseRiskConfig(
        complexity_weight=0.0,
        max_protected_regressions=0,
    )
    current = np.array([0.02, 0.50])
    proposal = np.array([0.20, 0.10])
    decision = evaluate_structure_proposal(
        current,
        proposal,
        current_anchor_count=100,
        proposal_anchor_count=101,
        config=config,
    )
    assert not decision["accepted"]
    assert not decision["gates"]["protected"]


def test_complexity_cost_prevents_zero_gain_growth():
    config = PoseRiskConfig(
        complexity_weight=0.1,
        reference_anchor_count=100,
    )
    errors = np.array([0.10, 0.20])
    current = summarize_pose_risk(
        errors, anchor_count=100, config=config
    )
    proposal = summarize_pose_risk(
        errors, anchor_count=120, config=config
    )
    assert proposal["objective"] > current["objective"]


def test_deployment_costs_are_part_of_objective():
    config = PoseRiskConfig(
        complexity_weight=0.0,
        hypotheses_weight=0.02,
        reference_hypotheses=1000,
        runtime_weight=0.03,
        reference_runtime_seconds=0.1,
    )
    errors = np.array([0.10, 0.20])
    fast = summarize_pose_risk(
        errors,
        anchor_count=100,
        config=config,
        hypotheses=np.array([1000, 1000]),
        runtime_seconds=np.array([0.1, 0.1]),
    )
    slow = summarize_pose_risk(
        errors,
        anchor_count=100,
        config=config,
        hypotheses=np.array([2000, 2000]),
        runtime_seconds=np.array([0.2, 0.2]),
    )
    assert fast["objective"] < slow["objective"]
    assert fast["hypotheses_cost"] == 0.02
    assert fast["runtime_cost"] == 0.03


def test_worst_group_risk_prevents_hiding_rare_trajectory():
    config = PoseRiskConfig(
        complexity_weight=0.0,
        cvar_weight=0.0,
        worst_group_weight=0.5,
    )
    groups = np.array(["common", "common", "rare"])
    balanced = summarize_pose_risk(
        np.array([0.10, 0.10, 0.10]),
        anchor_count=100,
        config=config,
        group_ids=groups,
    )
    rare_failure = summarize_pose_risk(
        np.array([0.05, 0.05, 0.25]),
        anchor_count=100,
        config=config,
        group_ids=groups,
    )
    assert rare_failure["worst_group_risk"] > balanced["worst_group_risk"]
    assert rare_failure["objective"] > balanced["objective"]


def test_soft_regression_mode_allows_continuous_pareto_tradeoff():
    config = PoseRiskConfig(
        complexity_weight=0.2,
        reference_anchor_count=100,
        cvar_weight=0.0,
        soft_regression_weight=0.001,
        hard_gate_mode=False,
        max_r5_regression=0.0,
    )
    current = np.array([0.049, 0.20, 0.20])
    proposal = np.array([0.051, 0.15, 0.15])
    decision = evaluate_structure_proposal(
        current,
        proposal,
        current_anchor_count=100,
        proposal_anchor_count=80,
        proposal_churn_count=20,
        config=config,
    )
    assert not decision["gates"]["r5"]
    assert decision["accepted"]
    assert decision["proposal"]["soft_regression_cost"] > 0


def test_affected_query_mask_uses_hard_top1_changes():
    mask = affected_query_mask(
        [torch.tensor([0.7, 0.8]), torch.tensor([0.5])],
        [torch.tensor([0.7, 0.9]), torch.tensor([0.4])],
    )
    assert mask.tolist() == [True, False]
