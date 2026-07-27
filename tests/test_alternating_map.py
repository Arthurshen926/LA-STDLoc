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


def test_affected_query_mask_uses_hard_top1_changes():
    mask = affected_query_mask(
        [torch.tensor([0.7, 0.8]), torch.tensor([0.5])],
        [torch.tensor([0.7, 0.9]), torch.tensor([0.4])],
    )
    assert mask.tolist() == [True, False]
