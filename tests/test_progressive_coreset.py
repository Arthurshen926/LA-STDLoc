import torch
import pytest

from localization_training.progressive_coreset import (
    active_group_representatives,
    coreset_soft_matching_loss,
    active_set_diagnostics,
    build_surface_groups,
    coreset_matching_loss,
    make_progressive_budget_schedule,
    progressive_coreset_regularizers,
    project_active_set,
)


def test_active_group_representatives_use_active_members_only():
    scores = torch.tensor([0.1, 0.9, 0.8, 0.2])
    groups = torch.tensor([0, 0, 1, 1])
    result = active_group_representatives(scores, groups, torch.tensor([0, 2, 3]), 2)
    assert result.tolist() == [0, 2]


def test_group_aware_projection_reserves_observed_groups():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0])
    groups = torch.tensor([0, 0, 1, 2])
    active = project_active_set(
        logits,
        2,
        group_ids=groups,
        group_priority=torch.tensor([1.0, 3.0, 2.0]),
    )
    assert set(groups[active].tolist()) == {1, 2}


def test_soft_matching_prefers_provenance_positive_mass():
    query = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], requires_grad=True)
    negatives = torch.tensor([[[0.5, 0.5]]])
    loss = coreset_soft_matching_loss(
        query,
        positives,
        torch.tensor([[0.8, 0.2]]),
        torch.zeros(1, 2),
        torch.tensor([[True, True]]),
        negatives,
        torch.zeros(1, 1),
        torch.tensor([[True]]),
        temperature=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert positives.grad.abs().sum() > 0


def test_progressive_schedule_is_monotonic_and_reaches_budget():
    schedule = make_progressive_budget_schedule(1_200_000, 30_000, 16_384)
    assert schedule.budgets[0] == 1_200_000
    assert schedule.budgets[-1] == 16_384
    assert all(a > b for a, b in zip(schedule.budgets, schedule.budgets[1:]))
    assert schedule.budget(30_000) == 16_384


def test_surface_groups_merge_nearby_points_and_split_normals():
    xyz = torch.tensor([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.2, 0.0, 0.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
    coarse, coarse_count = build_surface_groups(xyz, voxel_size=0.1)
    oriented, oriented_count = build_surface_groups(
        xyz, normals, voxel_size=0.1, normal_bins=4
    )
    assert coarse[0] == coarse[1]
    assert oriented[0] != oriented[1]
    assert oriented_count > coarse_count


def test_coverage_loss_reactivates_observed_group_representative():
    logits = torch.tensor([-4.0, -3.0, 4.0], requires_grad=True)
    groups = torch.tensor([0, 0, 1])
    losses = progressive_coreset_regularizers(logits, groups, torch.tensor([0]), 2)
    losses["coverage"].backward()
    # Exact group coverage distributes reactivation pressure to every member.
    assert logits.grad[1] < 0
    assert logits.grad[0] < 0
    assert logits.grad[2] == 0


def test_matching_loss_prefers_positive_descriptor_and_gate():
    query = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0]], requires_grad=True)
    negative = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]], requires_grad=True)
    positive_gate = torch.tensor([0.0], requires_grad=True)
    negative_gate = torch.zeros(1, 2, requires_grad=True)
    loss = coreset_matching_loss(
        query,
        positive,
        negative,
        positive_gate,
        negative_gate,
        torch.ones(1, 2, dtype=torch.bool),
    )
    loss.backward()
    assert positive_gate.grad < 0
    assert negative_gate.grad.sum() > 0


def test_projection_hysteresis_and_churn_diagnostics():
    logits = torch.tensor([1.0, 0.9, 0.8, 0.7])
    active = project_active_set(logits, 2, previous_active=torch.tensor([1, 2]), hysteresis=0.3)
    assert active.tolist() == [1, 2]
    diagnostics = active_set_diagnostics(active, torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1]))
    assert diagnostics["active_count"] == 2
    assert diagnostics["active_jaccard"] == pytest.approx(1 / 3)
