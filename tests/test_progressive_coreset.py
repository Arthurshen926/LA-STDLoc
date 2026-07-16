import torch
import pytest

from localization_training.progressive_coreset import (
    active_group_representatives,
    aggregate_atom_features,
    append_reprojection_positive,
    build_surface_patch_atoms,
    coreset_soft_matching_loss,
    discrete_select_atoms,
    deployment_soft_matching_loss,
    make_gradual_budget_schedule,
    active_set_diagnostics,
    build_surface_groups,
    coreset_matching_loss,
    make_progressive_budget_schedule,
    progressive_coreset_regularizers,
    project_active_set,
    provenance_mass_partition,
)


def test_surface_patch_atoms_separate_identity_coverage_and_redundancy():
    xyz = torch.tensor(
        [[0.000, 0.0, 0.0], [0.005, 0.0, 0.0], [0.060, 0.0, 0.0], [0.065, 0.0, 0.0]]
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1)
    atoms = build_surface_patch_atoms(
        xyz,
        normals,
        torch.tensor([3, 2, 3, 0]),
        identity_voxel_size=0.02,
        coverage_voxel_size=0.20,
        redundancy_voxel_size=0.05,
        min_observations=2,
    )
    assert atoms.representative_raw_indices.tolist() == [0, 2]
    assert atoms.raw_to_atom.tolist() == [0, 0, 1, 1]
    assert atoms.coverage_cell_ids[0] == atoms.coverage_cell_ids[1]
    assert atoms.diagnostics["identity_distance_m"]["max"] < 0.01


def test_surface_patch_atom_cap_preserves_coarse_coverage():
    xyz = torch.stack(
        [torch.arange(20).float() * 0.03, torch.zeros(20), torch.zeros(20)], dim=1
    )
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(20, 1)
    atoms = build_surface_patch_atoms(
        xyz,
        normals,
        torch.ones(20) * 3,
        identity_voxel_size=0.01,
        coverage_voxel_size=0.20,
        min_observations=2,
        max_atoms=10,
    )
    assert atoms.representative_raw_indices.numel() == 10
    assert torch.unique(atoms.coverage_cell_ids).numel() >= 3


def test_surface_patch_anchor_is_weighted_geometric_medoid():
    xyz = torch.tensor([[0.001, 0.0, 0.0], [0.009, 0.0, 0.0], [0.018, 0.0, 0.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1)
    atoms = build_surface_patch_atoms(
        xyz,
        normals,
        torch.tensor([1.0, 10.0, 1.0]),
        identity_voxel_size=0.02,
        min_observations=1,
        max_atoms=0,
    )
    assert atoms.representative_raw_indices.tolist() == [1]
    assert atoms.diagnostics["representative_mode"] == "weighted_geometric_medoid"


def test_patch_feature_initialization_aggregates_members():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    aggregated, mass = aggregate_atom_features(
        features,
        torch.tensor([0, 0, -1]),
        1,
        torch.tensor([3.0, 1.0, 1.0]),
    )
    assert mass.tolist() == [4.0]
    torch.testing.assert_close(
        aggregated[0], torch.nn.functional.normalize(torch.tensor([3.0, 1.0]), dim=0)
    )


def test_provenance_mass_keeps_missing_and_shadow_mass_separate():
    result = provenance_mass_partition(
        torch.tensor([[0, 1, -1], [1, -1, -1]]),
        torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.0, 0.0]]),
        torch.tensor([True, False]),
        atom_count=2,
    )
    torch.testing.assert_close(result["active_mass"], torch.tensor([0.2, 0.0]))
    torch.testing.assert_close(result["shadow_mass"], torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(result["missing_mass"], torch.tensor([0.5, 0.0]))


@pytest.mark.parametrize("valid", [False, True])
def test_reprojection_positive_keeps_fixed_schema_and_probability_mass(valid):
    labels, primitive_ids, weights = append_reprojection_positive(
        torch.tensor([[2, 3, -1, -1]]),
        torch.tensor([[12, 13, -1, -1]]),
        torch.tensor([[0.6, 0.4, 0.0, 0.0]]),
        torch.tensor([7 if valid else -1]),
        torch.tensor([17 if valid else -1]),
        torch.tensor([valid]),
        positive_weight=0.75,
    )
    assert labels.shape == primitive_ids.shape == weights.shape == (1, 5)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))
    if valid:
        assert labels[0, -1].item() == 7
        assert primitive_ids[0, -1].item() == 17
        torch.testing.assert_close(weights[0, -1], torch.tensor(0.75))
    else:
        assert labels[0, -1].item() == -1
        assert primitive_ids[0, -1].item() == -1
        torch.testing.assert_close(weights[0, -1], torch.tensor(0.0))


def test_discrete_selection_uses_utility_before_coverage_frequency():
    selected = discrete_select_atoms(
        torch.tensor([10.0, 1.0, 9.0, 0.5]),
        torch.tensor([0, 0, 1, 2]),
        torch.arange(4),
        2,
        coverage_priority=torch.tensor([1.0, 1.0, 1e6]),
        coverage_fraction=1.0,
    )
    assert set(selected.tolist()) == {0, 2}


def test_discrete_selection_reports_coverage_and_cutoff_attribution():
    selected, diagnostics = discrete_select_atoms(
        torch.tensor([4.0, 3.0, 2.0, 1.0]),
        torch.tensor([0, 0, 1, 1]),
        torch.arange(4),
        2,
        coverage_fraction=0.5,
        return_diagnostics=True,
    )
    assert selected.numel() == 2
    assert diagnostics["coverage_reserved_count"] == 1
    assert diagnostics["utility_fill_count"] == 1


def test_gradual_schedule_never_drops_more_than_keep_ratio_except_final_rounding():
    schedule = make_gradual_budget_schedule(100, 1000, 20, keep_ratio=0.75)
    assert schedule.budgets[0] == 100
    assert schedule.budgets[-1] == 20
    assert all(b < a for a, b in zip(schedule.budgets, schedule.budgets[1:]))
    assert schedule.boundaries[-1] == 1000


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


def test_deployment_matching_uses_cosine_without_selection_prior():
    loss = deployment_soft_matching_loss(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[1.0]]),
        torch.tensor([[True]]),
        torch.tensor([[[0.0, 1.0]]]),
        torch.tensor([[True]]),
        temperature=1.0,
    )
    expected = torch.log1p(torch.exp(torch.tensor(-1.0)))
    assert torch.allclose(loss, expected)


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
