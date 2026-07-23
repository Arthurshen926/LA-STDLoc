import torch


def test_bounded_surface_anchor_uses_meter_bounds_and_surface_basis():
    from localization_training.surface_anchor import (
        materialize_bounded_surface_anchors,
    )

    xyz = torch.zeros((1, 3))
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    raw = torch.tensor([[100.0, -100.0, 100.0]], requires_grad=True)
    anchor = materialize_bounded_surface_anchors(
        xyz,
        rotation,
        raw,
        tangent_bound_m=0.02,
        normal_bound_m=0.005,
    )
    tangent = anchor[:, :2]
    assert torch.allclose(
        torch.linalg.norm(tangent, dim=1), torch.tensor([0.02]), atol=1e-6
    )
    assert torch.allclose(
        tangent,
        torch.tensor([[0.02 / 2**0.5, -0.02 / 2**0.5]]),
        atol=1e-6,
    )
    assert torch.allclose(anchor[:, 2], torch.tensor([0.005]), atol=1e-6)
    anchor.sum().backward()
    assert raw.grad is not None


def test_bounded_surface_anchor_never_exceeds_total_tangent_bound():
    from localization_training.surface_anchor import (
        bounded_surface_local_offsets,
    )

    raw = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [100.0, 100.0, -100.0]]
    )
    tangent, normal = bounded_surface_local_offsets(
        raw,
        tangent_bound_m=0.003,
        normal_bound_m=0.001,
    )
    assert torch.all(torch.linalg.norm(tangent, dim=1) <= 0.0030001)
    assert torch.all(normal.abs() <= 0.0010001)


def test_bounded_surface_anchor_has_tangent_gradient_at_zero_offset():
    from localization_training.surface_anchor import bounded_surface_local_offsets

    raw = torch.zeros((1, 3), requires_grad=True)
    tangent, _ = bounded_surface_local_offsets(
        raw,
        tangent_bound_m=0.003,
        normal_bound_m=0.001,
    )
    tangent[:, 0].sum().backward()

    # A zero initialized surface anchor must be able to move tangentially.
    assert torch.allclose(raw.grad[0, 0], torch.tensor(0.003), atol=1e-8)
    assert torch.allclose(raw.grad[0, 1], torch.tensor(0.0), atol=1e-8)


def test_pure_geometric_scaffold_is_exact_deterministic_and_feature_free():
    from localization_training.surface_anchor import build_pure_geometric_scaffold

    axis = torch.linspace(-1.0, 1.0, 10)
    xyz = torch.cartesian_prod(axis, axis, torch.tensor([0.0]))
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(xyz.shape[0], 1)
    first = build_pure_geometric_scaffold(xyz, rotation, 24, seed=7)
    second = build_pure_geometric_scaffold(xyz, rotation, 24, seed=7)
    assert first.indices.numel() == 24
    assert torch.equal(first.indices, second.indices)
    assert torch.unique(first.indices).numel() == 24
    assert first.diagnostics["mode"] == "pure_geometry_normal_aware_voxel_medoid"


def test_pure_geometric_scaffold_exact_budget_when_pool_is_one_larger():
    from localization_training.surface_anchor import build_pure_geometric_scaffold

    xyz = torch.stack(
        [torch.arange(65, dtype=torch.float32), torch.zeros(65), torch.zeros(65)],
        dim=1,
    )
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(65, 1)
    scaffold = build_pure_geometric_scaffold(
        xyz,
        rotation,
        64,
        voxel_size=0.1,
    )
    assert scaffold.indices.numel() == 64
    assert torch.unique(scaffold.indices).numel() == 64
