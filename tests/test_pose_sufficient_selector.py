import torch

from localization_training.pose_sufficient_selector import (
    FEATURE_NAMES,
    basis_aware_core_reserve_mask,
    build_pose_sufficient_features,
    constrained_pose_sufficient_mask,
    image_grid_cells,
    predict_pose_sufficient_probability,
    spatial_octants,
)


def test_spatial_octants_cover_relative_geometry():
    xyz = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    bins = spatial_octants(xyz)
    assert len(set(bins.tolist())) >= 3


def test_pose_sufficient_selector_respects_budget_and_diversity():
    count = 64
    probabilities = torch.linspace(1.0, 0.0, count)
    cells = torch.arange(count) % 16
    dependencies = torch.arange(count) // 2
    sources = torch.arange(count) // 3
    xyz = torch.stack(
        (
            (torch.arange(count) % 4).float(),
            ((torch.arange(count) // 4) % 4).float(),
            (torch.arange(count) // 16).float(),
        ),
        dim=1,
    )
    selected = constrained_pose_sufficient_mask(
        probabilities,
        image_cells=cells,
        dependency_groups=dependencies,
        source_groups=sources,
        xyz=xyz,
        budget=32,
        minimum_per_image_cell=1,
        minimum_per_spatial_bin=1,
        maximum_per_dependency=2,
        maximum_per_source=2,
    )
    assert int(selected.sum()) == 32
    assert set(cells[selected].tolist()) == set(range(16))


def test_pose_sufficient_selector_returns_all_when_under_budget():
    selected = constrained_pose_sufficient_mask(
        torch.ones(5),
        image_cells=torch.arange(5),
        dependency_groups=torch.arange(5),
        source_groups=torch.arange(5),
        xyz=torch.zeros(5, 3),
        budget=8,
    )
    assert selected.all()


def test_pose_sufficient_runtime_feature_and_model_contract():
    scores = torch.tensor(
        [[0.9, 0.8, 0.7], [0.7, 0.6, 0.1], [0.6, 0.3, 0.2]]
    )
    indices = torch.tensor([[0, 1, 2], [1, 0, 2], [1, 2, 0]])
    statistics = {
        "attempts": torch.tensor([10.0, 20.0, 5.0]),
        "clean": torch.tensor([5.0, 4.0, 1.0]),
        "clean_inlier": torch.tensor([4.0, 3.0, 0.0]),
        "harmful_inlier": torch.tensor([1.0, 2.0, 0.0]),
    }
    features = build_pose_sufficient_features(
        scores,
        indices,
        keypoints=torch.tensor([[0.0, 0.0], [50.0, 25.0], [99.0, 49.0]]),
        keypoint_scores=torch.tensor([0.8, 0.7, 0.6]),
        image_hw=(50, 100),
        source_groups=torch.tensor([0, 1, 1]),
        dependency_groups=torch.tensor([0, 0, 1]),
        anchor_statistics=statistics,
    )
    assert features.shape == (3, len(FEATURE_NAMES))
    probabilities = predict_pose_sufficient_probability(
        features,
        {
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": torch.zeros(len(FEATURE_NAMES)),
            "feature_scale": torch.ones(len(FEATURE_NAMES)),
            "coefficients": torch.zeros(len(FEATURE_NAMES)),
            "intercept": 0.0,
        },
    )
    assert torch.allclose(probabilities, torch.full((3,), 0.5))
    assert image_grid_cells(
        torch.tensor([[0.0, 0.0], [99.0, 49.0]]), (50, 100)
    ).tolist() == [0, 15]


def test_basis_aware_core_reserve_is_bounded_and_rejects_harmful_rows():
    count = 96
    strict = torch.linspace(0.9, 0.2, count)
    solver = torch.linspace(0.95, 0.3, count)
    harmful = torch.zeros(count)
    harmful[:8] = 0.95
    x = torch.arange(count).float() % 8
    y = torch.arange(count).float() // 8
    points = torch.stack((x * 20 + 5, y * 20 + 5), dim=1)
    cells = (y.long() % 4) * 4 + (x.long() % 4)
    xyz = torch.stack((x, y, (x + y) % 5), dim=1)
    selected, diagnostics = basis_aware_core_reserve_mask(
        strict,
        solver,
        harmful,
        image_points=points,
        image_hw=(160, 180),
        image_cells=cells,
        dependency_groups=torch.arange(count),
        source_groups=torch.arange(count),
        xyz=xyz,
        core_budget=64,
        minimum_budget=72,
        maximum_budget=80,
        minimum_strict_lcb=0,
        minimum_dependency_groups=8,
        minimum_image_cells=8,
        minimum_log_expected_basis=0,
        representative_count=32,
        pair_count=32,
    )
    assert 72 <= int(selected.sum()) <= 80
    assert int(selected[:8].sum()) < 4
    assert diagnostics["selected_count"] == float(selected.sum())
    assert diagnostics["image_cell_count"] >= 8
