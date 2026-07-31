import torch

from localization_training.slps_residual_signatures import (
    RESIDUAL_SIGNATURE_FEATURE_NAMES,
    add_residual_statistics,
    empty_residual_statistics,
    residual_signature_features,
    residual_statistics_contribution,
    signed_reprojection_residual,
    subtract_residual_statistics,
)
from localization_training.slps_selector import (
    SLPS_BIAS_AWARE_FEATURE_NAMES,
    SLPSModelConfig,
    SLPSSelector,
    build_slps_features,
)


def test_residual_signature_statistics_support_leave_query_out():
    anchors = torch.tensor([0, 0, 1])
    keypoints = torch.tensor([[10.0, 10.0], [12.0, 10.0], [80.0, 80.0]])
    residual = torch.tensor([[2.0, 0.0], [4.0, 0.0], [-2.0, 0.0]])
    contribution = residual_statistics_contribution(
        anchor_indices=anchors,
        keypoints=keypoints,
        image_hw=(100, 100),
        signed_residual=residual,
        valid=torch.ones(3, dtype=torch.bool),
        anchor_count=2,
        grid_size=2,
    )
    total = empty_residual_statistics(2, grid_size=2)
    add_residual_statistics(total, contribution)
    leave_query_out = subtract_residual_statistics(total, contribution)
    assert leave_query_out["attempts"].sum() == 0
    assert leave_query_out["weighted_sum"].abs().sum() < 1e-6


def test_residual_signature_lookup_preserves_direction():
    statistics = empty_residual_statistics(1, grid_size=2)
    contribution = residual_statistics_contribution(
        anchor_indices=torch.zeros(4, dtype=torch.long),
        keypoints=torch.tensor(
            [[10.0, 10.0], [12.0, 10.0], [11.0, 12.0], [13.0, 11.0]]
        ),
        image_hw=(100, 100),
        signed_residual=torch.tensor(
            [[3.0, -1.0], [3.0, -1.0], [3.0, -1.0], [3.0, -1.0]]
        ),
        valid=torch.ones(4, dtype=torch.bool),
        anchor_count=1,
        grid_size=2,
    )
    add_residual_statistics(statistics, contribution)
    features = residual_signature_features(
        statistics,
        anchor_indices=torch.tensor([0]),
        keypoints=torch.tensor([[11.0, 11.0]]),
        image_hw=(100, 100),
        grid_size=2,
        anchor_prior=0.0,
        cell_prior=0.0,
    )
    assert features.shape == (1, len(RESIDUAL_SIGNATURE_FEATURE_NAMES))
    assert features[0, 0] > 0
    assert features[0, 1] < 0
    assert features[0, 4] > 0.99


def test_signed_reprojection_residual_uses_projected_minus_observed():
    xyz = torch.tensor([[1.0, 2.0, 10.0]])
    K = torch.tensor([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    pose = torch.eye(4)
    residual, valid = signed_reprojection_residual(
        xyz, torch.tensor([[59.0, 61.0]]), K, pose
    )
    assert valid.tolist() == [True]
    assert torch.allclose(residual, torch.tensor([[1.0, -1.0]]))


def test_build_slps_features_accepts_residual_extension():
    base = torch.zeros((2, 16))
    residual = torch.zeros((2, len(RESIDUAL_SIGNATURE_FEATURE_NAMES)))
    features = build_slps_features(
        base,
        xyz=torch.zeros((2, 3)),
        anchor_type=torch.zeros(2),
        track_groups=torch.arange(2),
        track_stability=torch.ones(2),
        anchor_map_support=torch.ones(2),
        residual_signature_features=residual,
    )
    assert features.shape == (2, len(SLPS_BIAS_AWARE_FEATURE_NAMES))


def test_bias_aware_greedy_prefers_cancelling_residuals():
    model = SLPSSelector(
        SLPSModelConfig(
            input_dim=len(SLPS_BIAS_AWARE_FEATURE_NAMES),
            bias_aware_utility=True,
            greedy_block_size=1,
        )
    ).eval()
    with torch.no_grad():
        model.harmful_weight_raw.fill_(-20.0)
        model.coverage_alpha_raw.fill_(-20.0)
        model.logdet_beta_raw.fill_(-20.0)
        model.bias_weight_raw.fill_(5.0)
    encoded = {
        "hidden": torch.zeros((3, model.config.hidden_dim)),
        "additive": torch.tensor([1.01, 1.0, 0.99]),
        "strict_probability": torch.ones(3),
        "solver_probability": torch.ones(3),
        "harmful_probability": torch.zeros(3),
        "coverage": torch.zeros((3, model.config.relation_count)),
        "complementarity": torch.zeros(
            (3, model.config.complementarity_dim)
        ),
        "bias_vector": torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0]]),
    }
    groups = torch.arange(3)[:, None].expand(-1, model.config.relation_count)
    order = model.greedy_order(encoded, groups, maximum_count=2)
    selected_bias = encoded["bias_vector"][order].sum(dim=0)
    assert torch.linalg.norm(selected_bias) < 1e-6
