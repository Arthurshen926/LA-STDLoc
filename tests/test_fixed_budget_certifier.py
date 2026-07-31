import torch

from localization_training.fixed_budget_certifier import (
    CERTIFIER_FEATURE_NAMES,
    calibrate_selective_risk,
    certify_fixed_budget,
    fit_linear_risk_certifier,
    fixed_budget_certifier_features,
    one_sided_binomial_upper_bound,
    predict_unsafe_probability,
)


def _diagnostics() -> dict:
    return {
        "sparse_diag_pose_sufficient_input_count": 2000,
        "sparse_diag_pose_sufficient_selected_count": 512,
        "sparse_diag_pose_sufficient_probability_mean": 0.25,
        "sparse_diag_pose_sufficient_probability_p10": 0.02,
        "sparse_diag_pose_sufficient_probability_p90": 0.80,
        "sparse_diag_pose_sufficient_strict_probability_mean": 0.18,
        "sparse_diag_pose_sufficient_harmful_probability_mean": 0.04,
        "sparse_diag_pose_sufficient_strict_lcb": 120.0,
        "sparse_diag_pose_sufficient_log_expected_basis": 14.0,
        "sparse_diag_pose_sufficient_dependency_group_count": 450,
        "sparse_diag_all_2d_occupancy_frac": 1.0,
        "sparse_diag_all_2d_entropy_norm": 0.9,
        "sparse_diag_all_2d_max_cell_frac": 0.1,
        "sparse_diag_all_3d_voxel_per_match": 0.5,
        "sparse_diag_all_3d_max_voxel_frac": 0.02,
    }


def test_fixed_budget_features_have_strict_pre_pnp_contract():
    features = fixed_budget_certifier_features(_diagnostics())
    assert features.shape == (len(CERTIFIER_FEATURE_NAMES),)
    assert torch.isfinite(features).all()
    diagnostics = _diagnostics()
    diagnostics.pop("sparse_diag_pose_sufficient_strict_lcb")
    try:
        fixed_budget_certifier_features(diagnostics)
    except KeyError as error:
        assert "strict_lcb" in str(error)
    else:
        raise AssertionError("missing deployment feature was accepted")


def test_linear_certifier_fits_and_serializes():
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(
        (200, len(CERTIFIER_FEATURE_NAMES)), generator=generator
    )
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0).float()
    state = fit_linear_risk_certifier(features, labels, steps=400)
    probability = predict_unsafe_probability(state.to_dict(), features)
    accuracy = ((probability > 0.5) == labels.bool()).float().mean()
    assert float(accuracy) > 0.9


def test_risk_calibration_falls_back_when_sample_is_insufficient():
    probabilities = torch.linspace(0.0, 1.0, 100)
    labels = torch.zeros(100, dtype=torch.bool)
    calibration = calibrate_selective_risk(
        probabilities, labels, risk_limit=0.02, confidence=0.95
    )
    assert not calibration.feasible
    assert calibration.accepted_count == 0


def test_risk_calibration_accepts_the_largest_risk_controlled_prefix():
    probabilities = torch.linspace(0.0, 1.0, 300)
    labels = torch.zeros(300, dtype=torch.bool)
    labels[250:] = True
    calibration = calibrate_selective_risk(
        probabilities, labels, risk_limit=0.02, confidence=0.95
    )
    assert calibration.feasible
    assert calibration.accepted_count >= 250
    assert calibration.failure_count <= 1
    assert calibration.false_safe_upper_bound <= 0.02

    features = torch.zeros((2, len(CERTIFIER_FEATURE_NAMES)))
    fit_labels = torch.tensor([0.0, 1.0])
    state = fit_linear_risk_certifier(features, fit_labels, steps=1)
    state = type(state)(
        **{
            **state.to_dict(),
            "unsafe_probability_threshold": 1.0,
        }
    )
    assert bool(certify_fixed_budget(state, features).all())


def test_one_sided_risk_bound_is_conservative():
    assert one_sided_binomial_upper_bound(0, 100) > 0.02
    assert one_sided_binomial_upper_bound(0, 250) < 0.02
    assert one_sided_binomial_upper_bound(5, 250) > 0.02
