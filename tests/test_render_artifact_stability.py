import pytest
import torch

from evidence.render_artifact_stability import (
    gaussian_opacity_multiplier,
    local_peak_stability,
    normalized_distortion_risk,
    observation_artifact_reliability,
    quantile_summary,
    sample_plane_nearest,
)


def test_distortion_risk_is_supported_finite_and_quantile_normalized():
    distortion = torch.tensor([[0.0, 1.0], [2.0, float("nan")]])
    alpha = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    risk, scale = normalized_distortion_risk(distortion, alpha, reference_quantile=0.5)
    assert float(scale) == 1.0
    assert torch.equal(risk, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    unsupported, zero = normalized_distortion_risk(
        distortion, torch.zeros_like(alpha), reference_quantile=0.5
    )
    assert torch.count_nonzero(unsupported) == 0
    assert float(zero) == 0.0


def test_gaussian_opacity_suppression_only_uses_visible_centres():
    risk = torch.tensor([[0.0, 0.5], [1.0, 0.25]])
    centres = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]])
    visible = torch.tensor([True, True, False])
    multiplier, gaussian_risk = gaussian_opacity_multiplier(
        risk, centres, visible, suppression_strength=0.5
    )
    assert torch.equal(gaussian_risk, torch.tensor([1.0, 0.5, 0.0]))
    assert torch.equal(multiplier, torch.tensor([0.5, 0.75, 1.0]))


def test_local_peak_stability_tracks_clean_peak_motion_and_score():
    score = torch.zeros(16, 16)
    score[8, 10] = 0.4
    keypoints = torch.tensor([[8.0, 8.0]])
    score_stability, position_stability, displacement = local_peak_stability(
        score,
        keypoints,
        torch.tensor([0.8]),
        nms_radius=1,
        search_radius=4,
        position_sigma_px=2.0,
    )
    assert torch.allclose(score_stability, torch.tensor([0.5]))
    assert torch.allclose(displacement, torch.tensor([2.0]))
    assert torch.allclose(
        position_stability, torch.tensor([torch.exp(torch.tensor(-0.5))])
    )


def test_artifact_reliability_is_geometric_mean_without_row_mutation():
    raw = torch.eye(2)
    clean = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    result = observation_artifact_reliability(
        raw,
        clean,
        torch.tensor([1.0, 0.5]),
        torch.tensor([1.0, 0.5]),
        torch.tensor([0.0, 0.5]),
    )
    assert torch.equal(result["descriptor_cosine"], torch.tensor([1.0, 0.8]))
    expected = torch.tensor([1.0, (0.8 * 0.5 * 0.5 * 0.5) ** 0.25])
    assert torch.allclose(result["reliability"], expected)


def test_artifact_helpers_reject_misaligned_shapes():
    with pytest.raises(ValueError, match="aligned"):
        normalized_distortion_risk(torch.zeros(2, 2), torch.zeros(4))
    with pytest.raises(ValueError, match="visibility"):
        gaussian_opacity_multiplier(
            torch.zeros(2, 2),
            torch.zeros(1, 2),
            torch.ones(1),
        )
    with pytest.raises(ValueError, match="align"):
        local_peak_stability(
            torch.zeros(16, 16), torch.zeros(2, 2), torch.ones(1), nms_radius=1
        )


def test_nearest_plane_sampling_uses_native_grid_round_semantics():
    plane = torch.arange(9).reshape(3, 3)
    sampled = sample_plane_nearest(
        plane, torch.tensor([[0.49, 0.51], [1.51, 1.49], [-3.0, 8.0]])
    )
    assert torch.equal(sampled, torch.tensor([3, 5, 6]))


def test_quantile_summary_matches_linear_order_statistic_interpolation(monkeypatch):
    values = torch.tensor([9.0, 0.0, 4.0, 1.0, 16.0, 25.0])
    expected_p10 = float(torch.quantile(values, 0.10))
    expected_p90 = float(torch.quantile(values, 0.90))

    def reject_quantile(*args, **kwargs):
        raise RuntimeError("formal Torch large-tensor quantile limit")

    monkeypatch.setattr(torch, "quantile", reject_quantile)
    summary = quantile_summary(values)
    assert summary["p10"] == pytest.approx(expected_p10)
    assert summary["p90"] == pytest.approx(expected_p90)
    assert summary["median"] == 4.0
