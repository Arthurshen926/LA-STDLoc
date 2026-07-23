import pytest
import torch

from localization_training.native_matchability import (
    FEATURE_NAMES,
    build_native_matchability_features,
    calibrated_native_matchability,
    validate_native_matchability_state,
)


def _state():
    return {
        "version": 1,
        "feature_names": list(FEATURE_NAMES),
        "topk": 3,
        "entropy_temperature": 0.1,
        "confidence_floor": 0.2,
        "feature_mean": torch.zeros(len(FEATURE_NAMES)),
        "feature_std": torch.ones(len(FEATURE_NAMES)),
        "weights": torch.tensor([2.0, 1.0, -1.0, -2.0, -0.5, 0.25]),
        "bias": 0.0,
        "landmark_false_attractor_rate": torch.tensor([0.1, 0.9, 0.4]),
        "landmark_incoming_count": torch.tensor([1.0, 20.0, 4.0]),
    }


def test_native_matchability_features_use_only_topk_context_and_selected_top1():
    features = build_native_matchability_features(
        torch.tensor([[0.9, 0.7, 0.1], [0.8, 0.79, 0.2]]),
        torch.tensor([[0, 1, 2], [1, 2, 0]]),
        false_attractor_rate=torch.tensor([0.1, 0.9, 0.4]),
        incoming_count=torch.tensor([1.0, 20.0, 4.0]),
        keypoint_scores=torch.tensor([0.7, 0.5]),
        entropy_temperature=0.1,
    )
    assert features.shape == (2, len(FEATURE_NAMES))
    assert torch.allclose(features[:, 0], torch.tensor([0.9, 0.8]))
    assert torch.allclose(features[:, 1], torch.tensor([0.2, 0.01]), atol=1e-6)
    assert torch.allclose(features[:, 3], torch.tensor([0.1, 0.9]))
    assert torch.allclose(features[:, 4], torch.log1p(torch.tensor([1.0, 20.0])))


def test_native_matchability_floor_preserves_probability_ranking():
    state = _state()
    validate_native_matchability_state(state, landmark_count=3)
    features = torch.zeros((3, len(FEATURE_NAMES)))
    features[:, 0] = torch.tensor([0.1, 0.2, 0.3])
    confidence = calibrated_native_matchability(features, state)
    assert torch.all(confidence >= 0.2)
    assert torch.all(confidence < 1.0)
    assert torch.equal(torch.argsort(confidence), torch.tensor([0, 1, 2]))


def test_native_matchability_rejects_schema_or_landmark_count_mismatch():
    state = _state()
    with pytest.raises(ValueError, match="count"):
        validate_native_matchability_state(state, landmark_count=2)
    with pytest.raises(ValueError, match="feature schema"):
        validate_native_matchability_state({**state, "feature_names": ["wrong"]})
