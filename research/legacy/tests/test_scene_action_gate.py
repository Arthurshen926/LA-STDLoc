import pytest
import torch

from features.scene_action_gate import (
    FEATURE_NAMES,
    SceneActionGate,
    feature_tensor,
    load_scene_action_gate,
    query_action_features,
)


def test_query_action_features_match_contract() -> None:
    value = query_action_features(
        native_keypoints=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        native_scores=torch.tensor([0.4, 0.2]),
        detector_keypoints=torch.tensor([[0.0, 0.0], [2.0, 0.0]]),
        detector_logits=torch.tensor([[0.0, 1.0, -1.0]]),
    )
    assert value.shape == (len(FEATURE_NAMES),)
    assert value[-1] == 0.5


def test_scene_action_gate_loads_only_bound_lineage() -> None:
    model = SceneActionGate(torch.zeros(len(FEATURE_NAMES)), torch.ones(len(FEATURE_NAMES)))
    checkpoint = {
        "schema": "lafgs_v12_scene_action_gate",
        "loo_used": False,
        "uses_test_queries": False,
        "feature_names": list(FEATURE_NAMES),
        "map_sha256": "map",
        "detector_sha256": "detector",
        "feature_mean": torch.zeros(len(FEATURE_NAMES)),
        "feature_std": torch.ones(len(FEATURE_NAMES)),
        "state_dict": model.state_dict(),
    }
    loaded = load_scene_action_gate(checkpoint, map_sha256="map", detector_sha256="detector")
    assert loaded(feature_tensor({name: 0.0 for name in FEATURE_NAMES})).ndim == 0
    with pytest.raises(ValueError, match="lineage"):
        load_scene_action_gate(checkpoint, map_sha256="wrong", detector_sha256="detector")
