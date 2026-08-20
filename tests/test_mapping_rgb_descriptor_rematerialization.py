import torch

from scripts.materialize_mapping_rgb_descriptors import (
    fuse_frozen_rows,
    validate_frozen_inputs,
)


def _fixture():
    cache = {
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "source_mapping_indices": torch.tensor([0, 1]),
        "queries": {
            "seq/a.png": {
                "native_keypoints": torch.tensor([[1.0, 2.0]]),
                "native_descriptors": torch.tensor([[1.0, 0.0]]),
                "native_scores": torch.tensor([0.8]),
                "native_alpha_at_keypoints": torch.tensor([0.9]),
                "native_valid_keypoint_mask": torch.tensor([True]),
                "native_appearance_reliability": torch.tensor([1.0]),
            },
            "seq/b.png": {
                "native_keypoints": torch.tensor([[3.0, 4.0]]),
                "native_descriptors": torch.tensor([[0.8, 0.2]]),
                "native_scores": torch.tensor([0.7]),
                "native_alpha_at_keypoints": torch.tensor([0.8]),
                "native_valid_keypoint_mask": torch.tensor([True]),
                "native_appearance_reliability": torch.tensor([1.0]),
            },
        },
    }
    payload = {
        "rendered_rgb_only": True,
        "query_names": ["seq/a.png", "seq/b.png"],
        "query_bins": torch.tensor([0, 1]),
        "tracks": {
            "track_index": torch.tensor([7, 7]),
            "query_index": torch.tensor([0, 1]),
            "keypoint_index": torch.tensor([0, 0]),
            "confidence": torch.tensor([1.0, 1.0]),
        },
    }
    state = {
        "anchor_ids": torch.tensor([5, 8]),
        "track_cluster_ids": torch.tensor([7, -1]),
        "anchor_features": torch.zeros(2, 2),
        "projective_anchor_observations": {
            "schema": "lafgs_projective_anchor_observations",
            "version": 1,
            "observation_offsets": torch.tensor([0, 2, 4]),
            "query_indices": torch.tensor([0, 1, 0, 1]),
            "keypoint_indices": torch.tensor([0, 0, 0, 0]),
        },
    }
    return state, payload, cache


def test_all_track_and_surface_rows_are_rematerialized():
    state, payload, cache = _fixture()
    names, indices = validate_frozen_inputs(state, payload, cache)
    assert names == ["seq/a.png", "seq/b.png"]
    assert indices.tolist() == [0, 1]
    features = fuse_frozen_rows(state, payload, cache, trim_fraction=0.2)
    assert features.shape == (2, 2)
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features, dim=1).tolist() == [2, 2]


def test_missing_surface_observations_fail_closed():
    state, payload, cache = _fixture()
    state["projective_anchor_observations"]["observation_offsets"][-1] = 2
    state["projective_anchor_observations"]["query_indices"] = torch.tensor([0, 1])
    state["projective_anchor_observations"]["keypoint_indices"] = torch.tensor([0, 0])
    try:
        validate_frozen_inputs(state, payload, cache)
    except ValueError as error:
        assert "every frozen Anchor row" in str(error)
    else:
        raise AssertionError("zero-observation completion row was accepted")


def test_test_or_source_rgb_cache_cannot_be_used_as_frozen_evidence():
    state, payload, cache = _fixture()
    cache["uses_test_queries"] = True
    try:
        validate_frozen_inputs(state, payload, cache)
    except ValueError as error:
        assert "mapping-only" in str(error)
    else:
        raise AssertionError("test evidence was accepted")
