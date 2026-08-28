import torch

from evidence.v2_filtered_observations import (
    build_v2_filtered_provider,
    remap_candidate_rows_to_source,
)


def _cache() -> dict:
    record = {
        "native_keypoints": torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        "native_descriptors": torch.eye(3),
        "native_scores": torch.tensor([0.3, 0.2, 0.1]),
        "native_K": torch.eye(3),
        "pose_w2c": torch.eye(4),
        "native_input_hw": torch.tensor([8, 8]),
        "native_valid_mask": torch.ones(8, 8, dtype=torch.bool),
        "native_alpha": torch.ones(8, 8),
        "native_depth": torch.ones(8, 8),
    }
    return {
        "schema": "render_observation_cache_v2", "version": 2,
        "uses_source_mapping_rgb": False, "uses_test_queries": False,
        "queries": {"seq/a.png": record},
    }


def test_v2_filter_is_applied_before_provider_consumers() -> None:
    provider, source, report = build_v2_filtered_provider(
        _cache(), rows_by_query=[torch.tensor([True, False, True])]
    )
    view = provider.build_view(0)
    assert view.keypoints.tolist() == [[1.0, 1.0], [3.0, 3.0]]
    assert view.keypoint_validity.tolist() == [True, True]
    assert source[0].tolist() == [0, 2]
    assert report["removed_row_count"] == 1
    assert report["filter_stage"] == "after_detection_before_pair_association"


def test_candidate_csr_is_remapped_to_original_cache_rows() -> None:
    candidate = {
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 2]),
            "query_indices": torch.tensor([0, 0]),
            "keypoint_indices": torch.tensor([0, 1]),
        },
        "contract": {},
    }
    remapped = remap_candidate_rows_to_source(
        candidate, [torch.tensor([4, 9])]
    )
    assert remapped["projective_anchor_observations"]["keypoint_indices"].tolist() == [4, 9]
    assert remapped["contract"]["observation_rows_remapped_to_source_cache"] is True
