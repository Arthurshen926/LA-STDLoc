import pytest
import torch

from common.v6_contracts import ANCHOR_CANDIDATE_SCHEMA
from map_learning.mapping_reliability_subset import select_mapping_reliable_anchors
from topology.v6_anchor_map import subset_projective_anchor_map


def _candidates():
    return {
        "schema": ANCHOR_CANDIDATE_SCHEMA,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "anchor_xyz": torch.zeros(4, 3),
        "identity_reliability": torch.tensor([0.8, 0.2, 0.8, 0.8]),
        "geometry_reliability": torch.tensor([0.8, 0.8, 0.2, 0.8]),
        "anchor_position_covariance": torch.stack(
            [torch.eye(3) * value for value in (0.001, 0.001, 0.001, 0.1)]
        ),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 4, 8, 12, 16]),
        },
    }


def test_mapping_reliability_gate_is_conjunctive():
    selected, report = select_mapping_reliable_anchors(
        _candidates(),
        minimum_identity_reliability=0.5,
        minimum_geometry_reliability=0.5,
        minimum_observations=4,
        maximum_covariance_trace_m2=0.01,
    )
    assert selected.tolist() == [0]
    assert report["selected_anchor_count"] == 1
    assert report["uses_test_queries"] is False
    assert report["localization_outcomes_consumed"] is False


def test_mapping_reliability_gate_fails_closed_when_empty():
    with pytest.raises(ValueError, match="retained no Anchor"):
        select_mapping_reliable_anchors(
            _candidates(),
            minimum_identity_reliability=0.99,
            minimum_geometry_reliability=0.99,
            minimum_observations=4,
            maximum_covariance_trace_m2=0.01,
        )


def test_subset_keeps_anchor_view_support_aligned():
    state = {
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.zeros(3, 3),
        "anchor_features": torch.zeros(3, 2),
        "anchor_position_covariance": torch.eye(3).repeat(3, 1, 1),
        "anchor_matchability": torch.ones(3),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 1, 2, 3]),
            "query_indices": torch.tensor([0, 1, 2]),
            "keypoint_indices": torch.tensor([3, 4, 5]),
        },
        "anchor_view_support": {
            "schema": "lafgs_v24_anchor_view_support",
            "direction_modes": torch.arange(18).reshape(3, 2, 3),
            "direction_radius_deg": torch.arange(6).reshape(3, 2),
            "mode_count": torch.tensor([1, 2, 1]),
            "minimum_distance_m": torch.tensor([1.0, 2.0, 3.0]),
            "maximum_distance_m": torch.tensor([4.0, 5.0, 6.0]),
            "observation_count": torch.tensor([1, 1, 1]),
        },
    }
    output = subset_projective_anchor_map(state, torch.tensor([0, 2]))
    assert output["anchor_view_support"]["mode_count"].tolist() == [1, 1]
    assert output["anchor_view_support"]["direction_modes"].shape == (2, 2, 3)
