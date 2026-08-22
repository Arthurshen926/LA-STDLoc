import pytest
import torch

from topology.v6_anchor_map import (
    materialize_projective_anchor_map,
    merge_projective_candidates,
)


def _part(start: float) -> dict:
    return {
        "schema": "projective_anchor_candidates_v2",
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": ["q0", "q1"],
        "query_bins": torch.tensor([0, 1]),
        "anchor_xyz": torch.tensor([[start, 0.0, 1.0]]),
        "anchor_features": torch.tensor([[1.0, 0.0]]),
        "anchor_position_covariance": torch.eye(3)[None],
        "identity_reliability": torch.tensor([0.8]),
        "geometry_reliability": torch.tensor([0.5]),
        "candidate_kind": "base" if start == 0 else "completion",
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 1]),
            "query_indices": torch.tensor([int(start)]),
            "keypoint_indices": torch.tensor([2]),
        },
    }


def test_merge_and_materialize_preserve_csr_and_pure_ray_contract() -> None:
    merged = merge_projective_candidates([_part(0), _part(1)])
    state = materialize_projective_anchor_map(merged, lineage={"round": 0})
    assert state["anchor_ids"].tolist() == [0, 1]
    assert state["projective_anchor_observations"]["observation_offsets"].tolist() == [0, 1, 2]
    assert state["anchor_matchability"].tolist() == pytest.approx([0.4, 0.4])
    assert state["projective_anchor_construction"]["direct_gaussian_surface_anchor"] is False
