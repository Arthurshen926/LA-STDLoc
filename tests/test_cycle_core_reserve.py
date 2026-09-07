import torch

from common.v6_contracts import ANCHOR_CANDIDATE_SCHEMA, ASSOCIATION_GRAPH_SCHEMA
from evidence.cycle_core_reserve import select_cycle_core_coverage_reserve
from evidence.cycle_core_reserve import _subset_candidate


def test_subset_preserves_query_registry_when_query_and_anchor_counts_coincide() -> None:
    candidate = _candidate()
    candidate["query_names"] = ["a.png", "b.png", "c.png"]
    candidate["query_bins"] = torch.tensor([4, 5, 6])
    output = _subset_candidate(candidate, torch.tensor([0, 2]))
    assert output["query_bins"].tolist() == [4, 5, 6]
    assert output["query_names"] == candidate["query_names"]
    assert output["anchor_xyz"].shape[0] == 2


def _candidate() -> dict:
    return {
        "schema": ANCHOR_CANDIDATE_SCHEMA,
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": ["a.png", "b.png"],
        "query_bins": torch.tensor([0, 1]),
        "anchor_ids": torch.arange(3),
        "source_component_ids": torch.arange(3),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        "anchor_features": torch.eye(3),
        "anchor_position_covariance": torch.eye(3).repeat(3, 1, 1),
        "identity_reliability": torch.tensor([1.0, 0.9, 0.8]),
        "geometry_reliability": torch.tensor([1.0, 0.9, 0.8]),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 1, 2, 3]),
            "query_indices": torch.tensor([0, 0, 1]),
            "keypoint_indices": torch.tensor([0, 1, 0]),
        },
        "geometry_diagnostics": {"score": torch.tensor([1.0, 2.0, 3.0])},
        "contract": {},
    }


def test_cycle_core_drops_redundant_chain_and_keeps_coverage_reserve() -> None:
    association = {
        "schema": ASSOCIATION_GRAPH_SCHEMA,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "component_statistics": {
            "cycle_supported": torch.tensor([True, False, False])
        },
    }
    output, report = select_cycle_core_coverage_reserve(
        _candidate(),
        association,
        keypoints_by_query=[
            torch.tensor([[10.0, 10.0], [11.0, 11.0]]),
            torch.tensor([[70.0, 70.0]]),
        ],
        image_hw_by_query=[(100, 100), (100, 100)],
        spatial_voxel_size_m=0.5,
        image_grid_shape=(2, 2),
    )

    assert output["source_component_ids"].tolist() == [0, 2]
    assert output["geometry_diagnostics"]["score"].tolist() == [1.0, 3.0]
    assert output["projective_anchor_observations"]["observation_offsets"].tolist() == [0, 1, 2]
    assert report["cycle_core_anchor_count"] == 1
    assert report["coverage_reserve_anchor_count"] == 1
    assert report["dropped_redundant_chain_anchor_count"] == 1


def test_chain_can_supply_a_new_image_cell_inside_an_existing_voxel() -> None:
    candidate = _candidate()
    candidate["anchor_xyz"][2] = torch.tensor([0.2, 0.0, 2.0])
    association = {
        "schema": ASSOCIATION_GRAPH_SCHEMA,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "component_statistics": {
            "cycle_supported": torch.tensor([True, False, False])
        },
    }
    output, report = select_cycle_core_coverage_reserve(
        candidate,
        association,
        keypoints_by_query=[
            torch.tensor([[10.0, 10.0], [11.0, 11.0]]),
            torch.tensor([[70.0, 70.0]]),
        ],
        image_hw_by_query=[(100, 100), (100, 100)],
        spatial_voxel_size_m=0.5,
        image_grid_shape=(2, 2),
    )
    assert output["source_component_ids"].tolist() == [0, 2]
    assert report["reserve_with_novel_spatial_voxel_count"] == 0
    assert report["reserve_with_novel_mapping_image_cell_count"] == 1
