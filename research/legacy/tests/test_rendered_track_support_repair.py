import torch

from scripts.materialize_rendered_track_support_repair import (
    _coverage_certification,
    _limit_children_after_triangulation,
    _remap_pair_sidecar_tracks,
)
from topology.adaptive_distillation import (
    _attach_support_repair_lineage,
    _track_only_source_capacity_ids,
)


def _geometry() -> dict:
    return {
        "triangulated_xyz": torch.tensor(
            [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]
        ),
        "triangulated": torch.tensor([True, True, True, True]),
        "triangulation_observation_count": torch.tensor([20, 8, 7, 6]),
        "triangulation_distinct_view_bin_count": torch.tensor([2, 2, 2, 4]),
        "triangulation_reprojection_median_px": torch.tensor([8.0, 1.0, 1.5, 0.2]),
        "triangulation_reprojection_p90_px": torch.tensor([30.0, 2.0, 3.0, 0.4]),
        "triangulation_parallax_deg": torch.tensor([2.0, 2.0, 1.5, 10.0]),
        "triangulation_condition_number": torch.ones(4),
        "triangulation_covariance_trace": torch.tensor([0.1, 0.1, 0.1, 0.001]),
        "triangulation_covariance_matrix": torch.eye(3).repeat(4, 1, 1),
        "track_confidence_level": torch.tensor([2, 2, 2, 2], dtype=torch.int8),
    }


def test_child_cap_runs_after_geometry_and_keeps_stable_smaller_child():
    tracks = {
        "track_index": torch.arange(4),
        "query_index": torch.zeros(4, dtype=torch.long),
        "keypoint_index": torch.arange(4),
        "confidence": torch.tensor([1.0, 0.8, 0.7, 0.6]),
        "track_level": torch.tensor([2, 2, 2, 2], dtype=torch.int8),
    }
    revised, geometry, old_to_new, diagnostics = _limit_children_after_triangulation(
        tracks,
        _geometry(),
        [torch.tensor([9, 9, 9, 9])],
        maximum_children=2,
    )
    # Child 0 has the most observations but fails the broad geometry gate.
    # Child 3 is smaller yet has the strongest triangulation and must survive.
    assert old_to_new[0].item() == -1
    assert old_to_new[3].item() >= 0
    assert revised["parent_source_track_ids"].tolist() == [9, 9]
    assert sorted(revised["repair_child_index"].tolist()) == [0, 1]
    assert geometry["triangulated_xyz"].shape[0] == 2
    assert diagnostics["dropped_geometry_ineligible_child_count"] == 1
    assert diagnostics["dropped_excess_child_count"] == 1
    assert diagnostics["child_cap_stage"].startswith("after_ray_triangulation")


def test_pair_sidecar_track_csr_is_remapped_after_child_filtering():
    sidecar = {
        "schema": "lafgs_mapping_track_pair_sidecar",
        "pair": {
            "left_query_index": torch.tensor([0]),
            "right_query_index": torch.tensor([1]),
            "final_track_offsets": torch.tensor([0, 3]),
            "final_track_indices": torch.tensor([0, 1, 3]),
            "final_component_edge_count": torch.tensor([3]),
        },
    }
    tracks = {"track_level": torch.tensor([2, 2], dtype=torch.int8)}
    geometry = {
        "triangulated_xyz": torch.tensor([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        "triangulated": torch.tensor([True, True]),
    }
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = -0.1
    revised = _remap_pair_sidecar_tracks(
        sidecar, torch.tensor([-1, 0, -1, 1]), tracks, geometry, poses
    )
    pair = revised["pair"]
    assert pair["final_track_indices"].tolist() == [0, 1]
    assert pair["final_track_offsets"].tolist() == [0, 2]
    assert pair["final_component_edge_count"].tolist() == [2]
    assert int(pair["final_track_indices"].max()) < 2


def test_coverage_requires_current_observation_reprojection_support():
    tracks = {
        "track_index": torch.tensor([0]),
        "query_index": torch.tensor([0]),
        "keypoint_index": torch.tensor([0]),
        "track_level": torch.tensor([2], dtype=torch.int8),
    }
    geometry = {
        "triangulated_xyz": torch.tensor([[0.0, 0.0, 2.0]]),
        "triangulated": torch.tensor([True]),
        "triangulation_distinct_view_bin_count": torch.tensor([2]),
        "triangulation_parallax_deg": torch.tensor([5.0]),
        "triangulation_reprojection_p90_px": torch.tensor([1.0]),
    }
    intrinsic = torch.tensor(
        [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]
    )
    result = _coverage_certification(
        tracks=tracks,
        geometry=geometry,
        support_records=[{"depth": torch.tensor([2.0]), "valid": torch.tensor([True])}],
        keypoints=[torch.tensor([[60.0, 50.0]])],
        intrinsics=intrinsic,
        poses=torch.eye(4).repeat(1, 1, 1),
        depth_uncertainty=[torch.tensor([0.0])],
        depth_abs_tolerance_m=0.05,
        depth_relative_tolerance=0.02,
        minimum_view_bins=2,
        minimum_parallax_deg=1.0,
        maximum_reprojection_px=2.0,
    )
    assert result["observation_reprojection_px"].item() == 10.0
    assert result["coverage_certified"].tolist() == [False]
    assert result["identity_positive_certified"].tolist() == [False]


def test_track_only_source_capacity_uses_repair_parent_identity():
    payload = {"tracks": {"parent_source_track_ids": torch.tensor([7, 7, 9])}}
    assert _track_only_source_capacity_ids(payload, 3).tolist() == [7, 7, 9]
    assert _track_only_source_capacity_ids({"tracks": {}}, 3).tolist() == [0, 1, 2]


def test_compact_map_propagates_selected_support_repair_lineage():
    state = {"track_centric_reconstruction": {}}
    payload = {
        "track_geometry": {"triangulated": torch.ones(3, dtype=torch.bool)},
        "tracks": {
            "parent_source_track_ids": torch.tensor([7, 7, 9]),
            "repair_child_index": torch.tensor([0, 1, 0]),
            "repair_parent_child_count": torch.tensor([2, 2, 1]),
        },
    }
    _attach_support_repair_lineage(state, payload, torch.tensor([1, 2]), base_count=1)
    assert state["parent_source_track_ids"].tolist() == [7, 9, -1]
    assert state["repair_child_index"].tolist() == [1, 0, -1]
    assert state["repair_parent_child_count"].tolist() == [2, 1, -1]
    assert (
        state["track_centric_reconstruction"]["support_repair_parent_lineage"][
            "base_sentinel"
        ]
        == -1
    )


def test_compact_map_rejects_partial_support_repair_lineage():
    state = {"track_centric_reconstruction": {}}
    payload = {
        "track_geometry": {"triangulated": torch.ones(1, dtype=torch.bool)},
        "tracks": {"parent_source_track_ids": torch.tensor([7])},
    }
    try:
        _attach_support_repair_lineage(state, payload, torch.tensor([0]), base_count=0)
    except ValueError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("partial support-repair lineage was accepted")
