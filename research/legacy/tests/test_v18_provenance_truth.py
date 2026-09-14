import torch

from map_learning.v18_provenance_truth import (
    TRUTH_AMBIGUOUS,
    TRUTH_EQUIVALENT,
    TRUTH_INVALID,
    TRUTH_NONE,
    TRUTH_UNIQUE,
    TruthAssignmentThresholds,
    aggregate_anchor_provenance,
    assign_full_map_projection_truth,
    assign_provenance_truth,
    backproject_query_surface,
    build_primitive_anchor_index,
    provenance_candidate_graph,
    query_anchor_geometry_evidence,
    transport_candidate_graph,
    truth_membership_mask,
)
from scripts.evaluate_v18_provenance_truth import _balanced_family_roles


def _provenance():
    # Anchor 0 is supported by primitive 7 in two independent view families.
    # Anchor 1 mixes primitive 7 and primitive 9. Anchor 2 only uses primitive 9.
    return aggregate_anchor_provenance(
        observation_offsets=torch.tensor([0, 2, 4, 5]),
        observation_primitive_ids=torch.tensor(
            [[7, -1], [7, -1], [7, 9], [7, 9], [9, -1]]
        ),
        observation_weights=torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [1.0, 0.0]]
        ),
        observation_view_family_ids=torch.tensor([0, 1, 0, 1, 2]),
    )


def test_anchor_provenance_is_view_balanced_and_invertible() -> None:
    provenance = _provenance()
    assert provenance["anchor_provenance_offsets"].tolist() == [0, 1, 3, 4]
    assert provenance["anchor_provenance_primitive_ids"].tolist() == [7, 7, 9, 9]
    assert torch.allclose(
        provenance["anchor_provenance_weights"],
        torch.tensor([1.0, 0.5, 0.5, 1.0]),
    )
    assert provenance["anchor_provenance_view_family_count"].tolist() == [2, 2, 1]
    inverse = build_primitive_anchor_index(provenance)
    assert inverse["primitive_ids"].tolist() == [7, 9]
    assert inverse["primitive_offsets"].tolist() == [0, 2, 4]
    assert inverse["anchor_rows"].tolist() == [0, 1, 1, 2]


def test_mapping_family_split_is_disjoint_and_count_balanced() -> None:
    roles = _balanced_family_roles(torch.arange(11), seed=1820260829)
    assert len(roles) == 11
    assert list(roles.values()).count("signature_design") == 7
    assert list(roles.values()).count("threshold_calibration") == 2
    assert list(roles.values()).count("independent_validation") == 2


def test_truth_candidates_are_full_map_and_descriptor_independent() -> None:
    inverse = build_primitive_anchor_index(_provenance())
    graph = provenance_candidate_graph(
        query_primitive_ids=torch.tensor([[7, -1], [9, -1], [8, -1]]),
        query_weights=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        primitive_anchor_index=inverse,
    )
    assert graph["uses_descriptor_scores"] is False
    assert graph["candidate_offsets"].tolist() == [0, 2, 4, 4]
    assert graph["candidate_anchor_rows"].tolist() == [0, 1, 2, 1]


def test_surface_transport_recovers_independent_track_identity() -> None:
    inverse = build_primitive_anchor_index(_provenance())
    graph = provenance_candidate_graph(
        query_primitive_ids=torch.tensor([[7]]),
        query_weights=torch.tensor([[1.0]]),
        primitive_anchor_index=inverse,
    )
    surface, valid = backproject_query_surface(
        torch.tensor([[10.0, 10.0]]),
        torch.tensor([5.0]),
        torch.tensor([[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]),
        torch.eye(4),
    )
    assert valid.tolist() == [True]
    mapping_keypoints = [
        torch.tensor([[10.0, 10.0], [30.0, 30.0]]),
        torch.tensor([[10.0, 10.0], [30.0, 30.0]]),
    ]
    transport = transport_candidate_graph(
        candidate_graph=graph,
        query_surface_xyz=surface,
        anchor_observation_offsets=torch.tensor([0, 2, 4]),
        observation_query_indices=torch.tensor([0, 1, 0, 1]),
        observation_keypoint_indices=torch.tensor([0, 0, 1, 1]),
        observation_enabled=None,
        mapping_keypoints=mapping_keypoints,
        mapping_intrinsics=torch.tensor(
            [
                [[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]],
                [[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]],
            ]
        ),
        mapping_poses_w2c=torch.eye(4).repeat(2, 1, 1),
        mapping_view_family_ids=torch.tensor([0, 1]),
    )
    # Candidate 0 transports exactly; candidate 1's observations are far away.
    assert transport["transport_view_family_count"].tolist() == [2, 0]
    truth = assign_provenance_truth(
        candidate_graph=graph,
        transport_evidence=transport,
        equivalence_class_ids=torch.arange(3),
        thresholds=TruthAssignmentThresholds(
            minimum_provenance_overlap=0.1,
            minimum_assignment_confidence=0.1,
            minimum_top1_top2_margin=0.01,
        ),
    )
    assert truth["truth_status"].tolist() == [TRUTH_UNIQUE]
    assert truth["truth_anchor_rows"].tolist() == [0]
    membership = truth_membership_mask(truth, torch.tensor([[1, 0]]))
    assert membership.tolist() == [[False, True]]


def test_query_geometry_rejects_same_provenance_wrong_track() -> None:
    graph = {
        "row_count": 1,
        "anchor_count": 2,
        "candidate_offsets": torch.tensor([0, 2]),
        "candidate_anchor_rows": torch.tensor([0, 1]),
        "bhattacharyya_overlap": torch.tensor([0.8, 0.8]),
        "query_valid": torch.tensor([True]),
    }
    geometry = query_anchor_geometry_evidence(
        candidate_graph=graph,
        query_keypoints=torch.tensor([[10.0, 10.0]]),
        query_depth=torch.tensor([5.0]),
        query_indices=torch.tensor([0]),
        anchor_xyz=torch.tensor([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]]),
        anchor_covariance=torch.eye(3).repeat(2, 1, 1) * 1e-4,
        query_intrinsics=torch.tensor(
            [[[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]]
        ),
        query_poses_w2c=torch.eye(4)[None],
    )
    assert torch.allclose(
        geometry["query_reprojection_residual_px"],
        torch.tensor([0.0, 20.0 * 2.0**0.5]),
    )
    transport = {
        "candidate_offsets": graph["candidate_offsets"],
        "candidate_anchor_rows": graph["candidate_anchor_rows"],
        "transport_view_family_count": torch.tensor([2, 2]),
        "transport_median_residual_px": torch.tensor([0.1, 0.1]),
    }
    truth = assign_provenance_truth(
        candidate_graph=graph,
        transport_evidence=transport,
        geometry_evidence=geometry,
        equivalence_class_ids=torch.arange(2),
        thresholds=TruthAssignmentThresholds(
            minimum_provenance_overlap=0.1,
            minimum_transport_view_families=2,
            minimum_assignment_confidence=0.1,
            minimum_top1_top2_margin=0.01,
            maximum_query_reprojection_px=4.0,
            maximum_query_normalized_depth_residual=1.0,
            maximum_query_projection_std_px=2.0,
        ),
    )
    assert truth["truth_status"].tolist() == [TRUTH_UNIQUE]
    assert truth["truth_anchor_rows"].tolist() == [0]


def test_truth_states_distinguish_equivalent_ambiguous_and_none() -> None:
    graph = {
        "row_count": 3,
        "anchor_count": 3,
        "candidate_offsets": torch.tensor([0, 2, 4, 4]),
        "candidate_anchor_rows": torch.tensor([0, 1, 0, 2]),
        "bhattacharyya_overlap": torch.tensor([0.8, 0.75, 0.8, 0.79]),
        "query_valid": torch.tensor([True, True, True]),
    }
    transport = {
        "candidate_offsets": graph["candidate_offsets"],
        "candidate_anchor_rows": graph["candidate_anchor_rows"],
        "transport_view_family_count": torch.tensor([2, 2, 2, 2]),
        "transport_median_residual_px": torch.tensor([0.1, 0.2, 0.1, 0.1]),
    }
    truth = assign_provenance_truth(
        candidate_graph=graph,
        transport_evidence=transport,
        equivalence_class_ids=torch.tensor([5, 5, 9]),
        thresholds=TruthAssignmentThresholds(
            minimum_provenance_overlap=0.1,
            minimum_assignment_confidence=0.1,
            minimum_top1_top2_margin=0.05,
            minimum_top1_top2_ratio=1.05,
        ),
    )
    assert truth["truth_status"].tolist() == [
        TRUTH_EQUIVALENT,
        TRUTH_AMBIGUOUS,
        TRUTH_NONE,
    ]
    assert truth["truth_anchor_rows"].tolist() == [0, 1]


def test_projection_comparator_scans_full_map_without_topl() -> None:
    truth = assign_full_map_projection_truth(
        keypoints=torch.tensor([[10.0, 10.0], [80.0, 80.0]]),
        rendered_depth=torch.tensor([5.0, 5.0]),
        query_indices=torch.tensor([0, 0]),
        anchor_xyz=torch.tensor([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]]),
        anchor_covariance=torch.eye(3).repeat(2, 1, 1) * 1e-4,
        observation_count=torch.tensor([3, 3]),
        mapping_intrinsics=torch.tensor(
            [[[100.0, 0.0, 10.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]]
        ),
        mapping_poses_w2c=torch.eye(4)[None],
        equivalence_class_ids=torch.arange(2),
    )
    assert truth["uses_topl_candidates"] is False
    assert truth["truth_status"].tolist() == [TRUTH_UNIQUE, TRUTH_NONE]
    assert truth["truth_anchor_rows"].tolist() == [0]


def test_entropy_and_depth_boundary_are_uncertain_not_forced_truth() -> None:
    inverse = build_primitive_anchor_index(_provenance())
    graph = provenance_candidate_graph(
        query_primitive_ids=torch.tensor([[7], [7]]),
        query_weights=torch.ones(2, 1),
        primitive_anchor_index=inverse,
        query_composition_entropy=torch.tensor([0.1, 1.5]),
        query_relative_depth_spread=torch.tensor([0.01, 0.5]),
    )
    transport = {
        "candidate_offsets": graph["candidate_offsets"],
        "candidate_anchor_rows": graph["candidate_anchor_rows"],
        "transport_view_family_count": torch.full(
            (graph["candidate_anchor_rows"].numel(),), 2
        ),
        "transport_median_residual_px": torch.zeros(
            graph["candidate_anchor_rows"].numel()
        ),
    }
    truth = assign_provenance_truth(
        candidate_graph=graph,
        transport_evidence=transport,
        equivalence_class_ids=torch.arange(3),
        thresholds=TruthAssignmentThresholds(
            minimum_provenance_overlap=0.1,
            minimum_assignment_confidence=0.1,
            minimum_top1_top2_margin=0.01,
            maximum_composition_entropy=1.0,
            maximum_relative_depth_spread=0.1,
        ),
    )
    assert truth["truth_status"].tolist() == [TRUTH_UNIQUE, TRUTH_INVALID]
