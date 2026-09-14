from __future__ import annotations

import torch

from map_learning.v18_provenance_truth import TRUTH_AMBIGUOUS, TRUTH_UNIQUE
from map_learning.v19_track_extension_teacher import (
    TRACK_EXTENSION_TIERS,
    assign_track_extension_truth,
    full_map_projection_candidate_graph,
    prepare_track_observation_bank,
    track_observation_consensus,
)


def test_projection_candidates_scan_full_map_without_topl() -> None:
    graph = full_map_projection_candidate_graph(
        keypoints=torch.tensor([[50.0, 50.0]]),
        rendered_depth=torch.tensor([5.0]),
        query_indices=torch.tensor([0]),
        anchor_xyz=torch.tensor([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]]),
        anchor_covariance=torch.eye(3).repeat(2, 1, 1) * 1e-6,
        observation_count=torch.tensor([3, 3]),
        query_intrinsics=torch.tensor(
            [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]
        ),
        query_poses_w2c=torch.eye(4).repeat(1, 1, 1),
    )
    assert graph["candidate_anchor_rows"].tolist() == [0]
    assert graph["uses_topl_candidates"] is False


def test_track_consensus_recovers_mapping_identity_without_writing_track() -> None:
    graph = {
        "row_count": 1,
        "candidate_offsets": torch.tensor([0, 1]),
        "candidate_anchor_rows": torch.tensor([0]),
        "query_reprojection_residual_px": torch.tensor([0.0]),
        "query_normalized_depth_residual": torch.tensor([0.0]),
        "query_projection_std_px": torch.tensor([0.0]),
        "query_valid": torch.tensor([True]),
    }
    poses = torch.eye(4).repeat(3, 1, 1)
    consensus = track_observation_consensus(
        candidate_graph=graph,
        query_surface_xyz=torch.tensor([[0.0, 0.0, 5.0]]),
        query_descriptors=torch.tensor([[1.0, 0.0]]),
        anchor_observation_offsets=torch.tensor([0, 3]),
        observation_query_indices=torch.tensor([0, 1, 2]),
        observation_keypoint_indices=torch.tensor([0, 0, 0]),
        observation_enabled=torch.tensor([True, True, True]),
        mapping_keypoints=[torch.tensor([[50.0, 50.0]])] * 3,
        mapping_descriptors=[torch.tensor([[1.0, 0.0]])] * 3,
        mapping_intrinsics=torch.tensor(
            [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]
        ).repeat(3, 1, 1),
        mapping_poses_w2c=poses,
        mapping_view_family_ids=torch.tensor([0, 1, 2]),
    )
    truth = assign_track_extension_truth(
        candidate_graph=graph,
        consensus=consensus,
        equivalence_class_ids=torch.tensor([0]),
        tier=TRACK_EXTENSION_TIERS["tier_a"],
    )
    assert int(truth["truth_status"][0]) == TRUTH_UNIQUE
    assert truth["truth_anchor_rows"].tolist() == [0]
    assert truth["feedback_enters_track_registry"] is False


def test_multiple_identity_classes_abstain_as_ambiguous() -> None:
    graph = {
        "row_count": 1,
        "candidate_offsets": torch.tensor([0, 2]),
        "candidate_anchor_rows": torch.tensor([0, 1]),
        "query_reprojection_residual_px": torch.tensor([0.0, 0.0]),
        "query_normalized_depth_residual": torch.tensor([0.0, 0.0]),
        "query_projection_std_px": torch.tensor([0.0, 0.0]),
        "query_valid": torch.tensor([True]),
    }
    consensus = {
        "candidate_offsets": torch.tensor([0, 2]),
        "candidate_anchor_rows": torch.tensor([0, 1]),
        "transport_view_family_count": torch.tensor([3, 3]),
        "transport_median_residual_px": torch.tensor([0.0, 0.0]),
        "descriptor_view_family_count": torch.tensor([2, 2]),
        "descriptor_best_cosine": torch.tensor([1.0, 1.0]),
    }
    truth = assign_track_extension_truth(
        candidate_graph=graph,
        consensus=consensus,
        equivalence_class_ids=torch.tensor([0, 1]),
        tier=TRACK_EXTENSION_TIERS["tier_a"],
    )
    assert int(truth["truth_status"][0]) == TRUTH_AMBIGUOUS


def test_missing_true_track_support_cannot_make_wrong_projection_unique() -> None:
    graph = {
        "row_count": 1,
        "candidate_offsets": torch.tensor([0, 2]),
        "candidate_anchor_rows": torch.tensor([0, 1]),
        "query_reprojection_residual_px": torch.tensor([0.0, 0.0]),
        "query_normalized_depth_residual": torch.tensor([0.0, 0.0]),
        "query_projection_std_px": torch.tensor([0.0, 0.0]),
        "query_valid": torch.tensor([True]),
    }
    consensus = {
        "candidate_offsets": torch.tensor([0, 2]),
        "candidate_anchor_rows": torch.tensor([0, 1]),
        # Candidate 0 has no bank support; candidate 1 looks strong.  This is
        # insufficient to prove candidate 0 false.
        "transport_view_family_count": torch.tensor([0, 3]),
        "transport_median_residual_px": torch.tensor([float("inf"), 0.0]),
        "descriptor_view_family_count": torch.tensor([0, 2]),
        "descriptor_best_cosine": torch.tensor([-1.0, 1.0]),
    }
    truth = assign_track_extension_truth(
        candidate_graph=graph,
        consensus=consensus,
        equivalence_class_ids=torch.tensor([0, 1]),
        tier=TRACK_EXTENSION_TIERS["tier_a"],
    )
    assert int(truth["truth_status"][0]) == TRUTH_AMBIGUOUS


def test_prepared_track_bank_is_reusable() -> None:
    bank = prepare_track_observation_bank(
        anchor_observation_offsets=torch.tensor([0, 2]),
        observation_query_indices=torch.tensor([0, 1]),
        observation_keypoint_indices=torch.tensor([0, 0]),
        observation_enabled=torch.tensor([True, True]),
        mapping_keypoints=[torch.tensor([[50.0, 50.0]])] * 2,
        mapping_descriptors=[torch.tensor([[1.0, 0.0]])] * 2,
        mapping_view_family_ids=torch.tensor([0, 1]),
    )
    graph = {
        "row_count": 1,
        "candidate_offsets": torch.tensor([0, 1]),
        "candidate_anchor_rows": torch.tensor([0]),
    }
    consensus = track_observation_consensus(
        candidate_graph=graph,
        query_surface_xyz=torch.tensor([[0.0, 0.0, 5.0]]),
        query_descriptors=torch.tensor([[1.0, 0.0]]),
        anchor_observation_offsets=torch.tensor([0, 2]),
        observation_query_indices=torch.tensor([0, 1]),
        observation_keypoint_indices=torch.tensor([0, 0]),
        observation_enabled=torch.tensor([True, True]),
        mapping_keypoints=[torch.tensor([[50.0, 50.0]])] * 2,
        mapping_descriptors=[torch.tensor([[1.0, 0.0]])] * 2,
        mapping_intrinsics=torch.tensor(
            [[[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]]
        ).repeat(2, 1, 1),
        mapping_poses_w2c=torch.eye(4).repeat(2, 1, 1),
        mapping_view_family_ids=torch.tensor([0, 1]),
        prepared_observation_bank=bank,
    )
    assert consensus["transport_view_family_count"].tolist() == [2]
