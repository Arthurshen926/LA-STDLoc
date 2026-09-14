from types import SimpleNamespace

import torch

from common.v6_contracts import (
    FEEDBACK_SCHEMA,
    FEEDBACK_VERSION,
    exact_identity_positive_contract,
)
from map_learning.v6_control_actions import (
    ControlActionUnavailable,
    control_oriented_descriptor_proposal,
    expand_probe_prototype_support,
    minimal_pose_correction_set,
    minimum_norm_score_boundary_action,
    nearest_mapping_observation_prototypes,
    pose_priority_prefix_correction_set,
    probe_conditioned_sparse_prototype_proposal,
)


def _counting_solver(_keypoints, points, _intrinsics, **_kwargs):
    corrected = int((torch.as_tensor(points)[:, 0] > 0.5).sum())
    pose = torch.eye(4)
    if corrected < 2:
        pose[0, 3] = 1.0
    return SimpleNamespace(pose_w2c=pose.numpy(), inliers=list(range(len(points))))


def test_minimal_pose_correction_set_replays_discrete_solver_boundary() -> None:
    result = minimal_pose_correction_set(
        keypoints=torch.zeros((4, 2)),
        xyz=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        winners=torch.zeros(4, dtype=torch.long),
        candidate_rows=torch.arange(4),
        candidate_positive_anchors=torch.ones(4, dtype=torch.long),
        candidate_priority=torch.tensor([4.0, 3.0, 2.0, 1.0]),
        intrinsics=torch.eye(3),
        ground_truth_pose_w2c=torch.eye(4),
        reprojection_error_px=4.0,
        maximum_set_size=3,
        beam_width=2,
        solver=_counting_solver,
    )
    assert result["correction_found"] is True
    assert result["selected_rows"].numel() == 2
    assert result["baseline"]["success"] is False
    assert result["best"]["success"] is True
    assert result["evaluated_action_set_count"] < result["enumerated_action_set_count"]


def test_pose_priority_prefix_crosses_large_discrete_basin() -> None:
    result = pose_priority_prefix_correction_set(
        keypoints=torch.zeros((16, 2)),
        xyz=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        winners=torch.zeros(16, dtype=torch.long),
        candidate_rows=torch.arange(16),
        candidate_positive_anchors=torch.ones(16, dtype=torch.long),
        candidate_priority=torch.arange(16).float(),
        intrinsics=torch.eye(3),
        ground_truth_pose_w2c=torch.eye(4),
        reprojection_error_px=4.0,
        initial_set_size=1,
        solver=_counting_solver,
    )
    assert result["correction_found"] is True
    assert result["selected_rows"].numel() == 2
    assert [row["prefix_size"] for row in result["prefix_evaluations"]] == [0, 1, 2]


def test_minimum_norm_action_crosses_requested_cosine_boundary() -> None:
    result = minimum_norm_score_boundary_action(
        anchor_features=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        query_descriptors=torch.tensor([[1.0, 0.0]]),
        positive_anchors=torch.tensor([0]),
        negative_anchors=torch.tensor([1]),
        target_margins=torch.tensor([0.1]),
    )
    assert result["feasible"] is True
    assert result["maximum_violation"] < 1e-5
    assert result["achieved_linearized_margins"][0] >= 0.1 - 1e-5
    assert torch.allclose(
        result["action"][1], torch.zeros(2), atol=1e-6
    )


class _Observations:
    names = ("q",)

    def __len__(self):
        return 1

    @staticmethod
    def build_view(_index):
        return SimpleNamespace(
            image_name="q",
            physical_keypoints=torch.zeros((4, 2)),
            descriptors=torch.tensor([[1.0, 0.0]]).repeat(4, 1),
            intrinsics=torch.eye(3),
            pose_w2c=torch.eye(4),
        )


def test_control_proposal_uses_only_pose_changing_controllable_rows() -> None:
    state = {
        "anchor_features": torch.tensor([[1.0, 0.0], [0.99, 0.14]]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
    }
    triplets = torch.tensor(
        [[row, 1, 0, 0] for row in range(4)], dtype=torch.long
    )
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "positive_identity_contract": exact_identity_positive_contract(),
        "query_names": ["q"],
        "records": [
            {
                "pose_success": False,
                "winner_anchor_ids": torch.zeros(4, dtype=torch.long),
                "descriptor_triplets": triplets,
                "descriptor_triplet_pose_weights": torch.ones(4),
                "certified_pose_valid_alternative_pairs": torch.empty((0, 2)),
                "top1_negative_mask": torch.ones(4, dtype=torch.bool),
            }
        ],
    }
    proposal = control_oriented_descriptor_proposal(
        state,
        _Observations(),
        feedback,
        training_query_indices=torch.tensor([0]),
        trust_region=0.2,
        margin=0.01,
        reprojection_error_px=4.0,
        maximum_correction_set_size=3,
        solver=_counting_solver,
    )
    report = proposal["v6_descriptor_distillation"]
    assert report["accepted_query_indices"].tolist() == [0]
    assert report["accepted_constraint_count"] == 2
    assert report["updated_anchor_count"] == 2
    assert not torch.equal(proposal["anchor_features"], state["anchor_features"])


def test_unavailable_control_action_preserves_per_query_audit() -> None:
    state = {
        "anchor_features": torch.tensor([[1.0, 0.0], [0.99, 0.14]]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
    }
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "positive_identity_contract": exact_identity_positive_contract(),
        "query_names": ["q"],
        "records": [
            {
                "pose_success": False,
                "winner_anchor_ids": torch.zeros(4, dtype=torch.long),
                "descriptor_triplets": torch.empty((0, 4), dtype=torch.long),
                "descriptor_triplet_pose_weights": torch.empty(0),
                "certified_pose_valid_alternative_pairs": torch.empty((0, 2)),
                "top1_negative_mask": torch.ones(4, dtype=torch.bool),
            }
        ],
    }
    try:
        control_oriented_descriptor_proposal(
            state,
            _Observations(),
            feedback,
            training_query_indices=torch.tensor([0]),
            trust_region=0.2,
            margin=0.01,
            reprojection_error_px=4.0,
            solver=_counting_solver,
        )
    except ControlActionUnavailable as error:
        assert len(error.audits) == 1
        assert error.audits[0]["controller_route"] == "structure_or_prior_limited"
        assert error.audits[0]["candidate_action_count"] == 0
    else:
        raise AssertionError("expected a preserved unavailable control audit")


def test_control_proposal_falls_back_to_verified_anchor_suppression() -> None:
    state = {
        "anchor_ids": torch.arange(3),
        "anchor_features": torch.tensor(
            [[1.0, 0.0], [0.99, 0.14], [0.999, 0.04]]
        ),
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 1, 2, 3]),
            "query_indices": torch.tensor([0, 0, 0]),
            "keypoint_indices": torch.tensor([0, 1, 2]),
        },
    }
    triplets = torch.tensor(
        [[row, 1, 0, 0] for row in range(4)], dtype=torch.long
    )
    feedback = {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "positive_identity_contract": exact_identity_positive_contract(),
        "query_names": ["q"],
        "records": [
            {
                "pose_success": False,
                "winner_anchor_ids": torch.zeros(4, dtype=torch.long),
                "winner_identity_correct_mask": torch.zeros(4, dtype=torch.bool),
                "descriptor_triplets": triplets,
                "descriptor_triplet_pose_weights": torch.ones(4),
                "certified_pose_valid_alternative_pairs": torch.empty((0, 2)),
                "top1_negative_mask": torch.ones(4, dtype=torch.bool),
            }
        ],
    }
    proposal = control_oriented_descriptor_proposal(
        state,
        _Observations(),
        feedback,
        training_query_indices=torch.tensor([0]),
        trust_region=0.001,
        margin=0.01,
        reprojection_error_px=4.0,
        maximum_correction_set_size=3,
        solver=_counting_solver,
    )
    report = proposal["v6_selection_distillation"]
    assert report["suppressed_source_anchor_rows"].tolist() == [0, 2]
    assert report["selected_query_indices"].tolist() == [0]
    assert proposal["anchor_ids"].numel() == 1


class _ProbeObservations:
    names = ("probe-train", "probe-validation")

    def __len__(self):
        return 2

    @staticmethod
    def build_view(index):
        return SimpleNamespace(
            image_name=_ProbeObservations.names[index],
            physical_keypoints=torch.zeros((4, 2)),
            descriptors=torch.tensor([[1.0, 0.0]]).repeat(4, 1),
            intrinsics=torch.eye(3),
            pose_w2c=torch.eye(4),
        )


def test_probe_sparse_prototype_uses_training_only_and_collapses_to_owner() -> None:
    state = {
        "anchor_ids": torch.arange(2),
        "anchor_features": torch.tensor([[0.99, 0.14], [0.0, 1.0]]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        "provenance": {
            "uses_source_mapping_rgb": False,
            "uses_test_queries": False,
        },
    }
    records = []
    for index, split in enumerate(("training", "validation")):
        records.append(
            {
                "query_index": index,
                "image_name": _ProbeObservations.names[index],
                "control_split": split,
                "pose_success": False,
                "oracle_available": True,
                "oracle_pose_success": True,
                "controller_route": "descriptor_controllable",
                "winner_anchor_ids": [0, 0, 0, 0],
                "winner_scores": [0.99, 0.99, 0.99, 0.99],
                "descriptor_triplets": [
                    [row, 1, 0, 0] for row in range(4)
                ],
                "descriptor_triplet_pose_weights": [1.0] * 4,
            }
        )
    feedback = {
        "schema": "lafgs_v6_fixed_map_virtual_probe_evaluation",
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "inputs": {
            "source_map_sha256": "a" * 64,
            "probe_cache_sha256": "b" * 64,
        },
        "control_split": {
            "training_query_indices": [0],
            "validation_query_indices": [1],
            "validation_used_by_controller": False,
            "sensor_variants_share_their_pose_partition": True,
        },
        "records": records,
    }
    proposal = probe_conditioned_sparse_prototype_proposal(
        state,
        _ProbeObservations(),
        feedback,
        source_map_sha256="a" * 64,
        probe_cache_sha256="b" * 64,
        probe_feedback_sha256="c" * 64,
        reprojection_error_px=4.0,
        maximum_correction_set_size=3,
        solver=_counting_solver,
    )
    report = proposal["v6_probe_prototype_control"]
    assert report["training_replay"]["recovered_query_indices"] == [0]
    assert report["validation_query_indices_used_by_controller"].numel() == 0
    assert proposal["anchor_extra_prototype_owner_rows"].tolist() == [1]


class _MappingObservations:
    names = ("seq1/q0", "seq2/q1")

    def __len__(self):
        return 2

    @staticmethod
    def build_view(index):
        descriptors = (
            torch.tensor([[0.0, 1.0], [1.0, 0.0]])
            if index == 0
            else torch.tensor([[1.0, 1.0], [-1.0, 0.0]])
        )
        return SimpleNamespace(descriptors=descriptors)


def test_probe_selects_nearest_eligible_real_mapping_observation() -> None:
    state = {
        "anchor_xyz": torch.zeros((1, 3)),
        "anchor_features": torch.tensor([[0.0, 1.0]]),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 3]),
            "query_indices": torch.tensor([0, 0, 1]),
            "keypoint_indices": torch.tensor([0, 1, 0]),
        },
    }
    selected = nearest_mapping_observation_prototypes(
        state,
        _MappingObservations(),
        owner_rows=torch.tensor([0]),
        target_descriptors=torch.tensor([[0.9, 0.1]]),
        eligible_query_indices=torch.tensor([0]),
    )
    assert torch.equal(selected, torch.tensor([[1.0, 0.0]]))


def test_probe_coverage_expands_training_support_without_validation() -> None:
    state = {
        "anchor_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "anchor_xyz": torch.zeros((2, 3)),
    }
    feedback = {
        "control_split": {
            "training_query_indices": [0],
            "validation_query_indices": [1],
            "validation_used_by_controller": False,
        },
        "records": [
            {
                "control_split": "training",
                "controller_route": "descriptor_controllable",
                "descriptor_triplets": [[0, 1, 0, 0]],
                "descriptor_triplet_pose_weights": [1.0],
            },
            {"control_split": "validation"},
        ],
    }
    result = expand_probe_prototype_support(
        state,
        _ProbeObservations(),
        feedback,
        maximum_total_prototypes=2,
        maximum_prototypes_per_anchor=1,
    )
    assert result["anchor_extra_prototype_owner_rows"].tolist() == [1]
    report = result["v6_probe_prototype_coverage"]
    assert report["added_prototype_count"] == 1
    assert report["validation_query_indices_used_by_controller"].numel() == 0
