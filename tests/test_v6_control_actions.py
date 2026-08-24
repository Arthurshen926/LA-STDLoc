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
    minimal_pose_correction_set,
    minimum_norm_score_boundary_action,
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
