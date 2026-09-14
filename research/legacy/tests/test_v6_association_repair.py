import torch

from common.v6_contracts import (
    FEEDBACK_SCHEMA,
    FEEDBACK_VERSION,
    exact_identity_positive_contract,
)
from map_learning.v6_association_repair import (
    deploy_association_repair_rule_globally,
    select_association_repair_pairs,
)


def _feedback(records):
    return {
        "schema": FEEDBACK_SCHEMA,
        "version": FEEDBACK_VERSION,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "positive_identity_contract": exact_identity_positive_contract(),
        "records": records,
    }


def test_association_repair_requires_repeated_training_evidence_and_no_view_conflict():
    state = {
        "anchor_xyz": torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "anchor_features": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 1, 2, 3]),
            "query_indices": torch.tensor([0, 2, 1]),
        },
    }
    records = []
    for _ in range(2):
        records.append(
            {
                "identity_inactive_pairs": torch.tensor([[3, 0]]),
                "certified_pose_valid_alternative_pairs": torch.tensor([[3, 1], [3, 2]]),
            }
        )
    pairs, report = select_association_repair_pairs(
        state,
        _feedback(records),
        training_query_indices=[0, 1],
        minimum_descriptor_similarity=0.9,
        maximum_xyz_distance_m=0.02,
        minimum_query_evidence=2,
    )
    assert pairs.tolist() == [[0, 1]]
    assert report["selected_pair_count"] == 1


def test_association_repair_rejects_overlapping_camera_support():
    state = {
        "anchor_xyz": torch.tensor([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]]),
        "anchor_features": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 1, 2]),
            "query_indices": torch.tensor([0, 0]),
        },
    }
    record = {
        "identity_inactive_pairs": torch.tensor([[0, 0]]),
        "certified_pose_valid_alternative_pairs": torch.tensor([[0, 1]]),
    }
    pairs, report = select_association_repair_pairs(
        state,
        _feedback([record]),
        training_query_indices=[0],
        minimum_descriptor_similarity=0.9,
        maximum_xyz_distance_m=0.02,
        minimum_query_evidence=1,
    )
    assert pairs.numel() == 0
    assert report["observation_conflict_pair_count"] == 1


def test_global_repair_deploys_certified_rule_without_validation_feedback():
    state = {
        "anchor_xyz": torch.tensor(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [1.0, 0.0, 0.0], [1.01, 0.0, 0.0]]
        ),
        "anchor_features": torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        "projective_anchor_observations": {
            "observation_offsets": torch.tensor([0, 1, 2, 3, 4]),
            "query_indices": torch.tensor([0, 1, 2, 3]),
        },
    }
    certified = torch.tensor([[0, 1]])
    report = {"selected_pair_evidence_counts": torch.tensor([5])}
    pairs, deployed = deploy_association_repair_rule_globally(
        state,
        certified,
        report,
        minimum_descriptor_similarity=0.9,
        maximum_xyz_distance_m=0.02,
    )
    assert pairs.tolist() == [[0, 1], [2, 3]]
    assert deployed["selected_pair_evidence_counts"].tolist() == [5, 0]
    assert deployed["selection_uses_validation_feedback"] is False
    assert deployed["feedback_certified_rule_deployed_globally"] is True
