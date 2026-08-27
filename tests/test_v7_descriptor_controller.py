import torch

from map_learning.v7_descriptor_controller import reconstruct_v7_descriptors


def _bank():
    return {
        10: {
            "descriptors": torch.tensor(
                [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.8, 0.0, 0.6]]
            ),
            "view_families": torch.tensor([0, 1, 2]),
        }
    }


def _evidence(families):
    return [
        {
            "pose_family_id": family,
            "query_descriptors": torch.tensor([[0.8, 0.6, 0.0]]),
            "positive_anchor_ids": torch.tensor([10]),
            "false_attractor_anchor_ids": torch.tensor([99]),
        }
        for family in families
    ]


def test_one_pose_family_is_an_exact_noop() -> None:
    current = torch.tensor([[2.0, 0.0, 0.0]])
    result = reconstruct_v7_descriptors(
        anchor_ids=torch.tensor([10]),
        current_descriptors=current,
        feedback_evidence=_evidence([3]),
        observation_banks={},
    )
    assert result["changed_anchor_count"] == 0
    assert torch.equal(result["anchor_features"], current)


def test_two_pose_families_reconstruct_only_from_mapping_observations() -> None:
    current = torch.tensor([[1.0, 0.0, 0.0]])
    query = _evidence([3, 4])[0]["query_descriptors"][0]
    result = reconstruct_v7_descriptors(
        anchor_ids=torch.tensor([10]),
        current_descriptors=current,
        feedback_evidence=_evidence([3, 4]),
        observation_banks=_bank(),
        maximum_descriptor_angle_deg=5.0,
    )
    assert result["changed_anchor_count"] == 1
    assert result["feedback_descriptors_copied_into_map"] is False
    assert not torch.equal(result["anchor_features"][0], query)
    assert result["audits"][0]["descriptor_angle_deg"] <= 5.0001


def test_conflicting_sign_in_same_family_blocks_update() -> None:
    evidence = _evidence([3, 4])
    evidence[0]["false_attractor_anchor_ids"] = torch.tensor([10])
    result = reconstruct_v7_descriptors(
        anchor_ids=torch.tensor([10]),
        current_descriptors=torch.tensor([[1.0, 0.0, 0.0]]),
        feedback_evidence=evidence,
        observation_banks={},
    )
    assert result["changed_anchor_count"] == 0
