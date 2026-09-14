import torch

from map_learning.v18_sparse_descriptor_audit import (
    audit_observation_convex_descriptor,
)


def test_sparse_descriptor_audit_requires_multifamily_safe_gain() -> None:
    native = torch.tensor([1.0, 0.0])
    observations = torch.tensor([[1.0, 0.0], [0.8, 0.6]])
    repair_query = torch.tensor([[0.8, 0.6], [0.75, 0.66]])
    result = audit_observation_convex_descriptor(
        anchor_row=4,
        native_descriptor=native,
        observation_descriptors=observations,
        repair_query_descriptors=repair_query,
        repair_competitor_scores=torch.tensor([0.85, 0.84]),
        repair_pose_family_ids=torch.tensor([10, 20]),
        repair_weights=torch.ones(2),
        protection_query_descriptors=torch.tensor([[1.0, 0.0]]),
        protection_competitor_scores=torch.tensor([0.7]),
        maximum_angle_deg=40.0,
        steps=200,
    )
    assert result["authorized"] is True
    assert result["improving_pose_family_count"] == 2
    assert result["broken_protection_row_count"] == 0
