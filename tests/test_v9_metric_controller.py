import torch

from map_learning.v9_metric_controller import (
    train_v9_shared_metric,
    transform_map_anchor_features,
)


def test_v9_metric_is_shared_bounded_and_does_not_create_prototypes() -> None:
    anchors = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.7, 0.7, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    query = torch.tensor([[0.75, 0.65, 0.05, 0.0]]).repeat(16, 1)
    metric, report = train_v9_shared_metric(
        anchor_features=anchors,
        query_descriptors=query,
        positive_anchor_rows=torch.full((16,), 2),
        negative_anchor_rows=torch.full((16,), 1),
        sample_weights=torch.ones(16),
        clean_query_descriptors=anchors[:1],
        clean_positive_anchor_rows=torch.tensor([0]),
        clean_negative_anchor_rows=torch.tensor([1]),
        clean_initial_margin=torch.tensor([0.2]),
        rank=2,
        maximum_residual_norm=0.2,
        steps=100,
        learning_rate=0.01,
        device="cpu",
    )
    transformed = transform_map_anchor_features(metric, anchors, device="cpu")
    assert transformed.shape == anchors.shape
    assert report["loo_used"] is False
    assert report["feedback_descriptors_copied_into_map"] is False
    assert report["maximum_observed_residual_norm"] <= 0.200001
