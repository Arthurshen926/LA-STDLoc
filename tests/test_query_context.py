import torch

from localization_training.full_primitive_retrieval import (
    ambiguity_gated_context_topk,
)
from localization_training.query_context import (
    spatial_pyramid_global_descriptor,
    visibility_context_bias,
)


def test_spatial_pyramid_context_is_fixed_size_and_normalized():
    descriptors = torch.eye(4)
    keypoints = torch.tensor(
        [[1.0, 1.0], [9.0, 1.0], [1.0, 9.0], [9.0, 9.0]]
    )
    context = spatial_pyramid_global_descriptor(
        descriptors,
        keypoints,
        torch.ones(4),
        (10, 10),
    )
    assert context.shape == (20,)
    assert torch.allclose(context.norm(), torch.tensor(1.0))


def test_visibility_context_bias_favors_visible_support_landmarks():
    bias, diagnostics = visibility_context_bias(
        torch.tensor([1.0, 0.0]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[True, False], [False, True]]),
        nearest_views=1,
        delta_max=0.02,
    )
    assert bias[0] > bias[1]
    assert diagnostics["nearest_similarity_max"] == 1.0


def test_global_lift_does_not_reward_universally_visible_landmarks():
    bias, _ = visibility_context_bias(
        torch.tensor([1.0, 0.0]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor(
            [[True, True, False], [True, False, True]]
        ),
        nearest_views=1,
        delta_max=0.02,
        prior_scale=1.0,
        normalization="global_lift",
    )
    assert torch.isclose(bias[0], torch.tensor(0.0))
    assert bias[1] > bias[0] > bias[2]


def test_context_retrieval_protects_high_margin_rows():
    query = torch.tensor([[1.0, 0.0], [0.92, 0.38]])
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]
    )
    result, ambiguous, _ = ambiguity_gated_context_topk(
        query,
        features,
        torch.tensor([-0.02, -0.02, 0.02]),
        margin_threshold=0.1,
        topk=1,
        chunk_size=2,
    )
    assert ambiguous.tolist() == [False, True]
    assert result.indices[:, 0].tolist() == [0, 2]
