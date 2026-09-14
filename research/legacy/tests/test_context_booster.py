import torch

from map_learning.context_booster import (
    FeatureBooster,
    SUPERPOINT_BOOST_F_CONFIG,
    normalize_keypoint_properties,
)
from map_learning.context_booster_crossfit import accumulate_view_descriptors


def test_official_superpoint_boost_f_shape_and_normalization():
    torch.manual_seed(7)
    model = FeatureBooster().eval()
    descriptors = torch.randn(5, 256)
    properties = torch.randn(5, 3)

    output = model(descriptors, properties)

    assert SUPERPOINT_BOOST_F_CONFIG["Attentional_layers"] == 9
    assert sum(parameter.numel() for parameter in model.parameters()) == 5_116_992
    assert output.shape == (5, 256)
    torch.testing.assert_close(output.norm(dim=1), torch.ones(5))


def test_normalize_keypoint_properties_matches_official_contract():
    keypoints = torch.tensor([[0.0, 0.0], [320.0, 240.0], [640.0, 480.0]])
    scores = torch.tensor([0.1, 0.5, 0.9])

    properties = normalize_keypoint_properties(keypoints, scores, (480, 640))

    scale = 640.0 * 0.7
    expected = torch.tensor(
        [
            [-320.0 / scale, -240.0 / scale, 0.1],
            [0.0, 0.0, 0.5],
            [320.0 / scale, 240.0 / scale, 0.9],
        ]
    )
    torch.testing.assert_close(properties, expected)


def test_observation_fusion_is_view_balanced_per_anchor():
    accumulator = torch.zeros((3, 2))
    counts = torch.zeros(3, dtype=torch.long)

    first_observed = accumulate_view_descriptors(
        accumulator,
        counts,
        torch.tensor([0, 0, 1]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]]),
    )
    second_observed = accumulate_view_descriptors(
        accumulator,
        counts,
        torch.tensor([0]),
        torch.tensor([[0.0, 3.0]]),
    )

    assert first_observed == 2
    assert second_observed == 1
    assert counts.tolist() == [2, 1, 0]
    torch.testing.assert_close(
        accumulator,
        torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]),
    )
