import torch


def test_local_correlation_patch_tracks_spatial_peak():
    from localization_training.pair_measurement import sample_local_correlation_patch

    feature_map = torch.zeros(2, 5, 5)
    feature_map[0] = 1.0
    feature_map[:, 2, 3] = torch.tensor([0.0, 1.0])
    patch = sample_local_correlation_patch(
        feature_map,
        torch.tensor([[2.0, 2.0]]),
        torch.tensor([[0.0, 1.0]]),
        radius=1,
    )

    assert patch.shape == (1, 9)
    assert int(patch.argmax(dim=1).item()) == 5


def test_pair_geometry_features_encode_image_and_map_coordinates():
    from localization_training.pair_measurement import build_pair_geometry_features

    features = build_pair_geometry_features(
        torch.tensor([[50.0, 25.0], [100.0, 50.0]]),
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]),
        torch.tensor(
            [[-2.0, -4.0, -6.0], [0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]
        ),
        (100, 200),
    )

    assert features.shape == (2, 5)
    assert torch.allclose(features[0, :2], torch.tensor([-0.5, -0.5]))
    assert torch.all(features[:, 2:].abs() <= 1.0)
    assert not torch.allclose(features[0, 2:], features[1, 2:])


def test_pair_measurement_outputs_positive_covariance_and_gradients():
    from localization_training.pair_measurement import (
        PairMeasurementHead,
        gaussian_measurement_nll,
    )

    head = PairMeasurementHead(
        descriptor_dim=4,
        patch_radius=1,
        hidden_dim=8,
        max_offset=2.0,
        covariance_floor=0.1,
        initial_sigma=0.8,
    )
    pair_features = torch.randn(6, 6)
    patch = torch.randn(6, 9)
    query = torch.randn(6, 4)
    landmark = torch.randn(6, 4)
    output = head(pair_features, patch, query, landmark)

    assert output.inlier_logits.shape == (6,)
    assert output.offset.shape == (6, 2)
    assert output.cholesky.shape == (6, 2, 2)
    assert torch.all(torch.diagonal(output.cholesky, dim1=-2, dim2=-1) > 0)
    assert torch.all(torch.linalg.eigvalsh(output.covariance) > 0)
    assert torch.allclose(output.offset, torch.zeros_like(output.offset))

    target = torch.randn(6, 2).clamp(-2.0, 2.0)
    loss = gaussian_measurement_nll(target, output)
    loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(
        output.inlier_logits, torch.ones(6)
    )
    loss.backward()
    gradients = [parameter.grad for parameter in head.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)


def test_measurement_nll_is_lower_at_target_mean():
    from localization_training.pair_measurement import (
        PairMeasurementHead,
        PairMeasurementOutput,
        gaussian_measurement_nll,
    )

    head = PairMeasurementHead(descriptor_dim=2, patch_radius=0, hidden_dim=4)
    base = head(
        torch.zeros(1, 6),
        torch.zeros(1, 1),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    target = torch.tensor([[0.5, -0.25]])
    at_target = PairMeasurementOutput(base.inlier_logits, target, base.cholesky)
    away = PairMeasurementOutput(base.inlier_logits, torch.zeros_like(target), base.cholesky)

    assert gaussian_measurement_nll(target, at_target) < gaussian_measurement_nll(
        target, away
    )


def test_pair_measurement_set_context_is_query_conditioned():
    from localization_training.pair_measurement import PairMeasurementHead

    torch.manual_seed(4)
    head = PairMeasurementHead(
        descriptor_dim=2,
        patch_radius=0,
        hidden_dim=4,
        use_set_context=True,
    )
    with torch.no_grad():
        head.set_network[0].weight.fill_(0.1)
        head.set_network[0].bias.zero_()
        head.set_network[-1].weight.zero_()
        head.set_network[-1].bias.zero_()
        head.set_network[-1].weight[0, 0] = 1.0
    pair = torch.randn(2, 6)
    pair[1] = pair[1] + 5.0
    patch = torch.randn(2, 1)
    query = torch.randn(2, 2)
    landmark = torch.randn(2, 2)
    together = head(pair, patch, query, landmark).inlier_logits[0]
    alone = head(pair[:1], patch[:1], query[:1], landmark[:1]).inlier_logits[0]

    assert not torch.allclose(together, alone)


def test_pair_measurement_geometry_context_changes_with_set_geometry():
    from localization_training.pair_measurement import PairMeasurementHead

    torch.manual_seed(5)
    head = PairMeasurementHead(
        descriptor_dim=2,
        patch_radius=0,
        hidden_dim=4,
        use_set_context=True,
        use_geometry_context=True,
    )
    with torch.no_grad():
        head.geometry_token_network[0].weight.fill_(0.1)
        head.geometry_token_network[0].bias.zero_()
        head.geometry_set_network[-1].weight.zero_()
        head.geometry_set_network[-1].bias.zero_()
        head.geometry_set_network[-1].weight[0, 0] = 1.0
    pair = torch.randn(2, 6)
    patch = torch.randn(2, 1)
    query = torch.randn(2, 2)
    landmark = torch.randn(2, 2)
    geometry = torch.zeros(2, 5)
    changed_geometry = geometry.clone()
    changed_geometry[1] = 3.0

    base = head(pair, patch, query, landmark, geometry).inlier_logits[0]
    changed = head(
        pair, patch, query, landmark, changed_geometry
    ).inlier_logits[0]

    assert not torch.allclose(base, changed)
