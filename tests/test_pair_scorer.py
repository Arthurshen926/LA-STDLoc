import torch


def test_sparse_pair_scorer_returns_one_logit_per_pair_and_backpropagates():
    from localization_training.pair_scorer import SparsePairScorer

    scorer = SparsePairScorer(input_dim=6, hidden_dim=8)
    features = torch.randn(5, 6, requires_grad=True)
    logits = scorer(features)
    logits.sum().backward()

    assert logits.shape == (5,)
    assert features.grad is not None
    assert scorer.export_config() == {
        "input_dim": 6,
        "hidden_dim": 8,
        "architecture": "cosine_residual_v1",
        "descriptor_dim": 0,
    }


def test_sparse_pair_scorer_initially_preserves_cosine_ranking():
    from localization_training.pair_scorer import SparsePairScorer

    scorer = SparsePairScorer(cosine_bias=0.65, cosine_scale=10.0)
    features = torch.randn(4, 6)
    features[:, 0] = torch.tensor([0.5, 0.8, 0.6, 0.7])
    logits = scorer(features)

    assert torch.equal(torch.argsort(logits), torch.argsort(features[:, 0]))
    assert torch.allclose(logits, 10.0 * (features[:, 0] - 0.65), atol=1e-6)


def test_descriptor_set_pair_scorer_uses_inference_available_pair_context():
    from localization_training.pair_scorer import SparsePairScorer

    scorer = SparsePairScorer(
        input_dim=6,
        hidden_dim=8,
        architecture="descriptor_set_residual_v2",
        descriptor_dim=4,
    )
    pair_features = torch.randn(5, 6, requires_grad=True)
    query = torch.randn(5, 4, requires_grad=True)
    landmark = torch.randn(5, 4, requires_grad=True)
    global_query = torch.randn(4)
    logits = scorer(pair_features, query, landmark, global_query)
    expected = 10.0 * (pair_features[:, 0] - 0.65)
    logits.sum().backward()

    assert logits.shape == (5,)
    assert torch.allclose(logits.detach(), expected.detach(), atol=1e-6)
    assert query.grad is not None
    assert landmark.grad is not None
    assert scorer.export_config()["architecture"] == "descriptor_set_residual_v2"
    assert scorer.export_config()["descriptor_dim"] == 4
