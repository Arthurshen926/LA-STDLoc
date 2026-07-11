import torch


def test_absolute_set_risk_preserves_physical_information_scale():
    from localization_training.absolute_set_risk import absolute_pose_set_risk

    torch.manual_seed(2)
    jacobian = torch.randn(32, 2, 6, dtype=torch.float64)
    residual = torch.full((32, 2), 0.2, dtype=torch.float64)
    cholesky = torch.eye(2, dtype=torch.float64)[None].repeat(32, 1, 1)
    logits = torch.full((32,), 2.0, dtype=torch.float64)

    base = absolute_pose_set_risk(jacobian, residual, cholesky, logits)
    more = absolute_pose_set_risk(
        torch.cat([jacobian, jacobian], dim=0),
        torch.cat([residual, residual], dim=0),
        torch.cat([cholesky, cholesky], dim=0),
        torch.cat([logits, logits], dim=0),
    )

    assert more.translation_covariance_trace_m2 < base.translation_covariance_trace_m2
    assert more.effective_pair_count > base.effective_pair_count
    assert torch.isfinite(base.translation_bias_loss)
    assert torch.isfinite(base.condition_number)


def test_absolute_set_risk_backpropagates_to_measurement_outputs():
    from localization_training.absolute_set_risk import absolute_pose_set_risk

    jacobian = torch.randn(12, 2, 6)
    residual = (torch.randn(12, 2) * 0.1).requires_grad_()
    raw_diagonal = torch.zeros(12, 2, requires_grad=True)
    cholesky = torch.diag_embed(torch.nn.functional.softplus(raw_diagonal) + 0.1)
    logits = torch.zeros(12, requires_grad=True)
    risk = absolute_pose_set_risk(jacobian, residual, cholesky, logits)
    loss = risk.translation_bias_loss + risk.translation_trace_loss
    loss.backward()

    assert residual.grad is not None
    assert raw_diagonal.grad is not None
    assert logits.grad is not None
