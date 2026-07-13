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


def test_absolute_set_risk_uses_probability_ess():
    from localization_training.absolute_set_risk import absolute_pose_set_risk

    jacobian = torch.randn(4, 2, 6, dtype=torch.float64)
    residual = torch.zeros(4, 2, dtype=torch.float64)
    cholesky = torch.eye(2, dtype=torch.float64)[None].repeat(4, 1, 1)
    probability = torch.tensor([0.9, 0.7, 0.2, 0.1], dtype=torch.float64)
    logits = torch.logit(probability)
    risk = absolute_pose_set_risk(jacobian, residual, cholesky, logits)
    expected = probability.sum().square() / probability.square().sum()
    assert torch.allclose(risk.effective_pair_count, expected)


def test_task_scaled_condition_is_invariant_to_translation_units():
    from localization_training.absolute_set_risk import absolute_pose_set_risk

    generator = torch.Generator().manual_seed(31)
    jacobian_m = torch.randn(24, 2, 6, generator=generator, dtype=torch.float64)
    residual = torch.randn(24, 2, generator=generator, dtype=torch.float64) * 0.1
    cholesky = torch.eye(2, dtype=torch.float64)[None].repeat(24, 1, 1)
    logits = torch.zeros(24, dtype=torch.float64)
    risk_m = absolute_pose_set_risk(
        jacobian_m,
        residual,
        cholesky,
        logits,
        translation_prior_sigma_m=1.0,
        translation_task_scale_m=0.02,
    )
    jacobian_cm = jacobian_m.clone()
    jacobian_cm[:, :, :3] *= 0.01
    risk_cm = absolute_pose_set_risk(
        jacobian_cm,
        residual,
        cholesky,
        logits,
        translation_prior_sigma_m=100.0,
        translation_task_scale_m=2.0,
        reference_translation_m=2.0,
    )
    assert torch.allclose(
        risk_m.condition_number,
        risk_cm.condition_number,
        atol=1e-10,
        rtol=1e-10,
    )
