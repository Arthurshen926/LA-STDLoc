import torch

from localization_training.map_information_bias import (
    counterfactual_pose_swap_utility,
    directional_candidate_set_risk,
    map_information_and_bias_risk,
)


def _inputs(count=12):
    generator = torch.Generator().manual_seed(7)
    jacobian = torch.randn(count, 2, 6, generator=generator)
    jacobian[:, :, :3] *= 20.0
    residual = torch.zeros(count, 2)
    logits = torch.zeros(count, requires_grad=True)
    labels = torch.tensor(([1.0, 0.0] * ((count + 1) // 2))[:count])
    return jacobian, residual, logits, labels


def test_map_information_bias_risk_is_finite_and_backpropagates():
    jacobian, residual, logits, labels = _inputs()
    risk = map_information_and_bias_risk(
        jacobian,
        residual,
        logits,
        labels,
    )
    loss = (
        risk.cleanliness_loss
        + 0.1 * risk.full_information_loss
        + 0.1 * risk.translation_information_loss
        + risk.capacity_loss
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert risk.target_match_count.item() == labels.sum().item()


def test_signed_reprojection_bias_increases_translation_bias():
    jacobian, residual, logits, labels = _inputs()
    clean = map_information_and_bias_risk(jacobian, residual, logits, labels)
    biased_residual = jacobian[:, :, 0] * 0.01
    biased = map_information_and_bias_risk(
        jacobian,
        biased_residual,
        logits,
        labels,
    )
    assert clean.translation_bias_m.item() < 1e-8
    assert biased.translation_bias_m.item() > clean.translation_bias_m.item()
    assert biased.bias_loss.item() > clean.bias_loss.item()


def test_cleanliness_gradient_separates_correct_and_false_candidates():
    jacobian, residual, logits, labels = _inputs()
    risk = map_information_and_bias_risk(jacobian, residual, logits, labels)
    risk.cleanliness_loss.backward()
    assert (logits.grad[labels > 0.5] < 0).all()
    assert (logits.grad[labels < 0.5] > 0).all()


def test_invalid_geometry_candidates_still_receive_cleanliness_supervision():
    jacobian, residual, logits, labels = _inputs()
    geometry_valid = torch.zeros(labels.shape[0], dtype=torch.bool)
    risk = map_information_and_bias_risk(
        jacobian,
        residual,
        logits,
        labels,
        valid_mask=geometry_valid,
    )
    loss = risk.cleanliness_loss + risk.capacity_loss
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0
    assert risk.full_logdet_gain.item() == 0.0
    assert risk.target_match_count.item() == labels.sum().item()


def test_capacity_is_bounded_when_query_has_no_clean_candidates():
    jacobian, residual, logits, labels = _inputs()
    labels.zero_()
    logits = torch.full_like(logits, 10.0, requires_grad=True)
    risk = map_information_and_bias_risk(jacobian, residual, logits, labels)
    assert 0.0 <= risk.capacity_loss.item() <= 1.0


def test_large_gt_residuals_are_removed_from_soft_inlier_pose_risk():
    jacobian, residual, logits, labels = _inputs()
    residual[:] = 1000.0
    risk = map_information_and_bias_risk(
        jacobian,
        residual,
        logits,
        labels,
        inlier_sigma_px=4.0,
    )
    assert risk.soft_inlier_expected_match_count.item() < 1e-8
    assert risk.translation_bias_m.item() < 1e-8


def _directional_inputs(rows=24):
    generator = torch.Generator().manual_seed(31)
    base_jacobian = torch.randn(rows, 2, 6, generator=generator)
    base_jacobian[:, :, :3] *= 12.0
    jacobian = base_jacobian[:, None].expand(-1, 2, -1, -1).clone()
    residual = torch.stack(
        [base_jacobian[:, :, 0] * 0.04] * 2,
        dim=1,
    )
    similarity = torch.zeros(rows, 2, requires_grad=True)
    valid = torch.ones(rows, 2, dtype=torch.bool)
    labels = torch.zeros(rows, 2, dtype=torch.bool)
    labels[:, 0] = True
    return jacobian, residual, similarity, valid, labels


def test_directional_candidate_set_risk_detects_coherent_signed_bias():
    jacobian, residual, similarity, valid, labels = _directional_inputs()
    coherent = directional_candidate_set_risk(
        jacobian,
        residual,
        similarity,
        valid,
        labels,
        dustbin_score=-1.0,
        temperature=0.2,
    )
    balanced_residual = residual.clone()
    balanced_residual[:, 1] *= -1.0
    balanced = directional_candidate_set_risk(
        jacobian,
        balanced_residual,
        similarity,
        valid,
        labels,
        dustbin_score=-1.0,
        temperature=0.2,
    )

    assert coherent.loss.item() > 0.0
    assert coherent.loss.item() > balanced.loss.item() + 1e-4
    assert coherent.translation_bias_m.item() > balanced.translation_bias_m.item()
    assert coherent.target_budget.item() == labels.any(dim=1).sum().item()


def test_directional_candidate_set_gradient_rebalances_opposite_directions():
    jacobian, residual, similarity, valid, labels = _directional_inputs()
    residual[:, 1] *= -1.0
    with torch.no_grad():
        similarity[:, 0] = 0.8
        similarity[:, 1] = 0.0
    risk = directional_candidate_set_risk(
        jacobian,
        residual,
        similarity,
        valid,
        labels,
        dustbin_score=-1.0,
        temperature=0.2,
    )
    risk.loss.backward()

    assert torch.isfinite(risk.loss)
    assert torch.isfinite(similarity.grad).all()
    assert similarity.grad[:, 0].mean().item() > 0.0
    assert similarity.grad[:, 1].mean().item() < 0.0


def test_directional_candidate_set_fixed_budget_prevents_all_dustbin_nan():
    jacobian, residual, similarity, valid, labels = _directional_inputs()
    similarity = torch.full_like(similarity, -20.0, requires_grad=True)
    risk = directional_candidate_set_risk(
        jacobian,
        residual,
        similarity,
        valid,
        labels,
        dustbin_score=0.0,
    )
    risk.loss.backward()

    assert torch.isfinite(risk.loss)
    assert torch.isfinite(similarity.grad).all()
    assert risk.target_budget.item() > 0.0


def test_counterfactual_swap_prioritizes_replacing_signed_bias():
    generator = torch.Generator().manual_seed(19)
    jacobian = torch.randn(16, 2, 6, generator=generator)
    jacobian[:, :, :3] *= 10.0
    residual = jacobian[:, :, 0] * 0.02
    add_jacobian = jacobian[:2].clone()
    add_residual = torch.stack(
        [torch.zeros_like(residual[0]), residual[1]], dim=0
    )

    utility = counterfactual_pose_swap_utility(
        jacobian,
        residual,
        torch.ones(16, dtype=torch.bool),
        torch.tensor([0, 1]),
        add_jacobian,
        add_residual,
        torch.tensor([True, True]),
        utility_floor=0.1,
    )

    assert utility.valid_mask.tolist() == [True, True]
    assert utility.bias_reduction_task2[0] > utility.bias_reduction_task2[1]
    assert utility.weights[0] > utility.weights[1]
    assert torch.allclose(utility.weights[1], torch.tensor(0.1), atol=1e-5)
    assert utility.current_translation_bias_m.item() > 0.0


def test_counterfactual_swap_accounts_for_quota_displacement():
    generator = torch.Generator().manual_seed(23)
    jacobian = torch.randn(12, 2, 6, generator=generator)
    residual = torch.randn(12, 2, generator=generator) * 0.5
    add_jacobian = jacobian[0:1].clone()
    add_residual = torch.zeros(1, 2)

    without_displacement = counterfactual_pose_swap_utility(
        jacobian,
        residual,
        torch.ones(12, dtype=torch.bool),
        torch.tensor([0]),
        add_jacobian,
        add_residual,
        torch.tensor([True]),
    )
    with_displacement = counterfactual_pose_swap_utility(
        jacobian,
        residual,
        torch.ones(12, dtype=torch.bool),
        torch.tensor([0]),
        add_jacobian,
        add_residual,
        torch.tensor([True]),
        displaced_indices=torch.tensor([1]),
    )

    assert not torch.allclose(
        without_displacement.counterfactual_translation_bias_task,
        with_displacement.counterfactual_translation_bias_task,
    )
    assert not torch.allclose(
        without_displacement.translation_logdet_gain,
        with_displacement.translation_logdet_gain,
    )


def test_counterfactual_swap_can_gate_to_joint_bias_and_information_gain():
    generator = torch.Generator().manual_seed(29)
    jacobian = torch.randn(20, 2, 6, generator=generator)
    jacobian[:, :, :3] *= 5.0
    residual = torch.randn(20, 2, generator=generator)
    add_jacobian = torch.randn(4, 2, 6, generator=generator)
    add_jacobian[:, :, :3] *= 5.0
    add_residual = torch.randn(4, 2, generator=generator)
    swap_mask = torch.ones(4, dtype=torch.bool)

    ungated = counterfactual_pose_swap_utility(
        jacobian,
        residual,
        torch.ones(20, dtype=torch.bool),
        torch.tensor([0, 1, 2, 3]),
        add_jacobian,
        add_residual,
        swap_mask,
        utility_floor=0.0,
    )
    gated = counterfactual_pose_swap_utility(
        jacobian,
        residual,
        torch.ones(20, dtype=torch.bool),
        torch.tensor([0, 1, 2, 3]),
        add_jacobian,
        add_residual,
        swap_mask,
        utility_floor=0.0,
        require_positive_bias_gain=True,
        require_nonnegative_translation_gain=True,
    )

    expected = (ungated.bias_reduction_task2 > 0) & (
        ungated.translation_logdet_gain >= 0
    )
    assert torch.equal(gated.valid_mask, expected)
    assert torch.equal(gated.weights[~expected], torch.zeros_like(gated.weights[~expected]))
