import pytest
import torch

from localization_training.confusion_directed_metric import (
    candidate_margin_loss,
    protected_top1_mask,
    select_stratified_rows,
    topk_distribution_distillation,
)
from scripts.train_lafgs_confusion_directed_metric import (
    _csr_positive_mask,
    _hierarchical_sampling_index,
    _sample_hierarchical,
)


def test_candidate_margin_loss_rewards_legal_candidate():
    mask = torch.tensor([[False, True, False]])
    good = candidate_margin_loss(
        torch.tensor([[0.1, 0.8, 0.2]]),
        mask,
        margin=0.02,
        temperature=0.05,
    )
    bad = candidate_margin_loss(
        torch.tensor([[0.8, 0.1, 0.2]]),
        mask,
        margin=0.02,
        temperature=0.05,
    )
    assert good < bad


def test_candidate_margin_loss_rejects_missing_class():
    with pytest.raises(ValueError, match="positive"):
        candidate_margin_loss(
            torch.zeros(1, 3),
            torch.zeros(1, 3, dtype=torch.bool),
            margin=0.02,
            temperature=0.05,
        )
    with pytest.raises(ValueError, match="negative"):
        candidate_margin_loss(
            torch.zeros(1, 3),
            torch.ones(1, 3, dtype=torch.bool),
            margin=0.02,
            temperature=0.05,
        )


def test_protected_top1_and_stratified_selection_are_well_formed():
    protected = protected_top1_mask(3, 4)
    assert protected[:, 0].all()
    assert protected[:, 1:].sum() == 0
    generator = torch.Generator().manual_seed(7)
    selected = select_stratified_rows(
        torch.tensor([True, False, True, True, False]),
        maximum=2,
        generator=generator,
    )
    assert selected.numel() == 2
    assert {int(value) for value in selected}.issubset({0, 2, 3})


def test_vectorized_csr_positive_mask():
    candidates = torch.tensor([[2, 7, 9], [1, 5, 8], [0, 3, 6]])
    offsets = torch.tensor([0, 2, 2, 3])
    positives = torch.tensor([7, 4, 6])
    assert torch.equal(
        _csr_positive_mask(candidates, offsets, positives),
        torch.tensor(
            [
                [False, True, False],
                [False, False, False],
                [False, False, True],
            ]
        ),
    )


def test_topk_distribution_distillation_is_zero_for_identical_scores():
    scores = torch.tensor([[0.8, 0.3, -0.1], [0.4, 0.2, 0.1]])
    loss = topk_distribution_distillation(
        scores, scores, temperature=0.05
    )
    assert float(loss) == pytest.approx(0.0, abs=1e-8)
    changed = topk_distribution_distillation(
        scores.flip(1), scores, temperature=0.05
    )
    assert changed > loss


def test_hierarchical_sampler_covers_eligible_trajectories():
    examples = {
        "kind": torch.tensor([2, 2, 2, 2]),
        "trajectory": torch.tensor([0, 0, 1, 1]),
        "query_index": torch.tensor([0, 1, 2, 3]),
    }
    index = _hierarchical_sampling_index(examples, 2)
    sampled = _sample_hierarchical(
        index, 100, torch.Generator().manual_seed(3)
    )
    assert sampled.numel() == 100
    assert set(examples["trajectory"][sampled].tolist()) == {0, 1}
