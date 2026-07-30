import torch

from scripts.train_lafgs_dual_context_encoder import (
    _sample_slots_by_category,
    _selected_training_batch,
)


def test_training_candidates_include_topk_and_directed_confusion_pair():
    teacher = {
        "query_rows": torch.tensor([4, 9]),
        "positive_offsets": torch.tensor([0, 1, 2]),
        "positive_indices": torch.tensor([1, 5]),
    }
    dynamic = {
        "query_rows": torch.tensor([4, 9]),
        "top1_anchor_indices": torch.tensor([2, 5]),
    }
    topk = {
        "query_rows": torch.tensor([9, 4]),
        "topk_anchor_indices": torch.tensor(
            [[5, 7, 8], [2, 3, 4]]
        ),
    }
    confusion = {
        4: [
            {
                "correct_anchor": 1,
                "confusing_anchor": 6,
                "pose_blame": 3.0,
            }
        ]
    }
    (
        rows,
        positive_mask,
        candidates,
        clean_top1,
        top1_positions,
        directed_rows,
        directed_correct,
        directed_confusing,
        directed_weights,
    ) = _selected_training_batch(
        teacher,
        dynamic,
        maximum_rows=0,
        generator=torch.Generator().manual_seed(1),
        anchor_count=10,
        topk=topk,
        confusion_events=confusion,
    )

    assert rows.tolist() == [4, 9]
    assert candidates.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert clean_top1.tolist() == [False, True]
    assert candidates[top1_positions].tolist() == [2, 5]
    assert positive_mask[0, candidates.tolist().index(1)]
    assert positive_mask[1, candidates.tolist().index(5)]
    assert directed_rows.tolist() == [0]
    assert candidates[directed_correct].tolist() == [1]
    assert candidates[directed_confusing].tolist() == [6]
    assert torch.allclose(directed_weights, torch.ones(1))


def test_training_candidates_reject_missing_topk_rows():
    teacher = {
        "query_rows": torch.tensor([4]),
        "positive_offsets": torch.tensor([0, 1]),
        "positive_indices": torch.tensor([1]),
    }
    dynamic = {
        "query_rows": torch.tensor([4]),
        "top1_anchor_indices": torch.tensor([2]),
    }
    topk = {
        "query_rows": torch.tensor([9]),
        "topk_anchor_indices": torch.tensor([[2, 3]]),
    }
    try:
        _selected_training_batch(
            teacher,
            dynamic,
            maximum_rows=0,
            generator=torch.Generator().manual_seed(1),
            anchor_count=10,
            topk=topk,
        )
    except ValueError as error:
        assert "miss teacher rows" in str(error)
    else:
        raise AssertionError("missing top-K rows must be rejected")


def test_stratified_sampling_preserves_confusion_quota():
    valid_slots = torch.arange(20)
    categories = (
        torch.tensor([True] * 8 + [False] * 12),
        torch.tensor([False] * 8 + [True] * 4 + [False] * 8),
        torch.tensor([False] * 12 + [True] * 4 + [False] * 4),
        torch.tensor([False] * 16 + [True] * 4),
    )
    selected = _sample_slots_by_category(
        categories,
        valid_slots,
        maximum_rows=10,
        generator=torch.Generator().manual_seed(4),
    )
    assert selected.numel() == 10
    assert int((selected < 8).sum()) == 4
