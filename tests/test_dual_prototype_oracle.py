import torch

from scripts.build_dual_prototype_oracle import csr_candidate_membership


def test_csr_membership_accepts_every_legal_positive():
    offsets = torch.tensor([0, 2, 3, 5])
    positives = torch.tensor([4, 7, 2, 1, 9])
    candidates = torch.tensor(
        [[7, 3], [2, 4], [9, 1]]
    )
    result = csr_candidate_membership(
        candidates,
        torch.tensor([0, 1, 2]),
        offsets,
        positives,
        landmark_count=10,
    )
    assert result.tolist() == [
        [True, False],
        [True, False],
        [True, True],
    ]


def test_csr_membership_supports_sampled_noncontiguous_rows():
    offsets = torch.tensor([0, 1, 2, 4])
    positives = torch.tensor([3, 4, 5, 6])
    result = csr_candidate_membership(
        torch.tensor([6, 3]),
        torch.tensor([2, 0]),
        offsets,
        positives,
        landmark_count=8,
    )
    assert result.tolist() == [True, True]
