import torch

from localization_training.positive_retrieval import (
    csr_candidate_positive_mask,
    strict_positive_rank_buckets,
)


def test_csr_candidate_positive_mask_and_rank_buckets():
    top16 = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    )
    top64 = torch.tensor(
        [[1, 2, 11], [3, 4, 12], [5, 6, 13], [7, 8, 14], [9, 10, 15]]
    )
    active_offsets = torch.tensor([0, 1, 2, 3, 3, 3])
    active_indices = torch.tensor([1, 12, 99])
    canonical_offsets = torch.tensor([0, 1, 2, 3, 4, 4])
    top16_mask = csr_candidate_positive_mask(
        top16, active_offsets, active_indices
    )
    top64_mask = csr_candidate_positive_mask(
        top64, active_offsets, active_indices
    )
    buckets = strict_positive_rank_buckets(
        active_offsets=active_offsets,
        canonical_offsets=canonical_offsets,
        top16_positive_mask=top16_mask,
        top64_positive_mask=top64_mask,
    )
    assert buckets.tolist() == [0, 1, 2, 3, 4]
