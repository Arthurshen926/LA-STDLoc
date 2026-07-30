import torch
import torch.nn.functional as F

from scripts.train_lafgs_candidate_context_pair_scorer import (
    _best_observed_context,
)


def test_observed_context_excludes_query_trajectory_and_falls_back():
    query = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1)
    prototypes = torch.zeros(2, 2, 2, 2)
    counts = torch.zeros(2, 2, 2, dtype=torch.long)
    prototypes[0, 0, 0] = torch.tensor([1.0, 0.0])
    counts[0, 0, 0] = 10
    prototypes[1, 1, 0] = torch.tensor([0.8, 0.2])
    counts[1, 1, 0] = 3
    fallback = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), dim=1)

    selected, similarity, available, observations = _best_observed_context(
        query,
        torch.tensor([0, 1]),
        heldout_trajectory_index=0,
        prototype_context=F.normalize(prototypes, dim=-1),
        observation_counts=counts,
        fallback_context=fallback,
    )

    assert available.tolist() == [True, False]
    assert observations.tolist() == [3, 0]
    assert torch.allclose(selected[0], F.normalize(torch.tensor([0.8, 0.2]), dim=0))
    assert torch.allclose(selected[1], fallback[1])
    assert torch.isfinite(similarity).all()
