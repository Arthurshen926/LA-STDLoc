import numpy as np
import torch

from localization_training.basin_distillation import (
    GOOD_SET,
    HARMFUL_SET,
    aggregate_edge_credit,
    expanded_positive_lookup,
    proposal_propensity,
)


def test_expand_positive_lookup_preserves_query_row_identity():
    record = {
        "query_rows": torch.tensor([3, 9]),
        "positive_offsets": torch.tensor([0, 2, 3]),
        "positive_indices": torch.tensor([4, 5, 8]),
    }
    assert expanded_positive_lookup(record) == {3: [4, 5], 9: [8]}


def test_proposal_propensity_is_category_weighted_uniform_triplet():
    assert np.isclose(proposal_propensity(5, 0.25), 0.025)
    assert proposal_propensity(2, 1.0) == 0.0


def test_edge_credit_is_query_local_and_inverse_propensity_capped():
    credit = aggregate_edge_credit(
        torch.tensor([[0, 1, 2], [0, 3, 4]]),
        torch.tensor([[5, 6, 7], [8, 9, 10]]),
        torch.tensor([GOOD_SET, HARMFUL_SET]),
        torch.tensor([True, False]),
        torch.tensor([0.1, 0.01]),
        torch.tensor([0.0, 2.0]),
        maximum_inverse_propensity=10.0,
    )
    assert credit["positive"]["rows"].tolist() == [0, 1, 2]
    assert credit["negative"]["rows"].tolist() == [0, 3, 4]
    assert torch.all(credit["negative"]["weights"] > 0)


def test_inverse_propensity_preserves_relative_weights_before_clipping():
    credit = aggregate_edge_credit(
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
        torch.tensor([[0, 1, 2], [3, 4, 5]]),
        torch.tensor([GOOD_SET, GOOD_SET]),
        torch.tensor([True, True]),
        torch.tensor([1e-9, 1e-7]),
        torch.zeros(2),
        maximum_inverse_propensity=100.0,
    )
    weights = credit["positive"]["weights"]
    assert float(weights[0]) > float(weights[-1])
