import torch

from map_learning.v18_provenance_truth import (
    TRUTH_EQUIVALENT,
    TRUTH_INVALID,
    TRUTH_NONE,
    TRUTH_UNIQUE,
)
from scripts.materialize_v18_certified_projection_truth import (
    _truth_from_positive_mask,
)


def test_certified_topl_fallback_preserves_set_valued_geometry() -> None:
    truth = _truth_from_positive_mask(
        candidates=torch.tensor([[3, 5], [7, 9], [11, 13], [15, 17]]),
        positive=torch.tensor(
            [[True, False], [True, True], [False, False], [True, False]]
        ),
        valid_rows=torch.tensor([True, True, True, False]),
    )
    assert truth["truth_status"].tolist() == [
        TRUTH_UNIQUE,
        TRUTH_EQUIVALENT,
        TRUTH_NONE,
        TRUTH_INVALID,
    ]
    assert truth["truth_offsets"].tolist() == [0, 1, 3, 3, 3]
    assert truth["truth_anchor_rows"].tolist() == [3, 7, 9]
    assert truth["uses_topl_candidates"] is True
    assert truth["fallback_only"] is True
