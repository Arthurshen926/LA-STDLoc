from __future__ import annotations

import torch

from map_learning.v18_provenance_truth import (
    TRUTH_NONE,
    TRUTH_UNIQUE,
)
from map_learning.v19_full_pool_sufficiency import (
    audit_full_pool_sufficiency_rows,
)


def _truth() -> dict:
    return {
        "row_count": 4,
        "truth_status": torch.tensor(
            [TRUTH_UNIQUE, TRUTH_UNIQUE, TRUTH_UNIQUE, TRUTH_NONE]
        ),
        "truth_offsets": torch.tensor([0, 1, 2, 3, 3]),
        "truth_anchor_rows": torch.tensor([0, 1, 2]),
    }


def test_full_pool_audit_separates_selection_retrieval_and_competition() -> None:
    report = audit_full_pool_sufficiency_rows(
        truth=_truth(),
        projection_candidate_graph={
            "candidate_offsets": torch.tensor([0, 1, 2, 3, 3])
        },
        retrieved_anchor_rows=torch.tensor(
            [[3, 0], [3, 1], [3, 4], [3, 4]]
        ),
        active_anchor_mask=torch.tensor([True, True, False, True, True]),
        equivalence_class_ids=torch.arange(5),
        candidate_pool_deficit_authorized=False,
    )
    assert report["counts"]["competition_miss"] == 2
    assert report["counts"]["selection_deficit"] == 1
    assert report["counts"]["retrieval_miss"] == 0
    assert report["counts"]["certified_candidate_pool_deficit"] == 0
    assert report["counts"]["unresolved_empty_projection"] == 1


def test_empty_projection_is_not_automatically_an_anchor_deficit() -> None:
    common = {
        "truth": _truth(),
        "projection_candidate_graph": {
            "candidate_offsets": torch.tensor([0, 1, 2, 3, 3])
        },
        "retrieved_anchor_rows": torch.tensor(
            [[0], [1], [2], [3]]
        ),
        "active_anchor_mask": torch.ones(5, dtype=torch.bool),
        "equivalence_class_ids": torch.arange(5),
    }
    unsafe = audit_full_pool_sufficiency_rows(
        **common, candidate_pool_deficit_authorized=False
    )
    certified = audit_full_pool_sufficiency_rows(
        **common, candidate_pool_deficit_authorized=True
    )
    assert unsafe["counts"]["certified_candidate_pool_deficit"] == 0
    assert unsafe["counts"]["unresolved_empty_projection"] == 1
    assert certified["counts"]["certified_candidate_pool_deficit"] == 1

