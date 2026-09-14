from __future__ import annotations

import torch

from map_learning.v18_provenance_truth import TRUTH_NONE, TRUTH_UNIQUE
from scripts.evaluate_v19_track_extension_teacher import (
    _calibration_safety,
    _wilson_lower_bound,
)


def _truth(status: list[int], predictions: list[int]) -> dict:
    offsets = [0]
    cursor = 0
    for value in status:
        cursor += int(value == TRUTH_UNIQUE)
        offsets.append(cursor)
    return {
        "truth_status": torch.tensor(status),
        "truth_offsets": torch.tensor(offsets),
        "truth_anchor_rows": torch.tensor(predictions),
    }


def test_wilson_bound_does_not_treat_one_success_as_certainty() -> None:
    assert _wilson_lower_bound(1, 1) < 0.5
    assert _wilson_lower_bound(300, 300) > 0.99


def test_destructive_authorization_requires_multiple_active_families() -> None:
    status = [TRUTH_UNIQUE] * 300 + [TRUTH_NONE] * 300
    report = _calibration_safety(
        truth=_truth(status, list(range(300))),
        ground_truth_anchor=torch.arange(600),
        equivalence=torch.arange(600),
        rows=torch.arange(600),
        row_families=torch.tensor([0] * 300 + [1] * 300),
        target_precision=0.99,
        minimum_decisive_assignments=300,
        minimum_active_families=2,
    )
    assert report["decisive_precision"] == 1.0
    assert report["active_family_count"] == 1
    assert report["authorized"] is False

