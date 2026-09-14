from __future__ import annotations

import numpy as np
import pytest
import torch

from localization.localizer import _mapping_quality_polish_rows


def test_mapping_quality_polish_keeps_best_owner_rows_stably() -> None:
    selected = _mapping_quality_polish_rows(
        np.arange(8),
        match_anchor_rows=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
        anchor_quality=torch.tensor([0.1, 0.8, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]),
        retention_fraction=0.5,
        minimum_count=4,
    )
    assert selected.tolist() == [1, 3, 5, 7]


def test_mapping_quality_polish_rejects_duplicate_rows() -> None:
    with pytest.raises(ValueError):
        _mapping_quality_polish_rows(
            np.array([0, 0, 1, 2]),
            match_anchor_rows=torch.arange(4),
            anchor_quality=torch.ones(4),
            retention_fraction=0.5,
            minimum_count=4,
        )
