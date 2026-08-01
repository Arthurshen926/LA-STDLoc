import pytest
import torch

from scripts.merge_lafgs_real_synthetic_training_bundle import (
    _validate_critical_teacher,
)


def test_critical_teacher_rejects_dynamic_outcome_payload() -> None:
    payload = {
        "query_names": ["query0"],
        "records": [
            {
                "query_rows": torch.tensor([0, 1]),
                "top1_anchor_indices": torch.tensor([2, 3]),
            }
        ],
    }

    with pytest.raises(ValueError, match="dynamic replay/outcome"):
        _validate_critical_teacher(payload, ["query0"])


def test_critical_teacher_accepts_pair_aligned_payload() -> None:
    payload = {
        "query_names": ["query0"],
        "records": [
            {
                "query_rows": torch.tensor([0, 1]),
                "positive_weights": torch.ones(3),
                "row_weights": torch.ones(2),
            }
        ],
    }

    _validate_critical_teacher(payload, ["query0"])
