import pytest
import torch

from scripts.extend_lafgs_metric_state import extend_metric_state


def test_extended_metric_state_preserves_query_metric_and_binds_new_registry():
    metric = {
        "landmark_indices": torch.tensor([0, 1]),
        "metric_state_dict": {"x": torch.tensor([3.0])},
    }
    output = extend_metric_state(
        metric, {"anchor_ids": torch.tensor([0, 1, 2])}, "map.pt"
    )
    assert output["landmark_indices"].tolist() == [0, 1, 2]
    assert output["extended_anchor_count"] == 1
    torch.testing.assert_close(
        output["metric_state_dict"]["x"], metric["metric_state_dict"]["x"]
    )


def test_extended_metric_state_rejects_reordered_prefix():
    with pytest.raises(ValueError, match="prefix"):
        extend_metric_state(
            {"landmark_indices": torch.tensor([0, 1])},
            {"anchor_ids": torch.tensor([1, 0, 2])},
            "map.pt",
        )
