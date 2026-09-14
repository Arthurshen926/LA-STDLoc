from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from evaluation.bootstrap import materialize_a0
from map_learning.metric import SharedLowRankMetric


def _stage_state(path: Path, *, duplicate_ids: bool = False) -> None:
    ids = torch.tensor([7, 9, 11, 13])
    if duplicate_ids:
        ids[-1] = ids[-2]
    torch.save(
        {
            "landmark_indices": ids,
            "landmark_xyz": torch.arange(12).reshape(4, 3).float(),
            "landmark_features": torch.randn(4, 8),
        },
        path,
    )


def test_materialize_a0_writes_exact_identity_metric(tmp_path: Path):
    state_path = tmp_path / "stage.pt"
    _stage_state(state_path)
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()
    map_path, metric_path = materialize_a0(
        state_path,
        tmp_path / "a0",
        Path(__file__).parents[1] / "configs/paper_mainline.yaml",
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)

    anchor_map = torch.load(map_path, map_location="cpu", weights_only=False)
    metric_state = torch.load(metric_path, map_location="cpu", weights_only=False)
    metric = SharedLowRankMetric(**metric_state["metric_config"])
    metric.load_state_dict(metric_state["metric_state_dict"])
    transformed, residual = metric(anchor_map["anchor_features"])
    assert torch.equal(residual, torch.zeros_like(residual))
    assert torch.allclose(
        transformed, F.normalize(anchor_map["anchor_features"], dim=-1)
    )
    assert metric_state["landmark_indices"].tolist() == [7, 9, 11, 13]
    assert metric_state["variant"] == "A0_bootstrap_identity_metric"


def test_materialize_a0_rejects_duplicate_landmark_ids(tmp_path: Path):
    state_path = tmp_path / "stage.pt"
    _stage_state(state_path, duplicate_ids=True)
    with pytest.raises(ValueError, match="unique"):
        materialize_a0(
            state_path,
            tmp_path / "a0",
            Path(__file__).parents[1] / "configs/paper_mainline.yaml",
        )
