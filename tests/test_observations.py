from __future__ import annotations

import pytest
import torch

from map_learning.observations import _sample_surface


def _cache(*, with_alpha: bool) -> dict:
    cache = {
        "native_keypoints": torch.tensor([[0.2, 0.8], [1.4, 1.2]]),
        "native_input_hw": [2, 2],
        "native_depth": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }
    if with_alpha:
        cache["native_alpha"] = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    return cache


def test_surface_samples_prefer_same_render_provenance() -> None:
    keypoints, depth, alpha, source = _sample_surface(
        _cache(with_alpha=False),
        torch.tensor([0, 1]),
        {
            "rendered_depth": torch.tensor([5.0, 6.0]),
            "rendered_alpha": torch.tensor([0.8, 0.9]),
        },
    )
    assert keypoints.shape == (2, 2)
    assert torch.equal(depth, torch.tensor([5.0, 6.0]))
    assert torch.allclose(alpha, torch.tensor([0.8, 0.9]))
    assert source == "raster_provenance"


def test_legacy_surface_samples_require_alpha() -> None:
    with pytest.raises(KeyError, match="rebuild raster provenance"):
        _sample_surface(_cache(with_alpha=False), torch.tensor([0]))


def test_legacy_surface_samples_remain_supported_with_alpha() -> None:
    _, depth, alpha, source = _sample_surface(
        _cache(with_alpha=True), torch.tensor([0, 1])
    )
    assert torch.equal(depth, torch.tensor([1.0, 4.0]))
    assert torch.allclose(alpha, torch.tensor([0.1, 0.4]))
    assert source == "query_cache"
