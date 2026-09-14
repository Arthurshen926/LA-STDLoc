from __future__ import annotations

import pytest
import torch

from evidence.observation_provider import (
    GaussianRenderObservationProvider,
    RealRGBObservationProvider,
)


def _record(seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    return {
        "native_keypoints": torch.rand((4, 2), generator=generator),
        "native_descriptors": torch.rand((4, 8), generator=generator).half(),
        "native_scores": torch.rand((4,), generator=generator),
        "native_K": torch.eye(3),
        "pose_w2c": torch.eye(4),
        "native_input_hw": [6, 9],
        "native_valid_mask": torch.ones((6, 9), dtype=torch.bool),
        "native_depth": torch.ones((6, 9)),
        "native_alpha": torch.full((6, 9), 0.75),
        "native_appearance_reliability": torch.linspace(0.5, 0.8, 4),
    }


def test_render_provider_reproduces_legacy_track_inputs_exactly() -> None:
    records = {
        "seq-01/frame-000001.color.png": _record(1),
        "seq-02/frame-000002.color.png": _record(2),
    }
    records["seq-01/frame-000001.color.png"]["sequence_id"] = "seq-01"
    records["seq-02/frame-000002.color.png"]["sequence_id"] = "seq-02"
    provider = GaussianRenderObservationProvider(
        {
            "schema": "lafgs_rendered_rgb_only_sparse_mapping_cache",
            "uses_source_mapping_rgb": False,
            "queries": records,
        },
        query_bins=torch.tensor([3, 7]),
    )
    inputs = provider.track_inputs()
    assert inputs["query_names"] == list(records)
    assert inputs["query_groups"] == ["seq-01", "seq-02"]
    for index, name in enumerate(records):
        record = records[name]
        assert torch.equal(
            inputs["descriptors"][index],
            torch.as_tensor(record["native_descriptors"]).float(),
        )
        assert torch.equal(
            inputs["keypoints"][index],
            torch.as_tensor(record["native_keypoints"]).float() + 0.5,
        )
        assert torch.equal(
            inputs["detector_scores"][index],
            torch.as_tensor(record["native_scores"]).float(),
        )
    view = provider.build_view(1)
    assert view.image_name == "seq-02/frame-000002.color.png"
    assert view.sequence_id == "seq-02"
    assert view.pose_bin == 7
    assert view.source_kind == "gaussian_render"
    assert view.depth is records[view.image_name]["native_depth"]
    assert view.alpha is records[view.image_name]["native_alpha"]
    assert view.surface_support is None


def test_real_provider_preserves_requested_query_order() -> None:
    records = {"b.png": _record(3), "a.png": _record(4)}
    provider = RealRGBObservationProvider(
        {"queries": records}, query_names=["a.png", "b.png"]
    )
    assert provider.names == ("a.png", "b.png")
    assert provider.track_inputs()["query_names"] == ["a.png", "b.png"]
    assert provider.build_view("a.png").source_kind == "real_rgb"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_keypoints", torch.zeros((4, 2, 1))),
        ("native_descriptors", torch.zeros((4, 8, 1))),
        ("native_scores", torch.zeros((4, 1))),
        ("native_K", torch.eye(4)),
        ("pose_w2c", torch.eye(3)),
        ("native_input_hw", [[6, 9]]),
        ("native_depth", torch.ones((1, 6, 9))),
        ("native_surface_support", torch.ones((4, 1))),
    ],
)
def test_provider_rejects_noncanonical_shapes(field: str, value) -> None:
    record = _record()
    record[field] = value
    with pytest.raises(ValueError):
        RealRGBObservationProvider({"queries": {"image.png": record}})


def test_provider_rejects_misaligned_sparse_rows() -> None:
    record = _record()
    record["native_scores"] = torch.ones(3)
    with pytest.raises(ValueError, match="not aligned"):
        RealRGBObservationProvider({"queries": {"image.png": record}})


def test_provider_fails_closed_on_source_kind() -> None:
    records = {"image.png": _record()}
    with pytest.raises(ValueError, match="source-image-free"):
        RealRGBObservationProvider(
            {"uses_source_mapping_rgb": False, "queries": records}
        )
    with pytest.raises(ValueError, match="uses_source_mapping_rgb=false"):
        GaussianRenderObservationProvider({"queries": records})
