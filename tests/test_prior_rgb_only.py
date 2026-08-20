from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement
import pytest
import torch

from priors import provenance
from priors.models import GaussianModel2D


def _write_minimal_2dgs(path, *, rows: int = 2, loc_values=None) -> None:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(3)]
    names += [f"f_rest_{index}" for index in range(45)]
    names += ["opacity", "scale_0", "scale_1"]
    names += [f"rot_{index}" for index in range(4)]
    if loc_values is not None:
        names += [f"loc_{index}" for index in range(loc_values.shape[1])]
    values = np.zeros(rows, dtype=[(name, "f4") for name in names])
    values["rot_0"] = 1.0
    if loc_values is not None:
        for index in range(loc_values.shape[1]):
            values[f"loc_{index}"] = loc_values[:, index]
    PlyData([PlyElement.describe(values, "vertex")], text=False).write(path)


def test_rgb_only_ply_load_omits_unused_random_localization_bank(
    tmp_path, monkeypatch
):
    path = tmp_path / "prior.ply"
    _write_minimal_2dgs(path)
    model = GaussianModel2D(sh_degree=3, device="cpu")
    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda *_args, **_kwargs: pytest.fail("RGB-only load generated loc features"),
    )

    model.load_ply(path, loc_feature_dim=0)

    assert model.get_xyz.shape == (2, 3)
    assert model.get_loc_feature.shape == (2, 1, 0)


def test_default_and_explicit_256_localization_banks_are_exact(tmp_path):
    path = tmp_path / "prior.ply"
    _write_minimal_2dgs(path)
    default = GaussianModel2D(sh_degree=3, device="cpu")
    explicit = GaussianModel2D(sh_degree=3, device="cpu")

    default.load_ply(path)
    explicit.load_ply(path, loc_feature_dim=256)

    assert default.state_dict().keys() == explicit.state_dict().keys()
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name]), name


def test_present_localization_fields_take_precedence_over_rgb_only_hint(tmp_path):
    path = tmp_path / "prior-with-loc.ply"
    expected = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    _write_minimal_2dgs(path, loc_values=expected)
    model = GaussianModel2D(sh_degree=3, device="cpu")

    model.load_ply(path, loc_feature_dim=0)

    assert torch.equal(model.get_loc_feature[:, 0], torch.from_numpy(expected))


def test_negative_localization_dimension_is_rejected(tmp_path):
    path = tmp_path / "prior.ply"
    _write_minimal_2dgs(path)
    model = GaussianModel2D(sh_degree=3, device="cpu")

    with pytest.raises(ValueError, match="non-negative"):
        model.load_ply(path, loc_feature_dim=-1)


@pytest.mark.parametrize("gaussian_type", ["2dgs", "3dgs"])
def test_raster_provenance_caller_requests_rgb_only_load(
    tmp_path, monkeypatch, gaussian_type
):
    calls = []

    class FakeGaussian:
        def __init__(self, degree):
            calls.append(("init", degree))

        def load_ply(self, path, *, loc_feature_dim):
            calls.append(("load", path, loc_feature_dim))

        def cuda(self):
            calls.append(("cuda",))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    monkeypatch.setattr(provenance, "GaussianModel2D", FakeGaussian)
    monkeypatch.setattr(provenance, "GaussianModel3D", FakeGaussian)
    ply = tmp_path / "prior.ply"

    result = provenance._load_rgb_only_gaussians(gaussian_type, 3, ply)

    assert isinstance(result, FakeGaussian)
    assert calls == [("init", 3), ("load", ply, 0), ("cuda",), ("eval",)]
