from __future__ import annotations

import numpy as np
from plyfile import PlyData, PlyElement

from priors.models import GaussianModel2D


def _write_minimal_2dgs(path, *, rows: int = 2) -> None:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{index}" for index in range(3)]
    names += [f"f_rest_{index}" for index in range(45)]
    names += ["opacity", "scale_0", "scale_1"]
    names += [f"rot_{index}" for index in range(4)]
    values = np.zeros(rows, dtype=[(name, "f4") for name in names])
    values["rot_0"] = 1.0
    PlyData([PlyElement.describe(values, "vertex")], text=False).write(path)


def test_rgb_only_ply_load_omits_unused_random_localization_bank(tmp_path):
    path = tmp_path / "prior.ply"
    _write_minimal_2dgs(path)
    model = GaussianModel2D(sh_degree=3, device="cpu")

    model.load_ply(path, loc_feature_dim=0)

    assert model.get_xyz.shape == (2, 3)
    assert model.get_loc_feature.shape == (2, 1, 0)
