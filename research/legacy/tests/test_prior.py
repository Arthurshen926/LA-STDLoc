import json
import hashlib
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import pytest

from priors.anysplat import load_anysplat
from priors.gaussian_2d import load_gaussian_2d
from priors.gaussian_3d import load_gaussian_3d
from priors.base import load_prior


def _write_prior(path: Path, scale_dimensions: int) -> Path:
    fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        *((f"scale_{index}", "f4") for index in range(scale_dimensions)),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    values = np.zeros(3, dtype=fields)
    values["z"] = [1, 2, 3]
    values["rot_0"] = 1
    PlyData([PlyElement.describe(values, "vertex")]).write(path)
    return path


def test_2d_and_3d_prior_shape_contract(tmp_path: Path):
    prior_2d = load_gaussian_2d(_write_prior(tmp_path / "2d.ply", 2))
    prior_3d = load_gaussian_3d(_write_prior(tmp_path / "3d.ply", 3))
    assert prior_2d.scales.shape == (3, 2)
    assert prior_3d.scales.shape == (3, 3)
    assert np.array_equal(prior_3d.primitive_ids, np.arange(3))


def test_anysplat_requires_3d_gaussians(tmp_path: Path):
    with pytest.raises(ValueError, match="3DGS"):
        load_anysplat(_write_prior(tmp_path / "2d.ply", 2))


def test_prior_manifest_rejects_stale_ply_sha(tmp_path: Path):
    ply = _write_prior(tmp_path / "3d.ply", 3)
    manifest = {
        "schema": "lafgs_gaussian_prior",
        "version": 1,
        "type": "3dgs",
        "source_method": "vanilla_3dgs",
        "primitive_count": 3,
        "ply_sha256": hashlib.sha256(b"different").hexdigest(),
    }
    path = tmp_path / "prior_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256"):
        load_prior(ply, manifest_path=path)
