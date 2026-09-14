"""Unified in-memory contract for 2DGS and 3DGS PLY priors."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData

from common.hashing import sha256_file
from common.schemas import PRIOR_SCHEMA, PRIOR_VERSION


@dataclass(frozen=True)
class GaussianPrior:
    path: Path
    prior_type: str
    source_method: str
    primitive_ids: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    appearance: np.ndarray
    properties: np.ndarray
    manifest: dict[str, Any]

    @property
    def primitive_count(self) -> int:
        return int(self.means.shape[0])

    def validate(self) -> None:
        count = self.primitive_count
        if self.prior_type not in {"2dgs", "3dgs"}:
            raise ValueError(f"unsupported prior type: {self.prior_type}")
        expected_scale_dim = 2 if self.prior_type == "2dgs" else 3
        expected = {
            "primitive_ids": (count,),
            "means": (count, 3),
            "scales": (count, expected_scale_dim),
            "rotations": (count, 4),
            "opacities": (count,),
        }
        actual = {
            "primitive_ids": self.primitive_ids.shape,
            "means": self.means.shape,
            "scales": self.scales.shape,
            "rotations": self.rotations.shape,
            "opacities": self.opacities.shape,
        }
        if actual != expected:
            raise ValueError(f"Gaussian field shape mismatch: {actual} != {expected}")
        if not np.array_equal(self.primitive_ids, np.arange(count, dtype=np.int64)):
            raise ValueError("source primitive IDs must preserve PLY row order")
        for name, value in (("means", self.means), ("scales", self.scales),
                            ("rotations", self.rotations), ("opacities", self.opacities)):
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite values in prior {name}")


def _stack(data: np.ndarray, names: list[str]) -> np.ndarray:
    return np.stack([np.asarray(data[name], dtype=np.float32) for name in names], axis=1)


def _manifest_type(manifest: dict[str, Any], data: np.ndarray) -> str:
    value = manifest.get("type", manifest.get("gaussian_type"))
    if value in {"2dgs", "3dgs"}:
        return str(value)
    scale_count = sum(name.startswith("scale_") for name in data.dtype.names or ())
    if scale_count == 2:
        return "2dgs"
    if scale_count == 3:
        return "3dgs"
    raise ValueError(f"cannot infer Gaussian type from {scale_count} scale fields")


def load_prior(
    ply_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    source_method: str | None = None,
) -> GaussianPrior:
    path = Path(ply_path).expanduser().resolve()
    element = PlyData.read(path)["vertex"]
    data = element.data
    names = list(data.dtype.names or ())
    if manifest_path is None:
        candidates = (path.parent / "prior_manifest.json", path.parents[2] / "rgb_prior_manifest.json")
        manifest_file = next((item for item in candidates if item.is_file()), None)
    else:
        manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text()) if manifest_file else {}
    if manifest.get("schema") == PRIOR_SCHEMA and manifest.get("version") != PRIOR_VERSION:
        raise ValueError("unsupported Gaussian prior manifest version")
    prior_type = _manifest_type(manifest, data)
    scale_names = [f"scale_{index}" for index in range(2 if prior_type == "2dgs" else 3)]
    required = ["x", "y", "z", *scale_names, "rot_0", "rot_1", "rot_2", "rot_3", "opacity"]
    missing = sorted(set(required) - set(names))
    if missing:
        raise ValueError(f"Gaussian PLY is missing fields: {missing}")
    appearance_names = sorted(
        (name for name in names if name.startswith(("f_dc_", "f_rest_"))),
        key=lambda name: (name.split("_")[1], int(name.rsplit("_", 1)[1])),
    )
    prior = GaussianPrior(
        path=path,
        prior_type=prior_type,
        source_method=str(source_method or manifest.get("source_method", "unknown")),
        primitive_ids=np.arange(len(data), dtype=np.int64),
        means=_stack(data, ["x", "y", "z"]),
        scales=_stack(data, scale_names),
        rotations=_stack(data, ["rot_0", "rot_1", "rot_2", "rot_3"]),
        opacities=np.asarray(data["opacity"], dtype=np.float32),
        appearance=_stack(data, appearance_names),
        properties=data,
        manifest={**manifest, "resolved_ply_sha256": sha256_file(path)},
    )
    prior.validate()
    expected_count = manifest.get("primitive_count")
    if expected_count is not None and int(expected_count) != prior.primitive_count:
        raise ValueError("prior manifest primitive count does not match PLY")
    expected_sha = manifest.get("ply_sha256", manifest.get("exported_ply_sha256"))
    if expected_sha is not None and expected_sha != prior.manifest["resolved_ply_sha256"]:
        raise ValueError("prior manifest PLY SHA-256 does not match")
    return prior
