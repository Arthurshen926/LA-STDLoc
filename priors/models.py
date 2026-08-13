"""Minimal frozen Gaussian models used by the LaFGS prior rasterizer.

The paper pipeline consumes an externally reconstructed RGB Gaussian map.  It
never optimizes Gaussian appearance or geometry, so the training-oriented
GraphDeco model (optimizers, densification and localization overlays) is not a
runtime dependency here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from torch import nn
from torch.nn import functional as F


def _ordered_properties(vertex, prefix: str) -> list[str]:
    names = [prop.name for prop in vertex.properties if prop.name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[-1]))


def _stack_properties(vertex, names: list[str], rows: int) -> np.ndarray:
    if not names:
        return np.empty((rows, 0), dtype=np.float32)
    return np.stack([np.asarray(vertex[name]) for name in names], axis=1).astype(
        np.float32, copy=False
    )


class FrozenGaussianModel(nn.Module):
    """Read-only 2DGS/3DGS PLY representation with gsplat-compatible fields."""

    def __init__(
        self,
        sh_degree: int = 3,
        *,
        expected_scale_dimensions: int | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.max_sh_degree = int(sh_degree)
        self.active_sh_degree = 0
        self.expected_scale_dimensions = expected_scale_dimensions
        self.target_device = torch.device(device)
        self._xyz = nn.Parameter(torch.empty((0, 3)), requires_grad=False)
        self._features_dc = nn.Parameter(torch.empty((0, 1, 3)), requires_grad=False)
        self._features_rest = nn.Parameter(torch.empty((0, 0, 3)), requires_grad=False)
        self._scaling = nn.Parameter(torch.empty((0, 0)), requires_grad=False)
        self._rotation = nn.Parameter(torch.empty((0, 4)), requires_grad=False)
        self._opacity = nn.Parameter(torch.empty((0, 1)), requires_grad=False)
        self._loc_feature = nn.Parameter(torch.empty((0, 1, 0)), requires_grad=False)

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return F.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_loc_feature(self) -> torch.Tensor:
        return self._loc_feature

    @property
    def get_loc_opacity(self) -> torch.Tensor:
        return self.get_opacity

    def load_ply(self, path: str | Path, loc_feature_dim: int | None = None) -> None:
        vertex = PlyData.read(str(Path(path).expanduser())).elements[0]
        rows = len(vertex.data)
        xyz = np.stack(
            [np.asarray(vertex[name]) for name in ("x", "y", "z")], axis=1
        ).astype(np.float32, copy=False)
        opacity = np.asarray(vertex["opacity"], dtype=np.float32)[:, None]

        dc_names = [f"f_dc_{index}" for index in range(3)]
        missing_dc = [name for name in dc_names if name not in vertex.data.dtype.names]
        if missing_dc:
            raise ValueError(f"Gaussian PLY is missing SH DC fields: {missing_dc}")
        features_dc = _stack_properties(vertex, dc_names, rows)[:, None, :]

        rest_names = _ordered_properties(vertex, "f_rest_")
        expected_rest = 3 * (self.max_sh_degree + 1) ** 2 - 3
        if len(rest_names) != expected_rest:
            raise ValueError(
                f"Expected {expected_rest} f_rest fields for SH degree "
                f"{self.max_sh_degree}, found {len(rest_names)}"
            )
        features_rest = _stack_properties(vertex, rest_names, rows).reshape(
            rows, 3, (self.max_sh_degree + 1) ** 2 - 1
        )
        features_rest = np.transpose(features_rest, (0, 2, 1)).copy()

        scales = _stack_properties(vertex, _ordered_properties(vertex, "scale_"), rows)
        if self.expected_scale_dimensions is not None and scales.shape[1] != int(
            self.expected_scale_dimensions
        ):
            raise ValueError(
                f"Expected {self.expected_scale_dimensions} scale fields, "
                f"found {scales.shape[1]}"
            )
        rotations = _stack_properties(vertex, _ordered_properties(vertex, "rot_"), rows)
        if rotations.shape[1] != 4:
            raise ValueError(
                f"Expected four quaternion fields, found {rotations.shape[1]}"
            )

        loc_names = _ordered_properties(vertex, "loc_")
        if loc_names:
            loc = _stack_properties(vertex, loc_names, rows)[:, None, :]
        else:
            dimension = 256 if loc_feature_dim is None else int(loc_feature_dim)
            if dimension < 0:
                raise ValueError("loc_feature_dim must be non-negative")
            if dimension == 0:
                # RGB-only consumers never rasterize localization features.
                # Avoid allocating and normalizing an otherwise unused
                # [Gaussian, 256] random bank for large appearance priors.
                loc = np.empty((rows, 1, 0), dtype=np.float32)
            else:
                generator = np.random.default_rng(0)
                loc = generator.standard_normal((rows, dimension)).astype(np.float32)
                loc /= np.clip(np.linalg.norm(loc, axis=1, keepdims=True), 1e-12, None)
                loc = loc[:, None, :]

        device = self.target_device
        self._xyz = nn.Parameter(torch.from_numpy(xyz).to(device), requires_grad=False)
        self._features_dc = nn.Parameter(
            torch.from_numpy(features_dc).to(device), requires_grad=False
        )
        self._features_rest = nn.Parameter(
            torch.from_numpy(features_rest).to(device), requires_grad=False
        )
        self._opacity = nn.Parameter(
            torch.from_numpy(opacity).to(device), requires_grad=False
        )
        self._scaling = nn.Parameter(
            torch.from_numpy(scales).to(device), requires_grad=False
        )
        self._rotation = nn.Parameter(
            torch.from_numpy(rotations).to(device), requires_grad=False
        )
        self._loc_feature = nn.Parameter(
            torch.from_numpy(loc).to(device), requires_grad=False
        )
        self.active_sh_degree = self.max_sh_degree


class GaussianModel3D(FrozenGaussianModel):
    def __init__(
        self, sh_degree: int = 3, *, device: str | torch.device = "cuda"
    ) -> None:
        super().__init__(sh_degree, expected_scale_dimensions=3, device=device)


class GaussianModel2D(FrozenGaussianModel):
    def __init__(
        self, sh_degree: int = 3, *, device: str | torch.device = "cuda"
    ) -> None:
        super().__init__(sh_degree, expected_scale_dimensions=2, device=device)


def gaussian_model(prior_type: str, sh_degree: int = 3) -> FrozenGaussianModel:
    normalized = str(prior_type).strip().lower()
    if normalized == "2dgs":
        return GaussianModel2D(sh_degree)
    if normalized == "3dgs":
        return GaussianModel3D(sh_degree)
    raise ValueError(f"Unsupported Gaussian prior type: {prior_type!r}")
