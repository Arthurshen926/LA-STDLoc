"""Attach mapping-calibrated Gaussian surface covariance to a Registry view."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from plyfile import PlyData
import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from topology.anchor_registry import SCHEMA as REGISTRY_SCHEMA


COVARIANCE_TRIANGULATION = 0
COVARIANCE_GAUSSIAN_SURFACE_PRIOR = 1
COVARIANCE_MISSING = 2


def _ordered_names(names: tuple[str, ...], prefix: str) -> list[str]:
    selected = [name for name in names if name.startswith(prefix)]
    return sorted(selected, key=lambda name: int(name.rsplit("_", 1)[-1]))


def _quaternion_frame(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = F.normalize(quaternion, dim=1)
    w, x, y, z = quaternion.unbind(dim=1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (y * x + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (z * x - w * y),
            2 * (z * y + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=1,
    ).reshape(-1, 3, 3)


def _selected_prior_geometry(
    path: str | Path,
    source_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int]:
    vertex = PlyData.read(str(Path(path).expanduser().resolve())).elements[0]
    data = vertex.data
    names = tuple(data.dtype.names or ())
    scale_names = _ordered_names(names, "scale_")
    rotation_names = _ordered_names(names, "rot_")
    if len(scale_names) not in {2, 3}:
        raise ValueError("Gaussian PLY must expose two or three scale fields")
    if len(rotation_names) != 4:
        raise ValueError("Gaussian PLY must expose four quaternion fields")
    source_ids = torch.as_tensor(source_ids).long().reshape(-1)
    if source_ids.numel() and (
        int(source_ids.min()) < 0 or int(source_ids.max()) >= len(data)
    ):
        raise ValueError("Anchor source primitive is outside the Gaussian prior")
    selected = source_ids.numpy()
    xyz = torch.from_numpy(
        np.stack([np.asarray(data[name])[selected] for name in ("x", "y", "z")], axis=1).astype(np.float32)
    )
    log_scaling = torch.from_numpy(
        np.stack([np.asarray(data[name])[selected] for name in scale_names], axis=1).astype(np.float32)
    )
    rotation = torch.from_numpy(
        np.stack([np.asarray(data[name])[selected] for name in rotation_names], axis=1).astype(np.float32)
    )
    return xyz, log_scaling.exp(), rotation, ("2dgs" if len(scale_names) == 2 else "3dgs"), len(data)


def attach_gaussian_prior_covariance(
    registry: Mapping,
    state: Mapping,
    gaussian_ply: str | Path,
) -> dict:
    """Fill base-Anchor covariance without changing localization tensors.

    The proxy follows the existing reserve-geometry surface bounds.  It is an
    audit prior, not an empirical localization posterior and not a merge rule.
    """
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("unsupported Anchor Registry schema")
    anchor_type = torch.as_tensor(registry["anchor_type"]).long()
    source = torch.as_tensor(registry["source_primitive_ids"]).long()
    xyz = torch.as_tensor(registry["anchor_xyz"]).float()
    count = int(anchor_type.numel())
    if source.numel() != count or xyz.shape != (count, 3):
        raise ValueError("Anchor Registry tensors do not align")
    parameters = (
        state.get("track_centric_reconstruction", {})
        .get("calibration", {})
        .get("parameters", {})
    )
    required = ("surface_max_distance_m", "surface_point_plane_m")
    missing = [key for key in required if key not in parameters]
    if missing:
        raise ValueError(f"mapping calibration lacks surface bounds: {missing}")
    tangent_cap = float(parameters["surface_max_distance_m"])
    normal_cap = float(parameters["surface_point_plane_m"])
    if tangent_cap <= 0.0 or normal_cap <= 0.0:
        raise ValueError("surface covariance bounds must be positive")
    base_rows = torch.nonzero(anchor_type == 0, as_tuple=False).reshape(-1)
    prior_xyz, scaling, rotation, gaussian_type, primitive_count = (
        _selected_prior_geometry(gaussian_ply, source[base_rows])
    )
    frame = _quaternion_frame(rotation)
    if gaussian_type == "2dgs":
        tangent = torch.minimum(
            torch.full_like(scaling, tangent_cap), 2.0 * scaling
        )
        bounds = torch.cat(
            (tangent, torch.full((base_rows.numel(), 1), normal_cap)), dim=1
        )
    else:
        bounds = torch.minimum(
            torch.full_like(scaling[:, :3], tangent_cap), 2.0 * scaling[:, :3]
        )
    sigma = (bounds.clamp_min(1e-4) / 3.0).float()
    base_covariance = frame @ torch.diag_embed(sigma.square()) @ frame.transpose(1, 2)

    output = dict(registry)
    covariance = torch.as_tensor(registry["anchor_position_covariance"]).float().clone()
    if covariance.shape != (count, 3, 3):
        raise ValueError("Anchor covariance does not align with Registry")
    covariance[base_rows] = base_covariance
    covariance_source = torch.full(
        (count,), COVARIANCE_MISSING, dtype=torch.int8
    )
    track_rows = torch.nonzero(anchor_type == 1, as_tuple=False).reshape(-1)
    track_finite = torch.isfinite(covariance[track_rows]).reshape(track_rows.numel(), -1).all(dim=1)
    covariance_source[track_rows[track_finite]] = COVARIANCE_TRIANGULATION
    covariance_source[base_rows] = COVARIANCE_GAUSSIAN_SURFACE_PRIOR
    center_distance = torch.full((count,), float("nan"), dtype=torch.float32)
    center_distance[base_rows] = torch.linalg.norm(xyz[base_rows] - prior_xyz, dim=1)
    output.update(
        {
            "anchor_position_covariance": covariance,
            "covariance_source": covariance_source,
            "gaussian_prior_center_distance_m": center_distance,
            "covariance_enrichment": {
                "schema": "lafgs_anchor_covariance_enrichment",
                "version": 1,
                "gaussian_ply": str(Path(gaussian_ply).expanduser().resolve()),
                "gaussian_ply_sha256": sha256_file(gaussian_ply),
                "gaussian_type": gaussian_type,
                "primitive_count": int(primitive_count),
                "base_anchor_count": int(base_rows.numel()),
                "tangent_bound_m": tangent_cap,
                "normal_bound_m": normal_cap,
                "covariance_semantics": "mapping_calibrated_gaussian_surface_prior_proxy",
                "changes_localization_tensors": False,
            },
        }
    )
    output["compatibility"] = {
        **dict(registry.get("compatibility", {})),
        "covariance_enrichment_changes_localization_tensors": False,
    }
    return output
