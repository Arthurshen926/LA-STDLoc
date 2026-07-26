#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from localization_training.gaussian_prior import GaussianPriorGeometry


def _properties(vertex, prefix):
    names = [name for name in vertex.data.dtype.names if name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def _selected_matrix(vertex, names, indices):
    value = np.stack(
        [np.asarray(vertex[name])[indices] for name in names], axis=1
    )
    return torch.from_numpy(value.copy()).float()


def _masked(value, valid):
    fill = torch.full_like(value, float("inf"))
    mask = valid
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask, value, fill)


def _add_residuals(geometry, prefix, prior, triangulated_xyz, valid):
    safe_xyz = torch.where(valid[:, None], triangulated_xyz, prior.xyz)
    residual = prior.surface_residual_components(safe_xyz)
    geometry.update(
        {
            f"triangulation_{prefix}_{name}": _masked(value, valid)
            for name, value in residual.items()
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Augment frozen triangulation evidence with Gaussian-frame "
            "surface residuals without rerunning matching or rendering"
        )
    )
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--rgb_ply", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    statistics_path = Path(args.statistics).expanduser().resolve()
    state_path = Path(args.state).expanduser().resolve()
    rgb_ply_path = Path(args.rgb_ply).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    payload = torch.load(
        statistics_path, map_location="cpu", weights_only=False
    )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    indices = torch.as_tensor(
        payload["landmark_indices"], dtype=torch.long
    ).reshape(-1)
    state_indices = torch.as_tensor(
        state["landmark_indices"], dtype=torch.long
    ).reshape(-1)
    if not torch.equal(indices, state_indices):
        raise ValueError("Statistics and state landmark IDs are not aligned")

    ply = PlyData.read(str(rgb_ply_path))
    vertex = ply.elements[0]
    maximum_index = int(indices.max().item()) if indices.numel() else -1
    if maximum_index >= len(vertex):
        raise ValueError("Statistics landmark ID exceeds RGB PLY primitive count")
    numpy_indices = indices.numpy()
    rgb_xyz = _selected_matrix(
        vertex, ["x", "y", "z"], numpy_indices
    )
    rotation_names = _properties(vertex, "rot_")
    scaling_names = _properties(vertex, "scale_")
    if len(rotation_names) != 4 or len(scaling_names) not in {2, 3}:
        raise ValueError(
            "RGB PLY does not contain a supported Gaussian rotation/scale"
        )
    rotation = _selected_matrix(vertex, rotation_names, numpy_indices)
    scaling = _selected_matrix(vertex, scaling_names, numpy_indices).exp()
    gaussian_type = "2dgs" if scaling.shape[1] == 2 else "3dgs"

    geometry = dict(payload["geometry_evidence"])
    recorded_type = str(geometry.get("gaussian_type", gaussian_type)).lower()
    if recorded_type != gaussian_type:
        raise ValueError(
            f"PLY implies {gaussian_type}, statistics record {recorded_type}"
        )
    triangulated_xyz = torch.as_tensor(
        geometry["triangulated_xyz"], dtype=torch.float32
    )
    triangulated = torch.as_tensor(
        geometry["triangulated"], dtype=torch.bool
    ).reshape(-1)
    current_xyz = torch.as_tensor(
        state["landmark_xyz"], dtype=torch.float32
    )
    if triangulated_xyz.shape != current_xyz.shape:
        raise ValueError("Triangulation and state geometry shapes differ")

    geometry["rotation"] = rotation
    geometry["scaling"] = scaling
    rgb_prior = GaussianPriorGeometry(
        gaussian_type=gaussian_type,
        xyz=rgb_xyz,
        rotation=rotation,
        scaling=scaling,
    )
    current_prior = GaussianPriorGeometry(
        gaussian_type=gaussian_type,
        xyz=current_xyz,
        rotation=rotation,
        scaling=scaling,
    )
    _add_residuals(
        geometry, "rgb", rgb_prior, triangulated_xyz, triangulated
    )
    _add_residuals(
        geometry, "current", current_prior, triangulated_xyz, triangulated
    )

    output = dict(payload)
    output["version"] = max(int(output.get("version", 0)), 5)
    output["geometry_evidence"] = geometry
    diagnostics = dict(output.get("diagnostics", {}))
    diagnostics.update(
        {
            "surface_geometry_rgb_ply": str(rgb_ply_path),
            "surface_geometry_source_state": str(state_path),
            "surface_geometry_augmented": True,
        }
    )
    output["diagnostics"] = diagnostics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        json.dumps(
            {
                "source_statistics": str(statistics_path),
                "state": str(state_path),
                "rgb_ply": str(rgb_ply_path),
                "output": str(output_path),
                "gaussian_type": gaussian_type,
                "landmark_count": int(indices.numel()),
                "triangulated_count": int(triangulated.sum().item()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
