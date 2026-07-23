#!/usr/bin/env python
"""Materialize a descriptor-only LaFGS factor-control state.

The historical strong-bank checkpoints can contain bounded-anchor offsets from
an earlier geometry stage.  Reusing those offsets in an IDs/descriptor factor
matrix would silently turn the "strong descriptor" control into a geometry
ablation.  This utility preserves descriptor rows and landmark identity while
resetting every localization anchor to the frozen source 2DGS primitive.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from plyfile import PlyData


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ply_xyz(path: Path) -> torch.Tensor:
    vertex = PlyData.read(str(path)).elements[0].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise ValueError(f"PLY lacks xyz vertex fields: {path}")
    return torch.stack(
        [
            torch.from_numpy(vertex["x"].copy()),
            torch.from_numpy(vertex["y"].copy()),
            torch.from_numpy(vertex["z"].copy()),
        ],
        dim=1,
    ).float()


def main():
    parser = argparse.ArgumentParser(
        description="Reset a historical LaFGS state to frozen source geometry."
    )
    parser.add_argument("--source-state", required=True)
    parser.add_argument("--source-ply", required=True)
    parser.add_argument("--output-state", required=True)
    parser.add_argument("--tangent-bound-m", type=float, default=0.005)
    parser.add_argument("--normal-bound-m", type=float, default=0.002)
    args = parser.parse_args()

    source_state = Path(args.source_state).resolve()
    source_ply = Path(args.source_ply).resolve()
    output_state = Path(args.output_state).resolve()
    if not source_state.is_file():
        raise FileNotFoundError(source_state)
    if not source_ply.is_file():
        raise FileNotFoundError(source_ply)

    state = torch.load(source_state, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("source state must be a dictionary")
    landmark_indices = torch.as_tensor(
        state.get("landmark_indices"), dtype=torch.long
    ).reshape(-1)
    features = torch.as_tensor(state.get("landmark_features"), dtype=torch.float32)
    if landmark_indices.numel() == 0 or features.ndim != 2:
        raise ValueError("source state does not contain a nonempty descriptor bank")
    if features.shape[0] != landmark_indices.numel():
        raise ValueError("source descriptor rows do not align with landmark indices")

    xyz = load_ply_xyz(source_ply)
    if int(landmark_indices.min().item()) < 0 or int(landmark_indices.max().item()) >= xyz.shape[0]:
        raise ValueError("landmark IDs are outside the source PLY primitive range")

    config = dict(state.get("config", {}))
    config.update(
        {
            "factor_control": "strong_ids_strong_descriptor_frozen_base_geometry_v1",
            "factor_source_state": str(source_state),
            "factor_source_state_sha256": file_sha256(source_state),
            "factor_source_ply": str(source_ply),
            "factor_source_ply_sha256": file_sha256(source_ply),
            "factor_geometry_policy": "frozen_source_2dgs_xyz",
            "geometry_frozen": True,
            "raw_xyz_trainable": False,
            "bounded_anchor_trainable": False,
            "tangent_bound_m": float(args.tangent_bound_m),
            "normal_bound_m": float(args.normal_bound_m),
        }
    )
    diagnostics = dict(state.get("diagnostics", {}))
    diagnostics.update(
        {
            "factor_control": "strong_descriptor_base_geometry",
            "factor_source_raw_anchor_offset_absmax": float(
                torch.as_tensor(state.get("raw_anchor_offset", torch.zeros(1))).abs().max().item()
            ),
            "factor_output_raw_anchor_offset_absmax": 0.0,
            "factor_landmark_count": int(landmark_indices.numel()),
        }
    )
    output = {
        "version": max(int(state.get("version", 0)), 6),
        "iteration": int(state.get("iteration", 0)),
        "landmark_indices": landmark_indices.cpu(),
        "landmark_features": torch.nn.functional.normalize(features, dim=1).cpu(),
        "landmark_xyz": xyz[landmark_indices].cpu(),
        "raw_anchor_offset": torch.zeros(
            (landmark_indices.numel(), 3), dtype=torch.float32
        ),
        "mvinit_observation_count": torch.as_tensor(
            state.get("mvinit_observation_count", torch.zeros(landmark_indices.numel()))
        ).cpu(),
        "config": config,
        "diagnostics": diagnostics,
    }
    output_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_state)
    provenance_path = output_state.with_suffix(output_state.suffix + ".json")
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "factor_matrix_descriptor_only_strong_control",
                "source_state": str(source_state),
                "source_state_sha256": file_sha256(source_state),
                "source_ply": str(source_ply),
                "source_ply_sha256": file_sha256(source_ply),
                "output_state": str(output_state),
                "landmark_count": int(landmark_indices.numel()),
                "geometry_policy": "frozen_source_2dgs_xyz",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(output_state)


if __name__ == "__main__":
    main()
