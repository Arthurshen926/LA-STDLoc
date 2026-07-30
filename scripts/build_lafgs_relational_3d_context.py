#!/usr/bin/env python3
"""Build a true tangent-normal relational context for an active LaFGS map."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from localization_training.relational_context import (
    relational_map_3d_context,
)
from localization_training.ulf_initializer import (
    quaternion_to_rotation_matrix,
)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--gaussian-ply", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--neighbor-count", type=int, default=16)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--minimum-normal-cosine", type=float, default=0.25)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    source_ids = torch.as_tensor(state["source_primitive_ids"]).long()
    vertex = PlyData.read(args.gaussian_ply)["vertex"].data
    if int(source_ids.max()) >= len(vertex):
        raise ValueError("active map source ID exceeds Gaussian PLY")
    rotations = torch.from_numpy(
        np.stack(
            [vertex[f"rot_{index}"] for index in range(4)], axis=1
        ).copy()
    ).float()
    scales = torch.from_numpy(
        np.stack((vertex["scale_0"], vertex["scale_1"]), axis=1).copy()
    ).float()
    source_rotation = quaternion_to_rotation_matrix(rotations[source_ids])
    source_scale = torch.exp(scales[source_ids].mean(dim=1))
    track_ids = torch.as_tensor(
        state.get(
            "track_cluster_ids",
            state.get(
                "coarse_dependency_group_ids",
                state["dependency_group_ids"],
            ),
        )
    ).long()
    device = torch.device(args.device)
    context = relational_map_3d_context(
        torch.as_tensor(state["anchor_features"]).float().to(device),
        torch.as_tensor(state["anchor_xyz"]).float().to(device),
        source_rotation[:, :, 0].to(device),
        source_rotation[:, :, 1].to(device),
        source_rotation[:, :, 2].to(device),
        source_ids=source_ids.to(device),
        track_ids=track_ids.to(device),
        surface_scale=source_scale.to(device),
        neighbor_count=args.neighbor_count,
        candidate_multiplier=args.candidate_multiplier,
        minimum_normal_cosine=args.minimum_normal_cosine,
        chunk_size=args.chunk_size,
    ).cpu()
    descriptor_dim = int(torch.as_tensor(state["anchor_features"]).shape[1])
    output = {
        "schema": "lafgs_fixed_3d_context_state",
        "version": 3,
        "anchor_ids": torch.as_tensor(state["anchor_ids"]).long().cpu(),
        "anchor_ids_sha256": _tensor_sha256(state["anchor_ids"]),
        "anchor_context": context.half(),
        "context_dim": int(context.shape[1]),
        "config": {
            "representation": "relational_tangent_normal_2d3d_v1",
            "query_representation": "relational_sparse_2d_v1",
            "query_neighbor_count": int(args.neighbor_count),
            "map_neighbor_count": int(args.neighbor_count),
            "candidate_multiplier": int(args.candidate_multiplier),
            "minimum_normal_cosine": float(args.minimum_normal_cosine),
            "context_chunk_size": int(args.chunk_size),
            "query_input_dim": int(6 * descriptor_dim + 4),
            "map_input_dim": int(context.shape[1]),
            "coordinate_contract": (
                "relative_2d_layout_to_2dgs_tangent_normal_frame"
            ),
            "graph_contract": (
                "fixed_geometry_source_track_aware_not_matcher_top1"
            ),
        },
        "provenance": {
            "map": str(Path(args.map).resolve()),
            "gaussian_ply": str(Path(args.gaussian_ply).resolve()),
        },
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    print(
        {
            "output": str(path),
            "anchor_count": len(context),
            "context_dim": int(context.shape[1]),
            "query_input_dim": output["config"]["query_input_dim"],
        }
    )


if __name__ == "__main__":
    main()
