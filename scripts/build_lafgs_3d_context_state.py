#!/usr/bin/env python3
"""Build a fixed, matcher-independent 3D neighborhood context map."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from localization_training.contextual_descriptor import (
    flatten_context,
    multiscale_map_3d_context,
)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--neighbor-counts", default="8,24,64")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    neighbor_counts = tuple(
        int(value) for value in args.neighbor_counts.split(",")
    )
    device = torch.device(args.device)
    context = multiscale_map_3d_context(
        torch.as_tensor(state["anchor_features"]).float().to(device),
        torch.as_tensor(state["anchor_xyz"]).float().to(device),
        neighbor_counts=neighbor_counts,
        chunk_size=args.chunk_size,
    )
    context = flatten_context(context).detach().cpu()
    output = {
        "schema": "lafgs_fixed_3d_context_state",
        "version": 1,
        "anchor_ids": torch.as_tensor(state["anchor_ids"]).long().cpu(),
        "anchor_ids_sha256": _tensor_sha256(state["anchor_ids"]),
        "anchor_context": context.half(),
        "context_dim": int(context.shape[1]),
        "config": {
            "map_neighbor_counts": list(neighbor_counts),
            "sparse_radii_px": [48.0, 96.0, 192.0],
            "maximum_sparse_neighbors": 48,
            "context_chunk_size": int(args.chunk_size),
            "coordinate_contract": "relative_knn_no_absolute_world_position",
            "graph_contract": "fixed_geometry_not_matcher_top1",
        },
        "provenance": {
            "map": str(Path(args.map).resolve()),
        },
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    print(
        {
            "output": str(path),
            "anchor_count": int(context.shape[0]),
            "context_dim": int(context.shape[1]),
            "neighbor_counts": neighbor_counts,
        }
    )


if __name__ == "__main__":
    main()
