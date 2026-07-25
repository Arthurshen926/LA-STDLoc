#!/usr/bin/env python
"""Build a zero-training native SuperPoint support-view context index."""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localization_training.query_context import (
    spatial_pyramid_global_descriptor,
)
from localization_training.ulf_initializer import sample_mask_at_grid_uv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_cache", required=True)
    parser.add_argument("--visibility_cache", required=True)
    parser.add_argument("--map_state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid_rows", type=int, default=2)
    parser.add_argument("--grid_cols", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    cache = torch.load(args.query_cache, map_location="cpu")["queries"]
    visibility = torch.load(args.visibility_cache, map_location="cpu")[
        "visibility"
    ]
    state = torch.load(args.map_state, map_location="cpu")
    names = sorted(set(cache) & set(visibility))
    embeddings = []
    visibility_rows = []
    valid_counts = []
    with torch.inference_mode():
        for name in tqdm(names, desc="Query context index"):
            query = cache[name]
            keypoints = torch.as_tensor(
                query["native_keypoints"], device=device
            ).float()
            descriptors = torch.as_tensor(
                query["native_descriptors"], device=device
            ).float()
            scores = torch.as_tensor(
                query["native_scores"], device=device
            ).float()
            valid_mask = torch.as_tensor(
                query["native_valid_mask"], device=device
            ).bool()
            keep = sample_mask_at_grid_uv(valid_mask, keypoints)
            embeddings.append(
                spatial_pyramid_global_descriptor(
                    descriptors[keep],
                    keypoints[keep],
                    scores[keep],
                    tuple(map(int, query["native_input_hw"])),
                    grid_rows=args.grid_rows,
                    grid_cols=args.grid_cols,
                ).half().cpu()
            )
            visibility_rows.append(
                torch.as_tensor(visibility[name]).bool().cpu()
            )
            valid_counts.append(int(keep.sum().item()))
    artifact = {
        "version": 1,
        "landmark_indices": torch.as_tensor(
            state["landmark_indices"]
        ).reshape(-1).cpu(),
        "support_names": names,
        "support_embeddings": torch.stack(embeddings),
        "support_visibility": torch.stack(visibility_rows),
        "config": {
            **vars(args),
            "support_count": len(names),
            "embedding_dim": int(embeddings[0].numel()),
            "valid_keypoint_mean": float(
                sum(valid_counts) / max(len(valid_counts), 1)
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    output.with_suffix(".json").write_text(
        json.dumps(artifact["config"], indent=2)
    )
    print(json.dumps(artifact["config"], indent=2))


if __name__ == "__main__":
    main()
