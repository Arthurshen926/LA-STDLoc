#!/usr/bin/env python3
"""Interpolate two ID-aligned LaFGS descriptor states."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_state", required=True)
    parser.add_argument("--updated_state", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    base = torch.load(args.base_state, map_location="cpu")
    updated = torch.load(args.updated_state, map_location="cpu")
    base_ids = torch.as_tensor(base["landmark_indices"]).reshape(-1)
    updated_ids = torch.as_tensor(updated["landmark_indices"]).reshape(-1)
    if not torch.equal(base_ids, updated_ids):
        raise ValueError("descriptor states have different landmark IDs")
    base_features = F.normalize(
        torch.as_tensor(base["landmark_features"]).float(), dim=1
    )
    updated_features = F.normalize(
        torch.as_tensor(updated["landmark_features"]).float(), dim=1
    )
    features = F.normalize(
        (1.0 - args.alpha) * base_features
        + args.alpha * updated_features,
        dim=1,
    )
    result = dict(updated)
    result["landmark_features"] = features
    result["config"] = {
        **dict(updated.get("config", {})),
        "descriptor_state_interpolation": {
            "base_state": str(Path(args.base_state).resolve()),
            "updated_state": str(Path(args.updated_state).resolve()),
            "alpha": args.alpha,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    delta = torch.linalg.norm(features - base_features, dim=1)
    print(
        f"Saved alpha={args.alpha:g}; descriptor delta "
        f"mean={delta.mean().item():.6f}, max={delta.max().item():.6f}"
    )


if __name__ == "__main__":
    main()
