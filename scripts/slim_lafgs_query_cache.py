#!/usr/bin/env python3
"""Remove rendered tensors from a LaFGS cache after geometry teachers exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


SPARSE_KEYS = (
    "pose_w2c",
    "pixel_center_offset",
    "native_keypoints",
    "native_descriptors",
    "native_scores",
    "native_K",
    "native_input_hw",
    "native_depth",
)

POSE_KEYS = (
    "pose_w2c",
    "pixel_center_offset",
    "native_keypoints",
    "native_K",
    "native_input_hw",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--profile",
        choices=("sparse", "pose"),
        default="sparse",
        help="pose omits descriptors, scores, and depth for exact PnP replay",
    )
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    cache = payload.get("queries", payload)
    names = list(cache)
    keys = SPARSE_KEYS if args.profile == "sparse" else POSE_KEYS
    slim = {}
    for name in names:
        missing = [key for key in keys if key not in cache[name]]
        if missing:
            raise ValueError(f"{name} missing sparse fields: {missing}")
        slim[name] = {key: cache[name][key] for key in keys}
    result = {
        "schema": (
            "lafgs_native_sparse_query_cache"
            if args.profile == "sparse"
            else "lafgs_pose_replay_query_cache"
        ),
        "version": 1,
        "source": str(source),
        "profile": args.profile,
        "queries": slim,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "query_count": len(slim),
                "fields": list(keys),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
