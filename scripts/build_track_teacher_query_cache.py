#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch


TRACK_FIELDS = (
    "native_descriptors",
    "native_keypoints",
    "native_scores",
    "native_K",
    "native_input_hw",
    "pose_w2c",
)


def main():
    parser = argparse.ArgumentParser(
        description="Strip a full query cache to track-teacher inputs"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    queries = payload.get("queries", payload)
    stripped = {}
    observation_count = 0
    for name, value in queries.items():
        if not isinstance(value, dict) or "native_descriptors" not in value:
            continue
        missing = [field for field in TRACK_FIELDS if field not in value]
        if missing:
            raise ValueError(f"{name} lacks track fields: {missing}")
        depth = torch.as_tensor(value["native_depth"])
        keypoints = torch.as_tensor(value["native_keypoints"]).float()
        xy = keypoints.round().long()
        xy[:, 0].clamp_(0, int(depth.shape[1]) - 1)
        xy[:, 1].clamp_(0, int(depth.shape[0]) - 1)
        item = {field: value[field] for field in TRACK_FIELDS}
        item["native_depth_at_keypoints"] = depth[xy[:, 1], xy[:, 0]].clone()
        stripped[name] = item
        observation_count += int(keypoints.shape[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": 1,
            "source_query_cache": str(source),
            "queries": stripped,
        },
        output,
    )
    print(
        json.dumps(
            {
                "source": str(source),
                "output": str(output),
                "query_count": len(stripped),
                "keypoint_count": observation_count,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
