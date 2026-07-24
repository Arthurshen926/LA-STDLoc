#!/usr/bin/env python3
"""Subset an ID-aligned landmark visibility cache."""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visibility_cache", required=True)
    parser.add_argument("--source_state", required=True)
    parser.add_argument("--target_state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = torch.load(args.source_state, map_location="cpu")
    target = torch.load(args.target_state, map_location="cpu")
    source_ids = torch.as_tensor(source["landmark_indices"]).reshape(-1)
    target_ids = torch.as_tensor(target["landmark_indices"]).reshape(-1)
    source_id_to_local = {
        int(value): index for index, value in enumerate(source_ids.tolist())
    }
    try:
        selected = torch.tensor(
            [source_id_to_local[int(value)] for value in target_ids.tolist()],
            dtype=torch.long,
        )
    except KeyError as error:
        raise ValueError(
            f"Target landmark {error.args[0]} is absent from the source bank"
        ) from error
    payload = torch.load(args.visibility_cache, map_location="cpu")
    visibility = payload["visibility"]
    subset = {}
    for name, value in visibility.items():
        value = torch.as_tensor(value)
        if value.shape[0] != source_ids.numel():
            raise ValueError(
                f"Visibility for {name} has {value.shape[0]} landmarks, "
                f"expected {source_ids.numel()}"
            )
        subset[name] = value[selected]
    result = dict(payload)
    result["visibility"] = subset
    result["source_landmark_indices"] = source_ids
    result["landmark_indices"] = target_ids
    result["source_visibility_cache"] = str(
        Path(args.visibility_cache).resolve()
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    print(
        f"Saved {len(subset)} views x {target_ids.numel()} landmarks to "
        f"{output}"
    )


if __name__ == "__main__":
    main()
