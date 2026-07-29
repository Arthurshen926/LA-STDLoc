#!/usr/bin/env python3
"""Bind a frozen query metric to an identity-preserving extended anchor map."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def extend_metric_state(metric: dict, anchor_map: dict, map_path: Path) -> dict:
    map_path = Path(map_path)
    old = torch.as_tensor(metric["landmark_indices"]).long()
    current = torch.as_tensor(anchor_map["anchor_ids"]).long()
    if current.numel() < old.numel() or not torch.equal(
        current[: old.numel()], old
    ):
        raise ValueError("extended map does not preserve the metric landmark prefix")
    output = dict(metric)
    output["schema"] = "lafgs_extended_shared_metric_state"
    output["version"] = 1
    output["landmark_indices"] = current
    output["map_path"] = str(map_path.resolve())
    output["extended_from_anchor_count"] = int(old.numel())
    output["extended_anchor_count"] = int(current.numel() - old.numel())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric-state", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    map_path = Path(args.map)
    metric = torch.load(
        args.metric_state, map_location="cpu", weights_only=False
    )
    anchor_map = torch.load(map_path, map_location="cpu", weights_only=False)
    output = extend_metric_state(metric, anchor_map, map_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
