#!/usr/bin/env python3
"""Split a sealed query plan for parallel rendering without changing its poses."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch


def _save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    source = torch.load(args.plan, map_location="cpu", weights_only=False)
    count = int(source["query_count"])
    args.output_dir.mkdir(parents=True)
    for shard_index in range(args.shard_count):
        indices = torch.arange(shard_index, count, args.shard_count).long()
        shard = dict(source)
        for key, value in list(source.items()):
            if torch.is_tensor(value) and value.ndim and value.shape[0] == count:
                shard[key] = value[indices].clone()
            elif isinstance(value, list) and len(value) == count:
                shard[key] = [value[int(index)] for index in indices]
        shard["query_indices"] = indices.clone()
        shard["query_count"] = int(indices.numel())
        shard["parent_plan"] = str(args.plan.resolve())
        shard["parent_plan_sha256"] = hashlib.sha256(args.plan.read_bytes()).hexdigest()
        shard["render_shard_index"] = shard_index
        shard["render_shard_count"] = args.shard_count
        _save(shard, args.output_dir / f"plan_shard{shard_index}.pt")


if __name__ == "__main__":
    main()
