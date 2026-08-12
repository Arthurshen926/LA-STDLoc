#!/usr/bin/env python3
"""Materialize and audit robust descriptors from real mapping observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from topology.observation_descriptor import materialize_observation_descriptor_audit


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help=(
            "Small per-Anchor reductions are fastest and most reproducible "
            "with one thread."
        ),
    )
    args = parser.parse_args()
    if int(args.cpu_threads) < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(int(args.cpu_threads))

    result = materialize_observation_descriptor_audit(
        _load(args.registry),
        _load(args.query_cache),
        trim_fraction=float(args.trim_fraction),
    )
    result["inputs"] = {
        "registry": str(args.registry.resolve()),
        "query_cache": str(args.query_cache.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {
                "schema": result["schema"],
                "version": result["version"],
                "uses_test_queries": result["uses_test_queries"],
                "audit_only": result["audit_only"],
                "deployment_descriptor_mutated": result[
                    "deployment_descriptor_mutated"
                ],
                "fusion_policy": result["fusion_policy"],
                "inputs": result["inputs"],
                "report": result["report"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(result["report"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
