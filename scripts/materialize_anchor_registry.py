#!/usr/bin/env python3
"""Materialize the V4-compatible Anchor Registry without changing a V3 map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from topology.anchor_covariance import attach_gaussian_prior_covariance
from topology.anchor_registry import build_anchor_registry


def _load(path: Path | None):
    if path is None:
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--track-payload", type=Path)
    parser.add_argument("--selection-provenance", type=Path)
    parser.add_argument("--gaussian-ply", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    state = _load(args.map)
    registry = build_anchor_registry(
        state,
        teacher=_load(args.teacher),
        track_payload=_load(args.track_payload),
        selection_provenance=_load(args.selection_provenance),
    )
    if args.gaussian_ply is not None:
        registry = attach_gaussian_prior_covariance(
            registry, state, args.gaussian_ply
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(registry, args.output)
    report = dict(registry["report"])
    if "covariance_enrichment" in registry:
        report["covariance_enrichment"] = registry["covariance_enrichment"]
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
