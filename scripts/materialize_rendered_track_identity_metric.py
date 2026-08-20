#!/usr/bin/env python3
"""Materialize a map-bound zero-residual shared metric for A0 evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.metric import (
    SharedLowRankMetric,
    validate_map_bound_identity_metric,
)


def materialize(*, map_path: Path, expected_map_sha256: str, output_path: Path) -> dict:
    map_path = map_path.resolve()
    output_path = output_path.resolve()
    if map_path == output_path:
        raise ValueError("identity metric output aliases its input map")
    if output_path.exists():
        raise FileExistsError(output_path)
    before_sha256 = sha256_file(map_path)
    if before_sha256 != str(expected_map_sha256):
        raise ValueError(
            f"map SHA differs: expected {expected_map_sha256}, got {before_sha256}"
        )
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    anchor_ids = torch.as_tensor(state.get("anchor_ids"))
    features = torch.as_tensor(state.get("anchor_features"))
    if anchor_ids.dtype != torch.int64 or anchor_ids.ndim != 1:
        raise ValueError("map anchor_ids must be an exact int64 vector")
    if anchor_ids.unique().numel() != anchor_ids.numel():
        raise ValueError("map anchor_ids are not unique")
    if features.ndim != 2 or features.shape[0] != anchor_ids.numel():
        raise ValueError("map features are not row-aligned with anchor_ids")
    if not features.is_floating_point() or not torch.isfinite(features).all():
        raise ValueError("map features must be finite floating-point rows")
    descriptor_dim = int(features.shape[1])
    if descriptor_dim <= 0:
        raise ValueError("map descriptor dimension must be positive")

    metric = SharedLowRankMetric(
        descriptor_dim=descriptor_dim, rank=1, max_residual_norm=0.0
    )
    with torch.no_grad():
        for parameter in metric.parameters():
            parameter.zero_()
    payload = {
        "schema": "lafgs_shared_metric_state",
        "version": 1,
        "landmark_indices": torch.arange(anchor_ids.numel(), dtype=torch.long),
        "metric_config": metric.export_config(),
        "metric_state_dict": {
            name: value.detach().cpu() for name, value in metric.state_dict().items()
        },
        "map_path": str(map_path),
        "map_sha256": before_sha256,
        "step": 0,
        "protocol": "rendered_track_map_bound_identity",
        "producer": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": sha256_file(Path(__file__).resolve()),
            "torch_version": torch.__version__,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        reloaded = torch.load(temporary, map_location="cpu", weights_only=False)
        validate_map_bound_identity_metric(
            reloaded,
            descriptor_dim=descriptor_dim,
            anchor_count=int(anchor_ids.numel()),
            map_path=str(map_path),
            map_sha256=before_sha256,
        )
        if sha256_file(map_path) != before_sha256:
            raise RuntimeError("input map changed during identity materialization")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "schema": "lafgs_rendered_track_identity_metric_materialization",
        "version": 1,
        "map": str(map_path),
        "map_sha256": before_sha256,
        "anchor_count": int(anchor_ids.numel()),
        "descriptor_dim": descriptor_dim,
        "descriptor_transform": "none_identity_only",
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = materialize(
        map_path=args.map,
        expected_map_sha256=args.expected_map_sha256,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
