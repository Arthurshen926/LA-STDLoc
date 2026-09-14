"""Materialize the frozen Stage-A scaffold as the A0 localization baseline."""

from __future__ import annotations

from pathlib import Path

import torch

from common.config import load_mainline_config
from map_learning.metric import SharedLowRankMetric


def materialize_a0(
    stage_state: str | Path,
    output: str | Path,
    config: str | Path,
) -> tuple[Path, Path]:
    """Write a deployment map and exact identity metric from Stage-A state."""
    stage_state = Path(stage_state).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    state = torch.load(stage_state, map_location="cpu", weights_only=False)
    required = ("landmark_indices", "landmark_xyz", "landmark_features")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"Stage-A state is missing fields: {missing}")

    anchor_ids = torch.as_tensor(state["landmark_indices"]).long().reshape(-1)
    anchor_xyz = torch.as_tensor(state["landmark_xyz"]).float()
    anchor_features = torch.as_tensor(state["landmark_features"]).float()
    if anchor_xyz.ndim != 2 or anchor_xyz.shape[1] != 3:
        raise ValueError("Stage-A landmark geometry must have shape [N, 3]")
    if anchor_features.ndim != 2:
        raise ValueError("Stage-A landmark descriptors must have shape [N, D]")
    if not (anchor_ids.numel() == anchor_xyz.shape[0] == anchor_features.shape[0]):
        raise ValueError("Stage-A anchor rows do not align")
    if anchor_ids.unique().numel() != anchor_ids.numel():
        raise ValueError("Stage-A landmark IDs must be unique")
    if not torch.isfinite(anchor_xyz).all() or not torch.isfinite(anchor_features).all():
        raise ValueError("Stage-A map contains non-finite geometry or descriptors")

    output.mkdir(parents=True, exist_ok=True)
    map_path = output / "a0_anchor_map.pt"
    metric_path = output / "a0_identity_metric.pt"
    torch.save(
        {
            "schema": "lafgs_materialized_anchor_map",
            "version": 1,
            "anchor_ids": anchor_ids,
            "anchor_xyz": anchor_xyz,
            "anchor_features": anchor_features,
            "provenance": {
                "variant": "A0_bootstrap",
                "stage_state": str(stage_state),
            },
        },
        map_path,
    )

    reconstruction = load_mainline_config(config).values["reconstruction"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        metric = SharedLowRankMetric(
            descriptor_dim=int(anchor_features.shape[1]),
            rank=int(reconstruction["metric_rank"]),
            max_residual_norm=float(reconstruction["metric_residual"]),
        )
    if torch.count_nonzero(metric.up.weight).item() != 0:
        raise RuntimeError("A0 metric initialization is not identity")
    torch.save(
        {
            "schema": "lafgs_shared_metric_state",
            "version": 1,
            "landmark_indices": anchor_ids,
            "metric_config": metric.export_config(),
            "metric_state_dict": metric.state_dict(),
            "map_path": str(map_path),
            "step": 0,
            "variant": "A0_bootstrap_identity_metric",
        },
        metric_path,
    )
    return map_path, metric_path
