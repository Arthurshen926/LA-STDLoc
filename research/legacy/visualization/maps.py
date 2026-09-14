"""Compact localization-map visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def render_localization_map(
    map_path: str | Path,
    output: str | Path,
    *,
    maximum_points: int = 20000,
) -> Path:
    """Render a deterministic 3D overview of active localization anchors."""
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    anchor_type = torch.as_tensor(
        state.get("anchor_type", torch.zeros(len(xyz)))
    ).long()
    if len(xyz) > int(maximum_points):
        rows = torch.linspace(0, len(xyz) - 1, int(maximum_points)).long()
        xyz, anchor_type = xyz[rows], anchor_type[rows]
    figure = plt.figure(figsize=(10, 8), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2], c=anchor_type, s=2, cmap="tab10"
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(f"LaFGS localization map: {len(xyz):,} displayed anchors")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output
