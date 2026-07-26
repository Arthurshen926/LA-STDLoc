#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate a frozen independent triangulation teacher against "
            "a different localization-center state"
        )
    )
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    statistics_path = Path(args.statistics).expanduser().resolve()
    state_path = Path(args.state).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    payload = torch.load(
        statistics_path, map_location="cpu", weights_only=False
    )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    statistics_indices = torch.as_tensor(
        payload["landmark_indices"], dtype=torch.long
    ).reshape(-1)
    state_indices = torch.as_tensor(
        state["landmark_indices"], dtype=torch.long
    ).reshape(-1)
    if not torch.equal(statistics_indices, state_indices):
        raise ValueError("Statistics and state landmark IDs are not aligned")
    geometry = dict(payload["geometry_evidence"])
    triangulated_xyz = torch.as_tensor(
        geometry["triangulated_xyz"], dtype=torch.float32
    )
    triangulated = torch.as_tensor(
        geometry["triangulated"], dtype=torch.bool
    ).reshape(-1)
    current_xyz = torch.as_tensor(
        state["landmark_xyz"], dtype=torch.float32
    )
    if triangulated_xyz.shape != current_xyz.shape:
        raise ValueError("Triangulation and state geometry shapes differ")
    offset = torch.full(
        (current_xyz.shape[0],), float("inf"), dtype=torch.float32
    )
    offset[triangulated] = torch.linalg.norm(
        triangulated_xyz[triangulated] - current_xyz[triangulated], dim=1
    )
    geometry["triangulation_current_center_offset_m"] = offset
    payload = dict(payload)
    payload["version"] = max(int(payload.get("version", 0)), 3)
    payload["geometry_evidence"] = geometry
    diagnostics = dict(payload.get("diagnostics", {}))
    diagnostics.update(
        {
            "geometry_teacher_rebound_to_state": str(state_path),
            "geometry_teacher_source_statistics": str(statistics_path),
        }
    )
    payload["diagnostics"] = diagnostics
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(
        json.dumps(
            {
                "source_statistics": str(statistics_path),
                "state": str(state_path),
                "output": str(output_path),
                "landmark_count": int(current_xyz.shape[0]),
                "triangulated_count": int(triangulated.sum().item()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
