#!/usr/bin/env python3
"""Compare K_mapping Track funnels without reading test queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evidence.mapping_density_factor import (
    compare_density_arms,
    summarize_track_payload,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-payload", type=Path, required=True)
    parser.add_argument("--high-payload", type=Path, required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = json.loads(args.scene_calibration.read_text())
    control = summarize_track_payload(
        torch.load(args.control_payload, map_location="cpu", weights_only=False),
        calibration,
    )
    high = summarize_track_payload(
        torch.load(args.high_payload, map_location="cpu", weights_only=False),
        calibration,
    )
    report = {
        "schema": "lafgs_mapping_density_track_funnel_factor",
        "version": 1,
        "uses_test_queries": False,
        "factor_axis": "k_mapping",
        "immutable_factors": {
            "nms_radius": 4,
            "pair_policy": "nearest_6_frozen_compatibility",
            "frontend": "frozen_native_superpoint",
            "seed": 2026,
            "selector_changed": False,
            "descriptor_policy_changed": False,
        },
        "control_k1024": control,
        "high_k2048": high,
        "comparison": compare_density_arms(control, high),
        "sources": {
            "control_payload": str(args.control_payload.resolve()),
            "high_payload": str(args.high_payload.resolve()),
            "scene_calibration": str(args.scene_calibration.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
