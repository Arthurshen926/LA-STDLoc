#!/usr/bin/env python3
"""Rebuild a Track-First payload with cross-fitted surface support."""

import argparse
import json
from pathlib import Path

import torch

from evidence.surface_tracks import rebuild_surface_supported_track_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--scene-calibration", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    query = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    calibration = json.loads(Path(args.scene_calibration).read_text())
    revised, report = rebuild_surface_supported_track_payload(
        payload=payload,
        query_payload=query,
        parameters=calibration["parameters"],
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(revised, output)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
