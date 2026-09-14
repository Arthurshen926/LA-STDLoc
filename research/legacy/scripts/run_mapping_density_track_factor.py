#!/usr/bin/env python3
"""Replay the frozen nearest-pair Track funnel for one K_mapping arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def _replace_value(command: list[str], flag: str, value: object) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index : index + 2]
    command.extend((flag, str(value)))


def density_track_command(
    manifest: dict,
    *,
    query_cache: Path,
    output_dir: Path,
    mapping_keypoints: int,
    nms_radius: int,
    python: str,
) -> list[str]:
    """Change only cache/output/K/NMS in a frozen Track invocation."""
    command = [str(value) for value in manifest["command"]]
    if "--save_track_micro_anchor_payload" not in command:
        raise ValueError("source manifest is not a Track-payload invocation")
    if "--steps" not in command or command[command.index("--steps") + 1] != "0":
        raise ValueError("density funnel requires a zero-step frozen Track replay")
    forbidden = tuple(
        flag
        for flag in command
        if "track_pair_policy" in flag or "parallax_stratified" in flag
    )
    if forbidden:
        raise ValueError(f"density-only replay rejects pair-policy flags: {forbidden}")
    command[0] = str(python)
    command[1:2] = ["-m", "map_learning.bootstrap"]
    overrides = {
        "--query_cache_path": query_cache.resolve(),
        "--query_cache_policy": "readonly",
        "--output_dir": output_dir.resolve(),
        "--native_keypoint_count": int(mapping_keypoints),
        "--native_nms_radius": int(nms_radius),
        "--max_observations": int(mapping_keypoints),
        "--validation_observations": int(mapping_keypoints),
    }
    for flag, value in overrides.items():
        _replace_value(command, flag, value)
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mapping-keypoints", type=int, choices=(1024, 2048), required=True
    )
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    command = density_track_command(
        manifest,
        query_cache=args.query_cache,
        output_dir=args.output_dir,
        mapping_keypoints=args.mapping_keypoints,
        nms_radius=args.nms_radius,
        python=args.python,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "lafgs_mapping_density_track_factor_invocation",
        "version": 1,
        "uses_test_queries": False,
        "factor_axis": "k_mapping",
        "mapping_keypoints": int(args.mapping_keypoints),
        "nms_radius": int(args.nms_radius),
        "pair_policy": "nearest_6_frozen_compatibility",
        "descriptor_policy_changed": False,
        "selector_policy_changed": False,
        "source_manifest": str(args.manifest.resolve()),
        "query_cache": str(args.query_cache.resolve()),
        "command": command,
    }
    (args.output_dir / "density_factor_invocation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
