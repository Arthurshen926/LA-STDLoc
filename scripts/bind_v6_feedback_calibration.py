#!/usr/bin/env python3
"""Bind one mapping-only calibration to the exact V6 query registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import torch

from common.hashing import sha256_file
from common.v6_contracts import (
    ordered_query_registry_sha256,
    require_mapping_only,
    validate_ordered_query_registry,
)
from common.v6_pipeline_contract import (
    FEEDBACK_CALIBRATION_BINDING_SCHEMA,
    validate_v6_feedback_scene_calibration,
)


_MAP_SCHEMA = "lafgs_materialized_anchor_map"
def _is_sha256(value: object) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _require_file_sha(path: Path, expected: str, *, label: str) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not _is_sha256(expected):
        raise ValueError(f"{label} expected SHA256 is invalid")
    actual = sha256_file(path)
    if actual != str(expected).lower():
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def build_binding(args: argparse.Namespace) -> dict:
    map_path = args.map.resolve()
    calibration_path = args.scene_calibration.resolve()
    map_sha256 = _require_file_sha(
        map_path,
        args.expected_map_sha256,
        label="V6 map",
    )
    calibration_sha256 = _require_file_sha(
        calibration_path,
        args.expected_scene_calibration_sha256,
        label="scene calibration",
    )
    observation_cache_sha256 = str(args.observation_cache_sha256).lower()
    if not _is_sha256(observation_cache_sha256):
        raise ValueError("observation cache SHA256 is invalid")

    state = torch.load(map_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("schema") != _MAP_SCHEMA:
        raise ValueError("input map is not a materialized V6 Anchor map")
    provenance = state.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("V6 map provenance is missing")
    require_mapping_only(provenance, label="V6 calibration-binding map")
    if provenance.get("mapping_source") != "gaussian_render_valid_projective_v6":
        raise ValueError("V6 map is not from the Gaussian-render projective path")
    if provenance.get("v6_observation_cache_sha256") != observation_cache_sha256:
        raise ValueError("V6 map and observation cache SHA lineage differ")
    names = validate_ordered_query_registry(state.get("v6_mapping_query_names", ()))

    calibration = json.loads(calibration_path.read_text())
    if not isinstance(calibration, dict):
        raise ValueError("scene calibration is not a JSON object")
    validate_v6_feedback_scene_calibration(calibration, query_count=len(names))
    return {
        "schema": FEEDBACK_CALIBRATION_BINDING_SCHEMA,
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "map_sha256": map_sha256,
        "observation_cache_sha256": observation_cache_sha256,
        "calibration_sha256": calibration_sha256,
        "ordered_query_registry_sha256": ordered_query_registry_sha256(names),
        "query_count": len(names),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    parser.add_argument("--observation-cache-sha256", required=True)
    parser.add_argument("--scene-calibration", type=Path, required=True)
    parser.add_argument("--expected-scene-calibration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = build_binding(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()) != payload:
            raise RuntimeError("temporary calibration binding did not reload exactly")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
