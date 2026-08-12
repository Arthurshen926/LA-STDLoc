#!/usr/bin/env python3
"""Audit exact control parity for Track payload and provenance assignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.hashing import sha256_file


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _tensor_parity(left, right, *, atol: float = 1e-7) -> dict:
    left = torch.as_tensor(left)
    right = torch.as_tensor(right)
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "equal": False,
            "shape_equal": left.shape == right.shape,
            "dtype_equal": left.dtype == right.dtype,
        }
    if left.dtype.is_floating_point:
        finite = torch.isfinite(left) & torch.isfinite(right)
        same_nonfinite = (
            torch.equal(torch.isnan(left), torch.isnan(right))
            and torch.equal(torch.isposinf(left), torch.isposinf(right))
            and torch.equal(torch.isneginf(left), torch.isneginf(right))
        )
        maximum = float(
            (left[finite] - right[finite]).abs().max() if bool(finite.any()) else 0.0
        )
        equal = same_nonfinite and maximum <= float(atol)
        return {
            "equal": bool(equal),
            "shape_equal": True,
            "dtype_equal": True,
            "finite_max_abs_difference": maximum,
            "absolute_tolerance": float(atol),
        }
    return {
        "equal": torch.equal(left, right),
        "shape_equal": True,
        "dtype_equal": True,
        "different_rows": int((left != right).sum()),
    }


def _table_parity(left: dict, right: dict, *, atol: float) -> dict:
    left_keys, right_keys = set(left), set(right)
    common = sorted(left_keys & right_keys)
    fields = {key: _tensor_parity(left[key], right[key], atol=atol) for key in common}
    return {
        "left_only": sorted(left_keys - right_keys),
        "right_only": sorted(right_keys - left_keys),
        "fields": fields,
        "equal": not (left_keys ^ right_keys)
        and all(value["equal"] for value in fields.values()),
    }


def audit_payload_parity(reference: dict, replay: dict, *, float_atol: float) -> dict:
    top_level_contract = {
        "reference_schema": reference.get("schema") == "lafgs_track_first_payload",
        "replay_schema": replay.get("schema") == "lafgs_track_first_payload",
        "reference_version": reference.get("version") == 1,
        "replay_version": replay.get("version") == 1,
    }
    required_assignment_fields = {
        "track_landmark_index",
        "track_assignment_cost",
        "landmark_best_track_index",
        "track_landmark_offsets",
        "track_landmark_indices",
        "track_landmark_costs",
    }
    assignment = _table_parity(
        reference["assignment"], replay["assignment"], atol=float_atol
    )
    assignment["required_fields_present"] = required_assignment_fields <= set(
        reference["assignment"]
    ) and required_assignment_fields <= set(replay["assignment"])
    tracks = _table_parity(reference["tracks"], replay["tracks"], atol=0.0)
    reference_geometry = dict(reference["track_geometry"])
    replay_required_geometry = {
        key: replay["track_geometry"][key]
        for key in reference_geometry
        if key in replay["track_geometry"]
    }
    geometry = _table_parity(
        reference_geometry, replay_required_geometry, atol=float_atol
    )
    geometry["replay_extra_fields"] = sorted(
        set(replay["track_geometry"]) - set(reference_geometry)
    )
    provenance_diagnostic_names = sorted(
        key
        for key in reference["diagnostics"]
        if key.startswith("geometry_teacher_provenance_")
    )
    diagnostic_parity = {
        key: reference["diagnostics"].get(key) == replay["diagnostics"].get(key)
        for key in provenance_diagnostic_names
    }
    query_support_diagnostic_names = sorted(
        key
        for key in reference["diagnostics"]
        if key.startswith("geometry_teacher_query_support_")
    )
    query_support_diagnostic_parity = {
        key: reference["diagnostics"].get(key) == replay["diagnostics"].get(key)
        for key in query_support_diagnostic_names
    }
    query_registry = {
        "query_names_equal": reference["query_names"] == replay["query_names"],
        "query_bins_equal": torch.equal(
            torch.as_tensor(reference["query_bins"]),
            torch.as_tensor(replay["query_bins"]),
        ),
        "train_camera_names_sha256_equal": (
            reference["train_camera_names_sha256"]
            == replay["train_camera_names_sha256"]
        ),
        "landmark_indices_equal": torch.equal(
            torch.as_tensor(reference["landmark_indices"]),
            torch.as_tensor(replay["landmark_indices"]),
        ),
    }
    valid = (
        all(top_level_contract.values())
        and assignment["equal"]
        and assignment["required_fields_present"]
        and tracks["equal"]
        and geometry["equal"]
        and all(diagnostic_parity.values())
        and all(query_support_diagnostic_parity.values())
        and all(query_registry.values())
    )
    return {
        "schema": "lafgs_track_payload_control_parity",
        "version": 1,
        "uses_test_queries": False,
        "valid": valid,
        "top_level_contract": top_level_contract,
        "assignment": assignment,
        "tracks": tracks,
        "geometry_common_fields": geometry,
        "provenance_diagnostics": diagnostic_parity,
        "query_support_diagnostics": query_support_diagnostic_parity,
        "query_registry": query_registry,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--expected-replay-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--float-atol", type=float, default=1e-7)
    args = parser.parse_args()
    reference_sha256 = sha256_file(args.reference)
    replay_sha256 = sha256_file(args.replay)
    if reference_sha256 != args.expected_reference_sha256.lower():
        raise ValueError("Reference Track payload SHA-256 differs from expected")
    if replay_sha256 != args.expected_replay_sha256.lower():
        raise ValueError("Replay Track payload SHA-256 differs from expected")
    reference, replay = _load(args.reference), _load(args.replay)
    report = audit_payload_parity(reference, replay, float_atol=float(args.float_atol))
    report.update(
        {
            "reference": {
                "path": str(args.reference.resolve()),
                "sha256": reference_sha256,
            },
            "replay": {
                "path": str(args.replay.resolve()),
                "sha256": replay_sha256,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
