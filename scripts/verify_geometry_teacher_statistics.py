#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch


TRACK_MODES = {"track_first", "track_first_provenance"}
REQUIRED_TRACK_DIAGNOSTICS = (
    "track_count",
    "geometry_teacher_triangulated_track_count",
    "geometry_teacher_high_confidence_track_count",
    "geometry_teacher_assigned_landmark_count",
)


def verify_statistics(path, expected_identity):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    diagnostics = dict(payload.get("diagnostics", {}))
    actual_identity = str(
        diagnostics.get("geometry_teacher_identity_mode", "")
    )
    if actual_identity != str(expected_identity):
        raise ValueError(
            "Geometry teacher identity mismatch: "
            f"expected {expected_identity!r}, found {actual_identity!r}"
        )
    summary = {"geometry_teacher_identity_mode": actual_identity}
    if actual_identity in TRACK_MODES:
        missing = [
            key for key in REQUIRED_TRACK_DIAGNOSTICS
            if key not in diagnostics
        ]
        if missing:
            raise ValueError(
                "Track-first statistics lack diagnostics: "
                + ", ".join(missing)
            )
        for key in REQUIRED_TRACK_DIAGNOSTICS:
            value = int(diagnostics[key])
            if value <= 0:
                raise ValueError(
                    f"Track-first diagnostic {key} must be positive"
                )
            summary[key] = value
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Fail closed on mislabeled geometry-teacher statistics"
    )
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--expected_identity", required=True)
    args = parser.parse_args()
    path = Path(args.statistics).expanduser().resolve()
    summary = verify_statistics(path, args.expected_identity)
    summary["statistics"] = str(path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
