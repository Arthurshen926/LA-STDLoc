#!/usr/bin/env python3
"""Classify selected harmful rows using complete active/canonical projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from localization_training.harmful_outcome_triage import (
    HarmfulTriageConfig,
    triage_harmful_outcomes,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-map", required=True)
    parser.add_argument("--canonical-map", required=True)
    parser.add_argument("--selected-outcomes", required=True)
    parser.add_argument("--dynamic-outcomes", required=True)
    parser.add_argument("--active-positive-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--raster-provenance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--completed-positive-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--strict-radius-px", type=float, default=2.0)
    parser.add_argument("--ambiguous-radius-px", type=float, default=6.0)
    parser.add_argument("--depth-abs-tolerance-m", type=float, default=0.05)
    parser.add_argument("--depth-rel-tolerance", type=float, default=0.02)
    parser.add_argument("--alpha-minimum", type=float, default=0.01)
    parser.add_argument("--maximum-candidates-per-row", type=int, default=8)
    parser.add_argument("--minimum-track-views", type=int, default=3)
    parser.add_argument("--minimum-track-view-bins", type=int, default=2)
    parser.add_argument(
        "--maximum-track-reprojection-p90-px", type=float, default=4.0
    )
    parser.add_argument(
        "--minimum-contribution-mass", type=float, default=0.02
    )
    parser.add_argument(
        "--maximum-depth-std-abs-m", type=float, default=0.05
    )
    parser.add_argument(
        "--maximum-depth-std-relative", type=float, default=0.02
    )
    parser.add_argument(
        "--geometry-xyz-threshold-m", type=float, default=0.02
    )
    parser.add_argument(
        "--geometry-reprojection-improvement-px",
        type=float,
        default=1.0,
    )
    args = parser.parse_args()
    paths = {
        "active_map": args.active_map,
        "canonical_map": args.canonical_map,
        "selected_outcomes": args.selected_outcomes,
        "dynamic_outcomes": args.dynamic_outcomes,
        "active_positive_teacher": args.active_positive_teacher,
        "query_cache": args.query_cache,
        "track_payload": args.track_payload,
        "raster_provenance": args.raster_provenance,
    }
    payloads = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    config = HarmfulTriageConfig(
        strict_radius_px=args.strict_radius_px,
        ambiguous_radius_px=args.ambiguous_radius_px,
        depth_abs_tolerance_m=args.depth_abs_tolerance_m,
        depth_rel_tolerance=args.depth_rel_tolerance,
        alpha_minimum=args.alpha_minimum,
        maximum_candidates_per_row=args.maximum_candidates_per_row,
        minimum_track_views=args.minimum_track_views,
        minimum_track_view_bins=args.minimum_track_view_bins,
        maximum_track_reprojection_p90_px=(
            args.maximum_track_reprojection_p90_px
        ),
        minimum_contribution_mass=args.minimum_contribution_mass,
        maximum_depth_std_abs_m=args.maximum_depth_std_abs_m,
        maximum_depth_std_relative=args.maximum_depth_std_relative,
        geometry_xyz_threshold_m=args.geometry_xyz_threshold_m,
        geometry_reprojection_improvement_px=(
            args.geometry_reprojection_improvement_px
        ),
    )

    def progress(completed: int, total: int, summary: dict) -> None:
        if completed % 25 == 0 or completed == total:
            print(
                json.dumps(
                    {
                        "completed_queries": completed,
                        "total_queries": total,
                        **summary,
                    }
                ),
                flush=True,
            )

    triage, completed = triage_harmful_outcomes(
        active_map=payloads["active_map"],
        canonical_map=payloads["canonical_map"],
        selected_outcomes=payloads["selected_outcomes"],
        dynamic_outcomes=payloads["dynamic_outcomes"],
        active_positive_teacher=payloads["active_positive_teacher"],
        query_cache=payloads["query_cache"],
        track_payload=payloads["track_payload"],
        raster_provenance=payloads["raster_provenance"],
        config=config,
        device=torch.device(args.device),
        progress=progress,
    )
    provenance = {
        name: {
            "path": str(Path(path).resolve()),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }
    triage["provenance"] = provenance
    completed["provenance"] = {
        **dict(completed.get("provenance", {})),
        "harmful_triage": provenance,
    }
    output = Path(args.output).resolve()
    completed_output = Path(args.completed_positive_output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed_output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(output, triage)
    _atomic_json(output.with_suffix(".json"), {
        "schema": triage["schema"],
        "version": triage["version"],
        "summary": triage["summary"],
        "config": triage["config"],
        "provenance": provenance,
    })
    _atomic_torch(completed_output, completed)
    print(json.dumps(triage["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
