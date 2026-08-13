#!/usr/bin/env python3
"""Materialize a hash-locked neutral Anchor Registry sibling artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from common.anchor_registry_artifact import materialize_anchor_registry


OPTIONAL_PARENTS = (
    ("compact_map", "compact-map"),
    ("positive_teacher", "teacher"),
    ("track_payload", "track-payload"),
    ("query_cache", "query-cache"),
    ("raster_provenance", "raster-provenance"),
    ("selection_provenance", "selection-provenance"),
    ("scene_calibration", "scene-calibration"),
    ("metric_state", "metric-state"),
    ("config", "config"),
    ("gaussian_ply", "gaussian-ply"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--expected-map-sha256", required=True)
    for _name, option in OPTIONAL_PARENTS:
        parser.add_argument(f"--{option}", type=Path)
        parser.add_argument(f"--expected-{option}-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path)
    parser.add_argument(
        "--require-pipeline-parents",
        action="store_true",
        help="Require the complete new-pipeline parent set and exact selection.",
    )
    parser.add_argument(
        "--allow-legacy-unresolved-audit",
        action="store_true",
        help=(
            "Explicitly permit an audit-only Registry whose legacy selector "
            "reason cannot be reconstructed. Never pipeline eligible."
        ),
    )
    return parser


def _parents(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    parents = {"trained_map": (args.map, args.expected_map_sha256)}
    for name, option in OPTIONAL_PARENTS:
        attribute = option.replace("-", "_")
        path = getattr(args, attribute)
        expected = getattr(args, f"expected_{attribute}_sha256")
        if (path is None) != (expected is None):
            parser.error(
                f"--{option} and --expected-{option}-sha256 must be supplied together"
            )
        if path is not None:
            parents[name] = (path, expected)
    return parents


def run(argv: Sequence[str] | None = None) -> dict:
    parser = build_parser()
    args = parser.parse_args(argv)
    return materialize_anchor_registry(
        parents=_parents(args, parser),
        output=args.output,
        contract_output=args.contract_output,
        require_pipeline_parents=bool(args.require_pipeline_parents),
        allow_legacy_unresolved_audit=bool(args.allow_legacy_unresolved_audit),
    )


def main(argv: Sequence[str] | None = None) -> None:
    result = run(argv)
    print(
        json.dumps(
            {
                "registry": str(result["registry"]),
                "registry_sha256": result["registry_sha256"],
                "contract": str(result["contract"]),
                "contract_sha256": result["contract_sha256"],
                "report": result["report"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
