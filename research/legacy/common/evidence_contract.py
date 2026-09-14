#!/usr/bin/env python3
"""Build or verify the logical LaFGS LocalizationEvidenceGraph contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.artifacts import (
    build_evidence_graph_contract,
    verify_evidence_graph_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--query-cache")
    parser.add_argument("--track-payload")
    parser.add_argument("--primitive-prior")
    parser.add_argument("--anchor-map")
    parser.add_argument("--function-graph")
    parser.add_argument("--raster-provenance")
    parser.add_argument("--positive-teacher")
    args = parser.parse_args()
    output = Path(args.output)
    if args.verify:
        verify_evidence_graph_contract(json.loads(output.read_text()))
        print(json.dumps({"verified": str(output.resolve())}, indent=2))
        return
    required = (
        args.query_cache,
        args.track_payload,
        args.primitive_prior,
        args.anchor_map,
        args.function_graph,
        args.raster_provenance,
        args.positive_teacher,
    )
    if any(value is None for value in required):
        parser.error("all evidence artifact paths are required when building")
    payload = build_evidence_graph_contract(
        query_cache_path=args.query_cache,
        track_payload_path=args.track_payload,
        primitive_prior_path=args.primitive_prior,
        anchor_map_path=args.anchor_map,
        function_graph_path=args.function_graph,
        raster_provenance_path=args.raster_provenance,
        positive_teacher_path=args.positive_teacher,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
