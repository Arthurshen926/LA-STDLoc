#!/usr/bin/env python3
"""Verify that a compact Map, graph, teacher, and metric share one registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.artifact_lineage import audit_compact_artifact_lineage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--function-graph", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_compact_artifact_lineage(
        anchor_map=args.map,
        function_graph=args.function_graph,
        complete_positive_teacher=args.complete_positive_teacher,
        metric_state=args.metric_state,
    )
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
