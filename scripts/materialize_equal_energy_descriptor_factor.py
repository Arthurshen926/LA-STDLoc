#!/usr/bin/env python3
"""Materialize the fixed mapping-only 320D SP-metric + XFeat factor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from map_learning.equal_energy_descriptor_factor import (
    materialize_equal_energy_descriptor_factor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-map",
        "source-metric",
        "source-query-cache",
        "refreshed-query-cache",
        "mechanism-report",
        "mechanism-gate",
        "deployment-extension",
        "teacher",
        "calibration",
        "probe",
        "xfeat-weights",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    kwargs = {}
    for name in (
        "source_map",
        "source_metric",
        "source_query_cache",
        "refreshed_query_cache",
        "mechanism_report",
        "mechanism_gate",
        "deployment_extension",
        "teacher",
        "calibration",
        "probe",
        "xfeat_weights",
    ):
        kwargs[f"{name}_path"] = getattr(args, name)
        kwargs[f"{name}_sha256"] = getattr(args, f"expected_{name}_sha256")
    result = materialize_equal_energy_descriptor_factor(
        **kwargs, output_dir=args.output
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
