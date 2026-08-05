#!/usr/bin/env python3
"""Aggregate only fully registered LaFGS benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.aggregate import (
    aggregate_registered_benchmark,
    iter_scene_names,
    latex_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = iter_scene_names(args.scenes)
    aggregate = aggregate_registered_benchmark(args.run_root, scenes)
    write_json(args.json_output, aggregate)
    rows = latex_rows(args.dataset_label, aggregate)
    if args.tex_output is None:
        print(rows, end="")
    else:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text(rows, encoding="utf-8")


if __name__ == "__main__":
    main()
