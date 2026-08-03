#!/usr/bin/env python3
"""Render a qualitative overview of a compact localization map."""

from __future__ import annotations

import argparse
from pathlib import Path

from visualization.maps import render_localization_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-points", type=int, default=20000)
    args = parser.parse_args()
    render_localization_map(
        args.map,
        args.output,
        maximum_points=args.maximum_points,
    )


if __name__ == "__main__":
    main()
