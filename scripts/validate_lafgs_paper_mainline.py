#!/usr/bin/env python3
"""Validate and materialize the frozen LaFGS release protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lafgs.protocol import load_mainline_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/lafgs_paper_mainline.yaml"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    manifest = load_mainline_protocol(args.config).manifest()
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
