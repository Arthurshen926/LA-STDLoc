#!/usr/bin/env python3
"""Fail-closed exact merge of mapping-cache or rendered-fullmap query shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.mapping_shards import merge_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-summary", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge_reports(args.shard_summary, args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
