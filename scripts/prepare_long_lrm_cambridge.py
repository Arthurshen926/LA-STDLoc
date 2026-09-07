#!/usr/bin/env python3
"""Prepare 32 pose-diverse mapping views for the released Long-LRM model."""

from __future__ import annotations

import argparse
import json

from priors.long_lrm_adapter import prepare_cambridge_mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--images", default="processed")
    args = parser.parse_args()
    result = prepare_cambridge_mapping(
        args.dataset,
        args.output,
        images=args.images,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
