#!/usr/bin/env python3
"""Export Long-LRM Gaussians into the Cambridge mapping coordinate frame."""

from __future__ import annotations

import argparse
import json

from priors.long_lrm_adapter import export_long_lrm_world_prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ply", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--output-ply", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()
    result = export_long_lrm_world_prior(
        args.source_ply,
        args.prepared_manifest,
        args.output_ply,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
