#!/usr/bin/env python3
"""Export i3DGS hierarchy leaves into an AnyGSLoc-compatible world PLY."""

from __future__ import annotations

import argparse
import json

from priors.i3dgs_adapter import export_i3dgs_world_prior


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy-ply", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--output-ply", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()
    result = export_i3dgs_world_prior(
        args.hierarchy_ply,
        args.prepared_manifest,
        args.output_ply,
        output_manifest=args.output_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
