#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from la_diagnostics.qualitative import BatchInputs, generate_qualitative_report


def _metadata(values):
    result = {}
    for value in values or []:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"Metadata must be key=value, got {value!r}.")
        key, raw = value.split("=", 1)
        result[key] = raw
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate a qualitative diagnostics report for one LA-STDLoc batch.")
    parser.add_argument("--batch_name", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--current_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_results", default=None)
    parser.add_argument("--artifact_audit_csv", default=None)
    parser.add_argument("--region_manifest_csv", default=None)
    parser.add_argument("--region_weight_root", default=None)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--registry_path", default=None)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--notes", default="")
    parser.add_argument("--metadata", nargs="*", default=[])
    args = parser.parse_args()

    summary = generate_qualitative_report(
        BatchInputs(
            batch_name=args.batch_name,
            scene=args.scene,
            current_results=Path(args.current_results),
            baseline_results=Path(args.baseline_results) if args.baseline_results else None,
            artifact_audit_csv=Path(args.artifact_audit_csv) if args.artifact_audit_csv else None,
            region_manifest_csv=Path(args.region_manifest_csv) if args.region_manifest_csv else None,
            region_weight_root=Path(args.region_weight_root) if args.region_weight_root else None,
            image_root=Path(args.image_root) if args.image_root else None,
            output_dir=Path(args.output_dir),
            registry_path=Path(args.registry_path) if args.registry_path else None,
            top_k=args.top_k,
            notes=args.notes,
            metadata=_metadata(args.metadata),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
