#!/usr/bin/env python3
import argparse
import json

from la_artifacts.nerfbaselines_visuals import build_predictions_grid


def main():
    parser = argparse.ArgumentParser(description="Create a GT/render grid from NerfBaselines predictions tarball.")
    parser.add_argument("--predictions", required=True, help="Path to predictions-*.tar.gz")
    parser.add_argument("--output", required=True, help="Output PNG path")
    parser.add_argument("--sample_count", type=int, default=24)
    parser.add_argument("--columns", type=int, default=4)
    args = parser.parse_args()

    summary = build_predictions_grid(
        args.predictions,
        args.output,
        sample_count=args.sample_count,
        columns=args.columns,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
