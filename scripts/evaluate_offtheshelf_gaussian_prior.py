#!/usr/bin/env python3
"""Evaluate or export held-out views from an RGB-only Gaussian prior."""

from __future__ import annotations

import argparse
import json

from lafgs.priors.quality import evaluate_prior_quality, summarize_quality

# Compatibility for existing analysis notebooks; new code should import the
# package function directly.
_summary = summarize_quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--gaussian-type", choices=("3dgs", "2dgs"), required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument(
        "--include-view",
        action="append",
        default=None,
        help="Evaluate only this image name; repeat for multiple views.",
    )
    parser.add_argument("--save-render-dir", default=None)
    parser.add_argument("--save-ground-truth", action="store_true")
    args = parser.parse_args()
    report = evaluate_prior_quality(
        args,
        include_views=args.include_view,
        save_render_dir=args.save_render_dir,
        save_ground_truth=args.save_ground_truth,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
