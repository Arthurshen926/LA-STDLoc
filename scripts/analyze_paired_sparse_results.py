#!/usr/bin/env python
import argparse
import json
import os

from localization_training.eval_analysis import paired_sparse_summary, threshold_curve


def _load_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of per-query results in {path}.")
    return data


def main():
    parser = argparse.ArgumentParser(description="Paired sparse-localization analysis for shared query sets.")
    parser.add_argument("--baseline_results", required=True)
    parser.add_argument("--la_results", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[2, 4, 6, 8, 10, 12, 16])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    baseline = _load_results(args.baseline_results)
    la = _load_results(args.la_results)
    result = {
        "baseline_results": args.baseline_results,
        "la_results": args.la_results,
        "paired": paired_sparse_summary(
            baseline,
            la,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "baseline_curve": threshold_curve(baseline, args.thresholds),
        "la_curve": threshold_curve(la, args.thresholds),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
            f.write("\n")
    print(text)


if __name__ == "__main__":
    main()
