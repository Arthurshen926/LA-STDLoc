#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from localization_training.eval_analysis import solver_threshold_sweep_summary


def _load_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of per-query results in {path}.")
    return data


def _parse_run_specs(specs):
    runs = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Run spec must be threshold:path, got {spec!r}.")
        threshold, path = spec.split(":", 1)
        runs[float(threshold)] = _load_results(path)
    return runs


def main():
    parser = argparse.ArgumentParser(description="Analyze actual solver-threshold sweep results.")
    parser.add_argument("--baseline_run", action="append", required=True, help="threshold:path/to/results.json")
    parser.add_argument("--la_run", action="append", required=True, help="threshold:path/to/results.json")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    summary = solver_threshold_sweep_summary(
        _parse_run_specs(args.baseline_run),
        _parse_run_specs(args.la_run),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    text = json.dumps(summary, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(text)
            f.write("\n")
    print(text)


if __name__ == "__main__":
    main()
