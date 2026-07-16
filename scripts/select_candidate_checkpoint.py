#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path
import shlex
import statistics


def linear_quantile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def checkpoint_metrics(results_path):
    records = json.loads(Path(results_path).read_text())
    translation_errors = [
        float(record["sparse_TE"])
        for record in records
        if record.get("sparse_TE") is not None
        and math.isfinite(float(record["sparse_TE"]))
    ]
    if not translation_errors:
        raise ValueError(f"no finite sparse_TE values in {results_path}")
    return {
        "query_count": len(translation_errors),
        "median_te_cm": statistics.median(translation_errors),
        "mean_te_cm": statistics.fmean(translation_errors),
        "p90_te_cm": linear_quantile(translation_errors, 0.90),
        "p95_te_cm": linear_quantile(translation_errors, 0.95),
        "max_te_cm": max(translation_errors),
    }


def select_checkpoint(evaluation_root, candidate_tag, iterations):
    checkpoints = []
    for iteration in iterations:
        result_dir = (
            Path(evaluation_root)
            / f"{candidate_tag}_{int(iteration)}_calibrated_validation"
        )
        results_path = result_dir / "results.json"
        if not results_path.is_file():
            raise FileNotFoundError(results_path)
        metrics = checkpoint_metrics(results_path)
        checkpoints.append(
            {
                "iteration": int(iteration),
                "result_dir": str(result_dir.resolve()),
                **metrics,
            }
        )

    selected = min(
        checkpoints,
        key=lambda item: (
            item["median_te_cm"],
            item["mean_te_cm"],
            item["p90_te_cm"],
            item["iteration"],
        ),
    )
    return {
        "selection_protocol": {
            "subset": "candidate_validation",
            "calibrated_frontend": True,
            "primary_metric": "median_te_cm",
            "tie_breakers": ["mean_te_cm", "p90_te_cm", "iteration"],
            "test_metrics_used": False,
        },
        "selected_iteration": selected["iteration"],
        "selected_metrics": selected,
        "checkpoints": checkpoints,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_root", required=True, type=Path)
    parser.add_argument("--candidate_tag", default="candidate_f0")
    parser.add_argument("--iterations", required=True, nargs="+", type=int)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    report = select_checkpoint(
        args.evaluation_root,
        args.candidate_tag,
        args.iterations,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if args.shell:
        print(
            "SELECTED_CANDIDATE_ITERATION="
            + shlex.quote(str(report["selected_iteration"]))
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
