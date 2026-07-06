#!/usr/bin/env python
import argparse
import csv
import json
import os

from localization_training.eval_analysis import paired_sparse_stage_rows, paired_sparse_stage_summary


def _load_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of per-query results in {path}.")
    return data


def _write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "image_name",
        "sequence",
        "baseline_te",
        "candidate_te",
        "delta_te",
        "baseline_ae",
        "candidate_ae",
        "delta_ae",
        "baseline_inliers",
        "candidate_inliers",
        "delta_inliers",
        "baseline_matches",
        "candidate_matches",
        "delta_matches",
        "baseline_keypoints",
        "candidate_keypoints",
        "delta_keypoints",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Compare paired sparse localization stage results.")
    parser.add_argument("--baseline_results", required=True)
    parser.add_argument("--candidate_results", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", default="")
    parser.add_argument("--te_ok_cm", type=float, default=5.0)
    parser.add_argument("--ae_ok_deg", type=float, default=5.0)
    parser.add_argument("--inlier_drop_threshold", type=float, default=50.0)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    baseline = _load_results(args.baseline_results)
    candidate = _load_results(args.candidate_results)
    summary = paired_sparse_stage_summary(
        baseline,
        candidate,
        te_ok_cm=args.te_ok_cm,
        ae_ok_deg=args.ae_ok_deg,
        inlier_drop_threshold=args.inlier_drop_threshold,
        top_k=args.top_k,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    if args.output_csv:
        _write_csv(args.output_csv, paired_sparse_stage_rows(baseline, candidate))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
