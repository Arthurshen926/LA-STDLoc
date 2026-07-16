#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import shlex

import numpy as np


def finite_values(records, key, *, scale=1.0):
    values = []
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        value = float(value) * float(scale)
        if np.isfinite(value) and value >= 0.0:
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def calibrate(records, fallback_translation_scale_m, fallback_inlier_sigma_px):
    translation_error_m = finite_values(records, "sparse_TE", scale=0.01)
    if translation_error_m.size:
        p50 = float(np.quantile(translation_error_m, 0.50))
        p75 = float(np.quantile(translation_error_m, 0.75))
        p90 = float(np.quantile(translation_error_m, 0.90))
    else:
        p50 = p75 = p90 = float(fallback_translation_scale_m)
    fallback = max(float(fallback_translation_scale_m), 1e-6)
    translation_scale = min(max(p50, 0.5 * fallback), 4.0 * fallback)
    bias_clip = min(8.0, max(2.0, p90 / max(translation_scale, 1e-8)))

    reprojection = []
    for record in records:
        sparse = record.get("sparse") or {}
        value = sparse.get("sparse_diag_inlier_gt_reproj_px_median")
        if value is not None and np.isfinite(float(value)) and float(value) > 0.0:
            reprojection.append(float(value))
    if reprojection:
        empirical_sigma = float(np.quantile(reprojection, 0.75))
        inlier_sigma = min(
            max(empirical_sigma, 0.5 * float(fallback_inlier_sigma_px)),
            2.0 * float(fallback_inlier_sigma_px),
        )
    else:
        inlier_sigma = float(fallback_inlier_sigma_px)

    return {
        "query_count": int(translation_error_m.size),
        "baseline_te_p50_m": p50,
        "baseline_te_p75_m": p75,
        "baseline_te_p90_m": p90,
        "translation_scale_m": translation_scale,
        "bias_huber_delta": 1.0,
        "bias_clip": bias_clip,
        "inlier_sigma_px": inlier_sigma,
        "residual_clip_px": 3.0 * inlier_sigma,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json", required=True, type=Path)
    parser.add_argument("--fallback_translation_scale_m", required=True, type=float)
    parser.add_argument("--fallback_inlier_sigma_px", required=True, type=float)
    parser.add_argument("--output_json", type=Path)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()

    result = calibrate(
        json.loads(args.results_json.read_text()),
        args.fallback_translation_scale_m,
        args.fallback_inlier_sigma_px,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.shell:
        for key, value in sorted(result.items()):
            print(f"CALIBRATED_{key.upper()}={shlex.quote(str(value))}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
