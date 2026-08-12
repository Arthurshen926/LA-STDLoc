#!/usr/bin/env python3
"""Compare paired K=1024 Track builds and apply the P7 mechanism gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _number(payload: dict, *path: str) -> float:
    value = payload
    for key in path:
        value = value[key]
    if value is None:
        raise ValueError(f"Missing comparison value: {'/'.join(path)}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    control = _read(args.control)
    variant = _read(args.variant)
    if control["pair_policy"] != "nearest":
        raise ValueError("Control must use the frozen nearest policy")
    if variant["pair_policy"] != "parallax_diverse":
        raise ValueError("Variant must use parallax_diverse")
    if control["mapping_keypoint_factor"] != variant["mapping_keypoint_factor"]:
        raise ValueError("Density factor changed between pair-policy arms")
    if control["mapping_query_count"] != variant["mapping_query_count"]:
        raise ValueError("Mapping query set changed between arms")

    metrics = {
        "pair_count": (
            _number(control, "pair", "pair_count"),
            _number(variant, "pair", "pair_count"),
        ),
        "pair_parallax_below_1deg_fraction": (
            _number(control, "pair", "mapping_point_parallax_below_1deg_fraction"),
            _number(variant, "pair", "mapping_point_parallax_below_1deg_fraction"),
        ),
        "pair_parallax_median_deg": (
            _number(control, "pair", "mapping_point_parallax_median_deg", "median"),
            _number(variant, "pair", "mapping_point_parallax_median_deg", "median"),
        ),
        "triangulated_tracks": (
            _number(control, "track", "triangulated_track_count"),
            _number(variant, "track", "triangulated_track_count"),
        ),
        "broad_eligible_tracks": (
            _number(control, "track", "broad_eligible_track_count"),
            _number(variant, "track", "broad_eligible_track_count"),
        ),
        "triangulated_covariance_p90_m2": (
            _number(
                control,
                "track",
                "triangulated_covariance_trace_m2",
                "p90",
            ),
            _number(
                variant,
                "track",
                "triangulated_covariance_trace_m2",
                "p90",
            ),
        ),
        "broad_support_query_p10": (
            _number(
                control,
                "track",
                "broad_track_support_per_mapping_query",
                "p10",
            ),
            _number(
                variant,
                "track",
                "broad_track_support_per_mapping_query",
                "p10",
            ),
        ),
        "surface_capacity_need_proxy": (
            _number(control, "track", "surface_capacity_need_proxy_at_7275"),
            _number(variant, "track", "surface_capacity_need_proxy_at_7275"),
        ),
    }
    comparisons = {
        name: {
            "control": baseline,
            "variant": revised,
            "delta": revised - baseline,
            "relative": (
                None if baseline == 0 else revised / baseline - 1.0
            ),
        }
        for name, (baseline, revised) in metrics.items()
    }
    gates = {
        "exact_global_pair_budget": metrics["pair_count"][0]
        == metrics["pair_count"][1],
        "low_parallax_fraction_reduced_by_10pp": (
            metrics["pair_parallax_below_1deg_fraction"][1]
            <= metrics["pair_parallax_below_1deg_fraction"][0] - 0.10
        ),
        "triangulated_tracks_retained_95pct": (
            metrics["triangulated_tracks"][1]
            >= 0.95 * metrics["triangulated_tracks"][0]
        ),
        "broad_tracks_retained_98pct": (
            metrics["broad_eligible_tracks"][1]
            >= 0.98 * metrics["broad_eligible_tracks"][0]
        ),
        "triangulated_covariance_p90_not_worse_5pct": (
            metrics["triangulated_covariance_p90_m2"][1]
            <= 1.05 * metrics["triangulated_covariance_p90_m2"][0]
        ),
        "broad_query_support_p10_retained_95pct": (
            metrics["broad_support_query_p10"][1]
            >= 0.95 * metrics["broad_support_query_p10"][0]
        ),
    }
    passed = all(gates.values())
    report = {
        "schema": "lafgs_pair_policy_mechanism_gate",
        "version": 1,
        "uses_test_queries": False,
        "single_factor": "camera_pair_policy",
        "mapping_keypoints": int(control["mapping_keypoint_factor"]),
        "comparisons": comparisons,
        "gates": gates,
        "mechanism_gate_passed": passed,
        "advance_to_full_pipeline_pose": passed,
        "decision": "GO_TO_PIPELINE" if passed else "STOP_BEFORE_PIPELINE",
        "limitations": [
            "The surface-capacity value is a Track-count proxy, not a selector result.",
            "A mechanism pass authorizes compact pipeline/mapping-pose validation; it is not a pose claim.",
        ],
        "inputs": {
            "control": str(args.control.resolve()),
            "variant": str(args.variant.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
