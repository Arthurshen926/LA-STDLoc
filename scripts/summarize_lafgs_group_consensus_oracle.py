#!/usr/bin/env python
"""Apply the cross-scene gate for group-saturated consensus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def scene_improved(summary):
    failure = summary.get("failure", {})
    control = summary.get("control", {})
    return (
        failure.get("win_rate_delta", 0.0) > 0.0
        and summary["all"]["recovered_count"]
        > summary["all"]["regressed_count"]
        and control.get("regressed_count", 0)
        <= control.get("recovered_count", 0) + 1
    )


def greatcourt_non_regression(summary):
    failure = summary.get("failure", {})
    control = summary.get("control", {})
    return (
        failure.get("win_rate_delta", 0.0) >= 0.0
        and summary["all"]["recovered_count"]
        >= summary["all"]["regressed_count"]
        and control.get("win_rate_delta", 0.0) >= -0.05
    )


def aggregate(reports):
    common = set.intersection(
        *[set(report["variant_summary"]) for report in reports]
    )
    common.discard("standard_msac")
    variants = {}
    for variant in sorted(common):
        per_scene = {
            report["scene"]: report["variant_summary"][variant]
            for report in reports
        }
        improved = [
            scene
            for scene, summary in per_scene.items()
            if scene_improved(summary)
        ]
        great_summary = per_scene.get("GreatCourt")
        great_ok = (
            greatcourt_non_regression(great_summary)
            if great_summary is not None
            else False
        )
        variants[variant] = {
            "improved_scenes": improved,
            "improved_scene_count": len(improved),
            "greatcourt_non_regression": great_ok,
            "gate_pass": len(improved) >= 3 and great_ok,
            "net_recoveries": sum(
                summary["all"]["recovered_count"]
                - summary["all"]["regressed_count"]
                for summary in per_scene.values()
            ),
            "per_scene": per_scene,
        }
    ranked = sorted(
        variants,
        key=lambda name: (
            variants[name]["gate_pass"],
            variants[name]["improved_scene_count"],
            variants[name]["net_recoveries"],
            name,
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    return {
        "schema": "lafgs_group_consensus_cross_scene_gate_v1",
        "scene_count": len(reports),
        "scenes": [report["scene"] for report in reports],
        "required_improved_scenes": 3,
        "requires_greatcourt_non_regression": True,
        "best_variant": best,
        "gate_pass": bool(best and variants[best]["gate_pass"]),
        "variants": variants,
    }


def _markdown(payload):
    lines = [
        "# Group-Saturated Consensus Cross-Scene Gate",
        "",
        f"Gate: **{'PASS' if payload['gate_pass'] else 'FAIL'}**",
        "",
        "| Variant | Improved scenes | GreatCourt safe | Net recoveries | Gate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in sorted(
        payload["variants"].items(),
        key=lambda item: (
            item[1]["gate_pass"],
            item[1]["improved_scene_count"],
            item[1]["net_recoveries"],
        ),
        reverse=True,
    ):
        lines.append(
            f"| {name} | {row['improved_scene_count']} | "
            f"{'yes' if row['greatcourt_non_regression'] else 'no'} | "
            f"{row['net_recoveries']:+d} | "
            f"{'PASS' if row['gate_pass'] else 'FAIL'} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in args.reports]
    payload = aggregate(reports)
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    Path(args.output_markdown).write_text(_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
