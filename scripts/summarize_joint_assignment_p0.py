#!/usr/bin/env python3
"""Aggregate the frozen five-scene joint-assignment P0 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCENES = (
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
)


def _gain(base: dict, method: dict) -> dict:
    return {
        "median_te_cm": base["median_te_cm"] - method["median_te_cm"],
        "mean_te_cm": base["mean_te_cm"] - method["mean_te_cm"],
        "p90_te_cm": base["p90_te_cm"] - method["p90_te_cm"],
        "r5_percentage_points": 100.0
        * (method["recall_5cm_5deg"] - base["recall_5cm_5deg"]),
    }


def _clear_pose_improvement(gain: dict) -> bool:
    broad = (
        gain["median_te_cm"] >= 0.2
        and gain["mean_te_cm"] >= 0.0
        and gain["p90_te_cm"] >= 0.0
        and gain["r5_percentage_points"] >= -0.5
    )
    tail = (
        gain["p90_te_cm"] >= 1.0
        and gain["mean_te_cm"] >= 0.2
        and gain["median_te_cm"] >= -0.1
        and gain["r5_percentage_points"] >= 0.0
    )
    return bool(broad or tail)


def summarize(report_paths: dict[str, Path]) -> dict:
    missing = [scene for scene in SCENES if scene not in report_paths]
    if missing:
        return {
            "schema": "lafgs_joint_assignment_p0_gate",
            "status": "INCOMPLETE",
            "missing_scenes": missing,
            "scenes": {},
            "recommendation": "WAIT_FOR_ALL_FIVE_SCENES",
        }

    rows = {}
    clear_count = 0
    rank_gate_count = 0
    difficult_tail_improved = 0
    difficult_tail_total = 0
    replay_failures = []
    for scene in SCENES:
        report = json.loads(report_paths[scene].read_text())
        base = report["pose"]["actual"]
        oracle = report["P0_oracles"]["one_of_k"]["16"]
        gains = _gain(base, oracle)
        clear = _clear_pose_improvement(gains)
        clear_count += int(clear)
        coverage = report["topk_identity_coverage"]["radius_2px"]["top_16"]
        rank_ok = coverage["positive_recall_matchable"] >= 0.5
        rank_gate_count += int(rank_ok)
        rescue = report["tail_rescue"]["OK16_one_of_k"]
        if scene in {"GreatCourt", "StMarysChurch"}:
            difficult_tail_improved += int(rescue["p90_tail_improved_count"])
            difficult_tail_total += int(rescue["p90_tail_query_count"])
        scene_replay_failures = list(report.get("pose_replay_failures", ()))
        if scene_replay_failures:
            replay_failures.append(
                {"scene": scene, "count": len(scene_replay_failures)}
            )
        rows[scene] = {
            "query_count": report["query_count"],
            "A1-All": base,
            "OK16-All": oracle,
            "OK16-S512-PoseSufficient": report["P0_oracles"][
                "one_of_k_fixed_set"
            ]["16"]["S512-PoseSufficient"],
            "OK16-S1024-Block8": report["P0_oracles"][
                "one_of_k_fixed_set"
            ]["16"]["S1024-Block8"],
            "A1-S512-PoseSufficient": report["P0_oracles"]["fixed_top1"][
                "S512-PoseSufficient"
            ],
            "A1-S1024-Block8": report["P0_oracles"]["fixed_top1"][
                "S1024-Block8"
            ],
            "OK16_gain": gains,
            "clear_pose_improvement": clear,
            "positive_row_rate_all_top16": coverage["positive_row_rate_all"],
            "positive_recall_matchable_top16": coverage[
                "positive_recall_matchable"
            ],
            "rank16_gate": rank_ok,
            "tail_rescue": rescue,
        }

    tail_fraction = difficult_tail_improved / max(difficult_tail_total, 1)
    checks = {
        "one_of_k_clear_improvement_at_least_3_of_5": clear_count >= 3,
        "rank16_recall_at_least_half_in_at_least_3_of_5": rank_gate_count >= 3,
        "greatcourt_stmary_p90_tail_rescue_at_least_25pct": tail_fraction >= 0.25,
        "exact_pose_replay_has_no_failures": not replay_failures,
    }
    passed = all(checks.values())
    return {
        "schema": "lafgs_joint_assignment_p0_gate",
        "status": "PASS" if passed else "NO_GO",
        "gate_definition": {
            "clear_pose_improvement": (
                "median gain >=0.2cm with non-worse mean/P90/R5, or P90 gain "
                ">=1cm with non-worse median/R5"
            ),
            "rank16": "strict-positive recall among matchable rows >=50%",
            "difficult_tail": (
                "at least 25% of the A1 P90 tail in GreatCourt and "
                "StMarysChurch improves under OK16"
            ),
        },
        "checks": checks,
        "clear_scene_count": clear_count,
        "rank16_scene_count": rank_gate_count,
        "difficult_tail_improved_fraction": tail_fraction,
        "pose_replay_failures": replay_failures,
        "scenes": rows,
        "recommendation": (
            "TRAIN_CROSS_SCENE_ASSIGNMENT_HEAD"
            if passed
            else "CLOSE_SELECTOR_DIRECTION_AFTER_P0"
        ),
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# Joint Assignment P0 Gate",
        "",
        f"Status: **{summary['status']}**",
        "",
    ]
    if summary["status"] == "INCOMPLETE":
        lines.append("Missing: " + ", ".join(summary["missing_scenes"]))
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "| Scene | A1 median | OK16 median | A1 P90 | OK16 P90 | Top16 strict recall | Clear |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for scene in SCENES:
        row = summary["scenes"][scene]
        base = row["A1-All"]
        oracle = row["OK16-All"]
        lines.append(
            f"| {scene} | {base['median_te_cm']:.3f} | "
            f"{oracle['median_te_cm']:.3f} | {base['p90_te_cm']:.3f} | "
            f"{oracle['p90_te_cm']:.3f} | "
            f"{100 * row['positive_recall_matchable_top16']:.1f}% | "
            f"{'yes' if row['clear_pose_improvement'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Recommendation: **{summary['recommendation']}**",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="SCENE=PATH",
        help="Repeat once per Cambridge scene.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report_paths = {}
    for value in args.report:
        scene, separator, path = value.partition("=")
        if not separator or scene not in SCENES:
            raise ValueError(f"invalid --report entry: {value}")
        report_paths[scene] = Path(path).resolve()
    summary = summarize(report_paths)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    output.with_suffix(".md").write_text(_markdown(summary))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
