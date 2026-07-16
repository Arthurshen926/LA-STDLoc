#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from summarize_cambridge_strict_2dgs import paired_bootstrap, read_json, result_metrics


DEFAULT_SCENES = [
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "StMarysChurch",
]


def latest_v14_result(results_root, scene, model_path):
    candidates = list(
        results_root.glob(f"strict2dgs-lafgs-v14f0-{scene}-*/summary.json")
    )
    expected = model_path.resolve()
    candidates = [
        path
        for path in candidates
        if Path(read_json(path).get("model_path", "")).resolve() == expected
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def reference_result(summary, scene, variant):
    item = summary.get("scenes", {}).get(scene, {}).get("variants", {}).get(variant)
    if not item:
        return None
    result_dir = Path(item["result_dir"])
    if not (result_dir / "summary.json").is_file():
        return None
    return result_metrics(result_dir)


def public_metrics(metrics):
    return {key: value for key, value in metrics.items() if key != "per_query"}


def fmt(value, scale=1.0, digits=3):
    if value is None:
        return "-"
    return f"{scale * value:.{digits}f}"


def markdown(report):
    lines = [
        "# Cambridge v14 F0 Cross-Scene Test",
        "",
        "The ShopFacade v14 F0 configuration is transferred unchanged to each "
        "strict from-SfM native-2DGS map. All rows use the full Cambridge test split.",
        "",
    ]
    labels = {
        "old_3dgs_baseline": "Old 3DGS baseline",
        "old_3dgs_field": "Old 3DGS field",
        "old_3dgs_best": "Old 3DGS best",
        "strict_2dgs_baseline": "Strict 2DGS baseline",
        "strict_2dgs_field": "Strict 2DGS old field",
        "v14_f0": "Strict 2DGS v14 F0",
    }
    for scene, scene_report in report["scenes"].items():
        lines.extend(
            [
                f"## {scene}",
                "",
                "| Method | Median TE cm | Mean | P95 | Max | AE deg | R5 % | R2 % | Raw P@2 % | Post P@2 % | Inlier P@2 % | Pose info | Matches |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, item in scene_report["variants"].items():
            lines.append(
                "| {label} | {median} | {mean} | {p95} | {maximum} | {ae} | "
                "{r5} | {r2} | {raw} | {post} | {inlier} | {info} | {matches} |".format(
                    label=labels[key],
                    median=fmt(item["median_te_cm"]),
                    mean=fmt(item["mean_te_cm"]),
                    p95=fmt(item["p95_te_cm"]),
                    maximum=fmt(item["max_te_cm"]),
                    ae=fmt(item["median_ae_deg"]),
                    r5=fmt(item["recall_5cm_5deg"], 100.0, 2),
                    r2=fmt(item["recall_2cm_2deg"], 100.0, 2),
                    raw=fmt(item["raw_gt_precision_2px"], 100.0, 2),
                    post=fmt(item["post_selector_gt_precision_2px"], 100.0, 2),
                    inlier=fmt(item["inlier_gt_precision_2px"], 100.0, 2),
                    info=fmt(item["inlier_pose_info_logdet"], 1.0, 2),
                    matches=fmt(item["post_selector_match_count"], 1.0, 1),
                )
            )
        lines.extend(
            [
                "",
                "| v14 F0 comparison | Median delta cm (CI95) | Mean delta cm (CI95) | Win % |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, item in scene_report["comparisons"].items():
            median_ci = item["marginal_median_delta_ci95_cm"]
            mean_ci = item["paired_mean_delta_ci95_cm"]
            lines.append(
                f"| vs {labels[key]} | {item['marginal_median_delta_cm']:.3f} "
                f"[{median_ci[0]:.3f}, {median_ci[1]:.3f}] | "
                f"{item['paired_mean_delta_cm']:.3f} "
                f"[{mean_ci[0]:.3f}, {mean_ci[1]:.3f}] | "
                f"{100.0 * item['query_win_rate']:.1f} |"
            )
        lines.append("")
    if report["missing"]:
        lines.extend(["## Missing", ""])
        lines.extend(f"- {item}" for item in report["missing"])
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument(
        "--experiment_root",
        type=Path,
        default=Path(
            "/mnt/pool/sqy/stdloc_lafgs_cambridge_matcha2dgs_strict_20260711"
        ),
    )
    parser.add_argument(
        "--old_3dgs_summary",
        type=Path,
        default=Path(
            "/mnt/pool/sqy/stdloc_lafgs_cambridge_best_crossscene_20260711/"
            "crossscene_summary.json"
        ),
    )
    parser.add_argument(
        "--strict_2dgs_summary",
        type=Path,
        default=Path(
            "/mnt/pool/sqy/stdloc_lafgs_cambridge_matcha2dgs_strict_20260711/"
            "reports/strict_2dgs_lafgs_summary.json"
        ),
    )
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    old_3dgs = read_json(args.old_3dgs_summary)
    strict_2dgs = read_json(args.strict_2dgs_summary)
    report = {
        "protocol": {
            "map": "Cambridge SfM step-0 native 2DGS at iteration 30000",
            "field": "ShopFacade v14 F0 transferred without scene tuning",
            "field_iterations": 2000,
            "detector": "scene baseline detector, frozen",
            "landmark_count": 16384,
            "detect_num": 4096,
            "dustbin_weight": 0.25,
            "map_cleanliness_weight": 0.5,
            "map_bias_weight": 0.75,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
        "scenes": {},
        "missing": [],
    }
    model_parent = args.experiment_root / "lafgs_from_sfm"
    for scene in args.scenes:
        variants = {}
        references = [
            ("old_3dgs_baseline", old_3dgs, "baseline"),
            ("old_3dgs_field", old_3dgs, "field"),
            ("old_3dgs_best", old_3dgs, "best"),
            ("strict_2dgs_baseline", strict_2dgs, "baseline"),
            ("strict_2dgs_field", strict_2dgs, "field"),
        ]
        for key, summary, variant in references:
            metrics = reference_result(summary, scene, variant)
            if metrics is None:
                report["missing"].append(f"{scene}/{key}")
            else:
                variants[key] = metrics
        v14_dir = latest_v14_result(
            args.results_root, scene, model_parent / scene
        )
        if v14_dir is None:
            report["missing"].append(f"{scene}/v14_f0")
        else:
            variants["v14_f0"] = result_metrics(v14_dir)

        comparisons = {}
        if "v14_f0" in variants:
            for key, item in variants.items():
                if key == "v14_f0":
                    continue
                comparisons[key] = paired_bootstrap(
                    item,
                    variants["v14_f0"],
                    seed=args.seed,
                    samples=args.bootstrap_samples,
                )
        report["scenes"][scene] = {
            "variants": {key: public_metrics(value) for key, value in variants.items()},
            "comparisons": comparisons,
        }

    if args.require_complete and report["missing"]:
        raise FileNotFoundError("Missing results: " + ", ".join(report["missing"]))

    output_dir = args.experiment_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cambridge_v14_f0_crossscene_summary.json"
    markdown_path = output_dir / "CAMBRIDGE_V14_F0_CROSSSCENE_REPORT.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    markdown_path.write_text(markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
