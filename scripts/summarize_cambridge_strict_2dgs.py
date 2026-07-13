#!/usr/bin/env python

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


DEFAULT_SCENES = [
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
]
DEFAULT_VARIANTS = ["baseline", "field", "pair", "best"]


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_result(results_root, map_kind, scene, variant, model_path=None):
    pattern = f"strict2dgs-{map_kind}-{variant}-{scene}-*/summary.json"
    candidates = list(results_root.glob(pattern))
    if model_path is not None:
        expected = Path(model_path).resolve()
        candidates = [
            path
            for path in candidates
            if Path(read_json(path).get("model_path", "")).resolve() == expected
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def result_metrics(result_dir):
    summary = read_json(result_dir / "summary.json")
    results = read_json(result_dir / "results.json")
    sparse = summary["sparse"]
    diagnostics = summary.get("sparse_diagnostics", {})
    per_query = {item["image_name"]: item for item in results}
    errors = np.asarray(
        [float(item["sparse_TE"]) for item in results], dtype=np.float64
    )
    return {
        "result_dir": str(result_dir.resolve()),
        "query_count": len(results),
        "median_te_cm": float(np.median(errors)),
        "mean_te_cm": float(np.mean(errors)),
        "p95_te_cm": float(np.quantile(errors, 0.95)),
        "max_te_cm": float(np.max(errors)),
        "median_ae_deg": float(sparse["median_ae"]),
        "recall_5cm_5deg": float(sparse["recall_5cm_5d"]),
        "recall_2cm_2deg": float(sparse["recall_2cm_2d"]),
        "avg_inliers": float(sparse["avg_inliers"]),
        "raw_gt_precision_2px": diagnostics.get(
            "sparse_diag_matcher_raw_all_gt_precision_2px_mean"
        ),
        "post_selector_gt_precision_2px": diagnostics.get(
            "sparse_diag_post_selector_all_gt_precision_2px_mean"
        ),
        "inlier_gt_precision_2px": diagnostics.get(
            "sparse_diag_post_selector_inlier_gt_precision_2px_mean"
        ),
        "inlier_pose_info_logdet": diagnostics.get(
            "sparse_diag_post_selector_inlier_pose_info_logdet_mean"
        ),
        "post_selector_match_count": diagnostics.get(
            "sparse_diag_post_selector_match_count_mean"
        ),
        "artifact_provenance": summary.get("artifact_provenance", {}),
        "per_query": per_query,
    }


def paired_bootstrap(baseline, candidate, seed, samples, batch_size=256):
    names = sorted(set(baseline["per_query"]) & set(candidate["per_query"]))
    if len(names) != baseline["query_count"] or len(names) != candidate["query_count"]:
        raise ValueError("Result variants do not contain the same query set")
    base = np.asarray(
        [baseline["per_query"][name]["sparse_TE"] for name in names],
        dtype=np.float64,
    )
    candidate_values = np.asarray(
        [candidate["per_query"][name]["sparse_TE"] for name in names],
        dtype=np.float64,
    )
    paired_delta = candidate_values - base
    rng = np.random.default_rng(seed)
    marginal_median_delta = []
    paired_mean_delta = []
    paired_median_delta = []
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        indices = rng.integers(0, len(names), size=(count, len(names)))
        sampled_delta = paired_delta[indices]
        marginal_median_delta.append(
            np.median(candidate_values[indices], axis=1)
            - np.median(base[indices], axis=1)
        )
        paired_mean_delta.append(np.mean(sampled_delta, axis=1))
        paired_median_delta.append(np.median(sampled_delta, axis=1))
    marginal_median_delta = np.concatenate(marginal_median_delta)
    paired_mean_delta = np.concatenate(paired_mean_delta)
    paired_median_delta = np.concatenate(paired_median_delta)
    return {
        "marginal_median_delta_cm": float(
            np.median(candidate_values) - np.median(base)
        ),
        "marginal_median_delta_ci95_cm": np.quantile(
            marginal_median_delta, [0.025, 0.975]
        ).tolist(),
        "marginal_median_improvement_probability": float(
            np.mean(marginal_median_delta < 0.0)
        ),
        "paired_mean_delta_cm": float(np.mean(paired_delta)),
        "paired_mean_delta_ci95_cm": np.quantile(
            paired_mean_delta, [0.025, 0.975]
        ).tolist(),
        "paired_median_delta_cm": float(np.median(paired_delta)),
        "paired_median_delta_ci95_cm": np.quantile(
            paired_median_delta, [0.025, 0.975]
        ).tolist(),
        "query_win_rate": float(np.mean(paired_delta < 0.0)),
    }


def window_median(history, key):
    values = [float(item[key]) for item in history if key in item]
    if not values:
        return None
    window = max(1, len(values) // 5)
    return {
        "first": float(statistics.median(values[:window])),
        "last": float(statistics.median(values[-window:])),
    }


def training_metrics(model_root):
    output = {}
    map_summary_path = model_root / "loc_training_summary.json"
    if map_summary_path.is_file():
        summary = read_json(map_summary_path)
        episodes = int(summary.get("diff_pnp_episodes", 0))
        output["map"] = {
            "initial_point_count": summary.get("geometry_initial_point_count"),
            "final_point_count": summary.get("geometry_final_point_count"),
            "pnp_episodes": episodes,
            "pnp_condition_pass_rate": (
                summary.get("diff_pnp_condition_guard_passed_total", 0.0)
                / episodes
                if episodes
                else None
            ),
            "pnp_geometry_correspondences_mean": (
                summary.get("diff_pnp_geometry_correspondences_total", 0.0)
                / episodes
                if episodes
                else None
            ),
            "raw_xyz_surviving_source_delta_mean_m": summary.get(
                "raw_xyz_delta_from_initial_mean"
            ),
            "all_source_delta_p95_m": summary.get(
                "raw_xyz_delta_from_initial_all_sources_p95"
            ),
            "loc_anchor_tangent_delta_p95_m": summary.get(
                "loc_anchor_tangent_delta_norm_p95"
            ),
            "loc_anchor_normal_delta_p95_m": summary.get(
                "loc_anchor_normal_delta_abs_p95"
            ),
        }

    r2_path = (
        model_root
        / "detector_strict2dgs_R2_flr2e4_2000"
        / "candidate_teacher_training_summary.json"
    )
    if r2_path.is_file():
        summary = read_json(r2_path)
        history = summary.get("history", [])
        output["r2"] = {
            "feature_lr": summary.get("config", {}).get("feature_lr"),
            "detector_lr": summary.get("config", {}).get("detector_lr"),
            "loss_assignment": window_median(history, "loss_assignment"),
            "predicted_gt_precision": window_median(
                history, "predicted_gt_precision"
            ),
            "false_negative_rate": window_median(
                history, "false_negative_rate"
            ),
            "assignment_top1_accuracy": window_median(
                history, "assignment_top1_accuracy"
            ),
        }

    head_path = (
        model_root
        / "detector_strict2dgs_pair_flr2e4_geometrycontext500"
        / "candidate_teacher_training_summary.json"
    )
    if head_path.is_file():
        summary = read_json(head_path)
        history = summary.get("validation_history", [])
        validation = history[-1] if history else {}
        keys = [
            "camera_count",
            "pair_measurement_ap_mean",
            "pair_measurement_auroc_mean",
            "pair_measurement_ece_mean",
            "pair_measurement_offset_epe_mean_mean",
            "pair_measurement_set_bias_m_mean",
            "pair_measurement_set_condition_mean",
            "pair_measurement_calibrated_threshold",
            "pair_measurement_calibrated_precision",
            "pair_measurement_calibrated_recall",
            "pair_measurement_calibrated_accepted_count",
        ]
        output["pair_measurement"] = {
            key: validation.get(key) for key in keys
        }
    return output


def fmt(value, scale=1.0, digits=3):
    if value is None:
        return "-"
    return f"{scale * value:.{digits}f}"


def make_markdown(report):
    lines = [
        "# Cambridge Strict From-SfM 2DGS Validation",
        "",
        "The latest result for each scene/variant is selected by modification time. "
        "All result paths and artifact hashes are retained in the JSON report.",
        "",
    ]
    for scene, scene_report in report["scenes"].items():
        lines.extend(
            [
                f"## {scene}",
                "",
                "| Variant | Median TE cm | Mean | P95 | Max | AE deg | R5 % | R2 % | Raw P@2 % | Post P@2 % | Inlier P@2 % | Pose info | Matches |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant, item in scene_report["variants"].items():
            lines.append(
                "| {variant} | {median} | {mean} | {p95} | {maximum} | {ae} | "
                "{r5} | {r2} | {raw} | {post} | {inlier} | {info} | {matches} |".format(
                    variant=variant,
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
        if scene_report["comparisons"]:
            lines.extend(
                [
                    "",
                    "| Variant vs baseline | Marginal median delta cm (CI95) | P(median improves) | Paired mean delta cm (CI95) | Query win % |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for variant, item in scene_report["comparisons"].items():
                median_ci = item["marginal_median_delta_ci95_cm"]
                mean_ci = item["paired_mean_delta_ci95_cm"]
                lines.append(
                    f"| {variant} | {item['marginal_median_delta_cm']:.3f} "
                    f"[{median_ci[0]:.3f}, {median_ci[1]:.3f}] | "
                    f"{100.0 * item['marginal_median_improvement_probability']:.1f} | "
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
    parser.add_argument("--map_kind", choices=["lafgs", "matcha"], default="lafgs")
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--require_complete", action="store_true")
    args = parser.parse_args()

    report = {
        "protocol": {
            "map_kind": args.map_kind,
            "map_type": "Cambridge SfM step-0 native 2DGS"
            if args.map_kind == "lafgs"
            else "MAtCha fixed native 2DGS geometry",
            "map_iteration": 30000 if args.map_kind == "lafgs" else 60000,
            "landmark_count": 16384,
            "detect_num": 4096,
            "nms_radius": 2,
            "reprojection_error_px": 12.0,
            "max_matches_per_landmark": 2,
            "best_refill_floor": 1024,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.seed,
        },
        "scenes": {},
        "missing": [],
    }
    model_parent = (
        args.experiment_root / "lafgs_from_sfm"
        if args.map_kind == "lafgs"
        else args.experiment_root / "matcha_feature_baseline"
    )
    for scene in args.scenes:
        variants = {}
        for variant in args.variants:
            result_dir = latest_result(
                args.results_root,
                args.map_kind,
                scene,
                variant,
                model_path=model_parent / scene,
            )
            if result_dir is None:
                report["missing"].append(f"{args.map_kind}/{scene}/{variant}")
                continue
            variants[variant] = result_metrics(result_dir)
        comparisons = {}
        if "baseline" in variants:
            for variant, item in variants.items():
                if variant == "baseline":
                    continue
                comparisons[variant] = paired_bootstrap(
                    variants["baseline"],
                    item,
                    seed=args.seed,
                    samples=args.bootstrap_samples,
                )
        for item in variants.values():
            item.pop("per_query")
        report["scenes"][scene] = {
            "variants": variants,
            "comparisons": comparisons,
            "training": training_metrics(model_parent / scene),
        }

    if args.require_complete and report["missing"]:
        raise FileNotFoundError(
            "Missing strict results: " + ", ".join(report["missing"])
        )

    output_dir = args.experiment_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"strict_2dgs_{args.map_kind}_summary.json"
    markdown_path = output_dir / f"STRICT_2DGS_{args.map_kind.upper()}_REPORT.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    markdown_path.write_text(make_markdown(report), encoding="utf-8")
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
