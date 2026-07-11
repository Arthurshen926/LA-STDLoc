#!/usr/bin/env python

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


SCENES = ["GreatCourt", "KingsCollege", "OldHospital", "StMarysChurch"]
VARIANTS = ["baseline", "field", "pair", "best"]


def latest_result(results_root, scene, variant):
    pattern = f"crossscene-{variant}-{scene}-*/summary.json"
    candidates = list(results_root.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No result matches {pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime).parent


def summary_metrics(result_dir):
    summary = json.loads((result_dir / "summary.json").read_text())
    results = json.loads((result_dir / "results.json").read_text())
    sparse = summary["sparse"]
    diagnostics = summary.get("sparse_diagnostics", {})
    translation_errors = np.asarray(
        [float(item["sparse_TE"]) for item in results], dtype=np.float64
    )
    return {
        "result_dir": str(result_dir),
        "query_count": len(results),
        "median_te_cm": float(sparse["median_te"]),
        "mean_te_cm": float(translation_errors.mean()),
        "p95_te_cm": float(np.quantile(translation_errors, 0.95)),
        "max_te_cm": float(translation_errors.max()),
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
        "per_query": {item["image_name"]: item for item in results},
    }


def paired_bootstrap(baseline, candidate, seed, samples):
    names = sorted(set(baseline["per_query"]) & set(candidate["per_query"]))
    if len(names) != baseline["query_count"] or len(names) != candidate["query_count"]:
        raise ValueError("Result variants do not contain the same query set")
    base_te = np.asarray(
        [baseline["per_query"][name]["sparse_TE"] for name in names],
        dtype=np.float64,
    )
    candidate_te = np.asarray(
        [candidate["per_query"][name]["sparse_TE"] for name in names],
        dtype=np.float64,
    )
    delta = candidate_te - base_te
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(names), size=(samples, len(names)))
    bootstrap_mean = delta[indices].mean(axis=1)
    bootstrap_median = np.median(delta[indices], axis=1)
    return {
        "paired_mean_delta_cm": float(delta.mean()),
        "paired_mean_delta_ci95_cm": np.quantile(
            bootstrap_mean, [0.025, 0.975]
        ).tolist(),
        "paired_median_delta_cm": float(np.median(delta)),
        "paired_median_delta_ci95_cm": np.quantile(
            bootstrap_median, [0.025, 0.975]
        ).tolist(),
        "win_rate": float((delta < 0).mean()),
        "mean_absolute_delta_cm": float(np.abs(delta).mean()),
    }


def window_change(history, key):
    values = [float(item[key]) for item in history if key in item]
    window = max(2, len(values) // 5)
    return {
        "first_window_median": float(statistics.median(values[:window])),
        "last_window_median": float(statistics.median(values[-window:])),
    }


def training_metrics(model_root):
    r2_path = (
        model_root
        / "detector_crossscene_best_R2_2000"
        / "candidate_teacher_training_summary.json"
    )
    head_path = (
        model_root
        / "detector_crossscene_pair_geometrycontext500"
        / "candidate_teacher_training_summary.json"
    )
    r2 = json.loads(r2_path.read_text())
    head = json.loads(head_path.read_text())
    validation = head["validation_history"][-1]
    return {
        "r2": {
            key: window_change(r2["history"], key)
            for key in [
                "loss_total",
                "loss_assignment",
                "predicted_gt_precision",
                "false_negative_rate",
                "assignment_top1_accuracy",
            ]
        },
        "pair_measurement_validation": {
            key: validation.get(key)
            for key in [
                "camera_count",
                "pair_measurement_ap_mean",
                "pair_measurement_auroc_mean",
                "pair_measurement_ece_mean",
                "pair_measurement_offset_epe_mean_mean",
                "pair_measurement_corrected_signed_bias_norm_px_mean",
                "pair_measurement_set_bias_m_mean",
                "pair_measurement_set_condition_mean",
                "pair_measurement_calibrated_threshold",
                "pair_measurement_calibrated_precision",
                "pair_measurement_calibrated_recall",
                "pair_measurement_calibrated_accepted_count",
            ]
        },
    }


def percent(value):
    return "-" if value is None else f"{100.0 * value:.2f}"


def make_markdown(report):
    lines = [
        "# Cambridge Cross-Scene Best-Protocol Validation",
        "",
        "All four scenes use their existing scene-specific 30k 3DGS baseline map. "
        "The ShopFacade best frontend protocol is transferred without scene-specific "
        "hyperparameter tuning: R2 feature/detector co-adaptation, geometry-context "
        "PairMeasurementHead, signed offset, calibrated threshold, and score refill "
        "to 1024 candidates.",
        "",
    ]
    for scene in SCENES:
        lines.extend(
            [
                f"## {scene}",
                "",
                "| Variant | TE cm | Mean cm | P95 cm | AE deg | R5 % | R2 % | Raw P@2 % | Inlier P@2 % | Pose info | Matches |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            item = report["scenes"][scene]["variants"][variant]
            lines.append(
                "| {variant} | {te:.3f} | {mean:.3f} | {p95:.3f} | {ae:.3f} | "
                "{r5:.2f} | {r2:.2f} | {raw} | {inlier} | {info:.2f} | {matches:.1f} |".format(
                    variant=variant,
                    te=item["median_te_cm"],
                    mean=item["mean_te_cm"],
                    p95=item["p95_te_cm"],
                    ae=item["median_ae_deg"],
                    r5=100.0 * item["recall_5cm_5deg"],
                    r2=100.0 * item["recall_2cm_2deg"],
                    raw=percent(item["raw_gt_precision_2px"]),
                    inlier=percent(item["inlier_gt_precision_2px"]),
                    info=item["inlier_pose_info_logdet"] or 0.0,
                    matches=item["post_selector_match_count"] or 0.0,
                )
            )
        lines.extend(
            [
                "",
                "| Variant vs baseline | Paired mean delta cm (CI95) | Paired median delta cm (CI95) | Win % |",
                "|---|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS[1:]:
            item = report["scenes"][scene]["paired"][variant]
            mean_ci = item["paired_mean_delta_ci95_cm"]
            median_ci = item["paired_median_delta_ci95_cm"]
            lines.append(
                f"| {variant} | {item['paired_mean_delta_cm']:.3f} "
                f"[{mean_ci[0]:.3f}, {mean_ci[1]:.3f}] | "
                f"{item['paired_median_delta_cm']:.3f} "
                f"[{median_ci[0]:.3f}, {median_ci[1]:.3f}] | "
                f"{100.0 * item['win_rate']:.2f} |"
            )
        lines.append("")

    field_median_wins = sum(
        report["scenes"][scene]["variants"]["field"]["median_te_cm"]
        < report["scenes"][scene]["variants"]["baseline"]["median_te_cm"]
        for scene in SCENES
    )
    pair_median_wins = sum(
        report["scenes"][scene]["variants"]["pair"]["median_te_cm"]
        < report["scenes"][scene]["variants"]["baseline"]["median_te_cm"]
        for scene in SCENES
    )
    best_median_wins = sum(
        report["scenes"][scene]["variants"]["best"]["median_te_cm"]
        < report["scenes"][scene]["variants"]["baseline"]["median_te_cm"]
        for scene in SCENES
    )
    refill_mean_wins = sum(
        report["scenes"][scene]["variants"]["best"]["mean_te_cm"]
        < report["scenes"][scene]["variants"]["pair"]["mean_te_cm"]
        for scene in SCENES
    )
    refill_median_wins = sum(
        report["scenes"][scene]["variants"]["best"]["median_te_cm"]
        < report["scenes"][scene]["variants"]["pair"]["median_te_cm"]
        for scene in SCENES
    )
    lines.extend(
        [
            "## Training Diagnostics",
            "",
            "| Scene | R2 assignment first->last | R2 raw P@2 first->last | Head AP | Head ECE | Set condition | Calibrated accepted/query |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scene in SCENES:
        training = report["scenes"][scene]["training"]
        r2 = training["r2"]
        head = training["pair_measurement_validation"]
        accepted_per_query = (
            head["pair_measurement_calibrated_accepted_count"]
            / head["camera_count"]
        )
        lines.append(
            "| {scene} | {a0:.3f}->{a1:.3f} | {p0:.2f}->{p1:.2f}% | "
            "{ap:.3f} | {ece:.3f} | {condition:.1f} | {accepted:.1f} |".format(
                scene=scene,
                a0=r2["loss_assignment"]["first_window_median"],
                a1=r2["loss_assignment"]["last_window_median"],
                p0=100.0
                * r2["predicted_gt_precision"]["first_window_median"],
                p1=100.0
                * r2["predicted_gt_precision"]["last_window_median"],
                ap=head["pair_measurement_ap_mean"],
                ece=head["pair_measurement_ece_mean"],
                condition=head["pair_measurement_set_condition_mean"],
                accepted=accepted_per_query,
            )
        )
    lines.extend(
        [
            "",
            "## Cross-Scene Findings",
            "",
            f"- Field-only improves marginal median TE in {field_median_wins}/4 scenes. "
            "Raw GT precision and pose-information logdet increase in all four scenes, "
            "so the representation learns more matchable/informative correspondences, "
            "but that is not sufficient to remove systematic pose bias.",
            f"- Pair filtering improves inlier GT precision in all four scenes but "
            f"improves marginal median TE in only {pair_median_wins}/4. It consistently "
            "reduces match count and pose information, and can create catastrophic "
            "low-information PnP solutions without refill.",
            f"- The full ShopFacade best configuration improves marginal median TE in "
            f"{best_median_wins}/4 scenes. GreatCourt and KingsCollege have significantly "
            "positive paired-median deltas (worse); OldHospital has a significantly "
            "negative paired-mean delta (better). The transfer is therefore not a "
            "universal localization improvement.",
            f"- Refill1024 lowers mean TE versus pair-no-refill in {refill_mean_wins}/4 "
            f"scenes, but lowers median TE in only {refill_median_wins}/4. It is a useful "
            "tail/failure guard, not a scene-independent median-optimal operating point.",
            "- The dominant bottleneck is objective alignment: AP/ECE improve while set "
            "condition numbers reach 2e5-7e5 in GreatCourt, KingsCollege, and OldHospital. "
            "The fixed 1 cm set-bias normalization and fixed 75% recall calibration do not "
            "directly protect translation observability or systematic signed bias.",
            "",
            "## Scope Boundary",
            "",
            "Other Cambridge scenes did not have current 2DGS/surfel reconstruction "
            "artifacts. This experiment transfers the best localization frontend onto "
            "the existing scene-specific 30k 3DGS baselines. It validates frontend/head "
            "cross-scene behavior, but it is not a full cross-scene replication of the "
            "from-SfM 2DGS reconstruction used by the ShopFacade main line.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=Path("results"))
    parser.add_argument(
        "--experiment_root",
        type=Path,
        default=Path(
            "/mnt/pool/sqy/stdloc_lafgs_cambridge_best_crossscene_20260711"
        ),
    )
    parser.add_argument(
        "--model_root",
        type=Path,
        default=Path("/mnt/pool/sqy/stdloc_la_full_runs"),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    report = {
        "protocol": {
            "map_type": "existing scene-specific 30k 3DGS baseline",
            "landmark_count": 16384,
            "detect_num": 4096,
            "nms": 2,
            "reprojection_error_px": 12.0,
            "max_matches_per_landmark": 2,
            "best_refill_floor": 1024,
        },
        "scenes": {},
    }
    for scene in SCENES:
        variants = {
            variant: summary_metrics(
                latest_result(args.results_root, scene, variant)
            )
            for variant in VARIANTS
        }
        paired = {
            variant: paired_bootstrap(
                variants["baseline"],
                variants[variant],
                seed=args.seed,
                samples=args.bootstrap_samples,
            )
            for variant in VARIANTS[1:]
        }
        for item in variants.values():
            item.pop("per_query")
        report["scenes"][scene] = {
            "variants": variants,
            "paired": paired,
            "training": training_metrics(
                args.model_root / f"{scene}_baseline"
            ),
        }

    args.experiment_root.mkdir(parents=True, exist_ok=True)
    json_path = args.experiment_root / "crossscene_summary.json"
    markdown_path = args.experiment_root / "CROSS_SCENE_VALIDATION_REPORT.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    markdown_path.write_text(make_markdown(report))
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
