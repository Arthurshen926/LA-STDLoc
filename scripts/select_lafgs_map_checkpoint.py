#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path


DIAGNOSTIC_KEYS = {
    "raw_gt_precision_2px": "sparse_diag_pre_selector_all_gt_precision_2px_mean",
    "inlier_gt_precision_2px": (
        "sparse_diag_pre_selector_inlier_gt_precision_2px_mean"
    ),
    "translation_pose_info_logdet": (
        "sparse_diag_pre_selector_inlier_pose_info_translation_logdet_mean"
    ),
}


def _mean_sparse_translation_error_cm(summary_path):
    results_path = Path(summary_path).with_name("results.json")
    if not results_path.is_file():
        raise ValueError(
            "performance selection requires per-query results.json next to "
            f"{summary_path}"
        )
    records = json.loads(results_path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"results.json is empty or malformed: {results_path}")
    translation_errors = []
    for index, record in enumerate(records):
        try:
            value = float(record["sparse_TE"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"missing sparse_TE for query {index} in {results_path}"
            ) from exc
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite sparse_TE for query {index} in {results_path}"
            )
        translation_errors.append(value)
    return sum(translation_errors) / len(translation_errors)


def load_metrics(path, *, selection_mode="safety"):
    path = Path(path)
    payload = json.loads(path.read_text())
    sparse = payload["sparse"]
    diagnostics = payload["sparse_diagnostics"]
    metrics = {
        "median_te_cm": float(sparse["median_te"]),
        "median_ae_deg": float(sparse["median_ae"]),
        **{
            name: float(diagnostics[key])
            for name, key in DIAGNOSTIC_KEYS.items()
        },
    }
    if selection_mode == "performance":
        metrics.update(
            {
                "mean_te_cm": _mean_sparse_translation_error_cm(path),
                "recall_2m_5deg": float(sparse["recall_2m_5d"]),
                "recall_5cm_5deg": float(sparse["recall_5cm_5d"]),
            }
        )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"non-finite selection metric in {path}")
    return {
        "results_summary": str(path.resolve()),
        "evaluation_camera_subset": payload.get("evaluation_camera_subset"),
        "evaluation_camera_count": int(payload["evaluation_camera_count"]),
        "evaluation_protocol_sha256": (
            payload.get("evaluation_protocol", {}).get("protocol_sha256")
        ),
        "metrics": metrics,
    }


def select_checkpoint(
    control_results,
    control_state,
    candidates,
    *,
    control_tag="control_strong",
    min_te_gain_cm=0.02,
    metric_tolerance=1e-9,
    selection_mode="safety",
    mean_te_weight=0.05,
    max_recall_2m_drop=0.01,
    max_recall_5cm_drop=0.01,
):
    if selection_mode not in {"safety", "performance"}:
        raise ValueError(f"unsupported selection mode: {selection_mode}")
    if mean_te_weight < 0.0:
        raise ValueError("mean_te_weight must be non-negative")
    if max_recall_2m_drop < 0.0 or max_recall_5cm_drop < 0.0:
        raise ValueError("recall-drop tolerances must be non-negative")
    control_tag = str(control_tag)
    if not control_tag:
        raise ValueError("control_tag must be non-empty")
    control = {
        "tag": control_tag,
        "state": str(Path(control_state).resolve()),
        **load_metrics(control_results, selection_mode=selection_mode),
    }
    if control["evaluation_protocol_sha256"] is None:
        raise ValueError(
            "selection requires evaluation_protocol.protocol_sha256; "
            "rerun legacy evaluations with the pinned input protocol"
        )
    if selection_mode == "performance":
        control["primary_score"] = (
            control["metrics"]["median_te_cm"]
            + float(mean_te_weight) * control["metrics"]["mean_te_cm"]
        )
    evaluated = []
    for tag, results_path, state_path in candidates:
        candidate = {
            "tag": str(tag),
            "state": str(Path(state_path).resolve()),
            **load_metrics(results_path, selection_mode=selection_mode),
        }
        if (
            candidate["evaluation_camera_subset"]
            != control["evaluation_camera_subset"]
            or candidate["evaluation_camera_count"]
            != control["evaluation_camera_count"]
            or candidate["evaluation_protocol_sha256"]
            != control["evaluation_protocol_sha256"]
        ):
            raise ValueError(
                f"{tag} does not use the same validation protocol as the control"
            )
        current = candidate["metrics"]
        baseline = control["metrics"]
        if selection_mode == "safety":
            checks = {
                "translation_gain": (
                    current["median_te_cm"]
                    <= baseline["median_te_cm"] - float(min_te_gain_cm)
                ),
                "rotation_not_worse": (
                    current["median_ae_deg"]
                    <= baseline["median_ae_deg"] + float(metric_tolerance)
                ),
                "raw_precision_not_worse": (
                    current["raw_gt_precision_2px"]
                    >= baseline["raw_gt_precision_2px"] - float(metric_tolerance)
                ),
                "inlier_precision_not_worse": (
                    current["inlier_gt_precision_2px"]
                    >= baseline["inlier_gt_precision_2px"] - float(metric_tolerance)
                ),
                "pose_information_not_worse": (
                    current["translation_pose_info_logdet"]
                    >= baseline["translation_pose_info_logdet"]
                    - float(metric_tolerance)
                ),
            }
        else:
            candidate["primary_score"] = (
                current["median_te_cm"]
                + float(mean_te_weight) * current["mean_te_cm"]
            )
            checks = {
                "primary_objective_gain": (
                    candidate["primary_score"]
                    <= control["primary_score"]
                    - float(min_te_gain_cm)
                    + float(metric_tolerance)
                ),
                "recall_2m_not_significantly_worse": (
                    current["recall_2m_5deg"]
                    >= baseline["recall_2m_5deg"] - float(max_recall_2m_drop)
                ),
                "recall_5cm_not_significantly_worse": (
                    current["recall_5cm_5deg"]
                    >= baseline["recall_5cm_5deg"] - float(max_recall_5cm_drop)
                ),
            }
        candidate["gate_checks"] = checks
        candidate["accepted"] = all(checks.values())
        evaluated.append(candidate)

    accepted = [candidate for candidate in evaluated if candidate["accepted"]]
    selected = min(
        [control, *accepted],
        key=lambda item: (
            (
                item["primary_score"]
                if selection_mode == "performance"
                else item["metrics"]["median_te_cm"]
            ),
            item["metrics"]["median_ae_deg"],
            item["tag"],
        ),
    )
    protocol = {
        "subset": control["evaluation_camera_subset"],
        "query_count": control["evaluation_camera_count"],
        "test_metrics_used": False,
        "selection_mode": selection_mode,
    }
    if selection_mode == "safety":
        protocol.update(
            {
                "min_translation_gain_cm": float(min_te_gain_cm),
                "metric_tolerance": float(metric_tolerance),
                "joint_gate": [
                    "translation_gain",
                    "rotation_not_worse",
                    "raw_precision_not_worse",
                    "inlier_precision_not_worse",
                    "pose_information_not_worse",
                ],
            }
        )
    else:
        protocol.update(
            {
                "primary_metric": "median_te_cm + mean_te_weight * mean_te_cm",
                "mean_te_weight": float(mean_te_weight),
                "min_primary_score_gain_cm": float(min_te_gain_cm),
                "max_recall_2m_drop": float(max_recall_2m_drop),
                "max_recall_5cm_drop": float(max_recall_5cm_drop),
                "deployment_constraints": [
                    "primary_objective_gain",
                    "recall_2m_not_significantly_worse",
                    "recall_5cm_not_significantly_worse",
                ],
            }
        )
    return {
        "selection_protocol": protocol,
        "control": control,
        "candidates": evaluated,
        "selected_tag": selected["tag"],
        "selected_state": selected["state"],
        "selected_metrics": selected["metrics"],
        # A later alternating stage can legitimately use the selected
        # residual or BA state as its control.  Keep that identity intact:
        # calling every control "strong" makes the final test provenance lie
        # even when the selected state path is correct.
        "control_tag": control_tag,
        "used_control_fallback": selected["tag"] == control_tag,
        # Backward-compatible alias retained for existing readers.  It means
        # "the control was retained", not necessarily that it was bootstrap.
        "used_strong_fallback": selected["tag"] == control_tag,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select a LaFGS map checkpoint on held-out queries with a joint "
            "cleanliness, pose-information, and pose-error gate."
        )
    )
    parser.add_argument("--control_results", required=True, type=Path)
    parser.add_argument("--control_state", required=True, type=Path)
    parser.add_argument(
        "--control_tag",
        default="control_strong",
        help="Provenance label for the validation control state.",
    )
    parser.add_argument(
        "--candidate",
        nargs=3,
        action="append",
        default=[],
        metavar=("TAG", "RESULTS_SUMMARY", "STATE"),
    )
    parser.add_argument("--min_te_gain_cm", type=float, default=0.02)
    parser.add_argument("--metric_tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--selection_mode",
        choices=["safety", "performance"],
        default="safety",
        help="Use the strict safety gate or the deployment-pose primary metric.",
    )
    parser.add_argument("--mean_te_weight", type=float, default=0.05)
    parser.add_argument("--max_recall_2m_drop", type=float, default=0.01)
    parser.add_argument("--max_recall_5cm_drop", type=float, default=0.01)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = select_checkpoint(
        args.control_results,
        args.control_state,
        args.candidate,
        control_tag=args.control_tag,
        min_te_gain_cm=args.min_te_gain_cm,
        metric_tolerance=args.metric_tolerance,
        selection_mode=args.selection_mode,
        mean_te_weight=args.mean_te_weight,
        max_recall_2m_drop=args.max_recall_2m_drop,
        max_recall_5cm_drop=args.max_recall_5cm_drop,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
