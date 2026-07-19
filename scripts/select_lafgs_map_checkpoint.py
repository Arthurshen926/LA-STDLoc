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


def load_metrics(path):
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
    min_te_gain_cm=0.02,
    metric_tolerance=1e-9,
):
    control = {
        "tag": "control_strong",
        "state": str(Path(control_state).resolve()),
        **load_metrics(control_results),
    }
    if control["evaluation_protocol_sha256"] is None:
        raise ValueError(
            "selection requires evaluation_protocol.protocol_sha256; "
            "rerun legacy evaluations with the pinned input protocol"
        )
    evaluated = []
    for tag, results_path, state_path in candidates:
        candidate = {
            "tag": str(tag),
            "state": str(Path(state_path).resolve()),
            **load_metrics(results_path),
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
        candidate["gate_checks"] = checks
        candidate["accepted"] = all(checks.values())
        evaluated.append(candidate)

    accepted = [candidate for candidate in evaluated if candidate["accepted"]]
    selected = (
        min(
            accepted,
            key=lambda item: (
                item["metrics"]["median_te_cm"],
                item["metrics"]["median_ae_deg"],
                item["tag"],
            ),
        )
        if accepted
        else control
    )
    return {
        "selection_protocol": {
            "subset": control["evaluation_camera_subset"],
            "query_count": control["evaluation_camera_count"],
            "test_metrics_used": False,
            "min_translation_gain_cm": float(min_te_gain_cm),
            "metric_tolerance": float(metric_tolerance),
            "joint_gate": [
                "translation_gain",
                "rotation_not_worse",
                "raw_precision_not_worse",
                "inlier_precision_not_worse",
                "pose_information_not_worse",
            ],
        },
        "control": control,
        "candidates": evaluated,
        "selected_tag": selected["tag"],
        "selected_state": selected["state"],
        "selected_metrics": selected["metrics"],
        "used_strong_fallback": not bool(accepted),
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
        "--candidate",
        nargs=3,
        action="append",
        default=[],
        metavar=("TAG", "RESULTS_SUMMARY", "STATE"),
    )
    parser.add_argument("--min_te_gain_cm", type=float, default=0.02)
    parser.add_argument("--metric_tolerance", type=float, default=1e-9)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = select_checkpoint(
        args.control_results,
        args.control_state,
        args.candidate,
        min_te_gain_cm=args.min_te_gain_cm,
        metric_tolerance=args.metric_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
