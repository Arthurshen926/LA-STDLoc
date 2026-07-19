#!/usr/bin/env python3

"""Choose an initialization family using only held-out map validation results."""

import argparse
import json
import math
from pathlib import Path


def _selected_record(payload):
    selected_tag = payload["selected_tag"]
    if selected_tag == payload["control"]["tag"]:
        return payload["control"]
    for candidate in payload["candidates"]:
        if candidate["tag"] == selected_tag:
            return candidate
    raise ValueError(f"selected tag {selected_tag!r} is absent from report")


def load_initialization_candidate(tag, report_path):
    report_path = Path(report_path)
    payload = json.loads(report_path.read_text())
    protocol = payload.get("selection_protocol", {})
    if protocol.get("test_metrics_used") is not False:
        raise ValueError(f"{report_path} is not a validation-only selection report")
    record = _selected_record(payload)
    metrics = record["metrics"]
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise ValueError(f"non-finite selected metric in {report_path}")
    selection_mode = protocol.get("selection_mode", "safety")
    primary_score = float(record.get("primary_score", metrics["median_te_cm"]))
    return {
        "tag": str(tag),
        "selection_report": str(report_path.resolve()),
        "selected_tag": payload["selected_tag"],
        "selected_state": str(Path(payload["selected_state"]).resolve()),
        "selected_results_summary": str(Path(record["results_summary"]).resolve()),
        "metrics": metrics,
        "primary_score": primary_score,
        "inner_selection_mode": selection_mode,
        "inner_primary_metric": protocol.get(
            "primary_metric", "median_te_cm"
        ),
        "inner_mean_te_weight": protocol.get("mean_te_weight"),
        "inner_max_recall_2m_drop": protocol.get("max_recall_2m_drop"),
        "inner_max_recall_5cm_drop": protocol.get("max_recall_5cm_drop"),
        "evaluation_camera_subset": record["evaluation_camera_subset"],
        "evaluation_camera_count": int(record["evaluation_camera_count"]),
        "evaluation_protocol_sha256": record["evaluation_protocol_sha256"],
        "used_within_initialization_fallback": bool(
            payload.get("used_strong_fallback", False)
        ),
    }


def select_initialization(candidates):
    if not candidates:
        raise ValueError("at least one initialization candidate is required")
    reference = candidates[0]
    protocol = (
        reference["evaluation_camera_subset"],
        reference["evaluation_camera_count"],
        reference["evaluation_protocol_sha256"],
    )
    if protocol[2] is None:
        raise ValueError("selection reports must pin an evaluation protocol hash")
    selection_protocol = (
        reference["inner_selection_mode"],
        reference["inner_primary_metric"],
        reference["inner_mean_te_weight"],
        reference["inner_max_recall_2m_drop"],
        reference["inner_max_recall_5cm_drop"],
    )
    for candidate in candidates[1:]:
        current = (
            candidate["evaluation_camera_subset"],
            candidate["evaluation_camera_count"],
            candidate["evaluation_protocol_sha256"],
        )
        if current != protocol:
            raise ValueError("initialization candidates use different validation protocols")
        current_selection_protocol = (
            candidate["inner_selection_mode"],
            candidate["inner_primary_metric"],
            candidate["inner_mean_te_weight"],
            candidate["inner_max_recall_2m_drop"],
            candidate["inner_max_recall_5cm_drop"],
        )
        if current_selection_protocol != selection_protocol:
            raise ValueError(
                "initialization candidates use different checkpoint selection protocols"
            )
    selected = min(
        candidates,
        key=lambda item: (
            float(item["primary_score"]),
            float(item["metrics"]["median_ae_deg"]),
            item["tag"],
        ),
    )
    return {
        "selection_protocol": {
            "subset": protocol[0],
            "query_count": protocol[1],
            "evaluation_protocol_sha256": protocol[2],
            "test_metrics_used": False,
            "checkpoint_selection_mode": selection_protocol[0],
            "primary_metric": selection_protocol[1],
            "mean_te_weight": selection_protocol[2],
            "max_recall_2m_drop": selection_protocol[3],
            "max_recall_5cm_drop": selection_protocol[4],
            "tie_breaker": "median_ae_deg",
        },
        "candidates": candidates,
        "selected_initialization": selected["tag"],
        "selected_state": selected["selected_state"],
        "selected_results_summary": selected["selected_results_summary"],
        "selected_metrics": selected["metrics"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select a clean-chain LaFGS initialization from held-out validation "
            "reports without reading test metrics."
        )
    )
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=2,
        metavar=("TAG", "SELECTION_REPORT"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = [
        load_initialization_candidate(tag, report)
        for tag, report in args.candidate
    ]
    selected = select_initialization(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
