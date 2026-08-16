#!/usr/bin/env python3
"""Choose baseline or revised render-only map from mapping PoseLib risk.

This is the closing decision of the render-only deployment-feedback loop.  It
uses one predeclared aggregate pose risk instead of a collection of independent
non-regression gates.  Test queries are neither loaded nor represented.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

import torch

from common.hashing import sha256_file
from map_learning.context_policy_oracle import pose_policy_loss


_SOURCE_PATHS = (
    "scripts/compare_rendered_track_pose_feedback.py",
    "map_learning/context_policy_oracle.py",
)


def _producer_identity() -> dict:
    repository = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("closed-loop selector worktree must be clean")
    return {
        "git_commit": commit,
        "worktree_clean": True,
        "source_sha256": {
            relative: sha256_file(repository / relative) for relative in _SOURCE_PATHS
        },
        "torch_version": torch.__version__,
    }


def _require_sha(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"{label} SHA differs: expected {expected}, got {actual}")
    return actual


def _load_report(path: Path, expected_sha256: str, label: str) -> tuple[dict, dict]:
    _require_sha(path, expected_sha256, label)
    report = json.loads(path.read_text())
    if report.get("schema") != "lafgs_rendered_track_full_mapping_loo_report":
        raise ValueError(f"{label} is not a rendered Track LOO report")
    if report.get("uses_source_mapping_rgb") is not False:
        raise ValueError(f"{label} used source mapping RGB")
    if report.get("uses_test_queries") is not False:
        raise ValueError(f"{label} used test queries")
    statistics_path = Path(str(report.get("statistics", ""))).resolve()
    statistics_sha256 = str(report.get("statistics_sha256", ""))
    _require_sha(statistics_path, statistics_sha256, f"{label} statistics")
    statistics = torch.load(statistics_path, map_location="cpu", weights_only=False)
    if statistics.get("schema") != "lafgs_rendered_track_full_mapping_loo_statistics":
        raise ValueError(f"{label} statistics schema differs")
    if (
        statistics.get("uses_source_mapping_rgb") is not False
        or statistics.get("uses_test_queries") is not False
    ):
        raise ValueError(f"{label} statistics are not mapping-only rendered evidence")
    return report, statistics


def _mean_pose_risk(statistics: dict) -> tuple[float, list[float]]:
    rows = list(statistics.get("queries", ()))
    if not rows:
        raise ValueError("pose feedback comparison requires query rows")
    risks = [
        pose_policy_loss(
            te_cm=float(row["te_cm"]),
            ae_deg=float(row["ae_deg"]),
            hypotheses=int(row["hypotheses"]),
        )
        for row in rows
    ]
    if not all(math.isfinite(value) for value in risks):
        raise ValueError("pose feedback comparison contains non-finite risk")
    return float(sum(risks) / len(risks)), risks


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if json.loads(temporary.read_text()).get("schema") != payload.get("schema"):
            raise RuntimeError("temporary closure result did not reload")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict:
    identity = _producer_identity()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    baseline_path = args.baseline_report.resolve()
    candidate_path = args.candidate_report.resolve()
    completion_path = args.revision_completion.resolve()
    baseline, baseline_statistics = _load_report(
        baseline_path, args.expected_baseline_report_sha256, "baseline report"
    )
    candidate, candidate_statistics = _load_report(
        candidate_path, args.expected_candidate_report_sha256, "candidate report"
    )
    completion_sha256 = _require_sha(
        completion_path,
        args.expected_revision_completion_sha256,
        "revision completion",
    )
    completion = json.loads(completion_path.read_text())
    if (
        completion.get("schema")
        != "lafgs_rendered_track_pose_feedback_revision_completion"
        or completion.get("uses_source_mapping_rgb") is not False
        or completion.get("uses_test_queries") is not False
    ):
        raise ValueError("revision completion schema or split differs")
    for role in ("map", "metric", "teacher"):
        path = Path(str(completion.get("outputs", {}).get(role, ""))).resolve()
        expected = str(completion.get("output_sha256", {}).get(role, ""))
        _require_sha(path, expected, f"revision {role}")
    candidate_inputs = candidate.get("inputs", {})
    candidate_hashes = candidate.get("input_sha256", {})
    for role in ("map", "metric", "teacher"):
        expected_path = Path(str(completion["outputs"][role])).resolve()
        if Path(str(candidate_inputs.get(role, ""))).resolve() != expected_path or str(
            candidate_hashes.get(role, "")
        ) != str(completion["output_sha256"][role]):
            raise ValueError(f"candidate report is not the completed revision {role}")
    baseline_names = [row["image_name"] for row in baseline_statistics["queries"]]
    candidate_names = [row["image_name"] for row in candidate_statistics["queries"]]
    if baseline_names != candidate_names:
        raise ValueError("baseline and candidate mapping query registries differ")
    baseline_risk, baseline_query_risk = _mean_pose_risk(baseline_statistics)
    candidate_risk, candidate_query_risk = _mean_pose_risk(candidate_statistics)
    selected_candidate = candidate_risk < baseline_risk
    selected = candidate if selected_candidate else baseline
    selected_label = "pose_feedback_revision" if selected_candidate else "v1_4_baseline"
    result = {
        "schema": "lafgs_rendered_track_closed_loop_selection",
        "version": 1,
        "valid": True,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "producer_identity": identity,
        "decision": (
            "SELECT_POSE_FEEDBACK_REVISION"
            if selected_candidate
            else "RETAIN_V1_4_BASELINE"
        ),
        "selection_objective": {
            "name": "mean_mapping_pose_policy_loss",
            "translation_scale_cm": 5.0,
            "rotation_scale_deg": 5.0,
            "catastrophe_cm": 100.0,
            "catastrophe_weight": 2.0,
            "hypothesis_scale": 1000.0,
            "hypothesis_weight": 0.05,
            "rule": "candidate_mean_risk_strictly_lower",
            "baseline_risk": baseline_risk,
            "candidate_risk": candidate_risk,
            "absolute_improvement": baseline_risk - candidate_risk,
            "improved_query_count": int(
                sum(
                    candidate_value < baseline_value
                    for baseline_value, candidate_value in zip(
                        baseline_query_risk, candidate_query_risk
                    )
                )
            ),
            "regressed_query_count": int(
                sum(
                    candidate_value > baseline_value
                    for baseline_value, candidate_value in zip(
                        baseline_query_risk, candidate_query_risk
                    )
                )
            ),
        },
        "baseline_summary": baseline["summary"],
        "candidate_summary": candidate["summary"],
        "selected_label": selected_label,
        "selected_artifacts": {
            "map": selected["inputs"]["map"],
            "metric": selected["inputs"]["metric"],
            "teacher": selected["inputs"]["teacher"],
            "query_cache": selected["inputs"]["query_cache"],
            "scene_calibration": selected["inputs"]["scene_calibration"],
        },
        "selected_artifact_sha256": {
            "map": selected["input_sha256"]["map"],
            "metric": selected["input_sha256"]["metric"],
            "teacher": selected["input_sha256"]["teacher"],
            "query_cache": selected["input_sha256"]["query_cache"],
            "scene_calibration": selected["input_sha256"]["scene_calibration"],
        },
        "inputs": {
            "baseline_report": str(baseline_path),
            "candidate_report": str(candidate_path),
            "revision_completion": str(completion_path),
        },
        "input_sha256": {
            "baseline_report": args.expected_baseline_report_sha256,
            "candidate_report": args.expected_candidate_report_sha256,
            "revision_completion": completion_sha256,
        },
        "authorization": {
            "mapping_selection_complete": True,
            "test_may_be_used_only_for_frozen_final_evaluation": True,
            "test_may_change_map_or_selection": False,
            "mixed_pipeline_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(result, output)
    if _producer_identity() != identity:
        output.unlink(missing_ok=True)
        raise RuntimeError("closed-loop selector producer identity changed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--expected-baseline-report-sha256", required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--expected-candidate-report-sha256", required=True)
    parser.add_argument("--revision-completion", type=Path, required=True)
    parser.add_argument("--expected-revision-completion-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
