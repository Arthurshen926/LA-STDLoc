#!/usr/bin/env python3
"""Aggregate the five-scene pure-base Cambridge online-feedback experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from common.hashing import sha256_file


ARMS = ("baseline", "topk_min64", "lgcv_min64")
REQUIRED_ARMS = ("baseline", "topk_min64")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/v23_cambridge_online_feedback.json"),
    )
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: 0.0 for key in ("mean", "median", "p90", "p95", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _r5(row: dict) -> bool:
    return bool(
        float(row["translation_error_cm"]) < 5.0
        and float(row["rotation_error_deg"]) < 5.0
    )


def _pose_summary(rows: list[dict]) -> dict:
    translation = [float(row["translation_error_cm"]) for row in rows]
    rotation = [float(row["rotation_error_deg"]) for row in rows]
    return {
        "query_count": len(rows),
        "translation_error_cm": _distribution(translation),
        "rotation_error_deg": _distribution(rotation),
        "r5_success_count": sum(_r5(row) for row in rows),
        "r5_percent": 100.0 * sum(_r5(row) for row in rows) / len(rows),
        "catastrophic_100cm_count": sum(value >= 100.0 for value in translation),
    }


def _timing_summary(rows: list[dict]) -> dict:
    fields = (
        "frontend_ms",
        "matching_ms",
        "ransac_ms",
        "feedback_geometry_ms",
        "feedback_ransac_ms",
        "total_ms",
    )
    output = {
        field: _distribution([float(row.get(field, 0.0)) for row in rows])
        for field in fields
    }
    output["feedback_geometry_ms_when_eligible"] = _distribution(
        [
            float(row.get("feedback_geometry_ms", 0.0))
            for row in rows
            if bool(row.get("sparse_feedback_eligible", 0))
        ]
    )
    output["feedback_ransac_ms_when_run"] = _distribution(
        [
            float(row.get("feedback_ransac_ms", 0.0))
            for row in rows
            if bool(row.get("sparse_feedback_gate_passed", 0))
        ]
    )
    return output


def _paired_summary(baseline: list[dict], candidate: list[dict]) -> dict:
    if [row["image_name"] for row in baseline] != [
        row["image_name"] for row in candidate
    ]:
        raise ValueError("V23 paired query registry differs")
    delta_te = [
        float(right["translation_error_cm"]) - float(left["translation_error_cm"])
        for left, right in zip(baseline, candidate)
    ]
    delta_re = [
        float(right["rotation_error_deg"]) - float(left["rotation_error_deg"])
        for left, right in zip(baseline, candidate)
    ]
    return {
        "translation_error_delta_cm": _distribution(delta_te),
        "rotation_error_delta_deg": _distribution(delta_re),
        "translation_improved_query_count": sum(value < -1e-9 for value in delta_te),
        "translation_worsened_query_count": sum(value > 1e-9 for value in delta_te),
        "r5_gain_count": sum(
            (not _r5(left)) and _r5(right)
            for left, right in zip(baseline, candidate)
        ),
        "r5_loss_count": sum(
            _r5(left) and (not _r5(right))
            for left, right in zip(baseline, candidate)
        ),
    }


def _feedback_summary(rows: list[dict]) -> dict:
    count = len(rows)
    eligible = sum(bool(row.get("sparse_feedback_eligible", 0)) for row in rows)
    gated = sum(bool(row.get("sparse_feedback_gate_passed", 0)) for row in rows)
    accepted = sum(bool(row.get("sparse_feedback_accepted", 0)) for row in rows)
    proposed = sum(int(row.get("sparse_feedback_proposed_rows", 0)) for row in rows)
    supported = sum(int(row.get("sparse_feedback_supported_rows", 0)) for row in rows)
    return {
        "eligible_query_count": eligible,
        "eligible_query_percent": 100.0 * eligible / count,
        "gate_passed_query_count": gated,
        "gate_passed_query_percent": 100.0 * gated / count,
        "accepted_query_count": accepted,
        "accepted_query_percent": 100.0 * accepted / count,
        "proposed_row_count": proposed,
        "supported_row_count": supported,
        "supported_row_fraction": float(supported / max(proposed, 1)),
        "pose_solve_count_per_query_mean": 1.0 + float(gated / count),
    }


def _apply_test_calibrated_acceptance(
    baseline: list[dict], candidate: list[dict], config: dict
) -> tuple[list[dict], list[bool]]:
    if [row["image_name"] for row in baseline] != [
        row["image_name"] for row in candidate
    ]:
        raise ValueError("V23 calibrated-gate query registry differs")
    selected = []
    accepted = []
    for baseline_row, candidate_row in zip(baseline, candidate):
        gain = int(candidate_row["inliers"]) - int(baseline_row["inliers"])
        relative = float(gain / max(int(baseline_row["inliers"]), 1))
        baseline_pose = np.asarray(baseline_row["pose_w2c"], dtype=np.float64)
        candidate_pose = np.asarray(candidate_row["pose_w2c"], dtype=np.float64)
        baseline_center = np.linalg.inv(baseline_pose)[:3, 3]
        candidate_center = np.linalg.inv(candidate_pose)[:3, 3]
        pose_translation_cm = float(
            np.linalg.norm(candidate_center - baseline_center) * 100.0
        )
        relative_rotation = candidate_pose[:3, :3] @ baseline_pose[:3, :3].T
        rotation_cosine = float(
            np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
        )
        pose_rotation_deg = float(np.degrees(np.arccos(rotation_cosine)))
        use_candidate = bool(
            gain >= int(config["minimum_candidate_inlier_gain"])
            and relative >= float(config["minimum_candidate_relative_inlier_gain"])
            and int(candidate_row["ransac_iterations"])
            <= int(config["maximum_candidate_ransac_iterations"])
            and pose_translation_cm
            <= float(config["maximum_pose_update_translation_cm"])
            and pose_rotation_deg <= float(config["maximum_pose_update_rotation_deg"])
        )
        selected.append(candidate_row if use_candidate else baseline_row)
        accepted.append(use_candidate)
    return selected, accepted


def _load_evaluation(path: Path, *, expected_map: Path, expected_metric: Path) -> dict:
    summary_path = path / "summary.json"
    rows_path = path / "results.json"
    contract_path = path / "deployment_contract.json"
    summary = json.loads(summary_path.read_text())
    rows = json.loads(rows_path.read_text())
    contract = json.loads(contract_path.read_text())
    artifacts = contract["input_artifacts"]
    if (
        Path(artifacts["map"]["path"]).resolve() != expected_map.resolve()
        or Path(artifacts["descriptor_state"]["path"]).resolve()
        != expected_metric.resolve()
        or artifacts["map"]["sha256"] != sha256_file(expected_map)
        or artifacts["descriptor_state"]["sha256"] != sha256_file(expected_metric)
        or int(summary["query_count"]) != len(rows)
        or contract["evaluated_split"] != "test"
    ):
        raise ValueError(f"V23 evaluation lineage differs: {path}")
    return {
        "summary": summary,
        "rows": rows,
        "contract": contract,
        "sources": {
            name: {
                "path": str(source.resolve()),
                "sha256": sha256_file(source),
                "size_bytes": int(source.stat().st_size),
            }
            for name, source in {
                "summary": summary_path,
                "results": rows_path,
                "deployment_contract": contract_path,
            }.items()
        },
    }


def _validate_base_report(path: Path, map_path: Path, metric_path: Path) -> dict:
    report = json.loads(path.read_text())
    scope = report["scientific_scope"]
    if not (
        report["schema"] == "lafgs_rendered_rgb_only_track_probe_report"
        and scope["test_queries_used_for_map_construction"] is False
        and scope["mapping_source_rgb_loaded"] is False
        and scope["mapping_source_rgb_used"] is False
        and scope["gaussian_rendered_rgb_used"] is True
        and Path(report["artifacts"]["anchor_map"]).resolve() == map_path.resolve()
        and Path(report["artifacts"]["identity_metric"]).resolve()
        == metric_path.resolve()
        and report["artifacts_sha256"]["anchor_map"] == sha256_file(map_path)
        and report["artifacts_sha256"]["identity_metric"]
        == sha256_file(metric_path)
    ):
        raise ValueError(f"V23 base-map scientific scope differs: {path}")
    return {
        "selected_anchor_count": int(report["selected_map_track_count"]),
        "mapping_render_query_count": int(report["query_count"]),
        "scientific_scope": scope,
        "source": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": int(path.stat().st_size),
        },
    }


def _atomic_json_fresh(payload: dict, output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    try:
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    args = _args()
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text())
    if (
        protocol.get("schema") != "lafgs_v23_cambridge_online_feedback_protocol"
        or protocol.get("version") != 1
        or set(protocol.get("scenes", {}))
        != {"GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"}
        or protocol["scientific_scope"]["offline_feedback_enabled"] is not False
    ):
        raise ValueError("V23 protocol is invalid")

    scene_outputs = {}
    pooled = {arm: [] for arm in (*ARMS, "topk_calibrated_gate")}
    paired_baseline = {arm: [] for arm in ARMS[1:]}
    paired_baseline["topk_calibrated_gate"] = []
    pooled_calibrated_acceptance: list[bool] = []
    root = args.evaluation_root.expanduser().resolve()
    for scene, scene_input in sorted(protocol["scenes"].items()):
        map_path = Path(scene_input["map"])
        metric_path = Path(scene_input["identity_metric"])
        base = _validate_base_report(
            Path(scene_input["map_report"]), map_path, metric_path
        )
        evaluations = {
            arm: _load_evaluation(
                root / scene / f"{arm}_seed2026",
                expected_map=map_path,
                expected_metric=metric_path,
            )
            for arm in ARMS
            if (root / scene / f"{arm}_seed2026" / "summary.json").is_file()
        }
        if any(arm not in evaluations for arm in REQUIRED_ARMS):
            raise ValueError(f"V23 {scene} misses a required evaluation arm")
        baseline_names = [row["image_name"] for row in evaluations["baseline"]["rows"]]
        if any(
            [row["image_name"] for row in evaluations[arm]["rows"]] != baseline_names
            for arm in evaluations
            if arm != "baseline"
        ):
            raise ValueError(f"V23 {scene} query registry differs")
        for arm in evaluations:
            pooled[arm].extend(evaluations[arm]["rows"])
            if arm != "baseline":
                paired_baseline[arm].extend(evaluations["baseline"]["rows"])
        calibrated_rows, calibrated_acceptance = _apply_test_calibrated_acceptance(
            evaluations["baseline"]["rows"],
            evaluations["topk_min64"]["rows"],
            protocol["test_calibrated_online_acceptance"],
        )
        pooled["topk_calibrated_gate"].extend(calibrated_rows)
        pooled_calibrated_acceptance.extend(calibrated_acceptance)
        paired_baseline["topk_calibrated_gate"].extend(
            evaluations["baseline"]["rows"]
        )
        scene_outputs[scene] = {
            "base_map": base,
            "arms": {
                arm: {
                    "pose": _pose_summary(evaluations[arm]["rows"]),
                    "timing_ms": _timing_summary(evaluations[arm]["rows"]),
                    "feedback": _feedback_summary(evaluations[arm]["rows"]),
                    "paired_vs_baseline": (
                        None
                        if arm == "baseline"
                        else _paired_summary(
                            evaluations["baseline"]["rows"],
                            evaluations[arm]["rows"],
                        )
                    ),
                    "sources": evaluations[arm]["sources"],
                }
                for arm in evaluations
            },
        }
        scene_outputs[scene]["arms"]["topk_calibrated_gate"] = {
            "pose": _pose_summary(calibrated_rows),
            "timing_ms": _timing_summary(evaluations["topk_min64"]["rows"]),
            "feedback": {
                **_feedback_summary(evaluations["topk_min64"]["rows"]),
                "accepted_query_count": sum(calibrated_acceptance),
                "accepted_query_percent": (
                    100.0 * sum(calibrated_acceptance) / len(calibrated_acceptance)
                ),
                "acceptance_configuration": protocol[
                    "test_calibrated_online_acceptance"
                ],
            },
            "paired_vs_baseline": _paired_summary(
                evaluations["baseline"]["rows"], calibrated_rows
            ),
            "sources": {
                "baseline": evaluations["baseline"]["sources"],
                "raw_topk": evaluations["topk_min64"]["sources"],
            },
        }

    pooled_micro = {
        arm: {
            "pose": _pose_summary(pooled[arm]),
            "timing_ms": _timing_summary(pooled[arm]),
            "feedback": _feedback_summary(pooled[arm]),
            "paired_vs_baseline": (
                None
                if arm == "baseline"
                else _paired_summary(paired_baseline[arm], pooled[arm])
            ),
        }
        for arm in (*ARMS, "topk_calibrated_gate")
        if pooled[arm]
    }
    calibrated_feedback = _feedback_summary(pooled["topk_min64"])
    calibrated_feedback.update(
        {
            "accepted_query_count": sum(pooled_calibrated_acceptance),
            "accepted_query_percent": (
                100.0
                * sum(pooled_calibrated_acceptance)
                / len(pooled_calibrated_acceptance)
            ),
            "acceptance_configuration": protocol[
                "test_calibrated_online_acceptance"
            ],
        }
    )
    pooled_micro["topk_calibrated_gate"]["timing_ms"] = _timing_summary(
        pooled["topk_min64"]
    )
    pooled_micro["topk_calibrated_gate"]["feedback"] = calibrated_feedback

    output = {
        "schema": "lafgs_v23_cambridge_online_feedback_aggregate",
        "version": 1,
        "protocol": protocol,
        "protocol_source": {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
            "size_bytes": int(protocol_path.stat().st_size),
        },
        "current_implementation_sources": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py",
                Path(__file__).resolve().parents[1]
                / "localization"
                / "localizer.py",
                Path(__file__).resolve().parents[1] / "localization" / "matcher.py",
                Path(__file__).resolve().parents[1]
                / "map_learning"
                / "v21_topk_geometric_feedback.py",
                Path(__file__).resolve().parents[1]
                / "map_learning"
                / "v22_sparse_lgcv_feedback.py",
            )
        ],
        "current_implementation_sources_are_not_claimed_as_historical_raw_run_hashes": (
            True
        ),
        "evaluation_root": str(root),
        "scene_count": len(scene_outputs),
        "scenes": scene_outputs,
        "pooled_micro": pooled_micro,
        "writes_map_or_metric": False,
        "offline_feedback_consumed": False,
        "ground_truth_used_for_online_selection": False,
    }
    output_path = _atomic_json_fresh(output, args.output)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(output_path)
    print(digest)


if __name__ == "__main__":
    main()
