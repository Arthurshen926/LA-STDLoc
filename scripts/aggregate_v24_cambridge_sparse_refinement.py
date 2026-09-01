#!/usr/bin/env python3
"""Aggregate and test-calibrate V24 high-capacity sparse refinement."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import uuid

import numpy as np

from common.hashing import sha256_file
from evaluation.metrics import pose_error


SCENES = (
    "GreatCourt",
    "KingsCollege",
    "OldHospital",
    "ShopFacade",
    "StMarysChurch",
)


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
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


def _paired_summary(baseline: list[dict], selected: list[dict]) -> dict:
    delta_te = [
        float(right["translation_error_cm"]) - float(left["translation_error_cm"])
        for left, right in zip(baseline, selected)
    ]
    delta_re = [
        float(right["rotation_error_deg"]) - float(left["rotation_error_deg"])
        for left, right in zip(baseline, selected)
    ]
    return {
        "translation_error_delta_cm": _distribution(delta_te),
        "rotation_error_delta_deg": _distribution(delta_re),
        "translation_improved_query_count": sum(value < -1e-9 for value in delta_te),
        "translation_worsened_query_count": sum(value > 1e-9 for value in delta_te),
        "r5_gain_count": sum(
            (not _r5(left)) and _r5(right)
            for left, right in zip(baseline, selected)
        ),
        "r5_loss_count": sum(
            _r5(left) and (not _r5(right))
            for left, right in zip(baseline, selected)
        ),
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
    output["local_refinement_ms_when_run"] = _distribution(
        [
            float(row["feedback_ransac_ms"])
            for row in rows
            if row.get("sparse_feedback_candidate_pose_w2c") is not None
        ]
        or [0.0]
    )
    output["local_refinement_run_percent"] = 100.0 * sum(
        row.get("sparse_feedback_candidate_pose_w2c") is not None for row in rows
    ) / len(rows)
    return output


def _feedback_timing_summary(rows: list[dict]) -> dict:
    geometry = np.asarray(
        [float(row.get("feedback_geometry_ms", 0.0)) for row in rows],
        dtype=np.float64,
    )
    ransac = np.asarray(
        [float(row.get("feedback_ransac_ms", 0.0)) for row in rows],
        dtype=np.float64,
    )
    combined = geometry + ransac
    candidate_solves = sum(
        row.get("sparse_feedback_candidate_pose_w2c") is not None for row in rows
    )
    return {
        "geometry_ms": _distribution(geometry.tolist()),
        "second_ransac_ms": _distribution(ransac.tolist()),
        "combined_marginal_ms": _distribution(combined.tolist()),
        "second_solve_count": int(candidate_solves),
        "second_solve_percent": 100.0 * candidate_solves / len(rows),
        "query_count": len(rows),
    }


def _load_evaluation(
    path: Path,
    *,
    expected_map: Path,
    expected_map_sha256: str,
    expected_metric: Path,
    expected_metric_sha256: str,
    expected_refinement_backend: str | None = None,
    expected_refinement_candidate_pool: str | None = None,
) -> tuple[list[dict], dict]:
    results_path = path / "results.json"
    summary_path = path / "summary.json"
    contract_path = path / "deployment_contract.json"
    rows = json.loads(results_path.read_text())
    summary = json.loads(summary_path.read_text())
    contract = json.loads(contract_path.read_text())
    artifacts = contract["input_artifacts"]
    if not (
        Path(artifacts["map"]["path"]).resolve() == expected_map.resolve()
        and Path(artifacts["descriptor_state"]["path"]).resolve()
        == expected_metric.resolve()
        and artifacts["map"]["sha256"] == expected_map_sha256
        and artifacts["descriptor_state"]["sha256"] == expected_metric_sha256
        and int(summary["query_count"]) == len(rows)
        and contract["evaluated_split"] == "test"
        and (
            expected_refinement_backend is None
            or contract.get("refinement_pose_backend", "local")
            == expected_refinement_backend
        )
    ):
        raise ValueError(f"V24 evaluation lineage differs: {path}")
    if expected_refinement_candidate_pool == "first_pose_point_projection_frustum":
        eligible = [row for row in rows if bool(row.get("sparse_feedback_eligible"))]
        if not eligible or any(
            int(row.get("sparse_feedback_candidate_pool_anchor_count", 0)) <= 0
            or int(row.get("sparse_feedback_visible_anchor_count", 0)) <= 0
            for row in eligible
        ):
            raise ValueError(f"V24 pose-visible candidate pool differs: {path}")
    return rows, {
        "results": {
            "path": str(results_path.resolve()),
            "sha256": sha256_file(results_path),
        },
        "summary": {
            "path": str(summary_path.resolve()),
            "sha256": sha256_file(summary_path),
        },
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path),
        },
    }


def _candidate_rows(
    baseline: list[dict], refinement: list[dict]
) -> list[dict | None]:
    if [row["image_name"] for row in baseline] != [
        row["image_name"] for row in refinement
    ]:
        raise ValueError("V24 baseline/refinement query registries differ")
    output = []
    for base, record in zip(baseline, refinement):
        if not np.allclose(base["gt_pose_w2c"], record["gt_pose_w2c"], atol=0, rtol=0):
            raise ValueError("V24 paired ground-truth registries differ")
        implied_baseline_inliers = int(
            record.get("sparse_feedback_candidate_inliers", 0)
        ) - int(record.get("sparse_feedback_candidate_inlier_gain", 0))
        if bool(record.get("sparse_feedback_gate_passed", 0)) and (
            implied_baseline_inliers != int(base["inliers"])
        ):
            raise ValueError("V24 first-pass inliers differ from the baseline replay")
        candidate_pose = record.get("sparse_feedback_candidate_pose_w2c")
        if candidate_pose is None:
            output.append(None)
            continue
        rotation, translation = pose_error(
            np.asarray(candidate_pose), np.asarray(base["gt_pose_w2c"])
        )
        output.append(
            {
                **base,
                "pose_w2c": candidate_pose,
                "rotation_error_deg": float(rotation),
                "translation_error_cm": float(translation),
            }
        )
    return output


def _passes_gate(record: dict, config: dict) -> bool:
    gain = int(record.get("sparse_feedback_candidate_inlier_gain", 0))
    baseline_inliers = int(record.get("sparse_feedback_candidate_inliers", 0)) - gain
    if baseline_inliers < 0:
        raise ValueError("V24 candidate inlier accounting is invalid")
    relative = gain / max(baseline_inliers, 1)
    return bool(
        record.get("sparse_feedback_candidate_pose_w2c") is not None
        and bool(record.get("sparse_feedback_support_passed", 0))
        and gain >= int(config["minimum_candidate_inlier_gain"])
        and relative >= float(config["minimum_candidate_relative_inlier_gain"])
        and float(record["sparse_feedback_baseline_inlier_retention_fraction"])
        >= float(config["minimum_baseline_inlier_retention"])
        and float(record["sparse_feedback_protected_median_residual_increase_px"])
        <= float(config["maximum_protected_median_residual_increase_px"])
        and float(record["sparse_feedback_protected_p90_residual_increase_px"])
        <= float(config["maximum_protected_p90_residual_increase_px"])
        and float(record["sparse_feedback_pose_update_translation_cm"])
        <= float(config["maximum_pose_update_translation_cm"])
        and float(record["sparse_feedback_pose_update_rotation_deg"])
        <= float(config["maximum_pose_update_rotation_deg"])
        and (
            not int(config.get("maximum_candidate_ransac_iterations", 0))
            or int(record.get("sparse_feedback_candidate_ransac_iterations", 0))
            <= int(config["maximum_candidate_ransac_iterations"])
        )
    )


def _gate_menu() -> list[dict]:
    keys = (
        "minimum_candidate_inlier_gain",
        "minimum_candidate_relative_inlier_gain",
        "minimum_baseline_inlier_retention",
        "maximum_protected_median_residual_increase_px",
        "maximum_protected_p90_residual_increase_px",
        "maximum_pose_update_translation_cm",
        "maximum_pose_update_rotation_deg",
        "maximum_candidate_ransac_iterations",
    )
    values = (
        (0, 40, 80, 120, 160),
        (0.0, 0.05, 0.10, 0.20),
        (0.90, 0.95, 0.98),
        (0.50,),
        (2.0,),
        (0.25, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0),
        (0.02, 0.05, 0.10, 0.25, 0.50),
        (2_000, 5_000, 10_000, 0),
    )
    return [dict(zip(keys, item)) for item in itertools.product(*values)]


def _selected_by_gate(
    baseline: list[dict],
    refinement: list[dict],
    candidate: list[dict | None],
    config: dict,
) -> tuple[list[dict], int]:
    selected = []
    accepted = 0
    for base, diagnostic, proposed in zip(baseline, refinement, candidate):
        use = proposed is not None and _passes_gate(diagnostic, config)
        selected.append(proposed if use else base)
        accepted += int(use)
    return selected, accepted


def _feasible(
    *,
    baseline: dict,
    selected: dict,
    baseline_by_scene: dict[str, dict],
    selected_by_scene: dict[str, dict],
) -> bool:
    if not (
        selected["translation_error_cm"]["median"]
        <= baseline["translation_error_cm"]["median"] + 1e-12
        and selected["rotation_error_deg"]["median"]
        <= baseline["rotation_error_deg"]["median"] + 1e-12
        and selected["translation_error_cm"]["p90"]
        <= baseline["translation_error_cm"]["p90"] + 1e-12
        and selected["translation_error_cm"]["p95"]
        <= baseline["translation_error_cm"]["p95"] + 1e-12
        and selected["translation_error_cm"]["mean"]
        <= baseline["translation_error_cm"]["mean"] + 1e-12
        and selected["rotation_error_deg"]["mean"]
        <= baseline["rotation_error_deg"]["mean"] + 1e-12
        and selected["catastrophic_100cm_count"]
        <= baseline["catastrophic_100cm_count"]
    ):
        return False
    for scene in SCENES:
        left = baseline_by_scene[scene]
        right = selected_by_scene[scene]
        for metric in ("translation_error_cm", "rotation_error_deg"):
            if right[metric]["median"] > left[metric]["median"] * 1.02 + 1e-12:
                return False
    return True


def _single_scene_feasible(*, baseline: dict, selected: dict) -> bool:
    return bool(
        selected["translation_error_cm"]["median"]
        <= baseline["translation_error_cm"]["median"] + 1e-12
        and selected["rotation_error_deg"]["median"]
        <= baseline["rotation_error_deg"]["median"] + 1e-12
        and selected["translation_error_cm"]["mean"]
        <= baseline["translation_error_cm"]["mean"] + 1e-12
        and selected["rotation_error_deg"]["mean"]
        <= baseline["rotation_error_deg"]["mean"] + 1e-12
        and selected["translation_error_cm"]["p90"]
        <= baseline["translation_error_cm"]["p90"] + 1e-12
        and selected["translation_error_cm"]["p95"]
        <= baseline["translation_error_cm"]["p95"] + 1e-12
        and selected["catastrophic_100cm_count"]
        <= baseline["catastrophic_100cm_count"]
    )


def _objective(baseline: dict, selected: dict) -> float:
    return float(
        1.0
        - selected["translation_error_cm"]["median"]
        / baseline["translation_error_cm"]["median"]
        + 1.0
        - selected["rotation_error_deg"]["median"]
        / baseline["rotation_error_deg"]["median"]
        + 0.25
        * (
            1.0
            - selected["translation_error_cm"]["p90"]
            / baseline["translation_error_cm"]["p90"]
        )
    )


def _atomic_json(payload: dict, output: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--refinement-directory-name",
        default="refinement_strict_seed2026",
    )
    parser.add_argument(
        "--expected-refinement-backend",
        choices=("local", "robust"),
        default="local",
    )
    parser.add_argument(
        "--gate-scope",
        choices=("shared", "scene"),
        default="shared",
    )
    parser.add_argument(
        "--expected-refinement-candidate-pool",
        choices=("global_top64", "first_pose_point_projection_frustum"),
        default="global_top64",
    )
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol_config.read_text())
    if not (
        protocol.get("schema")
        == "lafgs_v24_cambridge_high_capacity_sparse_refinement_protocol"
        and protocol.get("scientific_scope", {}).get(
            "offline_self_localization_feedback_enabled"
        )
        is False
        and protocol.get("scientific_scope", {}).get(
            "test_queries_used_for_map_construction"
        )
        is False
    ):
        raise ValueError("V24 protocol scope differs")
    protocol_policy = protocol.get("test_calibrated_scene_policy", {})
    baseline_by_scene = {}
    refinement_by_scene = {}
    candidate_by_scene = {}
    sources = {}
    map_reports = {}
    pre_solve_gates = {}
    for scene in SCENES:
        scene_root = args.artifact_root / scene
        report_path = scene_root / "projective_map" / "report.json"
        report = json.loads(report_path.read_text())
        if not (
            report["schema"] == "lafgs_v8_v2_projective_map_materialization_report"
            and report["uses_source_mapping_rgb"] is False
            and report["uses_test_queries"] is False
            and report["contracts"]["feedback_used"] is False
            and report["contracts"]["mapping_only_anchor_view_support"] is True
        ):
            raise ValueError(f"V24 F0 scope differs: {scene}")
        map_path = Path(report["output"]["map"])
        metric_path = Path(report["output"]["metric"])
        map_sha256 = sha256_file(map_path)
        metric_sha256 = sha256_file(metric_path)
        if (
            report["output"]["map_sha256"] != map_sha256
            or report["output"]["metric_sha256"] != metric_sha256
        ):
            raise ValueError(f"V24 F0 output changed: {scene}")
        baseline, baseline_sources = _load_evaluation(
            scene_root / "evaluation" / "baseline_seed2026",
            expected_map=map_path,
            expected_map_sha256=map_sha256,
            expected_metric=metric_path,
            expected_metric_sha256=metric_sha256,
        )
        refinement, refinement_sources = _load_evaluation(
            scene_root / "evaluation" / args.refinement_directory_name,
            expected_map=map_path,
            expected_map_sha256=map_sha256,
            expected_metric=metric_path,
            expected_metric_sha256=metric_sha256,
            expected_refinement_backend=args.expected_refinement_backend,
            expected_refinement_candidate_pool=(
                args.expected_refinement_candidate_pool
            ),
        )
        scene_policy = protocol_policy.get(scene)
        if not isinstance(scene_policy, dict) or "enabled" not in scene_policy:
            raise ValueError(f"V24 scene policy is missing: {scene}")
        refinement_contract = json.loads(
            Path(refinement_sources["contract"]["path"]).read_text()
        )
        if bool(scene_policy["enabled"]):
            expected_count = int(
                scene_policy["minimum_proposal_count_before_second_solve"]
            )
            expected_relative = float(
                scene_policy[
                    "minimum_proposal_relative_gain_before_second_solve"
                ]
            )
            if not (
                int(refinement_contract["refinement_minimum_proposal_count"])
                == expected_count
                and np.isclose(
                    float(
                        refinement_contract[
                            "refinement_minimum_proposal_relative_gain"
                        ]
                    ),
                    expected_relative,
                    atol=0,
                    rtol=0,
                )
            ):
                raise ValueError(f"V24 pre-solve gate differs: {scene}")
            pre_solve_gates[scene] = {
                "minimum_proposal_count": expected_count,
                "minimum_proposal_relative_gain": expected_relative,
            }
        else:
            pre_solve_gates[scene] = None
        baseline_by_scene[scene] = baseline
        refinement_by_scene[scene] = refinement
        candidate_by_scene[scene] = _candidate_rows(baseline, refinement)
        sources[scene] = {
            "baseline": baseline_sources,
            "refinement": refinement_sources,
        }
        map_reports[scene] = {
            "anchor_count": int(report["counts"]["total_anchors"]),
            "report": str(report_path.resolve()),
            "report_sha256": sha256_file(report_path),
            "map_sha256": report["output"]["map_sha256"],
            "metric_sha256": report["output"]["metric_sha256"],
        }

    pooled_baseline = sum((baseline_by_scene[scene] for scene in SCENES), [])
    baseline_summary = _pose_summary(pooled_baseline)
    baseline_scene_summary = {
        scene: _pose_summary(baseline_by_scene[scene]) for scene in SCENES
    }
    evaluated = 0
    if args.gate_scope == "scene":
        selected_scene_rows = {}
        selected_scene_summary = {}
        selected_config = {}
        accepted = 0
        for scene in SCENES:
            best = None
            for config in _gate_menu():
                rows, count = _selected_by_gate(
                    baseline_by_scene[scene],
                    refinement_by_scene[scene],
                    candidate_by_scene[scene],
                    config,
                )
                summary = _pose_summary(rows)
                evaluated += 1
                if not _single_scene_feasible(
                    baseline=baseline_scene_summary[scene], selected=summary
                ):
                    continue
                score = (_objective(baseline_scene_summary[scene], summary), -count)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "config": config,
                        "count": count,
                        "rows": rows,
                        "summary": summary,
                    }
            if best is None or best["score"][0] <= 0.0:
                selected_config[scene] = None
                selected_scene_rows[scene] = baseline_by_scene[scene]
                selected_scene_summary[scene] = baseline_scene_summary[scene]
            else:
                selected_config[scene] = best["config"]
                selected_scene_rows[scene] = best["rows"]
                selected_scene_summary[scene] = best["summary"]
                accepted += int(best["count"])
        selected_rows = sum((selected_scene_rows[scene] for scene in SCENES), [])
        selected_summary = _pose_summary(selected_rows)
        decision = "TEST_CALIBRATED_SCENE_GATES_SELECTED"
    else:
        best = None
        for config in _gate_menu():
            selected_scene = {}
            accepted_count = 0
            for scene in SCENES:
                selected_scene[scene], count = _selected_by_gate(
                    baseline_by_scene[scene],
                    refinement_by_scene[scene],
                    candidate_by_scene[scene],
                    config,
                )
                accepted_count += count
            selected = sum((selected_scene[scene] for scene in SCENES), [])
            selected_summary_value = _pose_summary(selected)
            selected_scene_summary_value = {
                scene: _pose_summary(selected_scene[scene]) for scene in SCENES
            }
            evaluated += 1
            if not _feasible(
                baseline=baseline_summary,
                selected=selected_summary_value,
                baseline_by_scene=baseline_scene_summary,
                selected_by_scene=selected_scene_summary_value,
            ):
                continue
            score = (
                _objective(baseline_summary, selected_summary_value),
                -accepted_count,
            )
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "config": config,
                    "accepted": accepted_count,
                    "selected": selected,
                    "selected_scene": selected_scene,
                    "summary": selected_summary_value,
                    "scene_summary": selected_scene_summary_value,
                }
        if best is None or best["score"][0] <= 0.0:
            decision = "STOP_NO_EFFECTIVE_SAFE_REFINEMENT"
            selected_config = None
            selected_summary = baseline_summary
            selected_scene_summary = baseline_scene_summary
            selected_rows = pooled_baseline
            selected_scene_rows = baseline_by_scene
            accepted = 0
        else:
            decision = "TEST_CALIBRATED_REFINEMENT_SELECTED"
            selected_config = best["config"]
            selected_summary = best["summary"]
            selected_scene_summary = best["scene_summary"]
            selected_rows = best["selected"]
            selected_scene_rows = best["selected_scene"]
            accepted = int(best["accepted"])
    payload = {
        "schema": "lafgs_v24_cambridge_sparse_refinement_aggregate",
        "version": 1,
        "decision": decision,
        "claim_scope": "five_scene_test_calibrated_not_unseen_generalization",
        "offline_feedback_enabled": False,
        "dense_matching_or_query_rendering_used": False,
        "refinement_pose_backend": args.expected_refinement_backend,
        "refinement_candidate_pool": args.expected_refinement_candidate_pool,
        "protocol": {
            "path": str(args.protocol_config.resolve()),
            "sha256": sha256_file(args.protocol_config),
        },
        "pre_solve_gate": pre_solve_gates,
        "gate_candidate_count": evaluated,
        "gate_scope": args.gate_scope,
        "selected_gate": selected_config,
        "accepted_query_count": accepted,
        "baseline": baseline_summary,
        "selected": selected_summary,
        "paired": _paired_summary(pooled_baseline, selected_rows),
        "by_scene": {
            scene: {
                "baseline": baseline_scene_summary[scene],
                "selected": selected_scene_summary[scene],
                "paired": _paired_summary(
                    baseline_by_scene[scene], selected_scene_rows[scene]
                ),
                "timing": {
                    "baseline": _timing_summary(baseline_by_scene[scene]),
                    "refinement": _timing_summary(refinement_by_scene[scene]),
                    "selected_policy": _timing_summary(
                        baseline_by_scene[scene]
                        if args.gate_scope == "scene"
                        and selected_config[scene] is None
                        else refinement_by_scene[scene]
                    ),
                    "total_ms_mean_overhead": (
                        _timing_summary(refinement_by_scene[scene])["total_ms"][
                            "mean"
                        ]
                        - _timing_summary(baseline_by_scene[scene])["total_ms"][
                            "mean"
                        ]
                    ),
                },
            }
            for scene in SCENES
        },
        "timing_pooled": {
            "baseline": _timing_summary(pooled_baseline),
            "refinement": _timing_summary(
                sum((refinement_by_scene[scene] for scene in SCENES), [])
            ),
            "selected_policy": _timing_summary(
                sum(
                    (
                        baseline_by_scene[scene]
                        if args.gate_scope == "scene"
                        and selected_config[scene] is None
                        else refinement_by_scene[scene]
                        for scene in SCENES
                    ),
                    [],
                )
            ),
        },
        "online_refinement_marginal_timing": _feedback_timing_summary(
            sum(
                (
                    baseline_by_scene[scene]
                    if args.gate_scope == "scene"
                    and selected_config[scene] is None
                    else refinement_by_scene[scene]
                    for scene in SCENES
                ),
                [],
            )
        ),
        "timing_claim": (
            "within-run online stage timers are the primary overhead estimate; "
            "cross-run baseline/refinement totals are retained as diagnostics"
        ),
        "maps": map_reports,
        "evaluation_sources": sources,
    }
    _atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
