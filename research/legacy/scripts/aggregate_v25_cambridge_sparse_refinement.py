#!/usr/bin/env python3
"""Finalize the V25 five-scene online sparse-refinement policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.hashing import sha256_file
from scripts.aggregate_v24_cambridge_sparse_refinement import (
    SCENES,
    _atomic_json,
    _candidate_rows,
    _distribution,
    _load_evaluation,
    _paired_summary,
    _passes_gate,
    _pose_summary,
    _r5,
)


def _validate_refinement_contract(
    contract: dict, scene_config: dict, scene: str
) -> None:
    adaptive = scene_config["adaptive_projection_gate"]
    expected = {
        "refinement_pose_backend": "robust",
        "refinement_candidate_pool": "first_pose_point_projection_frustum",
        "refinement_active_row_retrieval": True,
        "refinement_pre_topk_view_filter": True,
        "refinement_allow_soft_inliers": False,
        "refinement_common_candidate_grid_gate": False,
        "refinement_progressive_sampling": False,
    }
    if any(contract.get(key, False) != value for key, value in expected.items()):
        raise ValueError(f"V25 refinement contract differs: {scene}")
    if not np.isclose(
        float(contract.get("refinement_projection_gate_px", 8.0)),
        8.0,
        atol=0,
        rtol=0,
    ):
        raise ValueError(f"V25 base projection gate differs: {scene}")
    expected_wide = float(adaptive["wider_projection_gate_px"])
    expected_count = int(adaptive["maximum_baseline_inliers"])
    if not (
        np.isclose(
            float(
                contract.get("refinement_uncertainty_projection_gate_px", 0.0)
            ),
            expected_wide,
            atol=0,
            rtol=0,
        )
        and int(
            contract.get(
                "refinement_uncertainty_maximum_baseline_inliers", 0
            )
        )
        == expected_count
    ):
        raise ValueError(f"V25 adaptive projection gate differs: {scene}")


def _select_scene(
    baseline: list[dict],
    refinement: list[dict],
    candidates: list[dict | None],
    scene_config: dict,
) -> tuple[list[dict], int]:
    if not bool(scene_config["enabled"]):
        return list(baseline), 0
    selected = []
    accepted = 0
    adaptive = scene_config["adaptive_projection_gate"]
    base_gate = scene_config["post_solve_gate"]
    wide_gate = scene_config.get("wide_post_solve_gate", base_gate)
    for base, diagnostic, candidate in zip(baseline, refinement, candidates):
        projection_gate = float(
            diagnostic.get("sparse_feedback_projection_gate_px", 8.0)
        )
        if bool(adaptive["enabled"]):
            expected = (
                float(adaptive["wider_projection_gate_px"])
                if bool(diagnostic.get("sparse_feedback_eligible"))
                and int(base["inliers"])
                <= int(adaptive["maximum_baseline_inliers"])
                else 8.0
            )
            if not np.isclose(projection_gate, expected, atol=0, rtol=0):
                raise ValueError("V25 per-query adaptive projection route differs")
        elif not np.isclose(projection_gate, 8.0, atol=0, rtol=0):
            raise ValueError("V25 fixed projection route differs")
        gate = wide_gate if projection_gate > 8.0 else base_gate
        use = candidate is not None and _passes_gate(diagnostic, gate)
        selected.append(candidate if use else base)
        accepted += int(use)
    return selected, accepted


def _funnel(
    baseline: list[dict],
    refinement: list[dict],
    candidates: list[dict | None],
    selected: list[dict],
) -> dict:
    candidate_te_better = 0
    candidate_re_better = 0
    candidate_pareto_better = 0
    candidate_r5_gain = 0
    candidate_r5_loss = 0
    for base, candidate in zip(baseline, candidates):
        if candidate is None:
            continue
        te0 = float(base["translation_error_cm"])
        re0 = float(base["rotation_error_deg"])
        te1 = float(candidate["translation_error_cm"])
        re1 = float(candidate["rotation_error_deg"])
        candidate_te_better += int(te1 < te0)
        candidate_re_better += int(re1 < re0)
        candidate_pareto_better += int(te1 < te0 and re1 < re0)
        candidate_r5_gain += int((not _r5(base)) and _r5(candidate))
        candidate_r5_loss += int(_r5(base) and (not _r5(candidate)))
    return {
        "query_count": len(baseline),
        "eligible_query_count": sum(
            bool(row.get("sparse_feedback_eligible")) for row in refinement
        ),
        "proposal_gate_pass_query_count": sum(
            bool(row.get("sparse_feedback_gate_passed")) for row in refinement
        ),
        "second_solve_query_count": sum(value is not None for value in candidates),
        "candidate_te_better_query_count": candidate_te_better,
        "candidate_re_better_query_count": candidate_re_better,
        "candidate_pareto_better_query_count": candidate_pareto_better,
        "candidate_r5_gain_count": candidate_r5_gain,
        "candidate_r5_loss_count": candidate_r5_loss,
        "final_changed_query_count": sum(
            left["pose_w2c"] != right["pose_w2c"]
            for left, right in zip(baseline, selected)
        ),
        "correspondence_identity_oracle_available": False,
        "ground_truth_used_by_online_method": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if not (
        protocol.get("schema")
        == "lafgs_v25_cambridge_pose_conditioned_sparse_refinement_protocol"
        and protocol["scientific_scope"]["offline_feedback_enabled"] is False
        and protocol["scientific_scope"]["ground_truth_used_online"] is False
    ):
        raise ValueError("V25 protocol scope differs")
    root = Path(protocol["artifact_root"])
    baseline_by_scene = {}
    selected_by_scene = {}
    funnel_by_scene = {}
    sources = {}
    accepted_by_scene = {}
    selected_runtime_rows = []
    baseline_runtime_rows = []
    for scene in SCENES:
        scene_config = protocol["scene_policy"][scene]
        report_path = root / scene / "projective_map" / "report.json"
        report = json.loads(report_path.read_text())
        if not (
            report["schema"]
            == "lafgs_v8_v2_projective_map_materialization_report"
            and report["contracts"]["feedback_used"] is False
            and report["uses_test_queries"] is False
        ):
            raise ValueError(f"V25 F0 map scope differs: {scene}")
        map_path = Path(report["output"]["map"])
        metric_path = Path(report["output"]["metric"])
        map_sha = sha256_file(map_path)
        metric_sha = sha256_file(metric_path)
        baseline, baseline_source = _load_evaluation(
            root / scene / "evaluation" / "baseline_seed2026",
            expected_map=map_path,
            expected_map_sha256=map_sha,
            expected_metric=metric_path,
            expected_metric_sha256=metric_sha,
        )
        refinement, refinement_source = _load_evaluation(
            root / scene / "evaluation" / scene_config["evaluation_directory"],
            expected_map=map_path,
            expected_map_sha256=map_sha,
            expected_metric=metric_path,
            expected_metric_sha256=metric_sha,
            expected_refinement_backend="robust",
            expected_refinement_candidate_pool=(
                "first_pose_point_projection_frustum"
            ),
        )
        contract = json.loads(
            Path(refinement_source["contract"]["path"]).read_text()
        )
        _validate_refinement_contract(contract, scene_config, scene)
        candidates = _candidate_rows(baseline, refinement)
        selected, accepted = _select_scene(
            baseline, refinement, candidates, scene_config
        )
        baseline_by_scene[scene] = baseline
        selected_by_scene[scene] = selected
        accepted_by_scene[scene] = accepted
        funnel_by_scene[scene] = _funnel(
            baseline, refinement, candidates, selected
        )
        sources[scene] = {
            "map_report": {
                "path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
            },
            "baseline": baseline_source,
            "refinement": refinement_source,
        }
        baseline_runtime_rows.extend(baseline)
        selected_runtime_rows.extend(
            refinement if bool(scene_config["enabled"]) else baseline
        )

    pooled_baseline = sum((baseline_by_scene[s] for s in SCENES), [])
    pooled_selected = sum((selected_by_scene[s] for s in SCENES), [])
    marginal = [
        float(row.get("feedback_geometry_ms", 0.0))
        + float(row.get("feedback_ransac_ms", 0.0))
        + float(row.get("feedback_model_comparison_ms", 0.0))
        for row in selected_runtime_rows
    ]
    baseline_total = [float(row["total_ms"]) for row in baseline_runtime_rows]
    output = {
        "schema": "lafgs_v25_cambridge_sparse_refinement_aggregate",
        "version": 1,
        "decision": "TEST_CALIBRATED_ONLINE_POLICY_SELECTED",
        "claim_scope": "five_scene_test_calibrated_not_unseen_generalization",
        "offline_feedback_enabled": False,
        "dense_matching_or_query_rendering_used": False,
        "baseline": _pose_summary(pooled_baseline),
        "selected": _pose_summary(pooled_selected),
        "paired": _paired_summary(pooled_baseline, pooled_selected),
        "by_scene": {
            scene: {
                "enabled": bool(protocol["scene_policy"][scene]["enabled"]),
                "accepted_query_count": accepted_by_scene[scene],
                "baseline": _pose_summary(baseline_by_scene[scene]),
                "selected": _pose_summary(selected_by_scene[scene]),
                "paired": _paired_summary(
                    baseline_by_scene[scene], selected_by_scene[scene]
                ),
                "funnel": funnel_by_scene[scene],
            }
            for scene in SCENES
        },
        "marginal_online_stage_ms": _distribution(marginal),
        "baseline_total_ms": _distribution(baseline_total),
        "marginal_over_baseline_mean_percent": float(
            100.0 * np.mean(marginal) / np.mean(baseline_total)
        ),
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol),
        },
        "sources": sources,
    }
    _atomic_json(output, args.output)


if __name__ == "__main__":
    main()
