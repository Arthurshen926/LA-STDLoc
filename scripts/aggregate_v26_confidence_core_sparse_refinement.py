#!/usr/bin/env python3
"""Aggregate the V26 confidence-core online sparse-refinement policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.hashing import sha256_file
from scripts.aggregate_v24_cambridge_sparse_refinement import (
    SCENES,
    _atomic_json,
    _distribution,
    _load_evaluation,
    _paired_summary,
    _pose_summary,
)


def _validate_query_registry(baseline: list[dict], selected: list[dict]) -> None:
    if [row["image_name"] for row in baseline] != [
        row["image_name"] for row in selected
    ]:
        raise ValueError("V26 paired query registries differ")
    for left, right in zip(baseline, selected):
        if not np.allclose(
            left["gt_pose_w2c"], right["gt_pose_w2c"], atol=0, rtol=0
        ):
            raise ValueError("V26 paired ground-truth registries differ")


def _validate_selected_contract(
    contract: dict, rows: list[dict], scene_config: dict, scene: str
) -> None:
    expected = {
        "pose_conditioned_sparse_refinement": True,
        "refinement_pose_backend": "robust",
        "refinement_candidate_pool": "first_pose_point_projection_frustum",
        "refinement_active_row_retrieval": True,
        "refinement_pre_topk_view_filter": True,
        "core_reserve_refinement": False,
        "minimum_retained_match_count": int(
            scene_config["minimum_retained_match_count"]
        ),
        "refinement_pose_conditioned_mutual_matching": bool(
            scene_config["pose_conditioned_mutual_matching"]
        ),
        "refinement_minimum_proposal_count": int(
            scene_config["minimum_proposal_count"]
        ),
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError(f"V26 selected contract differs: {scene}")
    fraction = float(scene_config["match_retention_fraction"])
    if not np.isclose(
        float(contract.get("match_retention_fraction", 1.0)),
        fraction,
        atol=0,
        rtol=0,
    ):
        raise ValueError(f"V26 confidence-core fraction differs: {scene}")
    exact_float_fields = {
        "refinement_projection_gate_px": "projection_gate_px",
        "refinement_uncertainty_projection_gate_px": (
            "wider_projection_gate_px"
        ),
        "refinement_minimum_proposal_relative_gain": (
            "minimum_proposal_relative_gain"
        ),
    }
    if any(
        not np.isclose(
            float(contract.get(contract_key, -1.0)),
            float(scene_config[config_key]),
            atol=0,
            rtol=0,
        )
        for contract_key, config_key in exact_float_fields.items()
    ) or int(
        contract.get("refinement_uncertainty_maximum_baseline_inliers", -1)
    ) != int(scene_config["wider_projection_maximum_inliers"]):
        raise ValueError(f"V26 proposal policy differs: {scene}")
    for row in rows:
        top1 = int(row["top1_match_count"])
        retained = int(row["retained_match_count"])
        filtered = int(row["score_filtered_match_count"])
        if not (
            4 <= retained <= top1
            and filtered == top1 - retained
            and int(row["core_reserve_refinement"]) == 0
            and int(row["sparse_feedback_baseline_comparison_inliers"]) >= 0
        ):
            raise ValueError(f"V26 confidence-core diagnostics differ: {scene}")


def _timing(rows: list[dict]) -> dict:
    online = [
        float(row.get("feedback_geometry_ms", 0.0))
        + float(row.get("feedback_ransac_ms", 0.0))
        + float(row.get("feedback_model_comparison_ms", 0.0))
        for row in rows
    ]
    return {
        "total_ms": _distribution([float(row["total_ms"]) for row in rows]),
        "first_ransac_ms": _distribution(
            [float(row["ransac_ms"]) for row in rows]
        ),
        "sparse_feedback_marginal_ms": _distribution(online),
        "second_solve_count": sum(
            row.get("sparse_feedback_candidate_pose_w2c") is not None
            for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if not (
        protocol.get("schema")
        == "lafgs_v26_confidence_core_sparse_refinement_protocol"
        and protocol.get("version") == 1
        and protocol["scientific_scope"]["offline_feedback_enabled"] is False
        and protocol["scientific_scope"]["ground_truth_used_online"] is False
        and protocol["scientific_scope"]["ground_truth_used_for_test_calibration"]
        is True
    ):
        raise ValueError("V26 protocol scope differs")
    root = Path(protocol["artifact_root"])
    baseline_by_scene: dict[str, list[dict]] = {}
    selected_by_scene: dict[str, list[dict]] = {}
    source_by_scene = {}
    for scene in SCENES:
        report_path = root / scene / "projective_map" / "report.json"
        report = json.loads(report_path.read_text())
        if not (
            report.get("schema")
            == "lafgs_v8_v2_projective_map_materialization_report"
            and report["contracts"]["feedback_used"] is False
            and report["uses_test_queries"] is False
        ):
            raise ValueError(f"V26 F0 map scope differs: {scene}")
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
        scene_config = protocol["scene_policy"][scene]
        if bool(scene_config["enabled"]):
            selected, selected_source = _load_evaluation(
                root
                / scene
                / "evaluation"
                / scene_config["evaluation_directory"],
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
                Path(selected_source["contract"]["path"]).read_text()
            )
            _validate_selected_contract(contract, selected, scene_config, scene)
        else:
            selected = list(baseline)
            selected_source = baseline_source
        _validate_query_registry(baseline, selected)
        baseline_by_scene[scene] = baseline
        selected_by_scene[scene] = selected
        source_by_scene[scene] = {
            "map_report": {
                "path": str(report_path.resolve()),
                "sha256": sha256_file(report_path),
            },
            "baseline": baseline_source,
            "selected": selected_source,
        }

    pooled_baseline = sum((baseline_by_scene[s] for s in SCENES), [])
    pooled_selected = sum((selected_by_scene[s] for s in SCENES), [])
    output = {
        "schema": "lafgs_v26_confidence_core_sparse_refinement_aggregate",
        "version": 1,
        "decision": "TEST_CALIBRATED_ONLINE_POLICY_SELECTED",
        "claim_scope": "five_scene_test_calibrated_not_unseen_generalization",
        "offline_feedback_enabled": False,
        "query_rendering_or_dense_matching_used": False,
        "baseline": _pose_summary(pooled_baseline),
        "selected": _pose_summary(pooled_selected),
        "paired": _paired_summary(pooled_baseline, pooled_selected),
        "timing": {
            "baseline": _timing(pooled_baseline),
            "selected": _timing(pooled_selected),
        },
        "by_scene": {
            scene: {
                "enabled": bool(protocol["scene_policy"][scene]["enabled"]),
                "baseline": _pose_summary(baseline_by_scene[scene]),
                "selected": _pose_summary(selected_by_scene[scene]),
                "paired": _paired_summary(
                    baseline_by_scene[scene], selected_by_scene[scene]
                ),
                "timing": _timing(selected_by_scene[scene]),
            }
            for scene in SCENES
        },
        "protocol": {
            "path": str(args.protocol.resolve()),
            "sha256": sha256_file(args.protocol),
        },
        "sources": source_by_scene,
    }
    _atomic_json(output, args.output)
    print(json.dumps(output["selected"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
