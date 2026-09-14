#!/usr/bin/env python3
"""Aggregate fixed P0.5 shards and classify render--real gap mechanisms."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.v7_contracts import sha256_file
from evidence.v7_render_real_gap import (
    summarize_fixed_bias,
    summarize_pose_rows,
)


def _percent(correct: int, count: int) -> float:
    return 100.0 * int(correct) / max(int(count), 1)


def _relative_improvement(before: float, after: float) -> float:
    return (float(before) - float(after)) / max(abs(float(before)), 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema") != "lafgs_v7_render_real_gap_causal_diagnostic":
        raise ValueError("aggregate config schema differs")
    rows = []
    shard_inputs = []
    seen_queries: set[int] = set()
    expected_shard_count = None
    reference_mismatches = []
    for manifest_path in args.shard_manifest:
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text())
        result_path = Path(manifest["result"])
        if sha256_file(result_path) != manifest["result_sha256"]:
            raise ValueError("P0.5 shard result SHA mismatch")
        payload = json.loads(result_path.read_text())
        if not (
            payload.get("schema") == "lafgs_v7_render_real_gap_causal_diagnostic_shard"
            and payload.get("formal_protocol_eligible") is False
            and payload.get("posthoc_test_rgb_diagnostic") is True
            and payload.get("may_update_or_select_map") is False
            and payload.get("map_mutation_count") == 0
        ):
            raise ValueError("invalid P0.5 shard isolation flags")
        if expected_shard_count is None:
            expected_shard_count = int(payload["shard_count"])
        elif expected_shard_count != int(payload["shard_count"]):
            raise ValueError("P0.5 shard counts differ")
        for row in payload["rows"]:
            query_index = int(row["query_index"])
            if query_index in seen_queries:
                raise ValueError("duplicate P0.5 query index")
            seen_queries.add(query_index)
            rows.append(row)
        reference_mismatches.extend(payload["reference_replay_mismatch_query_indices"])
        shard_inputs.append({"path": str(manifest_path), "sha256": sha256_file(manifest_path)})
    if len(args.shard_manifest) != expected_shard_count or seen_queries != set(range(530)):
        raise ValueError("P0.5 aggregation requires the complete 530-query registry")
    rows.sort(key=lambda item: int(item["query_index"]))
    if reference_mismatches:
        raise RuntimeError("real+dataset-mask did not reproduce the frozen reference")

    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for condition in row["conditions"]:
            by_condition.setdefault(condition["condition"], []).append(condition)
    pose_metrics = {
        name: summarize_pose_rows(values) for name, values in sorted(by_condition.items())
    }
    fixed_bias = {
        name: summarize_fixed_bias(values)
        for name, values in by_condition.items()
        if name in {"real_dataset_masked", "render_unmasked"}
    }

    partition_names = ("real_inside", "real_outside", "render_inside", "render_outside")
    partition = {}
    for name in partition_names:
        correct = sum(int(row["correspondence_partition"][name]["correct"]) for row in rows)
        count = sum(int(row["correspondence_partition"][name]["count"]) for row in rows)
        partition[name] = {
            "correct": correct,
            "count": count,
            "gt_correct_4px_percent": _percent(correct, count),
        }
    descriptor_count = sum(int(row["descriptor_pairing"]["mutual_pair_count"]) for row in rows)
    descriptor = {
        "mutual_pair_count": descriptor_count,
        "real_mutual_repeatability_percent": _percent(
            descriptor_count,
            sum(
                int(row["correspondence_partition"]["real_inside"]["count"])
                + int(row["correspondence_partition"]["real_outside"]["count"])
                for row in rows
            ),
        ),
        "render_mutual_repeatability_percent": _percent(
            descriptor_count,
            sum(
                int(row["correspondence_partition"]["render_inside"]["count"])
                + int(row["correspondence_partition"]["render_outside"]["count"])
                for row in rows
            ),
        ),
        "mean_descriptor_cosine": sum(
            float(row["descriptor_pairing"]["descriptor_cosine_sum"]) for row in rows
        ) / max(descriptor_count, 1),
        "same_top1_anchor_percent": _percent(
            sum(int(row["descriptor_pairing"]["same_top1_anchor_count"]) for row in rows),
            descriptor_count,
        ),
        "real_mean_top1_margin": sum(
            float(row["descriptor_pairing"]["real_top1_margin_sum"]) for row in rows
        ) / max(descriptor_count, 1),
        "render_mean_top1_margin": sum(
            float(row["descriptor_pairing"]["render_top1_margin_sum"]) for row in rows
        ) / max(descriptor_count, 1),
        "real_pair_gt_correct_4px_percent": _percent(
            sum(int(row["descriptor_pairing"]["real_pair_correct_count"]) for row in rows),
            descriptor_count,
        ),
        "render_pair_gt_correct_4px_percent": _percent(
            sum(int(row["descriptor_pairing"]["render_pair_correct_count"]) for row in rows),
            descriptor_count,
        ),
        "mean_pair_distance_px": sum(
            float(row["descriptor_pairing"]["pair_distance_sum_px"]) for row in rows
        ) / max(descriptor_count, 1),
    }
    support_summary = {
        "dataset_mask_fraction_median": float(
            np.median([float(row["dataset_mask_fraction"]) for row in rows])
        ),
        "shared_support_pixel_fraction_median": float(
            np.median([float(row["shared_support_pixel_fraction"]) for row in rows])
        ),
        "oracle_correspondence_count_median": float(
            np.median([int(row["oracle_correspondence_count"]) for row in rows])
        ),
    }

    rules = config["decision_rules"]
    real_masked = pose_metrics["real_dataset_masked"]
    real_unmasked = pose_metrics["real_unmasked"]
    render_unmasked = pose_metrics["render_unmasked"]
    render_masked = pose_metrics["render_dataset_masked"]
    supported = pose_metrics["real_dataset_masked_support_rows"]
    oracle = pose_metrics["oracle_geometry"]
    geometry_sufficient = bool(
        oracle["median_translation_cm"]
        <= float(rules["geometry_ceiling_maximum_median_translation_cm"])
        and oracle["recall_5cm_5deg_percent"]
        >= float(rules["geometry_ceiling_minimum_recall_5cm_5deg_percent"])
    )
    outside_precision = float(partition["real_outside"]["gt_correct_4px_percent"])
    inside_precision = float(partition["real_inside"]["gt_correct_4px_percent"])
    contamination_ratio = inside_precision / max(outside_precision, 1e-12)
    content_contamination = bool(
        contamination_ratio >= float(rules["content_contamination_minimum_precision_ratio"])
        and outside_precision
        <= float(rules["content_contamination_maximum_outside_precision_percent"])
    )
    descriptor_gap_points = float(
        partition["render_inside"]["gt_correct_4px_percent"]
        - partition["real_inside"]["gt_correct_4px_percent"]
    )
    shared_descriptor_gap = bool(
        descriptor_gap_points
        >= float(rules["shared_descriptor_minimum_correctness_gap_points"])
    )
    real_mask_r5_delta = float(
        real_masked["recall_5cm_5deg_percent"] - real_unmasked["recall_5cm_5deg_percent"]
    )
    render_mask_r5_delta = float(
        render_masked["recall_5cm_5deg_percent"] - render_unmasked["recall_5cm_5deg_percent"]
    )
    support_filter_r5_delta = float(
        supported["recall_5cm_5deg_percent"] - real_masked["recall_5cm_5deg_percent"]
    )
    support_filter_median_relative = _relative_improvement(
        real_masked["median_translation_cm"], supported["median_translation_cm"]
    )
    material_support_pose_gain = bool(
        support_filter_r5_delta >= float(rules["material_pose_delta_minimum_r5_points"])
        or support_filter_median_relative
        >= float(rules["material_pose_delta_minimum_median_relative"])
    )
    real_bias_fixed = bool(
        fixed_bias["real_dataset_masked"]["unit_direction_resultant"]
        >= float(rules["fixed_bias_minimum_direction_resultant"])
    )

    hybrid_real = pose_metrics["real_shared_render_else_dataset_masked"]
    hybrid_render = pose_metrics["render_shared_real_else_dataset_masked"]
    hybrid_indices = {
        int(row["query_index"])
        for row in rows
        if any(
            item["condition"] == "real_shared_render_else_dataset_masked"
            for item in row["conditions"]
        )
    }
    hybrid_base = {}
    for base_name in ("real_dataset_masked", "render_dataset_masked"):
        selected = []
        for row in rows:
            if int(row["query_index"]) not in hybrid_indices:
                continue
            selected.extend(
                item for item in row["conditions"] if item["condition"] == base_name
            )
        hybrid_base[base_name] = summarize_pose_rows(selected)
    hybrid_effect = {
        "real_outside_replaced_by_render": {
            "median_translation_relative_improvement": _relative_improvement(
                hybrid_base["real_dataset_masked"]["median_translation_cm"],
                hybrid_real["median_translation_cm"],
            ),
            "r5_delta_points": float(
                hybrid_real["recall_5cm_5deg_percent"]
                - hybrid_base["real_dataset_masked"]["recall_5cm_5deg_percent"]
            ),
        },
        "real_outside_added_to_render": {
            "median_translation_relative_regression": -_relative_improvement(
                hybrid_base["render_dataset_masked"]["median_translation_cm"],
                hybrid_render["median_translation_cm"],
            ),
            "r5_delta_points": float(
                hybrid_render["recall_5cm_5deg_percent"]
                - hybrid_base["render_dataset_masked"]["recall_5cm_5deg_percent"]
            ),
        },
    }

    decisions = {
        "frozen_geometric_bias_is_primary": real_bias_fixed and not geometry_sufficient,
        "geometry_ceiling_is_sufficient": geometry_sufficient,
        "existing_dataset_mask_explains_render_advantage": bool(
            real_mask_r5_delta < -float(rules["material_pose_delta_minimum_r5_points"])
            and render_mask_r5_delta > float(rules["material_pose_delta_minimum_r5_points"])
        ),
        "content_correspondence_contamination_detected": content_contamination,
        "hard_support_filter_materially_improves_pose": material_support_pose_gain,
        "shared_content_descriptor_gap_detected": shared_descriptor_gap,
    }
    interpretation = []
    interpretation.append(
        "The frozen map geometry/PoseLib ceiling is sufficient."
        if geometry_sufficient
        else "The oracle geometry ceiling remains insufficient and geometry needs inspection."
    )
    interpretation.append(
        "Low-support real rows are correspondence contaminants."
        if content_contamination
        else "The preregistered correspondence-contamination rule did not pass."
    )
    interpretation.append(
        "Hard support filtering materially improves pose."
        if material_support_pose_gain
        else "Hard support filtering does not convert row-level cleanliness into a material pose gain."
    )
    interpretation.append(
        "A descriptor/matcher gap remains on shared content."
        if shared_descriptor_gap
        else "The preregistered shared-content descriptor-gap rule did not pass."
    )

    report = {
        "schema": "lafgs_v7_render_real_gap_causal_diagnostic_report",
        "version": 1,
        "status": "PASS",
        "formal_protocol_eligible": False,
        "posthoc_test_rgb_diagnostic": True,
        "may_update_or_select_map": False,
        "map_mutation_count": 0,
        "query_count": len(rows),
        "reference_replay_mismatch_count": 0,
        "pose_metrics": pose_metrics,
        "fixed_bias": fixed_bias,
        "correspondence_partition": partition,
        "content_contamination_precision_ratio": contamination_ratio,
        "shared_content_descriptor_correctness_gap_points": descriptor_gap_points,
        "descriptor_pairing": descriptor,
        "support_summary": support_summary,
        "existing_mask_effect": {
            "real_r5_delta_masked_minus_unmasked_points": real_mask_r5_delta,
            "render_r5_delta_masked_minus_unmasked_points": render_mask_r5_delta,
        },
        "support_filter_effect": {
            "r5_delta_points": support_filter_r5_delta,
            "median_translation_relative_improvement": support_filter_median_relative,
        },
        "hybrid_same_subset_baselines": hybrid_base,
        "hybrid_effect": hybrid_effect,
        "hybrid_interpretation_limit": (
            "The symmetric feathered hybrids are a detector-level intervention, "
            "but cross-domain seams and inconsistent scene content can create "
            "interactions. Their paired median/R5 effects support content causality "
            "and are not treated as a deployable mask result."
        ),
        "decisions": decisions,
        "interpretation": interpretation,
        "inputs": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "shard_manifests": shard_inputs,
        },
    }
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "report.json"
    temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, report_path)
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(
        "# V7 render--real causal diagnostic\n\n"
        + "- Status: PASS (post-hoc, non-formal, zero updates)\n"
        + f"- Frozen-reference replay mismatches: 0 / {len(rows)}\n"
        + f"- Geometry ceiling: median {oracle['median_translation_cm']:.3f}cm, "
        + f"R5 {oracle['recall_5cm_5deg_percent']:.3f}%\n"
        + f"- Real inside/outside support GT@4px: {inside_precision:.3f}% / "
        + f"{outside_precision:.3f}%\n"
        + f"- Render/real shared-support correctness gap: {descriptor_gap_points:.3f} points\n"
        + f"- Support-filter R5 delta: {support_filter_r5_delta:+.3f} points\n"
        + f"- Real dataset-mask R5 delta: {real_mask_r5_delta:+.3f} points\n"
        + f"- Render dataset-mask R5 delta: {render_mask_r5_delta:+.3f} points\n\n"
        + "## Decisions\n\n"
        + "\n".join(f"- {name}: {value}" for name, value in decisions.items())
        + "\n"
    )
    print(json.dumps({"report": str(report_path), "sha256": sha256_file(report_path), "decisions": decisions}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
