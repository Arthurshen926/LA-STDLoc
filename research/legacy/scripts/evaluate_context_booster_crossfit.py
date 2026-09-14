#!/usr/bin/env python3
"""Cross-fit official Boost-F against a support-matched raw-SP control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from map_learning.context_booster import (
    SUPERPOINT_BOOST_F_SHA256,
    load_superpoint_boost_f,
    resolve_featurebooster_weights,
)
from map_learning.context_booster_crossfit import (
    DEFAULT_TOPKS,
    build_observation_fused_banks,
    combine_additive_counts,
    compare_protocols,
    evaluate_context_banks,
    summarize_pose_rows,
    summarize_retrieval,
)
from topology.crossfit_swap_revision import temporal_crossfit_split


def _load_mmap(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _uniform_subset(values: list[int], requested: int) -> list[int]:
    if requested <= 0 or requested >= len(values):
        return list(values)
    positions = (
        torch.linspace(0, len(values) - 1, steps=int(requested))
        .round()
        .long()
        .unique(sorted=True)
        .tolist()
    )
    return [values[position] for position in positions]


def _parse_topks(value: str) -> tuple[int, ...]:
    topks = tuple(sorted(set(int(item) for item in value.split(",") if item)))
    if not topks or topks[0] < 1:
        raise argparse.ArgumentTypeError("top-K list must contain positive integers")
    return topks


def _reprojection_threshold(
    calibration_path: Path | None,
    fixed_threshold: float | None,
) -> tuple[float, dict]:
    if calibration_path is not None:
        calibration = json.loads(calibration_path.read_text())
        threshold = float(calibration["parameters"]["ransac_reprojection_px"])
        return threshold, {
            "source": "mapping_only_scene_calibration",
            "path": str(calibration_path.resolve()),
        }
    if fixed_threshold is None:
        raise ValueError(
            "provide --scene-calibration or the explicit legacy fallback "
            "--ransac-reprojection-px"
        )
    return float(fixed_threshold), {
        "source": "explicit_mapping_only_fixed_fallback",
        "value_px": float(fixed_threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--featurebooster-weights", type=Path)
    parser.add_argument("--scene-calibration", type=Path)
    parser.add_argument("--ransac-reprojection-px", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument(
        "--support-query-count-per-direction",
        type=int,
        default=0,
        help="Uniform support-fold sample for smoke tests; zero uses the full fold.",
    )
    parser.add_argument(
        "--gate-query-count",
        type=int,
        default=256,
        help="Total across both directions; zero evaluates both complete gate folds.",
    )
    parser.add_argument(
        "--pose-query-count",
        type=int,
        default=96,
        help="Total mapping-only PoseLib sample across both directions.",
    )
    parser.add_argument("--minimum-support-views", type=int, default=2)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--topks", type=_parse_topks, default=DEFAULT_TOPKS)
    parser.add_argument("--skip-pose-pnp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()

    for label, value in (
        ("support query count", args.support_query_count_per_direction),
        ("gate query count", args.gate_query_count),
        ("pose query count", args.pose_query_count),
        ("deployment row limit", args.deployment_row_limit),
    ):
        if int(value) < 0:
            raise ValueError(f"{label} must be non-negative")
    if args.minimum_support_views < 1:
        raise ValueError("minimum support views must be positive")

    device = torch.device(args.device)
    state = _load_mmap(args.map)
    teacher = _load_mmap(args.complete_positive_teacher)
    query_cache = _load_mmap(args.query_cache)
    if int(teacher["anchor_count"]) != len(state["anchor_ids"]):
        raise ValueError("map and complete-positive teacher anchor counts differ")
    names = list(teacher["query_names"])
    even, odd, split_report = temporal_crossfit_split(
        names, block_count=int(args.block_count)
    )
    threshold, threshold_source = _reprojection_threshold(
        args.scene_calibration, args.ransac_reprojection_px
    )
    weight_path = resolve_featurebooster_weights(args.featurebooster_weights)
    model = load_superpoint_boost_f(weight_path, device=device)

    directions = (
        ("even_blocks_to_odd_blocks", even, odd),
        ("odd_blocks_to_even_blocks", odd, even),
    )
    gate_budget = (
        [0, 0]
        if int(args.gate_query_count) == 0
        else [
            (int(args.gate_query_count) + 1) // 2,
            int(args.gate_query_count) // 2,
        ]
    )
    pose_budget = (
        [0, 0]
        if int(args.pose_query_count) == 0
        else [
            (int(args.pose_query_count) + 1) // 2,
            int(args.pose_query_count) // 2,
        ]
    )

    fold_reports = []
    all_pose_rows: list[dict] = []
    additive = {"raw_superpoint": [], "superpoint_boost_f": []}
    for direction_index, (direction_name, support_fold, gate_fold) in enumerate(
        directions
    ):
        support = _uniform_subset(
            support_fold, int(args.support_query_count_per_direction)
        )
        gate = _uniform_subset(gate_fold, gate_budget[direction_index])
        pose_queries = (
            []
            if args.skip_pose_pnp
            else _uniform_subset(gate, pose_budget[direction_index])
        )
        print(
            {
                "event": "context_booster_direction_start",
                "direction": direction_name,
                "support_query_count": len(support),
                "gate_query_count": len(gate),
                "pose_query_count": len(pose_queries),
            },
            flush=True,
        )
        banks, support_report = build_observation_fused_banks(
            teacher=teacher,
            query_cache=query_cache,
            support_query_indices=support,
            model=model,
            device=device,
            minimum_support_views=int(args.minimum_support_views),
            progress_interval=int(args.progress_interval),
        )
        supported_types = torch.as_tensor(state["anchor_type"]).long()[
            banks["anchor_indices"].cpu()
        ]
        support_report["supported_track_anchor_count"] = int(
            (supported_types != 0).sum()
        )
        support_report["supported_reserve_anchor_count"] = int(
            (supported_types == 0).sum()
        )
        retrieval, pose_rows = evaluate_context_banks(
            state=state,
            teacher=teacher,
            query_cache=query_cache,
            gate_query_indices=gate,
            pose_query_indices=pose_queries,
            banks=banks,
            model=model,
            device=device,
            topks=args.topks,
            deployment_row_limit=int(args.deployment_row_limit),
            ransac_reprojection_px=threshold,
            seed=int(args.seed),
            progress_interval=int(args.progress_interval),
        )
        fold_additive = retrieval.pop("additive_counts")
        for descriptor_name in additive:
            additive[descriptor_name].append(fold_additive[descriptor_name])
        fold_reports.append(
            {
                "direction": direction_name,
                "support_query_indices": support,
                "gate_query_indices": gate,
                "pose_query_indices": pose_queries,
                "support": support_report,
                "retrieval": retrieval,
                "pose": summarize_pose_rows(pose_rows),
            }
        )
        for row in pose_rows:
            row["direction"] = direction_name
        all_pose_rows.extend(pose_rows)
        del banks
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_retrieval = {}
    for descriptor_name, fold_counts in additive.items():
        combined = combine_additive_counts(fold_counts, args.topks)
        aggregate_retrieval[descriptor_name] = summarize_retrieval(
            combined, args.topks
        )
    aggregate_pose = summarize_pose_rows(all_pose_rows)
    if all_pose_rows:
        comparison = compare_protocols(aggregate_retrieval, aggregate_pose)
    else:
        raw_r1 = aggregate_retrieval["raw_superpoint"][
            "positive_recall_at_k"
        ]["1"]
        boost_r1 = aggregate_retrieval["superpoint_boost_f"][
            "positive_recall_at_k"
        ]["1"]
        comparison = {
            "boost_minus_raw_top1_positive_recall_percentage_points": float(
                100.0 * (boost_r1 - raw_r1)
            ),
            "routing_verdict": "pose_replay_skipped_no_final_routing_verdict",
        }

    report = {
        "schema": "lafgs_context_booster_crossfit",
        "version": 1,
        "uses_test_queries": False,
        "protocol": {
            "split": "bidirectional_trajectory_block_crossfit",
            "map_descriptor": (
                "view_balanced_mean_of_per_image_normalized_positive_observations"
            ),
            "support_parity": (
                "raw_superpoint_and_boost_f_use_identical_positive_edges_and_anchors"
            ),
            "online_matching": "one_descriptor_exact_global_cosine_top1",
            "pose_solver": "one_poselib_absolute_pose_call_per_descriptor_protocol",
            "map_topology": (
                "fixed_input_map_filtered_only_by_support_observation_availability"
            ),
        },
        "inputs": {
            "map": str(args.map.resolve()),
            "complete_positive_teacher": str(
                args.complete_positive_teacher.resolve()
            ),
            "query_cache": str(args.query_cache.resolve()),
            "featurebooster_weights": str(weight_path),
            "featurebooster_sha256": SUPERPOINT_BOOST_F_SHA256,
        },
        "config": {
            "block_count": int(args.block_count),
            "support_query_count_per_direction": int(
                args.support_query_count_per_direction
            ),
            "gate_query_count_total": int(args.gate_query_count),
            "pose_query_count_total": int(args.pose_query_count),
            "minimum_support_views": int(args.minimum_support_views),
            "deployment_row_limit": int(args.deployment_row_limit),
            "topks": list(args.topks),
            "ransac_reprojection_px": float(threshold),
            "ransac_threshold_source": threshold_source,
            "seed": int(args.seed),
        },
        "split": split_report,
        "directions": fold_reports,
        "aggregate": {
            "retrieval": aggregate_retrieval,
            "pose": aggregate_pose,
            "comparison": comparison,
        },
        "pose_queries": all_pose_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            "event": "context_booster_crossfit_complete",
            "output": str(args.output),
            "comparison": comparison,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
