#!/usr/bin/env python3
"""Cross-fit a metric-preserving contextual tangent uplift against frozen A1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from map_learning.context_booster_crossfit import (
    DEFAULT_TOPKS,
    combine_additive_counts,
    summarize_retrieval,
)
from map_learning.context_metric import CONTEXT_MODES
from map_learning.context_metric_crossfit import (
    build_raw_observation_bank,
    prepare_training_records,
)
from map_learning.metric_context_uplift import (
    MetricPreservingContextUplift,
    build_metric_uplift_bank,
    compare_metric_context_protocols,
    evaluate_metric_context_banks,
    load_frozen_metric_state,
    summarize_metric_context_pose,
    train_metric_context_stage,
)
from topology.crossfit_swap_revision import temporal_crossfit_split


def _load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


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
        raise argparse.ArgumentTypeError("top-K values must be positive")
    return topks


def _parse_kernels(value: str) -> tuple[int, ...]:
    kernels = tuple(int(item) for item in value.split(",") if item)
    if not kernels or any(item < 1 or item % 2 == 0 for item in kernels):
        raise argparse.ArgumentTypeError("context kernels must be positive odd values")
    return kernels


def _reprojection_threshold(
    calibration_path: Path | None,
    fixed_threshold: float | None,
) -> tuple[float, dict]:
    if calibration_path is not None:
        calibration = json.loads(calibration_path.read_text())
        return float(calibration["parameters"]["ransac_reprojection_px"]), {
            "source": "mapping_only_scene_calibration",
            "path": str(calibration_path.resolve()),
        }
    if fixed_threshold is None:
        raise ValueError("provide scene calibration or a fixed RANSAC threshold")
    return float(fixed_threshold), {
        "source": "explicit_mapping_only_fixed_fallback",
        "value_px": float(fixed_threshold),
    }


def _state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _pose_pairing(rows: list[dict]) -> dict:
    if not rows:
        return {"query_count": 0}
    delta = np.asarray(
        [row["uplift_te_cm"] - row["a1_te_cm"] for row in rows],
        dtype=np.float64,
    )
    worst = sorted(
        rows,
        key=lambda row: row["uplift_te_cm"] - row["a1_te_cm"],
        reverse=True,
    )[:5]
    return {
        "query_count": int(delta.size),
        "uplift_better_count": int((delta < 0).sum()),
        "uplift_worse_count": int((delta > 0).sum()),
        "tie_count": int((delta == 0).sum()),
        "median_te_delta_cm": float(np.median(delta)),
        "mean_te_delta_cm": float(delta.mean()),
        "p90_te_delta_cm": float(np.percentile(delta, 90)),
        "worst_regressions": [
            {
                "image_name": row["image_name"],
                "a1_te_cm": float(row["a1_te_cm"]),
                "uplift_te_cm": float(row["uplift_te_cm"]),
                "delta_cm": float(row["uplift_te_cm"] - row["a1_te_cm"]),
                "direction": row.get("direction"),
            }
            for row in worst
        ],
    }


def _combine_clean(rows: list[dict]) -> dict:
    keys = (
        "a1_clean_row_count",
        "uplift_positive_retained_count",
        "exact_winner_retained_count",
        "new_false_attractor_count",
        "clean_margin_violation_count",
    )
    combined = {key: sum(int(row[key]) for row in rows) for key in keys}
    denominator = max(combined["a1_clean_row_count"], 1)
    return {
        **combined,
        "positive_retention_percent": float(
            100.0 * combined["uplift_positive_retained_count"] / denominator
        ),
        "exact_winner_retention_percent": float(
            100.0 * combined["exact_winner_retained_count"] / denominator
        ),
        "new_false_attractor_percent": float(
            100.0 * combined["new_false_attractor_count"] / denominator
        ),
        "clean_margin_violation_percent": float(
            100.0 * combined["clean_margin_violation_count"] / denominator
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--metric-state", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--scene-calibration", type=Path)
    parser.add_argument("--ransac-reprojection-px", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-count", type=int, default=8)
    parser.add_argument("--support-query-count-per-direction", type=int, default=0)
    parser.add_argument("--gate-query-count", type=int, default=256)
    parser.add_argument("--pose-query-count", type=int, default=96)
    parser.add_argument("--minimum-support-views", type=int, default=2)
    parser.add_argument("--deployment-row-limit", type=int, default=0)
    parser.add_argument("--context-kernels", type=_parse_kernels, default=(3, 7, 15))
    parser.add_argument(
        "--context-mode", choices=CONTEXT_MODES, default="multi_scale_global"
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--maximum-angle-rad", type=float, default=0.05)
    parser.add_argument("--stage-one-epochs", type=int, default=1)
    parser.add_argument("--stage-two-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repair-topk", type=int, default=32)
    parser.add_argument("--training-topk", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--collision-weight", type=float, default=1.0)
    parser.add_argument("--clean-weight", type=float, default=2.0)
    parser.add_argument("--clean-margin-slack", type=float, default=0.01)
    parser.add_argument("--clean-task-scale", type=float, default=0.25)
    parser.add_argument("--trust-weight", type=float, default=1.0)
    parser.add_argument("--maximum-positives", type=int, default=8)
    parser.add_argument("--maximum-ignored", type=int, default=16)
    parser.add_argument("--topks", type=_parse_topks, default=DEFAULT_TOPKS)
    parser.add_argument("--skip-pose-pnp", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--progress-interval", type=int, default=100)
    args = parser.parse_args()

    if args.minimum_support_views < 1 or args.repair_topk < 2:
        raise ValueError("minimum support views and repair top-K are invalid")
    if args.stage_one_epochs < 1 or args.stage_two_epochs < 0:
        raise ValueError("stage-one must be positive; stage-two non-negative")
    for label in (
        "support_query_count_per_direction",
        "gate_query_count",
        "pose_query_count",
        "deployment_row_limit",
    ):
        if int(getattr(args, label)) < 0:
            raise ValueError(f"{label} must be non-negative")

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    state = _load(args.map)
    metric_state = _load(args.metric_state)
    teacher = _load(args.complete_positive_teacher)
    query_cache = _load(args.query_cache)
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
    if int(teacher["anchor_count"]) != anchor_ids.numel():
        raise ValueError("map and teacher anchor counts differ")
    metric = load_frozen_metric_state(
        metric_state, anchor_ids=anchor_ids, device=device
    )
    base_anchor_features = F.normalize(
        torch.as_tensor(state["anchor_features"], device=device).float(), dim=1
    )
    names = list(teacher["query_names"])
    even, odd, split_report = temporal_crossfit_split(
        names, block_count=int(args.block_count)
    )
    threshold, threshold_source = _reprojection_threshold(
        args.scene_calibration, args.ransac_reprojection_px
    )
    directions = (
        ("even_blocks_to_odd_blocks", even, odd),
        ("odd_blocks_to_even_blocks", odd, even),
    )
    gate_budget = [
        (int(args.gate_query_count) + 1) // 2,
        int(args.gate_query_count) // 2,
    ]
    pose_budget = [
        (int(args.pose_query_count) + 1) // 2,
        int(args.pose_query_count) // 2,
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fold_reports = []
    all_pose_rows = []
    additive = {"a1": [], "uplift": []}
    clean_rows = []
    for direction_index, (direction_name, support_fold, gate_fold) in enumerate(
        directions
    ):
        support = _uniform_subset(
            support_fold, int(args.support_query_count_per_direction)
        )
        gate = _uniform_subset(gate_fold, gate_budget[direction_index])
        pose_queries = (
            []
            if args.skip_pose_pnp or int(args.pose_query_count) == 0
            else _uniform_subset(gate, pose_budget[direction_index])
        )
        print(
            {
                "event": "metric_context_direction_start",
                "direction": direction_name,
                "support_query_count": len(support),
                "gate_query_count": len(gate),
                "pose_query_count": len(pose_queries),
            },
            flush=True,
        )
        support_banks, support_report = build_raw_observation_bank(
            teacher=teacher,
            query_cache=query_cache,
            support_query_indices=support,
            device=device,
            minimum_support_views=int(args.minimum_support_views),
            progress_interval=int(args.progress_interval),
        )
        anchor_indices = support_banks["anchor_indices"].to(device)
        base_bank = base_anchor_features[anchor_indices]
        supported_types = torch.as_tensor(state["anchor_type"]).long()[
            anchor_indices.cpu()
        ]
        support_report["supported_track_anchor_count"] = int(
            (supported_types != 0).sum()
        )
        support_report["supported_reserve_anchor_count"] = int(
            (supported_types == 0).sum()
        )
        records, training_data = prepare_training_records(
            teacher=teacher,
            support_query_indices=support,
            anchor_indices=anchor_indices.cpu(),
            maximum_positives=int(args.maximum_positives),
            maximum_ignored=int(args.maximum_ignored),
        )
        adapter = MetricPreservingContextUplift(
            hidden_dim=int(args.hidden_dim),
            context_kernels=args.context_kernels,
            context_mode=args.context_mode,
            maximum_angle_rad=float(args.maximum_angle_rad),
        ).to(device)
        common_training = {
            "adapter": adapter,
            "metric": metric,
            "teacher": teacher,
            "query_cache": query_cache,
            "support_query_indices": support,
            "records": records,
            "base_reference_bank": base_bank,
            "device": device,
            "batch_size": int(args.batch_size),
            "repair_topk": int(args.repair_topk),
            "training_topk": int(args.training_topk),
            "learning_rate": float(args.learning_rate),
            "temperature": float(args.temperature),
            "collision_weight": float(args.collision_weight),
            "clean_weight": float(args.clean_weight),
            "clean_margin_slack": float(args.clean_margin_slack),
            "clean_task_scale": float(args.clean_task_scale),
            "trust_weight": float(args.trust_weight),
            "progress_interval": int(args.progress_interval),
        }
        training_reports = [
            train_metric_context_stage(
                **common_training,
                task_bank=base_bank,
                epochs=int(args.stage_one_epochs),
                seed=int(args.seed) + 100 * direction_index,
                stage_name="frozen_a1_anchor_target",
            )
        ]
        interim_bank, interim_report = build_metric_uplift_bank(
            adapter=adapter,
            metric=metric,
            base_anchor_bank=base_bank,
            teacher=teacher,
            query_cache=query_cache,
            support_query_indices=support,
            anchor_indices=anchor_indices,
            expected_view_counts=support_banks["view_counts"],
            device=device,
            progress_interval=int(args.progress_interval),
        )
        if int(args.stage_two_epochs) > 0:
            training_reports.append(
                train_metric_context_stage(
                    **common_training,
                    task_bank=interim_bank,
                    epochs=int(args.stage_two_epochs),
                    seed=int(args.seed) + 100 * direction_index + 1,
                    stage_name="contextual_uplift_anchor_target",
                )
            )
        final_bank, final_report = build_metric_uplift_bank(
            adapter=adapter,
            metric=metric,
            base_anchor_bank=base_bank,
            teacher=teacher,
            query_cache=query_cache,
            support_query_indices=support,
            anchor_indices=anchor_indices,
            expected_view_counts=support_banks["view_counts"],
            device=device,
            progress_interval=int(args.progress_interval),
        )
        banks = {
            "anchor_indices": anchor_indices,
            "a1": base_bank,
            "uplift": final_bank,
        }
        retrieval, pose_rows = evaluate_metric_context_banks(
            state=state,
            teacher=teacher,
            query_cache=query_cache,
            gate_query_indices=gate,
            pose_query_indices=pose_queries,
            banks=banks,
            metric=metric,
            adapter=adapter,
            device=device,
            topks=args.topks,
            deployment_row_limit=int(args.deployment_row_limit),
            ransac_reprojection_px=float(threshold),
            clean_margin_slack=float(args.clean_margin_slack),
            seed=int(args.seed),
            progress_interval=int(args.progress_interval),
        )
        fold_additive = retrieval.pop("additive_counts")
        for descriptor_name in additive:
            additive[descriptor_name].append(fold_additive[descriptor_name])
        clean_rows.append(retrieval["clean_preservation"])
        adapter_state = {
            key: value.detach().cpu()
            for key, value in adapter.state_dict().items()
        }
        adapter_hash = _state_sha256(adapter_state)
        checkpoint_path = args.output.with_name(
            f"{args.output.stem}_{direction_name}.pt"
        )
        torch.save(
            {
                "schema": "lafgs_metric_preserving_context_uplift_fold",
                "version": 1,
                "uses_test_queries": False,
                "fit_split": "mapping_support_fold",
                "base_descriptor_protocol": "frozen_shared_metric_and_learned_anchor",
                "metric_state": str(args.metric_state.resolve()),
                "metric_state_sha256": sha256_file(args.metric_state),
                "adapter_config": adapter.export_config(),
                "adapter_state_dict": adapter_state,
                "adapter_state_sha256": adapter_hash,
                "support_query_indices": support,
                "anchor_indices": anchor_indices.cpu(),
                "anchor_ids": anchor_ids[anchor_indices.cpu()],
                "anchor_features": final_bank.cpu().half(),
            },
            checkpoint_path,
        )
        fold_pose = summarize_metric_context_pose(pose_rows)
        fold_reports.append(
            {
                "direction": direction_name,
                "support_query_indices": support,
                "gate_query_indices": gate,
                "pose_query_indices": pose_queries,
                "support": support_report,
                "training_data": training_data,
                "training": training_reports,
                "interim_uplift_bank": interim_report,
                "final_uplift_bank": final_report,
                "adapter_config": adapter.export_config(),
                "adapter_state_sha256": adapter_hash,
                "checkpoint": str(checkpoint_path.resolve()),
                "retrieval": retrieval,
                "pose": fold_pose,
            }
        )
        for row in pose_rows:
            row["direction"] = direction_name
        all_pose_rows.extend(pose_rows)
        del support_banks, banks, interim_bank, final_bank, adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_retrieval = {}
    for descriptor_name, fold_counts in additive.items():
        aggregate_retrieval[descriptor_name] = summarize_retrieval(
            combine_additive_counts(fold_counts, args.topks), args.topks
        )
    aggregate_pose = summarize_metric_context_pose(all_pose_rows)
    comparison = compare_metric_context_protocols(
        aggregate_retrieval, aggregate_pose
    )
    report = {
        "schema": "lafgs_metric_preserving_context_uplift_crossfit",
        "version": 1,
        "uses_test_queries": False,
        "protocol": {
            "split": "bidirectional_trajectory_block_crossfit",
            "base_query_descriptor": "frozen_shared_low_rank_metric",
            "base_map_descriptor": "frozen_learned_a1_anchor_feature",
            "context_update": "gated_unit_sphere_tangent_angle",
            "map_update": "view_balanced_residual_with_direction_coherence_gate",
            "zero_initialization": "exact_a1_query_and_map_identity",
            "training_scope": "clean_a1_and_a1_false_positive_in_repair_topk",
            "online_matching": "one_descriptor_exact_global_cosine_top1",
            "pose_solver": "one_poselib_absolute_pose_call_per_protocol",
        },
        "inputs": {
            "map": str(args.map.resolve()),
            "metric_state": str(args.metric_state.resolve()),
            "metric_state_sha256": sha256_file(args.metric_state),
            "complete_positive_teacher": str(
                args.complete_positive_teacher.resolve()
            ),
            "query_cache": str(args.query_cache.resolve()),
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
            "context_kernels": list(args.context_kernels),
            "context_mode": str(args.context_mode),
            "hidden_dim": int(args.hidden_dim),
            "maximum_angle_rad": float(args.maximum_angle_rad),
            "stage_one_epochs": int(args.stage_one_epochs),
            "stage_two_epochs": int(args.stage_two_epochs),
            "batch_size": int(args.batch_size),
            "repair_topk": int(args.repair_topk),
            "training_topk": int(args.training_topk),
            "learning_rate": float(args.learning_rate),
            "temperature": float(args.temperature),
            "collision_weight": float(args.collision_weight),
            "clean_weight": float(args.clean_weight),
            "clean_margin_slack": float(args.clean_margin_slack),
            "clean_task_scale": float(args.clean_task_scale),
            "trust_weight": float(args.trust_weight),
            "topks": list(args.topks),
            "ransac_reprojection_px": float(threshold),
            "ransac_threshold_source": threshold_source,
            "seed": int(args.seed),
        },
        "split": split_report,
        "directions": fold_reports,
        "aggregate": {
            "retrieval": aggregate_retrieval,
            "clean_preservation": _combine_clean(clean_rows),
            "pose": aggregate_pose,
            "paired_pose": _pose_pairing(all_pose_rows),
            "comparison": comparison,
        },
        "pose_queries": all_pose_rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            "event": "metric_context_crossfit_complete",
            "output": str(args.output),
            "comparison": comparison,
            "clean_preservation": report["aggregate"]["clean_preservation"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
