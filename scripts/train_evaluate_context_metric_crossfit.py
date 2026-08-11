#!/usr/bin/env python3
"""Train and gate the bounded dense-context descriptor in both directions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from map_learning.context_booster_crossfit import (
    DEFAULT_TOPKS,
    combine_additive_counts,
    summarize_retrieval,
)
from map_learning.context_metric import CONTEXT_MODES, MapConsistentContextAdapter
from map_learning.context_metric_crossfit import (
    build_context_observation_bank,
    build_raw_observation_bank,
    compare_mccd_protocols,
    evaluate_mccd_banks,
    prepare_training_records,
    summarize_mccd_pose,
    train_context_adapter_stage,
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


def _parse_kernels(value: str) -> tuple[int, ...]:
    kernels = tuple(int(item) for item in value.split(",") if item)
    if not kernels or any(item < 1 or item % 2 == 0 for item in kernels):
        raise argparse.ArgumentTypeError("context kernels must be positive odd integers")
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
        raise ValueError(
            "provide --scene-calibration or --ransac-reprojection-px"
        )
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
        [row["mccd_te_cm"] - row["raw_superpoint_te_cm"] for row in rows],
        dtype=np.float64,
    )
    worst = sorted(
        rows,
        key=lambda row: row["mccd_te_cm"] - row["raw_superpoint_te_cm"],
        reverse=True,
    )[:5]
    return {
        "query_count": int(delta.size),
        "mccd_better_count": int((delta < 0).sum()),
        "mccd_worse_count": int((delta > 0).sum()),
        "tie_count": int((delta == 0).sum()),
        "median_te_delta_cm": float(np.median(delta)),
        "mean_te_delta_cm": float(delta.mean()),
        "p90_te_delta_cm": float(np.percentile(delta, 90)),
        "worst_regressions": [
            {
                "image_name": row["image_name"],
                "raw_te_cm": float(row["raw_superpoint_te_cm"]),
                "mccd_te_cm": float(row["mccd_te_cm"]),
                "delta_cm": float(
                    row["mccd_te_cm"] - row["raw_superpoint_te_cm"]
                ),
                "direction": row.get("direction"),
            }
            for row in worst
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
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
    parser.add_argument("--maximum-residual-norm", type=float, default=0.10)
    parser.add_argument("--stage-one-epochs", type=int, default=1)
    parser.add_argument("--stage-two-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
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

    for label in (
        "support_query_count_per_direction",
        "gate_query_count",
        "pose_query_count",
        "deployment_row_limit",
    ):
        if int(getattr(args, label)) < 0:
            raise ValueError(f"{label} must be non-negative")
    if args.minimum_support_views < 1:
        raise ValueError("minimum support views must be positive")
    if args.stage_one_epochs < 1 or args.stage_two_epochs < 0:
        raise ValueError("stage-one epochs must be positive; stage-two non-negative")

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    state = _load_mmap(args.map)
    teacher = _load_mmap(args.complete_positive_teacher)
    query_cache = _load_mmap(args.query_cache)
    if int(teacher["anchor_count"]) != len(state["anchor_ids"]):
        raise ValueError("map and teacher anchor counts differ")
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
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fold_reports = []
    all_pose_rows = []
    additive = {"raw_superpoint": [], "mccd": []}
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
                "event": "mccd_direction_start",
                "direction": direction_name,
                "support_query_count": len(support),
                "gate_query_count": len(gate),
                "pose_query_count": len(pose_queries),
            },
            flush=True,
        )
        raw_banks, support_report = build_raw_observation_bank(
            teacher=teacher,
            query_cache=query_cache,
            support_query_indices=support,
            device=device,
            minimum_support_views=int(args.minimum_support_views),
            progress_interval=int(args.progress_interval),
        )
        supported_types = torch.as_tensor(state["anchor_type"]).long()[
            raw_banks["anchor_indices"].cpu()
        ]
        support_report["supported_track_anchor_count"] = int(
            (supported_types != 0).sum()
        )
        support_report["supported_reserve_anchor_count"] = int(
            (supported_types == 0).sum()
        )
        records, training_data_report = prepare_training_records(
            teacher=teacher,
            support_query_indices=support,
            anchor_indices=raw_banks["anchor_indices"],
            maximum_positives=int(args.maximum_positives),
            maximum_ignored=int(args.maximum_ignored),
        )
        adapter = MapConsistentContextAdapter(
            hidden_dim=int(args.hidden_dim),
            context_kernels=args.context_kernels,
            context_mode=args.context_mode,
            maximum_residual_norm=float(args.maximum_residual_norm),
        ).to(device)
        common_training = {
            "adapter": adapter,
            "teacher": teacher,
            "query_cache": query_cache,
            "support_query_indices": support,
            "records": records,
            "raw_reference_bank": raw_banks["raw_superpoint"],
            "device": device,
            "batch_size": int(args.batch_size),
            "topk": int(args.training_topk),
            "learning_rate": float(args.learning_rate),
            "temperature": float(args.temperature),
            "collision_weight": float(args.collision_weight),
            "clean_weight": float(args.clean_weight),
            "clean_margin_slack": float(args.clean_margin_slack),
            "clean_task_scale": float(args.clean_task_scale),
            "trust_weight": float(args.trust_weight),
            "progress_interval": int(args.progress_interval),
        }
        stage_reports = [
            train_context_adapter_stage(
                **common_training,
                task_bank=raw_banks["raw_superpoint"],
                epochs=int(args.stage_one_epochs),
                seed=int(args.seed) + direction_index * 100,
                stage_name="raw_support_target",
            )
        ]
        interim_bank, interim_report = build_context_observation_bank(
            adapter=adapter,
            teacher=teacher,
            query_cache=query_cache,
            support_query_indices=support,
            anchor_indices=raw_banks["anchor_indices"],
            expected_view_counts=raw_banks["view_counts"],
            device=device,
            progress_interval=int(args.progress_interval),
        )
        if int(args.stage_two_epochs) > 0:
            stage_reports.append(
                train_context_adapter_stage(
                    **common_training,
                    task_bank=interim_bank,
                    epochs=int(args.stage_two_epochs),
                    seed=int(args.seed) + direction_index * 100 + 1,
                    stage_name="context_support_target",
                )
            )
        final_bank, final_bank_report = build_context_observation_bank(
            adapter=adapter,
            teacher=teacher,
            query_cache=query_cache,
            support_query_indices=support,
            anchor_indices=raw_banks["anchor_indices"],
            expected_view_counts=raw_banks["view_counts"],
            device=device,
            progress_interval=int(args.progress_interval),
        )
        banks = {**raw_banks, "mccd": final_bank}
        retrieval, pose_rows = evaluate_mccd_banks(
            state=state,
            teacher=teacher,
            query_cache=query_cache,
            gate_query_indices=gate,
            pose_query_indices=pose_queries,
            banks=banks,
            adapter=adapter,
            device=device,
            topks=args.topks,
            deployment_row_limit=int(args.deployment_row_limit),
            ransac_reprojection_px=float(threshold),
            seed=int(args.seed),
            progress_interval=int(args.progress_interval),
        )
        fold_additive = retrieval.pop("additive_counts")
        for descriptor_name in additive:
            additive[descriptor_name].append(fold_additive[descriptor_name])
        adapter_state = {
            key: value.detach().cpu() for key, value in adapter.state_dict().items()
        }
        adapter_hash = _state_sha256(adapter_state)
        checkpoint_path = args.output.with_name(
            f"{args.output.stem}_{direction_name}.pt"
        )
        torch.save(
            {
                "schema": "lafgs_map_consistent_context_descriptor",
                "version": 1,
                "uses_test_queries": False,
                "adapter_config": adapter.export_config(),
                "adapter_state_dict": adapter_state,
                "adapter_state_sha256": adapter_hash,
                "support_query_indices": support,
                "anchor_indices": raw_banks["anchor_indices"].cpu(),
                "anchor_ids": torch.as_tensor(state["anchor_ids"])[
                    raw_banks["anchor_indices"].cpu()
                ],
                "anchor_features": final_bank.detach().cpu().half(),
            },
            checkpoint_path,
        )
        fold_reports.append(
            {
                "direction": direction_name,
                "support_query_indices": support,
                "gate_query_indices": gate,
                "pose_query_indices": pose_queries,
                "support": support_report,
                "training_data": training_data_report,
                "training": stage_reports,
                "interim_context_bank": interim_report,
                "final_context_bank": final_bank_report,
                "adapter_config": adapter.export_config(),
                "adapter_state_sha256": adapter_hash,
                "checkpoint": str(checkpoint_path.resolve()),
                "retrieval": retrieval,
                "pose": summarize_mccd_pose(pose_rows),
            }
        )
        for row in pose_rows:
            row["direction"] = direction_name
        all_pose_rows.extend(pose_rows)
        del banks, raw_banks, interim_bank, final_bank, adapter
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_retrieval = {}
    for descriptor_name, fold_counts in additive.items():
        combined = combine_additive_counts(fold_counts, args.topks)
        aggregate_retrieval[descriptor_name] = summarize_retrieval(
            combined, args.topks
        )
    aggregate_pose = summarize_mccd_pose(all_pose_rows)
    if all_pose_rows:
        comparison = compare_mccd_protocols(
            aggregate_retrieval, aggregate_pose
        )
    else:
        raw_r1 = aggregate_retrieval["raw_superpoint"][
            "positive_recall_at_k"
        ]["1"]
        mccd_r1 = aggregate_retrieval["mccd"]["positive_recall_at_k"]["1"]
        comparison = {
            "mccd_minus_raw_top1_positive_recall_percentage_points": float(
                100.0 * (mccd_r1 - raw_r1)
            ),
            "routing_verdict": "pose_replay_skipped_no_final_routing_verdict",
        }
    report = {
        "schema": "lafgs_mccd_crossfit",
        "version": 1,
        "uses_test_queries": False,
        "protocol": {
            "split": "bidirectional_trajectory_block_crossfit",
            "context": str(args.context_mode),
            "adapter": "identity_initialized_hard_bounded_residual_256d",
            "map_descriptor": "same_adapter_per_observation_then_view_balanced_fusion",
            "support_parity": "raw_and_mccd_use_identical_edges_and_anchor_mask",
            "online_matching": "one_descriptor_exact_global_cosine_top1",
            "pose_solver": "one_poselib_absolute_pose_call_per_protocol",
        },
        "inputs": {
            "map": str(args.map.resolve()),
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
            "maximum_residual_norm": float(args.maximum_residual_norm),
            "stage_one_epochs": int(args.stage_one_epochs),
            "stage_two_epochs": int(args.stage_two_epochs),
            "batch_size": int(args.batch_size),
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
            "pose": aggregate_pose,
            "paired_pose": _pose_pairing(all_pose_rows),
            "comparison": comparison,
        },
        "pose_queries": all_pose_rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            "event": "mccd_crossfit_complete",
            "output": str(args.output),
            "comparison": comparison,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
