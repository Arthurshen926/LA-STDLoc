#!/usr/bin/env python3
"""Fit the selected MCCD configuration on all mapping observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from map_learning.context_metric import CONTEXT_MODES, MapConsistentContextAdapter
from map_learning.context_metric_crossfit import (
    build_context_observation_bank,
    build_raw_observation_bank,
    prepare_training_records,
    train_context_adapter_stage,
)


def _load_mmap(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_kernels(value: str) -> tuple[int, ...]:
    kernels = tuple(int(item) for item in value.split(",") if item)
    if not kernels or any(item < 1 or item % 2 == 0 for item in kernels):
        raise argparse.ArgumentTypeError("context kernels must be positive odd integers")
    return kernels


def _state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        tensor = torch.as_tensor(value).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--complete-positive-teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--minimum-support-views", type=int, default=2)
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--progress-interval", type=int, default=200)
    args = parser.parse_args()
    if args.minimum_support_views < 1:
        raise ValueError("minimum support views must be positive")
    if args.stage_one_epochs < 0 or args.stage_two_epochs < 0:
        raise ValueError("training epochs must be non-negative")

    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    state = _load_mmap(args.map)
    teacher = _load_mmap(args.complete_positive_teacher)
    query_cache = _load_mmap(args.query_cache)
    if int(teacher["anchor_count"]) != len(state["anchor_ids"]):
        raise ValueError("map and teacher anchor counts differ")
    support = list(range(len(teacher["query_names"])))
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
    training_reports = []
    if int(args.stage_one_epochs) > 0:
        training_reports.append(
            train_context_adapter_stage(
                **common_training,
                task_bank=raw_banks["raw_superpoint"],
                epochs=int(args.stage_one_epochs),
                seed=int(args.seed),
                stage_name="raw_support_target",
            )
        )
    identity_only = (
        int(args.stage_one_epochs) == 0
        and int(args.stage_two_epochs) == 0
        and float(args.maximum_residual_norm) == 0.0
    )
    if identity_only:
        interim_bank = raw_banks["raw_superpoint"]
        interim_report = {
            "adapted_observation_count": int(
                support_report["positive_edge_count"]
            ),
            "residual_norm_mean": 0.0,
            "residual_norm_median": 0.0,
            "residual_norm_p90": 0.0,
            "residual_norm_maximum": 0.0,
        }
    else:
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
        training_reports.append(
            train_context_adapter_stage(
                **common_training,
                task_bank=interim_bank,
                epochs=int(args.stage_two_epochs),
                seed=int(args.seed) + 1,
                stage_name="context_support_target",
            )
        )
    if identity_only:
        final_bank, final_report = interim_bank, dict(interim_report)
    else:
        final_bank, final_report = build_context_observation_bank(
            adapter=adapter,
            teacher=teacher,
            query_cache=query_cache,
            support_query_indices=support,
            anchor_indices=raw_banks["anchor_indices"],
            expected_view_counts=raw_banks["view_counts"],
            device=device,
            progress_interval=int(args.progress_interval),
        )
    adapter_state = {
        key: value.detach().cpu() for key, value in adapter.state_dict().items()
    }
    adapter_hash = _state_sha256(adapter_state)
    artifact = {
        "schema": "lafgs_map_consistent_context_descriptor",
        "version": 1,
        "uses_test_queries": False,
        "fit_split": "all_mapping",
        "adapter_config": adapter.export_config(),
        "adapter_state_dict": adapter_state,
        "adapter_state_sha256": adapter_hash,
        "support_query_indices": support,
        "anchor_indices": raw_banks["anchor_indices"].cpu(),
        "anchor_ids": torch.as_tensor(state["anchor_ids"])[
            raw_banks["anchor_indices"].cpu()
        ],
        "anchor_features": final_bank.detach().cpu().half(),
        "base_map": str(args.map.resolve()),
        "complete_positive_teacher": str(
            args.complete_positive_teacher.resolve()
        ),
        "query_cache": str(args.query_cache.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    report_path = args.report or args.output.with_suffix(".json")
    report = {
        "schema": "lafgs_mccd_full_mapping_fit",
        "version": 1,
        "uses_test_queries": False,
        "output": str(args.output.resolve()),
        "adapter_state_sha256": adapter_hash,
        "adapter_config": adapter.export_config(),
        "support": support_report,
        "training_data": training_data_report,
        "training": training_reports,
        "identity_only": bool(identity_only),
        "interim_context_bank": interim_report,
        "final_context_bank": final_report,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        {
            "event": "mccd_full_mapping_fit_complete",
            "output": str(args.output),
            "report": str(report_path),
            "supported_anchor_count": support_report["supported_anchor_count"],
            "adapter_state_sha256": adapter_hash,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
