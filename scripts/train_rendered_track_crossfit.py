#!/usr/bin/env python3
"""Train and validate A1 on mapping sequences only.

Each fold consumes the bank materialized without the held sequence, trains the
shared metric on the remaining mapping sequences, and evaluates only the held
mapping sequence. No test query is loaded or scored.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common.hashing import sha256_file
from map_learning.trainer import train
from scripts.evaluate_rendered_track_crossfit import (
    COUNTER_NAMES,
    _combined_summary,
    _sequence_name,
)
from topology.deployment_revision import collect_deployment_statistics


def _atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _training_inputs(
    *,
    held_sequence: str,
    fold_teacher: dict,
    full_payload: dict,
) -> tuple[dict, dict, dict]:
    all_names = list(fold_teacher["query_names"])
    selected_indices = [
        index
        for index, name in enumerate(all_names)
        if _sequence_name(name) != held_sequence
    ]
    selected_names = [all_names[index] for index in selected_indices]
    sequence_names = sorted({_sequence_name(name) for name in selected_names})
    sequence_id = {sequence: index for index, sequence in enumerate(sequence_names)}
    records = []
    positive_rows = strong_pairs = ambiguous_pairs = 0
    for new_index, old_index in enumerate(selected_indices):
        source = fold_teacher["records"][old_index]
        record = dict(source)
        record["query_index"] = new_index
        record["query_name"] = selected_names[new_index]
        records.append(record)
        positive_offsets = torch.as_tensor(record["positive_offsets"]).long()
        positive_rows += int(((positive_offsets[1:] - positive_offsets[:-1]) > 0).sum())
        strong_pairs += int(torch.as_tensor(record["positive_indices"]).numel())
        ambiguous_pairs += int(torch.as_tensor(record["ambiguous_indices"]).numel())
    teacher = {
        **fold_teacher,
        "query_names": selected_names,
        "records": records,
        "diagnostics": {
            **fold_teacher["diagnostics"],
            "query_count": len(records),
            "positive_rows": positive_rows,
            "strong_pair_count": strong_pairs,
            "ambiguous_pair_count": ambiguous_pairs,
        },
        "crossfit": {
            "held_mapping_sequence": held_sequence,
            "uses_test_queries": False,
        },
    }
    graph = {
        "schema": "lafgs_rendered_track_training_graph",
        "version": 1,
        "query_names": selected_names,
        "records": [
            {
                "query_index": index,
                "query_rows": record["query_rows"].clone(),
                "ambiguous_training_policy": "ignore",
            }
            for index, record in enumerate(records)
        ],
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "held_mapping_sequence": held_sequence,
    }
    payload = {
        "schema": full_payload.get("schema", "lafgs_track_first_payload"),
        "version": full_payload.get("version", 1),
        "query_names": selected_names,
        "query_bins": torch.as_tensor(
            [sequence_id[_sequence_name(name)] for name in selected_names]
        ).long(),
        "training_sequence_names": sequence_names,
        "held_mapping_sequence": held_sequence,
        "uses_test_queries": False,
    }
    return teacher, graph, payload


def run(args) -> dict:
    teacher = torch.load(args.teacher, map_location="cpu", weights_only=False)
    payload = torch.load(args.track_payload, map_location="cpu", weights_only=False)
    cache = torch.load(args.query_cache, map_location="cpu", weights_only=False)
    if cache.get("uses_source_mapping_rgb") is not False:
        raise ValueError("training cache is not rendered-RGB-only")
    if cache.get("uses_test_queries") is not False:
        raise ValueError("training cache contains test queries")
    names = list(teacher["query_names"])
    sequences = sorted({_sequence_name(name) for name in names})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    full_count = int(teacher["anchor_count"])
    aggregate = {
        name: torch.zeros(full_count, dtype=torch.float64) for name in COUNTER_NAMES
    }
    all_rows = []
    folds = []
    for fold_index, held_sequence in enumerate(sequences):
        if args.held_sequence and held_sequence != args.held_sequence:
            continue
        fold_dir = args.output_dir / held_sequence
        fold_dir.mkdir()
        source_fold = args.identity_crossfit / held_sequence
        fold_map_path = source_fold / "anchor_map.pt"
        fold_teacher_path = source_fold / "positive_teacher.pt"
        fold_map = torch.load(fold_map_path, map_location="cpu", weights_only=False)
        fold_teacher = torch.load(
            fold_teacher_path, map_location="cpu", weights_only=False
        )
        train_teacher, graph, train_payload = _training_inputs(
            held_sequence=held_sequence,
            fold_teacher=fold_teacher,
            full_payload=payload,
        )
        teacher_path = fold_dir / "train_teacher.pt"
        graph_path = fold_dir / "train_graph.pt"
        payload_path = fold_dir / "train_payload.pt"
        _atomic_save(train_teacher, teacher_path)
        _atomic_save(graph, graph_path)
        _atomic_save(train_payload, payload_path)
        training_dir = fold_dir / "a1"
        train(
            map_path=fold_map_path,
            function_graph_path=graph_path,
            track_payload_path=payload_path,
            query_cache_path=args.query_cache,
            positive_teacher_path=teacher_path,
            output_dir=training_dir,
            steps=args.steps,
            checkpoint_steps=(args.steps,),
            batch_size=args.batch_size,
            topk=args.topk,
            max_positives=args.max_positives,
            rank=args.rank,
            metric_residual=args.metric_residual,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            harmful_weight=args.harmful_weight,
            trust_weight=args.trust_weight,
            group_dro_eta=args.group_dro_eta,
            group_dro_max_weight_ratio=args.group_dro_max_weight_ratio,
            anchor_feature_residual_max_norm=(args.anchor_feature_residual_max_norm),
            anchor_feature_residual_trust_weight=(
                args.anchor_feature_residual_trust_weight
            ),
            soft_pose_weight=args.soft_pose_weight,
            soft_pose_topk=args.soft_pose_topk,
            soft_pose_temperature=args.soft_pose_temperature,
            soft_pose_inlier_softness_px=args.soft_pose_inlier_softness_px,
            soft_pose_miss_weight=args.soft_pose_miss_weight,
            refresh_interval=0,
            refresh_shards=4,
            initial_ransac_refresh=False,
            seed=args.seed + fold_index,
        )
        trained_map_path = training_dir / f"anchor_map_step_{args.steps:04d}.pt"
        metric_path = training_dir / f"metric_state_step_{args.steps:04d}.pt"
        query_indices = [
            index
            for index, name in enumerate(names)
            if _sequence_name(name) == held_sequence
        ]
        statistics = collect_deployment_statistics(
            state=torch.load(trained_map_path, map_location="cpu", weights_only=False),
            metric_state_path=metric_path,
            teacher=fold_teacher,
            query_cache=cache,
            device=torch.device(args.device),
            ransac_reprojection_px=args.ransac_reprojection_px,
            clean_reprojection_px=args.clean_reprojection_px,
            task_translation_m=args.task_translation_m,
            task_rotation_deg=args.task_rotation_deg,
            seed=args.seed,
            query_indices=query_indices,
            progress_label=f"a1_held_{held_sequence}",
        )
        statistics_path = fold_dir / "statistics.pt"
        _atomic_save(statistics, statistics_path)
        source_keep = torch.as_tensor(fold_map["anchor_ids"]).long()
        # Fold anchor IDs were compacted, so recover full rows by Track ID.
        full_track = torch.as_tensor(payload.get("selected_track_ids", [])).long()
        if full_track.numel() == 0:
            # The training map preserves original selected Track IDs.
            source_track = torch.as_tensor(fold_map["track_cluster_ids"]).long()
            original_track = torch.as_tensor(
                torch.load(args.anchor_map, map_location="cpu", weights_only=False)[
                    "track_cluster_ids"
                ]
            ).long()
            lookup = {
                int(value): index for index, value in enumerate(original_track.tolist())
            }
            source_keep = torch.as_tensor(
                [lookup[int(value)] for value in source_track]
            )
        for name in COUNTER_NAMES:
            aggregate[name][source_keep] += torch.as_tensor(
                statistics["counters"][name]
            ).double()
        all_rows.extend(statistics["queries"])
        folds.append(
            {
                "held_sequence": held_sequence,
                "train_sequence_count": len(sequences) - 1,
                "train_supported_anchor_count": int(fold_map["anchor_ids"].numel()),
                "summary": statistics["summary"],
                "trained_map": str(trained_map_path.resolve()),
                "metric_state": str(metric_path.resolve()),
                "statistics": str(statistics_path.resolve()),
            }
        )
        print(json.dumps(folds[-1], sort_keys=True), flush=True)

    if not folds:
        raise RuntimeError("requested held mapping sequence was not found")

    combined = {
        "schema": "lafgs_rendered_track_a1_mapping_sequence_crossfit_statistics",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": all_rows,
        "counters": aggregate,
        "summary": _combined_summary(all_rows, aggregate),
        "folds": folds,
    }
    statistics_path = args.output_dir / "crossfit_statistics.pt"
    _atomic_save(combined, statistics_path)
    report = {
        "schema": "lafgs_rendered_track_a1_mapping_sequence_crossfit_report",
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "topk": args.topk,
            "max_positives": args.max_positives,
            "rank": args.rank,
            "metric_residual": args.metric_residual,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "harmful_weight": args.harmful_weight,
            "trust_weight": args.trust_weight,
            "group_dro_eta": args.group_dro_eta,
            "group_dro_max_weight_ratio": args.group_dro_max_weight_ratio,
            "anchor_feature_residual_max_norm": (args.anchor_feature_residual_max_norm),
            "anchor_feature_residual_trust_weight": (
                args.anchor_feature_residual_trust_weight
            ),
            "soft_pose_weight": args.soft_pose_weight,
            "soft_pose_topk": args.soft_pose_topk,
            "soft_pose_temperature": args.soft_pose_temperature,
            "soft_pose_inlier_softness_px": args.soft_pose_inlier_softness_px,
            "soft_pose_miss_weight": args.soft_pose_miss_weight,
        },
        "folds": folds,
        "combined_summary": combined["summary"],
        "statistics": str(statistics_path.resolve()),
        "statistics_sha256": sha256_file(statistics_path),
    }
    (args.output_dir / "crossfit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-map", type=Path, required=True)
    parser.add_argument("--track-payload", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--identity-crossfit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--held-sequence", default="")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--max-positives", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--metric-residual", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.04)
    parser.add_argument("--harmful-weight", type=float, default=0.1)
    parser.add_argument("--trust-weight", type=float, default=1.0)
    parser.add_argument("--group-dro-eta", type=float, default=0.03)
    parser.add_argument("--group-dro-max-weight-ratio", type=float, default=3.0)
    parser.add_argument("--anchor-feature-residual-max-norm", type=float, default=0.0)
    parser.add_argument(
        "--anchor-feature-residual-trust-weight", type=float, default=1.0
    )
    parser.add_argument("--soft-pose-weight", type=float, default=0.0)
    parser.add_argument("--soft-pose-topk", type=int, default=8)
    parser.add_argument("--soft-pose-temperature", type=float, default=0.05)
    parser.add_argument("--soft-pose-inlier-softness-px", type=float, default=1.0)
    parser.add_argument("--soft-pose-miss-weight", type=float, default=0.05)
    parser.add_argument("--ransac-reprojection-px", type=float, default=12.0)
    parser.add_argument("--clean-reprojection-px", type=float, default=4.0)
    parser.add_argument("--task-translation-m", type=float, default=0.05)
    parser.add_argument("--task-rotation-deg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for field in (
        "anchor_map",
        "track_payload",
        "teacher",
        "query_cache",
        "identity_crossfit",
    ):
        setattr(args, field, getattr(args, field).resolve())
    args.output_dir = args.output_dir.resolve()
    print(json.dumps(run(args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
