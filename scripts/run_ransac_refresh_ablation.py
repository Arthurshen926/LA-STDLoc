#!/usr/bin/env python3
"""Opt-in full-shard RANSAC refresh and detector-prefix A1 ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from common.config import load_mainline_config, materialize_keypoint_factor_config
from map_learning.trainer import full_refresh_interval, train


def _manifest(root: Path) -> dict[str, Path]:
    payload = json.loads((root / "pipeline_manifest.json").read_text())
    return {key: Path(value).expanduser().resolve() for key, value in payload.items()}


def _run_evaluation(
    *,
    dataset: Path,
    map_path: Path,
    metric_path: Path,
    calibration: Path | None,
    config: Path,
    output: Path,
    split: str,
    seed: int,
    device: str,
) -> dict:
    destination = output / f"{split}_seed{seed}"
    summary = destination / "summary.json"
    if not summary.is_file():
        command = [
            sys.executable,
            "scripts/evaluate.py",
            "--dataset",
            str(dataset),
            "--map",
            str(map_path),
            "--metric-state",
            str(metric_path),
            "--config",
            str(config),
            "--output",
            str(destination),
            "--split",
            split,
            "--seed",
            str(seed),
            "--device",
            device,
        ]
        if calibration is not None:
            command += ["--scene-calibration", str(calibration)]
        subprocess.run(command, check=True)
    return json.loads(summary.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refresh-shards", type=int, default=7)
    parser.add_argument("--deployment-keypoints", type=int, default=0)
    parser.add_argument("--soft-pose-weight", type=float, default=0.0)
    parser.add_argument("--density-prefix-fractions", default="1.0")
    parser.add_argument("--density-dro-eta", type=float, default=0.03)
    parser.add_argument("--alias-weight", type=float, default=0.0)
    parser.add_argument("--alias-margin", type=float, default=0.05)
    parser.add_argument("--alias-minimum-distinct-groups", type=int, default=2)
    parser.add_argument("--alias-minimum-queries", type=int, default=3)
    parser.add_argument("--alias-minimum-occurrences", type=int, default=6)
    parser.add_argument("--alias-minimum-rows-per-query", type=int, default=2)
    parser.add_argument("--alias-query-replay-fraction", type=float, default=0.5)
    parser.add_argument(
        "--alias-require-harmful-inlier",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--protected-clean-weight", type=float, default=0.0)
    parser.add_argument("--protected-clean-minimum-margin", type=float, default=0.05)
    parser.add_argument("--protected-clean-margin-slack", type=float, default=0.01)
    parser.add_argument("--protected-clean-task-scale", type=float, default=0.25)
    parser.add_argument("--anchor-feature-residual-max-norm", type=float, default=0.0)
    parser.add_argument(
        "--anchor-feature-residual-trust-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--anchor-feature-residual-alias-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--anchor-feature-residual-include-alias-positives",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--freeze-shared-metric",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seeds", default="2026")
    parser.add_argument("--splits", default="mapping,test")
    parser.add_argument(
        "--continue-current-metric",
        action="store_true",
        help="Continue the frozen A1 metric instead of retraining from compact A0.",
    )
    parser.add_argument("--initial-metric-state", type=Path)
    args = parser.parse_args()

    root = args.pipeline_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = _manifest(root)
    config = (
        args.config.resolve()
        if args.config is not None
        else next(root.glob("factor_config_k*.yaml"), Path("configs/paper_mainline.yaml")).resolve()
    )
    if int(args.deployment_keypoints) > 0:
        config = materialize_keypoint_factor_config(
            config,
            output / f"deployment_k{int(args.deployment_keypoints)}.yaml",
            int(args.deployment_keypoints),
        )
    values = load_mainline_config(config).values
    reconstruction = values["reconstruction"]
    calibration_path = artifacts.get("scene_calibration")
    if calibration_path is None:
        calibration_path = next(
            (
                path
                for path in (
                    root / "map_learning" / "scene_calibration.json",
                    root / "evidence" / "scene_calibration.json",
                )
                if path.is_file()
            ),
            None,
        )
    calibration = (
        json.loads(calibration_path.read_text())
        if calibration_path is not None
        else None
    )
    parameters = calibration["parameters"] if calibration is not None else {}
    steps = int(parameters.get("metric_steps", reconstruction["metric_steps"]))
    refresh_interval = full_refresh_interval(steps, int(args.refresh_shards))
    train_dir = output / "map_learning"
    trained_map = train_dir / f"anchor_map_step_{steps:04d}.pt"
    metric_state = train_dir / f"metric_state_step_{steps:04d}.pt"
    if not trained_map.is_file() or not metric_state.is_file():
        train(
            map_path=artifacts["compact_map"],
            function_graph_path=artifacts["function_graph"],
            track_payload_path=artifacts["track_payload"],
            query_cache_path=artifacts["query_cache"],
            positive_teacher_path=artifacts["compact_positive_teacher"],
            output_dir=train_dir,
            initial_metric_state_path=(
                args.initial_metric_state.resolve()
                if args.initial_metric_state is not None
                else artifacts["metric_state"]
                if args.continue_current_metric
                else None
            ),
            steps=steps,
            checkpoint_steps=(steps,),
            rank=int(reconstruction["metric_rank"]),
            metric_residual=float(reconstruction["metric_residual"]),
            learning_rate=float(reconstruction["learning_rate"]),
            temperature=float(reconstruction["temperature"]),
            harmful_weight=float(reconstruction["harmful_weight"]),
            trust_weight=float(reconstruction["trust_weight"]),
            group_dro_eta=float(reconstruction["group_dro_eta"]),
            group_dro_max_weight_ratio=float(
                reconstruction["group_dro_max_weight_ratio"]
            ),
            refresh_interval=refresh_interval,
            refresh_shards=int(args.refresh_shards),
            deployment_row_limit=int(args.deployment_keypoints),
            density_prefix_fractions=tuple(
                float(value)
                for value in args.density_prefix_fractions.split(",")
                if value
            ),
            density_dro_eta=float(args.density_dro_eta),
            alias_weight=float(args.alias_weight),
            alias_margin=float(args.alias_margin),
            alias_minimum_distinct_groups=int(args.alias_minimum_distinct_groups),
            alias_minimum_queries=int(args.alias_minimum_queries),
            alias_minimum_occurrences=int(args.alias_minimum_occurrences),
            alias_minimum_rows_per_query=int(args.alias_minimum_rows_per_query),
            alias_query_replay_fraction=float(args.alias_query_replay_fraction),
            alias_require_harmful_inlier=bool(
                args.alias_require_harmful_inlier
            ),
            protected_clean_weight=float(args.protected_clean_weight),
            protected_clean_minimum_margin=float(
                args.protected_clean_minimum_margin
            ),
            protected_clean_margin_slack=float(args.protected_clean_margin_slack),
            protected_clean_task_scale=float(args.protected_clean_task_scale),
            anchor_feature_residual_max_norm=float(
                args.anchor_feature_residual_max_norm
            ),
            anchor_feature_residual_trust_weight=float(
                args.anchor_feature_residual_trust_weight
            ),
            anchor_feature_residual_alias_only=bool(
                args.anchor_feature_residual_alias_only
            ),
            anchor_feature_residual_include_alias_positives=bool(
                args.anchor_feature_residual_include_alias_positives
            ),
            freeze_shared_metric=bool(args.freeze_shared_metric),
            soft_pose_weight=float(args.soft_pose_weight),
            task_translation_m=float(parameters.get("task_translation_m", 0.05)),
            task_rotation_deg=float(parameters.get("task_rotation_deg", 5.0)),
            ransac_reprojection_px=float(
                parameters.get(
                    "ransac_reprojection_px",
                    values["deployment"]["reprojection_error_px"],
                )
            ),
            clean_reprojection_px=float(parameters.get("clean_radius_px", 4.0)),
            seed=2026,
        )

    results = {}
    for split in [value for value in args.splits.split(",") if value]:
        for seed in [int(value) for value in args.seeds.split(",") if value]:
            results[f"{split}_seed{seed}"] = _run_evaluation(
                dataset=args.dataset.resolve(),
                map_path=trained_map,
                metric_path=metric_state,
                calibration=calibration_path,
                config=config,
                output=output / "evaluation",
                split=split,
                seed=seed,
                device=args.device,
            )
    report = {
        "schema": "lafgs_ransac_full_refresh_ablation",
        "version": 1,
        "changes_default_mainline": False,
        "source_pipeline": str(root),
        "trained_map": str(trained_map),
        "metric_state": str(metric_state),
        "steps": steps,
        "refresh_shards": int(args.refresh_shards),
        "refresh_interval": refresh_interval,
        "deployment_keypoints": int(args.deployment_keypoints),
        "soft_pose_weight": float(args.soft_pose_weight),
        "density_prefix_fractions": [
            float(value)
            for value in args.density_prefix_fractions.split(",")
            if value
        ],
        "density_dro_eta": float(args.density_dro_eta),
        "alias_weight": float(args.alias_weight),
        "alias_margin": float(args.alias_margin),
        "alias_minimum_distinct_groups": int(args.alias_minimum_distinct_groups),
        "alias_minimum_queries": int(args.alias_minimum_queries),
        "alias_minimum_occurrences": int(args.alias_minimum_occurrences),
        "alias_minimum_rows_per_query": int(args.alias_minimum_rows_per_query),
        "alias_query_replay_fraction": float(args.alias_query_replay_fraction),
        "alias_require_harmful_inlier": bool(args.alias_require_harmful_inlier),
        "protected_clean_weight": float(args.protected_clean_weight),
        "protected_clean_minimum_margin": float(
            args.protected_clean_minimum_margin
        ),
        "protected_clean_margin_slack": float(args.protected_clean_margin_slack),
        "protected_clean_task_scale": float(args.protected_clean_task_scale),
        "anchor_feature_residual_max_norm": float(
            args.anchor_feature_residual_max_norm
        ),
        "anchor_feature_residual_trust_weight": float(
            args.anchor_feature_residual_trust_weight
        ),
        "anchor_feature_residual_alias_only": bool(
            args.anchor_feature_residual_alias_only
        ),
        "anchor_feature_residual_include_alias_positives": bool(
            args.anchor_feature_residual_include_alias_positives
        ),
        "freeze_shared_metric": bool(args.freeze_shared_metric),
        "continued_current_metric": bool(args.continue_current_metric),
        "initial_metric_state": (
            str(args.initial_metric_state.resolve())
            if args.initial_metric_state is not None
            else None
        ),
        "results": results,
    }
    (output / "ablation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
