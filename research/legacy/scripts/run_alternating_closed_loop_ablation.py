#!/usr/bin/env python3
"""Two-round opt-in descriptor/topology self-localization experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from common.config import load_mainline_config
from map_learning.trainer import full_refresh_interval, train


def _load_paths(path: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text())
    return {key: Path(value).expanduser().resolve() for key, value in payload.items()}


def _run(*arguments: object) -> None:
    subprocess.run([str(value) for value in arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--round1-ablation-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--maximum-swaps", type=int, default=32)
    parser.add_argument("--maximum-prune-fraction", type=float, default=0.02)
    parser.add_argument(
        "--stop-if-no-structure-change",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    pipeline = args.pipeline_root.resolve()
    round1 = args.round1_ablation_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = _load_paths(pipeline / "pipeline_manifest.json")
    round1_report = json.loads((round1 / "ablation_report.json").read_text())
    round1_map = Path(round1_report["trained_map"]).resolve()
    round1_metric = Path(round1_report["metric_state"]).resolve()
    calibration = json.loads(artifacts["scene_calibration"].read_text())
    parameters = calibration["parameters"]

    swap_dir = output / "round1_crossfit_swap"
    swap_report_path = swap_dir / "crossfit_swap_report.json"
    if not swap_report_path.is_file():
        _run(
            sys.executable,
            "-m",
            "topology.crossfit_swap_revision",
            "--map",
            round1_map,
            "--metric-state",
            round1_metric,
            "--complete-positive-teacher",
            artifacts["compact_positive_teacher"],
            "--canonical-map",
            artifacts["canonical_map"],
            "--base-positive-teacher",
            artifacts["positive_teacher"],
            "--function-graph",
            artifacts["function_graph"],
            "--track-payload",
            artifacts["track_payload"],
            "--query-cache",
            artifacts["query_cache"],
            "--raster-provenance",
            artifacts["compact_provenance"],
            "--output-dir",
            swap_dir,
            "--matching-rows-target",
            int(parameters["matching_rows_target"]),
            "--ransac-reprojection-px",
            parameters["ransac_reprojection_px"],
            "--clean-reprojection-px",
            parameters["clean_radius_px"],
            "--maximum-swaps",
            args.maximum_swaps,
            "--seed",
            args.seed,
            "--device",
            args.device,
        )
    swap_report = json.loads(swap_report_path.read_text())
    active_map = round1_map
    active_metric = round1_metric
    active_teacher = artifacts["compact_positive_teacher"]
    if bool(swap_report["accepted"]):
        active_map = Path(swap_report["revised_map"])
        active_metric = Path(swap_report["revised_metric_state"])
        active_teacher = Path(swap_report["revised_teacher"])

    retire_dir = output / "round1_retire"
    retire_report_path = retire_dir / "deployment_revision_report.json"
    if not retire_report_path.is_file():
        _run(
            sys.executable,
            "-m",
            "topology.deployment_revision",
            "--map",
            active_map,
            "--metric-state",
            active_metric,
            "--complete-positive-teacher",
            active_teacher,
            "--query-cache",
            artifacts["query_cache"],
            "--output-dir",
            retire_dir,
            "--matching-rows-target",
            int(parameters["matching_rows_target"]),
            "--ransac-reprojection-px",
            parameters["ransac_reprojection_px"],
            "--clean-reprojection-px",
            parameters["clean_radius_px"],
            "--task-translation-m",
            parameters["task_translation_m"],
            "--task-rotation-deg",
            parameters["task_rotation_deg"],
            "--maximum-prune-fraction",
            args.maximum_prune_fraction,
            "--seed",
            args.seed,
            "--device",
            args.device,
        )
    retire_report = json.loads(retire_report_path.read_text())
    if bool(retire_report["accepted"]):
        active_map = Path(retire_report["revised_map"])
        active_metric = Path(retire_report["revised_metric_state"])
        active_teacher = Path(retire_report["revised_teacher"])

    structure_changed = bool(swap_report["accepted"]) or bool(
        retire_report["accepted"]
    )
    if args.stop_if_no_structure_change and not structure_changed:
        report = {
            "schema": "lafgs_two_round_alternating_closed_loop_ablation",
            "version": 1,
            "changes_default_mainline": False,
            "status": "stopped_after_rejected_structure_m_step",
            "swap_gate_accepted": False,
            "swap_count": int(swap_report["proposal"]["accepted_pair_count"]),
            "retire_gate_accepted": False,
            "retired_anchor_count": int(
                retire_report["selection"]["final_prune_count"]
            ),
            "round2_executed": False,
            "uses_test_queries": False,
        }
        (output / "alternating_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    values = load_mainline_config(args.config).values
    reconstruction = values["reconstruction"]
    deployment_row_limit = int(values["deployment"]["keypoints"])
    steps = int(parameters["metric_steps"])
    refresh_shards = 7
    round2_dir = output / "round2_map_learning"
    round2_map = round2_dir / f"anchor_map_step_{steps:04d}.pt"
    round2_metric = round2_dir / f"metric_state_step_{steps:04d}.pt"
    if not round2_map.is_file() or not round2_metric.is_file():
        train(
            map_path=active_map,
            function_graph_path=artifacts["function_graph"],
            track_payload_path=artifacts["track_payload"],
            query_cache_path=artifacts["query_cache"],
            positive_teacher_path=active_teacher,
            output_dir=round2_dir,
            initial_metric_state_path=active_metric,
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
            refresh_interval=full_refresh_interval(steps, refresh_shards),
            refresh_shards=refresh_shards,
            deployment_row_limit=deployment_row_limit,
            ransac_reprojection_px=float(parameters["ransac_reprojection_px"]),
            clean_reprojection_px=float(parameters["clean_radius_px"]),
            seed=args.seed,
        )

    evaluation = output / "test"
    if not (evaluation / "summary.json").is_file():
        _run(
            sys.executable,
            "scripts/evaluate.py",
            "--dataset",
            args.dataset.resolve(),
            "--map",
            round2_map,
            "--metric-state",
            round2_metric,
            "--scene-calibration",
            artifacts["scene_calibration"],
            "--config",
            args.config.resolve(),
            "--output",
            evaluation,
            "--split",
            "test",
            "--seed",
            args.seed,
            "--device",
            args.device,
        )
    report = {
        "schema": "lafgs_two_round_alternating_closed_loop_ablation",
        "version": 1,
        "changes_default_mainline": False,
        "round1": round1_report,
        "swap_gate_accepted": bool(swap_report["accepted"]),
        "swap_count": int(swap_report["proposal"]["accepted_pair_count"]),
        "retire_gate_accepted": bool(retire_report["accepted"]),
        "retired_anchor_count": int(
            retire_report["selection"]["final_prune_count"]
        ),
        "round2_source_map": str(active_map),
        "round2_map": str(round2_map),
        "round2_metric": str(round2_metric),
        "deployment_row_limit": deployment_row_limit,
        "test": json.loads((evaluation / "summary.json").read_text()),
        "round2_executed": True,
    }
    (output / "alternating_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
