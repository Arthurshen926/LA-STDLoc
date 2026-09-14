"""Registered multi-scene benchmark aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


_DEFAULT_SEEDS = (2026, 2027, 2028)


@dataclass(frozen=True)
class RegisteredScene:
    name: str
    root: Path
    marker: Mapping[str, object]


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_registered_scenes(
    run_root: Path,
    scene_names: Sequence[str],
) -> list[RegisteredScene]:
    scenes: list[RegisteredScene] = []
    reference_contract: tuple[object, ...] | None = None
    for name in scene_names:
        root = run_root / name
        marker_path = root / "full_benchmark_complete.json"
        if not marker_path.is_file():
            raise FileNotFoundError(f"missing registered marker: {marker_path}")
        marker = _read_json(marker_path)
        if marker.get("test_images_used_for_training") is not False:
            raise ValueError(f"scene {name} does not certify zero test leakage")
        if marker.get("mapping_only_prior") is not True:
            raise ValueError(f"scene {name} does not use a mapping-only prior")
        if tuple(marker.get("a0_seeds", ())) != _DEFAULT_SEEDS:
            raise ValueError(f"scene {name} has unexpected A0 seeds")
        if tuple(marker.get("a1_seeds", ())) != _DEFAULT_SEEDS:
            raise ValueError(f"scene {name} has unexpected A1 seeds")
        contract = (
            marker.get("schema"),
            marker.get("family"),
            marker.get("config_sha256"),
            marker.get("runner_sha256"),
            marker.get("function_graph_shards"),
            marker.get("provenance_shards"),
            marker.get("observation_shards"),
            marker.get("pose_scoring_shards"),
        )
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            raise ValueError(f"scene {name} has a mismatched benchmark contract")
        scenes.append(RegisteredScene(name=name, root=root, marker=marker))
    if not scenes:
        raise ValueError("at least one scene is required")
    return scenes


def _result_path(scene: RegisteredScene, stage: str, seed: int) -> Path:
    if stage == "a0":
        return scene.root / f"evaluation_a0_seed{seed}" / "results.json"
    if stage != "a1":
        raise ValueError(f"unsupported stage: {stage}")
    if seed == _DEFAULT_SEEDS[0]:
        return scene.root / "evaluation" / "results.json"
    return scene.root / f"evaluation_a1_seed{seed}" / "results.json"


def _summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot summarize an empty result set")
    te = np.asarray([row["translation_error_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row["rotation_error_deg"] for row in rows], dtype=np.float64)
    total_ms = np.asarray([row["total_ms"] for row in rows], dtype=np.float64)
    raw_count = sum(int(row["raw_count"]) for row in rows)
    inlier_count = sum(int(row["inlier_count"]) for row in rows)
    if raw_count <= 0 or inlier_count <= 0:
        raise ValueError("registered results contain no correspondences")
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "median_ae_deg": float(np.median(ae)),
        "mean_ae_deg": float(np.mean(ae)),
        "p90_ae_deg": float(np.percentile(ae, 90)),
        "recall_2cm_2deg_percent": 100.0 * float(np.mean((te <= 2) & (ae <= 2))),
        "recall_5cm_5deg_percent": 100.0 * float(np.mean((te <= 5) & (ae <= 5))),
        "raw_gt_precision_2px_percent": (
            100.0 * sum(int(row["raw_correct_2px"]) for row in rows) / raw_count
        ),
        "inlier_gt_precision_2px_percent": (
            100.0
            * sum(int(row["inlier_correct_2px"]) for row in rows)
            / inlier_count
        ),
        "solver_inlier_ratio_percent": 100.0 * inlier_count / raw_count,
        "mean_hypotheses": float(
            np.mean([row["ransac_iterations"] for row in rows])
        ),
        "catastrophic_100cm_count": int(np.sum(te > 100)),
        "total_ms_p50": float(np.median(total_ms)),
        "total_ms_p90": float(np.percentile(total_ms, 90)),
    }


def _mean_metrics(metrics: Sequence[Mapping[str, float | int]]) -> dict[str, float]:
    return {
        key: float(np.mean([float(item[key]) for item in metrics]))
        for key in metrics[0]
    }


def _anchor_count(scene: RegisteredScene) -> int:
    report = _read_json(scene.root / "map_learning" / "training_report.json")
    config = report["config"]
    return int(config["track_anchor_count"]) + int(config["base_anchor_count"])


def aggregate_registered_benchmark(
    run_root: Path,
    scene_names: Sequence[str],
) -> dict[str, object]:
    scenes = load_registered_scenes(run_root, scene_names)
    stages: dict[str, object] = {}
    per_scene: dict[str, object] = {}
    for stage in ("a0", "a1"):
        pooled_by_seed: list[dict[str, float | int]] = []
        scene_stage: dict[str, object] = {}
        for scene in scenes:
            seed_metrics = []
            for seed in _DEFAULT_SEEDS:
                path = _result_path(scene, stage, seed)
                if not path.is_file():
                    raise FileNotFoundError(f"missing registered results: {path}")
                seed_metrics.append(_summarize_rows(_read_json(path)))
            scene_stage[scene.name] = {
                "seed_mean": _mean_metrics(seed_metrics),
                "seeds": dict(zip(map(str, _DEFAULT_SEEDS), seed_metrics)),
            }
        for seed in _DEFAULT_SEEDS:
            pooled_rows = []
            for scene in scenes:
                pooled_rows.extend(_read_json(_result_path(scene, stage, seed)))
            pooled_by_seed.append(_summarize_rows(pooled_rows))
        stages[stage] = {
            "pooled_seed_mean": _mean_metrics(pooled_by_seed),
            "pooled_seeds": dict(zip(map(str, _DEFAULT_SEEDS), pooled_by_seed)),
        }
        for scene_name, values in scene_stage.items():
            per_scene.setdefault(scene_name, {})[stage] = values

    anchor_counts = {scene.name: _anchor_count(scene) for scene in scenes}
    for scene_name, count in anchor_counts.items():
        per_scene[scene_name]["a0_anchor_count"] = 48_000
        per_scene[scene_name]["a1_anchor_count"] = count
    first = scenes[0].marker
    return {
        "schema": "lafgs_registered_benchmark_aggregate_v1",
        "family": first["family"],
        "scene_count": len(scenes),
        "scenes": list(scene_names),
        "config_sha256": first["config_sha256"],
        "runner_sha256": first["runner_sha256"],
        "seeds": list(_DEFAULT_SEEDS),
        "stages": stages,
        "a0_anchor_count": 48_000,
        "a1_anchor_count_mean": float(np.mean(list(anchor_counts.values()))),
        "per_scene": per_scene,
    }


def latex_rows(dataset: str, aggregate: Mapping[str, object]) -> str:
    stages = aggregate["stages"]
    anchor_counts = {
        "a0": float(aggregate["a0_anchor_count"]),
        "a1": float(aggregate["a1_anchor_count_mean"]),
    }
    lines = []
    for stage in ("a0", "a1"):
        metrics = stages[stage]["pooled_seed_mean"]
        lines.append(
            f"{dataset} & {stage.upper()} & "
            f"{metrics['median_te_cm']:.3f} & {metrics['mean_te_cm']:.3f} & "
            f"{metrics['p90_te_cm']:.3f} & {metrics['median_ae_deg']:.3f} & "
            f"{metrics['recall_5cm_5deg_percent']:.2f} & "
            f"{metrics['raw_gt_precision_2px_percent']:.2f} & "
            f"{anchor_counts[stage] / 1000:.2f}K & "
            f"{metrics['total_ms_p90']:.1f} \\\\"
        )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def iter_scene_names(value: Iterable[str]) -> list[str]:
    names = [name.strip() for name in value if name.strip()]
    if len(names) != len(set(names)):
        raise ValueError("scene names must be unique")
    return names
