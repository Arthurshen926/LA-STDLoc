"""Freeze small, hash-addressed references for release parity."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import torch

from common.config import load_mainline_config
from common.hashing import sha256_file


@dataclass(frozen=True)
class SceneBaseline:
    name: str
    root: Path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True
    ).strip()


def _hash_record(path: Path, *, artifact_id: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "artifact_id": artifact_id,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _result_dir(root: Path, variant: str, seed: int) -> Path:
    pointer = root / "evaluation" / variant / f"seed{seed}" / "result.path"
    return Path(pointer.read_text().strip()).resolve()


def _scene_artifacts(scene: SceneBaseline, repo: Path) -> dict[str, Any]:
    root = scene.root.resolve()
    reconstructed = root / "self_localization_reconstruction"
    map_path = reconstructed / "anchor_map_step_0175.pt"
    metric_path = reconstructed / "metric_state_step_0175.pt"
    prior = root / "prior" / "rgb_matcha_2dgs"
    artifacts = {
        "prior_ply": prior / "point_cloud/iteration_30000/point_cloud.ply",
        "prior_manifest": prior / "rgb_prior_manifest.json",
        "query_cache": root / "runs/frozen_v1/query_cache_native_fullres_k2048.pt",
        "track_payload": root / (
            "runs/frozen_v1/statistics_combined_1000_frozen_"
            "g3_track_provenance_v1/track_micro_anchor_payload.pt"
        ),
        "function_graph": root / "function_graph/function_graph_v3.pt",
        "raster_provenance": root / "function_graph/raster_provenance.pt",
        "anchor_map": map_path,
        "metric_state": metric_path,
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {scene.name} baseline artifacts: {missing}")
    return {
        name: _hash_record(path, artifact_id=f"{scene.name}/{name}")
        for name, path in artifacts.items()
    }


def _without_machine_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_machine_paths(item)
            for key, item in value.items()
            if key != "result_path"
        }
    if isinstance(value, list):
        return [_without_machine_paths(item) for item in value]
    return value


def _scene_metrics(scene: SceneBaseline) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen = _read_json(scene.root / "frozen_results.json")
    expected = {
        "query_count": {},
        "anchor_count": int(frozen["anchor_count"]),
        "variants": {},
    }
    golden: list[dict[str, Any]] = []
    for variant in ("A0_bootstrap", "A1_reconstructed"):
        expected["variants"][variant] = _without_machine_paths(
            frozen["results"][variant]
        )
        result_dir = _result_dir(scene.root, variant, 2026)
        rows = _read_json(result_dir / "results.json")
        expected["query_count"][variant] = len(rows)
        if variant == "A1_reconstructed" and scene.name == "ShopFacade":
            for row in rows[:16]:
                sparse = row["sparse"]
                golden.append(
                    {
                        "image_name": row["image_name"],
                        "pose_w2c": sparse["pose_w2c"],
                        "gt_pose_w2c": row["gt_pose_w2c"],
                        "matches": int(sparse["matches"]),
                        "inliers": int(sparse["inliers"]),
                        "translation_error_cm": float(row["sparse_TE"]),
                        "rotation_error_deg": float(row["sparse_AE"]),
                    }
                )
    return expected, golden


def _dependency_versions() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "opencv-python", "pyyaml", "scipy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
    }


def freeze_baseline(
    *,
    repo: Path,
    output: Path,
    config_path: Path,
    scenes: list[SceneBaseline],
) -> None:
    repo = repo.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_mainline_config(config_path)
    metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    golden: list[dict[str, Any]] = []
    for scene in scenes:
        metrics[scene.name], scene_golden = _scene_metrics(scene)
        artifacts[scene.name] = _scene_artifacts(scene, repo)
        golden.extend(scene_golden)

    config_manifest = config.manifest()
    config_manifest["config_path"] = str(
        Path(config.path).resolve().relative_to(repo)
    )
    files = {
        "baseline_manifest.json": {
            "schema": "lafgs_paper_baseline",
            "version": 1,
            "git_commit": _git(repo, "rev-parse", "HEAD"),
            "git_describe": _git(repo, "describe", "--tags", "--always"),
            "branch": _git(repo, "branch", "--show-current"),
            "source_tag": "paper-mainline-baseline-v1",
            "config": config_manifest,
            "scenes": [scene.name for scene in scenes],
            "environment": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python_executable": Path(sys.executable).name,
            },
        },
        "dependency_versions.json": _dependency_versions(),
        "artifact_hashes.json": artifacts,
        "expected_metrics.json": metrics,
        "golden_queries.json": {
            "schema": "lafgs_golden_queries",
            "version": 1,
            "scene": "ShopFacade",
            "variant": "A1_reconstructed",
            "seed": 2026,
            "queries": golden,
        },
        "golden_camera_list.json": [row["image_name"] for row in golden],
    }
    for name, payload in files.items():
        (output / name).write_text(json.dumps(payload, indent=2) + "\n")
    (output / "resolved_config.yaml").write_text(
        Path(config.path).read_text()
    )
