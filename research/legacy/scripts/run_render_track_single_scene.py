#!/usr/bin/env python3
"""Run the frozen V4 rendered-RGB Track pipeline for one scene and one seed.

The mapping stages never read source mapping RGB or test queries.  The test
split is opened only after the rendered feature cache, Track payload, selector,
training map and identity metric have been atomically written.  Every stage is
restartable from a completed artifact, while partial output directories are
rejected instead of silently reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ENV_PYTHON = Path("/root/miniconda3/envs/g4splat/bin/python")
MAPPING_STAGE_NAMES = (
    "base",
    "appearance",
    "support",
    "surface_completion",
    "selector_inputs",
    "selector",
    "training",
    "identity",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_status(path: Path, value: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{int(value)}\n")
    os.replace(temporary, path)


def collect_scene_build_timing(
    scene_root: Path,
    *,
    invocation_started_unix: float,
    evaluation_stage: str,
) -> dict:
    """Aggregate restartable stage ledgers without rerunning any stage."""
    records = {}
    executed = []
    reused = []
    for stage in (*MAPPING_STAGE_NAMES, evaluation_stage):
        path = scene_root / "logs" / f"{stage}.timing.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing stage timing ledger: {path}")
        payload = json.loads(path.read_text())
        if set(payload) != {"returncode", "seconds"}:
            raise ValueError(f"invalid stage timing ledger: {path}")
        current = path.stat().st_mtime_ns >= int(invocation_started_unix * 1e9)
        record = {
            "seconds": float(payload["seconds"]),
            "returncode": int(payload["returncode"]),
            "executed_in_current_invocation": bool(current),
        }
        records[stage] = record
        (executed if current else reused).append(stage)
    current_mapping_seconds = sum(
        float(records[stage]["seconds"])
        for stage in MAPPING_STAGE_NAMES
        if bool(records[stage]["executed_in_current_invocation"])
    )
    return {
        "schema": "lafgs_v4_render_track_build_timing",
        "version": 1,
        "invocation_started_unix": float(invocation_started_unix),
        "current_invocation_wall_seconds": float(time.time() - invocation_started_unix),
        "current_mapping_subprocess_seconds": float(current_mapping_seconds),
        "executed_stages": executed,
        "cache_hit_stages": reused,
        "stages": records,
    }


def run_module(
    *,
    code_root: Path,
    scene_root: Path,
    stage: str,
    module: str,
    arguments: list[str],
    env: dict[str, str],
) -> None:
    logs = scene_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{stage}.stdout.log"
    stderr_path = logs / f"{stage}.stderr.log"
    status_path = logs / f"{stage}.status.txt"
    command = [str(ENV_PYTHON), "-u", "-m", module, *arguments]
    started = time.time()
    (logs / f"{stage}.command.json").write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(code_root),
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(
            command,
            cwd=code_root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    atomic_status(status_path, result.returncode)
    atomic_json(
        logs / f"{stage}.timing.json",
        {"returncode": result.returncode, "seconds": time.time() - started},
    )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def calibrate(code_root: Path, cache: Path, track: Path, env: dict[str, str]) -> dict:
    source = """
import json, sys
from pathlib import Path
from common.calibration import calibrate_scene
from common.config import load_mainline_config
policy = dict(load_mainline_config(Path(sys.argv[3])).values['adaptive'])
print(json.dumps(calibrate_scene(Path(sys.argv[1]), Path(sys.argv[2]), policy), sort_keys=True))
"""
    output = subprocess.check_output(
        [
            str(ENV_PYTHON),
            "-c",
            source,
            str(cache),
            str(track),
            str(code_root / "configs/paper_mainline.yaml"),
        ],
        cwd=code_root,
        env=env,
        text=True,
    )
    return json.loads(output)


def require_complete(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")


def reject_partial(path: Path, required: tuple[Path, ...], *, label: str) -> None:
    if not path.exists():
        return
    if all(item.exists() for item in required):
        return
    raise RuntimeError(f"partial {label} output; quarantine it before retry: {path}")


def build_base(
    *,
    code_root: Path,
    scene_root: Path,
    dataset: Path,
    prior: Path,
    env: dict[str, str],
) -> tuple[Path, Path, Path]:
    base = scene_root / "base"
    cache = base / "rendered_rgb_feature_cache.pt"
    track = base / "rendered_rgb_track_payload.pt"
    anchor_map = base / "rendered_rgb_track_map.pt"
    if cache.exists() and track.exists() and anchor_map.exists():
        return cache, track, anchor_map
    reject_partial(base, (cache, track, anchor_map), label="base")
    run_module(
        code_root=code_root,
        scene_root=scene_root,
        stage="base",
        module="scripts.probe_rendered_rgb_track_map",
        arguments=[
            "--dataset",
            str(dataset),
            "--images",
            "processed",
            "--gaussian-ply",
            str(prior),
            "--gaussian-type",
            "2dgs",
            "--sh-degree",
            "3",
            "--output-dir",
            str(base),
            "--keypoints",
            "2048",
            "--nms-radius",
            "4",
            "--pair-policy",
            "nearest",
            "--device",
            "cuda:0",
            "--progress-interval",
            "25",
        ],
        env=env,
    )
    for path, label in (
        (cache, "feature cache"),
        (track, "Track payload"),
        (anchor_map, "Track map"),
    ):
        require_complete(path, label=label)
    return cache, track, anchor_map


def run_scene(args: argparse.Namespace) -> dict:
    invocation_started = time.time()
    code_root = Path(args.code_root).expanduser().resolve()
    scene_root = Path(args.output).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    prior = Path(args.prior).expanduser().resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if not prior.is_file():
        raise FileNotFoundError(prior)
    scene_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(code_root)
    env["PATH"] = f"{ENV_PYTHON.parent}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"{ENV_PYTHON.parent.parent / 'lib'}:"
        f"{ENV_PYTHON.parent.parent / 'lib/python3.9/site-packages/torch/lib'}:"
        f"{env.get('LD_LIBRARY_PATH', '')}"
    )
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "8")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "8")
    base_cache, base_track, base_map = build_base(
        code_root=code_root,
        scene_root=scene_root,
        dataset=dataset,
        prior=prior,
        env=env,
    )

    appearance = scene_root / "appearance"
    appearance_cache = appearance / "appearance_ensemble_cache.pt"
    if not appearance_cache.exists():
        reject_partial(appearance, (appearance_cache,), label="appearance")
        run_module(
            code_root=code_root,
            scene_root=scene_root,
            stage="appearance",
            module="scripts.materialize_rendered_track_appearance_ensemble",
            arguments=[
                "--dataset",
                str(dataset),
                "--images",
                "processed",
                "--gaussian-ply",
                str(prior),
                "--gaussian-type",
                "2dgs",
                "--sh-degree",
                "3",
                "--source-cache",
                str(base_cache),
                "--track-payload",
                str(base_track),
                "--selected-map",
                str(base_map),
                "--output-dir",
                str(appearance),
                "--nms-radius",
                "4",
                "--alpha-minimum",
                "0.05",
                "--alpha-erosion-radius",
                "4",
                "--descriptor-trim-fraction",
                "0.2",
                "--progress-interval",
                "25",
            ],
            env=env,
        )
    require_complete(appearance_cache, label="appearance cache")

    initial = calibrate(code_root, appearance_cache, base_track, env)
    support = scene_root / "support_repair"
    repaired_track = support / "support_repaired_track_payload.pt"
    if not repaired_track.exists():
        reject_partial(support, (repaired_track,), label="support")
        run_module(
            code_root=code_root,
            scene_root=scene_root,
            stage="support",
            module="scripts.materialize_rendered_track_support_repair",
            arguments=[
                "--source-cache",
                str(base_cache),
                "--expected-source-cache-sha256",
                sha256(base_cache),
                "--support-cache",
                str(appearance_cache),
                "--expected-support-cache-sha256",
                sha256(appearance_cache),
                "--source-track-payload",
                str(base_track),
                "--expected-source-track-payload-sha256",
                sha256(base_track),
                "--output-dir",
                str(support),
                "--depth-abs-tolerance-m",
                str(initial["parameters"]["evidence_depth_abs_tolerance_m"]),
                "--device",
                "cuda:0",
                "--progress-interval",
                "25",
            ],
            env=env,
        )
    require_complete(repaired_track, label="repaired Track payload")
    final = calibrate(code_root, appearance_cache, repaired_track, env)
    parameters = final["parameters"]

    completion = scene_root / "surface_completion.pt"
    if not completion.exists():
        run_module(
            code_root=code_root,
            scene_root=scene_root,
            stage="surface_completion",
            module="scripts.materialize_rendered_surface_completion",
            arguments=[
                "--query-cache",
                str(appearance_cache),
                "--expected-query-cache-sha256",
                sha256(appearance_cache),
                "--track-payload",
                str(repaired_track),
                "--expected-track-payload-sha256",
                sha256(repaired_track),
                "--output",
                str(completion),
                "--voxel-size-m",
                str(parameters["surface_group_voxel_m"]),
                "--maximum-candidates",
                "512",
                "--maximum-rows-per-view",
                "256",
                "--alpha-minimum",
                "0.05",
                "--minimum-observations",
                "3",
                "--minimum-views",
                "2",
                "--minimum-pose-bins",
                "2",
                "--descriptor-trim-fraction",
                "0.2",
            ],
            env=env,
        )
    require_complete(completion, label="surface completion")

    selector_inputs = scene_root / "surface_selector_inputs"
    graph = selector_inputs / "surface_completion_function_graph.pt"
    if not graph.exists():
        reject_partial(selector_inputs, (graph,), label="selector inputs")
        run_module(
            code_root=code_root,
            scene_root=scene_root,
            stage="selector_inputs",
            module="scripts.materialize_surface_completion_selector_inputs",
            arguments=[
                "--surface-map",
                str(completion),
                "--expected-surface-map-sha256",
                sha256(completion),
                "--query-cache",
                str(appearance_cache),
                "--expected-query-cache-sha256",
                sha256(appearance_cache),
                "--track-payload",
                str(repaired_track),
                "--expected-track-payload-sha256",
                sha256(repaired_track),
                "--output-dir",
                str(selector_inputs),
            ],
            env=env,
        )
    selector = scene_root / "surface_unified_selector"
    selected_maps = sorted(selector.glob("adaptive_compact_total*.pt"))
    if not selected_maps:
        reject_partial(
            selector, (selector / "scene_calibration.json",), label="selector"
        )
        run_module(
            code_root=code_root,
            scene_root=scene_root,
            stage="selector",
            module="topology.adaptive_distillation",
            arguments=[
                "--canonical-map",
                str(completion),
                "--function-graph",
                str(graph),
                "--complete-positive-teacher",
                str(selector_inputs / "surface_completion_teacher.pt"),
                "--track-payload",
                str(repaired_track),
                "--query-cache",
                str(appearance_cache),
                "--rendered-track-pose-minimum-additions",
                "0",
                "--output-dir",
                str(selector),
            ],
            env=env,
        )
        selected_maps = sorted(selector.glob("adaptive_compact_total*.pt"))
    if len(selected_maps) != 1:
        raise RuntimeError(
            f"expected exactly one selected map in {selector}, got {selected_maps}"
        )
    compact_map = selected_maps[0]
    scene_calibration = selector / "scene_calibration.json"
    require_complete(scene_calibration, label="scene calibration")

    training = scene_root / "surface_unified_training"
    training_map = training / "rendered_track_training_map.pt"
    if not training_map.exists():
        reject_partial(training, (training_map,), label="training")
        run_module(
            code_root=code_root,
            scene_root=scene_root,
            stage="training",
            module="scripts.materialize_rendered_track_training",
            arguments=[
                "--anchor-map",
                str(compact_map),
                "--track-payload",
                str(repaired_track),
                "--query-cache",
                str(appearance_cache),
                "--output-dir",
                str(training),
                "--strong-radius-px",
                str(parameters["positive_radius_px"]),
                "--ambiguous-radius-px",
                str(parameters["negative_radius_px"]),
                "--single-trajectory-pose-cells",
                "3",
                "--alpha-minimum",
                "0.05",
                "--depth-abs-tolerance-m",
                str(parameters["evidence_depth_abs_tolerance_m"]),
                "--depth-relative-tolerance",
                "0.02",
                "--scene-calibration",
                str(scene_calibration),
                "--expected-scene-calibration-sha256",
                sha256(scene_calibration),
            ],
            env=env,
        )
    require_complete(training_map, label="training map")
    metric = scene_root / "surface_unified_identity_metric.pt"
    if not metric.exists():
        run_module(
            code_root=code_root,
            scene_root=scene_root,
            stage="identity",
            module="scripts.materialize_rendered_track_identity_metric",
            arguments=[
                "--map",
                str(training_map),
                "--expected-map-sha256",
                sha256(training_map),
                "--output",
                str(metric),
            ],
            env=env,
        )
    require_complete(metric, label="identity metric")

    evaluation = scene_root / f"surface_unified_test_seed{args.seed}"
    summary_path = evaluation / "summary.json"
    if not summary_path.exists():
        reject_partial(evaluation, (summary_path,), label="evaluation")
        run_module(
            code_root=code_root,
            scene_root=scene_root,
            stage=f"test_seed{args.seed}",
            module="scripts.evaluate",
            arguments=[
                "--dataset",
                str(dataset),
                "--images",
                "processed",
                "--map",
                str(training_map),
                "--metric-state",
                str(metric),
                "--scene-calibration",
                str(scene_calibration),
                "--output",
                str(evaluation),
                "--split",
                "test",
                "--device",
                "cuda:0",
                "--seed",
                str(args.seed),
            ],
            env=env,
        )
    require_complete(summary_path, label="test summary")
    summary = json.loads(summary_path.read_text())
    build_timing_path = scene_root / "build_timing.json"
    atomic_json(
        build_timing_path,
        collect_scene_build_timing(
            scene_root,
            invocation_started_unix=invocation_started,
            evaluation_stage=f"test_seed{args.seed}",
        ),
    )
    result = {
        "schema": "lafgs_v4_render_track_single_seed_scene_result",
        "version": 1,
        "scene": args.scene,
        "dataset": str(dataset),
        "prior": str(prior),
        "seed": int(args.seed),
        "code_root": str(code_root),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=code_root, text=True
        ).strip(),
        "uses_source_mapping_rgb": False,
        "uses_test_queries_for_map_construction": False,
        "completion_candidate_provider": "always_enabled",
        "descriptor_transform": "identity",
        "artifacts": {
            "base_cache": str(base_cache),
            "base_cache_sha256": sha256(base_cache),
            "base_track": str(base_track),
            "base_track_sha256": sha256(base_track),
            "appearance_cache": str(appearance_cache),
            "appearance_cache_sha256": sha256(appearance_cache),
            "repaired_track": str(repaired_track),
            "repaired_track_sha256": sha256(repaired_track),
            "selected_map": str(compact_map),
            "selected_map_sha256": sha256(compact_map),
            "training_map": str(training_map),
            "training_map_sha256": sha256(training_map),
            "metric": str(metric),
            "metric_sha256": sha256(metric),
            "scene_calibration": str(scene_calibration),
            "scene_calibration_sha256": sha256(scene_calibration),
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
            "build_timing": str(build_timing_path),
            "build_timing_sha256": sha256(build_timing_path),
        },
        "summary": summary,
    }
    atomic_json(scene_root / "single_seed_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        result = run_scene(args)
    except Exception as exc:  # status is persisted for the matrix supervisor
        root = Path(args.output).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(
            root / "single_seed_error.json", {"scene": args.scene, "error": repr(exc)}
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
