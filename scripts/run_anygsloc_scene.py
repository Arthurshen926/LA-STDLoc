#!/usr/bin/env python3
"""Build and evaluate the frozen, feedback-free AnyGSLoc Base for one scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from common.config import ANYGSLOC_SCHEMA, load_mainline_config


SCHEMA = "anygsloc_scene_experiment"
PYTHON = Path("/root/miniconda3/envs/g4splat/bin/python")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_prior(args: argparse.Namespace) -> dict[str, Any]:
    if (args.prior_manifest is None) == (args.gaussian_ply is None):
        raise ValueError("select exactly one of --prior-manifest and --gaussian-ply")
    if args.prior_manifest is not None:
        manifest_path = args.prior_manifest.expanduser().resolve()
        payload = json.loads(manifest_path.read_text())
        if payload.get("prior_kind") != "rgb_only":
            raise ValueError("prior manifest must describe a frozen RGB-only prior")
        if payload.get("prior_training_used_feature_loss") is not False:
            raise ValueError("feature-trained Gaussian priors are outside AnyGSLoc")
        ply = Path(payload["exported_ply"]).resolve()
        gaussian_type = str(payload["gaussian_type"])
        sh_degree = int(payload["sh_degree"])
        expected = str(payload["exported_ply_sha256"])
        if sha256_file(ply) != expected:
            raise ValueError("normalized prior PLY differs from its manifest")
        return {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "ply": str(ply),
            "ply_sha256": expected,
            "gaussian_type": gaussian_type,
            "sh_degree": sh_degree,
            "white_background": bool(payload.get("white_background", False)),
        }
    ply = args.gaussian_ply.expanduser().resolve()
    if args.gaussian_type is None or args.sh_degree is None:
        raise ValueError("explicit PLY requires --gaussian-type and --sh-degree")
    return {
        "manifest": None,
        "manifest_sha256": None,
        "ply": str(ply),
        "ply_sha256": sha256_file(ply),
        "gaussian_type": args.gaussian_type,
        "sh_degree": int(args.sh_degree),
        "white_background": bool(args.white_background),
    }


def command_plan(args: argparse.Namespace, prior: dict[str, Any]) -> list[dict[str, Any]]:
    root = args.output.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    observation = root / "observations" / "rendered_rgb_feature_cache.pt"
    audit_dir = root / "v2_audit"
    audit_shards = [audit_dir / f"render_quality_shard_{i:03d}_of_{args.audit_shards:03d}.pt" for i in range(args.audit_shards)]
    map_dir = root / "projective_map"
    base_eval = root / "evaluation" / f"base_seed{args.seed}"
    common_prior = [
        "--gaussian-ply", prior["ply"],
        "--gaussian-type", prior["gaussian_type"],
        "--sh-degree", str(prior["sh_degree"]),
    ]
    if prior["white_background"]:
        common_prior.append("--white-background")
    stages: list[dict[str, Any]] = [
        {
            "name": "observations",
            "output": str(observation),
            "command": [
                str(PYTHON), "-u", "-m", "scripts.probe_rendered_rgb_track_map",
                "--dataset", str(dataset), "--images", args.images,
                *common_prior,
                "--output-dir", str(observation.parent), "--render-only",
                "--keypoints", "2048", "--nms-radius", "4",
                "--detection-threshold", "0.0",
                "--render-valid-alpha-minimum", "0.05",
                "--render-valid-neighborhood-radius", "4",
                "--device", "cuda:0", "--cpu-threads", str(args.cpu_threads),
            ],
        }
    ]
    for index, output in enumerate(audit_shards):
        stages.append(
            {
                "name": f"v2_audit_{index:03d}",
                "output": str(output),
                "command": [
                    str(PYTHON), "-u", "-m", "scripts.audit_v7_mapping_render_quality_shard",
                    "--dataset", str(dataset), "--images", args.images,
                    *common_prior,
                    "--observation-cache", str(observation), "--output", str(output),
                    "--shard-index", str(index), "--shard-count", str(args.audit_shards),
                    "--device", "cuda:0",
                ],
            }
        )
    stages.extend(
        [
            {
                "name": "projective_map",
                "output": str(map_dir / "report.json"),
                "command": [
                    str(PYTHON), "-u", "-m", "scripts.materialize_v8_v2_projective_map",
                    "--observation-cache", str(observation),
                    "--v2-audit-shards", *map(str, audit_shards),
                    "--output-dir", str(map_dir), "--device", "cuda:0",
                    "--cpu-threads", str(args.cpu_threads),
                    "--triangulation-workers", str(args.triangulation_workers),
                ],
            },
            {
                "name": "base_evaluation",
                "output": str(base_eval / "summary.json"),
                "command": [
                    str(PYTHON), "-u", "-m", "scripts.evaluate",
                    "--dataset", str(dataset), "--images", args.images,
                    "--map", str(map_dir / "projective_anchor_map.pt"),
                    "--metric-state", str(map_dir / "identity_metric.pt"),
                    "--output", str(base_eval), "--config", str(args.config.resolve()),
                    "--split", "test", "--device", "cuda:0", "--seed", str(args.seed),
                    "--deployment-mode",
                ],
            },
        ]
    )
    return stages


def run_stage(stage: dict[str, Any], *, root: Path, env: dict[str, str]) -> dict[str, Any]:
    output = Path(stage["output"])
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return {"status": "reused", "seconds": 0.0, "output_sha256": sha256_file(output)}
    if stage["name"] == "projective_map" and output.parent.exists():
        raise RuntimeError(f"partial map output must be quarantined: {output.parent}")
    if stage["name"] == "base_evaluation" and output.parent.exists():
        raise RuntimeError(f"partial evaluation output must be quarantined: {output.parent}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command_path = log_dir / f"{stage['name']}.command.json"
    atomic_json(command_path, {"command": stage["command"], "cwd": str(Path.cwd())})
    started = time.perf_counter()
    with (log_dir / f"{stage['name']}.stdout.log").open("wb") as stdout, (
        log_dir / f"{stage['name']}.stderr.log"
    ).open("wb") as stderr:
        result = subprocess.run(stage["command"], env=env, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, stage["command"])
    if not output.is_file():
        raise RuntimeError(f"stage completed without declared output: {output}")
    record = {"status": "executed", "seconds": elapsed, "output_sha256": sha256_file(output)}
    atomic_json(log_dir / f"{stage['name']}.timing.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", default="processed")
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--gaussian-ply", type=Path)
    parser.add_argument("--gaussian-type", choices=("2dgs", "3dgs"))
    parser.add_argument("--sh-degree", type=int)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/anygsloc_mainline.yaml"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--audit-shards", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--triangulation-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mapping-only", action="store_true")
    args = parser.parse_args()
    if args.audit_shards < 1 or args.cpu_threads < 1 or args.triangulation_workers < 1:
        parser.error("worker and shard counts must be positive")
    args.config = args.config.expanduser().resolve()
    config = load_mainline_config(args.config)
    if config.values["schema"] != ANYGSLOC_SCHEMA:
        parser.error("--config must be the feedback-free AnyGSLoc schema")
    if not args.dataset.expanduser().resolve().is_dir():
        raise FileNotFoundError(args.dataset)
    prior = resolve_prior(args)
    plan = command_plan(args, prior)
    if args.mapping_only:
        plan = [stage for stage in plan if stage["name"] != "base_evaluation"]
    invocation = {
        "schema": SCHEMA,
        "version": 1,
        "scene": args.scene,
        "dataset": str(args.dataset.expanduser().resolve()),
        "prior": prior,
        "config": config.manifest(),
        "scientific_scope": {
            "mapping_uses_source_rgb": False,
            "mapping_uses_test_queries": False,
            "offline_self_localization_feedback": False,
            "descriptor_or_metric_training": False,
            "test_query_map_adaptation": False,
            "query_rendering": False,
            "dense_matching": False,
            "map_writeback": False,
        },
        "stages": plan,
    }
    if args.dry_run:
        print(json.dumps(invocation, indent=2, sort_keys=True))
        return
    root = args.output.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["OMP_NUM_THREADS"] = str(args.cpu_threads)
    env["MKL_NUM_THREADS"] = str(args.cpu_threads)
    started = time.time()
    results = {}
    for stage in plan:
        results[stage["name"]] = run_stage(stage, root=root, env=env)
    invocation["results"] = results
    invocation["wall_seconds"] = time.time() - started
    invocation["complete"] = True
    atomic_json(root / "anygsloc_scene_manifest.json", invocation)
    print(json.dumps(invocation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
